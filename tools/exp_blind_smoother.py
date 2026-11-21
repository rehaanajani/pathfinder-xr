"""Experiment: where should the blind rows of a drone recording be? (diagnosis + offline smoother candidates)

    python tools/exp_blind_smoother.py diagnose [--seeds s42 s1 s7] [--data euroc:MH_02_easy ...]
    python tools/exp_blind_smoother.py sweep                     # development recordings, seeds 42 and 1
    python tools/exp_blind_smoother.py check --setting NAME      # all 17, seeds 42, 1, 7, 2026

Inputs: the damaged files of tools/competitor_scoreboard.py (`inputs`). The entry runs twice per file: as shipped,
and with every pull off ("raw": un-pulled position). Only the BLIND rows (after the IMU failure) are replaced, and
only on drone recordings (rotor drag on); rows before the failure stay the shipped ones, so checkpoints and dropouts
cannot change. Scored with tools/score_track3.py (SE(3) alignment fitted on the rows before the failure, as the
scorer does). Raw rows go to runs/blind_smoother/<cmd>.json; the results page copies them to results/track3/.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402
from score_track3 import score, umeyama  # noqa: E402

INPUTS = ROOT / "runs" / "scoreboard" / "inputs"
WORK = ROOT / "runs" / "blind_smoother"
DEV = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
       "euroc:V2_01_easy", "euroc:V2_02_medium"] + [f"tumvi:room{i}" for i in range(1, 5)]
HELD = ["euroc:MH_04_difficult", "euroc:MH_05_difficult", "euroc:V1_03_difficult", "euroc:V2_03_difficult",
        "tumvi:room5", "tumvi:room6"]
OFF = ["--pos-tau", "0", "--pos-tau-nonet", "0", "--drag-pos-tau", "0", "--zp-tau-start", "0"]
SETS = {"s42": "s42_accel_first", "s1": "s1_accel_first", "s7": "s7_accel_first", "s2026": "s2026_accel_first"}
KEYS = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m"]


def run(d, est, extra):
    if not est.exists():
        p = subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *extra],
                           check=True, capture_output=True, text=True)
        est.with_suffix(".err").write_text(p.stderr)
    return est


def load_case(spec, s):
    d = INPUTS / SETS[s] / spec.replace(":", "_")
    tag = f"{s}_{spec.replace(':', '_')}"
    entry = run(d, WORK / f"{tag}_entry.txt", [])
    raw = run(d, WORK / f"{tag}_raw.txt", OFF)
    drone = "rotor-drag aiding on" in entry.with_suffix(".err").read_text()
    truth = json.loads((d / "truth.json").read_text())
    imu = np.loadtxt(d / "imu0" / "data.csv", delimiter=",", comments="#", skiprows=1)
    k = len(imu)                                     # first all-zero row of the dead tail = the failure
    while k > 0 and not np.any(imu[k - 1, 1:7]):
        k -= 1
    return d, tag, np.loadtxt(entry, comments="#"), np.loadtxt(raw, comments="#"), drone, truth, k


def aligned(est, gt_path, truth):
    """Scorer's frame: SE(3) fitted on the rows before the failure. Returns (R, t, rel time, gt at est rows)."""
    gt = np.loadtxt(gt_path, delimiter=",", comments="#")
    te, pe = est[:, 0], est[:, 1:4]
    tg = gt[:, 0] * 1e-9
    inside = (te >= tg[0]) & (te <= tg[-1])
    pgi = np.full_like(pe, np.nan)
    pgi[inside] = np.stack([np.interp(te[inside], tg, gt[:, 1 + i]) for i in range(3)], 1)
    rel = te - truth["t0_ns"] * 1e-9
    mission = inside & (rel < truth["t_fail_s"])
    R, t = umeyama(pe[mission], pgi[mission])
    return R, t, rel, pgi, inside


# ---------------------------------------------------------------------------------------------- the smoother
def kf_rts(t, z_v, s_v, centre, s_c, q_a, rate=2.5, t_end=None):
    """Per-axis Kalman filter + RTS smoother on a uniform grid at `rate` Hz, state (position, velocity).
    Measurements: velocity z_v (one value per grid step, NaN = none) with std s_v, and a weak prior "position =
    centre" with std s_c at every step. Process noise: white acceleration with spectral density q_a.
    Returns smoothed positions on the grid (n, 3)."""
    n = len(t)
    dt = 1.0 / rate
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q_a * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    out = np.zeros((n, 3))
    for ax in range(3):
        xs = np.zeros((n, 2)); Ps = np.zeros((n, 2, 2)); xp_ = np.zeros((n, 2)); Pp_ = np.zeros((n, 2, 2))
        x = np.array([centre[ax], 0.0]); P = np.diag([s_c[ax] ** 2, 1.0])
        for i in range(n):
            if i > 0:
                x = F @ x; P = F @ P @ F.T + Q
            xp_[i], Pp_[i] = x, P
            for H, z, r in ((np.array([0.0, 1.0]), z_v[i, ax], s_v[i]), (np.array([1.0, 0.0]), centre[ax], s_c[ax])):
                if not np.isfinite(z):
                    continue
                S = H @ P @ H + r * r
                K = P @ H / S
                x = x + K * (z - H @ x)
                P = P - np.outer(K, H @ P)
            xs[i], Ps[i] = x, P
        for i in range(n - 2, -1, -1):
            C = Ps[i] @ F.T @ np.linalg.inv(Pp_[i + 1])
            xs[i] = xs[i] + C @ (xs[i + 1] - xp_[i + 1])
        out[:, ax] = xs[:, 0]
    return out


def smoother_blind(entry, raw, k, cfg):
    """Blind rows from a whole-flight smoother. Velocity measurements: the un-pulled run's velocity (our network /
    filter velocity integrated, no pull), averaged per grid step; none in the last `cut` seconds before the failure
    (rotor drag is unreliable while landing). Centre and spread: from our own shipped trajectory before the failure.
    The smoothed position at the failure is expressed in the frame of the shipped rows by matching the smoother to
    the shipped rows over the mission (least-squares offset), then held (or coasted with the shipped blind motion)."""
    te = raw[:, 0]
    rate, cut, s_v, k_c, q_a, mode = cfg["rate"], cfg["cut"], cfg["s_v"], cfg["k_c"], cfg["q_a"], cfg["mode"]
    tf = te[k - 1]
    g = np.arange(te[0], tf + 1e-9, 1.0 / rate)
    pr = np.stack([np.interp(g, te[:k], raw[:k, 1 + i]) for i in range(3)], 1)
    z = np.full_like(pr, np.nan)
    z[1:] = (pr[1:] - pr[:-1]) * rate
    z[g > tf - cut] = np.nan
    ps = entry[:k, 1:4]                               # shipped rows before the failure
    c = ps.mean(0)
    sc = np.maximum(k_c * ps.std(0), 0.3)
    xs = kf_rts(g, z, np.full(len(g), s_v), c, sc, q_a, rate)
    # express in the shipped frame: offset so the smoother matches the shipped rows on average over the mission
    shp = np.stack([np.interp(g, te[:k], ps[:, i]) for i in range(3)], 1)
    xs = xs + (shp - xs).mean(0) if cfg.get("match", 1) else xs
    p_fail = xs[-1]
    out = entry.copy()
    if mode == "hold":
        out[k:, 1:4] = p_fail
    else:                                            # coast: the shipped blind motion, started from p_fail
        out[k:, 1:4] = p_fail + (entry[k:, 1:4] - entry[k - 1, 1:4])
    return out


SETTINGS = {"entry": None}
for rate in (2.5,):
    for cut in (0.0, 3.0, 6.0):
        for s_v in (0.3, 1.0):
            for k_c in (1.0, 2.0):
                for mode in ("hold", "coast"):
                    SETTINGS[f"rts r{rate} cut{cut:g} sv{s_v:g} kc{k_c:g} {mode}"] = dict(
                        rate=rate, cut=cut, s_v=s_v, k_c=k_c, q_a=1.0, mode=mode)


def evaluate(spec, s, name):
    d, tag, entry, raw, drone, truth, k = load_case(spec, s)
    est_path = WORK / f"{tag}_entry.txt"
    cfg = SETTINGS[name]
    if cfg is not None and drone and k < len(entry):
        est = smoother_blind(entry, raw, k, cfg)
        est_path = WORK / f"{tag}_{abs(hash(name)) % 10 ** 8}.txt"
        np.savetxt(est_path, est, fmt="%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    r = score(str(est_path), str(source(spec)[1]), truth, None)
    return {"data": spec, "set": s, "setting": name, "drone": drone, **{x: r.get(x) for x in KEYS}}


def diagnose(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    figdir = WORK / "plots"
    figdir.mkdir(parents=True, exist_ok=True)
    for spec in a.data:
        for s in a.seeds:
            d, tag, entry, raw, drone, truth, k = load_case(spec, s)
            gt_path = source(spec)[1]
            R, t, rel, pgi, inside = aligned(entry, gt_path, truth)
            A = lambda p: p @ R.T + t                           # noqa: E731  (into the scorer's frame)
            blind = np.arange(k, len(entry))
            blind = blind[inside[blind]]
            g_b = pgi[blind]
            # a raw (un-pulled) position expressed in the shipped frame: offset fitted over the mission
            off = (entry[:k, 1:4] - raw[:k, 1:4]).mean(0)
            cand = {
                "shipped blind rows": A(entry[blind, 1:4]),
                "hold last zero-phase row": np.repeat(A(entry[k - 1, 1:4])[None], len(blind), 0),
                "hold raw (un-pulled) at failure": np.repeat(A(raw[k - 1, 1:4] + off)[None], len(blind), 0),
                "oracle: hold true position at failure": np.repeat(pgi[k - 1][None], len(blind), 0),
                "oracle: best fixed point (blind mean)": np.repeat(g_b.mean(0)[None], len(blind), 0),
            }
            res = {"data": spec, "set": s, "drone": drone,
                   "true motion during blind (m)": float(np.linalg.norm(g_b[-1] - g_b[0])),
                   "error at failure, last zero-phase row (m)": float(np.linalg.norm(A(entry[k - 1, 1:4]) - pgi[k - 1])),
                   "error at failure, causal blind row (m)": float(np.linalg.norm(A(entry[k, 1:4]) - pgi[k]))}
            for n, p in cand.items():
                res[n] = float(np.linalg.norm(p - g_b, axis=1).mean())
            rows.append(res)
            print(json.dumps({x: (round(v, 2) if isinstance(v, float) else v) for x, v in res.items()}), flush=True)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
            for j, (lo, title) in enumerate(((0, "whole flight"), (max(0, k - 3000), "last 15 s + blind"))):
                sl = slice(lo, len(entry))
                ax[j].plot(pgi[sl, 0], pgi[sl, 1], "k-", lw=1, label="ground truth")
                pm = A(entry[lo:k, 1:4])
                ax[j].plot(pm[:, 0], pm[:, 1], "-", color="tab:blue", lw=0.8, label="shipped (zero-phase) rows")
                pr_ = A(raw[lo:k, 1:4] + off)
                ax[j].plot(pr_[:, 0], pr_[:, 1], "-", color="tab:orange", lw=0.6, alpha=0.7, label="un-pulled run")
                pb = A(entry[blind, 1:4])
                ax[j].plot(pb[:, 0], pb[:, 1], "-", color="tab:red", lw=2, label="shipped blind rows")
                ax[j].plot(*pgi[k - 1, :2], "k*", ms=12, label="true position at failure")
                ax[j].set_title(f"{spec} {s}: {title}")
                ax[j].axis("equal")
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figdir / f"{tag}.png", dpi=80)
            plt.close(fig)
    (WORK / "diagnose.json").write_text(json.dumps(rows, indent=1))


def table(rows, names, specs, seeds):
    print("| setting | " + " | ".join(f"{s} blind" for s in seeds) + " | mean blind | mean ckpt |")
    print("|---|" + "---:|" * (len(seeds) + 2))
    for n in names:
        vals = [np.mean([r["blind_mean_m"] for r in rows if r["setting"] == n and r["set"] == s and r["data"] in specs])
                for s in seeds]
        ck = np.mean([r["checkpoints_m"] for r in rows if r["setting"] == n and r["data"] in specs])
        print(f"| {n} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {np.mean(vals):.3f} | {ck:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "sweep", "check"])
    ap.add_argument("--setting")
    ap.add_argument("--seeds", nargs="+", default=["s42", "s1", "s7"])
    ap.add_argument("--data", nargs="+", default=["euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:MH_04_difficult",
                                                  "euroc:MH_05_difficult"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "diagnose":
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(lambda j: load_case(*j), [(sp, s) for sp in a.data for s in a.seeds]))
        diagnose(a)
        return
    seeds = ["s42", "s1"] if a.cmd == "sweep" else ["s42", "s1", "s7", "s2026"]
    specs = DEV if a.cmd == "sweep" else DEV + HELD
    names = list(SETTINGS) if a.cmd == "sweep" else ["entry", a.setting]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: load_case(*j), [(sp, s) for sp in specs for s in seeds]))
        rows = list(ex.map(lambda j: evaluate(*j), [(sp, s, n) for sp in specs for s in seeds for n in names]))
    (WORK / f"{a.cmd}.json").write_text(json.dumps(rows, indent=0))
    for part, sp in (("dev", DEV), ("held-out", HELD), ("all", DEV + HELD), ("MH_04 + MH_05", HELD[:2]),
                     ("drone dev", [x for x in DEV if "euroc" in x]),
                     ("Vicon", [x for x in DEV + HELD if ":V" in x]), ("TUM-VI", [x for x in DEV + HELD if "tumvi" in x])):
        if not any(r["data"] in sp for r in rows):
            continue
        print(f"\n### {part}\n")
        table(rows, names, sp, seeds)


if __name__ == "__main__" and sys.argv[1:2] != ["split"]:
    main()


# ---------------------------------------------------------------------------------------------- split diagnosis
def takeoff_index(imu_path, n):
    """First row where rotor vibration starts (median 1 s RMS of the accelerometer first difference > 1 m/s^2;
    VIO-Edge's drone test, D-055): before it the drone is carried or sits with motors off."""
    x = np.loadtxt(imu_path, delimiter=",", comments="#", skiprows=1)[:n]
    acc = x[:, 1:4]                                    # accel-first inputs
    d = np.linalg.norm(np.diff(acc, axis=0), axis=1)
    w = 200
    rms = np.sqrt(np.convolve(d ** 2, np.ones(w) / w, mode="same"))
    on = np.flatnonzero(rms > 1.0)
    return int(on[0]) if len(on) else 0


def split(a):
    rows = []
    NODRAG = OFF + ["--drag-k1", "0", "--drag-k2", "0"]
    for spec in a.data:
        for s in a.seeds:
            d, tag, entry, raw, drone, truth, k = load_case(spec, s)
            nod = np.loadtxt(run(d, WORK / f"{tag}_rawnodrag.txt", NODRAG), comments="#")
            R, t, rel, pgi, inside = aligned(entry, source(spec)[1], truth)
            A = lambda p: p @ R.T + t                   # noqa: E731
            te = entry[:, 0]
            res = {"data": spec, "set": s}
            for tau in (5, 15, 30, 60):
                j = int(np.searchsorted(te, te[k - 1] - tau))
                for nm, src in (("drag", raw), ("no drag", nod)):
                    late = pgi[j] + A(src[k - 1, 1:4]) - A(src[j, 1:4])          # true start, our displacement
                    res[f"late velocity {nm}, last {tau} s"] = float(np.linalg.norm(late - pgi[k - 1]))
                acc_ = A(entry[j, 1:4]) + pgi[k - 1] - pgi[j]                         # our start, true displacement
                res[f"accumulated, before last {tau} s"] = float(np.linalg.norm(acc_ - pgi[k - 1]))
            ko = takeoff_index(d / "imu0" / "data.csv", k)
            blind = np.arange(k, len(entry)); blind = blind[inside[blind]]
            res["hold take-off position (blind mean)"] = float(np.linalg.norm(A(entry[ko, 1:4]) - pgi[blind], axis=1).mean())
            res["shipped blind"] = float(np.linalg.norm(A(entry[blind, 1:4]) - pgi[blind], axis=1).mean())
            res["take-off time s"] = float(te[ko] - te[0])
            rows.append(res)
            print(json.dumps({x: (round(v, 2) if isinstance(v, float) else v) for x, v in res.items()}), flush=True)
    (WORK / "split.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "split":
    sys.argv.pop(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["s42", "s1"])
    ap.add_argument("--data", nargs="+", default=["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium",
                                                  "euroc:V1_01_easy", "euroc:V1_02_medium", "euroc:V2_01_easy",
                                                  "euroc:V2_02_medium"])
    split(ap.parse_args())
    sys.exit(0)
