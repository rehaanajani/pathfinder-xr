"""Experiment: an offline smoother for position (Kalman filter + RTS) in place of the causal pull back to the start.

    python tools/exp_smoother.py sweep | check | tables

main.py runs with the pull off (`--pos-tau 0 --pos-tau-nonet 0`); its position change over each block of `--block`
seconds is the input. Per axis, the smoother's state is position p, a slowly varying velocity error b and the centre
of the workspace c:
    p[k+1] = p[k] + d[k] - b[k] dt + w_p,   b[k+1] = b[k] + w_b,   c[k+1] = c[k]
    observation every block: p - c = 0 with standard deviation sigma (horizontal / vertical): a soft bounded space.
The velocity-error noise follows the organisers' damage ramp: q_b(t) = q_b0 (1 + a sev(t)), sev = 0.02 + 0.98 (t/T)^1.7.
The smoothed correction (smoothed p minus the raw position) is interpolated back to every row.
Settings are chosen on the 11 development recordings (seed 42); the frozen one is checked once on the held-out six and
seeds 1, 7, 2026. Nothing in src/ changes.
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from competitor_scoreboard import write_still  # noqa: E402
from eval_track3 import source  # noqa: E402
from exp_zero_phase import DEV, HELD, INPUTS, KEYS, SEEDS, entry_run  # noqa: E402
from score_track3 import score  # noqa: E402

WORK = ROOT / "runs" / "smoother"
RESULT = ROOT / "results" / "track3" / "rts_smoother.json"


def smooth(t, p, sigma_h=3.0, sigma_v=1.0, qp=1e-3, qb0=1e-3, a=0.0, block=0.2, b0=0.3, gamma=1.7, ap=0.0, sigma_end=None):
    """t (s), p (n,3) raw position -> smoothed position at every row."""
    t0, t1 = t[0], t[-1]
    tb = np.arange(t0, t1 + block, block)
    tb[-1] = min(tb[-1], t1) if len(tb) > 1 else tb[-1]
    pr = np.stack([np.interp(tb, t, p[:, i]) for i in range(3)], 1)            # raw position at block edges
    d = np.diff(pr, axis=0)                                                    # displacement per block
    nb = len(tb)
    sev = 0.02 + 0.98 * np.clip((tb - t0) / max(t1 - t0, 1e-9), 0, 1) ** gamma
    qb = qb0 * (1 + a * sev)
    qpk = qp * (1 + ap * sev)                                                  # late displacements are trusted less
    shrink = 1.0 if sigma_end is None else 1 + (sigma_end / sigma_h - 1) * (sev - sev[0]) / max(sev[-1] - sev[0], 1e-9)
    ps = np.empty((nb, 3))
    for ax in range(3):
        sig = (sigma_v if ax == 2 else sigma_h) * shrink
        F = np.array([[1.0, -block, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        H = np.array([1.0, 0.0, -1.0])
        x = np.zeros(3)
        P = np.diag([1e-4, b0 ** 2, 100.0])
        xf = np.empty((nb, 3)); Pf = np.empty((nb, 3, 3)); xp = np.empty((nb, 3)); Pp = np.empty((nb, 3, 3))
        for k in range(nb):
            if k > 0:
                x = F @ x + np.array([d[k - 1, ax], 0.0, 0.0])
                Q = np.diag([qpk[k - 1] * block, qb[k - 1] * block, 0.0])
                P = F @ P @ F.T + Q
            xp[k], Pp[k] = x, P
            S = H @ P @ H + (sig[k] if np.ndim(sig) else sig) ** 2
            K = P @ H / S
            x = x - K * (H @ x)
            P = P - np.outer(K, H @ P)
            xf[k], Pf[k] = x, P
        xs = xf[-1].copy()
        out = np.empty(nb); out[-1] = xs[0]
        for k in range(nb - 2, -1, -1):
            C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
            xs = xf[k] + C @ (xs - xp[k + 1])
            out[k] = xs[0]
        ps[:, ax] = out
    corr = ps - pr
    return p + np.stack([np.interp(t, tb, corr[:, i]) for i in range(3)], 1)


GRID = {"sigma_h": (1.5, 3.0, 5.0), "qb0": (1e-4, 1e-3, 1e-2), "a": (0.0, 30.0), "qp": (1e-3, 1e-2)}


def setting_name(kw):
    return ", ".join(f"{k} {v:g}" for k, v in kw.items())


def evaluate(spec, s, kw):
    d = INPUTS / s / spec.replace(":", "_")
    truth = json.loads((d / "truth.json").read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    name = kw if isinstance(kw, str) else name_kw(kw)
    tag = abs(hash((spec, s, name)))
    if kw == "entry":
        est = entry_run(spec, s, off=False)
    elif kw == "still":
        est = WORK / f"still_{tag}.txt"
        write_still(d / "imu0" / "data.csv", est)
    else:
        kw = dict(kw)
        taper = kw.pop("taper", None)
        e = np.loadtxt(entry_run(spec, s, off=True), comments="#")
        e[:, 1:4] = smooth(e[:, 0], e[:, 1:4], **kw)
        if taper is not None:                    # hand back to the causal entry over the last `taper` s before the IMU dies
            c = np.loadtxt(entry_run(spec, s, off=False), comments="#")
            rows = [ln for ln in (d / "imu0" / "data.csv").read_text().splitlines() if ln and ln[0].isdigit()]
            k = len(rows)
            while k > 0 and all(float(x) == 0 for x in rows[k - 1].split(",")[1:]):
                k -= 1
            t_dead = e[k - 1, 0] if 0 < k else e[-1, 0]
            w = np.clip((t_dead - e[:, 0]) / taper, 0, 1)[:, None] if taper > 0 else (e[:, 0] < t_dead)[:, None] * 1.0
            e[:, 1:4] = w * e[:, 1:4] + (1 - w) * c[:, 1:4]
        est = WORK / f"{tag}.txt"
        np.savetxt(est, e, fmt="%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    r = score(str(est), str(source(spec)[1]), truth, None)
    return {"data": spec, "set": s, "setting": name, **{k: r.get(k) for k, _ in KEYS}}


def run_jobs(jobs, phase):
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: entry_run(j[0], j[1], True), {(sp, s) for sp, s, _ in jobs}))
        rows = list(ex.map(lambda j: evaluate(*j), jobs))
    for r in rows:
        r["phase"] = phase
    old = json.loads(RESULT.read_text()) if RESULT.exists() else []
    RESULT.write_text(json.dumps([r for r in old if r["phase"] != phase] + rows, indent=0))


def table(rows, sel, names):
    out = ["| setting | " + " | ".join(n for _, n in KEYS) + " | worst ratio to standing still (ATE) |",
           "|---|" + "---:|" * (len(KEYS) + 1)]
    still = {(r["set"], r["data"]): r for r in rows if r["setting"] == "still" and sel(r)}
    for name in names:
        rs = [r for r in rows if r["setting"] == name and sel(r)]
        if not rs:
            continue
        worst = max(r["ate_rmse_m"] / still[(r["set"], r["data"])]["ate_rmse_m"] for r in rs)
        out.append(f"| {name} | " + " | ".join(f"{np.mean([r[k] for r in rs]):.3f}" for k, _ in KEYS) + f" | {worst:.2f} |")
    return "\n".join(out)


def name_kw(kw):
    return setting_name({k: v for k, v in kw.items() if v is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "sweep2", "sweep3", "check", "tables"])
    ap.add_argument("--frozen", default=None, help="JSON of the frozen setting for `check`")
    a = ap.parse_args()
    if a.cmd == "sweep":
        kws = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        run_jobs([(sp, "s42_accel_first", kw) for kw in ["entry", "still", *kws] for sp in DEV], "sweep")
    elif a.cmd == "sweep2":
        kws = [dict(sigma_h=sh, qb0=1e-4, qp=1e-3, ap=ap, sigma_end=se)
               for sh in (3.0, 5.0) for ap in (0.0, 30.0, 100.0, 300.0) for se in (None, 2.0, 1.0, 0.5)]
        run_jobs([(sp, "s42_accel_first", kw) for kw in kws for sp in DEV], "sweep2")
    elif a.cmd == "sweep3":
        kws = [dict(sigma_h=sh, qb0=1e-4, qp=1e-3, ap=ap, taper=tp)
               for sh in (3.0, 5.0) for ap in (0.0, 100.0) for tp in (0.0, 10.0, 30.0, 60.0)]
        run_jobs([(sp, "s42_accel_first", kw) for kw in kws for sp in DEV], "sweep3")
    elif a.cmd == "check":
        kw = json.loads(a.frozen)
        run_jobs([(sp, s, k) for s in SEEDS for sp in DEV + HELD for k in ("entry", "still", kw)], "check")
    rows = json.loads(RESULT.read_text())
    sw = [r for r in rows if r["phase"].startswith("sweep")]
    names = list(dict.fromkeys(r["setting"] for r in sw))
    print("### Development sweep (seed 42, 11 development recordings; mean metres)\n")
    print(table(sw, lambda r: True, names))


if __name__ == "__main__":
    main()
