"""Experiment: a better position for the blind window on the drone mount (an offline estimate of the failure state).

    python tools/exp_failure_state.py dump                 # run the entry once per file and record its internals
    python tools/exp_failure_state.py sweep                # development recordings, seed 42, every setting
    python tools/exp_failure_state.py check --setting S    # all 17 at seeds 42, 1, 7, 2026 (held-out checked once)
    python tools/exp_failure_state.py tables               # markdown from results/track3/failure_state.json

The shipped entry writes rows before the IMU failure from the offline zero-phase pass (drone mount) and the blind rows
from the causal filter. Here only the blind rows change; every row before the failure is the entry's own. The entry
is run in-process with its methods wrapped (nothing in src/ changes) to record, per row, the causal position, the
un-pulled position and the filter velocity, and every network velocity (with its standard deviation) and rotor-drag
velocity reading. Settings:

  * `anchor W, hold|coast`: position at the failure = the entry's (smoothed) row W seconds before the failure plus the
    filter's own un-pulled displacement since; then held, or continued with the causal blind motion.
  * `rts W, hold|coast`: velocity over the last W seconds re-estimated by a per-axis Kalman filter + RTS smoother
    whose measurements are the network velocity (its own std) and the rotor-drag velocity (std 0.5 m/s), with the
    filter's acceleration (velocity change) as input; the failure position is the smoothed row W seconds before plus
    the integral of that velocity. `rtsnet`: the same with the network velocity only (no drag readings).
"""
import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402

INPUTS = ROOT / "runs" / "scoreboard" / "inputs"
WORK = ROOT / "runs" / "failure_state"
RESULT = ROOT / "results" / "track3" / "failure_state.json"
DEV = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
       "euroc:V2_01_easy", "euroc:V2_02_medium"] + [f"tumvi:room{i}" for i in range(1, 5)]
HELD = ["euroc:MH_04_difficult", "euroc:MH_05_difficult", "euroc:V1_03_difficult", "euroc:V2_03_difficult",
        "tumvi:room5", "tumvi:room6"]
SEEDS = ["s42_accel_first", "s1_accel_first", "s7_accel_first", "s2026_accel_first"]
KEYS = [("ate_rmse_m", "ATE"), ("checkpoints_m", "checkpoints"), ("blind_mean_m", "blind 5 s"),
        ("dropout_m", "dropouts"), ("position_drift_1s_m", "drift 1 s")]
GROUPS = {"Machine Hall": lambda s: ":MH" in s, "Vicon": lambda s: ":V" in s, "TUM-VI": lambda s: "tumvi" in s}


def dump_one(job):
    """Run the shipped entry on one file in-process, recording per-row internals of the main pass."""
    spec, s = job
    import main as M
    d = INPUTS / s / spec.replace(":", "_")
    out = WORK / f"{s}_{spec.replace(':', '_')}.txt"
    pk = WORK / f"{s}_{spec.replace(':', '_')}.pkl"
    if pk.exists():
        return str(pk)
    rec = {}

    def slot(pf):
        return rec.setdefault(id(pf), {"rows": [], "net": [], "drag": []})

    orig_pose, orig_drag, orig_learned = M.PathFinder.pose, M.PathFinder.drag_update, M.PathFinder.learned_update

    def pose(self):
        slot(self)["rows"].append((list(self.p), list(self.p_raw), list(self.v), tuple(self.q), self.drag_on))
        return orig_pose(self)

    def drag_update(self, f, dt, still):
        orig_drag(self, f, dt, still)
        if self.f_drag is not None and not still and self.good_time > dt:
            fd = self.f_drag
            m = [(fd[0] * e[0] + fd[1] * e[1] + fd[2] * e[2]) / kd
                 for e, kd in ((M.DRAG_E1, self.a.drag_k1), (M.DRAG_E2, self.a.drag_k2))]
            r = slot(self)
            r["drag"].append((len(r["rows"]), tuple(self.q), m))

    def learned_update(self):
        net = self.net
        if net is not None and not hasattr(net, "_wrapped"):
            pred = net.predict

            def predict(ring, i, _pf=self):
                d_, sd_, v_, sv_ = pred(ring, i)
                v2, sv2 = (M.rot_cov(_pf.q, v_, sv_) if _pf.body else (list(v_), list(sv_)))
                k = math.exp(-0.5 * _pf.yaw_var)
                r = slot(_pf)
                r["net"].append((len(r["rows"]), [v2[0] * k, v2[1] * k, v2[2]], list(sv2)))
                return d_, sd_, v_, sv_
            net.predict = predict
            net._wrapped = True
        return orig_learned(self)

    M.PathFinder.pose, M.PathFinder.drag_update, M.PathFinder.learned_update = pose, drag_update, learned_update
    try:
        a = M.args([str(d), "-o", str(out)])
        M.run(str(d), str(out), a)
    finally:
        M.PathFinder.pose, M.PathFinder.drag_update, M.PathFinder.learned_update = orig_pose, orig_drag, orig_learned
    main_pass = max(rec.values(), key=lambda r: len(r["rows"]))
    with open(pk, "wb") as f:
        pickle.dump(main_pass, f)
    return str(pk)


def load(spec, s):
    base = WORK / f"{s}_{spec.replace(':', '_')}"
    with open(str(base) + ".pkl", "rb") as f:
        r = pickle.load(f)
    e = np.loadtxt(str(base) + ".txt", comments="#")
    rows = [ln for ln in (INPUTS / s / spec.replace(":", "_") / "imu0" / "data.csv").read_text().splitlines()
            if ln and ln[0].isdigit()]
    k = len(rows)
    while k > 0 and all(float(x) == 0 for x in rows[k - 1].split(",")[1:]):
        k -= 1
    return r, e, k                                    # k = number of rows up to the last live one


def qrot(q, v):
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def rts_velocity(t, v_f, meas, sig_proc):
    """Per-axis Kalman filter + RTS smoother on velocity over rows [0, n): state v, input = the filter's own velocity
    change (its acceleration), process noise sig_proc^2 per second; meas = list of (row, value(3), std(3))."""
    n = len(t)
    by_row = {}
    for i, val, sd in meas:
        by_row.setdefault(i, []).append((val, sd))
    xs = np.empty((n, 3))
    for ax in range(3):
        x, P = v_f[0, ax], 0.5 ** 2
        xf, Pf, xp, Pp = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
        for i in range(n):
            if i:
                x = x + (v_f[i, ax] - v_f[i - 1, ax])
                P = P + sig_proc ** 2 * max(t[i] - t[i - 1], 0.0)
            xp[i], Pp[i] = x, P
            for val, sd in by_row.get(i, ()):
                if np.isfinite(val[ax]) and sd[ax] > 0:
                    K = P / (P + sd[ax] ** 2)
                    x, P = x + K * (val[ax] - x), (1 - K) * P
            xf[i], Pf[i] = x, P
        s = xf[-1]
        xs[-1, ax] = s
        for i in range(n - 2, -1, -1):
            C = Pf[i] / Pp[i + 1]
            s = xf[i] + C * (s - xp[i + 1])
            xs[i, ax] = s
    return xs


def apply(spec, s, setting):
    """Positions of every row for `setting`; rows before the failure are the entry's own."""
    r, e, k = load(spec, s)
    if setting == "entry":
        return e
    rows = r["rows"]
    if len(rows) != len(e) or not rows[-1][4] or k >= len(e) or k < 10:
        return e                                       # not the drone mount, or no blind window: unchanged
    kind, W, blind = setting.split()
    W = float(W)
    t = e[:, 0]
    tf = t[k - 1]
    i0 = int(np.searchsorted(t, tf - W))
    i0 = max(0, min(i0, k - 2))
    raw = np.array([x[1] for x in rows])
    pc = np.array([x[0] for x in rows])
    if kind == "anchor":
        pf = e[i0, 1:4] + raw[k - 1] - raw[i0]
    else:                                              # rts: re-estimated velocity over [i0, k)
        sl = slice(i0, k)
        vf = np.array([x[2] for x in rows[sl]])
        meas = [(i - i0, v, sd) for i, v, sd in r["net"] if i0 <= i < k]
        for i, q, m in (r["drag"] if kind == "rts" else ()):
            if i0 <= i < k:
                import main as M
                vb = list(qrot((q[0], -q[1], -q[2], -q[3]), rows[i][2]))      # filter's body velocity
                for e_, mm in zip((M.DRAG_E1, M.DRAG_E2), m):
                    dd = mm - sum(vb[j] * e_[j] for j in range(3))
                    for j in range(3):
                        vb[j] += dd * e_[j]
                meas.append((i - i0, list(qrot(q, vb)), [float(os.environ.get("DRAG_SD", "0.5"))] * 3))
        vs = rts_velocity(t[sl], vf, meas, float(os.environ.get("RTS_Q", "0.5")))
        dtv = np.diff(t[sl])
        pf = e[i0, 1:4] + np.sum(0.5 * (vs[1:] + vs[:-1]) * dtv[:, None], axis=0)
    out = e.copy()
    if blind == "hold":
        out[k:, 1:4] = pf
    else:                                              # coast: continue the causal blind motion from pf
        out[k:, 1:4] = pf + (pc[k:] - pc[k - 1])
    return out


SETTINGS = ["entry"] + [f"{kind} {w} {b}" for kind in ("anchor", "rts", "rtsnet") for w in (5, 10, 20)
                        for b in ("hold", "coast")]


def evaluate(job):
    spec, s, setting = job
    d = INPUTS / s / spec.replace(":", "_")
    truth = json.loads((d / "truth.json").read_text())
    if setting == "still":
        tin = [int(ln.split(",", 1)[0]) for ln in (d / "imu0" / "data.csv").read_text().splitlines() if ln and ln[0].isdigit()]
        e = np.c_[np.array(tin) * 1e-9, np.zeros((len(tin), 6)), np.ones(len(tin))]
    else:
        e = apply(spec, s, setting)
    f = WORK / f"eval_{os.getpid()}.txt"
    np.savetxt(f, e, fmt="%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    sc = score(str(f), str(source(spec)[1]), truth, None)
    return {"data": spec, "set": s, "setting": setting, **{k: sc.get(k) for k, _ in KEYS}}


def run(jobs, phase):
    WORK.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(dump_one, sorted({(sp, s) for sp, s, _ in jobs})))
        rows = list(ex.map(evaluate, jobs))
    for r in rows:
        r["phase"] = phase
    old = json.loads(RESULT.read_text()) if RESULT.exists() else []
    RESULT.write_text(json.dumps([r for r in old if r["phase"] != phase] + rows, indent=0))


def tables():
    rows = json.loads(RESULT.read_text())
    out = []
    for phase, title in (("sweep", "Development sweep (11 development recordings, seed 42; mean metres)"),
                         ("check", "Check of the frozen setting (mean metres)")):
        rs = [r for r in rows if r["phase"] == phase]
        if not rs:
            continue
        labels = list(dict.fromkeys(r["setting"] for r in rs))
        sets = [("dev, seed 42", lambda r: r["data"] in DEV)] if phase == "sweep" else \
            [(f"{w}, seed {s[1:].split('_')[0]}", (lambda s, grp: lambda r: r["set"] == s and r["data"] in grp)(s, grp))
             for w, grp in (("held-out 6", HELD), ("all 17", DEV + HELD)) for s in SEEDS]
        if phase == "check":
            sets += [(f"{w}, mean of 4 seeds", (lambda grp: lambda r: r["data"] in grp)(grp))
                     for w, grp in (("held-out 6", HELD), ("all 17", DEV + HELD))]
        out += [f"\n### {title}\n", "| inputs | setting | " + " | ".join(n for _, n in KEYS) + " | "
                + " | ".join(f"{g} ckpt / blind / drop" for g in GROUPS) + " |",
                "|---|---|" + "---:|" * (len(KEYS) + len(GROUPS))]
        for lab, sel in sets:
            for name in labels:
                q = [r for r in rs if sel(r) and r["setting"] == name]
                if not q:
                    continue
                g = [" / ".join(f"{np.mean([r[k] for r in q if f(r['data'])]):.2f}"
                                for k in ("checkpoints_m", "blind_mean_m", "dropout_m")) for f in GROUPS.values()]
                out.append(f"| {lab} | {name} | " + " | ".join(f"{np.mean([r[k] for r in q]):.3f}" for k, _ in KEYS)
                           + " | " + " | ".join(g) + " |")
        if phase == "check":
            idx = {(r["set"], r["data"], r["setting"]): r for r in rs}
            fr = [x for x in labels if x not in ("entry", "still")]
            lose = [f"{s} {d} {k} {idx[(s, d, x)][k]:.2f} vs {idx[(s, d, 'still')][k]:.2f}"
                    for x in fr for (s, d, y) in idx if y == x for k, _ in KEYS
                    if idx[(s, d, x)][k] >= idx[(s, d, "still")][k] and idx[(s, d, "entry")][k] < idx[(s, d, "still")][k]]
            out.append("\nNewly not better than standing still: " + ("none" if not lose else "; ".join(lose)))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "sweep", "check", "tables"])
    ap.add_argument("--setting")
    a = ap.parse_args()
    if a.cmd == "dump":
        run([(sp, s, "entry") for s in SEEDS for sp in DEV + HELD], "dump")
    elif a.cmd == "sweep":
        run([(sp, "s42_accel_first", st) for st in SETTINGS + ["still"] for sp in DEV], "sweep")
    elif a.cmd == "check":
        run([(sp, s, st) for s in SEEDS for sp in DEV + HELD for st in ("entry", "still", a.setting)], "check")
    tables()


if __name__ == "__main__":
    main()
