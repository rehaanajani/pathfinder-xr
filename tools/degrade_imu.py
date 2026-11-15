"""Degrade a clean EuRoC/TUM-VI imu0/data.csv the way the Track 3 brief describes (our stand-in until the
organizer's script is available).

    python tools/degrade_imu.py <mav0 folder> <out folder> [--seed 0] [--death zeros|nan|frozen|missing]

Mission: the IMU degrades progressively up to the final checkpoint, then dies for the last ``--blind`` seconds.
  * noise: white noise whose std ramps linearly from (g0, a0) to (g1, a1) over the mission
  * bias drift: random-walk bias whose rate ramps up with the jamming
  * dropouts: ``--dropouts`` signal losses of ``--dropout-s`` seconds (default 3 x 2 s), as missing rows,
    zeros or frozen values (``--dropout-mode``), placed in the second half of the mission
  * death: the last ``--blind`` seconds are zeros / NaN / frozen at the last value, or missing rows
Writes <out>/imu0/data.csv (EuRoC header) and <out>/truth.json (the fault schedule) for scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
          "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", type=float, default=5.0)
    ap.add_argument("--dropouts", type=int, default=3)
    ap.add_argument("--dropout-s", type=float, default=2.0)
    ap.add_argument("--dropout-mode", default="missing", choices=["missing", "zeros", "frozen", "nan"])
    ap.add_argument("--death", default="zeros", choices=["zeros", "nan", "frozen", "missing"])
    ap.add_argument("--noise", nargs=4, type=float, default=[0.002, 0.03, 0.04, 0.6], metavar=("G0", "A0", "G1", "A1"))
    ap.add_argument("--bias-rate", nargs=2, type=float, default=[0.002, 0.03], metavar=("GYRO", "ACC"),
                    help="random-walk bias std per sqrt(s) at full jamming")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    src = Path(a.src)
    csv = src / "imu0" / "data.csv" if (src / "imu0").exists() else src / "mav0" / "imu0" / "data.csv"
    d = np.loadtxt(csv, delimiter=",", comments="#")
    t = d[:, 0].astype(np.int64)
    g, acc = d[:, 1:4].copy(), d[:, 4:7].copy()
    ts = (t - t[0]) * 1e-9
    T = ts[-1]
    t_fail = T - a.blind
    mission = ts < t_fail
    ramp = np.clip(ts / max(t_fail, 1e-9), 0, 1)[:, None]
    g0, a0, g1, a1 = a.noise
    g += rng.normal(size=g.shape) * (g0 + (g1 - g0) * ramp)
    acc += rng.normal(size=acc.shape) * (a0 + (a1 - a0) * ramp)
    dt = np.diff(ts, prepend=ts[0])
    bg = np.cumsum(rng.normal(size=g.shape) * a.bias_rate[0] * ramp * np.sqrt(dt)[:, None], 0)
    ba = np.cumsum(rng.normal(size=acc.shape) * a.bias_rate[1] * ramp * np.sqrt(dt)[:, None], 0)
    g += bg; acc += ba
    keep = np.ones(len(t), bool)
    drops = []
    lo, hi = 0.5 * t_fail, t_fail - a.dropout_s - 1.0
    for k in range(a.dropouts):
        for _ in range(100):
            s0 = float(rng.uniform(lo, hi))
            if all(abs(s0 - s) > a.dropout_s + 2 for s, _ in drops):
                break
        drops.append((s0, s0 + a.dropout_s))
    for s0, s1 in drops:
        m = (ts >= s0) & (ts < s1)
        if a.dropout_mode == "missing":
            keep &= ~m
        elif a.dropout_mode == "zeros":
            g[m] = 0; acc[m] = 0
        elif a.dropout_mode == "nan":
            g[m] = np.nan; acc[m] = np.nan
        else:
            i0 = np.argmax(m)
            g[m] = g[max(i0 - 1, 0)]; acc[m] = acc[max(i0 - 1, 0)]
    dead = ~mission
    if a.death == "missing":
        keep &= ~dead
    elif a.death == "zeros":
        g[dead] = 0; acc[dead] = 0
    elif a.death == "nan":
        g[dead] = np.nan; acc[dead] = np.nan
    else:
        i0 = np.argmax(dead)
        g[dead] = g[i0 - 1]; acc[dead] = acc[i0 - 1]
    out = Path(a.out)
    (out / "imu0").mkdir(parents=True, exist_ok=True)
    with open(out / "imu0" / "data.csv", "w") as f:
        f.write(HEADER + "\n")
        for i in np.flatnonzero(keep):
            vals = ",".join("nan" if not np.isfinite(x) else f"{x:.9g}" for x in (*g[i], *acc[i]))
            f.write(f"{t[i]},{vals}\n")
    json.dump({"t0_ns": int(t[0]), "t_fail_s": t_fail, "blind_s": a.blind, "dropouts_s": drops,
               "dropout_mode": a.dropout_mode, "death": a.death, "seed": a.seed, "source": str(csv)},
              open(out / "truth.json", "w"), indent=1)
    print(f"{out}: {keep.sum()} rows, mission {t_fail:.1f}s + blind {a.blind}s, dropouts {[(round(x,1),round(y,1)) for x,y in drops]}")


if __name__ == "__main__":
    main()
