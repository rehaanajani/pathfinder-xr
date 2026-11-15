"""Degrade + run + score several recordings in one go (development harness).

    python tools/eval_track3.py --data tumvi:room1 tumvi:room2 euroc:MH_01_easy ... --seeds 1 2 [-- main.py args]

Each recording is degraded with tools/degrade_imu.py (seeded), run through src/main.py, and scored with
tools/score_track3.py. Prints one row per recording plus the mean, and writes runs/track3/<tag>.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Desktop" / "pathfinder-xr" / "local_data"
sys.path.insert(0, str(ROOT / "tools"))
from score_track3 import score  # noqa: E402


def source(spec):
    kind, name = spec.split(":", 1)
    if kind == "tumvi":
        m = DATA / "tumvi" / f"dataset-{name}_512_16" / "mav0"
        return m, m / "mocap0" / "data.csv"
    if kind == "monado":
        m = DATA / "monado" / name / "mav0"
        return m, m / "gt" / "data.csv"
    m = DATA / "euroc" / name / "mav0"
    return m, m / "state_groundtruth_estimate0" / "data.csv"


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--death", default="zeros")
    ap.add_argument("--dropout-mode", default="missing")
    a = ap.parse_args(argv)
    work = ROOT / "runs" / "track3" / a.tag
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.data:
        src, gt = source(spec)
        for seed in a.seeds:
            d = work / f"{spec.replace(':', '_')}_s{seed}"
            subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d), "--seed", str(seed),
                            "--death", a.death, "--dropout-mode", a.dropout_mode], check=True, capture_output=True)
            est = d / "estimated_trajectory.txt"
            r = subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *extra],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(spec, seed, "FAILED", r.stderr[-400:]); continue
            res = score(est, gt, json.load(open(d / "truth.json")), d / "imu0" / "data.csv")
            res.update(data=spec, seed=seed)
            rows.append(res)
    keys = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m", "rpe_1s_m"]
    print(f"{'recording':28s}" + "".join(f"{k[:-2]:>14s}" for k in keys) + "   rows-ok")
    for r in rows:
        print(f"{r['data'] + ' s' + str(r['seed']):28s}" + "".join(f"{r.get(k, float('nan')):14.2f}" for k in keys)
              + f"   {r.get('rows_match_input')}")
    if rows:
        print(f"{'MEAN':28s}" + "".join(f"{np.nanmean([r.get(k, np.nan) for r in rows]):14.2f}" for k in keys))
    (ROOT / "runs" / "track3" / f"{a.tag}.json").write_text(json.dumps({"args": extra, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
