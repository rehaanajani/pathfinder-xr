"""Independent scoring of the Track 3 entry with the real `evo` library (not tools/score_track3.py).

    python tests/stress/track3_eval.py --out runs/stress_track3 [--quick]

For each recording x seed x death mode: degrade (tools/degrade_imu.py), run src/main.py under /usr/bin/time -l
(peak RSS, wall time), then with evo: ATE (translation RMSE, SE(3) alignment fitted on the mission part only so the
blind 5 s is a genuine prediction), RPE over 1 s, mean/final error over the blind 5 s, error inside the dropouts.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.metrics import PoseRelation
from evo.tools import file_interface

ROOT = Path(__file__).resolve().parents[2]
EUROC = ROOT / "local_data" / "euroc"


def gt_to_tum(csv, out):
    rows = []
    for l in open(csv):
        if l.startswith("#") or not l.strip():
            continue
        p = l.split(",")
        t = int(p[0]) * 1e-9
        rows.append(f"{t:.9f} {p[1]} {p[2]} {p[3]} {p[5]} {p[6]} {p[7]} {p[4]}")   # qw first in EuRoC -> qw last
    Path(out).write_text("\n".join(rows) + "\n")


def run_main(d, est, extra=()):
    cmd = ["/usr/bin/time", "-l", sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *extra]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return p, (int(m.group(1)) if m else None), wall


def score(est_path, gt_tum, truth, n_input_rows, imu_seconds):
    ref = file_interface.read_tum_trajectory_file(str(gt_tum))
    est = file_interface.read_tum_trajectory_file(str(est_path))
    t_fail = truth["t0_ns"] * 1e-9 + truth["t_fail_s"]
    out = {"rows_out": int(len(est.timestamps)), "rows_in": n_input_rows}
    r, e = sync.associate_trajectories(ref, est, max_diff=0.01)
    if len(e.timestamps) < 10:
        return {**out, "error": "no overlap with ground truth"}
    mission = e.timestamps < t_fail
    r_m, e_m = r.__class__(r.positions_xyz[mission], r.orientations_quat_wxyz[mission], r.timestamps[mission]), \
        e.__class__(e.positions_xyz[mission], e.orientations_quat_wxyz[mission], e.timestamps[mission])
    R, t, s = e_m.align(r_m, correct_scale=False)                 # fitted on the mission only
    e.transform(lie.se3(R, t))
    err = np.linalg.norm(e.positions_xyz - r.positions_xyz, axis=1)
    out["ate_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
    out["ate_mission_rmse_m"] = float(np.sqrt(np.mean(err[mission] ** 2)))
    blind = ~mission
    if blind.any():
        out["blind_mean_m"], out["blind_final_m"] = float(err[blind].mean()), float(err[blind][-1])
    d_err = np.zeros(len(err), bool)
    for a, b in truth["dropouts_s"]:
        d_err |= (e.timestamps >= truth["t0_ns"] * 1e-9 + a) & (e.timestamps < truth["t0_ns"] * 1e-9 + b + 1.0)
    if d_err.any():
        out["dropout_mean_m"] = float(err[d_err].mean())
    rpe = metrics.RPE(PoseRelation.translation_part, 200, metrics.Unit.frames, 0.0, all_pairs=False)
    rpe.process_data((r, e))
    out["rpe_1s_rmse_m"] = float(rpe.get_statistic(metrics.StatisticsType.rmse))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/stress_track3")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args()
    work = ROOT / a.out
    work.mkdir(parents=True, exist_ok=True)
    seqs = ["MH_01_easy", "MH_03_medium", "V1_01_easy", "V1_03_difficult", "V2_02_medium"]
    jobs = [(s, "zeros", "missing") for s in seqs] + [("MH_01_easy", d, "missing") for d in ("nan", "frozen", "missing")] \
        + [("MH_01_easy", "zeros", dm) for dm in ("zeros", "frozen")]
    if a.quick:
        jobs = jobs[:2]
    rows = []
    for seq, death, dmode in jobs:
        gt = work / f"{seq}_gt.tum"
        mav = EUROC / seq / "mav0"
        if not gt.exists():
            gt_to_tum(mav / "state_groundtruth_estimate0" / "data.csv", gt)
        for seed in a.seeds:
            d = work / f"{seq}_{death}_{dmode}_s{seed}"
            subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(mav), str(d), "--seed", str(seed),
                            "--death", death, "--dropout-mode", dmode], check=True, capture_output=True)
            truth = json.load(open(d / "truth.json"))
            lines = [l for l in open(d / "imu0" / "data.csv") if l.strip() and not l.startswith("#")]
            n_rows = len(lines)
            ts = [int(l.split(",")[0]) for l in lines if l.split(",")[0].lstrip("-").isdigit()]
            imu_s = (max(ts) - min(ts)) * 1e-9
            extra = ("--extend-blind", "5") if death == "missing" else ()
            est = d / "estimated_trajectory.txt"
            p, rss, wall = run_main(d, est, extra)
            row = {"seq": seq, "death": death, "dropout": dmode, "seed": seed, "imu_s": imu_s,
                   "rc": p.returncode, "peak_rss_mib": None if rss is None else rss / 2 ** 20,
                   "ms_per_imu_s": 1000 * wall / imu_s}
            if p.returncode == 0:
                row.update(score(est, gt, truth, n_rows, imu_s))
                row["rows_ok"] = row["rows_out"] == n_rows + (1000 if extra else 0)
            else:
                row["stderr"] = p.stderr[-300:]
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "stderr"}, flush=True)
    (work / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
