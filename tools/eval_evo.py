"""Score the Track 3 entry with the real `evo` command-line tools, next to tools/score_track3.py.

    python tools/eval_evo.py --data MH_01_easy V1_01_easy ... --seeds 1 2 3 --tag eval [--jobs 4]

Per recording and seed: degrade with tools/degrade_imu.py, run src/pathfinder (C++ entry) and the stand-still
baseline (--v-max 0.0000001), then run
    evo_ape tum gt est -a        (SE(3) alignment)
    evo_ape tum gt est -as       (Sim(3) alignment)
    evo_rpe tum gt est -d 200 -u f   (translation RPE over 200 frames = 1 s at EuRoC's 200 Hz)
and tools/score_track3.score on the same files. Writes runs/track3/<tag>/results.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from score_track3 import score  # noqa: E402

DATA = Path(os.environ.get("PFX_DATA", ROOT / "local_data"))
BIN = str(ROOT / "src" / "pathfinder")
STILL = ["--v-max", "0.0000001"]
EVO_BIN = Path(sys.executable).parent


def gt_tum(name, out):
    """EuRoC state_groundtruth_estimate0 (ns, p, q wxyz) -> TUM (s, p, q xyzw)."""
    d = np.loadtxt(DATA / "euroc" / name / "mav0" / "state_groundtruth_estimate0" / "data.csv", delimiter=",", comments="#")
    rows = np.c_[d[:, 0] * 1e-9, d[:, 1:4], d[:, 5:8], d[:, 4]]
    np.savetxt(out, rows, fmt="%.9f")


def evo(cmd, *args):
    r = subprocess.run([str(EVO_BIN / cmd), "tum", *args, "--t_max_diff", "0.01", "--no_warnings"],
                       capture_output=True, text=True)
    m = re.search(r"rmse\s+([0-9.eE+-]+)", r.stdout)
    if not m:
        return float("nan"), (r.stdout + r.stderr)[-300:]
    return float(m.group(1)), ""


def run_one(job):
    name, seed, work = job
    d = work / f"{name}_s{seed}"
    gt = work / f"{name}_gt.txt"
    src = DATA / "euroc" / name / "mav0"
    subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d), "--seed", str(seed)],
                   check=True, capture_output=True)
    truth = json.load(open(d / "truth.json"))
    csv = d / "imu0" / "data.csv"
    n_in = sum(1 for ln in open(csv) if ln.strip() and not ln.startswith("#"))
    out = {"data": name, "seed": seed, "rows_in": n_in, "t_fail_s": truth["t_fail_s"]}
    for label, extra in (("ours", []), ("still", STILL)):
        est = d / f"est_{label}.txt"
        t0 = time.perf_counter()
        r = subprocess.run([BIN, str(d), "-o", str(est), *extra], capture_output=True, text=True)
        wall = time.perf_counter() - t0
        res = {"returncode": r.returncode, "wall_s": wall}
        if r.returncode:
            res["stderr"] = r.stderr[-300:]
            out[label] = res
            continue
        res["rows_out"] = sum(1 for ln in open(est) if ln.strip() and not ln.startswith("#"))
        res["evo_ape_se3"], e1 = evo("evo_ape", str(gt), str(est), "-a")
        res["evo_ape_sim3"], e2 = evo("evo_ape", str(gt), str(est), "-as")
        res["evo_rpe_1s"], e3 = evo("evo_rpe", str(gt), str(est), "-d", "200", "-u", "f")
        if e1 or e2 or e3:
            res["evo_err"] = e1 or e2 or e3
        sc = score(est, d / "gt_unused" if False else DATA / "euroc" / name / "mav0" / "state_groundtruth_estimate0" / "data.csv",
                   truth, csv)
        res["score_track3"] = sc
        out[label] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    work = ROOT / "runs" / "track3" / a.tag
    work.mkdir(parents=True, exist_ok=True)
    for name in a.data:
        gt_tum(name, work / f"{name}_gt.txt")
    jobs = [(n, s, work) for n in a.data for s in a.seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run_one, jobs))
    (work / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {work / 'results.json'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
