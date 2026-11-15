"""Blind 5 s study: kill the IMU at random times and compare our dead reckoning with standing still.

    python tools/blind_study.py --data MH_01_easy V1_02_medium V2_03_difficult --kills 20 --seed 7 --tag blind

For each recording, `--kills` kill times are drawn uniformly in [20 s, T-10 s]. For each one the recording is degraded
with tools/degrade_imu.py (seed 1, `--blind T-t_kill`, so the jamming ramp ends at the kill and the IMU is zeros
afterwards), cut at t_kill+5 s, and run through src/pathfinder. The estimate is aligned to ground truth with SE(3)
fitted on the mission part only (before the kill), and the 5 s after it are scored:
    ours       src/pathfinder
    still      src/pathfinder --v-max 0.0000001 (the entry's own stand-still setting)
    hold       our own aligned position at the kill, held fixed for 5 s (stand still where we were)
Each is reported as absolute error (what the scorer sees; includes the drift already accumulated at the kill) and as
`rel_*`: error of the motion since the kill only (what dead reckoning itself adds; for `hold` it is the distance travelled).
Writes runs/track3/<tag>/blind.json.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from score_track3 import umeyama  # noqa: E402

DATA = Path(os.environ.get("PFX_DATA", ROOT / "local_data"))
BIN = str(ROOT / "src" / "pathfinder")


def one(job):
    name, i, tk, T, work = job
    src = DATA / "euroc" / name / "mav0"
    d = work / f"{name}_k{i:02d}"
    subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d), "--seed", "1",
                    "--blind", f"{T - tk:.6f}"], check=True, capture_output=True)
    truth = json.load(open(d / "truth.json"))
    csv = d / "imu0" / "data.csv"
    lines = open(csv).read().splitlines()
    head, rows = lines[0], [ln for ln in lines[1:] if ln.strip()]
    t0 = truth["t0_ns"]
    keep = [r for r in rows if (int(r.split(",")[0]) - t0) * 1e-9 <= tk + 5.0 + 0.05]
    cut = work / f"{name}_k{i:02d}_cut" / "imu0"
    cut.mkdir(parents=True, exist_ok=True)
    (cut / "data.csv").write_text("\n".join([head, *keep]) + "\n")
    gt = np.loadtxt(src / "state_groundtruth_estimate0" / "data.csv", delimiter=",", comments="#")
    tg, pg = gt[:, 0] * 1e-9, gt[:, 1:4]
    out = {"data": name, "kill_s": tk, "t_fail_s": truth["t_fail_s"]}
    for label, extra in (("ours", []), ("still", ["--v-max", "0.0000001"])):
        est = work / f"{name}_k{i:02d}_{label}.txt"
        r = subprocess.run([BIN, str(cut.parent), "-o", str(est), *extra], capture_output=True, text=True)
        if r.returncode:
            out[label] = {"error": r.stderr[-200:]}
            continue
        e = np.loadtxt(est, comments="#")
        te, pe = e[:, 0], e[:, 1:4]
        ok = (te >= tg[0]) & (te <= tg[-1])
        te, pe = te[ok], pe[ok]
        pgi = np.stack([np.interp(te, tg, pg[:, c]) for c in range(3)], 1)
        rel = te - t0 * 1e-9
        mission = rel < tk
        R, t = umeyama(pe[mission], pgi[mission])
        pa = pe @ R.T + t
        blind = ~mission
        err = np.linalg.norm(pa - pgi, axis=1)[blind]
        k0 = np.flatnonzero(mission)[-1]                      # last mission sample: motion since here is the blind part
        rel = np.linalg.norm((pa - pa[k0]) - (pgi - pgi[k0]), axis=1)[blind]
        out[label] = {"mean_m": float(err.mean()), "end_m": float(err[-1]),
                      "rel_mean_m": float(rel.mean()), "rel_end_m": float(rel[-1])}
        if label == "ours":
            hold = np.linalg.norm(pa[mission][-1] - pgi, axis=1)[blind]
            tr = np.linalg.norm(pgi - pgi[k0], axis=1)[blind]    # distance really travelled = error of standing still
            out["hold"] = {"mean_m": float(hold.mean()), "end_m": float(hold[-1]),
                           "rel_mean_m": float(tr.mean()), "rel_end_m": float(tr[-1])}
            k = np.argmin(abs(tg - (t0 * 1e-9 + tk)))
            out["gt_speed_mps"] = float(np.linalg.norm(pg[min(k + 100, len(pg) - 1)] - pg[max(k - 100, 0)]) / 1.0)
            out["gt_5s_travel_m"] = float(np.linalg.norm(pgi[blind][-1] - pgi[blind][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--kills", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="blind")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    work = ROOT / "runs" / "track3" / a.tag
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for n, name in enumerate(a.data):
        raw = np.loadtxt(DATA / "euroc" / name / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#", usecols=0)
        T = (raw[-1] - raw[0]) * 1e-9
        rng = np.random.default_rng(a.seed + n)
        for i, tk in enumerate(np.sort(rng.uniform(20.0, T - 10.0, a.kills))):
            jobs.append((name, i, float(tk), float(T), work))
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, jobs))
    (work / "blind.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {work / 'blind.json'} ({len(rows)} kills)")


if __name__ == "__main__":
    main()
