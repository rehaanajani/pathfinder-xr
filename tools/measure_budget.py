"""Measure the C++ entry's own peak RAM and CPU time per second of IMU on degraded recordings.

    python tools/measure_budget.py --data euroc:MH_05_difficult tumvi:room6 [--seeds 1] [--repeat 3] \
        [--out results/track3/budget.json]

Each recording is degraded with tools/degrade_imu.py (seeded), then src/pathfinder runs on it `--repeat` times. The
numbers are the process's own (os.wait4), not the shell's or the compiler's: peak RSS (max over repeats) and CPU time
(user + system, median over repeats) per second of IMU. Needs Linux or macOS. Run it on an otherwise idle machine.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402


def imu_seconds(csv):
    t = [int(l.split(",", 1)[0]) for l in Path(csv).read_text().splitlines() if l and l[0].isdigit()]
    return (t[-1] - t[0]) * 1e-9, len(t)


def run_once(binary, inp, out):
    proc = subprocess.Popen([str(binary), str(inp), "-o", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, ru = os.wait4(proc.pid, 0)
    err = proc.stderr.read().decode(errors="replace")
    if os.waitstatus_to_exitcode(status) != 0:
        raise SystemExit(f"entry failed on {inp}: {err}")
    peak = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return peak, ru.ru_utime + ru.ru_stime, err


def main():
    if not hasattr(os, "wait4"):
        raise SystemExit("needs Linux or macOS (os.wait4)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "track3" / "budget.json"))
    a = ap.parse_args()
    binary = ROOT / "src" / "pathfinder"
    subprocess.run(["make", "-C", str(ROOT / "src"), "-s"], check=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.data:
            src, _ = source(spec)
            for seed in a.seeds:
                d = Path(tmp) / f"{spec.replace(':', '_')}_s{seed}"
                subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d),
                                "--seed", str(seed)], check=True, capture_output=True)
                secs, n = imu_seconds(d / "imu0" / "data.csv")
                peaks, cpus = [], []
                for _ in range(a.repeat):
                    p, c, _ = run_once(binary, d, Path(tmp) / "out.txt")
                    peaks.append(p)
                    cpus.append(c)
                rows.append({"recording": spec, "seed": seed, "imu_rows": n, "imu_seconds": round(secs, 3),
                             "peak_rss_mb": round(max(peaks) / 2 ** 20, 3),
                             "cpu_ms_per_s_of_imu": round(1000 * statistics.median(cpus) / secs, 3)})
                print(json.dumps(rows[-1]), flush=True)
    res = {"entry": "src/pathfinder (C++), default settings: AirIO network + mount guard",
           "machine": {"os": platform.platform(), "cpu": platform.processor() or platform.machine()},
           "budget": {"peak_rss_mb": 8, "cpu_ms_per_s_of_imu": 20},
           "worst": {"peak_rss_mb": max(r["peak_rss_mb"] for r in rows),
                     "cpu_ms_per_s_of_imu": max(r["cpu_ms_per_s_of_imu"] for r in rows)},
           "rows": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"worst: {res['worst']}  -> {out}")


if __name__ == "__main__":
    main()
