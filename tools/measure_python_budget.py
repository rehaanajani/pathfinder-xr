"""Memory and CPU of main.py: peak heap of the algorithm (its data
structures, buffers and estimation logic, traced with tracemalloc; the interpreter and numpy's own import are not
counted), and CPU time per second of IMU, one thread.

    python tools/measure_python_budget.py --script <organisers' degrade_imu.py> --data euroc:MH_05_difficult ... \
        [--out results/track3/python_budget.json]

Each recording is damaged with the organisers' script (accel-first columns, their seed), then main.py runs in a fresh
process: once under tracemalloc for the heap peak, then three times untraced for CPU (best of three).
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_organiser import make_input  # noqa: E402
from eval_track3 import source  # noqa: E402

PROBE = r"""
import sys, time, tracemalloc, json, types
import numpy                                    # the runtime's own import: not part of the algorithm
sys.path.insert(0, sys.argv[1])
path = sys.argv[1] + "/main.py"
tracemalloc.start()
# Compile main.py from source, as `python main.py` does (scripts are never loaded from cached bytecode); standard-library
# modules it imports load as installed. Everything from here on is counted, embedded weights included.
entry = types.ModuleType("main")
entry.__file__ = path
sys.modules["main"] = entry
exec(compile(open(path).read(), path, "exec"), entry.__dict__)
loaded, import_peak = tracemalloc.get_traced_memory()
a = entry.args([sys.argv[2], "-o", sys.argv[3]])
tracemalloc.reset_peak()
entry.run(sys.argv[2], sys.argv[3], a)
run_peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
cpu = []
for _ in range(3):
    c0 = time.process_time()
    entry.run(sys.argv[2], sys.argv[3], a)
    cpu.append(time.process_time() - c0)
print(json.dumps({"module_bytes": loaded, "import_peak_bytes": import_peak, "run_peak_bytes": run_peak,
                  "cpu_s": min(cpu)}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "track3" / "python_budget.json"))
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.data:
            src, _ = source(spec)
            d = Path(tmp) / spec.replace(":", "_")
            (d / "imu0").mkdir(parents=True)
            t = make_input(src / "imu0" / "data.csv", d / "clean.csv", "accel_first")
            subprocess.run([sys.executable, a.script, str(d / "clean.csv"), str(d / "imu0" / "data.csv")],
                           check=True, capture_output=True)
            p = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "src"), str(d), str(d / "est.txt")],
                               check=True, capture_output=True, text=True)
            r = json.loads(p.stdout.strip().splitlines()[-1])
            secs = (t[-1] - t[0]) * 1e-9
            # Tracing starts before main.py is imported, so the run's peak already contains the module (with its
            # embedded network). The algorithm's own peak is that minus what the module holds.
            row = {"data": spec, "imu_seconds": round(float(secs), 1),
                   "heap_peak_algorithm_mib": round((r["run_peak_bytes"] - r["module_bytes"]) / 2 ** 20, 2),
                   "heap_peak_incl_module_mib": round(max(r["run_peak_bytes"], r["import_peak_bytes"]) / 2 ** 20, 2),
                   "import_peak_mib": round(r["import_peak_bytes"] / 2 ** 20, 2),
                   "cpu_ms_per_s": round(float(r["cpu_s"] / secs * 1000), 2)}
            rows.append(row)
            print(row, flush=True)
    worst = {k: max(r[k] for r in rows) for k in ("heap_peak_algorithm_mib", "heap_peak_incl_module_mib", "cpu_ms_per_s")}
    out = {"entry": "src/main.py (standalone, network weights embedded)", "python": platform.python_version(),
           "machine": platform.platform(), "processor": platform.processor() or platform.machine(),
           "method": "tracemalloc started after importing numpy and before importing main.py; "
                     "heap_peak_incl_module = the larger of the peak while loading main.py (compiled from source, as `python "
                     "main.py` does) and the peak while running (main.py and its embedded network included); "
                     "heap_peak_algorithm = that minus the memory main.py holds once imported; "
                     "CPU = process time, best of 3, one thread", "worst": worst, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("worst", worst, "->", a.out)


if __name__ == "__main__":
    main()
