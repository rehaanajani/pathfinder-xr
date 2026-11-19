"""Head-to-head scoreboard: our entry and other teams' public Track 3 entries on the SAME degraded files.

    python tools/competitor_scoreboard.py inputs --script <organisers' degrade_imu.py> --python <python with pandas>
    python tools/competitor_scoreboard.py run    --competitors <folder of cloned repos> --cpy <competitor venv python> \
        [--entries ours vortexxx ...] [--sets s42_accel_first ...] [--jobs 6]
    python tools/competitor_scoreboard.py heap   --competitors <...> --cpy <...>     # tracemalloc peak, 5 recordings

`inputs` damages the 17 recordings (11 EuRoC + TUM-VI room1-6) with the organisers' script once per set: seed 42 with
accel-first columns (their documented order), seed 42 with the EuRoC file unchanged (gyro first), and seeds 1 and 7
accel-first (a copy of the script with only `SEED = ` changed). Every entry then reads exactly those files.

`run` runs each entry under `/usr/bin/time -l` (one process per recording, one thread: OMP/BLAS threads set to 1),
scores the output with tools/score_track3.py (same alignment and windows for everyone), and checks the contract (one
pose per input row, timestamps equal to the input's, unit quaternions, finite). Standing still (the start pose on
every row) is written here, not run.

`heap` measures the peak Python heap (tracemalloc) of each entry: the script is compiled and executed once untraced
(numpy, lazily imported library modules and caches load there), then executed again under tracemalloc; that second
execution, module-level code such as decoding embedded weights included, is what counts. `--protocol cold` traces the
first execution instead (lazy imports of numpy submodules and the standard library then count too). This is the organisers' "algorithm / heap level" accounting, applied
the same way to every entry. Competitor code is never copied into this repository; it runs from its own clone.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_organiser import make_input  # noqa: E402
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402

WORK = ROOT / "runs" / "scoreboard"
RECORDINGS = ([f"euroc:{n}" for n in ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
                                      "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium",
                                      "V2_03_difficult")] + [f"tumvi:room{i}" for i in range(1, 7)])
SETS = {"s42_accel_first": (42, "accel_first"), "s42_euroc": (42, "euroc"),
        "s1_accel_first": (1, "accel_first"), "s7_accel_first": (7, "accel_first")}

# name -> (repo folder under --competitors, argv template). {csv} input file, {dir} its folder, {out} output file.
ENTRIES = {
    "ours": (None, ["{ourpy}", str(ROOT / "src" / "main.py"), "{dir}", "-o", "{out}"]),
    "vortexxx": ("RaktimChandra_pathfinder", ["{cpy}", "src/main.py", "{csv}", "{out}"]),
    "zerodrift": ("Gem2005_VITChennai_ZeroDrift", ["{cpy}", "src/main.py", "--input", "{csv}", "--output", "{out}",
                                                   "--no-memtrace"]),
    "vioedge": ("rohithkrishnaak_Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation",
                ["{cpy}", "src/main.py", "{csv}", "-o", "{out}"]),
    "vortex": ("ArnavJain2005_Vortex", ["{cpy}", "main.py", "{csv}", "{out}"]),
    "ridhima": ("ridhima936_PathFinder-IMU-Trajectory", ["{cpy}", "src/main.py", "{csv}", "-o", "{out}"]),
    "trustnav": ("Am-berneal_TrustNav-GPS-Denied-Navigation", ["{cpy}", "src/main.py", "{csv}", "{out}"]),
    "trustnav_rows": ("Am-berneal_TrustNav-GPS-Denied-Navigation",
                      ["{cpy}", "src/main.py", "{csv}", "{out}", "--no-fill", "--blind", "0"]),
}
ENV1 = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1"}


def set_dir(name, spec):
    return WORK / "inputs" / name / spec.replace(":", "_")


def cmd_inputs(a):
    scripts = WORK / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    text = Path(a.script).read_text()
    assert re.search(r"^SEED = 42$", text, re.M), "expected the organisers' script with SEED = 42"
    for name, (seed, layout) in SETS.items():
        s = scripts / f"degrade_seed{seed}.py"
        s.write_text(re.sub(r"^SEED = 42$", f"SEED = {seed}", text, flags=re.M))
        for spec in RECORDINGS:
            src, _ = source(spec)
            d = set_dir(name, spec)
            if (d / "truth.json").exists():
                continue
            (d / "imu0").mkdir(parents=True, exist_ok=True)
            t_orig = make_input(src / "imu0" / "data.csv", d / "clean.csv", layout)
            subprocess.run([a.python, str(s), str(d / "clean.csv"), str(d / "imu0" / "data.csv"),
                            "--meta", str(d / "meta.json")], check=True, capture_output=True)
            (d / "clean.csv").unlink()
            t_deg = np.array([int(ln.split(",", 1)[0]) for ln in (d / "imu0" / "data.csv").read_text().splitlines()
                              if ln and ln[0].isdigit()])
            nom = float(np.median(np.diff(t_orig))) * 1e-9
            gaps = [((t_deg[i] - t_orig[0]) * 1e-9, (t_deg[i + 1] - t_orig[0]) * 1e-9)
                    for i in range(len(t_deg) - 1) if (t_deg[i + 1] - t_deg[i]) * 1e-9 > 1.5 * nom]
            dur = (t_orig[-1] - t_orig[0]) * 1e-9
            truth = {"t0_ns": int(t_orig[0]), "t_fail_s": dur - 5.0, "dropouts_s": gaps, "duration_s": dur,
                     "seed": seed, "layout": layout}
            (d / "truth.json").write_text(json.dumps(truth))
            print(name, spec, "rows", len(t_deg), flush=True)


def contract(inp, est):
    """One pose per input row, same timestamps (s = ns * 1e-9), 8 finite columns, unit quaternions."""
    tin = np.array([int(ln.split(",", 1)[0]) for ln in Path(inp).read_text().splitlines() if ln and ln[0].isdigit()])
    rows = [ln.split() for ln in Path(est).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    out = {"rows_in": len(tin), "rows_out": len(rows), "cols_ok": all(len(r) == 8 for r in rows)}
    if not out["cols_ok"] or not rows:
        out["ok"] = False
        return out
    e = np.array(rows, float)
    out["finite"] = bool(np.isfinite(e).all())
    out["unit_quat"] = bool(np.allclose(np.linalg.norm(e[:, 4:8], axis=1), 1, atol=1e-5))
    out["same_times"] = bool(len(e) == len(tin) and np.allclose(e[:, 0], tin * 1e-9, atol=1e-6, rtol=0))
    out["ok"] = bool(out["rows_out"] == out["rows_in"] and out["same_times"] and out["finite"] and out["unit_quat"])
    return out


def write_still(inp, est):
    tin = [int(ln.split(",", 1)[0]) for ln in Path(inp).read_text().splitlines() if ln and ln[0].isdigit()]
    with open(est, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t in tin:
            f.write(f"{t * 1e-9:.9f} 0 0 0 0 0 0 1\n")


def parse_time_l(stderr):
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
    usr = re.search(r"([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", stderr)
    return (int(rss.group(1)) if rss else None,
            (float(usr.group(2)) + float(usr.group(3))) if usr else None,
            float(usr.group(1)) if usr else None)


def run_one(job):
    entry, set_name, spec, a = job
    d = set_dir(set_name, spec)
    inp = d / "imu0" / "data.csv"
    truth = json.loads((d / "truth.json").read_text())
    od = WORK / "out" / set_name / entry
    od.mkdir(parents=True, exist_ok=True)
    est = od / f"{spec.replace(':', '_')}.txt"
    res = {"entry": entry, "set": set_name, "data": spec}
    if entry == "still":
        write_still(inp, est)
    else:
        repo, tmpl = ENTRIES[entry]
        cwd = str(Path(a.competitors) / repo) if repo else str(ROOT)
        argv = [x.format(csv=str(inp), dir=str(d), out=str(est), cpy=a.cpy, ourpy=sys.executable) for x in tmpl]
        if est.exists():
            est.unlink()
        t0 = time.time()
        p = subprocess.run(["/usr/bin/time", "-l", *argv], cwd=cwd, capture_output=True, text=True,
                           env={**os.environ, **ENV1}, timeout=a.timeout)
        res["wall_s"] = time.time() - t0
        rss, cpu, real = parse_time_l(p.stderr)
        res.update(returncode=p.returncode, max_rss_bytes=rss, cpu_s=cpu,
                   cpu_ms_per_imu_s=(1e3 * cpu / truth["duration_s"]) if cpu else None)
        if p.returncode != 0 or not est.exists():
            res["error"] = p.stderr[-600:]
            return res
    try:
        res["contract"] = contract(inp, est)
        gt = source(spec)[1]
        res["score"] = score(str(est), str(gt), truth, None)
    except Exception as ex:                                           # noqa: BLE001
        res["error"] = f"scoring failed: {type(ex).__name__}: {ex}"
    return res


def cmd_run(a):
    entries = a.entries or ["still", *ENTRIES]
    sets = a.sets or list(SETS)
    jobs = [(e, s, spec, a) for s in sets for e in entries for spec in RECORDINGS]
    out = WORK / f"results_{a.tag}.json"
    done = {}
    if out.exists() and not a.fresh:
        for r in json.loads(out.read_text()):
            done[(r["entry"], r["set"], r["data"])] = r
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {out.name})", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            done[(r["entry"], r["set"], r["data"])] = r
            s = r.get("score", {})
            print(f"[{i + 1}/{len(todo)}] {r['entry']:13s} {r['set']:16s} {r['data']:24s} "
                  f"ATE {s.get('ate_rmse_m', float('nan')):6.2f} blind {s.get('blind_mean_m', float('nan')):6.2f} "
                  f"rows {r.get('contract', {}).get('ok')} {('ERR ' + r['error'][-120:]) if 'error' in r else ''}",
                  flush=True)
            if (i + 1) % 20 == 0:
                out.write_text(json.dumps(list(done.values()), indent=1))
    out.write_text(json.dumps(list(done.values()), indent=1))
    print("wrote", out)


PROBE = r"""
import json, os, sys, time, tracemalloc
import numpy                                     # the runtime's own import: not counted
script, argv = sys.argv[1], sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.argv = [script, *argv]
with open(script, "rb") as f:
    code = compile(f.read(), script, "exec")     # compiling the source text is not the algorithm: not counted


def once():
    g = {"__name__": "__main__", "__file__": os.path.abspath(script), "__builtins__": __builtins__}
    try:
        exec(code, g)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    g.clear()


if os.environ.get("HEAP_PROTOCOL", "warm") == "warm":
    once()                                       # warm-up, untraced: lazy library imports and caches load here
import gc; gc.collect()
tracemalloc.start()
c0 = time.process_time()
once()                                           # traced: the whole script again, module-level code included
cpu = time.process_time() - c0
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
print("HEAPPROBE " + json.dumps({"heap_peak_bytes": peak, "cpu_traced_s": cpu}))
"""


def cmd_heap(a):
    specs = a.data or ["euroc:MH_04_difficult", "euroc:V1_03_difficult", "euroc:MH_05_difficult",
                       "euroc:V2_03_difficult", "tumvi:room6"]
    entries = a.entries or list(ENTRIES)
    rows = []
    for entry in entries:
        repo, tmpl = ENTRIES[entry]
        cwd = str(Path(a.competitors) / repo) if repo else str(ROOT)
        for spec in specs:
            d = set_dir("s42_accel_first", spec)
            est = WORK / "heap_out.txt"
            argv = [x.format(csv=str(d / "imu0" / "data.csv"), dir=str(d), out=str(est), cpy=a.cpy,
                             ourpy=sys.executable) for x in tmpl]
            py, script, rest = argv[0], argv[1], argv[2:]
            p = subprocess.run([py, "-c", PROBE, script, *rest], cwd=cwd, capture_output=True, text=True,
                               env={**os.environ, **ENV1, "HEAP_PROTOCOL": a.protocol}, timeout=a.timeout)
            m = re.search(r"HEAPPROBE (\{.*\})", p.stdout)
            r = {"entry": entry, "data": spec, "protocol": a.protocol, **(json.loads(m.group(1)) if m else {"error": p.stderr[-400:]})}
            rows.append(r)
            print(entry, spec, r, flush=True)
    (WORK / f"heap_{a.protocol}.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--script", required=True)
    p.add_argument("--python", default=sys.executable, help="interpreter with pandas for the organisers' script")
    for name in ("run", "heap"):
        p = sub.add_parser(name)
        p.add_argument("--competitors", required=True)
        p.add_argument("--cpy", required=True, help="python of the separate competitor virtualenv")
        p.add_argument("--entries", nargs="*")
        p.add_argument("--timeout", type=float, default=900)
        if name == "run":
            p.add_argument("--sets", nargs="*")
            p.add_argument("--jobs", type=int, default=6)
            p.add_argument("--tag", default="all")
            p.add_argument("--fresh", action="store_true")
        else:
            p.add_argument("--data", nargs="*")
            p.add_argument("--protocol", choices=["warm", "cold"], default="warm",
                           help="warm: trace a second execution (library lazy imports excluded); cold: trace the first")
    a = ap.parse_args()
    {"inputs": cmd_inputs, "run": cmd_run, "heap": cmd_heap}[a.cmd](a)


if __name__ == "__main__":
    main()
