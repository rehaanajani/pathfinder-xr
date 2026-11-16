"""Build the submission zip in the layout of the brief (slide 18).

    python tools/make_bundle.py <input: folder with imu0/data.csv | .zip | csv> [--team CaliforniaBurrito] [--out dist]

Builds src/pathfinder, runs it once on <input> (measuring peak RAM and time), and writes <out>/<team>.zip:

    estimated_trajectory.txt   the trajectory for <input>
    metadata.json              runtime, peak RAM, rows, machine, versions
    src/                       main.cpp, Makefile, Dockerfile, main.py, velnet.py, velnet_weights.bin, README.md
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FILES = ["main.cpp", "Makefile", "Dockerfile", ".dockerignore", "main.py", "velnet.py", "velnet_weights.bin", "velnet_weights_world.bin",
         "README.md"]


def imu_csv(inp):
    p = Path(inp)
    if p.is_dir():
        for c in (p / "imu0" / "data.csv", p / "mav0" / "imu0" / "data.csv"):
            if c.exists():
                return c.read_bytes()
        raise SystemExit(f"no imu0/data.csv under {p}")
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        return z.read(next(n for n in z.namelist() if n.endswith("imu0/data.csv")))
    return p.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--team", default="CaliforniaBurrito")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--extend-blind", type=float, default=0.0,
                    help="only if the official input ends at the IMU failure instead of keeping invalid rows")
    ap.add_argument("--columns", choices=["auto", "gyro-first", "accel-first"], default="auto",
                    help="force the input column order if the organisers state it (default: detected)")
    ap.add_argument("--ts-unit", choices=["auto", "ns", "us", "ms", "s"], default="auto")
    ap.add_argument("--accel-unit", choices=["auto", "ms2", "g"], default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["make", "-C", str(SRC), "-s"], check=True)
    traj = out / "estimated_trajectory.txt"
    cmd = [str(SRC / "pathfinder"), a.input, "-o", str(traj)]
    if a.extend_blind > 0:
        cmd += ["--extend-blind", str(a.extend_blind)]
    forced = [f"--{k.replace('_', '-')}={v}" for k, v in (("columns", a.columns), ("ts_unit", a.ts_unit),
                                                          ("accel_unit", a.accel_unit)) if v != "auto"]
    cmd += forced
    # Measure the estimator process alone (wait4 gives that one child's own usage; RUSAGE_CHILDREN would also
    # count the compiler that make just ran).
    if not hasattr(os, "wait4"):
        raise SystemExit("make_bundle.py measures the entry with os.wait4, which needs Linux or macOS")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd)
    _, status, ru = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise SystemExit(f"the entry failed with exit code {proc.returncode}")
    # ru_maxrss: bytes on macOS, kilobytes on Linux
    peak = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    cpu = ru.ru_utime + ru.ru_stime

    data = imu_csv(a.input)
    rows = [l for l in data.decode(errors="replace").splitlines() if l.strip() and l.strip()[0].isdigit()]
    # Duration from the trajectory, which is always in seconds whatever the input's timestamp unit; the poses that
    # --extend-blind appends after the last input row are not part of the input's duration.
    ts = [float(l.split()[0]) for l in traj.read_text().splitlines() if l.strip() and not l.startswith("#")]
    poses = len(ts)
    ts_in = ts[:len(rows)] if len(ts) >= len(rows) else ts
    span = (ts_in[-1] - ts_in[0]) if len(ts_in) > 1 else 0.0

    meta = {
        "team": "California Burrito",
        "college": "SRM Institute of Science and Technology (SRMIST), Kattankulathur",
        "track": "Track 3: Path Finder, Trajectory Estimation Under Sensor Degradation",
        "entry": "src/main.cpp (C++17, single thread, offline); src/main.py is the reference implementation",
        "command": "src/pathfinder <input> -o estimated_trajectory.txt" + (
            f" --extend-blind {a.extend_blind}" if a.extend_blind > 0 else "") + "".join(" " + f for f in forced),
        "model": {"file": "src/velnet_weights.bin", "parameters": 42716, "input": "body frame (AirIO style), weights version 3",
                  "sha256": hashlib.sha256((SRC / "velnet_weights.bin").read_bytes()).hexdigest()},
        "input": {"imu_rows": len(rows), "duration_s": round(span, 3),
                  "sha256": hashlib.sha256(data).hexdigest()},
        "output": {"poses": poses},
        "runtime": {"wall_s": round(wall, 4), "cpu_s": round(cpu, 4),
                    "cpu_ms_per_s_of_imu": round(1000 * cpu / span, 3) if span > 0 else None,
                    "peak_rss_mb": round(peak / 2 ** 20, 2), "threads": 1},
        "machine": {"os": platform.platform(), "cpu": platform.processor() or platform.machine(),
                    "python": platform.python_version()},
        "repository": "https://github.com/rehaanajani/pathfinder-xr",
    }
    try:
        meta["commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    zpath = out / f"{a.team}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(traj, "estimated_trajectory.txt")
        z.write(out / "metadata.json", "metadata.json")
        for f in FILES:
            z.write(SRC / f, f"src/{f}")
    print(json.dumps(meta["runtime"]), file=sys.stderr)
    print(f"wrote {zpath} ({poses} poses for {len(rows)} IMU rows)", file=sys.stderr)
    if poses < len(rows):
        print("WARNING: fewer poses than IMU rows", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
