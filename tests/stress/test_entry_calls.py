"""The ways an evaluation harness might call the entry all give the same trajectory.

    PF_ENTRY=cpp pytest tests/stress/test_entry_calls.py    # the C++ port (make -C src first)
    pytest tests/stress/test_entry_calls.py                 # main.py
"""
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = os.environ.get("PF_ENTRY") == "cpp"
T0 = 1403636579758555392
N = 1500


def entry(cwd):
    if CPP:
        for f in ("pathfinder", "velnet_weights.bin", "velnet_weights_world.bin"):
            if (ROOT / "src" / f).exists():
                shutil.copy(ROOT / "src" / f, cwd / f)
        return ["./pathfinder"]
    return [sys.executable, str(ROOT / "src" / "main.py")]


def write_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(N):
            g = (0.02 * math.sin(i / 9), 0.01 * math.cos(i / 13), 0.03 * math.sin(i / 5))
            a = (0.2 * math.sin(i / 17), 0.1 * math.cos(i / 11), 9.81 + 0.05 * math.sin(i / 3))
            f.write(f"{T0 + i * 5_000_000}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def poses(path):
    return [l for l in path.read_text().splitlines() if l and l[0] != "#"]


def test_every_call_style_gives_the_same_trajectory(tmp_path):
    work = tmp_path / "work"
    write_csv(work / "imu0" / "data.csv")
    nested = tmp_path / "nested"
    write_csv(nested / "team" / "mav0" / "imu0" / "data.csv")
    with zipfile.ZipFile(tmp_path / "flat.zip", "w") as z:          # data.csv at the top of the zip
        z.write(work / "imu0" / "data.csv", "data.csv")
    cmd = entry(work)
    calls = {
        "no arguments (current folder)": ([], "estimated_trajectory.txt"),
        "the imu0 folder itself": (["imu0"], "estimated_trajectory.txt"),
        "output as a second argument": (["imu0/data.csv", "b.txt"], "b.txt"),
        "zip without an imu0 folder": ([str(tmp_path / "flat.zip"), "-o", "c.txt"], "c.txt"),
        "unpacked zip one level down": ([str(nested), "-o", "d.txt"], "d.txt"),
    }
    outs = {}
    for name, (args, out) in calls.items():
        subprocess.run([*cmd, *args], cwd=work, capture_output=True, text=True, check=True)
        outs[name] = poses(work / out)
        assert len(outs[name]) == N, name
        (work / out).unlink()
    first = next(iter(outs.values()))
    for name, rows in outs.items():
        assert rows == first, name
