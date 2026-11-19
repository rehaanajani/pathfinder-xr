"""Rotor-drag aiding turns on only for the drone mount (start-up gravity along the EuRoC thrust axis).

    PF_ENTRY=cpp pytest tests/stress/test_rotor_drag.py    # the C++ port (make -C src first)
    pytest tests/stress/test_rotor_drag.py                 # main.py
"""
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = [str(ROOT / "src" / "pathfinder")] if os.environ.get("PF_ENTRY") == "cpp" \
    else [sys.executable, str(ROOT / "src" / "main.py")]
T0 = 1403636579758555392
DRONE_UP = (0.9412, -0.0101, -0.3376)


def write(path, up, n=3000):
    s = math.sqrt(sum(c * c for c in up))
    up = [c / s for c in up]
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(n):
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = [9.81 * c + 0.05 * math.sin(i / (5 + k)) for k, c in enumerate(up)]
            f.write(f"{T0 + i * 5_000_000}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def run(tmp_path, up):
    write(tmp_path / "in.csv", up)
    p = subprocess.run([*ENTRY, str(tmp_path / "in.csv"), "-o", str(tmp_path / "out.txt")],
                       capture_output=True, text=True, check=True)
    rows = [l for l in (tmp_path / "out.txt").read_text().splitlines() if l and l[0] != "#"]
    assert len(rows) == 3000
    return p.stderr


def test_drag_on_for_the_drone_mount(tmp_path):
    assert "rotor-drag aiding on" in run(tmp_path, DRONE_UP)


def test_drag_off_for_an_upright_imu(tmp_path):
    assert "rotor-drag aiding on" not in run(tmp_path, (0.0, 0.0, 1.0))
