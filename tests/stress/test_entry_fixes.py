"""Regression tests for entry faults found in the PR #12 review (16 Nov 2026).

    PF_ENTRY=cpp pytest tests/stress/test_entry_fixes.py    # the C++ entry (make -C src first)
    pytest tests/stress/test_entry_fixes.py                 # the Python reference
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = [str(ROOT / "src" / "pathfinder")] if os.environ.get("PF_ENTRY") == "cpp" \
    else [sys.executable, str(ROOT / "src" / "main.py")]
T0 = 1403636579758555392
HEADER = "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"


def run(tmp_path, lines, *extra):
    (tmp_path / "in.csv").write_text(HEADER + "".join(lines))
    p = subprocess.run([*ENTRY, str(tmp_path / "in.csv"), "-o", str(tmp_path / "o.txt"), *extra],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    rows = [l for l in (tmp_path / "o.txt").read_text().splitlines() if l and l[0] != "#"]
    return rows, p.stderr


def row(i, g, a, t=None):
    return f"{T0 + i * 5_000_000 if t is None else t}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n"


def test_sustained_start_up_acceleration_is_not_a_rotated_mount(tmp_path):
    # TUM-VI mount (z up) accelerating at 5 m/s^2 for the first second: the mean specific force is 26 deg from
    # the training mounts, but it is not gravity alone, so the body-frame network must stay.
    lines = []
    for i in range(1500):
        ax = 5.0 if i < 200 else 0.0
        g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
        lines.append(row(i, g, (ax + 0.05 * math.sin(i / 5), 0.05 * math.sin(i / 6), 9.81 + 0.05 * math.sin(i / 7))))
    rows, err = run(tmp_path, lines)
    assert len(rows) == 1500
    assert "using velnet_weights_world.bin" not in err


@pytest.mark.parametrize("bad_t", ["1e309", "1e19", "9223372036854775808", "-1e19"])
def test_timestamps_outside_int64_are_skipped_without_crashing(tmp_path, bad_t):
    lines = [row(i, (0.0, 0.0, 0.001 * i), (0.0, 0.0, 9.81)) for i in range(4)]
    lines[2] = row(2, (0.0, 0.0, 0.002), (0.0, 0.0, 9.81), t=bad_t)
    rows, _ = run(tmp_path, lines)
    assert len(rows) == 3
    assert all(abs(float(r.split()[0])) < 1e10 for r in rows)


def test_zero_zupt_window_does_not_crash(tmp_path):
    lines = [row(i, (0.0, 0.0, 0.001 * (i % 3)), (0.0, 0.0, 9.81)) for i in range(50)]
    rows, _ = run(tmp_path, lines, "--zupt-window", "0", "--zupt-gyro-std", "0.01")
    assert len(rows) == 50
