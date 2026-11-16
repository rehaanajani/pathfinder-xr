"""Regression tests for the faults an independent review found (15 Nov 2026).

    PF_ENTRY=cpp pytest tests/stress/test_review_fixes.py    # the C++ entry (make -C src first)
    pytest tests/stress/test_review_fixes.py                 # the Python reference
"""
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = [str(ROOT / "src" / "pathfinder")] if os.environ.get("PF_ENTRY") == "cpp" \
    else [sys.executable, str(ROOT / "src" / "main.py")]
T0 = 1403636579758555392


def write(path, n, accel_zero_first=0):
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(n):
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = (0.0, 0.0, 0.0) if i < accel_zero_first else \
                (0.1 * math.sin(i / 13), 0.05 * math.cos(i / 9), 9.81 + 0.02 * math.sin(i / 5))
            f.write(f"{T0 + i * 5_000_000}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def run(inp, out, *extra):
    subprocess.run([*ENTRY, str(inp), "-o", str(out), *extra], check=True, capture_output=True)
    return [list(map(float, l.split())) for l in Path(out).read_text().splitlines() if l and l[0] != "#"]


def test_zero_acceleration_at_start_never_gives_nan(tmp_path):
    """First 1 s: accelerometer reads zero while the gyro moves. Gravity must come from later samples."""
    write(tmp_path / "in.csv", 2000, accel_zero_first=200)
    rows = run(tmp_path / "in.csv", tmp_path / "out.txt")
    assert len(rows) == 2000
    assert all(math.isfinite(x) for r in rows for x in r)
    assert all(abs(sum(x * x for x in r[4:]) - 1) < 1e-6 for r in rows)


@pytest.mark.parametrize("n", [199, 200, 201])
def test_extend_blind_at_the_start_up_boundary(tmp_path, n):
    """A file exactly as long as the start-up buffer (200 rows) must still get its 5 s of extra poses."""
    write(tmp_path / "in.csv", n)
    rows = run(tmp_path / "in.csv", tmp_path / "out.txt", "--extend-blind", "5")
    assert len(rows) == n + 1000


@pytest.mark.skipif(os.environ.get("PF_ENTRY") != "cpp" or sys.platform != "darwin",
                    reason="peak memory of the C++ entry, measured with macOS /usr/bin/time -l")
def test_zip_with_many_members_stays_small(tmp_path):
    """The zip directory is read one entry at a time: 100k extra files must not raise memory."""
    write(tmp_path / "imu.csv", 200)
    with zipfile.ZipFile(tmp_path / "big.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp_path / "imu.csv", "mav0/imu0/data.csv")
        for k in range(100_000):
            z.writestr(f"mav0/cam0/data/{k:06d}.png", b"")
    p = subprocess.run(["/usr/bin/time", "-l", *ENTRY, str(tmp_path / "big.zip"), "-o", str(tmp_path / "o.txt")],
                       capture_output=True, text=True, check=True)
    rss = int(next(l.split()[0] for l in p.stderr.splitlines() if "maximum resident set size" in l))
    assert rss < 4 * 2 ** 20, f"peak {rss / 2 ** 20:.1f} MiB"


def test_long_missing_gap_does_not_snap_to_the_start(tmp_path):
    """Rows missing for 3 s: the pull back to the start is exponential and split at the 1 s mark, so the position
    after the gap is not forced to exactly zero (it was with a linear step over the whole gap)."""
    path = tmp_path / "gap.csv"
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        t = T0
        for i in range(3000):
            if i == 2000:
                t += 3_000_000_000                       # 3 s of missing rows
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = (0.8 * math.sin(i / 60), 0.6 * math.cos(i / 45), 9.81 + 0.05 * math.sin(i / 5))
            f.write(f"{t}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")
            t += 5_000_000
    rows = run(path, tmp_path / "o.txt")
    assert len(rows) == 3000
    after = rows[2000]
    assert any(abs(c) > 0 for c in after[1:4]), "position snapped to the start point after the gap"
