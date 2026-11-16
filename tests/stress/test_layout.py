"""Input layout detection: column order (EuRoC gyro-first vs the organisers' documented accel-first), timestamp unit
and accelerometer unit, including the cases an independent review raised.

    PF_ENTRY=cpp pytest tests/stress/test_layout.py      # C++ (make -C src first)
    pytest tests/stress/test_layout.py                   # Python
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
G = 9.80665


def samples(n=1600, spin=0.0):
    out = []
    for i in range(n):
        g = (spin + 0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
        a = (0.1 * math.sin(i / 13), 0.05 * math.cos(i / 9), 9.81 + 0.02 * math.sin(i / 5))
        out.append((g, a))
    return out


def write(path, rows, order="gyro", t_unit="ns", g_units=False, rate=200, header=True):
    step_ns = int(round(1e9 / rate))
    with open(path, "w") as f:
        if header:
            f.write("timestamp,ax,ay,az,gx,gy,gz\n" if order == "accel" else "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i, (g, a) in enumerate(rows):
            t = T0 + i * step_ns
            ts = {"ns": str(t), "us": f"{t / 1e3:.3f}", "ms": f"{t / 1e6:.6f}", "s": f"{t // 10**9}.{t % 10**9:09d}"}[t_unit]
            aa = [c / G for c in a] if g_units else a
            vals = (*aa, *g) if order == "accel" else (*g, *aa)
            f.write(ts + "," + ",".join(f"{v:.9g}" for v in vals) + "\n")


def run(tmp_path, name, *extra):
    p = subprocess.run([*ENTRY, str(tmp_path / name), "-o", str(tmp_path / (name + ".txt")), *extra],
                       capture_output=True, text=True, check=True)
    rows = [list(map(float, l.split())) for l in (tmp_path / (name + ".txt")).read_text().splitlines() if l[0] != "#"]
    return rows, p.stderr


def close(a, b, tol=1e-5):
    return len(a) == len(b) and all(abs(x - y) <= tol for r, s in zip(a, b) for x, y in zip(r[1:4], s[1:4]))


def test_accel_first_gives_the_same_trajectory(tmp_path):
    s = samples()
    write(tmp_path / "g.csv", s, "gyro")
    write(tmp_path / "a.csv", s, "accel")
    g, _ = run(tmp_path, "g.csv")
    a, err = run(tmp_path, "a.csv")
    assert "accel columns first" in err
    assert close(g, a, 0.0)


@pytest.mark.parametrize("t_unit", ["s", "ms", "us"])
def test_other_timestamp_units_and_g(tmp_path, t_unit):
    s = samples()
    write(tmp_path / "g.csv", s, "gyro")
    write(tmp_path / "x.csv", s, "accel", t_unit=t_unit, g_units=True)
    g, _ = run(tmp_path, "g.csv")
    x, err = run(tmp_path, "x.csv")
    assert f"timestamps in {t_unit}" in err and "accel in g" in err
    assert close(g, x, 1e-5)


def test_fast_nanosecond_file_keeps_every_row(tmp_path):
    """2 kHz in ns: spacing alone would say microseconds and overflow; epoch-ns values must stay ns."""
    s = samples(4000)
    write(tmp_path / "fast.csv", s, "gyro", rate=2000)
    rows, err = run(tmp_path, "fast.csv")
    assert len(rows) == 4000 and "timestamps in" not in err


def test_spinning_gyro_is_not_taken_for_an_accelerometer_in_g(tmp_path):
    """Gyro magnitude ~1 rad/s at start must not look like 'acceleration in g'."""
    s = samples(spin=1.0)
    write(tmp_path / "spin.csv", s, "gyro")
    _, err = run(tmp_path, "spin.csv")
    assert "accel columns first" not in err and "accel in g" not in err


def test_forced_layout_flags(tmp_path):
    s = samples()
    write(tmp_path / "a.csv", s, "accel")
    write(tmp_path / "g.csv", s, "gyro")
    g, _ = run(tmp_path, "g.csv")
    a, _ = run(tmp_path, "a.csv", "--columns", "accel-first", "--ts-unit", "ns", "--accel-unit", "ms2")
    assert close(g, a, 0.0)


def test_forced_columns_still_detect_g(tmp_path):
    """--columns accel-first with an accelerometer in g: the g conversion must still happen."""
    s = samples()
    write(tmp_path / "g.csv", s, "gyro")
    write(tmp_path / "ag.csv", s, "accel", g_units=True)
    g, _ = run(tmp_path, "g.csv")
    a, err = run(tmp_path, "ag.csv", "--columns", "accel-first")
    assert "accel in g" in err and close(g, a, 1e-5)


def test_forced_g_unit_picks_the_columns(tmp_path):
    """--accel-unit g with a fast-spinning gyro (10 rad/s): the g-sized triplet is the accelerometer."""
    s = [((10.0 + g[0], g[1], g[2]), a) for g, a in samples()]
    write(tmp_path / "x.csv", s, "accel", g_units=True)
    _, err = run(tmp_path, "x.csv", "--accel-unit", "g")
    assert "accel columns first" in err


def test_jittered_1khz_relative_nanoseconds_stay_nanoseconds(tmp_path):
    """Device-uptime ns timestamps at ~1 kHz with jitter (median spacing just under 1e6 ns, as in the Monado VR
    recordings) must stay ns: a spacing threshold read them as microseconds."""
    s = samples(4000)
    with open(tmp_path / "j.csv", "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        t = 8324806068490
        for i, (g, a) in enumerate(s):
            t += 995200 if i % 4 else 1009600
            f.write(f"{t}," + ",".join(f"{v:.9g}" for v in (*g, *a)) + "\n")
    rows, err = run(tmp_path, "j.csv")
    assert "timestamps in" not in err
    assert abs(rows[0][0] - 8324806068490e-9) < 0.01
