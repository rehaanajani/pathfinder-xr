"""I/O contract of the Track 3 entry (docs/TRACK3_BRIEF.md): degraded EuRoC imu0/data.csv in,
TUM estimated_trajectory.txt out, exactly one pose per IMU row, seconds, unit quaternions,
<= 8 MB peak RAM, <= 20 ms of compute per 1 s of IMU, single thread.

Skipped until src/main.py exists. Command line (from src/main.py):
    python src/main.py <imu0/data.csv> -o <estimated_trajectory.txt>

Peak RAM is measured with `/usr/bin/time -l` (macOS: "maximum resident set size" in bytes).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "src" / "main.py"
# PF_ENTRY=cpp runs the same tests against the C++ build (make -C src) instead of main.py.
ENTRY = [str(ROOT / "src" / "pathfinder")] if os.environ.get("PF_ENTRY") == "cpp" else [sys.executable, str(MAIN)]
pytestmark = pytest.mark.skipif(not MAIN.exists(), reason="src/main.py not on this branch yet")

HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]," \
         "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
T0 = 1403636579758555392


def walk_rows(n=2000, rate=200, seed=0):
    """EuRoC-like IMU rows from a synthetic head walk (gravity included, body frame)."""
    sys.path.insert(0, str(ROOT))
    from pathfinder.synthetic import head_walk
    s = head_walk(n / rate, rate=float(rate), n_imus=1, seed=seed)
    ts = T0 + (np.arange(len(s.t)) * 1e9 / rate).astype(np.int64)
    return ts, s.gyro[0], s.acc[0], s


def write_csv(path, ts, g, a, header=HEADER, eol="\n", fmt="{:.6f}"):
    lines = [header.rstrip("\n") + eol] if header else []
    for t, gg, aa in zip(ts, g, a):
        lines.append(f"{int(t)}," + ",".join(fmt.format(v) for v in (*gg, *aa)) + eol)
    Path(path).write_text("".join(lines))


def run(csv, out, extra=()):
    """Returns (CompletedProcess, peak_rss_bytes or None, wall_s)."""
    import time
    cmd = ["/usr/bin/time", "-l", *ENTRY, str(csv), "-o", str(out), *extra] \
        if sys.platform == "darwin" else [*ENTRY, str(csv), "-o", str(out), *extra]
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1"})
    wall = time.perf_counter() - t
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return p, (int(m.group(1)) if m else None), wall


def read_traj(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    return np.array(rows, float) if rows else np.zeros((0, 8))


def check_format(traj, ts_ns):
    assert traj.shape == (len(ts_ns), 8), f"{traj.shape} vs {len(ts_ns)} input rows"
    assert np.isfinite(traj).all()
    assert np.allclose(traj[:, 0], np.asarray(ts_ns) * 1e-9, atol=1e-6, rtol=0), "timestamps must be seconds"
    q = traj[:, 4:8]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6), "quaternions must be unit (qx qy qz qw)"


@pytest.fixture()
def clean(tmp_path):
    ts, g, a, s = walk_rows()
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    return tmp_path, csv, ts, g, a


def test_clean_run_contract(clean):
    d, csv, ts, g, a = clean
    p, _, _ = run(csv, d / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    check_format(read_traj(d / "out.txt"), ts)


@pytest.mark.parametrize("name,kw", [
    ("no_header", dict(header="")),
    ("hash_header_crlf", dict(eol="\r\n")),
    ("plain_header_no_hash", dict(header=HEADER.lstrip("#"))),
    ("short_precision", dict(fmt="{:.2f}")),
    ("scientific", dict(fmt="{:.9e}")),
])
def test_header_and_format_variants(tmp_path, name, kw):
    ts, g, a, _ = walk_rows(1000)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a, **kw)
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    check_format(read_traj(tmp_path / "out.txt"), ts)


def test_trailing_blank_line_and_no_final_newline(tmp_path):
    ts, g, a, _ = walk_rows(500)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    csv.write_text(csv.read_text().rstrip("\n"))               # no final newline
    p, _, _ = run(csv, tmp_path / "o1.txt")
    assert p.returncode == 0
    check_format(read_traj(tmp_path / "o1.txt"), ts)
    csv.write_text(csv.read_text() + "\n\n")
    p, _, _ = run(csv, tmp_path / "o2.txt")
    assert p.returncode == 0
    check_format(read_traj(tmp_path / "o2.txt"), ts)


@pytest.mark.parametrize("kind", ["nan", "inf", "zeros", "frozen", "huge"])
def test_bad_values_still_one_pose_per_row(tmp_path, kind):
    ts, g, a, _ = walk_rows(2000)
    g, a = g.copy(), a.copy()
    sl = slice(800, 1200)                                      # 2 s
    if kind == "nan":
        g[sl] = np.nan; a[sl] = np.nan
    elif kind == "inf":
        g[sl] = np.inf; a[sl] = -np.inf
    elif kind == "zeros":
        g[sl] = 0; a[sl] = 0
    elif kind == "frozen":
        g[sl] = g[800]; a[sl] = a[800]
    else:
        g[sl] = 1e30; a[sl] = 1e30
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a, fmt="{}")
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    check_format(read_traj(tmp_path / "out.txt"), ts)


def test_nan_in_one_channel_only(tmp_path):
    ts, g, a, _ = walk_rows(1000)
    a = a.copy(); a[300:310, 2] = np.nan
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a, fmt="{}")
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0
    check_format(read_traj(tmp_path / "out.txt"), ts)


def test_two_second_gap_in_timestamps(tmp_path):
    """Rows missing (2 s signal loss): one pose per row that IS in the file, timestamps preserved."""
    ts, g, a, _ = walk_rows(2000)
    keep = np.r_[0:800, 1200:2000]
    csv = tmp_path / "data.csv"
    write_csv(csv, ts[keep], g[keep], a[keep])
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    traj = read_traj(tmp_path / "out.txt")
    check_format(traj, ts[keep])
    assert np.linalg.norm(traj[801, 1:4] - traj[800, 1:4]) < 10.0, "pose jumped across the gap"


def test_duplicate_timestamps(tmp_path):
    ts, g, a, _ = walk_rows(1000)
    ts = ts.copy(); ts[501] = ts[500]; ts[700:703] = ts[700]
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    check_format(read_traj(tmp_path / "out.txt"), ts)         # still one pose per row, timestamps as given


def test_out_of_order_timestamps(tmp_path):
    ts, g, a, _ = walk_rows(1000)
    order = np.arange(1000); order[300], order[301] = 301, 300
    csv = tmp_path / "data.csv"
    write_csv(csv, ts[order], g[order], a[order])
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    check_format(read_traj(tmp_path / "out.txt"), ts[order])


def test_file_ends_mid_row(tmp_path):
    ts, g, a, _ = walk_rows(1000)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    txt = csv.read_text()
    csv.write_text(txt[: len(txt) - 25])                       # last row cut
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0, p.stderr[-500:]
    traj = read_traj(tmp_path / "out.txt")
    assert len(ts) - 1 <= len(traj) <= len(ts)
    assert np.isfinite(traj).all()


def test_tiny_inputs(tmp_path):
    ts, g, a, _ = walk_rows(1000)
    for n in (1, 2, 3):
        csv = tmp_path / f"d{n}.csv"
        write_csv(csv, ts[:n], g[:n], a[:n])
        p, _, _ = run(csv, tmp_path / f"o{n}.txt")
        assert p.returncode == 0, (n, p.stderr[-300:])
        check_format(read_traj(tmp_path / f"o{n}.txt"), ts[:n])


def test_header_only_file_does_not_crash(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(HEADER)
    p, _, _ = run(csv, tmp_path / "o.txt")
    assert p.returncode == 0 or "Traceback" not in p.stderr


def test_nanosecond_timestamps_keep_full_precision(tmp_path):
    """1.4e18 does not fit a float64 exactly: seconds output must still be monotone to 1e-6 s."""
    ts, g, a, _ = walk_rows(1000)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    p, _, _ = run(csv, tmp_path / "out.txt")
    traj = read_traj(tmp_path / "out.txt")
    assert (np.diff(traj[:, 0]) > 0).all()


def test_unknown_gravity_rotated_input(tmp_path):
    """Brief: unknown gravity alignment. Whole IMU frame rotated by a fixed 90 deg about x."""
    ts, g, a, _ = walk_rows(1500)
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g @ Rx.T, a @ Rx.T)
    p, _, _ = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0
    traj = read_traj(tmp_path / "out.txt")
    check_format(traj, ts)
    assert np.abs(traj[:, 1:4]).max() < 200.0, "estimate flew away when gravity is not along +z"


def test_extend_blind_option(clean):
    d, csv, ts, g, a = clean
    p, _, _ = run(csv, d / "out.txt", extra=("--extend-blind", "5"))
    assert p.returncode == 0, p.stderr[-300:]
    traj = read_traj(d / "out.txt")
    assert len(traj) > len(ts)
    assert traj[-1, 0] - traj[len(ts) - 1, 0] == pytest.approx(5.0, abs=0.1)
    assert np.isfinite(traj).all()


# ---- budgets ----------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("PF_ENTRY") != "cpp",
                    reason="the 8 MB budget applies to the C++ entry; the Python reference alone is ~10 MB at start-up")
def test_peak_ram_under_8_mb(clean):
    d, csv, *_ = clean
    p, rss, _ = run(csv, d / "out.txt")
    if rss is None:
        pytest.skip("peak RSS only measured with /usr/bin/time -l on macOS")
    assert rss <= 8 * 1024 * 1024, f"peak RSS {rss / 2**20:.1f} MiB > 8 MiB (empty python is already ~10 MiB; needs the C++ port)"


def test_runtime_under_20ms_per_second_of_imu(tmp_path):
    ts, g, a, _ = walk_rows(200 * 120)                          # 120 s
    csv = tmp_path / "data.csv"
    write_csv(csv, ts, g, a)
    p, _, wall = run(csv, tmp_path / "out.txt")
    assert p.returncode == 0
    per_s = wall / 120
    assert per_s <= 0.020, f"{per_s * 1000:.1f} ms per second of IMU (whole process incl. startup and CSV parsing)"
