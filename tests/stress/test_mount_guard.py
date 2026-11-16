"""Mount guard: the body-frame network is replaced by the gravity-aligned one when start-up gravity points far from
every training mount (EuRoC, TUM-VI). See check_mount in src/main.py.

    PF_ENTRY=cpp pytest tests/stress/test_mount_guard.py      # C++ (make -C src first)
    pytest tests/stress/test_mount_guard.py                   # Python
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


def write(path, up, n=1200):
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(n):
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = [9.81 * c + 0.05 * math.sin(i / (5 + k)) for k, c in enumerate(up)]
            f.write(f"{T0 + i * 5_000_000}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def run(tmp_path, up, *extra):
    write(tmp_path / "in.csv", up)
    p = subprocess.run([*ENTRY, str(tmp_path / "in.csv"), "-o", str(tmp_path / "o.txt"), *extra],
                       capture_output=True, text=True, check=True)
    rows = [l for l in (tmp_path / "o.txt").read_text().splitlines() if l and l[0] != "#"]
    assert len(rows) == 1200
    return p.stderr


@pytest.mark.parametrize("up", [(0.9412, -0.0101, -0.3376), (0.0, 0.0, 1.0)], ids=["euroc", "tumvi"])
def test_training_mounts_keep_the_body_network(tmp_path, up):
    assert "IMU mount" not in run(tmp_path, up)


@pytest.mark.parametrize("up", [(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)], ids=["flipped", "y-up", "x-down"])
def test_rotated_mount_switches_to_the_gravity_aligned_network(tmp_path, up):
    assert "using velnet_weights_world.bin" in run(tmp_path, up)


def test_guard_can_be_turned_off(tmp_path):
    assert "IMU mount" not in run(tmp_path, (0.0, 0.0, -1.0), "--mount-guard-deg", "0")


@pytest.mark.parametrize("bad", ["missing", "truncated"])
def test_unusable_fallback_keeps_the_body_network(tmp_path, bad):
    """A missing or broken fallback file must not crash or drop rows: the body-frame network carries on."""
    fb = tmp_path / "fb.bin"
    if bad == "truncated":
        fb.write_bytes((ROOT / "src" / "velnet_weights_world.bin").read_bytes()[:500])
    err = run(tmp_path, (0.0, 0.0, -1.0), "--fallback-model", str(fb))
    assert "keeping the body-frame network" in err


def test_nan_threshold_is_rejected(tmp_path):
    write(tmp_path / "in.csv", (0.0, 0.0, 1.0))
    p = subprocess.run([*ENTRY, str(tmp_path / "in.csv"), "-o", str(tmp_path / "o.txt"), "--mount-guard-deg", "nan"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "finite" in p.stderr
