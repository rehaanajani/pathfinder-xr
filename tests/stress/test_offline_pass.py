"""The drone's offline pass (zero-phase smoothing + take-off / landing loop closure) and its memory cap.

    PF_ENTRY=cpp pytest tests/stress/test_offline_pass.py    # the C++ port (make -C src first)
    pytest tests/stress/test_offline_pass.py                 # main.py
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
N = 4000


def write(path):
    """A drone-mount file: 3 s resting on the pad, then vibrating flight with slow sideways motion, then the IMU
    dies (all-zero rows) for the last 5 s."""
    s = math.sqrt(sum(c * c for c in DRONE_UP))
    up = [c / s for c in DRONE_UP]
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(N):
            t = T0 + i * 5_000_000
            if i >= N - 1000:
                f.write(f"{t}," + ",".join(["0.0"] * 6) + "\n")
                continue
            fly = i >= 600
            vib = 1.2 if fly else 0.02
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = [9.81 * c + vib * math.sin(i * (1.7 + k)) + (0.3 * math.sin(i / 400) if fly else 0.0)
                 for k, c in enumerate(up)]
            f.write(f"{t}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def run(tmp_path, name, *opts):
    out = tmp_path / f"{name}.txt"
    p = subprocess.run([*ENTRY, str(tmp_path / "in.csv"), "-o", str(out), *opts],
                       capture_output=True, text=True, check=True)
    rows = [l for l in out.read_text().splitlines() if l and l[0] != "#"]
    assert len(rows) == N
    for r in rows:
        v = [float(x) for x in r.split()]
        assert len(v) == 8 and all(math.isfinite(x) for x in v)
    return rows, p.stderr


def test_offline_pass_runs_on_the_drone_and_keeps_one_pose_per_row(tmp_path):
    write(tmp_path / "in.csv")
    smoothed, err = run(tmp_path, "default")
    causal, _ = run(tmp_path, "causal", "--zp-tau-start", "0")
    assert "rotor-drag aiding on" in err
    assert smoothed != causal                  # the offline pass changed the positions


def test_memory_cap_falls_back_to_the_causal_output(tmp_path):
    write(tmp_path / "in.csv")
    capped, err = run(tmp_path, "capped", "--zp-max-rows", "500")
    causal, _ = run(tmp_path, "causal", "--zp-tau-start", "0")
    assert "offline smoothing skipped" in err
    assert capped == causal
