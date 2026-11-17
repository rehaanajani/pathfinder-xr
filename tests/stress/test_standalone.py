"""main.py runs on its own: the network weights are embedded in it.

    pytest tests/stress/test_standalone.py
"""
import base64
import math
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
T0 = 1403636579758555392


def write(path, n=3000):
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(n):
            g = (0.01 * math.sin(i / 7), 0.02 * math.cos(i / 11), 0.005 * math.sin(i / 3))
            a = (0.8 * math.sin(i / 60), 0.6 * math.cos(i / 45), 9.81 + 0.05 * math.sin(i / 5))
            f.write(f"{T0 + i * 5_000_000}," + ",".join(f"{v:.6f}" for v in (*g, *a)) + "\n")


def test_main_py_runs_alone_with_identical_output(tmp_path):
    """Copied into an empty folder, main.py gives exactly the output it gives next to the weight files."""
    write(tmp_path / "in.csv")
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(SRC / "main.py", alone / "main.py")
    subprocess.run([sys.executable, str(alone / "main.py"), str(tmp_path / "in.csv"), "-o", str(tmp_path / "a.txt")],
                   check=True, capture_output=True, cwd=alone)
    subprocess.run([sys.executable, str(SRC / "main.py"), str(tmp_path / "in.csv"), "-o", str(tmp_path / "b.txt"),
                    "--model", str(SRC / "velnet_weights.bin"), "--fallback-model", str(SRC / "velnet_weights_world.bin")],
                   check=True, capture_output=True)
    assert (tmp_path / "a.txt").read_text() == (tmp_path / "b.txt").read_text()


def test_embedded_weights_match_the_weight_files():
    """main.py's embedded copies are byte-identical to src/velnet_weights*.bin (which the C++ entry reads)."""
    sys.path.insert(0, str(SRC))
    import main
    for key, name in (("body", "velnet_weights.bin"), ("world", "velnet_weights_world.bin")):
        embedded = zlib.decompress(base64.b64decode(main.EMBEDDED_WEIGHTS[key]))
        assert embedded == (SRC / name).read_bytes(), f"re-run tools/embed_weights.py: {name} changed"
