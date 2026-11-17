"""Velocity-network weights file I/O for the training tools.

Inference (load_bin, VelNet) lives in main.py, which also embeds the weights so it runs as a standalone script; this
module re-exports it and adds save_bin for tools/train_*.py.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import VelNet, load_bin  # noqa: E402,F401


def save_bin(path, window, dec, rate, scale, convs, fcs, version=2):
    with open(path, "wb") as f:
        f.write(b"PFVN" + struct.pack("<iii", version, window, dec) + struct.pack("<f", rate))
        f.write(struct.pack("<i", len(scale) + 1) + np.asarray(scale, "<f4").tobytes())
        f.write(struct.pack("<i", len(convs)))
        for w, b, stride, pad in convs:
            f.write(struct.pack("<5i", w.shape[0], w.shape[1], w.shape[2], stride, pad))
            f.write(np.asarray(w, "<f4").tobytes() + np.asarray(b, "<f4").tobytes())
        f.write(struct.pack("<i", len(fcs)))
        for w, b in fcs:
            f.write(struct.pack("<2i", *w.shape) + np.asarray(w, "<f4").tobytes() + np.asarray(b, "<f4").tobytes())
