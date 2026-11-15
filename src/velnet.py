"""Forward pass of the displacement network with plain numpy (no PyTorch / ONNX at run time).

Weights come from tools/train_velocity.py (weights.npz, BatchNorm already folded into the convolutions).
Layout mirrors pathfinder/model.py: stem conv -> 4 residual blocks -> global average pool -> 2-layer head.
The C++ entry implements the same operations.
"""
import numpy as np


def conv1d(x, w, b, stride, pad):
    """x: (Cin, L); w: (Cout, Cin, K)."""
    cout, cin, k = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    lout = (x.shape[1] + 2 * pad - k) // stride + 1
    idx = np.arange(lout)[:, None] * stride + np.arange(k)[None]
    cols = xp[:, idx]
    return np.einsum("ilk,oik->ol", cols, w) + b[:, None]


class VelNet:
    def __init__(self, path):
        z = np.load(path)
        self.W = {k: z[k] for k in z.files}
        self.blocks = sorted({k.split(".")[0] for k in self.W if k[0] == "b" and k[1].isdigit()},
                             key=lambda s: int(s[1:]))

    def _cb(self, x, name, relu=True):
        y = conv1d(x, self.W[name + ".w"], self.W[name + ".b"], int(self.W[name + ".stride"]),
                   int(self.W[name + ".pad"]))
        return np.maximum(y, 0) if relu else y

    def __call__(self, win):
        """win: (6, 200) gravity-aligned [gyro, acc]. Returns displacement (3,) and std (3,)."""
        x = self._cb(np.asarray(win, np.float32), "stem")
        for b in self.blocks:
            y = self._cb(x, b + ".c1")
            y = self._cb(y, b + ".c2", relu=False)
            sk = self._cb(x, b + ".skip", relu=False) if (b + ".skip.w") in self.W else x
            x = np.maximum(y + sk, 0)
        z = x.mean(1)
        h = np.maximum(self.W["h1.w"] @ z + self.W["h1.b"], 0)
        out = self.W["h2.w"] @ h + self.W["h2.b"]
        return out[:3], np.exp(np.clip(out[3:], -7, 3))
