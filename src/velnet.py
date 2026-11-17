"""Forward pass of the learned velocity network with plain numpy (no PyTorch / ONNX at run time).

Weights come from tools/train_velocity.py as one little-endian binary file (BatchNorm already folded in):

    char[4] "PFVN" | i32 version | i32 window | i32 dec | f32 rate_hz | i32 n_in | f32 scale[n_in - 1]
    i32 n_conv | n_conv x { i32 cout, cin, k, stride, pad | f32 w[cout*cin*k] | f32 b[cout] }
    i32 n_fc   | n_fc   x { i32 out, in | f32 w[out*in] | f32 b[out] }

Network: conv1d + ReLU layers -> global average pool -> dense + ReLU layers -> 12 linear outputs:
displacement over the window (3, metres), its log-std (3), velocity at the end of the window (3, m/s), its
log-std (3). Input: (n_in, window): each of the `window` slots is the mean of `dec` consecutive samples at rate_hz
(so the window spans window * dec / rate_hz seconds) of 6 gravity-aligned IMU channels (bias-corrected gyro, specific
force minus gravity), each multiplied by its scale, plus the fraction of healthy samples in the slot; a slot with
no healthy sample is all zero.
Version 3 (AirIO-style input, arxiv 2501.15659): 10 input channels per slot in the IMU body frame: bias-corrected
gyro, specific force with gravity kept, the unit "up" direction in the body frame from the estimator's own
orientation, each scaled, plus the healthy fraction; the 12 outputs are then in the body frame at the window end.
The C++ entry reads the same file and does the same arithmetic.
"""
import struct

import numpy as np


def conv1d(x, w, b, stride, pad):
    """x: (Cin, L); w: (Cout, Cin, K) -> (Cout, Lout)."""
    cout, cin, k = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    lout = (x.shape[1] + 2 * pad - k) // stride + 1
    idx = np.arange(lout)[:, None] * stride + np.arange(k)[None]
    cols = xp[:, idx]                                           # (Cin, Lout, K)
    return np.tensordot(w, cols, axes=([1, 2], [0, 2])) + b[:, None]


def load_bin(path):
    buf = open(path, "rb").read()
    if buf[:4] != b"PFVN":
        raise ValueError(f"{path}: not a Path Finder velocity-network file")
    off = 4

    def i32(n=1):
        nonlocal off
        v = struct.unpack_from(f"<{n}i", buf, off)
        off += 4 * n
        return v if n > 1 else v[0]

    def f32(n):
        nonlocal off
        v = np.frombuffer(buf, "<f4", n, off).copy()
        off += 4 * n
        return v

    version, window, dec = i32(3)
    rate = float(f32(1)[0])
    n_in = i32()
    scale = f32(n_in - 1)
    convs, fcs = [], []
    for _ in range(i32()):
        cout, cin, k, stride, pad = i32(5)
        convs.append((f32(cout * cin * k).reshape(cout, cin, k), f32(cout), stride, pad))
    for _ in range(i32()):
        o, i = i32(2)
        fcs.append((f32(o * i).reshape(o, i), f32(o)))
    return {"version": version, "window": window, "dec": dec, "rate": rate, "n_in": n_in, "scale": scale,
            "convs": convs, "fcs": fcs}


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


class VelNet:
    def __init__(self, path):
        m = load_bin(path)
        self.window, self.dec, self.rate, self.scale = m["window"], m["dec"], m["rate"], m["scale"]
        self.version, self.n_in = m["version"], m["n_in"]
        self.convs, self.fcs = m["convs"], m["fcs"]
        self.n_params = sum(w.size + b.size for w, b, *_ in self.convs) + sum(w.size + b.size for w, b in self.fcs)
        self.x = np.zeros((m["n_in"], self.window), np.float32)
        # Same products as conv1d, packed once into a GEMM. Workspaces are reused so a forward does not allocate
        # a padded copy per layer. Checked bit-identical to conv1d on this network (float32).
        self._cbuf = []
        for w, b, stride, pad in self.convs:
            cout, cin, k = w.shape
            self._cbuf.append({
                "w": np.ascontiguousarray(w.reshape(cout, cin * k)),
                "b": np.ascontiguousarray(b.reshape(-1, 1)),
                "stride": int(stride), "pad": int(pad), "k": int(k), "cin": int(cin), "cout": int(cout),
                "ws": {},
            })

    def _conv_stack(self, x):
        """conv1d + ReLU for every layer. Same values as conv1d; buffers live for the life of the network."""
        for spec in self._cbuf:
            stride, pad, k, cin = spec["stride"], spec["pad"], spec["k"], spec["cin"]
            length = int(x.shape[1])
            ws = spec["ws"].get(length)
            if ws is None:
                lout = (length + 2 * pad - k) // stride + 1
                xp = np.zeros((cin, length + 2 * pad), np.float32)
                col = np.empty((cin * k, lout), np.float32)
                y = np.empty((spec["cout"], lout), np.float32)
                ws = (xp, col, y, lout)
                spec["ws"][length] = ws
            xp, col, y, lout = ws
            xp[:, pad:pad + length] = x
            colv = col.reshape(cin, k, lout)
            for ki in range(k):
                colv[:, ki, :] = xp[:, ki:ki + lout * stride:stride]
            np.matmul(spec["w"], col, out=y)
            y += spec["b"]
            np.maximum(y, 0, out=y)
            x = y
        return x

    def forward(self, x):
        """x: (n_in, window), already scaled. Returns the 12 raw outputs."""
        x = self._conv_stack(x)
        z = x.mean(1)
        for j, (w, b) in enumerate(self.fcs):
            z = w @ z + b
            if j < len(self.fcs) - 1:
                z = np.maximum(z, 0)
        return z

    def predict(self, ring, start):
        """ring: list of `window` slots (n_in-tuple or None), oldest slot at index `start`,
        or a float32 array (n_in, window) already in that same unscaled slot order.
        Returns displacement (3), its std (3), end velocity (3), its std (3) as Python floats."""
        x = self.x
        W = self.window
        if isinstance(ring, np.ndarray):
            if start:
                n = W - start
                x[:, :n] = ring[:, start:]
                x[:, n:] = ring[:, :start]
            else:
                x[:, :] = ring
        else:
            x[:] = 0
            for j in range(W):
                s = ring[(start + j) % W]
                if s is not None:
                    x[:, j] = s
        x[:-1] *= self.scale[:, None]
        o = self.forward(x)
        sd = np.exp(np.clip(o[3:6], -7, 3))
        sv = np.exp(np.clip(o[9:12], -7, 3))
        return o[0:3].tolist(), sd.tolist(), o[6:9].tolist(), sv.tolist()
