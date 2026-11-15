"""Train the small displacement network for the IMU-only Track 3 entry on EuRoC + TUM-VI.

    python tools/train_velocity.py --train euroc:MH_01_easy tumvi:room1 ... --val euroc:MH_04_difficult ...
                                   [--width 16 --epochs 30 --out models/vel16]

Each window: 1 s (200 samples at 200 Hz) of gyro + accelerometer rotated into a gravity-aligned frame with the
ground-truth orientation, then a random yaw (the network must not depend on heading) and a small tilt (our
runtime orientation has tilt errors). Target: the displacement over that second in the same frame. Training data is
degraded on the fly like the brief describes (rising noise, bias drift, frozen/zeroed stretches), so the network
reports more doubt on bad input. Loss: MSE + beta-NLL (Seitzer et al. 2022). Selection on the validation sequences.
Saves best.pt plus weights.npz (BatchNorm folded into the convolutions) for the numpy / C++ forward pass.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathfinder.model import WINDOW, DispNet, beta_nll, n_params  # noqa: E402

DATA = Path.home() / "Desktop" / "pathfinder-xr" / "local_data"
RATE = 200.0


def load(spec):
    """Gravity-aligned IMU (N,6) and ground-truth positions (N,3) on a uniform 200 Hz grid."""
    kind, name = spec.split(":", 1)
    if kind == "tumvi":
        m = DATA / "tumvi" / f"dataset-{name}_512_16" / "mav0"
        gt = np.loadtxt(m / "mocap0" / "data.csv", delimiter=",", comments="#")
    elif kind == "monado":
        m = DATA / "monado" / name / "mav0"
        gt = np.loadtxt(m / "gt" / "data.csv", delimiter=",", comments="#")
    else:
        m = DATA / "euroc" / name / "mav0"
        gt = np.loadtxt(m / "state_groundtruth_estimate0" / "data.csv", delimiter=",", comments="#")
    imu = np.loadtxt(m / "imu0" / "data.csv", delimiter=",", comments="#")
    ti, tg = imu[:, 0] * 1e-9, gt[:, 0] * 1e-9
    t0, t1 = max(ti[0], tg[0]) + 0.1, min(ti[-1], tg[-1]) - 0.1
    t = np.arange(t0, t1, 1 / RATE)
    g = np.stack([np.interp(t, ti, imu[:, 1 + i]) for i in range(3)], 1)
    a = np.stack([np.interp(t, ti, imu[:, 4 + i]) for i in range(3)], 1)
    p = np.stack([np.interp(t, tg, gt[:, 1 + i]) for i in range(3)], 1)
    rot = Slerp(tg, R.from_quat(gt[:, [5, 6, 7, 4]]))(t)
    up = rot.apply(a).mean(0)                                  # world frame may not be z-up: align to gravity
    to_z = R.align_vectors([[0, 0, 1.0]], [up / np.linalg.norm(up)])[0]
    rot = to_z * rot
    p = to_z.apply(p)
    Rm = rot.as_matrix()
    gw = np.einsum("nij,nj->ni", Rm, g)
    aw = np.einsum("nij,nj->ni", Rm, a)
    return np.concatenate([gw, aw], 1).astype(np.float32), p.astype(np.float32)


def rot_z(th):
    c, s = torch.cos(th), torch.sin(th)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def tilt(B, deg, dev):
    ax = torch.randn(B, 3, device=dev); ax[:, 2] = 0
    ax = ax / ax.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ang = torch.rand(B, 1, 1, device=dev) * math.radians(deg)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = ax[:, 1], -ax[:, 0], -ax[:, 1], ax[:, 0]
    return torch.eye(3, device=dev) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


def batch(imu, pos, starts, idx, dev, aug):
    s = starts[idx]
    ar = torch.arange(WINDOW, device=dev)
    win = imu[s[:, None] + ar[None]]                           # (B, W, 6)
    disp = pos[s + WINDOW] - pos[s]
    B = len(idx)
    if aug:
        Rz = rot_z(torch.rand(B, device=dev) * 2 * math.pi)
        T = Rz @ tilt(B, 3.0, dev)
        g, a = win[..., :3] @ T.transpose(1, 2), win[..., 3:] @ T.transpose(1, 2)
        disp = (Rz @ disp[..., None])[..., 0]
        sev = torch.rand(B, 1, 1, device=dev)                  # degradation level of this window
        g = g + torch.randn_like(g) * (0.002 + 0.05 * sev) + (torch.rand(B, 1, 3, device=dev) - 0.5) * 0.04 * sev
        a = a + torch.randn_like(a) * (0.02 + 0.6 * sev) + (torch.rand(B, 1, 3, device=dev) - 0.5) * 0.6 * sev
        # frozen or zeroed stretch (dropout) in some windows
        hit = (torch.rand(B, device=dev) < 0.15)[:, None, None].float()
        t0 = torch.randint(0, WINDOW // 2, (B, 1), device=dev)
        L = torch.randint(10, WINDOW // 2, (B, 1), device=dev)
        tt = torch.arange(WINDOW, device=dev)[None]
        m = ((tt >= t0) & (tt < t0 + L)).float()[..., None] * hit
        g, a = g * (1 - m), a * (1 - m)
        win = torch.cat([g, a], -1)
    return win.transpose(1, 2).contiguous(), disp


def stack(specs, stride):
    imus, poss, starts, off = [], [], [], 0
    for sp in specs:
        x, p = load(sp)
        imus.append(x); poss.append(p)
        starts.append(np.arange(off, off + len(x) - WINDOW - 1, stride))
        off += len(x)
    return (torch.from_numpy(np.concatenate(imus)), torch.from_numpy(np.concatenate(poss)),
            torch.from_numpy(np.concatenate(starts)))


def fold(net):
    """BatchNorm folded into the preceding conv -> plain arrays for numpy / C++."""
    net = net.eval()
    W = {}

    def conv_bn(conv, bn, name):
        w = conv.weight.detach().numpy().astype(np.float64)
        sc = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        W[name + ".w"] = (w * sc[:, None, None]).astype(np.float32)
        W[name + ".b"] = (bn.bias.detach().numpy() - bn.running_mean.detach().numpy() * sc).astype(np.float32)
        W[name + ".stride"] = np.array(conv.stride[0]); W[name + ".pad"] = np.array(conv.padding[0])
    conv_bn(net.stem[0], net.stem[1], "stem")
    for i, b in enumerate(net.body):
        conv_bn(b.c1, b.b1, f"b{i}.c1"); conv_bn(b.c2, b.b2, f"b{i}.c2")
        if not isinstance(b.skip, torch.nn.Identity):
            conv_bn(b.skip[0], b.skip[1], f"b{i}.skip")
    W["h1.w"], W["h1.b"] = net.head[0].weight.detach().numpy(), net.head[0].bias.detach().numpy()
    W["h2.w"], W["h2.b"] = net.head[2].weight.detach().numpy(), net.head[2].bias.detach().numpy()
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="models/vel16")
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    tr = [x.to(dev) for x in stack(a.train, a.stride)]
    va = [x.to(dev) for x in stack(a.val, 50)]
    net = DispNet(a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(tr[2]) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"params={n_params(net)} train_windows={len(tr[2])} val_windows={len(va[2])} device={dev}", flush=True)
    best, hist = 1e9, []
    for ep in range(a.epochs):
        net.train(); t0 = time.time()
        perm = torch.randperm(len(tr[2]), device=dev)
        for i in range(0, len(perm), a.batch):
            x, y = batch(*tr, perm[i:i + a.batch], dev, True)
            d, ls = net(x)
            loss = ((d - y) ** 2).mean() + beta_nll(d, ls.clamp_min(math.log(1e-3)), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            idx = torch.arange(len(va[2]), device=dev)
            x, y = batch(*va, idx, dev, True)                  # validate on degraded windows too
            d, ls = net(x)
            val = float(((d - y) ** 2).sum(1).mean().sqrt())
            still = float((y ** 2).sum(1).mean().sqrt())       # "stand still" predicts zero displacement
        hist.append({"epoch": ep, "val_rmse_m": val, "zero_rmse_m": still, "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if val < best:
            best = val
            torch.save({"state": net.state_dict(), "width": a.width}, out / "best.pt")
    net.load_state_dict(torch.load(out / "best.pt", weights_only=True)["state"])
    np.savez(out / "weights.npz", **{k: v for k, v in fold(net.cpu()).items()})
    (out / "history.json").write_text(json.dumps({"train": a.train, "val": a.val, "hist": hist}, indent=1))
    print(f"best val rmse {best:.3f} m per 1 s window (stand-still: {hist[0]['zero_rmse_m']:.3f})")


if __name__ == "__main__":
    main()
