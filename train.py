"""Train the displacement network on the TLIO headset dataset.

    python train.py --aug faults --epochs 12      # ours: trained on broken IMU data too
    python train.py --aug tlio   --epochs 12      # TLIO-style augmentation only (ablation)

Windows are cut on the GPU from in-memory sequences, so an epoch takes seconds-to-minutes on
an Apple M-series or any CUDA GPU.

Augmentation
  tlio   : random yaw, constant bias shift (acc 0.2, gyro 0.05), small gravity tilt (5 deg)
  faults : tlio + (with probability) heavy white noise, bias drift, dropped samples held at
           the last value, spikes, saturation, a dead (zero) sensor. The network is trained
           to report a larger standard deviation when its input is broken, which is what
           tells the filter to lean on it less.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pathfinder import data_tlio as D
from pathfinder.model import WINDOW, DispNet, n_params, nll


def load_split(name, limit=None, stride=10):
    ids = D.split(name)[:limit]
    imu, pos, starts = [], [], []
    off = 0
    for sid in ids:
        d = np.load(D.ROOT / sid / "imu0_resampled.npy")
        imu.append(np.concatenate([d[:, 1:4], d[:, 4:7]], 1).astype(np.float32))   # world frame
        pos.append(d[:, 11:14].astype(np.float32))
        n = len(d)
        starts.append(np.arange(off, off + n - WINDOW - 1, stride))
        off += n
    return (torch.from_numpy(np.concatenate(imu)), torch.from_numpy(np.concatenate(pos)),
            torch.from_numpy(np.concatenate(starts)), ids)


def rot_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], -2)


def small_tilt(B, deg, dev):
    ax = torch.randn(B, 3, device=dev)
    ax[:, 2] = 0
    ax = ax / ax.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ang = torch.rand(B, 1, device=dev) * math.radians(deg)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 2], K[:, 1, 2] = ax[:, 1], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    s, c = torch.sin(ang)[..., None], torch.cos(ang)[..., None]
    return torch.eye(3, device=dev) + s * K + (1 - c) * (K @ K)


def batch(imu, pos, starts, idx, aug, dev, p_fault=0.6):
    s = starts[idx]
    ar = torch.arange(WINDOW + 1, device=dev)
    win = imu[(s[:, None] + ar[None, :-1])]                         # (B, W, 6)
    disp = pos[s + WINDOW] - pos[s]                                  # (B, 3)
    B = len(idx)
    if aug != "none":
        Rz = rot_z(torch.rand(B, device=dev) * 2 * math.pi)
        T = Rz @ small_tilt(B, 5.0, dev)
        g, a = win[..., :3] @ T.transpose(1, 2), win[..., 3:] @ T.transpose(1, 2)
        disp = (Rz @ disp[..., None])[..., 0]
        g = g + (torch.rand(B, 1, 3, device=dev) - 0.5) * 2 * 0.05
        a = a + (torch.rand(B, 1, 3, device=dev) - 0.5) * 2 * 0.2
        if aug == "faults":
            kind = torch.randint(0, 7, (B,), device=dev)
            hit = torch.rand(B, device=dev) < p_fault
            sev = torch.rand(B, 1, 1, device=dev)
            m = lambda k: (hit & (kind == k))[:, None, None].float()
            # 0 heavy noise
            g = g + m(0) * torch.randn_like(g) * 0.15 * sev
            a = a + m(0) * torch.randn_like(a) * 1.5 * sev
            # 1 bias drift (random walk)
            g = g + m(1) * torch.cumsum(torch.randn_like(g) * 0.004 * sev, 1)
            a = a + m(1) * torch.cumsum(torch.randn_like(a) * 0.04 * sev, 1)
            # 2 dropout: a random stretch holds the last value
            t0 = torch.randint(0, WINDOW // 2, (B, 1), device=dev)
            L = (sev[:, 0] * WINDOW * 0.5).long()
            tt = torch.arange(WINDOW, device=dev)[None]
            drop = ((tt >= t0) & (tt < t0 + L)).float()[..., None] * m(2)
            hold_g = torch.gather(g, 1, t0[..., None].expand(-1, 1, 3))
            hold_a = torch.gather(a, 1, t0[..., None].expand(-1, 1, 3))
            g = g * (1 - drop) + hold_g * drop
            a = a * (1 - drop) + hold_a * drop
            # 3 spikes
            sp = (torch.rand(B, WINDOW, 1, device=dev) < 0.03).float() * m(3)
            g = g + sp * torch.randn_like(g) * 3.0
            a = a + sp * torch.randn_like(a) * 30.0
            # 4 saturation
            gm, am = 0.4 + 1.0 * sev, 10.0 + 4.0 * sev
            g = torch.where(m(4).bool().expand_as(g), torch.maximum(torch.minimum(g, gm), -gm), g)
            a = torch.where(m(4).bool().expand_as(a), torch.maximum(torch.minimum(a, am), -am), a)
            # 5 dead sensor (zeros) for part of the window
            dead = ((tt >= t0) & (tt < t0 + L)).float()[..., None] * m(5)
            g, a = g * (1 - dead), a * (1 - dead)
            # 6 big constant bias
            g = g + m(6) * (torch.rand(B, 1, 3, device=dev) - 0.5) * 0.4 * sev
            a = a + m(6) * (torch.rand(B, 1, 3, device=dev) - 0.5) * 4.0 * sev
        win = torch.cat([g, a], -1)
    return win.transpose(1, 2).contiguous(), disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="faults", choices=["none", "tlio", "faults"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    out = Path(args.out or f"models/dispnet_{args.aug}")
    out.mkdir(parents=True, exist_ok=True)

    tr_imu, tr_pos, tr_st, _ = load_split("train", args.limit, args.stride)
    va_imu, va_pos, va_st, _ = load_split("val", args.limit, stride=50)
    tr_imu, tr_pos, tr_st = tr_imu.to(dev), tr_pos.to(dev), tr_st.to(dev)
    va_imu, va_pos, va_st = va_imu.to(dev), va_pos.to(dev), va_st.to(dev)
    net = DispNet(args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr_st) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    print(f"device={dev} params={n_params(net)} train_windows={len(tr_st)} val_windows={len(va_st)}",
          flush=True)
    best, hist = 1e9, []
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        perm = torch.randperm(len(tr_st), device=dev)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            x, y = batch(tr_imu, tr_pos, tr_st, perm[i:i + args.batch], args.aug, dev)
            d, ls = net(x)
            loss = ((d - y) ** 2).mean() if ep < 2 else nll(d, ls.clamp_min(math.log(1e-3)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        net.eval()
        with torch.no_grad():
            errs, stds = [], []
            for i in range(0, len(va_st), 4096):
                idx = torch.arange(i, min(i + 4096, len(va_st)), device=dev)
                x, y = batch(va_imu, va_pos, va_st, idx, "none", dev)
                d, ls = net(x)
                errs.append((d - y).norm(dim=1))
                stds.append(torch.exp(ls).norm(dim=1))
            e = torch.cat(errs)
            val = float(e.pow(2).mean().sqrt())
        rec = {"epoch": ep, "train_loss": tot / len(perm), "val_rmse_m": val,
               "val_pred_std_m": float(torch.cat(stds).mean()), "sec": time.time() - t0}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if val < best:
            best = val
            torch.save({"state": net.state_dict(), "width": args.width, "aug": args.aug}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print("best val rmse (m over 1 s):", best)


if __name__ == "__main__":
    main()
