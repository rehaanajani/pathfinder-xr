"""Train the velocity network on AirIO-style input (body frame, gravity kept, attitude as input) for the IMU-only
Track 3 entry. Same weights file format as tools/train_velocity.py (version 3 = body-frame input and outputs).

    python tools/train_airio.py prep [--jobs 4]                  # degraded copies -> body-frame slot cache
    python tools/train_airio.py train --out runs/airio/a --window 200
    python tools/train_airio.py export --run runs/airio/a --out runs/airio/a/velnet_body.bin

AirIO (Qiu et al., RA-L 2025, arxiv 2501.15659): for a drone, IMU data rotated into a global frame loses what the
body-frame signal says about the motion (thrust along the body axis, attitude from the gravity component). The
network therefore reads, per 200 Hz slot, exactly what src/main.py has at run time with no ground truth:
  * gyro minus our gyro-bias estimate, body frame (before the gravity correction),
  * accelerometer (specific force) with gravity KEPT, body frame,
  * the "up" direction in the body frame from our own orientation (= AirIO's attitude input, without the
    unobservable yaw),
  * the fraction of healthy samples (0 = dropout / missing).
It predicts velocity over the last 0.25 s and displacement over the window, both in the body frame at the window
end, with log standard deviations; src/main.py rotates them into its world frame with its own orientation.
Targets: ground truth (EuRoC state_groundtruth_estimate0, TUM-VI mocap0: both the IMU frame) rotated into the body
frame with the ground-truth orientation.

Data and split as tools/train_velocity.py (train MH_01-03, V1_01-02, V2_01-02, room1-4; val MH_04, V1_03, room5;
held out MH_05, V2_03, room6). Degradation copies (noise ramps, bias drift, all dropout modes) are the same;
augmentation: mirror image, windows starting inside a dropout, random tilt error of the "up" channel.
"""
import argparse
import json
import math
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from eval_track3 import DATA, source  # noqa: E402
from train_velocity import TRAIN, VAL, TEST, copies  # noqa: E402

RATE = 200.0
VEL_T = 0.25
MIN_VALID = 0.25
N_IN = 10
SCALE = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
SPANS = (200, 400)                       # window lengths in slots (1 s and 2 s at 200 Hz, one sample per slot)
CACHE = Path(os.environ.get("PFX_CACHE", DATA / "velcache_body"))


# ----------------------------------------------------------------------------- data preparation
ORG_SCRIPT = os.environ.get("PFX_ORG_SCRIPT")      # the organisers' degrade_imu.py (16 Nov email; not in this repo)
ORG_COPIES = {"train": [(1000 + j, lvl) for j, lvl in enumerate((1.0, 1.0, 1.0, 0.7, 1.3, 1.6))],
              "val": [(1100, 1.0)], "test": [(1200, 1.0)]}


def organiser_copy(src, work, seed, max_level):
    """A copy damaged by the organisers' own degrade() (spikes, stuck runs, dropped rows and bursts, duplicates,
    ramped noise, bias and scale drift, last 5 s zero), with a training seed (>= 1000; their scoring seed is 42)
    and a varied final severity. Written in their documented column order (timestamp, accel, gyro)."""
    import importlib.util
    import pandas as pd
    spec_ = importlib.util.spec_from_file_location("org_degrade", ORG_SCRIPT)
    org = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(org)
    d = np.loadtxt(src / "imu0" / "data.csv", delimiter=",", comments="#")
    df = pd.DataFrame({"timestamp": d[:, 0].astype(np.int64), "ax": d[:, 4], "ay": d[:, 5], "az": d[:, 6],
                       "gx": d[:, 1], "gy": d[:, 2], "gz": d[:, 3]})
    cfg = dict(org.CONFIG, max_level=max_level)
    out, _ = org.degrade(df, cfg, seed)
    (work / "imu0").mkdir(parents=True, exist_ok=True)
    out.to_csv(work / "imu0" / "data.csv", index=False)


def all_copies(split):
    out = list(copies(split))
    if ORG_SCRIPT:
        out += [(f"org{s}_{lvl}", ["ORG", str(s), str(lvl)]) for s, lvl in ORG_COPIES[split]]
    return out


def prep_one(job):
    spec, tag, dargs = job
    from scipy.spatial.transform import Rotation as R, Slerp
    import main as entry
    name = f"{spec.replace(':', '_')}__{tag}"
    dst = CACHE / f"{name}.npz"
    if dst.exists():
        return name, None
    src, gt_csv = source(spec)
    work = CACHE / "degraded" / name
    if dargs and dargs[0] == "ORG":
        organiser_copy(src, work, seed=int(dargs[1]), max_level=float(dargs[2]))
    else:
        subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(work), *dargs],
                       check=True, capture_output=True)
    rec = []
    entry.run(str(work), str(work / "est.txt"),
              entry.args([str(work), "--preset", "learned", "--model", "none", "--model-input", "body",
                          "--model-window", str(max(SPANS)), "--model-dec", "1"]), rec=rec)
    n = len(rec)
    t = np.array([r[0] for r in rec], np.int64) * 1e-9
    X = np.zeros((n, N_IN), np.float32)
    for i, r in enumerate(rec):
        if r[2] is not None:
            X[i] = r[2]
    gt = np.loadtxt(gt_csv, delimiter=",", comments="#")
    tg = gt[:, 0] * 1e-9
    ok = np.r_[True, np.diff(tg) > 0]
    gt, tg = gt[ok], tg[ok]

    def pos(tt):
        tt = np.clip(tt, tg[0], tg[-1])
        return np.stack([np.interp(tt, tg, gt[:, 1 + i]) for i in range(3)], 1)
    r_gt = Slerp(tg, R.from_quat(gt[:, [5, 6, 7, 4]]))(np.clip(t, tg[0], tg[-1]))      # body -> GT world
    V = (r_gt.inv().apply(pos(t) - pos(t - VEL_T)) / VEL_T).astype(np.float32)
    out = {"X": X, "V": V, "t": t}
    # tilt error of our "up" channel against the true one (diagnostic)
    up_true = r_gt.inv().apply([0.0, 0.0, 1.0])
    has = X[:, 9] > 0
    cosang = np.clip((X[:, 6:9] * up_true).sum(1) / np.maximum(np.linalg.norm(X[:, 6:9], axis=1), 1e-9), -1, 1)
    tilt = np.degrees(np.arccos(cosang))[has]
    holes = tg[1:][np.diff(tg) > 0.05]
    cs = np.concatenate([[0], np.cumsum(has.astype(np.float64))])
    info = {"slots": n, "tilt_med_deg": float(np.median(tilt)), "tilt_p90_deg": float(np.percentile(tilt, 90))}
    for W in SPANS:
        span = W / RATE
        out[f"D{W}"] = r_gt.inv().apply(pos(t) - pos(t - span)).astype(np.float32)
        nvalid = np.zeros(n)
        nvalid[W - 1:] = cs[W:] - cs[:-W]
        M = has & (nvalid >= MIN_VALID * W)
        M[:W] = False
        M &= (t - span >= tg[0]) & (t <= tg[-1])
        for ge in holes:
            M &= ~((t > ge - 0.05) & (t < ge + span + 0.05))
        M[W:] &= np.abs((t[W:] - t[:-W]) / span - 1) < 0.02
        out[f"M{W}"] = M
        info[f"windows{W}"] = int(M.sum())
    np.savez(dst, **out)
    info["vel_rms"] = float(np.sqrt((V[out["M200"]] ** 2).sum(1).mean()))
    return name, info


def prep(a):
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, specs in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        for spec in specs:
            if not source(spec)[1].exists():
                print("missing data, skipped:", spec, flush=True)
                continue
            jobs += [(spec, tag, dargs) for tag, dargs in all_copies(split)]
    with Pool(a.jobs) as pool:
        for name, info in pool.imap_unordered(prep_one, jobs):
            if info:
                print(name, json.dumps(info), flush=True)


def load_split(specs, W, torch, dev, pattern="*"):
    Xs, Ds, Vs, idx, off = [], [], [], [], 0
    for spec in specs:
        files = sorted(CACHE.glob(f"{spec.replace(':', '_')}__{pattern}.npz"))
        if not files:
            raise SystemExit(f"no cache for {spec}: run `train_airio.py prep` first")
        for f in files:
            z = np.load(f)
            Xs.append(z["X"]); Ds.append(z[f"D{W}"]); Vs.append(z["V"])
            idx.append(np.flatnonzero(z[f"M{W}"]) + off)
            off += len(z["X"])
    X = np.concatenate(Xs)
    X[:, :-1] *= np.asarray(SCALE, np.float32)
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    return t(X), t(np.concatenate(Ds)), t(np.concatenate(Vs)), t(np.concatenate(idx))


# ----------------------------------------------------------------------------- model
def build(widths, hidden, kernels=(7, 5, 5, 3), dropout=0.0):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, cin = [], N_IN
            for w, k in zip(widths, kernels):
                layers += [nn.Conv1d(cin, w, k, 2, k // 2, bias=False), nn.BatchNorm1d(w), nn.ReLU()]
                cin = w
            self.conv = nn.Sequential(*layers)
            self.fc1 = nn.Linear(cin, hidden)
            self.fc2 = nn.Linear(hidden, 12)
            # Dropout on the pooled features and the hidden layer (VIO-Edge trains with 0.3; MIT licence). Identity
            # at run time, so the exported weights file and BatchNorm folding are unchanged.
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            z = self.drop(self.conv(x).mean(2))
            return self.fc2(self.drop(self.fc1(z).relu()))
    return Net()


def folded(net):
    import torch
    net = net.eval().cpu()
    convs = []
    mods = list(net.conv)
    for i in range(0, len(mods), 3):
        conv, bn = mods[i], mods[i + 1]
        sc = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        w = conv.weight.detach().numpy() * sc[:, None, None]
        b = bn.bias.detach().numpy() - bn.running_mean.detach().numpy() * sc
        convs.append((w.astype(np.float32), b.astype(np.float32), conv.stride[0], conv.padding[0]))
    fcs = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in (net.fc1, net.fc2)]
    return convs, fcs


def small_rot(B, sigma_rad, torch, dev):
    """Random rotation matrices (B, 3, 3) with rotation vectors ~ N(0, sigma^2) per axis (Rodrigues)."""
    r = torch.randn(B, 3, device=dev) * sigma_rad
    th = r.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = r / th
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=dev)[None] + s * K + (1 - c) * (K @ K)


def batch(data, sel, W, train, a, torch):
    X, D, V, _ = data
    dev = X.device
    ar = torch.arange(-W + 1, 1, device=dev)
    x = X[sel[:, None] + ar[None]]                                      # (B, W, 10)
    d, v = D[sel], V[sel]
    B = len(sel)
    if train:
        one = torch.ones(B, device=dev)
        if a.mirror:     # mirror image (y -> -y): vectors flip y, the angular rate (pseudo-vector) flips x and z
            m = (torch.rand(B, device=dev) < 0.5).float() * -2 + 1
            sx = torch.stack([m, one, m, one, m, one, one, m, one, one], 1)
            x = x * sx[:, None, :]
            sy = torch.stack([one, m, one], 1)
            d, v = d * sy, v * sy
        if a.tilt_aug > 0:  # our "up" is off by a few degrees (and more right after a dropout)
            Rt = small_rot(B, math.radians(a.tilt_aug), torch, dev)
            u = torch.einsum("bij,bwj->bwi", Rt, x[..., 6:9])
            x = torch.cat([x[..., :6], u, x[..., 9:]], -1)
        if a.rot_aug > 0:   # the IMU mounted another way: one random rotation of every body-frame vector
            q = torch.randn(B, 4, device=dev)
            q = q / q.norm(dim=1, keepdim=True)
            w_, x_, y_, z_ = q.unbind(1)
            Rr = torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_),
                              2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_),
                              2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)], 1).view(B, 3, 3)
            keep = torch.rand(B, device=dev) >= a.rot_aug
            Rr[keep] = torch.eye(3, device=dev)
            rot = lambda u: torch.einsum("bij,bwj->bwi", Rr, u)  # noqa: E731
            x = torch.cat([rot(x[..., 0:3]), rot(x[..., 3:6]), rot(x[..., 6:9]), x[..., 9:]], -1)
            d = torch.einsum("bij,bj->bi", Rr, d)
            v = torch.einsum("bij,bj->bi", Rr, v)
        if a.noise_aug > 0:  # per-window synthetic noise and bias on healthy slots (after VIO-Edge's AUG, MIT licence)
            u = torch.rand(B, 1, 1, device=dev) ** 2 * a.noise_aug
            ok = (x[..., 9:10] > 0).float()
            acc = (torch.randn(B, W, 3, device=dev) * u * 2.5 + torch.randn(B, 1, 3, device=dev)
                   * torch.rand(B, 1, 1, device=dev) * 0.3) * SCALE[3]
            gyr = torch.randn(B, W, 3, device=dev) * u * 0.05 + torch.randn(B, 1, 3, device=dev)                 * torch.rand(B, 1, 1, device=dev) * 0.05
            x = torch.cat([x[..., 0:3] + gyr * ok, x[..., 3:6] + acc * ok, x[..., 6:]], -1)
        cut = torch.where(torch.rand(B, device=dev) < 0.1,
                          torch.randint(0, int(W * (1 - MIN_VALID)), (B,), device=dev),
                          torch.zeros(B, dtype=torch.long, device=dev))
        x = x * (torch.arange(W, device=dev)[None] >= cut[:, None]).float()[..., None]
        x = x * (x[..., 9:10] > 0).float()                                 # a cut slot is empty in every channel
    return x.transpose(1, 2).contiguous(), d, v


def losses(out, d, v, torch, wd=1.0, mse_only=False):
    def part(mu, ls, y):
        if mse_only:                                # warm-up: the mean first (VIO-Edge trains 3 epochs on MSE)
            return ((mu - y) ** 2).mean()
        ls = ls.clamp(-7, 3)
        var = torch.exp(2 * ls)
        nll = (((mu - y) ** 2) / (2 * var) + ls) * var.detach() ** 0.5   # beta-NLL, beta = 0.5
        return ((mu - y) ** 2).mean() + nll.mean()
    return wd * part(out[:, 0:3], out[:, 3:6], d) + part(out[:, 6:9], out[:, 9:12], v)


def evaluate(net, data, W, a, torch, bs=4096):
    net.eval()
    se_d = se_v = z_d = z_v = n = cal = 0.0
    with torch.no_grad():
        idx = data[3]
        for i in range(0, len(idx), bs):
            x, d, v = batch(data, idx[i:i + bs], W, False, a, torch)
            o = net(x)
            se_d += float(((o[:, 0:3] - d) ** 2).sum()); z_d += float((d ** 2).sum())
            se_v += float(((o[:, 6:9] - v) ** 2).sum()); z_v += float((v ** 2).sum())
            cal += float((((o[:, 6:9] - v) / torch.exp(o[:, 9:12].clamp(-7, 3))) ** 2).sum())
            n += len(d)
    return {"disp_rmse_m": math.sqrt(se_d / n), "disp_zero_m": math.sqrt(z_d / n),
            "vel_rmse_mps": math.sqrt(se_v / n), "vel_zero_mps": math.sqrt(z_v / n), "vel_z2": cal / (3 * n)}


def train(a):
    import torch
    dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    W = a.window
    tr = load_split(a.train, W, torch, dev, a.train_copies)
    va = load_split(a.val, W, torch, dev, a.val_copies)
    va = (*va[:3], va[3][::a.val_stride])
    net = build(a.widths, a.hidden, a.kernels, a.dropout).to(dev)
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    steps_ep = a.windows_per_epoch // a.batch
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_ep)
    print(f"params={n_par} window={W} train_windows={len(tr[3])} val_windows={len(va[3])} dev={dev}", flush=True)
    best, hist = 1e9, []
    for ep in range(a.epochs):
        net.train(); t0 = time.time()
        for _ in range(steps_ep):
            sel = tr[3][torch.randint(0, len(tr[3]), (a.batch,), device=dev)]
            x, d, v = batch(tr, sel, W, True, a, torch)
            loss = losses(net(x), d, v, torch, a.disp_weight, mse_only=ep < a.mse_epochs)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        h = {"epoch": ep, **evaluate(net, va, W, a, torch), "sec": round(time.time() - t0, 1)}
        hist.append(h)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()}), flush=True)
        score = h["vel_rmse_mps"] / h["vel_zero_mps"]
        if score < best:
            best = score
            torch.save({"state": net.state_dict(), "widths": a.widths, "hidden": a.hidden, "kernels": a.kernels,
                        "window": W, "params": n_par}, out / "best.pt")
    net.load_state_dict(torch.load(out / "best.pt", weights_only=True)["state"])
    per = {}
    for spec in a.val:
        one = load_split([spec], W, torch, dev, a.val_copies)
        per[spec] = evaluate(net, (*one[:3], one[3][::a.val_stride]), W, a, torch)
        print("best", spec, json.dumps({k: round(v, 3) for k, v in per[spec].items()}), flush=True)
    (out / "history.json").write_text(json.dumps({"args": vars(a), "params": n_par, "hist": hist,
                                                  "best_per_recording": per}, indent=1))


def export(a):
    import torch
    from velnet import VelNet, save_bin
    ck = torch.load(Path(a.run) / "best.pt", weights_only=True, map_location="cpu")
    net = build(ck["widths"], ck["hidden"], ck["kernels"])
    net.load_state_dict(ck["state"])
    convs, fcs = folded(net)
    W = ck["window"]
    save_bin(a.out, W, 1, RATE, SCALE, convs, fcs, version=3)
    ref = VelNet(a.out)
    x = torch.randn(8, N_IN, W)
    with torch.no_grad():
        y = net.eval()(x).numpy()
    err = max(float(np.abs(ref.forward(x[i].numpy()) - y[i]).max()) for i in range(8))
    print(f"{a.out}: {ref.n_params} parameters, window {W}, {Path(a.out).stat().st_size} bytes, "
          f"numpy forward vs PyTorch max |diff| = {err:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--jobs", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", default=TRAIN); t.add_argument("--val", nargs="+", default=VAL)
    t.add_argument("--window", type=int, default=200, choices=SPANS)
    t.add_argument("--widths", nargs=4, type=int, default=[32, 48, 64, 64]); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--kernels", nargs=4, type=int, default=[7, 5, 5, 3])
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--batch", type=int, default=256)
    t.add_argument("--windows-per-epoch", type=int, default=150000); t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--dropout", type=float, default=0.0); t.add_argument("--mse-epochs", type=int, default=0)
    t.add_argument("--noise-aug", type=float, default=0.0, help="scale of per-window synthetic noise (1 = VIO-Edge's)")
    t.add_argument("--wd", type=float, default=1e-4); t.add_argument("--disp-weight", type=float, default=1.0)
    t.add_argument("--tilt-aug", type=float, default=2.0, help="deg, std of the random tilt error of the up channel")
    t.add_argument("--mirror", type=int, default=1)
    t.add_argument("--rot-aug", type=float, default=0.0, help="fraction of windows given a random IMU mounting")
    t.add_argument("--train-copies", default="*", help="glob of the cached copies to train on, e.g. 'org*'")
    t.add_argument("--val-copies", default="*", help="glob of the cached copies to validate on")
    t.add_argument("--val-stride", type=int, default=3); t.add_argument("--cpu", action="store_true")
    t.add_argument("--seed", type=int, default=0); t.add_argument("--out", default="runs/airio/a")
    e = sub.add_parser("export")
    e.add_argument("--run", default="runs/airio/a"); e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": prep, "train": train, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
