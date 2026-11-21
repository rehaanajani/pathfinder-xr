"""Train the small learned velocity network for the IMU-only Track 3 entry (EuRoC MAV + TUM-VI).

    python tools/train_velocity.py prep                 # degraded copies -> window cache (local_data/velcache)
    python tools/train_velocity.py train --out runs/vel/a [--widths 24 32 48 64 --hidden 64 --epochs 40]
    python tools/train_velocity.py export --run runs/vel/a --out src/velnet_weights.bin

What the network sees is exactly what src/main.py has at run time, with no ground truth:
  * every recording is degraded with tools/degrade_imu.py (many seeds, all dropout modes, three noise levels, plus
    one clean copy) and replayed through src/main.py itself, which records its window slots: gyro (bias-corrected)
    and specific force (minus g) rotated into main.py's own gravity-aligned frame, or an empty slot for an
    unhealthy / missing sample. Tilt errors, yaw drift and dropouts are therefore the real ones.
  * targets come from ground truth, expressed in the same (estimated) frame: the displacement over the window and
    the velocity over its last 0.25 s. A random yaw is applied to input and target (heading is unobservable).
  * loss: MSE + beta-NLL (Seitzer et al. 2022) so the network also reports a standard deviation (confidence).

Fixed split (everything is chosen on VAL; TEST is scored once at the end):
  TRAIN  MH_01-03, V1_01-02, V2_01-02, room1-4      VAL  MH_04, V1_03, room5      TEST  MH_05, V2_03, room6
"""
import argparse
import os
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from eval_track3 import DATA, source  # noqa: E402

TRAIN = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
         "euroc:V2_01_easy", "euroc:V2_02_medium", "tumvi:room1", "tumvi:room2", "tumvi:room3", "tumvi:room4"]
VAL = ["euroc:MH_04_difficult", "euroc:V1_03_difficult", "tumvi:room5"]
TEST = ["euroc:MH_05_difficult", "euroc:V2_03_difficult", "tumvi:room6"]

WINDOW = 200            # slots in the network window
DEC = int(os.environ.get("PFX_DEC", 1))                 # samples averaged per slot -> the window spans WINDOW * DEC / RATE = 1 s
RATE = 200.0
SPAN = WINDOW * DEC / RATE
VEL_T = 0.25            # end velocity = mean velocity over the last 0.25 s
MIN_VALID = 0.25        # fraction of the window that must be healthy (same as main.py --model-min-valid)
SCALE = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3]
CACHE = Path(os.environ.get("PFX_CACHE", DATA / "velcache"))


# ----------------------------------------------------------------------------- data preparation
def copies(split):
    """Degradation variants per recording: (tag, degrade_imu.py arguments). Seeds >= 100: never the scoring seeds."""
    out = [("clean", ["--noise", "0", "0", "0", "0", "--bias-rate", "0", "0", "--dropouts", "0", "--blind", "0"])]
    modes = ["missing", "zeros", "frozen", "nan"]
    n_seed = 2 if split == "train" else 1
    for k, mode in enumerate(modes):
        for j in range(n_seed):
            out.append((f"{mode}{j}", ["--seed", str(100 + 10 * k + j), "--dropout-mode", mode, "--death", mode]))
    if split == "train":
        for j, sc in enumerate((0.5, 0.5, 1.5, 1.5)):          # milder and harsher jamming than the default
            noise = [str(x * sc) for x in (0.002, 0.03, 0.04, 0.6)]
            out.append((f"x{sc}_{j}", ["--seed", str(200 + j), "--dropout-mode", modes[j], "--death", modes[j],
                                       "--noise", *noise, "--bias-rate", str(0.002 * sc), str(0.03 * sc),
                                       "--dropouts", "4"]))
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
    subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(work), *dargs],
                   check=True, capture_output=True)
    rec = []
    entry.run(str(work), str(work / "est.txt"),
              entry.args([str(work), "--preset", "learned", "--model", "none", "--model-input", "world", "--model-window", str(WINDOW), "--model-dec", str(DEC)]),
              rec=rec)
    n = len(rec)
    t = np.array([r[0] for r in rec], np.int64) * 1e-9                  # time of the last sample of each slot
    q = np.array([r[1] for r in rec])                                   # (w, x, y, z), body -> estimator frame
    X = np.zeros((n, 7), np.float32)
    for i, r in enumerate(rec):
        if r[2] is not None:
            X[i] = r[2]
    gt = np.loadtxt(gt_csv, delimiter=",", comments="#")
    tg = gt[:, 0] * 1e-9

    def pos(tt):
        tt = np.clip(tt, tg[0], tg[-1])
        return np.stack([np.interp(tt, tg, gt[:, 1 + i]) for i in range(3)], 1)
    r_gt = Slerp(tg, R.from_quat(gt[:, [5, 6, 7, 4]]))(np.clip(t, tg[0], tg[-1]))
    rel = R.from_quat(q[:, [1, 2, 3, 0]]) * r_gt.inv()                  # ground-truth world -> estimator frame
    D = rel.apply(pos(t) - pos(t - SPAN)).astype(np.float32)
    V = (rel.apply(pos(t) - pos(t - VEL_T)) / VEL_T).astype(np.float32)
    has = (X[:, 6] > 0).astype(np.float64)
    cs = np.concatenate([[0], np.cumsum(has)])
    nvalid = np.zeros(n)
    nvalid[WINDOW - 1:] = cs[WINDOW:] - cs[:-WINDOW]
    M = (X[:, 6] > 0) & (nvalid >= MIN_VALID * WINDOW)
    M[:WINDOW] = False
    M &= (t - SPAN >= tg[0]) & (t <= tg[-1])
    for ge in tg[1:][np.diff(tg) > 0.05]:                                # holes in the ground truth
        M &= ~((t > ge - 0.05) & (t < ge + SPAN + 0.05))
    M[WINDOW:] &= np.abs((t[WINDOW:] - t[:-WINDOW]) / SPAN - 1) < 0.02  # the window really spans SPAN seconds
    np.savez(dst, X=X, D=D, V=V, M=M)
    return name, {"slots": n, "windows": int(M.sum()), "disp_rms": float(np.sqrt((D[M] ** 2).sum(1).mean())),
                  "vel_rms": float(np.sqrt((V[M] ** 2).sum(1).mean()))}


def prep(a):
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, specs in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        for spec in specs:
            if not source(spec)[1].exists():
                print("missing data, skipped:", spec, flush=True)
                continue
            jobs += [(spec, tag, dargs) for tag, dargs in copies(split)]
    with Pool(a.jobs) as pool:
        for name, info in pool.imap_unordered(prep_one, jobs):
            if info:
                print(name, json.dumps(info), flush=True)


def load_split(specs, torch):
    Xs, Ds, Vs, idx, off = [], [], [], [], 0
    for spec in specs:
        files = sorted(CACHE.glob(f"{spec.replace(':', '_')}__*.npz"))
        if not files:
            raise SystemExit(f"no cache for {spec}: run `train_velocity.py prep` first")
        for f in files:
            z = np.load(f)
            Xs.append(z["X"]); Ds.append(z["D"]); Vs.append(z["V"])
            idx.append(np.flatnonzero(z["M"]) + off)
            off += len(z["X"])
    X = np.concatenate(Xs)
    X[:, :6] *= np.asarray(SCALE, np.float32)
    return (torch.from_numpy(X), torch.from_numpy(np.concatenate(Ds)), torch.from_numpy(np.concatenate(Vs)),
            torch.from_numpy(np.concatenate(idx)))


# ----------------------------------------------------------------------------- model
def build(widths, hidden, dropout=0.0):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            spec = [(7, 2, 3), (5, 2, 2), (5, 2, 2), (3, 2, 1)]
            layers, cin = [], 7
            for w, (k, s, p) in zip(widths, spec):
                layers += [nn.Conv1d(cin, w, k, s, p, bias=False), nn.BatchNorm1d(w), nn.ReLU()]
                cin = w
            self.conv = nn.Sequential(*layers)
            self.fc1 = nn.Linear(cin, hidden)
            self.drop = nn.Dropout(dropout)          # identity at export (VIO-Edge trains with 0.3; MIT licence)
            self.fc2 = nn.Linear(hidden, 12)

        def forward(self, x):                                           # (B, 7, WINDOW)
            z = self.conv(x).mean(2)
            return self.fc2(self.drop(self.fc1(self.drop(z)).relu()))
    return Net()


def folded(net):
    """BatchNorm folded into the convolutions -> plain arrays for src/velnet.py and the C++ port."""
    import torch
    net = net.eval()
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


def batch(data, sel, train, torch):
    X, D, V, _ = data
    ar = torch.arange(-WINDOW + 1, 1)
    x = X[sel[:, None] + ar[None]]                                      # (B, W, 7)
    d, v = D[sel], V[sel]
    B = len(sel)
    if train:
        th = torch.rand(B) * 2 * math.pi                                # heading is arbitrary
        c, s = torch.cos(th), torch.sin(th)

        def rz(u):                                                      # rotate (..., 3) about z, per batch item
            cc, ss = (c, s) if u.dim() == 2 else (c[:, None], s[:, None])
            return torch.stack([cc * u[..., 0] - ss * u[..., 1], ss * u[..., 0] + cc * u[..., 1], u[..., 2]], -1)
        x = torch.cat([rz(x[..., 0:3]), rz(x[..., 3:6]), x[..., 6:7]], -1)
        d, v = rz(d), rz(v)
        # mirror image of the motion (y -> -y): vectors flip y, the angular rate (a pseudo-vector) flips x and z
        m = (torch.rand(B) < 0.5).float() * -2 + 1
        sx = torch.stack([m, torch.ones(B), m, torch.ones(B), m, torch.ones(B), torch.ones(B)], 1)
        x = x * sx[:, None, :]
        d = d * torch.stack([torch.ones(B), m, torch.ones(B)], 1)
        v = v * torch.stack([torch.ones(B), m, torch.ones(B)], 1)
        # a window that starts inside a dropout (the first call after the signal returns)
        cut = torch.where(torch.rand(B) < 0.1, torch.randint(0, int(WINDOW * (1 - MIN_VALID)), (B,)),
                          torch.zeros(B, dtype=torch.long))
        x = x * (torch.arange(WINDOW)[None] >= cut[:, None]).float()[..., None]
    return x.transpose(1, 2).contiguous(), d, v


def losses(out, d, v, torch):
    def part(mu, ls, y):
        ls = ls.clamp(-7, 3)
        var = torch.exp(2 * ls)
        nll = (((mu - y) ** 2) / (2 * var) + ls) * var.detach() ** 0.5   # beta-NLL, beta = 0.5
        return ((mu - y) ** 2).mean() + nll.mean()
    return part(out[:, 0:3], out[:, 3:6], d) + part(out[:, 6:9], out[:, 9:12], v)


def evaluate(net, data, torch, bs=2048):
    net.eval()
    se_d = se_v = z_d = z_v = n = 0.0
    cal = 0.0
    with torch.no_grad():
        idx = data[3]
        for i in range(0, len(idx), bs):
            x, d, v = batch(data, idx[i:i + bs], False, torch)
            o = net(x)
            se_d += float(((o[:, 0:3] - d) ** 2).sum()); z_d += float((d ** 2).sum())
            se_v += float(((o[:, 6:9] - v) ** 2).sum()); z_v += float((v ** 2).sum())
            cal += float((((o[:, 0:3] - d) / torch.exp(o[:, 3:6].clamp(-7, 3))) ** 2).sum())
            n += len(d)
    return {"disp_rmse_m": math.sqrt(se_d / n), "disp_zero_m": math.sqrt(z_d / n),
            "vel_rmse_mps": math.sqrt(se_v / n), "vel_zero_mps": math.sqrt(z_v / n),
            "disp_z2": cal / (3 * n)}                                   # ~1 if the reported std is calibrated


def train(a):
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    tr = load_split(a.train, torch)
    va = load_split(a.val, torch)
    va = (*va[:3], va[3][::a.val_stride])
    net = build(a.widths, a.hidden, a.dropout)
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    steps_ep = a.windows_per_epoch // a.batch
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_ep)
    print(f"params={n_par} train_windows={len(tr[3])} val_windows={len(va[3])}", flush=True)
    best, hist = 1e9, []
    for ep in range(a.epochs):
        net.train(); t0 = time.time()
        for _ in range(steps_ep):
            sel = tr[3][torch.randint(0, len(tr[3]), (a.batch,))]
            x, d, v = batch(tr, sel, True, torch)
            loss = losses(net(x), d, v, torch)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        h = {"epoch": ep, **evaluate(net, va, torch), "sec": round(time.time() - t0, 1)}
        hist.append(h)
        print(json.dumps(h), flush=True)
        score = h["disp_rmse_m"] / h["disp_zero_m"] + h["vel_rmse_mps"] / h["vel_zero_mps"]
        if score < best:
            best = score
            torch.save({"state": net.state_dict(), "widths": a.widths, "hidden": a.hidden, "params": n_par}, out / "best.pt")
    net.load_state_dict(torch.load(out / "best.pt", weights_only=True)["state"])
    per = {}
    for spec in a.val:                                                  # best checkpoint, per validation recording
        one = load_split([spec], torch)
        per[spec] = evaluate(net, (*one[:3], one[3][::a.val_stride]), torch)
        print("best", spec, json.dumps({k: round(v, 3) for k, v in per[spec].items()}), flush=True)
    (out / "history.json").write_text(json.dumps({"train": a.train, "val": a.val, "params": n_par, "hist": hist,
                                                  "best_per_recording": per}, indent=1))


def export(a):
    import torch
    from velnet import VelNet, save_bin
    ck = torch.load(Path(a.run) / "best.pt", weights_only=True)
    net = build(ck["widths"], ck["hidden"])
    net.load_state_dict(ck["state"])
    convs, fcs = folded(net)
    save_bin(a.out, WINDOW, DEC, RATE, SCALE, convs, fcs)
    ref = VelNet(a.out)
    x = torch.randn(8, 7, WINDOW)
    with torch.no_grad():
        y = net.eval()(x).numpy()
    err = max(float(np.abs(ref.forward(x[i].numpy()) - y[i]).max()) for i in range(8))
    print(f"{a.out}: {ref.n_params} parameters, {Path(a.out).stat().st_size} bytes, "
          f"numpy forward vs PyTorch max |diff| = {err:.2e}")
    if a.header:
        flat = np.frombuffer(open(a.out, "rb").read(), np.uint8)
        with open(a.header, "w") as f:
            f.write(f"// generated by tools/train_velocity.py export: the bytes of {Path(a.out).name}\n"
                    f"static const unsigned int VELNET_BIN_LEN = {len(flat)};\nstatic const unsigned char VELNET_BIN[] = {{\n")
            for i in range(0, len(flat), 24):
                f.write(",".join(str(int(b)) for b in flat[i:i + 24]) + ",\n")
            f.write("};\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--jobs", type=int, default=2)
    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", default=TRAIN); t.add_argument("--val", nargs="+", default=VAL)
    t.add_argument("--dropout", type=float, default=0.0); t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--widths", nargs=4, type=int, default=[24, 32, 48, 64]); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--batch", type=int, default=256)
    t.add_argument("--windows-per-epoch", type=int, default=150000); t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--val-stride", type=int, default=3); t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--out", default="runs/vel/a")
    e = sub.add_parser("export")
    e.add_argument("--run", default="runs/vel/a"); e.add_argument("--out", default="src/velnet_weights.bin")
    e.add_argument("--header", default=None, help="also write the weights as a C header (unsigned char array)")
    a = ap.parse_args()
    {"prep": prep, "train": train, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
