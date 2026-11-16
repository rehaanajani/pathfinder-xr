"""Images for the deck, from real data (monochrome style).

    python deck/make_images.py --work <folder with clean.csv, run/imu0/data.csv, est.txt, terminal.txt> --gt <MH_05 GT csv>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "img"
BG = "#0A0E1A"
INK, MUTED, LINE, ACCENT, ALERT = "#F2F3F5", "#8A919E", "#262C38", "#5B8CFF", "#E5484D"
plt.rcParams.update({"font.family": "Helvetica", "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 13})


def signal(work):
    c = np.loadtxt(work / "clean.csv", delimiter=",", skiprows=1)
    d = np.loadtxt(work / "run" / "imu0" / "data.csv", delimiter=",", skiprows=1)
    t0 = c[0, 0]
    ct, dt_ = (c[:, 0] - t0) * 1e-9, (d[:, 0] - t0) * 1e-9
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), dpi=200, sharex=True, facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].plot(ct, c[:, 1], color=MUTED, lw=0.5)
    ax[0].set_ylabel("clean\naccel x (m/s²)")
    T = ct[-1]
    m = dt_ < T - 5
    ax[1].plot(dt_[m], d[m, 1], color=INK, lw=0.5)
    ax[1].plot(dt_[~m], d[~m, 1], color=ALERT, lw=1.5)
    ax[1].axvspan(T - 5, T, color=ALERT, alpha=0.12, lw=0)
    ax[1].text(T - 2.5, ax[1].get_ylim()[1] * 0.8, "IMU dead", color=ALERT, ha="center", fontsize=12)
    ax[1].set_ylabel("organisers'\ndamage")
    ax[1].set_xlabel("time (s), EuRoC MH_05")
    fig.tight_layout()
    fig.savefig(OUT / "signal.png", facecolor=BG)
    plt.close(fig)


def trajectory(work, gt_csv):
    est = np.loadtxt(work / "est.txt", comments="#")
    gt = np.loadtxt(gt_csv, delimiter=",", comments="#")
    tg = gt[:, 0] * 1e-9
    t, p = est[:, 0], est[:, 1:4]
    ok = (t >= tg[0]) & (t <= tg[-1])
    t, p = t[ok], p[ok]
    g = np.stack([np.interp(t, tg, gt[:, 1 + k]) for k in range(3)], 1)
    ms, md = p.mean(0), g.mean(0)
    U, _, Vt = np.linalg.svd((g - md).T @ (p - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ S @ Vt
    pa = p @ Rm.T + (md - Rm @ ms)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200, facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(g[:, 0], g[:, 1], color=MUTED, lw=1.4, label="motion capture (truth)")
    ax.plot(pa[:, 0], pa[:, 1], color=ACCENT, lw=1.8, label="Path Finder, IMU only")
    ax.scatter([md[0]], [md[1]], color=INK, s=40, zorder=5, label="standing still")
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=11)
    for tx in leg.get_texts():
        tx.set_color(INK)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.png", facecolor=BG)
    plt.close(fig)


def terminal(work):
    lines = (work / "terminal.txt").read_text().splitlines()
    f = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 30)
    W, H = 2000, 120 + 44 * len(lines)
    im = Image.new("RGB", (W, H), "#0D1117")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 64], fill="#161B22")
    for i, c in enumerate(("#4A4F58", "#4A4F58", "#4A4F58")):
        d.ellipse([28 + i * 40, 20, 52 + i * 40, 44], fill=c)
    d.text((W / 2, 32), "Terminal: the organisers' script, then the entry", font=ImageFont.truetype(
        "/System/Library/Fonts/SFNS.ttf", 26), fill="#8A919E", anchor="mm")
    y = 90
    for s in lines:
        col = "#5B8CFF" if s.startswith("$") else ("#F2F3F5" if ("wrote" in s or "resident" in s or " real" in s) else "#9AA0A6")
        d.text((40, y), s, font=f, fill=col)
        y += 44
    im.save(OUT / "terminal.png")
    # a shorter, larger crop for the architecture slide: the entry run only
    keep = [s for s in lines if not s.startswith(("1403", "# timestamp", "$ head", "clean.csv", " units", " severity"))]
    keep = [s.split("        #")[0].rstrip() for s in keep]          # drop our own annotation comment
    keep = [s for i, s in enumerate(keep) if s.strip() or (i and keep[i - 1].strip())]
    f2 = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 26)
    W2, H2 = 1180, 70 + 36 * len(keep)
    im2 = Image.new("RGB", (W2, H2), "#0D1117")
    d2 = ImageDraw.Draw(im2)
    d2.rectangle([0, 0, W2, 40], fill="#161B22")
    for i in range(3):
        d2.ellipse([18 + i * 28, 12, 34 + i * 28, 28], fill="#4A4F58")
    y = 55
    for s in keep:
        col = "#5B8CFF" if s.startswith("$") else ("#F2F3F5" if ("wrote" in s or "resident" in s or " real" in s) else "#9AA0A6")
        d2.text((22, y), s, font=f2, fill=col)
        y += 36
    im2.save(OUT / "terminal_short.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    w = Path(a.work)
    signal(w)
    trajectory(w, a.gt)
    terminal(w)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
