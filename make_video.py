"""Renders results/demo.mp4 (about 75 s, 1280x720, captions, no audio) from real runs.

Every path in the video is computed here on a held-out TLIO headset sequence; nothing is drawn by hand.
"""

import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from pathfinder import data_tlio as D
from pathfinder.faults import Fault
from pathfinder.pipeline import run_methods
from pathfinder.predictor import Predictor

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
BG, PANEL, INK, MUTED = "#0a0e1a", "#121a2e", "#eef1f6", "#9aa6bc"
CYAN, GREEN, ORANGE, RED, LINE = "#00b4d8", "#2ed39a", "#f08a34", "#f36b5f", "#232e4a"
FPS = 30
W, H = 12.8, 7.2


def text_card(fig, lines, seconds):
    frames = []
    for _ in range(int(seconds * FPS)):
        frames.append(("card", lines))
    return frames


def run_case(seq, faults, model, seed):
    r = run_methods(seq, faults, seed=seed, ours_model=model, methods=("naive_fusion", "ours"))
    step = 20
    o = r["ours"]
    return {
        "truth": seq.p[::step, :2], "naive": r["naive_fusion"]["p"][::step, :2], "ours": o["p"][::step, :2],
        "en": np.linalg.norm(r["naive_fusion"]["p"][::step] - seq.p[::step], axis=1),
        "eo": np.linalg.norm(o["p"][::step] - seq.p[::step], axis=1),
        "cam": np.where(np.isnan(r["_cam"]["fix"][:, 0]), 0, r["_cam"]["health"]),
        "imu": o["imu_weights"], "t": seq.t[::step] - seq.t[0],
    }


def main():
    model = Predictor("models/dispnet_faults/best.pt")
    sid = D.split("test")[4]
    base = D.load(sid)
    seq = D.redundant_imus(base.slice(30 * 200, 90 * 200 + 1), 3, seed=0)
    cases = [
        ("Camera off for 10 s, and IMU 1 develops a bias",
         [Fault("camera_blackout", 20, 10), Fault("imu_bias", 15, 1e9, 0, {"gyro": [0.08, -0.05, 0.04], "acc": [0.8, -0.5, 0.4]})]),
        ("Camera locks onto the wrong thing for 10 s",
         [Fault("camera_wrong_fix", 20, 10)]),
        ("Camera off 10 s while fast head turns saturate every IMU",
         [Fault("camera_blackout", 20, 10),
          Fault("imu_saturation", 20, 10, None, {"gyro_max": 0.5, "acc_max": 10.3})]),
    ]
    runs = [(title, run_case(seq, f, model, i)) for i, (title, f) in enumerate(cases)]
    S = json.loads(Path("results/summary.json").read_text())

    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    timeline = []
    timeline += [("card", ["Path Finder", "Tracking that finds the failing sensor and keeps going",
                           "Samsung PRISM XR Hackathon 2026  |  Track 3  |  Team California Burrito, SRM IST"])] * (4 * FPS)
    timeline += [("card", ["The problem", "A headset fuses cameras and IMUs to know where it is.",
                           "When a sensor fails, the virtual world slides away.",
                           "Trusting a real headset IMU alone: 1.4 m of drift in 10 s."])] * (6 * FPS)
    timeline += [("card", ["Our approach", "1  Score every camera frame before using it",
                           "2  Vote out the IMU that disagrees with the others",
                           "3  A 1.5 MB network predicts motion and how sure it is",
                           "4  A robust Kalman filter blends them by trust"])] * (7 * FPS)
    for title, r in runs:
        n = len(r["t"])
        idx = np.linspace(0, n - 1, int(16 * FPS)).astype(int)
        timeline += [("run", title, r, int(i)) for i in idx]
        timeline += [("run", title, r, n - 1)] * (2 * FPS)
    timeline += [("sweep",)] * (8 * FPS)
    timeline += [("card", ["Run it yourself", "bash reproduce.sh",
                           "github.com/rehaanajani/pathfinder-xr",
                           "All numbers from held-out TLIO headset data via evaluate.py"])] * (5 * FPS)

    def draw(frame):
        fig.clf()
        kind = frame[0]
        if kind == "card":
            lines = frame[1]
            fig.text(0.08, 0.68, lines[0], color=INK, fontsize=40, weight="bold")
            for i, l in enumerate(lines[1:]):
                fig.text(0.08, 0.55 - i * 0.08, l, color=CYAN if i == 0 and len(lines) <= 3 else MUTED, fontsize=20)
            return
        if kind == "sweep":
            sw = S["blackout_sweep"]
            ax = fig.add_axes([0.1, 0.18, 0.82, 0.62], facecolor=BG)
            xs = np.arange(len(sw["lengths_s"]))
            for off, (m, c, l) in zip((-0.2, 0.2), (("naive_fusion", ORANGE, "Standard camera + IMU fusion"),
                                                    ("ours", GREEN, "Ours"))):
                ax.bar(xs + off, sw[m], 0.38, color=c, label=l)
                for xx, v in zip(xs + off, sw[m]):
                    ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=12)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{L} s" for L in sw["lengths_s"]], color=MUTED, fontsize=14)
            ax.tick_params(colors=MUTED)
            for s_ in ax.spines.values():
                s_.set_color(LINE)
            ax.set_ylabel("error when the camera returns (m)", color=MUTED, fontsize=13)
            ax.legend(frameon=False, labelcolor=INK, fontsize=13)
            fig.text(0.1, 0.86, f"Camera blackout length vs error, {len(S['sequences'])} held-out headset recordings",
                     color=INK, fontsize=22, weight="bold")
            return
        _, title, r, i = frame
        fig.text(0.04, 0.92, title, color=INK, fontsize=24, weight="bold")
        ax = fig.add_axes([0.04, 0.06, 0.56, 0.8], facecolor=PANEL)
        allp = np.vstack([r["truth"], r["naive"], r["ours"]])
        lo, hi = allp.min(0), allp.max(0)
        pad = 0.08 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(LINE)
        fail = (r["cam"][: i + 1] < 0.3) | (r["imu"].min(0)[: i + 1] < 0.5)
        tr = r["truth"][: i + 1]
        ax.scatter(tr[fail, 0], tr[fail, 1], s=90, color=RED, alpha=0.18, lw=0)
        ax.plot(tr[:, 0], tr[:, 1], color=INK, lw=2.4, label="True path")
        ax.plot(r["naive"][: i + 1, 0], r["naive"][: i + 1, 1], color=ORANGE, lw=1.8, label="Standard fusion")
        ax.plot(r["ours"][: i + 1, 0], r["ours"][: i + 1, 1], color=GREEN, lw=2.4, label="Ours")
        for k, c in (("truth", INK), ("naive", ORANGE), ("ours", GREEN)):
            ax.plot(*r[k][i], "o", color=c, ms=8)
        ax.legend(loc="lower left", frameon=False, labelcolor=INK, fontsize=11)
        fig.text(0.64, 0.80, f"t = {r['t'][i]:4.1f} s", color=MUTED, fontsize=16)
        fig.text(0.62, 0.70, f"{r['en'][i]:.2f} m", color=ORANGE, fontsize=30, weight="bold")
        fig.text(0.64, 0.66, "standard fusion error", color=MUTED, fontsize=12)
        fig.text(0.84, 0.70, f"{r['eo'][i]:.2f} m", color=GREEN, fontsize=30, weight="bold")
        fig.text(0.84, 0.66, "our error", color=MUTED, fontsize=12)
        fig.text(0.64, 0.56, "How much we trust each sensor", color=INK, fontsize=15, weight="bold")
        bars = [("Camera", float(r["cam"][min(i, len(r["cam"]) - 1)]))] + \
               [(f"IMU {k + 1}", float(r["imu"][k, min(i, r["imu"].shape[1] - 1)])) for k in range(r["imu"].shape[0])]
        for j, (lbl, v) in enumerate(bars):
            y = 0.48 - j * 0.07
            fig.text(0.64, y, lbl, color=MUTED, fontsize=13)
            bx = fig.add_axes([0.72, y - 0.005, 0.22, 0.03])
            bx.set_xlim(0, 1)
            bx.axis("off")
            bx.barh(0, 1, color=LINE, height=1)
            bx.barh(0, v, color=RED if v < 0.3 else GREEN, height=1)

    anim = animation.FuncAnimation(fig, draw, frames=timeline, interval=1000 / FPS)
    out = Path("results/demo.mp4")
    anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p"]),
              savefig_kwargs={"facecolor": BG})
    print("wrote", out, f"{len(timeline) / FPS:.0f} s")


if __name__ == "__main__":
    main()
