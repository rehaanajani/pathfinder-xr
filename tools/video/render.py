"""Renders the narrated Track 3 film (about 2 min 55 s, 1080p, 25 fps) from real data: the organisers' damage applied
to EuRoC MH_05, the entry's own output and trajectory, the 17-recording results and the phone test.

    python tools/video/render.py --work <folder> --gt <MH_05 state_groundtruth_estimate0/data.csv> --out film.mp4

<folder> holds: clean.csv and run/imu0/data.csv (MH_05 before and after the organisers' degrade_imu.py), est.txt
(the entry's trajectory), terminal.txt (the captured run), naive.npz (error of plain integration), and audio/ with
one narration clip per scene (tools/video/narration.txt, spoken by tools/video/voice.sh). Needs imageio-ffmpeg.
"""
import argparse
import math
import subprocess
import wave
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 25
SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
INK = (240, 243, 250)
MUTED = (150, 162, 190)
CYAN = (90, 200, 255)
BLUE = (70, 120, 255)
VIOLET = (150, 90, 255)
GREEN = (60, 230, 170)
RED = (255, 95, 90)
AMBER = (255, 170, 70)
WORK = None
DATA = {}


# ----------------------------------------------------------------------------- helpers
def font(size, weight="Regular", mono=False):
    f = ImageFont.truetype(MONO if mono else SANS, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def fade(t, start, dur=0.6):
    return ease((t - start) / dur)


def background(t, hue=(BLUE, VIOLET), intensity=1.0, seed=0):
    """Deep navy with two slowly drifting light glows and a star field (cinematic, like a product film)."""
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    img = np.zeros((H // 4, W // 4, 3), np.float32)
    img[:] = (2, 3, 9)
    for k, col in enumerate(hue):
        cx = W / 8 * (1 + 0.55 * math.sin(0.11 * t + k * 2.4 + seed))
        cy = H / 8 * (1 + 0.45 * math.cos(0.09 * t + k * 1.7 + seed))
        r = (W / 4) * (0.42 + 0.06 * math.sin(0.2 * t + k))
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
        img += g ** 1.6 * np.array(col, np.float32) * 0.17 * intensity
    vig = 1 - 0.55 * (((xx - W / 8) / (W / 8)) ** 2 + ((yy - H / 8) / (H / 8)) ** 2) / 2
    img *= np.clip(vig, 0.25, 1)[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    rng = np.random.default_rng(7 + seed)
    stars = rng.random((260, 3))
    d = ImageDraw.Draw(im)
    for sx, sy, sb in stars:
        x = (sx * W + 8 * t * (0.3 + sb)) % W
        y = sy * H
        b = int(60 + 120 * sb * (0.6 + 0.4 * math.sin(t * 1.3 + sx * 40)))
        d.point((x, y), fill=(b, b, min(255, b + 30)))
    return im


def glow_layer(draw_fn, radius=12, strength=1.6):
    """Draw on a transparent layer, return (sharp, blurred) for additive glow."""
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    small = layer.resize((W // 2, H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius / 2))
    blur = small.resize((W, H), Image.BILINEAR)
    return layer, blur, strength


def add(base, layers):
    a = np.asarray(base, np.float32)
    for sharp, blur, s in layers:
        a = a + np.asarray(blur, np.float32) * s + np.asarray(sharp, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def text(d, xy, s, size, weight="Regular", fill=INK, alpha=1.0, anchor="la", mono=False):
    if alpha <= 0:
        return
    c = tuple(int(v * alpha) for v in fill)
    d.text(xy, s, font=font(size, weight, mono), fill=c, anchor=anchor)


KOREAN = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
KO_WEIGHT = {"Regular": 0, "Medium": 2, "SemiBold": 4, "Bold": 6, "Light": 8}


def ko_text(d, xy, s, size, weight="SemiBold", fill=INK, alpha=1.0, anchor="la"):
    """Korean text (Apple SD Gothic Neo)."""
    if alpha <= 0:
        return
    f = ImageFont.truetype(KOREAN, size, index=KO_WEIGHT[weight])
    d.text(xy, s, font=f, fill=tuple(int(v * alpha) for v in fill), anchor=anchor)


def title_block(im, t, start, lines, y=150, x=140):
    """Minimal headline that fades and rises in."""
    d = ImageDraw.Draw(im)
    for i, (s, size, weight, col) in enumerate(lines):
        a = fade(t, start + 0.25 * i)
        text(d, (x, y + (1 - a) * 18), s, size, weight, col, a)
        y += int(size * 1.25)


def subtitle(im, s, t, dur):
    if not s:
        return im
    a = min(fade(t, 0.1, 0.4), 1 - fade(t, dur - 0.35, 0.35))
    d = ImageDraw.Draw(im, "RGBA")
    f = font(30, "Regular")
    words, lines, cur = s.split(), [], ""
    for w in words:
        if d.textlength(cur + " " + w, font=f) > 1300 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    y = H - 60 - 40 * len(lines)
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.rounded_rectangle([(W - tw) / 2 - 14, y - 4, (W + tw) / 2 + 14, y + 38], 8, fill=(0, 0, 0, int(140 * a)))
        d.text(((W - tw) / 2, y), ln, font=f, fill=(235, 238, 245, int(255 * a)))
        y += 42
    return im


def polyline(d, pts, col, width):
    if len(pts) > 1:
        d.line([tuple(p) for p in pts], fill=col, width=width, joint="curve")


_FORMULA = {}


def formula(tex, size=34, color=(225, 232, 248)):
    """A LaTeX-style formula (matplotlib mathtext) as an RGBA image, cached."""
    key = (tex, size, color)
    if key not in _FORMULA:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        fig = plt.figure(figsize=(0.01, 0.01), dpi=100)
        fig.text(0, 0, f"${tex}$", fontsize=size, color=tuple(c / 255 for c in color))
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        _FORMULA[key] = Image.open(buf).convert("RGBA")
    return _FORMULA[key]


def put_formulas(im, t, items, x, y, gap=22, start=0.8, step=1.4, size=34, anchor_center=False):
    """Fade in a short stack of formulas, one after another."""
    for i, tex in enumerate(items):
        a = fade(t, start + step * i, 0.7)
        if a <= 0:
            continue
        f = formula(tex, size)
        fx = int(x - f.width / 2) if anchor_center else int(x)
        alpha = f.getchannel("A").point(lambda v: int(v * a))
        f2 = f.copy()
        f2.putalpha(alpha)
        im.paste(f2, (fx, int(y + (1 - a) * 10)), f2)
        y += f.height + gap
    return im


# ----------------------------------------------------------------------------- data
def load():
    import numpy as np
    w = WORK
    clean = np.loadtxt(w / "clean.csv", delimiter=",", skiprows=1)
    dmg = np.loadtxt(w / "run" / "imu0" / "data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(DATA["gt_path"], delimiter=",", comments="#")
    est = np.loadtxt(w / "est.txt", comments="#")
    t0 = clean[0, 0]
    DATA.update(
        c_t=(clean[:, 0] - t0) * 1e-9, c_a=clean[:, 1],
        d_t=(dmg[:, 0] - t0) * 1e-9, d_a=dmg[:, 1],
        gt_t=gt[:, 0] * 1e-9, gt_p=gt[:, 1:4], est_t=est[:, 0], est_p=est[:, 1:4],
        term=(w / "terminal.txt").read_text().splitlines(), naive=np.load(w / "naive.npz"))
    # SE(3)-align the estimate to ground truth (as evo does)
    t, p = DATA["est_t"], DATA["est_p"]
    tg, g = DATA["gt_t"], DATA["gt_p"]
    ok = (t >= tg[0]) & (t <= tg[-1])
    t, p = t[ok], p[ok]
    gi = np.stack([np.interp(t, tg, g[:, k]) for k in range(3)], 1)
    ms, md = p.mean(0), gi.mean(0)
    U, _, Vt = np.linalg.svd((gi - md).T @ (p - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ S @ Vt
    DATA.update(al_t=t, al_p=p @ Rm.T + (md - Rm @ ms), al_g=gi)


# ----------------------------------------------------------------------------- scenes
def sc_intro(t, dur):
    im = background(t, (BLUE, CYAN))
    g = DATA["al_g"]
    xy = g[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    sc = 700 / max(hi - lo)
    pts = (xy - (lo + hi) / 2) * sc * np.array([1, -1]) + np.array([W / 2 + 330, H / 2 + 20])
    n = int(len(pts) * ease(t / (dur * 0.95)))
    pts = pts[: max(n, 2): 6]

    def draw(d):
        polyline(d, pts, (120, 200, 255), 3)
        if len(pts):
            x, y = pts[-1]
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 255))
    im = add(im, [glow_layer(draw, 16, 2.2)])
    title_block(im, t, 0.6, [("One chip knows", 76, "Semibold", INK), ("where you are.", 76, "Semibold", INK),
                             ("The IMU: every turn, every step.", 34, "Regular", MUTED)], y=360)
    return im


def sc_jammed(t, dur):
    im = background(t, (VIOLET, RED), 0.8)
    ct, ca, dt_, da = DATA["c_t"], DATA["c_a"], DATA["d_t"], DATA["d_a"]
    T = ct[-1]
    prog = ease(t / (dur * 0.92)) * T
    x0, x1, yc, sy = 140, W - 140, 600, 22

    def X(tt):
        return x0 + (x1 - x0) * tt / T

    def draw(d):
        m = ct <= prog
        polyline(d, np.c_[X(ct[m][::8]), yc - 230 - sy * 0.6 * (ca[m][::8] - np.median(ca))], (60, 110, 160), 2)
        m = dt_ <= prog
        tt, aa = dt_[m], da[m]
        gaps = np.flatnonzero(np.diff(tt) > 0.02)
        segs = np.split(np.arange(len(tt)), gaps + 1)
        for s in segs:
            s = s[::3] if len(s) > 6 else s
            if len(s) > 1:
                col = RED if tt[s[0]] > T - 5 else (255, int(200 - 120 * tt[s[0]] / T), int(120 - 80 * tt[s[0]] / T))
                polyline(d, np.c_[X(tt[s]), yc + 80 - sy * np.clip(aa[s] - np.median(ca), -9, 9)], col, 2)
        for g in gaps:
            if tt[g + 1] - tt[g] > 0.08:
                d.rectangle([X(tt[g]), yc - 80, X(tt[g + 1]), yc + 240], fill=(60, 20, 30))
    im = add(im, [glow_layer(draw, 10, 1.3)])
    d = ImageDraw.Draw(im)
    text(d, (x0, yc - 330), "clean", 24, "Medium", (90, 140, 200), fade(t, 0.3))
    text(d, (x0, yc - 40), "jammed: the organisers' degradation, real data", 24, "Medium", AMBER, fade(t, 0.5))
    tags = [("noise", 0.15), ("spikes", 0.35), ("frozen", 0.55), ("dropped", 0.72), ("dead", 0.93)]
    for i, (s, at) in enumerate(tags):
        a = fade(t, at * dur * 0.92)
        text(d, (X(at * T) - 20, yc + 280), s, 30, "Semibold", RED if s == "dead" else INK, a)
    title_block(im, t, 0.2, [("What if it is being jammed?", 56, "Semibold", INK)], y=110)
    return im


def sc_explode(t, dur):
    im = background(t, (RED, VIOLET), 0.9)
    nv = DATA["naive"]
    tt, err = nv["t"], nv["err"]
    k = ease_out(t / (dur * 0.8))
    upto = tt[-1] * k
    m = tt <= upto
    x0, x1, yb, yt = 160, 1200, 900, 300
    lg = np.log10(np.maximum(err, 1))

    def draw(d):
        if m.sum() > 2:
            xs = x0 + (x1 - x0) * tt[m][::10] / tt[-1]
            ys = yb - (yb - yt) * lg[m][::10] / 5.0
            polyline(d, np.c_[xs, ys], RED, 4)
    im = add(im, [glow_layer(draw, 14, 2.0)])
    d = ImageDraw.Draw(im)
    for e, lab in ((1, "10 m"), (2, "100 m"), (3, "1 km"), (4, "10 km")):
        y = yb - (yb - yt) * e / 5.0
        d.line([x0, y, x1, y], fill=(40, 46, 70), width=1)
        text(d, (x0 - 20, y), lab, 20, "Regular", MUTED, 1, anchor="rm")
    cur = float(err[m][-1]) if m.any() else 0.0
    text(d, (1290, 430), f"{cur / 1000:,.1f} km" if cur > 1000 else f"{cur:,.0f} m", 120, "Bold", RED, fade(t, 0.2))
    text(d, (1295, 580), "off, after " + f"{upto:,.0f} s", 34, "Regular", MUTED, fade(t, 0.3))
    text(d, (1295, 640), "integrating the IMU directly (clean data drifts the same)", 24, "Regular", MUTED, fade(t, 0.5))
    title_block(im, t, 0.1, [("Trust it, and the error explodes.", 56, "Semibold", INK)], y=110, x=160)
    put_formulas(im, t, [r"\mathbf{p}(t)=\iint\left(R\,\mathbf{a}-\mathbf{g}\right)\,dt^{2}",
                         r"\delta\theta=1^{\circ}\ \Rightarrow\ g\,\sin\delta\theta\approx0.17\ \mathrm{m/s^{2}}",
                         r"\frac{1}{2}\,(0.17)\,(110\,\mathrm{s})^{2}\approx1\ \mathrm{km}"],
                 1295, 700, start=2.5, step=1.8, size=32)
    return im


def sc_title(t, dur):
    im = background(t, (BLUE, VIOLET), 1.4)
    burst = ease_out(t / 1.2)

    def draw(d):
        cx, cy = W / 2, H / 2 - 20
        for i in range(48):
            ang = i / 48 * 2 * math.pi + 0.05 * t
            r1, r2 = 60 + 30 * math.sin(i * 3.1), 220 + 520 * burst * (0.6 + 0.4 * math.sin(i * 7.7))
            c = (110, 150, 255) if i % 3 else (190, 140, 255)
            d.line([cx + r1 * math.cos(ang), cy + r1 * math.sin(ang), cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)],
                   fill=tuple(int(v * 0.35) for v in c), width=2)
    im = add(im, [glow_layer(draw, 40, 1.8)])
    core = Image.new("RGB", (W, H))
    ImageDraw.Draw(core).ellipse([W / 2 - 260, H / 2 - 200, W / 2 + 260, H / 2 + 160], fill=(40, 55, 120))
    core = core.filter(ImageFilter.GaussianBlur(90))
    im = Image.fromarray(np.clip(np.asarray(im, np.float32) + np.asarray(core, np.float32) * burst, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    a = fade(t, 0.3, 0.8)
    ko_text(d, (W / 2, H / 2 - 80), "길잡이", 170, "Bold", INK, a, anchor="mm")
    text(d, (W / 2, H / 2 + 52), "Giljabi  ·  \u201cthe one who finds the way\u201d", 44, "Medium", CYAN, fade(t, 0.7), anchor="mm")
    text(d, (W / 2, H / 2 + 120), "Path Finder track  ·  trajectory from a degraded IMU alone", 34, "Light", (200, 210, 235),
         fade(t, 0.9), anchor="mm")
    return im


STAGES = [("Check", "every sample"), ("Gravity & bias", "from the data"), ("Physics", "with brakes"),
          ("Learned velocity", "42,716 parameters"), ("Coast", "when the IMU dies")]


def pipeline(im, t, active):
    d = ImageDraw.Draw(im)
    xs = [260 + i * 350 for i in range(5)]
    y = 250

    def draw(dd):
        for i, x in enumerate(xs):
            on = i == active
            done = i < active
            col = (CYAN if on else (GREEN if done else (60, 70, 100)))
            r = 34 if on else 24
            dd.ellipse([x - r, y - r, x + r, y + r], outline=col, width=4 if on else 2)
            if on or done:
                dd.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
            if i < 4:
                dd.line([x + 40, y, xs[i + 1] - 40, y], fill=(50, 60, 90) if i >= active else (60, 160, 140), width=2)
    im2 = add(im, [glow_layer(draw, 12, 1.2)])
    d = ImageDraw.Draw(im2)
    for i, x in enumerate(xs):
        on = i == active
        text(d, (x, y + 62), STAGES[i][0], 30 if on else 24, "Semibold" if on else "Regular",
             INK if on else MUTED, 1.0, anchor="mm")
        text(d, (x, y + 98), STAGES[i][1], 21, "Regular", MUTED if on else (90, 100, 130), 1.0, anchor="mm")
    return im2


def sc_check(t, dur):
    im = pipeline(background(t, (BLUE, CYAN), 0.8), t, 0)
    rng = np.random.default_rng(3)
    n = 22
    kinds = rng.choice(["ok"] * 7 + ["nan", "frozen", "spike", "gap"], n)
    labels = {"nan": "NaN", "frozen": "frozen", "spike": "spike", "gap": "gap"}

    def draw(d):
        for i in range(n):
            x = 120 + ((t * 230 + i * 80) % 1760)
            k = kinds[i]
            y = 640
            if k != "ok" and x > 960:
                y += (x - 960) * 0.55
            col = GREEN if k == "ok" else RED
            d.rounded_rectangle([x - 16, y - 16, x + 16, y + 16], 6, outline=col, width=3)
        d.line([960, 560, 960, 720], fill=CYAN, width=3)
    im = add(im, [glow_layer(draw, 10, 1.3)])
    d = ImageDraw.Draw(im)
    for i in range(n):
        x = 120 + ((t * 230 + i * 80) % 1760)
        k = kinds[i]
        if k != "ok" and x > 960:
            text(d, (x + 26, 640 + (x - 960) * 0.55), labels[k], 26, "Semibold", RED, 1.0, anchor="lm")
    text(d, (960, 520), "health check, every row", 24, "Medium", CYAN, fade(t, 0.3), anchor="mm")
    return im


def sc_gravity(t, dur):
    im = pipeline(background(t, (BLUE, VIOLET), 0.8), t, 1)
    k = ease_out(t / (dur * 0.7))
    cx, cy = 960, 720
    wob = (1 - k) * 0.6 * math.sin(6 * t)
    ang = -math.pi / 2 + wob

    def draw(d):
        L = 190
        ex, ey = cx + L * math.cos(ang), cy + L * math.sin(ang)
        d.line([cx, cy, ex, ey], fill=GREEN, width=8)
        d.polygon([(ex, ey - 4), (ex - 22 * math.cos(ang + 0.5), ey - 22 * math.sin(ang + 0.5)),
                   (ex - 22 * math.cos(ang - 0.5), ey - 22 * math.sin(ang - 0.5))], fill=GREEN)
        d.ellipse([cx - 160, cy - 40, cx + 160, cy + 40], outline=(80, 100, 150), width=2)
    im = add(im, [glow_layer(draw, 14, 1.5)])
    d = ImageDraw.Draw(im)
    text(d, (cx + 230, cy - 120), "up, found from the data", 30, "Medium", INK, fade(t, 0.6))
    text(d, (cx + 230, cy - 75), "gyro bias, estimated while still", 26, "Regular", MUTED, fade(t, 1.0))
    text(d, (cx + 230, cy - 35), "no ground truth, no calibration", 26, "Regular", MUTED, fade(t, 1.4))
    put_formulas(im, t, [r"\hat{\mathbf{g}}=\frac{1}{N}\sum_{\mathrm{still}}\mathbf{a}_i,\qquad \mathbf{b}_{\omega}=\frac{1}{N}\sum_{\mathrm{still}}\boldsymbol{\omega}_i",
                         r"\mathbf{e}=\hat{\mathbf{a}}\times R^{\top}\hat{\mathbf{z}},\qquad \boldsymbol{\omega}\leftarrow\boldsymbol{\omega}-\mathbf{b}_{\omega}+k_p\,\mathbf{e}"],
                 cx + 200, cy + 30, start=1.6, step=1.6, size=26)
    return im


def sc_physics(t, dur):
    im = pipeline(background(t, (BLUE, CYAN), 0.8), t, 2)
    d = ImageDraw.Draw(im)
    items = [("Gravity correction", "tilt fixed from the accelerometer when it can be trusted"),
             ("Still detection", "velocity pulled to zero when you are not moving"),
             ("Speed limit", "no impossible speeds"),
             ("Pull to the start", "drift cannot run away forever")]
    for i, (h, s) in enumerate(items):
        a = fade(t, 0.4 + 1.2 * i)
        y = 470 + i * 115
        text(d, (520, y + (1 - a) * 14), h, 40, "Semibold", INK, a)
        text(d, (520, y + 50 + (1 - a) * 14), s, 26, "Regular", MUTED, a)
        d.rounded_rectangle([470, y + 10, 478, y + 70], 3, fill=tuple(int(v * a) for v in CYAN))
    put_formulas(im, t, [r"q_{k+1}=q_k\otimes\exp\!\left(\frac{1}{2}\,\boldsymbol{\omega}\,\Delta t\right)",
                         r"\mathbf{v}\leftarrow\mathbf{v}+\left(R\,\mathbf{a}-\mathbf{g}\right)\Delta t",
                         r"\|\boldsymbol{\omega}\|<\epsilon\ \Rightarrow\ \mathbf{v}\rightarrow\mathbf{0}",
                         r"\mathbf{p}\leftarrow\mathbf{p}\,e^{-\Delta t/\tau}"],
                 1180, 470, gap=34, start=0.6, step=1.25, size=34)
    return im


def sc_network(t, dur):
    im = pipeline(background(t, (VIOLET, BLUE), 0.9), t, 3)
    layers = [10, 32, 48, 64, 64, 6]
    xs = [360 + i * 250 for i in range(len(layers))]
    k = t / dur
    rng = np.random.default_rng(1)

    def draw(d):
        prev = None
        for li, (x, n) in enumerate(zip(xs, layers)):
            m = min(n, 12)
            ys = [700 + (j - (m - 1) / 2) * 32 for j in range(m)]
            if prev is not None:
                for y0 in prev[1][::2]:
                    for y1 in ys[::2]:
                        pulse = 0.5 + 0.5 * math.sin(6 * t - li * 1.2 + y0 * 0.03)
                        c = int(30 + 60 * pulse * (k > li / len(layers)))
                        d.line([prev[0], y0, x, y1], fill=(c // 2, c // 2, c), width=1)
            for y in ys:
                lit = k > li / len(layers)
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(VIOLET if li < len(layers) - 1 else GREEN) if lit else (50, 55, 80))
            prev = (x, ys)
    im = add(im, [glow_layer(draw, 10, 1.4)])
    d = ImageDraw.Draw(im)
    text(d, (xs[0], 460), "1 s of IMU", 26, "Medium", INK, fade(t, 0.3), anchor="mm")
    text(d, (xs[-1], 460), "velocity + how sure", 26, "Medium", GREEN, fade(t, 2.5), anchor="mm")
    text(d, (W / 2, 960), "runs inside main.py: no ML runtime", 26, "Regular", MUTED, fade(t, 4.0), anchor="mm")
    put_formulas(im, t, [r"(\hat{\mathbf{v}},\,\sigma)=f_{\theta}(\mathbf{x}_{t-1\,\mathrm{s}\,:\,t}),\qquad w=\frac{\sigma_{0}^{2}}{\sigma_{0}^{2}+\sigma^{2}}"],
                 W / 2, 405, start=3.0, size=30, anchor_center=True)
    return im


def sc_coast(t, dur):
    im = pipeline(background(t, (BLUE, VIOLET), 0.8), t, 4)
    k = ease(t / (dur * 0.9))

    def draw(d):
        pts = [(300 + 12 * i, 720 - 80 * math.sin(i / 14)) for i in range(70)]
        polyline(d, pts, GREEN, 5)
        cut = pts[-1]
        n = int(50 * k)
        for i in range(0, n, 2):
            a = (300 + 12 * (70 + i), 720 - 80 * math.sin((70 + i) / 14) * math.exp(-i / 30))
            b = (300 + 12 * (71 + i), 720 - 80 * math.sin((71 + i) / 14) * math.exp(-(i + 1) / 30))
            d.line([a, b], fill=(255, 200, 120), width=5)
        d.line([cut[0], 560, cut[0], 880], fill=RED, width=2)
    im = add(im, [glow_layer(draw, 12, 1.5)])
    d = ImageDraw.Draw(im)
    text(d, (300 + 12 * 70, 540), "IMU dies", 26, "Semibold", RED, fade(t, 0.2), anchor="mm")
    text(d, (1450, 800), "5 s blind:", 30, "Semibold", INK, fade(t, 0.8))
    text(d, (1450, 845), "turn rate and speed decay", 24, "Regular", MUTED, fade(t, 1.0))
    text(d, (1450, 880), "from the last trusted state", 24, "Regular", MUTED, fade(t, 1.2))
    text(d, (1450, 650), "On a drone, also:", 30, "Semibold", INK, fade(t, 3.0))
    text(d, (1450, 695), "rotor drag measures sideways speed", 24, "Regular", MUTED, fade(t, 3.4))
    text(d, (1450, 730), "the whole flight smoothed offline", 24, "Regular", MUTED, fade(t, 3.8))
    text(d, (1450, 765), "it lands where it took off", 24, "Regular", GREEN, fade(t, 4.2))
    put_formulas(im, t, [r"\boldsymbol{\omega}(t)=\boldsymbol{\omega}_0\,e^{-t/\tau_{\omega}},\quad \mathbf{v}(t)=\mathbf{v}_0\,e^{-t/\tau_{v}}"],
                 1120, 470, start=1.0, size=30)
    return im


def ring(d, cx, cy, r, frac, col, width=14):
    d.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=(35, 40, 60), width=width)
    if frac > 0:
        d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * frac, fill=col, width=width)


def sc_budget(t, dur):
    im = background(t, (BLUE, CYAN), 1.0)
    k = ease_out(t / 3.0)
    mem, cpu = 3.5 * k, 5.8 * k

    def draw(d):
        ring(d, 640, 520, 190, mem / 8, GREEN)
        ring(d, 1280, 520, 190, cpu / 20, CYAN)
    im = add(im, [glow_layer(draw, 12, 1.3)])
    d = ImageDraw.Draw(im)
    text(d, (640, 500), f"{mem:.1f} MB", 72, "Bold", INK, 1, anchor="mm")
    text(d, (640, 570), "of 8 MB memory", 26, "Regular", MUTED, 1, anchor="mm")
    text(d, (1280, 500), f"{cpu:.1f} ms", 72, "Bold", INK, 1, anchor="mm")
    text(d, (1280, 570), "per second of IMU, of 20", 26, "Regular", MUTED, 1, anchor="mm")
    title_block(im, t, 0.1, [("Tiny, and fast.", 56, "Semibold", INK)], y=110)
    a = fade(t, dur - 5.5)
    text(d, (W / 2, 830), "Its C++ port on a Galaxy phone's own CPU:  4.0 MB  ·  5.0 ms", 40, "Semibold", GREEN, a, anchor="mm")
    text(d, (W / 2, 885), "one Python file (main.py), one thread, network inside, no ML runtime", 26, "Regular", MUTED, fade(t, 2.0), anchor="mm")
    return im


def sc_terminal(t, dur):
    im = background(t, (BLUE, VIOLET), 0.7)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([110, 110, W - 110, H - 150], 22, fill=(12, 16, 30, 230), outline=(60, 70, 110, 255), width=2)
    for i, c in enumerate((RED, AMBER, GREEN)):
        d.ellipse([145 + i * 36, 140, 167 + i * 36, 162], fill=c)
    lines = DATA["term"]
    total = sum(len(s) for s in lines)
    shown = int(total * ease(t / (dur * 0.85)))
    y = 195
    f = font(25, mono=True)
    for s in lines:
        if shown <= 0:
            break
        part = s[:shown]
        shown -= len(s)
        col = CYAN if s.startswith("$") else (GREEN if ("wrote" in s or "resident" in s or " real" in s) else (190, 198, 215))
        d.text((160, y), part, font=f, fill=col)
        y += 34
    return im


def sc_results(t, dur):
    im = background(t, (BLUE, GREEN), 0.7)
    g, p, tt = DATA["al_g"], DATA["al_p"], DATA["al_t"]
    lo, hi = g[:, :2].min(0), g[:, :2].max(0)
    sc = 640 / max(hi - lo)
    off = np.array([520, 560])

    def P(a):
        return (a[:, :2] - (lo + hi) / 2) * sc * np.array([1, -1]) + off
    k = ease(t / (dur * 0.5))
    n = max(2, int(len(tt) * k))
    gp, pp = P(g[:n:5]), P(p[:n:5])

    def draw(d):
        polyline(d, gp, (210, 220, 240), 3)
        polyline(d, pp, GREEN, 4)
    im = add(im, [glow_layer(draw, 12, 1.4)])
    d = ImageDraw.Draw(im)
    text(d, (200, 160), "EuRoC MH_05, never trained on", 30, "Semibold", INK, fade(t, 0.2))
    text(d, (200, 200), "white: motion capture   green: Giljabi, IMU only", 24, "Regular", MUTED, fade(t, 0.4))
    # bars: 17 recordings, organisers' damage
    metrics = [("ATE", 2.62, 1.47), ("Checkpoints", 2.45, 1.35), ("Dropouts", 2.42, 1.36), ("Blind 5 s", 2.28, 1.37),
               ("Drift / 1 s", 0.77, 0.42)]
    a0 = fade(t, 1.5)
    text(d, (1060, 230), "All 17 public recordings, organisers' damage", 30, "Semibold", INK, a0)
    text(d, (1060, 270), "metres, lower is better", 22, "Regular", MUTED, a0)
    for i, (m, still, ours) in enumerate(metrics):
        y = 340 + i * 110
        kk = ease_out((t - 2.0 - 0.3 * i) / 1.2)
        sw, ow = 600 * still / 2.8 * kk, 600 * ours / 2.8 * kk
        text(d, (1060, y), m, 28, "Semibold", INK, a0)
        d.rounded_rectangle([1060, y + 34, 1060 + max(sw, 2), y + 52], 6, fill=(90, 98, 125))
        d.rounded_rectangle([1060, y + 60, 1060 + max(ow, 2), y + 78], 6, fill=GREEN)
        if kk > 0.95:
            text(d, (1070 + sw, y + 43), f"{still:.2f} still", 22, "Regular", MUTED, 1, anchor="lm")
            text(d, (1070 + ow, y + 69), f"{ours:.2f} ours", 22, "Semibold", GREEN, 1, anchor="lm")
    a = fade(t, dur - 7.0)
    text(d, (1060, 920), "45 % lower at checkpoints, 44 % through dropouts, 40 % in the blind 5 s.", 26, "Semibold", AMBER, a)
    return im


def sc_phone(t, dur):
    im = background(t, (VIOLET, BLUE), 0.9)
    k = ease_out((t - 1.0) / 4.0)
    std, ours = 227.25 * k, 3.51 * k
    scale = 1500 / 227.25

    def draw(d):
        d.rounded_rectangle([200, 520, 200 + max(4, std * scale), 560], 10, fill=RED)
        d.rounded_rectangle([200, 700, 200 + max(4, ours * scale), 740], 10, fill=GREEN)
        d.rounded_rectangle([200, 860, 200 + max(4, 1.78 * k * scale), 900], 10, fill=(200, 205, 220))
    im = add(im, [glow_layer(draw, 14, 1.4)])
    d = ImageDraw.Draw(im)
    title_block(im, t, 0.1, [("On a real Galaxy phone.", 56, "Semibold", INK),
                             ("Small room, camera covered for 48 s while walking.", 30, "Regular", MUTED)], y=140, x=200)
    text(d, (200, 470), "standard tracker", 28, "Medium", INK, fade(t, 0.8))
    text(d, (200, 650), "Giljabi", 28, "Medium", INK, fade(t, 0.8))
    text(d, (220 + std * scale, 540), f"{std:.0f} m", 44, "Bold", RED, fade(t, 1.0), anchor="lm")
    text(d, (230 + ours * scale, 720), f"{ours:.1f} m", 44, "Bold", GREEN, fade(t, 1.0), anchor="lm")
    text(d, (200, 810), "freezing in place", 28, "Medium", INK, fade(t, 0.8))
    text(d, (230 + 1.78 * k * scale, 880), f"{1.78 * k:.1f} m", 44, "Bold", (215, 220, 232), fade(t, 1.0), anchor="lm")
    text(d, (200, 950), "error against ARCore when the camera came back. Room about 6 x 9 m; Galaxy M36, live", 22, "Regular", MUTED, fade(t, 2))
    return im


def sc_outro(t, dur):
    im = background(t, (BLUE, VIOLET), 1.2)
    d = ImageDraw.Draw(im)
    ko_text(d, (W / 2, 370), "길잡이", 140, "Bold", INK, fade(t, 0.2, 0.8), anchor="mm")
    text(d, (W / 2, 468), "Giljabi  ·  \u201cthe one who finds the way\u201d", 36, "Medium", CYAN, fade(t, 0.6), anchor="mm")
    text(d, (W / 2, 545), "Know where you are, even when your sensors lie.", 40, "Light", (205, 215, 240), fade(t, 1.0), anchor="mm")
    text(d, (W / 2, 660), "Team California Burrito  ·  SRM Institute of Science and Technology", 30, "Medium", INK,
         fade(t, 2.5), anchor="mm")
    text(d, (W / 2, 710), "Rehaan Ajani  ·  Nikhil Solanki  ·  Debadrita Haldar  ·  Mayank Yadav", 26, "Regular", MUTED,
         fade(t, 3.0), anchor="mm")
    text(d, (W / 2, 800), "github.com/rehaanajani/pathfinder-xr", 30, "Medium", CYAN, fade(t, 3.6), anchor="mm")
    text(d, (W / 2, 850), "Samsung PRISM XR Hackathon 2026  ·  Track 3, Path Finder", 22, "Regular", MUTED, fade(t, 3.8), anchor="mm")
    return im


SCENES = [sc_intro, sc_jammed, sc_explode, sc_title, sc_check, sc_gravity, sc_physics, sc_network, sc_coast,
          sc_budget, sc_terminal, sc_results, sc_phone, sc_outro]
PAD = [0.8, 0.8, 0.8, 1.2, 0.5, 0.5, 0.6, 0.6, 0.8, 0.8, 1.5, 1.2, 1.0, 2.5]


# ----------------------------------------------------------------------------- audio
def read_audio(path, ffmpeg, sr=48000):
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def music(n, sr=48000):
    """A quiet ambient pad: slow chords of soft sines with a gentle swell."""
    t = np.arange(n) / sr
    chords = [(110.0, 164.81, 220.0, 277.18), (98.0, 146.83, 196.0, 246.94), (87.31, 130.81, 174.61, 261.63),
              (98.0, 146.83, 196.0, 293.66)]
    out = np.zeros(n, np.float32)
    seg = 8.0
    for i, ch in enumerate(chords * int(n / sr / seg / 4 + 2)):
        s0 = int(i * seg * sr)
        if s0 >= n:
            break
        s1 = min(n, int((i + 1.5) * seg * sr))
        tt = t[s0:s1] - t[s0]
        env = np.sin(np.pi * np.clip(tt / (1.5 * seg), 0, 1)) ** 2
        for f in ch:
            out[s0:s1] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + 0.3)) * env * 0.05
    out *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 23) ** 2
    return out / max(1e-6, np.abs(out).max()) * 0.12


# ----------------------------------------------------------------------------- render
def render_scene(args):
    i, dur, sub, out, work, gt = args
    global WORK
    WORK = Path(work)
    DATA["gt_path"] = gt
    load()
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(round(dur * FPS))
    for f in range(nf):
        t = f / FPS
        im = SCENES[i](t, dur)
        a = min(1.0, t / 0.4, (dur - t) / 0.4)          # fade through black between scenes
        if a < 1:
            im = Image.fromarray((np.asarray(im, np.float32) * max(a, 0)).astype(np.uint8))
        im = subtitle(im, sub, t, dur)
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--only", type=int, default=None, help="render one scene (for checking)")
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    work = Path(a.work)
    lines = (work / "narration.txt").read_text().splitlines()
    sr = 48000
    clips, durs = [], []
    for i in range(len(SCENES)):
        alt = {2: "a02", 11: "a11", 12: "a12"}
        files = sorted((work / alt[i]).glob("s*.caf")) if i in alt and (work / alt[i]).is_dir() else sorted((work / "audio").glob(f"s{i:02d}*.caf"))
        voice = np.concatenate([read_audio(f, ff, sr) for f in files])
        lead = int(0.35 * sr)
        clip = np.concatenate([np.zeros(lead, np.float32), voice, np.zeros(int(PAD[i] * sr), np.float32)])
        clips.append(clip)
        durs.append(len(clip) / sr)
    seg_dir = work / "segments"
    seg_dir.mkdir(exist_ok=True)
    jobs = [(i, durs[i], lines[i], str(seg_dir / f"seg{i:02d}.mp4"), str(work), a.gt)
            for i in range(len(SCENES)) if a.only is None or i == a.only]
    with Pool(a.jobs) as pool:
        for i in pool.imap_unordered(render_scene, jobs):
            print("scene", i, "done", flush=True)
    if a.only is not None:
        return
    voice = np.concatenate(clips)
    mix = voice * 0.95 + music(len(voice), sr)
    mix = np.clip(mix, -1, 1)
    wav = work / "mix.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{seg_dir / f'seg{i:02d}.mp4'}'\n" for i in range(len(SCENES))))
    subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], check=True)
    print(f"wrote {a.out}: {sum(durs):.1f} s")


if __name__ == "__main__":
    main()
