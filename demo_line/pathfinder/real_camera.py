"""Real camera frames: health score + real feature tracking (OpenCV KLT) on TUM-VI images.

Degradations (darkness, motion blur, blackout) are applied to the *real* frames. Two things are
then measured on each degraded frame:
  * our health score (camera.image_health), computed before fusion;
  * real trackability: Shi-Tomasi corners tracked into the next frame with pyramidal
    Lucas-Kanade, kept only if they pass a forward-backward check.

Visual position fixes are still built from motion-capture truth (we do not run a full SLAM
back end), but their error now follows what the real tracker achieves on the degraded frames:
fewer surviving tracks -> noisier fixes that drift and jump.
"""

import cv2
import numpy as np

from .camera import FIX_STD, image_health
from .faults import camera_schedule


def load_gray(path, size=256):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def degrade_real(img, brightness, blur_px, rng):
    out = img * brightness
    k = int(round(blur_px * img.shape[1] / 160))         # blur defined at 160 px width, rescale
    if k >= 2:
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k                          # horizontal motion blur
        out = cv2.filter2D(out, -1, kernel)
    noise = 0.01 + 0.03 * (1 - brightness)
    return np.clip(out + rng.normal(0, noise, out.shape).astype(np.float32), 0, 1)


def track_count(prev, cur, max_corners=150):
    """Number of corners in prev that KLT tracks into cur and back within 1 px."""
    a = (prev * 255).astype(np.uint8)
    b = (cur * 255).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(a, max_corners, 0.01, 7)
    if pts is None or len(pts) == 0:
        return 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(15, 15), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(15, 15), maxLevel=3)
    fb = np.linalg.norm(back - pts, axis=2).ravel()
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    return int(ok.sum())


def correct_tracks(clean_a, clean_b, deg_a, deg_b, max_corners=150):
    """Corners found on the CLEAN frame; tracked on clean and on degraded frames. A degraded track is
    correct if it passes forward-backward and lands within 2 px of the clean track. Returns
    (correct, reference) counts; correct/reference is the real trackability that survives."""
    u = lambda x: (x * 255).astype(np.uint8)
    a, b, da, db = u(clean_a), u(clean_b), u(deg_a), u(deg_b)
    pts = cv2.goodFeaturesToTrack(a, max_corners, 0.01, 7)
    if pts is None:
        return 0, 0
    lk = dict(winSize=(15, 15), maxLevel=3)
    ref, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, ref, None, **lk)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
    if good.sum() == 0:
        return 0, 0
    pts, ref = pts[good], ref[good]
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(da, db, pts, None, **lk)
    bk, st2, _ = cv2.calcOpticalFlowPyrLK(db, da, nxt, None, **lk)
    ok = ((st.ravel() == 1) & (st2.ravel() == 1)
          & (np.linalg.norm(bk - pts, axis=2).ravel() < 1.0)
          & (np.linalg.norm(nxt - ref, axis=2).ravel() < 2.0))
    return int(ok.sum()), int(len(pts))


def visual_fixes_real(seq, frame_t, frame_paths, faults, seed=0, every=2, health_w=None):
    """Fixes at the real camera timestamps (every 2nd frame = 10 Hz).

    Health: learned model (if weights given) or the hand-tuned score, computed on the real
    degraded frame. Fix quality: share of real KLT tracks that stay correct under the degradation
    (measured against the clean frames), which drives fix noise, drift and jumps."""
    rng = np.random.default_rng(seed + 991)
    ft = frame_t[::every]
    sel = list(range(0, len(frame_paths), every))
    sched = camera_schedule(ft, faults)
    idx = np.clip(np.round(ft / seq.dt).astype(int), 0, len(seq.t) - 1)
    n = len(ft)
    fix = np.full((n, 3), np.nan)
    health = np.zeros(n)
    quality = np.zeros(n)
    drift = np.zeros(3)
    for j in range(n):
        if not sched["available"][j]:
            continue
        i = sel[j]
        i2 = min(i + 1, len(frame_paths) - 1)
        b, bl = sched["brightness"][j], sched["blur"][j]
        if b >= 0.999 and bl <= 0:
            img = load_gray(frame_paths[i])
            q = 1.0
        else:
            ca, cc = load_gray(frame_paths[i]), load_gray(frame_paths[i2])
            img, nxt = degrade_real(ca, b, bl, rng), degrade_real(cc, b, bl, rng)
            good, ref = correct_tracks(ca, cc, img, nxt)
            q = good / max(ref, 1)
        health[j] = learned_health(img, health_w) if health_w is not None else image_health(img)["score"]
        quality[j] = q
        bad = float(np.clip(1.0 - q, 0, 1)) ** 1.0
        std = FIX_STD * (1 + 8 * bad)
        drift = drift * (0.5 if bad < 0.05 else 0.99) + rng.normal(0, 0.08 * bad, 3)
        jump = rng.normal(0, 0.8, 3) if rng.random() < 0.15 * bad else 0.0
        fix[j] = seq.p[idx[j]] + rng.normal(0, std, 3) + drift + jump + sched["wrong_fix"][j]
    return {"idx": idx, "fix": fix, "health": health, "quality": quality}


def validate_health(frame_paths, n_frames=120, seed=0):
    """Mean health score and mean real KLT tracks per degradation level, on real frames."""
    rng = np.random.default_rng(seed)
    levels = [("normal", 1.0, 0), ("dim 50%", 0.5, 0), ("dim 30%", 0.3, 0), ("dark 15%", 0.15, 0),
              ("very dark 6%", 0.06, 0), ("blur 5 px", 1.0, 5), ("blur 15 px", 1.0, 15),
              ("blur 30 px", 1.0, 30), ("dark 30% + blur 10 px", 0.3, 10)]
    step = max(2, len(frame_paths) // n_frames)
    starts = list(range(0, len(frame_paths) - 1, step))[:n_frames]
    rows = []
    for name, b, bl in levels:
        hs, ts = [], []
        for i in starts:                                   # consecutive pairs, as a tracker sees them
            ca, cc = load_gray(frame_paths[i]), load_gray(frame_paths[i + 1])
            a, c = degrade_real(ca, b, bl, rng), degrade_real(cc, b, bl, rng)
            hs.append(image_health(a)["score"])
            good, ref = correct_tracks(ca, cc, a, c)
            ts.append(good / max(ref, 1))
        rows.append({"condition": name, "health": float(np.mean(hs)), "correct_frac": float(np.mean(ts))})
    h = np.array([r["health"] for r in rows])
    t = np.array([r["correct_frac"] for r in rows])
    corr = float(np.corrcoef(h, t)[0, 1])
    return rows, corr


# ---------------------------------------------------------------------------------------------
# Learned health: predict the share of correct real tracks from cheap image statistics.

HEALTH_PATH = "models/camera_health_learned.json"


def health_features(img):
    """Cheap per-frame statistics, all computed before any tracking."""
    h = image_health(img)
    return np.array([np.log(h["brightness"] + 1e-3), np.log(h["sharpness"] + 1e-6),
                     np.log1p(h["corners"]), 1.0])


def build_health_dataset(frame_paths, n_pairs=300, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    idx = rng.integers(0, len(frame_paths) - 1, n_pairs)
    for i in idx:
        b = float(np.exp(rng.uniform(np.log(0.03), 0)))       # brightness 3 %..100 %, log-uniform
        bl = float(rng.choice([0, 0, rng.uniform(0, 35)]))
        ca, cc = load_gray(frame_paths[i]), load_gray(frame_paths[i + 1])
        a, c = degrade_real(ca, b, bl, rng), degrade_real(cc, b, bl, rng)
        good, ref = correct_tracks(ca, cc, a, c)
        if ref < 10:
            continue
        X.append(health_features(a))
        y.append(good / ref)
    return np.array(X), np.array(y)


def fit_health(X, y, out=HEALTH_PATH):
    from scipy.optimize import minimize
    import json
    from pathlib import Path

    def loss(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 1e-4 * np.sum(w[:-1] ** 2)

    w = minimize(loss, np.zeros(X.shape[1]), method="L-BFGS-B").x
    Path(out).write_text(json.dumps({"w": w.tolist(), "features": ["log_brightness", "log_sharpness",
                                                                      "log1p_corners", "bias"]}, indent=1))
    return w


def learned_health(img, w):
    return float(1 / (1 + np.exp(-(health_features(img) @ w))))
