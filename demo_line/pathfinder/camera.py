"""Camera side: a health score measured from the image, and the visual position fixes.

``image_health`` works on any grayscale frame (real Aria / EuRoC frames or our synthetic
ones). It looks at three things a tracker needs: enough light, sharp edges, and enough
trackable corners. It runs before fusion, so a bad frame is down-weighted before it can
pull the trajectory off.

On datasets without images (TLIO) we degrade a textured frame per the fault schedule and
measure its health with the same function; the fix noise also grows with darkness and blur,
as a real tracker's would.
"""

import numpy as np
from scipy import ndimage

CAM_RATE = 10.0          # visual fixes per second (typical for headset relocalisation)
FIX_STD = 0.03           # metres, clean visual fix noise


def image_health(img: np.ndarray) -> dict:
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    brightness = float(img.mean())
    smooth = ndimage.gaussian_filter(img, 1.0)          # ignore pixel noise, keep real edges
    smooth = smooth / max(brightness, 0.05) * 0.5       # contrast relative to exposure
    sharp = float(ndimage.laplace(smooth).var())
    gx, gy = ndimage.sobel(smooth, 1), ndimage.sobel(smooth, 0)
    sxx = ndimage.uniform_filter(gx * gx, 5)
    syy = ndimage.uniform_filter(gy * gy, 5)
    sxy = ndimage.uniform_filter(gx * gy, 5)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lam_min = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    peaks = (lam_min == ndimage.maximum_filter(lam_min, 7)) & (lam_min > 0.01)
    corners = int(peaks.sum())
    s_b = np.clip((brightness - 0.05) / 0.15, 0, 1)
    s_s = np.clip((sharp - 0.0002) / 0.0004, 0, 1)
    s_c = np.clip(corners / 40.0, 0, 1)
    return {"brightness": brightness, "sharpness": sharp, "corners": corners,
            "score": float(s_b * s_s * s_c)}


def textured_frame(rng: np.random.Generator, h: int = 120, w: int = 160) -> np.ndarray:
    """A cluttered indoor-like frame: blocks, edges and fine texture."""
    img = ndimage.gaussian_filter(rng.random((h, w)), 1.0) * 0.4 + 0.3
    for _ in range(25):
        y, x = rng.integers(0, h - 10), rng.integers(0, w - 10)
        img[y:y + rng.integers(5, 30), x:x + rng.integers(5, 40)] = rng.uniform(0.1, 0.9)
    return np.clip(img, 0, 1)


def degrade(img: np.ndarray, brightness: float, blur_px: float, rng) -> np.ndarray:
    out = img * brightness
    if blur_px > 0:
        out = ndimage.uniform_filter1d(out, int(blur_px), axis=1)   # horizontal motion blur
    sensor_noise = 0.01 + 0.03 * (1 - brightness)                    # dark frames are noisy
    return np.clip(out + rng.normal(0, sensor_noise, out.shape), 0, 1)


def visual_fixes(t: np.ndarray, p_true: np.ndarray, sched: dict, seed: int = 0) -> dict:
    """Position fixes at CAM_RATE with a measured health score per frame."""
    rng = np.random.default_rng(seed + 77)
    step = frame_step(t)
    idx = np.arange(0, len(t), step)
    n = len(idx)
    keys = {k: v for k, v in sched.items()}
    if len(keys["available"]) != n:
        raise ValueError("camera schedule length must match frame count")
    fix = np.full((n, 3), np.nan)
    health = np.zeros(n)
    drift = np.zeros(3)
    for j, i in enumerate(idx):
        if not keys["available"][j]:
            continue
        b, blur = keys["brightness"][j], keys["blur"][j]
        frame = degrade(textured_frame(rng), b, blur, rng)
        health[j] = image_health(frame)["score"]
        # How a real visual tracker degrades: with fewer, noisier features its fix gets
        # noisier AND it drifts (correlated error), with occasional jumps from mismatches.
        # Clean frames pull the drift back (relocalisation). Identical for every method.
        bad = float(np.clip((1 - b) ** 2 + blur / 20, 0, 1.5))
        std = FIX_STD * (1 + 4 * bad)
        drift = drift * (0.5 if bad < 0.05 else 0.995) + rng.normal(0, 0.06 * bad, 3)
        jump = rng.normal(0, 0.8, 3) if rng.random() < 0.08 * bad else 0.0
        fix[j] = p_true[i] + rng.normal(0, std, 3) + drift + jump + keys["wrong_fix"][j]
    return {"idx": idx, "fix": fix, "health": health}


def frame_step(t: np.ndarray) -> int:
    """IMU samples per camera frame (1 for a single-sample sequence, #19)."""
    if len(t) < 2:
        return 1
    return max(int(round((1 / CAM_RATE) / np.median(np.diff(t)))), 1)


def frame_times(t: np.ndarray) -> np.ndarray:
    step = frame_step(t)
    return t[::step]


# ---------------------------------------------------------------------------------------------
# Trust calibration: how wrong is a visual fix, as a function of the measured health score?
# Learned once on TRAINING sequences, then the filter uses the table instead of a hand-set rule.

TRUST_PATH = "models/camera_trust.json"
BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.97, 1.01]


def calibrate_trust(seqs, seed: int = 0, out: str = TRUST_PATH) -> dict:
    """Random darkness / blur episodes on training sequences -> RMS fix error per health bin."""
    import json
    from pathlib import Path

    from .faults import Fault, camera_schedule
    rng = np.random.default_rng(seed)
    h_all, e_all = [], []
    for i, s in enumerate(seqs):
        ft = frame_times(s.t)
        dur = ft[-1] - ft[0]
        faults = []
        for _ in range(4):
            start, length = rng.uniform(0, dur - 5), rng.uniform(3, 12)
            if rng.random() < 0.5:
                faults.append(Fault("camera_dark", start, length, params={"level": rng.uniform(0.04, 0.6)}))
            else:
                faults.append(Fault("camera_blur", start, length, params={"pixels": rng.uniform(3, 35)}))
        cam = visual_fixes(s.t, s.p, camera_schedule(ft, faults), seed + i)
        ok = ~np.isnan(cam["fix"][:, 0])
        err = np.linalg.norm(cam["fix"][ok] - s.p[cam["idx"][ok]], axis=1) / np.sqrt(3)
        h_all.append(cam["health"][ok])
        e_all.append(err)
    h, e = np.concatenate(h_all), np.concatenate(e_all)
    centers, stds, counts = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (h >= lo) & (h < hi)
        if m.sum() >= 20:
            centers.append(float(h[m].mean()))
            stds.append(float(np.sqrt(np.mean(e[m] ** 2))))
            counts.append(int(m.sum()))
    # Trust can only fall as health falls: enforce a monotone table.
    for j in range(len(stds) - 2, -1, -1):
        stds[j] = max(stds[j], stds[j + 1])
    table = {"health": centers, "std_m": stds, "count": counts, "n_frames": int(len(h))}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(table, indent=1))
    return table


def load_trust(path: str = TRUST_PATH):
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    return np.array(t["health"]), np.array(t["std_m"])
