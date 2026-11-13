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
    step = max(int(round((1 / CAM_RATE) / np.median(np.diff(t)))), 1)
    idx = np.arange(0, len(t), step)
    n = len(idx)
    keys = {k: v for k, v in sched.items()}
    if len(keys["available"]) != n:
        raise ValueError("camera schedule length must match frame count")
    fix = np.full((n, 3), np.nan)
    health = np.zeros(n)
    for j, i in enumerate(idx):
        if not keys["available"][j]:
            continue
        b, blur = keys["brightness"][j], keys["blur"][j]
        frame = degrade(textured_frame(rng), b, blur, rng)
        health[j] = image_health(frame)["score"]
        # A real tracker gets worse in the dark and with blur: its fix error grows.
        std = FIX_STD * (1 + 8 * (1 - b) ** 2 + blur / 3)
        fix[j] = p_true[i] + rng.normal(0, std, 3) + keys["wrong_fix"][j]
    return {"idx": idx, "fix": fix, "health": health}


def frame_times(t: np.ndarray) -> np.ndarray:
    step = max(int(round((1 / CAM_RATE) / np.median(np.diff(t)))), 1)
    return t[::step]
