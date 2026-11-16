"""Image-level corruption applied BEFORE the visual front end (issue #2 B/D).

A schedule is a list of ``ImageFault(kind, start, end, severity)`` in stream seconds. ``apply`` maps
(t, left, right) -> (t', left', right') or None (frame dropped). The schedule is evaluation
input only: the estimator and front end see just the corrupted stream, never the schedule.

Kinds (physical basis):
    dark       exposure loss: gain 1-0.97*severity plus shot/read noise
    blur       motion blur, horizontal kernel up to 25 px
    blackout   lens covered / camera off: zero images
    occlusion  a random rectangle covering up to 70 % of both images (hand in front of the rig)
    low_texture contrast collapse toward the mean (fog, blank wall)
    drop       frames missing (bandwidth / driver): frame removed
    jitter     timestamp noise up to 20 ms*severity (not corrected by the front end)
    frozen     stale frame repeated (driver hang): the scene appears static while the rig moves,
               which is a *wrong* visual measurement, not merely a missing one
    stereo_desync  right image from an older frame: wrong depth / scale
``randomized_schedule`` draws onset, duration, kind and severity from a seed, independent of training.
"""

from dataclasses import dataclass

import cv2
import numpy as np

KINDS = ("dark", "blur", "blackout", "occlusion", "low_texture", "drop", "jitter", "frozen", "stereo_desync")


@dataclass(frozen=True)
class ImageFault:
    kind: str
    start: float
    end: float
    severity: float = 1.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown image fault {self.kind!r}")
        if not (np.isfinite(self.start) and np.isfinite(self.end) and self.end > self.start):
            raise ValueError("fault interval must be finite with end > start")
        if not 0 < self.severity <= 1:
            raise ValueError("severity must be in (0, 1]")

    def active(self, t):
        return self.start <= t < self.end

    def as_dict(self):
        return {"kind": self.kind, "start": self.start, "end": self.end, "severity": self.severity}


class Corruptor:
    """Stateful corruption with a fixed composition order, independent of schedule order.

    Drop consumes no randomness. Jitter affects time only. Photometric/spatial faults
    precede blackout. Frozen repeats the previous delivered pair exactly, ignoring
    new image corruptions until it ends (at startup it captures the first pair).
    Desync uses the previous delivered right image unless the pair is frozen.
    """

    def __init__(self, schedule, seed: int = 0):
        self.schedule = list(schedule)
        self.rng = np.random.default_rng(seed)
        self.last = None        # last delivered (left, right)
        self.prev_right = None
        self.boxes = {}

    def apply(self, t, left, right):
        active = [f for f in self.schedule if f.active(t)]
        if any(f.kind == "drop" for f in active):
            return None
        order = ("jitter", "dark", "blur", "occlusion", "low_texture", "blackout", "frozen", "stereo_desync")
        active.sort(key=lambda f: (order.index(f.kind), f.start, f.end, f.severity))
        frozen = any(f.kind == "frozen" for f in active)
        if frozen and self.last is not None:
            for f in active:
                if f.kind == "jitter":
                    t += self.rng.uniform(-0.02, 0.02) * f.severity
            return t, self.last[0].copy(), self.last[1].copy()
        out_l, out_r = left, right
        for f in active:
            s = f.severity
            if f.kind == "drop":
                self.prev_right = right
                return None
            if f.kind == "jitter":
                t = t + self.rng.uniform(-0.02, 0.02) * s
            elif f.kind == "blackout":
                out_l, out_r = np.zeros_like(left), np.zeros_like(right)
            elif f.kind == "dark":
                out_l, out_r = (self._dark(x, s) for x in (out_l, out_r))
            elif f.kind == "blur":
                k = max(3, int(25 * s) | 1)
                ker = np.zeros((k, k), np.float32)
                ker[k // 2] = 1.0 / k
                out_l, out_r = (cv2.filter2D(x, -1, ker) for x in (out_l, out_r))
            elif f.kind == "occlusion":
                if f not in self.boxes:
                    self.boxes[f] = self._box(left.shape, s)
                box = self.boxes[f]
                out_l, out_r = out_l.copy(), out_r.copy()
                y0, y1, x0, x1 = box
                out_l[y0:y1, x0:x1] = 20
                out_r[y0:y1, x0:x1] = 20
            elif f.kind == "low_texture":
                out_l, out_r = (self._flatten(x, s) for x in (out_l, out_r))
            elif f.kind == "frozen" and self.last is not None:
                out_l, out_r = self.last
            elif f.kind == "stereo_desync" and self.prev_right is not None:
                out_r = self.prev_right
        self.last = (out_l.copy(), out_r.copy())
        self.prev_right = out_r.copy()
        return t, out_l, out_r

    def _dark(self, img, s):
        gain = 1.0 - 0.97 * s
        x = img.astype(np.float32) * gain
        x += self.rng.normal(0, 1.0, img.shape) * np.sqrt(np.maximum(x, 0) * 0.5 + 4.0)
        return np.clip(x, 0, 255).astype(np.uint8)

    def _flatten(self, img, s):
        m = float(img.mean())
        x = m + (img.astype(np.float32) - m) * (1.0 - 0.95 * s)
        x = cv2.GaussianBlur(x, (0, 0), 1.0 + 4 * s) + self.rng.normal(0, 1.0, img.shape)
        return np.clip(x, 0, 255).astype(np.uint8)

    def _box(self, shape, s):
        h, w = shape[:2]
        frac = np.sqrt(0.7 * s)
        bh, bw = int(h * frac), int(w * frac)
        y0, x0 = self.rng.integers(0, h - bh + 1), self.rng.integers(0, w - bw + 1)
        return int(y0), int(y0 + bh), int(x0), int(x0 + bw)


def randomized_schedule(duration: float, seed: int, n_faults=(0, 3), kinds=KINDS, dur_range=(0.5, 10.0),
                        allow_startup: bool = True):
    """Random camera faults over [0, duration): count, kind, onset, length and severity all drawn from
    ``seed``. Onset may be 0 (fault present at start-up) when ``allow_startup``."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("recording duration must be finite and positive")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_faults[0], n_faults[1] + 1))
    out = []
    for _ in range(n):
        kind = str(rng.choice(kinds))
        length = float(np.exp(rng.uniform(np.log(dur_range[0]), np.log(dur_range[1]))))
        start = 0.0 if (allow_startup and rng.random() < 0.1) else float(rng.uniform(0, max(duration - length, 0.0)))
        out.append(ImageFault(kind, start, min(start + length, duration), float(rng.uniform(0.3, 1.0))))
    return sorted(out, key=lambda f: f.start)
