"""Find the IMU that is lying, and fuse the rest.

All IMUs on a headset are bolted to the same rigid frame, so (after rotating into the body
frame) their angular rates must agree, and their specific forces agree up to small
lever-arm terms. A faulty IMU breaks that agreement.

Per short window (default 0.1 s) each IMU gets a trust weight in [0, 1] from four checks:
  1. dead / stuck: almost no variation while the others are moving
  2. saturation: readings pinned at the range limit
  3. spikes / noise: high-frequency energy far above its own clean baseline
  4. disagreement with the consensus (median of the others). With 3+ IMUs the median
     outvotes a single bad one, so the fault is isolated. With 2 IMUs a disagreement is
     detected, and checks 1-3 decide who is wrong; a smooth bias on one of two IMUs is
     detectable but not always attributable. That is why more IMUs (Galaxy XR has five)
     matter, and the benchmark reports isolation accuracy against the number of IMUs.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class VoteConfig:
    window: int = 20                 # samples (0.1 s at 200 Hz)
    gyro_tol: float = 0.05           # rad/s mean disagreement treated as healthy
    acc_tol: float = 0.6             # m/s^2 (covers lever-arm effects during head turns)
    stuck_std: float = 1e-4
    gyro_range: float | None = None  # if known, readings within 1% of it count as saturated
    acc_range: float | None = None
    noise_ratio: float = 6.0         # HF energy above this x baseline = noisy / spiky


def _hf(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.diff(x, axis=0) ** 2)))


def vote(gyro: np.ndarray, acc: np.ndarray, cfg: VoteConfig = VoteConfig(),
         baseline_hf: np.ndarray | None = None):
    """gyro, acc: (K, N, 3). Returns fused (N,3) gyro and acc, weights (K, n_windows), flags."""
    K, N, _ = gyro.shape
    W = cfg.window
    nw = int(np.ceil(N / W))
    weights = np.ones((K, nw))
    reasons = np.zeros((K, nw), dtype=object)
    reasons[:] = ""
    if baseline_hf is None:
        # Each IMU's own quiet-time high-frequency level, from the first second.
        first = min(N, 10 * W)
        baseline_hf = np.array([[_hf(gyro[k, :first]), _hf(acc[k, :first])] for k in range(K)])
    baseline_hf = np.maximum(baseline_hf, 1e-6)

    fused_g = np.zeros((N, 3))
    fused_a = np.zeros((N, 3))
    for w in range(nw):
        s = slice(w * W, min((w + 1) * W, N))
        g, a = gyro[:, s], acc[:, s]
        wts = np.ones(K)
        moving = np.array([g[k].std(0).max() for k in range(K)])
        for k in range(K):
            why = []
            if moving[k] < cfg.stuck_std and np.median(moving) > 10 * cfg.stuck_std:
                why.append("stuck")
            if cfg.gyro_range and np.any(np.abs(g[k]) > 0.99 * cfg.gyro_range):
                why.append("saturated")
            if cfg.acc_range and np.any(np.abs(a[k]) > 0.99 * cfg.acc_range):
                why.append("saturated")
            if len(g[k]) > 2:
                r = max(_hf(g[k]) / baseline_hf[k, 0], _hf(a[k]) / baseline_hf[k, 1])
                if r > cfg.noise_ratio:
                    why.append("noisy")
            if why:
                wts[k] = 0.0
                reasons[k, w] = "+".join(sorted(set(why)))
        if K >= 2:
            ok = wts > 0
            if ok.sum() < 2:
                ok[:] = True
            # Element-wise median of the IMUs still in play: with 3+ it outvotes one liar.
            ref_g = np.median(g[ok], axis=0)
            ref_a = np.median(a[ok], axis=0)
            for k in range(K):
                dg = np.linalg.norm(g[k] - ref_g, axis=1).mean()
                da = np.linalg.norm(a[k] - ref_a, axis=1).mean()
                score = max(dg / cfg.gyro_tol, da / cfg.acc_tol)
                if score > 1 and wts[k] > 0:
                    if K >= 3 or not reasons[k, w]:
                        # With 3+ IMUs the consensus excludes k, so a high score isolates k.
                        # With 2 IMUs both see the same score; mark as "disagree" for both
                        # and keep weights split unless another check already decided.
                        wts[k] = 1.0 / score ** 2 if K >= 3 else wts[k]
                        reasons[k, w] = reasons[k, w] or "disagree"
        if K == 2 and reasons[0, w] == "disagree" and reasons[1, w] == "disagree":
            # Can't tell who is right from the IMUs alone: keep both, flag low confidence.
            wts[:] = 0.5
        weights[:, w] = wts
        tot = wts.sum()
        if tot < 1e-9:
            # Every IMU looks bad: fall back to the median and let the filter's covariance grow.
            fused_g[s] = np.median(g, axis=0)
            fused_a[s] = np.median(a, axis=0)
        else:
            fused_g[s] = np.tensordot(wts / tot, g, 1)
            fused_a[s] = np.tensordot(wts / tot, a, 1)

    confidence = np.clip(weights.max(0), 0, 1)
    return fused_g, fused_a, {"weights": weights, "reasons": reasons, "confidence": confidence}
