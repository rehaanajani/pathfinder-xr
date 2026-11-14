"""Parity-space fault detection and isolation for redundant IMUs.

The standard aerospace method for redundant inertial units (Potter & Suman 1977; Sturza 1988,
"Navigation system integrity monitoring using redundant measurements"; also used in RAIM).

All IMUs measure the same rigid-body motion x (angular rate, or lever-arm-compensated specific
force), so stacking their axes gives z = H x + noise + fault. A matrix V whose rows span the left
null space of H (V H = 0) gives the parity vector p = V z, which **does not depend on the motion**:
it only contains noise and faults.

Detection (GLRT): D = pᵀp / σ² follows χ²(3K − 3) when every IMU is healthy.
Isolation: for IMU k, I_k = pᵀ V_k (V_kᵀ V_k)⁻¹ V_kᵀ p / σ²; the faulty IMU maximises it.
With 2 IMUs a fault can be detected but not isolated; with 3 or more it can be isolated.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2


def lever_arm_compensate(acc, gyro, r_b, dt):
    """Remove centripetal + tangential terms for an IMU offset r_b (metres) from the body origin."""
    w = gyro
    wdot = np.gradient(w, dt, axis=0)
    cent = np.cross(w, np.cross(w, r_b))
    tang = np.cross(wdot, r_b)
    return acc - cent - tang


@dataclass
class ParityConfig:
    window: int = 20            # samples per test (0.1 s at 200 Hz)
    p_false_alarm: float = 1e-4
    calib_s: float = 1.0        # seconds used to learn each IMU pair's nominal offset and noise


class ParityFDI:
    def __init__(self, K: int, cfg: ParityConfig = ParityConfig()):
        self.K, self.cfg = K, cfg
        H = np.vstack([np.eye(3)] * K)                 # every IMU measures the same 3-vector
        self.V = null_space(H.T).T                     # (3K-3) x 3K, orthonormal rows, V H = 0
        self.Vk = [self.V[:, 3 * k:3 * k + 3] for k in range(K)]
        self.Pk = [Vk @ np.linalg.pinv(Vk.T @ Vk) @ Vk.T for Vk in self.Vk]
        dof = self.V.shape[0]
        self.threshold = chi2.ppf(1 - cfg.p_false_alarm, 2 * dof)   # gyro + accel together
        self.can_isolate = K >= 3

    def run(self, gyro: np.ndarray, acc: np.ndarray, rate: float):
        """gyro, acc: (K, N, 3). Returns per-window weights (K, n_windows), D statistic, and flags."""
        K, N, _ = gyro.shape
        W = self.cfg.window
        nw = int(np.ceil(N / W))

        def parity(sig):
            out = np.zeros((nw, self.V.shape[0]))
            for w in range(nw):
                s = slice(w * W, min((w + 1) * W, N))
                z = sig[:, s].mean(1).reshape(-1)       # (3K,) window mean, IMU-major
                out[w] = self.V @ z
            return out

        pg, pa = parity(gyro), parity(acc)
        # Learn the nominal parity (healthy IMUs differ by small fixed biases) and its noise level.
        nc = max(2, int(self.cfg.calib_s * rate / W))
        mg, ma = np.median(pg[:nc], 0), np.median(pa[:nc], 0)
        sg = max(1.4826 * np.median(np.abs(pg[:nc] - mg)), 1e-5)
        sa = max(1.4826 * np.median(np.abs(pa[:nc] - ma)), 1e-4)
        pg, pa = (pg - mg) / sg, (pa - ma) / sa

        D = (pg ** 2).sum(1) + (pa ** 2).sum(1)
        weights = np.ones((K, nw))
        isolated = np.full(nw, -1)
        for w in range(nw):
            if D[w] <= self.threshold:
                continue
            if not self.can_isolate:
                weights[:, w] = 0.5                      # detected, cannot say which: low confidence
                continue
            I = [pg[w] @ self.Pk[k] @ pg[w] + pa[w] @ self.Pk[k] @ pa[w] for k in range(K)]
            k_bad = int(np.argmax(I))
            weights[k_bad, w] = 0.0
            isolated[w] = k_bad
        return weights, D, {"threshold": self.threshold, "isolated": isolated,
                            "detected": D > self.threshold}


def fuse(gyro, acc, weights, window):
    """Weighted mean of the IMUs, window by window."""
    K, N, _ = gyro.shape
    fg, fa = np.zeros((N, 3)), np.zeros((N, 3))
    for w in range(weights.shape[1]):
        s = slice(w * window, min((w + 1) * window, N))
        wt = weights[:, w]
        if wt.sum() < 1e-9:
            wt = np.ones(K)
        wt = wt / wt.sum()
        fg[s] = np.tensordot(wt, gyro[:, s], 1)
        fa[s] = np.tensordot(wt, acc[:, s], 1)
    return fg, fa
