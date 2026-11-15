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


def lever_arm_compensate(acc, gyro, r_b, dt, smooth: int = 5):
    """Move one IMU's specific force to the body origin, given its offset r_b (metres, body frame).

    An IMU at r_b feels f_k = f_0 + w x (w x r_b) + dw/dt x r_b. ``gyro`` should be the common
    angular rate (all IMUs on a rigid body share it); it is lightly smoothed before differentiating.
    """
    w = gyro
    if smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.stack([np.convolve(w[:, i], k, mode="same") for i in range(3)], 1)
    wdot = np.gradient(w, dt, axis=0)
    cent = np.cross(w, np.cross(w, r_b))
    tang = np.cross(wdot, r_b)
    return acc - cent - tang


@dataclass
class ParityConfig:
    window: int = 20            # samples per test (0.1 s at 200 Hz)
    p_false_alarm: float = 1e-4
    calib_s: float = 1.0        # warm-up: seconds before the data-driven noise level is used
    # Sensor priors (datasheet-style, per axis): white noise per sample and turn-on bias spread.
    # Used during warm-up, and as bounds on what warm-up may learn, so a fault that is already
    # present at start-up is tested against the spec instead of being learned as "healthy".
    gyro_noise: float = 0.003   # rad/s
    acc_noise: float = 0.03     # m/s^2
    gyro_bias: float = 0.0005   # rad/s
    acc_bias: float = 0.01      # m/s^2
    learn_range: tuple = (0.33, 3.0)   # learned noise level must stay within this x the prior


# Confidence levels handed to the filter (process noise grows as confidence falls).
CONF_OK = 1.0              # IMUs agree
CONF_ISOLATED = 0.95       # one IMU disagreed, was removed, and the rest agree again
CONF_WARMUP = 0.8          # first second: only the datasheet noise level is known
CONF_UNRESOLVED = 0.3      # disagreement we cannot attribute (2 IMUs, or several faults)
CONF_NONE = 0.0            # no usable IMU reading in this window


class ParityFDI:
    """Detects and isolates a faulty IMU. Note: a fault common to every IMU (shared vibration,
    temperature drift of the same model) cancels in the parity vector and is NOT detectable here;
    the filter's camera checks have to catch it."""

    def __init__(self, K: int, cfg: ParityConfig = ParityConfig()):
        self.K, self.cfg = K, cfg
        H = np.vstack([np.eye(3)] * K)                 # every IMU measures the same 3-vector
        self.V = null_space(H.T).T                     # (3K-3) x 3K, orthonormal rows, V H = 0
        self.Vk = [self.V[:, 3 * k:3 * k + 3] for k in range(K)]
        self.Pk = [Vk @ np.linalg.pinv(Vk.T @ Vk) @ Vk.T for Vk in self.Vk]
        dof = self.V.shape[0]
        self.threshold = chi2.ppf(1 - cfg.p_false_alarm, 2 * dof)   # gyro + accel together
        # After removing the isolated IMU, 3 dof per modality are explained by its fault.
        self.res_threshold = chi2.ppf(1 - cfg.p_false_alarm, 2 * (dof - 3)) if dof > 3 else np.inf
        self.can_isolate = K >= 3

    def run(self, gyro: np.ndarray, acc: np.ndarray, rate: float, lever_arms=None):
        """gyro, acc: (K, N, 3), already rotated into the body frame. lever_arms: (K, 3) metres or None
        (co-located IMUs). Causal: window w only uses samples up to the end of window w.

        Returns weights (K, n_windows), the detection statistic D, and info with ``confidence``
        (n_windows,), ``isolated`` (-1 or the IMU index), ``detected``, ``state`` per window."""
        K, N, _ = gyro.shape
        W, c = self.cfg.window, self.cfg
        nw = int(np.ceil(N / W))
        gyro, acc = np.array(gyro, float), np.array(acc, float)
        finite = np.isfinite(gyro).all(2) & np.isfinite(acc).all(2)          # (K, N)
        if lever_arms is not None:
            r = np.asarray(lever_arms, float)
            w_common = np.nan_to_num(np.nanmedian(np.where(finite[..., None], gyro, np.nan), 0))

        weights = np.ones((K, nw))
        zg, za = np.zeros((nw, 3 * K)), np.zeros((nw, 3 * K))
        for w in range(nw):
            s = slice(w * W, min((w + 1) * W, N))
            ok = finite[:, s].all(1)
            weights[~ok, w] = 0.0
            for k in range(K):
                if ok[k]:
                    zg[w, 3 * k:3 * k + 3] = gyro[k, s].mean(0)
                    za[w, 3 * k:3 * k + 3] = acc[k, s].mean(0)
            if lever_arms is not None:
                # Window-mean lever-arm terms: mean of w x (w x r), and mean angular acceleration
                # taken exactly as (w_end - w_start) / T, which avoids differentiating gyro noise.
                ws = w_common[s]
                i0 = max(s.start - 1, 0)                 # last sample before the window (causal)
                span = (s.stop - 1 - i0) / rate
                wdot = (w_common[s.stop - 1] - w_common[i0]) / max(span, 1e-9)
                for k in range(K):
                    if ok[k]:
                        cent = np.cross(ws, np.cross(ws, r[k])).mean(0)
                        za[w, 3 * k:3 * k + 3] -= cent + np.cross(wdot, r[k])
            if ok.any():                                # unusable IMUs: stand in with the others' mean
                mg = zg[w].reshape(K, 3)[ok].mean(0)
                ma = za[w].reshape(K, 3)[ok].mean(0)
                for k in np.where(~ok)[0]:
                    zg[w, 3 * k:3 * k + 3], za[w, 3 * k:3 * k + 3] = mg, ma
        pg, pa = zg @ self.V.T, za @ self.V.T

        # Warm-up uses the datasheet level (noise of a window mean + turn-on bias spread).
        sg0 = np.sqrt(c.gyro_noise ** 2 / W + c.gyro_bias ** 2)
        sa0 = np.sqrt(c.acc_noise ** 2 / W + c.acc_bias ** 2)
        nc = max(2, int(c.calib_s * rate / W))
        mg = np.zeros(pg.shape[1]); ma = np.zeros(pa.shape[1]); sg, sa = sg0, sa0

        D = np.zeros(nw)
        isolated = np.full(nw, -1)
        conf = np.ones(nw)
        state = np.empty(nw, dtype=object)
        lo, hi = c.learn_range
        for w in range(nw):
            if w == nc:
                # Learn the nominal offset and noise from warm-up windows that passed the spec test.
                good = [i for i in range(nc) if D[i] <= self.threshold and isolated[i] < 0]
                if len(good) >= 2:
                    mg, ma = np.median(pg[good], 0), np.median(pa[good], 0)
                    sg = float(np.clip(1.4826 * np.median(np.abs(pg[good] - mg)), lo * sg0, hi * sg0))
                    sa = float(np.clip(1.4826 * np.median(np.abs(pa[good] - ma)), lo * sa0, hi * sa0))
            ng, na = (pg[w] - mg) / sg, (pa[w] - ma) / sa
            D[w] = ng @ ng + na @ na
            usable = int((weights[:, w] > 0).sum())
            if usable == 0:
                conf[w], state[w] = CONF_NONE, "no_usable_imu"
                continue
            if D[w] <= self.threshold:
                conf[w] = CONF_WARMUP if w < nc else CONF_OK
                state[w] = "warmup" if w < nc else "ok"
                if usable < K:
                    state[w] = "ok_missing_imu"
                continue
            if not self.can_isolate:
                weights[:, w] = np.where(weights[:, w] > 0, 0.5, 0.0)   # detected, cannot say which
                conf[w], state[w] = CONF_UNRESOLVED, "detected_not_isolable"
                continue
            I = [ng @ self.Pk[k] @ ng + na @ self.Pk[k] @ na for k in range(K)]
            k_bad = int(np.argmax(I))
            weights[k_bad, w] = 0.0
            isolated[w] = k_bad
            if D[w] - I[k_bad] <= self.res_threshold and usable - 1 >= 2:
                conf[w], state[w] = CONF_ISOLATED, "isolated"
            else:
                conf[w], state[w] = CONF_UNRESOLVED, "unresolved"     # more than one fault, or too few left
        return weights, D, {"threshold": self.threshold, "isolated": isolated,
                            "detected": D > self.threshold, "confidence": conf, "state": state}


def fuse(gyro, acc, weights, window):
    """Weighted mean of the IMUs, window by window. A window with no usable IMU holds the last
    fused value (its confidence is 0, so the filter widens its uncertainty) instead of silently
    averaging readings that were all rejected."""
    K, N, _ = gyro.shape
    fg, fa = np.zeros((N, 3)), np.zeros((N, 3))
    last_g, last_a = np.zeros(3), np.array([0.0, 0.0, 9.81])
    for w in range(weights.shape[1]):
        s = slice(w * window, min((w + 1) * window, N))
        wt = weights[:, w]
        if wt.sum() < 1e-9:
            fg[s], fa[s] = last_g, last_a
            continue
        wt = wt / wt.sum()
        use = wt > 0
        fg[s] = np.tensordot(wt[use], gyro[use][:, s], 1)
        fa[s] = np.tensordot(wt[use], acc[use][:, s], 1)
        last_g, last_a = fg[s][-1], fa[s][-1]
    return fg, fa
