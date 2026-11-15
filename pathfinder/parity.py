"""Parity-space fault detection and isolation for redundant IMUs: the statistics.

The standard aerospace method for redundant inertial units (Potter & Suman 1977; Sturza 1988,
"Navigation system integrity monitoring using redundant measurements"; also used in RAIM).

Once every IMU's reading is expressed at the same point of the same rigid body (see
``to_body_frame``), all of them measure one motion vector x, so stacking their axes gives
z = H x + noise + fault with H = [I; I; ...]. A matrix V whose rows span the left null space of H
(V H = 0) gives the parity vector p = V z, which **does not depend on the motion**: it only
contains sensor noise, residual bias differences and faults.

Statistics used here (all per window of averaged samples, gyro and accelerometer blocks separately):

* The parity mean m and covariance S are *estimated* during a causal warm-up from n windows.
  For a new independent window, T^2 = (p - m)^T S^-1 (p - m) then follows the Hotelling
  prediction distribution  n/(n+1) * T^2 ~ d (n-1)/(n-d) F(d, n-d)  (Gaussian, independent windows).
  ``hotelling_crit`` gives that threshold, so the uncertainty of the estimated noise is accounted for
  instead of pretending S is the true covariance (which is what made a 10-window calibration
  produce ~2 % false alarms against a nominal 0.01 %).
* Isolation (GLRT for a bias on one sensor): after whitening with S, the statistic for sensor k is
  the energy of the whitened parity in the span of sensor k's columns. The largest wins; the
  remaining sensors are then re-tested so that a second fault, or a fault that does not fit a
  single-sensor model, is reported instead of hidden.
* With 2 IMUs a fault can be detected but not attributed; with 3 or more a single fault can be
  isolated. Several simultaneous faults need more sensors, and a common-mode error (identical on
  every sensor) is invisible to parity by construction.

``ParityFDI`` keeps the original batch interface; the streaming, causal implementation with
signal checks and fusion lives in :mod:`pathfinder.imu_health`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.stats import f as f_dist


def lever_arm_compensate(acc, gyro, r_b, dt=None, wdot=None):
    """Specific force at the body origin from an IMU at offset ``r_b`` (metres, body frame).

    An accelerometer at r measures f_r = f_0 + wdot x r + w x (w x r) (rigid body). ``gyro`` and
    ``acc`` must already be rotated into the body frame. ``wdot`` is the angular acceleration; if it
    is not given it is taken from ``np.gradient`` of the gyro, which needs the samples on both sides
    and is therefore only valid on a buffered window (see ``imu_health``).
    """
    w = np.asarray(gyro, float)
    if wdot is None:
        if dt is None:
            raise ValueError("need dt or wdot")
        wdot = np.gradient(w, dt, axis=0) if len(w) > 1 else np.zeros_like(w)
    r_b = np.asarray(r_b, float)
    cent = np.cross(w, np.cross(w, r_b))
    tang = np.cross(np.broadcast_to(wdot, w.shape), r_b)
    return np.asarray(acc, float) - cent - tang


def parity_matrix(m: int) -> np.ndarray:
    """(3m-3) x 3m matrix with orthonormal rows and V H = 0 for H = [I; ...; I]."""
    H = np.vstack([np.eye(3)] * m)
    return null_space(H.T).T


def hotelling_crit(d: int, n: int, alpha: float) -> float:
    """Threshold on T^2 for a new observation, mean and covariance estimated from n observations."""
    if n <= d:
        raise ValueError(f"need more calibration windows ({n}) than parity dimensions ({d})")
    return float((n + 1) / n * d * (n - 1) / (n - d) * f_dist.ppf(1 - alpha, d, n - d))


def hotelling_pvalue(t2: float, d: int, n: int) -> float:
    if n <= d or not np.isfinite(t2):
        return float("nan")
    x = t2 * n / (n + 1) * (n - d) / (d * (n - 1))
    return float(f_dist.sf(x, d, n - d))


def two_sample_shift_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Hotelling test for a mean shift between two windows of parity vectors."""
    n1, n2, d = len(a), len(b), a.shape[1]
    dof = n1 + n2 - d - 1
    if dof <= 0:
        return 1.0
    diff = a.mean(0) - b.mean(0)
    Sp = ((n1 - 1) * np.cov(a.T).reshape(d, d) + (n2 - 1) * np.cov(b.T).reshape(d, d)) / (n1 + n2 - 2)
    t2 = n1 * n2 / (n1 + n2) * float(diff @ np.linalg.solve(Sp + 1e-18 * np.eye(d), diff))
    fstat = dof / (d * (n1 + n2 - 2)) * t2
    return float(f_dist.sf(fstat, d, dof))


def robust_mean_cov(x: np.ndarray, cut: float, iters: int = 4):
    """Mean and covariance with iterative Mahalanobis trimming (start: median / MAD).

    Returns (mean, cov, inlier mask). Removes transient outliers (spikes, a short fault) from the
    calibration; it cannot remove a fault present for most of the warm-up, which is handled by the
    stationarity and prior-offset checks in ``imu_health``.
    """
    n, d = x.shape
    med = np.median(x, 0)
    mad = 1.4826 * np.median(np.abs(x - med), 0)
    mad = np.maximum(mad, 1e-12)
    keep = (((x - med) / mad) ** 2).sum(1) <= cut * 4.0
    if keep.sum() <= d + 1:
        keep = np.ones(n, bool)
    for _ in range(iters):
        mu = x[keep].mean(0)
        C = np.cov(x[keep].T).reshape(d, d)
        C = 0.5 * (C + C.T) + 1e-18 * np.eye(d)
        r = x - mu
        dist = np.einsum("ni,ni->n", r, np.linalg.solve(C, r.T).T)
        new = dist <= cut
        if new.sum() <= d + 1 or np.array_equal(new, keep):
            break
        keep = new
    mu = x[keep].mean(0)
    C = np.cov(x[keep].T).reshape(d, d)
    return mu, 0.5 * (C + C.T), keep


def subset_map(Vfull: np.ndarray, members_full: list[int], subset: list[int]) -> np.ndarray:
    """M with p_subset = M p_full, for parity vectors built from the same (bias-corrected) z.

    V_C E_C annihilates H, so it lies in the row space of the orthonormal V_full: M = V_C E_C V_full^T.
    """
    Vc = parity_matrix(len(subset))
    E = np.zeros((3 * len(subset), 3 * len(members_full)))
    for i, k in enumerate(subset):
        j = members_full.index(k)
        E[3 * i:3 * i + 3, 3 * j:3 * j + 3] = np.eye(3)
    return Vc @ E @ Vfull.T, Vc


def isolation_scores(p: np.ndarray, S: np.ndarray, Vc: np.ndarray, m: int) -> np.ndarray:
    """GLRT energy of the whitened parity explained by a bias on each of the m sensors."""
    L = np.linalg.cholesky(S)
    pw = np.linalg.solve(L, p)
    out = np.zeros(m)
    for k in range(m):
        A = np.linalg.solve(L, Vc[:, 3 * k:3 * k + 3])
        coef, *_ = np.linalg.lstsq(A, pw, rcond=None)
        out[k] = float(pw @ (A @ coef))
    return out


@dataclass
class ParityConfig:
    window: int = 20            # samples per test (0.1 s at 200 Hz)
    p_false_alarm: float = 1e-4  # per complete window under the stated Gaussian / independence model
    calib_s: float = 2.0        # minimum causal warm-up (seconds); extended until enough windows


class ParityFDI:
    """Batch wrapper kept for the original interface. Pure parity test, no per-sensor signal checks.

    Returns per-window weights (K, n_windows): 1 = used, 0 = isolated/excluded, 0.5 = fault detected
    but not attributable (two sensors). ``D`` is the larger of the gyro / accelerometer T^2 divided by
    its threshold, so ``D > 1`` means detected. Windows inside the warm-up are never tested
    (``info["calibrated"]`` is False there) instead of being judged with statistics learned from
    themselves.
    """

    def __init__(self, K: int, cfg: ParityConfig = ParityConfig()):
        self.K, self.cfg = K, cfg
        self.can_isolate = K >= 3

    def run(self, gyro: np.ndarray, acc: np.ndarray, rate: float):
        from .imu_health import ImuMonitor, MonitorConfig
        K, N, _ = gyro.shape
        mc = MonitorConfig(window_s=self.cfg.window / rate, warmup_s=self.cfg.calib_s,
                           p_false_alarm=self.cfg.p_false_alarm, signal_checks=False)
        t = np.arange(N) / rate
        _, _, info = ImuMonitor(K, mc).run_batch(t, gyro, acc)
        return info["weights"], info["D"], {"threshold": 1.0, "isolated": info["isolated"],
                                            "detected": info["detected"], "calibrated": info["calibrated"],
                                            "state": info["state"]}


def fuse(gyro, acc, weights, window):
    """Weighted mean of the IMUs, window by window (legacy helper, co-located sensors only).

    If every weight is zero there is no healthy sensor; the result is NaN rather than a silent
    equal average, so the caller must handle the lost state explicitly.
    """
    K, N, _ = gyro.shape
    fg, fa = np.full((N, 3), np.nan), np.full((N, 3), np.nan)
    for w in range(weights.shape[1]):
        s = slice(w * window, min((w + 1) * window, N))
        wt = weights[:, w]
        if wt.sum() < 1e-9:
            continue
        wt = wt / wt.sum()
        fg[s] = np.tensordot(wt, gyro[:, s], 1)
        fa[s] = np.tensordot(wt, acc[:, s], 1)
    return fg, fa
