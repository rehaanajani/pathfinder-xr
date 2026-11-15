"""Trajectory error metrics, as used in the inertial-odometry papers (RoNIN, TLIO).

ATE: RMSE of position after aligning yaw + translation (gravity fixes roll and pitch).
RTE: RMSE of displacement error over a fixed time window, after the same alignment.
Drift: final position error as a percentage of distance walked.
"""

import numpy as np


def align_yaw(est: np.ndarray, gt: np.ndarray):
    """Best yaw rotation + translation mapping est onto gt (closed-form 4-DoF Umeyama)."""
    me, mg = est.mean(0), gt.mean(0)
    e, g = est - me, gt - mg
    s = (e[:, 0] * g[:, 1] - e[:, 1] * g[:, 0]).sum()
    c = (e[:, 0] * g[:, 0] + e[:, 1] * g[:, 1]).sum()
    th = np.arctan2(s, c)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    return Rz, mg - Rz @ me


def ate(est: np.ndarray, gt: np.ndarray, align: bool = True) -> float:
    ok = np.isfinite(est).all(axis=1) & np.isfinite(gt).all(axis=1)
    est, gt = est[ok], gt[ok]
    if not len(est):
        return float("nan")
    if align:
        Rz, tr = align_yaw(est, gt)
        est = est @ Rz.T + tr
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, 1))))


def rte(est: np.ndarray, gt: np.ndarray, window: int, align: bool = True) -> float:
    if window < 1 or len(gt) <= window:
        return float("nan")
    ok = np.isfinite(est).all(axis=1) & np.isfinite(gt).all(axis=1)
    # Never compress the time axis by deleting failed/initializing samples. Score only windows
    # whose entire original interval contains finite outputs, at its original sample spacing.
    missing = np.r_[0, np.cumsum(~ok)]
    usable = missing[window + 1:] - missing[:len(ok) - window] == 0
    if not usable.any():
        return float("nan")
    if align:
        Rz, _ = align_yaw(est[ok], gt[ok])
        est = est @ Rz.T
    de = est[window:] - est[:-window]
    dg = gt[window:] - gt[:-window]
    return float(np.sqrt(np.mean(np.sum((de[usable] - dg[usable]) ** 2, 1))))


def drift_pct(est: np.ndarray, gt: np.ndarray) -> float:
    if not len(est) or not np.isfinite(est[-1]).all() or not np.isfinite(gt).all():
        return float("nan")
    length = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    return float(100 * np.linalg.norm(est[-1] - gt[-1]) / max(length, 1e-9))


def summary(est: np.ndarray, gt: np.ndarray, rate: float) -> dict:
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    if est.ndim != 2 or est.shape[1:] != (3,) or gt.shape != est.shape:
        raise ValueError("estimated and reference positions must have matching (N, 3) shape")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    finite = np.isfinite(est).all(axis=1) & np.isfinite(gt).all(axis=1)
    return {
        "ate_m": ate(est, gt),
        "rte_1s_m": rte(est, gt, int(rate)),
        "rte_10s_m": rte(est, gt, int(10 * rate)),
        "drift_pct": drift_pct(est, gt),
        "final_err_m": float(np.linalg.norm(est[-1] - gt[-1])) if len(est) else float("nan"),
        "samples": len(est),
        "finite_samples": int(finite.sum()),
        "missing_samples": int((~finite).sum()),
        "finite_fraction": float(finite.mean()) if len(est) else 0.0,
    }


# ---------------------------------------------------------------------------------------------
# Uncertainty evaluation (scoring only: uses the reference, never called by the estimator).
#
# For a d-dimensional Gaussian error e ~ N(0, P), the squared Mahalanobis distance e^T P^-1 e
# follows chi2(d). A scalar test |e| <= 3 sqrt(trace(P)) is NOT a 99.7 % region in 3-D: its
# probability depends on the shape of P and is not tied to any chi-square level. Coverage is
# therefore reported as the share of samples inside the chi2(3) region at a stated level.
# Consecutive samples are strongly correlated, so the effective sample size is far below N; treat
# the reported coverage as descriptive, not as a test with N independent draws.

def nees(est: np.ndarray, gt: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared per sample (NaN where P or est is not finite)."""
    e = est - gt
    out = np.full(len(e), np.nan)
    for i in range(len(e)):
        if np.all(np.isfinite(e[i])) and np.all(np.isfinite(P[i])):
            try:
                if not np.allclose(P[i], P[i].T, atol=1e-12, rtol=1e-8):
                    continue
                # A finite indefinite matrix is not a valid uncertainty region.
                L = np.linalg.cholesky(P[i])
                whitened = np.linalg.solve(L, e[i])
                out[i] = float(whitened @ whitened)
            except np.linalg.LinAlgError:
                pass
    return out


def coverage(nees_vals: np.ndarray, dof: int = 3, levels=(0.5, 0.9, 0.99, 0.997)) -> dict:
    """Empirical share of finite samples inside the chi2(dof) region at each nominal level."""
    from scipy.stats import chi2
    v = nees_vals[np.isfinite(nees_vals)]
    if len(v) == 0:
        return {lv: float("nan") for lv in levels}
    return {lv: float(np.mean(v <= chi2.ppf(lv, dof))) for lv in levels}
