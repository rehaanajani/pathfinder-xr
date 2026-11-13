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
    if align:
        Rz, tr = align_yaw(est, gt)
        est = est @ Rz.T + tr
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, 1))))


def rte(est: np.ndarray, gt: np.ndarray, window: int, align: bool = True) -> float:
    if len(gt) <= window:
        return float("nan")
    if align:
        Rz, _ = align_yaw(est, gt)
        est = est @ Rz.T
    de = est[window:] - est[:-window]
    dg = gt[window:] - gt[:-window]
    return float(np.sqrt(np.mean(np.sum((de - dg) ** 2, 1))))


def drift_pct(est: np.ndarray, gt: np.ndarray) -> float:
    length = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    return float(100 * np.linalg.norm(est[-1] - gt[-1]) / max(length, 1e-9))


def summary(est: np.ndarray, gt: np.ndarray, rate: float) -> dict:
    return {
        "ate_m": ate(est, gt),
        "rte_1s_m": rte(est, gt, int(rate)),
        "rte_10s_m": rte(est, gt, int(10 * rate)),
        "drift_pct": drift_pct(est, gt),
        "final_err_m": float(np.linalg.norm(est[-1] - gt[-1])),
    }
