"""Score an estimated_trajectory.txt against ground truth, mirroring the Track 3 metrics.

    python tools/score_track3.py <estimated_trajectory.txt> <ground-truth csv> [--truth truth.json] [--input imu csv]

Ground truth: EuRoC state_groundtruth_estimate0/data.csv or TUM-VI mocap0/data.csv (#t_ns, p xyz, q wxyz, ...).
Reports (metres): ATE RMSE after SE(3) alignment (as `evo_ape -a`), translational RPE over 1 s, and the three
closeness parts of the brief: checkpoints (10 evenly spaced mission times), the blind 5 s after IMU failure, and
inside dropouts. Alignment is fitted on the mission part only (before the failure), so the blind 5 s is scored as a
genuine prediction. Also checks the output contract (one pose per input row, seconds, unit quaternions).
"""
import argparse
import json
import sys

import numpy as np


def umeyama(src, dst):
    """R, t minimising |R src + t - dst| (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, md - R @ ms


def score(est_path, gt_path, truth=None, input_csv=None):
    est = np.loadtxt(est_path, comments="#")
    gt = np.loadtxt(gt_path, delimiter=",", comments="#")
    te, pe, qe = est[:, 0], est[:, 1:4], est[:, 4:8]
    tg, pg = gt[:, 0] * 1e-9, gt[:, 1:4]
    res = {"rows": int(len(est))}
    if input_csv:
        tin = np.loadtxt(input_csv, delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
        res["rows_match_input"] = bool(len(tin) == len(te) and np.allclose(tin, te, atol=1e-6))
    res["quat_unit"] = bool(np.allclose(np.linalg.norm(qe, axis=1), 1, atol=1e-6))
    inside = (te >= tg[0]) & (te <= tg[-1])
    te, pe = te[inside], pe[inside]
    pgi = np.stack([np.interp(te, tg, pg[:, i]) for i in range(3)], 1)
    t0 = truth["t0_ns"] * 1e-9 if truth else te[0]
    rel = te - t0
    t_fail = truth["t_fail_s"] if truth else rel[-1]
    mission = rel < t_fail
    R, t = umeyama(pe[mission], pgi[mission])
    pa = pe @ R.T + t
    err = np.linalg.norm(pa - pgi, axis=1)
    R2, t2 = umeyama(pe, pgi)
    res["ate_rmse_m"] = float(np.sqrt(np.mean(np.sum((pe @ R2.T + t2 - pgi) ** 2, 1))))
    # RPE: translation error over 1 s windows (frame-independent)
    k = int(round(1.0 / np.median(np.diff(te))))
    if len(te) > k:
        d_est = pa[k:] - pa[:-k]
        d_gt = pgi[k:] - pgi[:-k]
        res["rpe_1s_m"] = float(np.sqrt(np.mean(np.sum((d_est - d_gt) ** 2, 1))))
    cps = np.linspace(0, t_fail, 12)[1:-1]
    res["checkpoints_m"] = float(np.mean([err[np.argmin(abs(rel - c))] for c in cps]))
    blind = ~mission
    if blind.any():
        res["blind_mean_m"] = float(err[blind].mean())
        res["blind_end_m"] = float(err[blind][-1])
    if truth and truth.get("dropouts_s"):
        m = np.zeros(len(rel), bool)
        for s0, s1 in truth["dropouts_s"]:
            m |= (rel >= s0) & (rel < s1 + 1.0)
        if m.any():
            res["dropout_m"] = float(err[m].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est"); ap.add_argument("gt")
    ap.add_argument("--truth"); ap.add_argument("--input")
    a = ap.parse_args()
    truth = json.load(open(a.truth)) if a.truth else None
    print(json.dumps(score(a.est, a.gt, truth, a.input), indent=1))


if __name__ == "__main__":
    main()
