"""Score an estimated_trajectory.txt against ground truth.

    python tools/score_track3.py <estimated_trajectory.txt> <ground-truth csv> \\
        [--truth truth.json] [--input imu csv] [--mode legacy|support]

Two explicit versions (see docs/TRACK3_SCORE.md):

* ``legacy`` (``pathfinder.track3-score/1``, the default) is the scorer that produced
  ``results/runs/20261116-main-c382494-benchmark``. It linearly interpolates motion-capture
  gaps, including the 1.058 s and 2.492 s gaps in those rooms. A clean run with no failure
  time still emits ``blind_*`` for the last sample. Kept so those numbers can be reproduced.
* ``support`` (``pathfinder.track3-score/2``) scores only samples with temporal reference
  support. A bracket wider than ``--max-gap`` (default 50 ms) is unsupported unless the
  estimate timestamp hits a reference timestamp (``rtol=0``, ``atol=1e-6`` s). Samples
  before the first or after the last reference sample are unsupported and are not
  extrapolated. Coverage is reported. ``blind_*`` errors are omitted unless a failure time
  was declared and at least one supported sample lies in that window.

Both modes compare timestamps with ``rtol=0`` and ``atol=1e-6`` s. Ground truth is an
EuRoC ``state_groundtruth_estimate0/data.csv`` or a TUM-VI ``mocap0/data.csv``
(``#t_ns``, position xyz, quaternion wxyz, ...).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA_LEGACY = "pathfinder.track3-score/1"
SCHEMA_SUPPORT = "pathfinder.track3-score/2"
TS_ATOL = 1e-6
DEFAULT_MAX_GAP_S = 0.05


def umeyama(src, dst):
    """R, t minimising |R src + t - dst| (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, md - R @ ms


def _load_legacy(path, **kwargs):
    return np.loadtxt(path, **kwargs)


def _timestamps_match(tin, te):
    return bool(len(tin) == len(te) and np.allclose(tin, te, atol=TS_ATOL, rtol=0))


def _score_legacy(est_path, gt_path, truth=None, input_csv=None):
    est = _load_legacy(est_path, comments="#")
    gt = _load_legacy(gt_path, delimiter=",", comments="#")
    te, pe, qe = est[:, 0], est[:, 1:4], est[:, 4:8]
    tg, pg = gt[:, 0] * 1e-9, gt[:, 1:4]
    res = {"schema": SCHEMA_LEGACY, "mode": "legacy", "rows": int(len(est))}
    if input_csv:
        tin = _load_legacy(input_csv, delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
        res["rows_match_input"] = _timestamps_match(tin, te)
    res["quat_unit"] = bool(np.allclose(np.linalg.norm(qe, axis=1), 1, atol=1e-6))
    inside = (te >= tg[0]) & (te <= tg[-1])
    if not inside.any():
        res["status"] = "no_overlap"
        return res
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
    # 1 s drift: error of the displacement over each 1 s interval (pairs chosen by TIME, so missing rows do not
    # stretch the interval). Position only, in the aligned world frame: NOT evo's pose RPE, which compares
    # relative poses in the body frame and includes rotation. Use evo_rpe for that number.
    j = np.searchsorted(te, te + 1.0)
    ok = j < len(te)
    ok[ok] &= np.abs(te[j[ok]] - te[ok] - 1.0) < 0.05
    if ok.any():
        i, jj = np.flatnonzero(ok), j[ok]
        d_est = pa[jj] - pa[i]
        d_gt = pgi[jj] - pgi[i]
        res["position_drift_1s_m"] = float(np.sqrt(np.mean(np.sum((d_est - d_gt) ** 2, 1))))
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


def temporal_support(te, tg, max_gap):
    """Return supported, endpoint-outside, and inside-but-gapped masks over estimate times.

    ``supported`` is inside the reference span and either the bracketing reference gap is
    at most ``max_gap`` or the estimate time matches a bracketing reference timestamp.
    Nothing outside ``[tg[0], tg[-1]]`` is supported.
    """
    te = np.asarray(te, float)
    tg = np.asarray(tg, float)
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    if not np.isfinite(te).all() or not np.isfinite(tg).all():
        raise ValueError("timestamps must be finite")
    if np.any(np.diff(te) <= 0) or np.any(np.diff(tg) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    n = len(te)
    supported = np.zeros(n, bool)
    endpoint = np.ones(n, bool)
    gapped = np.zeros(n, bool)
    if n == 0 or len(tg) == 0:
        return supported, endpoint, gapped
    if len(tg) == 1:
        supported = np.isclose(te, tg[0], rtol=0, atol=TS_ATOL)
        return supported, ~supported, gapped
    inside = (te >= tg[0]) & (te <= tg[-1])
    endpoint = ~inside
    j = np.clip(np.searchsorted(tg, te, side="left"), 1, len(tg) - 1)
    gap = tg[j] - tg[j - 1]
    exact = (np.isclose(te, tg[j], rtol=0, atol=TS_ATOL)
             | np.isclose(te, tg[j - 1], rtol=0, atol=TS_ATOL))
    supported = inside & ((gap <= max_gap) | exact)
    gapped = inside & ~supported
    return supported, endpoint, gapped


def _fit(pe, pref, mask):
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return None
    rot, trans = umeyama(pe[idx], pref[idx])
    aligned = pe @ rot.T + trans
    err = np.linalg.norm(aligned - pref, axis=1)
    err[~np.isfinite(pref).all(axis=1)] = np.nan
    return aligned, err


def _read_table(path, what, **kwargs):
    if Path(path).stat().st_size == 0:
        raise ValueError(f"{path} has no {what} rows")
    try:
        arr = np.loadtxt(path, **kwargs)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} has no {what} rows") from exc
    arr = np.atleast_2d(arr)
    if arr.size == 0:
        raise ValueError(f"{path} has no {what} rows")
    return arr


def _score_support(est_path, gt_path, truth=None, input_csv=None, max_gap=DEFAULT_MAX_GAP_S):
    est = _read_table(est_path, "pose", comments="#")
    gt = _read_table(gt_path, "ground-truth", delimiter=",", comments="#")
    if est.shape[1] < 8:
        raise ValueError(f"{est_path} has no pose rows")
    if gt.shape[1] < 4:
        raise ValueError(f"{gt_path} has no ground-truth rows")
    if not np.isfinite(est[:, :8]).all() or not np.isfinite(gt[:, :4]).all():
        raise ValueError("pose and reference samples must be finite")
    te, pe, qe = est[:, 0], est[:, 1:4], est[:, 4:8]
    tg, pg = gt[:, 0] * 1e-9, gt[:, 1:4]
    supported, endpoint, gapped = temporal_support(te, tg, max_gap)
    res = {
        "schema": SCHEMA_SUPPORT,
        "mode": "support",
        "max_gap_s": float(max_gap),
        "rows": int(len(est)),
        "supported_rows": int(supported.sum()),
        "unsupported_rows": int((~supported).sum()),
        "endpoint_unsupported_rows": int(endpoint.sum()),
        "gap_unsupported_rows": int(gapped.sum()),
        "support_coverage": float(supported.mean()) if len(te) else 0.0,
        "failure_declared": bool(truth is not None and "t_fail_s" in truth),
    }
    if input_csv:
        tin = np.atleast_1d(np.loadtxt(input_csv, delimiter=",", comments="#", usecols=0, dtype=np.int64)) * 1e-9
        res["rows_match_input"] = _timestamps_match(tin, te)
    res["quat_unit"] = bool(np.allclose(np.linalg.norm(qe, axis=1), 1, atol=1e-6, rtol=0))
    if not supported.any():
        res["status"] = "no_supported_samples"
        return res
    inside = ~endpoint
    if truth and "t0_ns" in truth:
        t0 = float(truth["t0_ns"]) * 1e-9
    elif inside.any():
        t0 = float(te[np.flatnonzero(inside)[0]])
    else:
        t0 = float(te[0])
    rel = te - t0
    if res["failure_declared"]:
        t_fail = float(truth["t_fail_s"])
        if not np.isfinite(t_fail) or t_fail < 0:
            raise ValueError("t_fail_s must be finite and nonnegative")
        mission = supported & (rel < t_fail)
        blind = rel >= t_fail
        if "blind_s" in truth:
            blind_s = float(truth["blind_s"])
            if not np.isfinite(blind_s) or blind_s <= 0:
                raise ValueError("blind_s must be finite and positive")
            blind &= rel <= t_fail + blind_s + TS_ATOL
    else:
        scored_rel = rel[inside] if inside.any() else rel
        t_fail = float(scored_rel[-1])
        mission = supported
        blind = np.zeros(len(te), bool)
    pgi = np.full((len(te), 3), np.nan)
    pgi[supported] = np.column_stack([np.interp(te[supported], tg, pg[:, i]) for i in range(3)])
    ate = _fit(pe, pgi, supported)
    if ate is None:
        res["status"] = "insufficient_support"
        return res
    _, ate_err = ate
    res["ate_rmse_m"] = float(np.sqrt(np.mean(ate_err[supported] ** 2)))
    res["status"] = "ok"
    mission_fit = _fit(pe, pgi, mission)
    if mission_fit is None:
        res["mission_alignment"] = "insufficient_support"
        _blind_coverage(res, blind, supported)
        return res
    pa, err = mission_fit
    res["mission_alignment"] = "se3_supported_mission"
    ts = te[supported]
    sup_i = np.flatnonzero(supported)
    if len(ts) >= 2:
        j = np.searchsorted(ts, ts + 1.0)
        ok = j < len(ts)
        ok[ok] &= np.abs(ts[j[ok]] - ts[ok] - 1.0) < 0.05
        if ok.any():
            i, jj = np.flatnonzero(ok), j[ok]
            # Both ends are supported. Pairing is by timestamp, so a dropped middle row
            # does not stretch the 1 s interval. Alignment is the mission fit.
            d_est = pa[sup_i[jj]] - pa[sup_i[i]]
            d_gt = pgi[sup_i[jj]] - pgi[sup_i[i]]
            res["position_drift_1s_m"] = float(np.sqrt(np.mean(np.sum((d_est - d_gt) ** 2, 1))))
            res["position_drift_1s_pairs"] = int(ok.sum())
    cps = np.linspace(0, t_fail, 12)[1:-1]
    found = []
    mission_i = np.flatnonzero(mission)
    for c in cps:
        if not len(mission_i):
            break
        k = int(np.argmin(np.abs(rel[mission_i] - c)))
        if abs(rel[mission_i[k]] - c) <= max_gap:
            found.append(float(err[mission_i[k]]))
    res["checkpoints_declared"] = int(len(cps))
    res["checkpoints_supported"] = int(len(found))
    if found:
        res["checkpoints_m"] = float(np.mean(found))
    _blind_coverage(res, blind, supported, err)
    if truth and truth.get("dropouts_s"):
        m = np.zeros(len(rel), bool)
        for s0, s1 in truth["dropouts_s"]:
            m |= (rel >= s0) & (rel < s1 + 1.0)
        covered = m & supported
        res["dropout_rows"] = int(m.sum())
        res["dropout_supported_rows"] = int(covered.sum())
        res["dropout_support_coverage"] = float(covered.sum() / m.sum()) if m.any() else 0.0
        if covered.any():
            res["dropout_m"] = float(err[covered].mean())
    return res


def _blind_coverage(res, blind, supported, err=None):
    if not res["failure_declared"]:
        return
    covered = blind & supported
    res["blind_rows"] = int(np.asarray(blind).sum())
    res["blind_supported_rows"] = int(covered.sum())
    res["blind_support_coverage"] = float(covered.sum() / blind.sum()) if np.asarray(blind).any() else 0.0
    if err is not None and covered.any():
        res["blind_status"] = "supported"
        res["blind_mean_m"] = float(err[covered].mean())
        res["blind_last_supported_m"] = float(err[np.flatnonzero(covered)[-1]])
        # An earlier reference sample is not evidence for the end of the window.
        end = np.flatnonzero(blind)[-1]
        res["blind_end_supported"] = bool(supported[end])
        if supported[end]:
            res["blind_end_m"] = float(err[end])
    else:
        res["blind_status"] = "unsupported"


def score(est_path, gt_path, truth=None, input_csv=None, mode="legacy", max_gap=DEFAULT_MAX_GAP_S):
    if mode == "legacy":
        return _score_legacy(est_path, gt_path, truth, input_csv)
    if mode == "support":
        return _score_support(est_path, gt_path, truth, input_csv, max_gap)
    raise ValueError(f"unknown scoring mode {mode!r}; expected 'legacy' or 'support'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est")
    ap.add_argument("gt")
    ap.add_argument("--truth")
    ap.add_argument("--input")
    ap.add_argument("--mode", choices=("legacy", "support"), default="legacy")
    ap.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_S)
    a = ap.parse_args()
    truth = json.load(open(a.truth)) if a.truth else None
    print(json.dumps(score(a.est, a.gt, truth, a.input, a.mode, a.max_gap), indent=1))


if __name__ == "__main__":
    sys.exit(main())
