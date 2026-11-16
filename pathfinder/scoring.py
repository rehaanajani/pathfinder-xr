"""Trajectory scoring against a reference (evaluation only; never imported by runtime code).

Declared alignment policies (protocol ``docs/PROTOCOL.md``):
    start_se3   rigid SE(3) at the first initialised estimate: the reference pose at that time defines
                the odometry frame. Nothing is fitted afterwards, so drift is fully visible. PRIMARY.
    full_4dof   yaw + translation fitted once over all valid samples (gravity makes roll/pitch
                observable for VIO). SECONDARY, comparable with VIO literature.
Alignment is never redone per outage. Samples without reference support (mocap gap > ``max_gap``
or outside the reference interval) are excluded and counted, never interpolated across.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.stats import chi2


def reference_at(ref_t, ref_p, ref_q, t, max_gap=0.05):
    """Interpolate reference poses at t; returns (p, q, supported mask)."""
    t = np.asarray(t, float)
    inside = (t >= ref_t[0]) & (t <= ref_t[-1])
    j = np.clip(np.searchsorted(ref_t, t), 1, len(ref_t) - 1)
    gap = ref_t[j] - ref_t[j - 1]
    exact = np.searchsorted(ref_t, t)
    exact_hit = (exact < len(ref_t)) & (ref_t[np.minimum(exact, len(ref_t) - 1)] == t)
    ok = inside & ((gap <= max_gap) | exact_hit)
    p = np.full((len(t), 3), np.nan)
    q = np.full((len(t), 4), np.nan)
    if ok.any():
        p[ok] = np.stack([np.interp(t[ok], ref_t, ref_p[:, i]) for i in range(3)], 1)
        q[ok] = Slerp(ref_t, Rot.from_quat(ref_q))(t[ok]).as_quat()
    return p, q, ok


def align_start_se3(p, q, gp, gq, i0):
    A = Rot.from_quat(gq[i0]) * Rot.from_quat(q[i0]).inv()
    pa = A.apply(p - p[i0]) + gp[i0]
    good_q = np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) > 0)
    qa = (A * Rot.from_quat(np.where(good_q[:, None], q, [0, 0, 0, 1]))).as_quat()
    return pa, qa, A


def align_4dof(p, gp, mask):
    """Least-squares yaw + translation (Umeyama restricted to rotation about z)."""
    x, y = p[mask], gp[mask]
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    s = np.sum(xc[:, 0] * yc[:, 1] - xc[:, 1] * yc[:, 0])
    c = np.sum(xc[:, 0] * yc[:, 0] + xc[:, 1] * yc[:, 1])
    yaw = np.arctan2(s, c)
    A = Rot.from_euler("z", yaw)
    return A.apply(p - mx) + my, A


def rotation_error_deg(qa, gq):
    return np.rad2deg(np.linalg.norm((Rot.from_quat(gq).inv() * Rot.from_quat(qa)).as_rotvec(), axis=1))


def relative_errors(t, pa, qa, gp, gq, mask, delta):
    """RPE over time delta (s): translation (m) and rotation (deg) of the motion between t and t+delta,
    expressed in the reference body frame at t; pairs must both be scored samples."""
    j = np.searchsorted(t, t + delta)
    ok = (j < len(t)) & mask
    ok[ok] &= mask[j[ok]] & (np.abs(t[j[ok]] - t[ok] - delta) < 0.05)
    i, j = np.where(ok)[0], j[ok]
    if len(i) == 0:
        return np.zeros(0), np.zeros(0)
    Rg_i, Re_i = Rot.from_quat(gq[i]), Rot.from_quat(qa[i])
    dg = Rg_i.inv().apply(gp[j] - gp[i])
    de = Re_i.inv().apply(pa[j] - pa[i])
    rg = Rg_i.inv() * Rot.from_quat(gq[j])
    re = Re_i.inv() * Rot.from_quat(qa[j])
    return np.linalg.norm(de - dg, axis=1), np.rad2deg(np.linalg.norm((rg.inv() * re).as_rotvec(), axis=1))


def nees_pos(err, P, A=None):
    """Position NEES with the covariance rotated into the aligned frame."""
    if A is not None:
        M = A.as_matrix()
        P = np.einsum("ij,njk,lk->nil", M, P, M)
    out = np.full(len(err), np.nan)
    ok = np.all(np.isfinite(P).reshape(len(P), -1), 1) & np.all(np.isfinite(err), 1)
    for k in np.where(ok)[0]:
        try:
            out[k] = err[k] @ np.linalg.solve(P[k], err[k])
        except np.linalg.LinAlgError:
            pass
    return out


def coverage(nees, levels=(0.5, 0.9, 0.99, 0.997), dof=3):
    n = nees[np.isfinite(nees)]
    return {str(l): (float(np.mean(n <= chi2.ppf(l, dof))) if len(n) else None) for l in levels}


def recovery_time(t, err, t_end, bound, hold=2.0, max_gap=0.05):
    """First time after t_end from which err stays <= bound for ``hold`` s (None = never recovers).

    This is a sustained criterion: crossing the bound once does not count."""
    after = t >= t_end
    ta, ea = t[after], err[after]
    ok = np.isfinite(ea) & (ea <= bound)
    for k in np.where(ok)[0]:
        win = (ta >= ta[k]) & (ta <= ta[k] + hold)
        stop = np.searchsorted(ta, ta[k] + hold)
        if (stop < len(ta) and ta[stop] - (ta[k] + hold) <= max_gap
                and np.all(ok[k:stop + 1]) and np.all(np.diff(ta[k:stop + 1]) <= max_gap)):
            return float(ta[k] - t_end)
    return None


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "rmse": float(np.sqrt(np.mean(x ** 2))), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def score(out, ref, faults=(), rec_bound=0.10, rec_hold=2.0, rpe_deltas=(1.0, 10.0), max_gap=0.05,
          jump_window=2.0, segments=None):
    """All metrics for one estimator output (dict of arrays with t, p, q, P_pos, valid)."""
    t = out["t"]
    gp, gq, sup = reference_at(ref.t, ref.p, ref.q, t, max_gap)
    est = (np.all(np.isfinite(out["p"]), 1) & np.all(np.isfinite(out["q"]), 1)
           & (np.linalg.norm(out["q"], axis=1) > 0))
    valid = out["valid"].astype(bool) & est
    res = {"samples": int(len(t)), "reference_supported": int(sup.sum()),
           "unsupported_excluded": int((~sup).sum()), "estimated": int((est & sup).sum()),
           "valid": int((valid & sup).sum()), "valid_fraction": float((valid & sup).sum() / max(sup.sum(), 1))}
    if "tracked" in out:
        res["tracked_fraction"] = float(np.sum(out["tracked"] & sup) / max(sup.sum(), 1))
    # Alignment anchor: the first initialised estimate with reference support. Outputs may be
    # best-effort (valid=False, e.g. uncertified covariance); they are scored over *all* estimated
    # samples and separately over valid ones, so an invalid flag never hides drift.
    started = est & sup
    if "track_state" in out:
        started &= np.asarray(out["track_state"]) != 0          # TRACK_INITIALIZING
    cand = np.where(started)[0]
    if not len(cand):
        res["failure"] = "no initialised estimate with reference support"
        return res, None
    i0 = int(cand[0])
    res["first_estimate_t"] = float(t[i0] - t[0])
    vv = np.where(valid & sup)[0]
    res["first_valid_t"] = float(t[vv[0]] - t[0]) if len(vv) else None
    res["estimated_fraction_after_start"] = float(np.mean(est[i0:] & sup[i0:])) if sup[i0:].any() else None
    pa, qa, A = align_start_se3(out["p"], out["q"], gp, gq, i0)
    scored = est & sup & (np.arange(len(t)) >= i0)
    err = np.linalg.norm(pa - gp, axis=1)
    err[~scored] = np.nan
    rot = rotation_error_deg(qa[scored], gq[scored])
    res["start_se3"] = {"ate_all": stats(err), "ate_valid": stats(np.where(valid, err, np.nan)),
                        "rot_deg_all": stats(rot), "final_err": float(err[np.where(scored)[0][-1]])}
    p4, _ = align_4dof(out["p"], gp, scored)
    res["full_4dof"] = {"ate_all": stats(np.linalg.norm(p4 - gp, axis=1)[scored])}
    for d in rpe_deltas:
        et, er = relative_errors(t, pa, qa, gp, gq, scored, d)
        res[f"rpe_{d:g}s"] = {"trans_m": stats(et), "rot_deg": stats(er)}
    nees = nees_pos(pa - gp, out["P_pos"], A)
    res["nees"] = {"valid": coverage(nees[valid & scored]), "all": coverage(nees[scored]),
                   "interpretation": "gauge-unadjusted marginal-covariance diagnostic, not calibrated coverage",
                   "mean_all": float(np.nanmean(nees[scored])) if np.isfinite(nees[scored]).any() else None,
                   "mean_valid": float(np.mean(nv)) if len(nv := nees[valid & scored & np.isfinite(nees)]) else None,
                   "n_finite": int(np.isfinite(nees[scored]).sum())}
    false_valid = valid & scored & (nees > chi2.ppf(0.997, 3))
    res["valid_outside_99.7"] = int(false_valid.sum())
    rel = t - t[0]
    step_dev = np.linalg.norm(np.diff(pa, axis=0) - np.diff(gp, axis=0), axis=1)
    sc = np.where(scored)[0]
    connected = scored[:-1] & scored[1:] & (np.diff(t) <= max_gap)
    path = float(np.sum(np.linalg.norm(np.diff(gp, axis=0)[connected], axis=1)))
    res["path_length_m"] = path
    res["final_drift_pct_of_path"] = 100 * res["start_se3"]["final_err"] / path if path > 0 else None
    # Declared time segments (s since the first sample), e.g. calibration vs within-recording hold-out.
    # Same alignment as the full run: the segment inherits the drift accumulated before it.
    res["segments"] = {}
    for name, (s0, s1) in (segments or {}).items():
        w = scored & (rel >= s0) & (rel < s1)
        res["segments"][name] = {"ate_all": stats(err[w]), "valid_fraction": float(valid[w].mean()) if w.any() else None,
                                 "rot_deg_all": stats(rotation_error_deg(qa[w], gq[w])) if w.any() else {"n": 0}}
    fr = []
    for f in faults:
        inside = scored & (rel >= f["start"]) & (rel < f["end"])
        before = np.where(scored & (rel < f["start"]) & (rel >= f["start"] - max_gap))[0]
        ending = np.where(scored & (np.abs(rel - f["end"]) <= max_gap))[0]
        e_start = float(err[before[-1]]) if len(before) else None
        e_end = float(err[ending[np.argmin(np.abs(rel[ending] - f["end"]))]]) if len(ending) else None
        # Discontinuity: largest per-sample position step beyond the reference's own step, within
        # ``jump_window`` s after the fault ends (re-anchoring jumps; normal motion is subtracted).
        w = scored & (rel >= f["end"]) & (rel < f["end"] + jump_window)
        jump = float(np.nanmax(step_dev[w[1:]])) if w[1:].any() and np.isfinite(step_dev[w[1:]]).any() else None
        recovery = recovery_time(rel, err, f["end"], e_start + rec_bound, rec_hold, max_gap) if e_start is not None else None
        fr.append({**f, "err_at_start": e_start, "err_at_end": e_end,
                   "boundary_status": "supported" if e_start is not None and e_end is not None else "insufficient_support",
                   "err_growth": e_end - e_start if e_end is not None and e_start is not None else None,
                   "max_err_inside": float(np.nanmax(err[inside])) if inside.any() else None,
                   "valid_fraction_inside": float(valid[inside].mean()) if inside.any() else None,
                   "recovery_s": recovery,
                   "recovery_status": "unknown_baseline" if e_start is None else ("recovered" if recovery is not None else "not_observed"),
                   "max_step_dev_after_m": jump, "nees_cov_inside_valid": coverage(nees[inside & valid]),
                   "nees_cov_inside_all": coverage(nees[inside])})
    res["faults"] = fr
    return res, {"t": rel, "err": err, "valid": valid, "nees": nees, "p_aligned": pa, "p_ref": gp}
