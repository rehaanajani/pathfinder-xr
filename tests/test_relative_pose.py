"""Relative-pose (visual odometry) fusion between camera keyframe clones (issue #2 B).

The visual events here are generated from the synthetic reference trajectory: a labelled test
fixture that isolates the fusion mathematics. Real events come from ``pathfinder.stereo_vo``.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pathfinder import synthetic
from pathfinder.estimator import (REASONS, ReferenceFreeInit, StreamingEstimator, TRACK_INERTIAL, TRACK_LOST, replay)
from pathfinder.filter import CAM_NOMINAL, CAM_OUTAGE, ESKF, FilterConfig, FilterLog, valid_covariance


def vo_fixture(seq, rate=20.0, std_t=0.005, std_r=0.002, seed=0, drop=(), wrong=(), start=0.0):
    """Frame-to-frame relative poses from the reference (DIAGNOSTIC fixture), with a keyframe at
    every frame and a fresh keyframe (no relative measurement) after every dropped interval."""
    rng = np.random.default_rng(seed)
    R = Rot.from_quat(seq.q)
    ev, kf, prev = [], 0, None
    for tf in np.arange(start, seq.t[-1], 1.0 / rate):
        k = int(round(tf * seq.rate))
        if any(a <= tf < b for a, b in drop):
            prev = None
            continue
        if prev is not None:
            kp, kprev = prev
            Ri = R[kprev].as_matrix()
            dp = Ri.T @ (seq.p[k] - seq.p[kprev]) + rng.normal(0, std_t, 3)
            dR = (R[kprev].inv() * R[k] * Rot.from_rotvec(rng.normal(0, std_r, 3))).as_matrix()
            if any(a <= tf < b for a, b in wrong):
                dp = dp + np.array([0.3, 0, 0])
            cov = np.diag([std_t ** 2] * 3 + [std_r ** 2] * 3)
            ev.append({"t": seq.t[k], "kind": "rel", "kf": kp, "dp": dp, "dR": dR, "cov": cov, "health": 1.0})
        kf += 1
        ev.append({"t": seq.t[k], "kind": "kf", "kf": kf})
        prev = (kf, k)
    return ev


def _random_state(seed=1):
    rng = np.random.default_rng(seed)
    f = ESKF(FilterConfig(), rng.normal(size=3), rng.normal(size=3), Rot.random(random_state=seed).as_quat())
    f.add_clone(("cam", 1), 3)
    f.p, f.R = f.p + rng.normal(size=3), f.R * Rot.from_rotvec(rng.normal(size=3) * 0.5)
    return f


def test_relative_pose_jacobian_matches_finite_differences():
    f = _random_state()
    h_t, h_R, H = f.relative_pose_measurement(("cam", 1))
    n = f.P.shape[0]
    eps = 1e-6
    num = np.zeros((6, n))
    for j in range(n):
        g = _random_state()
        dx = np.zeros(n)
        dx[j] = eps
        g.R = g.R * Rot.from_rotvec(dx[0:3])
        g.p = g.p + dx[6:9]
        g.clone_R[0] = g.clone_R[0] * Rot.from_rotvec(dx[15:18])
        g.clone_p[0] = g.clone_p[0] + dx[18:21]
        ht2, hR2, _ = g.relative_pose_measurement(("cam", 1))
        num[:3, j] = (ht2 - h_t) / eps
        num[3:, j] = Rot.from_matrix(h_R.T @ hR2).as_rotvec() / eps
    assert np.max(np.abs(num - H)) < 1e-4


def test_mixed_clone_kinds_are_pruned_independently():
    f = ESKF(FilterConfig(), np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1.0]))
    for k in range(5):
        f.add_clone(k, 2)
        f.add_clone(("cam", k), 3)
    assert [c for c in f.clone_k if isinstance(c, int)] == [3, 4]
    assert [c for c in f.clone_k if isinstance(c, tuple)] == [("cam", 2), ("cam", 3), ("cam", 4)]
    assert f.P.shape == (15 + 6 * 5, 15 + 6 * 5)
    assert np.allclose(f.P, f.P.T) and np.linalg.eigvalsh(f.P).min() > -1e-12   # clones are exact copies: PSD


def _run(seq, ev, **kw):
    cfg = FilterConfig(**kw)
    return replay(seq.t, seq.gyro[0], seq.acc[0], cfg, ReferenceFreeInit(), vision=ev)


def _start_frame_error(seq, out):
    """Error in the odometry frame defined by the first valid output (declared alignment: SE(3) at the
    first estimated pose, nothing fitted afterwards)."""
    i0 = int(np.argmax(np.isfinite(out["p"][:, 0]) & (out["track_state"] != 0)))
    Re, Rt = Rot.from_quat(out["q"][i0]), Rot.from_quat(seq.q[i0])
    A = Rt * Re.inv()
    p = A.apply(out["p"] - out["p"][i0]) + seq.p[i0]
    return np.linalg.norm(p - seq.p, axis=1), i0


def _start_frame_nees(seq, out, i0):
    Re, Rt = Rot.from_quat(out["q"][i0]), Rot.from_quat(seq.q[i0])
    A = Rt * Re.inv()
    p = A.apply(out["p"] - out["p"][i0]) + seq.p[i0]
    Am = A.as_matrix()
    e = p - seq.p
    P = np.einsum("ij,njk,lk->nil", Am, out["P_pos"], Am)
    return np.array([ei @ np.linalg.solve(Pi, ei) if np.all(np.isfinite(Pi)) else np.nan for ei, Pi in zip(e, P)])


def test_vision_only_relative_fusion_bounds_drift_and_stays_consistent():
    seq = synthetic.head_walk(40, seed=3)
    # The fixture's VO noise is drawn from exactly the covariance it reports, so asserting a validated
    # VO covariance is true here by construction (never for a real front end without held-out evidence).
    _, log_vo, out = _run(seq, vo_fixture(seq), vo_covariance_validated=True)
    _, _, out_imu = replay(seq.t, seq.gyro[0], seq.acc[0], FilterConfig(), ReferenceFreeInit(), cam=None)
    e_vo, i0 = _start_frame_error(seq, out)
    e_imu, _ = _start_frame_error(seq, out_imu)
    assert log_vo.vo_used > 600 and log_vo.vo_rejected < 0.02 * log_vo.vo_used
    assert np.nanmax(e_imu[i0:]) > 5 * np.nanmax(e_vo[i0:])
    # Odometry drifts (heading random walk from the per-frame rotation noise); what must hold is that
    # the reported covariance covers that drift in the start frame.
    i1 = i0 + 200
    nees = _start_frame_nees(seq, out, i0)[i1:]
    assert np.mean(nees < 14.16) > 0.9
    assert np.all(out["camera_state"][i0 + 50:] == CAM_NOMINAL)
    # Relative-pose NIS should follow chi2(6): mean near 6 once velocity has converged.
    nis = np.array([n for t, n, ok in log_vo.vo_nis if t > 10])
    assert 3.5 < nis.mean() < 9.0
    assert out["reasons"][-1] & REASONS["POSITION_FRAME_ARBITRARY"]
    assert out["valid"][i0 + 200:].mean() > 0.95
    # Default: VO covariance not validated -> best-effort poses, explicitly invalid with a reason.
    _, _, out_default = _run(seq, vo_fixture(seq))
    assert not out_default["valid"].any()
    started = out_default["track_state"] != 0
    assert np.all(out_default["reasons"][started] & REASONS["COVARIANCE_UNRELIABLE"])
    np.testing.assert_allclose(out_default["p"], out["p"])


def test_restart_needs_new_keyframe_and_outage_is_reported():
    seq = synthetic.head_walk(30, seed=4)
    ev = vo_fixture(seq, drop=[(12.0, 17.0)])
    _, log, out = _run(seq, ev)
    in_gap = (seq.t > 13.0) & (seq.t < 17.0)
    assert np.all(out["camera_state"][in_gap] == CAM_OUTAGE)
    assert np.all(np.isin(out["track_state"][in_gap], (TRACK_INERTIAL, TRACK_LOST)))
    assert not np.any(out["valid"][in_gap & (out["track_state"] == TRACK_LOST)])
    after = seq.t > 18.0
    assert np.mean(out["camera_state"][after] == CAM_NOMINAL) > 0.95
    # A relative measurement against a keyframe that is no longer in the state is never applied.
    est = StreamingEstimator(FilterConfig(), ReferenceFreeInit(), camera_mode="relative")
    for i in range(int(3.5 * 200)):
        est.add_imu(seq.t[i], seq.gyro[0, i], seq.acc[0, i])
    est.add_camera_relative(seq.t[701], 999, np.zeros(3), np.eye(3), np.eye(6) * 1e-4)
    est.add_imu(seq.t[701], seq.gyro[0, 701], seq.acc[0, 701])
    assert est.log.vo_no_anchor == 1 and est.log.vo_used == 0


def test_wrong_visual_motion_is_gated_not_fused():
    seq = synthetic.head_walk(30, seed=5)
    clean = _run(seq, vo_fixture(seq))[2]
    _, log, out = _run(seq, vo_fixture(seq, wrong=[(15.0, 16.0)]))
    assert log.vo_rejected >= 18
    e_c, i0 = _start_frame_error(seq, clean)
    e_w, _ = _start_frame_error(seq, out)
    assert np.nanmax(e_w[i0:]) < np.nanmax(e_c[i0:]) + 0.05


def test_relative_mode_rejects_absolute_fixes_and_mixing():
    est = StreamingEstimator(camera_mode="relative")
    with pytest.raises(ValueError):
        est.add_camera(0.1, np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        StreamingEstimator().add_camera_keyframe(0.1, 1)
    seq = synthetic.head_walk(5, seed=0)
    with pytest.raises(ValueError):
        replay(seq.t, seq.gyro[0], seq.acc[0], FilterConfig(), ReferenceFreeInit(),
               cam={"t": np.zeros(0), "fix": np.zeros((0, 3)), "health": np.zeros(0)}, vision=[])
    log = FilterLog()
    f = _random_state()
    assert f.relative_pose_update(("cam", 1), np.zeros(3), np.eye(3) * 2, np.eye(6), 1.0, log) == "invalid"


def _bias_after_outage(seq, reanchor):
    """Accelerometer bias step during a 5 s VO outage (as in the room1 'blackout + IMU bias' scenario),
    with the small datasheet-style IMU noise used for TUM-VI, so the gate rejects VO after the outage."""
    acc = seq.acc[0].copy()
    acc[seq.t >= 12.0] += np.array([0.3, -0.2, 0.15])
    ev = vo_fixture(seq, drop=[(12.0, 17.0)])
    cfg = FilterConfig(vo_reanchor=reanchor, sigma_g=0.00016, sigma_a=0.0028, sigma_bg=2.2e-5, sigma_ba=8.6e-4)
    return replay(seq.t, seq.gyro[0], acc, cfg, ReferenceFreeInit(), vision=ev)


def test_persistent_healthy_vo_after_an_outage_reanchors_a_diverged_inertial_state():
    seq = synthetic.head_walk(35, seed=6)
    _, log, out = _bias_after_outage(seq, True)
    _, log_off, out_off = _bias_after_outage(seq, False)
    e, i0 = _start_frame_error(seq, out)
    e_off, _ = _start_frame_error(seq, out_off)
    assert log.vo_reanchored >= 1 and log_off.vo_reanchored == 0
    first = min(t for t, kind, _ in log.events if kind == "vo_reanchored")
    assert 17.0 < first < 19.5                         # after the outage: inertial state suspect after ~1 s
    late = seq.t > 25
    assert np.nanmax(e[late]) < 0.5 * np.nanmax(e_off[late])
    assert log_off.vo_rejected > 10 * log.vo_rejected   # without recovery every later VO update is rejected
    k = np.searchsorted(out["t"], first)
    assert out["reasons"][k] & REASONS["CAMERA_REANCHORED"]


def test_wrong_vo_while_anchored_is_held_out_not_reanchored():
    # Shorter than vo_recover_after_jump_s (2 s): rejected, never re-anchored. A wrong but healthy-looking
    # relative stream that persists beyond the hold-out IS eventually accepted: with one camera and one
    # IMU that case is indistinguishable from an inertial fault (documented restriction).
    seq = synthetic.head_walk(30, seed=7)
    _, log, out = _run(seq, vo_fixture(seq, wrong=[(15.0, 16.8)]))
    assert log.vo_reanchored == 0 and log.vo_rejected >= 30
