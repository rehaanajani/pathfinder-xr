"""Issue #1 sections D, E: filter mathematics, numerical stability, reference-free streaming API."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from pathfinder import metrics, pipeline
from pathfinder.estimator import (REASONS, TRACK_INITIALIZING, TRACK_LOST, ReferenceFreeInit,
                                  StreamingEstimator, TruthInit, reason_names, replay)
from pathfinder.faults import Fault
from pathfinder.filter import ESKF, FilterConfig, G, run
from pathfinder.synthetic import head_walk

from .helpers import assert_recovered_validity, displacement_oracle, err, run_shipped

FS = 200


def _state(f):
    return f.R, f.v.copy(), f.p.copy(), f.bg.copy(), f.ba.copy()


def _err(nom, tru):
    R, v, p, bg, ba = nom
    Rt, vt, pt, bgt, bat = tru
    return np.r_[(R.inv() * Rt).as_rotvec(), vt - v, pt - p, bgt - bg, bat - ba]


def test_propagation_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    cfg = FilterConfig()
    f = ESKF(cfg, rng.normal(size=3), rng.normal(size=3), Rot.random(random_state=1).as_quat())
    f.bg, f.ba = rng.normal(0, 0.01, 3), rng.normal(0, 0.1, 3)
    gyro, acc, dt = np.array([0.8, -1.2, 2.0]), np.array([0.5, -0.3, 9.9]), 0.005
    nom0 = _state(f)
    P0 = f.P.copy()
    f.P = np.eye(15)
    f.propagate(gyro, acc, dt)
    nom1 = _state(f)
    # F from the filter: P1 = F I F^T + Q  ->  recover F from columns by propagating unit errors.
    eps = 1e-6
    Fnum = np.zeros((15, 15))
    for j in range(15):
        dx = np.zeros(15)
        dx[j] = eps
        g = ESKF(cfg, nom0[2] + dx[6:9], nom0[1] + dx[3:6], (nom0[0] * Rot.from_rotvec(dx[0:3])).as_quat())
        g.bg, g.ba = nom0[3] + dx[9:12], nom0[4] + dx[12:15]
        g.propagate(gyro, acc, dt)
        Fnum[:, j] = _err(nom1, _state(g)) / eps
    # Rebuild the filter's F by the same formulas (propagate() exposes it only through P).
    Rm = nom0[0].as_matrix()
    w, fb = gyro - nom0[3], acc - nom0[4]
    from pathfinder.filter import skew
    F = np.eye(15)
    F[0:3, 0:3] = Rot.from_rotvec(w * dt).as_matrix().T
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 0:3] = -Rm @ skew(fb) * dt
    F[3:6, 12:15] = -Rm * dt
    F[6:9, 0:3] = -0.5 * Rm @ skew(fb) * dt * dt
    F[6:9, 3:6] = np.eye(3) * dt
    F[6:9, 12:15] = -0.5 * Rm * dt * dt
    assert np.allclose(Fnum, F, atol=5e-5)
    Q = f.P - F @ F.T
    assert np.allclose(Q, np.diag(np.diag(Q)), atol=1e-12) and np.all(np.diag(Q) >= -1e-14)
    del P0


def test_reset_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    R = Rot.random(random_state=2)
    dx = rng.normal(0, 0.05, 3)
    R_new = R * Rot.from_rotvec(dx)
    eps = 1e-7
    J = np.zeros((3, 3))
    th0 = dx
    for j in range(3):
        th = th0.copy()
        th[j] += eps
        Rt = R * Rot.from_rotvec(th)
        J[:, j] = ((R_new.inv() * Rt).as_rotvec() - (R_new.inv() * (R * Rot.from_rotvec(th0))).as_rotvec()) / eps
    from pathfinder.filter import skew
    assert np.allclose(J, np.eye(3) - skew(0.5 * dx), atol=2e-3)


@pytest.mark.parametrize("yaw_frame", [False, True])
def test_clone_measurement_jacobian(yaw_frame):
    cfg = FilterConfig(cloning=True, yaw_frame=yaw_frame)
    f = ESKF(cfg, [1.0, 2.0, 1.5], [0.3, -0.2, 0.0], Rot.from_euler("ZYX", [0.7, 0.1, -0.05]).as_quat())
    f.add_clone(0, 5)
    f.p = f.p + np.array([0.8, -0.4, 0.05])
    f.R = f.R * Rot.from_rotvec([0.0, 0.0, 0.3])
    h0, H = f.clone_measurement(0)
    eps = 1e-6
    for j in list(range(0, 3)) + list(range(6, 9)) + list(range(15, 21)):
        g = ESKF(cfg, f.p, f.v, f.R.as_quat())
        g.clone_k, g.clone_p, g.clone_R, g.P = [0], [f.clone_p[0].copy()], [f.clone_R[0]], f.P.copy()
        dx = np.zeros(21)
        dx[j] = eps
        g.p = g.p + dx[6:9]
        g.R = g.R * Rot.from_rotvec(dx[0:3])
        g.clone_R[0] = g.clone_R[0] * Rot.from_rotvec(dx[15:18])
        g.clone_p[0] = g.clone_p[0] + dx[18:21]
        h1, _ = g.clone_measurement(0)
        assert np.allclose((h1 - h0) / eps, H[:, j], atol=1e-4), j


def test_clone_removal_keeps_cross_covariance():
    f = ESKF(FilterConfig(cloning=True), np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1.0]))
    for k in range(8):
        f.propagate(np.array([0.1, 0, 0.2]), np.array([0.1, 0, 9.81]), 0.005)
        f.add_clone(k, 3)
    assert f.P.shape == (15 + 18, 15 + 18) and f.clone_k == [5, 6, 7]
    assert np.allclose(f.P, f.P.T) and np.linalg.eigvalsh(f.P).min() > -1e-12


def test_long_run_cloning_rejection_recovery_numerically_stable():
    s = head_walk(90, n_imus=3, seed=2)
    cfg = replace(pipeline.ours_cfg(), cloning=True, yaw_frame=True)
    faults = [Fault("camera_wrong_fix", a, 4, params={"offset": [1.0, 0, 0], "step": 0.0}) for a in (15, 35, 55)] + \
             [Fault("camera_blackout", 70, 8)]
    bad, cam = pipeline.prepare(s, faults, 2)
    fg, fa, info = pipeline.fuse_imus(bad)
    est = StreamingEstimator(cfg, TruthInit(s.p[0], s.v[0], s.q[0]), displacement_oracle(s))
    q = info["quality"]
    ci = 0
    worst_asym, min_eig = 0.0, np.inf
    for i in range(len(s.t)):
        while ci < len(cam["idx"]) and cam["idx"][ci] <= i:
            est.add_camera(s.t[cam["idx"][ci]], cam["fix"][ci], cam["health"][ci])
            ci += 1
        est.add_imu(s.t[i], fg[i], fa[i], q["q_add_g"][i], q["q_add_a"][i], q["state"][i])
        if i % 500 == 0:
            P = est.f.P
            assert np.all(np.isfinite(P))
            worst_asym = max(worst_asym, float(np.abs(P - P.T).max()))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(P).min()))
    assert worst_asym < 1e-12 and min_eig > -1e-12
    assert np.linalg.norm(est.f.p - s.p[-1]) < 0.15


def test_reference_free_run_does_not_read_reference_fields():
    s = head_walk(30, n_imus=3, seed=1)
    bad, cam = pipeline.prepare(s, [Fault("camera_blackout", 15, 4)], 1)   # fixture uses the reference
    fg, fa, info = pipeline.fuse_imus(bad)
    blind = s.copy()
    blind.p[:], blind.v[:], blind.q[:] = np.nan, np.nan, np.nan
    kw = dict(imu_quality=info["quality"], init="reference_free", return_outputs=True)
    a, _, oa = run(s, fg, fa, None, cam, None, pipeline.ours_cfg(), **kw)
    b, _, ob = run(blind, fg, fa, None, cam, None, pipeline.ours_cfg(), **kw)
    assert np.array_equal(a, b, equal_nan=True) and np.array_equal(oa["reasons"], ob["reasons"])
    ok = oa["valid"]
    assert not ok[:2 * FS].any()
    assert_recovered_validity(oa, slice(4 * FS, 15 * FS))
    assert_recovered_validity(oa, slice(21 * FS, None))
    # Inside the blackout, without a learned model, the estimate is withdrawn once its own
    # uncertainty passes max_valid_pos_std, and it says why.
    for i in np.flatnonzero(~ok[4 * FS:]) + 4 * FS:
        assert oa["reasons"][i] & (REASONS["UNCERTAINTY_LIMIT"] | REASONS["COVARIANCE_UNRELIABLE"])
    assert err(a, s)[ok].mean() < 0.06
    assert not any(oa["reasons"] & REASONS["TRUTH_INIT_DIAGNOSTIC"])


def test_truth_init_is_labelled_diagnostic():
    s = head_walk(5, n_imus=1, seed=0)
    _, _, out = run(s, s.gyro[0], s.acc[0], None, None, None, FilterConfig(), return_outputs=True)
    assert np.all(out["reasons"] & REASONS["TRUTH_INIT_DIAGNOSTIC"])


def test_imu_only_reference_free_start_is_honest():
    s = head_walk(20, n_imus=1, seed=0)
    _, log, out = run(s, s.gyro[0], s.acc[0], None, None, None, FilterConfig(), init="reference_free",
                      return_outputs=True)
    names = reason_names(out["reasons"][-1])
    assert {"HEADING_ARBITRARY", "VELOCITY_ASSUMED_ZERO", "POSITION_FRAME_ARBITRARY", "NO_CAMERA"} <= set(names)
    assert not out["valid"][-1] and out["track_state"][-1] == TRACK_LOST    # uncertainty grew past the limit


def test_direct_streaming_api_matches_replay_and_accepts_out_of_order_camera():
    s = head_walk(12, n_imus=1, seed=3)
    bad, cam = pipeline.prepare(s, [])
    cam_t = s.t[cam["idx"]]
    cfg = pipeline.ours_cfg()
    _, _, ref = replay(s.t, bad.gyro[0], bad.acc[0], cfg, ReferenceFreeInit(),
                       cam={"t": cam_t, "fix": cam["fix"], "health": cam["health"]})
    est = StreamingEstimator(cfg, ReferenceFreeInit())
    rows, ci = [], 0
    for i in range(len(s.t)):
        # Deliver camera events in pairs, the later one first (late delivery inside the lookahead).
        while ci + 1 < len(cam_t) and cam_t[ci + 1] <= s.t[i]:
            est.add_camera(cam_t[ci + 1], cam["fix"][ci + 1], cam["health"][ci + 1])
            est.add_camera(cam_t[ci], cam["fix"][ci], cam["health"][ci])
            ci += 2
        while ci < len(cam_t) and cam_t[ci] <= s.t[i]:
            est.add_camera(cam_t[ci], cam["fix"][ci], cam["health"][ci])
            ci += 1
        rows += est.add_imu(s.t[i], bad.gyro[0, i], bad.acc[0, i])
    rows += est.finish()
    assert len(rows) == len(s.t)
    p = np.array([r["p"] for r in rows])
    ok = ref["valid"]
    assert np.allclose(p[ok], ref["p"][ok], atol=1e-9)


def test_invalid_and_missing_inputs():
    s = head_walk(10, n_imus=1, seed=4)
    est = StreamingEstimator(FilterConfig(), TruthInit(s.p[0], s.v[0], s.q[0]))
    with pytest.raises(ValueError):
        est.add_imu(0.0, np.zeros(2), np.zeros(3))
    est.add_imu(s.t[0], s.gyro[0, 0], s.acc[0, 0])
    with pytest.raises(ValueError):
        est.add_imu(s.t[0], s.gyro[0, 1], s.acc[0, 1])                  # non-increasing time
    est.add_camera(s.t[1] + 1e-4, np.array([np.nan, 0, 0]), 1.0)          # invalid fix
    est.add_camera(s.t[1] + 2e-4, s.p[1], np.nan)                         # invalid health
    r = est.add_imu(s.t[1], np.full(3, np.nan), s.acc[0, 1])[0]           # non-finite IMU sample
    assert "IMU_LOST" in reason_names(r["reasons"]) and np.all(np.isfinite(r["p"]))
    est.add_imu(s.t[2], s.gyro[0, 2], s.acc[0, 2])
    assert est.log.cam_invalid == 2
    est.add_camera(s.t[1], s.p[1], 1.0)                                   # older than the state
    assert est.log.cam_late == 1
    P_before = est.f.P[6:9, 6:9].trace()
    r = est.add_imu(s.t[2] + 0.5, s.gyro[0, 3], s.acc[0, 3])[0]           # 0.5 s IMU gap
    assert "IMU_GAP" in reason_names(r["reasons"]) and est.f.P[6:9, 6:9].trace() > P_before + 0.05


def test_empty_and_single_sample():
    pos, log, out = replay(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), FilterConfig(),
                           TruthInit(np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1.0])))
    assert pos.shape == (0, 3)
    pos, log, out = replay(np.zeros(1), np.zeros((1, 3)), -G[None], FilterConfig(),
                           TruthInit(np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1.0])))
    assert pos.shape == (1, 3) and out["valid"][0]
    pos, log, out = replay(np.arange(5) / FS, np.zeros((5, 3)), np.tile(-G, (5, 1)), FilterConfig(),
                           ReferenceFreeInit())
    assert np.all(out["track_state"] == TRACK_INITIALIZING) and np.all(np.isnan(pos))
    with pytest.raises(ValueError):
        replay(np.arange(3) / FS, np.zeros((2, 3)), np.zeros((3, 3)), FilterConfig(), ReferenceFreeInit())


def test_irregular_imu_timestamps():
    s = head_walk(25, n_imus=3, seed=5)
    rng = np.random.default_rng(1)
    keep = np.sort(rng.choice(len(s.t), int(0.8 * len(s.t)), replace=False))   # 20 % dropped samples
    keep[0] = 0
    sub = s.copy()
    sub.t, sub.gyro, sub.acc, sub.p, sub.q, sub.v = s.t[keep], s.gyro[:, keep], s.acc[:, keep], s.p[keep], s.q[keep], s.v[keep]
    bad, cam = pipeline.prepare(s, [])
    fg, fa, info = pipeline.fuse_imus(sub)
    pos_of = {k: j for j, k in enumerate(keep)}
    cam2 = {"idx": np.array([pos_of[i] for i in cam["idx"] if i in pos_of]),
            "fix": np.array([cam["fix"][n] for n, i in enumerate(cam["idx"]) if i in pos_of]),
            "health": np.array([cam["health"][n] for n, i in enumerate(cam["idx"]) if i in pos_of])}
    est, log = run(sub, fg, fa, None, cam2, None, pipeline.ours_cfg(), imu_quality=info["quality"])
    assert metrics.ate(est, sub.p) < 0.06


def test_validity_and_reasons_when_both_modalities_degrade():
    s = head_walk(45, n_imus=3, seed=6)
    f = [Fault("camera_blackout", 12, 25), Fault("imu_noise", 12, 25, None, {"gyro_std": 0.08, "acc_std": 0.8})]
    est, log, out, _ = run_shipped(s, f, 6)
    sl = slice(14 * FS, 36 * FS)
    assert np.all(out["reasons"][sl] & REASONS["CAMERA_OUTAGE"])
    assert np.mean((out["reasons"][sl] & REASONS["BOTH_DEGRADED"]) > 0) > 0.8
    assert not out["valid"][30 * FS:36 * FS].any()                      # uncertainty limit reached
    nv = metrics.nees(est[sl], s.p[sl], out["P_pos"][sl])
    assert np.mean(nv <= 14.16) > 0.95                                  # conservative, not overconfident
    assert_recovered_validity(out, slice(40 * FS, None))
    assert err(est, s)[40 * FS:].max() < 0.15


def test_clean_uncertainty_is_not_overconfident():
    s = head_walk(40, n_imus=3, seed=7)
    est, log, out, _ = run_shipped(s, [], 7)
    cov = metrics.coverage(metrics.nees(est, s.p, out["P_pos"]))
    assert cov[0.997] > 0.99 and cov[0.9] > 0.85
