"""Regressions for the issue #1 completion pass; no dataset or trained network required."""

import numpy as np
import pytest

from pathfinder import metrics, pipeline
from pathfinder.estimator import REASONS, StreamingEstimator, TruthInit, replay
from pathfinder.filter import ESKF, FilterConfig, FilterLog
from pathfinder.imu_health import LOST, WARMUP, ImuGeometry, ImuMonitor, MonitorConfig


def start(cfg=None, predictor=None):
    return StreamingEstimator(cfg, TruthInit(np.zeros(3), np.zeros(3), np.array([0., 0., 0., 1.])), predictor)


@pytest.mark.parametrize("noise", [-1., np.inf, np.nan])
def test_bad_noise_rejected_before_state_mutation(noise):
    e = start()
    with pytest.raises(ValueError, match="noise densities"):
        e.add_imu(0., np.zeros(3), [0., 0., 9.81], q_add_a=noise)
    assert e.f is None and e.boot == []


@pytest.mark.parametrize("cov", [np.zeros((3, 3)), -np.eye(3), np.ones(3),
                               np.diag([1., 1., -1.]), np.full((3, 3), np.nan),
                               np.array([[1., .5, 0.], [0., 1., 0.], [0., 0., 1.]])])
def test_invalid_model_covariance_never_changes_filter(cov):
    f = ESKF(FilterConfig(), np.zeros(3), np.zeros(3), [0., 0., 0., 1.])
    f.last_good_t = 0.
    before = f.P.copy()
    log = FilterLog()
    f.displacement_update(np.ones(3), cov, np.zeros(3), np.eye(3), log, .1)
    assert log.disp_skipped == 1 and log.disp_used == 0
    np.testing.assert_array_equal(f.P, before)
    np.testing.assert_array_equal(f.p, np.zeros(3))
    assert f.model_scale == 1.


def test_default_withholds_known_inconsistent_model_during_outage():
    f = ESKF(pipeline.ours_cfg(), np.zeros(3), np.zeros(3), [0., 0., 0., 1.])
    f.model_scale = 100.
    log = FilterLog()
    before = f.P.copy()
    f.displacement_update(np.ones(3), np.eye(3) * .01, np.zeros(3), np.eye(3), log, 10.)
    assert log.disp_used == 0 and log.disp_withheld == 1
    np.testing.assert_array_equal(f.P, before)


def test_healthy_visual_tracking_checks_model_without_fusing_correlated_prediction():
    f = ESKF(pipeline.ours_cfg(), np.zeros(3), np.zeros(3), [0., 0., 0., 1.])
    f.last_good_t = 0.
    before = f.P.copy()
    log = FilterLog()
    f.displacement_update(np.ones(3), np.eye(3) * .001, np.zeros(3), np.eye(3) * .001, log, .1)
    assert log.disp_used == 0 and log.disp_withheld == 1
    assert f.model_scale > 1.  # passive self-check remains active
    np.testing.assert_array_equal(f.P, before)


def test_uncalibrated_model_cannot_certify_an_outage_even_before_self_check():
    def predictor(_):
        return np.zeros(3), np.eye(3) * .01
    predictor.window, predictor.every = 2, 1
    e = start(predictor=predictor)
    rows = [e.add_imu(i / 200., np.zeros(3), [0., 0., 9.81])[0] for i in range(5)]
    assert not any(r["valid"] for r in rows)
    assert all(r["reasons"] & REASONS["COVARIANCE_UNRELIABLE"] for r in rows)


def test_lost_sensor_is_invalid_immediately_and_warmup_is_unverified():
    e = start()
    r = e.add_imu(0., np.zeros(3), [0., 0., 9.81], imu_state=WARMUP)[0]
    assert r["reasons"] & REASONS["IMU_UNVERIFIED"]
    r = e.add_imu(.005, np.full(3, np.nan), np.full(3, np.nan), imu_state=LOST)[0]
    assert not r["valid"] and r["reasons"] & REASONS["IMU_LOST"]


def test_rejected_finite_stale_sensors_are_not_silently_reused():
    t = np.arange(40) / 200.
    g = np.zeros((3, len(t), 3))
    a = np.broadcast_to([0., 0., 9.81], g.shape).copy()
    fg, fa, info = ImuMonitor(3).run_batch(t, g, a)
    assert np.all(info["state"] == LOST)
    assert np.isnan(fg).all() and np.isnan(fa).all()


def test_buffered_inputs_are_owned_not_aliased_to_callers():
    e = start()
    fix, gyro, acc = np.array([.1, 0., 0.]), np.zeros(3), np.array([0., 0., 9.81])
    e.add_camera(.1, fix, 1.)
    e.add_imu(0., gyro, acc)
    fix[:] = gyro[:] = acc[:] = 99.
    assert e.pending[0][1] == "fix" and e.pending[0][2][0][0] == .1   # (t, kind, (fix, health))
    np.testing.assert_array_equal(e.held[0], np.zeros(3))
    m = ImuMonitor(1)
    g, a = np.zeros((1, 3)), np.array([[0., 0., 9.81]])
    m.push(0., g, a)
    g[:] = a[:] = 99.
    np.testing.assert_array_equal(m._buf_g[0], np.zeros((1, 3)))


def test_invalid_camera_event_never_enters_initialization_or_pending_queue():
    e = StreamingEstimator()
    for fix, health in (([0., 0.], 1.), ([np.nan, 0., 0.], 1.), ([0., 0., 0.], 2.)):
        e.add_camera(.1, fix, health)
    assert not e.pending and e.log.cam_invalid == 3


@pytest.mark.parametrize("kw", [{"disp_cov_scale": 0.}, {"fix_std": np.nan}, {"q_max": np.inf},
                               {"model_max_scale": -1.}, {"trust": ([0., 0.], [.1, .2])},
                               {"trust": ([0., 1.], [.1, -.2])}, {"recover_after_jump_s": 31.}])
def test_invalid_filter_configuration_fails_at_boundary(kw):
    with pytest.raises(ValueError):
        FilterConfig(**kw)


def test_configs_are_not_shared_between_estimators():
    cfg = FilterConfig()
    a, b = start(cfg), start(cfg)
    a.cfg.fix_std = 10.
    assert b.cfg.fix_std == cfg.fix_std == .03


def test_invalid_geometry_is_rejected_instead_of_appearing_as_sensor_fault():
    with pytest.raises(ValueError, match="proper rotations"):
        ImuGeometry(np.repeat((2. * np.eye(3))[None], 3, 0), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        MonitorConfig(window_s=0.)


def test_metric_windows_do_not_bridge_missing_tracking_samples():
    gt = np.column_stack((np.arange(5), np.zeros((5, 2)))).astype(float)
    est = gt.copy()
    est[2] = np.nan
    assert np.isnan(metrics.rte(est, gt, 2))
    summary = metrics.summary(est, gt, 1.)
    assert summary["missing_samples"] == 1 and summary["finite_fraction"] == .8
    est[-1] = np.nan
    assert np.isnan(metrics.summary(est, gt, 1.)["final_err_m"])
    assert np.isnan(metrics.drift_pct(est, gt))
    empty = metrics.summary(np.empty((0, 3)), np.empty((0, 3)), 1.)
    assert empty["finite_fraction"] == 0. and np.isnan(empty["ate_m"])


def test_nees_rejects_indefinite_covariance_instead_of_negative_error_score():
    est, gt = np.ones((2, 3)), np.zeros((2, 3))
    cov = np.array([-np.eye(3), np.eye(3)])
    result = metrics.nees(est, gt, cov)
    assert np.isnan(result[0]) and result[1] == pytest.approx(3.)


def test_uncalibrated_degraded_vision_is_not_certified_as_accurate():
    e = start()
    e.add_camera(0., np.zeros(3), .3)
    r = e.add_imu(0., np.zeros(3), [0., 0., 9.81])[0]
    assert not r["valid"] and r["reasons"] & REASONS["COVARIANCE_UNRELIABLE"]


def test_rank_deficient_sensor_calibration_stays_unverified():
    t = np.arange(700) / 200.
    g = np.zeros((3, len(t), 3))
    a = np.broadcast_to([0., 0., 9.81], g.shape).copy()
    # Signal checks disabled to exercise the statistical calibration boundary itself.
    _, _, info = ImuMonitor(3, MonitorConfig(signal_checks=False)).run_batch(t, g, a)
    assert not info["calibrated"].any()


def test_fault_isolation_preserves_original_ensemble_pvalue():
    from pathfinder.synthetic import head_walk
    from pathfinder.faults import Fault, apply_imu_faults
    s = head_walk(9., n_imus=3, seed=20)
    bad = apply_imu_faults(s, [Fault("imu_bias", 6., 2., 1, {"gyro": [.1, 0., 0.], "acc": [1., 0., 0.]})])
    _, _, info = ImuMonitor(3).run_batch(s.t, bad.gyro, bad.acc)
    # After initial isolation, hysteresis tests only the retained sensors. The diagnostic must
    # preserve the failed initial test on detection windows, not the selected survivors' value.
    ix = info["detected"] & (info["window_t0"] >= 6.) & (info["window_t0"] < 8.)
    assert ix.any()
    assert (info["agreement_p"][ix] < MonitorConfig().p_false_alarm / 2).all()


def test_replay_rejects_misaligned_camera_and_quality_arrays():
    t = np.arange(3) / 200.
    g, a = np.zeros((3, 3)), np.tile([0., 0., 9.81], (3, 1))
    init = TruthInit(np.zeros(3), np.zeros(3), np.array([0., 0., 0., 1.]))
    with pytest.raises(ValueError, match="camera timestamps"):
        replay(t, g, a, FilterConfig(), init, cam={"t": [0.], "fix": np.zeros((2, 3)), "health": [1.]})
    with pytest.raises(ValueError, match="quality"):
        replay(t, g, a, FilterConfig(), init, imu_quality={"q_add_g": [0.], "q_add_a": [0.], "state": [0]})
