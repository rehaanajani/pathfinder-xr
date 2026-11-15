"""Regression tests for the issue #1 findings on multi-IMU fault handling."""

import numpy as np

from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.parity import CONF_ISOLATED, CONF_OK, CONF_UNRESOLVED, ParityFDI, fuse
from pathfinder.synthetic import head_walk

WARM = 10          # warm-up windows (1 s)


def test_confidence_is_one_only_when_imus_agree():
    s = head_walk(10, n_imus=3)
    w, D, info = ParityFDI(3).run(s.gyro, s.acc, s.rate)
    assert np.mean(info["confidence"][WARM:] == CONF_OK) > 0.98


def test_one_fault_cleanly_removed_keeps_high_confidence():
    s = head_walk(10, n_imus=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 1, {"gyro": [0.05, 0, 0], "acc": [0.5, 0, 0]})])
    w, D, info = ParityFDI(3).run(bad.gyro, bad.acc, s.rate)
    c = info["confidence"][30:]
    assert np.mean(c == CONF_ISOLATED) > 0.9 and w[1, 30:].mean() < 0.1


def test_two_simultaneous_faults_lower_confidence():
    # Old behaviour: confidence = max weight = 1.0 whenever any IMU was kept.
    s = head_walk(10, n_imus=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 0, {"gyro": [0.05, 0, 0], "acc": [0.5, 0, 0]}),
                               Fault("imu_bias", 2, 1e9, 1, {"gyro": [0, -0.08, 0], "acc": [0, 0.8, 0]})])
    w, D, info = ParityFDI(3).run(bad.gyro, bad.acc, s.rate)
    assert np.mean(info["confidence"][30:] <= CONF_UNRESOLVED) > 0.9


def test_two_imus_disagreeing_is_unresolved():
    s = head_walk(10, n_imus=2)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 0, {"gyro": [0.1, 0, 0], "acc": [1.0, 0, 0]})])
    w, D, info = ParityFDI(2).run(bad.gyro, bad.acc, s.rate)
    assert np.mean(info["confidence"][30:] == CONF_UNRESOLVED) > 0.9


def test_fault_present_at_startup_is_not_learned_as_healthy():
    # Old behaviour: the first second set the "nominal" offset, so a bias there was invisible forever.
    s = head_walk(10, n_imus=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 0, 1e9, 2, {"gyro": [0.03, 0, 0], "acc": [0.3, 0, 0]})])
    w, D, info = ParityFDI(3).run(bad.gyro, bad.acc, s.rate)
    assert w[2, WARM:].mean() < 0.1
    assert w[0, WARM:].mean() > 0.9 and w[1, WARM:].mean() > 0.9


def _rigid_body_imus(s, r):
    """Three IMUs at offsets r (K,3) on the same rigid head: exact lever-arm specific force."""
    g = s.gyro[0]
    wdot = np.gradient(g, s.dt, axis=0)
    rng = np.random.default_rng(1)
    gy, ac = [], []
    for k in range(len(r)):
        f = s.acc[0] + np.cross(g, np.cross(g, r[k])) + np.cross(wdot, r[k])
        gy.append(g + rng.normal(0, 0.003, g.shape))
        ac.append(f + rng.normal(0, 0.03, f.shape))
    return np.stack(gy), np.stack(ac)


def test_lever_arms_are_compensated_when_mounting_is_known():
    s = head_walk(20, n_imus=1, imu_noise=False)
    r = np.array([[0.09, 0.0, 0.02], [-0.09, 0.0, 0.02], [0.0, 0.07, -0.04]])
    gy, ac = _rigid_body_imus(s, r)
    _, _, without = ParityFDI(3).run(gy, ac, s.rate)
    _, _, with_ = ParityFDI(3).run(gy, ac, s.rate, lever_arms=r)
    fa_without = without["detected"][WARM:].mean()
    fa_with = with_["detected"][WARM:].mean()
    # Measured envelope (3 seeds: 1.0-2.6 %), not the nominal 0.01 %: the noise level is learned
    # from 1 s of warm-up and residual gyro noise times the lever arm is motion-dependent.
    assert fa_with < 0.03, fa_with
    assert fa_without > 0.5, fa_without


def test_non_finite_imu_is_dropped_and_output_stays_finite():
    s = head_walk(10, n_imus=3)
    g, a = s.gyro.copy(), s.acc.copy()
    g[1, 400:800] = np.nan
    w, D, info = ParityFDI(3).run(g, a, s.rate)
    fg, fa = fuse(g, a, w, 20)
    assert np.isfinite(fg).all() and np.isfinite(fa).all()
    assert (w[1, 20:40] == 0).all()


def test_no_usable_imu_gives_zero_confidence_not_equal_averaging():
    s = head_walk(10, n_imus=3)
    g, a = s.gyro.copy(), s.acc.copy()
    g[:, 400:800] = np.nan
    w, D, info = ParityFDI(3).run(g, a, s.rate)
    fg, fa = fuse(g, a, w, 20)
    assert (info["confidence"][20:40] == 0).all()
    assert np.isfinite(fg).all()
