import numpy as np

from pathfinder import metrics
from pathfinder.camera import degrade, image_health, textured_frame
from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.imu_vote import vote
from pathfinder.integrator import strapdown
from pathfinder.synthetic import head_walk


def test_clean_imu_integrates_back_to_truth():
    s = head_walk(20, n_imus=1, imu_noise=False)
    est = strapdown(s.gyro[0], s.acc[0], s.dt, s.p[0], s.v[0], s.q[0])
    assert metrics.ate(est, s.p) < 0.01


def test_bias_makes_strapdown_drift_metres():
    s = head_walk(20, n_imus=1, imu_noise=False)
    bad = apply_imu_faults(s, [Fault("imu_bias", 0, 1e9, 0)])
    est = strapdown(bad.gyro[0], bad.acc[0], s.dt, s.p[0], s.v[0], s.q[0])
    assert metrics.ate(est, s.p) > 2.0


def test_metrics_alignment_removes_yaw_and_offset():
    s = head_walk(10, imu_noise=False)
    th = 0.7
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = s.p @ Rz.T + np.array([3.0, -2.0, 0.5])
    assert metrics.ate(moved, s.p) < 1e-6
    assert metrics.ate(moved, s.p, align=False) > 1.0


def test_faults_never_touch_ground_truth():
    s = head_walk(10, n_imus=2)
    bad = apply_imu_faults(s, [Fault("imu_noise", 0, 1e9, None)])
    assert np.array_equal(bad.p, s.p) and np.array_equal(bad.q, s.q)
    assert not np.array_equal(bad.gyro, s.gyro)


def test_camera_health_drops_in_dark_and_blur():
    rng = np.random.default_rng(0)
    clean = np.mean([image_health(degrade(textured_frame(rng), 1, 0, rng))["score"] for _ in range(5)])
    dark = np.mean([image_health(degrade(textured_frame(rng), 0.08, 0, rng))["score"] for _ in range(5)])
    blur = np.mean([image_health(degrade(textured_frame(rng), 1, 30, rng))["score"] for _ in range(5)])
    assert clean > 0.8 and dark < 0.1 and blur < 0.1


def test_vote_isolates_one_bad_imu_out_of_three():
    s = head_walk(10, n_imus=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 1, {"gyro": [0.3, 0, 0], "acc": [2, 0, 0]})])
    fg, fa, info = vote(bad.gyro, bad.acc)
    after = info["weights"][:, 30:]
    assert after[1].mean() < 0.1 and after[0].mean() > 0.9 and after[2].mean() > 0.9
    assert np.abs(fg[600:] - s.gyro[0, 600:]).mean() < 0.02


def test_vote_drops_stuck_imu_even_with_two():
    s = head_walk(10, n_imus=2)
    bad = apply_imu_faults(s, [Fault("imu_stuck", 3, 4, 0)])
    fg, fa, info = vote(bad.gyro, bad.acc)
    w = info["weights"][:, 35:65]
    assert w[0].mean() < 0.1 and w[1].mean() > 0.9


def test_parity_isolates_small_bias_that_voting_misses():
    from pathfinder.parity import ParityFDI
    s = head_walk(10, n_imus=3)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 1, {"gyro": [0.02, 0, 0], "acc": [0.3, 0, 0]})])
    w, D, info = ParityFDI(3).run(bad.gyro, bad.acc, s.rate)
    after = w[:, 30:]
    assert after[1].mean() < 0.1 and after[0].mean() > 0.9 and after[2].mean() > 0.9


def test_parity_detects_but_cannot_isolate_with_two_imus():
    from pathfinder.parity import ParityFDI
    s = head_walk(10, n_imus=2)
    bad = apply_imu_faults(s, [Fault("imu_bias", 2, 1e9, 0, {"gyro": [0.1, 0, 0], "acc": [1.0, 0, 0]})])
    w, D, info = ParityFDI(2).run(bad.gyro, bad.acc, s.rate)
    assert info["detected"][30:].mean() > 0.9
    assert np.allclose(w[:, 30:], 0.5)
