"""Saturation guard: clipped IMU readings are detected from the data and handled."""

from dataclasses import replace

import numpy as np

from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.filter import FilterConfig, run, saturation_mask
from pathfinder.pipeline import prepare
from pathfinder.synthetic import head_walk


def test_detector_flags_clipping_and_nothing_on_clean_data():
    s = head_walk(30, n_imus=1, seed=1)
    assert saturation_mask(s.gyro[0], s.acc[0]).mean() == 0.0
    bad = apply_imu_faults(s, [Fault("imu_saturation", 10, 10, None, {"gyro_max": 0.3, "acc_max": 9.0})])
    m = saturation_mask(bad.gyro[0], bad.acc[0])
    t = np.arange(len(m)) / 200
    assert m[(t >= 10) & (t < 20)].mean() > 0.4 and m[(t < 10) | (t >= 20.5)].mean() == 0.0


def test_guard_keeps_a_model_from_running_away_on_saturated_input():
    # A learned model that faithfully integrates the clipped stream (here: displacement from the
    # bad IMU itself) would follow it; the guard distrusts it and keeps the speed limit on.
    s = head_walk(40, n_imus=1, seed=2)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 10),
                           Fault("imu_saturation", 15, 10, None, {"gyro_max": 0.3, "acc_max": 9.0})])

    def wrong_model(win):
        return np.array([1.5, 0.0, 0.0]), np.eye(3) * 0.05 ** 2     # confidently wrong: 1.5 m/s east
    wrong_model.window, wrong_model.every = 200, 20
    base = FilterConfig(speed_limit=True, outage_v_std=2.0)
    off, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, wrong_model, base)
    on, log = run(s, bad.gyro[0], bad.acc[0], None, cam, wrong_model, replace(base, saturation_guard=True))
    k = int(24.9 * 200)
    e_off, e_on = np.linalg.norm(off[k] - s.p[k]), np.linalg.norm(on[k] - s.p[k])
    assert log.sat_samples > 0 and e_on < 0.7 * e_off, (e_on, e_off)


def test_dropped_packets_are_not_mistaken_for_saturation():
    s = head_walk(30, n_imus=1, seed=3)
    bad = apply_imu_faults(s, [Fault("imu_dropout", 10, 10, None, {})])
    assert saturation_mask(bad.gyro[0], bad.acc[0]).mean() == 0.0
