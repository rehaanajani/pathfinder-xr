"""Outage speed limit: rescues a broken IMU during a camera outage, leaves healthy tracking alone."""

from dataclasses import replace

import numpy as np

from pathfinder.faults import Fault
from pathfinder.filter import FilterConfig, run
from pathfinder.pipeline import prepare
from pathfinder.synthetic import head_walk

ON = FilterConfig(speed_limit=True)


def end_err(est, s, t):
    k = int(t * 200)
    return float(np.linalg.norm(est[k] - s.p[k]))


def test_speed_limit_caps_runaway_velocity_from_saturated_imu():
    s = head_walk(40, n_imus=1, seed=2)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 10),
                           Fault("imu_saturation", 15, 10, None, {"gyro_max": 0.3, "acc_max": 9.0})])
    off, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig())
    on, log = run(s, bad.gyro[0], bad.acc[0], None, cam, None, ON)
    assert log.v_limited > 0
    assert end_err(on, s, 24.9) < 0.5 * end_err(off, s, 24.9), (end_err(on, s, 24.9), end_err(off, s, 24.9))


def test_speed_limit_is_inactive_while_the_camera_works():
    s = head_walk(30, n_imus=1, seed=3)
    bad, cam = prepare(s, [])
    off, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig())
    on, log = run(s, bad.gyro[0], bad.acc[0], None, cam, None, ON)
    assert log.v_limited == 0 and log.v_prior == 0
    assert np.allclose(on, off)


def test_speed_limit_does_not_hurt_a_healthy_imu_in_an_outage():
    # The synthetic walk peaks at 2.1-2.5 m/s (faster than real headset data), so the limit is set
    # above its true speed: then a correct IMU must be left alone.
    s = head_walk(40, n_imus=1, seed=4)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 10)])
    off, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig())
    on, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, replace(ON, v_max=3.0))
    assert end_err(on, s, 24.9) <= 1.1 * end_err(off, s, 24.9) + 0.02
