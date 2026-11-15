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


def test_long_outage_tightening_holds_a_biased_imu_close():
    # 40 s blind with a biased IMU: tightening the stand-still prior after 3 s keeps us far closer.
    s = head_walk(70, n_imus=1, seed=6)
    s.p[:] = s.p[0]; s.v[:] = 0                                    # a person standing in place
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 40),
                           Fault("imu_bias", 15, 40, 0, {"gyro": [0, 0, 0.002], "acc": [0.08, 0, 0]})])
    base = FilterConfig(speed_limit=True, outage_v_std=2.0)
    loose, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, base)
    tight, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, replace(base, outage_tighten_s=3.0))
    assert end_err(tight, s, 54.9) < 0.5 * end_err(loose, s, 54.9)     # measured 1.09 vs 3.31 m


def test_reported_uncertainty_is_honest_in_long_outages():
    from pathfinder.stream import StreamingTracker
    s = head_walk(30, n_imus=1, seed=7)
    bad, cam = prepare(s, [Fault("camera_blackout", 10, 15)])
    trk = StreamingTracker(replace(ON, outage_std_rate=0.15), s.p[0], s.q[0], s.v[0])
    at = {int(i): j for j, i in enumerate(cam["idx"])}
    outs = []
    for k in range(len(s.t)):
        if k in at:
            trk.push_camera_fix(cam["fix"][at[k]], cam["health"][at[k]])
        outs.append(trk.push_imu(bad.gyro[0][k], bad.acc[0][k]))
    o = outs[int(22 * 200)]                                        # 12 s blind
    assert o["position_std_m"] >= 0.15 * 11 * 3 ** 0.5 and not o["valid"]
