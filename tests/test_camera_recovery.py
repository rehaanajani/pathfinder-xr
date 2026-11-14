"""Issue #1 A: a wrong camera must not talk its way back in, and a genuine correction must still land.

Uses the shipped configuration (pipeline.ours_cfg), durations longer and shorter than any hold-out,
dropped frames and repeated faults. The displacement source here is a labelled ORACLE (true
displacement + noise) so the tests isolate the filter's camera logic from the network."""

import numpy as np
import pytest

from pathfinder.faults import Fault
from pathfinder.filter import FilterConfig, run
from pathfinder.pipeline import ours_cfg, prepare
from pathfinder.synthetic import head_walk


def oracle(s, std=0.05, seed=0):
    rng = np.random.default_rng(seed)

    def p(win):
        k = p.k
        p.k += 20
        return s.p[k] - s.p[k - 200] + rng.normal(0, std, 3), np.eye(3) * std ** 2
    p.k, p.window, p.every = 200, 200, 20
    return p


def err(est, s, t):
    k = int(t * 200)
    return float(np.linalg.norm(est[k] - s.p[k]))


def drop_frames(cam, keep, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random(len(cam["idx"])) < keep
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in cam.items()}


@pytest.mark.parametrize("dur", [3, 10, 20])
@pytest.mark.parametrize("offset", [[3.0, 0, 0], [0.6, -0.4, 0]])
def test_wrong_camera_is_rejected_by_the_shipped_config(dur, offset):
    s = head_walk(15 + dur + 10, n_imus=1, seed=5)
    bad, cam = prepare(s, [Fault("camera_wrong_fix", 15, dur, params={"offset": offset, "step": 0.0})])
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, oracle(s), ours_cfg())
    # Held out during the fault: the remaining error is our own dead-reckoning drift (oracle noise),
    # a few cm per 10 s, never the camera's offset.
    assert err(est, s, 15 + dur - 0.5) < min(0.03 * dur + 0.05, 0.3 * np.linalg.norm(offset))
    assert err(est, s, 15 + dur + 3) < 0.15          # and the camera is trusted again once it is right


def test_wrong_camera_with_dropped_frames_and_repeats():
    s = head_walk(60, n_imus=1, seed=7)
    faults = [Fault("camera_wrong_fix", 12, 8, params={"offset": [2.0, 1.0, 0], "step": 0.0}),
              Fault("camera_wrong_fix", 30, 12, params={"offset": [-1.5, 0.5, 0], "step": 0.0})]
    bad, cam = prepare(s, faults)
    cam = drop_frames(cam, keep=0.5)
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, oracle(s), ours_cfg())
    for t in (19, 41, 50):
        assert err(est, s, t) < 0.2, t


def test_genuine_correction_after_drift_is_accepted_quickly():
    # No learned model and a biased IMU: during a 10 s blackout we drift metres. When the camera returns
    # it is right and we are wrong; we must re-anchor within a couple of seconds, not hold it out.
    s = head_walk(40, n_imus=1, seed=3)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 10), Fault("imu_bias", 15, 10, 0,
                                                                     {"gyro": [0, 0, 0.01], "acc": [0.15, 0, 0]})])
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, ours_cfg())
    assert err(est, s, 24.9) > 1.0                   # we really did drift
    assert err(est, s, 27.5) < 0.2                   # and the returning camera corrected us


def test_genuine_correction_with_dropped_frames():
    s = head_walk(40, n_imus=1, seed=4)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 10), Fault("imu_bias", 15, 10, 0,
                                                                     {"gyro": [0, 0, 0.01], "acc": [0.15, 0, 0]})])
    cam = drop_frames(cam, keep=0.4, seed=2)
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, ours_cfg())
    assert err(est, s, 28.5) < 0.25


def test_gradual_visual_drift_does_not_drag_us_far():
    # A wrong fix that creeps away slowly (no single jump) while the model knows our motion.
    s = head_walk(40, n_imus=1, seed=6)
    bad, cam = prepare(s, [Fault("camera_wrong_fix", 10, 20, params={"offset": [0.0, 0, 0], "step": 0.08})])
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, oracle(s), ours_cfg())
    naive, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig(use_health=False, use_gate=False))
    assert err(est, s, 29.5) < 0.5 * err(naive, s, 29.5)
