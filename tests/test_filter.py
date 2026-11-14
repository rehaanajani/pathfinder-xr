import numpy as np

from pathfinder import metrics
from pathfinder.faults import Fault
from pathfinder.filter import FilterConfig, run
from pathfinder.pipeline import prepare
from pathfinder.synthetic import head_walk


def test_filter_with_clean_camera_tracks_truth():
    s = head_walk(30, n_imus=1, seed=3)
    bad, cam = prepare(s, [])
    est, log = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig())
    assert metrics.ate(est, s.p) < 0.06
    assert log.cam_used > 250


def test_perfect_displacement_holds_through_blackout():
    s = head_walk(40, n_imus=1, seed=4)
    bad, cam = prepare(s, [Fault("camera_blackout", 15, 20)])


    def predictor(win):
        k = predictor.k
        predictor.k += 20
        d = s.p[k] - s.p[k - 200]
        return d, np.eye(3) * 0.05 ** 2

    predictor.k = 200
    predictor.window, predictor.every = 200, 20
    with_model, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, predictor, FilterConfig())
    without, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig())
    end = 35 * 200
    assert np.linalg.norm(with_model[end] - s.p[end]) < 0.5 * np.linalg.norm(without[end] - s.p[end])


def test_gate_plus_model_ignores_a_camera_tracking_the_wrong_thing():
    s = head_walk(40, n_imus=1, seed=5)
    bad, cam = prepare(s, [Fault("camera_wrong_fix", 15, 10, params={"offset": [3.0, 0, 0], "step": 0.0})])

    def predictor(win):
        k = predictor.k
        predictor.k += 20
        return s.p[k] - s.p[k - 200], np.eye(3) * 0.05 ** 2

    predictor.k, predictor.window, predictor.every = 200, 200, 20
    ours, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, predictor, FilterConfig())
    naive, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, None, FilterConfig(use_health=False, use_gate=False))
    mid = 22 * 200
    assert np.linalg.norm(ours[mid] - s.p[mid]) < 0.2
    assert np.linalg.norm(naive[mid] - s.p[mid]) > 2.0
