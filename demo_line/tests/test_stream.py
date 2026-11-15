"""The streaming tracker reproduces the offline filter sample for sample, and its flags make sense."""

from dataclasses import replace

import numpy as np

from pathfinder.faults import Fault
from pathfinder.filter import FilterConfig, run
from pathfinder.pipeline import ours_cfg, prepare
from pathfinder.stream import StreamingTracker
from pathfinder.synthetic import head_walk
from test_vo_update import vo_from_truth


def _oracle(s):
    def p(win):
        k = p.k
        p.k += 20
        return s.p[k] - s.p[k - 200] + 0.02, np.eye(3) * 0.05 ** 2
    p.window, p.every = 200, 20
    return p


def _stream(s, gyro, acc, cam, vo, predictor, cfg):
    trk = StreamingTracker(cfg, s.p[0], s.q[0], s.v[0], predictor=predictor)
    cam_at = {int(i): j for j, i in enumerate(cam["idx"])}
    vo_at = {int(i): j for j, i in enumerate(vo["frame_idx"])}
    outs = []
    for k in range(len(s.t)):
        if k in cam_at:
            trk.push_camera_fix(cam["fix"][cam_at[k]], cam["health"][cam_at[k]])
        if k in vo_at:
            j = vo_at[k]
            trk.push_vo(vo["t"][j] if j > 0 else None, vo["R"][j], vo["cov_t"][j], vo["cov_r"][j])
        outs.append(trk.push_imu(gyro[k], acc[k]))
    return outs


def test_streaming_matches_offline_filter_exactly():
    s = head_walk(30, n_imus=1, seed=4)
    bad, cam = prepare(s, [Fault("camera_blackout", 10, 6),
                           Fault("imu_saturation", 12, 4, None, {"gyro_max": 0.4, "acc_max": 10.5})])
    vo = vo_from_truth(s, every=10)
    cfg = replace(ours_cfg(), trust=None)
    p1 = _oracle(s); p1.k = 200
    est, _ = run(s, bad.gyro[0], bad.acc[0], None, cam, p1, cfg, vo=vo)
    p2 = _oracle(s); p2.k = 200
    outs = _stream(s, bad.gyro[0], bad.acc[0], cam, vo, p2, cfg)
    live = np.array([o["position"] for o in outs])
    assert np.allclose(live, est, atol=1e-9), np.abs(live - est).max()


def test_flags_follow_openxr_semantics():
    s = head_walk(30, n_imus=1, seed=5)
    bad, cam = prepare(s, [Fault("camera_blackout", 10, 8)])
    vo = {"frame_idx": np.array([], int), "t": np.zeros((0, 3)), "R": np.zeros((0, 3, 3)),
          "cov_t": np.zeros((0, 3, 3)), "cov_r": np.zeros((0, 3, 3))}
    outs = _stream(s, bad.gyro[0], bad.acc[0], cam, vo, None, FilterConfig())
    at = lambda t: outs[int(t * 200)]
    assert at(5)["tracked"] and at(5)["reason"] == "ok"
    assert not at(15)["tracked"] and at(15)["reason"] in ("visual_outage", "too_uncertain")
    assert at(15)["position_std_m"] > at(5)["position_std_m"]          # uncertainty grows in the outage
    assert at(25)["tracked"] and at(25)["reason"] == "ok"              # and recovers
