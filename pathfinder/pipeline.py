"""Runs one sequence through one fault scenario with every method, and scores them.

Methods
  imu_only      : average the IMUs and integrate blindly (what happens with no correction)
  naive_fusion  : Kalman filter with camera + averaged IMUs, every frame trusted equally
                  (no health check, no gate, no IMU voting). Typical simple VIO fusion.
  ours          : IMU voting + camera health weighting + chi-square gate
                  (+ learned displacement when a model is given)
"""

import numpy as np

from . import camera, metrics
from .faults import Fault, apply_imu_faults, camera_schedule
from .filter import FilterConfig, run
from .imu_vote import vote
from .integrator import strapdown


def prepare(seq, faults: list[Fault], seed: int = 0):
    bad = apply_imu_faults(seq, faults, seed)
    sched = camera_schedule(camera.frame_times(seq.t), faults)
    cam = camera.visual_fixes(seq.t, seq.p, sched, seed)
    return bad, cam


def run_methods(seq, faults, seed=0, learned=None, methods=("imu_only", "naive_fusion", "ours")):
    bad, cam = prepare(seq, faults, seed)
    results = {}
    avg_g, avg_a = bad.gyro.mean(0), bad.acc.mean(0)
    if "imu_only" in methods:
        est = strapdown(avg_g, avg_a, seq.dt, seq.p[0], seq.v[0], seq.q[0])
        results["imu_only"] = {"p": est, **metrics.summary(est, seq.p, seq.rate)}
    if "naive_fusion" in methods:
        cfg = FilterConfig(use_health=False, use_gate=False)
        est, log = run(seq, avg_g, avg_a, None, cam, None, cfg)
        results["naive_fusion"] = {"p": est, **metrics.summary(est, seq.p, seq.rate),
                                   "cam_used": log.cam_used}
    if "ours" in methods:
        fg, fa, info = vote(bad.gyro, bad.acc)
        conf = np.repeat(info["confidence"], 20)[: len(seq.t)]
        disp = learned(bad, fg, fa) if learned is not None else None
        est, log = run(seq, fg, fa, conf, cam, disp, FilterConfig())
        results["ours"] = {"p": est, **metrics.summary(est, seq.p, seq.rate),
                           "cam_used": log.cam_used, "cam_rejected": log.cam_rejected,
                           "cam_skipped": log.cam_skipped, "disp_used": log.disp_used,
                           "disp_rejected": log.disp_rejected,
                           "imu_weights": info["weights"], "imu_reasons": info["reasons"],
                           "p_std": np.array(log.p_std)}
    results["_cam"] = cam
    return results
