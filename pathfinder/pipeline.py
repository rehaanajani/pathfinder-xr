"""Runs one sequence through one fault scenario with every method, and scores them.

Methods
  imu_only        : average the IMUs and integrate blindly (no correction at all)
  naive_fusion    : Kalman filter with camera + averaged IMUs, every frame trusted equally
                    (no health check, no gate, no IMU voting, no learned model)
  tlio_style      : learned displacement (TLIO-style training) + filter, IMU only, no camera
  ours_no_model   : ours without the learned model (ablation)
  ours_tlio_model : ours with the TLIO-style-trained network swapped in (isolates the training)
  ours            : IMU voting + camera health weighting + chi-square gate + learned
                    displacement from the fault-aware network
"""

import numpy as np

from . import camera, metrics
from .faults import Fault, apply_imu_faults, camera_schedule
from .filter import FilterConfig, run
from .imu_vote import vote
from .parity import ParityFDI, fuse
from .integrator import strapdown

from .camera import load_trust

TRUST = load_trust()      # measured health -> fix-error table; None until calibrated


OURS_OVERRIDES: dict = {}          # set by evaluate.py --set, for validation sweeps


def ours_cfg():
    from dataclasses import replace
    # Outage speed limit + weak zero-velocity prior (2 m/s): chosen on the validation split for runs
    # without the learned model; it switches itself off when the model runs (results/diagnostics).
    # Saturation guard: chosen on the validation split (saturation 2.00 -> 0.93 m without the model,
    # no change elsewhere).
    return replace(FilterConfig(trust=TRUST, adaptive_q=True, bias_q_gain=10.0, speed_limit=True,
                                outage_v_std=2.0, saturation_guard=True), **OURS_OVERRIDES)


ALL = ("imu_only", "naive_fusion", "tlio_style", "ours_no_model", "ours_tlio_model", "ours")


def prepare(seq, faults: list[Fault], seed: int = 0):
    bad = apply_imu_faults(seq, faults, seed)
    sched = camera_schedule(camera.frame_times(seq.t), faults)
    cam = camera.visual_fixes(seq.t, seq.p, sched, seed)
    return bad, cam


def _pack(est, seq, log=None, **extra):
    r = {"p": est, **metrics.summary(est, seq.p, seq.rate), **extra}
    if log is not None:
        r.update(cam_used=log.cam_used, cam_rejected=log.cam_rejected, cam_skipped=log.cam_skipped,
                 cam_recovered=log.cam_recovered, disp_used=log.disp_used,
                 disp_rejected=log.disp_rejected, p_std=np.array(log.p_std),
                 disp_std=np.array(log.disp_std))
    return r


def fuse_imus(bad):
    """3+ IMUs: parity-space GLRT detection and isolation. 2 IMUs: parity can only detect,
    so per-IMU signal checks (stuck, saturated, spiky, noisy) decide who is wrong."""
    if bad.n_imus >= 3:
        w, D, pinfo = ParityFDI(bad.n_imus).run(bad.gyro, bad.acc, bad.rate,
                                                lever_arms=bad.meta.get("lever_arms"))
        fg, fa = fuse(bad.gyro, bad.acc, w, 20)
        # Confidence = can the fused stream be trusted (agreement, or one fault cleanly removed),
        # not "was any IMU kept". See parity.CONF_* for the levels.
        return fg, fa, {"weights": w, "reasons": pinfo["isolated"], "confidence": pinfo["confidence"],
                        "state": pinfo["state"], "parity_D": D, "method": "parity"}
    fg, fa, info = vote(bad.gyro, bad.acc)
    return fg, fa, {**info, "method": "vote"}


def run_methods(seq, faults, seed=0, ours_model=None, tlio_model=None, methods=ALL):
    bad, cam = prepare(seq, faults, seed)
    res = {"_cam": cam}
    avg_g, avg_a = bad.gyro.mean(0), bad.acc.mean(0)
    fg = fa = info = conf = None
    if any(m.startswith("ours") for m in methods):
        fg, fa, info = fuse_imus(bad)
        conf = np.repeat(info["confidence"], 20)[: len(seq.t)]
    if "imu_only" in methods:
        est = strapdown(avg_g, avg_a, seq.dt, seq.p[0], seq.v[0], seq.q[0])
        res["imu_only"] = _pack(est, seq)
    if "naive_fusion" in methods:
        est, log = run(seq, avg_g, avg_a, None, cam, None, FilterConfig(use_health=False, use_gate=False))
        res["naive_fusion"] = _pack(est, seq, log)
    if "tlio_style" in methods and tlio_model is not None:
        est, log = run(seq, avg_g, avg_a, None, None, tlio_model, FilterConfig())
        res["tlio_style"] = _pack(est, seq, log)
    if "ours_no_model" in methods:
        est, log = run(seq, fg, fa, conf, cam, None, ours_cfg())
        res["ours_no_model"] = _pack(est, seq, log)
    if "ours_tlio_model" in methods and tlio_model is not None:
        est, log = run(seq, fg, fa, conf, cam, tlio_model, ours_cfg())
        res["ours_tlio_model"] = _pack(est, seq, log)
    if "ours" in methods and ours_model is not None:
        est, log = run(seq, fg, fa, conf, cam, ours_model, ours_cfg())
        res["ours"] = _pack(est, seq, log, imu_weights=info["weights"], imu_reasons=info["reasons"])
    return res
