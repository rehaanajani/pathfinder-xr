"""Runs one sequence through one fault scenario with every method, and scores them.

Methods
  imu_only        : average the IMUs and integrate blindly (no correction at all)
  naive_fusion    : Kalman filter with camera + averaged IMUs, every frame trusted equally
                    (no health check, no gate, no IMU monitor, no learned model)
  tlio_style      : learned displacement (TLIO-style training) + filter, IMU only, no camera
  ours_no_model   : ours without the learned model (ablation)
  ours_tlio_model : ours with the TLIO-style-trained network swapped in (isolates the training)
  ours            : IMU health monitor + camera health weighting + chi-square gate + camera
                    supervision + learned displacement from the fault-aware network

Benchmark protocol: every filter method here starts from the reference pose at the first sample
(``init="truth"``, labelled ``truth_init_diagnostic`` in the outputs) so methods are compared on
equal footing. That is a benchmark convenience, not the runtime path; ``init="reference_free"``
runs the same methods with the reference-free initialisation of ``estimator.py``. Visual fixes in
``prepare`` are *simulated* from the reference trajectory (there is no visual front end yet).
"""

import numpy as np

from . import camera, metrics
from .faults import Fault, apply_imu_faults, camera_schedule
from .filter import FilterConfig, run
from .imu_health import ImuMonitor, MonitorConfig
from .integrator import strapdown

from .camera import load_trust

TRUST = load_trust()      # measured health -> fix-error table; None until calibrated


def ours_cfg():
    """The configuration every 'ours' method uses (the shipped configuration)."""
    return FilterConfig(trust=TRUST, adaptive_q=True, bias_q_gain=10.0)


def monitor_cfg():
    return MonitorConfig()


ALL = ("imu_only", "naive_fusion", "tlio_style", "ours_no_model", "ours_tlio_model", "ours")


def prepare(seq, faults: list[Fault], seed: int = 0):
    """Benchmark fixture: faulty IMU streams and SIMULATED visual fixes (uses the reference)."""
    bad = apply_imu_faults(seq, faults, seed)
    sched = camera_schedule(camera.frame_times(seq.t), faults)
    cam = camera.visual_fixes(seq.t, seq.p, sched, seed)
    return bad, cam


def _pack(est, seq, log=None, **extra):
    ok = np.all(np.isfinite(est), 1)
    r = {"p": est, **(metrics.summary(est[ok], seq.p[ok], seq.rate) if ok.any() else {}), **extra}
    if log is not None:
        r.update(cam_used=log.cam_used, cam_rejected=log.cam_rejected, cam_skipped=log.cam_skipped,
                 cam_recovered=log.cam_recovered, disp_used=log.disp_used,
                 disp_rejected=log.disp_rejected, disp_skipped=log.disp_skipped, p_std=np.array(log.p_std),
                 disp_std=np.array(log.disp_std))
        out = getattr(log, "outputs", None)
        if out is not None:
            r.update(P_pos=out["P_pos"], valid=out["valid"], track_state=out["track_state"],
                     camera_state=out["camera_state"], reasons=out["reasons"])
    return r


def fuse_imus(bad, cfg: MonitorConfig | None = None, geometry=None):
    """Causal IMU health monitor for any number of IMUs (see ``imu_health``).

    3+ IMUs: parity detection + isolation. 2 IMUs: parity detects; per-sensor signal checks
    (stale, stuck, pinned, noisy, non-finite) attribute what they can. 1 IMU: signal checks only.
    Returns fused gyro, acc and an info dict whose ``quality`` entry is the per-sample contract the
    filter consumes (``q_add_g``, ``q_add_a``, ``state``). There is no scalar "confidence": see the
    module docstring of ``imu_health`` for what relative agreement can and cannot establish.
    """
    mon = ImuMonitor(bad.n_imus, cfg or monitor_cfg(), geometry)
    fg, fa, info = mon.run_batch(bad.t, bad.gyro, bad.acc)
    info["quality"] = {"q_add_g": info["q_add_g"], "q_add_a": info["q_add_a"], "state": info["state"]}
    info["method"] = "monitor"
    return fg, fa, info


def run_methods(seq, faults, seed=0, ours_model=None, tlio_model=None, methods=ALL, init="truth"):
    bad, cam = prepare(seq, faults, seed)
    res = {"_cam": cam, "_init": init}
    avg_g, avg_a = bad.gyro.mean(0), bad.acc.mean(0)
    fg = fa = info = None
    if any(m.startswith("ours") for m in methods):
        fg, fa, info = fuse_imus(bad)
    if "imu_only" in methods:
        est = strapdown(avg_g, avg_a, seq.dt, seq.p[0], seq.v[0], seq.q[0])
        res["imu_only"] = _pack(est, seq)
    if "naive_fusion" in methods:
        est, log = run(seq, avg_g, avg_a, None, cam, None, FilterConfig(use_health=False, use_gate=False),
                       init=init)
        res["naive_fusion"] = _pack(est, seq, log)
    if "tlio_style" in methods and tlio_model is not None:
        est, log = run(seq, avg_g, avg_a, None, None, tlio_model, FilterConfig(), init=init)
        res["tlio_style"] = _pack(est, seq, log)
    q = info["quality"] if info is not None else None
    if "ours_no_model" in methods:
        est, log = run(seq, fg, fa, None, cam, None, ours_cfg(), imu_quality=q, init=init)
        res["ours_no_model"] = _pack(est, seq, log)
    if "ours_tlio_model" in methods and tlio_model is not None:
        est, log = run(seq, fg, fa, None, cam, tlio_model, ours_cfg(), imu_quality=q, init=init)
        res["ours_tlio_model"] = _pack(est, seq, log)
    if "ours" in methods and ours_model is not None:
        est, log = run(seq, fg, fa, None, cam, ours_model, ours_cfg(), imu_quality=q, init=init)
        res["ours"] = _pack(est, seq, log, imu_weights=info["weights"], imu_reasons=info["reasons"])
    return res
