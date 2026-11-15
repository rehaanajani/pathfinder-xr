"""Shared helpers for the stress tests: build a case, run the offline filter and the streaming
tracker on identical inputs, and check the invariants that must hold on every sample.

A case is fully described by a small dict, so a failure prints a one-line repro.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from pathfinder.faults import IMU_KINDS, CAMERA_KINDS, Fault
from pathfinder.filter import run
from pathfinder.pipeline import fuse_imus, ours_cfg, prepare
from pathfinder.stream import StreamingTracker
from pathfinder.synthetic import head_walk

ALL_KINDS = sorted(IMU_KINDS | CAMERA_KINDS)


def make_case(seed, duration=20.0, rate=200.0, n_imus=3, faults=(), cam_keep=1.0):
    return dict(seed=seed, duration=duration, rate=rate, n_imus=n_imus,
                faults=[(f.kind, f.start, f.duration, f.imu, dict(f.params)) for f in faults],
                cam_keep=cam_keep)


def faults_of(case):
    return [Fault(k, s, d, i, dict(p)) for k, s, d, i, p in case["faults"]]


def repro(case):
    return f"case={case!r}"


def inputs(case, mutate=None):
    """Sequence, broken IMUs, fused IMU stream, per-sample confidence, camera fixes."""
    s = head_walk(case["duration"], rate=case["rate"], n_imus=case["n_imus"], seed=case["seed"])
    bad, cam = prepare(s, faults_of(case), case["seed"])
    if mutate is not None:
        bad = mutate(bad)
    fg, fa, info = fuse_imus(bad)
    conf = np.repeat(info["confidence"], 20)[: len(s.t)]
    keep = case.get("cam_keep", 1.0)
    if keep < 1.0:
        rng = np.random.default_rng(case["seed"] + 5)
        m = rng.random(len(cam["idx"])) < keep
        cam = {k: v[m] for k, v in cam.items()}
    return s, bad, fg, fa, conf, cam, info


def run_pair(case, cfg=None, predictor_factory=None, check_every=1, mutate=None, exact_dt=True):
    """Offline ``run`` and ``StreamingTracker`` on the same inputs. Returns a result dict."""
    cfg = cfg or ours_cfg()
    s, bad, fg, fa, conf, cam, info = inputs(case, mutate)
    pr_off = predictor_factory() if predictor_factory else None
    pr_on = predictor_factory() if predictor_factory else None
    est, log = run(s, fg, fa, conf, cam, pr_off, replace(cfg))
    trk = StreamingTracker(replace(cfg), s.p[0], s.q[0], s.v[0], rate=(1.0 / s.dt if exact_dt else case["rate"]), predictor=pr_on)
    cam_at = {int(i): j for j, i in enumerate(cam["idx"])}
    outs, worst_sym, worst_psd = [], 0.0, 0.0
    t0 = time.perf_counter()
    lat = []
    for k in range(len(s.t)):
        if k in cam_at:
            j = cam_at[k]
            trk.push_camera_fix(cam["fix"][j], cam["health"][j])
        a = time.perf_counter()
        outs.append(trk.push_imu(fg[k], fa[k], conf[k]))
        lat.append(time.perf_counter() - a)
        if k % check_every == 0:
            P = trk.f.P
            if np.isfinite(P).all():
                scale = max(np.abs(P).max(), 1e-12)
                worst_sym = max(worst_sym, np.abs(P - P.T).max() / scale)
                ev = np.linalg.eigvalsh(0.5 * (P + P.T))
                worst_psd = max(worst_psd, -ev.min() / scale)
            else:
                worst_sym = worst_psd = np.inf
    wall = time.perf_counter() - t0
    live = np.array([o["position"] for o in outs])
    return dict(case=case, seq=s, est=est, log=log, outs=outs, live=live, cam=cam, fg=fg, fa=fa, conf=conf,
                wall=wall, rt_factor=(len(s.t) / case["rate"]) / max(wall, 1e-9), lat=np.array(lat),
                sym=worst_sym, psd=worst_psd,
                max_diff=float(np.nanmax(np.abs(live - est))) if live.size else 0.0)


def check_invariants(r, tol=1e-9, psd_tol=1e-9):
    """Returns a list of violated-invariant strings (empty = all good)."""
    bad = []
    outs, s = r["outs"], r["seq"]
    pos = np.array([o["position"] for o in outs])
    cov = np.array([o["position_cov"] for o in outs])
    std = np.array([o["position_std_m"] for o in outs])
    if not np.isfinite(pos).all():
        bad.append("non-finite position")
    if not np.isfinite(cov).all() or not np.isfinite(std).all():
        bad.append("non-finite covariance/std")
    if r["sym"] > 1e-7:
        bad.append(f"covariance not symmetric (rel {r['sym']:.2e})")
    if r["psd"] > psd_tol:
        bad.append(f"covariance not PSD (rel min eig -{r['psd']:.2e})")
    if not r["max_diff"] <= tol:
        bad.append(f"stream != run (max |dp| {r['max_diff']:.3e})")
    lc = np.array([np.asarray(c) for c in r["log"].p_cov])
    if lc.shape == cov.shape and np.isfinite(lc).all() and np.isfinite(cov).all():
        d = np.abs(lc - cov).max()
        if d > 1e-9 * max(1.0, np.abs(lc).max()):
            bad.append(f"stream cov != run cov (max {d:.3e})")
    if r["rt_factor"] < 1.0:
        bad.append(f"slower than real time ({r['rt_factor']:.2f}x)")
    for o in outs:
        if o["valid"] and not (np.isfinite(o["position"]).all() and np.isfinite(o["position_std_m"])):
            bad.append("valid=True with non-finite output")
            break
    return bad
