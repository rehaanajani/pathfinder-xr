"""How honest is the tracker about its own error?  Runs the streaming tracker (the code that ships:
position_std_m with the outage floor, valid/reason flags) on TLIO validation clips with faults,
and compares reported uncertainty to the true error.

Definitions (fixed before running, see docs/STRESS_REPORT.md):
  e          true 3-D position error (m)
  std        reported position_std_m (sqrt of trace of the 3x3 position covariance, with the floor)
  overconf   e > 3 * std  and  e > 0.10 m      ("reported std far below the real error")
  NEES       e^T P^-1 e with P = position_cov (3 dof; mean should be ~3, 95% bound 7.81)
  coverage   share of samples with e <= 1.61 * std  (95% for an isotropic Gaussian: chi2_3,0.95 / 3 = 2.60)
  valid-lead t(first e > 1 m) - t(first valid = False after the fault starts); >= 0 means the flag
             went false before (or when) the error passed 1 m.
"""
from __future__ import annotations

import numpy as np

from pathfinder import data_tlio as D
from pathfinder.faults import Fault
from pathfinder.pipeline import fuse_imus, ours_cfg, prepare
from pathfinder.stream import StreamingTracker
from pathfinder.synthetic import head_walk

SCENARIOS = {
    "blackout_10s": [Fault("camera_blackout", 20, 10)],
    "blackout_30s": [Fault("camera_blackout", 20, 30)],
    "blackout_10s+imu_bias": [Fault("camera_blackout", 20, 10), Fault("imu_bias", 15, 1e9, 0)],
    "wrong_camera_10s": [Fault("camera_wrong_fix", 20, 10)],
    "camera_dark_10s": [Fault("camera_dark", 20, 10)],
}


def val_sequences(n=12, clip=60.0, k_imus=3):
    out = []
    for sid in D.split("val")[:n]:
        s = D.load(sid)
        start = int(min(30, max(0, s.t[-1] - clip - 1)) * 200)
        out.append(D.redundant_imus(s.slice(start, start + int(clip * 200) + 1), k_imus, seed=len(out)))
    return out


def stream_run(seq, faults, seed, predictor=None, cfg=None, mutate=None):
    bad, cam = prepare(seq, faults, seed)
    if mutate:
        bad = mutate(bad)
    fg, fa, info = fuse_imus(bad)
    conf = np.repeat(info["confidence"], 20)[: len(seq.t)]
    trk = StreamingTracker(cfg or ours_cfg(), seq.p[0], seq.q[0], seq.v[0], rate=1 / seq.dt, predictor=predictor)
    cam_at = {int(i): j for j, i in enumerate(cam["idx"])}
    outs = []
    for k in range(len(seq.t)):
        if k in cam_at:
            trk.push_camera_fix(cam["fix"][cam_at[k]], cam["health"][cam_at[k]])
        outs.append(trk.push_imu(fg[k], fa[k], conf[k]))
    return outs


def stats(outs, truth, fault_start_s, rate=200.0, window=None):
    """Per-run honesty numbers over the samples from the fault start to the end (or ``window`` seconds)."""
    pos = np.array([o["position"] for o in outs])
    std = np.array([o["position_std_m"] for o in outs])
    valid = np.array([o["valid"] for o in outs])
    cov = np.array([o["position_cov"] for o in outs])
    e = np.linalg.norm(pos - truth, axis=1)
    k0 = int(fault_start_s * rate)
    k1 = len(e) if window is None else min(len(e), k0 + int(window * rate))
    sl = slice(k0, k1)
    d = (pos - truth)[sl]
    try:
        nees = np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov[sl] + 1e-12 * np.eye(3)), d)
    except np.linalg.LinAlgError:
        nees = np.full(len(d), np.nan)
    over = (e[sl] > 3 * std[sl]) & (e[sl] > 0.10)
    over_valid = over & valid[sl]
    big_valid = (e[sl] > 1.0) & valid[sl]
    first_big = np.flatnonzero(e[k0:] > 1.0)
    first_inv = np.flatnonzero(~valid[k0:])
    lead = np.nan
    if len(first_big):
        t_big = first_big[0] / rate
        t_inv = first_inv[0] / rate if len(first_inv) else np.inf
        lead = t_big - t_inv                       # >= 0: flag dropped no later than the error passed 1 m
    return {"n": int(len(d)), "overconf_frac": float(over.mean()), "overconf_valid_frac": float(over_valid.mean()),
            "valid_wrong_frac": float(big_valid.mean()), "cover95": float((e[sl] <= 1.61 * std[sl]).mean()),
            "nees_mean": float(np.nanmean(nees)), "nees_in95": float(np.nanmean(nees <= 7.81)),
            "max_err": float(e[sl].max()), "reached_1m": bool(len(first_big)), "lead_s": float(lead),
            "valid_late": bool(len(first_big) and lead < 0)}
