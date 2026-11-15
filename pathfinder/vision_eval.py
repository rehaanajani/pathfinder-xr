"""Real-vision evaluation: systems, scenarios and one scored run (issue #2 C/D).

Everything a system receives is measured or corrupted *before* it is consumed:

* the real single IMU of the recording (optionally corrupted by a declared, seeded IMU fault), and
* visual-odometry events from ``vision_frontend`` on the real (optionally corrupted) stereo images.

Reference poses are read only by ``scoring.score`` after the run. No system is started from the
reference: Path Finder and the conventional filter use the reference-free gravity start (odometry
frame = start pose, velocity unknown), and alignment for scoring is declared in ``scoring``.

Systems (``SYSTEMS``)
    vo_only        relative poses chained keyframe to keyframe; across a tracking break the chain holds
                   its last pose (zero motion) and restarts from the next keyframe. No IMU.
    imu_only       Path Finder's estimator with no camera: inertial dead reckoning from the gravity start.
    conventional   loosely coupled ESKF with the same VO relative updates: front-end covariance as
                   claimed (scale 1), no chi-square gate, no health weighting, no adaptive process noise.
    pathfinder     Path Finder (``pipeline.ours_cfg``) with relative VO updates, development-calibrated VO
                   covariance scale and the 6-DoF gate. No learned model.
    pathfinder_model   + the fault-aware learned displacement network (trained on Meta TLIO headset data,
                   so on TUM-VI it is a cross-device evaluation), conservative fallback defaults.
    ablations      pathfinder_no_gate, pathfinder_uncal_cov, pathfinder_no_adaptive_q, pathfinder_no_health,
                   pathfinder_model_no_selfcheck (model_max_scale=None), pathfinder_model_biascorr.
An established external VIO (OpenVINS / Basalt / ORB-SLAM3) is not run here; see docs/PROTOCOL.md.
"""

import time
import tracemalloc
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import scoring
from .estimator import ReferenceFreeInit, replay, reason_names
from .filter import FilterConfig

# Sensor noise from the recording's own calibration file (TUM-VI dso/imu_config.yaml: Allan-variance
# continuous noise densities with the dataset authors' inflation, white noise x2 and bias walk x10).
# Units match FilterConfig: rad/s/sqrt(Hz), m/s^2/sqrt(Hz), rad/s^2/sqrt(Hz), m/s^3/sqrt(Hz).
def imu_noise_from_config(path) -> dict:
    import yaml
    c = yaml.safe_load(open(path))
    return {"sigma_g": float(c["gyroscope_noise_density"]), "sigma_a": float(c["accelerometer_noise_density"]),
            "sigma_bg": float(c["gyroscope_random_walk"]), "sigma_ba": float(c["accelerometer_random_walk"])}


@dataclass
class Frozen:
    """Selected parameters. Changing any value requires a new protocol version (docs/PROTOCOL.md)."""
    protocol: str = "pathfinder.vision-protocol/2"
    imu_noise: dict = field(default_factory=dict)        # from the dataset's imu_config.yaml
    vo_cov_scale: float = 1.0                            # selected on the calibration segment
    camera_time_offset_s: float = 0.0                    # applied only if measured |offset| > 1 ms
    model: str | None = None                             # checkpoint path
    model_sha256: str | None = None


def base_cfg(fz: Frozen) -> FilterConfig:
    from .pipeline import ours_cfg
    return replace(ours_cfg(), **fz.imu_noise, vo_cov_scale=fz.vo_cov_scale)


def system_configs(fz: Frozen) -> dict:
    """name -> (FilterConfig, uses_camera, uses_model)."""
    ours = base_cfg(fz)
    conv = replace(FilterConfig(**fz.imu_noise), use_gate=False, use_health=False, vo_cov_scale=1.0)
    return {
        "imu_only": (replace(ours), False, False),
        "conventional": (conv, True, False),
        "pathfinder": (ours, True, False),
        "pathfinder_no_gate": (replace(ours, use_gate=False), True, False),
        "pathfinder_uncal_cov": (replace(ours, vo_cov_scale=1.0), True, False),
        "pathfinder_no_adaptive_q": (replace(ours, adaptive_q=False), True, False),
        "pathfinder_no_health": (replace(ours, use_health=False), True, False),
        "pathfinder_model": (ours, True, True),
        "pathfinder_model_no_selfcheck": (replace(ours, model_max_scale=None), True, True),
        "pathfinder_model_biascorr": (replace(ours, model_input_bias_corrected=True), True, True),
    }


# ---------------------------------------------------------------------------------------------- IMU faults
IMU_FAULT_KINDS = ("bias", "saturation", "noise", "dropout", "spikes")


def corrupt_imu(t, gyro, acc, faults, seed=0):
    """Declared IMU faults on the real IMU stream. faults: [{"kind", "start", "end", ...}] in seconds
    since the first IMU sample. Physically plausible: bias step, saturation (clipping), dropout (NaN
    samples = missing); stress tests: 10x noise burst, spikes."""
    g, a = gyro.copy(), acc.copy()
    rel = t - t[0]
    rng = np.random.default_rng(seed)
    for f in faults:
        m = (rel >= f["start"]) & (rel < f["end"])
        k = f["kind"]
        if k == "bias":
            g[m] += np.asarray(f.get("gyro", [0.02, -0.015, 0.01]))
            a[m] += np.asarray(f.get("acc", [0.3, -0.2, 0.15]))
        elif k == "saturation":
            g[m] = np.clip(g[m], -f.get("gyro_max", 1.0), f.get("gyro_max", 1.0))
            a[m] = np.clip(a[m], -f.get("acc_max", 11.0), f.get("acc_max", 11.0))
        elif k == "noise":
            g[m] += rng.normal(0, f.get("gyro_std", 0.05), (m.sum(), 3))
            a[m] += rng.normal(0, f.get("acc_std", 0.5), (m.sum(), 3))
        elif k == "dropout":
            g[m] = np.nan
            a[m] = np.nan
        elif k == "spikes":
            idx = np.where(m)[0]
            pick = idx[rng.random(len(idx)) < f.get("rate", 0.02)]
            g[pick] += rng.normal(0, 2.0, (len(pick), 3))
            a[pick] += rng.normal(0, 20.0, (len(pick), 3))
        else:
            raise ValueError(f"unknown IMU fault {k!r}")
    return g, a


# ---------------------------------------------------------------------------------------------- VO only
def vo_only_trajectory(t, events):
    """Chain relative poses; output a pose at each IMU time (held between frames). Frame = start pose.

    Returns dict like an estimator output: p, q (body->odom), valid (False before the first keyframe
    and while the chain is broken), P_pos NaN (no covariance claimed)."""
    poses = {}              # kf id -> (R, p) in the odometry frame
    cur = None              # current (R, p)
    measured = False        # was the latest frame linked to its predecessor by a relative solve?
    linked = False
    times, Ps, Qs, ok = [], [], [], []
    for e in events:
        if e["kind"] == "rel":
            if e["kf"] in poses:
                Rk, pk = poses[e["kf"]]
                cur = (Rk * Rot.from_matrix(e["dR"]), pk + Rk.apply(e["dp"]))
                linked = True
        else:
            if cur is None:
                cur = (Rot.identity(), np.zeros(3))      # first keyframe defines the odometry frame
                linked = True
            # A keyframe that no relative solve reached keeps the last pose (zero-motion hold).
            measured, linked = linked, False
            poses = {k: v for k, v in poses.items() if k >= e["kf"] - 3}
            poses[e["kf"]] = cur
        if cur is not None:
            times.append(e["t"])
            Ps.append(cur[1])
            Qs.append(cur[0].as_quat())
            ok.append(measured or linked)
    times = np.array(times)
    N = len(t)
    p = np.full((N, 3), np.nan)
    q = np.full((N, 4), np.nan)
    valid = np.zeros(N, bool)
    if len(times):
        j = np.searchsorted(times, t, side="right") - 1
        have = j >= 0
        p[have] = np.array(Ps)[j[have]]
        q[have] = np.array(Qs)[j[have]]
        valid[have] = np.array(ok)[j[have]]
    return {"t": t, "p": p, "q": q, "valid": valid, "P_pos": np.full((N, 3, 3), np.nan)}


# Declared room1 split (data/exposure.json): calibration segment and within-recording hold-out.
SEGMENTS = {"calibration_0_60": (0.0, 60.0), "heldout_60_end": (60.0, 1e9)}


# ---------------------------------------------------------------------------------------------- one run
def run_system(name, cfg, uses_camera, uses_model, imu_t, gyro, acc, events, predictor=None):
    """Runs one system; returns (outputs, diagnostics). Timing covers the estimator only; the front-end
    cost is measured per frame by ``vision_frontend`` and combined in the report."""
    tracemalloc.start()
    c0 = time.perf_counter()
    if name == "vo_only":
        out = vo_only_trajectory(imu_t, events)
        diag = {}
    else:
        pos, log, out = replay(imu_t, gyro, acc, cfg, ReferenceFreeInit(),
                               vision=events if uses_camera else None,
                               predictor=predictor if uses_model else None)
        reasons = {}
        for r in np.unique(out["reasons"]):
            for nm in reason_names(int(r)):
                reasons[nm] = reasons.get(nm, 0) + int(np.sum(out["reasons"] == r))
        nis = np.array([x[1] for x in log.vo_nis]) if log.vo_nis else np.zeros(0)
        diag = {"vo_used": log.vo_used, "vo_rejected": log.vo_rejected, "vo_no_anchor": log.vo_no_anchor,
                "vo_keyframes": log.vo_keyframes, "cam_skipped": log.cam_skipped, "cam_late": log.cam_late,
                "cam_invalid": log.cam_invalid, "disp_used": log.disp_used, "disp_rejected": log.disp_rejected,
                "disp_skipped": log.disp_skipped, "disp_withheld": log.disp_withheld,
                "vo_nis_mean": float(nis.mean()) if len(nis) else None,
                "vo_nis_median": float(np.median(nis)) if len(nis) else None,
                "reason_counts": reasons, "events": [list(map(str, e)) for e in log.events[:50]]}
    wall = time.perf_counter() - c0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diag["estimator_wall_s"] = wall
    diag["estimator_us_per_imu_sample"] = 1e6 * wall / max(len(imu_t), 1)
    diag["estimator_peak_traced_mb"] = peak / 2 ** 20
    return out, diag


def evaluate_scenario(scn, systems, fz, streams, ref, rig, predictor=None, frontend=None):
    """scn: {"name", "image_faults": [...], "imu_faults": [...], "seed"} -> list of result rows."""
    from .vision_frontend import run_frontend
    from . import data_tumvi_raw as D
    t0 = time.perf_counter()
    fe = frontend or run_frontend(streams.frame_t, streams.frame_paths, rig, streams.imu_t[0],
                                  schedule=scn.get("image_faults", []), seed=scn.get("seed", 0),
                                  index_sha=D.index_sha256(streams.name),
                                  time_shift=rig.time_shift + fz.camera_time_offset_s)
    fe_wall = time.perf_counter() - t0
    g, a = corrupt_imu(streams.imu_t, streams.gyro, streams.acc, scn.get("imu_faults", []), scn.get("seed", 0))
    cfgs = system_configs(fz)
    frames = fe["frames"]
    fe_ms = frames[:, 2:5].sum(1) if len(frames) else np.zeros(0)
    outage_list = [{"start": f["start"], "end": f["end"], "kind": "image:" + f["kind"]}
                   for f in fe["schedule"]] + [{"start": f["start"], "end": f["end"], "kind": "imu:" + f["kind"]}
                                               for f in scn.get("imu_faults", [])]
    rows = []
    for sysname in systems:
        if sysname == "vo_only":
            cfg, cam, mdl = None, True, False
        else:
            cfg, cam, mdl = cfgs[sysname]
        try:
            out, diag = run_system(sysname, cfg, cam, mdl, streams.imu_t, g, a, fe["events"], predictor)
            res, series = scoring.score(out, ref, outage_list, segments=SEGMENTS)
            status = "ok" if "failure" not in res else "no_valid_estimate"
        except Exception as exc:                 # retained as a harness failure, never dropped
            import traceback
            res, series, diag, status = {}, None, {"traceback": traceback.format_exc()}, \
                f"harness_error: {type(exc).__name__}: {exc}"
        rows.append({"scenario": scn["name"], "system": sysname, "status": status, "metrics": res,
                     "diagnostics": diag, "series": _thin(series),
                     "frontend": {"key": fe["key"], "stats": fe["stats"], "cached": fe.get("cached", False),
                                  "wall_s": fe_wall, "frames": int(len(frames)),
                                  "delivered": int(frames[:, 1].sum()) if len(frames) else 0,
                                  "ms_per_frame": _pct(fe_ms), "vo_ms_per_frame": _pct(frames[:, 4]) if len(frames) else {},
                                  "load_ms_per_frame": _pct(frames[:, 2]) if len(frames) else {}},
                     "scenario_def": {k: v for k, v in scn.items() if k != "name"}})
    return rows


def _pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "mean": float(x.mean())}


def _thin(series, every=20):
    """10 Hz copy of the per-sample error series for plots (full series are not stored)."""
    if series is None:
        return None
    return {k: (np.asarray(v)[::every].tolist()) for k, v in series.items()}
