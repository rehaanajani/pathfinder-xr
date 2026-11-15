"""Causal whole-pipeline latency, throughput and memory on the machine it runs on (issue #2 B/D, #3 E).

    python tools/profile_pipeline.py --seq room1 --seconds 60 [--kf-every K] [--model models/dispnet_faults/best.pt]

Replays the recording in timestamp order on ONE thread (BLAS/OMP/torch threads = 1), exactly as a
single-threaded live system would see it: each stereo frame is decoded and passed through the
camera front end (image load, VO), each IMU sample through the estimator (including any learned-model
inference it triggers). Every call is timed separately, so the report gives per-stage and per-input
latency distributions (p50/p95/p99/max), missed deadlines against the input period (frame 50 ms,
IMU 5 ms), the first-call warm-up, throughput relative to real time and process memory.

This is *laptop* evidence for the named CPU only. It says nothing about a headset/NPU, thermal limits
or a multi-threaded production scheduler (issue #3 real-device phase).
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.estimator import ReferenceFreeInit, StreamingEstimator  # noqa: E402
from pathfinder.provenance import run_bundle  # noqa: E402
from pathfinder.stereo_vo import StereoVO, VOConfig  # noqa: E402
from pathfinder.vision_eval import Frozen, imu_noise_from_config, system_configs  # noqa: E402
from pathfinder.vision_frontend import load_pair  # noqa: E402


def pct(x, budget=None):
    x = np.asarray(x, float)
    if not len(x):
        return {}
    out = {"n": int(len(x)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
           "p99": float(np.percentile(x, 99)), "max": float(x.max()), "mean": float(x.mean())}
    if budget is not None:
        out["missed_deadline"] = int(np.sum(x > budget))
        out["budget_ms"] = budget
    return out


def cpu_name():
    if platform.system() == "Windows":
        try:
            import subprocess
            return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            pass
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", default="room1")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--kf-every", type=int, default=1)
    ap.add_argument("--vo-cov-scale", type=float, default=1.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import psutil
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    s = D.load_streams(a.seq)
    rig = RigCalibration.from_kalibr(s.calib_path)
    fz = Frozen(imu_noise=imu_noise_from_config(s.dir / "dso/imu_config.yaml"), vo_cov_scale=a.vo_cov_scale)
    cfg, _, _ = system_configs(fz)["pathfinder"]
    pred = None
    if a.model:
        import torch
        torch.set_num_threads(1)
        from pathfinder.predictor import Predictor
        pred = Predictor(a.model)
    t_start = time.perf_counter()
    vo = StereoVO(rig, VOConfig(kf_every=a.kf_every))
    est = StreamingEstimator(cfg, ReferenceFreeInit(), pred, expect_camera=True, camera_mode="relative")
    startup_ms = 1e3 * (time.perf_counter() - t_start)
    hi = s.imu_t[0] + a.seconds
    imu_idx = np.where(s.imu_t <= hi)[0]
    fr_idx = np.where(s.frame_t <= hi)[0]
    load_ms, vo_ms, imu_us, imu_ms_list, model_calls = [], [], [], [], 0
    fi = 0
    rss_peak = rss0
    wall0 = time.perf_counter()
    for i in imu_idx:
        while fi < len(fr_idx) and s.frame_t[fr_idx[fi]] <= s.imu_t[i]:
            k = fr_idx[fi]
            c0 = time.perf_counter()
            left, right = load_pair(*s.frame_paths[k])
            c1 = time.perf_counter()
            for e in vo.step(s.frame_t[k] + rig.time_shift, left, right):
                if e["kind"] == "kf":
                    est.add_camera_keyframe(e["t"], e["kf"])
                else:
                    est.add_camera_relative(e["t"], e["kf"], e["dp"], e["dR"], e["cov"], e["health"])
            c2 = time.perf_counter()
            load_ms.append(1e3 * (c1 - c0))
            vo_ms.append(1e3 * (c2 - c1))
            fi += 1
            if fi % 50 == 0:
                rss_peak = max(rss_peak, proc.memory_info().rss)
        n_disp = est.log.disp_used + est.log.disp_rejected + est.log.disp_skipped + est.log.disp_withheld
        c0 = time.perf_counter()
        est.add_imu(s.imu_t[i], s.gyro[i], s.acc[i])
        dt = time.perf_counter() - c0
        imu_us.append(1e6 * dt)
        if est.log.disp_used + est.log.disp_rejected + est.log.disp_skipped + est.log.disp_withheld != n_disp:
            model_calls += 1
            imu_ms_list.append(1e3 * dt)
    wall = time.perf_counter() - wall0
    rss_peak = max(rss_peak, proc.memory_info().rss)
    imu_ms = np.array(imu_us) / 1e3
    # The estimator replays its 3 s initialisation buffer in one call; report it separately.
    init_k = int(np.argmax(imu_ms)) if len(imu_ms) else 0
    steady = np.delete(imu_ms, init_k)
    res = {
        "hardware": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "platform": platform.platform(),
                     "python": platform.python_version(), "threads": "single thread (BLAS/OMP/torch = 1)",
                     "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)},
        "label": "laptop CPU replay from disk; not headset/NPU evidence",
        "config": vars(a), "seconds_replayed": float(s.imu_t[imu_idx[-1]] - s.imu_t[0]),
        "frames": int(len(load_ms)), "imu_samples": int(len(imu_us)),
        "startup_ms": startup_ms,
        "frame_decode_ms": pct(load_ms),
        "frame_vo_ms": pct(vo_ms, 50.0),
        "frame_total_ms": pct(np.add(load_ms, vo_ms), 50.0),
        "imu_estimator_ms_steady": pct(steady, 5.0),
        "imu_estimator_init_replay_ms": float(imu_ms[init_k]) if len(imu_ms) else None,
        "imu_samples_with_model_inference_ms": pct(imu_ms_list, 5.0) if imu_ms_list else None,
        "model_calls": model_calls,
        "wall_s": wall, "real_time_factor": wall / max(float(s.imu_t[imu_idx[-1]] - s.imu_t[0]), 1e-9),
        "rss_mb_start": rss0 / 2 ** 20, "rss_mb_peak_sampled": rss_peak / 2 ** 20,
        "first_frame_vo_ms": vo_ms[0] if vo_ms else None,
        "first_10_frames_vo_ms": vo_ms[:10],
    }
    kw = {"root": Path(a.out)} if a.out else {}
    with run_bundle(f"profile-{a.seq}", vars(a), {"camchain": s.calib_path, **({"model": a.model} if a.model else {})},
                    protocol="profiling") as run:
        run.write_json("profile.json", res)
        print(json.dumps({k: res[k] for k in ("hardware", "frame_total_ms", "frame_vo_ms", "imu_estimator_ms_steady",
                                              "imu_samples_with_model_inference_ms", "real_time_factor",
                                              "rss_mb_peak_sampled")}, indent=1))
        print("run:", run.path)


if __name__ == "__main__":
    main()
