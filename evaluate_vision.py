"""Real-vision evaluation under the frozen protocol (docs/PROTOCOL.md, issue #2).

    python evaluate_vision.py --calibration calibration/vo_room1_v2.json [--seq room1]
                              [--scenarios all|slice] [--workers 8] [--no-model]

Writes one immutable run bundle ``results/runs/<utc>-vision-<seq>-<sha>/`` with ``manifest.json``,
``status.json``, ``protocol.json`` (frozen parameters, scenario definitions), ``rows.jsonl`` (one line
per scenario x system, including harness failures) and ``summary.json``. ``tools/vision_report.py``
renders the report and figures from that bundle only.

Inputs to every system: the real IMU and the real stereo images of the recording, each optionally
corrupted *before* use. The motion-capture reference is read by the scorer after each run, never by a
system. No system is started from the reference.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import cv2

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from pathfinder import data_tumvi_raw as D  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.image_faults import randomized_schedule  # noqa: E402
from pathfinder.provenance import run_bundle, sha256_file  # noqa: E402
from pathfinder.stereo_vo import VOConfig  # noqa: E402
from pathfinder.vision_eval import (IMU_FAULT_KINDS, Frozen, evaluate_scenario,  # noqa: E402
                                    imu_noise_from_config, system_configs)
from pathfinder.vision_frontend import _src_hash  # noqa: E402

PROTOCOL = "pathfinder.vision-protocol/3"
SYSTEMS = ("vo_only", "imu_only", "conventional", "pathfinder", "pathfinder_no_gate", "pathfinder_uncal_cov",
           "pathfinder_no_adaptive_q", "pathfinder_no_health", "pathfinder_model", "pathfinder_model_no_selfcheck",
           "pathfinder_model_biascorr", "conventional_gated", "pathfinder_no_reanchor")
MODEL_SYSTEMS = tuple(s for s in SYSTEMS if s.startswith("pathfinder_model"))
RANDOM_SEEDS = tuple(range(100, 112))


def _img(kind, start, dur, sev=1.0):
    return {"kind": kind, "start": float(start), "end": float(start + dur), "severity": float(sev)}


def scenarios(duration: float) -> list:
    """Fixed scenarios (declared onset/duration) plus seeded randomized ones (numpy seeds 100-111).
    Independence from training: the network's augmentation is drawn by ``torch.manual_seed(0)`` on TLIO
    headset IMU windows; these schedules corrupt TUM-VI images and IMU. The IMU fault *types* noise,
    spikes, saturation and dropout also occur in that augmentation (disclosed in docs/PROTOCOL.md);
    image faults and the camera/IMU combinations do not."""
    bias = {"kind": "bias", "start": 35.0, "end": duration + 1, "gyro": [0.02, -0.015, 0.01], "acc": [0.3, -0.2, 0.15]}
    fixed = [
        {"name": "clean"},
        {"name": "blackout_5s_at40", "image_faults": [_img("blackout", 40, 5)]},
        {"name": "blackout_5s_startup", "image_faults": [_img("blackout", 0, 5)]},
        {"name": "blackout_20s_at60", "image_faults": [_img("blackout", 60, 20)]},
        {"name": "dark_10s_at50", "image_faults": [_img("dark", 50, 10, 0.9)]},
        {"name": "blur_5s_at70", "image_faults": [_img("blur", 70, 5, 1.0)]},
        {"name": "occlusion_8s_at30", "image_faults": [_img("occlusion", 30, 8, 1.0)]},
        {"name": "low_texture_6s_at90", "image_faults": [_img("low_texture", 90, 6, 1.0)]},
        {"name": "drop_3s_at100", "image_faults": [_img("drop", 100, 3)]},
        {"name": "jitter_10s_at20", "image_faults": [_img("jitter", 20, 10, 1.0)]},
        {"name": "frozen_3s_at60", "image_faults": [_img("frozen", 60, 3)]},
        {"name": "stereo_desync_5s_at80", "image_faults": [_img("stereo_desync", 80, 5)]},
        {"name": "blackout_5s_at40+imu_bias", "image_faults": [_img("blackout", 40, 5)], "imu_faults": [bias]},
        {"name": "blackout_5s_at40+imu_saturation", "image_faults": [_img("blackout", 40, 5)],
         "imu_faults": [{"kind": "saturation", "start": 40.0, "end": 45.0, "gyro_max": 0.5, "acc_max": 10.5}]},
        {"name": "imu_dropout_0.5s_at50", "imu_faults": [{"kind": "dropout", "start": 50.0, "end": 50.5}]},
        {"name": "imu_noise_5s_at60", "imu_faults": [{"kind": "noise", "start": 60.0, "end": 65.0}]},
    ]
    rnd = []
    for seed in RANDOM_SEEDS:
        img = [f.as_dict() for f in randomized_schedule(duration, seed, n_faults=(1, 3))]
        rng = np.random.default_rng(10_000 + seed)
        imu = []
        if rng.random() < 0.3 and duration > 5:
            kind = str(rng.choice(IMU_FAULT_KINDS))
            start = float(rng.uniform(0, duration - 5))
            imu.append({"kind": kind, "start": start, "end": min(duration, start + float(rng.uniform(0.5, 10.0)))})
        rnd.append({"name": f"random_s{seed}", "image_faults": img, "imu_faults": imu, "seed": seed})
    return fixed + rnd


def _worker(args):
    scn, systems, fz_dict, seq, vo_cfg, model_path, data_root = args
    fz = Frozen(**fz_dict)
    s = D.load_streams(seq, Path(data_root) if data_root else D.ROOT)
    ref = D.load_reference(seq, Path(data_root) if data_root else D.ROOT)
    rig = RigCalibration.from_kalibr(s.calib_path)
    pred = None
    if model_path and any(x in MODEL_SYSTEMS for x in systems):
        import torch
        torch.set_num_threads(1)
        from pathfinder.predictor import Predictor
        pred = Predictor(model_path)
    from pathfinder.vision_frontend import run_frontend
    frontend_start = time.perf_counter()
    fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], schedule=scn.get("image_faults", []),
                      seed=scn.get("seed", 0), cfg=VOConfig(**vo_cfg),
                      index_sha=D.index_sha256(seq, Path(data_root) if data_root else D.ROOT),
                      time_shift=rig.time_shift + fz.camera_time_offset_s)
    fe["retrieval_wall_s"] = time.perf_counter() - frontend_start
    return evaluate_scenario(scn, systems, fz, s, ref, rig, pred, frontend=fe)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", default="room1")
    ap.add_argument("--data-root", default=None, help="directory that contains dataset-<seq>_512_16")
    ap.add_argument("--calibration", required=True, help="JSON from tools/calibrate_vo.py")
    ap.add_argument("--sync", default=None, help="sync.json from tools/check_sync.py (records the measured offset)")
    ap.add_argument("--model", default="models/dispnet_faults/best.pt")
    ap.add_argument("--model-sha256", default="611e97fdad4ac7735a3086179e059edb772f6d93d13a40c3b88b8d4bfec3eba3")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--allow-calibration-transfer", action="store_true", help="explicit cross-rig development transfer, not validated calibration")
    ap.add_argument("--scenarios", default="all", help="all | slice | comma-separated names")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cal = json.loads(Path(a.calibration).read_text())
    if cal.get("schema") != "pathfinder.vo-calibration/1":
        raise SystemExit("calibration schema mismatch")
    if cal["frontend_src_sha256"] != _src_hash():
        raise SystemExit("front-end source changed since calibration: recalibrate (new protocol version)")
    if cal.get("opencv") is not None and cal["opencv"] != cv2.__version__:
        raise SystemExit(f"OpenCV {cv2.__version__} != calibration fingerprint {cal['opencv']}")
    if cal.get("vo_covariance_validated") is True:
        raise SystemExit("calibration must not assert vo_covariance_validated")
    root = Path(a.data_root) if a.data_root else D.ROOT
    s = D.load_streams(a.seq, root)
    duration = float(s.imu_t[-1] - s.imu_t[0])
    noise = imu_noise_from_config(s.dir / "dso" / "imu_config.yaml")
    rig_check = RigCalibration.from_kalibr(s.calib_path)
    calibration_mismatch = (cal.get("calibration_sha256") != rig_check.meta.get("sha256")
        or cal.get("imu_config_sha256") != sha256_file(s.dir / "dso/imu_config.yaml")
        or cal.get("imu_noise") != noise)
    if calibration_mismatch and not a.allow_calibration_transfer:
        raise SystemExit("calibration rig/noise mismatch; explicit --allow-calibration-transfer required")
    offset = float(cal.get("camera_time_offset_s", 0.0))
    sync = None
    if a.sync:
        sync = json.loads(Path(a.sync).read_text())
        measurement = sync.get("camera_imu", {})
        if (measurement.get("status") == "unsupported" or measurement.get("offset_s") is None
                or not np.isfinite(measurement["offset_s"])):
            raise SystemExit("sync measurement has no supported finite camera/IMU offset")
        measured = float(sync["camera_imu"]["offset_s"])
        offset = measured if abs(measured) > 1e-3 else 0.0          # protocol: apply only if > 1 ms
    if offset != float(cal.get("camera_time_offset_s", 0.0)):
        raise SystemExit("sync offset differs from calibration; create a new calibration")
    model = None
    if not a.no_model:
        model = a.model
        got = sha256_file(model)
        if got is None:
            raise SystemExit(f"model checkpoint {model} missing: run tools/fetch_models.py (no silent retraining)")
        if got != a.model_sha256:
            raise SystemExit(f"model checkpoint hash {got} != expected {a.model_sha256}")
    fz = Frozen(protocol=PROTOCOL, imu_noise=noise, vo_cov_scale=float(cal["selected_vo_cov_scale"]),
                camera_time_offset_s=offset, model=model, model_sha256=a.model_sha256 if model else None)
    for name, (cfg, _cam, _mdl) in system_configs(fz).items():
        if cfg.vo_covariance_validated:
            raise SystemExit(f"{name} asserts vo_covariance_validated without independent evidence")
    vo_cfg = asdict(VOConfig(kf_every=int(cal["selected_kf_every"])))
    all_scn = scenarios(duration)
    if a.scenarios == "slice":
        chosen = [x for x in all_scn if x["name"] in ("clean", "blackout_5s_at40")]
    elif a.scenarios == "all":
        chosen = all_scn
    else:
        names = set(a.scenarios.split(","))
        unknown = names - {x["name"] for x in all_scn}
        if unknown:
            raise SystemExit("unknown scenarios: " + ", ".join(sorted(unknown)))
        chosen = [x for x in all_scn if x["name"] in names]
    systems = tuple(x for x in SYSTEMS if model or x not in MODEL_SYSTEMS)
    inputs = {"imu": s.dir / "mav0/imu0/data.csv", "cam0_index": s.dir / "mav0/cam0/data.csv",
              "cam1_index": s.dir / "mav0/cam1/data.csv", "camchain": s.calib_path,
              "imu_config": s.dir / "dso/imu_config.yaml", "reference_scoring_only": s.dir / "mav0/mocap0/data.csv",
              "vo_calibration": Path(a.calibration)}
    if model:
        inputs["model"] = Path(model)
    if a.sync:
        inputs["sync"] = Path(a.sync)
    kw = {"root": Path(a.out)} if a.out else {}
    with run_bundle(f"vision-{a.seq}", {**vars(a), "systems": systems}, inputs, protocol=PROTOCOL,
                    seeds=list(RANDOM_SEEDS), **kw) as run:
        run.write_json("protocol.json", {"protocol": PROTOCOL, "frozen": asdict(fz), "vo_config": vo_cfg,
                                         "scenarios": chosen, "systems": systems, "sequence": a.seq,
                                         "split_role": "development recording (previously exposed); see docs/PROTOCOL.md",
                                         "duration_s": duration, "sync_measurement": sync and sync.get("camera_imu"),
                                         "calibration_run": cal.get("run"),
                                         "frontend_src_sha256": cal["frontend_src_sha256"],
                                         "calibration_opencv": cal.get("opencv"),
                                         "runtime_opencv": cv2.__version__,
                                         "vo_covariance_validated": False,
                                         "data_root": str(root),
                                         "external_vio": "not an input; a separate baseline must not be fused (IMU would be counted twice)"})
        rows_path = run.file("rows.jsonl")
        t0 = time.time()
        jobs = [(scn, systems, asdict(fz), a.seq, vo_cfg, model, str(root)) for scn in chosen]
        n_rows = 0
        with open(rows_path, "w") as fh, ProcessPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(_worker, j): j[0]["name"] for j in jobs}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    rows = fut.result()
                except Exception as exc:             # a whole-scenario harness failure is kept, not dropped
                    import traceback
                    rows = [{"scenario": name, "system": sysname, "status": f"harness_error: {type(exc).__name__}: {exc}",
                             "metrics": {}, "diagnostics": {"traceback": traceback.format_exc()}} for sysname in systems]
                for r in rows:
                    fh.write(json.dumps(r, default=_default) + "\n")
                    n_rows += 1
                fh.flush()
                print(f"[{time.time() - t0:7.1f}s] {name}: " + ", ".join(
                    f"{r['system']}={_ate(r)}" for r in rows), flush=True)
        run.write_json("summary.json", {"rows": n_rows, "wall_s": time.time() - t0, "scenarios": len(chosen),
                                        "systems": systems})
        print("run:", run.path)


def _ate(r):
    v = r.get("metrics", {}).get("start_se3", {}).get("ate_all", {}).get("rmse")
    return "ERR" if v is None else f"{v:.3f}"


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
