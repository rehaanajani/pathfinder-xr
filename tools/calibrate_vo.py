"""Select the visual-odometry covariance scale on a declared CALIBRATION segment (issue #2 E).

    python tools/calibrate_vo.py --seq room1 --segment 0 60 [--write calibration/vo_<seq>_vN.json]

Two declared selections, both on the calibration segment only:
1. For each keyframe interval ``kf_every`` in KF_GRID, the covariance scale is chosen reference-free:
   Path Finder (no learned model) runs with the frozen IMU noise for each candidate scale; the scale
   whose median and 95th-percentile NIS best match chi-square(6), in log distance, is kept.
2. ``kf_every`` is then chosen by the smallest calibration-segment ATE (start-SE(3) alignment) of that
   vision-only trajectory. This step uses the motion-capture reference of the calibration segment, which is
   allowed for calibration data and is never repeated on evaluation data.
Two diagnostics are reported alongside it:
  * rotation-only consistency of each VO increment against the gyro integrated over the same interval
    (raw gyro is bias-contaminated: this is a diagnostic, never a truth or covariance certificate);
  * position NEES of the fused trajectory against mocap on the same calibration segment.
The output is a versioned JSON (schema ``pathfinder.vo-calibration/1``) with input hashes. A
different recording, a different front-end version or a different IMU noise model needs a new
calibration; the loader checks the front-end source hash.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder import data_tumvi_raw as D, scoring  # noqa: E402
from pathfinder.camera_model import RigCalibration  # noqa: E402
from pathfinder.provenance import run_bundle, sha256_file  # noqa: E402
from pathfinder.vision_eval import Frozen, imu_noise_from_config, run_system, system_configs  # noqa: E402
from pathfinder.stereo_vo import VOConfig  # noqa: E402
from pathfinder.vision_frontend import _src_hash, run_frontend  # noqa: E402

GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
KF_GRID = (1, 2, 4)
SCHEMA = "pathfinder.vo-calibration/1"


def gyro_rotation_check(s, events, t_lo, t_hi):
    """Per-increment rotation NEES (3 dof) of the VO against the integrated gyro."""
    kf_t = {e["kf"]: e["t"] for e in events if e["kind"] == "kf"}
    out = []
    for e in events:
        if e["kind"] != "rel" or e["kf"] not in kf_t or not (t_lo <= e["t"] <= t_hi):
            continue
        ta, tb = kf_t[e["kf"]], e["t"]
        if not s.imu_t[0] <= ta < tb <= s.imu_t[-1]:
            continue
        knots = np.r_[ta, s.imu_t[(s.imu_t > ta) & (s.imu_t < tb)], tb]
        rates = np.column_stack([np.interp(knots, s.imu_t, s.gyro[:, i]) for i in range(3)])
        rotation = Rot.identity()
        for dt, rate in zip(np.diff(knots), .5 * (rates[1:] + rates[:-1])):
            rotation = rotation * Rot.from_rotvec(dt * rate)
        r = (rotation.inv() * Rot.from_matrix(e["dR"])).as_rotvec()
        C = e["cov"][3:, 3:]
        out.append((float(r @ np.linalg.solve(C, r)), float(np.linalg.norm(r))))
    if not out:
        return {"n": 0, "label": "raw-gyro diagnostic, not reference truth"}
    a = np.array(out)
    return {"n": len(a), "label": "raw-gyro diagnostic, not reference truth", "nees_rot_mean_at_scale1": float(a[:, 0].mean()), "nees_rot_median_at_scale1": float(np.median(a[:, 0])),
            "rot_err_rms_deg": float(np.rad2deg(np.sqrt(np.mean(a[:, 1] ** 2)))),
            "expected_mean": 3.0}


def scale_score(row):
    """Distribution-shape diagnostic, not a covariance-validity certificate."""
    med, tail = row.get("vo_nis_median"), row.get("vo_nis_p95")
    if med is None or tail is None or med <= 0 or tail <= 0:
        return np.inf
    return abs(np.log(med / chi2.ppf(.5, 6))) + abs(np.log(tail / chi2.ppf(.95, 6)))


def select_keyframe(per_kf):
    """Keyframe interval from vision-only calibration error.

    Fused ATE is recorded and is not the selection key: it depends on the
    covariance scale and would mix the two declared choices.
    """
    def key(row):
        ate = row.get("vo_only_ate_rmse_m")
        rot = row.get("vo_only_rot_rmse_deg")
        return (ate if ate is not None else np.inf, rot if rot is not None else np.inf)
    return min(per_kf, key=key)


def _one_kf(args):
    """Front end at one keyframe interval, then every covariance scale (worker process)."""
    seq, kf_every, segment, noise, offset, data_root = args
    root = Path(data_root) if data_root else D.ROOT
    s = D.load_streams(seq, root)
    ref = D.load_reference(seq, root)
    rig = RigCalibration.from_kalibr(s.calib_path)
    lo, hi = s.imu_t[0] + segment[0], s.imu_t[0] + segment[1]
    m = (s.imu_t >= lo) & (s.imu_t <= hi)
    n_frames = int(np.searchsorted(s.frame_t + rig.time_shift + offset, hi, side="right"))
    fe = run_frontend(s.frame_t, s.frame_paths, rig, s.imu_t[0], cfg=VOConfig(kf_every=kf_every),
                      index_sha=D.index_sha256(seq, root), frame_range=(0, n_frames), time_shift=rig.time_shift + offset)
    ev = [e for e in fe["events"] if lo <= e["t"] <= hi]
    rows = []
    for scale in GRID:
        fz = Frozen(imu_noise=noise, vo_cov_scale=scale)
        cfg, _, _ = system_configs(fz)["pathfinder"]
        out, diag = run_system("pathfinder", cfg, True, False, s.imu_t[m], s.gyro[m], s.acc[m], ev)
        res, _ = scoring.score(out, ref)
        rows.append({"scale": scale, "vo_nis_mean": diag["vo_nis_mean"], "vo_nis_median": diag["vo_nis_median"], "vo_nis_p95": diag["vo_nis_p95"],
                     "vo_used": diag["vo_used"], "vo_rejected": diag["vo_rejected"],
                     "ate_rmse_m": res.get("start_se3", {}).get("ate_all", {}).get("rmse"),
                     "rot_rmse_deg": res.get("start_se3", {}).get("rot_deg_all", {}).get("rmse"),
                     "nees_all": res.get("nees", {}).get("all")})
    best = min(rows, key=scale_score)
    vo_out = run_system("vo_only", None, True, False, s.imu_t[m], s.gyro[m], s.acc[m], ev)[0]
    vo_res, _ = scoring.score(vo_out, ref)
    return {"kf_every": kf_every, "selected_scale": best["scale"], "candidates": rows,
            "selected_ate_rmse_m": best["ate_rmse_m"], "frontend_key": fe["key"], "frontend_stats": fe["stats"],
            "vo_only_ate_rmse_m": vo_res.get("start_se3", {}).get("ate_all", {}).get("rmse"),
            "vo_only_rot_rmse_deg": vo_res.get("start_se3", {}).get("rot_deg_all", {}).get("rmse"),
            "gyro_rotation_check": gyro_rotation_check(s, ev, lo, hi)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", default="room1")
    ap.add_argument("--data-root", default=None, help="directory that contains dataset-<seq>_512_16")
    ap.add_argument("--segment", nargs=2, type=float, default=(0.0, 60.0), help="seconds since first IMU sample")
    ap.add_argument("--write", default=None, help="write the selected calibration JSON here (refuses overwrite)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sync", default=None, help="development sync measurement JSON")
    a = ap.parse_args(argv)
    offset = 0.
    if a.sync:
        sync = json.loads(Path(a.sync).read_text(encoding="utf-8"))["camera_imu"]
        value = sync.get("offset_s")
        if sync.get("status") == "unsupported" or value is None or not np.isfinite(value):
            raise ValueError("sync measurement has no supported finite offset")
        offset = float(value) if abs(value) > 1e-3 else 0.
    root = Path(a.data_root) if a.data_root else D.ROOT
    s = D.load_streams(a.seq, root)
    ref = D.load_reference(a.seq, root)
    rig = RigCalibration.from_kalibr(s.calib_path)
    noise = imu_noise_from_config(s.dir / "dso" / "imu_config.yaml")
    kw = {"root": Path(a.out)} if a.out else {}
    inputs = {"imu": s.dir / "mav0/imu0/data.csv", "cam_index": s.dir / "mav0/cam0/data.csv",
              "camchain": s.calib_path, "imu_config": s.dir / "dso/imu_config.yaml",
              "mocap_diagnostic_and_kf_selection": s.dir / "mav0/mocap0/data.csv"}
    with run_bundle(f"calibrate-vo-{a.seq}", vars(a), inputs, protocol="calibration", **kw) as run:
        src_hash = _src_hash()                 # computed once, from the code this run executes
        with ProcessPoolExecutor(max_workers=min(len(KF_GRID), os.cpu_count() or 1)) as pool:
            per_kf = list(pool.map(_one_kf, [(a.seq, kf, tuple(a.segment), noise, offset, str(root)) for kf in KF_GRID]))
        for r in per_kf:
            for c in r["candidates"]:
                print(r["kf_every"], c)
        chosen = select_keyframe(per_kf)
        best = {"scale": chosen["selected_scale"]}
        result = {"schema": SCHEMA, "sequence": a.seq, "segment_s": list(a.segment),
                  "split_role": "calibration (development recording, first segment)",
                  "camera_time_offset_s": offset,
                  "selection_rule": "per kf_every: scale minimizes log deviation of median and p95 NIS from chi-square(6); "
                                    "kf_every minimizes vision-only calibration ATE, rotation RMSE breaks ties (development only)",
                  "grid": list(GRID), "kf_grid": list(KF_GRID), "selected_vo_cov_scale": best["scale"],
                  "selected_kf_every": chosen["kf_every"], "per_kf_every": per_kf,
                  "imu_noise": noise, "imu_noise_source": "dataset dso/imu_config.yaml (inflated by the dataset authors)",
                  "frontend_src_sha256": src_hash,
                  "calibration_sha256": rig.meta["sha256"], "imu_config_sha256": sha256_file(s.dir / "dso/imu_config.yaml"),
                  "opencv": cv2.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
                  "vo_covariance_validated": False,
                  "run": str(run.path.name),
                  "limits": "Single development recording; covariance validity is NOT asserted "
                            "(FilterConfig.vo_covariance_validated stays False). "
                            "OpenCV version is part of the fingerprint because KLT and feature detection are not stable across it. "
                            "Historical v1/v2 files predate this fingerprint and do not describe the current front end."}
        run.write_json("vo_calibration.json", result)
        if a.write:
            p = Path(a.write)
            if p.exists():
                raise FileExistsError(f"{p} exists; calibrations are versioned, write a new file")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
        print("selected scale", best["scale"], "kf_every", chosen["kf_every"], "run", run.path)


if __name__ == "__main__":
    main()
