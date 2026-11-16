"""Run the stereo visual front end over a recorded image stream (issue #2 B).

``run_frontend`` reads each stereo pair from disk, applies the *image* corruption schedule (if any)
and only then hands the corrupted pair to ``StereoVO``. It returns the estimator events plus a
per-frame timing log. Nothing here reads IMU samples, reference poses or clean images once a fault
is active: the corruptor sees only the frame it corrupts (and its own previous outputs for
frozen / desynchronised frames).

Time conventions
    Fault schedules are in seconds since ``stream_t0`` (the first IMU sample of the recording), the
    same origin the scorer uses. Camera timestamps are mapped onto the IMU clock with the rig's
    Kalibr ``time_shift`` (t_imu = t_cam + shift). If the calibration file has no
    ``timeshift_cam_imu`` the shift is 0 *by assumption*; ``tools/check_sync.py`` measures it on
    development data and the evaluation records that measurement.

Results are cached under ``local_data/cache/vo`` keyed by the SHA-256 of the front-end source, its
configuration, the image index, calibration, fault schedule and seed, so a code change always
invalidates the cache. The cache is a local convenience (pickle of our own outputs), not a
published artifact.
"""

import hashlib
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import camera_model, image_faults, stereo_vo
from .camera_model import RigCalibration
from .image_faults import Corruptor, ImageFault
from .stereo_vo import StereoVO, VOConfig

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "local_data" / "cache" / "vo"
FRONTEND_VERSION = "pathfinder.frontend/1"


def load_pair(left: Path, right: Path):
    """8-bit grey stereo pair. TUM-VI ``512_16`` PNGs are 16-bit; OpenCV's grey conversion keeps the
    top 8 bits (value >> 8). Symlinks are refused: only regular extracted files are read."""
    for p in (left, right):
        if Path(p).is_symlink() or not Path(p).is_file():
            raise FileNotFoundError(f"not a regular image file: {p}")
    a = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise OSError(f"unreadable image pair {left}, {right}")
    return a, b


def _src_hash() -> str:
    """SHA-256 of the front-end sources with line endings normalised (a Windows CRLF checkout and a
    Linux checkout of the same commit hash identically)."""
    h = hashlib.sha256()
    for path in (stereo_vo.__file__, camera_model.__file__, image_faults.__file__, __file__):
        h.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def cache_key(index_sha: str, calib_sha: str, cfg: VOConfig, schedule, seed: int, time_shift: float,
              frame_range) -> str:
    blob = json.dumps({"v": FRONTEND_VERSION, "src": _src_hash(), "index": index_sha, "calib": calib_sha,
                       "cfg": asdict(cfg), "schedule": [f.as_dict() for f in schedule], "seed": int(seed),
                       "shift": float(time_shift), "range": list(frame_range)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_frontend(frame_t, frame_paths, rig: RigCalibration, stream_t0: float, schedule=(), seed: int = 0,
                 cfg: VOConfig | None = None, index_sha: str = "", time_shift: float | None = None,
                 frame_range=(0, None), cache: bool = True, cache_dir: Path = CACHE):
    """Returns {"events", "frames", "stats", "key"}.

    frame_t: camera timestamps (s, camera clock); frame_paths: [(left, right)].
    events: estimator events on the IMU clock (``kf`` / ``rel``), in emission order.
    frames: per input frame [t_rel, delivered, load_ms, corrupt_ms, vo_ms, n_rel, n_kf].
    """
    cfg = cfg or VOConfig()
    schedule = [f if isinstance(f, ImageFault) else ImageFault(**f) for f in schedule]
    shift = rig.time_shift if time_shift is None else float(time_shift)
    i0, i1 = frame_range[0], len(frame_t) if frame_range[1] is None else frame_range[1]
    calibration = {"cam0": asdict(rig.cam0), "cam1": asdict(rig.cam1),
                   "T_c0_i": rig.T_c0_i.tolist(), "T_c1_c0": rig.T_c1_c0.tolist(),
                   "time_shift": rig.time_shift}
    calib_sha = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    # Programmatic rigs and streams may not carry file hashes. Never let the
    # empty default identify unrelated recordings or calibrations as the same input.
    stream_hash = hashlib.sha256(index_sha.encode())
    stream_hash.update(np.asarray(frame_t, dtype="<f8").tobytes())
    stream_hash.update(np.asarray([stream_t0], dtype="<f8").tobytes())
    for pair in frame_paths:
        for image_path in pair:
            stream_hash.update(str(Path(image_path).resolve()).encode())
            if cache and not index_sha:
                with Path(image_path).open("rb") as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        stream_hash.update(block)
    key = cache_key(stream_hash.hexdigest(), calib_sha, cfg, schedule, seed, shift, (i0, i1))
    path = Path(cache_dir) / f"{key}.pkl"
    if cache and path.is_file():
        with open(path, "rb") as fh:
            out = pickle.load(fh)
        out["cached"] = True
        return out
    vo = StereoVO(rig, cfg)
    corr = Corruptor(schedule, seed) if schedule else None
    events, frames = [], []
    last_t = -np.inf
    for i in range(i0, i1):
        t_imu = float(frame_t[i]) + shift
        a0 = time.perf_counter()
        left, right = load_pair(*frame_paths[i])
        a1 = time.perf_counter()
        delivered = True
        if corr is not None:
            res = corr.apply(t_imu - stream_t0, left, right)
            if res is None:
                delivered = False
            else:
                t_rel, left, right = res
                t_imu = t_rel + stream_t0
        a2 = time.perf_counter()
        ev = []
        if delivered and t_imu <= last_t:
            delivered = False          # a jittered stamp that would go backwards: the driver drops it
        if delivered:
            ev = vo.step(t_imu, left, right)
            last_t = t_imu
        a3 = time.perf_counter()
        events.extend(ev)
        frames.append([t_imu - stream_t0, delivered, 1e3 * (a1 - a0), 1e3 * (a2 - a1), 1e3 * (a3 - a2),
                       sum(e["kind"] == "rel" for e in ev), sum(e["kind"] == "kf" for e in ev)])
    out = {"events": events, "frames": np.array(frames, float), "stats": dict(vo.stats), "key": key,
           "cfg": asdict(cfg), "schedule": [f.as_dict() for f in schedule], "seed": seed, "time_shift": shift,
           "cached": False}
    if cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(out, fh)
        tmp.replace(path)
    return out
