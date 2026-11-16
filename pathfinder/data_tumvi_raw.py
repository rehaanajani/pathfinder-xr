"""TUM-VI raw streams for the real-vision path, with runtime inputs and reference kept apart.

``load_streams`` returns only what a live system would have: IMU samples on the IMU clock, stereo
frame timestamps/paths and the calibration. ``load_reference`` returns the motion-capture poses for
scoring and declared alignment only. The evaluation runner never passes the reference to the
estimator or the front end.

Conventions (checked from the files, see ``check_conventions``):
* all ``data.csv`` timestamps are integer nanoseconds; IMU columns are
  ``w_RS_S_[xyz]`` (rad/s) then ``a_RS_S_[xyz]`` (m/s^2) in the IMU frame S;
* mocap columns are ``p_RS_R_[xyz]`` and ``q_RS_[wxyz]``: the pose of S in the mocap frame R
  (names follow the EuRoC convention); whether that frame is gravity aligned and whether the mocap
  clock matches the IMU clock is *measured* on development data by ``tools/check_sync.py``, not
  assumed;
* the calibration is ``dso/camchain.yaml`` (Kalibr), if present in the export.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "local_data" / "tumvi"


@dataclass
class Streams:
    name: str
    imu_t: np.ndarray        # (N,) s, IMU clock
    gyro: np.ndarray         # (N, 3) rad/s, IMU frame
    acc: np.ndarray          # (N, 3) m/s^2 specific force, IMU frame
    frame_t: np.ndarray      # (M,) s, camera clock (cam0 timestamps)
    frame_paths: list        # [(left path, right path)]
    calib_path: Path | None
    dir: Path


@dataclass
class Reference:
    t: np.ndarray            # (L,) s, mocap clock
    p: np.ndarray            # (L, 3) m, IMU position in mocap frame
    q: np.ndarray            # (L, 4) xyzw, IMU -> mocap frame


def seq_dir(name: str, root: Path = ROOT) -> Path:
    return Path(root) / f"dataset-{name}_512_16"


def _csv(path):
    return np.loadtxt(path, delimiter=",", comments="#", ndmin=2)


def header(path) -> list[str]:
    with open(path) as fh:
        first = fh.readline().lstrip("#").strip()
    return [c.strip() for c in first.split(",")]


def load_streams(name: str, root: Path = ROOT) -> Streams:
    d = seq_dir(name, root)
    m = d / "mav0"
    imu = _csv(m / "imu0" / "data.csv")
    if imu.shape[1] != 7:
        raise ValueError("IMU index must have seven columns")
    _check_times(imu[:, 0], "IMU")
    rows = [ln.split(",") for ln in (m / "cam0" / "data.csv").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    ts = np.array([int(r[0]) for r in rows], np.int64)
    right_rows = [ln.split(",") for ln in (m / "cam1" / "data.csv").read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
    if [[v.strip() for v in r] for r in rows] != [[v.strip() for v in r] for r in right_rows]:
        raise ValueError("stereo indices must have matching timestamps and filenames")
    _check_times(ts, "camera")
    pairs, keep = [], []
    for i, (_, fname) in enumerate(rows):
        fname = fname.strip()
        if not fname or fname in (".", "..") or any(c in fname for c in ("/", "\\", ":")):
            raise ValueError("camera filename must be a plain basename")
        a, b = m / "cam0" / "data" / fname.strip(), m / "cam1" / "data" / fname.strip()
        for path in (a, b):
            base = path.parent.resolve()
            if not path.resolve().is_relative_to(base):
                raise ValueError("camera path escapes its data directory")
            if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                   for p in (path, *path.parents)):
                raise ValueError("camera paths must not traverse links")
        # Only regular extracted files: archive symlinks (e.g. dso/cam*/images) are never followed.
        if a.is_file() and b.is_file() and not a.is_symlink() and not b.is_symlink():
            pairs.append((a, b))
            keep.append(i)
    calib = d / "dso" / "camchain.yaml"
    if len(keep) < len(rows):
        import warnings
        warnings.warn(f"{name}: {len(rows) - len(keep)} of {len(rows)} stereo pairs missing or not regular files")
    # float64 seconds since the epoch resolve ~0.2 us, far below the 5 ms IMU period.
    return Streams(name, imu[:, 0] * 1e-9, imu[:, 1:4], imu[:, 4:7], ts[keep] * 1e-9, pairs,
                   calib if calib.is_file() else None, d)


def index_sha256(name: str, root: Path = ROOT) -> str:
    """Hash of the camera index, IMU and calibration files: identifies the runtime input recording."""
    import hashlib
    d = seq_dir(name, root)
    h = hashlib.sha256()
    for rel in ("mav0/cam0/data.csv", "mav0/cam1/data.csv", "mav0/imu0/data.csv", "dso/camchain.yaml"):
        p = d / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()


def reference_sha256(name: str, root: Path = ROOT) -> str:
    import hashlib
    return hashlib.sha256((seq_dir(name, root) / "mav0" / "mocap0" / "data.csv").read_bytes()).hexdigest()


def load_reference(name: str, root: Path = ROOT) -> Reference:
    gt = _csv(seq_dir(name, root) / "mav0" / "mocap0" / "data.csv")
    q = gt[:, [5, 6, 7, 4]]
    _check_times(gt[:, 0], "reference")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(gt).all() or np.any(norm <= 0):
        raise ValueError("reference poses must be finite with nonzero quaternions")
    q = q / norm
    return Reference(gt[:, 0] * 1e-9, gt[:, 1:4], q)


def _check_times(t, name):
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name} timestamps must be finite, nonempty and strictly increasing")


def check_conventions(name: str, root: Path = ROOT) -> dict:
    """Column-name and timestamp checks on the actual files (raises on mismatch)."""
    m = seq_dir(name, root) / "mav0"
    out = {"imu_header": header(m / "imu0" / "data.csv"), "mocap_header": header(m / "mocap0" / "data.csv"),
           "cam_header": header(m / "cam0" / "data.csv")}
    imu_h, mo_h = out["imu_header"], out["mocap_header"]
    if not (imu_h[1].startswith("w_RS_S_x") and imu_h[4].startswith("a_RS_S_x")):
        raise ValueError(f"unexpected IMU columns {imu_h}")
    if not (mo_h[1].startswith("p_RS_R_x") and mo_h[4].startswith("q_RS_w")):
        raise ValueError(f"unexpected mocap columns {mo_h}")
    s = load_streams(name, root)
    r = load_reference(name, root)
    for k, t in (("imu", s.imu_t), ("frames", s.frame_t), ("mocap", r.t)):
        dt = np.diff(t)
        out[k] = {"n": int(len(t)), "t0": float(t[0]), "t1": float(t[-1]), "median_dt": float(np.median(dt)),
                  "max_gap": float(dt.max()), "non_increasing": int(np.sum(dt <= 0))}
    out["calibration"] = str(s.calib_path) if s.calib_path else None
    return out
