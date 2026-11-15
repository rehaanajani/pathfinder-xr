"""Loader for TUM-VI (Schubert et al., IROS 2018), EuRoC export at 512x512.

Real sensors: Bosch BMI160 IMU at 200 Hz, global-shutter fisheye camera at 20 Hz, OptiTrack
motion capture at 120 Hz. The exported ground truth (p_RS_R, q_RS) is already the IMU pose in the
mocap frame (z up), so no extra extrinsic is needed for trajectory evaluation.

Download (no sign-up): https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar
The room sequences have motion capture over the whole run.
"""

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

from .sequence import Sequence

ROOT = Path(__file__).resolve().parent.parent / "local_data" / "tumvi"


def available(root: Path = ROOT) -> list[str]:
    return sorted(p.name.replace("dataset-", "").replace("_512_16", "")
                  for p in root.glob("dataset-*_512_16") if (p / "mav0" / "mocap0" / "data.csv").exists())


def _dir(name: str, root: Path = ROOT) -> Path:
    return root / f"dataset-{name}_512_16" / "mav0"


def load(name: str, root: Path = ROOT, rate: float = 200.0) -> Sequence:
    d = _dir(name, root)
    imu = np.loadtxt(d / "imu0" / "data.csv", delimiter=",", comments="#")
    gt = np.loadtxt(d / "mocap0" / "data.csv", delimiter=",", comments="#")
    t_imu, t_gt = imu[:, 0] * 1e-9, gt[:, 0] * 1e-9
    t0, t1 = max(t_imu[0], t_gt[0]) + 0.5, min(t_imu[-1], t_gt[-1]) - 0.5
    t = np.arange(t0, t1, 1.0 / rate)
    gyro = np.stack([np.interp(t, t_imu, imu[:, 1 + i]) for i in range(3)], 1)
    acc = np.stack([np.interp(t, t_imu, imu[:, 4 + i]) for i in range(3)], 1)
    # Ground truth: positions interpolated, rotations slerped, velocity from smoothed positions.
    p = np.stack([np.interp(t, t_gt, gt[:, 1 + i]) for i in range(3)], 1)
    q_wxyz = gt[:, 4:8]
    rot_gt = R.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    q = Slerp(t_gt, rot_gt)(t).as_quat()
    p_s = savgol_filter(p, 41, 3, axis=0)
    v = np.gradient(p_s, t, axis=0)
    return Sequence(name=f"tumvi_{name}", t=t - t[0], gyro=gyro[None], acc=acc[None], p=p, q=q, v=v,
                    meta={"source": "tumvi", "t_abs0": float(t[0]), "dir": str(d)})


def frames(seq: Sequence):
    """Sorted (absolute time in seconds, path) of the real left-camera frames inside the sequence."""
    d = Path(seq.meta["dir"]) / "cam0" / "data"
    files = sorted(d.glob("*.png"))
    ts = np.array([int(f.stem) * 1e-9 for f in files])
    t0 = seq.meta["t_abs0"]
    m = (ts >= t0) & (ts <= t0 + seq.t[-1])
    return ts[m] - t0, [f for f, k in zip(files, m) if k]
