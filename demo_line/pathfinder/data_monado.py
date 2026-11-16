"""Loader for the Monado SLAM Dataset (MSD, Collabora, CC-BY 4.0): real consumer VR headsets.

    Mateo de Mayo et al., "The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking",
    arXiv 2508.00088. Data: https://huggingface.co/datasets/collabora/monado-slam-datasets

Headsets used here: Samsung Odyssey+ (2 x 640x480 @ 30 fps, IMU 1000 Hz) and Valve Index
(2 x 960x960 @ 54 fps, IMU 1000 Hz). Ground truth: three SteamVR Lighthouse base stations
(~1 cm), already transformed to the IMU pose and time-aligned to the IMU clock by the authors.

World frame: SteamVR is y-up; ``load`` rotates the ground truth so gravity is along -z, as the
filter expects (the rotation is estimated from the data's own mean specific force).

Layout (EuRoC/ASL): local_data/monado/<SEQ>/mav0/{imu0,cam0,cam1,gt}/data.csv, images under
cam*/data/. Frame times come from the CSV's first column (IMU-aligned clock); the image FILE
NAMES carry the raw camera clock, so they must not be used as timestamps.
Calibration (Basalt format, T_imu_cam per camera): local_data/monado/calib/<headset>.yaml.
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

from .sequence import Sequence
from .stereo_vo import StereoCalib

ROOT = Path(__file__).resolve().parent.parent / "local_data" / "monado"
HEADSET = {"MO": "odyssey_plus", "MI": "valve_index", "MG": "reverb_g2"}


def available(root: Path = ROOT) -> list[str]:
    return sorted(p.name for p in root.glob("M*") if (p / "mav0" / "gt" / "data.csv").exists())


def headset(name: str) -> str:
    return HEADSET[name[:2]]


def _csv(path):
    return np.loadtxt(path, delimiter=",", comments="#", ndmin=2)


def load(name: str, root: Path = ROOT, rate: float = 200.0) -> Sequence:
    """IMU resampled to ``rate`` Hz (the filter's 200 Hz), ground truth interpolated / slerped."""
    d = root / name / "mav0"
    imu, gt = _csv(d / "imu0" / "data.csv"), _csv(d / "gt" / "data.csv")
    t_imu, t_gt = imu[:, 0] * 1e-9, gt[:, 0] * 1e-9
    t0, t1 = max(t_imu[0], t_gt[0]) + 0.2, min(t_imu[-1], t_gt[-1]) - 0.2
    t = np.arange(t0, t1, 1.0 / rate)
    # 1000 Hz -> 200 Hz: average each 5 ms bin (anti-aliasing) instead of point sampling.
    k = max(1, int(round(1.0 / rate / np.median(np.diff(t_imu)))))
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(imu[:, 1 + i], ker, mode="same") for i in range(6)], 1)
    gyro = np.stack([np.interp(t, t_imu, sm[:, i]) for i in range(3)], 1)
    acc = np.stack([np.interp(t, t_imu, sm[:, 3 + i]) for i in range(3)], 1)
    p = np.stack([np.interp(t, t_gt, gt[:, 1 + i]) for i in range(3)], 1)
    rot = Slerp(t_gt, R.from_quat(gt[:, [5, 6, 7, 4]]))(t)                   # file is w,x,y,z
    # SteamVR's world frame is y-up; our filter assumes z-up (gravity along -z). Rotate the
    # ground-truth world so the measured mean specific force (gravity reaction) points along +z.
    up = rot.apply(acc).mean(0)
    up /= np.linalg.norm(up)
    to_z = R.align_vectors([[0.0, 0.0, 1.0]], [up])[0]
    p = to_z.apply(p)
    q = (to_z * rot).as_quat()
    v = np.gradient(savgol_filter(p, 41, 3, axis=0), t, axis=0)
    return Sequence(name=f"monado_{name}", t=t - t[0], gyro=gyro[None], acc=acc[None], p=p, q=q, v=v,
                    meta={"source": "monado", "headset": headset(name), "t_abs0": float(t[0]), "dir": str(d)})


def frames(seq: Sequence):
    """(time in seconds relative to the sequence, left image path) for frames inside the sequence."""
    d = Path(seq.meta["dir"]) / "cam0"
    rows = [r.split(",") for r in (d / "data.csv").read_text().splitlines() if r and not r.startswith("#")]
    ts = np.array([int(r[0]) * 1e-9 for r in rows]) - seq.meta["t_abs0"]
    paths = [d / "data" / r[1].strip() for r in rows]
    m = (ts >= 0) & (ts <= seq.t[-1])
    return ts[m], [p for p, k in zip(paths, m) if k]


def calib(seq_or_headset, root: Path = ROOT) -> StereoCalib:
    """Stereo calibration from the dataset's Basalt yaml (T_SC = T_imu_cam: camera -> IMU)."""
    h = seq_or_headset.meta["headset"] if isinstance(seq_or_headset, Sequence) else seq_or_headset
    c = yaml.safe_load((root / "calib" / f"{h}.yaml").read_text())["cameras"]

    def K(cam):
        (fx, fy), (cx, cy) = cam["focal_length"], cam["principal_point"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    T_i_c0 = np.array(c[0]["T_SC"], float).reshape(4, 4)
    T_i_c1 = np.array(c[1]["T_SC"], float).reshape(4, 4)
    return StereoCalib(K(c[0]), np.array(c[0]["distortion_coefficients"], float), K(c[1]),
                       np.array(c[1]["distortion_coefficients"], float),
                       np.linalg.inv(T_i_c0), np.linalg.inv(T_i_c1) @ T_i_c0)
