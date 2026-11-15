"""Calibrated camera models and stereo/IMU rig calibration (issue #2 B).

Projection models are selected from the calibration file, never assumed:

equidistant  Kalibr ``pinhole-equi`` = OpenCV ``cv2.fisheye`` = Kannala-Brandt, 4 coefficients:
             theta = atan(r), theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8),
             x_d = (theta_d / r) x. TUM-VI and the Monado Valve Index recordings use it.
radtan       Kalibr ``pinhole-radtan`` = OpenCV standard / rational model, 4, 5 or 8 coefficients
             ordered k1, k2, p1, p2[, k3[, k4, k5, k6]]. The Monado Samsung Odyssey+ calibration is
             the 8-coefficient rational form; the fisheye model is wrong for it.

``unproject`` gives undistorted normalised coordinates (x/z, y/z); ``project`` maps camera-frame
points to pixels. Both are numpy and are tested against OpenCV.

Rig convention (Kalibr): ``T_c0_i`` maps IMU-frame points into cam0, ``T_c1_c0`` maps cam0 points into
cam1. ``time_shift`` follows Kalibr: t_imu = t_cam + time_shift.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    model: str                     # "equidistant" | "radtan"
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple
    width: int
    height: int

    def __post_init__(self):
        if self.model not in ("equidistant", "radtan"):
            raise ValueError(f"unsupported camera model {self.model!r}")
        n = len(self.dist)
        if (self.model == "equidistant" and n != 4) or (self.model == "radtan" and n not in (4, 5, 8)):
            raise ValueError(f"{self.model} needs {'4' if self.model == 'equidistant' else '4, 5 or 8'} "
                             f"coefficients, got {n}")
        vals = (self.fx, self.fy, self.cx, self.cy, *self.dist)
        if not np.all(np.isfinite(vals)) or self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("camera intrinsics must be finite with positive focal lengths and image size")

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def focal(self) -> float:
        return float(np.sqrt(self.fx * self.fy))

    # ------------------------------------------------------------------ distortion of normalised points
    def _distort(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        d = np.zeros(8)
        d[:len(self.dist)] = self.dist
        if self.model == "equidistant":
            r = np.hypot(x, y)
            th = np.arctan(r)
            th2 = th * th
            thd = th * (1 + d[0] * th2 + d[1] * th2 ** 2 + d[2] * th2 ** 3 + d[3] * th2 ** 4)
            s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
            return xy * s[:, None]
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        r2 = x * x + y * y
        radial = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], 1)

    def project(self, X):
        """Camera-frame points (N, 3) -> pixels (N, 2); NaN for points not in front of the camera."""
        X = np.asarray(X, float).reshape(-1, 3)
        z = X[:, 2]
        front = z > 1e-9
        xy = np.full((len(X), 2), np.nan)
        xy[front] = X[front, :2] / z[front, None]
        xd = self._distort(np.nan_to_num(xy))
        uv = np.stack([self.fx * xd[:, 0] + self.cx, self.fy * xd[:, 1] + self.cy], 1)
        uv[~front] = np.nan
        return uv

    def unproject(self, uv, iters: int = 20):
        """Pixels (N, 2) -> undistorted normalised coordinates (N, 2) by Gauss-Newton on _distort.

        Points whose inversion does not converge to 1e-3 px are returned as NaN (outside the valid
        field of view of the model) rather than silently mapped."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        xd = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], 1)
        if self.model == "equidistant":
            thd = np.hypot(xd[:, 0], xd[:, 1])
            th = np.clip(thd, 0, np.pi / 2 - 1e-6)
            d = self.dist
            for _ in range(iters):
                t2 = th * th
                f = th * (1 + d[0] * t2 + d[1] * t2 ** 2 + d[2] * t2 ** 3 + d[3] * t2 ** 4) - thd
                df = 1 + 3 * d[0] * t2 + 5 * d[1] * t2 ** 2 + 7 * d[2] * t2 ** 3 + 9 * d[3] * t2 ** 4
                th = np.clip(th - f / np.where(np.abs(df) > 1e-12, df, 1e-12), 0, np.pi / 2 - 1e-6)
            r = np.tan(th)
            s = np.where(thd > 1e-12, r / np.maximum(thd, 1e-12), 1.0)
            xy = xd * s[:, None]
        else:
            xy = xd.copy()
            eps = 1e-7
            for _ in range(iters):
                f = self._distort(xy) - xd
                J = np.empty((len(xy), 2, 2))
                for j in range(2):
                    step = np.zeros(2)
                    step[j] = eps
                    J[:, :, j] = (self._distort(xy + step) - self._distort(xy - step)) / (2 * eps)
                xy = xy - np.linalg.solve(J, f[:, :, None])[:, :, 0]
        err = np.linalg.norm((self._distort(xy) - xd) * [self.fx, self.fy], axis=1)
        xy[~(np.isfinite(err) & (err < 1e-3))] = np.nan
        return xy


@dataclass
class RigCalibration:
    cam0: CameraModel
    cam1: CameraModel
    T_c0_i: np.ndarray             # IMU -> cam0
    T_c1_c0: np.ndarray            # cam0 -> cam1
    time_shift: float = 0.0        # t_imu = t_cam + time_shift (Kalibr)
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("T_c0_i", "T_c1_c0"):
            T = np.asarray(getattr(self, name), float)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError(f"{name} must be a finite homogeneous 4x4 transform")
            R = T[:3, :3]
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.999:
                raise ValueError(f"{name} rotation is not orthonormal")
            setattr(self, name, T)
        if not np.isfinite(self.time_shift) or abs(self.time_shift) > 0.5:
            raise ValueError("camera-IMU time shift must be finite and below 0.5 s")

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T_c1_c0[:3, 3]))

    @staticmethod
    def _camera(entry: dict) -> CameraModel:
        cm, dm = entry.get("camera_model", "pinhole"), entry["distortion_model"]
        if cm != "pinhole":
            raise ValueError(f"unsupported Kalibr camera model {cm!r}")
        model = {"equidistant": "equidistant", "radtan": "radtan", "radtan8": "radtan"}.get(dm)
        if model is None:
            raise ValueError(f"unsupported Kalibr distortion model {dm!r}")
        fx, fy, cx, cy = (float(v) for v in entry["intrinsics"])
        w, h = (int(v) for v in entry["resolution"])
        return CameraModel(model, fx, fy, cx, cy, tuple(float(v) for v in entry["distortion_coeffs"]), w, h)

    @staticmethod
    def from_kalibr(path) -> "RigCalibration":
        """Load a Kalibr camchain (cam0/cam1 with T_cam_imu, T_cn_cnm1, timeshift_cam_imu)."""
        import yaml
        text = Path(path).read_text()
        c = yaml.safe_load(text)
        if "cam0" not in c or "cam1" not in c:
            raise ValueError("camchain needs cam0 and cam1")
        T_c0_i = np.array(c["cam0"]["T_cam_imu"], float)
        if "T_cn_cnm1" in c["cam1"]:
            T_c1_c0 = np.array(c["cam1"]["T_cn_cnm1"], float)
        else:
            T_c1_c0 = np.array(c["cam1"]["T_cam_imu"], float) @ np.linalg.inv(T_c0_i)
        shift = float(c["cam0"].get("timeshift_cam_imu", 0.0))
        shift1 = float(c["cam1"].get("timeshift_cam_imu", shift))
        if abs(shift1 - shift) > 1e-4:
            raise ValueError("cameras with different IMU time shifts are not supported")
        import hashlib
        return RigCalibration(RigCalibration._camera(c["cam0"]), RigCalibration._camera(c["cam1"]), T_c0_i,
                              T_c1_c0, shift, str(path), {"sha256": hashlib.sha256(text.encode()).hexdigest()})
