"""Real stereo visual odometry on TUM-VI (issue #2 B): motion measured from the images themselves.

Front end (OpenCV only, so it builds anywhere ``pip install opencv-python-headless`` does):
  1. Shi-Tomasi corners in the left image, tracked into the right image with pyramidal KLT and a
     forward-backward check, undistorted with the dataset's equidistant (Kannala-Brandt) fisheye
     calibration, and triangulated with the calibrated stereo baseline (about 10 cm): metric depth.
  2. Those 3D points are tracked into the next left image (KLT, forward-backward) and the relative
     camera motion is solved with RANSAC PnP on undistorted (normalised) coordinates.
  3. The camera motion is converted to the IMU frame with the calibrated camera-IMU extrinsics:
     the output is the IMU's relative pose between two frames, expressed in the earlier IMU frame.

What it is not: no bundle adjustment, no map, no loop closure. It is frame-to-frame odometry, so it
drifts slowly; Path Finder fuses it as a *relative* measurement between cloned states, never as an
absolute position fix. Nothing here reads ground truth, clean images or fault schedules.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


def _mat(rows):
    return np.array(rows, dtype=float)


@dataclass
class StereoCalib:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    T_c0_i: np.ndarray      # IMU -> cam0 (Kalibr T_cam_imu)
    T_c1_c0: np.ndarray     # cam0 -> cam1

    @staticmethod
    def load(path):
        c = yaml.safe_load(Path(path).read_text())

        def K(cam):
            fx, fy, cx, cy = cam["intrinsics"]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        return StereoCalib(K(c["cam0"]), np.array(c["cam0"]["distortion_coeffs"], float), K(c["cam1"]),
                           np.array(c["cam1"]["distortion_coeffs"], float), _mat(c["cam0"]["T_cam_imu"]),
                           _mat(c["cam1"]["T_cn_cnm1"]))


@dataclass
class VOConfig:
    max_corners: int = 300
    min_distance: int = 12
    quality: float = 0.003
    fb_px: float = 1.0          # forward-backward KLT tolerance
    ransac_px: float = 1.5      # PnP inlier threshold in pixels
    min_inliers: int = 12
    depth_range: tuple = (0.2, 15.0)
    guess_depth: float = 2.0    # metres: where the stereo prediction assumes a corner is


class StereoVO:
    def __init__(self, calib: StereoCalib, cfg: VOConfig = VOConfig()):
        self.c, self.cfg = calib, cfg
        self.f = float(calib.K0[0, 0])
        self.P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P1 = calib.T_c1_c0[:3]
        self.T_i_c0 = np.linalg.inv(calib.T_c0_i)
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.prev = None          # (img0, pts2d (N,2), pts3d (N,3) in cam0)

    def _klt(self, a, b, pts, guess=None):
        if len(pts) == 0:
            return pts, np.zeros(0, bool)
        p = pts.reshape(-1, 1, 2).astype(np.float32)
        if guess is not None:
            q0 = guess.reshape(-1, 1, 2).astype(np.float32)
            q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, q0, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        else:
            q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **self.lk)
        if q is None:
            return pts, np.zeros(len(pts), bool)
        # Backward check starts from the original corner (needed when the forward search used a
        # large predicted offset; identical result for small motions).
        r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, p.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((r - p)[:, 0], axis=1) < self.cfg.fb_px)
        return q[:, 0], ok

    def _norm(self, pts, K, D):
        return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D)[:, 0]

    def triangulate(self, img0, img1):
        """3D points (cam0 frame) from one stereo pair."""
        cfg = self.cfg
        pts = cv2.goodFeaturesToTrack(img0, cfg.max_corners, cfg.quality, cfg.min_distance)
        if pts is None:
            return np.zeros((0, 2)), np.zeros((0, 3))
        pts = pts[:, 0]
        # Predict each corner's position in the right image from the calibration (rotation between
        # the cameras + baseline at a nominal depth). Needed for headsets whose cameras point apart
        # (Samsung Odyssey+: 47 degrees); harmless for parallel rigs.
        rays = np.hstack([self._norm(pts, self.c.K0, self.c.D0), np.ones((len(pts), 1))])
        X1 = (self.P1[:, :3] @ (rays * self.cfg.guess_depth).T).T + self.P1[:, 3]
        front = X1[:, 2] > 0.05
        guess = pts.copy()
        if front.any():
            proj, _ = cv2.fisheye.projectPoints(X1[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                                self.c.K1, self.c.D1)
            guess[front] = proj[:, 0]
        h, w = img1.shape[:2]
        inside = front & (guess[:, 0] >= 0) & (guess[:, 0] < w) & (guess[:, 1] >= 0) & (guess[:, 1] < h)
        pts, guess = pts[inside], guess[inside]
        q, ok = self._klt(img0, img1, pts, guess)
        pts, q = pts[ok], q[ok]
        if len(pts) < 8:
            return np.zeros((0, 2)), np.zeros((0, 3))
        n0, n1 = self._norm(pts, self.c.K0, self.c.D0), self._norm(q, self.c.K1, self.c.D1)
        X = cv2.triangulatePoints(self.P0, self.P1, n0.T, n1.T)
        X = (X[:3] / X[3]).T
        x1 = (self.P1 @ np.hstack([X, np.ones((len(X), 1))]).T).T
        rep = np.linalg.norm(x1[:, :2] / x1[:, 2:3] - n1, axis=1) * self.f
        lo, hi = cfg.depth_range
        keep = (X[:, 2] > lo) & (X[:, 2] < hi) & (rep < 1.0)
        return pts[keep], X[keep]

    def step(self, img0, img1):
        """Feed one stereo pair (uint8 grey). Returns None for the first frame or a failed solve, else
        dict(R, t, inliers, tracked): the IMU pose at this frame relative to the previous frame,
        expressed in the previous IMU frame (t in metres)."""
        out = None
        if self.prev is not None:
            img_p, pts_p, X_p = self.prev
            q, ok = self._klt(img_p, img0, pts_p)
            tracked = int(ok.sum())
            if tracked >= self.cfg.min_inliers:
                nq = self._norm(q[ok], self.c.K0, self.c.D0)
                good, rvec, tvec, inl = cv2.solvePnPRansac(
                    X_p[ok], nq, np.eye(3), None, iterationsCount=200,
                    reprojectionError=self.cfg.ransac_px / self.f, confidence=0.999,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                n_in = 0 if inl is None else len(inl)
                if good and n_in >= self.cfg.min_inliers:
                    idx = inl[:, 0]
                    rvec, tvec = cv2.solvePnPRefineLM(X_p[ok][idx], nq[idx], np.eye(3), None, rvec, tvec)
                    T_cc_cp = np.eye(4)
                    T_cc_cp[:3, :3] = cv2.Rodrigues(rvec)[0]
                    T_cc_cp[:3, 3] = tvec[:, 0]
                    # IMU(current) <- IMU(previous), then invert: pose of current IMU in previous IMU frame.
                    T_ic_ip = self.T_i_c0 @ T_cc_cp @ self.c.T_c0_i
                    T_ip_ic = np.linalg.inv(T_ic_ip)
                    out = {"R": T_ip_ic[:3, :3], "t": T_ip_ic[:3, 3], "inliers": n_in, "tracked": tracked}
        pts, X = self.triangulate(img0, img1)
        self.prev = (img0, pts, X)
        return out


def load_pair(path0, cam1_dir):
    """Left and right uint8 images for one TUM-VI timestamp (same file name in cam0/cam1)."""
    a = cv2.imread(str(path0), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(Path(cam1_dir) / Path(path0).name), cv2.IMREAD_GRAYSCALE)
    return a, b
