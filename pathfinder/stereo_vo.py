"""Stereo visual odometry front end: relative motion measured from images only (issue #2 B).

Adapted from the frame-to-frame design in PR #5 (branch ``issue-2-real-vo``, KLT + stereo
triangulation + RANSAC PnP), with these changes:

* camera projection follows the calibration's model (``camera_model``: equidistant *or* rational
  radtan), instead of ``cv2.fisheye`` for every camera;
* every relative pose carries a covariance from the PnP information matrix, including the
  stereo-depth uncertainty of each landmark (not a constant);
* events follow the estimator's keyframe protocol: a ``kf`` event clones the pose at a frame with a
  usable stereo map, a ``rel`` event is the IMU motion since the previous keyframe (``kf_every``
  frames of KLT tracking, then one PnP solve);
* landmarks are anchored on their left-image ray (see ``triangulate``). After a failure
  (no relative solve), the next frame is still a keyframe, so recovery has an anchor and the first
  measurement after it is relative to a pose that the filter actually has.

Not done: bundle adjustment, a map, loop closure, IMU-aided feature prediction. No IMU sample, reference
pose, clean image or fault label is read here, so its output is an independent visual measurement.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .camera_model import RigCalibration


@dataclass
class VOConfig:
    max_corners: int = 300
    min_distance: int = 12
    quality: float = 0.003
    fb_px: float = 1.0           # forward-backward KLT tolerance, px
    ransac_px: float = 1.5       # PnP inlier threshold, px
    min_inliers: int = 15
    good_inliers: int = 80       # health 1 at this many inliers
    depth_range: tuple = (0.2, 15.0)
    guess_depth: float = 2.0     # m, nominal depth for predicting the right-image position
    stereo_rep_px: float = 1.0   # max stereo reprojection error accepted for a landmark
    pixel_std: float = 0.5       # assumed corner localisation noise, px (floor of the residual scale)
    max_rot_rad: float = 0.6     # larger frame-to-frame rotations are treated as failed solves
    max_trans_m: float = 0.5     # ... and translations (5 m/s at 10 Hz); both per tracked frame
    kf_every: int = 1            # frames tracked against one keyframe before the next relative solve
    kf_min_track_ratio: float = 0.5   # ... or earlier, when fewer than this fraction of its tracks survive


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


class StereoVO:
    def __init__(self, rig: RigCalibration, cfg: VOConfig | None = None):
        self.rig, self.cfg = rig, cfg or VOConfig()
        self.T_c1_c0 = rig.T_c1_c0
        self.T_i_c0 = np.linalg.inv(rig.T_c0_i)
        self.f = rig.cam0.focal
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # (last image, tracked px (N, 2), keyframe landmarks X (N, 3) in keyframe cam0, Sigma_X, kf_id,
        #  landmarks at the keyframe, frames since the keyframe)
        self.prev = None
        self.kf_id = 0
        self.stats = {"frames": 0, "rel": 0, "fail_track": 0, "fail_pnp": 0, "fail_stereo": 0, "fail_motion": 0}

    # ------------------------------------------------------------------ tracking primitives
    def _klt(self, a, b, pts, guess=None):
        if len(pts) == 0:
            return pts, np.zeros(0, bool)
        p = pts.reshape(-1, 1, 2).astype(np.float32)
        if guess is not None:
            q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, guess.reshape(-1, 1, 2).astype(np.float32),
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        else:
            q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **self.lk)
        if q is None:
            return pts, np.zeros(len(pts), bool)
        r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, p.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((r - p)[:, 0], axis=1) < self.cfg.fb_px)
        return q[:, 0].astype(float), ok

    def triangulate(self, img0, img1):
        """Landmarks in cam0 from one stereo pair: (pixels, X (N, 3), Sigma_X (N, 3, 3))."""
        cfg, rig = self.cfg, self.rig
        empty = (np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3, 3)))
        pts = cv2.goodFeaturesToTrack(img0, cfg.max_corners, cfg.quality, cfg.min_distance)
        if pts is None:
            return empty
        pts = pts[:, 0].astype(float)
        n0 = rig.cam0.unproject(pts)
        ok0 = np.isfinite(n0).all(1)
        pts, n0 = pts[ok0], n0[ok0]
        # Predict each corner in the right image from the calibration at a nominal depth (needed for
        # rigs whose cameras point apart, e.g. Odyssey+; harmless for parallel rigs).
        X1 = (self.T_c1_c0[:3, :3] @ (np.c_[n0, np.ones(len(n0))] * cfg.guess_depth).T).T + self.T_c1_c0[:3, 3]
        guess = rig.cam1.project(X1)
        h, w = img1.shape[:2]
        inside = np.isfinite(guess).all(1) & (guess[:, 0] >= 0) & (guess[:, 0] < w) & (guess[:, 1] >= 0) \
            & (guess[:, 1] < h)
        pts, n0, guess = pts[inside], n0[inside], guess[inside]
        q, ok = self._klt(img0, img1, pts, guess)
        pts, n0, q = pts[ok], n0[ok], q[ok]
        n1 = rig.cam1.unproject(q) if len(q) else np.zeros((0, 2))
        ok1 = np.isfinite(n1).all(1)
        pts, n0, n1 = pts[ok1], n0[ok1], n1[ok1]
        if len(pts) < 8:
            return empty
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.T_c1_c0[:3]
        Xh = cv2.triangulatePoints(P0, P1, n0.T, n1.T)
        X = (Xh[:3] / Xh[3]).T
        x1 = (P1 @ np.c_[X, np.ones(len(X))].T).T
        rep = np.linalg.norm(x1[:, :2] / x1[:, 2:3] - n1, axis=1) * rig.cam1.focal
        lo, hi = cfg.depth_range
        keep = (X[:, 2] > lo) & (X[:, 2] < hi) & (rep < cfg.stereo_rep_px) & (x1[:, 2] > lo)
        pts, X, n0 = pts[keep], X[keep], n0[keep]
        # Anchor each landmark on its left-image ray (keep the triangulated depth). DLT triangulation
        # splits any residual epipolar error between the two views, so X would not reproject onto its
        # own left pixel: on TUM-VI room1 that offset is systematic (vertical) and produced a constant
        # ~0.02 deg/frame pitch bias against the gyro. With the anchor, zero motion gives exactly zero
        # PnP residual at the identity, as it must.
        X = np.c_[n0, np.ones(len(n0))] * X[:, 2:3]
        # Landmark covariance: lateral pixel noise and stereo depth noise sigma_z = z^2 sigma_d / (f b),
        # with sigma_d = sqrt(2) pixel_std (one corner in each image).
        z = np.linalg.norm(X, axis=1)
        sig_lat = cfg.pixel_std * z / self.f
        sig_z = z ** 2 * np.sqrt(2) * cfg.pixel_std / (self.f * max(rig.baseline, 1e-3))
        ray = X / z[:, None]
        S = (sig_lat ** 2)[:, None, None] * (np.eye(3)[None] - ray[:, :, None] * ray[:, None, :]) \
            + (sig_z ** 2)[:, None, None] * ray[:, :, None] * ray[:, None, :]
        return pts, X, S

    # ------------------------------------------------------------------ relative pose
    def _pnp(self, X, S, uv):
        """RANSAC PnP + weighted Gauss-Newton refinement. Returns (T_cur_prev, cov6 on right-perturbed
        [rho (3), phi (3)] of T_cur_prev, inliers) or None."""
        cfg, cam = self.cfg, self.rig.cam0
        nq = cam.unproject(uv)
        ok = np.isfinite(nq).all(1)
        X, S, nq = X[ok], S[ok], nq[ok]
        if len(X) < cfg.min_inliers:
            return None
        good, rvec, tvec, inl = cv2.solvePnPRansac(X, nq, np.eye(3), None, iterationsCount=200,
                                                   reprojectionError=cfg.ransac_px / self.f, confidence=0.999,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not good or inl is None or len(inl) < cfg.min_inliers:
            return None
        idx = inl[:, 0]
        X, S, nq = X[idx], S[idx], nq[idx]
        R = cv2.Rodrigues(rvec)[0]
        t = tvec[:, 0].astype(float)
        sig2 = (cfg.pixel_std / self.f) ** 2
        for _ in range(8):                     # Gauss-Newton with per-landmark covariance weights
            Y = X @ R.T + t
            zc = Y[:, 2]
            if np.any(zc <= 1e-6):
                return None
            r = (Y[:, :2] / zc[:, None] - nq).reshape(-1)
            Jp = np.zeros((len(X), 2, 3))       # d proj / dY
            Jp[:, 0, 0] = Jp[:, 1, 1] = 1 / zc
            Jp[:, 0, 2] = -Y[:, 0] / zc ** 2
            Jp[:, 1, 2] = -Y[:, 1] / zc ** 2
            # Right perturbation T Exp([rho, phi]): Y = R (X + rho + phi x X) + t -> dY = R drho - R [X]x dphi
            JX = Jp @ R                          # (N, 2, 3) wrt landmark / rho
            Jphi = -np.einsum("nij,njk->nik", JX, _skew_n(X))
            J = np.concatenate([JX, Jphi], 2)    # (N, 2, 6)
            C = sig2 * np.eye(2)[None] + np.einsum("nij,njk,nlk->nil", JX, S, JX)
            W = np.linalg.inv(C)
            A = np.einsum("nji,njk,nkl->il", J, W, J)
            b = np.einsum("nji,njk,nk->i", J, W, r.reshape(-1, 2))
            try:
                dx = -np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
            dR = Rot.from_rotvec(dx[3:]).as_matrix()
            # T Exp(dx): R' = R dR, t' = R drho + t (first order in rho)
            t = R @ dx[:3] + t
            R = R @ dR
            if np.linalg.norm(dx) < 1e-10:
                break
        Y = X @ R.T + t
        res = Y[:, :2] / Y[:, 2:3] - nq
        chi2 = float(np.einsum("ni,nij,nj->", res, W, res))
        dof = max(2 * len(X) - 6, 1)
        scale = max(chi2 / dof, 1.0)           # inflate if residuals exceed the assumed noise
        try:
            cov = np.linalg.inv(A) * scale
        except np.linalg.LinAlgError:
            return None
        return _se3(R, t), cov, len(X), float(np.sqrt(np.mean(np.sum(res ** 2, 1)))) * self.f

    def _imu_relative(self, T_cc_cp, cov_c):
        """Camera relative pose -> IMU pose now in the previous IMU frame, with propagated covariance."""
        def f(xi):
            T = T_cc_cp @ _se3(Rot.from_rotvec(xi[3:]).as_matrix(), xi[:3])
            T_ip_ic = np.linalg.inv(self.T_i_c0 @ T @ self.rig.T_c0_i)
            return T_ip_ic
        T0 = f(np.zeros(6))
        J = np.zeros((6, 6))
        eps = 1e-6
        for j in range(6):
            d = np.zeros(6)
            d[j] = eps
            T1 = f(d)
            J[:3, j] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, j] = Rot.from_matrix(T0[:3, :3].T @ T1[:3, :3]).as_rotvec() / eps
        cov = J @ cov_c @ J.T
        return T0[:3, 3], T0[:3, :3], 0.5 * (cov + cov.T)

    def step(self, t, img0, img1) -> list:
        """One stereo pair (uint8 grey, timestamp on the IMU clock). Returns estimator events.

        Features of the current keyframe are tracked frame to frame (KLT); every ``kf_every`` frames (or
        when too few tracks survive) the pose relative to the keyframe is solved once by PnP against the
        keyframe's stereo landmarks, a ``rel`` event is emitted, and the current frame becomes the next
        keyframe (``kf`` event, same timestamp, after the ``rel``). Consecutive relative measurements
        therefore use disjoint stereo triangulations, so their errors are not shared through common
        landmarks (no measurement is fused twice)."""
        cfg = self.cfg
        self.stats["frames"] += 1
        events = []
        new_kf = self.prev is None
        if self.prev is not None:
            img_p, pts_p, X_p, S_p, kf_p, n_kf, k = self.prev
            q, ok = self._klt(img_p, img0, pts_p)
            k += 1
            if ok.sum() < cfg.min_inliers:
                self.stats["fail_track"] += 1
                new_kf = True
            elif k >= cfg.kf_every or ok.sum() < cfg.kf_min_track_ratio * n_kf:
                new_kf = True
                sol = self._pnp(X_p[ok], S_p[ok], q[ok])
                if sol is None:
                    self.stats["fail_pnp"] += 1
                else:
                    T_cc_cp, cov_c, n_in, rms_px = sol
                    dp, dR, cov6 = self._imu_relative(T_cc_cp, cov_c)
                    ang = np.linalg.norm(Rot.from_matrix(dR).as_rotvec())
                    if ang > cfg.max_rot_rad * k or np.linalg.norm(dp) > cfg.max_trans_m * k                             or not np.isfinite(cov6).all():
                        self.stats["fail_motion"] += 1
                    else:
                        health = float(np.clip((n_in - cfg.min_inliers) / (cfg.good_inliers - cfg.min_inliers), 0, 1))
                        events.append({"t": float(t), "kind": "rel", "kf": kf_p, "dp": dp, "dR": dR, "cov": cov6,
                                       "health": health, "inliers": int(n_in), "tracked": int(ok.sum()),
                                       "rms_px": rms_px, "frames": int(k)})
                        self.stats["rel"] += 1
            else:
                self.prev = (img0, q[ok], X_p[ok], S_p[ok], kf_p, n_kf, k)
        if new_kf:
            pts, X, S = self.triangulate(img0, img1)
            if len(pts) >= cfg.min_inliers:
                self.kf_id += 1
                events.append({"t": float(t), "kind": "kf", "kf": self.kf_id, "landmarks": int(len(pts))})
                self.prev = (img0, pts, X, S, self.kf_id, len(pts), 0)
            else:
                self.stats["fail_stereo"] += 1
                self.prev = None              # nothing to track from: the next usable frame restarts
        return events


def _skew_n(X):
    S = np.zeros((len(X), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -X[:, 2], X[:, 1]
    S[:, 1, 0], S[:, 1, 2] = X[:, 2], -X[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -X[:, 1], X[:, 0]
    return S
