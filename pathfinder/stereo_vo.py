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
from .filter import valid_covariance


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
    stale_mad: float = 0.25      # mean |difference| (grey levels) below which a frame repeats the last one
    max_frame_gap_s: float = 0.5  # restart tracking after a delivery/stale gap


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
        self.stats = {"frames": 0, "rel": 0, "fail_track": 0, "fail_pnp": 0, "fail_stereo": 0, "fail_motion": 0,
                      "stale": 0}
        self.last_raw = None      # previous delivered left image (stale-frame check)
        self.last_track_t = None
        self.keyframe_t = None

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
        valid = np.isfinite(Xh).all(0) & (np.abs(Xh[3]) > 1e-12)
        Xh, pts, n0, n1 = Xh[:, valid], pts[valid], n0[valid], n1[valid]
        X = (Xh[:3] / Xh[3]).T
        x1 = (P1 @ np.c_[X, np.ones(len(X))].T).T
        rep = np.full(len(X), np.inf)
        front = x1[:, 2] > 1e-9
        rep[front] = np.linalg.norm(rig.cam1.project(x1[front]) -
            rig.cam1.project(np.c_[n1[front], np.ones(front.sum())]), axis=1)
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
        uv = uv[ok]
        # Generate minimal-pose hypotheses on rays, but score them in distorted
        # image pixels: a normalized-plane threshold is invalid for wide-angle rays.
        rng = np.random.default_rng(0)
        best, best_mask, best_score = None, None, (-1, -np.inf)
        limit = 200
        trial = 0
        while trial < limit:
            trial += 1
            sample = rng.choice(len(X), 4, replace=False)
            try:
                solved = cv2.solvePnPGeneric(X[sample], nq[sample], np.eye(3), None,
                                            flags=cv2.SOLVEPNP_AP3P)
            except cv2.error:
                continue
            if not solved[0]:
                continue
            for rv, tv in zip(solved[1], solved[2]):
                R0, t0 = cv2.Rodrigues(rv)[0], tv.ravel()
                Y = X @ R0.T + t0
                err = np.linalg.norm(cam.project(Y) - uv, axis=1)
                mask = np.isfinite(err) & (err <= cfg.ransac_px) & (Y[:, 2] > 1e-6)
                score = (int(mask.sum()), -float(np.minimum(np.nan_to_num(err, nan=1e6), cfg.ransac_px).sum()))
                if score > best_score:
                    best, best_mask, best_score = (R0, t0), mask, score
                    fraction = score[0] / len(X)
                    if fraction == 1:
                        limit = trial
                    elif fraction > 0:
                        limit = min(limit, max(trial, int(np.ceil(np.log(.001) / np.log1p(-fraction ** 4)))))
        if best is None or best_score[0] < cfg.min_inliers:
            return None
        X, S, uv = X[best_mask], S[best_mask], uv[best_mask]
        R, t = best

        def linearize(R, t):
            Y = X @ R.T + t
            if not np.isfinite(Y).all() or np.any(Y[:, 2] <= 1e-6):
                return None
            residual = cam.project(Y) - uv
            Jp = np.empty((len(X), 2, 3))
            eps = 1e-6 * np.maximum(1., np.linalg.norm(Y, axis=1))
            for j in range(3):
                step = np.zeros_like(Y)
                step[:, j] = eps
                Jp[:, :, j] = (cam.project(Y + step) - cam.project(Y - step)) / (2 * eps[:, None])
            JX = Jp @ R
            J = np.concatenate([JX, -np.einsum("nij,njk->nik", JX, _skew_n(X))], 2)
            C = cfg.pixel_std ** 2 * np.eye(2)[None] + np.einsum("nij,njk,nlk->nil", JX, S, JX)
            if not np.isfinite(J).all() or not np.isfinite(C).all():
                return None
            try:
                W = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                return None
            A = np.einsum("nji,njk,nkl->il", J, W, J)
            b = np.einsum("nji,njk,nk->i", J, W, residual)
            return residual, W, A, b

        converged = False
        for _ in range(20):
            lin = linearize(R, t)
            if lin is None:
                return None
            residual, W, A, b = lin
            if not np.isfinite(A).all() or np.linalg.cond(A) > 1e12:
                return None
            cost = float(np.einsum("ni,nij,nj->", residual, W, residual))
            accepted = False
            for damping in (0., 1e-6, 1e-4, .01, 1., 100.):
                try:
                    dx = -np.linalg.solve(A + damping * np.diag(np.maximum(np.diag(A), 1e-12)), b)
                except np.linalg.LinAlgError:
                    continue
                if np.linalg.norm(dx) < 1e-9:
                    converged = True
                    break
                R1 = R @ Rot.from_rotvec(dx[3:]).as_matrix()
                t1 = t + R @ dx[:3]
                Y1 = X @ R1.T + t1
                if np.any(Y1[:, 2] <= 1e-6):
                    continue
                r1 = cam.project(Y1) - uv
                cost1 = float(np.einsum("ni,nij,nj->", r1, W, r1))
                if np.isfinite(cost1) and cost1 < cost:
                    R, t = R1, t1
                    accepted = True
                    converged = (cost - cost1) <= 1e-8 * max(cost, 1.)
                    break
            if converged:
                break
            if not accepted:
                return None
        if not converged:
            return None
        lin = linearize(R, t)  # final-state Hessian, weights and cheirality
        if lin is None:
            return None
        residual, W, A, _ = lin
        cost = float(np.einsum("ni,nij,nj->", residual, W, residual))
        try:
            cov = np.linalg.inv(A) * max(cost / max(2 * len(X) - 6, 1), 1.)
        except np.linalg.LinAlgError:
            return None
        cov = .5 * (cov + cov.T)
        if not valid_covariance(cov, 6):
            return None
        return _se3(R, t), cov, len(X), float(np.sqrt(np.mean(np.sum(residual ** 2, 1))))

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
        if not np.isfinite(t) or (self.last_track_t is not None and t <= self.last_track_t):
            raise ValueError("frame timestamps must increase and be finite")
        if self.last_track_t is not None and t - self.last_track_t > cfg.max_frame_gap_s:
            self.prev = None
        self.stats["frames"] += 1
        # Stale-frame check (camera health from the images alone): a real sensor never delivers two
        # bit-identical consecutive exposures, because shot and read noise change every pixel. A
        # repeated frame (driver hang, frozen stream) would otherwise read as "no motion" with perfect
        # tracks, a confidently wrong measurement. It is treated as a dropped frame.
        last, self.last_raw = self.last_raw, img0
        if last is not None and last.shape == img0.shape and                 float(np.mean(np.abs(img0.astype(np.int16) - last.astype(np.int16)))) < cfg.stale_mad:
            self.stats["stale"] += 1
            self.last_raw = last
            return []
        self.last_track_t = float(t)
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
                    elapsed = t - self.keyframe_t
                    if not valid_covariance(cov6, 6):
                        self.stats["fail_pnp"] += 1
                    elif ang > cfg.max_rot_rad * elapsed / .1 or np.linalg.norm(dp) > cfg.max_trans_m * elapsed / .1:
                        self.stats["fail_motion"] += 1
                    else:
                        health = .005 + .995 * float(np.clip((n_in - cfg.min_inliers) / (cfg.good_inliers - cfg.min_inliers), 0, 1))
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
                self.keyframe_t = float(t)
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
