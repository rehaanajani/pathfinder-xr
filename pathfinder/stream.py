"""Streaming Path Finder (issue #1 E / #3 E): push sensor data as it arrives, read the pose back.

    trk = StreamingTracker(cfg, p0, q0, rate=200, predictor=model)   # start pose: benchmark mode
    trk.push_camera_fix(fix, health)          # optional, any time: queued for the next IMU sample
    trk.push_vo(t_rel, R_rel, cov_t, cov_r)   # optional: motion since the previous push_vo call
    out = trk.push_imu(gyro, acc, conf=1.0)   # one IMU sample (rate Hz); returns the current output

``out`` follows OpenXR's semantics (XrSpaceLocationFlags) plus a graded signal OpenXR lacks:
    position, orientation (xyzw), velocity, position_cov (3x3), position_std_m
    tracked: a camera / visual-odometry update was accepted in the last ``vis_gap_s`` seconds
             (OpenXR POSITION_TRACKED_BIT)
    valid:   the position is finite and its uncertainty is below ``valid_std_m``
             (OpenXR POSITION_VALID_BIT: keep rendering with it, but it may be inferred)
    reason:  "ok" | "visual_outage" | "imu_saturated" | "imu_disagreement" | "too_uncertain"

Causal: every output uses only data pushed so far. It reproduces ``filter.run`` sample for sample
(tests/test_stream.py), so the offline benchmark numbers describe this live code path.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .filter import ESKF, FilterConfig, FilterLog, saturation_mask


@dataclass
class StreamConfig:
    valid_std_m: float = 1.0


class StreamingTracker:
    def __init__(self, cfg: FilterConfig, p0, q0, v0=(0.0, 0.0, 0.0), rate: float = 200.0,
                 predictor=None, scfg: StreamConfig = StreamConfig()):
        self.cfg, self.scfg, self.rate, self.dt = cfg, scfg, rate, 1.0 / rate
        self.f = ESKF(cfg, np.asarray(p0, float), np.asarray(v0, float), np.asarray(q0, float))
        self.f.dt = self.dt
        self.log = FilterLog()
        self.predictor = predictor
        self.W = getattr(predictor, "window", 200)
        self.every = getattr(predictor, "every", 20)
        n = max(self.W, int(2.0 * rate)) + 2
        self.hist = deque(maxlen=n)                 # (p, Pp, R, gyro, acc) per sample
        self.sat_hist = deque(maxlen=n)             # saturated-recently flag per sample
        self.raw = deque(maxlen=int(2.0 * rate) + 1)
        self.k = 0
        self.last_sat_k = -10 ** 9
        self.pending_cam, self.pending_vo = [], []
        self.last_vo_k = None
        self.out = None

    # ---- inputs -------------------------------------------------------------------------------
    def push_camera_fix(self, fix, health):
        self.pending_cam.append((np.asarray(fix, float), float(health)))

    def push_vo(self, t_rel, R_rel, cov_t, cov_r):
        """Relative pose since the previous push_vo call (None for t_rel = first frame / no solve)."""
        self.pending_vo.append((t_rel, R_rel, cov_t, cov_r))

    def push_imu(self, gyro, acc, conf: float = 1.0):
        f, cfg, k = self.f, self.cfg, self.k
        gyro, acc = np.asarray(gyro, float), np.asarray(acc, float)
        # Saturation: decided from the last 2 s of raw samples (same rule as the offline filter).
        self.raw.append(np.r_[gyro, acc])
        sat_now = False
        if cfg.saturation_guard and len(self.raw) >= 3:
            r = np.array(self.raw)
            sat_now = bool(saturation_mask(r[:, :3], r[:, 3:], self.rate)[-1])
        if sat_now:
            self.last_sat_k = k
        sat_recent = k - self.last_sat_k <= int(cfg.sat_hold_s / self.dt)

        out = self._snapshot(sat_recent, conf)
        self.hist.append((f.p.copy(), f.P[6:9, 6:9].copy(), f.R.as_matrix(), gyro, acc))
        self.sat_hist.append(sat_recent)
        self.log.p_std.append(out["position_std_m"])
        self.log.p_cov.append(out["position_cov"])

        for fix, health in self.pending_cam:
            if not np.any(np.isnan(fix)):
                f.camera_update(fix, health, self.log, k)
        self.pending_cam = []

        if self.predictor is not None and k >= self.W and k % self.every == 0:
            h = list(self.hist)[-self.W - 1:-1]                       # samples k-W .. k-1
            Rs = np.array([x[2] for x in h])
            g = np.array([x[3] for x in h])
            a = np.array([x[4] for x in h])
            gw, aw = np.einsum("nij,nj->ni", Rs, g), np.einsum("nij,nj->ni", Rs, a)
            k0 = k - self.W
            if cfg.cloning and cfg.yaw_frame and k0 in f.clone_k:
                Rz = f.clone_yaw(k0)
                gw, aw = gw @ Rz, aw @ Rz
            d, cov = self.predictor(np.concatenate([gw, aw], 1))
            cov = cov * cfg.disp_cov_scale
            if cfg.saturation_guard and any(list(self.sat_hist)[-self.W - 1:-1]):
                cov = cov * cfg.sat_model_inflate
            if cfg.cloning:
                f.displacement_update_clone(d, cov, k0, self.log, k)
            else:
                p0, P0 = list(self.hist)[-self.W - 1][0:2]
                f.displacement_update(d, cov, p0, P0, self.log, k)
        if self.predictor is not None and cfg.cloning and k % self.every == 0:
            f.add_clone(k, self.W // self.every + 1)

        for t_rel, R_rel, cov_t, cov_r in self.pending_vo:
            if self.last_vo_k is not None and t_rel is not None and np.isfinite(t_rel).all():
                f.vo_update(("vo", self.last_vo_k), t_rel, R_rel, cov_t, cov_r, self.log, k)
            f.add_clone(("vo", k), 2)
            self.last_vo_k = k
        self.pending_vo = []

        if cfg.speed_limit and (self.predictor is None or cfg.speed_limit_with_model or sat_recent) \
                and k % 20 == 0:
            f.speed_check(k, conf, self.log)
        c = min(conf, cfg.sat_imu_conf) if sat_recent else conf
        f.propagate(gyro, acc, self.dt, c)
        self.k += 1
        self.out = out
        return out

    # ---- output -------------------------------------------------------------------------------
    def _snapshot(self, sat_recent, conf):
        f, cfg = self.f, self.cfg
        P = f.P[6:9, 6:9].copy()
        std = float(np.sqrt(np.trace(P)))
        if cfg.outage_std_rate > 0:
            # Honest reporting in long outages: never claim less than rate x time blind per axis.
            std = max(std, np.sqrt(3.0) * cfg.outage_std_rate * f.outage_time(self.k))
        t = self.k * self.dt
        last_vis = max(getattr(f, "last_good_cam_t", -1e9), getattr(f, "last_vo_t", -1e9))
        tracked = t - last_vis <= cfg.vis_gap_s
        valid = bool(np.isfinite(f.p).all() and std < self.scfg.valid_std_m)
        if not valid:
            reason = "too_uncertain"
        elif sat_recent:
            reason = "imu_saturated"
        elif conf < 0.5:
            reason = "imu_disagreement"
        elif not tracked:
            reason = "visual_outage"
        else:
            reason = "ok"
        return {"t": t, "position": f.p.copy(), "orientation": f.R.as_quat(), "velocity": f.v.copy(),
                "position_cov": P, "position_std_m": std, "tracked": bool(tracked), "valid": valid,
                "reason": reason}
