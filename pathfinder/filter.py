"""Error-state Kalman filter that fuses IMU, camera fixes and learned displacement.

Nominal state: rotation R (body->world), velocity v, position p, gyro bias bg, accel bias ba.
Error state (15): [dtheta, dv, dp, dbg, dba], with R_true = R * Exp(dtheta).

Sources of trust:
  * IMU: process noise grows when the voting layer is unsure (``imu_conf`` < 1).
  * Camera fix: noise scales with 1 / health, frames below ``min_health`` are skipped.
  * Learned displacement: the network's own predicted covariance.
  * Every update passes a chi-square gate (Mahalanobis distance), so a confident but wrong
    measurement (camera locked on the wrong thing) is rejected instead of pulling the path.

Simplification: the learned displacement d = p(t) - p(t - w) is applied as a measurement on
the current position with the past position held at its estimate (TLIO uses stochastic
cloning for this; we inflate the noise by the past position covariance instead).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import GRAVITY

G = np.array([0.0, 0.0, -GRAVITY])
CHI2_3DOF_999 = 16.27


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@dataclass
class FilterConfig:
    sigma_g: float = 0.005        # gyro noise, rad/s/sqrt(Hz)-ish per step scaling
    sigma_a: float = 0.08         # accel noise
    sigma_bg: float = 1e-4        # bias random walk
    sigma_ba: float = 1e-3
    min_health: float = 0.005     # only frames with essentially nothing visible are ignored
    recover_health: float = 0.6   # frames this healthy can re-anchor a drifted estimate
    recover_after: int = 5        # ... after this many consecutive rejections (0.5 s)
    use_health: bool = True       # off = naive fusion that trusts every frame equally
    use_gate: bool = True
    gate: float = CHI2_3DOF_999
    fix_std: float = 0.03
    low_conf_inflate: float = 25.0   # process-noise multiplier when IMU confidence is 0


@dataclass
class FilterLog:
    p: list = field(default_factory=list)
    p_std: list = field(default_factory=list)
    cam_used: int = 0
    cam_rejected: int = 0
    cam_skipped: int = 0
    cam_recovered: int = 0
    disp_used: int = 0
    disp_rejected: int = 0


class ESKF:
    def __init__(self, cfg: FilterConfig, p0, v0, q0, P0_scale: float = 1e-4):
        self.cfg = cfg
        self.R = Rot.from_quat(q0)
        self.v = np.array(v0, float)
        self.p = np.array(p0, float)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.eye(15) * P0_scale
        self.P[9:12, 9:12] = np.eye(3) * 1e-4
        self.P[12:15, 12:15] = np.eye(3) * 1e-2

    def propagate(self, gyro, acc, dt, imu_conf: float = 1.0):
        w = gyro - self.bg
        f = acc - self.ba
        Rm = self.R.as_matrix()
        a = Rm @ f + G
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.R = self.R * Rot.from_rotvec(w * dt)

        F = np.eye(15)
        F[0:3, 0:3] -= skew(w) * dt
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 0:3] = -Rm @ skew(f) * dt
        F[3:6, 12:15] = -Rm * dt
        F[6:9, 3:6] = np.eye(3) * dt
        c = self.cfg
        k = 1.0 + (c.low_conf_inflate - 1.0) * (1.0 - np.clip(imu_conf, 0, 1))
        Q = np.zeros(15)
        Q[0:3] = (c.sigma_g * k) ** 2 * dt
        Q[3:6] = (c.sigma_a * k) ** 2 * dt
        Q[9:12] = c.sigma_bg ** 2 * dt
        Q[12:15] = c.sigma_ba ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag(Q)

    def _update_position(self, z, Rmeas):
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        y = z - self.p
        S = H @ self.P @ H.T + Rmeas
        nis = float(y @ np.linalg.solve(S, y))
        if self.cfg.use_gate and nis > self.cfg.gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rmeas @ K.T
        self.R = self.R * Rot.from_rotvec(dx[0:3])
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        return True

    def camera_update(self, fix, health, log: FilterLog):
        c = self.cfg
        if c.use_health:
            if health < c.min_health:
                log.cam_skipped += 1
                return
            # Trust falls off fast with health: a frame at 0.1 counts ~100x less than a clean one.
            std = c.fix_std / max(health, 0.02)
        else:
            std = c.fix_std
        Rm = np.eye(3) * std ** 2
        if self._update_position(fix, Rm):
            log.cam_used += 1
            self.cam_fail_streak = 0
            return
        log.cam_rejected += 1
        self.cam_fail_streak = getattr(self, "cam_fail_streak", 0) + 1
        # Gate lock-out guard: if *healthy* frames keep disagreeing for a while, our own
        # estimate is the thing that drifted. Re-anchor to the camera, cautiously.
        if health >= c.recover_health and self.cam_fail_streak >= c.recover_after:
            gate, self.cfg.use_gate = self.cfg.use_gate, False
            self._update_position(fix, Rm * 4)
            self.cfg.use_gate = gate
            self.cam_fail_streak = 0
            log.cam_recovered += 1

    def displacement_update(self, d, cov, p_past, P_past_pos, log: FilterLog):
        z = p_past + d
        if self._update_position(z, cov + P_past_pos):
            log.disp_used += 1
        else:
            log.disp_rejected += 1


def run(seq, gyro, acc, imu_conf=None, cam=None, disp=None, cfg: FilterConfig = FilterConfig(),
        start: int = 0):
    """Run the filter over a sequence.

    gyro, acc: (N,3) fused IMU stream. imu_conf: (N,) or None.
    cam: dict from camera.visual_fixes (idx, fix, health) or None.
    disp: dict with idx (end sample), window (samples), d (M,3), cov (M,3,3) or None.
    """
    N = len(seq.t)
    dt = seq.dt
    f = ESKF(cfg, seq.p[start], seq.v[start], seq.q[start])
    log = FilterLog()
    cam_at = {} if cam is None else {int(i): j for j, i in enumerate(cam["idx"])}
    disp_at = {} if disp is None else {int(i): j for j, i in enumerate(disp["idx"])}
    hist_p = np.zeros((N, 3))
    hist_Pp = np.zeros((N, 3, 3))
    out = np.zeros((N, 3))
    for k in range(start, N):
        out[k] = f.p
        hist_p[k] = f.p
        hist_Pp[k] = f.P[6:9, 6:9]
        log.p_std.append(float(np.sqrt(np.trace(f.P[6:9, 6:9]))))
        if k in cam_at:
            j = cam_at[k]
            if not np.any(np.isnan(cam["fix"][j])):
                f.camera_update(cam["fix"][j], cam["health"][j], log)
        if k in disp_at:
            j = disp_at[k]
            k0 = k - disp["window"]
            if k0 >= start:
                f.displacement_update(disp["d"][j], disp["cov"][j], hist_p[k0], hist_Pp[k0], log)
        if k < N - 1:
            c = 1.0 if imu_conf is None else float(imu_conf[k])
            f.propagate(gyro[k], acc[k], dt, c)
    out[:start] = seq.p[:start]
    return out, log
