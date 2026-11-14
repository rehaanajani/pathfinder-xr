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
    recover_after: int = 10       # after an outage: re-anchor after 1 s of healthy disagreement
    recover_after_jump: int = 100 # after a sudden camera jump: hold out for 10 s
    outage_gap: int = 100         # samples (0.5 s) without a good frame counts as an outage
    model_check_up: float = 0.1     # learned-model self-check: how fast trust drops (per update)
    model_check_down: float = 0.003 # ... and how slowly it comes back
    use_health: bool = True       # off = naive fusion that trusts every frame equally
    trust: tuple | None = None    # (health, std_m) table from camera.calibrate_trust
    cloning: bool = False         # TLIO-style stochastic cloning (mixed results, see README)
    adaptive_q: bool = False      # inflate IMU process noise when healthy camera disagrees
    q_up: float = 0.2
    q_down: float = 0.01
    q_max: float = 400.0
    bias_q_gain: float = 1.0
    use_gate: bool = True
    gate: float = CHI2_3DOF_999
    gate_mode: str = "soft"       # "soft" = robust down-weighting, "hard" = reject + re-anchor
    soft_power: float = 2.0
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
    disp_std: list = field(default_factory=list)
    model_scale: list = field(default_factory=list)


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
        self.clone_k: list[int] = []      # stochastic cloning: past positions kept in the state
        self.clone_p: list[np.ndarray] = []

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
        # Adaptive process noise (Mehra 1970 style): q_scale grows when healthy camera fixes keep
        # disagreeing with the IMU prediction more than the filter expected.
        q = getattr(self, "q_scale", 1.0) if c.adaptive_q else 1.0
        Q[0:3] = (c.sigma_g * k) ** 2 * dt * q
        Q[3:6] = (c.sigma_a * k) ** 2 * dt * q
        Q[9:12] = c.sigma_bg ** 2 * dt * q * c.bias_q_gain
        Q[12:15] = c.sigma_ba ** 2 * dt * q * c.bias_q_gain
        P = self.P
        P[:15, :15] = F @ P[:15, :15] @ F.T + np.diag(Q)
        if P.shape[0] > 15:                       # clones are static; only cross terms move
            P[:15, 15:] = F @ P[:15, 15:]
            P[15:, :15] = P[:15, 15:].T

    def _update(self, y, H, Rmeas):
        n = self.P.shape[0]
        S = H @ self.P @ H.T + Rmeas
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        accepted = True
        if self.cfg.use_gate and nis > self.cfg.gate:
            if self.cfg.gate_mode == "hard":
                return False
            # Soft gate (robust Kalman): inflate the measurement noise so its pull is capped,
            # growing with how surprising it is. Far-off measurements barely move the state.
            Rmeas = Rmeas * (nis / self.cfg.gate) ** self.cfg.soft_power
            S = H @ self.P @ H.T + Rmeas
            accepted = False
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rmeas @ K.T
        self.R = self.R * Rot.from_rotvec(dx[0:3])
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        for i in range(len(self.clone_p)):
            self.clone_p[i] = self.clone_p[i] + dx[15 + 3 * i:18 + 3 * i]
        return accepted

    def _update_position(self, z, Rmeas):
        H = np.zeros((3, self.P.shape[0]))
        H[:, 6:9] = np.eye(3)
        return self._update(z - self.p, H, Rmeas)

    def add_clone(self, k: int, max_clones: int):
        """Stochastic cloning (Roumeliotis & Burdick 2002): copy the current position into the state."""
        n = self.P.shape[0]
        J = np.zeros((n + 3, n))
        J[:n, :n] = np.eye(n)
        J[n:, 6:9] = np.eye(3)
        self.P = J @ self.P @ J.T
        self.clone_k.append(k)
        self.clone_p.append(self.p.copy())
        while len(self.clone_k) > max_clones:
            keep = np.r_[0:15, 18:self.P.shape[0]]
            self.P = self.P[np.ix_(keep, keep)]
            self.clone_k.pop(0)
            self.clone_p.pop(0)

    def displacement_update_clone(self, d, cov, k0, log, k: int = 0):
        """Learned displacement d = p(k) - p(k0), with p(k0) a clone in the state (as in TLIO)."""
        if k0 not in self.clone_k:
            return
        i = self.clone_k.index(k0)
        n = self.P.shape[0]
        H = np.zeros((3, n))
        H[:, 6:9] = np.eye(3)
        H[:, 15 + 3 * i:18 + 3 * i] = -np.eye(3)
        y = d - (self.p - self.clone_p[i])
        anchored = k - getattr(self, "last_good_cam_k", -10 ** 9) <= self.cfg.outage_gap
        if anchored:
            S = H @ self.P @ H.T + cov
            ratio = float(y @ np.linalg.solve(S, y)) / 3.0
            cur = getattr(self, "model_scale", 1.0)
            target = max(ratio, 1.0)
            a = self.cfg.model_check_up if target > cur else self.cfg.model_check_down
            self.model_scale = (1 - a) * cur + a * target
        cov = cov * getattr(self, "model_scale", 1.0)
        if self._update(y, H, cov):
            log.disp_used += 1
        else:
            log.disp_rejected += 1

    def camera_update(self, fix, health, log: FilterLog, k: int = 0):
        c = self.cfg
        if c.use_health:
            if c.trust is not None:
                # Calibrated on training data: measured fix error for this health level.
                std = float(np.interp(health, c.trust[0], c.trust[1]))
            else:
                if health < c.min_health:
                    log.cam_skipped += 1
                    return
                std = c.fix_std / max(health, 0.02)
        else:
            std = c.fix_std
        Rm = np.eye(3) * std ** 2
        healthy = health >= c.recover_health
        if c.adaptive_q and healthy:
            # Use the innovation of healthy frames to check whether the IMU model is too optimistic.
            H = np.zeros((3, 15)); H[:, 6:9] = np.eye(3)
            y = fix - self.p
            S = H @ self.P @ H.T + Rm
            ratio = float(y @ np.linalg.solve(S, y)) / 3.0
            cur = getattr(self, "q_scale", 1.0)
            target = float(np.clip(ratio, 1.0, c.q_max))
            a = c.q_up if target > cur else c.q_down
            self.q_scale = (1 - a) * cur + a * target
        if self._update_position(fix, Rm):
            log.cam_used += 1
            self.cam_fail_streak = 0
            if healthy:
                self.last_good_cam_k = k
            return
        log.cam_rejected += 1
        if not healthy:
            return
        streak = getattr(self, "cam_fail_streak", 0)
        if streak == 0:
            # Why does a healthy camera disagree with us? If we just came back from an outage,
            # we are the one that drifted. If the camera was agreeing a moment ago and suddenly
            # jumped, it has probably locked onto the wrong thing.
            gap = k - getattr(self, "last_good_cam_k", -10 ** 9)
            self.streak_origin = "outage" if gap > c.outage_gap else "jump"
        self.cam_fail_streak = streak + 1
        needed = c.recover_after if self.streak_origin == "outage" else c.recover_after_jump
        if self.cam_fail_streak >= needed:
            # Re-anchor: open our position uncertainty to the size of the disagreement, then
            # take the camera fix with the gate off.
            y = fix - self.p
            self.P[6:9, 6:9] += np.eye(3) * float(y @ y)
            gate, self.cfg.use_gate = self.cfg.use_gate, False
            self._update_position(fix, Rm)
            self.cfg.use_gate = gate
            self.cam_fail_streak = 0
            self.last_good_cam_k = k
            log.cam_recovered += 1

    def displacement_update(self, d, cov, p_past, P_past_pos, log: FilterLog, k: int = 0):
        z = p_past + d
        # Self-check of the learned model: while the camera is healthy our state is anchored,
        # so a large innovation here means the *model* is wrong on this kind of motion. Track
        # how far off it is relative to its own claimed uncertainty and scale its covariance.
        anchored = k - getattr(self, "last_good_cam_k", -10 ** 9) <= self.cfg.outage_gap
        if anchored:
            y = z - self.p
            S = self.P[6:9, 6:9] + cov + P_past_pos
            ratio = float(y @ np.linalg.solve(S, y)) / 3.0
            cur = getattr(self, "model_scale", 1.0)
            target = max(ratio, 1.0)
            # Lose trust fast, regain it slowly: one bad stretch should be remembered.
            a = self.cfg.model_check_up if target > cur else self.cfg.model_check_down
            self.model_scale = (1 - a) * cur + a * target
        cov = cov * getattr(self, "model_scale", 1.0)
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
    disp_at = {} if disp is None or callable(disp) else {int(i): j for j, i in enumerate(disp["idx"])}
    predictor = disp if callable(disp) else None
    hist_p = np.zeros((N, 3))
    hist_Pp = np.zeros((N, 3, 3))
    hist_R = np.zeros((N, 3, 3))
    out = np.zeros((N, 3))
    W = getattr(predictor, "window", 200)
    every = getattr(predictor, "every", 20)
    for k in range(start, N):
        out[k] = f.p
        hist_p[k] = f.p
        hist_Pp[k] = f.P[6:9, 6:9]
        hist_R[k] = f.R.as_matrix()
        log.p_std.append(float(np.sqrt(np.trace(f.P[6:9, 6:9]))))
        if k in cam_at:
            j = cam_at[k]
            if not np.any(np.isnan(cam["fix"][j])):
                f.camera_update(cam["fix"][j], cam["health"][j], log, k)
        if k in disp_at:
            j = disp_at[k]
            k0 = k - disp["window"]
            if k0 >= start:
                f.displacement_update(disp["d"][j], disp["cov"][j], hist_p[k0], hist_Pp[k0], log)
        if predictor is not None and k - start >= W and (k - start) % every == 0:
            k0 = k - W
            # Gravity-align the last second with the filter's OWN orientation history.
            Rs = hist_R[k0:k]
            gw = np.einsum("nij,nj->ni", Rs, gyro[k0:k])
            aw = np.einsum("nij,nj->ni", Rs, acc[k0:k])
            d, cov = predictor(np.concatenate([gw, aw], 1))
            log.disp_std.append(float(np.sqrt(np.trace(cov))))
            if cfg.cloning:
                f.displacement_update_clone(d, cov, k0, log, k)
            else:
                f.displacement_update(d, cov, hist_p[k0], hist_Pp[k0], log, k)
            log.model_scale.append(getattr(f, "model_scale", 1.0))
        if predictor is not None and cfg.cloning and (k - start) % every == 0:
            f.add_clone(k, W // every + 1)
        if k < N - 1:
            c = 1.0 if imu_conf is None else float(imu_conf[k])
            f.propagate(gyro[k], acc[k], dt, c)
    out[:start] = seq.p[:start]
    return out, log
