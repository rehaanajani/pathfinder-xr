"""Error-state Kalman filter that fuses IMU, camera fixes and learned displacement.

Nominal state: rotation R (body->world), velocity v, position p, gyro bias bg, accel bias ba.
Error state (15): [dtheta, dv, dp, dbg, dba], with R_true = R * Exp(dtheta).

Sources of trust:
  * IMU: process noise grows when the voting layer is unsure (``imu_conf`` < 1).
  * Camera fix: noise scales with 1 / health, frames below ``min_health`` are skipped.
  * Learned displacement: the network's own predicted covariance.
  * Every update passes a chi-square gate (Mahalanobis distance), so a confident but wrong
    measurement (camera locked on the wrong thing) is rejected instead of pulling the path.

Learned displacement d = p(t) - p(t - w), two forms:
  * default: a measurement on the current position with the past position held at its
    estimate (noise inflated by the past position covariance);
  * ``cloning=True, yaw_frame=True``: TLIO's form. Past poses are cloned into the state, the
    network sees the window in the clone's yaw frame, and the measurement
    h = Rz(yaw_i)^T (p - p_i) carries a Jacobian on the clone's yaw, so the network corrects
    heading drift instead of only position.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import GRAVITY

G = np.array([0.0, 0.0, -GRAVITY])
CHI2_3DOF_999 = 16.27
CHI2_6DOF_999 = 22.46


def saturation_mask(gyro, acc, rate=200.0, flat_tol=1e-6, top=0.95, win_s=2.0):
    """Per sample: is any IMU axis pinned flat at the top of its recent range (clipped)?

    A saturated axis repeats the same full-scale value; healthy MEMS noise never repeats exactly.
    A sample counts when it equals its predecessor (within ``flat_tol``) AND its magnitude is at
    least ``top`` x the largest magnitude of that axis over the last ``win_s`` seconds (causal),
    and not every axis is frozen (that is a dropped packet held at its last value, not clipping)."""
    x = np.hstack([np.asarray(gyro, float), np.asarray(acc, float)])
    n = len(x)
    flat = np.zeros_like(x, bool)
    flat[1:] = np.abs(np.diff(x, axis=0)) <= flat_tol * np.maximum(1.0, np.abs(x[1:]))
    w = max(1, int(win_s * rate))
    from scipy.ndimage import maximum_filter1d
    a = np.abs(x)
    # trailing window; zero padding (not reflect) so the first 2 s never see later samples (causal, #16)
    runmax = maximum_filter1d(a, size=w, axis=0, origin=(w - 1) // 2, mode="constant", cval=0.0)
    near_top = a >= top * np.maximum(runmax, 1e-9)
    # Dropped packets repeat the whole sample (all six axes frozen); clipping freezes only the
    # clipped axes while the others keep moving. Only the latter counts as saturation.
    hit = (flat & near_top & (a > 1e-6)).any(1) & ~flat.all(1)
    # A single flat pair can happen by chance at low resolution; need 3 in a row.
    run3 = np.zeros(n, bool)
    run3[2:] = hit[2:] & hit[1:-1] & hit[:-2]
    return run3


class CausalSaturation:
    """One-sample equivalent of ``saturation_mask(...)[-1]``.

    The streaming tracker used to stack the last ``int(win_s * rate) + 3`` samples and
    call ``saturation_mask`` on that whole block at every IMU sample. The flag only
    depends on a trailing ``win_s`` maximum plus the two previous hits, so this keeps
    the same horizon and the same comparisons without rebuilding the block.
    """

    def __init__(self, rate, win_s=2.0, flat_tol=1e-6, top=0.95):
        self.w = max(1, int(win_s * float(rate)))
        self.cap = int(win_s * float(rate)) + 3
        if self.cap < self.w:
            self.cap = self.w
        if self.cap < 3:
            self.cap = 3
        self.flat_tol = float(flat_tol)
        self.top = float(top)
        self.buf = np.zeros((self.cap, 6))
        self.n = 0
        self.pos = 0
        self._hit_prev = False
        self._hit_prev2 = False
        # Decreasing deques of (sample index, |axis|): the front is the trailing-window max.
        self._mx = [deque() for _ in range(6)]
        self._rm = np.zeros(6)

    def push(self, gyro, acc):
        cur = self.buf[self.pos]
        cur[:3] = gyro
        cur[3:] = acc
        self.pos = (self.pos + 1) % self.cap
        self.n += 1
        rm = self._roll_max(cur)
        if self.n < 2:
            hit = False
        else:
            prev = self.buf[(self.pos - 2) % self.cap]
            flat = np.abs(cur - prev) <= self.flat_tol * np.maximum(1.0, np.abs(cur))
            av = np.abs(cur)
            near = av >= self.top * np.maximum(rm, 1e-9)
            hit = bool((flat & near & (av > 1e-6)).any() and not flat.all())
        flagged = bool(hit and self._hit_prev and self._hit_prev2) if self.n >= 3 else False
        self._hit_prev2 = self._hit_prev
        self._hit_prev = hit
        return flagged

    def _roll_max(self, cur):
        """Trailing maximum of |sample| over ``w`` samples, equal to ``maximum_filter1d``."""
        limit = self.n - self.w
        rm = self._rm
        for axis in range(6):
            value = float(abs(cur[axis]))
            window = self._mx[axis]
            while window and window[0][0] <= limit:
                window.popleft()
            while window and window[-1][1] <= value:
                window.pop()
            window.append((self.n, value))
            rm[axis] = window[0][1]
        return rm


IMU_GYRO_MAX = 100.0      # rad/s; beyond any consumer IMU's range: the reading is garbage
IMU_ACC_MAX = 1000.0      # m/s^2 (~100 g)
T_EPS = 1e-6              # s; time comparisons tolerate float noise so offline dt and 1/rate agree (#17)


class ImuCleaner:
    """One IMU sample at a time: non-finite, all-zero or impossibly large readings are replaced by the
    last good sample (zeros if there was none yet) and flagged. Used by both ``run`` and the streaming
    tracker, so the two stay identical on bad input (#13)."""

    def __init__(self):
        self.last = None

    def __call__(self, gyro, acc):
        g, a = np.asarray(gyro, float), np.asarray(acc, float)
        ok = bool(np.isfinite(g).all() and np.isfinite(a).all() and (g.any() or a.any())
                  and np.linalg.norm(g) <= IMU_GYRO_MAX and np.linalg.norm(a) <= IMU_ACC_MAX)
        if ok:
            self.last = (g, a)
            return g, a, True, True
        if self.last is not None:
            return self.last[0], self.last[1], False, True
        return np.zeros(3), np.zeros(3), False, False


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
    # Camera recovery, in seconds (independent of frame rate and dropped frames):
    outage_gap_s: float = 2.0     # this long without an accepted healthy frame counts as an outage
                                  # (shorter gaps, e.g. dropped frames, cannot explain a big jump)
    recover_after_s: float = 1.0  # after an outage: re-anchor once a healthy camera has disagreed
                                  # steadily for this long (we are the one that drifted)
    max_holdout_s: float = 30.0   # after a sudden jump while tracking well: reject the camera until
                                  # our own uncertainty makes it plausible, or at most this long
    min_recover_frames: int = 3   # and never on fewer frames than this
    steady_ratio: float = 0.5     # "steady": spread of the disagreement < this x its size
    model_check_up: float = 0.1     # learned-model self-check: how fast trust drops (per update)
    model_check_down: float = 0.003 # ... and how slowly it comes back
    use_health: bool = True       # off = naive fusion that trusts every frame equally
    trust: tuple | None = None    # (health, std_m) table from camera.calibrate_trust
    cloning: bool = False         # TLIO-style stochastic cloning of past (rotation, position)
    disp_cov_scale: float = 1.0   # calibration of the network's claimed variance (>1: it is overconfident)
    # Outage speed limit (issue #2 follow-up): while no camera/VO update has been accepted for
    # ``vis_gap_s``, a person's head cannot move faster than ``v_max``; with a broken IMU the
    # integrated velocity otherwise runs away (saturation, drift). Optionally also a weak
    # zero-velocity prior that tightens as IMU confidence falls.
    speed_limit: bool = False
    v_max: float = 2.0            # m/s: TLIO train split head speed 99.9th pct 1.62, max 2.13
    v_limit_std: float = 0.3      # m/s, how firmly the limit is applied
    outage_v_std: float = 0.0     # m/s; 0 = off. Zero-velocity prior std at IMU confidence 1
    vis_gap_s: float = 0.5
    # Long outages (phone test: 32 s blind -> 50 m off while claiming +-1.2 m):
    outage_tighten_s: float = 0.0  # >0: after this long blind, tighten the zero-velocity prior ...
    outage_tau_s: float = 3.0      # ... with this time constant ...
    outage_v_floor: float = 0.3    # ... down to this std (m/s): someone who can't be seen mostly stays put
    outage_std_rate: float = 0.0   # >0 (m/s per axis): during an outage the REPORTED position
                                   # uncertainty is never below rate x time blind (reporting only:
                                   # inflating the filter's own covariance hurt accuracy on validation)
    speed_limit_with_model: bool = False
    # Saturation guard: a clipped IMU reading sits flat at the top of its range. While that is seen
    # (from the data alone), keep the speed limit on even with the learned model, distrust the
    # model (its input is clipped), and widen IMU process noise.
    saturation_guard: bool = False
    sat_model_inflate: float = 100.0   # learned-update variance multiplier while saturated
    sat_imu_conf: float = 0.3          # IMU confidence cap while saturated
    sat_hold_s: float = 0.5            # keep the guard on this long after the last clipped sample  # validation: with the learned model running the limit did not
                                          # help (it already bounds velocity), so it switches itself off
    yaw_frame: bool = False       # with cloning: measure displacement in the clone's yaw frame,
                                  # with the rotation Jacobian (TLIO, Liu et al. 2020, eq. 9-10)
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
    low_conf_inflate: float = 10.0   # process-noise multiplier when IMU confidence is 0 (chosen on the
                                     # validation split from 1, 4, 10, 25; see results/diagnostics)


@dataclass
class FilterLog:
    p: list = field(default_factory=list)
    p_std: list = field(default_factory=list)
    p_cov: list = field(default_factory=list)      # full 3x3 position covariance per sample
    cam_used: int = 0
    cam_rejected: int = 0
    cam_skipped: int = 0
    cam_recovered: int = 0
    disp_used: int = 0
    disp_rejected: int = 0
    disp_std: list = field(default_factory=list)
    v_limited: int = 0
    sat_samples: int = 0
    v_prior: int = 0
    vo_used: int = 0
    vo_rejected: int = 0
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
        self.clone_k: list[int] = []      # stochastic cloning: past poses kept in the state
        self.clone_p: list[np.ndarray] = []
        self.clone_R: list = []           # error state per clone: [dtheta (3), dp (3)]

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

    def propagate_sample(self, gyro, acc, dt, imu_conf, ok, held):
        """Propagate with a cleaned sample: a bad one counts with zero confidence; with no good sample
        ever, coast (keep velocity and orientation, grow the uncertainty)."""
        if ok:
            self.propagate(gyro, acc, dt, imu_conf)
        elif held:
            self.propagate(gyro, acc, dt, 0.0)
        else:
            self.propagate(self.bg.copy(), self.ba + self.R.as_matrix().T @ -G, dt, 0.0)

    def _update(self, y, H, Rmeas, gate=None):
        n = self.P.shape[0]
        gate = self.cfg.gate if gate is None else gate
        S = H @ self.P @ H.T + Rmeas
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        accepted = True
        if self.cfg.use_gate and nis > gate:
            if self.cfg.gate_mode == "hard":
                return False
            # Soft gate (robust Kalman): inflate the measurement noise so its pull is capped,
            # growing with how surprising it is. Far-off measurements barely move the state.
            Rmeas = Rmeas * (nis / gate) ** self.cfg.soft_power
            S = H @ self.P @ H.T + Rmeas
            accepted = False
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rmeas @ K.T
        self.P = 0.5 * (self.P + self.P.T)          # keep it symmetric (long cloned runs drift)
        self.R = self.R * Rot.from_rotvec(dx[0:3])
        self.v += dx[3:6]
        self.p += dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        for i in range(len(self.clone_p)):
            o = 15 + 6 * i
            self.clone_R[i] = self.clone_R[i] * Rot.from_rotvec(dx[o:o + 3])
            self.clone_p[i] = self.clone_p[i] + dx[o + 3:o + 6]
        return accepted

    def _update_position(self, z, Rmeas):
        H = np.zeros((3, self.P.shape[0]))
        H[:, 6:9] = np.eye(3)
        return self._update(z - self.p, H, Rmeas)

    def add_clone(self, k: int, max_clones: int):
        """Stochastic cloning (Roumeliotis & Burdick 2002): copy the current pose into the state."""
        n = self.P.shape[0]
        J = np.zeros((n + 6, n))
        J[:n, :n] = np.eye(n)
        J[n:n + 3, 0:3] = np.eye(3)
        J[n + 3:, 6:9] = np.eye(3)
        self.P = J @ self.P @ J.T
        self.clone_k.append(k)
        self.clone_p.append(self.p.copy())
        self.clone_R.append(self.R)
        # Keep at most max_clones of this kind (int key: learned-displacement clone; ("vo", k): visual).
        kind = type(k)
        while sum(type(c) is kind for c in self.clone_k) > max_clones:
            self.drop_clone(next(i for i, c in enumerate(self.clone_k) if type(c) is kind))

    def drop_clone(self, i: int):
        o = 15 + 6 * i
        keep = np.r_[0:o, o + 6:self.P.shape[0]]
        self.P = self.P[np.ix_(keep, keep)]
        self.clone_k.pop(i)
        self.clone_p.pop(i)
        self.clone_R.pop(i)

    def vo_update(self, key_prev, t_rel, R_rel, cov_t, cov_r, log, k: int = 0):
        """Visual odometry between the clone taken at the previous frame and now (issue #2 B).

        Measured: pose of the IMU now relative to the IMU at the previous frame, in that frame:
          translation  h_t = R_i^T (p - p_i)       H: p -> R_i^T, p_i -> -R_i^T, dtheta_i -> [h_t]x
          rotation     h_R = R_i^T R               residual Log(h_R^T R_meas) ~ dtheta - h_R^T dtheta_i
        Six degrees of freedom, so the visual rotation also corrects heading and gyro bias."""
        if key_prev not in self.clone_k:
            return False
        self.release_outage_assumption(k)
        i = self.clone_k.index(key_prev)
        n = self.P.shape[0]
        o = 15 + 6 * i
        Ri = self.clone_R[i].as_matrix()
        Rc = self.R.as_matrix()
        h_t = Ri.T @ (self.p - self.clone_p[i])
        M = Ri.T @ Rc
        y = np.r_[t_rel - h_t, Rot.from_matrix(M.T @ R_rel).as_rotvec()]
        H = np.zeros((6, n))
        H[0:3, 6:9] = Ri.T
        H[0:3, o + 3:o + 6] = -Ri.T
        H[0:3, o:o + 3] = skew(h_t)
        H[3:6, 0:3] = np.eye(3)
        H[3:6, o:o + 3] = -M.T
        Rm = np.zeros((6, 6))
        Rm[:3, :3], Rm[3:, 3:] = cov_t, cov_r
        ok = self._update(y, H, Rm, gate=CHI2_6DOF_999)
        if ok:
            log.vo_used += 1
            self.last_vo_t = self._t(k)
        else:
            log.vo_rejected += 1
        return ok

    def clone_yaw(self, k0):
        """Yaw-only rotation (world <- local gravity-aligned frame) of the clone taken at sample k0."""
        Rm = self.clone_R[self.clone_k.index(k0)].as_matrix()
        g = np.arctan2(Rm[1, 0], Rm[0, 0])
        c, s_ = np.cos(g), np.sin(g)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def displacement_update_clone(self, d, cov, k0, log, k: int = 0):
        """Learned displacement d = p(k) - p(k0), with p(k0) a clone in the state (as in TLIO)."""
        if k0 not in self.clone_k:
            return
        i = self.clone_k.index(k0)
        n = self.P.shape[0]
        o = 15 + 6 * i
        H = np.zeros((3, n))
        if self.cfg.yaw_frame:
            # d is in the clone's local gravity-aligned frame: h = Rz(yaw_i)^T (p - p_i).
            # A yaw error dgamma at the clone gives dh = -[e_z]x h dgamma. With the clone's
            # rotation right-perturbed, R Exp(dtheta), and yaw = atan2(R10, R00):
            # dgamma/dtheta_j = (R00 dR10 - R10 dR00) / (R00^2 + R10^2), dR = R [e_j]x.
            A = self.clone_yaw(k0).T
            h = A @ (self.p - self.clone_p[i])
            ez = np.array([0.0, 0.0, 1.0])
            Rm = self.clone_R[i].as_matrix()
            dg = np.array([(Rm[0, 0] * dR[1, 0] - Rm[1, 0] * dR[0, 0]) for dR in
                           (Rm @ skew(e) for e in np.eye(3))]) / max(Rm[0, 0] ** 2 + Rm[1, 0] ** 2, 1e-9)
            H[:, o:o + 3] = np.outer(-skew(ez) @ h, dg)
        else:
            A = np.eye(3)
            h = self.p - self.clone_p[i]
        H[:, 6:9] = A
        H[:, o + 3:o + 6] = -A
        y = d - h
        anchored = self._anchored(k)
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

    def speed_check(self, k, imu_conf, log):
        """Velocity pseudo-measurements during a visual outage (see FilterConfig.speed_limit)."""
        c = self.cfg
        t = self._t(k)
        last = max(getattr(self, "last_good_cam_t", -1e9), getattr(self, "last_vo_t", -1e9))
        if t - last <= c.vis_gap_s + T_EPS:
            return
        n = self.P.shape[0]
        H = np.zeros((3, n)); H[:, 3:6] = np.eye(3)
        gate, self.cfg.use_gate = self.cfg.use_gate, False      # a physical bound, not a sensor reading
        try:
            sp = float(np.linalg.norm(self.v))
            if sp > c.v_max:
                self._update(self.v * (c.v_max / sp) - self.v, H, np.eye(3) * c.v_limit_std ** 2)
                log.v_limited += 1
            if c.outage_v_std > 0:
                std = c.outage_v_std * (0.25 + 0.75 * float(np.clip(imu_conf, 0, 1)))
                t_out = t - last
                if c.outage_tighten_s > 0 and t_out > c.outage_tighten_s:
                    std = max(c.outage_v_floor, std * np.exp(-(t_out - c.outage_tighten_s) / c.outage_tau_s))
                    self._assumed_still = True           # velocity is now an assumption, not a measurement
                self._update(-self.v, H, np.eye(3) * std ** 2)
                log.v_prior += 1
        finally:
            self.cfg.use_gate = gate

    def release_outage_assumption(self, k):
        """Back from a long outage: the velocity was held by the stand-still assumption, not measured.
        Re-open its uncertainty so the returning visual data sets it at once."""
        c = self.cfg
        if c.outage_tighten_s <= 0:
            return
        if getattr(self, "_assumed_still", False):
            self.P[3:6, 3:6] += np.eye(3) * c.v_max ** 2
            self._assumed_still = False                  # once per outage, only if the prior was used

    def outage_time(self, k):
        """Seconds since the last accepted visual update, or since the start if there was none (0 inside vis_gap_s)."""
        last = max(getattr(self, "last_good_cam_t", -1e9), getattr(self, "last_vo_t", -1e9))
        if last < -1e8:                                  # no visual fix yet: blind since the start (#15)
            last = self._t(getattr(self, "k_start", 0))
        t_out = self._t(k) - last
        return t_out if t_out > self.cfg.vis_gap_s + T_EPS else 0.0

    def _t(self, k):
        return k * getattr(self, "dt", 1.0 / 200.0)

    def _anchored(self, k):
        """A healthy camera fix was accepted recently, so our state is pinned to the world."""
        return self._t(k) - getattr(self, "last_good_cam_t", -1e9) <= self.cfg.outage_gap_s + T_EPS

    def camera_update(self, fix, health, log: FilterLog, k: int = 0):
        """Camera position fix with a small state machine (see ``self.cam_state``):

        tracking  - fixes agree; adaptive process noise may learn from healthy innovations.
        recovering- a healthy camera disagrees after an outage: we drifted. Adaptive noise may grow;
                    re-anchor once the disagreement has been steady for ``recover_after_s``.
        suspect   - a healthy camera jumped away while we were tracking well: probably locked onto
                    the wrong thing. Rejected outright and never allowed to inflate our noise, so it
                    cannot talk its way back in. It is accepted only when our own uncertainty has
                    grown enough to make it plausible (normal gate), or after ``max_holdout_s`` of
                    steady disagreement.
        """
        c = self.cfg
        t = self._t(k)
        if not (np.isfinite(fix).all() and np.isfinite(health)):     # #14: never let NaN/inf in
            log.cam_skipped += 1
            return
        health = float(np.clip(health, 0.0, 1.0))
        if health >= c.recover_health:
            self.release_outage_assumption(k)
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
        n = self.P.shape[0]
        H = np.zeros((3, n)); H[:, 6:9] = np.eye(3)
        y = fix - self.p
        S = H @ self.P @ H.T + Rm
        nis = float(y @ np.linalg.solve(S, y))
        state = getattr(self, "cam_state", "tracking")
        consistent = (not c.use_gate) or nis <= c.gate

        def adapt_q():
            if c.adaptive_q and healthy:
                # Innovations of trusted healthy frames tell us whether the IMU model is too optimistic.
                cur = getattr(self, "q_scale", 1.0)
                target = float(np.clip(nis / 3.0, 1.0, c.q_max))
                a = c.q_up if target > cur else c.q_down
                self.q_scale = (1 - a) * cur + a * target

        if consistent:
            adapt_q()
            self._update(y, H, Rm)
            log.cam_used += 1
            if healthy:
                self.last_good_cam_t = t
                self.cam_state, self.streak = "tracking", []
            return

        if not healthy:
            # A dim or blurred frame that disagrees: soft-gated (heavily down-weighted), no state change.
            if state != "suspect":
                self._update(y, H, Rm)
            log.cam_rejected += 1
            return

        # Healthy frame that disagrees with us. Why?
        if state == "tracking":
            outage = t - getattr(self, "last_good_cam_t", -1e9) > c.outage_gap_s + T_EPS
            self.cam_state = state = "recovering" if outage else "suspect"
            self.streak, self.streak_t0 = [], t
        self.streak.append(y.copy())
        log.cam_rejected += 1
        if state == "recovering":
            adapt_q()
            self._update(y, H, Rm)                   # soft-gated pull while we confirm
            wait = c.recover_after_s
        else:
            wait = c.max_holdout_s                   # suspect: no update, no noise inflation
        Y = np.array(self.streak)
        steady = np.linalg.norm(Y.std(0)) < c.steady_ratio * max(np.linalg.norm(Y.mean(0)), 1e-6)
        if t - self.streak_t0 >= wait - T_EPS and len(Y) >= c.min_recover_frames and steady:
            # Re-anchor: open our position uncertainty to the size of the disagreement, then take the
            # camera fix with the gate off.
            yy = fix - self.p
            self.P[6:9, 6:9] += np.eye(3) * float(yy @ yy)
            gate, self.cfg.use_gate = self.cfg.use_gate, False
            self._update_position(fix, Rm)
            self.cfg.use_gate = gate
            self.last_good_cam_t = t
            self.cam_state, self.streak = "tracking", []
            log.cam_recovered += 1

    def displacement_update(self, d, cov, p_past, P_past_pos, log: FilterLog, k: int = 0):
        z = p_past + d
        # Self-check of the learned model: while the camera is healthy our state is anchored,
        # so a large innovation here means the *model* is wrong on this kind of motion. Track
        # how far off it is relative to its own claimed uncertainty and scale its covariance.
        anchored = self._anchored(k)
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
        start: int = 0, vo=None):
    """Run the filter over a sequence.

    gyro, acc: (N,3) fused IMU stream. imu_conf: (N,) or None.
    cam: dict from camera.visual_fixes (idx, fix, health) or None.
    disp: dict with idx (end sample), window (samples), d (M,3), cov (M,3,3) or None.
    vo: dict from stereo visual odometry or None: frame_idx (sample of every processed frame),
        and per frame t (3), R (3,3), cov_t, cov_r (NaN t = no solve for that frame pair).
    """
    N = len(seq.t)
    dt = seq.dt
    f = ESKF(cfg, seq.p[start], seq.v[start], seq.q[start])
    f.dt = seq.dt
    f.k_start = start
    log = FilterLog()
    # Bad samples (#13): replaced by the last good one, propagated with zero confidence.
    gyro, acc = np.array(gyro, float), np.array(acc, float)
    clean = ImuCleaner()
    imu_ok, imu_held = np.ones(N, bool), np.ones(N, bool)
    for k in range(N):
        gyro[k], acc[k], imu_ok[k], imu_held[k] = clean(gyro[k], acc[k])
    cam_at = {} if cam is None else {int(i): j for j, i in enumerate(cam["idx"])}
    disp_at = {} if disp is None or callable(disp) else {int(i): j for j, i in enumerate(disp["idx"])}
    predictor = disp if callable(disp) else None
    vo_at = {} if vo is None else {int(i): j for j, i in enumerate(vo["frame_idx"])}
    hist_p = np.zeros((N, 3))
    hist_Pp = np.zeros((N, 3, 3))
    hist_R = np.zeros((N, 3, 3))
    out = np.zeros((N, 3))
    W = getattr(predictor, "window", 200)
    sat_recent = np.zeros(N, bool)
    if cfg.saturation_guard:
        sat = saturation_mask(gyro, acc, 1.0 / dt)
        hold = int(cfg.sat_hold_s / dt)
        idx = np.flatnonzero(sat)
        for i in idx:                                    # extend each hit forward by the hold time
            sat_recent[i:i + hold + 1] = True
        log.sat_samples = int(sat.sum())
    every = getattr(predictor, "every", 20)
    for k in range(start, N):
        out[k] = f.p
        hist_p[k] = f.p
        hist_Pp[k] = f.P[6:9, 6:9]
        hist_R[k] = f.R.as_matrix()
        log.p_std.append(float(np.sqrt(np.trace(f.P[6:9, 6:9]))))
        log.p_cov.append(f.P[6:9, 6:9].copy())
        if k in cam_at:
            j = cam_at[k]
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
            if cfg.cloning and cfg.yaw_frame and k0 in f.clone_k:
                Rz = f.clone_yaw(k0)              # into the clone's yaw frame (row vectors: v @ Rz)
                gw, aw = gw @ Rz, aw @ Rz
            d, cov = predictor(np.concatenate([gw, aw], 1))
            cov = cov * cfg.disp_cov_scale
            if cfg.saturation_guard and sat_recent[k0:k].any():
                cov = cov * cfg.sat_model_inflate
            log.disp_std.append(float(np.sqrt(np.trace(cov))))
            if cfg.cloning:
                f.displacement_update_clone(d, cov, k0, log, k)
            else:
                f.displacement_update(d, cov, hist_p[k0], hist_Pp[k0], log, k)
            log.model_scale.append(getattr(f, "model_scale", 1.0))
        if predictor is not None and cfg.cloning and (k - start) % every == 0:
            f.add_clone(k, W // every + 1)
        if k in vo_at:
            j = vo_at[k]
            if j > 0 and np.isfinite(vo["t"][j]).all():
                f.vo_update(("vo", int(vo["frame_idx"][j - 1])), vo["t"][j], vo["R"][j], vo["cov_t"][j],
                            vo["cov_r"][j], log, k)
            f.add_clone(("vo", k), 2)
        if cfg.speed_limit and (predictor is None or cfg.speed_limit_with_model or sat_recent[k]) \
                and (k - start) % 20 == 0:
            f.speed_check(k, 1.0 if imu_conf is None else float(imu_conf[k]), log)
        if k < N - 1:
            c = 1.0 if imu_conf is None else float(imu_conf[k])
            if sat_recent[k]:
                c = min(c, cfg.sat_imu_conf)
            f.propagate_sample(gyro[k], acc[k], dt, c, imu_ok[k], imu_held[k])
    out[:start] = seq.p[:start]
    return out, log
