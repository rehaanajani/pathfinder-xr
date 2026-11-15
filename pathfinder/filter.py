"""Error-state Kalman filter that fuses IMU, camera fixes and learned displacement.

Conventions (shared with ``sequence.py``): world frame z-up, gravity (0, 0, -9.81) m/s^2; IMU gyro in
rad/s and specific force f = R^T (a - g) in m/s^2, both in the body frame; R is body -> world.
Nominal state: rotation R, velocity v, position p, gyro bias bg, accel bias ba.
Error state (15): [dtheta, dv, dp, dbg, dba] with R_true = R Exp(dtheta) (right / body-frame
perturbation). Stochastic clones add [dtheta_i, dp_i] (6) per kept past pose.

Prediction holds each IMU sample constant until the next timestamp (variable dt is supported).
Updates use a Joseph-form covariance update, linear solves instead of explicit inverses, and the
error-reset Jacobian G = I - [dtheta/2]x after every rotation injection (Sola 2017, eq. 287).

Sources of trust
  * IMU: process noise = filter noise density x legacy confidence scaling + the extra density
    measured by the IMU monitor (``imu_health``): excess noise, unattributed disagreement, lost
    sensors. See that module for what is and is not observable.
  * Camera fix: noise from the calibrated health -> error table (or 1 / health). Every fix passes a
    chi-square gate on the innovation. A fix outside the gate is rejected outright (no "soft"
    partial pull: under a persistent wrong fix a soft pull drags the state until the fix looks
    consistent, which is how a wrong camera used to legitimise itself).
  * Learned displacement: the network's own covariance x a self-check scale x a calibration factor.

Camera supervision (all durations in seconds of measurement time, not frame counts)
  nominal        healthy fixes are being accepted
  degraded       fixes are accepted but with low health (large noise)
  outage         no accepted fix for more than ``outage_gap_s``
  suspect_jump   healthy fixes disagree while the filter was anchored a moment ago: the camera most
                 likely jumped (wrong match, moving object). Fixes are rejected for at least
                 ``recover_after_jump_s``.
  recovering     healthy fixes disagree after an outage: the inertial estimate most likely drifted.
                 Fixes are rejected for at least ``recover_after_s``.
In both disagreement episodes the filter only re-anchors to the camera when the episode's
innovations are mutually consistent (constant offset plus linear trend, chi-square test on the
residuals at ``consistency_alpha``), i.e. the camera is tracking *something* rigidly. A recovering
episode is downgraded to suspect_jump if its offset is larger than the inertial drift envelope
(``outage_envelope`` x the filter's position covariance) allows. Re-anchoring inflates position and
velocity covariance by the observed offset and trend before the update. None of this can tell a
rigid wrong fix that lasts longer than the hold-out from a genuine correction: there is no second
absolute position source. That case is reported (``cam_recovered`` events, ``reanchored`` reasons).

Adaptive process noise (``adaptive_q``, covariance matching after Mehra 1970) is fed only by
healthy fixes that *passed* the gate outside a disagreement episode, so a rejected fix can never
raise its own future influence. It is capped at ``q_max`` and decays at ``q_down``. It still cannot
distinguish slow visual drift inside the gate from IMU model error; see the docs.

Learned displacement d = p(t) - p(t - w), two forms:
  * default: a measurement on the current position with the past position held at its estimate
    (noise inflated by the past position covariance; cross-covariance ignored, so conservative);
  * ``cloning=True`` (optionally ``yaw_frame=True``, TLIO eq. 9-10): past poses are cloned into the
    state and h = Rz(yaw_i)^T (p - p_i) carries Jacobians on the clone.
Overlapping windows are correlated; ``disp_cov_scale`` is the validation-calibrated correction.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import chi2

from . import GRAVITY

G = np.array([0.0, 0.0, -GRAVITY])
CHI2_3DOF_999 = 16.27
CHI2_6DOF_999 = 22.458

CAM_NONE, CAM_NOMINAL, CAM_DEGRADED, CAM_OUTAGE, CAM_SUSPECT, CAM_RECOVERING = range(6)
CAM_STATE_NAMES = ("none", "nominal", "degraded", "outage", "suspect_jump", "recovering")


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def valid_covariance(value, size=3):
    """Reject malformed/indefinite measurement covariance instead of legitimising it by jitter."""
    value = np.asarray(value, float)
    if value.shape != (size, size) or not np.isfinite(value).all():
        return False
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-8):
        return False
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError:
        return False
    return True


@dataclass
class FilterConfig:
    sigma_g: float = 0.005        # gyro white-noise density used by the filter, rad/s/sqrt(Hz)
    sigma_a: float = 0.08         # accel white-noise density, m/s^2/sqrt(Hz)
    sigma_bg: float = 1e-4        # bias random walks
    sigma_ba: float = 1e-3
    min_health: float = 0.005     # only frames with essentially nothing visible are ignored
    recover_health: float = 0.6   # frames this healthy can start / end disagreement episodes
    recover_after_s: float = 1.0  # after an outage: re-anchor after 1 s of consistent disagreement
    recover_after_jump_s: float = 10.0  # after a sudden camera jump: hold out at least 10 s
    outage_gap_s: float = 0.5     # no accepted healthy fix for this long counts as an outage
    consistency_alpha: float = 1e-3   # episode innovations must fit offset + trend at this level
    min_episode_fixes: int = 3
    anchor_min_s: float = 1.0         # healthy fixes accepted this long = anchored (a later disagreement is a jump)
    step_back_ratio: float = 0.25     # after a jump: camera moved back by >= half the offset (Mahalanobis) = recovery
    provisional_factor: float = 2.0   # a re-anchor that ended a jump stays provisional this x the jump hold-out
    max_jump_holdout_s: float = 30.0  # a consistent healthy camera is re-anchored after this at the latest
    jump_cover: float = 1.0           # jump re-anchor needs nis <= gate * jump_cover (state cannot rule it out)
    max_episode_fixes: int = 600      # memory bound per episode (oldest dropped)
    episode_end_lr: float = 100.0     # "camera unbiased again" must be this much likelier than "offset persists"
    outage_envelope: float = 100.0    # recovering offset must fit (envelope x P_pos + R)
    model_check_up: float = 0.1     # learned-model self-check: how fast trust drops (per update)
    model_check_down: float = 0.003 # ... and how slowly it comes back
    model_doubt_scale: float = 10.0 # self-check scale above which the model is "distrusted" (see model_distrusted)
    model_max_scale: float | None = 10.0  # withhold a measured-inconsistent model without a healthy camera
    model_covariance_validated: bool = False  # requires held-out calibration evidence for this model/domain
    model_during_visual_tracking: bool = False  # research override: correlated learned updates while camera anchors pose
    use_health: bool = True       # off = naive fusion that trusts every frame equally
    trust: tuple | None = None    # (health, std_m) table from camera.calibrate_trust
    cloning: bool = False         # TLIO-style stochastic cloning of past (rotation, position)
    disp_cov_scale: float = 1.0   # calibration of the network's claimed variance (>1: it is overconfident)
    yaw_frame: bool = False       # with cloning: displacement in the clone's yaw frame (TLIO eq. 9-10)
    adaptive_q: bool = False      # inflate IMU process noise when accepted healthy fixes disagree
    q_up: float = 0.2
    q_down: float = 0.01
    q_max: float = 400.0
    bias_q_gain: float = 1.0
    use_gate: bool = True
    gate: float = CHI2_3DOF_999
    gate_mode: str = "soft"       # learned displacement only: "soft" = robust down-weighting, "hard" = reject
    soft_power: float = 2.0
    fix_std: float = 0.03
    low_conf_inflate: float = 25.0   # legacy imu_conf input: process-noise std multiplier at conf 0
    max_imu_gap_s: float = 0.02      # longer IMU gaps are propagated in sub-steps with envelope noise
    gap_sigma_g: float = 0.5         # motion envelope while IMU data is missing, rad/s/sqrt(Hz)
    gap_sigma_a: float = 2.0         # ... m/s^2/sqrt(Hz)
    max_valid_pos_std: float = 1.0   # largest position std (m, worst axis) still reported as valid
    disp_rate_tol: float = 0.05      # learned window skipped if its sample spacing deviates more
    vo_cov_scale: float = 1.0        # calibration of the visual front end's claimed relative-pose covariance
    vo_gate: float = CHI2_6DOF_999   # hard chi-square gate on the 6-DoF relative-pose innovation
    vo_max_clones: int = 3           # camera keyframe clones kept in the state
    # The front end's covariance (PnP information x residual scale x vo_cov_scale) has been fitted on
    # development data only. Like the learned model's, it certifies nothing until held-out coverage
    # evidence exists: relative-mode outputs are COVARIANCE_UNRELIABLE / invalid unless asserted.
    vo_covariance_validated: bool = False
    # Recovery from persistent VO rejection (relative mode). Healthy VO that keeps failing the gate means
    # either the camera is wrong (frozen/desynchronised frames) or the inertial state has diverged (IMU
    # bias or saturation during an outage). After an outage the inertial state is the suspect after
    # recover_after_s of consecutive healthy rejections; while VO was anchored a disagreement is first
    # treated as a camera fault and held out for vo_recover_after_jump_s (shorter than the absolute-fix
    # hold-out: a rejected relative stream lets inertial drift grow without bound). Then velocity,
    # attitude and bias covariance are inflated and VO is applied ungated for recover_after_s (reported).
    vo_reanchor: bool = True
    vo_recover_after_jump_s: float = 2.0  # hold-out for a disagreement that starts while VO is anchored
    vo_reanchor_vel_std: float = 1.0      # m/s added to the velocity std
    vo_reanchor_att_deg: float = 5.0
    vo_reanchor_bg_std: float = 0.05      # rad/s
    vo_reanchor_ba_std: float = 0.5       # m/s^2
    # TLIO training windows are offline bias-compensated; raw-IMU devices (e.g. TUM-VI) are not. When
    # true, the learned model sees IMU samples minus the filter's current bias estimate. Off by default
    # (issue #1 behaviour on the bias-compensated TLIO data).
    model_input_bias_corrected: bool = False

    def __post_init__(self):
        positive = ("fix_std", "gate", "max_imu_gap_s", "max_valid_pos_std", "disp_cov_scale",
                    "outage_gap_s", "recover_after_s", "recover_after_jump_s", "model_doubt_scale",
                    "vo_reanchor_vel_std", "vo_recover_after_jump_s", "vo_reanchor_bg_std", "vo_reanchor_ba_std",
                    "max_jump_holdout_s", "episode_end_lr", "jump_cover", "outage_envelope", "vo_cov_scale",
                    "vo_gate")
        nonnegative = ("vo_reanchor_att_deg", "sigma_g", "sigma_a", "sigma_bg", "sigma_ba", "gap_sigma_g", "gap_sigma_a",
                       "bias_q_gain", "anchor_min_s", "provisional_factor", "soft_power", "disp_rate_tol")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"invalid {name}: {value}")
        if self.model_max_scale is not None and (not np.isfinite(self.model_max_scale) or self.model_max_scale <= 0):
            raise ValueError("model_max_scale must be positive, finite or None (research only)")
        for name in ("model_check_up", "model_check_down", "q_up", "q_down"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.min_health <= self.recover_health <= 1:
            raise ValueError("health thresholds must satisfy 0 <= min <= recovery <= 1")
        if not 0 < self.consistency_alpha < 1 or not 0 < self.step_back_ratio < 1:
            raise ValueError("consistency_alpha and step_back_ratio must be in (0, 1)")
        if (self.gate_mode not in ("soft", "hard") or not np.isfinite(self.q_max) or self.q_max < 1
                or not np.isfinite(self.low_conf_inflate) or self.low_conf_inflate < 1):
            raise ValueError("invalid gating or process-noise configuration")
        if int(self.vo_max_clones) != self.vo_max_clones or self.vo_max_clones < 1:
            raise ValueError("vo_max_clones must be a positive integer")
        if self.min_episode_fixes < 3 or self.max_episode_fixes < self.min_episode_fixes + 2:
            raise ValueError("episode buffer is too short for the consistency fit")
        if self.max_jump_holdout_s < self.recover_after_jump_s:
            raise ValueError("maximum jump hold-out must include the minimum hold-out")
        if self.trust is not None:
            h, std = (np.array(x, dtype=float, copy=True) for x in self.trust)
            if (h.ndim != 1 or std.shape != h.shape or len(h) < 2 or not np.isfinite(h).all()
                    or not np.isfinite(std).all() or np.any(np.diff(h) <= 0) or np.any(std <= 0)):
                raise ValueError("trust must contain increasing health bins and positive finite standard deviations")
            self.trust = (h, std)


@dataclass
class FilterLog:
    p: list = field(default_factory=list)
    p_std: list = field(default_factory=list)   # sqrt(trace(P_pos)): a scalar summary, NOT a 3-sigma radius
    cam_used: int = 0
    cam_rejected: int = 0
    cam_skipped: int = 0
    cam_recovered: int = 0
    cam_invalid: int = 0
    cam_late: int = 0
    disp_used: int = 0
    disp_rejected: int = 0
    disp_skipped: int = 0
    disp_withheld: int = 0                      # skipped by the optional activation policy
    disp_std: list = field(default_factory=list)
    vo_used: int = 0                            # accepted relative-pose (visual odometry) updates
    vo_rejected: int = 0                        # failed the 6-DoF gate
    vo_no_anchor: int = 0                       # relative measurement whose keyframe clone is not in the state
    vo_keyframes: int = 0
    vo_reanchored: int = 0                      # inertial state re-anchored to persistent healthy VO
    vo_nis: list = field(default_factory=list)  # (t, nis, accepted) per relative update
    model_scale: list = field(default_factory=list)
    q_scale: list = field(default_factory=list)  # adaptive process-noise scale per output sample
    events: list = field(default_factory=list)  # (t, kind, detail) for re-anchors and episodes


@dataclass
class _Episode:
    origin: str                   # "jump" or "outage"
    t_start: float
    ts: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    Rs: list = field(default_factory=list)
    Ps: list = field(default_factory=list)

    def add(self, t, y, R, P, cap: int = 600):
        self.ts.append(t)
        self.ys.append(y)
        self.Rs.append(R)
        self.Ps.append(P)
        if len(self.ts) > cap:
            del self.ts[0], self.ys[0], self.Rs[0], self.Ps[0]

    def fit(self):
        """Weighted polynomial fit of the episode's innovations; returns (b, rate, chi2 residual, dof).

        After a jump the camera is the suspect and a rigid wrong fix is an offset plus a linear trend
        (order 1). After an outage the inertial estimate is the suspect: its error grows with velocity
        error and accelerometer bias, i.e. quadratically over the episode (order 2). b and rate are
        evaluated at the newest fix.
        """
        n = len(self.ts)
        order = 2 if self.origin == "outage" else 1
        ts = np.array(self.ts)
        tau = ts - ts.mean()
        m = 3 * (order + 1)
        A = np.zeros((m, m))
        rhs = np.zeros(m)
        Ws = [np.linalg.inv(R) for R in self.Rs]
        basis = lambda x: np.hstack([x ** j * np.eye(3) for j in range(order + 1)])
        for i in range(n):
            J = basis(tau[i])
            A += J.T @ Ws[i] @ J
            rhs += J.T @ Ws[i] @ self.ys[i]
        th = np.linalg.lstsq(A, rhs, rcond=None)[0]
        r2 = 0.0
        for i in range(n):
            e = self.ys[i] - basis(tau[i]) @ th
            r2 += float(e @ Ws[i] @ e)
        x = ts[-1] - ts.mean()
        b = basis(x) @ th
        rate = th[3:6] + (2 * x * th[6:9] if order == 2 else 0.0)
        return b, rate, r2, max(3 * n - m, 1)


class ESKF:
    def __init__(self, cfg: FilterConfig, p0, v0, q0, P0_scale: float = 1e-4, P0=None):
        self.cfg = cfg
        for name, value, shape in (("p0", p0, (3,)), ("v0", v0, (3,)), ("q0", q0, (4,))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if P0 is not None:
            P0 = np.asarray(P0, float)
            if (P0.shape != (15, 15) or not np.isfinite(P0).all() or not np.allclose(P0, P0.T)
                    or np.linalg.eigvalsh(P0).min() < -1e-12):
                raise ValueError("initial covariance must be a finite symmetric positive semidefinite 15x15 matrix")
        self.R = Rot.from_quat(q0)
        self.v = np.array(v0, float)
        self.p = np.array(p0, float)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        if P0 is not None:
            self.P = np.array(P0, float)
        else:
            self.P = np.eye(15) * P0_scale
            self.P[9:12, 9:12] = np.eye(3) * 1e-4
            self.P[12:15, 12:15] = np.eye(3) * 1e-2
        self.clone_k: list[int] = []      # stochastic cloning: past poses kept in the state
        self.clone_p: list[np.ndarray] = []
        self.clone_R: list = []           # error state per clone: [dtheta (3), dp (3)]
        self.q_scale = 1.0
        self.model_scale = 1.0
        self.last_good_t = -np.inf        # last accepted healthy camera fix
        self.last_used_t = -np.inf        # last accepted camera fix of any health
        self.last_used_healthy = False
        self.episode: _Episode | None = None
        self.good_since = None
        self.vo_rej_t0 = None             # start of the current run of healthy VO rejections
        self.vo_rej_hold = 0.0
        self.vo_bypass_until = -np.inf    # end of the re-anchoring window
        self.provisional_until = -np.inf
        self._nis_b = 0.0
        self.diverged = False
        self.last_nis = np.nan

    # ------------------------------------------------------------------ prediction
    def propagate(self, gyro, acc, dt, imu_conf: float = 1.0, q_add_g: float = 0.0, q_add_a: float = 0.0):
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError(f"propagation step must be positive and finite, got {dt}")
        w = gyro - self.bg
        f = acc - self.ba
        Rm = self.R.as_matrix()
        a = Rm @ f + G
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.R = self.R * Rot.from_rotvec(w * dt)

        F = np.eye(15)
        F[0:3, 0:3] = Rot.from_rotvec(w * dt).as_matrix().T      # exact for constant w over dt
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 0:3] = -Rm @ skew(f) * dt
        F[3:6, 12:15] = -Rm * dt
        F[6:9, 0:3] = -0.5 * Rm @ skew(f) * dt * dt
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -0.5 * Rm * dt * dt
        c = self.cfg
        k = 1.0 + (c.low_conf_inflate - 1.0) * (1.0 - float(np.clip(imu_conf, 0, 1)))
        q = self.q_scale if c.adaptive_q else 1.0
        Q = np.zeros(15)
        Q[0:3] = ((c.sigma_g * k) ** 2 * q + q_add_g) * dt
        Q[3:6] = ((c.sigma_a * k) ** 2 * q + q_add_a) * dt
        Q[9:12] = c.sigma_bg ** 2 * dt * q * c.bias_q_gain
        Q[12:15] = c.sigma_ba ** 2 * dt * q * c.bias_q_gain
        P = self.P
        P[:15, :15] = F @ P[:15, :15] @ F.T + np.diag(Q)
        if P.shape[0] > 15:                       # clones are static; only cross terms move
            P[:15, 15:] = F @ P[:15, 15:]
            P[15:, :15] = P[:15, 15:].T
        self.P = 0.5 * (P + P.T)
        self._check()

    # ------------------------------------------------------------------ updates
    def _check(self):
        if not (np.all(np.isfinite(self.P)) and np.all(np.isfinite(self.p)) and np.all(np.isfinite(self.v))):
            self.diverged = True

    def _inject(self, dx):
        n = self.P.shape[0]
        Gm = np.eye(n)
        Gm[0:3, 0:3] -= skew(0.5 * dx[0:3])
        self.R = self.R * Rot.from_rotvec(dx[0:3])
        self.v = self.v + dx[3:6]
        self.p = self.p + dx[6:9]
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]
        for i in range(len(self.clone_p)):
            o = 15 + 6 * i
            self.clone_R[i] = self.clone_R[i] * Rot.from_rotvec(dx[o:o + 3])
            self.clone_p[i] = self.clone_p[i] + dx[o + 3:o + 6]
            Gm[o:o + 3, o:o + 3] -= skew(0.5 * dx[o:o + 3])
        self.P = Gm @ self.P @ Gm.T

    def nis(self, y, H, Rmeas):
        S = H @ self.P @ H.T + Rmeas
        return float(y @ np.linalg.solve(S, y))

    def _update(self, y, H, Rmeas, mode: str = "soft"):
        """mode: "soft" (robust down-weighting outside the gate), "hard" (reject), "off" (no gate).
        Returns "accepted", "softened" or "rejected"."""
        if not (np.all(np.isfinite(y)) and valid_covariance(Rmeas, len(y))):
            return "rejected"
        n = self.P.shape[0]
        S = H @ self.P @ H.T + Rmeas
        S = 0.5 * (S + S.T)
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis
        result = "accepted"
        if mode != "off" and self.cfg.use_gate and nis > self.cfg.gate:
            if mode == "hard":
                return "rejected"
            Rmeas = Rmeas * (nis / self.cfg.gate) ** self.cfg.soft_power
            S = H @ self.P @ H.T + Rmeas
            S = 0.5 * (S + S.T)
            result = "softened"
        K = np.linalg.solve(S, H @ self.P).T          # K = P H^T S^-1 (S, P symmetric)
        dx = K @ y
        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rmeas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)
        self._check()
        return result

    def _pos_H(self):
        H = np.zeros((3, self.P.shape[0]))
        H[:, 6:9] = np.eye(3)
        return H

    def _update_position(self, z, Rmeas, mode="soft"):
        return self._update(z - self.p, self._pos_H(), Rmeas, mode)

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
        # Clones of one kind are pruned independently: int keys are learned-displacement window
        # starts, ("cam", id) keys are camera keyframes. Mixing them must not evict the other kind.
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

    def relative_pose_measurement(self, key):
        """h and H for the IMU pose now relative to the clone ``key``, expressed in the clone's frame.

        translation  h_t = R_i^T (p - p_i)   dh_t = R_i^T dp - R_i^T dp_i + [h_t]x dtheta_i
        rotation     h_R = R_i^T R          residual r = Log(h_R^T z_R) ~ dtheta - h_R^T dtheta_i
        (right perturbations R = R_hat Exp(dtheta) for the current state and every clone)."""
        i = self.clone_k.index(key)
        n = self.P.shape[0]
        o = 15 + 6 * i
        Ri = self.clone_R[i].as_matrix()
        h_t = Ri.T @ (self.p - self.clone_p[i])
        h_R = Ri.T @ self.R.as_matrix()
        H = np.zeros((6, n))
        H[0:3, 6:9] = Ri.T
        H[0:3, o + 3:o + 6] = -Ri.T
        H[0:3, o:o + 3] = skew(h_t)
        H[3:6, 0:3] = np.eye(3)
        H[3:6, o:o + 3] = -h_R.T
        return h_t, h_R, H

    def relative_pose_update(self, key, dp, dR, cov6, health, log: FilterLog, t: float = 0.0):
        """Visual odometry between a camera keyframe clone and now (vision-only relative motion).

        dp (3,) metres and dR (3x3) are the IMU pose now in the keyframe's IMU frame, cov6 the front
        end's covariance of [dp, dtheta] (rotation as a right perturbation of dR). Returns
        "no_anchor", "skipped", "invalid", "accepted", "rejected" or "reanchored". A relative measurement does not observe
        absolute position or heading, so it never re-anchors the world frame."""
        if key not in self.clone_k:
            log.vo_no_anchor += 1
            return "no_anchor"
        if self.cfg.use_health and not health >= self.cfg.min_health:
            log.cam_skipped += 1               # front end reports essentially nothing usable
            return "skipped"
        dR = np.asarray(dR, float)
        cov6 = np.asarray(cov6, float) * self.cfg.vo_cov_scale
        if (np.shape(dp) != (3,) or not np.all(np.isfinite(dp)) or dR.shape != (3, 3) or not np.isfinite(dR).all()
                or not np.allclose(dR @ dR.T, np.eye(3), atol=1e-6) or not valid_covariance(cov6, 6)):
            log.cam_invalid += 1
            return "invalid"
        h_t, h_R, H = self.relative_pose_measurement(key)
        y = np.r_[np.asarray(dp, float) - h_t, Rot.from_matrix(h_R.T @ dR).as_rotvec()]
        S = H @ self.P @ H.T + cov6
        S = 0.5 * (S + S.T)
        nis = float(y @ np.linalg.solve(S, y))
        result = "accepted"
        c = self.cfg
        if c.use_gate and nis > c.vo_gate and t <= self.vo_bypass_until:
            result = "reanchored"               # inside the re-anchoring window: applied without the gate
        elif c.use_gate and nis > c.vo_gate:
            if not (c.vo_reanchor and health >= c.recover_health):
                log.vo_rejected += 1
                log.vo_nis.append((t, nis, False))
                return "rejected"
            if self.vo_rej_t0 is None:
                anchored = (t - self.last_good_t <= c.outage_gap_s and self.good_since is not None
                            and self.last_good_t - self.good_since >= c.anchor_min_s - 1e-9)
                self.vo_rej_t0, self.vo_rej_hold = t, (c.vo_recover_after_jump_s if anchored else c.recover_after_s)
            log.vo_rejected += 1
            log.vo_nis.append((t, nis, False))
            if t - self.vo_rej_t0 >= self.vo_rej_hold:
                # Persistent healthy VO outlasted the hold-out: the inertial state is the suspect. Inflate what
                # the IMU alone could have corrupted (velocity, attitude, biases). This measurement's relative
                # position was propagated with the old covariance, so it is not applied; the following
                # measurements (relative to the next keyframe clone, which carries the inflated covariance)
                # are applied without the gate for recover_after_s. Frozen frames, the main *camera* fault
                # that looks like healthy VO, are removed earlier by the front end's stale-frame check.
                infl = np.zeros(self.P.shape[0])
                infl[0:3] = np.deg2rad(c.vo_reanchor_att_deg) ** 2
                infl[3:6] = c.vo_reanchor_vel_std ** 2
                infl[9:12] = c.vo_reanchor_bg_std ** 2
                infl[12:15] = c.vo_reanchor_ba_std ** 2
                self.P = self.P + np.diag(infl)
                self.vo_bypass_until = t + c.recover_after_s
                log.vo_reanchored += 1
                log.events.append((t, "vo_reanchored", f"held {t - self.vo_rej_t0:.2f}s nis {nis:.1f}"))
                self.vo_rej_t0 = None
                return "reanchored"
            return "rejected"
        self.vo_rej_t0 = None
        self._update(y, H, cov6, mode="off")
        log.vo_used += 1
        log.vo_nis.append((t, nis, True))
        self.last_used_t = t
        self.last_used_healthy = health >= self.cfg.recover_health
        if self.last_used_healthy:
            if self.good_since is None or t - self.last_good_t > self.cfg.outage_gap_s:
                self.good_since = t
            self.last_good_t = t
        return result

    def clone_yaw(self, k0):
        """Yaw-only rotation (world <- local gravity-aligned frame) of the clone taken at sample k0."""
        Rm = self.clone_R[self.clone_k.index(k0)].as_matrix()
        g = np.arctan2(Rm[1, 0], Rm[0, 0])
        c, s_ = np.cos(g), np.sin(g)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def clone_measurement(self, k0):
        """h and H for d = p(k) - p(k0) with p(k0) a clone (yaw frame if configured)."""
        i = self.clone_k.index(k0)
        n = self.P.shape[0]
        o = 15 + 6 * i
        H = np.zeros((3, n))
        if self.cfg.yaw_frame:
            # h = Rz(yaw_i)^T (p - p_i). A yaw error dgamma gives dh = -[e_z]x h dgamma. With the
            # clone's rotation right-perturbed, R Exp(dtheta), and yaw = atan2(R10, R00):
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
        return h, H

    def _anchored(self, t):
        return t - self.last_good_t <= self.cfg.outage_gap_s and self.episode is None

    def _model_check(self, y, S, t):
        # Self-check of the learned model: while healthy fixes are being accepted the state is
        # anchored by the camera, so a large innovation means the *model* is wrong on this motion.
        # Lose trust fast, regain slowly. Not run during camera disagreement episodes, so a
        # suspect camera cannot poison it.
        if not self._anchored(t):
            return
        ratio = float(y @ np.linalg.solve(S, y)) / 3.0
        target = max(ratio, 1.0)
        a = self.cfg.model_check_up if target > self.model_scale else self.cfg.model_check_down
        self.model_scale = (1 - a) * self.model_scale + a * target

    def model_distrusted(self):
        """The self-check says the network is far outside its claimed uncertainty on this motion.

        model_scale is an asymmetric EWMA of NIS/3 with correlated windows. Its threshold is a
        conservative engineering guard, not a chi-square confidence level; the EWMA does not
        itself have a chi-square law. A low value does not certify model covariance calibration.
        """
        return self.model_scale > self.cfg.model_doubt_scale

    def _model_cov(self, cov, t, log):
        """Covariance actually used for a learned update, or None if the update is withheld."""
        c = self.cfg
        if self._anchored(t) and not c.model_during_visual_tracking:
            # The camera already observes translation. Use these windows to check the model,
            # without counting its IMU-derived prediction as independent position evidence.
            log.disp_skipped += 1
            log.disp_withheld += 1
            return None
        if self.episode is not None and self.episode.origin == "jump" and self.model_distrusted():
            # Camera and state disagree and the camera is being held out. If the self-check has
            # already measured the network as inconsistent, its updates would be the only thing
            # keeping the state covariance tight enough to veto the camera; a distrusted model must not
            # arbitrate. The state falls back to inertial-only, whose covariance grows honestly, until
            # the episode ends. (Reproduced: synthetic camera_wrong_fix with the models-v1 network went
            # to 4.8 m ATE without this rule, baseline_evidence/synthetic_eval_after.txt.)
            log.disp_skipped += 1
            log.disp_withheld += 1
            return None
        if c.model_max_scale is not None and self.model_scale > c.model_max_scale and not self._anchored(t):
            # Conservative default: an observed-inconsistent model cannot anchor an outage.
            # None is an explicit research override, not a claim of calibrated model uncertainty.
            log.disp_skipped += 1
            log.disp_withheld += 1
            return None
        return cov * self.model_scale

    def displacement_update_clone(self, d, cov, k0, log, t: float = 0.0):
        """Learned displacement d = p(k) - p(k0), with p(k0) a clone in the state (as in TLIO)."""
        if k0 not in self.clone_k:
            log.disp_skipped += 1
            return
        if not (np.shape(d) == (3,) and np.all(np.isfinite(d)) and valid_covariance(cov)):
            log.disp_skipped += 1
            return
        h, H = self.clone_measurement(k0)
        y = d - h
        self._model_check(y, H @ self.P @ H.T + cov, t)
        cov = self._model_cov(cov, t, log)
        if cov is None:
            return
        if self._update(y, H, cov, self.cfg.gate_mode) == "accepted":
            log.disp_used += 1
        else:
            log.disp_rejected += 1

    def displacement_update(self, d, cov, p_past, P_past_pos, log: FilterLog, t: float = 0.0):
        if not (np.shape(d) == (3,) and np.all(np.isfinite(d)) and valid_covariance(cov)
                and np.all(np.isfinite(p_past))):
            log.disp_skipped += 1
            return
        z = p_past + d
        y = z - self.p
        self._model_check(y, self.P[6:9, 6:9] + cov + P_past_pos, t)
        cov = self._model_cov(cov, t, log)
        if cov is None:
            return
        if self._update_position(z, cov + P_past_pos, self.cfg.gate_mode) == "accepted":
            log.disp_used += 1
        else:
            log.disp_rejected += 1

    # ------------------------------------------------------------------ camera supervision
    def camera_std(self, health):
        c = self.cfg
        if not c.use_health:
            return c.fix_std
        if c.trust is not None:
            # Calibrated on training data: measured fix error for this health level.
            return float(np.interp(health, c.trust[0], c.trust[1]))
        if health < c.min_health:
            return None
        return c.fix_std / max(health, 0.02)

    def camera_state(self, t):
        if self.episode is not None:
            return CAM_SUSPECT if self.episode.origin == "jump" else CAM_RECOVERING
        if t - self.last_used_t > self.cfg.outage_gap_s:
            return CAM_OUTAGE
        return CAM_NOMINAL if (self.last_used_healthy and t - self.last_good_t <= self.cfg.outage_gap_s) \
            else CAM_DEGRADED

    def camera_update(self, fix, health, log: FilterLog, t: float = 0.0):
        """Returns one of: invalid, skipped, accepted, rejected, reanchored."""
        c = self.cfg
        fix = np.asarray(fix, float)
        if fix.shape != (3,) or not np.all(np.isfinite(fix)) or not np.isfinite(health):
            log.cam_invalid += 1
            return "invalid"
        health = float(np.clip(health, 0.0, 1.0))
        std = self.camera_std(health)
        if std is None:
            log.cam_skipped += 1
            return "skipped"
        Rm = np.eye(3) * std ** 2
        healthy = health >= c.recover_health
        H = self._pos_H()
        y = fix - self.p
        Ppos = self.P[6:9, 6:9].copy()
        S = Ppos + Rm
        nis = float(y @ np.linalg.solve(S, y))
        mode = "off" if not c.use_gate else "hard"
        if self.episode is None and (not c.use_gate or nis <= c.gate):
            if c.adaptive_q and healthy:
                # Covariance matching on an innovation that passed the gate (never a rejected one).
                ratio = nis / 3.0
                target = float(np.clip(ratio, 1.0, c.q_max))
                a = c.q_up if target > self.q_scale else c.q_down
                self.q_scale = (1 - a) * self.q_scale + a * target
            self._update(y, H, Rm, mode)
            log.cam_used += 1
            self.last_used_t = t
            self.last_used_healthy = healthy
            if healthy:
                if t - self.last_good_t > c.outage_gap_s or self.good_since is None:
                    self.good_since = t             # start of the current run of healthy fixes
                self.last_good_t = t
            return "accepted"
        if self.episode is not None:
            if self.episode.origin == "outage":
                # Recovering: the state is the suspect, so agreement with it is all that is needed.
                end, step_back = nis <= c.gate, False
            else:
                lr_ok, step_back = self._episode_over(y, Rm, Ppos, S, nis, t)
                end = nis <= c.gate and lr_ok
            if end:
                # The camera agrees again AND (after a jump) the jump offset no longer explains it.
                log.events.append((t, "episode_end_agree", self.episode.origin))
                self.episode = None
                return self.camera_update(fix, health, log, t)
            if step_back and healthy:
                # The camera left the suspicious offset and moved back towards the inertial
                # prediction, but the inertial estimate has drifted beyond the gate meanwhile.
                # Treat the rest as a recovery after an outage: re-anchor only after recover_after_s
                # of mutually consistent fixes.
                log.events.append((t, "episode_step_back", "jump->outage"))
                self.episode = _Episode("outage", t)
        log.cam_rejected += 1
        if not healthy:
            return "rejected"                       # low-health frames never steer recovery
        if self.episode is None:
            # Why does a healthy camera disagree? If we just came back from an outage, we probably
            # drifted. If it was agreeing a moment ago and suddenly jumped, it is probably wrong.
            # "Agreeing a moment ago" needs a sustained run of accepted healthy fixes, not a single one:
            # after degraded vision a lone healthy fix can scrape through the gate while the state is
            # still wrong, and the next healthy fixes must not then be treated as a camera jump.
            origin = self._classify_origin(t, y, Ppos, Rm)
            self.episode = _Episode(origin, t)
            log.events.append((t, "episode_start", origin))
        ep = self.episode
        ep.add(t, y, Rm, Ppos, c.max_episode_fixes)
        needed = c.recover_after_s if ep.origin == "outage" else c.recover_after_jump_s
        n = len(ep.ts)
        if n >= c.min_episode_fixes + (2 if ep.origin == "outage" else 0):
            b, rate, r2, dof = ep.fit()
            if r2 > chi2.ppf(1 - c.consistency_alpha, dof):
                # Not a rigid offset any more. If, after a jump, the camera moved most of the way back
                # to the inertial prediction, it is a (partial) recovery: continue as "outage" with the
                # short consistency hold-out. Otherwise the camera is scattering: start again.
                # The restarted episode is classified like a new one: once the state has not been
                # anchored for longer than outage_gap_s, the inertial estimate is the suspect, so a
                # camera that changes its offset is treated as a possible recovery (short, consistency-
                # checked hold-out), not as a second jump that needs another full jump hold-out.
                # Inside an episode `last_good_t` is stale by construction, so the restart is not
                # classified by elapsed time. It becomes "outage" only on evidence: the camera moved
                # back towards the prediction, or the state's own uncertainty can no longer rule the
                # camera out (nis within the gate). With independent evidence keeping P_pos tight
                # (learned displacement), a wandering wrong fix stays a jump.
                origin, t0 = ep.origin, t
                if ep.origin == "jump" and nis < c.step_back_ratio * self._nis_b:
                    origin = "outage"
                if origin != ep.origin:
                    log.events.append((t, "episode_step_back", f"{ep.origin}->{origin}"))
                else:
                    log.events.append((t, "episode_inconsistent", ep.origin))
                    if origin == "jump":
                        # The camera has disagreed since ep.t_start; the hold-out keeps counting from
                        # there, while the consistency fit uses only the new segment.
                        t0 = ep.t_start
                self.episode = _Episode(origin, t0)
                self.episode.add(t, y, Rm, Ppos)
                return "rejected"
            # After a jump the hold-out alone is not evidence: a rigid wrong fix that outlasts it
            # would be taken. Also require that the state's own uncertainty can no longer rule the
            # camera out (nis within the gate). While independent evidence (learned displacement,
            # small inertial drift) keeps P_pos tight, a large persistent offset stays rejected.
            # Test the episode's fitted offset b (many fixes, precise), not one fix: a single
            # lower-health frame claims a large noise and can make any offset look plausible.
            Cb = np.mean(ep.Rs, 0) / n
            nis_ep = float(b @ np.linalg.solve(Ppos + Cb, b))
            covered = ep.origin == "outage" or nis_ep <= c.gate * c.jump_cover
            elapsed = t - ep.t_start
            # Past the jump hold-out the camera has been rigidly self-consistent; it is still kept out
            # while the state's covariance rules it out. That covariance can be held tight by a
            # learned model that is confidently wrong, and with no third position source both are
            # suspect, so the hold-out is capped at max_jump_holdout_s.
            forced = ep.origin == "jump" and elapsed >= c.max_jump_holdout_s - 1e-9
            if elapsed >= needed - 1e-9 and (covered or forced):
                # Re-anchor: open position / velocity uncertainty to the observed disagreement, then
                # take the fix without the gate.
                self.P[6:9, 6:9] += np.outer(b, b)
                self.P[3:6, 3:6] += np.outer(rate, rate)
                self._update(y, H, Rm, "off")
                log.cam_recovered += 1
                log.events.append((t, "reanchored_forced" if (forced and not covered) else "reanchored", ep.origin))
                self.episode = None
                self.last_good_t = self.last_used_t = t
                self.good_since = t
                self.last_used_healthy = True
                if ep.origin == "jump":
                    self.provisional_until = t + c.provisional_factor * c.recover_after_jump_s
                return "reanchored"
        return "rejected"

    def _classify_origin(self, t, y, Ppos, Rm):
        """"jump" if the state was anchored by a sustained run of accepted healthy fixes until a
        moment ago (the camera is the suspect), else "outage" (the inertial estimate is the suspect),
        unless the offset exceeds what inertial drift can explain."""
        c = self.cfg
        # "Agreeing a moment ago" needs a sustained run of accepted healthy fixes, not a single one:
        # after degraded vision a lone healthy fix can scrape through the gate while the state is still
        # wrong, and the next healthy fixes must not then be treated as a camera jump.
        # A re-anchor that ended a *jump* episode is provisional: it may have been onto a wrong fix
        # that outlasted the hold-out. Until recover_after_jump_s has passed, a later disagreement is
        # treated as a possible recovery (consistency-checked, envelope-bounded), not as a new jump.
        anchored = (t - self.last_good_t <= c.outage_gap_s and self.good_since is not None
                    and self.last_good_t - self.good_since >= c.anchor_min_s - 1e-9
                    and t >= self.provisional_until)
        if anchored:
            return "jump"
        if self.model_distrusted():
            # The drift envelope below rests on P_pos, which a distrusted learned model has held
            # artificially tight while the camera was away: it cannot be used to call the camera wrong.
            return "outage"
        env = c.outage_envelope * Ppos + Rm
        return "jump" if float(y @ np.linalg.solve(env, y)) > c.gate else "outage"

    def _episode_over(self, y, Rm, Ppos, S0, nis0, t):
        """Two-hypothesis test for ending a disagreement episode.

        H0: the camera is unbiased again, y ~ N(0, P_pos + R).
        H1: the episode's offset b persists, y ~ N(b_pred, R + var(b_pred) + inertial drift since the
            episode started).
        Passing the chi-square gate is not enough: while fixes are rejected the inertial covariance
        grows, and a persistent wrong fix eventually falls inside the gate (this is what let a
        repeated wrong fix take over in the first implementation, see docs/ESTIMATOR.md). The episode
        ends only if H0 is ``episode_end_lr`` times more likely than H1.
        """
        ep = self.episode
        n = len(ep.ts)
        if n == 0:
            return True, False
        if n >= self.cfg.min_episode_fixes:
            b, rate, _, _ = ep.fit()
            b = b + rate * (t - ep.ts[-1])
            Cb = np.mean(ep.Rs, 0) / n * 4.0 + ep.Rs[-1]   # extrapolated offset: conservative variance
        else:
            b = ep.ys[-1]
            Cb = ep.Rs[-1]
        # Change of the inertial error since the last episode fix. This is NOT P_pos(t) - P_pos(t_last):
        # that difference is dominated by the cross-covariance of an error that already existed.
        # The increment is (velocity error) x dt; bound it by the current velocity covariance with a
        # factor 2 in std for the unmodelled acceleration error. The offset fit absorbs a mean trend.
        dt_last = max(t - ep.ts[-1], 0.0)
        growth = 4.0 * self.P[3:6, 3:6] * dt_last ** 2
        S1 = Rm + Cb + growth
        e1 = y - b
        nis1 = float(e1 @ np.linalg.solve(S1, e1))
        log_lr = 0.5 * (nis1 - nis0) + 0.5 * (np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])
        self._nis_b = float(b @ np.linalg.solve(S0, b))
        step_back = nis1 > self.cfg.gate and nis0 < self.cfg.step_back_ratio * self._nis_b
        return log_lr > np.log(self.cfg.episode_end_lr), step_back

    def position_cov(self):
        return self.P[6:9, 6:9].copy()


def run(seq, gyro, acc, imu_conf=None, cam=None, disp=None, cfg: FilterConfig = FilterConfig(),
        start: int = 0, *, imu_quality=None, init="truth", return_outputs: bool = False):
    """Offline replay of one sequence through the streaming estimator (same code path as live use).

    gyro, acc: (N, 3) fused IMU stream. imu_conf: legacy (N,) confidence in [0, 1] or None.
    imu_quality: dict with per-sample ``q_add_g``, ``q_add_a``, ``state`` from ``ImuMonitor``.
    cam: dict from camera.visual_fixes (idx, fix, health) or None. disp: learned predictor or None.
    init: ``"truth"`` = **diagnostic / benchmark** start from the reference pose at ``start``;
          ``"reference_free"`` or a ``ReferenceFreeInit`` = uses only IMU and camera data.
    Returns (positions (N, 3), FilterLog). Samples without a valid estimate are NaN.
    """
    from .estimator import ReferenceFreeInit, TruthInit, replay
    if isinstance(disp, dict):
        raise NotImplementedError("precomputed displacement dicts are no longer supported; pass a predictor")
    t = np.asarray(seq.t, float)
    if init == "truth":
        init_obj = TruthInit(seq.p[start], seq.v[start], seq.q[start])
    elif init == "reference_free":
        init_obj = ReferenceFreeInit()
    else:
        init_obj = init
    sl = slice(start, None)
    q = None if imu_quality is None else {k: np.asarray(v)[sl] for k, v in imu_quality.items()
                                          if k in ("q_add_g", "q_add_a", "state")}
    conf = None if imu_conf is None else np.asarray(imu_conf)[sl]
    cam_s = None
    if cam is not None:
        keep = np.asarray(cam["idx"]) >= start
        cam_s = {"t": t[np.asarray(cam["idx"])[keep]], "fix": np.asarray(cam["fix"])[keep],
                 "health": np.asarray(cam["health"])[keep]}
    out, log, outputs = replay(t[sl], np.asarray(gyro)[sl], np.asarray(acc)[sl], cfg, init_obj, cam=cam_s,
                               predictor=disp, imu_quality=q, imu_conf=conf)
    full = np.full((len(t), 3), np.nan)
    full[start:] = out
    if return_outputs:
        return full, log, outputs
    return full, log
