"""Reference-free streaming estimator: the runtime interface of Path Finder.

Inputs (all timestamps in seconds on one clock, strictly increasing for the IMU):
    add_camera(t, fix, health)   visual position fix in the world (map) frame, metres, and its
                                 image-health score in [0, 1]. May arrive before or after the IMU
                                 samples that bracket it; it is applied once the IMU has reached t.
    add_imu(t, gyro, acc, ...)   one fused body-frame IMU sample (rad/s, m/s^2 specific force) plus
                                 the monitor's quality fields (``imu_health`` contract).
Output per IMU sample (dict, see ``OUTPUT_FIELDS``): pose p (m, world), velocity v (m/s, world),
orientation q (xyzw, body -> world), position covariance P_pos (3x3, m^2, the filter's own),
velocity covariance P_vel, ``valid``, ``track_state`` / ``camera_state`` / ``imu_state`` codes,
``reasons`` bit mask (``REASONS``) and the counters in ``log``.

Nothing here reads reference trajectories, clean images, injected fault schedules or labels.

Initialisation
    ``ReferenceFreeInit`` buffers the first ``duration_s`` of data and solves position, velocity
    and full orientation from healthy camera fixes and the IMU pre-integrated in the first body
    frame (Gauss-Newton on p_j = p0 + v0 t_j + g t_j^2 / 2 + R0 alpha_j). Gravity fixes roll and
    pitch strongly; yaw is only as observable as the horizontal acceleration in the window, and its
    uncertainty is carried into the initial covariance. Without enough fixes it falls back to an
    IMU-only start (roll/pitch from the mean specific force, yaw = 0, velocity assumed zero, frame
    origin at the start) and says so in ``reasons``. The buffered samples are replayed through the
    filter once the state exists; they are reported invalid (no pose was available in real time).
    Gyro and accelerometer biases start at zero with the filter's prior covariance.
    ``TruthInit`` starts from a given reference pose. It exists for benchmarks and diagnostics and
    marks every output with ``TRUTH_INIT_DIAGNOSTIC``.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import GRAVITY
from .filter import (CAM_DEGRADED, CAM_NOMINAL, CAM_NONE, CAM_OUTAGE, CAM_RECOVERING, CAM_SUSPECT, ESKF,
                     FilterConfig, FilterLog, skew)
from .imu_health import COMMON_MODE, ISOLATED, LOST, SINGLE, UNISOLATED

G = np.array([0.0, 0.0, -GRAVITY])

TRACK_INITIALIZING, TRACK_TRACKING, TRACK_DEGRADED, TRACK_INERTIAL, TRACK_LOST = range(5)
TRACK_NAMES = ("initializing", "tracking", "degraded", "inertial_only", "lost")
REASONS = {
    "CAMERA_OUTAGE": 1 << 0, "CAMERA_DEGRADED": 1 << 1, "CAMERA_SUSPECT_JUMP": 1 << 2,
    "CAMERA_RECOVERING": 1 << 3, "CAMERA_REANCHORED": 1 << 4, "IMU_FAULT_EXCLUDED": 1 << 5,
    "IMU_UNISOLATED": 1 << 6, "IMU_LOST": 1 << 7, "IMU_GAP": 1 << 8, "UNCERTAINTY_LIMIT": 1 << 9,
    "HEADING_ARBITRARY": 1 << 10, "TRUTH_INIT_DIAGNOSTIC": 1 << 11, "NUMERICAL_FAILURE": 1 << 12,
    "VELOCITY_ASSUMED_ZERO": 1 << 13, "POSITION_FRAME_ARBITRARY": 1 << 14, "IMU_COMMON_MODE": 1 << 15,
    "BOTH_DEGRADED": 1 << 16, "IMU_UNVERIFIED": 1 << 17, "NO_CAMERA": 1 << 18, "MODEL_DISTRUSTED": 1 << 19, "COVARIANCE_UNRELIABLE": 1 << 20,
}
OUTPUT_FIELDS = ("t", "p", "v", "q", "P_pos", "P_vel", "valid", "track_state", "camera_state", "imu_state",
                 "reasons")


def reason_names(mask: int) -> list[str]:
    return [k for k, b in REASONS.items() if int(mask) & b]


@dataclass
class TruthInit:
    """Benchmark / diagnostic only: start from a reference pose."""
    p0: np.ndarray
    v0: np.ndarray
    q0: np.ndarray
    label: str = "truth_init_diagnostic"


@dataclass
class ReferenceFreeInit:
    duration_s: float = 3.0        # data buffered before the first state is solved
    max_wait_s: float = 10.0       # with a camera configured, wait this long for enough fixes
    min_fixes: int = 10
    min_health: float = 0.6
    cov_inflate: float = 4.0       # unmodelled errors (biases, fix correlation) on the LS covariance
    tilt_floor_deg: float = 1.0
    yaw_floor_deg: float = 3.0
    vel_floor: float = 0.05
    pos_floor: float = 0.03
    fallback_vel_std: float = 1.5  # IMU-only start: velocity assumed zero with this std (m/s)
    label: str = "reference_free"


def _gravity_tilt(f_body0: np.ndarray) -> Rot:
    """Rotation body0 -> world with zero yaw that maps the mean specific force to +z."""
    u = f_body0 / np.linalg.norm(f_body0)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(u @ z)
    if s < 1e-12:
        R = Rot.identity() if c > 0 else Rot.from_rotvec([np.pi, 0, 0])
    else:
        R = Rot.from_rotvec(v / s * np.arctan2(s, c))
    # remove the yaw this rotation introduces so heading starts at 0
    yaw = np.arctan2(R.as_matrix()[1, 0], R.as_matrix()[0, 0])
    return Rot.from_euler("z", -yaw) * R


def preintegrate(t, gyro, acc):
    """Rotations R_0i (body_i -> body_0) and alpha_i, beta_i (body_0 frame) with held samples."""
    n = len(t)
    Rs = [Rot.identity()]
    alpha = np.zeros((n, 3))
    beta = np.zeros((n, 3))
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        f0 = Rs[i].apply(acc[i])
        alpha[i + 1] = alpha[i] + beta[i] * dt + 0.5 * f0 * dt * dt
        beta[i + 1] = beta[i] + f0 * dt
        Rs.append(Rs[i] * Rot.from_rotvec(gyro[i] * dt))
    return Rs, alpha, beta


def solve_alignment(t, gyro, acc, cam_t, cam_fix, cam_std, cfg: ReferenceFreeInit):
    """Gauss-Newton visual-inertial alignment. Returns dict(p0, v0, R0, P0 (15x15), ok, reasons)."""
    Rs, alpha, beta = preintegrate(t, gyro, acc)
    f0_mean = np.mean([Rs[i].apply(acc[i]) for i in range(len(t))], 0)
    R_tilt = _gravity_tilt(f0_mean)
    tau = cam_t - t[0]
    al = np.stack([np.interp(cam_t, t, alpha[:, j]) for j in range(3)], 1)
    w = 1.0 / np.asarray(cam_std) ** 2
    best = None
    for yaw0 in np.deg2rad([0, 90, 180, 270]):
        R = Rot.from_euler("z", yaw0) * R_tilt
        x = np.zeros(6)
        for _ in range(15):
            Rm = R.as_matrix()
            pred = x[:3] + np.outer(tau, x[3:]) + 0.5 * np.outer(tau ** 2, G) + al @ Rm.T
            r = (cam_fix - pred).reshape(-1)
            J = np.zeros((3 * len(tau), 9))
            for j in range(len(tau)):
                J[3 * j:3 * j + 3, 0:3] = np.eye(3)
                J[3 * j:3 * j + 3, 3:6] = tau[j] * np.eye(3)
                J[3 * j:3 * j + 3, 6:9] = -Rm @ skew(al[j])
            W = np.repeat(w, 3)
            A = J.T @ (J * W[:, None])
            b = J.T @ (W * r)
            try:
                dx = np.linalg.solve(A + 1e-9 * np.eye(9), b)
            except np.linalg.LinAlgError:
                break
            x[:6] += dx[:6]
            R = R * Rot.from_rotvec(dx[6:9])
            if np.linalg.norm(dx) < 1e-9:
                break
        Rm = R.as_matrix()
        pred = x[:3] + np.outer(tau, x[3:]) + 0.5 * np.outer(tau ** 2, G) + al @ Rm.T
        cost = float(np.sum(w[:, None] * (cam_fix - pred) ** 2))
        if best is None or cost < best[0]:
            best = (cost, x.copy(), R, A)
    cost, x, R, A = best
    dof = max(3 * len(tau) - 9, 1)
    s2 = max(cost / dof, 1.0)
    try:
        C = np.linalg.inv(A) * s2 * cfg.cov_inflate
    except np.linalg.LinAlgError:
        return {"ok": False}
    if not np.all(np.isfinite(C)) or np.max(np.diag(C)) > 1e6:
        return {"ok": False}
    P0 = np.zeros((15, 15))
    P0[0:3, 0:3] = C[6:9, 6:9]
    P0[3:6, 3:6] = C[3:6, 3:6]
    P0[6:9, 6:9] = C[0:3, 0:3]
    P0[0:3, 3:6] = C[6:9, 3:6]
    P0[3:6, 0:3] = C[3:6, 6:9]
    P0[0:3, 6:9] = C[6:9, 0:3]
    P0[6:9, 0:3] = C[0:3, 6:9]
    P0[3:6, 6:9] = C[3:6, 0:3]
    P0[6:9, 3:6] = C[0:3, 3:6]
    floor = np.r_[[np.deg2rad(cfg.tilt_floor_deg)] * 2, np.deg2rad(cfg.yaw_floor_deg), [cfg.vel_floor] * 3,
                  [cfg.pos_floor] * 3]
    P0[:9, :9] += np.diag(floor ** 2)
    P0[9:12, 9:12] = np.eye(3) * 1e-4
    P0[12:15, 12:15] = np.eye(3) * 1e-2
    return {"ok": True, "p0": x[:3], "v0": x[3:6], "R0": R, "P0": 0.5 * (P0 + P0.T),
            "rms_norm": float(np.sqrt(cost / max(len(tau), 1)))}


class StreamingEstimator:
    def __init__(self, cfg: FilterConfig = FilterConfig(), init=None, predictor=None,
                 expect_camera: bool = True):
        self.cfg = cfg
        self.init = init if init is not None else ReferenceFreeInit()
        self.predictor = predictor
        self.expect_camera = expect_camera
        self.log = FilterLog()
        self.f: ESKF | None = None
        self.pending: list = []                 # camera events (t, fix, health), sorted by t
        self.cur_t = None
        self.held = None                        # (gyro, acc, conf, q_add_g, q_add_a) for propagation
        self.k = 0
        self.W = int(getattr(predictor, "window", 200))
        self.every = int(getattr(predictor, "every", 20))
        self.pred_rate = float(getattr(predictor, "rate", 200.0))
        self.hist = deque(maxlen=self.W + 1)    # per sample: (t, R, p, P_pos, gyro, acc, imu_bad)
        self.boot: list = []                    # samples buffered before initialisation
        self.base_reasons = 0
        self.last_reanchor_t = -np.inf
        self.gap_flag = False

    # ------------------------------------------------------------------ inputs
    def add_camera(self, t, fix, health):
        if not np.isfinite(t):
            self.log.cam_invalid += 1
            return
        if self.cur_t is not None and self.f is not None and t <= self.cur_t:
            self.log.cam_late += 1                 # older than the state: cannot be applied causally
            return
        ev = (float(t), np.asarray(fix, float), float(health))
        i = len(self.pending)
        while i > 0 and self.pending[i - 1][0] > ev[0]:
            i -= 1
        self.pending.insert(i, ev)

    def add_imu(self, t, gyro, acc, q_add_g: float = 0.0, q_add_a: float = 0.0, imu_state=None,
                imu_conf: float = 1.0) -> list:
        t = float(t)
        gyro = np.asarray(gyro, float).reshape(-1)
        acc = np.asarray(acc, float).reshape(-1)
        if gyro.shape != (3,) or acc.shape != (3,):
            raise ValueError("gyro and acc must each have 3 components")
        if not np.isfinite(t):
            raise ValueError("IMU timestamp must be finite")
        last = self.boot[-1][0] if self.f is None and self.boot else self.cur_t
        if last is not None and t <= last:
            raise ValueError(f"IMU timestamps must increase strictly ({t} after {last})")
        sample = (t, gyro, acc, float(q_add_g), float(q_add_a), imu_state, float(imu_conf))
        if self.f is None:
            self.boot.append(sample)
            return self._try_init()
        return [self._step(*sample)]

    def finish(self) -> list:
        """End of stream: samples still waiting for initialisation are reported invalid."""
        if self.f is None and self.boot:
            out = [self._invalid(s[0], s[5]) for s in self.boot]
            self.boot = []
            return out
        return []

    # ------------------------------------------------------------------ initialisation
    def _try_init(self) -> list:
        ini = self.init
        if isinstance(ini, TruthInit):
            s0 = self.boot[0]
            self.f = ESKF(self.cfg, ini.p0, ini.v0, ini.q0)
            self.base_reasons |= REASONS["TRUTH_INIT_DIAGNOSTIC"]
            self.cur_t = s0[0]
            boot, self.boot = self.boot, []
            return [self._step(*s, first=(i == 0)) for i, s in enumerate(boot)]
        t0, tl = self.boot[0][0], self.boot[-1][0]
        if tl - t0 < ini.duration_s:
            return []
        ts = np.array([s[0] for s in self.boot])
        g = np.array([s[1] for s in self.boot])
        a = np.array([s[2] for s in self.boot])
        fin = np.all(np.isfinite(g), 1) & np.all(np.isfinite(a), 1)
        if fin.sum() < 0.9 * len(ts):
            if tl - t0 < ini.max_wait_s:
                return []
        g = np.where(fin[:, None], g, 0.0)
        a = np.where(fin[:, None], a, np.nanmean(a[fin], 0) if fin.any() else np.array([0, 0, GRAVITY]))
        cams = [e for e in self.pending if t0 <= e[0] <= tl and np.all(np.isfinite(e[1])) and e[2] >= ini.min_health]
        sol = None
        if len(cams) >= ini.min_fixes:
            ct = np.array([e[0] for e in cams])
            cf = np.array([e[1] for e in cams])
            cstd = np.array([self._cam_std(e[2]) for e in cams])
            sol = solve_alignment(ts, g, a, ct, cf, cstd, ini)
            if not sol.get("ok"):
                sol = None
        if sol is None and self.expect_camera and tl - t0 < ini.max_wait_s:
            return []                                    # keep waiting for usable fixes
        if sol is not None:
            self.f = ESKF(self.cfg, sol["p0"], sol["v0"], sol["R0"].as_quat(), P0=sol["P0"])
            # The alignment was anchored by healthy fixes up to the last one it used.
            self.f.last_good_t = self.f.last_used_t = float(cams[-1][0])
            self.f.good_since = float(cams[0][0])
            self.f.last_used_healthy = True
            self.log.events.append((tl, "init_aligned", f"rms={sol['rms_norm']:.3f}"))
        else:
            Rs, _, _ = preintegrate(ts, g, a)
            f0 = np.mean([Rs[i].apply(a[i]) for i in range(len(ts))], 0)
            R0 = _gravity_tilt(f0)
            P0 = np.zeros((15, 15))
            P0[0:2, 0:2] = np.eye(2) * np.deg2rad(3.0) ** 2
            P0[2, 2] = np.pi ** 2
            P0[3:6, 3:6] = np.eye(3) * ini.fallback_vel_std ** 2
            P0[6:9, 6:9] = np.eye(3) * 1e-6
            P0[9:12, 9:12] = np.eye(3) * 1e-4
            P0[12:15, 12:15] = np.eye(3) * 1e-2
            self.f = ESKF(self.cfg, np.zeros(3), np.zeros(3), R0.as_quat(), P0=P0)
            self.base_reasons |= (REASONS["HEADING_ARBITRARY"] | REASONS["VELOCITY_ASSUMED_ZERO"]
                                  | REASONS["POSITION_FRAME_ARBITRARY"])
            self.log.events.append((tl, "init_imu_only", "no usable camera fixes"))
        # Fixes used by the alignment are consumed: replaying them would count them twice.
        self.pending = [e for e in self.pending if e[0] > tl]
        boot, self.boot = self.boot, []
        self.cur_t = boot[0][0]
        out = []
        for i, s in enumerate(boot):
            r = self._step(*s, first=(i == 0))
            out.append(self._invalid(r["t"], s[5], r))
        return out

    def _cam_std(self, health):
        s = ESKF(self.cfg, np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1.0])).camera_std(health)
        return s if s is not None else 1.0

    def _invalid(self, t, imu_state, r=None):
        o = {"t": t, "p": np.full(3, np.nan), "v": np.full(3, np.nan), "q": np.full(4, np.nan),
             "P_pos": np.full((3, 3), np.nan), "P_vel": np.full((3, 3), np.nan), "valid": False,
             "track_state": TRACK_INITIALIZING, "camera_state": CAM_NONE,
             "imu_state": -1 if imu_state is None else int(imu_state), "reasons": self.base_reasons}
        return o

    # ------------------------------------------------------------------ main step
    def _advance(self, t_target):
        dt = t_target - self.cur_t
        if dt <= 0 or self.held is None:
            self.cur_t = max(self.cur_t, t_target)
            return
        g, a, conf, qg, qa = self.held
        c = self.cfg
        if dt > c.max_imu_gap_s:
            # Missing IMU data: hold the last sample in sub-steps and add the motion envelope.
            self.gap_flag = True
            n = int(np.ceil(dt / c.max_imu_gap_s))
            for _ in range(n):
                self.f.propagate(g, a, dt / n, conf, qg + c.gap_sigma_g ** 2, qa + c.gap_sigma_a ** 2)
        else:
            self.f.propagate(g, a, dt, conf, qg, qa)
        self.cur_t = t_target

    def _step(self, t, gyro, acc, q_add_g, q_add_a, imu_state, imu_conf, first=False):
        f, c, log = self.f, self.cfg, self.log
        reasons = self.base_reasons
        self.gap_flag = False
        while self.pending and self.pending[0][0] <= t:
            tc, fix, health = self.pending.pop(0)
            if (not first and tc <= self.cur_t) or (first and tc < t):
                log.cam_late += 1
                continue
            self._advance(tc)
            if f.camera_update(fix, health, log, tc) == "reanchored":
                reasons |= REASONS["CAMERA_REANCHORED"]
        self._advance(t)
        if self.gap_flag:
            reasons |= REASONS["IMU_GAP"]
        finite = np.all(np.isfinite(gyro)) and np.all(np.isfinite(acc))
        lost = (imu_state == LOST) or not finite
        if finite:
            self.held = (gyro, acc, imu_conf, q_add_g, q_add_a)
        elif self.held is not None:
            g, a, cf, _, _ = self.held
            self.held = (g, a, cf, max(q_add_g, c.gap_sigma_g ** 2), max(q_add_a, c.gap_sigma_a ** 2))
        else:
            self.held = (np.zeros(3), -G.copy(), 1.0, c.gap_sigma_g ** 2, c.gap_sigma_a ** 2)

        k = self.k
        pred = self.predictor
        if pred is not None and k >= self.W and k % self.every == 0:
            k0 = k - self.W
            win = list(self.hist)[-self.W:]
            ts = np.array([h[0] for h in win] + [t])
            dts = np.diff(ts)
            nominal = 1.0 / self.pred_rate
            bad = any(h[6] for h in win) or np.max(np.abs(dts - nominal)) > c.disp_rate_tol * nominal
            if bad:
                log.disp_skipped += 1
            else:
                Rs = np.array([h[1] for h in win])
                gw = np.einsum("nij,nj->ni", Rs, np.array([h[4] for h in win]))
                aw = np.einsum("nij,nj->ni", Rs, np.array([h[5] for h in win]))
                if c.cloning and c.yaw_frame and k0 in f.clone_k:
                    Rz = f.clone_yaw(k0)          # into the clone's yaw frame (row vectors: v @ Rz)
                    gw, aw = gw @ Rz, aw @ Rz
                d, cov = pred(np.concatenate([gw, aw], 1))
                cov = np.asarray(cov) * c.disp_cov_scale
                log.disp_std.append(float(np.sqrt(np.trace(cov))) if np.all(np.isfinite(cov)) else np.nan)
                if c.cloning:
                    f.displacement_update_clone(d, cov, k0, log, t)
                else:
                    h0 = win[0]
                    f.displacement_update(d, cov, h0[2], h0[3], log, t)
                log.model_scale.append(f.model_scale)
        if pred is not None and c.cloning and k % self.every == 0:
            f.add_clone(k, self.W // self.every + 1)

        Rm = f.R.as_matrix()
        self.hist.append((t, Rm, f.p.copy(), f.P[6:9, 6:9].copy(),
                          gyro if finite else self.held[0], acc if finite else self.held[1], lost))
        self.k += 1
        return self._output(t, imu_state, lost, reasons)

    def _output(self, t, imu_state, lost, reasons):
        f, c = self.f, self.cfg
        cam = f.camera_state(t) if self.expect_camera else CAM_NONE
        Pp = f.position_cov()
        if cam == CAM_OUTAGE:
            reasons |= REASONS["CAMERA_OUTAGE"]
        elif cam == CAM_DEGRADED:
            reasons |= REASONS["CAMERA_DEGRADED"]
        elif cam == CAM_SUSPECT:
            reasons |= REASONS["CAMERA_SUSPECT_JUMP"]
        elif cam == CAM_RECOVERING:
            reasons |= REASONS["CAMERA_RECOVERING"]
        elif cam == CAM_NONE:
            reasons |= REASONS["NO_CAMERA"]
        imu_bad = False
        if lost:
            reasons |= REASONS["IMU_LOST"]
            imu_bad = True
        elif imu_state == ISOLATED:
            reasons |= REASONS["IMU_FAULT_EXCLUDED"]
            imu_bad = True
        elif imu_state == UNISOLATED:
            reasons |= REASONS["IMU_UNISOLATED"]
            imu_bad = True
        elif imu_state == COMMON_MODE:
            reasons |= REASONS["IMU_COMMON_MODE"]
            imu_bad = True
        elif imu_state is None or imu_state == SINGLE:
            reasons |= REASONS["IMU_UNVERIFIED"]
        distrusted = self.predictor is not None and f.model_distrusted()
        if distrusted:
            reasons |= REASONS["MODEL_DISTRUSTED"]
        if cam not in (CAM_NOMINAL,) and imu_bad:
            reasons |= REASONS["BOTH_DEGRADED"]
        if cam == CAM_NOMINAL:
            state = TRACK_DEGRADED if imu_bad else TRACK_TRACKING
        elif cam == CAM_DEGRADED:
            state = TRACK_DEGRADED
        else:
            state = TRACK_INERTIAL
        worst_std = float(np.sqrt(max(np.linalg.eigvalsh(0.5 * (Pp + Pp.T)).max(), 0.0))) \
            if np.all(np.isfinite(Pp)) else np.inf
        if f.diverged:
            reasons |= REASONS["NUMERICAL_FAILURE"]
            state = TRACK_LOST
        elif worst_std > c.max_valid_pos_std:
            reasons |= REASONS["UNCERTAINTY_LIMIT"]
            state = TRACK_LOST
        # Without the camera anchoring the state, its covariance rests on the learned model. If the self-check
        # has measured that model as inconsistent, the covariance cannot be trusted: the estimate is still
        # output (it may be the best available), but it is not declared valid.
        unreliable = distrusted and cam not in (CAM_NOMINAL, CAM_DEGRADED) and state != TRACK_LOST
        if unreliable:
            reasons |= REASONS["COVARIANCE_UNRELIABLE"]
        self.log.p.append(f.p.copy())
        self.log.q_scale.append(f.q_scale)
        self.log.p_std.append(float(np.sqrt(np.trace(Pp))) if np.all(np.isfinite(Pp)) else np.inf)
        return {"t": t, "p": f.p.copy(), "v": f.v.copy(), "q": f.R.as_quat(), "P_pos": Pp,
                "P_vel": f.P[3:6, 3:6].copy(), "valid": state not in (TRACK_INITIALIZING, TRACK_LOST) and not unreliable,
                "track_state": state, "camera_state": cam,
                "imu_state": -1 if imu_state is None else int(imu_state), "reasons": int(reasons)}


def stack_outputs(rows: list) -> dict:
    if not rows:
        return {"t": np.zeros(0), "p": np.zeros((0, 3)), "v": np.zeros((0, 3)), "q": np.zeros((0, 4)),
                "P_pos": np.zeros((0, 3, 3)), "P_vel": np.zeros((0, 3, 3)), "valid": np.zeros(0, bool),
                "track_state": np.zeros(0, int), "camera_state": np.zeros(0, int),
                "imu_state": np.zeros(0, int), "reasons": np.zeros(0, int)}
    return {k: np.array([r[k] for r in rows]) for k in OUTPUT_FIELDS}


def replay(t, gyro, acc, cfg: FilterConfig, init, cam=None, predictor=None, imu_quality=None, imu_conf=None):
    """Feed recorded arrays through ``StreamingEstimator`` in timestamp order.

    cam: dict with ``t``, ``fix`` (M, 3), ``health`` (M,) or None. Each camera event is added just
    before the first IMU sample at or after its timestamp, as a live system would see it.
    Returns (positions (N, 3) with NaN where invalid-at-init, FilterLog, stacked outputs).
    """
    t = np.asarray(t, float)
    gyro, acc = np.asarray(gyro, float), np.asarray(acc, float)
    N = len(t)
    if gyro.shape != (N, 3) or acc.shape != (N, 3):
        raise ValueError(f"gyro and acc must be (N={N}, 3)")
    est = StreamingEstimator(cfg, init, predictor, expect_camera=cam is not None)
    rows = []
    ci = 0
    ct = np.asarray(cam["t"], float) if cam is not None else np.zeros(0)
    order = np.argsort(ct, kind="stable")
    for i in range(N):
        while ci < len(order) and ct[order[ci]] <= t[i]:
            j = order[ci]
            est.add_camera(ct[j], cam["fix"][j], cam["health"][j])
            ci += 1
        qg = qa = 0.0
        st = None
        if imu_quality is not None:
            qg, qa = float(imu_quality["q_add_g"][i]), float(imu_quality["q_add_a"][i])
            st = int(imu_quality["state"][i])
        cf = 1.0 if imu_conf is None else float(imu_conf[i])
        rows += est.add_imu(t[i], gyro[i], acc[i], qg, qa, st, cf)
    rows += est.finish()
    out = stack_outputs(rows)
    if len(rows) != N:
        raise RuntimeError("estimator output does not align with the IMU input")
    pos = np.where(out["track_state"][:, None] == TRACK_INITIALIZING, np.nan, out["p"]) if N else np.zeros((0, 3))
    est.log.outputs = out
    return pos, est.log, out
