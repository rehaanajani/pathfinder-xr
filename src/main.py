#!/usr/bin/env python3
"""Path Finder — Track 3 entry: degraded IMU CSV -> 6-DoF trajectory (TUM format).

    python src/main.py <input> [-o estimated_trajectory.txt] [--extend-blind 5]

<input> is an EuRoC-format folder/zip containing imu0/data.csv, or the CSV itself:
    #timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z [rad/s], a_RS_S_x, a_RS_S_y, a_RS_S_z [m/s^2, incl. gravity]
Output: one line per input IMU row, "timestamp_s tx ty tz qx qy qz qw" (space-separated, world frame,
gravity along -z; origin and yaw are fixed by the first sample, since an IMU alone cannot observe them).

Pipeline (single pass over the samples, single thread, standard library only):
  1. Sample health: non-finite, all-zero, frozen (bit-identical repeats) or physically impossible readings are
     marked invalid; timestamp gaps > 3x the nominal period count as dropouts.
  2. Initialisation without ground truth: gravity direction and gyro bias from the first still period
     (unknown gravity alignment), yaw = 0, position = 0.
  3. Strapdown integration on SO(3) with gyro-bias correction. While the IMU is still (zero-velocity detector),
     velocity is pulled to zero and the gyro bias keeps being refined.
  4. Drift control: a soft speed limit and a velocity leak that grows while readings are unhealthy, so noise and
     bias drift can't run the position away.
  5. Dropouts and the final IMU failure: dead reckoning from the last trusted state — constant turn rate decaying
     to zero, velocity decaying with a time constant, never faster than the speed limit.
  6. Learned velocity (default when src/velnet_weights.bin is present, needs numpy; --model none turns it off): a
     small learned network reads the last 1 s of IMU (body frame with gravity kept plus our "up" direction,
     docs/AIRIO.md; velnet_weights_world.bin: the earlier gravity-aligned input) and its velocity is used as a measurement; this also switches the filter to the
     "learned" preset (see PRESETS and docs/LEARNED_VELOCITY.md). --model none --preset physics is the original filter.
"""

import argparse
import io
import math
import sys
import zipfile
from pathlib import Path

G = 9.80665
MOUNT_G_TOL = 0.05                            # |mean specific force| / g - 1 above which the mount guard holds off
# Gravity ("up") in the IMU frame at the start of the recordings the body-frame network was trained on: EuRoC (all 11
# sequences within ~6 deg of this) and TUM-VI (rooms 1-6). The mount guard compares a new file against these.
TRAIN_UP = ((0.9412, -0.0101, -0.3376), (0.0160, -0.0040, 0.9999))


# ---------------------------------------------------------------------------- small vector / quaternion helpers
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / n for c in q)


def qexp(w):                      # rotation vector -> quaternion (w, x, y, z)
    th = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    if th < 1e-12:
        return (1.0, 0.5 * w[0], 0.5 * w[1], 0.5 * w[2])
    s = math.sin(0.5 * th) / th
    return (math.cos(0.5 * th), w[0] * s, w[1] * s, w[2] * s)


def qrot(q, v):                   # rotate v by q (body -> world)
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def q_from_two(u, v):             # shortest rotation taking unit u to unit v
    d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    c = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    if d < -0.999999:
        axis = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        c = (u[1] * axis[2] - u[2] * axis[1], u[2] * axis[0] - u[0] * axis[2], u[0] * axis[1] - u[1] * axis[0])
        return qnorm((0.0,) + c)
    return qnorm((1.0 + d,) + c)


def norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


# ---------------------------------------------------------------------------- input
def open_csv(path):
    p = Path(path)
    if p.is_dir():
        for cand in (p / "imu0" / "data.csv", p / "mav0" / "imu0" / "data.csv"):
            if cand.exists():
                return open(cand, "r", newline="")
        raise SystemExit(f"no imu0/data.csv under {p}")
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        name = next(n for n in z.namelist() if n.endswith("imu0/data.csv"))
        return io.TextIOWrapper(z.open(name), newline="")
    return open(p, "r", newline="")


def detect_layout(inp, n_max=2000, columns="auto", ts_unit="auto", accel_unit="auto"):
    """Column order and units (EuRoC: ns, gyro first, accel m/s^2; the organisers' degradation script documents
    accel-first columns and accepts s/ms/us/ns timestamps and accel in g). Each can be forced; "auto" decides from the
    first data rows:
    - accelerometer: a triplet whose median magnitude is 6-14 (m/s^2, gravity) and the other's is not; only if neither
      looks like m/s^2, a triplet at 0.6-1.4 (g) while the other is not. Otherwise the EuRoC order.
    - timestamps: values above 1e17 can only be epoch nanoseconds; otherwise the unit follows the median spacing of
      all rows (as the organisers' script does). A unit that would overflow int64 falls back to ns.
    Returns (accel_first, t_scale_to_ns, accel_scale_to_m_s2, note)."""
    t, x, y = [], [], []
    with open_csv(inp) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] == "#" or s[0].isalpha():
                continue
            p = s.replace(";", ",").split(",")
            try:
                t0 = float(p[0])
            except ValueError:
                continue
            if not math.isfinite(t0):
                continue
            t.append(t0)
            try:
                v = [float(c) for c in p[1:7]]
            except ValueError:
                v = []
            if len(v) == 6 and all(math.isfinite(c) for c in v) and any(v):
                x.append(norm3(v[0:3])); y.append(norm3(v[3:6]))
            if len(t) >= n_max:
                break
    med = lambda z: sorted(z)[len(z) // 2]
    notes = []
    accel_first, a_scale = False, 1.0
    si = lambda n: 6.0 <= n <= 14.0                 # gravity in m/s^2
    gu = lambda n: 0.6 <= n <= 1.4                  # gravity in g
    mx, my = (med(x), med(y)) if len(x) >= 10 else (None, None)
    if columns != "auto":
        accel_first = columns == "accel-first"
    elif mx is None:
        notes.append("too few rows to check columns, using gyro first")
    else:
        # a forced accel unit narrows the test to that unit; otherwise m/s^2 first, then g
        bands = [si] if accel_unit == "ms2" else [gu] if accel_unit == "g" else [si, gu]
        decided = False
        for band in bands:
            if band(mx) != band(my):
                accel_first, decided = band(mx), True
                if band is gu and accel_unit == "auto":
                    a_scale = G
                break
            if band is si and (si(mx) or si(my)):  # both look like m/s^2: g cannot settle it
                break
        if not decided:
            notes.append("column order unclear, using gyro first")
    if accel_unit == "g":
        a_scale = G
    elif accel_unit == "ms2":
        a_scale = 1.0
    elif columns != "auto" and mx is not None:     # forced order, automatic unit: judge the accel triplet
        na = mx if accel_first else my
        a_scale = G if gu(na) and not si(na) else 1.0
    t_scale = 1
    if ts_unit != "auto":
        t_scale = {"ns": 1, "us": 10 ** 3, "ms": 10 ** 6, "s": 10 ** 9}[ts_unit]
    elif t and max(abs(c) for c in t) < 1e17:
        dts = [b - a for a, b in zip(t, t[1:]) if b > a]
        if dts:
            dt = med(dts)
            # the unit that makes the sampling rate believable for an IMU (10 Hz - 10 kHz); units are 1000x apart,
            # so exactly one fits. (A spacing threshold alone fails on jittered ~1 kHz ns data, e.g. 995,200 ns.)
            t_scale = 1
            for sc in (1, 10 ** 3, 10 ** 6, 10 ** 9):
                if 10.0 <= 1e9 / (dt * sc) < 10000.0:
                    t_scale = sc
                    break
            if max(abs(c) for c in t) * t_scale >= 9.2e18:
                t_scale = 1
                notes.append("timestamp unit would overflow, using ns")
    return accel_first, t_scale, a_scale, "; ".join(notes)


def parse_rows(f, layout=(False, 1, 1.0, "")):
    """Yield (t_ns, gyro(3), acc(3)) per data row, in our order and units whatever the file's (see detect_layout).
    Malformed numbers become NaN (row kept, marked bad)."""
    accel_first, t_scale, a_scale = layout[:3]
    for line in f:
        s = line.strip()
        if not s or s[0] == "#" or s[0].isalpha():
            continue
        parts = s.replace(";", ",").split(",")
        try:
            if t_scale == 1:
                t = int(float(parts[0])) if "e" in parts[0].lower() or "." in parts[0] else int(parts[0])
            else:
                t = int(round(float(parts[0]) * t_scale))
        except (ValueError, OverflowError):
            continue
        if not -(1 << 63) <= t < (1 << 63):         # as the C++ entry: timestamps must fit in int64
            continue
        vals = []
        for x in parts[1:7]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float("nan"))
        while len(vals) < 6:
            vals.append(float("nan"))
        g, a = (vals[3:6], vals[:3]) if accel_first else (vals[:3], vals[3:6])
        if a_scale != 1.0:
            a = [c * a_scale for c in a]
        yield t, tuple(g), tuple(a)


# ---------------------------------------------------------------------------- the estimator
class PathFinder:
    def __init__(self, a):
        self.a = a
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.bg = [0.0, 0.0, 0.0]
        self.ba = [0.0, 0.0, 0.0]
        self.w_last = (0.0, 0.0, 0.0)
        self.t_prev = None
        self.init_buf = []
        self.initialised = False
        self.prev_raw = None
        self.frozen_run = 0
        self.still_g, self.still_a = [], []       # short sliding window for the zero-velocity detector
        self.still_w, self.still_s, self.still_ss = [], [0.0] * 6, [0.0] * 6
        self.bad_time = 0.0                       # seconds since the last trusted sample
        self.good_time = 1e9                      # seconds of healthy data since the last dropout
        self.bias_known = False                   # gyro bias already taken from a still stretch (prescan)
        self.f_lp = None                          # low-passed specific force (body frame)
        self.yaw_var = 0.0                        # rad^2, how far our heading may have drifted since the start
        self.dt_nom = None
        self.acc_res = [0.0, 0.0, 0.0]            # slow world-frame residual acceleration (bias + tilt leak)
        # Learned velocity model (optional). The last `window` time slots of gravity-aligned IMU are kept in a
        # ring; a slot is all-zero (valid flag 0) for unhealthy or missing samples, so the window is always on a
        # uniform time grid and the network can be called again shortly after a dropout.
        self.net = None
        self.rec = None                           # tools/train_velocity.py sets a list here to record slots
        self.W, self.dec = a.model_window, a.model_dec
        self.rate_mult = 1                        # input samples averaged per network sample (1 kHz input, 200 Hz net: 5)
        self.v_blind_tau = a.blind_vel_tau
        # Network input: "world" (weights version 2, 7 channels: gravity-aligned gyro, specific force minus g, health)
        # or "body" (version 3, 10 channels, AirIO style: body-frame gyro and specific force with gravity kept, the
        # gravity direction in the body frame from our own orientation, health; outputs in the body frame).
        self.body = a.model_input == "body"
        if a.model:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from velnet import VelNet
            self.net = VelNet(a.model)
            self.W, self.dec = self.net.window, self.net.dec
            self.body = self.net.version >= 3
            if a.model_blind_tau > 0:
                self.v_blind_tau = a.model_blind_tau
        self.ring = None
        self.ring_i = 0                           # next slot to write
        self.ring_valid = 0                       # valid slots currently in the window
        self.since = 0
        self.net_ok = False                       # the network has produced a velocity since the last long gap

    # -- health of one reading
    def healthy(self, g, a):
        vals = g + a
        if any(not math.isfinite(x) for x in vals):
            return False
        if all(x == 0.0 for x in vals):
            return False
        raw = vals
        if self.prev_raw is not None and raw == self.prev_raw:
            self.frozen_run += 1
        else:
            self.frozen_run = 0
        self.prev_raw = raw
        if self.frozen_run >= 3:                  # bit-identical repeats: dropout held at the last value
            return False
        na = norm3(a)
        if na > self.a.acc_max or norm3(g) > self.a.gyro_max:
            return False
        if na < self.a.acc_min:                   # no gravity at all: a dead accelerometer, not motion
            return False
        return True

    def init_from(self, buf):
        n = len(buf)
        ma = [sum(b[1][i] for b in buf) / n for i in range(3)]
        mg = [sum(b[0][i] for b in buf) / n for i in range(3)]
        # Gravity alignment: rotate the measured specific force (body) onto +z (world).
        if norm3(ma) > 1e-6:                      # (healthy samples always carry gravity; guard anyway)
            self.q = q_from_two(tuple(c / norm3(ma) for c in ma), (0.0, 0.0, 1.0))
            self.check_mount(ma)
        var_g = sum(sum((b[0][i] - mg[i]) ** 2 for i in range(3)) for b in buf) / n
        if var_g < self.a.still_gyro_var and not self.bias_known:   # still at start: the mean gyro is its bias
            self.bg = mg
        self.initialised = True

    def check_mount(self, ma):
        """Mount guard: the body-frame network learned the training IMUs' axes. If gravity at the start points far
        from every training mount (the IMU axes were rotated), use the gravity-aligned network instead, which does
        not depend on how the IMU is mounted. A rotation about gravity itself cannot be seen here."""
        A = self.a
        if self.net is None or not self.body or A.mount_guard_deg <= 0 or not A.fallback_model:
            return
        n = norm3(ma)
        u = [c / n for c in ma]
        cos = max(sum(u[i] * m[i] for i in range(3)) / norm3(m) for m in TRAIN_UP)
        self.mount_angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        if self.mount_angle > A.mount_guard_deg:
            if abs(n / G - 1.0) > MOUNT_G_TOL:     # sustained acceleration tilts the mean; that is not a mount
                print(f"mount guard: start-up specific force {n:.2f} m/s^2 is not gravity alone; "
                      "keeping the body-frame network", file=sys.stderr)
                return
            from velnet import VelNet
            try:
                fb = VelNet(A.fallback_model)
            except Exception as e:                # unusable fallback: keep the body-frame network (as the C++ entry)
                print(f"mount guard: cannot load {A.fallback_model} ({e}); keeping the body-frame network",
                      file=sys.stderr)
                return
            self.net = fb
            # The heavier network coupling of the "learned" preset was chosen with the body-frame network on its
            # training mounts; an unfamiliar mount gets the gravity-aligned network and the coupling it was validated
            # with (fresh VR-headset recordings: 32 % better than standing still; the heavier coupling was worse).
            import copy
            self.a = A = copy.copy(A)
            for k, v in FALLBACK_COUPLING.items():
                setattr(A, k, v)
            self.W, self.dec = self.net.window, self.net.dec
            self.body = self.net.version >= 3
            print(f"IMU mount {self.mount_angle:.0f} deg from the training mounts: using {Path(A.fallback_model).name}",
                  file=sys.stderr)

    def zero_velocity(self, g, a):
        A = self.a
        w = A.zupt_window
        if A.zupt_gyro_std > 0 and w <= 0:
            return False
        if A.zupt_gyro_std > 0:
            # Still = the readings barely change (per-axis standard deviation over the window). Unlike a test on
            # the gyro magnitude this also works when the gyro has a large constant bias, and the window mean is
            # then exactly that bias.
            x = g + a
            self.still_w.append(x)
            for i in range(6):
                self.still_s[i] += x[i]
                self.still_ss[i] += x[i] * x[i]
            if len(self.still_w) > w:
                y = self.still_w.pop(0)
                for i in range(6):
                    self.still_s[i] -= y[i]
                    self.still_ss[i] -= y[i] * y[i]
            if len(self.still_w) < w:
                return False
            var = [max(0.0, self.still_ss[i] / w - (self.still_s[i] / w) ** 2) for i in range(6)]
            if max(var[:3]) < A.zupt_gyro_std ** 2 and max(var[3:]) < A.zupt_acc_std ** 2:
                self.still_gyro = (self.still_s[0] / w, self.still_s[1] / w, self.still_s[2] / w)
                return True
            return False
        self.still_g.append(norm3(g))
        self.still_a.append(norm3(a) - G)
        if len(self.still_g) > w:
            self.still_g.pop(0)
            self.still_a.pop(0)
        if len(self.still_g) < w:
            return False
        mg = sum(self.still_g) / w
        ma = sum(abs(x) for x in self.still_a) / w
        self.still_gyro = g
        return mg < self.a.zupt_gyro and ma < self.a.zupt_acc

    def step(self, t, g, a, ok):
        """Advance to time t; if anything turns non-finite, keep the last finite state (never emit NaN)."""
        saved = (self.q, list(self.p), list(self.v), list(self.bg), tuple(self.w_last), list(self.acc_res))
        self._step(t, g, a, ok)
        if not all(math.isfinite(c) for c in (*self.q, *self.p, *self.v, *self.bg, *self.w_last, *self.acc_res)):
            self.q, self.p, self.v, self.bg, self.w_last, self.acc_res = saved
            self.v = [0.0, 0.0, 0.0]

    def _step(self, t, g, a, ok):
        """Advance to time t with reading (g, a); ok=False means dead-reckon through this sample."""
        A = self.a
        if self.t_prev is None:
            self.t_prev = t
            return
        dt = (t - self.t_prev) * 1e-9
        self.t_prev = t
        if dt <= 0:
            return
        if self.dt_nom is None or (ok and dt < 2 * self.dt_nom):
            self.dt_nom = dt if self.dt_nom is None else 0.99 * self.dt_nom + 0.01 * dt
        gap = dt > 3 * self.dt_nom
        if ok and not gap:
            self.bad_time = 0.0
            self.yaw_var += A.yaw_q * dt
            w = [g[0] - self.bg[0], g[1] - self.bg[1], g[2] - self.bg[2]]
            w_raw = (w[0], w[1], w[2])            # bias-corrected gyro before the gravity correction (body input)
            f = (a[0] - self.ba[0], a[1] - self.ba[1], a[2] - self.ba[2])
            # Gravity correction (Mahony complementary filter): when the specific force is close to 1 g, pull the
            # estimated "down" toward the measured one and learn the gyro bias from the disagreement.
            fn = norm3(f)
            # Right after a dropout the tilt is stale (the head kept turning while we saw nothing): for a short
            # while follow the accelerometer much faster, then settle back to the slow, smooth correction.
            boost = A.kp_boost * math.exp(-self.good_time / A.kp_boost_tau) if A.kp_boost > 0 else 0.0
            fl = f
            if A.tilt_lp > 0:                     # smooth out vibration before using gravity as a reference
                if self.f_lp is None or self.good_time < dt:
                    self.f_lp = f                 # (re)start after a dropout
                else:
                    kl = min(1.0, dt / A.tilt_lp)
                    self.f_lp = tuple(self.f_lp[i] + kl * (f[i] - self.f_lp[i]) for i in range(3))
                fl = self.f_lp
                fn = norm3(fl)
            self.good_time += dt
            if fn > 1e-6 and A.kp + boost > 0:
                trust = max(0.0, 1.0 - abs(fn - G) / A.tilt_acc_tol)
                if boost > 0:
                    trust = max(trust, 1.0 - abs(fn - G) / (3.0 * A.tilt_acc_tol), 0.0)
                if trust > 0:
                    up_b = qrot((self.q[0], -self.q[1], -self.q[2], -self.q[3]), (0.0, 0.0, 1.0))
                    fu = (fl[0] / fn, fl[1] / fn, fl[2] / fn)
                    e = (fu[1] * up_b[2] - fu[2] * up_b[1], fu[2] * up_b[0] - fu[0] * up_b[2],
                         fu[0] * up_b[1] - fu[1] * up_b[0])
                    tb = trust
                    if A.ki_calm > 0:             # learn the gyro bias only in calm moments (slow turn, ~1 g):
                        wn = norm3(w)             # fast hand motion would otherwise leak into the bias
                        tb = max(0.0, 1.0 - abs(fn - G) / (A.tilt_acc_tol / 3.0)) * max(0.0, 1.0 - wn / A.ki_calm)
                    for i in range(3):
                        w[i] += (A.kp + boost) * trust * e[i]
                        self.bg[i] -= A.ki * tb * e[i] * dt
            q_mid = qnorm(qmul(self.q, qexp((0.5 * w[0] * dt, 0.5 * w[1] * dt, 0.5 * w[2] * dt))))
            acc_w = qrot(q_mid, f)
            acc_w = [acc_w[0], acc_w[1], acc_w[2] - G]
            # Remove the slowly varying residual (accelerometer bias + leftover tilt): real motion in a bounded
            # space averages to ~zero acceleration over tens of seconds.
            ka = min(1.0, dt / A.acc_res_tau) if A.acc_res_tau > 0 else 0.0
            for i in range(3):
                self.acc_res[i] += ka * (acc_w[i] - self.acc_res[i])
                acc_w[i] -= self.acc_res[i]
            self.q = qnorm(qmul(self.q, qexp((w[0] * dt, w[1] * dt, w[2] * dt))))
            for i in range(3):
                self.p[i] += self.v[i] * dt + 0.5 * acc_w[i] * dt * dt
                self.v[i] += acc_w[i] * dt
            self.w_last = w
            if self.net is not None or self.rec is not None:
                if self.push_slot(t, w_raw if self.body else w, f):
                    self.learned_update()
            if self.zero_velocity(g, a):
                k = min(1.0, dt / A.zupt_tau)
                for i in range(3):
                    self.v[i] *= 1.0 - k
                    self.bg[i] += min(1.0, A.bias_rate * dt) * (self.still_gyro[i] - self.bg[i])
            leak = A.leak
        else:
            # Dead reckoning (dropout or failed IMU): turn rate and velocity decay, position keeps moving.
            self.bad_time += dt
            self.yaw_var += A.yaw_q_blind * dt
            if self.bad_time > 0.25:
                self.good_time = 0.0
            dk = math.exp(-dt / A.blind_turn_tau)
            self.w_last = tuple(c * dk for c in self.w_last)
            w = self.w_last
            self.q = qnorm(qmul(self.q, qexp((w[0] * dt, w[1] * dt, w[2] * dt))))
            vk = math.exp(-dt / (self.v_blind_tau if self.net_ok else A.blind_vel_tau))
            for i in range(3):
                self.p[i] += self.v[i] * dt
                self.v[i] *= vk
            leak = 0.0
            if self.net is not None or self.rec is not None:
                n = max(1, int(round(dt / self.dt_nom))) if self.dt_nom else 1
                for j in range(min(n, self.W * self.dec)):   # one empty period per missing / unusable sample
                    self.push_slot(t - int((n - 1 - j) * self.dt_nom * 1e9), None, None)
        # Drift control: soft speed limit and a small velocity leak (mean-reverting velocity).
        if leak > 0:
            k = min(1.0, leak * dt)
            for i in range(3):
                self.v[i] *= 1.0 - k
        # Bounded-space prior: position slowly reverts to the start point (exponential, so a long gap cannot overshoot);
        # once the IMU has been unusable for over 1 s the pull is firmer. The part of this step past the 1 s mark gets
        # the firmer pull, the rest the normal one.
        tb = min(dt, max(0.0, self.bad_time - 1.0)) if A.pos_tau_blind > 0 else 0.0
        tau = A.pos_tau
        if A.pos_tau_nonet > 0 and not self.net_usable():
            tau = A.pos_tau_nonet                 # network not running: the firmer pull tuned without it
        f = 1.0
        if tau > 0:
            f *= math.exp(-(dt - tb) / tau)
        if tb > 0:
            f *= math.exp(-tb / A.pos_tau_blind)
        if f != 1.0:
            for i in range(3):
                self.p[i] *= f
        sp = norm3(self.v)
        if sp > A.v_max:
            s = A.v_max / sp
            self.v = [c * s for c in self.v]

    def push_slot(self, t, w, f):
        """Feed one sample period to the network window (w, f = bias-corrected gyro and specific force, or None for
        an unhealthy / missing sample). `dec` periods are averaged into one slot: gravity-aligned gyro, specific
        force minus g, and the fraction of healthy samples; a slot with no healthy sample is empty (None)."""
        W = self.W
        if self.ring is None:
            if self.net is not None and self.dt_nom:  # the network's sample rate, whatever the input's (e.g. 1 kHz)
                self.rate_mult = max(1, int(round(1.0 / (self.dt_nom * self.net.rate))))
                self.dec = self.net.dec * self.rate_mult
            self.ring = [None] * W                # features per slot (None = empty)
            self.ring_p = [(0.0, 0.0, 0.0)] * W   # position when the slot was written
            self.nb = 9 if self.body else 6
            self.bin = [0.0] * self.nb
            self.bin_n = self.bin_k = 0
            self.slots = 0
        if w is not None:
            b = self.bin
            if self.body:                         # body frame, gravity kept, plus our "up" seen from the body
                u = qrot((self.q[0], -self.q[1], -self.q[2], -self.q[3]), (0.0, 0.0, 1.0))
                b[0] += w[0]; b[1] += w[1]; b[2] += w[2]
                b[3] += f[0]; b[4] += f[1]; b[5] += f[2]
                b[6] += u[0]; b[7] += u[1]; b[8] += u[2]
            else:
                gw, aw = qrot(self.q, w), qrot(self.q, f)
                b[0] += gw[0]; b[1] += gw[1]; b[2] += gw[2]
                b[3] += aw[0]; b[4] += aw[1]; b[5] += aw[2] - G
            self.bin_n += 1
            self.since += 1
        self.bin_k += 1
        if self.bin_k < self.dec:
            return False
        feat = None
        if self.bin_n:
            n = self.bin_n
            feat = tuple(x / n for x in self.bin) + (n / self.dec,)
        self.bin = [0.0] * self.nb
        self.bin_n = self.bin_k = 0
        if self.ring[self.ring_i] is not None:
            self.ring_valid -= 1
        if feat is not None:
            self.ring_valid += 1
        self.p_then = self.ring_p[self.ring_i]    # position W slots ago (start of the window)
        self.ring[self.ring_i] = feat
        self.ring_p[self.ring_i] = (self.p[0], self.p[1], self.p[2])
        self.ring_i = (self.ring_i + 1) % W
        self.slots += 1
        if self.rec is not None:
            self.rec.append((t, self.q, feat))
        return feat is not None

    def learned_update(self):
        """Every `model_every` healthy samples the network reads the window and returns the displacement over it
        and the velocity at its end (both in our gravity-aligned frame) with their standard deviations. They are
        used as measurements: position is pulled toward (position one window ago + displacement), velocity toward
        the predicted velocity, each weighted by the network's own confidence."""
        A = self.a
        if self.net is None or self.since < A.model_every * self.rate_mult:
            return
        if self.slots < self.W or self.ring_valid < A.model_min_valid * self.W:
            return
        if not self.net_usable():
            return                                # rate the network cannot use: fall back to physics
        self.since = 0
        d, sd, v, sv = self.net.predict(self.ring, self.ring_i)
        if self.body:                             # body-frame outputs -> our world frame (diagonal covariance rotated)
            d, sd = rot_cov(self.q, d, sd)
            v, sv = rot_cov(self.q, v, sv)
        # Heading is unobservable for an IMU: gyro drift and, above all, every dropout (the head keeps turning
        # while we see nothing) rotate our frame by an unknown yaw. With a heading error of variance s2 the best
        # estimate of a horizontal vector is the predicted one shrunk by E[cos] = exp(-s2 / 2).
        k = math.exp(-0.5 * self.yaw_var)
        d = [d[0] * k, d[1] * k, d[2]]
        v = [v[0] * k, v[1] * k, v[2]]
        for i in range(3):
            gp = A.model_pos_gain * A.model_ref ** 2 / (A.model_ref ** 2 + sd[i] ** 2)
            gv = A.model_vel_gain * A.model_vref ** 2 / (A.model_vref ** 2 + sv[i] ** 2)
            self.p[i] += gp * (self.p_then[i] + d[i] - self.p[i])
            self.v[i] += gv * (v[i] - self.v[i])
        self.net_ok = True

    def net_usable(self):
        """The network runs at its own sample rate (input averaged down by rate_mult); otherwise physics only."""
        return (self.net is not None and self.dt_nom is not None
                and abs(self.dt_nom * self.net.rate * self.rate_mult - 1.0) <= 0.15)

    def pose(self):
        w, x, y, z = self.q
        return self.p[0], self.p[1], self.p[2], x, y, z, w


def rot_cov(q, m, s):
    """Rotate a vector m with independent per-axis std s from the body frame into the world frame; returns the
    rotated vector and the per-axis world std (square root of the diagonal of R diag(s^2) R^T)."""
    mw = qrot(q, m)
    c = [qrot(q, e) for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]   # columns of R
    sw = [math.sqrt(sum((c[j][i] * s[j]) ** 2 for j in range(3))) for i in range(3)]
    return list(mw), sw


def prescan_bias(inp, a, layout=(False, 1, 1.0, "")):
    """First pass over the start of the file: gyro bias = mean gyro of the first still stretch (>= 0.5 s).

    The gyro bias about the vertical cannot be seen from gravity, and a recording does not have to begin at rest
    (EuRoC: the drone is carried around first, then put down). Left uncorrected the heading drifts by degrees per
    second from the first sample on. Reads at most --prescan seconds, keeps a handful of numbers."""
    pf = PathFinder(a)
    acc_sum, n_still, t0 = [0.0, 0.0, 0.0], 0, None
    with open_csv(inp) as f:
        for t, g, acc in parse_rows(f, layout):
            t0 = t if t0 is None else t0
            if (t - t0) * 1e-9 > a.prescan:
                break
            if pf.healthy(g, acc) and pf.zero_velocity(g, acc):
                for i in range(3):
                    acc_sum[i] += pf.still_gyro[i]
                n_still += 1
            elif n_still >= 100:
                break                              # the first still stretch has ended
            else:
                acc_sum, n_still = [0.0, 0.0, 0.0], 0
    return [c / n_still for c in acc_sum] if n_still >= 100 else None


def run(inp, out, a, rec=None):
    layout = detect_layout(inp, columns=a.columns, ts_unit=a.ts_unit, accel_unit=a.accel_unit)
    accel_first, t_scale, a_scale, note = layout
    if accel_first or t_scale != 1 or a_scale != 1.0 or note:
        unit = {1: "ns", 10 ** 3: "us", 10 ** 6: "ms", 10 ** 9: "s"}[t_scale]
        print(f"input layout: {'accel' if accel_first else 'gyro'} columns first, timestamps in {unit}, "
              f"accel in {'g' if a_scale != 1.0 else 'm/s^2'}" + (f" ({note})" if note else ""), file=sys.stderr)
    pf = PathFinder(a)
    if a.prescan > 0 and a.zupt_gyro_std > 0:
        bias = prescan_bias(inp, a, layout)
        if bias is not None:
            pf.bg = bias
            pf.bias_known = True
    pf.rec = rec                                   # optional slot recorder (training data preparation)
    n_out = 0
    last_t = None
    pending = []                                   # rows seen before initialisation
    with open_csv(inp) as f, open(out, "w") as o:
        o.write("# timestamp tx ty tz qx qy qz qw\n")

        def emit(t):
            x, y, z, qx, qy, qz, qw = pf.pose()
            o.write(f"{t * 1e-9:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")

        for t, g, acc in parse_rows(f, layout):
            ok = pf.healthy(g, acc)
            if not pf.initialised:
                pending.append((t, g, acc, ok))
                if ok:
                    pf.init_buf.append((g, acc))
                if len(pf.init_buf) >= a.init_samples or len(pending) >= 4 * a.init_samples:
                    if pf.init_buf:
                        pf.init_from(pf.init_buf)
                    else:
                        pf.initialised = True
                    for tt, gg, aa, oo in pending:     # replay the buffered rows from the start
                        pf.step(tt, gg, aa, oo)
                        emit(tt)
                        n_out += 1
                        last_t = tt
                    pending = []
                    pf.init_buf = []
                continue
            pf.step(t, g, acc, ok)
            emit(t)
            n_out += 1
            last_t = t
        if pending:                                # very short files
            if pf.init_buf:
                pf.init_from(pf.init_buf)
            for tt, gg, aa, oo in pending:
                pf.step(tt, gg, aa, oo)
                emit(tt)
                n_out += 1
                last_t = tt
        if a.extend_blind > 0 and last_t is not None and pf.dt_nom:
            step = int(round(pf.dt_nom * 1e9))
            t = last_t
            for _ in range(int(round(a.extend_blind / pf.dt_nom))):
                t += step
                pf.step(t, (0.0,) * 3, (0.0,) * 3, False)
                emit(t)
                n_out += 1                         # the message counts every pose written
    return n_out


# Option defaults. "physics" is the filter as first submitted; "learned" was chosen on the validation recordings
# (docs/LEARNED_VELOCITY.md) for use with the velocity network, whose input is rotated with this orientation.
# Network coupling used with the fallback (gravity-aligned) network after the mount guard switches.
FALLBACK_COUPLING = {"model_pos_gain": 0.0, "pos_tau": 3.0, "model_ref": 0.1, "pos_tau_blind": 0.0}

PRESETS = {
    "physics": {"kp": 1.0, "ki": 0.02, "kp_boost": 0.0, "tilt_lp": 0.0, "ki_calm": 0.0, "zupt_gyro_std": 0.0, "bias_rate": 0.2,
                "prescan": 0.0, "pos_tau": 0.0, "pos_tau_nonet": 0.0, "pos_tau_blind": 0.0, "model_pos_gain": 0.0, "model_ref": 0.1},
    "learned": {"kp": 0.3, "ki": 0.1, "kp_boost": 10.0, "tilt_lp": 0.1, "ki_calm": 0.5, "zupt_gyro_std": 0.01, "bias_rate": 3.0,
                "prescan": 60.0, "pos_tau": 3.0, "pos_tau_nonet": 3.0, "pos_tau_blind": 0.0, "model_pos_gain": 0.0, "model_ref": 0.1},
}


def args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="estimated_trajectory.txt")
    ap.add_argument("--extend-blind", type=float, default=0.0,
                    help="append this many seconds of dead-reckoned poses after the last input row")
    ap.add_argument("--init-samples", type=int, default=200)
    ap.add_argument("--still-gyro-var", type=float, default=4e-4)
    ap.add_argument("--zupt-window", type=int, default=40)
    ap.add_argument("--zupt-gyro", type=float, default=0.04)
    ap.add_argument("--zupt-acc", type=float, default=0.15)
    ap.add_argument("--zupt-tau", type=float, default=0.05)
    ap.add_argument("--zupt-gyro-std", type=float, default=None,
                    help="rad/s; if > 0, 'still' means per-axis gyro std below this (and --zupt-acc-std)")
    ap.add_argument("--zupt-acc-std", type=float, default=0.15)
    ap.add_argument("--prescan", type=float, default=None,
                    help="s; look this far ahead for the first still stretch to get the gyro bias (0 = off)")
    ap.add_argument("--bias-rate", type=float, default=None, help="1/s; how fast gyro bias follows the still-period mean")
    ap.add_argument("--leak", type=float, default=1.0, help="velocity mean reversion rate (1/s) on healthy data")
    ap.add_argument("--kp", type=float, default=None, help="gravity-correction gain (1/s)")
    ap.add_argument("--ki", type=float, default=None, help="gyro-bias learning gain from gravity correction")
    ap.add_argument("--kp-boost", type=float, default=None, help="extra gravity-correction gain right after a dropout")
    ap.add_argument("--ki-calm", type=float, default=None,
                    help="rad/s; learn gyro bias from gravity only below this turn rate (0 = always)")
    ap.add_argument("--tilt-lp", type=float, default=None,
                    help="s; low-pass on the specific force used for gravity correction (0 = off)")
    ap.add_argument("--kp-boost-tau", type=float, default=0.7, help="s; how fast that extra gain fades")
    ap.add_argument("--tilt-acc-tol", type=float, default=1.5, help="m/s^2 from 1 g where gravity is still trusted")
    ap.add_argument("--acc-res-tau", type=float, default=10.0, help="s; 0 disables residual-acceleration removal")
    ap.add_argument("--v-max", type=float, default=3.0)
    ap.add_argument("--pos-tau", type=float, default=None,
                    help="s; time constant of the pull of position back to the start point (0 = off)")
    ap.add_argument("--pos-tau-nonet", type=float, default=None,
                    help="s; pull back to the start while the network is not running (no model, unusable rate)")
    ap.add_argument("--pos-tau-blind", type=float, default=None,
                    help="s; pull back to the start once the IMU has been unusable for > 1 s (0 = same as --pos-tau)")
    ap.add_argument("--blind-vel-tau", type=float, default=1.5)
    ap.add_argument("--blind-turn-tau", type=float, default=0.5)
    ap.add_argument("--model", default="auto",
                    help="weights of the learned velocity network (src/velnet.py); 'auto' (default) = velnet_weights.bin "
                         "next to this script if present, 'none' = physics only")
    ap.add_argument("--model-every", type=int, default=20, help="healthy samples between network updates")
    ap.add_argument("--model-min-valid", type=float, default=0.25, help="fraction of the window that must be healthy")
    ap.add_argument("--model-ref", type=float, default=None, help="m; displacement std at which the position gain halves")
    ap.add_argument("--model-vref", type=float, default=1.0, help="m/s; velocity std at which the velocity gain halves")
    ap.add_argument("--model-pos-gain", type=float, default=None)
    ap.add_argument("--model-vel-gain", type=float, default=1.0)
    ap.add_argument("--model-window", type=int, default=200, help=argparse.SUPPRESS)   # taken from the weights file;
    ap.add_argument("--model-dec", type=int, default=1, help=argparse.SUPPRESS)        # only set when recording
    ap.add_argument("--fallback-model", default="auto",
                    help="network used when the mount guard fires; 'auto' = velnet_weights_world.bin next to this script")
    ap.add_argument("--mount-guard-deg", type=float, default=25.0,
                    help="deg; switch to --fallback-model if start-up gravity is this far from the training mounts (0 = off)")
    ap.add_argument("--model-input", choices=["world", "body"], default="world", help=argparse.SUPPRESS)
    ap.add_argument("--yaw-q", type=float, default=0.0, help="rad^2/s; heading uncertainty growth on healthy data")
    ap.add_argument("--yaw-q-blind", type=float, default=2.0, help="rad^2/s; heading uncertainty growth in dropouts")
    ap.add_argument("--model-blind-tau", type=float, default=0.7,
                    help="s; velocity decay during dropouts / the blind 5 s when the model is active (0: --blind-vel-tau)")
    ap.add_argument("--columns", choices=["auto", "gyro-first", "accel-first"], default="auto",
                    help="order of the sensor columns (EuRoC: gyro first; the organisers' script: accel first)")
    ap.add_argument("--ts-unit", choices=["auto", "ns", "us", "ms", "s"], default="auto")
    ap.add_argument("--accel-unit", choices=["auto", "ms2", "g"], default="auto")
    ap.add_argument("--acc-max", type=float, default=16 * G)
    ap.add_argument("--acc-min", type=float, default=1.0, help="m/s^2; a weaker reading has no gravity: unhealthy")
    ap.add_argument("--gyro-max", type=float, default=35.0)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="default values of the options below that depend on it: 'physics' (original filter) or "
                         "'learned' (retuned for the velocity network; the default when --model is given)")
    a = ap.parse_args(argv)
    if a.model == "auto":
        cand = Path(__file__).resolve().parent / "velnet_weights.bin"
        a.model = str(cand) if cand.exists() else None
    elif a.model in ("none", ""):
        a.model = None
    if not math.isfinite(a.mount_guard_deg):
        ap.error("--mount-guard-deg must be a finite number")
    if a.fallback_model == "auto":
        cand = Path(__file__).resolve().parent / "velnet_weights_world.bin"
        a.fallback_model = str(cand) if cand.exists() else None
    elif a.fallback_model in ("none", ""):
        a.fallback_model = None
    for k, v in PRESETS[a.preset or ("learned" if a.model else "physics")].items():
        cur = getattr(a, k)
        if cur is None or (isinstance(cur, float) and math.isnan(cur)):  # as the C++ entry: NaN means "use the preset"
            setattr(a, k, v)
    return a


def main(argv=None):
    a = args(argv)
    n = run(a.input, a.output, a)
    print(f"wrote {n} poses to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
