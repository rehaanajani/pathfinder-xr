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
  6. Optional (--model src/velnet_weights.bin, needs numpy): a small learned network reads the last 1 s of
     gravity-aligned IMU and its velocity is used as a measurement; this also switches the filter to the
     "learned" preset (see PRESETS and docs/LEARNED_VELOCITY.md). Without --model nothing changes.
"""

import argparse
import io
import math
import sys
import zipfile
from pathlib import Path

G = 9.80665


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


def parse_rows(f):
    """Yield (t_ns, gyro(3), acc(3), ok) per data row. Malformed numbers become NaN (row kept, marked bad)."""
    for line in f:
        s = line.strip()
        if not s or s[0] == "#" or s[0].isalpha():
            continue
        parts = s.replace(";", ",").split(",")
        try:
            t = int(float(parts[0])) if "e" in parts[0].lower() or "." in parts[0] else int(parts[0])
        except ValueError:
            continue
        vals = []
        for x in parts[1:7]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float("nan"))
        while len(vals) < 6:
            vals.append(float("nan"))
        yield t, tuple(vals[:3]), tuple(vals[3:6])


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
        self.v_blind_tau = a.blind_vel_tau
        if a.model:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from velnet import VelNet
            self.net = VelNet(a.model)
            self.W, self.dec = self.net.window, self.net.dec
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
        if norm3(a) > self.a.acc_max or norm3(g) > self.a.gyro_max:
            return False
        return True

    def init_from(self, buf):
        n = len(buf)
        ma = [sum(b[1][i] for b in buf) / n for i in range(3)]
        mg = [sum(b[0][i] for b in buf) / n for i in range(3)]
        # Gravity alignment: rotate the measured specific force (body) onto +z (world).
        self.q = q_from_two(tuple(c / norm3(ma) for c in ma), (0.0, 0.0, 1.0))
        var_g = sum(sum((b[0][i] - mg[i]) ** 2 for i in range(3)) for b in buf) / n
        if var_g < self.a.still_gyro_var and not self.bias_known:   # still at start: the mean gyro is its bias
            self.bg = mg
        self.initialised = True

    def zero_velocity(self, g, a):
        A = self.a
        w = A.zupt_window
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
                if self.push_slot(t, w, f):
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
        if A.pos_tau > 0:                         # bounded-space prior: position slowly reverts to the start point
            k = min(1.0, dt / A.pos_tau)
            for i in range(3):
                self.p[i] *= 1.0 - k
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
            self.ring = [None] * W                # features per slot (None = empty)
            self.ring_p = [(0.0, 0.0, 0.0)] * W   # position when the slot was written
            self.bin = [0.0] * 6
            self.bin_n = self.bin_k = 0
            self.slots = 0
        if w is not None:
            gw, aw = qrot(self.q, w), qrot(self.q, f)
            b = self.bin
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
            feat = (self.bin[0] / n, self.bin[1] / n, self.bin[2] / n, self.bin[3] / n, self.bin[4] / n,
                    self.bin[5] / n, n / self.dec)
        self.bin = [0.0] * 6
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
        if self.net is None or self.since < A.model_every:
            return
        if self.slots < self.W or self.ring_valid < A.model_min_valid * self.W:
            return
        if self.dt_nom is None or abs(self.dt_nom * self.net.rate - 1.0) > 0.15:
            return                                # trained for one sample rate only: fall back to physics
        self.since = 0
        d, sd, v, sv = self.net.predict(self.ring, self.ring_i)
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

    def pose(self):
        w, x, y, z = self.q
        return self.p[0], self.p[1], self.p[2], x, y, z, w


def prescan_bias(inp, a):
    """First pass over the start of the file: gyro bias = mean gyro of the first still stretch (>= 0.5 s).

    The gyro bias about the vertical cannot be seen from gravity, and a recording does not have to begin at rest
    (EuRoC: the drone is carried around first, then put down). Left uncorrected the heading drifts by degrees per
    second from the first sample on. Reads at most --prescan seconds, keeps a handful of numbers."""
    pf = PathFinder(a)
    acc_sum, n_still, t0 = [0.0, 0.0, 0.0], 0, None
    with open_csv(inp) as f:
        for t, g, acc in parse_rows(f):
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
    pf = PathFinder(a)
    if a.prescan > 0 and a.zupt_gyro_std > 0:
        bias = prescan_bias(inp, a)
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

        for t, g, acc in parse_rows(f):
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
    return n_out


# Option defaults. "physics" is the filter as first submitted; "learned" was chosen on the validation recordings
# (docs/LEARNED_VELOCITY.md) for use with the velocity network, whose input is rotated with this orientation.
PRESETS = {
    "physics": {"kp": 1.0, "ki": 0.02, "kp_boost": 0.0, "tilt_lp": 0.0, "ki_calm": 0.0, "zupt_gyro_std": 0.0, "bias_rate": 0.2,
                "prescan": 0.0, "pos_tau": 0.0},
    "learned": {"kp": 0.3, "ki": 0.1, "kp_boost": 10.0, "tilt_lp": 0.1, "ki_calm": 0.5, "zupt_gyro_std": 0.01, "bias_rate": 3.0,
                "prescan": 60.0, "pos_tau": 3.0},
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
    ap.add_argument("--blind-vel-tau", type=float, default=1.5)
    ap.add_argument("--blind-turn-tau", type=float, default=0.5)
    ap.add_argument("--model", default=None, help="weights.bin of the learned velocity network (src/velnet.py)")
    ap.add_argument("--model-every", type=int, default=20, help="healthy samples between network updates")
    ap.add_argument("--model-min-valid", type=float, default=0.25, help="fraction of the window that must be healthy")
    ap.add_argument("--model-ref", type=float, default=0.1, help="m; displacement std at which the position gain halves")
    ap.add_argument("--model-vref", type=float, default=1.0, help="m/s; velocity std at which the velocity gain halves")
    ap.add_argument("--model-pos-gain", type=float, default=0.0)
    ap.add_argument("--model-vel-gain", type=float, default=1.0)
    ap.add_argument("--model-window", type=int, default=200, help=argparse.SUPPRESS)   # taken from the weights file;
    ap.add_argument("--model-dec", type=int, default=1, help=argparse.SUPPRESS)        # only set when recording
    ap.add_argument("--yaw-q", type=float, default=0.0, help="rad^2/s; heading uncertainty growth on healthy data")
    ap.add_argument("--yaw-q-blind", type=float, default=2.0, help="rad^2/s; heading uncertainty growth in dropouts")
    ap.add_argument("--model-blind-tau", type=float, default=0.7,
                    help="s; velocity decay during dropouts / the blind 5 s when the model is active (0: --blind-vel-tau)")
    ap.add_argument("--acc-max", type=float, default=16 * G)
    ap.add_argument("--gyro-max", type=float, default=35.0)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="default values of the options below that depend on it: 'physics' (original filter) or "
                         "'learned' (retuned for the velocity network; the default when --model is given)")
    a = ap.parse_args(argv)
    for k, v in PRESETS[a.preset or ("learned" if a.model else "physics")].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main(argv=None):
    a = args(argv)
    n = run(a.input, a.output, a)
    print(f"wrote {n} poses to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
