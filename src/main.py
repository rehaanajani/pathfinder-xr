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
        self.bad_time = 0.0                       # seconds since the last trusted sample
        self.dt_nom = None
        self.acc_res = [0.0, 0.0, 0.0]            # slow world-frame residual acceleration (bias + tilt leak)
        self.net = None
        if a.model:
            from collections import deque
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from velnet import VelNet
            self.net = VelNet(a.model)
            self.hist = deque(maxlen=201)         # (gravity-aligned gyro+acc, position) per healthy sample
            self.since = 0

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
        if var_g < self.a.still_gyro_var:         # still at start: the mean gyro is its bias
            self.bg = mg
        self.initialised = True

    def zero_velocity(self, g, a):
        self.still_g.append(norm3(g))
        self.still_a.append(norm3(a) - G)
        w = self.a.zupt_window
        if len(self.still_g) > w:
            self.still_g.pop(0)
            self.still_a.pop(0)
        if len(self.still_g) < w:
            return False
        mg = sum(self.still_g) / w
        ma = sum(abs(x) for x in self.still_a) / w
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
            w = [g[0] - self.bg[0], g[1] - self.bg[1], g[2] - self.bg[2]]
            f = (a[0] - self.ba[0], a[1] - self.ba[1], a[2] - self.ba[2])
            # Gravity correction (Mahony complementary filter): when the specific force is close to 1 g, pull the
            # estimated "down" toward the measured one and learn the gyro bias from the disagreement.
            fn = norm3(f)
            if fn > 1e-6 and A.kp > 0:
                trust = max(0.0, 1.0 - abs(fn - G) / A.tilt_acc_tol)
                if trust > 0:
                    up_b = qrot((self.q[0], -self.q[1], -self.q[2], -self.q[3]), (0.0, 0.0, 1.0))
                    fu = (f[0] / fn, f[1] / fn, f[2] / fn)
                    e = (fu[1] * up_b[2] - fu[2] * up_b[1], fu[2] * up_b[0] - fu[0] * up_b[2],
                         fu[0] * up_b[1] - fu[1] * up_b[0])
                    for i in range(3):
                        w[i] += A.kp * trust * e[i]
                        self.bg[i] -= A.ki * trust * e[i] * dt
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
            if self.net is not None:
                self.learned_update(g, a)
            if self.zero_velocity(g, a):
                k = min(1.0, dt / A.zupt_tau)
                for i in range(3):
                    self.v[i] *= 1.0 - k
                    self.bg[i] += A.bias_rate * dt * (g[i] - self.bg[i])
            leak = A.leak
        else:
            # Dead reckoning (dropout or failed IMU): turn rate and velocity decay, position keeps moving.
            self.bad_time += dt
            dk = math.exp(-dt / A.blind_turn_tau)
            self.w_last = tuple(c * dk for c in self.w_last)
            w = self.w_last
            self.q = qnorm(qmul(self.q, qexp((w[0] * dt, w[1] * dt, w[2] * dt))))
            vk = math.exp(-dt / A.blind_vel_tau)
            for i in range(3):
                self.p[i] += self.v[i] * dt
                self.v[i] *= vk
            leak = 0.0
            if self.net is not None:
                self.hist.clear()                    # a window must be continuous, healthy data
        # Drift control: soft speed limit and a small velocity leak (mean-reverting velocity).
        if leak > 0:
            k = min(1.0, leak * dt)
            for i in range(3):
                self.v[i] *= 1.0 - k
        sp = norm3(self.v)
        if sp > A.v_max:
            s = A.v_max / sp
            self.v = [c * s for c in self.v]

    def learned_update(self, g, a):
        """Every 0.1 s: the network predicts how far we moved in the last 1 s; correct p and v toward it."""
        A = self.a
        gw, aw = qrot(self.q, g), qrot(self.q, a)
        self.hist.append((gw + aw, tuple(self.p)))
        self.since += 1
        if len(self.hist) < 201 or self.since < A.model_every:
            return
        self.since = 0
        import numpy as np
        win = np.array([h[0] for h in list(self.hist)[1:]], np.float32).T          # (6, 200)
        d, std = self.net(win)
        p_then = self.hist[0][1]
        k = 0
        for i in range(3):
            e = float(d[i]) - (self.p[i] - p_then[i])
            gain = A.model_ref ** 2 / (A.model_ref ** 2 + float(std[i]) ** 2)
            self.p[i] += A.model_pos_gain * gain * e
            self.v[i] += A.model_vel_gain * gain * e

    def pose(self):
        w, x, y, z = self.q
        return self.p[0], self.p[1], self.p[2], x, y, z, w


def run(inp, out, a):
    pf = PathFinder(a)
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
    ap.add_argument("--bias-rate", type=float, default=0.2)
    ap.add_argument("--leak", type=float, default=1.0, help="velocity mean reversion rate (1/s) on healthy data")
    ap.add_argument("--kp", type=float, default=1.0, help="gravity-correction gain (1/s)")
    ap.add_argument("--ki", type=float, default=0.02, help="gyro-bias learning gain from gravity correction")
    ap.add_argument("--tilt-acc-tol", type=float, default=1.5, help="m/s^2 from 1 g where gravity is still trusted")
    ap.add_argument("--acc-res-tau", type=float, default=10.0, help="s; 0 disables residual-acceleration removal")
    ap.add_argument("--v-max", type=float, default=3.0)
    ap.add_argument("--blind-vel-tau", type=float, default=1.5)
    ap.add_argument("--blind-turn-tau", type=float, default=0.5)
    ap.add_argument("--model", default=None, help="weights.npz of the learned displacement network (numpy)")
    ap.add_argument("--model-every", type=int, default=20, help="samples between network updates")
    ap.add_argument("--model-ref", type=float, default=0.1, help="m; network std at which its gain is 0.5")
    ap.add_argument("--model-pos-gain", type=float, default=0.3)
    ap.add_argument("--model-vel-gain", type=float, default=0.5)
    ap.add_argument("--acc-max", type=float, default=16 * G)
    ap.add_argument("--gyro-max", type=float, default=35.0)
    return ap.parse_args(argv)


def main(argv=None):
    a = args(argv)
    n = run(a.input, a.output, a)
    print(f"wrote {n} poses to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
