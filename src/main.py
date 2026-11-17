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
import ctypes
import hashlib
import io
import json
import math
import os
import sys
import zipfile
import warnings
from itertools import chain, islice
from pathlib import Path

# The entry is specified as one thread. Set this before numpy (imported with the velocity network) starts its pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

G = 9.80665
MOUNT_G_TOL = 0.05                            # |mean specific force| / g - 1 above which the mount guard holds off
# Gravity ("up") in the IMU frame at the start of the recordings the body-frame network was trained on: EuRoC (all 11
# sequences within ~6 deg of this) and TUM-VI (rooms 1-6). The mount guard compares a new file against these.
TRAIN_UP = ((0.9412, -0.0101, -0.3376), (0.0160, -0.0040, 0.9999))


def _load_hotstep():
    """The strapdown step, compiled. None keeps the pure-Python step (training recorder, or no DLL)."""
    if os.environ.get("PF_NO_HOT"):
        return None
    suffix = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    dll = Path(__file__).resolve().with_name("hotstep" + suffix)
    if not dll.exists():
        return None
    try:
        metadata = json.loads(dll.with_suffix(dll.suffix + ".json").read_text())
        digest = hashlib.sha256(dll.with_name("hotstep.c").read_bytes()).hexdigest()
        if metadata["source_sha256"] != digest or metadata["binary_sha256"] != hashlib.sha256(dll.read_bytes()).hexdigest():
            raise ValueError("native helper hashes do not match; rebuild with src/build_hotstep.py")
        lib = ctypes.CDLL(str(dll))
        lib.hot_abi.restype = ctypes.c_int
        if lib.hot_abi() != 1:
            raise ValueError("unsupported native helper ABI")
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        warnings.warn(f"Native helper unavailable; using Python: {exc}", RuntimeWarning)
        return None
    lib.hot_create.restype = ctypes.c_void_p
    lib.hot_create.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int]
    lib.hot_free.argtypes = [ctypes.c_void_p]
    lib.hot_free.restype = None
    lib.hot_pull.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    lib.hot_seed.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                             ctypes.c_int, ctypes.c_longlong]
    lib.hot_integrate.restype = ctypes.c_int
    lib.hot_integrate.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.hot_rest.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                             ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.hot_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_longlong),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int)]
    lib.hot_push_learned.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                     ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_double]
    lib.hot_batch.restype = ctypes.c_int
    lib.hot_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    return lib


_HOT = _load_hotstep()


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


def qexp_out(w0, w1, w2, o):
    """qexp into a 4-list. Same products as qexp."""
    th = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
    if th < 1e-12:
        o[0] = 1.0
        o[1] = 0.5 * w0
        o[2] = 0.5 * w1
        o[3] = 0.5 * w2
        return
    s = math.sin(0.5 * th) / th
    o[0] = math.cos(0.5 * th)
    o[1] = w0 * s
    o[2] = w1 * s
    o[3] = w2 * s


def qmul_out(a, b, o):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    o[0] = aw * bw - ax * bx - ay * by - az * bz
    o[1] = aw * bx + ax * bw + ay * bz - az * by
    o[2] = aw * by - ax * bz + ay * bw + az * bx
    o[3] = aw * bz + ax * by - ay * bx + az * bw


def qnorm_out(q, o):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) or 1.0
    o[0] = q[0] / n
    o[1] = q[1] / n
    o[2] = q[2] / n
    o[3] = q[3] / n


def qrot_out(q, v, o):
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    o[0] = vx + w * tx + y * tz - z * ty
    o[1] = vy + w * ty + z * tx - x * tz
    o[2] = vz + w * tz + x * ty - y * tx


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


def _parse_slow(s):
    """One data row. Malformed numbers become NaN (row kept). None if the line is not a sample."""
    parts = s.replace(";", ",").split(",")
    try:
        t = int(float(parts[0])) if "e" in parts[0].lower() or "." in parts[0] else int(parts[0])
    except (ValueError, OverflowError):
        return None
    if not -(1 << 63) <= t < (1 << 63):             # as the C++ entry: timestamps must fit in int64
        return None
    vals = []
    for x in parts[1:7]:
        try:
            vals.append(float(x))
        except ValueError:
            vals.append(float("nan"))
    while len(vals) < 6:
        vals.append(float("nan"))
    return t, tuple(vals[:3]), tuple(vals[3:6])


def parse_rows(f):
    """Yield (t_ns, gyro(3), acc(3)) per data row. Malformed numbers become NaN (row kept, marked bad)."""
    lo, hi = -(1 << 63), 1 << 63
    for line in f:
        s = line.strip()
        if not s or s[0] == "#" or s[0].isalpha():
            continue
        # Plain decimal rows (the runtime files) skip the scientific / NaN / semicolon checks.
        if ";" not in s and "e" not in s and "E" not in s and "n" not in s and "N" not in s:
            parts = s.split(",")
            if len(parts) >= 7 and "." not in parts[0]:
                try:
                    t = int(parts[0])
                    if lo <= t < hi:
                        yield (t,
                               (float(parts[1]), float(parts[2]), float(parts[3])),
                               (float(parts[4]), float(parts[5]), float(parts[6])))
                        continue
                except (ValueError, OverflowError):
                    pass
        row = _parse_slow(s)
        if row is not None:
            yield row


# ---------------------------------------------------------------------------- the estimator
class PathFinder:
    def __init__(self, a, load_net=True):
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
        # Network input: "world" (weights version 2, 7 channels: gravity-aligned gyro, specific force minus g, health)
        # or "body" (version 3, 10 channels, AirIO style: body-frame gyro and specific force with gravity kept, the
        # gravity direction in the body frame from our own orientation, health; outputs in the body frame).
        self.body = a.model_input == "body"
        if a.model and load_net:
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
        self.ring_f = None                        # float32 copy of the network window, unscaled; None until the first slot
        self._zring = None                        # zero-velocity ring (6 * window), same sums as the old list
        self._qe = [0.0, 0.0, 0.0, 0.0]           # scratch quaternions; not part of the saved state
        self._qm = [0.0, 0.0, 0.0, 0.0]
        self._qn = [0.0, 0.0, 0.0, 0.0]
        self._qv = [0.0, 0.0, 0.0]
        self._qi = [0.0, 0.0, 0.0, 0.0]
        self._aw = [0.0, 0.0, 0.0]             # world acceleration scratch; not part of the saved state
        self._hot = None                          # compiled step state (hotstep.dll); None until the first sample

    # -- health of one reading
    def healthy(self, g, a):
        g0, g1, g2 = g
        a0, a1, a2 = a
        if not (math.isfinite(g0) and math.isfinite(g1) and math.isfinite(g2)
                and math.isfinite(a0) and math.isfinite(a1) and math.isfinite(a2)):
            return False
        if g0 == 0.0 and g1 == 0.0 and g2 == 0.0 and a0 == 0.0 and a1 == 0.0 and a2 == 0.0:
            return False
        raw = (g0, g1, g2, a0, a1, a2)
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
            # then exactly that bias. The ring keeps the same running sums as a list that drops the oldest sample.
            x0, x1, x2 = g
            x3, x4, x5 = a
            s, ss = self.still_s, self.still_ss
            ring = self._zring
            if ring is None or len(ring) != w * 6:
                ring = self._zring = [0.0] * (w * 6)
                self._zn = 0
                self._zi = 0
                for i in range(6):
                    s[i] = 0.0
                    ss[i] = 0.0
            zi = self._zi
            base = zi * 6
            if self._zn == w:
                y0, y1, y2 = ring[base], ring[base + 1], ring[base + 2]
                y3, y4, y5 = ring[base + 3], ring[base + 4], ring[base + 5]
                s[0] -= y0; ss[0] -= y0 * y0
                s[1] -= y1; ss[1] -= y1 * y1
                s[2] -= y2; ss[2] -= y2 * y2
                s[3] -= y3; ss[3] -= y3 * y3
                s[4] -= y4; ss[4] -= y4 * y4
                s[5] -= y5; ss[5] -= y5 * y5
            else:
                self._zn += 1
            ring[base] = x0; ring[base + 1] = x1; ring[base + 2] = x2
            ring[base + 3] = x3; ring[base + 4] = x4; ring[base + 5] = x5
            s[0] += x0; ss[0] += x0 * x0
            s[1] += x1; ss[1] += x1 * x1
            s[2] += x2; ss[2] += x2 * x2
            s[3] += x3; ss[3] += x3 * x3
            s[4] += x4; ss[4] += x4 * x4
            s[5] += x5; ss[5] += x5 * x5
            self._zi = zi + 1 if zi + 1 < w else 0
            if self._zn < w:
                return False
            vg = va = 0.0
            for i in range(3):
                var = ss[i] / w - (s[i] / w) ** 2
                if var > vg:
                    vg = var
            for i in range(3, 6):
                var = ss[i] / w - (s[i] / w) ** 2
                if var > va:
                    va = var
            if vg < 0.0:
                vg = 0.0
            if va < 0.0:
                va = 0.0
            if vg < A.zupt_gyro_std ** 2 and va < A.zupt_acc_std ** 2:
                self.still_gyro = (s[0] / w, s[1] / w, s[2] / w)
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
        q = self.q
        p, v, bg, w, r = self.p, self.v, self.bg, self.w_last, self.acc_res
        p0, p1, p2 = p[0], p[1], p[2]
        v0, v1, v2 = v[0], v[1], v[2]
        b0, b1, b2 = bg[0], bg[1], bg[2]
        w0 = w
        r0, r1, r2 = r[0], r[1], r[2]
        self._step(t, g, a, ok)
        qn, pn, vn, bn, wn, rn = self.q, self.p, self.v, self.bg, self.w_last, self.acc_res
        if (math.isfinite(qn[0]) and math.isfinite(qn[1]) and math.isfinite(qn[2]) and math.isfinite(qn[3])
                and math.isfinite(pn[0]) and math.isfinite(pn[1]) and math.isfinite(pn[2])
                and math.isfinite(vn[0]) and math.isfinite(vn[1]) and math.isfinite(vn[2])
                and math.isfinite(bn[0]) and math.isfinite(bn[1]) and math.isfinite(bn[2])
                and math.isfinite(wn[0]) and math.isfinite(wn[1]) and math.isfinite(wn[2])
                and math.isfinite(rn[0]) and math.isfinite(rn[1]) and math.isfinite(rn[2])):
            return
        self.q = q
        p[0], p[1], p[2] = p0, p1, p2
        bg[0], bg[1], bg[2] = b0, b1, b2
        self.w_last = w0
        r[0], r[1], r[2] = r0, r1, r2
        self.v = [0.0, 0.0, 0.0]

    def _hot_ready(self):
        """Compiled step matches this filter when nothing is recording slots and the window is not decimated.
        dec > 1 can call the learned update in the middle of a dropout burst; that stays on the Python step."""
        return _HOT is not None and self.rec is None and (self.net is None or self.dec == 1)

    def _hot_open(self):
        """Create the compiled state from the current Python estimate. The network window is shared."""
        if self._hot is not None:
            return
        lib = _HOT
        nb = (9 if self.body else 6) if self.net is not None else 0
        if self.net is not None and self.ring_f is None:
            import numpy as np
            self.ring_f = np.zeros((nb + 1, self.W), np.float32)
        A = self.a
        prm = (ctypes.c_double * 24)(
            A.kp, A.ki, A.kp_boost, A.kp_boost_tau, A.tilt_lp, A.ki_calm, A.tilt_acc_tol,
            A.acc_res_tau, A.zupt_tau, A.bias_rate, A.leak, A.pos_tau, A.v_max,
            A.blind_turn_tau, A.blind_vel_tau, self.v_blind_tau, A.yaw_q, A.yaw_q_blind,
            A.zupt_gyro_std, A.zupt_acc_std, A.zupt_gyro, A.zupt_acc,
            A.model_min_valid, float(self.net.rate) if self.net is not None else 0.0)
        ring = None
        if self.ring_f is not None:
            ring = self.ring_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._hot = lib.hot_create(prm, int(self.body), int(self.dec), int(self.W), int(nb),
                                   1 if self.net is not None else 0, int(A.zupt_window), ring,
                                   int(A.model_every))
        if not self._hot:
            raise MemoryError("native estimator allocation failed")
        self._hq = (ctypes.c_double * 4)()
        self._h3 = [(ctypes.c_double * 3)() for _ in range(6)]
        self._ho = (ctypes.c_double * 26)()
        self._hi = [(ctypes.c_int)() for _ in range(7)]
        self._ht = ctypes.c_longlong()
        lib.hot_seed(self._hot, self.bad_time, self.good_time, self.yaw_var,
                     0 if self.t_prev is None else 1, 0 if self.t_prev is None else int(self.t_prev))
        self._hot_pull()

    def _hot_pull(self):
        q = self._hq
        p, v, bg, ba, w, ar = self._h3
        qq = self.q
        q[0], q[1], q[2], q[3] = qq[0], qq[1], qq[2], qq[3]
        for buf, src in ((p, self.p), (v, self.v), (bg, self.bg), (ba, self.ba), (w, self.w_last), (ar, self.acc_res)):
            buf[0], buf[1], buf[2] = src[0], src[1], src[2]
        _HOT.hot_pull(self._hot, q, p, v, bg, ba, w, ar, int(self.since), 1 if self.net_ok else 0)

    def _step_hot(self, t, g, a, ok):
        """Same step as below, in hotstep.dll. The learned update still runs here, in Python."""
        lib = _HOT
        self._hot_open()
        h = self._hot
        self._hot_pull()
        want = lib.hot_integrate(h, int(t), g[0], g[1], g[2], a[0], a[1], a[2], 1 if ok else 0)
        if want:
            self._hot_export()
            self.learned_update()
            pv, vv = self.p, self.v
            lib.hot_push_learned(h, pv[0], pv[1], pv[2], vv[0], vv[1], vv[2], int(self.since),
                                 1 if self.net_ok else 0, float(self.yaw_var))
        lib.hot_rest(h, g[0], g[1], g[2], a[0], a[1], a[2])
        self._hot_export()

    def _hot_export(self):
        """Copy the compiled state back onto this object (pose, bias, and the network bookkeeping)."""
        o, ht = self._ho, self._ht
        has_t, has_dt, since, slots, ring_i, ring_valid, net_ok = self._hi
        _HOT.hot_read(self._hot, o, ctypes.byref(ht), ctypes.byref(has_t), ctypes.byref(has_dt),
                      ctypes.byref(since), ctypes.byref(slots), ctypes.byref(ring_i),
                      ctypes.byref(ring_valid), ctypes.byref(net_ok))
        self.q = (o[0], o[1], o[2], o[3])
        for i in range(3):
            self.p[i] = o[4 + i]
            self.v[i] = o[7 + i]
            self.bg[i] = o[10 + i]
            self.acc_res[i] = o[16 + i]
        self.w_last = (o[13], o[14], o[15])
        self.p_then = (o[19], o[20], o[21])
        self.yaw_var = o[22]
        if has_dt.value:
            self.dt_nom = o[23]
        self.good_time = o[24]
        self.bad_time = o[25]
        if has_t.value:
            self.t_prev = int(ht.value)
        self.since = int(since.value)
        self.slots = int(slots.value)
        self.ring_i = int(ring_i.value)
        self.ring_valid = int(ring_valid.value)
        self.net_ok = bool(net_ok.value)

    def drive(self, samples, emit):
        """Step every sample in the compiled loop, returning to Python only for a learned update.

        `samples` is (t, gyro, acc, ok). `emit(t, tx, ty, tz, qx, qy, qz, qw)` receives one finished pose.
        The learned update itself stays in Python and sees the same window the pure-Python step would.
        """
        import numpy as np
        n = len(samples)
        if n == 0:
            return
        self._hot_open()
        ts = np.empty(n, np.int64)
        imu = np.empty((n, 6), np.float64)
        ok = np.empty(n, np.int32)
        for i, (t, g, acc, good) in enumerate(samples):
            ts[i] = t
            imu[i, 0], imu[i, 1], imu[i, 2] = g[0], g[1], g[2]
            imu[i, 3], imu[i, 4], imu[i, 5] = acc[0], acc[1], acc[2]
            ok[i] = 1 if good else 0
        poses = np.empty((n, 7), np.float64)
        lib = _HOT
        hold = ctypes.c_int()
        ts_p = ts.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        imu_p = imu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ok_p = ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pose_p = poses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        i = 0
        while i < n:
            got = lib.hot_batch(self._hot, ts_p, imu_p, ok_p, i, n, pose_p, ctypes.byref(hold))
            if got <= 0:
                raise RuntimeError("native estimator made no progress")
            nfin = got - (1 if hold.value else 0)
            for k in range(nfin):
                p = poses[k]
                emit(samples[i + k][0], p[0], p[1], p[2], p[3], p[4], p[5], p[6])
            i += got
            if hold.value:
                self._hot_export()
                self.learned_update()
                pv, vv = self.p, self.v
                lib.hot_push_learned(self._hot, pv[0], pv[1], pv[2], vv[0], vv[1], vv[2],
                                     int(self.since), 1 if self.net_ok else 0, float(self.yaw_var))
                g, acc = samples[i - 1][1], samples[i - 1][2]
                lib.hot_rest(self._hot, g[0], g[1], g[2], acc[0], acc[1], acc[2])
                self._hot_export()
                x, y, z, qx, qy, qz, qw = self.pose()
                emit(samples[i - 1][0], x, y, z, qx, qy, qz, qw)
        self._hot_export()

    def close(self):
        if self._hot:
            _HOT.hot_free(self._hot)
            self._hot = None

    def __del__(self):
        if getattr(self, "_hot", None) and globals().get("_HOT") is not None:
            self.close()

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
            bg = self.bg
            w = [g[0] - bg[0], g[1] - bg[1], g[2] - bg[2]]
            w_raw = (w[0], w[1], w[2])            # bias-corrected gyro before the gravity correction (body input)
            ba = self.ba
            f = (a[0] - ba[0], a[1] - ba[1], a[2] - ba[2])
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
                    qi = self._qi
                    q = self.q
                    qi[0], qi[1], qi[2], qi[3] = q[0], -q[1], -q[2], -q[3]
                    up_b = self._qv
                    qrot_out(qi, (0.0, 0.0, 1.0), up_b)
                    fu0, fu1, fu2 = fl[0] / fn, fl[1] / fn, fl[2] / fn
                    e = (fu1 * up_b[2] - fu2 * up_b[1], fu2 * up_b[0] - fu0 * up_b[2],
                         fu0 * up_b[1] - fu1 * up_b[0])
                    tb = trust
                    if A.ki_calm > 0:             # learn the gyro bias only in calm moments (slow turn, ~1 g):
                        wn = norm3(w)             # fast hand motion would otherwise leak into the bias
                        tb = max(0.0, 1.0 - abs(fn - G) / (A.tilt_acc_tol / 3.0)) * max(0.0, 1.0 - wn / A.ki_calm)
                    for i in range(3):
                        w[i] += (A.kp + boost) * trust * e[i]
                        bg[i] -= A.ki * tb * e[i] * dt
            qe, qm, qn = self._qe, self._qm, self._qn
            qexp_out(0.5 * w[0] * dt, 0.5 * w[1] * dt, 0.5 * w[2] * dt, qe)
            qmul_out(self.q, qe, qm)
            qnorm_out(qm, qn)
            qv = self._qv
            qrot_out(qn, f, qv)
            acc_w = self._aw
            acc_w[0] = qv[0]
            acc_w[1] = qv[1]
            acc_w[2] = qv[2] - G
            # Remove the slowly varying residual (accelerometer bias + leftover tilt): real motion in a bounded
            # space averages to ~zero acceleration over tens of seconds.
            ka = min(1.0, dt / A.acc_res_tau) if A.acc_res_tau > 0 else 0.0
            ar = self.acc_res
            for i in range(3):
                ar[i] += ka * (acc_w[i] - ar[i])
                acc_w[i] -= ar[i]
            qexp_out(w[0] * dt, w[1] * dt, w[2] * dt, qe)
            qmul_out(self.q, qe, qm)
            qnorm_out(qm, qn)
            self.q = (qn[0], qn[1], qn[2], qn[3])
            p, v = self.p, self.v
            for i in range(3):
                p[i] += v[i] * dt + 0.5 * acc_w[i] * dt * dt
                v[i] += acc_w[i] * dt
            self.w_last = w
            if self.net is not None or self.rec is not None:
                if self.push_slot(t, w_raw if self.body else w, f):
                    self.learned_update()
            if self.zero_velocity(g, a):
                k = min(1.0, dt / A.zupt_tau)
                br = min(1.0, A.bias_rate * dt)
                sg = self.still_gyro
                one_k = 1.0 - k
                for i in range(3):
                    v[i] *= one_k
                    bg[i] += br * (sg[i] - bg[i])
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
            qe, qm, qn = self._qe, self._qm, self._qn
            qexp_out(w[0] * dt, w[1] * dt, w[2] * dt, qe)
            qmul_out(self.q, qe, qm)
            qnorm_out(qm, qn)
            self.q = (qn[0], qn[1], qn[2], qn[3])
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
            self.nb = 9 if self.body else 6
            self.bin = [0.0] * self.nb
            self.bin_n = self.bin_k = 0
            self.slots = 0
            if self.net is not None:
                import numpy as np
                self.ring_f = np.zeros((self.nb + 1, W), np.float32)
        if w is not None:
            b = self.bin
            if self.body:                         # body frame, gravity kept, plus our "up" seen from the body
                qi, u = self._qi, self._qv
                q = self.q
                qi[0], qi[1], qi[2], qi[3] = q[0], -q[1], -q[2], -q[3]
                qrot_out(qi, (0.0, 0.0, 1.0), u)
                b[0] += w[0]; b[1] += w[1]; b[2] += w[2]
                b[3] += f[0]; b[4] += f[1]; b[5] += f[2]
                b[6] += u[0]; b[7] += u[1]; b[8] += u[2]
            else:
                gw, aw = self._qv, self._qm
                qrot_out(self.q, w, gw)
                # _qm is a 4-list; only the first three entries are the rotated specific force
                qrot_out(self.q, f, aw)
                b[0] += gw[0]; b[1] += gw[1]; b[2] += gw[2]
                b[3] += aw[0]; b[4] += aw[1]; b[5] += aw[2] - G
            self.bin_n += 1
            self.since += 1
        self.bin_k += 1
        if self.bin_k < self.dec:
            return False
        feat = None
        slot = self.ring_i
        rf = self.ring_f
        if self.bin_n:
            n = self.bin_n
            b = self.bin
            nb = self.nb
            health = n / self.dec
            if rf is not None:
                for i in range(nb):
                    rf[i, slot] = b[i] / n         # same float64 quotient, then float32 store, as the tuple path
                rf[nb, slot] = health
            if self.rec is not None or rf is None:
                feat = tuple(b[i] / n for i in range(nb)) + (health,)
            else:
                feat = ()                          # values live in ring_f; only None vs set matters here
            for i in range(nb):
                b[i] = 0.0
        elif rf is not None:
            rf[:, slot] = 0
        self.bin_n = self.bin_k = 0
        if self.ring[slot] is not None:
            self.ring_valid -= 1
        if feat is not None:
            self.ring_valid += 1
        self.p_then = self.ring_p[slot]           # position W slots ago (start of the window)
        self.ring[slot] = feat
        pp = self.p
        self.ring_p[slot] = (pp[0], pp[1], pp[2])
        self.ring_i = (slot + 1) % W
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
        d, sd, v, sv = self.net.predict(self.ring_f if self.ring_f is not None else self.ring, self.ring_i)
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


def prescan_bias(inp, a):
    """First pass over the start of the file: gyro bias = mean gyro of the first still stretch (>= 0.5 s).

    The gyro bias about the vertical cannot be seen from gravity, and a recording does not have to begin at rest
    (EuRoC: the drone is carried around first, then put down). Left uncorrected the heading drifts by degrees per
    second from the first sample on. Reads at most --prescan seconds, keeps a handful of numbers."""
    pf = PathFinder(a, load_net=False)            # bias only; the network is not used on this pass
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


def _run_hot(pf, inp, out, a):
    """Bounded-memory native batches; health and learned updates stay in Python."""
    try:
        with open_csv(inp) as f:
            rows = ((t, g, acc, pf.healthy(g, acc)) for t, g, acc in parse_rows(f))
            return _run_hot_rows(pf, rows, out, a)
    finally:
        pf.close()


def _run_hot_rows(pf, rows, out, a):
    parsed = []
    init_buf = []
    triggered = False
    limit = 4 * a.init_samples
    for i, row in enumerate(rows):
        parsed.append(row)
        _t, g, acc, ok = row
        if ok:
            init_buf.append((g, acc))
        if len(init_buf) >= a.init_samples or (i + 1) >= limit:
            triggered = True
            break
    if init_buf and (triggered or parsed):
        pf.init_from(init_buf)
    elif triggered:
        pf.initialised = True
    parsed = chain(parsed, rows)
    if not pf._hot_ready():
        # Mount guard swapped in a decimated network: replay on the Python step. Health was already decided.
        return _run_parsed(pf, parsed, out, a)
    n_out = 0
    last_t = None
    buf = []
    with open(out, "w") as o:
        o.write("# timestamp tx ty tz qx qy qz qw\n")

        def emit(t, x, y, z, qx, qy, qz, qw):
            nonlocal n_out, last_t
            buf.append(f"{t * 1e-9:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
            if len(buf) >= 512:
                o.write("".join(buf))
                buf.clear()
            n_out += 1
            last_t = t

        while batch := list(islice(parsed, 512)):
            pf.drive(batch, emit)
        if a.extend_blind > 0 and last_t is not None and pf.dt_nom:
            step = int(round(pf.dt_nom * 1e9))
            extra = []
            t = last_t
            for _ in range(int(round(a.extend_blind / pf.dt_nom))):
                t += step
                extra.append((t, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False))
                if len(extra) == 512:
                    pf.drive(extra, emit)
                    extra.clear()
            if extra:
                pf.drive(extra, emit)
        if buf:
            o.write("".join(buf))
    return n_out


def _run_parsed(pf, parsed, out, a):
    """Python step over rows whose health was already decided."""
    n_out = 0
    last_t = None
    buf = []
    with open(out, "w") as o:
        o.write("# timestamp tx ty tz qx qy qz qw\n")

        def emit(t):
            nonlocal n_out, last_t
            x, y, z, qx, qy, qz, qw = pf.pose()
            buf.append(f"{t * 1e-9:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
            if len(buf) >= 512:
                o.write("".join(buf))
                buf.clear()
            n_out += 1
            last_t = t

        for t, g, acc, ok in parsed:
            pf.step(t, g, acc, ok)
            emit(t)
        if a.extend_blind > 0 and last_t is not None and pf.dt_nom:
            step = int(round(pf.dt_nom * 1e9))
            t = last_t
            for _ in range(int(round(a.extend_blind / pf.dt_nom))):
                t += step
                pf.step(t, (0.0,) * 3, (0.0,) * 3, False)
                emit(t)
        if buf:
            o.write("".join(buf))
    return n_out


def run(inp, out, a, rec=None):
    pf = PathFinder(a)
    if a.prescan > 0 and a.zupt_gyro_std > 0:
        bias = prescan_bias(inp, a)
        if bias is not None:
            pf.bg = bias
            pf.bias_known = True
    pf.rec = rec                                   # optional slot recorder (training data preparation)
    if pf._hot_ready():
        return _run_hot(pf, inp, out, a)
    n_out = 0
    last_t = None
    pending = []                                   # rows seen before initialisation
    with open_csv(inp) as f, open(out, "w") as o:
        o.write("# timestamp tx ty tz qx qy qz qw\n")
        buf = []

        def emit(t):
            x, y, z, qx, qy, qz, qw = pf.pose()
            buf.append(f"{t * 1e-9:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
            if len(buf) >= 512:
                o.write("".join(buf))
                buf.clear()

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
        if buf:
            o.write("".join(buf))
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
    ap.add_argument("--model", default="auto",
                    help="weights of the learned velocity network (src/velnet.py); 'auto' (default) = velnet_weights.bin "
                         "next to this script if present, 'none' = physics only")
    ap.add_argument("--model-every", type=int, default=20, help="healthy samples between network updates")
    ap.add_argument("--model-min-valid", type=float, default=0.25, help="fraction of the window that must be healthy")
    ap.add_argument("--model-ref", type=float, default=0.1, help="m; displacement std at which the position gain halves")
    ap.add_argument("--model-vref", type=float, default=1.0, help="m/s; velocity std at which the velocity gain halves")
    ap.add_argument("--model-pos-gain", type=float, default=0.0)
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
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main(argv=None):
    a = args(argv)
    n = run(a.input, a.output, a)
    print(f"wrote {n} poses to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
