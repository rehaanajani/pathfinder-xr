"""Causal multi-IMU health monitor and fusion: the IMU quality contract consumed by the filter.

What it can and cannot know
---------------------------
* **Relative agreement** between sensors (parity test, ``parity.py``) is observable when two or more
  sensors are usable. Agreement does *not* prove the sensors are right: an error common to every
  sensor (shared vibration, temperature-driven bias, a calibration error copied to all units,
  identical bias drift) produces zero parity and is invisible here.
* **Fault attribution** needs three or more agreeing sensors for one faulty sensor; two sensors can
  only detect. Several simultaneous faults need correspondingly more sensors.
* **Signal-level evidence** (non-finite values, stale/repeated packets, a sensor that stops moving
  while the others move, readings pinned at a range limit, high-frequency noise far above the
  sensor's own warm-up level) is observable per sensor, including some common-mode failures
  (noise or saturation on every sensor).
* **Absolute motion confidence** is *not* produced here. The monitor never reports "the motion is
  correct"; it reports which problems it has evidence of, and converts measured problems into extra
  process noise for the filter. Absence of detected problems is reported as absence of evidence
  (states ``WARMUP`` / ``SINGLE`` are explicitly unverified).

Contract (per fused sample, see ``ImuMonitor.run_batch``):
    t, gyro (3), acc (3)        fused body-frame angular rate (rad/s) and specific force (m/s^2)
    state                       one of STATE_NAMES (worst state of the window the sample belongs to)
    q_add_g, q_add_a            extra white-noise density for the filter, (rad/s)^2/Hz and (m/s^2)^2/Hz,
                                added to the filter's own process noise. Measured, not hand-set:
                                excess high-frequency noise, the size of an unattributed disagreement
                                over ``horizon_s``, the averaging lost when sensors are excluded, or a
                                motion envelope (``lost_sigma_*``) when nothing usable is left.
    n_used                      sensors in the fused value

Timing: decisions are made per window (``window_s``). Samples are released when their window is
complete, so the fused stream has a fixed latency of one window (0.1 s by default). Nothing is ever
decided with data from a later window. The batch helper reproduces exactly what the stream emits.

Geometry: ``ImuGeometry`` rotates each sensor into the body frame and moves its specific force to
the body origin (rigid body: f_0 = R f_s - w x (w x r) - wdot x r, wdot from the buffered window).
Sensor streams must share timestamps; inter-sensor time offsets are **not** estimated or
compensated and show up as disagreement during fast rotation. Default geometry is co-located and
aligned, which is what the simulated redundant IMUs in this repository are. Real mounting geometry
is unvalidated until measured on hardware (``validated`` stays False).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from .parity import (hotelling_crit, hotelling_pvalue, isolation_scores, parity_matrix,
                     robust_mean_cov, subset_map, two_sample_shift_pvalue)

WARMUP, NOMINAL, SINGLE, ISOLATED, UNISOLATED, COMMON_MODE, LOST = range(7)
STATE_NAMES = ("warmup", "nominal", "single_unverified", "isolated", "unisolated", "common_mode", "lost")
# Severity order used when a window must be summarised by one state.
_SEVERITY = {NOMINAL: 0, WARMUP: 1, SINGLE: 1, ISOLATED: 2, COMMON_MODE: 3, UNISOLATED: 4, LOST: 5}
HARD_FLAGS = {"nonfinite", "stale", "stuck"}
SOFT_FLAGS = {"noisy", "pinned", "saturated"}


@dataclass
class ImuGeometry:
    R_bs: np.ndarray            # (K, 3, 3) rotation sensor -> body
    r_b: np.ndarray             # (K, 3) sensor origin in the body frame, metres
    validated: bool = False     # only True with measured hardware calibration evidence

    @classmethod
    def colocated(cls, K: int) -> "ImuGeometry":
        return cls(np.repeat(np.eye(3)[None], K, 0), np.zeros((K, 3)))

    @property
    def is_colocated(self) -> bool:
        return bool(np.allclose(self.R_bs, np.eye(3)) and np.allclose(self.r_b, 0.0))

    def to_body(self, t, g, a):
        """g, a: (K, n, 3) raw sensor readings of one buffered window -> body frame at the origin."""
        if self.is_colocated:
            return g, a
        gb = np.einsum("kij,knj->kni", self.R_bs, g)
        ab = np.einsum("kij,knj->kni", self.R_bs, a)
        out = np.empty_like(ab)
        for k in range(g.shape[0]):
            w = gb[k]
            # Mean angular acceleration over the window (the window is buffered, so central
            # differences only use samples that have already arrived).
            wdot = np.gradient(w, t, axis=0).mean(0) if len(t) > 1 else np.zeros(3)
            r = self.r_b[k]
            out[k] = ab[k] - np.cross(w, np.cross(w, r)) - np.cross(wdot, r)
        return gb, out


@dataclass
class MonitorConfig:
    window_s: float = 0.1
    warmup_s: float = 2.0           # minimum warm-up; extended until enough clean windows exist
    calib_extra: int = 20           # calibration windows >= parity dimension + this
    p_false_alarm: float = 1e-4     # per complete window (gyro and accel blocks Bonferroni-split)
    trim_alpha: float = 1e-3        # calibration outlier trimming
    stationarity_alpha: float = 1e-3  # warm-up halves must not differ (a fault starting mid warm-up)
    max_offset_gyro: float = 0.02   # prior bound on calibrated inter-sensor bias differences, rad/s
    max_offset_acc: float = 0.3     # ... m/s^2 (set from the device datasheet / factory calibration)
    max_warmup_s: float = 10.0      # still uncalibrated after this: report unattributed disagreement
    readmit_s: float = 1.0          # an isolated sensor must agree this long before it is used again
    signal_checks: bool = True
    stuck_std: float = 1e-4
    noise_ratio: float = 6.0        # HF energy above this x own warm-up level = noisy
    gyro_range: float | None = None  # known datasheet range -> saturation check
    acc_range: float | None = None
    plateau_run: int = 5            # identical consecutive extreme readings = pinned (range unknown)
    horizon_s: float = 1.0          # unattributed disagreement -> extra process noise over this horizon
    lost_sigma_g: float = 0.5       # motion envelope used when no sensor is usable, rad/s/sqrt(Hz)
    lost_sigma_a: float = 2.0       # ... m/s^2/sqrt(Hz)


@dataclass
class _Calib:
    members: list
    V: np.ndarray
    mg: np.ndarray
    Sg: np.ndarray
    ma: np.ndarray
    Sa: np.ndarray
    n: int
    n_samples: float
    dev_g: np.ndarray
    dev_a: np.ndarray
    hf: np.ndarray
    ambiguous: bool = False
    cache: dict = field(default_factory=dict)


def _hf(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.diff(x, axis=0) ** 2, 1)))) if len(x) > 1 else 0.0


def _longest_extreme_run(v: np.ndarray) -> int:
    best = 0
    for ax in range(v.shape[1]):
        x = np.abs(v[:, ax])
        m = x.max()
        if m <= 0:
            continue
        hit = (x == m).astype(int)
        run = cur = 0
        for h in hit:
            cur = cur + 1 if h else 0
            run = max(run, cur)
        best = max(best, run)
    return best


class ImuMonitor:
    def __init__(self, K: int, cfg: MonitorConfig = MonitorConfig(), geometry: ImuGeometry | None = None):
        if K < 1:
            raise ValueError("need at least one IMU")
        self.K, self.cfg = K, cfg
        self.geometry = geometry or ImuGeometry.colocated(K)
        if self.geometry.R_bs.shape != (K, 3, 3) or self.geometry.r_b.shape != (K, 3):
            raise ValueError("geometry does not match the number of IMUs")
        self.members = list(range(K))
        self.calib: _Calib | None = None
        self.store: list = []                # clean warm-up windows: (zg, za, hf, n)
        self.restarts = 0
        self.calib_skipped = 0
        self.excluded: dict[int, float] = {}  # isolated sensor -> clean time accumulated
        self.permanent: dict[int, str] = {}  # excluded at calibration, with reason
        self._buf_t: list = []
        self._buf_g: list = []
        self._buf_a: list = []
        self._w0 = None
        self._t_start = None
        self._last_t = None
        self.windows: list = []
        self.last_fused = None

    # ------------------------------------------------------------------ streaming interface
    def push(self, t: float, gyro: np.ndarray, acc: np.ndarray) -> list:
        """Add one synchronized sample (gyro, acc: (K, 3)). Returns fused samples of any completed window."""
        gyro = np.asarray(gyro, float).reshape(self.K, 3)
        acc = np.asarray(acc, float).reshape(self.K, 3)
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("IMU timestamp must be finite")
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(f"IMU timestamps must increase strictly ({t} after {self._last_t})")
        out = []
        W = self.cfg.window_s
        if self._w0 is None:
            self._w0 = t
        elif t >= self._w0 + W - 1e-9:
            out = self._close()
            self._w0 += W * max(1, int(np.floor((t - self._w0) / W + 1e-9)))
        self._buf_t.append(t)
        self._buf_g.append(gyro)
        self._buf_a.append(acc)
        self._last_t = t
        return out

    def flush(self) -> list:
        """Close the final (possibly incomplete) window."""
        return self._close()

    def run_batch(self, t, gyro, acc):
        """t: (N,), gyro/acc: (K, N, 3). Same results as streaming push() + flush()."""
        t = np.asarray(t, float)
        gyro, acc = np.asarray(gyro, float), np.asarray(acc, float)
        if gyro.ndim != 3 or gyro.shape[0] != self.K or gyro.shape[2] != 3 or acc.shape != gyro.shape:
            raise ValueError(f"gyro/acc must have shape (K={self.K}, N, 3) and match")
        if len(t) != gyro.shape[1]:
            raise ValueError("timestamps and samples differ in length")
        rows = []
        for i in range(len(t)):
            rows += self.push(t[i], gyro[:, i], acc[:, i])
        rows += self.flush()
        N = len(t)
        fg, fa = np.full((N, 3), np.nan), np.full((N, 3), np.nan)
        state = np.zeros(N, int)
        qg, qa, nu, widx = np.zeros(N), np.zeros(N), np.zeros(N, int), np.zeros(N, int)
        ts = np.array([r["t"] for r in rows])
        if len(rows) != N or not np.array_equal(ts, t):
            raise RuntimeError("fused stream does not align with the input timestamps")
        for i, r in enumerate(rows):
            fg[i], fa[i] = r["gyro"], r["acc"]
            state[i], qg[i], qa[i], nu[i], widx[i] = r["state"], r["q_add_g"], r["q_add_a"], r["n_used"], r["window"]
        win = self.windows
        weights = np.array([w["weights"] for w in win]).T if win else np.zeros((self.K, 0))
        info = {
            "t": t, "state": state, "q_add_g": qg, "q_add_a": qa, "n_used": nu, "window": widx,
            "weights": weights,
            "window_state": np.array([w["state"] for w in win], int),
            "window_t0": np.array([w["t0"] for w in win]),
            "window_n": np.array([w["n"] for w in win], int),
            "detected": np.array([w["detected"] for w in win], bool),
            "isolated": np.array([w["isolated"][0] if w["isolated"] else -1 for w in win], int),
            "isolated_all": [w["isolated"] for w in win],
            "D": np.array([w["D"] for w in win]),
            "agreement_p": np.array([w["p_value"] for w in win]),
            "calibrated": np.array([w["calibrated"] for w in win], bool),
            "reasons": [w["flags"] for w in win],
            "restarts": self.restarts, "calib_skipped": self.calib_skipped,
            "permanent_exclusions": dict(self.permanent),
            "calibration_ambiguous": bool(self.calib.ambiguous) if self.calib else False,
            "geometry_validated": self.geometry.validated,
        }
        return fg, fa, info

    # ------------------------------------------------------------------ internals
    def _close(self) -> list:
        if not self._buf_t:
            return []
        t = np.array(self._buf_t)
        g = np.stack(self._buf_g, 1)          # (K, n, 3)
        a = np.stack(self._buf_a, 1)
        self._buf_t, self._buf_g, self._buf_a = [], [], []
        return self._process(t, g, a)

    def _flags(self, t, g, a, gb, ab):
        c, K = self.cfg, self.K
        n = len(t)
        flags = [set() for _ in range(K)]
        finite = np.array([np.isfinite(g[k]).all() and np.isfinite(a[k]).all() for k in range(K)])
        for k in range(K):
            if not finite[k]:
                flags[k].add("nonfinite")
        if not c.signal_checks:
            return flags, finite
        moving = np.array([g[k].std(0).max() if finite[k] else np.nan for k in range(K)])
        med_moving = np.nanmedian(moving) if finite.any() else np.nan
        for k in range(K):
            if not finite[k]:
                continue
            if n >= 3 and np.all(np.diff(g[k], axis=0) == 0) and np.all(np.diff(a[k], axis=0) == 0):
                flags[k].add("stale")
            elif K >= 2 and moving[k] < c.stuck_std and med_moving > 10 * c.stuck_std:
                flags[k].add("stuck")
            if c.gyro_range and np.any(np.abs(g[k]) >= 0.99 * c.gyro_range):
                flags[k].add("saturated")
            if c.acc_range and np.any(np.abs(a[k]) >= 0.99 * c.acc_range):
                flags[k].add("saturated")
            if c.plateau_run and n >= c.plateau_run and "stale" not in flags[k]:
                if max(_longest_extreme_run(g[k]), _longest_extreme_run(a[k])) >= c.plateau_run:
                    flags[k].add("pinned")
            if self.calib is not None and n > 2:
                base = self.calib.hf[k]
                if np.all(base > 0):
                    r = max(_hf(g[k]) / base[0], _hf(a[k]) / base[1])
                    if r > c.noise_ratio:
                        flags[k].add("noisy")
        return flags, finite

    def _window_means(self, gb, ab, idx):
        return gb[idx].mean(1).reshape(-1), ab[idx].mean(1).reshape(-1)

    def _try_calibrate(self):
        c = self.cfg
        m = len(self.members)
        d = 3 * (m - 1)
        need = max(int(round(c.warmup_s / c.window_s)), d + c.calib_extra if m >= 2 else 0, 3)
        if len(self.store) < need:
            return
        if len(self.store) > 4 * need:              # persistent contamination: keep the recent part
            self.store = self.store[-2 * need:]
        mem = self.members
        cols = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in mem])
        Zg = np.array([s[0] for s in self.store])[:, cols]
        Za = np.array([s[1] for s in self.store])[:, cols]
        hf = np.array([s[2] for s in self.store])          # (W, K, 2)
        nsamp = float(np.median([s[3] for s in self.store]))
        if m == 1:
            self.calib = _Calib(mem, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)), np.zeros(0),
                                np.zeros((0, 0)), len(self.store), nsamp, np.zeros((self.K, 3)),
                                np.zeros((self.K, 3)), np.median(hf, 0))
            return
        V = parity_matrix(m)
        pg, pa = Zg @ V.T, Za @ V.T
        cut = chi2.ppf(1 - c.trim_alpha, d)
        mg, Sg, kg = robust_mean_cov(pg, cut)
        ma, Sa, ka = robust_mean_cov(pa, cut)
        keep = kg & ka
        if keep.sum() < d + c.calib_extra:
            return                                      # too many outliers: keep collecting
        half = int(keep.sum() // 2)
        ii = np.flatnonzero(keep)
        if half > d + 1:
            pmin = min(two_sample_shift_pvalue(pg[ii[:half]], pg[ii[half:]]),
                       two_sample_shift_pvalue(pa[ii[:half]], pa[ii[half:]]))
            if pmin < c.stationarity_alpha:
                # The parity mean moved during warm-up: something changed (a fault started).
                # Do not learn it as healthy; restart from the later half.
                self.store = self.store[ii[half]:]
                self.restarts += 1
                return
        mg, Sg = pg[keep].mean(0), np.cov(pg[keep].T)
        ma, Sa = pa[keep].mean(0), np.cov(pa[keep].T)
        zg = Zg[keep].mean(0).reshape(m, 3)
        za = Za[keep].mean(0).reshape(m, 3)
        dev_g, dev_a = zg - zg.mean(0), za - za.mean(0)
        over = np.maximum(np.abs(dev_g).max(1) / c.max_offset_gyro, np.abs(dev_a).max(1) / c.max_offset_acc)
        ambiguous = False
        if over.max() > 1.0:
            if m >= 3:
                medg, meda = np.median(zg, 0), np.median(za, 0)
                score = np.maximum(np.abs(zg - medg).max(1) / c.max_offset_gyro,
                                   np.abs(za - meda).max(1) / c.max_offset_acc)
                worst = mem[int(np.argmax(score))]
                self.permanent[worst] = "calibration_offset_exceeds_prior"
                self.members = [k for k in mem if k != worst]
                self._try_calibrate()
                return
            ambiguous = True                            # two sensors: cannot say which one is off
        DG, DA = np.zeros((self.K, 3)), np.zeros((self.K, 3))
        for i, k in enumerate(mem):
            DG[k], DA[k] = dev_g[i], dev_a[i]
        self.calib = _Calib(mem, V, mg, 0.5 * (Sg + Sg.T), ma, 0.5 * (Sa + Sa.T), int(keep.sum()), nsamp,
                            DG, DA, np.median(hf[keep], 0), ambiguous)

    def _subset_stats(self, subset):
        cb = self.calib
        key = tuple(subset)
        if key not in cb.cache:
            M, Vc = subset_map(cb.V, cb.members, list(subset))
            d = Vc.shape[0]
            alpha = self.cfg.p_false_alarm / 2.0
            cb.cache[key] = (Vc, M @ cb.mg, M @ cb.Sg @ M.T, M @ cb.ma, M @ cb.Sa @ M.T,
                             hotelling_crit(d, cb.n, alpha), d)
        return cb.cache[key]

    def _test(self, subset, zg_all, za_all, n):
        """Hotelling test of one subset; returns (detected, ratio, p_value, pg, pa, stats)."""
        Vc, mg, Sg, ma, Sa, crit, d = self._subset_stats(subset)
        cols = np.concatenate([np.arange(3 * k, 3 * k + 3) for k in subset])
        scale = self.calib.n_samples / max(n, 1)        # incomplete window: mean is noisier
        pg = Vc @ zg_all[cols] - mg
        pa = Vc @ za_all[cols] - ma
        tg = float(pg @ np.linalg.solve(Sg * scale, pg))
        ta = float(pa @ np.linalg.solve(Sa * scale, pa))
        ratio = max(tg, ta) / crit
        pval = min(hotelling_pvalue(tg, d, self.calib.n), hotelling_pvalue(ta, d, self.calib.n))
        return ratio > 1.0, ratio, pval, pg, pa, (Vc, Sg * scale, Sa * scale)

    def _process(self, t, g, a):
        c, K = self.cfg, self.K
        n = len(t)
        dt = float(np.median(np.diff(t))) if n > 1 else c.window_s
        gb, ab = self.geometry.to_body(t, np.nan_to_num(g), np.nan_to_num(a))
        flags, finite = self._flags(t, g, a, gb, ab)
        hard = [k for k in range(K) if flags[k] & HARD_FLAGS]
        for k in self.permanent:
            flags[k].add("excluded_at_calibration")
        usable = [k for k in self.members if k not in hard]
        soft = [k for k in usable if flags[k] & SOFT_FLAGS]
        clean = [k for k in usable if k not in soft]
        common = False
        if clean:
            cand = clean
        else:
            cand, common = usable, bool(usable)
        zg_all, za_all = gb.mean(1).reshape(-1), ab.mean(1).reshape(-1)
        hf_now = np.array([[_hf(g[k]), _hf(a[k])] if finite[k] else [np.nan, np.nan] for k in range(K)])

        detected, isolated, D, pval = False, [], 0.0, float("nan")
        calibrated = self.calib is not None
        if not calibrated:
            if all(not flags[k] for k in range(K)) and n >= 2:
                self.store.append((zg_all, za_all, hf_now, n))
            else:
                self.calib_skipped += 1
            used = cand
            state = WARMUP if len(used) >= 2 else SINGLE if len(used) == 1 else LOST
            if self._t_start is None:
                self._t_start = float(t[0])
            if len(used) >= 2 and t[-1] - self._t_start > c.max_warmup_s:
                # The sensors never agreed long enough to calibrate (e.g. drifting relative to each
                # other from power-on). Their disagreement is real but cannot be attributed.
                state = UNISOLATED
            self._try_calibrate()
        else:
            # Isolated sensors stay out until they agree again for readmit_s (hysteresis).
            excluded = [k for k in self.excluded if k in cand]
            base = [k for k in cand if k not in excluded]
            forced = not base and bool(cand)
            if forced:
                # Every remaining sensor was isolated earlier: do not drop to zero sensors on
                # history alone; use them, but report the disagreement as unattributed.
                base, excluded = list(cand), []
            used = list(base)
            if len(used) >= 2:
                detected, D, pval, pg, pa, st = self._test(used, zg_all, za_all, n)
                det = detected
                # A common-mode anomaly (every sensor noisy / pinned) changes the noise level the
                # parity test was calibrated for, so isolation would pick a sensor at random.
                while det and len(used) >= 3 and not common:
                    Vc, Sg, Sa = st
                    sc = isolation_scores(pg, Sg, Vc, len(used)) + isolation_scores(pa, Sa, Vc, len(used))
                    j = used[int(np.argmax(sc))]
                    isolated.append(j)
                    used = [k for k in used if k != j]
                    self.excluded[j] = 0.0
                    det, _, pval, pg, pa, st = self._test(used, zg_all, za_all, n)
                unisolated = (det and not common) or forced
                for j in excluded:
                    ok = not unisolated and not self._test(sorted(used + [j]), zg_all, za_all, n)[0]
                    self.excluded[j] = self.excluded[j] + n * dt if ok else 0.0
                    if self.excluded[j] >= c.readmit_s - 1e-9:
                        del self.excluded[j]
                if unisolated or self.calib.ambiguous:
                    state = UNISOLATED
                elif common:
                    state = COMMON_MODE
                elif isolated or excluded or len(used) < len(self.members):
                    state = ISOLATED
                else:
                    state = NOMINAL
            elif len(used) == 1:
                state = COMMON_MODE if common else ISOLATED if K > 1 else SINGLE
                for j in excluded:
                    self.excluded[j] = 0.0
            else:
                state = LOST
        if common and state not in (LOST, UNISOLATED):
            state = COMMON_MODE

        # ---- fused output and the extra process noise it needs
        q_g = q_a = 0.0
        if used:
            off_g = self.calib.dev_g if calibrated else np.zeros((K, 3))
            off_a = self.calib.dev_a if calibrated else np.zeros((K, 3))
            # Equal weights over the used sensors, each corrected by its calibrated offset from the
            # ensemble, so excluding one sensor does not step the fused bias.
            fg = np.mean([gb[k] - off_g[k] for k in used], 0)
            fa = np.mean([ab[k] - off_a[k] for k in used], 0)
            if state == UNISOLATED:
                sg = max(np.linalg.norm(zg_all[3 * k:3 * k + 3] - off_g[k] - fg.mean(0)) for k in used)
                sa = max(np.linalg.norm(za_all[3 * k:3 * k + 3] - off_a[k] - fa.mean(0)) for k in used)
                if calibrated and self.calib.ambiguous:
                    sg = max(sg, float(np.abs(off_g).max()))
                    sa = max(sa, float(np.abs(off_a).max()))
                q_g += sg ** 2 * c.horizon_s
                q_a += sa ** 2 * c.horizon_s
            if calibrated:
                base_hf = self.calib.hf
                ratio_used = np.median([max(hf_now[k, 0] / max(base_hf[k, 0], 1e-12),
                                            hf_now[k, 1] / max(base_hf[k, 1], 1e-12)) for k in used])
                if ratio_used > 2.0:
                    # Measured excess white noise (per axis, per-sample variance -> density).
                    q_g += float(np.median([max(hf_now[k, 0] ** 2 - base_hf[k, 0] ** 2, 0) for k in used])) / 6 * dt
                    q_a += float(np.median([max(hf_now[k, 1] ** 2 - base_hf[k, 1] ** 2, 0) for k in used])) / 6 * dt
                nm = len(self.members)
                if len(used) < nm:
                    vg = float(np.median(base_hf[used, 0])) ** 2 / 6 * dt
                    va = float(np.median(base_hf[used, 1])) ** 2 / 6 * dt
                    q_g += vg * (1 / len(used) - 1 / nm)
                    q_a += va * (1 / len(used) - 1 / nm)
            if common and any(flags[k] & {"pinned", "saturated"} for k in used):
                q_g += c.lost_sigma_g ** 2
                q_a += c.lost_sigma_a ** 2
        else:
            fin = [k for k in range(K) if finite[k]]
            if fin:
                fg = np.median(gb[fin], 0)
                fa = np.median(ab[fin], 0)
            else:
                fg = np.full((n, 3), np.nan)
                fa = np.full((n, 3), np.nan)
            q_g, q_a = c.lost_sigma_g ** 2, c.lost_sigma_a ** 2

        weights = np.zeros(K)
        if state == UNISOLATED and len(used) == 2 and K == 2:
            weights[used] = 0.5
        else:
            weights[used] = 1.0
        widx = len(self.windows)
        self.windows.append({"t0": float(t[0]), "n": n, "state": state, "weights": weights, "detected": detected,
                             "isolated": isolated, "D": D, "p_value": pval, "calibrated": calibrated,
                             "flags": ["+".join(sorted(f)) for f in flags], "used": list(used)})
        rows = []
        for i in range(n):
            rows.append({"t": float(t[i]), "gyro": fg[i], "acc": fa[i], "state": state, "q_add_g": q_g,
                         "q_add_a": q_a, "n_used": len(used), "window": widx})
        return rows


def state_name(code: int) -> str:
    return STATE_NAMES[int(code)]
