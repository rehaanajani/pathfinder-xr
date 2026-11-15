"""Statistical evaluation of the causal IMU monitor (pathfinder/imu_health.py), issue #1 section C.

    python results/diagnostics/imu_fdi_evaluation.py [--seeds 30]

SYNTHETIC ONLY: head_walk trajectories with simulated co-located redundant IMUs (independent white
noise + constant bias per copy). Nothing here is evidence about real multi-IMU hardware.
Writes results/diagnostics/imu_fdi_evaluation.json.

Reported
  * false alarms per tested window on clean data, with a two-sided 95 % Clopper-Pearson interval;
  * calibration-assumption checks: uniformity of per-window p-values (KS) and lag-1 autocorrelation
    of the parity statistic (the Hotelling threshold assumes independent Gaussian windows);
  * per fault type / axis / severity: detection rate, correct isolation, false isolation (a healthy
    sensor excluded), missed windows and detection delay, with Wilson 95 % intervals.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, kstest

from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.imu_health import ISOLATED, NOMINAL, ImuMonitor, MonitorConfig
from pathfinder.synthetic import head_walk

OUT = Path("results/diagnostics/imu_fdi_evaluation.json")


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def clean(K, seeds, dur):
    alarms = tested = 0
    pv, lag = [], []
    for seed in range(seeds):
        s = head_walk(dur, n_imus=K, seed=1000 + seed)
        _, _, info = ImuMonitor(K).run_batch(s.t, s.gyro, s.acc)
        c = info["calibrated"]
        alarms += int(info["detected"][c].sum())
        tested += int(c.sum())
        p = info["agreement_p"][c]
        pv.append(p[np.isfinite(p)])
        D = info["D"][c]
        if len(D) > 2:
            lag.append(float(np.corrcoef(D[:-1], D[1:])[0, 1]))
    pv = np.concatenate(pv)
    # agreement_p is the min of the gyro and accel p-values: if both are independent U(0,1) under H0,
    # P(min <= x) = 1 - (1 - x)^2, so u = 1 - (1 - p)^2 should be uniform. Departures mean the Gaussian /
    # independence / known-covariance assumptions behind the nominal false-alarm rate do not hold.
    u = 1 - (1 - pv) ** 2
    ks = kstest(u, "uniform")
    lo, hi = clopper_pearson(alarms, tested)
    return {"K": K, "seeds": seeds, "duration_s": dur, "tested_windows": tested, "false_alarms": alarms,
            "rate": alarms / tested, "rate_95ci": [lo, hi], "nominal": MonitorConfig().p_false_alarm,
            "pvalue_uniformity_ks": {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)},
            "parity_lag1_autocorr_mean": float(np.mean(lag)) if lag else None,
            "note": "A 95 % upper bound above the nominal rate means the nominal rate is NOT verified by this sample."}


FAULTS = {
    "bias_small_x": lambda k: Fault("imu_bias", 10, 1e9, k, {"gyro": [0.01, 0, 0], "acc": [0.15, 0, 0]}),
    "bias_small_y": lambda k: Fault("imu_bias", 10, 1e9, k, {"gyro": [0, 0.01, 0], "acc": [0, 0.15, 0]}),
    "bias_small_z": lambda k: Fault("imu_bias", 10, 1e9, k, {"gyro": [0, 0, 0.01], "acc": [0, 0, 0.15]}),
    "bias_tiny_xyz": lambda k: Fault("imu_bias", 10, 1e9, k, {"gyro": [0.003, -0.002, 0.002], "acc": [0.04, -0.03, 0.02]}),
    "bias_large": lambda k: Fault("imu_bias", 10, 1e9, k, {"gyro": [0.08, -0.05, 0.04], "acc": [0.8, -0.5, 0.4]}),
    "bias_drift": lambda k: Fault("imu_bias_drift", 10, 1e9, k),
    "stuck": lambda k: Fault("imu_stuck", 10, 1e9, k),
    "dropout_hold": lambda k: Fault("imu_dropout", 10, 1e9, k),
    "spikes": lambda k: Fault("imu_spikes", 10, 1e9, k),
    "noise": lambda k: Fault("imu_noise", 10, 1e9, k),
    "saturation": lambda k: Fault("imu_saturation", 10, 1e9, k, {"gyro_max": 0.6, "acc_max": 10.5}),
}


def faulty(K, seeds, dur):
    res = {}
    for name, mk in FAULTS.items():
        hit = det = iso = false_iso = n = 0
        delays = []
        for seed in range(seeds):
            s = head_walk(dur, n_imus=K, seed=2000 + seed)
            k_bad = seed % K
            bad = apply_imu_faults(s, [mk(k_bad)], seed)
            _, _, info = ImuMonitor(K).run_batch(bad.t, bad.gyro, bad.acc)
            changed = np.abs(bad.gyro[k_bad] - s.gyro[k_bad]).max(1) + np.abs(bad.acc[k_bad] - s.acc[k_bad]).max(1)
            nw = info["weights"].shape[1]
            aff = np.array([changed[j * 20:(j + 1) * 20].max() > 1e-9 for j in range(nw)]) & info["calibrated"]
            w = info["weights"]
            st = info["window_state"]
            n += int(aff.sum())
            det += int((aff & (st != NOMINAL)).sum())
            iso += int((aff & (w[k_bad] == 0) & np.all(np.delete(w, k_bad, 0) > 0, 0)).sum())
            false_iso += int((aff & np.any(np.delete(w, k_bad, 0) == 0, 0)).sum())
            first = np.flatnonzero(aff & (w[k_bad] == 0))
            if len(first):
                delays.append(float(info["window_t0"][first[0]] + 0.1 - 10.0))
            hit += int(len(first) > 0)
        res[name] = {"affected_windows": n, "detected": det / max(n, 1), "detected_95ci": wilson(det, n),
                     "correct_isolation": iso / max(n, 1), "correct_isolation_95ci": wilson(iso, n),
                     "false_isolation": false_iso / max(n, 1), "false_isolation_95ci": wilson(false_iso, n),
                     "missed": 1 - det / max(n, 1), "runs_isolated": hit, "runs": seeds,
                     "delay_s_median": float(np.median(delays)) if delays else None,
                     "delay_s_max": float(np.max(delays)) if delays else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    out = {"generator": "results/diagnostics/imu_fdi_evaluation.py", "data": "synthetic head_walk, simulated co-located IMUs",
           "config": MonitorConfig().__dict__, "clean": [], "faults": {}}
    for K in (3, 5):
        out["clean"].append(clean(K, a.seeds, 60.0))
        print(json.dumps(out["clean"][-1]), flush=True)
    for K in (2, 3, 5):
        out["faults"][str(K)] = faulty(K, max(a.seeds // 3, 6), 30.0)
        for kname, v in out["faults"][str(K)].items():
            print(K, kname, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()
                             if not x.endswith("95ci")}, flush=True)
    out["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
