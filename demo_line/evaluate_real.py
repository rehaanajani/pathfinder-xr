"""Real-data benchmark on TUM-VI (real IMU, real camera frames, motion-capture truth).

    python evaluate_real.py            # needs local_data/tumvi/dataset-room1_512_16 and room2

Protocol: the camera-health model and the trust table are fitted on one room and tested on the
other (both ways). The learned motion network is the one trained on Meta's TLIO headset data, so
TUM-VI is also a cross-device test (different IMU, handheld rig). Writes results/real_summary.json.
"""

import json
import time
from pathlib import Path

import numpy as np

from pathfinder import data_tumvi as T
from pathfinder import metrics
from pathfinder import real_camera as RC
from pathfinder.data_tlio import redundant_imus
from pathfinder.faults import Fault, apply_imu_faults
from pathfinder.filter import FilterConfig, run
from pathfinder.integrator import strapdown
from pathfinder.pipeline import fuse_imus
from pathfinder.predictor import Predictor

CLIP = 60.0
STARTS = [5.0, 40.0, 75.0]          # three 60 s clips per room


def scenarios():
    m = 20.0
    big_bias = {"gyro": [0.08, -0.05, 0.04], "acc": [0.8, -0.5, 0.4]}
    return {
        "clean": [],
        "camera_blackout": [Fault("camera_blackout", m, 10)],
        "camera_dark": [Fault("camera_dark", m, 10, params={"level": 0.06})],
        "camera_blur": [Fault("camera_blur", m, 10, params={"pixels": 30})],
        "dark_plus_imu_drift": [Fault("camera_dark", m, 10, params={"level": 0.06}),
                                Fault("imu_bias_drift", 0, 1e9, None)],
        "blackout_plus_one_imu_bias": [Fault("camera_blackout", m, 10), Fault("imu_bias", m - 5, 1e9, 0, big_bias)],
        "blackout_plus_saturation_all": [Fault("camera_blackout", m, 10),
                                         Fault("imu_saturation", m, 10, None, {"gyro_max": 1.0, "acc_max": 11.0})],
    }


def calibrate_trust_real(seq, ft, fps, hw, seed=0):
    """Measured fix error per learned-health bin on the calibration room."""
    rng = np.random.default_rng(seed)
    H, E = [], []
    for _ in range(3):
        faults = []
        for _ in range(5):
            st, L = rng.uniform(0, seq.t[-1] - 8), rng.uniform(3, 8)
            if rng.random() < 0.5:
                faults.append(Fault("camera_dark", st, L, params={"level": float(np.exp(rng.uniform(np.log(0.03), np.log(0.6))))}))
            else:
                faults.append(Fault("camera_blur", st, L, params={"pixels": float(rng.uniform(5, 35))}))
        cam = RC.visual_fixes_real(seq, ft, fps, faults, seed=int(rng.integers(1e6)), health_w=hw)
        ok = ~np.isnan(cam["fix"][:, 0])
        H.append(cam["health"][ok])
        E.append(np.linalg.norm(cam["fix"][ok] - seq.p[cam["idx"][ok]], axis=1) / np.sqrt(3))
    h, e = np.concatenate(H), np.concatenate(E)
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
    c, s = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (h >= lo) & (h < hi)
        if m.sum() >= 15:
            c.append(float(h[m].mean()))
            s.append(float(np.sqrt(np.mean(e[m] ** 2))))
    for j in range(len(s) - 2, -1, -1):
        s[j] = max(s[j], s[j + 1])
    return np.array(c), np.array(s)


def main():
    t0 = time.time()
    rooms = T.available()
    assert len(rooms) >= 2, "need TUM-VI room1 and room2 in local_data/tumvi"
    model = Predictor("models/dispnet_faults/best.pt")
    full = {r: T.load(r) for r in rooms}
    frames = {r: T.frames(full[r]) for r in rooms}
    results = {}
    for test in rooms:
        cal = [r for r in rooms if r != test][0]
        X, y = RC.build_health_dataset(frames[cal][1], 300, seed=1)
        hw = RC.fit_health(X, y, out=f"models/camera_health_learned_{cal}.json")
        trust = calibrate_trust_real(full[cal], *frames[cal], hw)
        print(f"test={test} calibrated on {cal}: trust {np.round(trust[1], 3).tolist()}", flush=True)
        seq_full = full[test]
        ft_all, fp_all = frames[test]
        for name, faults in scenarios().items():
            for st in STARTS:
                a, b = int(st * 200), int((st + CLIP) * 200) + 1
                if b > len(seq_full.t):
                    continue
                seq = redundant_imus(seq_full.slice(a, b), 3, seed=int(st))
                seq.t = seq.t - seq.t[0]
                seq.meta = dict(seq_full.meta, t_abs0=seq_full.meta["t_abs0"] + st)
                m = (ft_all >= st) & (ft_all <= st + CLIP)
                ft, fp = ft_all[m] - st, [p for p, k in zip(fp_all, m) if k]
                bad = apply_imu_faults(seq, faults, seed=int(st))
                cam_naive = RC.visual_fixes_real(seq, ft, fp, faults, seed=int(st), health_w=hw)
                avg_g, avg_a = bad.gyro.mean(0), bad.acc.mean(0)
                fg, fa, info = fuse_imus(bad)
                conf = np.repeat(info["confidence"], 20)[: len(seq.t)]
                rows = {}
                est = strapdown(avg_g, avg_a, seq.dt, seq.p[0], seq.v[0], seq.q[0])
                rows["imu_only"] = metrics.ate(est, seq.p)
                est, _ = run(seq, avg_g, avg_a, None, cam_naive, None, FilterConfig(use_health=False, use_gate=False))
                rows["naive_fusion"] = metrics.ate(est, seq.p)
                est, _ = run(seq, avg_g, avg_a, None, None, model, FilterConfig())
                rows["learned_imu_only"] = metrics.ate(est, seq.p)
                cfg = FilterConfig(trust=trust, adaptive_q=True, bias_q_gain=10.0)
                est, _ = run(seq, fg, fa, conf, cam_naive, None, cfg)
                rows["ours_no_model"] = metrics.ate(est, seq.p)
                est, _ = run(seq, fg, fa, conf, cam_naive, model, cfg)
                rows["ours"] = metrics.ate(est, seq.p)
                results.setdefault(name, []).append({"room": test, "start": st, **rows})
            r = results[name][-len(STARTS):]
            print(f"  {test} {name:28s} " + "  ".join(f"{k}={np.mean([x[k] for x in r]):.3f}"
                                                      for k in ("naive_fusion", "ours_no_model", "ours", "learned_imu_only")), flush=True)
    summary = {n: {k: float(np.mean([x[k] for x in v])) for k in v[0] if k not in ("room", "start")}
               for n, v in results.items()}
    for n, v in results.items():
        summary[n]["n_clips"] = len(v)
        summary[n]["ours_median"] = float(np.median([x["ours"] for x in v]))
        summary[n]["naive_median"] = float(np.median([x["naive_fusion"] for x in v]))
    out = {"dataset": "TUM-VI room1 + room2 (real IMU, real frames, OptiTrack truth)", "clip_s": CLIP,
           "summary": summary, "per_clip": results, "runtime_s": time.time() - t0}
    Path("results/real_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
