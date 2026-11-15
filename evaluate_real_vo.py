"""Real visual odometry + real IMU on TUM-VI (issue #2, first vertical slice).

    python evaluate_real_vo.py [--workers 9] [--out results]

Everything the estimators see is measured: the real BMI160 IMU and motion estimated from the real
stereo images by ``pathfinder.stereo_vo``. Faults are applied to the IMAGES before tracking (dark,
blur, blackout) and to the IMU stream. Motion-capture truth is used only to initialise the start
pose (benchmark mode, labelled) and for scoring.

Protocol
  * Clips: two non-overlapping 60 s clips per room (5-65 s, 70-130 s), faults at 20 s.
  * VO noise model (std vs. number of PnP inliers) is calibrated on the OTHER room, then frozen.
  * Primary setting: the single real IMU. A separate, labelled study adds two simulated IMU copies
    (noise + bias each) to exercise fault isolation.
  * These recordings were already used during development (health model, earlier benchmark), so the
    results are development evidence, not an untouched final test set.

Methods
  vo_only       visual odometry alone (holds still when tracking fails)
  conventional  real IMU + VO, fixed VO noise, no gating: a standard loosely coupled filter
  ours_1imu     real IMU + VO with calibrated noise, robust gate (single real IMU, primary)
  ours_3imu     + parity fault isolation over 3 IMUs (two are simulated copies, labelled)
  ours_1imu_net ours_1imu + the learned motion network (trained on Meta's TLIO headset data, so on
                TUM-VI this is a cross-device test of the network)
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pathfinder import data_tumvi as T
from pathfinder import metrics
from pathfinder.data_tlio import redundant_imus
from pathfinder.faults import Fault, apply_imu_faults, camera_schedule
from pathfinder.filter import FilterConfig, run
from pathfinder.pipeline import fuse_imus
from pathfinder.real_camera import degrade_real
from pathfinder.stereo_vo import StereoCalib, StereoVO, load_pair

CLIP = 60.0
STARTS = [5.0, 70.0]
EVERY = 1                 # use every frame (20 Hz)


def scenarios(m=20.0, dur=10.0):
    big_bias = {"gyro": [0.08, -0.05, 0.04], "acc": [0.8, -0.5, 0.4]}
    return {
        "clean": [],
        "camera_blackout": [Fault("camera_blackout", m, dur)],
        "camera_dark": [Fault("camera_dark", m, dur, params={"level": 0.06})],
        "camera_blur": [Fault("camera_blur", m, dur, params={"pixels": 30})],
        "dark_plus_imu_drift": [Fault("camera_dark", m, dur, params={"level": 0.06}),
                                Fault("imu_bias_drift", 0, 1e9, None)],
        "blackout_plus_one_imu_bias": [Fault("camera_blackout", m, dur), Fault("imu_bias", max(m - 5, 0), 1e9, 0, big_bias)],
        "blackout_plus_saturation_all": [Fault("camera_blackout", m, dur),
                                         Fault("imu_saturation", m, dur, None, {"gyro_max": 1.0, "acc_max": 11.0})],
    }


def _degrade(img, brightness, blur, rng):
    if brightness >= 1.0 and blur <= 0:
        return img                                       # clean frames stay exactly the real image
    out = degrade_real(img.astype(np.float32) / 255.0, brightness, blur, rng)
    return (out * 255).astype(np.uint8)


def stereo_calib(seq):
    if seq.meta.get("source") == "monado":
        from pathfinder import data_monado as M
        return M.calib(seq)
    return StereoCalib.load(Path(seq.meta["dir"]).parent / "dso" / "camchain.yaml")


def run_vo(seq, ft, fp, faults, seed=0):
    """Visual odometry over the clip's (degraded) real frames. Returns per-frame arrays."""
    d = Path(seq.meta["dir"])
    vo = StereoVO(stereo_calib(seq))
    sched = camera_schedule(ft, faults)
    rng = np.random.default_rng(seed)
    n = len(ft)
    out = {"frame_idx": np.minimum(np.round(ft * seq.rate).astype(int), len(seq.t) - 1),
           "t": np.full((n, 3), np.nan), "R": np.tile(np.eye(3), (n, 1, 1)), "inliers": np.zeros(n, int)}
    for j in range(n):
        if not sched["available"][j]:
            vo.prev = None                               # no image: the next frame cannot be chained
            continue
        a, b = load_pair(fp[j], d / "cam1" / "data")
        a = _degrade(a, sched["brightness"][j], sched["blur"][j], rng)
        b = _degrade(b, sched["brightness"][j], sched["blur"][j], rng)
        r = vo.step(a, b)
        if r is not None:
            out["t"][j], out["R"][j], out["inliers"][j] = r["t"], r["R"], r["inliers"]
    return out


def truth_rel(seq, vo):
    Rg = R.from_quat(seq.q)
    idx = vo["frame_idx"]
    et, er = np.full((len(idx), 3), np.nan), np.full((len(idx), 3), np.nan)
    for j in range(1, len(idx)):
        if not np.isfinite(vo["t"][j]).all():
            continue
        a, b = idx[j - 1], idx[j]
        et[j] = vo["t"][j] - Rg[a].as_matrix().T @ (seq.p[b] - seq.p[a])
        er[j] = (R.from_matrix((Rg[a].inv() * Rg[b]).as_matrix()).inv() * R.from_matrix(vo["R"][j])).as_rotvec()
    return et, er


def collect_noise(seq, ft, fp, seed=0):
    """(inliers, translation error, rotation error) per frame on a calibration recording, under random
    dark/blur episodes."""
    rng = np.random.default_rng(seed)
    T = float(seq.t[-1])
    faults = []
    for _ in range(max(2, int(12 * T / 140))):
        L = rng.uniform(1, min(6, max(1.5, T / 4)))
        st = rng.uniform(0, max(0.1, T - L))
        if rng.random() < 0.5:
            faults.append(Fault("camera_dark", st, L, params={"level": float(np.exp(rng.uniform(np.log(0.03), np.log(0.6))))}))
        else:
            faults.append(Fault("camera_blur", st, L, params={"pixels": float(rng.uniform(5, 35))}))
    vo = run_vo(seq, ft, fp, faults, seed=seed)
    et, er = truth_rel(seq, vo)
    ok = np.isfinite(et).all(1)
    return vo["inliers"][ok], et[ok], er[ok]


def calibrate_noise(seq, ft, fp, seed=0):
    """VO error vs. inlier count on one calibration recording."""
    return bin_noise(*collect_noise(seq, ft, fp, seed))


def bin_noise(n_in, et, er):
    ok = np.ones(len(n_in), bool)
    edges = np.unique(np.quantile(n_in, np.linspace(0, 1, 9)).astype(int))
    c, st, sr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n_in >= lo) & (n_in <= hi)
        if m.sum() >= 20:
            # Robust per-axis spread (MAD), so a few gross failures set the gate, not the noise level.
            c.append(float(np.median(n_in[m])))
            st.append(float(1.4826 * np.median(np.abs(et[m]))))
            sr.append(float(1.4826 * np.median(np.abs(er[m]))))
    for j in range(len(st) - 2, -1, -1):                 # fewer inliers never means less noise
        st[j], sr[j] = max(st[j], st[j + 1]), max(sr[j], sr[j + 1])
    return {"inliers": c, "std_t": st, "std_r": sr, "frames": int(ok.sum())}


def with_cov(vo, noise, fixed=False):
    n = len(vo["t"])
    if fixed:
        st = np.full(n, float(np.median(noise["std_t"])))
        sr = np.full(n, float(np.median(noise["std_r"])))
    else:
        st = np.interp(vo["inliers"], noise["inliers"], noise["std_t"])
        sr = np.interp(vo["inliers"], noise["inliers"], noise["std_r"])
    return dict(vo, cov_t=np.eye(3)[None] * (st ** 2)[:, None, None], cov_r=np.eye(3)[None] * (sr ** 2)[:, None, None])


def vo_only(seq, vo):
    est = np.zeros_like(seq.p)
    p, Rw = seq.p[0].copy(), R.from_quat(seq.q[0]).as_matrix()
    last = 0
    for j, k in enumerate(vo["frame_idx"]):
        est[last:k] = p
        if j > 0 and np.isfinite(vo["t"][j]).all():
            p = p + Rw @ vo["t"][j]
            Rw = Rw @ vo["R"][j]
        last = k
    est[last:] = p
    return est


_W = {}


def _init(clips, noise, overrides=None, fault_t=(20.0, 10.0)):
    import torch
    from pathfinder.predictor import Predictor
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _W["clips"], _W["noise"] = clips, noise
    _W["overrides"] = overrides or {}
    _W["fault_t"] = fault_t
    _W["net"] = Predictor("models/dispnet_faults/best.pt")


def _job(job):
    room, st, name = job
    seq1, ft, fp = _W["clips"][(room, st)]
    faults = scenarios(*_W["fault_t"])[name]
    noise = _W["noise"][(room, st)]
    t0 = time.time()
    vo = run_vo(seq1, ft, fp, faults, seed=int(st))
    vo_s = time.time() - t0
    bad1 = apply_imu_faults(seq1, faults, seed=int(st))
    seq3 = redundant_imus(seq1, 3, seed=int(st))
    bad3 = apply_imu_faults(seq3, faults, seed=int(st))
    fg, fa, info = fuse_imus(bad3)
    conf = np.repeat(info["confidence"], 20)[: len(seq1.t)]
    from dataclasses import replace
    ours_cfg = replace(FilterConfig(), **_W["overrides"])  # robust soft gate on; VO noise from calibration
    rows, end = {}, min(int(sum(_W["fault_t"]) * seq1.rate), len(seq1.t) - 1)
    ests = {
        "vo_only": vo_only(seq1, vo),
        "conventional": run(seq1, bad1.gyro[0], bad1.acc[0], None, None, None,
                            FilterConfig(use_gate=False), vo=with_cov(vo, noise, fixed=True))[0],
        "ours_1imu": run(seq1, bad1.gyro[0], bad1.acc[0], None, None, None, ours_cfg, vo=with_cov(vo, noise))[0],
        "ours_3imu": run(seq1, fg, fa, conf, None, None, ours_cfg, vo=with_cov(vo, noise))[0],
        "ours_1imu_net": run(seq1, bad1.gyro[0], bad1.acc[0], None, None, _W["net"], ours_cfg,
                             vo=with_cov(vo, noise))[0],
    }
    for m, est in ests.items():
        rows[m] = {"ate_m": metrics.ate(est, seq1.p), "ate_raw_m": metrics.ate(est, seq1.p, align=False),
                   "end_of_fault_err_m": float(np.linalg.norm(est[end] - seq1.p[end])),
                   "rte_10s_m": metrics.rte(est, seq1.p, int(10 * seq1.rate))}
    solved = float(np.isfinite(vo["t"]).all(1).mean())
    return {"room": room, "start": st, "scenario": name, "vo_solved_frac": solved,
            "vo_seconds": vo_s, "frames": len(ft), **rows}


def _clip(full, frames, st, length):
    a, b = int(st * 200), min(int((st + length) * 200) + 1, len(full.t))
    s = full.slice(a, b)
    s.t = s.t - s.t[0]
    s.meta = dict(full.meta, t_abs0=full.meta["t_abs0"] + st)
    ft_all, fp_all = frames
    m = (ft_all >= st) & (ft_all <= st + length)
    return s, ft_all[m] - st, [p for p, k in zip(fp_all, m) if k]


def tumvi_protocol():
    rooms = T.available()
    full = {r: T.load(r) for r in rooms}
    frames = {r: T.frames(full[r]) for r in rooms}
    noise = {}
    with ProcessPoolExecutor(2) as ex:
        cal_for = {test: [r for r in rooms if r != test][0] for test in rooms}
        futs = {test: ex.submit(calibrate_noise, full[cal], *frames[cal]) for test, cal in cal_for.items()}
        for test, f in futs.items():
            noise[test] = f.result()
            print(f"VO noise for {test} (calibrated on {cal_for[test]}): std_t {np.round(noise[test]['std_t'], 4).tolist()}", flush=True)
    clips = {(r, st): _clip(full[r], frames[r], st, CLIP) for r in rooms for st in STARTS}
    protocol = {"clips": [f"{r} {st:.0f}-{st + CLIP:.0f} s" for r in rooms for st in STARTS],
                "faults": "at 20 s for 10 s", "vo_noise": "calibrated on the other room",
                "status": "development evidence (recordings seen before)",
                "init": "truth start pose (benchmark mode)", "simulated": "ours_3imu adds 2 simulated IMU copies"}
    return ({(r, st): clips[(r, st)] for (r, st) in clips}, {(r, st): noise[r] for (r, st) in clips} | noise,
            (20.0, 10.0), "TUM-VI room1 + room2: real IMU, real stereo VO, OptiTrack truth for scoring only", protocol)


MONADO_CAL = {"odyssey_plus": ["MOO09_short_1_updown", "MOO11_short_3_backandforth"],
              "valve_index": ["MIO09_short_1_updown", "MIO11_short_3_backandforth"]}
MONADO_TEST = ["MOO05_inspect_easy", "MOO07_mapping_easy", "MOO08_mapping_hard", "MIO08_mapping_hard"]
MONADO_CLIP = 30.0


def monado_protocol():
    from pathfinder import data_monado as M
    have = set(M.available())
    noise = {}
    for h, seqs in MONADO_CAL.items():
        parts = [collect_noise(s, *M.frames(s), seed=i) for i, s in enumerate(M.load(n) for n in seqs if n in have)]
        if parts:
            noise[h] = bin_noise(*[np.concatenate(x) for x in zip(*parts)])
            print(f"VO noise for {h} (calibrated on {', '.join(seqs)}): std_t {np.round(noise[h]['std_t'], 4).tolist()}", flush=True)
    clips = {}
    for n in MONADO_TEST:
        if n not in have:
            continue
        full = M.load(n)
        fr = M.frames(full)
        st = 0.0
        while full.t[-1] - st >= 19.0:                       # non-overlapping clips of up to 30 s
            clips[(n, st)] = _clip(full, fr, st, min(MONADO_CLIP, full.t[-1] - st))
            st += MONADO_CLIP + 3.0
    noise_by_clip = {k: noise[M.headset(k[0])] for k in clips}
    protocol = {"clips": [f"{n} {st:.0f}-{st + len(c[0].t) / 200:.0f} s" for (n, st), c in clips.items()],
                "faults": "at 8 s for 8 s",
                "vo_noise": "calibrated per headset on separate short recordings (never the test recordings)",
                "status": "first use of these recordings; settings were chosen earlier on TLIO validation",
                "init": "truth start pose (benchmark mode)", "simulated": "ours_3imu adds 2 simulated IMU copies",
                "ground_truth": "3 SteamVR Lighthouse base stations (~1 cm), transformed to the IMU pose by the dataset authors"}
    return clips, noise_by_clip, (8.0, 8.0), \
        "Monado SLAM Dataset (real VR headsets: Samsung Odyssey+, Valve Index), real IMU + real stereo VO", protocol


def _parse_set(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        t = type(getattr(FilterConfig(), k))
        out[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dataset", default="tumvi", choices=["tumvi", "monado"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a FilterConfig field for the 'ours' methods (not the conventional baseline)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.dataset == "monado":
        clips, noise, fault_t, desc, protocol = monado_protocol()
    else:
        clips, noise, fault_t, desc, protocol = tumvi_protocol()
    names = list(scenarios(*fault_t))
    jobs = [(g, st, n) for (g, st) in clips for n in names]
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(clips, noise, _parse_set(args.set), fault_t)) as ex:
        rows = list(ex.map(_job, jobs))
    methods = ["vo_only", "conventional", "ours_1imu", "ours_3imu", "ours_1imu_net"]
    summary = {}
    for n in names:
        rs = [r for r in rows if r["scenario"] == n]
        summary[n] = {m: {k: float(np.mean([r[m][k] for r in rs])) for k in rs[0][m]} for m in methods}
        summary[n]["per_clip_ate"] = {m: [round(r[m]["ate_m"], 4) for r in rs] for m in methods}
        summary[n]["vo_solved_frac"] = float(np.mean([r["vo_solved_frac"] for r in rs]))
        print(f"{n:30s} " + "  ".join(f"{m}={summary[n][m]['ate_m']:.3f}" for m in methods), flush=True)
    vo_fps = float(sum(r["frames"] for r in rows) / sum(r["vo_seconds"] for r in rows))
    res = {"dataset": desc, "protocol": protocol,
           "vo_noise": {(k if isinstance(k, str) else f"{k[0]} @{k[1]:.0f}s"): v for k, v in noise.items()}, "vo_fps_1core": vo_fps, "summary": summary, "per_run": rows,
           "runtime_s": time.time() - t0}
    (out / f"real_vo_{args.dataset}_summary.json").write_text(json.dumps(res, indent=1))
    from pathfinder.manifest import write as write_manifest
    write_manifest(out / f"real_vo_{args.dataset}_manifest.json", args=args, models=["models/dispnet_faults/best.pt"])
    print(f"VO speed {vo_fps:.0f} stereo frames/s on one core; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
