"""Writes docs/WIDER_SYSTEM.md (the camera + multi-IMU system, not scored) from README_template.md and results/summary.json (no hand-typed numbers)."""
import re

import json

import numpy as np
from pathlib import Path

S = json.loads(Path("results/summary.json").read_text())
T = Path("README_template.md").read_text()
SC = S["scenarios"]
NAMES = {
    "clean": "No fault", "camera_blackout": "Camera off 10 s", "camera_dark": "Dark room 10 s",
    "camera_blur": "Motion blur 10 s", "camera_wrong_fix": "Camera tracks the wrong thing 10 s",
    "imu_noise_all": "Noise on all IMUs", "one_imu_bias": "One IMU biased", "one_imu_stuck": "One IMU dies",
    "one_imu_spikes": "One IMU spikes", "one_imu_saturation": "One IMU saturates",
    "blackout_plus_one_imu_bias": "Camera off + one IMU biased", "dark_plus_imu_drift": "Dark room + all IMUs drift",
    "blackout_plus_vibration_all": "Camera off + vibration on all IMUs",
    "blackout_plus_spikes_all": "Camera off + spikes on all IMUs",
    "blackout_plus_saturation_all": "Camera off + all IMUs saturate",
    "blackout_plus_dropout_all": "Camera off + IMU packets dropped",
}


def x(a, b):
    return f"{a / b:.1f}x" if b > 0 else "-"


lines = []
key = ["camera_blackout", "blackout_plus_one_imu_bias", "blackout_plus_vibration_all", "dark_plus_imu_drift"]
best = max(key, key=lambda k: SC[k]["naive_fusion"]["ate_m"] / max(SC[k]["ours"]["ate_m"], 1e-9))
sw0 = S["blackout_sweep"]
def verdict(v):
    o, n = v["ours"]["ate_m"], v["naive_fusion"]["ate_m"]
    tol = 0.05 * n + 0.005                      # 5 % or 5 mm counts as level
    return "win" if o < n - tol else "loss" if o > n + tol else "tie"


vs = [verdict(v) for v in SC.values()]
wins, ties = vs.count("win"), vs.count("tie")
lines.append(f"**Headline.** {NAMES[best]}: naive fusion drifts **{SC[best]['naive_fusion']['ate_m']:.2f} m**, ours "
             f"**{SC[best]['ours']['ate_m']:.2f} m** ({x(SC[best]['naive_fusion']['ate_m'], SC[best]['ours']['ate_m'])} lower). "
             f"After a {sw0['lengths_s'][-1]} s blackout: **{sw0['naive_fusion'][-1]:.2f} m vs {sw0['ours'][-1]:.2f} m**. "
             f"Ours is clearly better in {wins} of {len(SC)} scenarios, level in {ties}, worse in "
             f"{len(SC) - wins - ties} (see Honest limits).\n")
lines.append("| Scenario | IMU only | Naive fusion | Ours w/o model | Ours w/ TLIO-trained model | **Ours** | Ours median | Gain vs naive |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for k, v in SC.items():
    g = lambda m: v[m]["ate_m"] if m in v else float("nan")
    lines.append(f"| {NAMES.get(k, k)} | {g('imu_only'):.1f} | {g('naive_fusion'):.2f} | {g('ours_no_model'):.2f} | "
                 f"{g('ours_tlio_model'):.2f} | **{g('ours'):.2f}** | {v['ours']['ate_median_m']:.2f} | "
                 f"{x(g('naive_fusion'), g('ours'))} |")
lines.append("\nATE in metres, mean over sequences (median for ours alongside). Per-sequence values are in `results/summary.json`.\n")

sw = S["blackout_sweep"]
lines.append("**Error when the camera comes back, by blackout length (m, mean):**\n")
lines.append("| Blackout | " + " | ".join(f"{L} s" for L in sw["lengths_s"]) + " |")
lines.append("|---|" + "---:|" * len(sw["lengths_s"]))
for m, lbl in [("imu_only", "IMU only"), ("naive_fusion", "Naive fusion"), ("ours", "**Ours**")]:
    lines.append(f"| {lbl} | " + " | ".join(f"{v:.2f}" for v in sw[m]) + " |")

iso = S["imu_isolation"]
lines.append("\n**Finding the faulty IMU** (share of faulty moments where the bad IMU is dropped and every good one kept):\n")
lines.append("| Fault | 2 IMUs | 3 IMUs | 5 IMUs |")
lines.append("|---|---:|---:|---:|")
for k, v in iso.items():
    lines.append(f"| {k} | " + " | ".join(f"{100 * v[c]:.0f}%" for c in ("2", "3", "5")) + " |")

extra = []
P = Path("results")
if (P / "real_summary.json").exists():
    R = json.loads((P / "real_summary.json").read_text())["summary"]
    extra.append("### On real sensor data (TUM-VI)\n")
    extra.append("Real Bosch BMI160 IMU, real fisheye camera frames, OptiTrack truth. Degradations are applied to the real "
                 "frames; camera health and trust are fitted on one room and tested on the other. Visual fixes come "
                 "from motion capture plus error driven by how many real KLT feature tracks survive. The motion network "
                 "was trained on Meta's headset data, so this is also a cross-device test. 60 s clips, ATE in metres.\n")
    extra.append("| Scenario | Naive fusion | **Ours** | Ours w/o model | Gain |")
    extra.append("|---|---:|---:|---:|---:|")
    for k, v in R.items():
        extra.append(f"| {NAMES.get(k, k)} | {v['naive_fusion']:.3f} | **{v['ours']:.3f}** | {v['ours_no_model']:.3f} | "
                     f"{x(v['naive_fusion'], v['ours'])} |")
    extra.append("")
if (P / "learned_health_eval.json").exists():
    L = json.loads((P / "learned_health_eval.json").read_text())
    V = json.loads((P / "real_camera_validation.json").read_text()) if (P / "real_camera_validation.json").exists() else None
    line = (f"**Camera health on real frames:** the learned score predicts the share of correct real KLT tracks with "
            f"correlation **{L['test_corr']:.2f}** on a room it never saw ({L['n_test']} frame pairs)")
    if V:
        line += f", up from {np.mean([v['corr'] for v in V.values()]):.2f} for the hand-tuned score"
    extra.append(line + ".\n")
if (P / "tlio_full_split.json").exists():
    F = json.loads((P / "tlio_full_split.json").read_text())
    o, t = F["summary"]["ours"], F["summary"]["tlio"]
    extra.append("### IMU only, full TLIO test split, next to published results\n")
    extra.append(f"All {F['n_sequences']} test sequences start to end (mean {F['mean_seconds']:.0f} s), one real IMU, no camera. "
                 "Published numbers use their own filter and alignment, so this is indicative, not like-for-like.\n")
    extra.append("| Method | ATE (m) |")
    extra.append("|---|---:|")
    for k, v in F["published"].items():
        extra.append(f"| {k} (published) | {v:.3f} |")
    extra.append(f"| Our network + our filter, TLIO-style training | {t['ate_raw_m']:.3f} |")
    extra.append(f"| Our network + our filter, fault-aware training (first filter) | {o['ate_raw_m']:.3f} |")
    lg = Path("results/logs/covscale_test.log")
    if lg.exists():
        import re
        m = re.search(r"cloning_x(\S+)\s+ATE raw mean ([\d.]+) m\s+median ([\d.]+)", lg.read_text())
        if m:
            extra.append(f"| **Same network, cloned poses + uncertainty x{m.group(1)} (chosen on validation)** | "
                         f"**{float(m.group(2)):.3f}** (median {float(m.group(3)):.2f}) |")
    extra.append("")
if (P / "onnx_benchmark.json").exists():
    O = json.loads((P / "onnx_benchmark.json").read_text())
    extra.append(f"**On-device readiness:** exported to ONNX ({O['fp32_mb']:.1f} MB, {O['fp32_ms_1core']:.2f} ms) and INT8 "
                 f"({O['int8_mb']:.2f} MB, {O['int8_ms_1core']:.2f} ms) per prediction on one CPU core; INT8 window error "
                 f"{O['rmse_int8_m']:.3f} m vs {O['rmse_fp32_m']:.3f} m. Both files are in the `models-v1` release.\n")

m = S["model"]
out = (T.replace("__RESULTS__", "\n".join(lines)).replace("__EXTRA__", "\n".join(extra))
        .replace("__NSEQ__", str(len(S["sequences"])))
        .replace("__CLIP__", f"{S['clip_s']:.0f}")
        .replace("__PARAMS__", f"{m['params']:,}")
        .replace("__SIZE__", f"{m['size_mb']:.1f}")
        .replace("__MS__", f"{m['ms_per_prediction_cpu']:.1f}"))
def _relink(md):
    """The page lives in docs/: make repository-relative links work from there."""
    def f(m):
        url = m.group(2)
        if re.match(r"(https?:|#|mailto:|\.\./|/)", url):
            return m.group(0)
        url = url[5:] if url.startswith("docs/") else "../" + url
        return f"{m.group(1)}({url})"
    return re.sub(r"(\]|!\[[^\]]*\])\(([^)\s]+)\)", f, md)


out = ("> The camera + multi-IMU system built before the official Track 3 brief. Not the scored entry; see the "
       "[main README](../README.md).\n\n" + _relink(out))
Path("docs/WIDER_SYSTEM.md").write_text(out)
print("wrote docs/WIDER_SYSTEM.md")
