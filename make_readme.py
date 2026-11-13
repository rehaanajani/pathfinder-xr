"""Writes README.md from README_template.md and results/summary.json (no hand-typed numbers)."""

import json
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

m = S["model"]
out = (T.replace("__RESULTS__", "\n".join(lines))
        .replace("__NSEQ__", str(len(S["sequences"])))
        .replace("__CLIP__", f"{S['clip_s']:.0f}")
        .replace("__PARAMS__", f"{m['params']:,}")
        .replace("__SIZE__", f"{m['size_mb']:.1f}")
        .replace("__MS__", f"{m['ms_per_prediction_cpu']:.1f}"))
Path("README.md").write_text(out)
print("wrote README.md")
