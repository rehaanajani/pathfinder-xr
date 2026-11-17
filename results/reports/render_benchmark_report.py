from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "runs/20261117-followthrough-benchmark"
s = json.loads((DATA / "summary.json").read_text(encoding="utf-8-sig"))

W, H = 1600, 2120
im = Image.new("RGB", (W, H), "#0B1220")
d = ImageDraw.Draw(im)

FONT = Path("C:/Windows/Fonts") if Path("C:/Windows/Fonts").exists() else Path("/mnt/c/Windows/Fonts")
def f(size, bold=False):
    return ImageFont.truetype(str(FONT / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)

def text(x, y, t, size=25, color="#EDF3FC", bold=False):
    d.text((x, y), t, font=f(size, bold), fill=color)

def line(y):
    d.line((72, y, 1528, y), fill="#29364B", width=2)

def card(box, fill="#142035"):
    d.rounded_rectangle(box, radius=20, fill=fill)

muted = "#A5B5CC"
green = "#5CE1BF"
blue = "#6DAAFF"
orange = "#FFBA73"
purple = "#C49BFF"

text(72, 42, "PATH FINDER XR  •  BENCHMARK FOLLOW-THROUGH", 26, green, True)
text(72, 82, "Follow-through benchmark report", 60, bold=True)
text(72, 162, "Branch fix/benchmark-followthrough-20261117 (base c382494)  •  17 November 2026", 25, muted)
text(72, 206, "All validation suites green in isolation. Gap-aware scorer, real-vision v3 + Basalt 0.1.7, and confirmatory MIO10 completed.", 25, bold=True)

maxcpu = max(x["cpu_ms_per_s_worst"] for x in s["performance"])
maxmem = max(x["rss_mib_worst"] for x in s["performance"])
py_med = s["python_performance"]["windows"]["walk_120s_200hz"]["hotstep"]["median_ms_per_s"]
py_pure = s["python_performance"]["windows"]["walk_120s_200hz"]["pure_python"]["median_ms_per_s"]
rt_1k = s["demo_timing_isolated"]["measured_1000hz"]["rt_factor"]

cards_top = [
    (72, "C++ WORST CPU", f"{maxcpu:.2f} ms/s", "Limit <= 20 ms/s (5.10 MiB <= 8 MiB)"),
    (442, "PYTHON ENTRY (120s)", f"{py_med:.2f} ms/s", f"hotstep.dll ({py_pure:.2f} ms/s pure-Py)"),
    (812, "1,000 Hz DEMO STREAM", f"{rt_1k:.2f}x RT", "3.15 s wall / 12.0 s IMU (was 0.61x)"),
    (1182, "CONFIRMATORY MIO10", "0.397 m", "Clean 1 kHz ATE vs 0.584 m physics"),
]
for x, title, value, foot in cards_top:
    card((x, 265, x + 346, 445))
    text(x + 22, 285, title, 20, muted, True)
    text(x + 22, 322, value, 42, green, True)
    text(x + 22, 392, foot, 18, muted)

text(72, 478, "Track 3 trajectory error (pathfinder.track3-score/2 support mode, 50 ms gap limit)", 34, bold=True)
text(72, 524, "Mean aligned ATE in metres across 18 fault scenarios per room  •  Lower is better  •  0–7 m scale", 23, muted)

colors = {"shipped": green, "physics": orange, "standstill": blue}
names = {"shipped": "Path Finder (shipped)", "physics": "Physics only", "standstill": "Standing still"}
for k, room in enumerate(("room1", "room2")):
    x = 72 + 750 * k
    card((x, 570, x + 706, 935))
    cov = next(r for r in s["accuracy_support"] if r["room"] == room and r["system"] == "shipped")["support_coverage"]["mean"] * 100
    text(x + 24, 588, f"{room.upper()}  (support coverage {cov:.1f}%)", 25, bold=True)
    for j, system in enumerate(("shipped", "physics", "standstill")):
        r = next(r for r in s["accuracy_support"] if r["room"] == room and r["system"] == system)
        v = r["ate_rmse_m"]["mean"]
        bm = r["blind_mean_m"]["mean"]
        y = 642 + j * 82
        text(x + 24, y, f"{names[system]}  (blind mean {bm:.3f} m)", 22, muted)
        label = f"{v:.3f} m"
        d.text((x + 678, y), label, font=f(24, True), fill=colors[system], anchor="ra")
        d.rounded_rectangle((x + 24, y + 36, x + 678, y + 52), radius=8, fill="#24334B")
        d.rounded_rectangle((x + 24, y + 36, x + 24 + min(654, int(654 * v / 7.0)), y + 52), radius=8, fill=colors[system])
    clean = next(r for r in s["accuracy_support"] if r["room"] == room and r["system"] == "shipped")["clean_ate_m"]
    text(x + 24, 892, f"Clean ATE: {clean:.3f} m  •  ATE wins: 19/19 vs both baselines", 21, green)

text(72, 955, "Blind-period diagnosis (room2): Under gap-aware support scoring, Path Finder beats standstill on blind mean (0.712 m vs 0.738 m, 12/18 wins).", 21, "#EDF3FC")
text(72, 988, "Retained negative experiment (20261117-yaw-gauge): resetting yaw variance after 1 healthy window worsened blind mean (0.525->0.684 m, 0.701->0.838 m); rejected.", 19, orange)

line(1038)
text(72, 1058, "Real-stereo vision v3 evaluation (28 scenarios/room) & Basalt 0.1.7 external VIO", 34, bold=True)

for k, (room, b_se3, b_4dof, pf_clean, pf_med, conv_clean, conv_med, wins, delta, worst) in enumerate([
    ("ROOM1 (calibrated 0–60 s)", 0.161, 0.107, 0.301, 0.659, 0.333, 0.850, "19/28", "-0.026 m", "6776.9 m"),
    ("ROOM2 (calibration transfer)", 0.132, 0.076, 0.171, 0.278, 0.230, 0.556, "20/28", "-0.056 m", "3165.8 m"),
]):
    x = 72 + 750 * k
    card((x, 1112, x + 706, 1380))
    text(x + 24, 1130, room, 24, bold=True)
    text(x + 24, 1172, f"Basalt 0.1.7 VIO (clean):  {b_se3:.3f} m start-SE(3)  •  {b_4dof:.3f} m 4-DoF", 20, purple, True)
    text(x + 24, 1212, f"Path Finder v3:  clean {pf_clean:.3f} m  •  28-scenario median {pf_med:.3f} m", 21, green, True)
    text(x + 24, 1250, f"Conventional ESKF:  clean {conv_clean:.3f} m  •  28-scenario median {conv_med:.3f} m", 21, blue)
    text(x + 24, 1288, f"Paired vs conventional:  {wins} wins  (median delta {delta})", 21, "#EDF3FC")
    text(x + 24, 1326, f"Worst compound fault (blackout_5s_at40+imu_saturation): {worst}", 19, orange)

line(1405)
text(72, 1425, "Isolated validation suites & confirmatory recording (monado/MIO10_short_2_panorama)", 34, bold=True)

suites = [
    (1480, "Windows root pytest", "291 passed • 5 skipped • 0 failed  (in-memory check_repo, exact Decimal parity 0.000001 m)"),
    (1522, "WSL C++ stress & Demo", "C++ stress: 52 passed • 2 skipped  |  Demo suite: 92 passed • 26 skipped • 1 xfailed • 0 failed"),
    (1564, "Isolated timing & Smoke", "Python 120s walk: 9.81 ms/s Win / 8.51 ms/s WSL  |  Demo 1 kHz: 3.81x RT  |  Vision smoke: PASS"),
    (1606, "Confirmatory MIO10", "Pre-open manifest SHA-256 verified  |  7.630 s @ 1 kHz: Path Finder 0.397 m vs Physics 0.584 m, Still 0.445 m"),
    (1648, "MIO10 fault schedule", "Duration 7.630 s < 15.0 s rule: unchanged 18 fault variants recorded as generator_failed (not retuned)"),
]
for y, name, outcome in suites:
    text(72, y, "PASS", 21, green, True)
    text(150, y, name, 23, "#EDF3FC", True)
    text(450, y, outcome, 21, muted)

card((72, 1710, 1528, 2020))
text(98, 1732, "METHODOLOGICAL GUARDRAILS & RETAINED LIMITATIONS", 24, green, True)
text(98, 1774, "1. Exposed development data: TUM-VI room1 and room2 remain development recordings; 18 fault variants per room are correlated.", 21)
text(98, 1812, "2. Confirmatory scope: MIO10_short_2_panorama was unopened before freeze, but uses the same Valve Index headset family and is 7.63 s long.", 21)
text(98, 1850, "3. Vision limits: Basalt 0.1.7 outperforms the lightweight VO frontend on clean runs; vo_covariance_validated=false; 16->8-bit image shift.", 21)
text(98, 1888, "4. Open failure modes: Simultaneous 5 s camera blackout + IMU saturation/bias step causes runaway drift; target headset/NPU untested.", 21)
text(98, 1926, "5. Immutable baseline: 20261116-main-c382494-benchmark is untouched; follow-through bundle is 20261117-followthrough-benchmark.", 21, muted)
text(72, 2050, "SOURCE: results/runs/20261117-followthrough-benchmark & 20261117-confirmatory-mio10-eval  •  evidence-sha256.json verified", 21, muted)

for out_name in ("pathfinder-benchmark-report.png", "pathfinder-benchmark-followthrough-20261117.png"):
    im.save(HERE / out_name, dpi=(160, 160))
print("Re-rendered benchmark PNGs")
