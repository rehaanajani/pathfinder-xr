"""Experiment: an offline zero-phase pull back to the start, on the drone mount only (rotor drag on).

    python tools/exp_zero_phase_drone.py sweep               # development recordings, seed 42
    python tools/exp_zero_phase_drone.py check --setting S   # all 17, seeds 42, 1, 7 (held-out checked once)

The entry runs twice per file: as shipped, and with every pull off. On recordings where rotor drag turns on (the
drone mount), position from the pull-off run is high-passed forwards and backwards (zero phase) with a time constant
that shortens along the organisers' damage ramp; elsewhere the shipped output is kept. Inputs come from
tools/competitor_scoreboard.py (`inputs`). Rows go to runs/zp_drone/<phase>.json.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402

INPUTS = ROOT / "runs" / "scoreboard" / "inputs"
WORK = ROOT / "runs" / "zp_drone"
DEV = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
       "euroc:V2_01_easy", "euroc:V2_02_medium"] + [f"tumvi:room{i}" for i in range(1, 5)]
HELD = ["euroc:MH_04_difficult", "euroc:MH_05_difficult", "euroc:V1_03_difficult", "euroc:V2_03_difficult",
        "tumvi:room5", "tumvi:room6"]
KEYS = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m", "position_drift_1s_m"]
OFF = ["--pos-tau", "0", "--pos-tau-nonet", "0", "--drag-pos-tau", "0"]


def ema(t, x, taus, reverse=False):
    idx = range(len(t) - 1, -1, -1) if reverse else range(len(t))
    y = np.empty_like(x)
    s = prev = None
    for i in idx:
        s = x[i].copy() if s is None else s + (1 - np.exp(-abs(t[i] - t[prev]) / taus[i])) * (x[i] - s)
        y[i] = s
        prev = i
    return y


def zero_phase(t, p, tau0, tau1, gamma=1.7):
    u = np.clip((t - t[0]) / max(t[-1] - t[0], 1e-9), 0, 1) ** gamma
    taus = tau0 + (tau1 - tau0) * u
    return p - ema(t, ema(t, p, taus), taus, reverse=True)


SETTINGS = {"entry": None}
for a, b in ((24, 6), (48, 6), (48, 12), (96, 6), (96, 12), (24, 12), (48, 24), (96, 24), (12, 12), (24, 24)):
    SETTINGS[f"zp {a}->{b}"] = (a, b)
    SETTINGS[f"zpb {a}->{b} entry"] = (a, b, "entry")
    SETTINGS[f"zpb {a}->{b} off"] = (a, b, "off")
    SETTINGS[f"zpb {a}->{b} keep"] = (a, b, "keep")


def run(d, est, extra):
    if not est.exists():
        p = subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *extra],
                           check=True, capture_output=True, text=True)
        est.with_suffix(".err").write_text(p.stderr)
    return est


def evaluate(spec, s, name):
    d = INPUTS / s / spec.replace(":", "_")
    tag = f"{s}_{spec.replace(':', '_')}"
    entry = run(d, WORK / f"{tag}_entry.txt", [])
    drone = "rotor-drag aiding on" in entry.with_suffix(".err").read_text()
    est = entry
    if SETTINGS[name] is not None and drone:
        e = np.loadtxt(run(d, WORK / f"{tag}_off.txt", OFF), comments="#")
        cfg = SETTINGS[name]
        if len(cfg) == 2:
            e[:, 1:4] = zero_phase(e[:, 0], e[:, 1:4], *cfg)
        else:   # smooth only up to the failure, then the blind motion of `cfg[2]` ("entry" or "off") from there
            raw = np.loadtxt(d / "imu0" / "data.csv", delimiter=",", comments="#", skiprows=1)
            k = len(raw)
            while k > 0 and not np.any(raw[k - 1, 1:7]):
                k -= 1
            src = e.copy() if cfg[2] == "off" else np.loadtxt(entry, comments="#")
            e[:k, 1:4] = zero_phase(e[:k, 0], e[:k, 1:4], cfg[0], cfg[1])
            if cfg[2] == "keep":   # the shipped entry's own blind positions, untouched
                e[k:, 1:4] = np.loadtxt(entry, comments="#")[k:, 1:4]
            else:
                e[k:, 1:4] = e[k - 1, 1:4] + (src[k:, 1:4] - src[k - 1, 1:4])
        est = WORK / f"{tag}_{name.replace(' ', '_').replace('>', '')}.txt"
        np.savetxt(est, e, fmt="%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    truth = json.loads((d / "truth.json").read_text())
    r = score(str(est), str(source(spec)[1]), truth, None)
    return {"data": spec, "set": s, "setting": name, "drone": drone, **{k: r.get(k) for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "check"])
    ap.add_argument("--setting")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "sweep":
        jobs = [(sp, "s42_accel_first", n) for sp in DEV for n in SETTINGS]
    else:
        jobs = [(sp, s, n) for sp in DEV + HELD for s in ("s42_accel_first", "s1_accel_first", "s7_accel_first")
                for n in ("entry", a.setting)]
    def prepare(job):   # both runs of each file, once, before any thread reads them
        sp, s = job
        d, tag = INPUTS / s / sp.replace(":", "_"), f"{s}_{sp.replace(':', '_')}"
        e = run(d, WORK / f"{tag}_entry.txt", [])
        if "rotor-drag aiding on" in e.with_suffix(".err").read_text():
            run(d, WORK / f"{tag}_off.txt", OFF)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(prepare, {(j[0], j[1]) for j in jobs}))
        rows = list(ex.map(lambda j: evaluate(*j), jobs))
    (WORK / f"{a.cmd}.json").write_text(json.dumps(rows, indent=0))
    groups = {"MH": lambda x: ":MH" in x, "Vicon": lambda x: ":V" in x, "TUM": lambda x: "tumvi" in x, "all": lambda x: True}
    for s in sorted({r["set"] for r in rows}):
        for part, specs in (("dev", DEV), ("held", HELD), ("all", DEV + HELD)):
            sub = [r for r in rows if r["set"] == s and r["data"] in specs]
            if not sub:
                continue
            print(f"\n== {s} {part}  (ATE / ckpt / blind / drop / drift)")
            for n in dict.fromkeys(r["setting"] for r in sub):
                rr = [r for r in sub if r["setting"] == n]
                line = " | ".join(f"{g} " + " ".join(f"{np.mean([r[k] for r in rr if f(r['data'])]):.3f}" for k in KEYS[:4])
                                  for g, f in groups.items() if any(f(r["data"]) for r in rr))
                print(f"{n:12s} {line}")


if __name__ == "__main__":
    main()
