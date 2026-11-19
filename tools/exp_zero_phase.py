"""Experiment: replace the causal pull back to the start (a first-order high-pass of position, --pos-tau 3 s) by a
zero-phase (forward-backward) one, offline, with a time constant that shortens as the organisers' damage ramps up.

    python tools/exp_zero_phase.py sweep      # development recordings, seed 42, every setting below
    python tools/exp_zero_phase.py check      # the frozen setting: all 17, seeds 42, 1, 7, 2026 (held-out checked once)
    python tools/exp_zero_phase.py tables     # markdown tables from results/track3/zero_phase_pull.json

Needs the damaged inputs of tools/competitor_scoreboard.py (`inputs`, plus seed 2026). main.py runs with the pull off
(`--pos-tau 0 --pos-tau-nonet 0`); position is then filtered here, which reproduces the shipped entry exactly when
the filter is the same causal high-pass (row "causal 3 s (= entry)"). Nothing in src/ changes.
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
from competitor_scoreboard import write_still  # noqa: E402
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402

INPUTS = ROOT / "runs" / "scoreboard" / "inputs"
WORK = ROOT / "runs" / "zero_phase"
RESULT = ROOT / "results" / "track3" / "zero_phase_pull.json"
DEV = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
       "euroc:V2_01_easy", "euroc:V2_02_medium"] + [f"tumvi:room{i}" for i in range(1, 5)]
HELD = ["euroc:MH_04_difficult", "euroc:MH_05_difficult", "euroc:V1_03_difficult", "euroc:V2_03_difficult",
        "tumvi:room5", "tumvi:room6"]
SEEDS = ["s42_accel_first", "s1_accel_first", "s7_accel_first", "s2026_accel_first"]
KEYS = [("ate_rmse_m", "ATE"), ("checkpoints_m", "checkpoints"), ("blind_mean_m", "blind 5 s"),
        ("dropout_m", "dropouts"), ("position_drift_1s_m", "drift 1 s")]


def ema(t, x, taus, reverse=False):
    """First-order low-pass with a per-row time constant, run forwards or backwards over uneven timestamps."""
    idx = range(len(t) - 1, -1, -1) if reverse else range(len(t))
    y = np.empty_like(x)
    s = prev = None
    for i in idx:
        if s is None:
            s = x[i].copy()
        else:
            s = s + (1 - np.exp(-abs(t[i] - t[prev]) / taus[i])) * (x[i] - s)
        y[i] = s
        prev = i
    return y


def ramp_taus(t, tau0, tau1, gamma=1.7):
    s = np.clip((t - t[0]) / max(t[-1] - t[0], 1e-9), 0, 1) ** gamma        # the organisers' damage ramp shape
    return tau0 + (tau1 - tau0) * s


def causal(t, p, tau):
    return p - ema(t, p, np.full(len(t), float(tau)))


def zero_phase(t, p, tau0, tau1=None, gamma=1.7):
    taus = ramp_taus(t, tau0, tau0 if tau1 is None else tau1, gamma)
    return p - ema(t, ema(t, p, taus), taus, reverse=True)


SETTINGS = {
    "entry (causal 3 s)": None,
    "causal 3 s (= entry)": lambda t, p: causal(t, p, 3.0),
    "no pull": lambda t, p: p,
    **{f"zero-phase {tau} s": (lambda tau: lambda t, p: zero_phase(t, p, tau))(tau) for tau in (1.5, 3, 6, 12, 24, 48, 96)},
    **{f"zero-phase {a}→{b} s, γ {g}": (lambda a, b, g: lambda t, p: zero_phase(t, p, a, b, g))(a, b, g)
       for a, b, g in ((24, 3, 1.7), (24, 6, 1.7), (24, 12, 1.7), (48, 3, 1.7), (48, 6, 1.7), (48, 12, 1.7),
                       (72, 3, 1.7), (96, 3, 1.7), (72, 2, 1.7), (48, 3, 1.0), (48, 3, 3.0), (72, 3, 3.0))},
    "standing still": "still",
}
FROZEN = "zero-phase 48→3 s, γ 1.7"


def entry_run(spec, s, off):
    d = INPUTS / s / spec.replace(":", "_")
    est = WORK / f"{s}_{spec.replace(':', '_')}_{'off' if off else 'entry'}.txt"
    if not est.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        extra = ["--pos-tau", "0", "--pos-tau-nonet", "0"] if off else []
        subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *extra],
                       check=True, capture_output=True)
    return est


def evaluate(spec, s, name):
    d = INPUTS / s / spec.replace(":", "_")
    truth = json.loads((d / "truth.json").read_text())
    f = SETTINGS[name]
    if f is None:
        est = entry_run(spec, s, off=False)
    elif f == "still":
        est = WORK / f"{s}_{spec.replace(':', '_')}_still.txt"
        write_still(d / "imu0" / "data.csv", est)
    else:
        e = np.loadtxt(entry_run(spec, s, off=True), comments="#")
        e[:, 1:4] = f(e[:, 0], e[:, 1:4])
        est = WORK / f"{s}_{spec.replace(':', '_')}_{abs(hash(name))}.txt"
        np.savetxt(est, e, fmt="%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    r = score(str(est), str(source(spec)[1]), truth, None)
    return {"data": spec, "set": s, "setting": name, **{k: r.get(k) for k, _ in KEYS}}


def run_jobs(jobs, phase):
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: entry_run(j[0], j[1], True), {(sp, s) for sp, s, _ in jobs}))
        rows = list(ex.map(lambda j: evaluate(*j), jobs))
    for r in rows:
        r["phase"] = phase
    old = json.loads(RESULT.read_text()) if RESULT.exists() else []
    old = [r for r in old if r["phase"] != phase]
    RESULT.write_text(json.dumps(old + rows, indent=0))


def tables():
    rows = json.loads(RESULT.read_text())

    def mean(sel, name, k):
        v = [r[k] for r in rows if sel(r) and r["setting"] == name]
        return np.mean(v) if v else np.nan

    out = ["### Development sweep (seed 42, 11 development recordings; mean metres)\n",
           "| setting | " + " | ".join(n for _, n in KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
    dev = lambda r: r["phase"] == "sweep"
    for name in SETTINGS:
        if any(r["setting"] == name for r in rows if dev(r)):
            out.append(f"| {name} | " + " | ".join(f"{mean(dev, name, k):.3f}" for k, _ in KEYS) + " |")
    chk = [r for r in rows if r["phase"] == "check"]
    if chk:
        out += ["\n### Check of the frozen setting (mean metres)\n",
                "| inputs | setting | " + " | ".join(n for _, n in KEYS) + " |", "|---|---|" + "---:|" * len(KEYS)]
        groups = [("held-out 6, seed 42", lambda r: r["set"] == "s42_accel_first" and r["data"] in HELD)]
        groups += [(f"all 17, seed {s.split('_')[0][1:]}", (lambda s: lambda r: r["set"] == s)(s)) for s in SEEDS]
        for lab, sel in groups:
            for name in ("entry (causal 3 s)", FROZEN, "standing still"):
                out.append(f"| {lab} | {name} | " + " | ".join(
                    f"{mean(lambda r: r['phase'] == 'check' and sel(r), name, k):.3f}" for k, _ in KEYS) + " |")
        out += ["\n### Recordings where the frozen setting is not better than standing still\n",
                "| inputs | recording | metric | frozen | entry | standing still |", "|---|---|---|---:|---:|---:|"]
        idx = {(r["set"], r["data"], r["setting"]): r for r in chk}
        for s in SEEDS:
            for spec in DEV + HELD:
                for k, n in KEYS[:2] + KEYS[3:4]:
                    z, e, st = (idx[(s, spec, x)][k] for x in (FROZEN, "entry (causal 3 s)", "standing still"))
                    if z >= st:
                        out.append(f"| seed {s.split('_')[0][1:]} | {spec} | {n} | {z:.2f} | {e:.2f} | {st:.2f} |")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "check", "tables"])
    a = ap.parse_args()
    if a.cmd == "sweep":
        run_jobs([(sp, "s42_accel_first", name) for name in SETTINGS for sp in DEV], "sweep")
    elif a.cmd == "check":
        run_jobs([(sp, s, name) for s in SEEDS for sp in DEV + HELD
                  for name in ("entry (causal 3 s)", FROZEN, "standing still")], "check")
    tables()


if __name__ == "__main__":
    main()
