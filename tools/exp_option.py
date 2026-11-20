"""Compare main.py option sets against the shipped defaults and standing still, on the organisers' damage.

    python tools/exp_option.py --name NAME sweep --variants '{"bridge 0.3": ["--bridge-max", "0.3"], ...}'
    python tools/exp_option.py --name NAME check --variant '["--bridge-max", "0.3"]'
    python tools/exp_option.py --name NAME tables

Inputs are the damaged files of tools/competitor_scoreboard.py (accel-first; seeds 42, 1, 7, 2026). `sweep` runs the
11 development recordings at seed 42; `check` runs the frozen variant once on all 17 recordings at all four seeds.
Raw rows go to results/track3/<NAME>.json; `tables` prints markdown tables from that file only.
"""
import argparse
import hashlib
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
WORK = ROOT / "runs" / "exp_option"
DEV = ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
       "euroc:V2_01_easy", "euroc:V2_02_medium"] + [f"tumvi:room{i}" for i in range(1, 5)]
HELD = ["euroc:MH_04_difficult", "euroc:MH_05_difficult", "euroc:V1_03_difficult", "euroc:V2_03_difficult",
        "tumvi:room5", "tumvi:room6"]
SEEDS = ["s42_accel_first", "s1_accel_first", "s7_accel_first", "s2026_accel_first"]
KEYS = [("ate_rmse_m", "ATE"), ("checkpoints_m", "checkpoints"), ("blind_mean_m", "blind 5 s"),
        ("dropout_m", "dropouts"), ("position_drift_1s_m", "drift 1 s")]
GROUPS = {"Machine Hall": lambda s: ":MH" in s, "Vicon": lambda s: ":V" in s, "TUM-VI": lambda s: "tumvi" in s}


def write_still(inp, est):
    tin = [int(ln.split(",", 1)[0]) for ln in Path(inp).read_text().splitlines() if ln and ln[0].isdigit()]
    with open(est, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t in tin:
            f.write(f"{t * 1e-9:.9f} 0 0 0 0 0 0 1\n")


def one(job):
    spec, s, label, opts = job
    d = INPUTS / s / spec.replace(":", "_")
    truth = json.loads((d / "truth.json").read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha1((ROOT / "src" / "main.py").read_bytes()).hexdigest()   # a cached run is tied to this main.py
    h = hashlib.sha1(json.dumps([spec, s, opts, code]).encode()).hexdigest()[:12]
    est = WORK / f"{h}.txt"
    if opts == "still":
        write_still(d / "imu0" / "data.csv", est)
    elif not est.exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "main.py"), str(d), "-o", str(est), *opts],
                       check=True, capture_output=True)
    r = score(str(est), str(source(spec)[1]), truth, str(d / "imu0" / "data.csv"))
    return {"data": spec, "set": s, "setting": label, "options": opts, "rows_ok": r["rows_match_input"],
            **{k: r.get(k) for k, _ in KEYS}}


def run(name, jobs, phase):
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    for r in rows:
        r["phase"] = phase
    path = ROOT / "results" / "track3" / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else []
    keep = [r for r in old if not (r["phase"] == phase and r["setting"] in {j[2] for j in jobs})]
    path.write_text(json.dumps(keep + rows, indent=0))


def tables(name):
    rows = json.loads((ROOT / "results" / "track3" / f"{name}.json").read_text())
    out = []
    for phase, title in (("sweep", "Development sweep (11 development recordings; mean metres)"),
                         ("check", "Check of the frozen variant (mean metres)")):
        rs = [r for r in rows if r["phase"] == phase]
        if not rs:
            continue
        labels = list(dict.fromkeys(r["setting"] for r in rs))
        sets = [(f"dev, seed {x.split('_')[0][1:]}", (lambda x: lambda r: r["set"] == x and r["data"] in DEV)(x))
                for x in dict.fromkeys(r["set"] for r in rs)] if phase == "sweep" else \
            [("held-out 6, seed 42", lambda r: r["set"] == "s42_accel_first" and r["data"] in HELD)] + \
            [(f"all 17, seed {s.split('_')[0][1:]}", (lambda s: lambda r: r["set"] == s)(s)) for s in SEEDS]
        out += [f"\n### {title}\n", "| inputs | setting | " + " | ".join(n for _, n in KEYS) +
                " | " + " | ".join(f"{g} ATE / blind" for g in GROUPS) + " | rows ok |",
                "|---|---|" + "---:|" * (len(KEYS) + len(GROUPS) + 1)]
        for lab, sel in sets:
            for name_ in labels:
                q = [r for r in rs if sel(r) and r["setting"] == name_]
                if not q:
                    continue
                grp = [f"{np.mean([r['ate_rmse_m'] for r in q if f(r['data'])]):.2f} / "
                       f"{np.mean([r['blind_mean_m'] for r in q if f(r['data'])]):.2f}" for f in GROUPS.values()]
                out.append(f"| {lab} | {name_} | " + " | ".join(f"{np.mean([r[k] for r in q]):.3f}" for k, _ in KEYS)
                           + " | " + " | ".join(grp) + f" | {sum(r['rows_ok'] for r in q)}/{len(q)} |")
        if phase == "check":
            idx = {(r["set"], r["data"], r["setting"]): r for r in rs}
            frozen = [x for x in labels if x not in ("entry", "standing still")]
            lose = [(s, d, k, idx[(s, d, f)][k], idx[(s, d, "standing still")][k]) for f in frozen
                    for (s, d, x) in idx if x == f for k, _ in KEYS if k != "blind_mean_m"
                    and idx[(s, d, f)][k] >= idx[(s, d, "standing still")][k]]
            out.append("\nNot better than standing still (ATE, checkpoints, dropouts, drift): " +
                       ("none" if not lose else "; ".join(f"{s} {d} {k} {a:.2f} vs {b:.2f}" for s, d, k, a, b in lose)))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("cmd", choices=["sweep", "check", "tables"])
    ap.add_argument("--variants", help="JSON {label: [main.py options]}")
    ap.add_argument("--variant", help="JSON [main.py options] of the frozen variant")
    ap.add_argument("--label", default="frozen")
    ap.add_argument("--sweep-sets", nargs="+", default=["s42_accel_first"],
                    help="input sets for `sweep` (development recordings only); e.g. s42_accel_first s1_accel_first")
    a = ap.parse_args()
    if a.cmd == "sweep":
        v = {"entry": [], "standing still": "still", **json.loads(a.variants)}
        run(a.name, [(sp, st, lab, o) for st in a.sweep_sets for lab, o in v.items() for sp in DEV], "sweep")
    elif a.cmd == "check":
        v = {"entry": [], "standing still": "still", a.label: json.loads(a.variant)}
        run(a.name, [(sp, s, lab, o) for lab, o in v.items() for s in SEEDS for sp in DEV + HELD], "check")
    tables(a.name)


if __name__ == "__main__":
    main()
