"""Regenerate the per-recording results of the shipped entry on the organisers' script.

    python tools/make_per_recording.py --script <organisers' degrade_imu.py>

Runs tools/eval_organiser.py on the 11 EuRoC + 6 TUM-VI recordings with `--entry python` and `--entry cpp`, then
writes results/track3/organiser_per_recording.json (path_finder rows from the Python run), ..._cpp.json and
per_recording.md. Standing still does not depend on our code, so its rows are kept from the existing file. Monado
rows are re-run only if local_data/monado exists; otherwise they are kept and marked with the commit they came from.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "track3"
DATA = [f"euroc:{n}" for n in ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
                               "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium",
                               "V2_03_difficult")] + [f"tumvi:room{i}" for i in range(1, 7)]
KEEP = ("data", "ate_rmse_m", "checkpoints_m", "blind_mean_m", "blind_end_m", "dropout_m", "position_drift_1s_m",
        "rows", "rows_match_input")
COLS = [("ate_rmse_m", "ATE"), ("checkpoints_m", "Checkpoints"), ("blind_mean_m", "Blind 5 s"),
        ("dropout_m", "Dropouts"), ("position_drift_1s_m", "Drift 1 s")]


def evaluate(script, entry, specs, tag):
    subprocess.run([sys.executable, str(ROOT / "tools" / "eval_organiser.py"), "--entry", entry, "--script", script,
                    "--tag", tag, "--data", *specs], check=True)
    rows = json.loads((ROOT / "runs" / "organiser" / f"{tag}.json").read_text())["rows"]
    return [{k: r.get(k) for k in KEEP} for r in rows]


def table(pf, st, monado_note):
    out = ["| Recording | " + " | ".join(n for _, n in COLS) + " |", "|---|" + "---:|" * len(COLS)]
    sti = {r["data"]: r for r in st}
    for r in pf:
        s = sti[r["data"]]
        cells = [(f"**{r[k]:.2f}** / {s[k]:.2f}" if r[k] < s[k] else f"{r[k]:.2f} / {s[k]:.2f}") for k, _ in COLS]
        out.append(f"| {r['data'].split(':')[1]} | " + " | ".join(cells) + " |")
    means = [(np.mean([r[k] for r in pf]), np.mean([sti[r['data']][k] for r in pf])) for k, _ in COLS]
    out.append("| **Mean** | " + " | ".join(f"**{a:.2f}** / {b:.2f}" if a < b else f"{a:.2f} / {b:.2f}"
                                            for a, b in means) + " |")
    out.append("| **Recordings better** | " + " | ".join(
        f"{sum(r[k] < sti[r['data']][k] for r in pf)} / {len(pf)}" for k, _ in COLS) + " |")
    return "\n".join(out) + "\n\n" + monado_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--commit", required=True, help="the commit these results describe")
    a = ap.parse_args()
    old = json.loads((RES / "organiser_per_recording.json").read_text())
    py = evaluate(a.script, "python", DATA, "final_python")
    cpp = evaluate(a.script, "cpp", DATA, "final_cpp")
    st = old["euroc_tumvi"]["standing_still"]
    have_monado = (ROOT / "local_data" / "monado").exists()
    monado = old["monado"]
    if have_monado:
        specs = [r["data"] for r in monado["path_finder"]]
        monado = {"path_finder": evaluate(a.script, "python", specs, "final_monado"),
                  "standing_still": monado["standing_still"]}
    else:
        monado = dict(monado, note=f"not re-run for {a.commit}: no Monado data on this machine; measured at 85b3a30 "
                                   "(rotor drag, before dropout bridging)")
    desc = old["description"]
    doc = {"description": desc, "produced_by": "tools/make_per_recording.py (tools/eval_organiser.py --entry python)",
           "commit": a.commit, "euroc_tumvi": {"path_finder": py, "standing_still": st}, "monado": monado}
    (RES / "organiser_per_recording.json").write_text(json.dumps(doc, indent=1))
    (RES / "organiser_per_recording_cpp.json").write_text(json.dumps(
        {"description": desc.replace("--entry python", "--entry cpp"), "commit": a.commit,
         "euroc_tumvi": {"path_finder": cpp, "standing_still": st}}, indent=1))
    diff = max(abs(p[k] - c[k]) for p, c in zip(py, cpp) for k, _ in COLS)
    mp = monado["path_finder"]
    ms = {r["data"]: r for r in monado["standing_still"]}
    mon = (f"Monado headset recordings (8, IMU mounted unlike the training data, so the mount guard selects the "
           f"gravity-aligned network; rotor drag stays off): ATE {np.mean([r['ate_rmse_m'] for r in mp]):.2f} vs "
           f"{np.mean([ms[r['data']]['ate_rmse_m'] for r in mp]):.2f} m, better on "
           f"{sum(r['ate_rmse_m'] < ms[r['data']]['ate_rmse_m'] for r in mp)} / {len(mp)}"
           + ("." if have_monado else " (measured at `85b3a30`, before dropout bridging; not re-run here)."))
    head = f"""# Per-recording results, organisers' degradation script

All 11 EuRoC and 6 TUM-VI recordings, damaged by the organisers' `degrade_imu.py` (their seed, accel-first input),
run with `src/main.py` at commit `{a.commit}` and scored with `tools/score_track3.py`. Each cell: Path Finder /
standing still, metres; **bold** = Path Finder lower. Raw numbers: [`organiser_per_recording.json`](organiser_per_recording.json);
the C++ port gives the same scores within {diff:.0e} m ([`organiser_per_recording_cpp.json`](organiser_per_recording_cpp.json)).
Regenerate with `python tools/make_per_recording.py --script <degrade_imu.py> --commit <hash>`.

"""
    (RES / "per_recording.md").write_text(head + table(py, st, mon) + "\n")
    print(f"python vs cpp max difference {diff:.2e} m")


if __name__ == "__main__":
    main()
