"""Why the blind 5 s window trails SRMIST_vortexxx on the Machine Hall: per recording and seed, split the blind error
into the error at the moment of the failure and the error added during the 5 s, and plot the last 15 s.

    python tools/analyse_mh_blind.py --tag mhblind        # reads runs/scoreboard/results_<tag>.json and its outputs

Alignment exactly as tools/score_track3.py scores blind: SE(3) Umeyama fitted on the mission part (before the IMU
dies), applied to the whole trajectory. Writes results/track3/mh_blind_diagnosis.json and the plots
results/track3/mh_blind_<recording>_<seed>.png for the worst cases.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402
from score_track3 import umeyama  # noqa: E402

INPUTS = ROOT / "runs" / "scoreboard" / "inputs"
OUT = ROOT / "runs" / "scoreboard" / "out"
MH = [f"euroc:{n}" for n in ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult")]
SEEDS = ["s42_accel_first", "s1_accel_first", "s7_accel_first", "s2026_accel_first"]


def aligned(spec, s, entry):
    d = INPUTS / s / spec.replace(":", "_")
    truth = json.loads((d / "truth.json").read_text())
    g = np.loadtxt(source(spec)[1], delimiter=",", comments="#")
    tg, pg = g[:, 0] * 1e-9, g[:, 1:4]
    e = np.loadtxt(OUT / s / entry / f"{spec.replace(':', '_')}.txt", comments="#")
    te, pe = e[:, 0], e[:, 1:4]
    ok = (te >= tg[0]) & (te <= tg[-1])
    te, pe = te[ok], pe[ok]
    pgi = np.stack([np.interp(te, tg, pg[:, i]) for i in range(3)], 1)
    rel = te - truth["t0_ns"] * 1e-9
    m = rel < truth["t_fail_s"]
    R, t = umeyama(pe[m], pgi[m])
    return rel, pe @ R.T + t, pgi, m, truth["t_fail_s"], (pe[0] @ R.T + t)


def analyse(spec, s, entry):
    rel, pa, pgi, m, tf, start = aligned(spec, s, entry)
    k0 = np.flatnonzero(m)[-1]
    b = ~m
    err = np.linalg.norm(pa - pgi, axis=1)
    motion = np.linalg.norm((pa - pa[k0]) - (pgi - pgi[k0]), axis=1)
    return {"blind_mean_m": float(err[b].mean()), "err_at_failure_m": float(err[k0]),
            "motion_err_mean_m": float(motion[b].mean()),
            "est_moved_in_blind_m": float(np.linalg.norm(pa[b][-1] - pa[k0])),
            "gt_moved_in_blind_m": float(np.linalg.norm(pgi[b][-1] - pgi[k0])),
            "est_failure_to_own_start_m": float(np.linalg.norm(pa[k0] - start)),
            "gt_failure_to_start_m": float(np.linalg.norm(pgi[k0] - pgi[0])),
            "gt_speed_last_2s_mps": float(np.linalg.norm(pgi[k0] - pgi[np.searchsorted(rel, rel[k0] - 2.0)]) / 2.0)}


def plot(spec, s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for entry, col, lab in (("ours", "#1f6f5c", "Path Finder"), ("vortexxx", "#b0562b", "SRMIST_vortexxx")):
        rel, pa, pgi, m, tf, _ = aligned(spec, s, entry)
        w = rel >= tf - 15
        if entry == "ours":
            axes[0].plot(pgi[w, 0], pgi[w, 1], color="#555", lw=2, label="ground truth")
            axes[0].plot(*pgi[m & w][-1, :2], "o", color="#555")
            axes[1].plot(rel[w] - tf, np.linalg.norm(pgi[w] - pgi[m][-1], axis=1), color="#555", lw=2,
                         label="ground truth: distance from its position at failure")
        axes[0].plot(pa[w, 0], pa[w, 1], color=col, lw=1.3, label=lab)
        axes[0].plot(*pa[m & w][-1, :2], "o", color=col)
        axes[1].plot(rel[w] - tf, np.linalg.norm(pa[w] - pgi[w], axis=1), color=col, lw=1.3, label=f"{lab}: error")
    axes[0].set_title(f"{spec.split(':')[1]}, seed {s[1:].split('_')[0]}: last 15 s (x-y, aligned; dot = failure)")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)"); axes[0].axis("equal"); axes[0].legend(fontsize=8)
    axes[1].axvline(0, color="#999", ls="--", lw=1)
    axes[1].set_xlabel("time from IMU failure (s)"); axes[1].set_ylabel("m"); axes[1].legend(fontsize=8)
    axes[1].set_title("error to ground truth (blind window right of the dashed line)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mhblind")
    a = ap.parse_args()
    rows = []
    for s in SEEDS:
        for spec in MH:
            for entry in ("ours", "vortexxx"):
                rows.append({"data": spec, "set": s, "entry": entry, **analyse(spec, s, entry)})
    (ROOT / "results" / "track3" / "mh_blind_diagnosis.json").write_text(json.dumps(rows, indent=1))
    by = {(r["data"], r["set"], r["entry"]): r for r in rows}
    print("| recording | seed | blind ours / vortexxx | error at failure ours / vortexxx | motion error ours / vortexxx"
          " | estimate moved in blind ours / vortexxx / truth | estimate at failure to own start: ours / vortexxx;"
          " truth to start |")
    print("|---|---|---:|---:|---:|---:|---:|")
    gaps = []
    for s in SEEDS:
        for spec in MH:
            o, v = by[(spec, s, "ours")], by[(spec, s, "vortexxx")]
            gaps.append((o["blind_mean_m"] - v["blind_mean_m"], spec, s))
            print(f"| {spec.split(':')[1]} | {s[1:].split('_')[0]} | {o['blind_mean_m']:.2f} / {v['blind_mean_m']:.2f} | "
                  f"{o['err_at_failure_m']:.2f} / {v['err_at_failure_m']:.2f} | {o['motion_err_mean_m']:.2f} / "
                  f"{v['motion_err_mean_m']:.2f} | {o['est_moved_in_blind_m']:.2f} / {v['est_moved_in_blind_m']:.2f} / "
                  f"{o['gt_moved_in_blind_m']:.2f} | {o['est_failure_to_own_start_m']:.2f} / "
                  f"{v['est_failure_to_own_start_m']:.2f}; {o['gt_failure_to_start_m']:.2f} |")
    for g, spec, s in sorted(gaps, reverse=True)[:3]:
        p = ROOT / "results" / "track3" / f"mh_blind_{spec.split(':')[1]}_s{s[1:].split('_')[0]}.png"
        plot(spec, s, p)
        print(f"worst gap {g:+.2f} m: {spec} {s} -> {p.name}")


if __name__ == "__main__":
    main()
