"""One command to reproduce every number in the README.

    python evaluate.py --seqs 36       # the full TLIO test split, ~10 min on 9 cores
    python evaluate.py --seqs 36       # the full TLIO test split
    python evaluate.py --synthetic     # no dataset needed: synthetic head walks

Writes results/summary.json, results/report.html and results/*.png.
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathfinder import data_tlio as D
from pathfinder.faults import Fault, scenarios
from pathfinder.imu_vote import vote
from pathfinder.faults import apply_imu_faults
from pathfinder.pipeline import ALL, fuse_imus, run_methods
from pathfinder.synthetic import head_walk

OUT = Path("results")
LABEL = {"imu_only": "IMU only", "naive_fusion": "Naive camera+IMU fusion",
         "tlio_style": "Learned IMU only (TLIO-style)", "ours_no_model": "Ours without learned model",
         "ours_tlio_model": "Ours with TLIO-trained model",
         "ours": "Ours"}
COLOR = {"imu_only": "#9aa0a6", "naive_fusion": "#e8710a", "tlio_style": "#1a73e8",
         "ours_no_model": "#a142f4", "ours_tlio_model": "#80868b", "ours": "#188038"}


def load_models():
    from pathfinder.predictor import Predictor
    ours = tl = None
    if Path("models/dispnet_faults/best.pt").exists():
        ours = Predictor("models/dispnet_faults/best.pt")
    if Path("models/dispnet_tlio/best.pt").exists():
        tl = Predictor("models/dispnet_tlio/best.pt")
    return ours, tl


def sequences(n, clip, synthetic, k_imus=3, split="test"):
    if synthetic:
        return [head_walk(clip, n_imus=k_imus, seed=i) for i in range(n)]
    out = []
    for sid in D.split(split)[:n]:
        s = D.load(sid)
        start = int(min(30, max(0, s.t[-1] - clip - 1)) * 200)     # skip the first 30 s settle
        out.append(D.redundant_imus(s.slice(start, start + int(clip * 200) + 1), k_imus, seed=len(out)))
    return out




_W = {}


def _parse_set(items):
    from pathfinder.filter import FilterConfig
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        t = type(getattr(FilterConfig(), k))
        out[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return out


def _init(seqs, methods, overrides=None):
    import torch
    from pathfinder import pipeline
    torch.set_num_threads(1)
    pipeline.OURS_OVERRIDES.update(overrides or {})
    _W["seqs"], _W["methods"] = seqs, methods
    _W["ours"], _W["tl"] = load_models()


def _job(job):
    kind, key, i = job
    s = _W["seqs"][i]
    faults = scenarios(10.0)[key] if kind == "scenario" else [Fault("camera_blackout", 20.0, key)]
    r = run_methods(s, faults, seed=i, ours_model=_W["ours"], tlio_model=_W["tl"], methods=_W["methods"])
    out = {}
    for m in _W["methods"]:
        out[m] = {k: r[m][k] for k in ("ate_m", "rte_1s_m", "rte_10s_m", "drift_pct", "final_err_m")}
        if kind == "sweep":
            b = min(int((20 + key) * 200), len(s.p) - 1)
            out[m]["end_err_m"] = float(np.linalg.norm(r[m]["p"][b] - s.p[b]))
    return out


def fig_to_b64(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=10)
    ap.add_argument("--clip", type=float, default=60.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="choose settings on val; test is for final reporting")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a field of the shipped filter config (pipeline.ours_cfg), e.g. low_conf_inflate=6")
    ap.add_argument("--methods", default=None, help="comma-separated subset of methods")
    ap.add_argument("--out", default="results", help="output folder (use a run folder for checks, so "
                    "committed results are only replaced on purpose)")
    args = ap.parse_args()
    global OUT
    from pathfinder import pipeline as _pl
    _pl.OURS_OVERRIDES.update(_parse_set(args.set))
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    ours, tl = load_models()
    seqs = sequences(args.seqs, args.clip, args.synthetic, split=args.split)
    methods = [m for m in ALL if (m != "ours" or ours) and (m not in ("tlio_style", "ours_tlio_model") or tl)]
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(",")]
    print(f"{len(seqs)} sequences x {args.clip:.0f}s, methods: {methods}", flush=True)

    # 1) every named scenario and 2) the blackout-length sweep, all sequences in parallel
    scen = scenarios(10.0)
    lengths = [1, 2, 5, 10, 20]
    jobs = [("scenario", name, i) for name in scen for i in range(len(seqs))]
    jobs += [("sweep", L, i) for L in lengths for i in range(len(seqs))]
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"{len(jobs)} runs on {workers} processes", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(seqs, methods, _parse_set(args.set))) as ex:
        out = list(ex.map(_job, jobs, chunksize=1))
    res = dict(zip(jobs, out))
    table = {}
    for name in scen:
        rows = {m: [res[("scenario", name, i)][m] for i in range(len(seqs))] for m in methods}
        table[name] = {m: {**{k: float(np.mean([x[k] for x in v])) for k in v[0]},
                           "ate_median_m": float(np.median([x["ate_m"] for x in v])),
                           "ate_per_seq_m": [round(x["ate_m"], 3) for x in v]} for m, v in rows.items()}
        print(f"{name:28s} " + "  ".join(f"{m}={table[name][m]['ate_m']:.2f}" for m in methods), flush=True)
    sweep = {m: [float(np.mean([res[("sweep", L, i)][m]["end_err_m"] for i in range(len(seqs))]))
                 for L in lengths] for m in methods}
    print("blackout sweep (error at end of blackout, m):", json.dumps(sweep), flush=True)
    example = (seqs[0], run_methods(seqs[0], scen["camera_blackout"], seed=0, ours_model=ours,
                                    tlio_model=tl, methods=methods))

    # 3) faulty-IMU isolation vs number of IMUs
    iso = {}
    kinds = {"bias": Fault("imu_bias", 5, 1e9, 0, {"gyro": [0.08, -0.05, 0.04], "acc": [0.8, -0.5, 0.4]}),
             "small bias": Fault("imu_bias", 5, 1e9, 0, {"gyro": [0.02, -0.015, 0.01], "acc": [0.3, -0.2, 0.15]}),
             "slow drift": Fault("imu_bias_drift", 5, 1e9, 0),
             "stuck": Fault("imu_stuck", 5, 1e9, 0), "spikes": Fault("imu_spikes", 5, 1e9, 0),
             "noise": Fault("imu_noise", 5, 1e9, 0)}
    base = [D.load(sid, max_seconds=40) for sid in D.split("test")[:5]] if not args.synthetic else \
        [head_walk(40, n_imus=1, seed=i) for i in range(5)]
    for kname, f in kinds.items():
        iso[kname] = {}
        for k in (2, 3, 5):
            vals = []
            for b in base:
                s = D.redundant_imus(b, k, seed=3) if not args.synthetic else head_walk(40, n_imus=k, seed=1)
                bad = apply_imu_faults(s, [f])
                _, _, info = fuse_imus(bad)
                # Only score the 0.1 s windows where the fault actually changed the readings.
                diff = np.abs(bad.gyro[0] - s.gyro[0]).max(1) + np.abs(bad.acc[0] - s.acc[0]).max(1)
                n_w = info["weights"].shape[1]
                hit = np.array([diff[j * 20:(j + 1) * 20].max() > 1e-6 for j in range(n_w)])
                w = info["weights"][:, hit]
                others = [j for j in range(k) if j != 0]
                vals.append(float(((w[0] < 0.5) & np.all(w[others] >= 0.5, axis=0)).mean()))
            iso[kname][k] = float(np.mean(vals))
    print("isolation:", json.dumps(iso), flush=True)

    # 4) model size and speed
    perf = {}
    if ours:
        import torch
        from pathfinder.model import n_params
        x = np.random.randn(200, 6)
        for _ in range(20):
            ours(x)
        t = time.perf_counter()
        for _ in range(200):
            ours(x)
        perf = {"params": n_params(ours.net), "size_mb": n_params(ours.net) * 4 / 1e6,
                "ms_per_prediction_cpu": (time.perf_counter() - t) / 200 * 1000}

    summary = {"sequences": [s.name for s in seqs], "clip_s": args.clip, "scenarios": table,
               "blackout_sweep": {"lengths_s": lengths, **sweep}, "imu_isolation": iso,
               "model": perf, "runtime_s": time.time() - t0, "synthetic": args.synthetic}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from pathfinder.manifest import write as write_manifest
    write_manifest(OUT / "manifest.json", args=args, models=["models/dispnet_faults/best.pt",
                   "models/dispnet_tlio/best.pt", "models/camera_trust.json"],
                   extra={"filter_overrides": _parse_set(args.set), "split": args.split})
    write_report(summary, example, methods)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}/report.html")


def write_report(S, example, methods):
    imgs = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        ax.plot(S["blackout_sweep"]["lengths_s"], S["blackout_sweep"][m], "o-", label=LABEL[m], color=COLOR[m])
    ax.set_xlabel("camera blackout length (s)")
    ax.set_ylabel("position error when camera returns (m)")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    imgs["sweep"] = fig_to_b64(fig, "blackout_sweep")

    if example:
        s, r = example
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(s.p[:, 0], s.p[:, 1], "k-", lw=2, label="True path")
        for m in methods:
            if m == "imu_only":
                continue
            ax.plot(r[m]["p"][:, 0], r[m]["p"][:, 1], "-", color=COLOR[m], lw=1.2, label=LABEL[m])
        a, b = 20 * 200, 30 * 200
        ax.plot(s.p[a:b, 0], s.p[a:b, 1], "-", color="#d93025", lw=5, alpha=.25, label="Camera off (10 s)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title("Top view, one test sequence")
        imgs["traj"] = fig_to_b64(fig, "trajectory_example")

    names = list(S["scenarios"].keys())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    show = [m for m in methods if m != "imu_only"]
    wdt = 0.8 / len(show)
    for i, m in enumerate(show):
        ax.bar(x + i * wdt, [S["scenarios"][n][m]["ate_m"] for n in names], wdt, label=LABEL[m], color=COLOR[m])
    ax.set_xticks(x + 0.4 - wdt / 2)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ATE (m), lower is better")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    imgs["bars"] = fig_to_b64(fig, "scenarios_ate")

    rows = "".join(
        f"<tr><td>{n.replace('_', ' ')}</td>" + "".join(
            f"<td class='{'best' if S['scenarios'][n][m]['ate_m'] == min(S['scenarios'][n][q]['ate_m'] for q in methods) else ''}'>"
            f"{S['scenarios'][n][m]['ate_m']:.2f}</td>" for m in methods) + "</tr>" for n in names)
    iso_rows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{100 * v[c]:.0f}%</td>" for c in v) + "</tr>"
                       for k, v in S["imu_isolation"].items())
    perf = S.get("model") or {}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Path Finder results</title>
<style>body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#202124}}
table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #dadce0;padding:6px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}td.best{{background:#e6f4ea;font-weight:600}}img{{max-width:100%}}</style></head><body>
<h1>Path Finder: trajectory estimation under sensor degradation</h1>
<p>{len(S['sequences'])} held-out sequences x {S['clip_s']:.0f} s{' (synthetic)' if S['synthetic'] else ' from the TLIO headset test split'}.
Faults start at 20 s. ATE in metres, after yaw + translation alignment. Generated by <code>python evaluate.py</code> in {S['runtime_s']:.0f} s.</p>
<h2>Error when the camera comes back</h2><img src="data:image/png;base64,{imgs['sweep']}">
<h2>Every scenario (ATE, m)</h2><table><tr><th>Scenario</th>{''.join(f'<th>{LABEL[m]}</th>' for m in methods)}</tr>{rows}</table>
<img src="data:image/png;base64,{imgs['bars']}">
{f'<h2>Example</h2><img src="data:image/png;base64,{imgs["traj"]}">' if 'traj' in imgs else ''}
<h2>Finding the faulty IMU</h2><p>Share of 0.1 s windows where the faulty IMU is down-weighted and every healthy one is kept.</p>
<table><tr><th>Fault</th><th>2 IMUs</th><th>3 IMUs</th><th>5 IMUs</th></tr>{iso_rows}</table>
<h2>On-device cost</h2><p>{perf.get('params', '-')} parameters, {perf.get('size_mb', 0):.2f} MB (float32),
{perf.get('ms_per_prediction_cpu', 0):.2f} ms per prediction on one laptop CPU core, 10 predictions per second.</p>
</body></html>"""
    (OUT / "report.html").write_text(html)


if __name__ == "__main__":
    main()
