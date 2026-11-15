"""Render the real-vision report and figures from ONE evaluation run bundle (issue #2 D, #3 C/F).

    python tools/vision_report.py results/runs/<run> [--profile results/runs/<profile-run>]
                                  [--out results/vision/<name>]

The output directory is a promoted, committed bundle: ``bundle.json`` (schema
``pathfinder.result-bundle/1``: source run id, manifest copy, SHA-256 of every source and output file),
``REPORT.md``, ``tables.json`` and figures. Every number in REPORT.md is computed here from
``rows.jsonl``; nothing is typed by hand. Refuses to overwrite an existing output directory.

Statistics: one recording. Paired deltas are over *scenarios* (identical corrupted inputs for both
systems) and the bootstrap resamples scenarios, so the intervals describe fault-schedule variability
on room1 only. They are not recording-level confidence intervals and support no general claim.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ("vo_only", "imu_only", "conventional", "pathfinder", "pathfinder_model")
PAIRS = (("pathfinder", "conventional"), ("pathfinder", "vo_only"), ("pathfinder_model", "pathfinder"),
         ("pathfinder_no_gate", "pathfinder"), ("pathfinder_uncal_cov", "pathfinder"),
         ("pathfinder_no_adaptive_q", "pathfinder"), ("pathfinder_no_health", "pathfinder"),
         ("pathfinder_model_no_selfcheck", "pathfinder_model"), ("pathfinder_model_biascorr", "pathfinder_model"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{n}f}"


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    m = np.array([np.median(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--sync", default=None)
    ap.add_argument("--model-shift", default=None, help="run dir of tools/model_shift.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    status = json.loads((run / "status.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    rows = [json.loads(ln) for ln in (run / "rows.jsonl").read_text().splitlines() if ln.strip()]
    out = Path(a.out or ROOT / "results" / "vision" / run.name)
    if out.exists():
        sys.exit(f"{out} exists; promoted bundles are immutable")
    out.mkdir(parents=True)
    by = {(r["scenario"], r["system"]): r for r in rows}
    scen = [s["name"] for s in protocol["scenarios"]]
    systems = list(protocol["systems"])
    failures = [r for r in rows if r["status"] != "ok"]

    def ate(sc, sy, key=("start_se3", "ate_all", "rmse")):
        r = by.get((sc, sy))
        return g(r["metrics"], *key) if r else None

    # ------------------------------------------------------------------ tables
    T = {"per_run": [], "pairs": [], "faults": [], "failures": [], "runtime": {}}
    for r in rows:
        m = r["metrics"]
        T["per_run"].append({
            "scenario": r["scenario"], "system": r["system"], "status": r["status"],
            "ate_rmse_m": g(m, "start_se3", "ate_all", "rmse"), "ate_max_m": g(m, "start_se3", "ate_all", "max"),
            "ate4_rmse_m": g(m, "full_4dof", "ate_all", "rmse"), "rot_rmse_deg": g(m, "start_se3", "rot_deg_all", "rmse"),
            "rpe1_trans_rmse_m": g(m, "rpe_1s", "trans_m", "rmse"), "rpe1_rot_rmse_deg": g(m, "rpe_1s", "rot_deg", "rmse"),
            "rpe10_trans_rmse_m": g(m, "rpe_10s", "trans_m", "rmse"), "final_err_m": g(m, "start_se3", "final_err"),
            "drift_pct": g(m, "final_drift_pct_of_path"), "valid_fraction": g(m, "valid_fraction"),
            "estimated_fraction": g(m, "estimated_fraction_after_start"), "first_estimate_s": g(m, "first_estimate_t"),
            "nees99_all": g(m, "nees", "all", "0.99"), "nees997_all": g(m, "nees", "all", "0.997"),
            "heldout_ate_rmse_m": g(m, "segments", "heldout_60_end", "ate_all", "rmse"),
            "calib_ate_rmse_m": g(m, "segments", "calibration_0_60", "ate_all", "rmse"),
            "vo_used": g(r, "diagnostics", "vo_used"), "vo_rejected": g(r, "diagnostics", "vo_rejected"),
            "vo_no_anchor": g(r, "diagnostics", "vo_no_anchor"), "disp_used": g(r, "diagnostics", "disp_used"),
            "estimator_us_per_sample": g(r, "diagnostics", "estimator_us_per_imu_sample")})
        for f in g(m, "faults") or []:
            T["faults"].append({"scenario": r["scenario"], "system": r["system"], "kind": f["kind"], "start": f["start"],
                                "end": f["end"], "err_growth_m": f.get("err_growth"), "err_at_end_m": f.get("err_at_end"),
                                "recovery_s": f.get("recovery_s"), "step_dev_m": f.get("max_step_dev_after_m"),
                                "valid_inside": f.get("valid_fraction_inside"),
                                "nees99_inside_all": g(f, "nees_cov_inside_all", "0.99")})
    for r in failures:
        T["failures"].append({"scenario": r["scenario"], "system": r["system"], "status": r["status"]})
    for x, y in PAIRS:
        if x not in systems or y not in systems:
            continue
        d, worst = [], None
        for sc in scen:
            ax, ay = ate(sc, x), ate(sc, y)
            if ax is None or ay is None:
                continue
            d.append(ax - ay)
            if worst is None or ax - ay > worst[1]:
                worst = (sc, ax - ay)
        d = np.array(d)
        lo, hi = boot_ci(d)
        T["pairs"].append({"a": x, "b": y, "n_scenarios": int(len(d)), "median_delta_m": float(np.median(d)) if len(d) else None,
                           "mean_delta_m": float(d.mean()) if len(d) else None, "a_better": int((d < 0).sum()),
                           "b_better": int((d > 0).sum()), "boot95_median": [lo, hi], "worst_for_a": worst,
                           "delta_definition": "ATE RMSE(a) - ATE RMSE(b), same corrupted inputs; negative = a better"})
    fe_ms = [g(r, "frontend", "ms_per_frame") for r in rows if r["system"] == "pathfinder" and r["scenario"] == "clean"]
    T["runtime"]["frontend_clean_ms_per_frame"] = fe_ms[0] if fe_ms else None
    T["runtime"]["estimator_us_per_imu_sample"] = {sy: g(by.get(("clean", sy)) or {}, "diagnostics", "estimator_us_per_imu_sample")
                                                  for sy in systems}
    if a.profile:
        T["runtime"]["profile"] = json.loads((Path(a.profile) / "profile.json").read_text())
    if a.model_shift:
        T["model_shift"] = json.loads((Path(a.model_shift) / "model_shift.json").read_text())
        T["model_shift"]["run"] = Path(a.model_shift).name
    (out / "tables.json").write_text(json.dumps(T, indent=1))

    # ------------------------------------------------------------------ figures
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"vo_only": "#8c8c8c", "imu_only": "#c49a00", "conventional": "#1f77b4", "pathfinder": "#d62728",
                  "pathfinder_model": "#2ca02c"}
        for sc in ("clean", "blackout_5s_at40", "blackout_20s_at60", "frozen_3s_at60"):
            if (sc, "pathfinder") not in by:
                continue
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for sy in MAIN:
                r = by.get((sc, sy))
                if not r or not r.get("series"):
                    continue
                t, e = np.array(r["series"]["t"]), np.array(r["series"]["err"], float)
                if sy != "imu_only":           # diverges to hundreds of metres; reported in the tables
                    ax[0].plot(t, e, label=sy, color=colors.get(sy), lw=1.2)
                if "nees" in r["series"] and sy not in ("vo_only", "imu_only"):
                    ax[1].semilogy(t, np.array(r["series"]["nees"], float), color=colors.get(sy), lw=0.8)
            for f in by[(sc, "pathfinder")]["scenario_def"].get("image_faults", []):
                for axx in ax:
                    axx.axvspan(f["start"], f["end"], color="k", alpha=0.08)
            ax[0].set_ylabel("position error (m)\nstart-SE(3) alignment")
            ax[0].set_ylim(0, None)
            ax[0].legend(fontsize=8, ncol=3)
            ax[1].axhline(11.34, color="k", ls=":", lw=0.8)
            ax[1].set_ylabel("position NEES (3 dof)\ndotted: chi2 99%")
            ax[1].set_xlabel("time since first IMU sample (s)")
            ax[0].set_title(f"TUM-VI room1 (development), scenario {sc}: real images + real IMU, offline replay")
            fig.tight_layout()
            name = f"error_{sc}.png"
            fig.savefig(out / name, dpi=110)
            plt.close(fig)
            figs.append(name)
        if ("clean", "pathfinder") in by:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            r0 = by[("clean", "pathfinder")]["series"]
            pr = np.array(r0["p_ref"], float)
            ax.plot(pr[:, 0], pr[:, 1], color="k", lw=1.5, label="mocap reference")
            for sy in MAIN:
                r = by.get(("clean", sy))
                if r and r.get("series"):
                    pa = np.array(r["series"]["p_aligned"], float)
                    if sy != "imu_only":
                        ax.plot(pa[:, 0], pa[:, 1], color=colors.get(sy), lw=1, label=sy)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(fontsize=8)
            ax.set_title("room1 clean, start-SE(3) aligned (imu_only omitted: diverges)")
            fig.tight_layout()
            fig.savefig(out / "trajectory_clean.png", dpi=110)
            plt.close(fig)
            figs.append("trajectory_clean.png")
        fig, ax = plt.subplots(figsize=(10, 7))
        ys = np.arange(len(scen))
        for k, sy in enumerate(("vo_only", "conventional", "pathfinder", "pathfinder_model")):
            v = [ate(sc, sy) for sc in scen]
            ax.scatter([x if x is not None else np.nan for x in v], ys + 0.15 * (k - 1.5), s=14, label=sy,
                       color=colors.get(sy))
        ax.set_yticks(ys)
        ax.set_yticklabels(scen, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("ATE RMSE (m), log scale, start-SE(3) alignment")
        ax.legend(fontsize=8)
        ax.set_title("Every scenario, room1 (one development recording)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out / "ate_by_scenario.png", dpi=110)
        plt.close(fig)
        figs.append("ate_by_scenario.png")
    except ImportError:
        figs = []

    # ------------------------------------------------------------------ report
    L = []
    w = L.append
    fz = protocol["frozen"]
    w(f"# Real-vision evaluation: {protocol['sequence']} ({protocol['protocol']})\n")
    w("**Generated by `tools/vision_report.py` from run `" + run.name + "` (status: " + status["state"] + ").** "
      "Real stereo images and the real single IMU of TUM-VI room1, replayed offline; faults are injected into "
      "images/IMU *before* the front end/estimator. No system receives the reference; no system is started "
      "from it. room1 is a **development** recording (see `data/exposure.json`): these are integration and "
      "development results on one recording, not an untouched final test and not evidence of general superiority.\n")
    w("## Frozen configuration\n")
    w(f"- VO covariance scale {fz['vo_cov_scale']} and keyframe interval {protocol['vo_config']['kf_every']} "
      f"(selected on room1 0–60 s, calibration run `{protocol.get('calibration_run')}`)")
    w(f"- IMU noise from the dataset's `imu_config.yaml`: {json.dumps(fz['imu_noise'])}")
    w(f"- camera–IMU time offset applied: {fz['camera_time_offset_s']} s "
      f"(measured {fmt(g(protocol, 'sync_measurement', 'offset_s'), 5)} s; protocol applies it only if > 1 ms)")
    w(f"- learned model: `{fz.get('model')}` sha256 `{fz.get('model_sha256')}` (trained on TLIO headset data: cross-device here)")
    w("- initialisation: reference-free gravity start for every filter (odometry frame = start pose, velocity unknown)")
    w("- scoring: start-SE(3) alignment at the first initialised estimate (primary; drift fully visible), "
      "4-DoF full-trajectory fit (secondary); never re-aligned per outage; mocap gaps > 50 ms excluded")
    w("- validity: VO covariance is not validated on held-out data, so every relative-mode output is "
      "`COVARIANCE_UNRELIABLE` and `valid=False` by contract; errors below are over **all estimated** samples\n")
    w("## Main results (ATE RMSE, m; all estimated samples)\n")
    head = "| scenario | " + " | ".join(MAIN) + " |"
    w(head)
    w("|" + "---|" * (len(MAIN) + 1))
    for sc in scen:
        w(f"| {sc} | " + " | ".join(fmt(ate(sc, sy)) for sy in MAIN) + " |")
    w("")
    w("## Paired scenario-level deltas (negative = first system better)\n")
    w("| a vs b | n | median Δ m | mean Δ m | a better | b better | bootstrap 95% of median | worst case for a |")
    w("|---|---|---|---|---|---|---|---|")
    for p in T["pairs"]:
        lo, hi = p["boot95_median"]
        wc = p["worst_for_a"]
        w(f"| {p['a']} vs {p['b']} | {p['n_scenarios']} | {fmt(p['median_delta_m'])} | {fmt(p['mean_delta_m'])} | "
          f"{p['a_better']} | {p['b_better']} | [{fmt(lo)}, {fmt(hi)}] | {wc[0] if wc else '–'} ({fmt(wc[1] if wc else None)}) |")
    w("\nIntervals resample scenarios on one recording (fault-schedule variability), not recordings.\n")
    w("## Clean run detail\n")
    w("| system | ATE | max err | 4-DoF ATE | rot RMSE deg | RPE 1 s m | RPE 1 s deg | final drift % path | "
      "held-out 60–141 s ATE | NEES≤χ²99% (all) | VO used/rejected |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for pr in T["per_run"]:
        if pr["scenario"] != "clean":
            continue
        w(f"| {pr['system']} | {fmt(pr['ate_rmse_m'])} | {fmt(pr['ate_max_m'])} | {fmt(pr['ate4_rmse_m'])} | "
          f"{fmt(pr['rot_rmse_deg'], 2)} | {fmt(pr['rpe1_trans_rmse_m'])} | {fmt(pr['rpe1_rot_rmse_deg'], 2)} | "
          f"{fmt(pr['drift_pct'], 2)} | {fmt(pr['heldout_ate_rmse_m'])} | {fmt(pr['nees99_all'], 2)} | "
          f"{pr['vo_used'] if pr['vo_used'] is not None else '–'}/{pr['vo_rejected'] if pr['vo_rejected'] is not None else '–'} |")
    w("\n## Outage / fault behaviour (Path Finder, conventional, VO only)\n")
    w("Recovery: first time after the fault ends from which the error stays ≤ (error at fault start + 0.10 m) "
      "for 2 s; '–' = never within the recording. Step deviation: largest per-sample position step beyond the "
      "reference's own step within 2 s after the fault (re-anchoring discontinuity).\n")
    w("| scenario | fault | system | error growth m | error at end m | recovery s | step dev m | NEES≤χ²99% inside |")
    w("|---|---|---|---|---|---|---|---|")
    for f in T["faults"]:
        if f["system"] in ("pathfinder", "conventional", "vo_only", "pathfinder_model"):
            w(f"| {f['scenario']} | {f['kind']} {f['start']:.1f}–{f['end']:.1f} s | {f['system']} | {fmt(f['err_growth_m'])} | "
              f"{fmt(f['err_at_end_m'])} | {fmt(f['recovery_s'], 1)} | {fmt(f['step_dev_m'], 4)} | {fmt(f['nees99_inside_all'], 2)} |")
    w("\n## Failures and harness errors\n")
    if T["failures"]:
        for f in T["failures"]:
            w(f"- {f['scenario']} / {f['system']}: {f['status']}")
    else:
        w("- none: every scenario × system produced scored output")
    w("\n## Runtime\n")
    rt = T["runtime"]
    if rt.get("frontend_clean_ms_per_frame"):
        x = rt["frontend_clean_ms_per_frame"]
        w(f"- front end (decode + corrupt + VO), batch evaluation, clean room1: p50 {fmt(x['p50'], 1)} ms, "
          f"p95 {fmt(x['p95'], 1)} ms, p99 {fmt(x['p99'], 1)} ms per frame")
    for sy, v in rt["estimator_us_per_imu_sample"].items():
        if v is not None:
            w(f"- estimator `{sy}`: {fmt(v, 0)} µs per IMU sample (mean, offline replay incl. Python overhead)")
    if rt.get("profile"):
        p = rt["profile"]
        w(f"- causal single-thread profile on **{p['hardware']['cpu']}** ({p['hardware']['platform']}, "
          f"Python {p['hardware']['python']}, {p['hardware']['threads']}): frame total p50/p95/p99 "
          f"{fmt(g(p, 'frame_total_ms', 'p50'), 1)}/{fmt(g(p, 'frame_total_ms', 'p95'), 1)}/{fmt(g(p, 'frame_total_ms', 'p99'), 1)} ms "
          f"({g(p, 'frame_total_ms', 'missed_deadline')} of {g(p, 'frame_total_ms', 'n')} frames over 50 ms); IMU step "
          f"p50/p99 {fmt(g(p, 'imu_estimator_ms_steady', 'p50'), 2)}/{fmt(g(p, 'imu_estimator_ms_steady', 'p99'), 2)} ms; "
          f"real-time factor {fmt(p['real_time_factor'], 2)}; peak RSS {fmt(p['rss_mb_peak_sampled'], 0)} MB. {p['label']}.")
    w("\n## Figures\n")
    for f in figs:
        w(f"![{f}]({f})")
    w("\n## What this does not show\n")
    w("- No established external VIO (OpenVINS/Basalt/ORB-SLAM3) was run on the same inputs; the comparison "
      "systems are the in-repo VO-only chain and a conventional loosely coupled filter.")
    w("- One development recording: no recording-level statistics, no cross-device result, no untouched final test.")
    w("- VO covariance validity is not established; the learned model is evaluated cross-device without calibration.")
    w("- Runtime is a laptop replay; no headset, NPU, thermal or multi-threaded scheduler measurement.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ provenance
    src = {n: sha(run / n) for n in ("manifest.json", "protocol.json", "rows.jsonl", "summary.json", "status.json")
           if (run / n).exists()}
    shutil.copy(run / "manifest.json", out / "source_manifest.json")
    shutil.copy(run / "protocol.json", out / "source_protocol.json")
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    bundle = {"schema": "pathfinder.result-bundle/1", "source_run": run.name, "source_status": status["state"],
              "source_sha256": src, "protocol": protocol["protocol"], "code": manifest.get("code"),
              "generator": "tools/vision_report.py", "generator_sha256": sha(Path(__file__)),
              "profile_run": Path(a.profile).name if a.profile else None, "files": files,
              "rows_jsonl_note": "rows.jsonl (full per-run metrics + 10 Hz series) stays in the source run directory; "
                                 "tables.json carries every reported number"}
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    print("bundle:", out)


if __name__ == "__main__":
    main()
