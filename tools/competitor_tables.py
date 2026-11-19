"""Markdown tables for results/track3/competitors.md from the raw scoreboard JSON (tools/competitor_scoreboard.py).

    python tools/competitor_tables.py results/track3/competitors_raw.json \
        --heap results/track3/competitors_heap_warm.json --heap-cold results/track3/competitors_heap_cold.json

Prints the tables; the page quotes them verbatim (nothing typed by hand).
"""
import argparse
import json
from collections import defaultdict

import numpy as np

METRICS = [("ate_rmse_m", "ATE"), ("checkpoints_m", "checkpoints"), ("blind_mean_m", "blind 5 s"),
           ("dropout_m", "dropouts"), ("position_drift_1s_m", "drift 1 s")]
GROUPS = {"Vicon rooms": lambda s: s.startswith("euroc:V"), "Machine Hall": lambda s: s.startswith("euroc:MH"),
          "TUM-VI": lambda s: s.startswith("tumvi:")}
ORDER = ["ours", "vortexxx", "zerodrift", "vioedge", "vortex", "trustnav", "trustnav_rows", "ridhima", "still"]
LABEL = {"ours": "**Path Finder (ours)**", "vortexxx": "SRMIST_vortexxx", "zerodrift": "ZeroDrift", "vioedge": "VIO-Edge",
         "vortex": "Vortex", "trustnav": "TrustNav (as shipped)", "trustnav_rows": "TrustNav `--no-fill --blind 0`",
         "ridhima": "Ridhima Chopra", "still": "standing still"}
SET_LABEL = {"s42_accel_first": "seed 42, accel-first (documented order)", "s42_euroc": "seed 42, EuRoC file as is (gyro first)",
             "s1_accel_first": "seed 1, accel-first", "s7_accel_first": "seed 7, accel-first"}


def short(spec):
    return spec.split(":", 1)[1].replace("_easy", "").replace("_medium", "").replace("_difficult", "")


def load(path):
    rows = json.load(open(path))
    by = defaultdict(dict)                         # (entry, set) -> spec -> row
    for r in rows:
        by[(r["entry"], r["set"])][r["data"]] = r
    return rows, by


def val(r, k):
    return r.get("score", {}).get(k, np.nan) if r and "error" not in r else np.nan


def entries_in(by, s):
    return [e for e in ORDER if (e, s) in by]


def group_table(by, s, metric):
    specs = sorted({d for (e, ss), m in by.items() if ss == s for d in m})
    lines = [f"| entry | " + " | ".join(GROUPS) + " | all 17 | beats standing still |", "|---|" + "---:|" * (len(GROUPS) + 2)]
    still = by.get(("still", s), {})
    for e in entries_in(by, s):
        m = by[(e, s)]
        cells = []
        for g, f in GROUPS.items():
            v = [val(m.get(d), metric) for d in specs if f(d)]
            cells.append(f"{np.nanmean(v):.2f}" if np.isfinite(v).any() else "–")
        allv = [val(m.get(d), metric) for d in specs]
        beat = sum(val(m.get(d), metric) < val(still.get(d), metric) for d in specs) if e != "still" else None
        fails = sum(1 for d in specs if not np.isfinite(val(m.get(d), metric)))
        cells.append(f"{np.nanmean(allv):.2f}" + (f" ({fails} missing)" if fails else ""))
        cells.append("–" if beat is None else f"{beat}/{len(specs)}")
        lines.append(f"| {LABEL[e]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_recording(by, s, metric):
    specs = sorted({d for (e, ss), m in by.items() if ss == s for d in m}, key=lambda d: (d.startswith("tumvi"), d))
    es = entries_in(by, s)
    lines = ["| recording | " + " | ".join(LABEL[e].replace("**", "") for e in es) + " |", "|---|" + "---:|" * len(es)]
    for d in specs:
        vals = [val(by[(e, s)].get(d), metric) for e in es]
        best = np.nanmin([v for e, v in zip(es, vals) if e != "still"])
        cells = [("**%.2f**" % v if v == best else "%.2f" % v) if np.isfinite(v) else "fail" for v in vals]
        lines.append(f"| {short(d)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contract_table(by, rows):
    lines = ["| entry | runs | one pose per row, same timestamps | crashes | rows out vs in (typical) |", "|---|---:|---:|---:|---|"]
    for e in ORDER:
        rs = [r for r in rows if r["entry"] == e]
        if not rs or e == "still":
            continue
        ok = sum(1 for r in rs if r.get("contract", {}).get("ok"))
        crash = sum(1 for r in rs if "error" in r)
        diffs = [r["contract"]["rows_out"] - r["contract"]["rows_in"] for r in rs if "contract" in r]
        typ = f"{int(np.median(diffs)):+d} (range {min(diffs):+d} to {max(diffs):+d})" if diffs else "–"
        lines.append(f"| {LABEL[e]} | {len(rs)} | {ok}/{len(rs)} | {crash} | {typ} |")
    return "\n".join(lines)


def resource_table(rows, heap, heap_cold=None):
    lines = ["| entry | CPU per s of IMU, median (max), ms | process peak RSS, median (max), MB | heap peak, warm protocol, max of 5, MiB | heap peak, cold protocol, max of 5, MiB |",
             "|---|---:|---:|---:|---:|"]
    hp, hc = defaultdict(list), defaultdict(list)
    for src, dst in ((heap, hp), (heap_cold, hc)):
        for h in src or []:
            if "heap_peak_bytes" in h:
                dst[h["entry"]].append(h["heap_peak_bytes"])
    for e in ORDER:
        rs = [r for r in rows if r["entry"] == e and r.get("cpu_ms_per_imu_s")]
        if not rs:
            continue
        c = [r["cpu_ms_per_imu_s"] for r in rs]
        m = [r["max_rss_bytes"] / 1e6 for r in rs if r.get("max_rss_bytes")]
        h = f"{max(hp[e]) / 2**20:.2f}" if hp.get(e) else "–"
        hcs = f"{max(hc[e]) / 2**20:.2f}" if hc.get(e) else "–"
        lines.append(f"| {LABEL[e]} | {np.median(c):.1f} ({max(c):.1f}) | {np.median(m):.0f} ({max(m):.0f}) | {h} | {hcs} |")
    return "\n".join(lines)


def gaps(by, s, rival):
    """Where `rival` beats ours, per group and metric (mean over the group, metres)."""
    out = []
    specs = sorted({d for (e, ss), m in by.items() if ss == s for d in m})
    for g, f in GROUPS.items():
        for k, name in METRICS:
            o = np.nanmean([val(by[("ours", s)].get(d), k) for d in specs if f(d)])
            r = np.nanmean([val(by[(rival, s)].get(d), k) for d in specs if f(d)])
            if np.isfinite(r) and r < o:
                out.append((g, name, o, r, o - r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--heap")
    ap.add_argument("--heap-cold")
    a = ap.parse_args()
    rows, by = load(a.raw)
    heap = json.load(open(a.heap)) if a.heap else None
    heap_cold = json.load(open(a.heap_cold)) if a.heap_cold else None
    sets = [s for s in SET_LABEL if any(ss == s for (_, ss) in by)]
    for s in sets:
        print(f"\n### {SET_LABEL[s]}\n")
        for k, name in METRICS:
            print(f"\n**{name}** (mean metres per group)\n\n{group_table(by, s, k)}")
    print("\n### Per recording, ATE (m), seed 42 accel-first; best non-baseline in bold\n")
    print(per_recording(by, "s42_accel_first", "ate_rmse_m"))
    print("\n### Per recording, blind 5 s (m), seed 42 accel-first\n")
    print(per_recording(by, "s42_accel_first", "blind_mean_m"))
    print("\n### Contract\n")
    print(contract_table(by, rows))
    print("\n### Resources\n")
    print(resource_table(rows, heap, heap_cold))
    for rival in ("vortexxx", "zerodrift", "vioedge"):
        for s in sets:
            g = gaps(by, s, rival)
            if g:
                print(f"\n**{LABEL[rival]} beats ours ({SET_LABEL[s]})**: " +
                      "; ".join(f"{gg} {n}: {o:.2f} vs {r:.2f} (−{d:.2f})" for gg, n, o, r, d in g))


if __name__ == "__main__":
    main()
