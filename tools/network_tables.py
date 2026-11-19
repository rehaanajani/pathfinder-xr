"""Markdown tables for results/track3/network_*.md from the raw JSON in results/track3/network/.

    python tools/network_tables.py <tag> [<tag> ...] [--labels "a" "b" ...] [--per-recording]
    python tools/network_tables.py --seeds sel_s{seed}_vref1.0 sel_s{seed}_vref0.5 --seed-list 101 102 103 104 105

Each <tag>.json is a tools/eval_organiser.py result (copied from runs/organiser/). The first form prints the mean of
ATE, checkpoints, blind 5 s and dropouts per tag (and per recording with --per-recording). --seeds compares two tag
patterns over several degradation seeds: per seed and recording, the paired difference (second minus first).
"""
import argparse
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "track3" / "network"
K = [("ate_rmse_m", "ATE"), ("checkpoints_m", "Checkpoints"), ("blind_mean_m", "Blind 5 s"), ("dropout_m", "Dropouts")]


def load(tag):
    return json.load(open(D / f"{tag}.json"))["rows"]


def means(rows):
    return [float(np.mean([r[k] for r in rows])) for k, _ in K]


def table(tags, labels, per_recording):
    out = ["| | " + " | ".join(n for _, n in K) + " |", "|---|" + "---:|" * len(K)]
    for t, lab in zip(tags, labels):
        out.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in means(load(t))) + " |")
    if per_recording:
        out += ["", "| Recording | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(tags)]
        names = list(dict.fromkeys(r["data"] for r in load(tags[0])))
        for n in names:
            cells = []
            for t in tags:
                r = [x for x in load(t) if x["data"] == n]
                cells.append(" / ".join(f"{x[k]:.2f}" for x in r for k, _ in K[:2]) if r else "")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("Per-recording cells: ATE / checkpoints.")
    return "\n".join(out)


def seeds(a, b, seed_list):
    out = ["| Seed | " + " | ".join(f"{n} (first)" for _, n in K) + " | " + " | ".join(f"{n} (second)" for _, n in K)
           + " |", "|---|" + "---:|" * (2 * len(K))]
    diffs = []
    for s in seed_list:
        ra, rb = load(a.format(seed=s)), load(b.format(seed=s))
        out.append(f"| {s} | " + " | ".join(f"{v:.3f}" for v in means(ra)) + " | "
                   + " | ".join(f"{v:.3f}" for v in means(rb)) + " |")
        for x in ra:
            y = next(r for r in rb if r["data"] == x["data"])
            diffs.append([y[k] - x[k] for k, _ in K])
    d = np.array(diffs)
    out += ["", f"Paired differences (second minus first) over {len(d)} recording-seed pairs:", "",
            "| | " + " | ".join(n for _, n in K) + " |", "|---|" + "---:|" * len(K),
            "| mean | " + " | ".join(f"{v:+.3f}" for v in d.mean(0)) + " |",
            "| standard error | " + " | ".join(f"{v:.3f}" for v in d.std(0, ddof=1) / np.sqrt(len(d))) + " |",
            "| pairs where second is lower | " + " | ".join(f"{int((d[:, j] < 0).sum())} / {len(d)}" for j in range(len(K)))
            + " |"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--per-recording", action="store_true")
    ap.add_argument("--seeds", nargs=2)
    ap.add_argument("--seed-list", nargs="*", type=int)
    a = ap.parse_args()
    if a.seeds:
        print(seeds(a.seeds[0], a.seeds[1], a.seed_list))
    else:
        print(table(a.tags, a.labels or a.tags, a.per_recording))


if __name__ == "__main__":
    main()
