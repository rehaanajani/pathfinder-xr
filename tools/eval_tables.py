"""Markdown tables for docs/TRACK3_EVAL.md from runs/track3/<tag>/results.json and runs/track3/blind/blind.json."""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "runs" / "track3"


def evo_tables(tag="eval"):
    r = json.load(open(R / tag / "results.json"))
    names = list(dict.fromkeys(x["data"] for x in r))
    out = ["| recording | evo ATE -a (ours) | ATE -as (ours) | evo RPE 1 s (ours) | ATE -a (still) | ATE -as (still) | RPE 1 s (still) |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        x = [q for q in r if q["data"] == n]
        m = lambda lab, k: np.mean([q[lab][k] for q in x])
        out.append(f"| {n} | {m('ours','evo_ape_se3'):.2f} | {m('ours','evo_ape_sim3'):.2f} | {m('ours','evo_rpe_1s'):.2f} | "
                   f"{m('still','evo_ape_se3'):.2f} | {m('still','evo_ape_sim3'):.2f} | {m('still','evo_rpe_1s'):.2f} |")
    mm = lambda lab, k: np.mean([q[lab][k] for q in r])
    out.append(f"| **mean (33 runs)** | **{mm('ours','evo_ape_se3'):.2f}** | **{mm('ours','evo_ape_sim3'):.2f}** | **{mm('ours','evo_rpe_1s'):.2f}** | "
               f"**{mm('still','evo_ape_se3'):.2f}** | **{mm('still','evo_ape_sim3'):.2f}** | **{mm('still','evo_rpe_1s'):.2f}** |")
    t1 = "\n".join(out)
    out = ["| recording | seed | evo ATE -a | score_track3 ATE | diff | evo RPE | score_track3 RPE | diff |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for q in r:
        o = q["ours"]
        da = 100 * (o["score_track3"]["ate_rmse_m"] / o["evo_ape_se3"] - 1)
        dr = 100 * (o["score_track3"]["rpe_1s_m"] / o["evo_rpe_1s"] - 1)
        for lab in ("still",):
            s = q[lab]
            da = max(da, 100 * (s["score_track3"]["ate_rmse_m"] / s["evo_ape_se3"] - 1), key=abs)
            dr = max(dr, 100 * (s["score_track3"]["rpe_1s_m"] / s["evo_rpe_1s"] - 1), key=abs)
        if abs(da) > 5 or abs(dr) > 5:
            out.append(f"| {q['data']} | {q['seed']} | {o['evo_ape_se3']:.2f} | {o['score_track3']['ate_rmse_m']:.2f} | {da:+.1f} % | "
                       f"{o['evo_rpe_1s']:.2f} | {o['score_track3']['rpe_1s_m']:.2f} | {dr:+.1f} % |")
    t2 = "\n".join(out)
    wins = {k: sum(q["ours"][k] < q["still"][k] for q in r) for k in ("evo_ape_se3", "evo_ape_sim3", "evo_rpe_1s")}
    return t1, t2, wins, r


def blind_table():
    r = json.load(open(R / "blind" / "blind.json"))
    out = ["| recording | kills | ours abs (m) | still abs (m) | hold abs (m) | ours moved-since-kill err (m) | still/hold moved-since-kill err (m) | kills ours worse than still (abs) | worse than hold (rel) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in dict.fromkeys(x["data"] for x in r):
        x = [q for q in r if q["data"] == n]
        f = lambda a, b: np.array([q[a][b] for q in x])
        out.append(f"| {n} | {len(x)} | {f('ours','mean_m').mean():.2f} | {f('still','mean_m').mean():.2f} | {f('hold','mean_m').mean():.2f} | "
                   f"{f('ours','rel_mean_m').mean():.2f} | {f('hold','rel_mean_m').mean():.2f} | "
                   f"{(f('ours','mean_m') > f('still','mean_m')).sum()} | {(f('ours','rel_mean_m') > f('hold','rel_mean_m')).sum()} |")
    f = lambda a, b: np.array([q[a][b] for q in r])
    d_abs = f("ours", "mean_m") - f("still", "mean_m")
    d_rel = f("ours", "rel_mean_m") - f("hold", "rel_mean_m")
    out.append(f"| **all** | {len(r)} | {f('ours','mean_m').mean():.2f} | {f('still','mean_m').mean():.2f} | {f('hold','mean_m').mean():.2f} | "
               f"{f('ours','rel_mean_m').mean():.2f} | {f('hold','rel_mean_m').mean():.2f} | {(d_abs > 0).sum()} | {(d_rel > 0).sum()} |")
    extra = {"worst_loss_abs": d_abs.max(), "best_gain_abs": d_abs.min(), "worst_loss_rel": d_rel.max(), "best_gain_rel": d_rel.min(),
             "mean_gain_rel_pct": 100 * (1 - f('ours', 'rel_mean_m').mean() / f('hold', 'rel_mean_m').mean())}
    return "\n".join(out), extra, r


if __name__ == "__main__":
    t1, t2, wins, _ = evo_tables()
    print(t1, "\n\n", t2, "\n", wins)
    t3, ex, _ = blind_table()
    print(t3, "\n", ex)
