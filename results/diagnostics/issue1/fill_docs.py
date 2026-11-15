"""Fill the before/after tables in docs/ESTIMATOR.md from saved evidence (no numbers typed by hand)."""
import re
import sys
from pathlib import Path

EV = Path(sys.argv[1])
doc = Path("docs/ESTIMATOR.md")


def rd(p):
    raw = p.read_bytes()
    return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")


def parse_matrix(p):
    rows = {}
    for line in rd(p).splitlines():
        m = re.match(r"(.{24}) (oracle|none)\s+fault mean ([\d.]+) max ([\d.]+) \| 2-10 s after ([\d.na]+) \| last 10 s ([\d.]+) \| reanchors (\[.*\])", line)
        if m:
            rows[(m.group(1).strip(), m.group(2))] = m.groups()[2:]
    return rows


b, a = parse_matrix(EV / "wrongfix_matrix_baseline.txt"), parse_matrix(EV / "wrongfix_matrix_final.txt")
out = ["| Case | displacement | before mean / max | after mean / max | after: error 2–10 s later | after: re-anchors per seed |",
       "|---|---|---:|---:|---:|---|"]
for k, v in a.items():
    bb = b[k]
    out.append(f"| {k[0]} | {k[1]} | {bb[0]} / {bb[1]} | **{v[0]} / {v[1]}** | {v[2]} | {v[4]} |")
matrix = "\n".join(out) + (
    "\n\nBefore = `ff7ec44` with the pipeline's own confidence input; after = this branch. Without the "
    "network, the wrong-fix rows match the blackout row because a rejected camera leaves the inertial "
    "estimate alone; the 20 s no-model case re-anchors onto the wrong fix after the 10 s hold-out (see limits) "
    "and back after it ends. `blackout + bias` puts the same bias on every IMU, which no IMU comparison can see.")


def parse_scen(text, pat):
    rows = {}
    for line in text.splitlines():
        m = re.match(r"(\w+)\s+imu_only=", line)
        if m:
            vals = dict(re.findall(r"(\w+)=([\d.]+)", line.split("valid")[0]))
            rows[m.group(1)] = vals
    return rows


sb = parse_scen(rd(EV.parent / "synthetic_eval.log"), None)
sa = parse_scen(rd(EV / "synthetic_eval_after.txt"), None)
out = ["| Scenario | naive | ours_no_model before → after | ours (models-v1 net) before → after |",
       "|---|---:|---:|---:|"]
for n, v in sa.items():
    out.append(f"| {n} | {v['naive_fusion']} | {sb[n]['ours_no_model']} → {v['ours_no_model']} | {sb[n]['ours']} → {v['ours']} |")
scen = "\n".join(out) + (
    "\n\nBefore = the audit's `synthetic_eval.log` (same seeds, same scenarios); after = "
    "`probes/synthetic_eval_after.py` (writes nothing under `results/`). ATE over finite samples, truth-initialised "
    "benchmark mode. The network rows are dominated by its out-of-domain error on synthetic motion (§5).")
s = doc.read_text(encoding="utf-8")
assert "FILLED_BY_MATRIX" in s and "FILLED_BY_SCENARIOS" in s
s = s.replace("FILLED_BY_MATRIX", matrix).replace("FILLED_BY_SCENARIOS", scen)
doc.write_text(s, encoding="utf-8", newline="")
print(matrix)
print(scen)
