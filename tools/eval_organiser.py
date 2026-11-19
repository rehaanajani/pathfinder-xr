"""Score the entry on the ORGANISERS' degradation script (sent to teams on 16 Nov 2026; not in this repo).

    python tools/eval_organiser.py --script <path to their degrade_imu.py> --data euroc:MH_05_difficult tumvi:room6 \
        [--layout accel_first|euroc] [--tag NAME] [-- extra main.py / C++ options]

Their script reads `timestamp, accel xyz, gyro xyz` (its docstring) with a fixed seed (42). We feed it either
`accel_first` (that documented order, header "timestamp,ax,ay,az,gx,gy,gz") or `euroc` (the EuRoC file unchanged,
gyro first), run the C++ entry on the result (it detects the column order itself) and score with
tools/score_track3.py. The failure is the last 5 s of the ORIGINAL timeline (their `failure_seconds`), kept as
all-zero rows; dropouts are the timestamp gaps their script leaves (> 1.5 x the nominal period).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402
from score_track3 import score  # noqa: E402


def make_input(src_csv, dst_csv, layout):
    d = np.loadtxt(src_csv, delimiter=",", comments="#")
    if layout == "euroc":
        Path(dst_csv).write_text(Path(src_csv).read_text())
        return d[:, 0]
    with open(dst_csv, "w") as f:
        f.write("timestamp,ax,ay,az,gx,gy,gz\n")
        for r in d:
            f.write(f"{int(r[0])}," + ",".join(f"{x:.9g}" for x in (*r[4:7], *r[1:4])) + "\n")
    return d[:, 0]


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--layout", choices=["accel_first", "euroc"], default="accel_first")
    ap.add_argument("--tag", default="organiser")
    ap.add_argument("--entry", choices=["cpp", "python"], default="cpp",
                    help="score the C++ port (src/pathfinder, fast) or src/main.py (same output within 1 um)")
    a = ap.parse_args(argv)
    cmd = [str(ROOT / "src" / "pathfinder")] if a.entry == "cpp" else [sys.executable, str(ROOT / "src" / "main.py")]
    subprocess.run(["make", "-C", str(ROOT / "src"), "-s"], check=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.data:
            src, gt = source(spec)
            d = Path(tmp) / spec.replace(":", "_")
            (d / "imu0").mkdir(parents=True)
            clean = d / "clean.csv"
            t_orig = make_input(src / "imu0" / "data.csv", clean, a.layout)
            p = subprocess.run([sys.executable, a.script, str(clean), str(d / "imu0" / "data.csv"),
                                "--meta", str(d / "meta.json")], capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"organiser script failed on {spec}: {p.stderr}")
            t_deg = np.array([int(l.split(",", 1)[0]) for l in (d / "imu0" / "data.csv").read_text().splitlines()
                              if l and l[0].isdigit()])
            dur = (t_orig[-1] - t_orig[0]) * 1e-9
            nom = float(np.median(np.diff(t_orig))) * 1e-9
            gaps = [((t_deg[i] - t_orig[0]) * 1e-9, (t_deg[i + 1] - t_orig[0]) * 1e-9)
                    for i in range(len(t_deg) - 1) if (t_deg[i + 1] - t_deg[i]) * 1e-9 > 1.5 * nom]
            truth = {"t0_ns": int(t_orig[0]), "t_fail_s": dur - 5.0, "dropouts_s": gaps}
            est = d / "est.txt"
            r = subprocess.run([*cmd, str(d), "-o", str(est), *extra], capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"entry failed on {spec}: {r.stderr}")
            res = score(str(est), str(gt), truth, str(d / "imu0" / "data.csv"))
            meta = json.loads((d / "meta.json").read_text())
            res.update(data=spec, layout=a.layout, entry_stderr=[l for l in r.stderr.splitlines() if "wrote" not in l],
                       degradation={k: meta[k] for k in ("rows_in", "rows_out", "spikes", "stuck_samples",
                                                          "dropped_total", "duplicated_rows", "failure_frames")})
            rows.append(res)
            print(f"{spec:28s} ATE {res['ate_rmse_m']:6.2f}  ckpt {res['checkpoints_m']:6.2f}  "
                  f"blind {res.get('blind_mean_m', float('nan')):6.2f}  drop {res.get('dropout_m', float('nan')):6.2f}  "
                  f"rows-ok {res['rows_match_input']}  {'; '.join(res['entry_stderr'])}", flush=True)
    keys = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m"]
    mean = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    print("MEAN " + "  ".join(f"{k.split('_')[0]} {v:.3f}" for k, v in mean.items()))
    out = ROOT / "runs" / "organiser" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"layout": a.layout, "extra": extra, "mean": mean, "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
