"""Record what the mount guard does on real recordings.

    python tools/check_mount_guard.py --data euroc:MH_01_easy ... tumvi:room6 [--seed 1] \
        [--out results/track3/mount_guard_check.json]

For every recording (degraded with tools/degrade_imu.py, seeded): run the C++ entry with the guard on (default) and off
(--mount-guard-deg 0) and record whether the guard said anything and whether the two outputs are byte-identical. Then
rotate the same degraded IMU file by 90 deg about x and record whether the guard switches to the gravity-aligned
network. Expected: normal recordings quiet and identical; rotated copies switch (unless the device is not at rest at
start-up, where the guard holds off by design).
"""
import argparse
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from eval_track3 import source  # noqa: E402

RX = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


def run(binary, inp, out, *extra):
    p = subprocess.run([str(binary), str(inp), "-o", str(out), *extra], capture_output=True, text=True, check=True)
    return p.stderr


def rotate(src_csv, dst_csv):
    lines = Path(src_csv).read_text().splitlines()
    with open(dst_csv, "w") as f:
        for l in lines:
            if not l or not l[0].isdigit():
                f.write(l + "\n")
                continue
            c = l.split(",")
            try:
                v = [float(x) for x in c[1:7]]
            except ValueError:                          # degraded rows (NaN etc.) are kept as they are
                f.write(l + "\n")
                continue
            g = [sum(RX[i][j] * v[j] for j in range(3)) for i in range(3)]
            a = [sum(RX[i][j] * v[3 + j] for j in range(3)) for i in range(3)]
            f.write(c[0] + "," + ",".join(f"{x:.9g}" for x in g + a) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "results" / "track3" / "mount_guard_check.json"))
    a = ap.parse_args()
    binary = ROOT / "src" / "pathfinder"
    subprocess.run(["make", "-C", str(ROOT / "src"), "-s"], check=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.data:
            src, _ = source(spec)
            d = Path(tmp) / spec.replace(":", "_")
            subprocess.run([sys.executable, str(ROOT / "tools" / "degrade_imu.py"), str(src), str(d),
                            "--seed", str(a.seed)], check=True, capture_output=True)
            on, off = Path(tmp) / "on.txt", Path(tmp) / "off.txt"
            err = run(binary, d, on)
            run(binary, d, off, "--mount-guard-deg", "0")
            r = Path(tmp) / "rot" / "imu0"
            r.mkdir(parents=True, exist_ok=True)
            rotate(d / "imu0" / "data.csv", r / "data.csv")
            rerr = run(binary, r.parent, Path(tmp) / "rot.txt")
            rows.append({"recording": spec, "seed": a.seed,
                         "guard_message_normal": "IMU mount" in err,
                         "identical_to_guard_off": filecmp.cmp(on, off, shallow=False),
                         "switches_when_rotated_90x": "using velnet_weights_world.bin" in rerr,
                         "rotated_message": next((l for l in rerr.splitlines() if "mount" in l), "")})
            print(json.dumps(rows[-1]), flush=True)
    res = {"normal_all_quiet_and_identical": all(not r["guard_message_normal"] and r["identical_to_guard_off"] for r in rows),
           "rotated_switched": sum(r["switches_when_rotated_90x"] for r in rows), "recordings": len(rows), "rows": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"normal quiet+identical: {res['normal_all_quiet_and_identical']}; rotated switched "
          f"{res['rotated_switched']}/{len(rows)} -> {out}")


if __name__ == "__main__":
    main()
