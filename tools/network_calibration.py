"""How well the network's stated velocity standard deviation matches its real error, per recording and body axis.

    python tools/network_calibration.py --run runs/airio/r0_s2 --out results/track3/network/calibration.json

Uses the body-frame window cache of tools/train_airio.py (every third window). z2 = mean of (error / stated std)^2:
1 means calibrated, above 1 over-confident. Also reports the RMS error and the RMS stated std (m/s).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_airio as T  # noqa: E402

SPECS = {"train": ["euroc:MH_01_easy", "euroc:MH_02_easy", "euroc:MH_03_medium", "euroc:V1_01_easy", "euroc:V1_02_medium",
                   "euroc:V2_01_easy", "euroc:V2_02_medium", "tumvi:room1", "tumvi:room2", "tumvi:room3", "tumvi:room4"],
         "validation": ["euroc:MH_04_difficult", "euroc:V1_03_difficult", "tumvi:room5"]}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/airio/r0_s2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(Path(a.run) / "best.pt", weights_only=True, map_location=dev)
    net = T.build(ck["widths"], ck["hidden"], ck["kernels"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    opts = argparse.Namespace(mirror=1, tilt_aug=0, rot_aug=0)
    out = {"run": a.run, "recordings": []}
    for split, specs in SPECS.items():
        for spec in specs:
            X, D, V, idx = T.load_split([spec], ck["window"], torch, dev)
            idx = idx[::3]
            z2, e2, s2 = [], [], []
            with torch.no_grad():
                for i in range(0, len(idx), 4096):
                    x, d, v = T.batch((X, D, V, idx), idx[i:i + 4096], ck["window"], False, opts, torch)
                    o = net(x)
                    s = torch.exp(o[:, 9:12].clamp(-7, 3))
                    e = o[:, 6:9] - v
                    z2.append(((e / s) ** 2).cpu()); e2.append((e ** 2).cpu()); s2.append((s ** 2).cpu())
            rec = {"data": spec, "split": split, "windows": int(len(idx)),
                   "vel_z2": torch.cat(z2).mean(0).tolist(),
                   "vel_rmse_mps": torch.cat(e2).mean(0).sqrt().tolist(),
                   "vel_std_rms_mps": torch.cat(s2).mean(0).sqrt().tolist()}
            out["recordings"].append(rec)
            print(spec, split, np.round(rec["vel_z2"], 2), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
