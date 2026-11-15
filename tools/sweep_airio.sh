#!/bin/bash
# Train + export + score one configuration of tools/train_airio.py (validation only).
#   bash tools/sweep_airio.sh <name> [train_airio.py train options]
# Scores: MH_04 + V1_03 (validation), room5, and MH_04 + V1_03 with the IMU mounted 90 deg rotated
# (local_data/rot, see docs/AIRIO.md). Results in runs/track3/<name>_{val,r5,rot}.json.
set -e
cd "$(dirname "$0")/.."
n=$1; shift
python tools/train_airio.py train --out runs/airio/$n "$@" > runs/airio/$n.log 2>&1
python tools/train_airio.py export --run runs/airio/$n --out runs/airio/$n/velnet_body.bin > /dev/null
M="--model runs/airio/$n/velnet_body.bin"
PF_ENTRY=cpp python tools/eval_track3.py --data euroc:MH_04_difficult euroc:V1_03_difficult --seeds 1 2 3 --tag ${n}_val -- $M > /dev/null
PF_ENTRY=cpp python tools/eval_track3.py --data tumvi:room5 --seeds 1 2 3 --tag ${n}_r5 -- $M > /dev/null
PFX_DATA=$PWD/local_data/rot PF_ENTRY=cpp python tools/eval_track3.py --data euroc:MH_04_difficult euroc:V1_03_difficult --seeds 1 2 3 --tag ${n}_rot -- $M > /dev/null
python - "$n" <<'PY'
import json, sys
n = sys.argv[1]
K = ["ate_rmse_m", "checkpoints_m", "blind_mean_m", "dropout_m"]
out = []
for part in ("val", "r5", "rot"):
    r = json.load(open(f"runs/track3/{n}_{part}.json"))["rows"]
    out.append(part + " " + " ".join(f"{sum(x[k] for x in r) / len(r):.3f}" for k in K))
print(n, " | ".join(out), flush=True)
PY
