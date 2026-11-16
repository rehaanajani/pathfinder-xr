#!/usr/bin/env bash
# Frozen benchmark of the Track 3 entry on this machine: resources, accuracy (our degradation stand-in and, if given,
# the organisers' script), and every test suite. Writes results/reports/<date>-<host>-<commit>/.
#
#   PFX_DATA=<datasets> bash tools/benchmark_report.sh [path to the organisers' degrade_imu.py]
#
# Needs: a C++17 compiler + zlib, python with numpy/scipy/pandas/pytest, and EuRoC + TUM-VI (tools/fetch_*.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ORG=${1:-}
COMMIT=$(git rev-parse --short HEAD)
OUT="results/reports/$(date +%Y%m%d)-$(hostname -s | tr ' ' '-')-$COMMIT"
mkdir -p "$OUT"
make -C src -s
export PF_ENTRY=cpp OMP_NUM_THREADS=1

HO="euroc:MH_05_difficult euroc:V2_03_difficult tumvi:room6"
VAL="euroc:MH_04_difficult euroc:V1_03_difficult"

echo "== resources"
$PY tools/measure_budget.py --data $HO $VAL --repeat 5 --out "$OUT/budget.json" | tail -1

echo "== held-out accuracy (our degradation stand-in, seeds 1-3)"
for cfg in "entry:" "still:--v-max 0.0000001" "oldnet:--model src/velnet_weights_world.bin"; do
  tag=${cfg%%:*}; args=${cfg#*:}
  $PY tools/eval_track3.py --data $HO --seeds 1 2 3 --tag "report_heldout_$tag" -- $args | tail -1 | sed "s/^MEAN/$tag/"
  cp "runs/track3/report_heldout_$tag.json" "$OUT/heldout_$tag.json"
done

if [ -n "$ORG" ]; then
  echo "== organisers' degradation script"
  for lay in accel_first euroc; do for cfg in "entry:" "still:--v-max 0.0000001"; do
    tag=${cfg%%:*}; args=${cfg#*:}
    $PY tools/eval_organiser.py --script "$ORG" --data $VAL $HO --layout $lay --tag "report_org_${lay}_$tag" -- $args \
      | tail -1 | sed "s/^MEAN/$lay $tag/"
    cp "runs/organiser/report_org_${lay}_$tag.json" "$OUT/organiser_${lay}_$tag.json"
  done; done
fi

echo "== tests"
{
  echo "cpp_stress: $($PY -m pytest -q -p no:cacheprovider tests/stress 2>&1 | tail -1)"
  echo "root: $(PF_ENTRY= $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1)"
  echo "demo_line: $(cd demo_line && PF_ENTRY= $PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1)"
} | tee "$OUT/tests.txt"

$PY - "$OUT" <<'EOF'
import json, platform, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1])
cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
    if platform.system() == "Darwin" else platform.processor()
info = {"commit": out.name.rsplit("-", 1)[-1], "os": platform.platform(), "cpu": cpu,
        "compiler": subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]}
(out / "machine.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info))
EOF
echo "wrote $OUT"
