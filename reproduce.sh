#!/usr/bin/env bash
# Everything in one command: environment, data, models, tests, benchmark, report.
#
#   bash reproduce.sh               # uses our trained models from the GitHub release if present,
#                                   # otherwise trains them (~25 min on an Apple M-series / any GPU)
#   bash reproduce.sh --synthetic   # no download: synthetic head walks, ~3 min
#   bash reproduce.sh --train       # always retrain both models
set -euo pipefail
cd "$(dirname "$0")"

SYN=0; TRAIN=0
for a in "$@"; do
  case "$a" in --synthetic) SYN=1 ;; --train) TRAIN=1 ;; esac
done

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.11 .venv; else python3 -m venv .venv; fi
fi
PY=.venv/bin/python
if command -v uv >/dev/null; then uv pip install -q --python $PY -r requirements.txt; else $PY -m pip install -q -r requirements.txt; fi

echo "== tests"
$PY -m pytest -q tests

if [ "$SYN" = 1 ]; then
  echo "== benchmark (synthetic)"
  $PY evaluate.py --synthetic --seqs 5 --clip 60
  exit 0
fi

if [ ! -f local_data/tlio_golden/test_list.txt ]; then
  echo "== downloading the TLIO headset dataset (Meta, CC BY-NC 4.0, 3.8 GB zip, 8.5 GB unpacked)"
  mkdir -p local_data
  .venv/bin/gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg -O local_data/tlio_golden.zip
  unzip -q local_data/tlio_golden.zip -d local_data && rm local_data/tlio_golden.zip
fi

for aug in faults tlio; do
  if [ "$TRAIN" = 1 ] || [ ! -f models/dispnet_$aug/best.pt ]; then
    echo "== training dispnet_$aug"
    $PY train.py --aug $aug --epochs 10
  fi
done

echo "== benchmark"
$PY evaluate.py --seqs 10 --clip 60
echo "== demo"
$PY make_demo.py
echo "Open results/report.html and demo/index.html"
