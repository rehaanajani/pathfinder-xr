#!/usr/bin/env bash
# Everything in one command: environment, data, models, tests, benchmark, report.
#
#   bash reproduce.sh               # downloads and SHA-256-verifies the released models (fails if it cannot;
#                                   # it never retrains silently)
#   bash reproduce.sh --synthetic   # no download: synthetic head walks, ~3 min
#   bash reproduce.sh --train       # intentionally retrain both models (new model identity, see train.py)
# Real-vision (TUM-VI) workflow: docs/REPRODUCIBILITY.md.
set -euo pipefail
cd "$(dirname "$0")"

SYN=0; TRAIN=0
for a in "$@"; do
  case "$a" in --synthetic) SYN=1 ;; --train) TRAIN=1 ;; esac
done

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.11 .venv; else python3 -m venv .venv; fi
fi
# Windows venvs put the interpreter in Scripts/, POSIX venvs in bin/.
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
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
  $PY -m gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg -O local_data/tlio_golden.zip
  unzip -q local_data/tlio_golden.zip -d local_data && rm local_data/tlio_golden.zip
fi

if [ "$TRAIN" = 1 ]; then
  for aug in faults tlio; do
    echo "== training dispnet_$aug (intentional retraining: results are a new model identity)"
    $PY train.py --aug $aug --epochs 10
  done
else
  $PY tools/fetch_models.py          # exits non-zero on any missing/mismatched checkpoint
fi

echo "== benchmark"
$PY evaluate.py --seqs 36 --clip 60
echo "== demo"
$PY make_demo.py
echo "Open results/report.html and demo/index.html"
