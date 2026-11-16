#!/usr/bin/env bash
# Everything in one command: locked core environment, tests, and an isolated evaluation run.
#
#   bash reproduce.sh               # downloads and SHA-256-verifies the released models (fails if it cannot;
#                                   # it never retrains silently)
#   bash reproduce.sh --synthetic   # no dataset/model download: generated stereo smoke evaluation
#   bash reproduce.sh --train       # intentionally retrain both models (new model identity, see train.py)
# Real-vision (TUM-VI) workflow: docs/REPRODUCE.md.
set -euo pipefail
cd "$(dirname "$0")"

SYN=0; TRAIN=0
for a in "$@"; do
  case "$a" in --synthetic) SYN=1 ;; --train) TRAIN=1 ;; *) echo "Unknown option: $a" >&2; exit 2 ;; esac
done

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.13 .venv; else python3 -m venv .venv; fi
fi
# Windows venvs put the interpreter in Scripts/, POSIX venvs in bin/.
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 13), "Use a Python 3.13 venv for the locked environment"'
if command -v uv >/dev/null; then
  uv pip install -q --python "$PY" --require-hashes -r requirements/core-dev-py313.lock
else
  "$PY" -m pip install -q --require-hashes -r requirements/core-dev-py313.lock
fi

echo "== tests"
$PY -m pytest -q tests

if [ "$SYN" = 1 ]; then
  echo "== synthetic stereo smoke evaluation"
  "$PY" tools/smoke_eval.py --out results/runs
  exit 0
fi

# Legacy learned-model evaluation needs optional packages beyond the hash-locked core.
# These installs are not hash-locked; preserve the tested core versions using constraints.
if command -v uv >/dev/null; then
  uv pip install -q --python "$PY" -c requirements/core-dev.in 'torch==2.8.0' 'gdown>=5,<6'
else
  "$PY" -m pip install -q -c requirements/core-dev.in 'torch==2.8.0' 'gdown>=5,<6'
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
"$PY" evaluate.py --seqs 36 --clip 60
# Evaluation prints its new output directory; historical committed reports are preserved.
