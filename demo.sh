#!/usr/bin/env bash
# Path Finder demo: one word per step. Run from the project folder:
#
#   ./demo.sh            show this menu
#   ./demo.sh check      is everything installed? (data, models, phone)
#   ./demo.sh tests      run all tests (about 20 s)
#   ./demo.sh page       open the interactive demo page in the browser
#   ./demo.sh quick      fast check on 3 real headset recordings (about 1 min), opens the report
#   ./demo.sh headset    benchmark on 12 real headset recordings from Meta (about 4 min)
#   ./demo.sh camera     real cameras + real IMU on TUM-VI, with dark/blur/blackout (about 3 min)
#   ./demo.sh phone      speed test on a USB-connected Android phone (about 3 min)
#   ./demo.sh live       LIVE: phone as the headset's sensors; cover the camera / shake it (browser view)
#   ./demo.sh replay F   replay a saved live session (runs/live/*.jsonl) without a phone
#
# Results go to runs/demo_* so the committed results are never overwritten.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ok()   { printf "  \033[32mOK\033[0m   %s\n" "$1"; }
miss() { printf "  \033[31mMISSING\033[0m %s\n" "$1"; }

need_py() {
  if [ ! -x "$PY" ]; then
    echo "Python setup missing. Creating it (one time, about 2 min)..."
    if command -v uv >/dev/null; then uv venv -q -p 3.11 .venv && uv pip install -q -p $PY -r requirements.lock
    else python3 -m venv .venv && $PY -m pip install -q -r requirements.lock; fi
  fi
}

open_file() { if command -v open >/dev/null; then open "$1"; else echo "open $1 in a browser"; fi; }

case "${1:-}" in
  check)
    echo "Path Finder: what's in place"
    [ -x "$PY" ] && ok "Python setup (.venv)" || miss "Python setup: run ./demo.sh tests and it installs itself"
    [ -d local_data/tlio_golden ] && ok "Meta TLIO headset data" || miss "Meta TLIO data (needed for 'headset'): bash reproduce.sh downloads it"
    ls -d local_data/tumvi/dataset-room1_512_16 >/dev/null 2>&1 && ok "TUM-VI camera data" || miss "TUM-VI data (needed for 'camera')"
    [ -f models/dispnet_faults/best.pt ] && ok "trained motion model" || miss "motion model: bash reproduce.sh fetches it"
    [ -f models/dispnet_faults.onnx ] && ok "phone model (ONNX)" || miss "phone model: run .venv/bin/python export_onnx.py"
    [ -f runs/phone/bench ] && ok "phone speed program" || miss "phone speed program: bash tools/phone_bench/build.sh (needs Android NDK + runs/phone/ort)"
    if command -v adb >/dev/null && adb get-state >/dev/null 2>&1; then
      ok "phone connected: $(adb shell getprop ro.product.model | tr -d '\r')"
    else
      miss "phone not connected (only needed for 'phone'): USB debugging on, plug in, tap Allow"
    fi
    ;;
  tests)
    need_py
    $PY -m pytest -q tests
    ;;
  page)
    open_file demo/index.html
    echo "Pick a failure on the page and watch: grey = true path, orange = normal tracker, green = ours."
    ;;
  quick)
    need_py
    $PY evaluate.py --seqs 3 --clip 60 --methods naive_fusion,ours --out runs/demo_quick && \
      open_file runs/demo_quick/report.html
    ;;
  headset)
    need_py
    $PY evaluate.py --seqs 12 --clip 60 --out runs/demo_headset && open_file runs/demo_headset/report.html
    ;;
  camera)
    need_py
    $PY evaluate_real_vo.py --set speed_limit=true --set outage_v_std=2.0 --out runs/demo_camera
    echo
    echo "Read the table above: lower is better (metres off). vo_only = camera alone, conventional = normal"
    echo "tracker, ours_1imu = ours with the real IMU, ours_3imu = ours + 2 simulated extra IMUs."
    ;;
  phone)
    if ! adb get-state >/dev/null 2>&1; then
      echo "No phone found. On the phone: Settings > About phone > Software information > tap Build number 7x;"
      echo "then Developer options > USB debugging ON. Plug in with a data cable and tap Allow."
      exit 1
    fi
    [ -f runs/phone/bench ] || bash tools/phone_bench/build.sh
    bash tools/phone_bench/run_on_phone.sh && cat results/phone_benchmark.json
    ;;
  live)
    bash tools/live_demo/run.sh "${@:2}"
    ;;
  replay)
    (sleep 2; open_file http://localhost:8080) &
    $PY tools/live_demo/server.py --replay "$2"
    ;;
  *)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
