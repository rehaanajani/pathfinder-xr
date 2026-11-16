#!/usr/bin/env bash
# Build the C++ entry for a 64-bit Android phone (NDK), run it there on degraded recordings and report its own peak
# RSS and CPU per second of IMU. Phone: USB debugging on. Nothing is installed; files go to /data/local/tmp/pf3.
#
#   bash tools/android/bench_on_phone.sh <degraded folder with imu0/data.csv> [...]   -> results/reports/phone_<model>.json
set -euo pipefail
cd "$(dirname "$0")/../.."
NDK=${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Library/Android/sdk/ndk/* | tail -1)}
CXX=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android34-clang++ | head -1)
CC=${CXX%++}
ADB=${ADB:-$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")}
B=build/android; mkdir -p "$B"
"$CXX" -O3 -std=c++17 -static-libstdc++ -o "$B/pathfinder" src/main.cpp -lz
"$CC" -O2 -o "$B/measure" tools/android/measure.c
D=/data/local/tmp/pf3
"$ADB" shell "mkdir -p $D/in"
"$ADB" push -q "$B/pathfinder" "$B/measure" src/velnet_weights.bin src/velnet_weights_world.bin "$D/" >/dev/null
"$ADB" shell "chmod 755 $D/pathfinder $D/measure"
MODEL=$("$ADB" shell getprop ro.product.model | tr -d '\r' | tr ' ' '_')
SOC=$("$ADB" shell getprop ro.soc.model | tr -d '\r')
ABI=$("$ADB" shell getprop ro.product.cpu.abi | tr -d '\r')
ANDROID=$("$ADB" shell getprop ro.build.version.release | tr -d '\r')
ROWS_JSON=""
for d in "$@"; do
  name=$(basename "$d")
  "$ADB" push -q "$d/imu0/data.csv" "$D/in/$name.csv" >/dev/null
  rows=$(grep -c '^[0-9]' "$d/imu0/data.csv")
  secs=$(awk -F, '/^[0-9]/{if(!f){f=$1} l=$1} END{printf "%.3f", (l-f)/1e9}' "$d/imu0/data.csv")
  best_cpu=""; peak=0
  for i in 1 2 3 4 5; do
    r=$("$ADB" shell "$D/measure $D/pathfinder $D/in/$name.csv -o $D/out.txt" | tail -1 | tr -d '\r')
    kb=$(echo "$r" | awk '{print $1}'); cpu=$(echo "$r" | awk '{print $2+$3}'); ex=$(echo "$r" | awk '{print $4}')
    [ "$ex" = "0" ] || { echo "entry failed on the phone: $r"; exit 1; }
    peak=$(( kb > peak ? kb : peak ))
    best_cpu=$(awk -v a="$cpu" -v b="${best_cpu:-1e9}" 'BEGIN{print (a<b)?a:b}')
  done
  poses=$("$ADB" shell "grep -c '^[0-9]' $D/out.txt" | tr -d '\r')
  ms=$(awk -v c="$best_cpu" -v s="$secs" 'BEGIN{printf "%.3f", 1000*c/s}')
  echo "$name: rows $rows, poses $poses, peak $(awk -v k=$peak 'BEGIN{printf "%.2f", k/1024}') MB, $ms ms per s of IMU"
  ROWS_JSON+="{\"recording\":\"$name\",\"imu_rows\":$rows,\"poses\":$poses,\"imu_seconds\":$secs,\"peak_rss_mb\":$(awk -v k=$peak 'BEGIN{printf "%.3f", k/1024}'),\"cpu_ms_per_s_of_imu\":$ms},"
done
mkdir -p results/reports
OUT="results/reports/phone_${MODEL}.json"
printf '{"device":"%s","soc":"%s","abi":"%s","android":"%s","commit":"%s","build":"NDK aarch64 clang++ -O3, static libc++","method":"5 runs each; peak RSS max, CPU (user+sys) min, from wait4 on the phone","rows":[%s]}\n' \
  "$MODEL" "$SOC" "$ABI" "$ANDROID" "$(git rev-parse --short HEAD)" "${ROWS_JSON%,}" > "$OUT"
"$ADB" shell "rm -rf $D"
echo "wrote $OUT"
