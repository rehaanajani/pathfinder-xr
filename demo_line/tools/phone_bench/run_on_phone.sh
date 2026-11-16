#!/bin/bash
# Times Path Finder on a USB-connected Android phone, then removes everything it copied.
# Only touches /data/local/tmp/pathfinder. Writes results/phone_benchmark.json.
set -e
D=/data/local/tmp/pathfinder
adb get-state >/dev/null
adb shell mkdir -p $D
adb push -q runs/phone/bench runs/phone/ort/jni/arm64-v8a/libonnxruntime.so \
         models/dispnet_faults.onnx models/dispnet_faults_int8.onnx $D/
trap 'adb shell rm -rf $D' EXIT
dev=$(adb shell 'printf "{\"model\":\"%s\",\"soc\":\"%s %s\",\"android\":\"%s\",\"abi\":\"%s\"}" \
  "$(getprop ro.product.model)" "$(getprop ro.soc.manufacturer)" "$(getprop ro.soc.model)" \
  "$(getprop ro.build.version.release)" "$(getprop ro.product.cpu.abi)"')
out="["
for m in dispnet_faults.onnx dispnet_faults_int8.onnx; do
  for t in 1 2; do
    r=$(adb shell "cd $D && chmod 755 bench && LD_LIBRARY_PATH=$D ./bench $m 2000 $t")
    echo "$r"
    out="$out$r,"
  done
done
out="${out%,}]"
printf '{"device":%s,"runs":%s}\n' "$dev" "$out" | python3 -m json.tool > results/phone_benchmark.json
echo "wrote results/phone_benchmark.json; removed $D from the phone"
