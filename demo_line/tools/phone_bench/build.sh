#!/bin/bash
# Builds the Android arm64 benchmark. Needs the Android NDK and the ONNX Runtime Android AAR
# (Maven Central: com.microsoft.onnxruntime:onnxruntime-android:1.32.0) unzipped into $ORT.
set -e
NDK=${NDK:-$HOME/Library/Android/sdk/ndk/27.1.12297006}
ORT=${ORT:-runs/phone/ort}
CXX=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang++
$CXX -O2 -std=c++17 -I$ORT/headers tools/phone_bench/bench.cpp -L$ORT/jni/arm64-v8a -lonnxruntime \
     -o runs/phone/bench -static-libstdc++
echo built runs/phone/bench
