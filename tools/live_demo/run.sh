#!/bin/bash
# Live demo: phone streams its sensors over USB, the laptop tracks live, the browser shows it.
#   bash tools/live_demo/run.sh            build+install the phone app if needed, then start
#   bash tools/live_demo/run.sh --model    also use the learned motion model
set -e
cd "$(dirname "$0")/../.."
adb get-state >/dev/null 2>&1 || { echo "No phone: turn on USB debugging, plug in, tap Allow."; exit 1; }
APK=android/app/build/outputs/apk/debug/app-debug.apk
if [ ! -f "$APK" ]; then
  export JAVA_HOME=${JAVA_HOME:-"/Applications/Android Studio.app/Contents/jbr/Contents/Home"}
  [ -f android/local.properties ] || echo "sdk.dir=$HOME/Library/Android/sdk" > android/local.properties
  (cd android && ./gradlew -q assembleDebug)
fi
adb shell pm list packages | grep -q xr.pathfinder.bridge || adb install -r "$APK"
adb reverse tcp:7777 tcp:7777
adb shell am start -n xr.pathfinder.bridge/.MainActivity >/dev/null
echo "Phone app started. Opening http://localhost:8080 ..."
(sleep 2; open http://localhost:8080 2>/dev/null) &
.venv/bin/python tools/live_demo/server.py "$@"
