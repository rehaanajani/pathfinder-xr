#!/bin/bash
# Speaks tools/video/narration.txt with the Mac's system voice (System Settings > Accessibility > Read & Speak),
# one clip per scene, into <work>/audio/sNN.caf. "Giljabi" is spelled out so the voice says it right.
set -e
work="$1"; mkdir -p "$work/audio"
i=0
while IFS= read -r line; do
  spoken="${line//Giljabi/Gil-jah-bee}"
  say ${RATE:+-r $RATE} --data-format=LEF32@48000 -o "$work/audio/$(printf 's%02d' $i).caf" "$spoken"
  i=$((i + 1))
done < "$(dirname "$0")/narration.txt"
echo "$i clips in $work/audio"
