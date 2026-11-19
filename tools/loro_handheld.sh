#!/bin/bash
# Leave-one-room-out gate for a handheld network (results/track3/network_handheld.md).
#   bash tools/loro_handheld.sh body|world
# For each of TUM-VI rooms 1-4: train without it (two seeds) and report the window-level velocity error on that room
# and room5. body = tools/train_airio.py (cache $PFX_DATA/velcache_body), world = tools/train_velocity.py (gravity-
# aligned input, cache from `PFX_CACHE=$PFX_DATA/velcache_world python tools/train_velocity.py prep`).
# Each run picks its best epoch on a pool of the left-out room, room5, MH_04 and V1_03 (the same for both inputs).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATA=${PFX_DATA:-local_data}
ALL="euroc:MH_01_easy euroc:MH_02_easy euroc:MH_03_medium euroc:V1_01_easy euroc:V1_02_medium euroc:V2_01_easy euroc:V2_02_medium tumvi:room1 tumvi:room2 tumvi:room3 tumvi:room4"
mkdir -p runs/loro
for k in 1 2 3 4; do
  TR=$(echo $ALL | tr ' ' '\n' | grep -v "tumvi:room$k$" | tr '\n' ' ')
  VA="tumvi:room$k tumvi:room5 euroc:MH_04_difficult euroc:V1_03_difficult"
  for s in 0 1; do
    out=runs/loro/$1_room${k}_s$s
    if [ "$1" = body ]; then
      $PY tools/train_airio.py train --train $TR --val $VA --seed $s --out $out > $out.log 2>&1
      $PY tools/train_airio.py export --run $out --out $out/w.bin > /dev/null
    else
      PFX_CACHE=$DATA/velcache_world $PY tools/train_velocity.py train --train $TR --val $VA --seed $s --epochs 20 --threads 8 --out $out > $out.log 2>&1
      PFX_CACHE=$DATA/velcache_world $PY tools/train_velocity.py export --run $out --out $out/w.bin > /dev/null
    fi
    echo "$1 room$k seed $s: $(grep '^best' $out.log | grep room | tr '\n' ' ')"
  done
done
# End to end: room k with the networks that never saw it, e.g.
#   python tools/eval_organiser.py --script degrade_imu.py --tag loro_world_room1_s0_d42 --data tumvi:room1 -- --model runs/loro/world_room1_s0/w.bin
