#!/usr/bin/env bash
# Keep this attached to a wsl.exe client: WSL shuts its VM down when none is attached.
source /mnt/e/clipping-pipeline/xr_track3/experiments/scripts/env.sh
cat > tmp/queue.txt <<'QUEUE'
--name eu2s --scheme loso --T 100 --center 1
--name tum --scheme loro --T 100 --center 1
--name tumc --scheme loro --T 100 --center 0
--name eu2s --scheme loeo --T 100 --center 1
--name eu2sc --scheme loso --T 100 --center 0
--name eu2s --scheme tum2euroc --T 100 --center 1
--name eu2s --scheme euroc2tum --T 100 --center 1
--name xd --scheme tum2euroc --T 100 --center 1 --yaw_aug 1 --no_up 1
--name xd --scheme euroc2tum --T 100 --center 1 --yaw_aug 1 --no_up 1
--name xd --scheme both --T 100 --center 1 --yaw_aug 1 --no_up 1 --per_room 1
--name eu2sy --scheme loso --T 100 --center 1 --yaw_aug 1
QUEUE
xargs -P 3 -L 1 sh -c 'echo "start $*"; $PY scripts/20_cross.py "$@" >> logs/cross.log 2>&1; echo "end $* rc $?"' _ < tmp/queue.txt
echo CROSS_DONE
