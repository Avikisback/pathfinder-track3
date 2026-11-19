#!/usr/bin/env bash
# Keep this attached to a wsl.exe client: WSL shuts its VM down when none is attached, which kills children.
source /mnt/e/clipping-pipeline/xr_track3/experiments/scripts/env.sh
job() { $PY scripts/10_train_vel.py --name "$1" "${@:2}" --steps 6000 >> logs/$1.log 2>&1; echo "$1 exit $?"; }
job center2s --T 100 --center 1 &
job causal2s --T 100 --center 0 &
job center2s_seg --T 100 --center 1 --seg 50 &
wait
job causal2s_seg --T 100 --center 0 --seg 50 &
job center4s --T 200 --center 1 &
job center1s --T 50 --center 1 &
wait
echo H1_DONE
