#!/bin/bash
source /mnt/e/clipping-pipeline/xr_track3/experiments/scripts/env.sh
export OMP_NUM_THREADS=1
$PY scripts/54_seq4.py make_pool > logs/s4_pool.log 2>&1
export OMP_NUM_THREADS=2
$PY scripts/54_seq4.py train --scheme loeo > logs/s4_loeo.log 2>&1 &
$PY scripts/54_seq4.py train --scheme loro > logs/s4_loro.log 2>&1 &
wait
$PY scripts/54_seq4.py train --scheme loso > logs/s4_loso.log 2>&1
echo S4_ALL
