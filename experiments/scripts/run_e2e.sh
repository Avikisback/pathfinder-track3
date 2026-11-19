#!/bin/bash
source /mnt/e/clipping-pipeline/xr_track3/experiments/scripts/env.sh
mkdir -p logs
$PY scripts/32_e2e.py train --scheme loeo > logs/e2e_loeo.log 2>&1 &
$PY scripts/32_e2e.py train --scheme loro > logs/e2e_loro.log 2>&1 &
wait
$PY scripts/32_e2e.py train --scheme loso > logs/e2e_loso.log 2>&1
echo E2E_ALL_DONE
