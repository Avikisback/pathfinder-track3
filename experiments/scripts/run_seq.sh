#!/bin/bash
source /mnt/e/clipping-pipeline/xr_track3/experiments/scripts/env.sh
$PY scripts/50_seqnet.py --scheme loro > logs/seq_loro.log 2>&1
$PY scripts/50_seqnet.py --scheme loso > logs/seq_loso.log 2>&1
echo SEQ_ALL_DONE
