#!/usr/bin/env bash
# Scores src/main.py with each switch set on its own, on the official sets (seed 42 both column orders, seeds 1-3).
# Usage: bash scripts/score_variants.sh PF_FIX_ROTOR PF_FIX_TAIL ...   (run from experiments/, after env.sh)
M=/mnt/e/clipping-pipeline/xr_track3/src/main.py
for var in "$@"; do
    echo "=== $var"
    for d in cpp_off cpp_off_s1 cpp_off_s2 cpp_off_s3; do
        env "$var=1" OFF_DIR=$d "$PY" scripts/score_off.py python3 "$M" -- "v_${var}_${d}" 2>&1 | grep "ATE" | cut -c1-150 | sed "s/^/$d /"
    done
done
echo "ALL VARIANTS DONE"
