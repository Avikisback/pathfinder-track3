#!/usr/bin/env bash
# Score one trajectory with evo. Position uses SE(3) alignment (-a) because the deck gives no
# flags; change this as soon as the organisers publish their exact commands.
#
# Attitude is lined up on the first pose (--align_origin), not with -a. With -a the rotation is
# solved from positions, which means nothing when positions are nearly stationary.
#
# RPE delta is in ground-truth frames: 120 = 1 s on TUM-VI mocap, use RPE_FRAMES=200 for EuRoC.
# Usage: tools/score.sh <ground_truth.tum> <estimated_trajectory.txt>
set -u
cd "$(dirname "$0")/.."
EVO=.venv-wsl/bin
GT=$1
EST=$2
F=${RPE_FRAMES:-120}
rmse() { grep -E '^\s*rmse' | awk '{printf "%.3f", $2}'; }
ape_t=$($EVO/evo_ape tum "$GT" "$EST" -a --t_max_diff 0.01 --no_warnings 2>&1 | rmse)
ape_r=$($EVO/evo_ape tum "$GT" "$EST" --align_origin --t_max_diff 0.01 -r angle_deg --no_warnings 2>&1 | rmse)
rpe_t=$($EVO/evo_rpe tum "$GT" "$EST" -a --t_max_diff 0.01 -r trans_part --delta "$F" --delta_unit f --no_warnings 2>&1 | rmse)
rpe_r=$($EVO/evo_rpe tum "$GT" "$EST" -a --t_max_diff 0.01 -r angle_deg --delta "$F" --delta_unit f --no_warnings 2>&1 | rmse)
printf 'ATE %s m | attitude %s deg | RPE(1s) %s m, %s deg\n' "${ape_t:-ERR}" "${ape_r:-ERR}" "${rpe_t:-ERR}" "${rpe_r:-ERR}"
