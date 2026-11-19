"""File-by-file difference between two score_bin.py runs. Usage: score_diff.py <labelA> <labelB> <set> [metric ...]"""
import sys, pickle
import numpy as np
A, B, SET = sys.argv[1:4]
keys = sys.argv[4:] or ["ate_se3", "att", "rpe_r", "blind_rel"]
D = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp"
a = pickle.load(open(f"{D}/score_{A}_{SET}.pkl", "rb")); b = pickle.load(open(f"{D}/score_{B}_{SET}.pkl", "rb"))
print(f"{'file':34s} " + " ".join(f"{k + ' ' + A + '>' + B:>26s}" for k in keys))
for n in sorted(a):
    if n in b and "error" not in a[n] and "error" not in b[n]:
        print(f"{n:34s} " + " ".join(f"{a[n].get(k, np.nan):12.3f} {b[n].get(k, np.nan):12.3f} " for k in keys))
