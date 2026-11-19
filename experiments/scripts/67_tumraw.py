"""Frozen entry on handheld: how often does the self-check fire, and what do the unfiltered tracks score?"""
import sys, pickle
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
L = lambda n, k: m57.load(n, *k, LVL)
for names, folds, title in ((DEV_ROOMS, TU_FOLDS, "TUM"),):
    r = evaluate(names, folds, ALLK, L)
    table(r, f"{title} frozen L15")
    for kd in NAMES:
        ks = [k for k in r if k[1] == kd]
        rv = [r[k]["rvs"] for k in ks]
        print(f"  {kd:10s} slow rms min {min(rv):.2f} median {np.median(rv):.2f} max {max(rv):.2f} share above 0.4: {np.mean(np.array(rv)>0.4):.2f}")
    pickle.dump(r, open(f"{RES}/frozen_tum_detail.pkl", "wb"))
