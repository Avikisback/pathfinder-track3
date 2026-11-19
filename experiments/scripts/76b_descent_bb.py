"""Descent prior on the unseen airframe (Blackbird, cache v17) and on the TUM-VI walks (gate must not fire)."""
import sys, os, pickle, importlib
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m75 = importlib.import_module("75_blackbird")       # patches load_room for bb_ files, sets rwe.OUT to v17
import numpy as np
from ev import *
import rw_evidence as rwe
m76 = importlib.import_module("76_descent")
V13 = EXP + "/cache/v13"

if __name__ == "__main__":
    UN = m75.UN
    names = [f"bb_{x}" for x in sorted(d for d in os.listdir(UN) if d.split("_")[0] in ("train", "test"))]
    rooms = {n: m75.load2(n) for n in SEQS}
    fit = m57.spec_fit([(n, pickle.load(open(f"{V13}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl", "rb"))) for n in SEQS for k in rwe.KEYS_KNOWN], rooms)
    rows = {}
    for n in names:
        for k in m75.KS:
            f = f"{m75.OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl"
            if not os.path.exists(f):
                continue
            rows[(n, *k)] = m76.score(n, k, pickle.load(open(f, "rb")), fit, file_for)
    m76.show(rows, "Blackbird (unseen airframe), clean + full")
    for n_, r in sorted(rows.items()):
        print(f"  {n_[0]:28s} {n_[1]:5s} drone {r['drone']} on_end {r['on_end']:.2f} truth drop {r['gt_drop']:+.2f} | blind_rel hold {r[(0.0, 3.0)].get('blind_rel', np.nan):.3f} descent {r[(0.25, 3.0)].get('blind_rel', np.nan):.3f}")
    pickle.dump(rows, open(f"{RES}/descent_bb.pkl", "wb"))
