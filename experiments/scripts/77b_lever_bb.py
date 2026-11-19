"""Lever fix + 0.2 Hz drag filter on the unseen airframe (Blackbird). EuRoC constants (all 11 flights) are applied unchanged,
as the shipped entry would. Only files flagged as drones change; the others are skipped."""
import sys, os, pickle, importlib
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m75 = importlib.import_module("75_blackbird")       # patches load_room for bb_ files; rwe.OUT = v17
m77 = importlib.import_module("77_lever")
import numpy as np
from ev import *
import rw_evidence as rwe
V13 = EXP + "/cache/v13"

if __name__ == "__main__":
    C = pickle.load(open(f"{m77.OUT}/constants.pkl", "rb"))
    # all-flight constants: average of the three folds (they agree to 1-3 mm)
    allc = {k: np.mean([C[n][k] for n in ("MH01", "V101", "V201")], 0) for k in ("k", "r", "d")}
    print("constants", {k: np.round(v, 3) for k, v in allc.items()})
    names = [f"bb_{x}" for x in sorted(d for d in os.listdir(m75.UN) if d.split("_")[0] in ("train", "test"))]
    rooms = {n: m75.load2(n) for n in SEQS}
    fit_base = m57.spec_fit([(n, pickle.load(open(f"{V13}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl", "rb"))) for n in SEQS for k in rwe.KEYS_KNOWN], rooms)
    fit_lev = m57.spec_fit([(n, pickle.load(open(m77.fname(n, k, "lev2"), "rb"))) for n in SEQS for k in rwe.KEYS_KNOWN], rooms)
    rows = []
    for n in names:
        for k in m75.KS:
            fb = f"{m75.OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl"
            d0 = pickle.load(open(fb, "rb"))
            if not d0["drone"]:
                continue
            fl = f"{m75.OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_lev2.pkl"
            if not os.path.exists(fl):
                m77.VARIANT.clear()
                m77.VARIANT.update(m77.VARS["lev2"])
                m77.VARIANT.update(allc)
                rwe.VARIANTS["lev2"] = rwe.VARIANTS["L8rw_eb1"]
                rwe.work((n, k, "lev2"))
            d1 = pickle.load(open(fl, "rb"))
            t, imu, room = file_for(n, k)
            P = prepare(t, imu)
            o = {}
            for nm, d, fit in (("base", d0, fit_base), ("lev2", d1, fit_lev)):
                sc = PosScorer(P, room, d["q"].astype(np.float64))
                o[nm] = sc(m57.track(P, d, m58.final_v(d, fit, *m77.CFG2)))
            o["stat"] = sc(np.zeros((len(P["t"]), 3)))
            rows.append(o)
            print(f"{n:26s} {k[0]:5s} stationary {o['stat']['ate_se3']:.2f} | base {o['base']['ate_se3']:.2f} lev2 {o['lev2']['ate_se3']:.2f} | rpe {o['base']['rpe_t']:.3f} {o['lev2']['rpe_t']:.3f}", flush=True)
    print(f"MEAN over {len(rows)} drone files: stationary {np.mean([o['stat']['ate_se3'] for o in rows]):.3f} base {np.mean([o['base']['ate_se3'] for o in rows]):.3f} lev2 {np.mean([o['lev2']['ate_se3'] for o in rows]):.3f}")
