"""Heading drift of the output attitude: the gyro-bias curve is anchored only by a weak amplitude prior (0.1 rad/s), so the solve
can turn the heading to shrink velocity. On TUM rooms (gyro already compensated) the heading then wanders 40-150 deg over a file.
Test a realistic anchor (0.01 and 0.003 rad/s) on the shipped entry (lever + 0.2 Hz for drones): position (same folds and filter)
and attitude (first-pose aligned RMS, heading range), EuRoC and TUM, known and hostile shapes.
Usage: 78_gyroprior.py solve | score"""
import sys, os, pickle, time, importlib
STAGE = sys.argv[1] if len(sys.argv) > 1 else "score"
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m77 = importlib.import_module("77_lever")
import multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import rw_evidence as rwe
import ev
from ev import *

V13, V18 = EXP + "/cache/v13", EXP + "/cache/v18"
VARS = {"lev2_g01": 0.01, "lev2_g003": 0.003, "lev2_g1": 0.1}  # lev2_g1 = shipped (handheld baseline)
KINDS = rwe.KEYS


def fname(n, k, var):
    return f"{V18}/{n}_{k[0]}_{k[1]}_{k[2]}_{var}.pkl"


def work(job):
    n, k, var, consts = job
    if os.path.exists(fname(n, k, var)):
        return job[:3]
    m77.VARIANT.clear()
    m77.VARIANT.update(m77.VARS["lev2"])
    m77.VARIANT.update({kk: consts[kk] for kk in ("k", "r", "d")})
    t, imu, room = m77.m72.file_for2(n, k)
    P = prepare(t, imu)
    m77.m73.CUR["b0"] = m77.m73.b0_rest(P["t"], P["imu"], P["ok"], n)
    m41.calib_excl = m77.m73.calib_patch
    rwe.m41.calib_excl = m77.m73.calib_patch
    rwe.VARIANTS[var] = dict(level="L8", pen="rw", eb_passes=1, kappa=1.0, amp_g_sig=VARS[var])
    rwe.OUT = V18
    rwe.work((n, k, var))
    return job[:3]


def loader_for(var):
    def fn(n, k):
        if var == "base":
            f = fname(n, k, "lev2") if isinstance(n, str) else fname(n, k, "lev2_g1")
        else:
            f = fname(n, k, var)
        return pickle.load(open(f, "rb"))
    return fn


def att(n, k, d):
    t, imu, room = m77.m72.file_for2(n, k)
    P = prepare(t, imu)
    keep = P["ids"] != -1
    te = P["t_ns"][keep] * 1e-9
    q = d["q"].astype(np.float64)[keep]
    gt_t = room["gt_t_ns"] * 1e-9
    gi = np.clip(np.searchsorted(te, gt_t), 1, len(te) - 1)
    gi = gi - (gt_t - te[gi - 1] < te[gi] - gt_t)
    m = (np.abs(te[gi] - gt_t) < 0.01) & (te[gi] <= P["t_fail_ns"] * 1e-9)
    qg = Rot.from_quat(room["gt_q"][m])
    qe = Rot.from_quat(q[gi[m]][:, [1, 2, 3, 0]])
    W = qg[0] * qe[0].inv()
    e = (qg.inv() * (W * qe)).magnitude() * 180 / np.pi
    D = (qg * qe.inv()).as_matrix()
    yaw = np.degrees(np.unwrap(np.arctan2(D[:, 1, 0] - D[:, 0, 1], D[:, 0, 0] + D[:, 1, 1])))
    return float(np.sqrt((e ** 2).mean())), float(yaw.max() - yaw.min())


def att_job(a):
    n, k, var = a
    return (n, k, var), att(n, k, loader_for(var)(n, k))


if __name__ == "__main__":
    if STAGE == "solve":
        C = pickle.load(open(f"{V18}/constants.pkl", "rb"))
        tum_c = C["MH01"]  # unused on handheld files (no rotors), any fold's constants
        jobs = [(n, k, var, C.get(n, tum_c)) for var in VARS for n in SEQS + DEV_ROOMS for k in KINDS if not (var == "lev2_g1" and isinstance(n, str))]
        t0 = time.time()
        with mp.get_context("fork").Pool(20) as p:
            for i, j in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
                if i % 40 == 0:
                    print(i, len(jobs), j, f"{time.time()-t0:.0f}s", flush=True)
    ev.KNOWN = rwe.KEYS_KNOWN
    for var in ("base", "lev2_g01", "lev2_g003"):
        L = loader_for(var)
        for names, folds, pl in ((SEQS, EU_FOLDS, "EuRoC"), (DEV_ROOMS, TU_FOLDS, "TUM")):
            r = evaluate(names, folds, KINDS, L, cfg=m77.CFG2, procs=12)
            s = summary(r, "final")
            with mp.get_context("fork").Pool(12) as p:
                A = dict(p.map(att_job, [(n, k, var) for n in names for k in KINDS]))
            ak = [A[(n, k, var)] for n in names for k in rwe.KEYS_KNOWN]
            ah = [A[(n, k, var)] for n in names for k in rwe.KEYS_NEW]
            print(f"{pl:5s} {var:9s} ATE " + " ".join(f"{a} {b:.2f}" for a, b in s.items())
                  + f" | blind {np.nanmean([r[k]['final'].get('blind_rel', np.nan) for k in r]):.3f}"
                  + f" | attitude known {np.mean([x[0] for x in ak]):.1f} deg (heading range {np.mean([x[1] for x in ak]):.0f}), hostile {np.mean([x[0] for x in ah]):.1f} deg", flush=True)
