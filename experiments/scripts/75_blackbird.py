"""Unseen airframe: Blackbird quadrotor (100 Hz, different drone, yaw-forward flights). Band gains from EuRoC. Nothing tuned here.
start and end (in the mocap room), so ATE here measures whether the track comes back to the right place after a long walk.

Compares the old frozen entry and the new one (evidence drift, thresholds 0.4 -> 0.5 handheld), with band gains from TUM rooms 1-5.
Also times the solver per second of IMU (Python, so only a relative number).
"""
import sys, os, pickle, time, multiprocessing as mp
OUT = "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v17"
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common, pipeline, phys, ev
import rw_evidence as rwe
from ev import *

UN = "/mnt/e/clipping-pipeline/xr_track3/data/unseen/blackbird/conv"
V13 = EXP + "/cache/v13"
_load = common.load_room


def load2(n):
    if isinstance(n, str) and n.startswith("bb_"):
        c = f"{CACHE}/{n}.npz"
        if os.path.exists(c):
            d = np.load(c)
            return {k: d[k] for k in d.files}
        t, imu = common._read_csv_ns(f"{UN}/{n[3:]}/imu0/data.csv")
        gt_t, gt = common._read_csv_ns(f"{UN}/{n[3:]}/gt.csv")
        out = dict(t_ns=t, imu=imu, gt_t_ns=gt_t, gt_p=gt[:, :3], gt_q=gt[:, [4, 5, 6, 3]])
        np.savez(c, **out)
        return out
    return _load(n)


for mod in (common, pipeline, phys, ev, rwe, m57, m58, m41, m48):
    if hasattr(mod, "load_room"):
        mod.load_room = load2
_cal = m41.calib_excl
m41.calib_excl = lambda n: np.zeros(3) if (isinstance(n, str) and n.startswith("bb_")) else _cal(n)
rwe.m41.calib_excl = m41.calib_excl
ev.file_for.__globals__["load_room"] = load2
m57.get_file.__globals__["load_room"] = load2

CFG_NEW = (1.0, 0.5, 8.0, 1.0)
KS = [("clean", "delete", 1), ("full", "zero", 1)]


def solve_old(job):
    n, k = job
    f = f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_old.pkl"
    if not os.path.exists(f):
        t, imu, room = file_for(n, k)
        P = prepare(t, imu)
        t0 = time.time()
        q, v, tb, last, info = m48.solve_level(P, m41.calib_excl(n), m48.LEVELS["L15"])
        d = to_cache(q, v, tb, info["drone"])
        d["sec"] = time.time() - t0
        pickle.dump(d, open(f, "wb"))
    return job


def solve_new(job):
    n, k = job
    t0 = time.time()
    rwe.work((n, k, "L8rw_eb1"))
    return job, time.time() - t0


if __name__ == "__main__":
    names = sorted(d for d in os.listdir(UN) if d.split("_")[0] in ("train", "test"))
    names = [f"bb_{x}" for x in names]
    print("sequences:", names, flush=True)
    for n in names:
        load2(n)
        for k in KS:
            file_for(n, k)
    jobs = [(n, k) for n in names for k in KS]
    t0 = time.time()
    with mp.get_context("fork").Pool(12) as p:
        p.map(solve_old, jobs, chunksize=1)
        tnew = dict(p.map(solve_new, jobs, chunksize=1))
    print(f"solved {time.time()-t0:.0f}s", flush=True)
    rooms = {n: load2(n) for n in SEQS}
    fit_old = m57.spec_fit([(n, m57.load(n, *k, LVL)) for n in SEQS for k in rwe.KEYS_KNOWN], rooms)
    fit_new = m57.spec_fit([(n, pickle.load(open(f"{V13}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl", "rb"))) for n in SEQS for k in rwe.KEYS_KNOWN], rooms)
    rows = {}
    for n in names:
        room = load2(n)
        gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
        for k in KS:
            t, imu, _ = file_for(n, k)
            P = prepare(t, imu)
            dur = P["t"][P["last_good"]]
            o = {"dur": dur, "gt_cover": float(np.sum(np.diff(gt_t)[np.diff(gt_t) < 0.1]))}
            for nm, f, fit, cfg in (("old", f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_old.pkl", fit_old, CFG), ("new", rwe.fname(n, k, "L8rw_eb1"), fit_new, CFG_NEW)):
                d = pickle.load(open(f, "rb"))
                sc = PosScorer(P, room, d["q"].astype(np.float64))
                o["stationary"] = sc(np.zeros((len(P["t"]), 3)))["ate_se3"]
                o[nm] = sc(m57.track(P, d, m58.final_v(d, fit, *cfg)))["ate_se3"]
                o[nm + "_raw"] = sc(m57.track(P, d, d["v"] - d["v"].mean(0)))["ate_se3"]
                o[nm + "_check"] = m57.slow_rms(d)
            o["sec_new_per_s"] = tnew[(n, k)] / dur
            rows[(n, k[0])] = o
            print(f"{n:16s} {k[0]:5s} {dur:6.0f}s gt {o['gt_cover']:5.0f}s | stationary {o['stationary']:.2f} old {o['old']:.2f} new {o['new']:.2f} (raw {o['old_raw']:.2f}/{o['new_raw']:.2f}, check {o['old_check']:.2f}/{o['new_check']:.2f}) | python {o['sec_new_per_s']*1000:.0f} ms per s", flush=True)
    for kk in ("clean", "full"):
        r = [rows[x] for x in rows if x[1] == kk]
        print(f"MEAN {kk}: stationary {np.mean([x['stationary'] for x in r]):.2f} old {np.mean([x['old'] for x in r]):.2f} new {np.mean([x['new'] for x in r]):.2f} | beats stationary old {np.mean([x['old'] < x['stationary'] for x in r]):.2f} new {np.mean([x['new'] < x['stationary'] for x in r]):.2f}")
    pickle.dump(rows, open(f"{RES}/unseen_tumvi.pkl", "wb"))
