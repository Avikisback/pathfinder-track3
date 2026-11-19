"""Freeze the evidence-selected drift model (random-walk penalty on 8 s knots, drift rate chosen per file by evidence) with
self-check thresholds picked on the earlier shapes (drones 1.0 m/s, handheld 0.5 m/s, same rule as 58: known + 0.25 x hostile).
Then test it, and the old frozen entry, on five degradations nobody has tuned on (robust_shapes.py)."""
import sys, os, pickle, time, multiprocessing as mp
OUT = "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v13"
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import rw_evidence as rwe
import ev
from ev import *
import robust_shapes as rs

CFG2 = (1.0, 0.5, 8.0, 1.0)
RK = [(k, "x", 1) for k in rs.KINDS]
_ff = ev.file_for


def file_for2(n, k):
    if k[0] in rs.KINDS:
        room = load_room(n)
        t, imu = rs.degrade_robust(room, k[2], k[0])
        return t, imu, room
    return _ff(n, k)


ev.file_for = file_for2
rwe.file_for = file_for2


def solve_old(job):
    n, k = job
    f = f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_old.pkl"
    if not os.path.exists(f):
        t, imu, room = file_for2(n, k)
        P = prepare(t, imu)
        q, v, tb, last, info = m48.solve_level(P, m41.calib_excl(n), m48.LEVELS["L15"])
        pickle.dump(to_cache(q, v, tb, info["drone"]), open(f, "wb"))
    return job


def solve_new(job):
    return rwe.work((job[0], job[1], "L8rw_eb1"))


if __name__ == "__main__":
    t0 = time.time()
    names = SEQS + DEV_ROOMS
    KN = rwe.KEYS_KNOWN
    jobs = [(n, k) for n in names for k in RK + KN]
    with mp.get_context("fork").Pool(16) as p:
        p.map(solve_old, [j for j in jobs if j[1] in RK], chunksize=1)
        p.map(solve_new, jobs, chunksize=1)
    print(f"solved {time.time()-t0:.0f}s", flush=True)
    old = lambda n, k: m57.load(n, *k, LVL) if k in KN else pickle.load(open(f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_old.pkl", "rb"))
    new = lambda n, k: pickle.load(open(rwe.fname(n, k, "L8rw_eb1"), "rb"))
    ev.NAMES = tuple(rs.KINDS)
    res = {}
    for nm, L, cfg in (("old frozen", old, CFG), ("evidence drift", new, CFG2)):
        for names_, folds, pl in ((SEQS, EU_FOLDS, "EuRoC"), (DEV_ROOMS, TU_FOLDS, "TUM")):
            r = evaluate(names_, folds, RK, L, cfg=cfg)
            res[(nm, pl)] = r
            print(f"{pl:5s} {nm:15s} final " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, 'final').items())
                  + " | stationary " + " ".join(f"{a} {np.mean([r[k]['stationary']['ate_se3'] for k in r if k[1]==a]):.2f}" for a in rs.KINDS), flush=True)
            print(f"{'':21s} rpe   " + " ".join(f"{a} {np.mean([r[k]['final']['rpe_t'] for k in r if k[1]==a]):.2f}" for a in rs.KINDS)
                  + " | beats stationary " + " ".join(f"{np.mean([r[k]['final']['ate_se3'] < r[k]['stationary']['ate_se3'] for k in r if k[1]==a]):.2f}" for a in rs.KINDS), flush=True)
    pickle.dump(res, open(f"{RES}/freeze2_robust.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
