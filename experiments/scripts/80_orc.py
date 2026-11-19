"""Oracle for one error source in the drone drag target. The target subtracts the file mean of the in-plane specific force,
which also removes the drone's mean in-plane BODY velocity c (79_dragoffset.py: 0.04-0.37 m/s on EuRoC). Here the true c is
added back to the target inside the entry's solver (lever + 0.2 Hz, L8rw_eb1, CFG2, rot fix). This is a ceiling, not a
method: c comes from ground truth. Variants: lev2 (the entry, cache v18) and orc (entry + true c, cache v19).
The 'rot' kind is not meaningful for orc (c is in the original sensor axes).
Usage: 80_orc.py solve | score"""
import sys, os, pickle, time, importlib
STAGE = sys.argv[1] if len(sys.argv) > 1 else "score"
sys.argv = [sys.argv[0], "score"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
m77 = importlib.import_module("77_lever")
from ev import *
import ev
import common
from phys import rotors_on, _lpf
from scipy.spatial.transform import Rotation as Rot
rwe, m72, m73, rs, m46 = m77.rwe, m77.m72, m77.m73, m77.rs, m77.m46

V18 = EXP + "/cache/v18"
OUT = EXP + "/cache/v19"
KINDS = m77.KINDS


def true_c(name):
    r = common.load_room(name)
    t = r["t_ns"] * 1e-9
    fs = round(1.0 / np.median(np.diff(t)))
    gt_t = r["gt_t_ns"] * 1e-9
    k = (t >= gt_t[0]) & (t <= gt_t[-1])
    t, imu = t[k], r["imu"][k]
    p = np.stack([np.interp(t, gt_t, r["gt_p"][:, i]) for i in range(3)], 1)
    j = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
    vb = Rot.from_quat(r["gt_q"][j]).inv().apply(np.gradient(_lpf(p, 5.0, fs), t, axis=0))
    on = rotors_on(imu[:, 3:6], fs)
    return vb[on].mean(0)


def set_target_o(self, k=m46.K_DRAG, fc=0.1):
    m77.set_target_v(self, k, fc)
    c3 = m77.VARIANT.get("c3")
    if self.drone and c3 is not None:
        E = self.E
        self.vb_t = self.vb_t + (np.asarray(c3) @ E) @ E.T


m46.DragCal.set_target = set_target_o


def fname(n, k, var):
    return f"{OUT if var == 'orc' else V18}/{n}_{k[0]}_{k[1]}_{k[2]}_{var}.pkl"


def work(job):
    n, k, var, consts, c3 = job
    if os.path.exists(fname(n, k, var)):
        return job[:3], 0.0
    t0 = time.time()
    m77.OUT = OUT
    m77.VARIANT.clear()
    m77.VARIANT.update(dict(lever=True, fc=0.2, c3=c3))
    m77.VARIANT.update({kk: consts[kk] for kk in ("k", "r", "d")})
    t, imu, room = m72.file_for2(n, k)
    P = prepare(t, imu)
    m73.CUR["b0"] = m73.b0_rest(P["t"], P["imu"], P["ok"], n)
    m77.m41.calib_excl = m73.calib_patch
    rwe.m41.calib_excl = m73.calib_patch
    rwe.VARIANTS[var] = rwe.VARIANTS["L8rw_eb1"]
    rwe.OUT = OUT
    rwe.work((n, k, var))
    return job[:3], time.time() - t0


def loader_for(var):
    def fn(n, k):
        return pickle.load(open(fname(n, k, var), "rb"))
    return fn


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if STAGE == "solve":
        C = pickle.load(open(f"{V18}/constants.pkl", "rb"))
        c = {n: true_c(n) for n in SEQS}
        for n in SEQS:
            print(n, "true mean body velocity (IMU axes)", np.round(c[n], 3), flush=True)
        jobs = [(n, k, "orc", C[n], c[n]) for n in SEQS for k in KINDS]
        t0 = time.time()
        with mp.get_context("fork").Pool(20) as p:
            for i, (job, dt) in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
                if i % 30 == 0:
                    print(i, job, f"{dt:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
        print("solved", time.time() - t0, flush=True)
    ev.KNOWN = rwe.KEYS_KNOWN
    ev.NAMES = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine") + tuple(rs.KINDS)
    for var in ("lev2", "orc"):
        r = evaluate(SEQS, EU_FOLDS, KINDS, loader_for(var), cfg=m77.CFG2, procs=12)
        s = summary(r, "final")
        print(f"{var:5s} final " + " ".join(f"{a} {b:.2f}" for a, b in s.items()) + f" | known {np.mean([s[x] for x in ('clean','mild','full')]):.3f}", flush=True)
        s2 = summary(r, "nocheck")
        print(f"{'':5s} nocheck " + " ".join(f"{a} {b:.2f}" for a, b in s2.items()), flush=True)
        s3 = summary(r, "raw")
        print(f"{'':5s} raw " + " ".join(f"{a} {b:.2f}" for a, b in s3.items()), flush=True)
        print(f"{'':5s} per seq clean " + " ".join(f"{k[0]} {r[k]['final']['ate_se3']:.2f}" for k in sorted(r) if k[1] == 'clean')
              + f" | rpe {np.nanmean([r[k]['final']['rpe_t'] for k in r]):.3f}", flush=True)
    print("done", flush=True)
