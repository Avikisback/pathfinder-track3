"""Lever-arm and rotor-rate correction of the drone drag target, inside the entry's solver (L8rw_eb1 + CFG2 + rot fix).

Research notes (drift_immune_velocity.md): with truth velocity, subtracting the IMU lever-arm acceleration (w' x r + w x (w x r))
and a rotor-rate drag term (w x d) from the in-plane specific force raised drag/truth coherence above 1/4 Hz and cut the
1/64-1/4 Hz drag-velocity error 0.312 -> 0.256 m/s. Untested inside the solver until now.

Constants k, r, d are fitted per fold on the OTHER EuRoC environments (truth used for this fit only, like k always was), with the
researcher's own fitting code. Variants (all drones; handheld files are untouched by construction):
  base  : no correction, fc 0.1 Hz          (only for the 3 robust kinds missing from cache v15; the rest is v15)
  lev   : correction, fold k/r/d, fc 0.1 Hz
  lev2  : correction, fold k/r/d, fc 0.2 Hz  (notes: the crossover may move up once the channel is cleaner)
  base2 : no correction, fc 0.2 Hz           (separates the fc change from the correction)
Usage: 77_lever.py solve | score
"""
import sys, os, pickle, time
OUT = "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v18"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "score"
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import importlib, multiprocessing as mp
import numpy as np
import rw_evidence as rwe
import ev
from ev import *
m73 = importlib.import_module("73_rotfix")
m72 = m73.m72
import robust_shapes as rs
from phys import rotors_on, _lpf
from scipy.ndimage import uniform_filter1d

m46 = rwe.m46
V13, V15 = EXP + "/cache/v13", EXP + "/cache/v15"
CFG2 = m72.CFG2
RK = [(k, "x", 1) for k in rs.KINDS]
KINDS = rwe.KEYS + RK
DM = "/mnt/e/clipping-pipeline/xr_track3/experiments/research_scripts/scripts/drag_models.py"


def fold_constants():
    src = open(DM).read().split("\ndata = {n: load(n)")[0].replace('ROOT = "E:/', 'ROOT = "/mnt/e/')
    ns = {}
    exec(src, ns)
    data = {n: ns["load"](n) for n in ns["SEQ"]}
    out = {}
    for test, train in EU_FOLDS:
        th, names = ns["fit"]([data[n] for n in train], dict(lever=True, rate=True))
        th0, _ = ns["fit"]([data[n] for n in train], dict())
        c = dict(k=float(th[0]), r=[float(x) for x in th[1:4]], d=[float(x) for x in th[4:7]], k_only=float(th0[0]))
        print(f"fold test {test}: k only {c['k_only']:.3f} | k {c['k']:.3f} r {np.round(c['r'], 3)} d {np.round(c['d'], 3)}", flush=True)
        for n in test:
            out[n] = c
    return out


VARIANT = {}


def set_target_v(self, k=m46.K_DRAG, fc=0.1):
    C = VARIANT
    k = C.get("k", k) if C.get("lever") else k
    fc = C.get("fc", fc)
    t, acc, ok = self.t, self.acc, self.ok
    fs = round(1.0 / np.median(np.diff(t)))
    idx = np.arange(self.n)
    a = np.stack([np.interp(idx, idx[ok], acc[ok, i]) for i in range(3)], 1)
    on = rotors_on(a, fs)
    last = self.last
    self.drone = on[: last + 1].mean() > 0.3
    if not self.drone:
        self.vb_t = np.zeros((len(self.idx), 3))
        self.on_b = np.zeros(len(self.idx), bool)
        return
    ta = a[: last + 1][on[: last + 1]].mean(0)
    ta /= np.linalg.norm(ta)
    e1 = np.cross(ta, [0, 1, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ta, e1)
    E = np.stack([e1, e2], 1)
    if C.get("lever"):
        w = np.stack([np.interp(idx, idx[ok], self.gyro[ok, i]) for i in range(3)], 1) - self.b0
        nb = max(1, int(round(0.05 * fs)))
        sm = lambda x, n_: uniform_filter1d(x, n_, axis=0, mode="nearest")
        wl = sm(sm(w, nb), nb)
        wd = sm(sm(np.gradient(sm(w, 2 * nb), 1.0 / fs, axis=0), nb), nb)
        r, d = np.array(C["r"]), np.array(C["d"])
        a = a - (np.cross(wd, r) + np.cross(wl, np.cross(wl, r))) - np.cross(wl, d)
    fp = _lpf(a, fc, fs) @ E
    off = fp[: last + 1][on[: last + 1]].mean(0)
    vp = -(fp - off) / k
    vb = vp @ E.T
    self.vb_t = vb[self.mid]
    self.on_b = on[self.mid].astype(bool)
    self.E = E


m46.DragCal.set_target = set_target_v
VARS = {"base": dict(), "lev": dict(lever=True), "lev2": dict(lever=True, fc=0.2), "base2": dict(fc=0.2)}


def fname(n, k, var):
    return f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_{var}.pkl"


def work(job):
    n, k, var, consts = job
    if os.path.exists(fname(n, k, var)):
        return job[:3], 0.0
    t0 = time.time()
    VARIANT.clear()
    VARIANT.update(VARS[var])
    VARIANT.update({kk: consts[kk] for kk in ("k", "r", "d")})
    t, imu, room = m72.file_for2(n, k)
    P = prepare(t, imu)
    m73.CUR["b0"] = m73.b0_rest(P["t"], P["imu"], P["ok"], n)
    m41.calib_excl = m73.calib_patch
    rwe.m41.calib_excl = m73.calib_patch
    rwe.VARIANTS[var] = rwe.VARIANTS["L8rw_eb1"]
    rwe.work((n, k, var))
    return job[:3], time.time() - t0


def loader_for(var):
    def fn(n, k):
        cands = [fname(n, k, var)] + ([f"{V15}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl"] if var == "base" else [])
        for f in cands:
            if os.path.exists(f):
                return pickle.load(open(f, "rb"))
        raise FileNotFoundError((n, k, var))
    return fn


if __name__ == "__main__":
    stage = STAGE
    os.makedirs(OUT, exist_ok=True)
    if stage == "solve":
        C = fold_constants()
        pickle.dump(C, open(f"{OUT}/constants.pkl", "wb"))
        jobs = []
        for var in VARS:
            for n in SEQS:
                for k in KINDS:
                    if var == "base" and os.path.exists(f"{V15}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl"):
                        continue
                    jobs.append((n, k, var, C[n]))
        # long files first
        dur = {n: len(load_room(n)["t_ns"]) for n in SEQS}
        jobs.sort(key=lambda j: -dur[j[0]])
        print(len(jobs), "solves", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(20) as p:
            for i, (job, dt) in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
                if i % 20 == 0:
                    print(i, job, f"{dt:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
        print("solved", time.time() - t0, flush=True)
    if stage in ("solve", "score"):
        ev.KNOWN = rwe.KEYS_KNOWN
        ev.NAMES = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine") + tuple(rs.KINDS)
        res = {}
        for var in VARS:
            r = evaluate(SEQS, EU_FOLDS, KINDS, loader_for(var), cfg=CFG2, procs=12)
            res[var] = r
            s = summary(r, "final")
            known = np.mean([s[x] for x in ("clean", "mild", "full")])
            print(f"{var:6s} final " + " ".join(f"{a} {b:.2f}" for a, b in s.items()) + f" | mean known {known:.3f} all {np.mean(list(s.values())):.3f}", flush=True)
            s2 = summary(r, "nocheck")
            print(f"{'':6s} nocheck " + " ".join(f"{a} {b:.2f}" for a, b in s2.items()), flush=True)
            print(f"{'':6s} blind_rel {np.nanmean([r[k]['final'].get('blind_rel', np.nan) for k in r]):.3f} rpe {np.nanmean([r[k]['final']['rpe_t'] for k in r]):.3f}"
                  f" | per seq clean " + " ".join(f"{k[0]} {r[k]['final']['ate_se3']:.2f}" for k in sorted(r) if k[1] == 'clean'), flush=True)
        pickle.dump(res, open(f"{RES}/lever.pkl", "wb"))
        print("done", flush=True)
