"""Two cheap upgrades to test.

(a) A better self-check on drones. The frozen check (5 s average speed above 0.9 m/s means drift won) fires on some clean fast flights
    and costs about 0.1-0.2 m there. On drones a second witness exists: the drag velocity. Real slow motion shows in both, drift
    does not show in both the same way. Print how well each statistic separates the known schedules from the hostile shapes.
(b) The blind tail. Truth-state tests say velocity persistence beats hold at the end of drone flights, if the velocity is right. Sweep
    the tail's gain and time constant on the frozen output, per platform, other environments only for the choice.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from scipy.ndimage import uniform_filter1d
m46 = m57.m46
mf = importlib.import_module("69_fuse")


def track_tail(P, d, vf, gain, tau):
    tk = d["tk"]
    x = np.cumsum(vf, axis=0) * DT
    out = np.stack([np.interp(P["t"], tk, x[:, i]) for i in range(3)], 1)
    t_last = tk[-1]
    tail = P["t"] > t_last
    if tail.any():
        v0 = vf[-5:].mean(0)
        s = P["t"][tail] - t_last
        out[tail] = x[-1] + gain * v0 * (tau * (1 - np.exp(-s / tau)))[:, None]
    return out


GAINS = (0.0, 0.5, 1.0, 1.5)
TAUS = (0.35, 1.0, 2.0)


def work(a):
    n, k, fit = a
    d = m57.load(n, *k, LVL)
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    sc = PosScorer(P, room, d["q"].astype(np.float64))
    o = {}
    if d["drone"]:
        vd = mf.drag_world(n, k, d)
        sv = uniform_filter1d(d["v"] - d["v"].mean(0), 50, axis=0, mode="nearest")
        sd = uniform_filter1d(vd - vd.mean(0), 50, axis=0, mode="nearest")
        o["s1"] = m57.slow_rms(d)
        o["s2"] = float(np.sqrt(((sv - sd) ** 2).sum(1).mean()) / (np.sqrt((sd ** 2).sum(1).mean()) + 0.05))
        o["s3"] = float((sv[:, :2] * sd[:, :2]).sum() / np.sqrt((sv[:, :2] ** 2).sum() * (sd[:, :2] ** 2).sum() + 1e-9))
        o["s4"] = float(np.sqrt(((sv - sd) ** 2).sum(1).mean()))
    vf = m58.final_v(d, fit, *CFG)
    for g in GAINS:
        for tau in TAUS:
            o[(g, tau)] = sc(track_tail(P, d, vf, g, tau))["blind_rel"]
    o["stat"] = sc(np.zeros((len(P["t"]), 3)))["blind_rel"]
    return (n, *k), o


if __name__ == "__main__":
    t0 = time.time()
    kinds = KNOWN + UNSEEN + STEPS
    allr = {}
    for names, folds, title in ((SEQS, EU_FOLDS, "EuRoC"), (DEV_ROOMS, TU_FOLDS, "TUM")):
        rooms = {n: load_room(n) for n in names}
        F = fits(folds, lambda n, k: m57.load(n, *k, LVL), rooms)
        jobs = [(n, k, F[n]) for n in names for k in kinds]
        with mp.get_context("spawn").Pool(16) as p:
            res = dict(p.map(work, jobs, chunksize=2))
        allr[title] = res
        if title == "EuRoC":
            print("=== (a) self-check statistics on drones: min / median / max per drift shape")
            for s in ("s1", "s2", "s3", "s4"):
                for kd in NAMES:
                    v = [res[k][s] for k in res if k[1] == kd]
                    print(f"  {s} {kd:10s} {np.min(v):6.2f} {np.median(v):6.2f} {np.max(v):6.2f}")
        print(f"=== (b) {title} blind error relative to failure, mean over known schedules / hostile shapes; stationary {np.mean([res[k]['stat'] for k in res]):.3f}")
        for g in GAINS:
            for tau in TAUS:
                kn = np.mean([res[k][(g, tau)] for k in res if k[1] in ("clean", "mild", "full", "burst")])
                ho = np.mean([res[k][(g, tau)] for k in res if k[1] in ("steps", "randomwalk", "sine")])
                print(f"  gain {g:3.1f} tau {tau:4.2f}: known {kn:.3f} hostile {ho:.3f}", flush=True)
    pickle.dump(allr, open(f"{RES}/tail_check.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
