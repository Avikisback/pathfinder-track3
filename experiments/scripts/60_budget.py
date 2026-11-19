"""Error budget. Which single error, if it vanished, would help most? Each row replaces one part of the frozen entry's estimate
with ground truth. This is analysis only: it tells us where to dig, it is not a method.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from scipy.spatial.transform import Slerp


def gt_at(room, P, tk):
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    h = 0.05
    f = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    v = (f(tk + h) - f(tk - h)) / (2 * h)
    idx = np.clip(np.searchsorted(gt_t, tk), 1, len(gt_t) - 1)
    # inside mocap holes the straight line across the hole is used: it still integrates to the true position at the far side
    good = (tk - h >= gt_t[0]) & (tk + h <= gt_t[-1])
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(tk, gt_t[0], gt_t[-1]))
    return v, good, Rg


def lowpass(e, fc):
    X, f = eb.spec(e)
    X[f > fc] = 0
    return np.fft.irfft(X, n=2 * len(e), axis=0)[: len(e)]


FCS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)


def work(a):
    n, k, fit = a
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    d = m57.load(n, *k, LVL)
    q = d["q"].astype(np.float64)
    sc = PosScorer(P, room, q)
    vf = m58.final_v(d, fit, *CFG)
    tk = d["tk"]
    vt, good, Rg = gt_at(room, P, tk)
    idx = np.clip(np.searchsorted(P["t"], tk), 0, len(P["t"]) - 1)
    Re = R.from_quat(q[idx][:, [1, 2, 3, 0]])
    Q = Rg * Re.inv()
    Q0 = Q[good].mean()
    dQ = Q * Q0.inv()
    M = dQ.as_matrix()
    dpsi = np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1])
    vs = Q0.apply(vf)
    v2 = np.where(good[:, None], vt, vs)
    c, s = np.cos(dpsi), np.sin(dpsi)
    vy = np.stack([c * vs[:, 0] - s * vs[:, 1], s * vs[:, 0] + c * vs[:, 1], vs[:, 2]], 1)
    va = dQ.apply(vs)
    hv, ht = vs[:, :2], v2[:, :2]
    nv, nt = np.linalg.norm(hv, axis=1, keepdims=True), np.linalg.norm(ht, axis=1, keepdims=True)
    uv, ut = hv / np.maximum(nv, 1e-6), ht / np.maximum(nt, 1e-6)
    tr = lambda v: sc(m57.track(P, d, v))
    o = dict(stationary=sc(np.zeros((len(P["t"]), 3))), base=tr(vs), yaw=tr(vy), att=tr(va))
    o["speed"] = tr(np.c_[uv * nt, vs[:, 2]])
    o["dir"] = tr(np.c_[ut * nv, vs[:, 2]])
    o["z"] = tr(np.c_[vs[:, :2], v2[:, 2]])
    o["xy"] = tr(np.c_[v2[:, :2], vs[:, 2]])
    o["yaw_speed"] = tr(np.c_[(vy[:, :2] / np.maximum(np.linalg.norm(vy[:, :2], axis=1, keepdims=True), 1e-6)) * nt, vs[:, 2]])
    e = v2 - vs
    for fc in FCS:
        o[f"low{1/fc:.0f}"] = tr(vs + lowpass(e, fc))        # truth below fc
    o["att_rms"] = float(np.degrees(np.sqrt(np.mean(dpsi[good] ** 2))))
    W = int(20 / DT)
    dd = np.unwrap(dpsi)
    o["yaw_rate_err"] = float(np.sqrt(np.mean(((dd[W:] - dd[:-W]) / 20.0) ** 2)))
    return (n, *k), o


def yaw_prior_test(n):
    """If yaw rate over a window is taken to be zero, or its median zero, how wrong is that? Truth attitude, clean gyro."""
    room = load_room(n)
    t = (room["t_ns"] - room["t_ns"][0]) * 1e-9
    gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(t, gt_t[0], gt_t[-1]))
    wup = Rg.apply(room["imu"][:, :3])[:, 2]
    wup = wup[::4]
    out = {}
    for W in (5, 10, 20, 40):
        w = int(W * 50)
        hop = w // 2
        med = [np.median(wup[i:i + w]) for i in range(0, len(wup) - w, hop)]
        mean = [np.mean(wup[i:i + w]) for i in range(0, len(wup) - w, hop)]
        out[W] = (float(np.sqrt(np.mean(np.square(med)))), float(np.sqrt(np.mean(np.square(mean)))))
    yaw = np.cumsum(wup) * 0.02
    return n, out, float(np.ptp(yaw)), float(np.degrees(np.std(yaw)))


if __name__ == "__main__":
    t0 = time.time()
    rows = {}
    for names, folds, title in ((SEQS, EU_FOLDS, "EuRoC"), (DEV_ROOMS, TU_FOLDS, "TUM")):
        rooms = {n: load_room(n) for n in names}
        F = fits(folds, lambda n, k: m57.load(n, *k, LVL), rooms)
        jobs = [(n, k, F[n]) for n in names for k in ALLK]
        res = {}
        with mp.get_context("fork").Pool(14) as p:
            for key, o in p.imap_unordered(work, jobs, chunksize=4):
                res[key] = o
        rows[title] = res
        print(f"\n=== {title}: mean ATE SE3 when one part of the estimate is replaced by truth ({time.time()-t0:.0f}s)")
        cols = ["stationary", "base", "yaw", "att", "speed", "dir", "yaw_speed", "z", "xy"] + [f"low{1/fc:.0f}" for fc in FCS]
        print(f"{'':11s}" + " ".join(f"{c:>9s}" for c in cols) + "   yaw rms deg, yaw rate err")
        for kind in NAMES:
            ks = [k for k in res if k[1] == kind]
            print(f"{kind:11s}" + " ".join(f"{np.nanmean([res[k][c]['ate_se3'] for k in ks]):9.2f}" for c in cols) + f"   {np.mean([res[k]['att_rms'] for k in ks]):6.1f} {np.mean([res[k]['yaw_rate_err'] for k in ks]):.4f}", flush=True)
        print("RPE 1 s")
        for kind in NAMES:
            ks = [k for k in res if k[1] == kind]
            print(f"{kind:11s}" + " ".join(f"{np.nanmean([res[k][c]['rpe_t'] for k in ks]):9.2f}" for c in cols), flush=True)
    pickle.dump(rows, open(f"{RES}/budget.pkl", "wb"))
    print("\n=== How wrong is 'yaw rate is zero over a window'? rms rad/s for median, mean, per window length; then total yaw travelled rad, yaw std deg")
    for n in SEQS + DEV_ROOMS:
        n_, out, ptp, sd = yaw_prior_test(n)
        print(f"{str(n_):5s} " + " ".join(f"W{W}: med {out[W][0]:.3f} mean {out[W][1]:.3f}" for W in out) + f"  yaw range {ptp:6.1f} rad std {sd:6.1f} deg", flush=True)
