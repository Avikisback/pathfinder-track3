"""How predictable are the next 5 s at all? Uses ground-truth state at the moment of failure, so this is an upper bound on any
forecast method. Failure points: the real end of each file, and also many points inside the file, which gives far more samples
and shows what happens if the organisers do not cut at the very end.

Predictors are per-horizon scalar coefficients on yaw-free vector features, fitted on other environments.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Slerp

FS = 10
HOR = np.arange(1, 51) / FS   # 0.1 .. 5 s
SEQS = list(EUROC)
ENVS = dict(ENV)


def states(n):
    room = load_room(n)
    t0 = room["t_ns"][0]
    gt_t = (room["gt_t_ns"] - t0) * 1e-9
    t_imu = (room["t_ns"] - t0) * 1e-9
    a, b = max(gt_t[0], t_imu[0]) + 0.5, min(gt_t[-1], t_imu[-1]) - 0.5
    T = np.arange(a, b, 1 / FS)
    p = np.stack([np.interp(T, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    ps = uniform_filter1d(p, 3, axis=0, mode="nearest")
    v = np.gradient(ps, 1 / FS, axis=0)
    vs = uniform_filter1d(v, 5, axis=0, mode="nearest")
    acc = np.gradient(vs, 1 / FS, axis=0)
    acc = uniform_filter1d(acc, 5, axis=0, mode="nearest")
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(T, gt_t[0], gt_t[-1]))
    gy = np.stack([np.interp(T, t_imu, uniform_filter1d(room["imu"][:, i], 40)) for i in range(3)], 1)
    wup = Rg.apply(gy)[:, 2]
    M = Rg.as_matrix()
    return dict(T=T, p=p, v=vs, a=acc, wup=wup, M=M, n=n)


def feats(s, i, mu):
    """Vector features at row i, N x 3 per feature, with mu the running mean position up to row i."""
    v0, a0, p0 = s["v"][i], s["a"][i], s["p"][i]
    pull = mu - p0
    w = s["wup"][i]
    turn = np.array([-v0[1], v0[0], 0.0]) * w
    M = s["M"][i]
    bx, by, bz = M[:, 0] * [1, 1, 0], M[:, 1] * [1, 1, 0], M[:, 2] * [1, 1, 0]
    z = np.array([0, 0, 1.0]) * (mu[2] - p0[2])
    return dict(v=v0, a=a0, pull=pull, turn=turn, bx=bx, by=by, bz=bz, zpull=z)


def points(s, stride=20):
    """(rows of failure points inside the file, row of the end point). All need 5 s of truth after them."""
    n = len(s["T"])
    end = n - 51
    inner = np.arange(10 * FS, end - 5 * FS, stride)
    return inner, end


def dataset(s, keys):
    inner, end = points(s)
    rows = np.append(inner, end)
    cp = np.cumsum(s["p"], axis=0) / np.arange(1, len(s["p"]) + 1)[:, None]
    X = np.zeros((len(rows), 3, len(keys)))
    Y = np.zeros((len(rows), 3, len(HOR)))
    for j, i in enumerate(rows):
        f = feats(s, i, cp[i])
        for c, k in enumerate(keys):
            X[j, :, c] = f[k]
        Y[j] = (s["p"][i + 1 : i + 51] - s["p"][i]).T
    return X, Y, len(inner)


def fit(Xs, Ys, lam=1e-3):
    X = np.concatenate(Xs).reshape(-1, Xs[0].shape[2])
    Y = np.concatenate(Ys).transpose(0, 1, 2).reshape(-1, Ys[0].shape[2])
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)  # coeff x horizon


def err(Yhat, Y):
    """rms over horizons per point; returns per-point squared error"""
    return ((Yhat - Y) ** 2).sum(1).mean(1)


def analog(s, K=30, stride=20):
    """History-only analogue forecast: find when this file was in a similar state, and copy what happened next. Uses
    yaw-free state in the body-yaw frame. No training."""
    M = s["M"]
    yaw = np.arctan2(M[:, 1, 0], M[:, 0, 0])
    cp = np.cumsum(s["p"], axis=0) / np.arange(1, len(s["p"]) + 1)[:, None]
    c, sn = np.cos(-yaw), np.sin(-yaw)

    def rot(x):
        return np.stack([c * x[:, 0] - sn * x[:, 1], sn * x[:, 0] + c * x[:, 1]], 1)

    F = np.c_[rot(s["v"]), rot(s["a"]) * 0.5, s["v"][:, 2], s["a"][:, 2] * 0.5, s["wup"], rot(cp - s["p"]) * 0.3, (cp - s["p"])[:, 2] * 0.3]
    inner, end = points(s, stride)
    out = {}
    for i in np.append(inner, end):
        H = np.arange(0, i - 50, 2)
        if len(H) < 2 * K:
            continue
        d = ((F[H] - F[i]) ** 2).sum(1)
        o = np.argsort(d)[:K]
        h = H[o]
        w = np.exp(-d[o] / (np.median(d[o]) + 1e-6))[:, None, None]
        # future displacement rotated to the yaw now
        Y = np.stack([s["p"][j + 1 : j + 51] - s["p"][j] for j in h])
        ca, sa = np.cos(yaw[i] - yaw[h]), np.sin(yaw[i] - yaw[h])
        Yr = np.stack([ca[:, None] * Y[:, :, 0] - sa[:, None] * Y[:, :, 1], sa[:, None] * Y[:, :, 0] + ca[:, None] * Y[:, :, 1], Y[:, :, 2]], 2)
        Yh = (w * Yr).sum(0) / w.sum()
        out[i] = Yh.T
    return out


def report(names, folds, title):
    S = {n: states(n) for n in names}
    sets = {"hold": [], "v": ["v"], "v+a": ["v", "a"], "v+a+pull": ["v", "a", "pull"], "v+a+pull+turn": ["v", "a", "pull", "turn"],
            "all+axes": ["v", "a", "pull", "turn", "bx", "by", "bz"]}
    res = {}
    for nm, keys in sets.items():
        inner_e, end_e = [], []
        coefs = []
        for test, train in folds:
            if keys:
                data = [dataset(S[n], keys) for n in train]
                C = fit([d[0] for d in data], [d[1] for d in data])
                coefs.append(C)
            for n in test:
                X, Y, ni = dataset(S[n], keys or ["v"])
                Yh = np.einsum("nic,ch->nih", X, C) if keys else np.zeros_like(Y)
                e = err(Yh, Y)
                inner_e.append(e[:ni])
                end_e.append(e[ni:])
        res[nm] = (float(np.sqrt(np.concatenate(inner_e).mean())), float(np.sqrt(np.concatenate(end_e).mean())))
        if nm == "v+a+pull":
            res["coef_v+a+pull"] = np.mean(coefs, 0)[:, [4, 9, 19, 29, 39, 49]]
    # analogue
    ae_i, ae_e, hold_i, hold_e = [], [], [], []
    for n in names:
        s = S[n]
        an = analog(s)
        inner, end = points(s)
        for i, Yh in an.items():
            Y = (s["p"][i + 1 : i + 51] - s["p"][i]).T
            e = ((Yh - Y) ** 2).sum(0).mean()
            e0 = (Y ** 2).sum(0).mean()
            (ae_e if i == end else ae_i).append(e)
            (hold_e if i == end else hold_i).append(e0)
    res["analog"] = (float(np.sqrt(np.mean(ae_i))), float(np.sqrt(np.mean(ae_e)) if ae_e else np.nan))
    res["hold_same"] = (float(np.sqrt(np.mean(hold_i))), float(np.sqrt(np.mean(hold_e)) if hold_e else np.nan))
    print(f"\n=== {title}: rms error of the 5 s forecast, truth state at failure. inner points / real end of file")
    for k, v in res.items():
        if k.startswith("coef"):
            continue
        print(f"  {k:16s} inner {v[0]:.3f}  end {v[1]:.3f}")
    print("  coefficients at horizons 0.5, 1, 2, 3, 4, 5 s (v, a, pull):")
    print(np.round(res["coef_v+a+pull"], 3))
    # what fraction of 5 s displacement is motion at the end of file per sequence
    for n in names:
        s = S[n]
        i = points(s)[1]
        Y = s["p"][i + 1 : i + 51] - s["p"][i]
        print(f"   {str(n):5s} last 5 s: speed at failure {np.linalg.norm(s['v'][i]):.2f} m/s, hold error {np.sqrt((Y**2).sum(1).mean()):.2f} m, path length {np.linalg.norm(np.diff(s['p'][i:i+51],axis=0),axis=1).sum():.2f} m")
    return res


if __name__ == "__main__":
    t0 = time.time()
    r1 = report(SEQS, [(v, [x for x in SEQS if x not in v]) for v in ENVS.values()], "EuRoC")
    r2 = report(DEV_ROOMS, [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS], "TUM")
    pickle.dump(dict(eu=r1, tum=r2), open(f"{RES}/forecast_bound.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
