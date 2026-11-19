"""Can the *fast* part of the IMU tell speed? Bias and slow drift do not touch vibration or gait bands, so if band power predicts
speed we get a speedometer that drift cannot corrupt: the handheld analogue of the drone drag model.

Features: log band power per 2 s window, for vertical and horizontal parts of accel and gyro (yaw-free by construction).
Target: true speed over the same window. Trained only on other environments. Tested on clean and on degraded rows.
"""
import importlib, multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from scipy.signal import butter, filtfilt

BANDS = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 10, 15, 25, 40, 70, 100.01])
NB = len(BANDS) - 1
WIN, HOP = 2.0, 1.0


def signals(t, imu, ok):
    fs = round(1.0 / np.median(np.diff(t)))
    idx = np.arange(len(t))
    v = np.stack([np.interp(idx, idx[ok], imu[ok, i]) for i in range(6)], 1)
    b, a = butter(2, 0.5 / (fs / 2))
    up = filtfilt(b, a, v[:, 3:], axis=0)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    av = (v[:, 3:] * up).sum(1)
    ah = v[:, 3:] - av[:, None] * up
    wv = (v[:, :3] * up).sum(1)
    wh = v[:, :3] - wv[:, None] * up
    return fs, av, ah, wv, wh


def features(t, imu, ok, last):
    fs, av, ah, wv, wh = signals(t, imu, ok)
    w = int(WIN * fs)
    h = int(HOP * fs)
    st = np.arange(0, last - w, h)
    win = np.hanning(w)
    fr = np.fft.rfftfreq(w, 1.0 / fs)
    bin_of = np.digitize(fr, BANDS) - 1
    X, good, cen = [], [], []
    for s in st:
        e = s + w
        cols = []
        for sig in (av[s:e, None], ah[s:e], wv[s:e, None], wh[s:e]):
            S = np.abs(np.fft.rfft((sig - sig.mean(0)) * win[:, None], axis=0)) ** 2
            P = np.bincount(np.clip(bin_of, 0, NB), S.sum(1), minlength=NB + 1)[:NB] / w
            cols.append(np.log(P + 1e-7))
        X.append(np.concatenate(cols))
        good.append(ok[s:e].mean() > 0.95)
        cen.append(t[s + w // 2])
    return np.array(X), np.array(good), np.array(cen)


def target(room, P, cen):
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    tt = np.arange(gt_t[0], gt_t[-1], 0.02)
    p = np.stack([np.interp(tt, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    v = np.gradient(p, 0.02, axis=0)
    from scipy.ndimage import uniform_filter1d
    v = uniform_filter1d(v, 5, axis=0)
    sp = np.linalg.norm(v, axis=1)
    sph = np.linalg.norm(v[:, :2], axis=1)
    out, outh, good = [], [], []
    for c in cen:
        m = (tt >= c - WIN / 2) & (tt <= c + WIN / 2)
        if m.sum() < 50:
            out.append(0.0); outh.append(0.0); good.append(False)
        else:
            out.append(sp[m].mean()); outh.append(sph[m].mean()); good.append(True)
    return np.array(out), np.array(outh), np.array(good)


def one(key):
    n, k = key
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    X, g1, cen = features(P["t"], P["imu"], P["ok"], P["last_good"])
    y, yh, g2 = target(room, P, cen)
    return key, dict(X=X.astype(np.float32), y=y, yh=yh, good=g1 & g2)


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda Xq: ((Xq - mu) / sd) @ w + y.mean()


def scores(pred, y):
    r = np.corrcoef(pred, y)[0, 1]
    r2 = 1 - ((pred - y) ** 2).mean() / ((y - y.mean()) ** 2).mean()
    return r, r2, float(np.sqrt(((pred - y) ** 2).mean()))


if __name__ == "__main__":
    t0 = time.time()
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    except Exception as e:
        print("no sklearn", e)
        HGB = None
    KS = [("clean", "delete", 1), ("full", "zero", 1), ("randomwalk", "x", 1)]
    names = SEQS + DEV_ROOMS
    jobs = [(n, k) for n in names for k in KS]
    with mp.get_context("fork").Pool(14) as p:
        D = dict(p.map(one, jobs))
    pickle.dump(D, open(f"{RES}/speedo_feats.pkl", "wb"))
    print(f"features {time.time()-t0:.0f}s", flush=True)
    for plat, group, folds in (("EuRoC", SEQS, EU_FOLDS), ("TUM", DEV_ROOMS, TU_FOLDS)):
        for k in KS:
            for target_name in ("y", "yh"):
                Pr, Y, Pg = [], [], []
                for test, train in folds:
                    # train on clean + degraded of the other environments
                    Xtr = np.concatenate([D[(n, kk)]["X"][D[(n, kk)]["good"]] for n in train for kk in KS])
                    ytr = np.concatenate([D[(n, kk)][target_name][D[(n, kk)]["good"]] for n in train for kk in KS])
                    f = ridge(Xtr, ytr, 1e-2)
                    if HGB is not None and target_name == "y":
                        gb = HGB(max_depth=4, max_iter=200, learning_rate=0.05).fit(Xtr, ytr)
                    for n in test:
                        d = D[(n, k)]
                        m = d["good"]
                        Pr.append(f(d["X"][m]))
                        if HGB is not None and target_name == "y":
                            Pg.append(gb.predict(d["X"][m]))
                        Y.append(d[target_name][m])
                pr, y = np.concatenate(Pr), np.concatenate(Y)
                r, r2, e = scores(pr, y)
                txt = f"{plat:5s} {k[0]:10s} {target_name:2s} ridge corr {r:5.2f} R2 {r2:5.2f} rms {e:.3f} (speed mean {y.mean():.2f} std {y.std():.2f})"
                if Pg:
                    r, r2, e = scores(np.concatenate(Pg), y)
                    txt += f" | boosted corr {r:5.2f} R2 {r2:5.2f} rms {e:.3f}"
                print(txt, flush=True)
    # per-file correlation for clean: which sequences work
    print("\nper file on clean (ridge, 3D speed), and share of variance by feature family")
    for plat, group, folds in (("EuRoC", SEQS, EU_FOLDS), ("TUM", DEV_ROOMS, TU_FOLDS)):
        for test, train in folds:
            Xtr = np.concatenate([D[(n, kk)]["X"][D[(n, kk)]["good"]] for n in train for kk in KS])
            ytr = np.concatenate([D[(n, kk)]["y"][D[(n, kk)]["good"]] for n in train for kk in KS])
            f = ridge(Xtr, ytr, 1e-2)
            for n in test:
                d = D[(n, KS[0])]
                m = d["good"]
                print(f"  {str(n):5s} corr {np.corrcoef(f(d['X'][m]), d['y'][m])[0,1]:5.2f} rms {np.sqrt(((f(d['X'][m])-d['y'][m])**2).mean()):.3f} speed {d['y'][m].mean():.2f}")
    print(f"done {time.time()-t0:.0f}s")
