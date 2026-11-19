"""The no-network entry: self-calibrating solver, then a filter with very few numbers, then a physical plausibility clamp.

Clamp: velocity power per octave band cannot exceed what real platforms show, with a margin. If it does, that band is
mostly drift, so scale it down. This bounds the damage from any drift shape. Numbers come from training files only, and
only from the known schedules. The other drift shapes are test-only.

Usage: 57_nonet.py [levels...]
"""
import importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
m46 = importlib.import_module("46_drag_energy")
m48 = importlib.import_module("48_ladder")
shapes = importlib.import_module("56_shapes")
eb = importlib.import_module("52_ebwiener")
from phys import *

D8 = EXP + "/cache/v8"
os.makedirs(D8, exist_ok=True)
SEQS = m41.SEQS
KNOWN = [("clean", "delete", 1), ("mild", "zero", 1), ("mild", "delete", 2), ("full", "zero", 1), ("full", "delete", 2), ("full", "zero", 3)]
UNSEEN = [(k, "x", sd) for k in ("randomwalk", "sine", "burst") for sd in (1, 2)]
TAIL_TAU, TAIL_GAIN = 0.35, 0.5
DT = 0.1


def get_file(n, kind, mode, sd):
    room = load_room(n)
    if kind in SCHEDULES:
        t, imu, _ = eval_file(n, kind, mode, sd)
    else:
        t, imu = shapes.degrade_custom(room, sd, kind)
    return t, imu, room


def one(key):
    n, kind, mode, sd, lvl = key
    f = f"{D8}/{n}_{kind}_{mode}_{sd}_{lvl}.pkl"
    if os.path.exists(f):
        return key
    t, imu, room = get_file(n, kind, mode, sd)
    P = prepare(t, imu)
    q, v, tb, last, info = m48.solve_level(P, m41.calib_excl(n), m48.LEVELS[lvl])
    tk = np.arange(tb[0], tb[-1], DT)
    vk = np.stack([np.interp(tk, tb, v[:, i]) for i in range(3)], 1)
    with open(f, "wb") as fh:
        pickle.dump(dict(q=q.astype(np.float32), tk=tk, v=vk, drone=info["drone"]), fh)
    return key


def load(n, kind, mode, sd, lvl):
    return pickle.load(open(f"{D8}/{n}_{kind}_{mode}_{sd}_{lvl}.pkl", "rb"))


def spec_fit(items, room_of):
    """From training files: wiener gains and error power per band, and the plausible velocity power per band."""
    Sxy, Sz, Txy, Tz, gxy, gz = [], [], [], [], [], []
    for (n, d) in items:
        room = room_of[n]
        t0 = room["t_ns"][0]
        gt_t = (room["gt_t_ns"] - t0) * 1e-9
        tk = d["tk"]
        h = 0.05
        fn = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
        vg = (fn(tk + h) - fn(tk - h)) / (2 * h)
        idx = np.clip(np.searchsorted(gt_t, tk), 1, len(gt_t) - 1)
        good = (tk - h >= gt_t[0]) & (tk + h <= gt_t[-1]) & ((gt_t[idx] - gt_t[idx - 1]) < 0.15)
        vh = d["v"]
        Rg = eb.yaw_gauge(vh[good], vg[good])
        vs = (Rg @ vh.T).T
        vg2 = np.where(good[:, None], vg, vs)
        e = vs - vg2
        N = len(e)
        X, f = eb.spec(e - e.mean(0))
        a, b = eb.band_power(X, f, N)
        Sxy.append(a)
        Sz.append(b)
        Y, _ = eb.spec(vg2 - vg2.mean(0))
        a, b = eb.band_power(Y, f, N)
        Txy.append(a)
        Tz.append(b)
        V, _ = eb.spec(vs - vs.mean(0))
        bins = np.digitize(f, eb.EDGES_B) - 1
        for arr, c in ((gxy, (0, 1)), (gz, (2,))):
            num = sum(np.bincount(bins, (np.conj(V[:, i]) * Y[:, i]).real, minlength=eb.NB)[: eb.NB] for i in c)
            den = sum(np.bincount(bins, (np.abs(V[:, i]) ** 2), minlength=eb.NB)[: eb.NB] for i in c)
            arr.append(num / np.maximum(den, 1e-12))
    return dict(Sxy=np.median(Sxy, 0), Sz=np.median(Sz, 0), Txy=np.percentile(Txy, 95, axis=0), Tz=np.percentile(Tz, 95, axis=0),
                gxy=np.clip(np.median(gxy, 0), 0, 1.1), gz=np.clip(np.median(gz, 0), 0, 1.1))


def slow_rms(d, win_s=5.0):
    """The self-check. Real platforms in bounded spaces have a small 5 s average velocity; uncorrected drift does not."""
    from scipy.ndimage import uniform_filter1d
    vs = uniform_filter1d(d["v"], int(win_s / DT), axis=0, mode="nearest")
    return float(np.sqrt((vs ** 2).sum(1).mean()))


def apply(d, fit, mode, margin=2.0):
    if mode.startswith("auto"):
        # self-check passes: use the platform's filter. Fails: fall back to keeping fast motion only.
        thr = float(mode[4:])
        if slow_rms(d) > thr:
            return apply(d, fit, "safe")
        return apply(d, fit, "eb" if d["drone"] else "band")
    if mode == "safe":
        v = d["v"]
        vm = v - v.mean(0)
        X, f = eb.spec(vm)
        G = (2 * np.pi * f * 4.0) ** 2 / (1 + (2 * np.pi * f * 4.0) ** 2)
        return np.fft.irfft(X * G[:, None], n=2 * len(vm), axis=0)[: len(vm)]
    v = d["v"]
    vm = v - v.mean(0)
    n = len(vm)
    X, f = eb.spec(vm)
    Pxy, Pz = eb.band_power(X, f, n)
    if mode.startswith("eb"):
        gxy = np.clip(1 - fit["Sxy"] / np.maximum(Pxy, 1e-12), 0, 1)
        gz = np.clip(1 - fit["Sz"] / np.maximum(Pz, 1e-12), 0, 1)
    elif mode.startswith("band"):
        gxy, gz = fit["gxy"].copy(), fit["gz"].copy()
    else:
        gxy, gz = np.ones(eb.NB), np.ones(eb.NB)
    if mode.endswith("clamp"):
        gxy = np.minimum(gxy, np.sqrt(margin * fit["Txy"] / np.maximum(Pxy, 1e-12)))
        gz = np.minimum(gz, np.sqrt(margin * fit["Tz"] / np.maximum(Pz, 1e-12)))
    return eb.apply_gains(X, f, gxy, gz, n) + (0.0 if mode != "raw" else v.mean(0))


def track(P, d, vf):
    tk = d["tk"]
    x = np.cumsum(vf, axis=0) * DT
    out = np.stack([np.interp(P["t"], tk, x[:, i]) for i in range(3)], 1)
    t_last = tk[-1]
    tail = P["t"] > t_last
    if tail.any():
        v0 = vf[-5:].mean(0)
        s = P["t"][tail] - t_last
        out[tail] = x[-1] + TAIL_GAIN * v0 * (TAIL_TAU * (1 - np.exp(-s / TAIL_TAU)))[:, None]
    return out


METHODS = ("raw", "band", "eb", "safe", "auto0.5", "auto0.6", "auto0.7", "auto0.8")


def evaluate(names, folds, lvl):
    rooms = {n: load_room(n) for n in names}
    res = {}
    for test, train in folds:
        items = [(n, load(n, *k, lvl)) for n in train for k in KNOWN]
        fit = spec_fit(items, rooms)
        for n in test:
            for k in KNOWN + UNSEEN:
                t, imu, room = get_file(n, *k)
                P = prepare(t, imu)
                d = load(n, *k, lvl)
                q = d["q"].astype(np.float64)
                sc = PosScorer(P, room, q)
                o = {"stationary": sc(np.zeros((len(P["t"]), 3)))}
                for m in METHODS:
                    o[m] = sc(track(P, d, apply(d, fit, m)))
                res[(n, k[0], k[1], k[2])] = o
    return res


def table(res, title):
    print(f"\n=== {title}")
    for sc_ in ("clean", "mild", "full", "burst", "randomwalk", "sine"):
        ks = [k for k in res if k[1] == sc_]
        if not ks:
            continue
        print(f"-- {sc_} ({len(ks)} files)")
        for m in ("stationary",) + METHODS:
            print(f"  {m:10s} " + " ".join(f"{x} {np.nanmean([res[k][m][x] for k in ks]):.3f}/{np.nanmax([res[k][m][x] for k in ks]):.3f}" for x in ("ate_se3", "ate_org", "rpe_t", "blind_rel")))
    sys.stdout.flush()


if __name__ == "__main__":
    levels = sys.argv[1:] or ["L60", "L15", "L8"]
    keys = [(n, *k, lvl) for lvl in levels for n in SEQS + DEV_ROOMS + [FINAL_ROOM] for k in KNOWN + UNSEEN]
    t0 = time.time()
    with mp.get_context("spawn").Pool(16) as p:
        for _ in p.imap_unordered(one, keys):
            pass
    print(f"solved {len(keys)} in {time.time()-t0:.0f}s", flush=True)
    for lvl in levels:
        r1 = evaluate(SEQS, [(v, [x for x in SEQS if x not in v]) for v in ENV.values()], lvl)
        table(r1, f"{lvl} EuRoC, unseen environment")
        r2 = evaluate(DEV_ROOMS, [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS], lvl)
        table(r2, f"{lvl} TUM, unseen room")
        r3 = evaluate([1, 2, 3, 4, 5, 6], [([6], DEV_ROOMS)], lvl)
        table(r3, f"{lvl} locked room 6, trained on rooms 1-5")
        pickle.dump(dict(eu=r1, tum=r2, room6=r3), open(f"{RES}/nonet_{lvl}.pkl", "wb"))
