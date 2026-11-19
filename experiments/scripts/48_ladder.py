"""A ladder of drift models, from stiff to flexible, and a self-check to choose between them with no ground truth.
Prints ATE per level and per drift shape, and the diagnostics a chooser could use.
"""
import importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
m46 = importlib.import_module("46_drag_energy")
from phys import *
from scipy.ndimage import uniform_filter1d

LEVELS = {"L60": dict(knot=60.0, sig_g=0.01, sig_a=0.1), "L30": dict(knot=30.0, sig_g=0.02, sig_a=0.2),
          "L15": dict(knot=15.0, sig_g=0.03, sig_a=0.3), "L8": dict(knot=8.0, sig_g=0.05, sig_a=0.5)}
KINDS = ("clean", "mild", "full", "randomwalk", "sine", "burst")


def degrade_any(n, kind, sd=1):
    room = load_room(n)
    if kind in ("clean", "mild", "full"):
        t, imu, _ = eval_file(n, kind, "zero" if kind != "clean" else "delete", sd)
        return t, imu, room
    deg = importlib.import_module("56_shapes")
    t, imu = deg.degrade_custom(room, sd, kind)
    return t, imu, room


def solve_level(P, b0, lv):
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=b0, knot_a=lv["knot"], knot_g=lv["knot"])
    ec.set_target()
    if ec.drone:
        q, v, tb, last, info = m46.solve_drag(ec, sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_a=lv["sig_a"], sig_g=lv["sig_g"])
    else:
        q, v, tb, last, info = ec.solve(sig_a=lv["sig_a"], sig_g=lv["sig_g"], sig_p=5.0)
    info["drone"] = bool(ec.drone)
    return q, v, tb, last, info


def diag(v, tb):
    fs = 1.0 / np.median(np.diff(tb))
    vs = uniform_filter1d(v, int(5 * fs), axis=0, mode="nearest")
    p = np.cumsum(v, axis=0) / fs
    return dict(rv=float(np.sqrt((v ** 2).sum(1).mean())), rvs=float(np.sqrt((vs ** 2).sum(1).mean())), rp=float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())),
                vmax=float(np.percentile(np.linalg.norm(v, axis=1), 99)))


def one(key):
    n, kind = key
    t, imu, room = degrade_any(n, kind)
    P = prepare(t, imu)
    b0 = m41.calib_excl(n)
    out = {}
    for nm, lv in LEVELS.items():
        q, v, tb, last, info = solve_level(P, b0, lv)
        fs = 1.0 / np.median(np.diff(tb))
        a, st = m41.ate(np.cumsum(v, axis=0) / fs, tb, room)
        out[nm] = dict(a=a, st=st, **diag(v, tb))
    return key, out


if __name__ == "__main__":
    files = [(n, k) for n in m41.SEQS + DEV_ROOMS for k in KINDS]
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(16) as p:
        for k, o in p.imap_unordered(one, files):
            res[k] = o
    pickle.dump(res, open(f"{RES}/ladder.pkl", "wb"))
    print(f"{len(res)} files {time.time()-t0:.0f}s")
    for plat, ns in (("EuRoC", m41.SEQS), ("TUM", DEV_ROOMS)):
        for kind in KINDS:
            ks = [(n, kind) for n in ns]
            st = np.mean([res[k]["L60"]["st"] for k in ks])
            print(f"{plat:5s} {kind:10s} stat {st:.2f} | " + " | ".join(f"{nm} ATE {np.mean([res[k][nm]['a'] for k in ks]):6.2f} rvs {np.mean([res[k][nm]['rvs'] for k in ks]):5.2f} rv {np.mean([res[k][nm]['rv'] for k in ks]):5.2f}" for nm in LEVELS), flush=True)
