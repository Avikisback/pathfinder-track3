"""Shared evaluation for the second round of trials. Same folds, same scorer, same filter as the frozen entry (58_freeze.py),
so any new number can be compared with it directly. Ground truth is used only for scoring and for the band gains fitted on
other environments, exactly as before.
"""
import importlib, os, pickle, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m57 = importlib.import_module("57_nonet")
m58 = importlib.import_module("58_freeze")
m41, m48, eb = m57.m41, m57.m48, m57.eb
from phys import *

CFG = (0.9, 0.4, 8.0, 1.0)         # frozen self-check thresholds and fallback filters
LVL = "L15"
SEQS = m57.SEQS
DT = m57.DT
KNOWN = m57.KNOWN
UNSEEN = m57.UNSEEN
STEPS = [("steps", "x", 1), ("steps", "x", 2)]
ALLK = KNOWN + UNSEEN + STEPS
EU_FOLDS = [(v, [x for x in SEQS if x not in v]) for v in ENV.values()]
TU_FOLDS = [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
NAMES = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine")


def file_for(n, k):
    """Degraded rows for a file key. Returns t_ns, imu, room."""
    room = load_room(n)
    if k[0] == "steps":
        t, imu = m58.steps_file(room, k[2])
    else:
        t, imu, _ = m57.get_file(n, *k)
    return t, imu, room


def to_cache(q, v, tb, drone, **extra):
    tk = np.arange(tb[0], tb[-1], DT)
    vk = np.stack([np.interp(tk, tb, v[:, i]) for i in range(3)], 1)
    return dict(q=q.astype(np.float32), tk=tk, v=vk, drone=drone, **extra)


def blind_abs(sc, p_grid):
    """Error over the last 5 s under the alignment of the whole file: what a plain evo run restricted to the tail would see."""
    e = p_grid[sc.keep][sc.ei]
    if np.ptp(e, axis=0).max() < 1e-9:
        tr = sc.g.mean(0) - e.mean(0)
        al = e + tr
    else:
        r, tr = umeyama(e, sc.g)
        al = (r @ e.T).T + tr
    return float(np.sqrt(((al[sc.tail] - sc.g[sc.tail]) ** 2).sum(1).mean())) if sc.tail.sum() else float("nan")


def fits(folds, loader, rooms):
    """Band gains per fold, from the training environments of that fold, on the known schedules only."""
    out = {}
    for test, train in folds:
        fit = m57.spec_fit([(n, loader(n, k)) for n in train for k in KNOWN], rooms)
        for n in test:
            out[n] = fit
    return out


def score_file(n, k, d, fit, room=None, cfg=CFG, extra=None):
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    sc = PosScorer(P, room, d["q"].astype(np.float64))
    st = np.zeros((len(P["t"]), 3))
    o = dict(stationary=sc(st))
    o["stationary"]["blind_abs"] = blind_abs(sc, st)
    vs = {"final": m58.final_v(d, fit, *cfg), "nocheck": m57.apply(d, fit, "eb" if d["drone"] else "band"), "raw": d["v"] - d["v"].mean(0)}
    if extra:
        vs.update(extra(d, fit))
    for nm, v in vs.items():
        tr = m57.track(P, d, v)
        o[nm] = sc(tr)
        o[nm]["blind_abs"] = blind_abs(sc, tr)
    o["rvs"] = m57.slow_rms(d)
    return o


_LOADER = [None]


def _work(a):
    n, k, fit, cfg = a
    return (n, *k), score_file(n, k, _LOADER[0](n, k), fit, cfg=cfg)


def evaluate(names, folds, kinds, loader, cfg=CFG, procs=12):
    import multiprocessing as mp
    rooms = {n: load_room(n) for n in names}
    F = fits(folds, loader, rooms)
    _LOADER[0] = loader
    jobs = [(n, k, F[n], cfg) for n in names for k in kinds]
    rows = {}
    with mp.get_context("fork").Pool(procs) as p:
        for key, o in p.imap_unordered(_work, jobs, chunksize=4):
            rows[key] = o
    return rows


def table(rows, title, methods=("final", "nocheck", "raw"), metrics=("ate_se3", "rpe_t", "blind_rel", "blind_abs")):
    print(f"\n=== {title}")
    for kind in NAMES:
        ks = [k for k in rows if k[1] == kind]
        if not ks:
            continue
        s = [rows[k]["stationary"] for k in ks]
        print(f" {kind:10s} ({len(ks):2d}) stationary " + " ".join(f"{m} {np.nanmean([x.get(m, np.nan) for x in s]):.3f}" for m in metrics))
        for me in methods:
            if me not in rows[ks[0]]:
                continue
            f = [rows[k][me] for k in ks]
            print(f"   {me:12s} " + " ".join(f"{m} {np.nanmean([x.get(m, np.nan) for x in f]):.3f}" for m in metrics)
                  + f" worst {np.nanmax([x['ate_se3'] for x in f]):.2f} beat {np.mean([a['ate_se3'] < b['ate_se3'] for a, b in zip(f, s)]):.2f}", flush=True)
    sys.stdout.flush()


def summary(rows, me="final"):
    """Compact numbers: mean ATE per drift shape."""
    return {kind: float(np.nanmean([rows[k][me]["ate_se3"] for k in rows if k[1] == kind])) for kind in NAMES if any(k[1] == kind for k in rows)}
