"""Trial of the whitened cost and per-file drift selection by evidence. Spectra are always from other environments.

Usage: 64_gls.py <tag> [options: couple=0/1 select=0/1 knot=10] [names: eu,tu]
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
import gls

D10 = EXP + "/cache/v10"
os.makedirs(D10, exist_ok=True)
TAB = f"{RES}/spec_tables.pkl"


def train_of(n):
    for test, train in EU_FOLDS + TU_FOLDS:
        if n in test:
            return train
    return DEV_ROOMS   # room 6


def one(a):
    key, tag, opt = a
    n, k = key
    f = f"{D10}/{n}_{k[0]}_{k[1]}_{k[2]}_{tag}.pkl"
    if os.path.exists(f):
        return key
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    tables = pickle.load(open(TAB, "rb"))
    spec = gls.fold_spec(tables, train_of(n))
    q, v, tb, last, info = gls.solve_file(P, m41.calib_excl(n), spec, **opt)
    d = to_cache(q, v, tb, info["drone"], sa=info["sa"], sg=info["sg"], ax=info["ax"], az=info["az"], chi2=info["chi2"], hist=info["hist"], tab=info.get("tab"))
    with open(f, "wb") as fh:
        pickle.dump(d, fh)
    return key


def loader(tag):
    def load(n, k):
        return pickle.load(open(f"{D10}/{n}_{k[0]}_{k[1]}_{k[2]}_{tag}.pkl", "rb"))
    return load


if __name__ == "__main__":
    tag = sys.argv[1]
    opt = {}
    sets = ["eu", "tu"]
    for a in sys.argv[2:]:
        if a.startswith("sets="):
            sets = a[5:].split(",")
            continue
        kk, vv = a.split("=")
        opt[kk] = bool(int(vv)) if kk in ("couple", "select") else float(vv)
    t0 = time.time()
    gls.spec_tables(SEQS + ALL_ROOMS, TAB)
    print(f"tables {time.time()-t0:.0f}s", flush=True)
    names = (SEQS if "eu" in sets else []) + (DEV_ROOMS if "tu" in sets else [])
    jobs = [((n, k), tag, opt) for n in names for k in ALLK]
    with mp.get_context("spawn").Pool(16) as p:
        for _ in p.imap_unordered(one, jobs, chunksize=2):
            pass
    print(f"solved {len(jobs)} in {time.time()-t0:.0f}s", flush=True)
    out = {}
    L = loader(tag)
    if "eu" in sets:
        r = evaluate(SEQS, EU_FOLDS, ALLK, L)
        table(r, f"EuRoC whitened {tag}")
        out["eu"] = r
        print("  drift picks (sa, sg) per shape, mean:")
        for kd in NAMES:
            ks = [(n, k) for n in SEQS for k in ALLK if k[0] == kd]
            print(f"   {kd:10s} sa {np.mean([np.log10(L(*x)['sa']) for x in ks]):.2f} sg {np.mean([np.log10(L(*x)['sg']) for x in ks]):.2f} ax {np.mean([L(*x)['ax'] for x in ks]):.2f} chi2 {np.mean([L(*x)['chi2'] for x in ks]):.2f}", flush=True)
    if "tu" in sets:
        r = evaluate(DEV_ROOMS, TU_FOLDS, ALLK, L)
        table(r, f"TUM whitened {tag}")
        out["tu"] = r
        print("  drift picks (sa, sg) per shape, mean:")
        for kd in NAMES:
            ks = [(n, k) for n in DEV_ROOMS for k in ALLK if k[0] == kd]
            print(f"   {kd:10s} sa {np.mean([np.log10(L(*x)['sa']) for x in ks]):.2f} sg {np.mean([np.log10(L(*x)['sg']) for x in ks]):.2f} ax {np.mean([L(*x)['ax'] for x in ks]):.2f} chi2 {np.mean([L(*x)['chi2'] for x in ks]):.2f}", flush=True)
    pickle.dump(out, open(f"{RES}/gls_{tag}.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
