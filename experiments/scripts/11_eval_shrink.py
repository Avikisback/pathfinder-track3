"""Scores stored net predictions. Shrinkage settings for a test room are chosen only from the OTHER rooms."""
import argparse, glob, json, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
a = ap.parse_args()
refs = references(eval_set(DEV_ROOMS), f"{RES}/refs_dev.json")
print("reference rows (mean/worst over rooms, seeds, gap modes)")
show("stationary", summarise([(k, v["stationary"]) for k, v in refs.items()]))
show("baseline", summarise([(k, v["baseline"]) for k, v in refs.items()]))

TAUS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 40]
ALPHAS = [0.0, 0.5, 1.0]
_P = {}
def getP(key):
    if key not in _P:
        n, s, m, sd = key.split("|")
        t, imu, _ = eval_file(int(n), s, m, int(sd))
        P = prepare(t, imu)
        q = baseline_attitude(P["t"], P["imu"], P["ok"])
        _P[key] = (P, q, load_room(int(n)))
    return _P[key]

for name in a.names:
    preds = {}
    for r in DEV_ROOMS:
        preds.update(pickle.load(open(f"{RES}/pred_{name}_r{r}.pkl", "rb")))
    t0 = time.time()
    grid = {}          # (tau, alpha) -> {key: score}
    for tau in TAUS:
        for al in ALPHAS:
            grid[(tau, al)] = {}
            for key, o in preds.items():
                P, q, room = getP(key)
                x = smooth_prior(o["v"], o["var"], tau, al)
                grid[(tau, al)][key] = score_all(P, room, to_rows(P, o["tk"], x), q)
    raw, causal = {}, {}
    for key, o in preds.items():
        P, q, room = getP(key)
        raw[key] = score_all(P, room, to_rows(P, o["tk"], integrate(o["v"])), q)
    print(f"== {name} ({time.time()-t0:.0f}s)")
    show("no shrink", summarise(list(raw.items())))
    # fixed-tau view: chosen per test room from the other rooms' files, on the full schedule and mild together
    def pick(al_set, metric):
        out = {}
        chosen = {}
        for r in DEV_ROOMS:
            best, bv = None, 1e9
            for (tau, al), d in grid.items():
                if al not in al_set:
                    continue
                v = np.mean([x[metric] for k, x in d.items() if int(k.split("|")[0]) != r])
                if v < bv:
                    best, bv = (tau, al), v
            chosen[r] = best
            for k, x in grid[best].items():
                if int(k.split("|")[0]) == r:
                    out[k] = x
        return out, chosen
    for lbl, als in (("fixed tau (cross-fit)", [0.0]), ("sigma-driven (cross-fit)", ALPHAS)):
        res, ch = pick(als, "ate_se3")
        show(lbl, summarise(list(res.items())))
        print("     chosen", ch)
        wins = np.mean([res[k]["ate_se3"] < refs[k]["stationary"]["ate_se3"] for k in res])
        wino = np.mean([res[k]["ate_org"] < refs[k]["stationary"]["ate_org"] for k in res])
        winr = np.mean([res[k]["rpe_t"] < refs[k]["stationary"]["rpe_t"] for k in res])
        print(f"     beats stationary on {100*wins:.0f}% of files for ATE SE3, {100*wino:.0f}% for ATE origin, {100*winr:.0f}% for RPE 1 s")
    res, ch = pick(ALPHAS, "ate_org")
    show("sigma-driven, picked on origin", summarise(list(res.items())))
    print("     chosen", ch)
    # sensitivity view, in-sample by construction, labelled as such
    print("   ATE SE3 full+mild mean by tau, alpha=0 (all dev rooms, sensitivity only):",
          " ".join(f"{t}:{np.mean([x['ate_se3'] for x in grid[(t,0.0)].values()]):.3f}" for t in TAUS))
    print("   ATE origin by tau, alpha=0:", " ".join(f"{t}:{np.mean([x['ate_org'] for x in grid[(t,0.0)].values()]):.3f}" for t in TAUS))
    print("   RPE1s by tau, alpha=0:", " ".join(f"{t}:{np.mean([x['rpe_t'] for x in grid[(t,0.0)].values()]):.3f}" for t in TAUS))
    pickle.dump(dict(grid=grid, raw=raw), open(f"{RES}/shrink_{name}.pkl", "wb"))
