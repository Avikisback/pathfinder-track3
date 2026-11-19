"""Scores stored predictions from 20_cross.py. Shrinkage is chosen only from sequences outside the test group:
for loso the other EuRoC sequences, for loeo the other environments (their predictions come from models that
did not see them). For one-model schemes (tum2euroc, euroc2tum) it is chosen from the TRAINING domain's own
held-out predictions if given by --tau_from, otherwise reported per tau as sensitivity only.

Usage: 21_eval_cross.py --tag name_scheme [--att auto]
"""
import argparse, glob, json, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--att", default="auto")
ap.add_argument("--rfold", type=int, default=0)
a = ap.parse_args()

TAUS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 40, 80]
ALPHAS = [0.0, 0.5, 1.0]
scheme = a.tag.split("_")[-1]


def key_room(k):
    n = k.split("|")[0]
    return int(n) if n.isdigit() else n


def group_of(n):
    if isinstance(n, int):
        return "TUM"
    return [e for e, v in ENV.items() if n in v][0]


def load_preds(tag):
    preds = {}
    for f in sorted(glob.glob(f"{RES}/pred_{tag}_f*.pkl") + glob.glob(f"{RES}/pred_{tag}_r?.pkl")):
        preds.update(pickle.load(open(f, "rb")))
    return preds


_P = {}
def getP(key):
    if key not in _P:
        n, s, m, sd = key.split("|")
        n = int(n) if n.isdigit() else n
        t, imu, _ = eval_file(n, s, m, int(sd))
        P = prepare(t, imu)
        q = attitude_auto(P["t"], P["imu"], P["ok"]) if a.att == "auto" else baseline_attitude(P["t"], P["imu"], P["ok"])
        room = load_room(n)
        _P[key] = (P, q, room, PosScorer(P, room, q))
    return _P[key]


def grid_scores(preds):
    grid = {(tau, al): {} for tau in TAUS for al in ALPHAS}
    raw = {}
    for key, o in preds.items():
        P, q, room, sc = getP(key)
        for (tau, al) in grid:
            x = smooth_prior(o["v"], o["var"], tau, al)
            grid[(tau, al)][key] = sc(to_rows(P, o["tk"], x))
        raw[key] = sc(to_rows(P, o["tk"], integrate(o["v"])))
    return grid, raw


def pick(grid, als, metric, loeo):
    """Per test item, best setting on items outside its group."""
    out, chosen = {}, {}
    rooms = sorted({key_room(k) for k in next(iter(grid.values()))}, key=str)
    for r in rooms:
        best, bv = None, 1e18
        def other(k):
            n = key_room(k)
            return group_of(n) != group_of(r) if loeo else n != r
        for (tau, al), d in grid.items():
            if al not in als:
                continue
            vals = [x[metric] for k, x in d.items() if other(k)]
            if not vals:
                continue
            v = np.mean(vals)
            if v < bv:
                best, bv = (tau, al), v
        chosen[r] = best
        for k, x in grid[best].items():
            if key_room(k) == r:
                out[k] = x
    return out, chosen


def wins(res, refs):
    return {m: float(np.mean([res[k][m] < refs[k]["stationary"][m] for k in res])) for m in ("ate_se3", "ate_org", "rpe_t")}


def per_seq(res, refs, metric="ate_se3"):
    rows = []
    rooms = sorted({key_room(k) for k in res}, key=str)
    for r in rooms:
        ks = [k for k in res if key_room(k) == r and k.split("|")[1] == "full"]
        rows.append(f"{r}: {np.mean([res[k][metric] for k in ks]):.2f} vs {np.mean([refs[k]['stationary'][metric] for k in ks]):.2f}")
    return " | ".join(rows)


preds = load_preds(a.tag)
rooms = {key_room(k) for k in preds}
refs = {}
if any(isinstance(r, str) for r in rooms):
    refs.update(references(eval_set(list(EUROC)), f"{RES}/refs_euroc.json"))
if any(isinstance(r, int) for r in rooms):
    refs.update(references(eval_set(DEV_ROOMS), f"{RES}/refs_dev.json"))
refs = {k: v for k, v in refs.items() if k in preds}
KEYS = ("ate_se3", "ate_org", "rpe_t", "blind_rel")
print(f"== {a.tag}: {len(preds)} files, {len(rooms)} sequences")
show("stationary", summarise([(k, v["stationary"]) for k, v in refs.items()]), KEYS)
show("baseline binary", summarise([(k, v["baseline"]) for k, v in refs.items()]), KEYS)
t0 = time.time()
grid, raw = grid_scores(preds)
show("net, no shrink", summarise(list(raw.items())), KEYS)
cross = scheme in ("loso", "loeo", "both") or a.att == "baseline"
if cross:
    loeo = scheme == "loeo"
    for lbl, als in (("net, fixed tau (cross-fit)", [0.0]), ("net, sigma-driven (cross-fit)", ALPHAS)):
        res, ch = pick(grid, als, "ate_se3", loeo)
        show(lbl, summarise(list(res.items())), KEYS)
        print("     chosen", ch)
        print("     beats stationary on share of files:", {k: round(v, 2) for k, v in wins(res, refs).items()})
        print("     full schedule by sequence, ATE SE3 net vs stationary:", per_seq(res, refs))
    res, ch = pick(grid, ALPHAS, "ate_org", loeo)
    show("net, picked on origin ATE", summarise(list(res.items())), KEYS)
    print("     chosen", ch)
    print("     full schedule by sequence, ATE origin:", per_seq(res, refs, "ate_org"))
for m in ("ate_se3", "ate_org", "rpe_t"):
    print(f"   sensitivity {m} by tau alpha=0, all schedules:", " ".join(f"{t}:{np.mean([x[m] for x in grid[(t,0.0)].values()]):.3f}" for t in TAUS))
    print(f"   sensitivity {m} by tau alpha=1, all schedules:", " ".join(f"{t}:{np.mean([x[m] for x in grid[(t,1.0)].values()]):.3f}" for t in TAUS))
if any(isinstance(r, str) for r in rooms):
    for e, v in ENV.items():
        print(f"   {e} ATE SE3 by tau alpha=0:", " ".join(f"{t}:{np.mean([x['ate_se3'] for k, x in grid[(t,0.0)].items() if key_room(k) in v]):.3f}" for t in TAUS))
pickle.dump(dict(grid=grid, raw=raw), open(f"{RES}/shrink_{a.tag}.pkl", "wb"))
print(f"   ({time.time()-t0:.0f}s)")
