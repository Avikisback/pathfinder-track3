"""H7: output gain and ensembling on stored held-out predictions.

Why this one: under an aligned ATE the MMSE estimate of an uncertain shape is a shrunken copy of it. A plain gain under 1
on the high-passed track should remove the cases where the net is worse than stationary, at the cost of 1 s RPE.
It needs almost no code and adapts to nothing about the room except the fitted gain. Ensembling uses more of
the unused CPU budget. Settings are chosen on sequences outside the test item only.

Usage: 22_h7.py --tags a,b --att auto --group loso|loeo
"""
import argparse, glob, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--att", default="auto")
ap.add_argument("--group", default="loso")
ap.add_argument("--metric", default="ate_se3")
a = ap.parse_args()
TAUS = [1, 2, 3, 4, 6, 8, 12, 24]
GAINS = [1.0, 0.8, 0.6, 0.4, 0.2]
KEYS = ("ate_se3", "ate_org", "rpe_t", "blind_rel")


def key_room(k):
    n = k.split("|")[0]
    return int(n) if n.isdigit() else n


def group_of(n):
    return "TUM" if isinstance(n, int) else [e for e, v in ENV.items() if n in v][0]


def load(tag):
    d = {}
    for f in glob.glob(f"{RES}/pred_{tag}_f*.pkl") + glob.glob(f"{RES}/pred_{tag}_r?.pkl"):
        d.update(pickle.load(open(f, "rb")))
    return d


sets = [load(t) for t in a.tags.split(",")]
keys = sorted(set.intersection(*[set(s) for s in sets]))
preds = {k: dict(tk=sets[0][k]["tk"], v=np.mean([s[k]["v"] for s in sets], 0), var=np.mean([s[k]["var"] for s in sets], 0)) for k in keys}
refs = {}
for f in ("refs_euroc.json", "refs_dev.json"):
    if os.path.exists(f"{RES}/{f}"):
        refs.update(json.load(open(f"{RES}/{f}")))
print(f"== {a.tags}: {len(keys)} files")
grid = {(t, g): {} for t in TAUS for g in GAINS}
for k in keys:
    n, s, m, sd = k.split("|")
    n = int(n) if n.isdigit() else n
    t, imu, _ = eval_file(n, s, m, int(sd))
    P = prepare(t, imu)
    q = attitude_auto(P["t"], P["imu"], P["ok"]) if a.att == "auto" else baseline_attitude(P["t"], P["imu"], P["ok"])
    room = load_room(n)
    sc = PosScorer(P, room, q)
    o = preds[k]
    for tau in TAUS:
        x = to_rows(P, o["tk"], smooth_prior(o["v"], o["var"], tau, 0.0))
        for g in GAINS:
            grid[(tau, g)][k] = sc(x * g)
show("stationary", summarise([(k, refs[k]["stationary"]) for k in keys]), KEYS)
show("baseline binary", summarise([(k, refs[k]["baseline"]) for k in keys]), KEYS)
rooms = sorted({key_room(k) for k in keys}, key=str)


def pick(allowed, metric):
    out, ch = {}, {}
    for r in rooms:
        best, bv = None, 1e18
        for kk, d in grid.items():
            if kk[1] not in allowed:
                continue
            other = (lambda x: group_of(key_room(x)) != group_of(r)) if a.group == "loeo" else (lambda x: key_room(x) != r)
            vals = [v[metric] for x, v in d.items() if other(x)]
            if vals and np.mean(vals) < bv:
                best, bv = kk, np.mean(vals)
        ch[r] = best
        out.update({x: v for x, v in grid[best].items() if key_room(x) == r})
    return out, ch


for lbl, allowed, metric in (("gain 1, tau cross-fit", [1.0], a.metric), ("gain+tau cross-fit", GAINS, a.metric),
                             ("gain+tau cross-fit on ATE origin", GAINS, "ate_org")):
    res, ch = pick(allowed, metric)
    show(lbl, summarise(list(res.items())), KEYS)
    print("     chosen", ch)
    print("     beats stationary:", {m: round(float(np.mean([res[k][m] < refs[k]["stationary"][m] for k in res])), 2) for m in ("ate_se3", "ate_org", "rpe_t")}, flush=True)
