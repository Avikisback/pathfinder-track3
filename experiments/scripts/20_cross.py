"""EuRoC and cross-domain velocity nets. Stores predictions on held-out files only.

Schemes:
  loso       leave one EuRoC sequence out (11 folds)
  loeo       leave one EuRoC environment out: MH, V1, V2 (3 folds)
  tum2euroc  train on TUM-VI rooms 1-5, test on all EuRoC
  euroc2tum  train on all EuRoC, test on TUM-VI rooms 1-5
  both       train on both datasets, leaving out the test sequence (EuRoC) or room (rooms 1-5)
Room 6 is never used.
"""
import argparse, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--scheme", required=True)
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--center", type=int, default=1)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--yaw_aug", type=int, default=0)
ap.add_argument("--no_up", type=int, default=0)
ap.add_argument("--per_room", type=int, default=2)
ap.add_argument("--ch", type=str, default="24,32,48,64")
a = ap.parse_args()

SEQS = list(EUROC)
if a.scheme == "loso":
    folds = [([s], [x for x in SEQS if x != s]) for s in SEQS]
elif a.scheme == "loeo":
    folds = [(v, [x for x in SEQS if x not in v]) for v in ENV.values()]
elif a.scheme == "tum2euroc":
    folds = [(SEQS, list(DEV_ROOMS))]
elif a.scheme == "euroc2tum":
    folds = [(list(DEV_ROOMS), SEQS)]
elif a.scheme == "both":
    folds = [([s], [x for x in SEQS if x != s] + list(DEV_ROOMS)) for s in SEQS]
    folds += [([r], SEQS + [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
elif a.scheme == "loro":
    folds = [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
else:
    raise SystemExit("unknown scheme")

cfg = dict(T=a.T, center=bool(a.center), steps=a.steps, ch=tuple(int(x) for x in a.ch.split(",")), yaw_aug=a.yaw_aug,
           no_up=a.no_up, pool_per_room=a.per_room, attitude="auto", cached=True)
tag = f"{a.name}_{a.scheme}"
for fi, (test, train) in enumerate(folds):
    out = f"{RES}/pred_{tag}_f{fi}.pkl"
    if os.path.exists(out):
        continue
    t0 = time.time()
    net = train_fold(train, cfg, seed=200 + fi, log=None)
    net.center = cfg["center"]
    net.no_up = bool(a.no_up)
    torch.save(dict(state=net.state_dict(), cfg=net.cfg, center=cfg["center"], no_up=bool(a.no_up), train=train),
               f"{WGT}/{tag}_f{fi}.pt")
    preds = {}
    for (n, s, m, sd) in eval_set(test):
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        q = attitude_auto(P["t"], P["imu"], P["ok"])
        o = infer(net, P, q=q)
        preds[f"{n}|{s}|{m}|{sd}"] = dict(tk=o["tk"], v=o["v"], var=o["var"])
    pickle.dump(preds, open(out, "wb"))
    print(f"{tag} fold {fi} test {test[:3]} done {time.time()-t0:.0f}s", flush=True)
print(f"{tag} ALL_DONE", flush=True)
