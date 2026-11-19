"""H1/H2: trains the velocity net leave-one-room-out on the dev rooms and stores predictions on held-out files.

Usage: 10_train_vel.py --name n --T 100 --center 1 [--seg 50] [--final]
--final trains on all dev rooms and predicts on the locked room. Only run it when choices are frozen.
"""
import argparse, json, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--center", type=int, default=1)
ap.add_argument("--seg", type=int, default=0)
ap.add_argument("--seg_w", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--ch", type=str, default="24,32,48,64")
ap.add_argument("--drop", type=float, default=0.05)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--final", type=int, default=0)
a = ap.parse_args()

cfg = dict(T=a.T, center=bool(a.center), seg=a.seg, seg_w=a.seg_w, steps=a.steps, ch=tuple(int(x) for x in a.ch.split(",")),
           drop=a.drop, wd=a.wd)
folds = [(FINAL_ROOM, DEV_ROOMS)] if a.final else [(r, [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
for test, train in folds:
    if os.path.exists(f"{RES}/pred_{a.name}_r{test}.pkl"):
        continue  # restartable: finished folds are kept
    t0 = time.time()
    net = train_fold(train, cfg, seed=100 + test, log=f"{a.name} r{test}")
    net.center = cfg["center"]
    torch.save(dict(state=net.state_dict(), cfg=net.cfg, center=cfg["center"], train_rooms=train),
               f"{WGT}/{a.name}_r{test}.pt")
    nparams = sum(p.numel() for p in net.parameters())
    preds = {}
    for (n, s, m, sd) in eval_set([test]):
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        o = infer(net, P)
        preds[f"{n}|{s}|{m}|{sd}"] = dict(tk=o["tk"], v=o["v"], var=o["var"])
    pickle.dump(preds, open(f"{RES}/pred_{a.name}_r{test}.pkl", "wb"))
    print(f"{a.name} room{test} done in {time.time()-t0:.0f}s params {nparams} macs/forward {macs(net.cfg)}", flush=True)
