"""H3 part 1: classical whole-file attitude against the baseline tilt filter. Dev rooms only (no training here).

Extra drift shapes check that the method is not just fitted to the ramp in tools/degrade.py:
  sine:  bias drifts as a slow sine of random phase, peak 0.02 rad/s per axis, noise and gaps as 'full'
  euroc: constant gyro bias of EuRoC size, (-0.002, 0.021, 0.077) rad/s, plus 'full' damage
"""
import json
import sys
import time

sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *


def variant_file(n, kind, mode, seed):
    room = load_room(n)
    t, imu, gaps = eval_file(n, "full", mode, seed)
    t = t.copy()
    imu = imu.copy()
    rng = np.random.default_rng(seed + 1000 * n)
    live = ~(imu == 0).all(1)
    ts = (t - t[0]) * 1e-9
    x = ts / ts[-1]
    if kind == "sine":
        # remove nothing from the ramp file: build from clean rows instead, so only the shape differs
        rc = dict(zip(room["t_ns"].tolist(), range(len(room["t_ns"]))))
        idx = np.array([rc[int(v)] for v in t])
        clean = room["imu"][idx]
        per = rng.uniform(40, 120, 3)
        ph = rng.uniform(0, 2 * np.pi, 3)
        drift = 0.02 * np.sin(2 * np.pi * ts[:, None] / per + ph)
        out = clean.copy()
        out[:, :3] += drift + rng.standard_normal((len(t), 3)) * (0.002 + 0.008 * x)[:, None]
        out[:, 3:] += rng.standard_normal((len(t), 3)) * (0.02 + 0.08 * x)[:, None]
        out[~live] = 0
        return t, out
    if kind == "euroc":
        imu[live, :3] += np.array([-0.002, 0.021, 0.077])
        return t, imu
    return t, imu


def run(P, q, room):
    return score_all(P, room, np.zeros((len(P["t"]), 3)), q)


METHODS = {
    "baseline tilt filter": lambda P: baseline_attitude(P["t"], P["imu"], P["ok"]),
    "low-gain tilt kp=0.02": lambda P: baseline_attitude(P["t"], P["imu"], P["ok"], kp=0.02, boost=0.5),
    "gyro only": lambda P: baseline_attitude(P["t"], P["imu"], P["ok"], kp=0.0, boost=0.0),
    "offline no bias fit, gaps decay": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], fit_bias=False, gap_mode="decay"),
    "offline no bias fit, gaps lerp": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], fit_bias=False, gap_mode="lerp"),
    "offline bias 60 s knots, gaps lerp": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=60),
    "offline bias 30 s knots, gaps lerp": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=30),
    "offline bias 15 s knots, gaps lerp": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=15),
    "offline bias 30 s, gaps decay": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=30, gap_mode="decay"),
    "offline bias 30 s + residual 5 s": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=30, residual_s=5.0),
    "offline bias 30 s + residual 15 s": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=30, residual_s=15.0),
}

KEYS = ("att_org", "tilt", "rpe_r", "rpe1_r")
results = {}
files = eval_set(DEV_ROOMS)
for name, fn in METHODS.items():
    t0 = time.time()
    rows = []
    for key in files:
        n, s, m, sd = key
        room = load_room(n)
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        rows.append((f"{n}|{s}|{m}|{sd}", run(P, fn(P), room)))
    results[name] = rows
    show(name, summarise(rows), KEYS)
    print(f"     {(time.time()-t0)/len(files)*1e3:.0f} ms per file in Python", flush=True)

print("\nother bias shapes: mean/worst over dev rooms, seeds, gap modes")
extra = {}
for kind in ("sine", "euroc"):
    for name in ("baseline tilt filter", "low-gain tilt kp=0.02", "offline bias 60 s knots, gaps lerp",
                 "offline bias 30 s knots, gaps lerp", "offline bias 15 s knots, gaps lerp", "offline bias 30 s + residual 15 s"):
        rows = []
        for n in DEV_ROOMS:
            for m in ("delete", "zero"):
                for sd in (1, 2, 3):
                    room = load_room(n)
                    t, imu = variant_file(n, kind, m, sd)
                    P = prepare(t, imu)
                    rows.append(run(P, METHODS[name](P), room))
        v = {k: (float(np.mean([r[k] for r in rows])), float(np.max([r[k] for r in rows]))) for k in KEYS}
        extra[f"{kind}|{name}"] = v
        print(f"  {kind:6s}{name:36s} " + " ".join(f"{k} {v[k][0]:.2f}/{v[k][1]:.2f}" for k in KEYS), flush=True)
json.dump(dict(results=results, extra=extra), open(f"{RES}/h3_attitude.json", "w"))
