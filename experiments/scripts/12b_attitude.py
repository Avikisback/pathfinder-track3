"""H3: prior and noise settings for the bias fit. Scores with the blind tail left out. A third damage shape,
'ramp only', removes gaps to isolate bias from gap losses."""
import sys, time, json
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *
KEYS = ("att_org_pre", "tilt_pre", "rpe_r_pre", "att_org", "rpe_r")
M = {"baseline tilt filter": lambda P: baseline_attitude(P["t"], P["imu"], P["ok"]),
     "gaps lerp only + residual 5 s": lambda P: offline_attitude(P["t"], P["imu"], P["ok"], fit_bias=False, gap_mode="lerp")}
for sr in (0.03, 0.1, 0.3):
    for sb in (0.003, 0.01, 0.03):
        for ks in (60,):
            M[f"fit knots {ks} s_r {sr} s_b {sb} resid 5"] = (lambda sr, sb, ks: lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=ks, s_r=sr, sigma_b=sb, residual_s=5.0))(sr, sb, ks)
M["fit knots 30 s_r 0.1 s_b 0.01 resid 5"] = lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=30, s_r=0.1, sigma_b=0.01, residual_s=5.0)
M["no fit, gaps lerp, tilt only resid 5"] = lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=60, s_r=0.1, sigma_b=1e-5, residual_s=5.0)
M["no fit, gaps decay, tilt only resid 5"] = lambda P: offline_attitude(P["t"], P["imu"], P["ok"], knot_s=60, s_r=0.1, sigma_b=1e-5, residual_s=5.0, gap_mode="decay")
files = eval_set(DEV_ROOMS)
res = {}
for name, fn in M.items():
    rows = []
    for (n, s, m, sd) in files:
        room = load_room(n); t, imu, _ = eval_file(n, s, m, sd); P = prepare(t, imu)
        rows.append((f"{n}|{s}|{m}|{sd}", score_all(P, room, np.zeros((len(P["t"]), 3)), fn(P))))
    res[name] = rows
    show(name, summarise(rows), KEYS)
json.dump(res, open(f"{RES}/h3b.json", "w"))
