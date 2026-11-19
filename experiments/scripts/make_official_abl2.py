"""Which part of the organisers' degradation costs the most? The official script with one part switched off at a time
(columns in the designed order). Rooms 1-5 and four flights. Usage: make_official_abl.py"""
import sys, os
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/tools")
import pandas as pd
import common
import degrade_imu as D
OUT = common.EXP + "/tmp/cpp_abl2"
os.makedirs(OUT, exist_ok=True)
Z = dict(scale_rw_max=0.0, accel_bias_rw_max=0.0, gyro_bias_rw_max=0.0, accel_white_max=0.0, gyro_white_max=0.0, spike_prob_max=0.0, stuck_prob_max=0.0, drop_prob_max=0.0, burst_prob_max=0.0, dup_prob_max=0.0)
C = D.CONFIG
VAR = {
    "xdrop": dict(Z, drop_prob_max=C["drop_prob_max"]),
    "xburst": dict(Z, burst_prob_max=C["burst_prob_max"]),
    "xdup": dict(Z, dup_prob_max=C["dup_prob_max"]),
    "xstuck": dict(Z, stuck_prob_max=C["stuck_prob_max"]),
    "xspike": dict(Z, spike_prob_max=C["spike_prob_max"]),
}
files = [(f"room{n}", f"{common.ROOT}/data/tumvi_room{n}/imu0/data.csv") for n in common.DEV_ROOMS] + []
for name, path in files:
    df = pd.read_csv(path, header=0)
    cols = list(df.columns)
    d = df[[cols[0]] + cols[4:7] + cols[1:4]]
    for tag, ch in VAR.items():
        cfg = dict(D.CONFIG); cfg.update(ch)
        out, st = D.degrade(d, cfg, D.SEED)
        out[cols].to_csv(f"{OUT}/{name}_{tag}.csv", index=False, header=True)
    print(name, flush=True)
