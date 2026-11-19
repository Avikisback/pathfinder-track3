"""The 22 TUM-VI walks (corridor, magistrale, slides, outdoors; 1.5-24 min; nothing was tuned on them) degraded with
the organisers' script (columns swapped for it and back, "sw" as in make_official.py), seed 42. TUM-VI has motion
capture only where the walks start and end, so the scores use those stretches.
Writes tmp/tv_off/<name>_sw.csv and <name>_sw.gt.npz for score_off.py (OFF_DIR=tv_off). Usage: make_tv_off.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/tools")
import common               # noqa: E402
import degrade_imu as D     # noqa: E402

OUT = common.EXP + "/tmp/tv_off"
os.makedirs(OUT, exist_ok=True)
names = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
         "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
for f in sorted(glob.glob(common.CACHE + "/tv_*.npz")):
    r = np.load(f)
    name = os.path.basename(f)[3:-4] + "_sw"            # e.g. corridor1_sw
    df = pd.DataFrame({"#timestamp [ns]": r["t_ns"].astype(np.int64)})
    for k, nm in enumerate(names):
        df[nm] = r["imu"][:, k]
    cols = list(df.columns)
    out, st = D.degrade(df[[cols[0]] + cols[4:7] + cols[1:4]], dict(D.CONFIG), D.SEED)
    out[cols].to_csv(f"{OUT}/{name}.csv", index=False, header=True)
    np.savez(f"{OUT}/{name}.gt.npz", gt_t_ns=r["gt_t_ns"], gt_p=r["gt_p"], gt_q=r["gt_q"])
    print(name, "%.0f s" % ((r["t_ns"][-1] - r["t_ns"][0]) * 1e-9), "rows", st["rows_in"], "->", st["rows_out"], flush=True)
