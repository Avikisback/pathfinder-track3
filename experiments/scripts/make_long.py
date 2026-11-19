"""Long handheld test files with ground truth, for the windowed solve (files over WIN_S seconds).
A recording played backwards is real motion: positions retrace the path, specific force is unchanged, the gyro changes
sign. Joining forward, reversed, forward copies of a TUM-VI room (which starts and ends at rest) gives a long, continuous
recording with exact ground truth. Each long file is then degraded with the organisers' script (columns swapped for it
and back, as in make_official.py "sw"), seed 42.
Writes tmp/long_off/<name>_sw.csv and <name>_sw.gt.npz (gt_t_ns, gt_p, gt_q x y z w). Usage: make_long.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/tools")
import common               # noqa: E402
import degrade_imu as D     # noqa: E402

OUT = common.EXP + "/tmp/long_off"
os.makedirs(OUT, exist_ok=True)


def build(n, copies):
    r = common.load_room(n)
    t, imu = r["t_ns"].astype(np.int64), r["imu"]
    gt_t, gp, gq = r["gt_t_ns"].astype(np.int64), r["gt_p"], r["gt_q"]
    keep = (gt_t >= t[0]) & (gt_t <= t[-1])
    gt_t, gp, gq = gt_t[keep], gp[keep], gq[keep]
    span = int(t[-1] - t[0]) + int(np.median(np.diff(t)))
    T, I, GT, GP, GQ = [], [], [], [], []
    for c in range(copies):
        base = c * span
        if c % 2 == 0:
            T.append(t - t[0] + base)
            I.append(imu)
            GT.append(gt_t - t[0] + base)
            GP.append(gp)
            GQ.append(gq)
        else:
            T.append(t[-1] - t[::-1] + base)
            I.append(imu[::-1] * np.array([-1, -1, -1, 1, 1, 1.0]))     # gyro changes sign, accel does not
            GT.append(t[-1] - gt_t[::-1] + base)
            GP.append(gp[::-1])
            GQ.append(gq[::-1])
    t0 = int(r["t_ns"][0])
    T = np.concatenate(T) + t0
    I = np.concatenate(I)
    df = pd.DataFrame({"#timestamp [ns]": T})
    names = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
             "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
    for k, nm in enumerate(names):
        df[nm] = I[:, k]
    cols = list(df.columns)
    d = df[[cols[0]] + cols[4:7] + cols[1:4]]
    out, st = D.degrade(d, dict(D.CONFIG), D.SEED)
    out = out[cols]
    name = f"long{n}x{copies}_sw"
    out.to_csv(f"{OUT}/{name}.csv", index=False, header=True)
    np.savez(f"{OUT}/{name}.gt.npz", gt_t_ns=np.concatenate(GT) + t0, gt_p=np.concatenate(GP), gt_q=np.concatenate(GQ))
    print(name, "%.0f s" % ((T[-1] - T[0]) * 1e-9), "rows", st["rows_in"], "->", st["rows_out"], "bursts", st["dropped_burst_gaps"], flush=True)


for n, copies in ((1, 2), (2, 3), (3, 3), (4, 5), (5, 2)):
    build(n, copies)
