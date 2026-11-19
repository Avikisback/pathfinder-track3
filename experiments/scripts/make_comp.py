"""EuRoC test files with the gyro bias already removed ("compensated angular velocity", as the deck describes the input).
The public EuRoC gyro carries a bias of up to 0.08 rad/s, and the entry starts drones from a declared calibration of that
size. If the organisers hand out compensated data, that calibration is wrong. This writes clean and full versions of every
flight with the true constant bias (median of the ground-truth bias states) subtracted from every live row.
Usage: make_comp.py"""
import sys, os, importlib
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from ev import *
m77 = importlib.import_module("77_lever")
OUT = EXP + "/tmp/cpp_comp"
os.makedirs(OUT, exist_ok=True)
for n in SEQS:
    b = np.median(load_room(n)["gt_bw"], 0)
    for k in (("clean", "delete", 1), ("full", "zero", 1)):
        t, imu, room = m77.m72.file_for2(n, k)
        imu = np.array(imu, float)
        live = np.abs(imu).sum(1) > 0
        imu[live, :3] -= b
        common.write_csv(f"{OUT}/{n}_{k[0]}.csv", t, imu)
    print(n, "bias removed", np.round(b, 4), flush=True)
