"""Samsung Odyssey+ headset recordings (Monado SLAM dataset, tools/get_monado.py) as test files for the C++ entry.
For every recording and both IMU versions (raw = data.raw.csv, cal = data.csv with the dataset's own bias calibration
applied) this caches the ground truth and writes two CSVs: clean (only the last 5 s cut off) and full (our guessed
degradation, tools/degrade.py). The IMU is 1 kHz, so this is also the first real test of the fast-IMU path.
Nothing in the entry was tuned on this data. Usage: make_monado.py"""
import sys, os
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from ev import *

SRC = "/mnt/e/clipping-pipeline/xr_track3/data/unseen/monado"
OUT = EXP + "/tmp/cpp_monado"
os.makedirs(OUT, exist_ok=True)
for d in sorted(os.listdir(SRC)):
    seq = d.split("_")[0]
    gt_t, gt = common._read_csv_ns(f"{SRC}/{d}/mav0/gt/data.csv")
    for tag, fn in (("raw", "data.raw.csv"), ("cal", "data.csv")):
        f = f"{SRC}/{d}/mav0/imu0/{fn}"
        if not os.path.exists(f):
            continue
        t, imu = common._read_csv_ns(f)
        room = dict(t_ns=t, imu=imu, gt_t_ns=gt_t, gt_p=gt[:, :3], gt_q=gt[:, [4, 5, 6, 3]])
        name = f"mo_{seq}_{tag}"
        np.savez(f"{CACHE}/{name}.npz", **room)
        for kind, mode in (("clean", "delete"), ("full", "zero")):
            csv = f"{OUT}/{name}_{kind}.csv"
            if not os.path.exists(csv):
                tt, im, _ = common.degrade(room, 1, mode, kind)
                common.write_csv(csv, tt, im)
        dt = np.median(np.diff(t)) * 1e-9
        spread = float(np.sqrt(((room["gt_p"] - room["gt_p"].mean(0)) ** 2).sum(1).mean()))
        print(f"{name}: {len(t)} rows, {1/dt:.0f} Hz, {(t[-1]-t[0])*1e-9:.0f} s, path spread {spread:.2f} m, mean gyro {np.round(imu[:, :3].mean(0), 4)}", flush=True)
