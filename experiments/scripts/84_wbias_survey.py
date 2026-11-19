"""Survey of wbias.py over the parity files and the bias-compensated files: what the estimator returns, how far it is from the
declared calibration (drones) or from zero (handheld), and whether the file has a rest block. Usage: 84_wbias_survey.py"""
import sys, os, glob, pickle
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from ev import *
from phys import hardened_rest, is_drone
from wbias import wbias, agree
C = pickle.load(open(EXP + "/cache/entry_constants.pkl", "rb"))


def one(csv):
    name = os.path.basename(csv)[:-4]
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    drone = is_drone(P["t"], P["imu"], P["ok"])
    cal = np.asarray(C["b0"]) if drone else np.zeros(3)
    tb_, mb = hardened_rest(P["t"], P["imu"], P["ok"])
    rest = None if tb_ is None else np.median(mb, 0)
    b, N, nw = wbias(P["t"], P["imu"][:, :3], P["imu"][:, 3:], P["ok"])
    n = P["imu"][P["ok"], 3:].mean(0)
    n /= np.linalg.norm(n)
    perp = lambda v: v - (v @ n) * n
    st = os.path.basename(os.path.dirname(csv))
    return st + "/" + name, dict(drone=drone, rest=rest, b=b, d=agree(b, N, cal), dp=float(np.linalg.norm(perp(b - cal))), bp=perp(b), nw=nw, cal=cal, set=st)


if __name__ == "__main__":
    files = sorted(glob.glob(EXP + "/tmp/cpp_tests/*.csv")) + sorted(glob.glob(EXP + "/tmp/cpp_comp/*.csv"))
    with mp.get_context("fork").Pool(16) as p:
        R = dict(p.map(one, files, chunksize=1))
    pickle.dump(R, open(EXP + "/tmp/wbias_survey.pkl", "wb"))
    print(len(R), "files")
