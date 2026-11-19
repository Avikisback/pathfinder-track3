"""EuRoC test files for the degradation shapes that the parity set does not hold (mild, burst, steps, random walk, sine,
warm-up, long gaps, triple noise), so that a C++ build can be scored on all twelve shapes. Usage: make_kinds.py"""
import sys, os, importlib
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import common
from ev import *
m77 = importlib.import_module("77_lever")
OUT = EXP + "/tmp/cpp_kinds"
KS = [("mild", "zero", 1), ("burst", "x", 1), ("steps", "x", 1), ("randomwalk", "x", 1), ("sine", "x", 1),
      ("warmup", "x", 1), ("longgap", "x", 1), ("noisy", "x", 1)]


def one(job):
    n, k = job
    f = f"{OUT}/{n if isinstance(n, str) else chr(114) + chr(111) + chr(111) + chr(109) + str(n)}_{k[0]}.csv"
    if not os.path.exists(f):
        t, imu, room = m77.m72.file_for2(n, k)
        common.write_csv(f, t, imu)
    return f


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with mp.get_context("fork").Pool(12) as p:
        r = p.map(one, [(n, k) for n in SEQS + DEV_ROOMS for k in KS], chunksize=1)
    print(len(r), "files in", OUT)
