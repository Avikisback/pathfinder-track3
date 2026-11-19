"""Rotated sensor axes break the drone path. Suspect: the declared gyro calibration b0 is in the original sensor frame.
Fix candidate: take b0 from the file's own rest on the ground (rotors off) when one exists, else zero."""
import sys, os, pickle, time, multiprocessing as mp
OUT = "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v15"
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import rw_evidence as rwe
import ev
from ev import *
import robust_shapes as rs
import importlib
m72 = importlib.import_module("72_freeze2")
ev.file_for = m72.file_for2
rwe.file_for = m72.file_for2
from phys import hardened_rest


def own_b0(n):
    return None


def b0_rest(t, imu, ok, n=None):
    """Keep the declared calibration when the rest on the ground agrees with it; otherwise the axes are not the ones it was
    made for, and the rest reading is used instead."""
    cal = _calib(n)
    tb, mb = hardened_rest(t, imu, ok)
    if tb is None:
        return cal
    rest = np.median(mb, 0)
    gap = float(np.linalg.norm(rest - cal))
    print("gap", n, round(gap, 4), flush=True)
    return cal if gap < 0.02 else rest


_calib = m41.calib_excl
MODE = {"m": "rest"}


def calib_patch(n):
    return CUR["b0"]


CUR = {}


def work(job):
    n, k = job
    t, imu, room = m72.file_for2(n, k)
    P = prepare(t, imu)
    CUR["b0"] = b0_rest(P["t"], P["imu"], P["ok"], n)
    m41.calib_excl = calib_patch
    rwe.m41.calib_excl = calib_patch
    return rwe.work((n, k, "L8rw_eb1"))


if __name__ == "__main__":
    kinds = [("rot", "x", 1), ("hold", "x", 1)] + rwe.KEYS
    jobs = [(n, k) for n in SEQS for k in kinds]
    with mp.get_context("fork").Pool(16) as p:
        p.map(work, jobs, chunksize=1)
    new = lambda n, k: pickle.load(open(rwe.fname(n, k, "L8rw_eb1"), "rb"))
    ev.KNOWN = rwe.KEYS_KNOWN
    ev.NAMES = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine", "rot", "hold")
    r = evaluate(SEQS, EU_FOLDS, kinds, new, cfg=m72.CFG2)
    print("EuRoC evidence drift, b0 from own rest: final " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, 'final').items()))
    print("                                      nocheck " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, 'nocheck').items()))
