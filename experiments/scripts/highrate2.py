"""Fast-IMU check that isolates the averaged solve: each 200 Hz test file is upsampled by holding every sample D times with a
zero-sum jitter of a few 1e-6 (so rows are not exact repeats, and the group average is the original sample). The result should
be close to the 200 Hz run of the same file. Usage: highrate2.py <binary>"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import common
from pipeline import prepare, PosScorer

BIN = sys.argv[1]
T = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/cpp_tests"
D_ = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/cpp_edge"
JIT = {2: [1, -1], 5: [2, 1, 0, -1, -2]}


def score(csv, out, room):
    tt, vv = common._read_csv_ns(csv)
    P = prepare(tt, vv)
    a = np.loadtxt(out, comments="#")
    k = P["ids"] != -1
    if k.sum() != len(a):
        return f"ROWS {len(a)} != {k.sum()}"
    pg = np.zeros((len(P["t"]), 3))
    pg[k] = a[:, 1:4]
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1))
    qg[k] = a[:, [7, 4, 5, 6]]
    s = PosScorer(P, room, qg)(pg)
    return f"ATE {s['ate_se3']:.3f} blind {s.get('blind_rel', np.nan):.3f}"


def run(csv):
    out = csv[:-4] + "_out.txt"
    r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, csv, out], capture_output=True, text=True)
    lines = r.stderr.strip().splitlines()
    sec, kb = map(float, lines[-1].split())
    return out, sec, kb, lines[-2] if len(lines) > 1 else ""


for name, rn in (("V101_clean", "V101"), ("MH03_full", "MH03"), ("room1_clean", 1), ("room3_full", 3)):
    room = common.load_room(rn)
    src = f"{T}/{name}.csv"
    t, v = common._read_csv_ns(src)
    out, sec, kb, info = run(src)
    dur = (t[-1] - t[0]) * 1e-9
    print(f"{name:12s}  200 Hz: {score(src, out, room)} | {kb/1024:.2f} MB {1000*sec/dur:.2f} ms/s | {info}", flush=True)
    dt = np.diff(t, append=t[-1] + int(np.median(np.diff(t))))
    for D in (2, 5):
        p = f"{D_}/{name}_x{D}.csv"
        if not os.path.exists(p):
            tu = (t[:, None] + (dt[:, None] * np.arange(D)[None]) // D).ravel()
            dead = (v == 0).all(1)
            vu = np.repeat(v, D, axis=0) + np.tile(np.array(JIT[D], float)[:, None] * 1e-6, (len(t), 6))
            vu[np.repeat(dead, D)] = 0.0
            common.write_csv(p, tu, vu)
        out, sec, kb, info = run(p)
        print(f"{name:12s} {200*D:4d} Hz: {score(p, out, room)} | {kb/1024:.2f} MB {1000*sec/dur:.2f} ms/s | {info}", flush=True)
