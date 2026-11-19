"""High-rate IMUs (XR headsets run at 500-1000 Hz): EuRoC flights resampled to 400 Hz and 1000 Hz by linear interpolation,
then the 'full' degradation pattern is not applied (clean only). Checks memory, time, row count and ATE against the 200 Hz run.
Usage: highrate.py <binary>"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import common
from pipeline import prepare, PosScorer

BIN = sys.argv[1]
D = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/cpp_edge"
os.makedirs(D, exist_ok=True)
for n, hz in (("V101", 400), ("MH01", 1000), ("V101", 200)):
    room = common.load_room(n)
    t, imu = room["t_ns"], room["imu"]
    keep = t <= t[-1] - 5 * 10 ** 9
    t, imu = t[keep], imu[keep]
    tn = np.arange(t[0], t[-1], int(1e9 / hz), dtype=np.int64)
    v = np.stack([np.interp(tn, t, imu[:, i]) for i in range(6)], 1)
    p = f"{D}/{n}_{hz}hz.csv"
    if not os.path.exists(p):
        common.write_csv(p, tn, v)
    out = f"{D}/{n}_{hz}hz_out.txt"
    r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, p, out], capture_output=True, text=True)
    lines = r.stderr.strip().splitlines()
    sec, kb = map(float, lines[-1].split())
    tt, vv = common._read_csv_ns(p)
    P = prepare(tt, vv)
    a = np.loadtxt(out, comments="#")
    k = P["ids"] != -1
    pg = np.zeros((len(P["t"]), 3))
    pg[k] = a[:, 1:4]
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1))
    qg[k] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)
    dur = P["t"][P["last_good"]]
    print(f"{n} {hz:4d} Hz: {len(tt)} rows, {len(a)} poses (expected {k.sum()}), ATE {sc(pg)['ate_se3']:.3f}, "
          f"{kb/1024:.2f} MB, {1000*sec/dur:.2f} ms per s | {lines[-2] if len(lines) > 1 else ''}", flush=True)
