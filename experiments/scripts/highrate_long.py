"""Windowed path at 1 kHz: a 334 s TUM-VI walk held 5x with zero-sum jitter (as highrate2.py), against its 200 Hz run.
Usage: highrate_long.py <binary>"""
import sys, os, subprocess, importlib
import numpy as np
BIN = sys.argv[1]
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m74 = importlib.import_module("74_unseen")
import common
from pipeline import prepare, PosScorer

L = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/cpp_long"
D, J = 5, [2, 1, 0, -1, -2]
for name in ("tv_corridor2_clean", "tv_corridor2_full"):
    src = f"{L}/{name}.csv"
    t, v = common._read_csv_ns(src)
    p = f"/tmp/bench/{name}_x5.csv"
    if not os.path.exists(p):
        dt = np.diff(t, append=t[-1] + int(np.median(np.diff(t))))
        tu = (t[:, None] + (dt[:, None] * np.arange(D)[None]) // D).ravel()
        dead = (v == 0).all(1)
        vu = np.repeat(v, D, axis=0) + np.tile(np.array(J, float)[:, None] * 1e-6, (len(t), 6))
        vu[np.repeat(dead, D)] = 0.0
        common.write_csv(p, tu, vu)
    room = m74.load2("tv_corridor2")
    for csv in (src, p):
        out = "/tmp/bench/hl_out.txt"
        r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, csv, out], capture_output=True, text=True)
        lines = r.stderr.strip().splitlines()
        sec, kb = map(float, lines[-1].split())
        tt, vv = common._read_csv_ns(csv)
        P = prepare(tt, vv)
        a = np.loadtxt(out, comments="#")
        k = P["ids"] != -1
        pg = np.zeros((len(P["t"]), 3))
        pg[k] = a[:, 1:4]
        qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1))
        qg[k] = a[:, [7, 4, 5, 6]]
        ate = PosScorer(P, room, qg)(pg)["ate_se3"]
        dur = P["t"][P["last_good"]]
        print(f"{os.path.basename(csv):28s} rows {len(tt):7d} poses {len(a)} (expected {k.sum()}) ATE {ate:.3f} | "
              f"{kb/1024:.2f} MB {1000*sec/dur:.2f} ms/s | {lines[-2] if len(lines) > 1 else ''}", flush=True)
