"""Long files: the C++ entry (windows of 4 min for files over 5 min) against the Python whole-file solve (74_unseen.py results)
on the unseen TUM-VI walks, clean and 'full' degradation. Also peak memory and time per file.
Usage: compare_long.py <binary> [name-filter]
"""
import sys, os, subprocess, pickle, importlib
BIN = sys.argv[1]
FILT = sys.argv[2] if len(sys.argv) > 2 else ""
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m74 = importlib.import_module("74_unseen")
import numpy as np
import common
from ev import *

OUTD = EXP + "/tmp/cpp_long"
os.makedirs(OUTD, exist_ok=True)


def read_tum(p):
    a = np.loadtxt(p, comments="#")
    return a[:, 0], a[:, 1:4], a[:, 4:8]


if __name__ == "__main__":
    py = pickle.load(open(f"{RES}/unseen_tumvi.pkl", "rb"))
    names = sorted({k[0] for k in py if FILT in k[0]})
    rows = []
    for n in names:
        for k in m74.KS:
            csv = f"{OUTD}/{n}_{k[0]}.csv"
            if not os.path.exists(csv):
                t, imu, _ = m74.file_for(n, k)
                common.write_csv(csv, t, imu)
            out = f"{OUTD}/{n}_{k[0]}_cpp.txt"
            r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, csv, out], capture_output=True, text=True)
            lines = r.stderr.strip().splitlines()
            sec, kb = map(float, lines[-1].split())
            info = lines[-2] if len(lines) > 1 else ""
            t_ns, imu = common._read_csv_ns(csv)
            P = prepare(t_ns, imu)
            room = m74.load2(n)
            ta, pa, qa = read_tum(out)
            keep = P["ids"] != -1
            if keep.sum() != len(pa):
                print(n, k[0], "ROW COUNT", keep.sum(), len(pa), info)
                continue
            pg = np.zeros((len(P["t"]), 3))
            pg[keep] = pa
            qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1))
            qg[keep] = qa[:, [3, 0, 1, 2]]
            sc = PosScorer(P, room, qg)
            ate = sc(pg)["ate_se3"]
            o = py[(n, k[0])]
            dur = P["t"][P["last_good"]]
            rows.append(dict(n=n, k=k[0], cpp=ate, py=o["new"], stat=o["stationary"], dur=dur, kb=kb, ms=1000 * sec / dur))
            print(f"{n:18s} {k[0]:5s} {dur:6.0f}s | ATE cpp {ate:.2f} python {o['new']:.2f} stationary {o['stationary']:.2f} | "
                  f"{kb/1024:.2f} MB {1000*sec/dur:.2f} ms/s | {info}", flush=True)
    for kk in ("clean", "full"):
        r = [x for x in rows if x["k"] == kk]
        if r:
            print(f"MEAN {kk} ({len(r)}): cpp {np.mean([x['cpp'] for x in r]):.3f} python {np.mean([x['py'] for x in r]):.3f} "
                  f"stationary {np.mean([x['stat'] for x in r]):.3f} | cpp beats stationary {np.mean([x['cpp'] < x['stat'] for x in r]):.2f}")
    if rows:
        print(f"peak memory {max(x['kb'] for x in rows)/1024:.2f} MB, worst {max(x['ms'] for x in rows):.2f} ms per s")
    pickle.dump(rows, open(f"{OUTD}/compare_long.pkl", "wb"))
