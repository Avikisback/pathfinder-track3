"""Runs the C++ entry on every test file in tmp/cpp_tests and compares it with the Python reference (ref_entry.py):
position and attitude differences row by row, ATE / RPE / blind error of both against ground truth, peak memory and time.
Usage: compare_cpp.py <binary> [name-filter]
"""
import sys, os, re, subprocess, time, glob, pickle
sys.argv = sys.argv[:1] + ["/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"] + sys.argv[1:]
BIN = sys.argv[2]
FILT = sys.argv[3] if len(sys.argv) > 3 else ""
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
from ev import *
from scipy.spatial.transform import Rotation as Rot

TESTS = EXP + "/tmp/cpp_tests"


def read_tum(p):
    a = np.loadtxt(p, comments="#")
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def room_of(name):
    n = name.split("_")[0]
    return int(n[4:]) if n.startswith("room") else n


def main():
    rows = []
    for csv in sorted(glob.glob(f"{TESTS}/*.csv")):
        name = os.path.basename(csv)[:-4]
        if FILT not in name:
            continue
        ref = f"{TESTS}/{name}_py.txt"
        if not os.path.exists(ref):
            continue
        out = f"/tmp/cmp_{name}.txt"
        t0 = time.time()
        r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, csv, out], capture_output=True, text=True)
        wall = time.time() - t0
        tail = r.stderr.strip().splitlines()
        sec, kb = map(float, tail[-1].split())
        ta, pa, qa = read_tum(out)
        tb, pb, qb = read_tum(ref)
        if len(ta) != len(tb):
            print(name, "ROW COUNT DIFFERS", len(ta), len(tb))
            continue
        dp = np.sqrt(((pa - pb) ** 2).sum(1))
        ang = (Rot.from_quat(qa).inv() * Rot.from_quat(qb)).magnitude() * 180 / np.pi
        # scores against ground truth
        t_ns, imu = common._read_csv_ns(csv)
        P = prepare(t_ns, imu)
        room = load_room(room_of(name))
        keep = P["ids"] != -1
        sc = {}
        for nm, p, q in (("cpp", pa, qa), ("py", pb, qb)):
            pg = np.zeros((len(P["t"]), 3))
            pg[keep] = p
            qg = np.zeros((len(P["t"]), 4))
            qg[keep] = q[:, [3, 0, 1, 2]]
            qg[~keep] = [1, 0, 0, 0]
            s = PosScorer(P, room, qg)
            sc[nm] = s(pg)
        dur = P["t"][P["last_good"]]
        o = dict(name=name, dp_rms=float(np.sqrt((dp ** 2).mean())), dp_max=float(dp.max()), ang_max=float(ang.max()),
                 ate_cpp=sc["cpp"]["ate_se3"], ate_py=sc["py"]["ate_se3"], blind_cpp=sc["cpp"].get("blind_rel", np.nan),
                 blind_py=sc["py"].get("blind_rel", np.nan), kb=kb, sec=sec, ms_per_s=1000 * sec / dur, dur=dur)
        rows.append(o)
        print(f"{name:22s} dp rms {o['dp_rms']*1000:8.3f} mm max {o['dp_max']*1000:8.2f} mm | ang max {o['ang_max']:.4f} deg | "
              f"ATE cpp {o['ate_cpp']:.3f} py {o['ate_py']:.3f} | blind {o['blind_cpp']:.3f} {o['blind_py']:.3f} | "
              f"{kb/1024:.2f} MB {o['ms_per_s']:.2f} ms/s", flush=True)
    if rows:
        print(f"\nSUMMARY {len(rows)} files: worst position diff {max(r['dp_max'] for r in rows)*1000:.2f} mm, "
              f"mean ATE cpp {np.mean([r['ate_cpp'] for r in rows]):.4f} py {np.mean([r['ate_py'] for r in rows]):.4f}, "
              f"peak memory {max(r['kb'] for r in rows)/1024:.2f} MB, worst {max(r['ms_per_s'] for r in rows):.2f} ms per s of IMU")
        pickle.dump(rows, open(f"{TESTS}/compare.pkl", "wb"))


if __name__ == "__main__":
    main()
