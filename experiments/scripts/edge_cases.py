"""Inputs that must not crash the entry. Writes CSVs into tmp/cpp_edge, runs the binary on each, checks one pose per input row
plus the blind tail, finite values and unit quaternions. Usage: edge_cases.py <binary>"""
import sys, os, subprocess
import numpy as np
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import common

BIN = sys.argv[1]
D = "/mnt/e/clipping-pipeline/xr_track3/experiments/tmp/cpp_edge"
os.makedirs(D, exist_ok=True)
room = common.load_room("V101")
t, imu = room["t_ns"], room["imu"]
rng = np.random.default_rng(3)


def write(name, tt, vv, raw_lines=None):
    p = f"{D}/{name}.csv"
    if raw_lines is not None:
        open(p, "w").write(raw_lines)
    else:
        common.write_csv(p, tt, vv)
    return p


cases = {}
cases["short_8s"] = write("short_8s", t[:1600], imu[:1600])
cases["short_25s"] = write("short_25s", t[:5000], imu[:5000])
v = imu[:20000].copy()
v[rng.random(len(v)) < 0.02] = np.nan
cases["nan_rows"] = write("nan_rows", t[:20000], v)
v = imu[:20000].copy()
v[12000:] = 0.0
cases["dead_from_middle"] = write("dead_from_middle", t[:20000], v)
cases["header_only"] = write("header_only", None, None, "#timestamp [ns],wx,wy,wz,ax,ay,az\n")
v = imu[:20000].copy()
v[5000:5400] = v[4999]
cases["frozen_block"] = write("frozen_block", t[:20000], v)
tt = t[:20000].copy()
tt[8000:] += 3_000_000_000  # 3 s hole in the timestamps
cases["time_hole"] = write("time_hole", tt, imu[:20000])
bb = "/mnt/e/clipping-pipeline/xr_track3/data/unseen/blackbird/conv/test_egg_maxSpeed8p0/imu0/data.csv"
if os.path.exists(bb):
    cases["blackbird_100hz"] = bb

for name, p in cases.items():
    out = f"{D}/{name}_out.txt"
    r = subprocess.run([BIN, p, out], capture_output=True, text=True)
    n_in = sum(1 for line in open(p) if line.strip() and not line.startswith("#"))
    msg = f"exit {r.returncode}"
    if os.path.exists(out) and r.returncode == 0:
        a = np.loadtxt(out, comments="#", ndmin=2)
        if a.size == 0:
            msg += f", 0 poses for {n_in} rows"
        else:
            qn = np.linalg.norm(a[:, 4:8], axis=1)
            msg += (f", {len(a)} poses for {n_in} rows, finite {np.isfinite(a).all()}, |q|-1 max {np.abs(qn - 1).max():.1e}, "
                    f"time increasing {bool((np.diff(a[:, 0]) > 0).all())}, max |p| {np.abs(a[:, 1:4]).max():.2f} m")
    print(f"{name:18s} {msg} | {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
