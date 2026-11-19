"""Error inside dropouts, for a C++ binary. For every stretch of dead rows inside a file (2 s gaps of the degradation):
  pos     position error of the rows inside the gap under the whole-file SE3 alignment (what ATE sees), against the rows outside
  local   error of the displacement since the last row before the gap (the part the bridge is responsible for), against
          two simple bridges built from the same output: hold the position, or a straight line between the two ends
  rot     error of the rotation across the gap (last row before to first row after)
Rows deleted from the file get no pose, so position inside a gap is scored only for files that keep the dead rows.
Usage: score_gaps.py <binary> <label> <tests|long> [name-filter]"""
import sys, os, glob, pickle, subprocess, importlib
BIN, LABEL, SET = sys.argv[1], sys.argv[2], sys.argv[3]
FILT = sys.argv[4] if len(sys.argv) > 4 else ""
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from scipy.spatial.transform import Rotation as Rot
if SET == "long":
    mX = importlib.import_module("74_unseen")
    loader = mX.load2
from ev import *
if SET == "tests":
    loader = load_room
OUTD = f"{EXP}/tmp/score_out/{LABEL}_{SET}"


def cases():
    out = []
    pat = f"{EXP}/tmp/cpp_tests/*.csv" if SET == "tests" else f"{EXP}/tmp/cpp_long/*.csv"
    for csv in sorted(glob.glob(pat)):
        name = os.path.basename(csv)[:-4]
        if name.endswith("_clean") or FILT not in name:
            continue
        n = name.split("_")[0]
        rn = (int(n[4:]) if n.startswith("room") else n) if SET == "tests" else name.rsplit("_", 1)[0]
        out.append((name, csv, rn))
    return out


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def one(case):
    name, csv, rn = case
    out = f"{OUTD}/{name}.txt"
    subprocess.run([BIN, csv, out], capture_output=True, text=True)
    a = np.loadtxt(out, comments="#")
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    room = loader(rn)
    keep = P["ids"] != -1
    pg = np.zeros((len(P["t"]), 3)); pg[keep] = a[:, 1:4]
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1)); qg[keep] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)
    e, g = pg[keep][sc.ei], sc.g
    r, tr = common.umeyama(e, g)
    al = (r @ e.T).T + tr
    row = np.nonzero(keep)[0][sc.ei]           # grid row of every matched sample
    tm = sc.t
    Rg = Rot.from_quat(room["gt_q"][sc.gi])
    Re = Rot.from_quat(qg[keep][:, [1, 2, 3, 0]][sc.ei])
    L = P["last_good"]
    dead = ~P["ok"]
    dead[L + 1:] = False
    o = dict(pos=[], out=[], loc=[], hold=[], line=[], rot=[], rot_zero=[], n_gap=0)
    ingap = np.zeros(len(row), bool)
    for s, en in runs(dead):
        if s == 0 or en > L or P["t"][en] - P["t"][s] < 0.5:
            continue
        m0 = np.searchsorted(row, s) - 1
        m1 = np.searchsorted(row, en)
        if m0 < 0 or m1 >= len(row) or tm[m1] - tm[m0] > (P["t"][en] - P["t"][s]) + 0.5:
            continue                          # ground truth missing around this gap
        o["n_gap"] += 1
        rel = (Rg[m0].inv() * Rg[m1]).inv() * (Re[m0].inv() * Re[m1])
        o["rot"].append(float(rel.magnitude() * 180 / np.pi))
        o["rot_zero"].append(float((Rg[m0].inv() * Rg[m1]).magnitude() * 180 / np.pi))
        m = np.arange(m0 + 1, m1)
        if len(m) < 5:
            continue                          # rows were deleted: no poses inside
        ingap[m] = True
        o["pos"].extend(np.linalg.norm(al[m] - g[m], axis=1))
        dg = g[m] - g[m0]
        o["loc"].extend(np.linalg.norm((al[m] - al[m0]) - dg, axis=1))
        o["hold"].extend(np.linalg.norm(dg, axis=1))
        w = ((tm[m] - tm[m0]) / (tm[m1] - tm[m0]))[:, None]
        o["line"].extend(np.linalg.norm((al[m1] - al[m0]) * w - dg, axis=1))
    pre = sc.pre & ~ingap
    o["out"] = np.linalg.norm(al[pre] - g[pre], axis=1)
    return name, {k: (np.asarray(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in o.items()}


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    with mp.get_context("fork").Pool(10) as p:
        R = dict(p.map(one, cases(), chunksize=1))
    pickle.dump(R, open(f"{EXP}/tmp/gaps_{LABEL}_{SET}.pkl", "wb"))
    rms = lambda x: float(np.sqrt((np.asarray(x) ** 2).mean())) if len(x) else float("nan")

    def grp(name):
        kind = name.rsplit("_", 1)[1]
        plat = "room" if name.startswith("room") else ("walk" if name.startswith("tv_") else "euroc")
        return f"{plat}_{kind}"
    G = {}
    for k, v in R.items():
        G.setdefault(grp(k), []).append(v)
    print(f"{LABEL} / {SET}")
    for gname in sorted(G):
        V = G[gname]
        cat = lambda key: np.concatenate([v[key] for v in V]) if any(len(v[key]) for v in V) else np.zeros(0)
        ng = sum(v["n_gap"] for v in V)
        if ng == 0:
            continue
        rot = cat("rot")
        print(f"  {gname:12s} gaps {ng:3d} | inside {rms(cat('pos')):.3f} m, outside {rms(cat('out')):.3f} m | since gap start: entry {rms(cat('loc')):.3f} "
              f"hold {rms(cat('hold')):.3f} line {rms(cat('line')):.3f} m | rotation across gap: rms {rms(rot):.1f} median {np.median(rot):.1f} deg "
              f"(no rotation: rms {rms(cat('rot_zero')):.1f} median {np.median(cat('rot_zero')):.1f})")
