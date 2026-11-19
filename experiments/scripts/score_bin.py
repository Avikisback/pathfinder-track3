"""Runs a C++ binary over a test set and scores every output against ground truth: ATE (SE3 and first-pose aligned), RPE 1 s in
translation and rotation, attitude error (first-pose aligned, before the IMU dies), blind 5 s error, memory and time.
Results go to tmp/score_<label>_<set>.pkl so that two builds can be compared file by file (score_diff.py).
Sets: tests (the 64 parity files), bb (Blackbird, clean and full), long (22 unseen TUM-VI walks, clean and full).
Usage: score_bin.py <binary> <label> <tests|bb|long> [name-filter]"""
import sys, os, glob, pickle, subprocess, importlib
BIN, LABEL, SET = sys.argv[1], sys.argv[2], sys.argv[3]
FILT = sys.argv[4] if len(sys.argv) > 4 else ""
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from scipy.spatial.transform import Rotation as Rot
if SET == "bb":
    mX = importlib.import_module("75_blackbird")
    loader = mX.load2
elif SET == "long":
    mX = importlib.import_module("74_unseen")
    loader = mX.load2
from ev import *
if SET == "monado":
    def loader(n):
        d = np.load(f"{CACHE}/{n}.npz")
        return {k: d[k] for k in d.files}
elif SET not in ("bb", "long"):
    loader = load_room
OUTD = f"{EXP}/tmp/score_out/{LABEL}_{SET}"


def cases():
    out = []
    if SET not in ("bb", "long"):
        for csv in sorted(glob.glob(f"{EXP}/tmp/cpp_{SET}/*.csv")):
            name = os.path.basename(csv)[:-4]
            n = name.split("_")[0]
            out.append((name, csv, name.rsplit("_", 1)[0] if SET == "monado" else (int(n[4:]) if n.startswith("room") else n)))
    elif SET == "long":
        for csv in sorted(glob.glob(f"{EXP}/tmp/cpp_long/*.csv")):
            name = os.path.basename(csv)[:-4]
            out.append((name, csv, name.rsplit("_", 1)[0]))
    else:
        d = f"{EXP}/tmp/cpp_bb"
        os.makedirs(d, exist_ok=True)
        names = sorted("bb_" + x for x in os.listdir(mX.UN) if x.startswith(("test_", "train_")))
        for n in names:
            for k in mX.KS:
                csv = f"{d}/{n}_{k[0]}.csv"
                if not os.path.exists(csv):
                    t, imu, _ = mX.file_for(n, k)
                    common.write_csv(csv, t, imu)
                out.append((f"{n}_{k[0]}", csv, n))
    return [c for c in out if FILT in c[0]]


def one(case):
    name, csv, rn = case
    out = f"{OUTD}/{name}.txt"
    r = subprocess.run(["/usr/bin/time", "-f", "%e %M", BIN, csv, out], capture_output=True, text=True)
    lines = r.stderr.strip().splitlines()
    sec, kb = map(float, lines[-1].split())
    a = np.loadtxt(out, comments="#")
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    room = loader(rn)
    keep = P["ids"] != -1
    if keep.sum() != len(a):
        return name, dict(error=f"rows {len(a)} expected {int(keep.sum())}")
    pg = np.zeros((len(P["t"]), 3))
    pg[keep] = a[:, 1:4]
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1))
    qg[keep] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)
    o = sc(pg)
    st = sc(np.zeros_like(pg))
    # attitude, first-pose aligned, rows before the IMU dies; rotation RPE over 1 s
    Rg = Rot.from_quat(room["gt_q"][sc.gi])
    Re = Rot.from_quat(qg[keep][:, [1, 2, 3, 0]][sc.ei])
    pre = sc.pre
    W = Rg[0] * Re[0].inv()
    e = (Rg.inv() * (W * Re)).magnitude()[pre] * 180 / np.pi
    i, j = sc.i, sc.j
    m = pre[j]
    er = ((Rg[i].inv() * Rg[j]).inv() * (Re[i].inv() * Re[j])).magnitude()[m] * 180 / np.pi
    up_g = Rg.inv().apply([0, 0, 1.0])[pre]
    up_e = Re.inv().apply([0, 0, 1.0])[pre]
    cosang = np.clip((up_g * up_e).sum(1), -1, 1)
    o.update(att=float(np.sqrt((e ** 2).mean())), rpe_r=float(np.sqrt((er ** 2).mean())) if m.sum() else float("nan"),
             stat=st["ate_se3"], stat_blind=st.get("blind_rel", float("nan")), stat_org=st["ate_org"], stat_rpe=st["rpe_t"],
             stat_rpe_r=float(np.sqrt(((Rg[i].inv() * Rg[j]).magnitude()[m] ** 2).mean()) * 180 / np.pi) if m.sum() else float("nan"), sec=sec, kb=kb,
             ms_per_s=1000 * sec / P["t"][P["last_good"]], info=lines[-2] if len(lines) > 1 else "")
    return name, o


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    cs = cases()
    with mp.get_context("fork").Pool(10) as p:
        R = dict(p.map(one, cs, chunksize=1))
    pickle.dump(R, open(f"{EXP}/tmp/score_{LABEL}_{SET}.pkl", "wb"))
    bad = {k: v for k, v in R.items() if "error" in v}
    for k, v in bad.items():
        print("ERROR", k, v["error"])
    R = {k: v for k, v in R.items() if "error" not in v}

    def grp(name):
        kind = name.rsplit("_", 1)[1]
        plat = "room" if name.startswith("room") else ("walk" if name.startswith("tv_") else ("bb" if name.startswith("bb_") else ("headset_" + name.split("_")[2] if name.startswith("mo_") else "euroc")))
        return f"{plat}_{kind}"
    G = {}
    for k, v in R.items():
        G.setdefault(grp(k), []).append(v)
    print(f"{LABEL} / {SET}: {len(R)} files")
    for g in sorted(G):
        V = G[g]
        mean = lambda key: float(np.nanmean([v.get(key, np.nan) for v in V]))
        print(f"  {g:14s} n {len(V):2d} | ATE {mean('ate_se3'):.3f} (still {mean('stat'):.3f}) origin-ATE {mean('ate_org'):.3f} ({mean('stat_org'):.3f}) | RPE 1 s {mean('rpe_t'):.3f} m ({mean('stat_rpe'):.3f}) {mean('rpe_r'):.2f} deg ({mean('stat_rpe_r'):.1f}) | "
              f"attitude {mean('att'):.1f} deg | blind {mean('blind_rel'):.3f} (hold {mean('stat_blind'):.3f}) | {max(v['kb'] for v in V)/1024:.2f} MB {max(v['ms_per_s'] for v in V):.1f} ms/s")
