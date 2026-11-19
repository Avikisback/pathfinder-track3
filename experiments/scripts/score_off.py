"""Scores an entry on the files made with the organisers' script (make_official.py), by the organisers' rules (FAQ, 17 Nov):
exactly one pose per input row, in order, with the input timestamp, and nothing appended. Independent of our own pipeline code.
  ATE        SE(3)-aligned (Umeyama) position RMSE over every row that has ground truth, blind rows included
  RPE 1 s    position and rotation, rows before the failure
  blind      error of the displacement since the last pose before the failure, aligned on the rows before it
  attitude   first-pose aligned, rows before the failure
"still" is the same score for one constant pose.
Usage: score_off.py <command...> -- <label> [name-filter]     e.g.  score_off.py ./pathfinder -- cpp _sw
       the command is run as: <command...> <in.csv> <out.txt>"""
import sys, os, glob, pickle, subprocess, time
i = sys.argv.index("--")
CMD, LABEL = sys.argv[1:i], sys.argv[i + 1]
FILT = sys.argv[i + 2] if len(sys.argv) > i + 2 else ""
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from scipy.spatial.transform import Rotation as Rot

SRC = common.EXP + "/tmp/" + os.environ.get("OFF_DIR", "cpp_off")
OUTD = common.EXP + f"/tmp/score_out/{LABEL}_off"


def ate(e, g):
    if np.ptp(e, axis=0).max() < 1e-9:
        return float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean())), np.eye(3), g.mean(0) - e.mean(0)
    r, tr = common.umeyama(e, g)
    return float(np.sqrt((((r @ e.T).T + tr - g) ** 2).sum(1).mean())), r, tr


def one(csv):
    name = os.path.basename(csv)[:-4]
    n = name.split("_")[0]
    out = f"{OUTD}/{name}.txt"
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-f", "%e %M"] + CMD + [csv, out], capture_output=True, text=True)
    lines = r.stderr.strip().splitlines()
    try:
        sec, kb = map(float, lines[-1].split())
    except Exception:
        sec, kb = time.time() - t0, 0.0
    raw = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
    tin = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    o = dict(name=name, rows=len(tin), sec=sec, kb=kb, info=lines[-2] if len(lines) > 1 else "")
    if not os.path.exists(out):
        return name, dict(o, error="no output: " + (lines[0][:120] if lines else ""))
    rows = [l.split() for l in open(out) if l.strip() and l[0] != "#"]
    if len(rows) != len(tin):
        return name, dict(o, error=f"{len(rows)} poses for {len(tin)} rows")
    tout = np.array([int(x[0].split(".")[0]) * 10**9 + int((x[0].split(".")[1] + "000000000")[:9]) if "." in x[0] else int(x[0]) * 10**9 for x in rows], dtype=np.int64)
    o["ts_mismatch"] = int((tout != tin).sum())
    a = np.array([[float(v) for v in x[1:8]] for x in rows])
    o["finite"] = bool(np.isfinite(a).all())
    if not o["finite"]:
        return name, dict(o, error="non-finite output")
    if os.path.exists(csv[:-4] + ".gt.npz"):           # files with their own ground truth (make_long.py)
        room = dict(np.load(csv[:-4] + ".gt.npz"))
    else:
        room = common.load_room(int(n[4:]) if n.startswith("room") else n)
    gt_t = room["gt_t_ns"]
    j = np.clip(np.searchsorted(gt_t, tin), 1, len(gt_t) - 1)
    j = j - ((tin - gt_t[j - 1]) < (gt_t[j] - tin))
    m = np.abs(gt_t[j] - tin) < 5_000_000
    t = (tin[m] - tin[0]) * 1e-9
    e, g = a[m, :3], room["gt_p"][j[m]]
    Re, Rg = Rot.from_quat(a[m, 3:7]), Rot.from_quat(room["gt_q"][j[m]])
    tf = (tin[-1] - tin[0]) * 1e-9 - 5.0
    pre = t < tf
    o["ate"], _, _ = ate(e, g)
    o["ate_pre"], r2, tr2 = ate(e[pre], g[pre])
    o["still"] = float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))
    al = (r2 @ e.T).T + tr2
    k = np.nonzero(pre)[0][-1]
    tail = ~pre
    if tail.sum() >= 10:
        o["blind"] = float(np.sqrt((((al[tail] - al[k]) - (g[tail] - g[k])) ** 2).sum(1).mean()))
        o["blind_still"] = float(np.sqrt(((g[tail] - g[k]) ** 2).sum(1).mean()))
    # 1 s pairs on a 10 Hz pick of the rows before the failure
    pick = np.unique(np.searchsorted(t[pre], np.arange(t[0], t[pre][-1], 0.1)))
    tp = t[pre][pick]
    jj = np.searchsorted(tp, tp + 1.0)
    ok = (jj < len(tp))
    ok[ok] &= np.abs(tp[jj[ok]] - tp[ok] - 1.0) < 0.06
    ii, jj = np.nonzero(ok)[0], jj[ok]
    ep, gp, Rep, Rgp = e[pre][pick], g[pre][pick], Re[np.nonzero(pre)[0][pick]], Rg[np.nonzero(pre)[0][pick]]
    de = Rep[ii].inv().apply(ep[jj] - ep[ii]); dg = Rgp[ii].inv().apply(gp[jj] - gp[ii])
    o["rpe_t"] = float(np.sqrt(((de - dg) ** 2).sum(1).mean()))
    o["rpe_t_still"] = float(np.sqrt((dg ** 2).sum(1).mean()))
    o["rpe_r"] = float(np.degrees(np.sqrt((((Rgp[ii].inv() * Rgp[jj]).inv() * (Rep[ii].inv() * Rep[jj])).magnitude() ** 2).mean())))
    o["rpe_r_still"] = float(np.degrees(np.sqrt(((Rgp[ii].inv() * Rgp[jj]).magnitude() ** 2).mean())))
    W = Rg[0] * Re[0].inv()
    o["att"] = float(np.degrees(np.sqrt(((Rg.inv() * (W * Re)).magnitude()[pre] ** 2).mean())))
    o["dur"] = (tin[-1] - tin[0]) * 1e-9
    o["ms_per_s"] = 1000 * sec / o["dur"]
    return name, o


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    files = [f for f in sorted(glob.glob(SRC + "/*.csv")) if FILT in os.path.basename(f)]
    with mp.get_context("fork").Pool(8) as p:
        R = dict(p.map(one, files, chunksize=1))
    pickle.dump(R, open(common.EXP + f"/tmp/score_{LABEL}_off.pkl", "wb"))
    for k, v in R.items():
        if "error" in v:
            print("ERROR", k, v["error"], "|", v.get("info", "")[:100])
    good = {k: v for k, v in R.items() if "error" not in v}
    G = {}
    for k, v in good.items():
        G.setdefault(("room" if k.startswith("room") else "euroc") + "_" + k.split("_")[1], []).append(v)
    print(f"{LABEL}: {len(good)} of {len(R)} files scored")
    for g in sorted(G):
        V = G[g]
        mean = lambda key: float(np.nanmean([v.get(key, np.nan) for v in V]))
        print(f"  {g:9s} n {len(V):2d} | ATE {mean('ate'):.3f} (still {mean('still'):.3f}; better on {sum(v['ate'] < v['still'] for v in V)}/{len(V)}) | RPE 1 s {mean('rpe_t'):.3f} m ({mean('rpe_t_still'):.3f}) {mean('rpe_r'):.1f} deg ({mean('rpe_r_still'):.1f}) | "
              f"blind {mean('blind'):.3f} (hold {mean('blind_still'):.3f}) | attitude {mean('att'):.0f} deg | ts mismatch {sum(v['ts_mismatch'] for v in V)} | {max(v['kb'] for v in V)/1024:.1f} MB {max(v['ms_per_s'] for v in V):.1f} ms/s")
