"""Oracle: how much of the handheld loss under dropouts is the heading lost across each gap?

Clean rooms score 0.44 m, the same rooms with three 2 s gaps 0.68 m, and the size of the noise and bias barely matters
(mild = full). Across a gap the gyro is gone, so the turn made inside it is a guess (26 deg median error). Here the
true turn across every gap is taken from ground truth and the part of the path after the gap is rotated about the
vertical by the difference, on saved outputs. No re-solve: the handheld solve does not care about heading, so turning a
whole segment is consistent. If the score returns to the clean level, heading across gaps is the thing to work on.
Usage: 89_gapyaw_oracle.py"""
import sys, os, glob
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from ev import *
from scipy.spatial.transform import Rotation as Rot


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def one(case):
    name, csv, out = case
    n = name.split("_")[0]
    room = load_room(int(n[4:]) if n.startswith("room") else n)
    a = np.loadtxt(out, comments="#")
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    keep = P["ids"] != -1
    te = P["t_ns"][keep] * 1e-9
    grid_of = np.nonzero(keep)[0]
    p = a[:, 1:4]
    Re = Rot.from_quat(a[:, 4:8])
    gt_t = room["gt_t_ns"] * 1e-9
    Rg_all = Rot.from_quat(room["gt_q"])
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1)); qg[keep] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)

    def gt_at(tq):
        j = int(np.clip(np.searchsorted(gt_t, tq), 0, len(gt_t) - 1))
        return Rg_all[j], abs(gt_t[j] - tq)
    L = P["last_good"]
    dead = ~P["ok"]
    dead[L + 1:] = False
    yaw_fix = np.zeros(len(te))          # cumulative yaw correction per output row
    errs = []
    for s, e in runs(dead):
        if s == 0 or e > L or P["t"][e] - P["t"][s] < 0.5:
            continue
        ib = np.searchsorted(grid_of, s) - 1          # output row just before the gap
        ia = np.searchsorted(grid_of, e)              # first output row after it
        if ib < 0 or ia >= len(te):
            continue
        gb, db = gt_at(te[ib]); ga, da = gt_at(te[ia])
        if max(db, da) > 0.05:
            continue
        W = (Re[ib] * (Re[ib].inv() * Re[ia]) * (gb.inv() * ga).inv() * Re[ib].inv()).as_matrix()
        eps = np.arctan2(W[1, 0] - W[0, 1], W[0, 0] + W[1, 1])
        errs.append(np.degrees(eps))
        yaw_fix[ia:] -= eps
    v = np.diff(p, axis=0)
    c, s_ = np.cos(yaw_fix[1:]), np.sin(yaw_fix[1:])
    vf = np.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
    pf = np.vstack([p[:1], p[0] + np.cumsum(vf, axis=0)])
    pg = np.zeros((len(P["t"]), 3)); pg[keep] = p
    pg2 = np.zeros((len(P["t"]), 3)); pg2[keep] = pf
    return name, sc(pg)["ate_se3"], sc(pg2)["ate_se3"], errs


if __name__ == "__main__":
    cases = []
    for lab, folder in (("v6_tests", "cpp_tests"), ("v6_kinds", "cpp_kinds")):
        for out in sorted(glob.glob(f"{EXP}/tmp/score_out/{lab}/*.txt")):
            n = os.path.basename(out)[:-4]
            if n.rsplit("_", 1)[1] in ("full", "mild", "longgap", "burst", "noisy"):
                cases.append((n, f"{EXP}/tmp/{folder}/{n}.csv", out))
    with mp.get_context("fork").Pool(12) as p:
        R = p.map(one, cases, chunksize=1)
    for plat in ("room", "euroc"):
        print(plat)
        for kind in ("mild", "full", "longgap", "burst", "noisy"):
            v = [r for r in R if r[0].rsplit("_", 1)[1] == kind and r[0].startswith("room") == (plat == "room")]
            if not v:
                continue
            e = np.abs(np.concatenate([r[3] for r in v])) if any(len(r[3]) for r in v) else np.zeros(1)
            print(f"  {kind:8s} n {len(v):2d} | entry {np.mean([r[1] for r in v]):.3f} m | true turn across gaps {np.mean([r[2] for r in v]):.3f} m | "
                  f"heading error per gap: median {np.median(e):.0f} deg, rms {np.sqrt((e ** 2).mean()):.0f} deg ({len(e)} gaps)")
