"""How do recordings end? Ground truth only, no estimator.

The blind 5 s starts where the IMU dies, 5 s before the end of the recording. If recordings tend to finish where they
began (a walk back to the start, a landing at the take-off spot) or drift back towards the middle of where they have been,
a forecast can use that. For every sequence this prints the distance from the start at the cut and at the end, the share
of the way home covered in the last 5 s, and the blind error (rms over the 5 s of the displacement since the cut) of
  hold      no displacement
  home      move a share b of the way to the start, linearly over 5 s, with b fitted on the OTHER sequences of the set
  centre    the same towards the centroid of the path so far
Usage: 86_endings.py"""
import sys, os, importlib
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
import common
m74 = importlib.import_module("74_unseen")
load74 = m74.load2
from ev import *

BB_DIR = "/mnt/e/clipping-pipeline/xr_track3/data/unseen/blackbird/conv"


def load_bb(n):
    d = np.load(f"{CACHE}/{n}.npz")
    return {k: d[k] for k in d.files}


def one(room):
    t = room["t_ns"] * 1e-9
    gt_t = room["gt_t_ns"] * 1e-9
    p = room["gt_p"]
    t_end = min(t[-1], gt_t[-1])
    t_cut = t[-1] - 5.0
    if gt_t[-1] < t_cut + 2.0 or gt_t[0] > t[0] + 30:
        return None
    at = lambda tq: np.array([np.interp(tq, gt_t, p[:, i]) for i in range(3)]).T
    p0 = at(gt_t[0] + 0.5)
    pc = at(t_cut)
    s = np.arange(0.1, t_end - t_cut, 0.1)
    d = at(t_cut + s) - pc                                 # true displacement since the cut
    before = (gt_t <= t_cut) & (gt_t >= t[0])
    # ground truth on the long walks exists only near the start and the end; use what there is before the cut
    cen = p[before].mean(0)
    return dict(pc_p0=pc - p0, cen=cen - pc, d=d, s=s, dist_cut=float(np.linalg.norm(pc - p0)), dist_end=float(np.linalg.norm(at(t_end) - p0)),
                gap=float(np.max(np.diff(gt_t[before]))) if before.sum() > 1 else 0.0)


def rms(x):
    return float(np.sqrt((np.asarray(x) ** 2).sum(-1).mean()))


SETS = {
    "EuRoC drone": ([s for v in common.ENV.values() for s in v], load_room),
    "TUM rooms 1-5": (DEV_ROOMS, load_room),
    "TUM walks": (sorted("tv_" + d[6:] for d in os.listdir(m74.UN) if d.startswith("tumvi_") and os.path.exists(f"{m74.UN}/{d}/gt_imu.csv")), load74),
    "Blackbird": (sorted("bb_" + x for x in os.listdir(BB_DIR) if x.startswith(("test_", "train_"))), load_bb),
}
for title, (names, loader) in SETS.items():
    R = {}
    for n in names:
        try:
            o = one(loader(n))
        except Exception as ex:
            print(n, "failed", repr(ex)[:100]); continue
        if o is not None:
            R[n] = o
    if not R:
        continue
    print(f"\n{title}: {len(R)} sequences")
    hold, home, cent = [], [], []
    for n, o in R.items():
        prof = (o["s"] / 5.0)[:, None]
        # share of the way home (and to the centre), fitted on the other sequences by least squares
        num = den = numc = denc = 0.0
        for m, q in R.items():
            if m == n:
                continue
            pr = (q["s"] / 5.0)[:, None]
            num += (q["d"] * (-q["pc_p0"])[None] * pr).sum(); den += ((pr * (-q["pc_p0"])[None]) ** 2).sum()
            numc += (q["d"] * q["cen"][None] * pr).sum(); denc += ((pr * q["cen"][None]) ** 2).sum()
        b, bc = num / max(den, 1e-9), numc / max(denc, 1e-9)
        hold.append(rms(o["d"])); home.append(rms(o["d"] - b * prof * (-o["pc_p0"])[None])); cent.append(rms(o["d"] - bc * prof * o["cen"][None]))
        own = (o["d"][-1] @ (-o["pc_p0"])) / max(o["pc_p0"] @ o["pc_p0"], 1e-9)
        print(f"  {str(n):30s} from start: at cut {o['dist_cut']:5.2f} m, at end {o['dist_end']:5.2f} m | own share home {own:+.2f} | hold {hold[-1]:.3f}  home (b {b:+.2f}) {home[-1]:.3f}  centre (b {bc:+.2f}) {cent[-1]:.3f}")
    print(f"  MEAN blind error: hold {np.mean(hold):.3f} | home {np.mean(home):.3f} ({100*(np.mean(home)/np.mean(hold)-1):+.0f}%) | centre {np.mean(cent):.3f} ({100*(np.mean(cent)/np.mean(hold)-1):+.0f}%)"
          f" | better than hold: home {sum(a < b for a, b in zip(home, hold))}/{len(hold)}, centre {sum(a < b for a, b in zip(cent, hold))}/{len(hold)}")
