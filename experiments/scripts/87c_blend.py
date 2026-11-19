"""Follow-up to 87b_rule.py: files where the evidence chose a high drift rate, one line each, and two more rules.
  two-number rule   fall back if slow speed > c0, or drift rate > A and slow speed > C
  blend             for drift rate > A, write the average of the two outputs (weight w on the solve) instead of switching
Blends are scored from the saved outputs (positions are linear in the velocity blend).
Usage: 87c_blend.py"""
import pickle, sys, os
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from ev import *
R = pickle.load(open(EXP + "/tmp/selfcheck.pkl", "rb"))
OUTD = EXP + "/tmp/score_out/selfcheck"
DESIGN = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine")
VALID = ("hold", "rot", "warmup", "longgap", "noisy")
kind = lambda r: r["name"].split("_")[1]
WS = (0.7, 0.5, 0.3)


def blend(r):
    name = r["name"]
    n = name.split("_")[0]
    folder = "cpp_tests" if os.path.exists(f"{EXP}/tmp/cpp_tests/{name}.csv") else "cpp_kinds"
    t_ns, imu = common._read_csv_ns(f"{EXP}/tmp/{folder}/{name}.csv")
    P = prepare(t_ns, imu)
    room = load_room(int(n[4:]) if n.startswith("room") else n)
    keep = P["ids"] != -1
    a = np.loadtxt(f"{OUTD}/{name}_nofb.txt", comments="#")
    b = np.loadtxt(f"{OUTD}/{name}_fb.txt", comments="#")
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1)); qg[keep] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)
    out = {}
    for w in WS:
        pg = np.zeros((len(P["t"]), 3)); pg[keep] = w * a[:, 1:4] + (1 - w) * b[:, 1:4]
        out[w] = sc(pg)["ate_se3"]
    return name, out


if __name__ == "__main__":
    hi = [r for r in R if r["sig_a"] > 0.15]
    with mp.get_context("fork").Pool(12) as p:
        BL = dict(p.map(blend, hi, chunksize=1))
    for plat, c0 in ((True, 1.0), (False, 0.5)):
        rows = [r for r in R if r["drone"] == plat]
        print(f"\n{'DRONE' if plat else 'HANDHELD'}: files with drift rate over 0.15")
        print(f"  {'file':18s} rate   slow   solve  fallback  still   blend 0.7/0.5/0.3")
        for r in sorted(rows, key=lambda r: (kind(r), r["name"])):
            if r["sig_a"] > 0.15:
                bl = BL[r["name"]]
                print(f"  {r['name']:18s} {r['sig_a']:.3f}  {r['check']:.2f}   {r['nofb']:.2f}   {r['fb']:.2f}      {r['stat']:.2f}    {bl[0.7]:.2f} {bl[0.5]:.2f} {bl[0.3]:.2f}")

        def rule(fn, group):
            return float(np.mean([fn(r) for r in rows if kind(r) in group]))
        now = lambda r: r["fb"] if r["check"] > c0 else r["nofb"]
        cands = {"now": now}
        for A in (0.2, 0.3, 0.4):
            cands[f"rate>{A}"] = (lambda A: lambda r: r["fb"] if (r["check"] > c0 or r["sig_a"] > A) else r["nofb"])(A)
            for C in (0.5, 0.6, 0.7) if plat else (0.35, 0.4, 0.45):
                cands[f"rate>{A} & slow>{C}"] = (lambda A, C: lambda r: r["fb"] if (r["check"] > c0 or (r["sig_a"] > A and r["check"] > C)) else r["nofb"])(A, C)
            for w in WS:
                cands[f"rate>{A}: blend {w}"] = (lambda A, w: lambda r: r["fb"] if r["check"] > c0 else (BL[r["name"]][w] if r["sig_a"] > A else r["nofb"]))(A, w)
        print(f"  {'rule':26s} design  unseen  all 12 | sine   random walk  warm-up")
        for nm, fn in cands.items():
            d, u = rule(fn, DESIGN), rule(fn, VALID)
            nd, nu = sum(kind(r) in DESIGN for r in rows), sum(kind(r) in VALID for r in rows)
            print(f"  {nm:26s} {d:.3f}   {u:.3f}   {(d * nd + u * nu) / (nd + nu):.3f}  | {rule(fn, ('sine',)):.3f}  {rule(fn, ('randomwalk',)):.3f}        {rule(fn, ('warmup',)):.3f}")
