"""Decision rule for the self-check, from 87_selfcheck.py. Fall back to the high-pass when the slow speed is over the old
threshold OR the accel drift rate chosen by the evidence is over a new threshold. The threshold is picked on the seven
design shapes (clean, mild, full, burst, steps, random walk, sine) and then read off on the five shapes nobody tuned on
(hold, rot, warm-up, long gaps, triple noise). Also a graded version: blend the two outputs with a weight that falls
from 1 to 0 as the drift rate rises from lo to hi (positions are linear in the blend).
Usage: 87b_rule.py"""
import pickle, sys
import numpy as np
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
EXP = "/mnt/e/clipping-pipeline/xr_track3/experiments"
R = pickle.load(open(EXP + "/tmp/selfcheck.pkl", "rb"))
DESIGN = ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine")
VALID = ("hold", "rot", "warmup", "longgap", "noisy")
kind = lambda r: r["name"].split("_")[1]
for plat, c0 in ((True, 1.0), (False, 0.5)):
    rows = [r for r in R if r["drone"] == plat]
    print(f"\n{'DRONE' if plat else 'HANDHELD'} (old rule: fall back if slow speed > {c0})")

    def score(thr, group, cc=c0):
        v = [(r["fb"] if (r["check"] > cc or r["sig_a"] > thr) else r["nofb"]) for r in rows if kind(r) in group]
        return float(np.mean(v))
    for name, group in (("design", DESIGN), ("unseen", VALID)):
        base = score(1e9, group)
        best = np.mean([min(r["fb"], r["nofb"]) for r in rows if kind(r) in group])
        still = np.mean([r["stat"] for r in rows if kind(r) in group])
        line = " ".join(f"{t:g}:{score(t, group):.3f}" for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0))
        print(f"  {name:7s} now {base:.3f} | perfect choice {best:.3f} | still {still:.3f} | by drift-rate threshold  {line}")
    print("  per shape at thresholds: now / 0.2 / 0.3 / 0.4 / 0.5  (always fall back, standing still)")
    for k in DESIGN + VALID:
        g = (k,)
        print(f"    {k:11s} {score(1e9, g):.3f} / {score(0.2, g):.3f} / {score(0.3, g):.3f} / {score(0.4, g):.3f} / {score(0.5, g):.3f}   "
              f"({np.mean([r['fb'] for r in rows if kind(r) == k]):.3f}, {np.mean([r['stat'] for r in rows if kind(r) == k]):.3f})"
              f"   files with drift rate > 0.3: {sum(r['sig_a'] > 0.3 for r in rows if kind(r) == k)}/{sum(kind(r) == k for r in rows)}")
