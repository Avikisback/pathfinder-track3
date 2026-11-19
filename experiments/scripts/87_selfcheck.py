"""Self-check study. For every test file: the solver's own diagnostics (slow speed, the drift rates the evidence chose) and
the ATE of the three outputs the entry could write: band-shrunk solve (never fall back), high-pass fallback (always fall
back), and standing still. The question is which label-free number tells when the solve should not be trusted.
Needs three builds in tmp/bin: pf_v5 (entry), pf_fb (-DPF_FORCE_CHECK=1), pf_nofb (-DPF_FORCE_CHECK=2).
Usage: 87_selfcheck.py"""
import sys, os, glob, pickle, re, subprocess
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
from ev import *

B = EXP + "/tmp/bin"
OUTD = EXP + "/tmp/score_out/selfcheck"


def one(csv):
    name = os.path.basename(csv)[:-4]
    n = name.split("_")[0]
    room = load_room(int(n[4:]) if n.startswith("room") else n)
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    keep = P["ids"] != -1
    o = dict(name=name)
    sc = None
    for tag, exe, dbg in (("entry", "pf_v5", True), ("fb", "pf_fb", False), ("nofb", "pf_nofb", False)):
        out = f"{OUTD}/{name}_{tag}.txt"
        r = subprocess.run([f"{B}/{exe}", csv, out] + (["--debug"] if dbg else []), capture_output=True, text=True)
        a = np.loadtxt(out, comments="#")
        pg = np.zeros((len(P["t"]), 3)); pg[keep] = a[:, 1:4]
        if sc is None:
            qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1)); qg[keep] = a[:, [7, 4, 5, 6]]
            sc = PosScorer(P, room, qg)
            o["stat"] = sc(np.zeros_like(pg))["ate_se3"]
        s = sc(pg)
        o[tag] = s["ate_se3"]
        o[tag + "_blind"] = s.get("blind_rel", np.nan)
        if dbg:
            e = r.stderr
            m = re.search(r"platform=(\w+) self_check=([0-9.]+)", e)
            o["drone"] = m.group(1) == "drone"
            o["check"] = float(m.group(2))
            sig = re.findall(r"sig_a ([0-9.eE+-]+) sig_g ([0-9.eE+-]+)", e)
            o["sig_a"], o["sig_g"] = (float(sig[-1][0]), float(sig[-1][1])) if sig else (np.nan, np.nan)
    return o


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    files = sorted(glob.glob(EXP + "/tmp/cpp_tests/*.csv")) + sorted(glob.glob(EXP + "/tmp/cpp_kinds/*.csv"))
    with mp.get_context("fork").Pool(12) as p:
        R = p.map(one, files, chunksize=1)
    pickle.dump(R, open(EXP + "/tmp/selfcheck.pkl", "wb"))
    for plat in (True, False):
        rows = [r for r in R if r["drone"] == plat]
        print(f"\n{'drone' if plat else 'handheld'}: {len(rows)} files")
        kinds = sorted({r["name"].split("_")[1] for r in rows})
        print(f"  {'kind':11s}  n  entry  never-fb  always-fb  still | best-of-3 | check  sig_a   sig_g   | fb used")
        for k in kinds:
            v = [r for r in rows if r["name"].split("_")[1] == k]
            mean = lambda key: float(np.mean([r[key] for r in v]))
            best = float(np.mean([min(r["nofb"], r["fb"], r["stat"]) for r in v]))
            used = sum(abs(r["entry"] - r["fb"]) < 1e-9 and abs(r["nofb"] - r["fb"]) > 1e-9 for r in v)
            print(f"  {k:11s} {len(v):2d}  {mean('entry'):.3f}  {mean('nofb'):.3f}     {mean('fb'):.3f}      {mean('stat'):.3f} | {best:.3f}     | {mean('check'):.2f}  {mean('sig_a'):.4f}  {mean('sig_g'):.5f} | {used}")
