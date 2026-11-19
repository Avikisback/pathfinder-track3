"""Blind 5 s on handheld files: which forecast should follow the last pose?

Rooms (5 endings) say hold, long walks (21 endings) say carry a little velocity on. This lab replays alternative tails on
saved C++ outputs (hold tail, tmp/score_out/new_*), so no solve is needed:
  fixed   p_last + G * tau * (1 - exp(-s/tau)) * v0 for a grid of (G, tau); (0.5, 0.35) is the old tail
  self    G and tau fitted on the file's own estimated path: at many earlier moments, how well did the velocity then predict
          the displacement over the next 5 s? No truth is used. "self half" halves that gain.
v0 is the mean velocity of the output over its last 0.5 s. Scored with blind_rel (error of the displacement since the failure).
Usage: 85_tail_lab.py"""
import sys, os, glob, pickle, importlib
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v19"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import multiprocessing as mp
import numpy as np
import common
m74 = importlib.import_module("74_unseen")
from ev import *

FIXED = [(0.5, 0.35), (1.0, 0.35), (1.0, 0.7), (1.0, 1.5), (0.5, 1.5), (1.0, 3.0)]
TAUS = [0.35, 0.7, 1.5, 3.0]
SJ = np.arange(5, 51, 5)          # 0.5 s steps out to 5 s, in 10 Hz samples


def self_fit(pk, lo=None):
    """Gain and time constant that best predicted this path's own next 5 s from its velocity."""
    cuts = np.arange(100, len(pk) - 50, 10)
    if lo is not None:
        cuts = cuts[cuts >= lo]
    if len(cuts) < 5:
        return 0.0, 0.35
    v = (pk[cuts] - pk[cuts - 5]) / 0.5
    d = np.stack([pk[cuts + j] - pk[cuts] for j in SJ], 1)          # cuts x steps x 3
    best = (np.inf, 0.0, 0.35)
    for tau in TAUS:
        a = tau * (1 - np.exp(-(SJ * 0.1) / tau))
        num = (a[None, :] * (d * v[:, None, :]).sum(2)).sum()
        den = (a[None, :] ** 2 * (v ** 2).sum(1)[:, None]).sum()
        G = float(np.clip(num / max(den, 1e-12), 0.0, 1.0))
        err = ((d - G * a[None, :, None] * v[:, None, :]) ** 2).sum()
        if err < best[0]:
            best = (err, G, tau)
    return best[1], best[2]


def one(case):
    name, csv, out, kind = case
    a = np.loadtxt(out, comments="#")
    t_ns, imu = common._read_csv_ns(csv)
    P = prepare(t_ns, imu)
    room = m74.load2(name.rsplit("_", 1)[0]) if kind == "walk" else load_room(int(name.split("_")[0][4:]))
    keep = P["ids"] != -1
    te = P["t_ns"][keep] * 1e-9
    tf = P["t_fail_ns"] * 1e-9
    tail = te > tf
    p = a[:, 1:4]
    qg = np.tile([1.0, 0, 0, 0], (len(P["t"]), 1)); qg[keep] = a[:, [7, 4, 5, 6]]
    sc = PosScorer(P, room, qg)
    pre_t, pre_p = te[~tail], p[~tail]
    at = lambda tq: np.stack([np.interp(tq, pre_t, pre_p[:, i]) for i in range(3)], -1)
    v0 = (at(tf) - at(tf - 0.5)) / 0.5
    s = te[tail] - tf
    tk = np.arange(pre_t[0], tf, 0.1)
    pk = at(tk)

    def score(G, tau):
        pg = np.zeros((len(P["t"]), 3))
        q = p.copy()
        q[tail] = p[tail] + G * tau * (1 - np.exp(-s / tau))[:, None] * v0[None]
        pg[keep] = q
        return sc(pg).get("blind_rel", np.nan)

    o = {"hold": score(0.0, 0.35)}
    for G, tau in FIXED:
        o[f"G{G} t{tau}"] = score(G, tau)
    G, tau = self_fit(pk)
    o["self"] = score(G, tau)
    o["self half"] = score(0.5 * G, tau)
    o["self x1.5"] = score(min(1.0, 1.5 * G), tau)
    o["self x2"] = score(min(1.0, 2.0 * G), tau)
    o["self gain t0.7"] = score(G, 0.7)
    o["self x2 t0.7"] = score(min(1.0, 2.0 * G), 0.7)
    G2, tau2 = self_fit(pk, lo=len(pk) - 650)
    o["self last 60 s"] = score(G2, tau2)
    o["_fit"] = (G, tau, G2, tau2, float(np.linalg.norm(v0)))
    return name, kind, o


if __name__ == "__main__":
    cases = []
    for out in sorted(glob.glob(f"{EXP}/tmp/score_out/new_tests/room*.txt")):
        n = os.path.basename(out)[:-4]
        cases.append((n, f"{EXP}/tmp/cpp_tests/{n}.csv", out, "room"))
    for out in sorted(glob.glob(f"{EXP}/tmp/score_out/new_long/*.txt")):
        n = os.path.basename(out)[:-4]
        cases.append((n, f"{EXP}/tmp/cpp_long/{n}.csv", out, "walk"))
    with mp.get_context("fork").Pool(12) as p:
        R = p.map(one, cases, chunksize=1)
    pickle.dump(R, open(f"{EXP}/tmp/tail_lab.pkl", "wb"))
    keys = [k for k in R[0][2] if not k.startswith("_")]
    for kind in ("room", "walk"):
        rows = [o for n, k, o in R if k == kind and np.isfinite(o["hold"])]
        print(f"\n{kind}: {len(rows)} files (all degradations)")
        for k in keys:
            v = np.array([o[k] for o in rows]); h = np.array([o["hold"] for o in rows])
            print(f"  {k:16s} mean {v.mean():.3f} | better than hold on {(v < h - 1e-4).sum():2d}, worse on {(v > h + 1e-4).sum():2d}")
        fits = np.array([o["_fit"] for o in rows])
        print(f"  self-fit gain: median {np.median(fits[:, 0]):.2f} (range {fits[:, 0].min():.2f}-{fits[:, 0].max():.2f}), tau median {np.median(fits[:, 1]):.2f}; end speed median {np.median(fits[:, 4]):.2f} m/s")
    both = [o for n, k, o in R if np.isfinite(o["hold"])]
    print("\nrooms and walks pooled by file:")
    for k in keys:
        print(f"  {k:16s} mean {np.mean([o[k] for o in both]):.3f}")
