"""Descent prior for the drone's blind 5 s, scored on the entry's saved outputs (L8rw_eb1 + CFG2 + rot fix). No re-solve needed.

The research notes found every EuRoC flight descending to land at the end of file, and that a fixed descent
z(h) = -D min(h/T, 1) added to hold took the blind error 0.335 -> 0.277 m on truth-free tails. Here it is added to the
entry's own tail (hold plus a small damped velocity), on drones only (rotor signature), and checked:
  - EuRoC, all 12 degradations (7 known/hostile + 5 robust), LOEO band gains as in the entry
  - gated variant: only if the rotors are still on in the last 2 s before the IMU dies (a landed drone does not descend)
  - Blackbird (unseen airframe), clean and full: does the prior hurt a drone whose files do not end in a landing?
  - TUM rooms and TUM-VI walks: the gate must never fire (handheld)
D is swept for information only; the shipped value stays the one declared before this run (0.25 m over 3 s).
"""
import sys, os, pickle
sys.argv = [sys.argv[0], "/mnt/e/clipping-pipeline/xr_track3/experiments/cache/v15"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import importlib
import numpy as np
import rw_evidence as rwe
import ev
from ev import *
m73 = importlib.import_module("73_rotfix")
m72 = m73.m72
import robust_shapes as rs
from phys import rotors_on

V13, V15, V17 = EXP + "/cache/v13", EXP + "/cache/v15", EXP + "/cache/v17"
CFG2 = m72.CFG2
RK = [(k, "x", 1) for k in rs.KINDS]
DS = (0.0, 0.15, 0.25, 0.35)
TS = (2.0, 3.0)


def loader(n, k):
    for d in (V15, V13):
        f = f"{d}/{n}_{k[0]}_{k[1]}_{k[2]}_L8rw_eb1.pkl"
        if os.path.exists(f):
            return pickle.load(open(f, "rb"))
    raise FileNotFoundError((n, k))


def rotors_end(P, sec=2.0):
    """Fraction of the last `sec` seconds before failure with the rotor signature on."""
    t, imu, ok = P["t"], P["imu"], P["ok"]
    fs = round(1.0 / np.median(np.diff(t)))
    idx = np.arange(len(t))
    a = np.stack([np.interp(idx, idx[ok], imu[ok, 3 + i]) for i in range(3)], 1)
    on = rotors_on(a, fs)
    L = P["last_good"]
    return float(on[max(0, L - int(sec * fs)): L + 1].mean())


def descend(P, d, tr, D, T):
    out = tr.copy()
    tail = P["t"] > d["tk"][-1]
    s = P["t"][tail] - d["tk"][-1]
    out[tail, 2] -= D * np.minimum(s / T, 1.0)
    return out


def score(n, k, d, fit, file_for):
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    sc = PosScorer(P, room, d["q"].astype(np.float64))
    tr = m57.track(P, d, m58.final_v(d, fit, *CFG2))
    o = {"drone": bool(d["drone"]), "on_end": rotors_end(P)}
    for D in DS:
        for T in TS:
            x = descend(P, d, tr, D, T) if d["drone"] else tr
            s = sc(x)
            s["blind_abs"] = blind_abs(sc, x)
            o[(D, T)] = s
    st = np.zeros((len(P["t"]), 3))
    o["stationary"] = sc(st)
    # truth: vertical drop over the blind window, for the record
    gt_t = room["gt_t_ns"] * 1e-9
    tf = P["t_fail_ns"] * 1e-9
    m = (gt_t >= tf) & (gt_t <= tf + 5.0)
    o["gt_drop"] = float(room["gt_p"][m][-1, 2] - room["gt_p"][m][0, 2]) if m.sum() > 10 else float("nan")
    return o


def show(rows, title):
    print(f"\n=== {title}  ({len(rows)} files; drone flagged on {np.mean([r['drone'] for r in rows.values()]):.2f})")
    gate_on = {k: r["on_end"] > 0.5 for k, r in rows.items()}
    print(f"  rotors on at the end in {np.mean(list(gate_on.values())):.2f} of files; truth drop over blind 5 s: median {np.nanmedian([r['gt_drop'] for r in rows.values()]):+.2f} m, range {np.nanmin([r['gt_drop'] for r in rows.values()]):+.2f} to {np.nanmax([r['gt_drop'] for r in rows.values()]):+.2f}")
    for D in DS:
        for T in TS:
            if D == 0 and T != 3.0:
                continue
            br = np.nanmean([r[(D, T)].get("blind_rel", np.nan) for r in rows.values()])
            ba = np.nanmean([r[(D, T)]["blind_abs"] for r in rows.values()])
            at = np.nanmean([r[(D, T)]["ate_se3"] for r in rows.values()])
            # gated: descent only where rotors are on at the end
            g = [r[(D, T)] if gate_on[k] else r[(0.0, 3.0)] for k, r in rows.items()]
            gbr = np.nanmean([x.get("blind_rel", np.nan) for x in g])
            print(f"  D {D:.2f} T {T:.0f}: blind_rel {br:.3f} (gated {gbr:.3f})  blind_abs {ba:.3f}  ate {at:.3f}", flush=True)


if __name__ == "__main__":
    out = {}
    # EuRoC
    rooms = {n: load_room(n) for n in SEQS}
    F = fits(EU_FOLDS, loader, rooms)
    rows = {}
    for n in SEQS:
        for k in rwe.KEYS + RK:
            rows[(n, *k)] = score(n, k, loader(n, k), F[n], m72.file_for2)
    out["eu"] = rows
    show(rows, "EuRoC, all degradations")
    for grp in (rwe.KEYS_KNOWN, rwe.KEYS_NEW, RK):
        nm = {tuple(rwe.KEYS_KNOWN): "known", tuple(rwe.KEYS_NEW): "hostile", tuple(RK): "robust"}[tuple(grp)]
        show({k: v for k, v in rows.items() if tuple(k[1:]) in grp}, f"EuRoC, {nm} only")
    # TUM rooms: the gate must not fire
    rooms_t = {n: load_room(n) for n in DEV_ROOMS}
    Ft = fits(TU_FOLDS, loader, rooms_t)
    rows_t = {(n, *k): score(n, k, loader(n, k), Ft[n], m72.file_for2) for n in DEV_ROOMS for k in rwe.KEYS_KNOWN}
    out["tum"] = rows_t
    print(f"\nTUM rooms: drone flagged on {sum(r['drone'] for r in rows_t.values())} of {len(rows_t)} files")
    pickle.dump(out, open(f"{RES}/descent.pkl", "wb"))
    print("done eu/tum", flush=True)
