"""Probe, handheld only: is the walking direction a fixed axis of the device within one recording, and can the file find that
axis by itself? If so, speed (gait speedometer, 62_speedo.py) times that axis is a full velocity measurement that does not
drift, the handheld counterpart of rotor drag. No solver change here: model velocities are integrated and scored directly,
and also blended into the entry's output below a cut-off. TUM-VI rooms 1-5, clean and full, cached entry outputs (v18).
  M1 true speed, axis fitted on truth, true attitude      (ceiling of the model class)
  M3 true speed, axis from the entry's own velocity, entry attitude
  M4 speedometer speed (other rooms), axis from the entry's own velocity, entry attitude   (nothing from truth)
  F4 entry output with its band below fc replaced by M4's
Usage: 81_carryaxis.py"""
import sys, pickle, importlib
sys.argv = [sys.argv[0], "score"]
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
m78 = importlib.import_module("78_gyroprior")
from ev import *
import ev
m62 = importlib.import_module("62_speedo")
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import uniform_filter1d
from phys import _lpf
rwe, m77 = m78.rwe, m78.m77
KS = [("clean", "delete", 1), ("full", "zero", 1)]
L = m78.loader_for("base")
ev.KNOWN = rwe.KEYS_KNOWN


def unit(x):
    return x / max(np.linalg.norm(x), 1e-9)


def hdir(R, u):
    w = R.apply(u)
    w[:, 2] = 0
    n = np.linalg.norm(w, axis=1, keepdims=True)
    return w / np.maximum(n, 1e-6)


def box(x, n=20):
    return uniform_filter1d(x, n, axis=0, mode="nearest")


def run(n, k, fit, D):
    t, imu, room = m77.m72.file_for2(n, k)
    P = prepare(t, imu)
    d = L(n, k)
    tk = d["tk"]
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    ing = (tk >= gt_t[0]) & (tk <= min(gt_t[-1], P["t"][P["last_good"]]))
    pt = np.stack([np.interp(tk, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    vt = np.gradient(pt, tk, axis=0) * ing[:, None]
    vts = box(vt)
    st = box(np.linalg.norm(vt[:, :2], axis=1))
    Rt = Rot.from_quat(room["gt_q"][np.clip(np.searchsorted(gt_t, tk), 0, len(gt_t) - 1)])
    gi = np.clip(np.searchsorted(P["t"], tk), 0, len(P["t"]) - 1)
    Rh = Rot.from_quat(d["q"].astype(np.float64)[gi][:, [1, 2, 3, 0]])
    mov = ing & (st > 0.3)
    u_orc = unit(Rt.inv().apply(vts)[mov].mean(0))
    res_len = np.linalg.norm(Rt.inv().apply(vts)[mov].mean(0)) / np.sqrt((vts[mov] ** 2).sum(1).mean())
    # speedometer, trained on the other rooms
    Xtr = np.concatenate([D[(r, kk)]["X"][D[(r, kk)]["good"]] for r in DEV_ROOMS if r != n for kk in D_KS])
    ytr = np.concatenate([D[(r, kk)]["yh"][D[(r, kk)]["good"]] for r in DEV_ROOMS if r != n for kk in D_KS])
    f = m62.ridge(Xtr, ytr, 1e-2)
    X, good, cen = m62.features(P["t"], P["imu"], P["ok"], P["last_good"])
    sh = np.clip(np.interp(tk, cen, np.where(good, f(X), np.nan_to_num(np.nanmean(f(X[good]))))), 0, 2.5)
    sh = sh * (tk <= P["t"][P["last_good"]])
    # the file's own axis: mean body-frame velocity of the entry's raw solve, where the speedometer says it is moving
    vr = d["v"] - d["v"].mean(0)
    vb = Rh.inv().apply(box(vr))
    mh = sh > 0.3
    u_self = unit(vb[mh].mean(0))
    ang = np.degrees(np.arccos(np.clip(u_self @ u_orc, -1, 1)))
    # direction error of the model class against truth, moving samples
    dm = hdir(Rt, u_orc)[mov]
    dv = vts[mov, :2] / np.linalg.norm(vts[mov, :2], axis=1, keepdims=True)
    de = np.degrees(np.arccos(np.clip((dm[:, :2] * dv).sum(1), -1, 1)))
    vfin = m58.final_v(d, fit, *m77.CFG2)
    sc = PosScorer(P, room, d["q"].astype(np.float64))

    def ate(v):
        v = v.copy()
        v[:, 2] = vfin[:, 2]
        return sc(m57.track(P, d, v))["ate_se3"]

    M1 = st[:, None] * hdir(Rt, u_orc)
    M3 = st[:, None] * hdir(Rh, u_self)
    M4 = sh[:, None] * hdir(Rh, u_self)
    o = dict(stat=sc(np.zeros((len(P["t"]), 3)))["ate_se3"], entry=ate(vfin), M1=ate(M1), M3=ate(M3), M4=ate(M4),
             res_len=float(res_len), ang=float(ang), de50=float(np.median(de)), de75=float(np.percentile(de, 75)),
             sp_rmse=float(np.sqrt(((sh - st)[ing] ** 2).mean())))
    for fc in (1 / 32, 1 / 16, 1 / 8):
        for nm, M in (("F3", M3), ("F4", M4)):
            o[f"{nm}_{round(1/fc)}"] = ate(vfin - _lpf(vfin, fc, 10.0) + _lpf(M, fc, 10.0))
        o[f"R4_{round(1/fc)}"] = ate(vr - _lpf(vr, fc, 10.0) + _lpf(M4, fc, 10.0))
    return o


if __name__ == "__main__":
    rooms = {n: load_room(n) for n in DEV_ROOMS}
    F = ev.fits(TU_FOLDS, L, rooms)
    D = pickle.load(open(f"{RES}/speedo_feats.pkl", "rb"))
    D_KS = sorted({k for (n, k) in D if n in DEV_ROOMS})
    globals()["D_KS"] = D_KS
    keys = None
    for k in KS:
        rows = []
        for n in DEV_ROOMS:
            o = run(n, k, F[n], D)
            rows.append(o)
            keys = list(o)
            print(f"room {n} {k[0]:5s} " + " ".join(f"{a} {o[a]:.2f}" for a in keys), flush=True)
        print(f"MEAN   {k[0]:5s} " + " ".join(f"{a} {np.mean([r[a] for r in rows]):.2f}" for a in keys), flush=True)
