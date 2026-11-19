"""Tests energy_cal on real data, raw (no filter at all) and through a leave-one-sequence-out band filter."""
import importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
m42 = importlib.import_module("42_tilt_aid")
from phys import *
from energy2 import energy_cal

D5 = EXP + "/cache/v7"
os.makedirs(D5, exist_ok=True)
FV, DECV = m41.FV, m41.DECV
VARS = {"p5": dict(), "p3": dict(sig_p=3.0), "p10": dict(sig_p=10.0), "p5a1": dict(sig_a=0.1), "p5dv": dict(sig_dv=0.5), "p5v1": dict(sig_v=1.0)}


def one(key):
    n, s, m, sd = key
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    room = load_room(n)
    b0 = m41.calib_excl(n)
    out = {}
    tk = P["t"][np.arange(0, P["last_good"] + 1, DECV)]
    for nm, kw in VARS.items():
        q, v, tb, last, info = energy_cal(P["t"], P["imu"], P["ok"], b0=b0, return_info=True, **kw)
        v20 = np.stack([np.interp(tk, tb, v[:, i]) for i in range(3)], 1)
        pre = P["ids"] != -1
        pre[last + 1 :] = False
        d = score(P["t_ns"][pre] * 1e-9, np.zeros((int(pre.sum()), 3)), q[pre][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        out[nm] = dict(v=v20.astype(np.float32), att=d["att_org"], tilt=d["tilt"], rpe_r=d["rpe_r"], info=info)
    vg, good = m41.gt_v(room, tk)
    with open(f"{D5}/{n}_{s}_{m}_{sd}.pkl", "wb") as f:
        pickle.dump(dict(out=out, tk=tk, vg=vg.astype(np.float32), good=good, last=last, key=key), f)
    return key


def raw_pos(d, var):
    return np.cumsum(d["out"][var]["v"].astype(np.float64), axis=0) / FV


def file_gains(d, var):
    vh = d["out"][var]["v"].astype(np.float64)
    v = np.nan_to_num(d["vg"].astype(np.float64))
    g = d["good"]
    Rg = m41.yaw_gauge(vh[g], v[g])
    vs = (Rg @ vh.T).T
    v2 = v.copy()
    v2[~g] = vs[~g]
    X, f = m42.spec(vs)
    Y, _ = m42.spec(v2)
    b = np.digitize(f, m42.EDG) - 1
    num = np.bincount(b, (np.conj(X) * Y).real.sum(1), minlength=9)[:9]
    den = np.bincount(b, (np.abs(X) ** 2).sum(1), minlength=9)[:9]
    return num / np.maximum(den, 1e-12)


def pos_low(d, var, gains):
    vh = d["out"][var]["v"].astype(np.float64)
    X, f = m42.spec(vh)
    cen = np.sqrt(np.maximum(m42.EDG[:-1], 1 / 512) * np.minimum(m42.EDG[1:], 4))
    G = np.interp(np.log(np.maximum(f, 1e-6)), np.log(cen), np.clip(gains, 0, 1.2))
    vb = np.fft.irfft(X * G[:, None], n=2 * len(vh), axis=0)[: len(vh)]
    return np.cumsum(vb, axis=0) / FV


if __name__ == "__main__":
    files = eval_set(m41.ALLN, seeds=(1, 2), modes=("zero",))
    todo = [k for k in files if not os.path.exists(f"{D5}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl")]
    t0 = time.time()
    if todo:
        with mp.get_context("spawn").Pool(14) as p:
            for k in p.imap_unordered(one, todo):
                pass
    print(f"built {len(todo)} files {time.time()-t0:.0f}s", flush=True)
    data = {k: pickle.load(open(f"{D5}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl", "rb")) for k in files}
    rooms = {n: load_room(n) for n in m41.ALLN}
    for plat, names in (("EuRoC", m41.SEQS), ("TUM", DEV_ROOMS)):
        for var in VARS:
            for sc in ("clean", "mild", "full"):
                A, Ar, att, tl, Gs, rv = [], [], [], [], [], []
                for n in names:
                    gains = np.median([file_gains(data[k], var) for k in files if k[0] in names and k[0] != n and k[1] == sc], 0)
                    Gs.append(gains)
                    for k in files:
                        if k[0] == n and k[1] == sc:
                            d = data[k]
                            A.append(m41.ate(pos_low(d, var, gains), d["tk"], rooms[n])[0])
                            Ar.append(m41.ate(raw_pos(d, var), d["tk"], rooms[n])[0])
                            att.append(d["out"][var]["att"])
                            tl.append(d["out"][var]["tilt"])
                            rv.append(d["out"][var]["info"]["rms_v"])
                print(f"{plat:5s} {var:8s} {sc:5s} raw ATE {np.mean(Ar):.3f}/{np.max(Ar):.3f} band ATE {np.mean(A):.3f}/{np.max(A):.3f} att {np.mean(att):6.1f} tilt {np.mean(tl):5.2f} rmsv {np.mean(rv):5.2f} gains {np.round(np.mean(Gs,0),2)}", flush=True)
