"""New idea 4: velocity-aided tilt. Accelerometer = gravity + motion. Tilt fits treat motion as noise, or worse, absorb it.
Once a first velocity estimate exists, its derivative predicts the motion part, so it can be subtracted and tilt re-fitted.
Compared through one band filter, leave-one-sequence-out, with robust gains (median over files, lowest bands zero).
"""
import importlib, json, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
from phys import *
from phys import _hpf, _lpf

D4 = EXP + "/cache/v4"
os.makedirs(D4, exist_ok=True)
DECV, FV = m41.DECV, m41.FV
VARS = ("r5", "r10", "r20", "fit0", "aid5", "aid10", "gt")
LOW = 1 / 40     # Hz, velocity below this is not trusted when building the aid


def aided_imu(P, q):
    """IMU with the predicted motion acceleration removed from the accel channels, in the body frame."""
    v, last = m41.velocity(P, q)
    fs = round(1.0 / np.median(np.diff(P["t"])))
    vb = _hpf(v, LOW, fs)
    am = np.gradient(_lpf(vb, 2.0, fs), 1.0 / fs, axis=0)
    Rm = R.from_quat(q[: last + 1][:, [1, 2, 3, 0]]).as_matrix()
    ab = np.einsum("nji,nj->ni", Rm, am)
    out = P["imu"].copy()
    good = P["ok"][: last + 1]
    out[: last + 1][good, 3:] -= ab[good]
    return out


def one(key):
    n, s, m, sd = key
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    room = load_room(n)
    b0 = m41.calib_excl(n)
    fit = dict(b0=b0, knot_s=60, s_r=0.3, sigma_b=0.003)
    qs = {}
    qs["r5"] = offline_attitude(P["t"], P["imu"], P["ok"], residual_s=5.0, **fit)
    qs["r10"] = offline_attitude(P["t"], P["imu"], P["ok"], residual_s=10.0, **fit)
    qs["r20"] = offline_attitude(P["t"], P["imu"], P["ok"], residual_s=20.0, **fit)
    qs["fit0"] = offline_attitude(P["t"], P["imu"], P["ok"], residual_s=0.0, **fit)
    a = aided_imu(P, qs["r10"])
    qs["aid5"] = offline_attitude(P["t"], a, P["ok"], residual_s=5.0, **fit)
    qs["aid10"] = offline_attitude(P["t"], a, P["ok"], residual_s=10.0, **fit)
    qs["gt"] = m41.gt_attitude(P, room)
    out = {}
    for nm, q in qs.items():
        v, last = m41.velocity(P, q)
        idx = np.arange(0, last + 1, DECV)
        pre = P["ids"] != -1
        pre[last + 1 :] = False
        d = score(P["t_ns"][pre] * 1e-9, np.zeros((int(pre.sum()), 3)), q[pre][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        out[nm] = dict(v=v[idx].astype(np.float32), att=d["att_org"], tilt=d["tilt"])
    tk = P["t"][idx]
    vg, good = m41.gt_v(room, tk)
    with open(f"{D4}/{n}_{s}_{m}_{sd}.pkl", "wb") as f:
        pickle.dump(dict(out=out, tk=tk, vg=vg.astype(np.float32), good=good, last=last, key=key), f)
    return key


EDG = np.array([0, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 1e9])


def spec(x):
    return np.fft.rfft(np.concatenate([x, x[::-1]], 0), axis=0), np.fft.rfftfreq(2 * len(x), 1 / FV)


def file_gains(d, var):
    vh = d["out"][var]["v"].astype(np.float64)
    v = np.nan_to_num(d["vg"].astype(np.float64))
    g = d["good"]
    Rg = m41.yaw_gauge(vh[g] - vh[g].mean(0), v[g] - v[g].mean(0))
    vs = (Rg @ (vh - vh.mean(0)).T).T
    v2 = v - v[g].mean(0)
    v2[~g] = vs[~g]
    X, f = spec(vs)
    Y, _ = spec(v2)
    b = np.digitize(f, EDG) - 1
    num = np.bincount(b, (np.conj(X) * Y).real.sum(1), minlength=9)[:9]
    den = np.bincount(b, (np.abs(X) ** 2).sum(1), minlength=9)[:9]
    return num / np.maximum(den, 1e-12)


def pos(d, var, gains):
    vh = d["out"][var]["v"].astype(np.float64)
    X, f = spec(vh - vh.mean(0))
    cen = np.sqrt(np.maximum(EDG[:-1], 1 / 512) * np.minimum(EDG[1:], 4))
    G = np.interp(np.log(np.maximum(f, 1e-6)), np.log(cen), np.clip(gains, 0, 1.2))
    G[f < 1 / 64] = 0
    vb = np.fft.irfft(X * G[:, None], n=2 * len(vh), axis=0)[: len(vh)]
    return np.cumsum(vb, axis=0) / FV


if __name__ == "__main__":
    files = eval_set(m41.ALLN, seeds=(1, 2), modes=("zero",))
    todo = [k for k in files if not os.path.exists(f"{D4}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl")]
    t0 = time.time()
    if todo:
        with mp.get_context("spawn").Pool(12) as p:
            for k in p.imap_unordered(one, todo):
                pass
    print(f"built {len(todo)} files {time.time()-t0:.0f}s", flush=True)
    data = {k: pickle.load(open(f"{D4}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl", "rb")) for k in files}
    rooms = {n: load_room(n) for n in m41.ALLN}
    for plat, names in (("EuRoC", m41.SEQS), ("TUM", DEV_ROOMS)):
        for var in VARS:
            for sc in ("clean", "mild", "full"):
                A, att, tl, Gs = [], [], [], []
                for n in names:
                    gains = np.median([file_gains(data[k], var) for k in files if k[0] in names and k[0] != n and k[1] == sc], 0)
                    Gs.append(gains)
                    for k in files:
                        if k[0] == n and k[1] == sc:
                            d = data[k]
                            a, st = m41.ate(pos(d, var, gains), d["tk"], rooms[n])
                            A.append(a)
                            att.append(d["out"][var]["att"])
                            tl.append(d["out"][var]["tilt"])
                print(f"{plat:5s} {var:6s} {sc:5s} ATE {np.mean(A):.3f}/{np.max(A):.3f} att {np.mean(att):6.1f} tilt {np.mean(tl):5.2f} gains {np.round(np.mean(Gs,0),2)}", flush=True)
