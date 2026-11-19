"""Which attitude gives the best velocity? Each variant goes through the same leave-one-sequence-out band filter.

Earlier threads disagree: 0.39 m on clean TUM (gyro + low-gain Mahony) against 0.78 m (offline fit with tilt residual).
This separates attitude from everything else. Includes a new two-sided tilt: forward and backward low-gain filters have
opposite bias errors, so their midpoint cancels them, and their difference measures the bias.
"""
import json, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *

D3 = EXP + "/cache/v3"
os.makedirs(D3, exist_ok=True)
SEQS = list(EUROC)
ALLN = SEQS + DEV_ROOMS
DECV = 10   # velocity stored at 20 Hz
FV = 20.0


def calib_excl(n):
    """Declared sensor calibration: mean bias of the OTHER sequences of that platform. Zero for TUM (export is compensated)."""
    if not isinstance(n, str):
        return np.zeros(3)
    return np.mean([np.median(load_room(x)["gt_bw"], 0) for x in SEQS if x != n], 0)


def fill_gaps(t, imu, ok):
    """Rows of gyro and accel with gaps filled by a straight line across the gap, tail held at zero rate."""
    idx = np.arange(len(t))
    last = np.nonzero(ok)[0][-1]
    out = imu.copy()
    for i in range(6):
        out[:, i] = np.interp(idx, idx[ok], imu[ok, i])
    out[last + 1 :, :3] = 0.0
    return out, last


def rot_int(t, gyro, q0, reverse=False):
    return _integrate(t, gyro, q0)


from numba import njit
from common import _qexp, _qmul, _up_in_body


@njit(cache=True)
def _mahony(t, gyro, acc, ok, q0, kp, g, gate, forward):
    n = len(t)
    out = np.zeros((n, 4))
    q = q0.copy()
    rng = range(n) if forward else range(n - 1, -1, -1)
    prev = -1
    for k in rng:
        if prev >= 0:
            dt = abs(t[k] - t[prev])
            # the rate that applies across the step is the same for both directions
            w = gyro[k] if forward else -gyro[prev]
            corr = np.zeros(3)
            j = k
            if ok[j]:
                a = acc[j]
                am = np.sqrt(np.sum(a * a))
                if abs(am - g) < gate:
                    up = _up_in_body(q)
                    u = a / am
                    corr = np.array([u[1] * up[2] - u[2] * up[1], u[2] * up[0] - u[0] * up[2], u[0] * up[1] - u[1] * up[0]]) * kp
            q = _qmul(q, _qexp((w + corr) * dt))
            q /= np.sqrt(np.sum(q * q))
        out[k] = q
        prev = k
    return out


def tilt_fuse(qf, qb):
    """Midpoint in the tilt directions only: a world-horizontal rotation half way from forward to backward.
    Heading stays that of the forward pass."""
    Rf, Rb = R.from_quat(qf[:, [1, 2, 3, 0]]), R.from_quat(qb[:, [1, 2, 3, 0]])
    uf = Rf.as_matrix()[:, 2, :]       # world up in the body, forward
    ub = Rb.as_matrix()[:, 2, :]       # world up in the body, backward
    u = uf + ub
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # body-frame rotation taking uf to u
    ax = np.cross(uf, u)
    return (Rf * R.from_rotvec(-ax)).as_quat()[:, [3, 0, 1, 2]]


def two_sided(t, imu, ok, kp, b0, q0=None, iters=2, gate=2.0, fit_ramp=True):
    """Forward and backward low-gain filters. Their heading is not used. The tilt midpoint cancels constant and slowly
    changing bias. Then the horizontal part of bias from (forward - backward) can be removed and the pass repeated."""
    gy = imu[:, :3] - b0
    fill, last = fill_gaps(t, np.hstack([gy, imu[:, 3:]]), ok)
    an = np.linalg.norm(imu[ok, 3:], axis=1)
    g = np.median(an)
    if q0 is None:
        imu2 = imu.copy()
        imu2[ok, :3] -= b0
        q0, _ = quiet_init(t, imu2, ok)
    qf = _mahony(t, fill[:, :3], imu[:, 3:], ok, q0, kp, g, gate, True)
    # the backward pass begins where the forward pass ended
    qb = _mahony(t, fill[:, :3], imu[:, 3:], ok, qf[-1], kp, g, gate, False)
    qf = _mahony(t, fill[:, :3], imu[:, 3:], ok, qb[0], kp, g, gate, True)
    return tilt_fuse(qf, qb), qf, qb


def velocity(P, q):
    """World-frame gravity-free velocity at DECV, with the file mean removed. Only rows up to failure mean anything."""
    t, imu, ok = P["t"], P["imu"], P["ok"]
    fill, last = fill_gaps(t, imu, ok)
    Rm = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    g = np.median(np.linalg.norm(imu[ok, 3:], axis=1))
    aw = np.einsum("nij,nj->ni", Rm, fill[:, 3:]) - np.array([0, 0, g])
    aw = aw[: last + 1]
    aw = aw - aw.mean(0)
    dt = np.diff(t[: last + 1], prepend=t[0])
    v = np.cumsum(aw * dt[:, None], axis=0)
    return v, last


def gt_v(room, t):
    """Ground-truth world velocity at seconds t relative to room start (training and scoring only)."""
    gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
    p = room["gt_p"]
    h = 0.05
    f = lambda x: np.stack([np.interp(x, gt_t, p[:, i]) for i in range(3)], 1)
    v = (f(t + h) - f(t - h)) / (2 * h)
    idx = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1)
    good = (t - h >= gt_t[0]) & (t + h <= gt_t[-1]) & ((gt_t[idx] - gt_t[idx - 1]) < 0.1)
    return v, good


def gt_attitude(P, room):
    gt_t = (room["gt_t_ns"] - P["t_ns"][0]) * 1e-9
    tc = np.clip(P["t"], gt_t[0], gt_t[-1])
    from scipy.spatial.transform import Slerp
    return Slerp(gt_t, R.from_quat(room["gt_q"]))(tc).as_quat()[:, [3, 0, 1, 2]]


VARIANTS = ("v2", "maho02", "fit0", "ts01", "ts02", "ts05", "gt")


def one(key):
    n, s, m, sd = key
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    room = load_room(n)
    b0 = calib_excl(n)
    qs = {}
    qs["v2"], _i = attitude_v2(P["t"], P["imu"], P["ok"], calib=b0 if isinstance(n, str) else None)
    im2 = P["imu"].copy()
    im2[P["ok"], :3] -= b0
    q0, g = quiet_init(P["t"], im2, P["ok"])
    fill, last = fill_gaps(P["t"], im2, P["ok"])
    gmed = np.median(np.linalg.norm(P["imu"][P["ok"], 3:], axis=1))
    qs["maho02"] = _mahony(P["t"], fill[:, :3], P["imu"][:, 3:], P["ok"], q0, 0.02, gmed, 2.0, True)
    qs["fit0"] = offline_attitude(P["t"], P["imu"], P["ok"], b0=b0, knot_s=60, s_r=0.3, sigma_b=0.003, residual_s=0.0)
    for nm, kp in (("ts01", 0.01), ("ts02", 0.02), ("ts05", 0.05)):
        qs[nm], _, _ = two_sided(P["t"], P["imu"], P["ok"], kp, b0)
    qs["gt"] = gt_attitude(P, room)
    out = {}
    tv = None
    for nm, q in qs.items():
        v, last = velocity(P, q)
        idx = np.arange(0, last + 1, DECV)
        d = score(P["t_ns"][P["ids"] != -1] * 1e-9, np.zeros((int((P["ids"] != -1).sum()), 3)), q[P["ids"] != -1][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        pre = P["ids"] != -1
        pre[last + 1 :] = False
        d2 = score(P["t_ns"][pre] * 1e-9, np.zeros((int(pre.sum()), 3)), q[pre][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        out[nm] = dict(v=v[idx].astype(np.float32), att=d2["att_org"], tilt=d2["tilt"], rpe_r=d2["rpe_r"])
    tk = P["t"][idx]
    vg, good = gt_v(room, tk)
    with open(f"{D3}/{n}_{s}_{m}_{sd}.pkl", "wb") as f:
        pickle.dump(dict(out=out, tk=tk, vg=vg.astype(np.float32), good=good, last=last, key=key), f)
    return key


def yaw_gauge(vh, v):
    cov = vh[:, :2].T @ v[:, :2]
    th = np.arctan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fit_gains(items, var):
    pairs = []
    for d in items:
        vh = d["out"][var]["v"].astype(np.float64)
        v = np.nan_to_num(d["vg"].astype(np.float64))
        g = d["good"]
        Rg = yaw_gauge(vh[g], v[g])
        vs = (Rg @ vh.T).T
        v2 = v.copy()
        v2[~g] = vs[~g]
        pairs.append((vs - vs.mean(0), v2 - v2.mean(0)))
    return band_fit(pairs, 1 / FV)


def pos_from(d, var, gains, tau=1.0):
    vh = d["out"][var]["v"].astype(np.float64)
    vb = band_apply(vh - vh.mean(0), gains, 1 / FV)
    p = np.cumsum(vb, axis=0) / FV
    return p, vb


def ate(p, tk, room, tail=None):
    gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
    # compare at velocity rate on rows inside the mocap span
    m = (tk >= gt_t[0]) & (tk <= gt_t[-1])
    g = np.stack([np.interp(tk[m], gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    e = p[m]
    r, tr = umeyama(e, g)
    return float(np.sqrt((((r @ e.T).T + tr - g) ** 2).sum(1).mean())), float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))


if __name__ == "__main__":
    files = eval_set(ALLN, seeds=(1, 2), modes=("zero",))
    todo = [k for k in files if not os.path.exists(f"{D3}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl")]
    t0 = time.time()
    if todo:
        with mp.get_context("spawn").Pool(12) as p:
            for k in p.imap_unordered(one, todo):
                pass
    print(f"built {len(todo)} files {time.time()-t0:.0f}s", flush=True)
    data = {k: pickle.load(open(f"{D3}/{k[0]}_{k[1]}_{k[2]}_{k[3]}.pkl", "rb")) for k in files}
    rooms = {n: load_room(n) for n in ALLN}
    for plat, names in (("EuRoC", SEQS), ("TUM", DEV_ROOMS)):
        for var in VARIANTS:
            for sc in ("clean", "mild", "full"):
                A, S, att, tl, Gs = [], [], [], [], []
                for n in names:
                    train = [data[k] for k in files if k[0] in names and k[0] != n and k[1] == sc]
                    g = fit_gains(train, var)
                    Gs.append(g)
                    for k in files:
                        if k[0] == n and k[1] == sc:
                            d = data[k]
                            p, _ = pos_from(d, var, g)
                            a, st = ate(p, d["tk"], rooms[n])
                            A.append(a)
                            S.append(st)
                            att.append(d["out"][var]["att"])
                            tl.append(d["out"][var]["tilt"])
                print(f"{plat:5s} {var:7s} {sc:5s} ATE {np.mean(A):.3f}/{np.max(A):.3f} stat {np.mean(S):.3f} att {np.mean(att):6.1f} tilt {np.mean(tl):5.2f} gains {np.round(np.mean(Gs,0),2)}", flush=True)
