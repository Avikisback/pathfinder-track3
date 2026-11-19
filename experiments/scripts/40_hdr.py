"""New idea 1: yaw-rate mode tracking ("heuristic drift reduction" made offline).

Gravity cannot see the gyro bias around the vertical. But most of the time a platform is not turning, so the MODE of
the world-vertical rate over a long window sits at the bias, and real turns only add tails. Nothing here reads
ground truth at estimate time. Scoring is in score_all.
"""
import json, multiprocessing as mp, os, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *


def world_up_rate(t, imu, ok, q, hop_s=0.2):
    """Gyro rate about world up, in blocks. Returns block times, rate, block valid."""
    fs = round(1.0 / np.median(np.diff(t)))
    hop = int(hop_s * fs)
    n = len(t) // hop
    Rm = R.from_quat(q[hop // 2 :: hop][:n][:, [1, 2, 3, 0]]).as_matrix()
    m = ok[: n * hop].reshape(n, hop, 1).astype(float)
    cnt = m.sum(1)
    w = (imu[: n * hop, :3].reshape(n, hop, 3) * m).sum(1) / np.maximum(cnt, 1)
    up = Rm[:, 2, :]                   # world up seen in the body frame
    return t[hop // 2 :: hop][:n], (w * up).sum(1), cnt[:, 0] > 0.8 * hop


def kde_mode(x, bw, lo=-0.25, hi=0.25, grid=0.002):
    """Mode of a kernel density, refined by mean shift. Returns mode and the density peak height per sample."""
    g = np.arange(lo, hi, grid)
    h, _ = np.histogram(x, bins=np.append(g, g[-1] + grid))
    kx = np.arange(-int(4 * bw / grid), int(4 * bw / grid) + 1) * grid
    k = np.exp(-0.5 * (kx / bw) ** 2)
    d = np.convolve(h, k, "same")
    m = g[int(np.argmax(d))] + grid / 2
    for _ in range(6):
        w = np.exp(-0.5 * ((x - m) / bw) ** 2)
        m = float((w * x).sum() / max(w.sum(), 1e-9))
    return m, float(d.max() / (len(x) * bw * 2.5066))


def yaw_drift(tb, wu, okb, win_s=60.0, step_s=10.0, bw=0.012, smooth=True):
    """Drift of the vertical rate over time. Knots every step_s, each from a window of win_s. Returns drift at tb."""
    T = tb[-1]
    win_s = min(win_s, max(T, 20.0))
    cs = np.arange(0.0, T + step_s, step_s)
    vals, conf = [], []
    for c in cs:
        s = (tb >= c - win_s / 2) & (tb <= c + win_s / 2) & okb
        if s.sum() < 20:
            vals.append(np.nan)
            conf.append(0.0)
            continue
        m, p = kde_mode(wu[s], bw)
        vals.append(m)
        conf.append(p)
    vals, conf = np.array(vals), np.array(conf)
    g = np.isfinite(vals)
    if not g.any():
        return np.zeros_like(tb)
    vals = np.interp(cs, cs[g], vals[g])
    if smooth and len(vals) >= 5:
        # slow prior: median then a light triangular smooth
        v2 = vals.copy()
        for i in range(len(vals)):
            v2[i] = np.median(vals[max(0, i - 2) : i + 3])
        vals = np.convolve(np.pad(v2, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    return np.interp(tb, cs, vals)


def apply_yaw(t, q, tb, d):
    """Removes the integral of the drift as a world-vertical rotation."""
    dr = np.interp(t, tb, d)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (dr[1:] + dr[:-1]) * np.diff(t))])
    rot = R.from_rotvec(np.stack([np.zeros_like(phi), np.zeros_like(phi), -phi], 1))
    return (rot * R.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]


def attitude_hdr(t, imu, ok, q=None, **kw):
    if q is None:
        q, _ = attitude_v2(t, imu, ok)
    tb, wu, okb = world_up_rate(t, imu, ok, q)
    d = yaw_drift(tb, wu, okb, **kw)
    return apply_yaw(t, q, tb, d), d


def pos_fixed(P, q):
    """A fixed position recipe, only to compare attitudes against each other."""
    ph = phys_channels(P["t"], P["imu"], P["ok"], q)
    if ph["drone"]:
        v, hp = ph["v_drag"] * 0.63, 1 / 60
    else:
        v, hp = ph["v_int"] * 0.36, 1 / 15
    v = decay_tail(v, ph["last"], P["t"])
    return hp_position(v, P["t"], hp)


def one(key):
    n, s, m, sd = key
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    room = load_room(n)
    out = {}
    q0, info = attitude_v2(P["t"], P["imu"], P["ok"])
    variants = {"v2": q0}
    for nm, kw in (("hdr60", dict(win_s=60)), ("hdr30", dict(win_s=30, step_s=5)), ("hdr120", dict(win_s=120, step_s=20)),
                   ("hdrwide", dict(win_s=60, bw=0.03))):
        variants[nm], _ = attitude_hdr(P["t"], P["imu"], P["ok"], q=q0, **kw)
    for nm, q in variants.items():
        p = pos_fixed(P, q)
        d = score_all(P, room, p, q)
        out[nm] = {k: d[k] for k in ("ate_se3", "ate_org", "rpe_t", "rpe_r", "att_org_pre", "tilt_pre", "ate_se3_pre")}
    return f"{n}|{s}|{m}|{sd}", out


if __name__ == "__main__":
    files = eval_set(list(EUROC) + DEV_ROOMS, seeds=(1, 2), modes=("zero",))
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(14) as p:
        for k, o in p.imap_unordered(one, files):
            res[k] = o
    json.dump(res, open(f"{RES}/h40_hdr.json", "w"))
    print(f"{len(res)} files {time.time()-t0:.0f}s")
    for grp, sel in (("EuRoC", lambda k: not k.split("|")[0].isdigit()), ("TUM", lambda k: k.split("|")[0].isdigit())):
        ks = [k for k in res if sel(k)]
        for sc in ("clean", "mild", "full"):
            sk = [k for k in ks if k.split("|")[1] == sc]
            print(f"-- {grp} {sc} ({len(sk)} files)")
            for nm in res[sk[0]]:
                print(f"  {nm:10s} " + " ".join(f"{m} {np.mean([res[k][nm][m] for k in sk]):.3f}/{np.max([res[k][nm][m] for k in sk]):.3f}" for m in ("att_org_pre", "tilt_pre", "ate_se3", "rpe_t")))
