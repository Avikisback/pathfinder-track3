"""Handheld questions.

A. Is walking direction tied to the body? Resultant length of the horizontal velocity direction in the body-yaw frame (1 = always the same
   direction relative to the rig, 0 = no relation), truth only.
B. Gait-band PCA: does the principal axis of horizontal accel in the step band line up with the direction of travel? Truth attitude, clean
   rows, angle modulo 180 degrees.
C. New idea: yaw across a gap. A handheld rig turns at about 0.7 rad/s, so a 2 s gap can hide over 80 degrees of heading, and each gap
   then rotates everything after it. Same principle as the solver: pick the yaw jumps that keep the whole track most compact. The search is
   cheap because a yaw jump only rotates already-known position pieces.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Slerp
mb = importlib.import_module("60_budget")


def direction_stats(n):
    room = load_room(n)
    t = (room["t_ns"] - room["t_ns"][0]) * 1e-9
    gt_t = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
    T = np.arange(gt_t[0] + 1, gt_t[-1] - 1, 0.1)
    f = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    v = (f(T + 0.25) - f(T - 0.25)) / 0.5
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(T).as_matrix()
    sp = np.linalg.norm(v[:, :2], axis=1)
    m = sp > 0.3
    # direction of travel relative to the horizontal projection of each body axis
    out = {}
    for name, c in (("x", 0), ("y", 1), ("z", 2)):
        ax = Rg[:, :, c][:, :2]
        na = np.linalg.norm(ax, axis=1)
        ok = m & (na > 0.3)
        ang = np.arctan2(ax[:, 1], ax[:, 0])
        d = np.arctan2(v[:, 1], v[:, 0]) - ang
        out[name] = (float(np.abs(np.mean(np.exp(1j * d[ok])))), float(np.degrees(np.angle(np.mean(np.exp(1j * d[ok]))))), float(ok.mean()))
    # gait band PCA
    fs = 200
    idx = np.arange(len(t))
    b, a = butter(2, [1.0 / (fs / 2), 3.5 / (fs / 2)], btype="band")
    Rr = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(t, gt_t[0], gt_t[-1])).as_matrix()
    aw = np.einsum("nij,nj->ni", Rr, room["imu"][:, 3:]) - [0, 0, np.median(np.linalg.norm(room["imu"][:, 3:], axis=1))]
    ag = filtfilt(b, a, aw[:, :2], axis=0)
    W = int(3 * fs)
    errs, weights, band_share = [], [], []
    for s in range(0, len(t) - W, W // 3):
        e = s + W
        tc = t[s + W // 2]
        if tc < gt_t[0] + 2 or tc > gt_t[-1] - 2:
            continue
        vv = (f(np.array([tc + 1.0])) - f(np.array([tc - 1.0])))[0] / 2.0
        if np.linalg.norm(vv[:2]) < 0.3:
            continue
        C = ag[s:e].T @ ag[s:e]
        w, U = np.linalg.eigh(C)
        ax = U[:, 1]
        d = np.arctan2(vv[1], vv[0]) - np.arctan2(ax[1], ax[0])
        d = (d + np.pi / 2) % np.pi - np.pi / 2
        errs.append(d)
        weights.append(w[1] / max(w[0], 1e-9))
    errs = np.array(errs)
    return n, out, float(np.degrees(np.sqrt(np.mean(errs ** 2)))), float(np.degrees(np.median(np.abs(errs)))), float(np.mean(weights))


def gap_edges(P, tk):
    """Times (mid) of gaps with data on both sides."""
    from attitude import find_gaps
    g = [(s, e) for s, e in find_gaps(P["ok"]) if e < len(P["t"]) and s > 0 and e <= P["last_good"]]
    return [(P["t"][s], P["t"][e - 1]) for s, e in g]


def rot_z(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]]


def track_with_jumps(vf, tk, gaps, psi):
    """Velocity after each gap is rotated by the sum of the jumps so far; integrate."""
    v = vf.copy()
    for (a, b), ps in zip(gaps, psi):
        m = tk > 0.5 * (a + b)
        v[m] = rot_z(v[m], ps)
    return v


def spread(vf, dt=0.1):
    x = np.cumsum(vf, axis=0) * dt
    x = x - x.mean(0)
    return float((x[:, :2] ** 2).sum(1).mean())


def solve_jumps(vf, tk, gaps, sig, rounds=3, grid=np.radians(np.arange(-180, 180, 10.0))):
    psi = np.zeros(len(gaps))
    # pieces between gaps as position increments, so a candidate only rotates and re-adds pieces
    if not len(gaps):
        return psi
    edges = [0] + [int(np.searchsorted(tk, 0.5 * (a + b))) for a, b in gaps] + [len(tk)]
    pieces = [np.cumsum(vf[edges[i]:edges[i + 1]], axis=0) * 0.1 for i in range(len(edges) - 1)]
    n = len(tk)

    def build(psi):
        out = np.zeros((n, 3))
        base = np.zeros(3)
        ang = 0.0
        for i, pc in enumerate(pieces):
            if i > 0:
                ang += psi[i - 1]
            q = rot_z(pc, ang)
            out[edges[i]:edges[i + 1]] = q + base
            base = base + q[-1] if len(q) else base
        return out

    def cost(psi):
        x = build(psi)
        x = x - x.mean(0)
        return (x[:, :2] ** 2).sum(1).mean()

    for r in range(rounds):
        for j in range(len(gaps)):
            cand = []
            for g in grid:
                p2 = psi.copy()
                p2[j] = g
                cand.append(cost(p2) + (g / sig[j]) ** 2 * 0.05)
            psi[j] = grid[int(np.argmin(cand))]
    return psi


def gap_trial(a):
    n, k, fit, sig_scale = a
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    d = m57.load(n, *k, LVL)
    q = d["q"].astype(np.float64)
    sc = PosScorer(P, room, q)
    vf = m58.final_v(d, fit, *CFG)
    tk = d["tk"]
    gaps = [g for g in gap_edges(P, tk) if g[1] < tk[-1] - 1]
    # truth of yaw jumps
    vt, good, Rg = mb.gt_at(room, P, tk)
    idx = np.clip(np.searchsorted(P["t"], tk), 0, len(P["t"]) - 1)
    Re = R.from_quat(q[idx][:, [1, 2, 3, 0]])
    Q = Rg * Re.inv()
    Q0 = Q[good].mean()
    M = (Q * Q0.inv()).as_matrix()
    dpsi = np.unwrap(np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1]))
    true_j = []
    for a_, b_ in gaps:
        i0, i1 = np.searchsorted(tk, a_ - 0.3), np.searchsorted(tk, b_ + 0.3)
        true_j.append(dpsi[i1] - dpsi[i0])
    true_j = np.array(true_j)
    # prior width from the rig's own turn rate
    from attitude import find_gaps
    wrms = float(np.sqrt(np.mean((P["imu"][P["ok"], :3] ** 2).sum(1))))
    sig = np.array([max(0.2, sig_scale * wrms * (b_ - a_)) for a_, b_ in gaps])
    psi = solve_jumps(vf, tk, gaps, sig) if len(gaps) else np.zeros(0)
    tr = lambda v: sc(m57.track(P, d, v))
    o = dict(base=tr(vf), stationary=sc(np.zeros((len(P["t"]), 3))))
    o["solved"] = tr(track_with_jumps(vf, tk, gaps, psi))
    o["oracle"] = tr(track_with_jumps(vf, tk, gaps, true_j))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    o["jump_true"] = true_j
    o["jump_err"] = wrap(psi - true_j)
    return (n, *k), o


if __name__ == "__main__":
    t0 = time.time()
    print("=== A/B direction of travel against the body, truth attitude, per room")
    for n in DEV_ROOMS + SEQS:
        n_, out, rms, med, ratio = direction_stats(n)
        print(f"  {str(n):5s} resultant length & mean angle: " + " ".join(f"{a}: {out[a][0]:.2f} @ {out[a][1]:6.1f} ({out[a][2]:.2f})" for a in out) + f" | gait PCA vs travel axis: rms {rms:.1f} deg, median {med:.1f} deg, eigen ratio {ratio:.1f}", flush=True)
    print("\n=== C. yaw jump across gaps chosen by compactness")
    for names, folds, title in ((DEV_ROOMS, TU_FOLDS, "TUM"), (SEQS, EU_FOLDS, "EuRoC")):
        rooms = {n: load_room(n) for n in names}
        F = fits(folds, lambda n, k: m57.load(n, *k, LVL), rooms)
        for scale in (0.5, 1.0, 2.0):
            jobs = [(n, k, F[n], scale) for n in names for k in KNOWN + STEPS + UNSEEN if k[0] != "clean"]
            res = {}
            with mp.get_context("fork").Pool(14) as p:
                for key, o in p.imap_unordered(gap_trial, jobs, chunksize=4):
                    res[key] = o
            print(f"-- {title}, prior width factor {scale}")
            for kd in ("mild", "full", "burst", "steps", "randomwalk", "sine"):
                ks = [k for k in res if k[1] == kd]
                je = np.concatenate([res[k]["jump_err"] for k in ks])
                jt = np.concatenate([res[k]["jump_true"] for k in ks])
                print(f"   {kd:10s} stat {np.mean([res[k]['stationary']['ate_se3'] for k in ks]):.2f} base {np.mean([res[k]['base']['ate_se3'] for k in ks]):.2f} solved {np.mean([res[k]['solved']['ate_se3'] for k in ks]):.2f} oracle {np.mean([res[k]['oracle']['ate_se3'] for k in ks]):.2f} | true jump rms {np.degrees(np.sqrt(np.mean(jt**2))):5.1f} deg, error after search rms {np.degrees(np.sqrt(np.mean(je**2))):5.1f} deg", flush=True)
    print(f"done {time.time()-t0:.0f}s")
