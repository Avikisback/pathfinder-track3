"""Three follow-ups from the error budget.

1. Drones: all remaining error is horizontal velocity below 1/16 Hz, attitude is not the limit, and the drag velocity has coherence
   0.5-0.9 with truth there but the wrong gain. So fuse per octave band: v = a_b * solver + c_b * drag, two numbers per band learned
   on other environments (a two-input Wiener filter).
2. Handheld: does the gap yaw search help once the band gains are refitted on tracks treated the same way? With the true jumps as the
   upper limit.
3. Handheld: are gaps the cause of the clean-to-full loss? Schedules with only gaps, and with everything except gaps.
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from common import SCHEDULES
m46 = m57.m46
mb = importlib.import_module("60_budget")
mw = importlib.import_module("66_walk")
SCHEDULES["gaponly"] = dict(sg=(0.0, 0.0), sa=(0.0, 0.0), bg=0.0, ba=0.0, gaps=3, gap_s=2.0)
SCHEDULES["nogap"] = dict(sg=(0.002, 0.008), sa=(0.02, 0.08), bg=0.02, ba=0.2, gaps=0, gap_s=2.0)
D12 = EXP + "/cache/v12"
os.makedirs(D12, exist_ok=True)


# ------------------------------------------------------------------ 1. two-input band fusion on drones
def drag_world(n, k, d):
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=m41.calib_excl(n))
    ec.set_target()
    q = d["q"].astype(np.float64)
    i = np.clip(np.searchsorted(P["t"], ec.tb), 0, len(P["t"]) - 1)
    Rm = R.from_quat(q[i][:, [1, 2, 3, 0]]).as_matrix()
    w = np.einsum("nij,nj->ni", Rm, ec.vb_t) * ec.on_b[:, None]
    tk = d["tk"]
    return np.stack([np.interp(tk, ec.tb, w[:, j]) for j in range(3)], 1)


def file_pack(a):
    n, k = a
    d = m57.load(n, *k, LVL)
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    vd = drag_world(n, k, d)
    vt, good, _ = mb.gt_at(room, P, d["tk"])
    return (n, k), dict(vd=vd, vt=vt, good=good)


def bands_of(f):
    return np.clip(np.digitize(f, eb.EDGES_B) - 1, 0, eb.NB - 1)


def fit_fusion(items):
    """items: list of (vs, vd, vt) already in the truth frame, mean removed. Per band: 2x2 least squares for xy pooled."""
    A = np.zeros((eb.NB, 2, 2))
    b = np.zeros((eb.NB, 2))
    for vs, vd, vt in items:
        S, f = eb.spec(vs)
        Dd, _ = eb.spec(vd)
        T, _ = eb.spec(vt)
        bi = bands_of(f)
        for ax in (0, 1):
            X = np.stack([S[:, ax], Dd[:, ax]], 1)
            for j in range(eb.NB):
                m = bi == j
                Xm, Tm = X[m], T[m, ax]
                A[j] += (np.conj(Xm).T @ Xm).real
                b[j] += (np.conj(Xm).T @ Tm).real
    return np.stack([np.linalg.solve(A[j] + 1e-9 * np.eye(2), b[j]) for j in range(eb.NB)])


def apply_fusion(vs, vd, C, gz):
    S, f = eb.spec(vs)
    Dd, _ = eb.spec(vd)
    bi = bands_of(f)
    Y = S.copy()
    Y[:, 0] = C[bi, 0] * S[:, 0] + C[bi, 1] * Dd[:, 0]
    Y[:, 1] = C[bi, 0] * S[:, 1] + C[bi, 1] * Dd[:, 1]
    cen = np.sqrt(np.maximum(eb.EDGES_B[:-1], 1 / 512) * np.minimum(eb.EDGES_B[1:], 4))
    Y[:, 2] = S[:, 2] * np.interp(np.log(np.maximum(f, 1e-6)), np.log(cen), gz)
    Y[0] = 0
    return np.fft.irfft(Y, n=2 * len(vs), axis=0)[: len(vs)]


def to_truth_frame(vs, vt, good):
    Rg = eb.yaw_gauge(vs[good], vt[good])
    return (Rg @ vs.T).T


def part1():
    kinds = KNOWN + UNSEEN + STEPS
    jobs = [(n, k) for n in SEQS for k in kinds]
    with mp.get_context("spawn").Pool(16) as p:
        packs = dict(p.map(file_pack, jobs, chunksize=2))
    rows = {}
    rooms = {n: load_room(n) for n in SEQS}
    F = fits(EU_FOLDS, lambda n, k: m57.load(n, *k, LVL), rooms)
    for test, train in EU_FOLDS:
        items = []
        for n in train:
            for k in KNOWN:
                d = m57.load(n, *k, LVL)
                pk = packs[(n, k)]
                vs, vd = d["v"] - d["v"].mean(0), pk["vd"] - pk["vd"].mean(0)
                Rg = eb.yaw_gauge(vs[pk["good"]], pk["vt"][pk["good"]])
                items.append(((Rg @ vs.T).T, (Rg @ vd.T).T, np.where(pk["good"][:, None], pk["vt"], (Rg @ vs.T).T) - pk["vt"][pk["good"]].mean(0)))
        C = fit_fusion(items)
        print(f"fold {test[0][:2]} fusion coefficients per band (solver, drag):", np.round(C.T, 2).tolist(), flush=True)
        for n in test:
            for k in kinds:
                d = m57.load(n, *k, LVL)
                pk = packs[(n, k)]
                t, imu, room = file_for(n, k)
                P = prepare(t, imu)
                sc = PosScorer(P, room, d["q"].astype(np.float64))
                vs, vd = d["v"] - d["v"].mean(0), pk["vd"] - pk["vd"].mean(0)
                fit = F[n]
                vf = apply_fusion(vs, vd, C, fit["gz"])
                tr = lambda v: sc(m57.track(P, d, v))
                o = dict(stationary=sc(np.zeros((len(P["t"]), 3))), frozen=tr(m58.final_v(d, fit, *CFG)), nocheck=tr(m57.apply(d, fit, "eb")), fused=tr(vf))
                # same self-check as the frozen entry, fallback unchanged
                o["fused_check"] = tr(m58.hp(d, CFG[2]) if m57.slow_rms(d) > CFG[0] else vf)
                rows[(n, *k)] = o
    print("\n=== 1. EuRoC two-input band fusion (unseen environment), mean ATE SE3 / RPE 1 s")
    for kd in NAMES:
        ks = [k for k in rows if k[1] == kd]
        print(f"  {kd:10s} " + " | ".join(f"{m} {np.mean([rows[k][m]['ate_se3'] for k in ks]):.2f}/{np.mean([rows[k][m]['rpe_t'] for k in ks]):.2f}" for m in ("stationary", "frozen", "nocheck", "fused", "fused_check")), flush=True)
    return rows


# ------------------------------------------------------------------ 2. gap yaw with refitted gains
def jump_versions(a):
    n, k, fit = a
    d = m57.load(n, *k, LVL)
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    tk = d["tk"]
    gaps = [g for g in mw.gap_edges(P, tk) if g[1] < tk[-1] - 1]
    v = d["v"]
    out = {"none": v}
    if not gaps:
        out["solved"] = v
        out["oracle"] = v
        return (n, k), out
    q = d["q"].astype(np.float64)
    vt, good, Rg = mb.gt_at(room, P, tk)
    idx = np.clip(np.searchsorted(P["t"], tk), 0, len(P["t"]) - 1)
    Re = R.from_quat(q[idx][:, [1, 2, 3, 0]])
    Q = Rg * Re.inv()
    M = (Q * Q[good].mean().inv()).as_matrix()
    dpsi = np.unwrap(np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1]))
    tj = np.array([dpsi[np.searchsorted(tk, b_ + 0.3)] - dpsi[np.searchsorted(tk, a_ - 0.3)] for a_, b_ in gaps])
    wrms = float(np.sqrt(np.mean((P["imu"][P["ok"], :3] ** 2).sum(1))))
    sig = np.array([max(0.2, wrms * (b_ - a_)) for a_, b_ in gaps])
    # search on the file's own band-filtered track, then apply the jumps to the raw velocity
    vf = m57.apply(d, fit, "band")
    psi = mw.solve_jumps(vf, tk, gaps, sig)
    out["solved"] = mw.track_with_jumps(v, tk, gaps, psi)
    out["oracle"] = mw.track_with_jumps(v, tk, gaps, -tj)
    out["oracle_pos"] = mw.track_with_jumps(v, tk, gaps, tj)
    return (n, k), out


def part2():
    rooms = {n: load_room(n) for n in DEV_ROOMS}
    F = fits(TU_FOLDS, lambda n, k: m57.load(n, *k, LVL), rooms)
    kinds = KNOWN + UNSEEN + STEPS
    jobs = [(n, k, F[n]) for n in DEV_ROOMS for k in kinds]
    with mp.get_context("spawn").Pool(16) as p:
        V = dict(p.map(jump_versions, jobs, chunksize=2))
    print("\n=== 2. TUM gap yaw, band gains refitted on tracks treated the same way (unseen room)")
    for var in ("none", "solved", "oracle", "oracle_pos"):
        def L(n, k, var=var):
            d = dict(m57.load(n, *k, LVL))
            d["v"] = V[(n, k)][var]
            return d
        r = evaluate(DEV_ROOMS, TU_FOLDS, kinds, L)
        print(f"  {var:10s} final: " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, "final").items()))
        print(f"  {'':10s} nocheck: " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, "nocheck").items()), flush=True)


# ------------------------------------------------------------------ 3. what does each part of the degradation cost on handheld?
def solve_v12(a):
    n, k = a
    f = f"{D12}/{n}_{k[0]}_{k[1]}_{k[2]}.pkl"
    if not os.path.exists(f):
        t, imu, room = file_for(n, k)
        P = prepare(t, imu)
        q, v, tb, last, info = m48.solve_level(P, m41.calib_excl(n), m48.LEVELS[LVL])
        pickle.dump(to_cache(q, v, tb, info["drone"]), open(f, "wb"))
    return a


def part3():
    new = [("gaponly", "zero", 1), ("gaponly", "zero", 2), ("nogap", "zero", 1), ("nogap", "zero", 2)]
    jobs = [(n, k) for n in DEV_ROOMS + SEQS for k in new]
    with mp.get_context("spawn").Pool(16) as p:
        p.map(solve_v12, jobs, chunksize=2)

    def L(n, k):
        if k[0] in ("gaponly", "nogap"):
            return pickle.load(open(f"{D12}/{n}_{k[0]}_{k[1]}_{k[2]}.pkl", "rb"))
        return m57.load(n, *k, LVL)

    import ev as _ev
    _ev.NAMES = ("clean", "gaponly", "nogap", "full")
    print("\n=== 3. which part of the degradation costs what")
    for names, folds, title in ((DEV_ROOMS, TU_FOLDS, "TUM"), (SEQS, EU_FOLDS, "EuRoC")):
        r = evaluate(names, folds, [("clean", "delete", 1)] + new + [("full", "zero", 1)], L)
        for me in ("final", "nocheck", "raw"):
            print(f"  {title:5s} {me:8s} " + " ".join(f"{a} {b:.2f}" for a, b in summary(r, me).items()), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        r1 = part1()
        pickle.dump(r1, open(f"{RES}/fuse_eu.pkl", "wb"))
    if "2" in which:
        part2()
    if "3" in which:
        part3()
    print(f"done {time.time()-t0:.0f}s")
