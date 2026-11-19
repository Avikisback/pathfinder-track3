"""Fix for the drone weakness on hostile drift.

Body-plane accel is f = -k v + b. The drag velocity target used so far was -f/k, which carries an error of -b/k: 0.2 m/s^2 of
drift is about 1 m/s. Meanwhile the integrated velocity carries the integral of b. The solver saw a fixed target, so it could not
tell these apart. The two errors have a known relation: put the solved bias into the target as well, residual = v_int - R(v_drag + b/k).
Now the bias enters the residual as (integral of e) + e/k, which has only a fast-decaying null direction. Bias becomes observable
from the disagreement between the two velocities, with no assumption about the true motion.

Optional second idea: solve for one drag scale per file too, so k does not need to be known in advance.

Usage: 63_dragbias.py <vars comma separated> <levels comma separated>
"""
import multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from ev import *
from energy2 import cross_mat
from scipy.ndimage import uniform_filter1d

m46 = m57.m46
D9 = EXP + "/cache/v9"
os.makedirs(D9, exist_ok=True)


def solve_shift2(ec, Rfix, couple=True, fit_k=False, iters=8, sig_a=0.1, sig_g=0.01, sig_dv=1.0, sig_v=0.5, sig_p=5.0, smooth_s=5.0, row_s=1.0, sig_k=0.3, k=m46.K_DRAG, s0=0.0):
    fs_b = 1.0 / np.median(ec.dtb)
    smooth = max(1, int(smooth_s * fs_b))
    stride = max(1, int(row_s * fs_b))
    ec.sw = 1.0 / stride
    pr = dict(a=1 / sig_a ** 2, g=1 / sig_g ** 2, dv=1 / sig_dv ** 2)
    x = {kk: np.array(v, float) for kk, v in ec.x.items()}
    Ka, Kg, G, K = ec.Ka, ec.Kg, ec.G, ec.K
    nk = 1 if fit_k else 0
    np_total = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G + nk
    rows = np.arange(smooth // 2, len(ec.idx), stride)
    m = len(rows)
    dtb = ec.dtb
    Id = np.tile(np.eye(3), (len(dtb), 1, 1))
    sv, sp = 1.0 / sig_v, 1.0 / sig_p
    on = ec.on_b.astype(float)
    Pl = ec.E @ ec.E.T                     # body-plane projector
    base = np.einsum("nij,nj->ni", Rfix, ec.vb_t)
    RP = Rfix @ Pl                         # n x 3 x 3

    def shift_of(x, s):
        out = base * (1 + s)
        if couple:
            bb = ec.Ba @ x["ca"]
            out = out + np.einsum("nij,nj->ni", RP, bb) / k
        return out * on[:, None]

    def obj(v, x, s):
        d = uniform_filter1d(v - shift_of(x, s), smooth, axis=0, mode="nearest")
        p = np.cumsum(v * dtb[:, None], axis=0)
        p = p - p.mean(0)
        return ((d ** 2).sum() * sv ** 2 + (p ** 2).sum() * sp ** 2) * ec.sw + (x["ca"] ** 2).sum() * pr["a"] + (x["cg"] ** 2).sum() * pr["g"] + (x["dv"] ** 2).sum() * pr["dv"] + s * s / sig_k ** 2

    s = float(s0)
    v, Rg, Rm, fw, Sm = ec.forward(x)
    best = obj(v, x, s)
    lam = 1e-3
    for it in range(iters):
        Fx = cross_mat(fw)
        A = np.zeros((6 * m, np_total))

        def put(col, M, Ms=None):
            Mv = M if Ms is None else M - Ms
            sm = uniform_filter1d(Mv, smooth, axis=0, mode="nearest")
            A[: 3 * m, col : col + M.shape[2]] = sv * sm[rows].reshape(3 * m, M.shape[2])
            P = np.cumsum(M * dtb[:, None, None], axis=0)
            P = P - P.mean(0, keepdims=True)
            A[3 * m :, col : col + M.shape[2]] = sp * P[rows].reshape(3 * m, M.shape[2])

        put(0, Id)
        col = 3
        for kk in range(K):
            mk = (ec.seg_of == kk).astype(float)
            put(col, np.cumsum(-Fx * (dtb * mk)[:, None, None], axis=0)[:, :, :2])
            col += 2
        wdt = dtb * ec.okb
        for j in range(Ka):
            M = np.cumsum(-Rm * (ec.Ba[:, j] * wdt)[:, None, None], axis=0)
            put(col, M, RP * (ec.Ba[:, j] * on)[:, None, None] / k if couple else None)
            col += 3
        Sb = np.zeros_like(Rm)
        Sb[:] = np.eye(3)
        for kk in range(K):
            Sb[ec.seg_of == kk] = Sm[kk]
        for j in range(Kg):
            Phi = Sb @ np.cumsum(-Rg * (ec.Bg[:, j] * dtb)[:, None, None], axis=0)
            put(col, np.cumsum(-np.einsum("nij,njk->nik", Fx, Phi) * dtb[:, None, None], axis=0))
            col += 3
        for j, (a, b) in enumerate(ec.gaps):
            put(col, Id * ec._ramp(a, b)[:, None, None])
            col += 3
        if fit_k:
            sm = uniform_filter1d(base * on[:, None], smooth, axis=0, mode="nearest")
            A[: 3 * m, col] = -sv * sm[rows].ravel()
            col += 1
        d = uniform_filter1d(v - shift_of(x, s), smooth, axis=0, mode="nearest")
        pc = np.cumsum(v * dtb[:, None], axis=0)
        pc = pc - pc.mean(0)
        y = np.concatenate([-sv * d[rows].ravel(), -sp * pc[rows].ravel()])
        pri_list = [np.full(3, 1 / 25.0), np.zeros(2 * K), np.full(3 * Ka, pr["a"]), np.full(3 * Kg, pr["g"]), np.full(3 * G, pr["dv"])]
        cur_list = [x["v0"], np.zeros(2 * K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()]
        if fit_k:
            pri_list.append(np.array([1 / sig_k ** 2]))
            cur_list.append(np.array([s]))
        pri, cur = np.concatenate(pri_list), np.concatenate(cur_list)
        H = A.T @ A + np.diag(pri)
        rhs = A.T @ y - pri * cur
        D = np.diag(np.maximum(np.diag(H), 1e-9))
        ok = False
        for _ in range(6):
            dx = np.linalg.solve(H + lam * D, rhs)
            xn = ec._apply(x, dx[: np_total - nk])
            sn = s + (dx[-1] if fit_k else 0.0)
            r = ec.forward(xn)
            o = obj(r[0], xn, sn)
            if o < best:
                x, best, s = xn, o, sn
                v, Rg, Rm, fw, Sm = r
                lam = max(lam / 3.0, 1e-6)
                ok = True
                break
            lam *= 8.0
        if not ok:
            break
    ec.x = x
    q, vout, tb, last, info = ec.finish(x, v, float(best))
    info["scale"] = s
    info["target"] = shift_of(x, s)
    return q, vout, tb, last, info


def solve_drag2(ec, auto_p=1.5, outer=4, **kw):
    s = 0.0
    for o in range(outer):
        v, Rg, Rm, fw, Sm = ec.forward(ec.x)
        if auto_p and o > 0:
            pos = np.cumsum(v * ec.dtb[:, None], axis=0)
            kw["sig_p"] = float(np.clip(auto_p * np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()), 1.0, 20.0))
        q, vs, tb, last, info = solve_shift2(ec, Rm, s0=s, **kw)
        s = info["scale"]
    info["sig_p"] = kw.get("sig_p")
    return q, vs, tb, last, info


VARS = {
    "c": dict(couple=True, fit_k=False, sig_v=0.5),
    "ck": dict(couple=True, fit_k=True, sig_v=0.5),
    "c3": dict(couple=True, fit_k=False, sig_v=0.3),
    "c8": dict(couple=True, fit_k=False, sig_v=0.8),
}


def solve_file(n, k, lvl, var):
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    lv = m48.LEVELS[lvl]
    b0 = m41.calib_excl(n)
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=b0, knot_a=lv["knot"], knot_g=lv["knot"])
    ec.set_target()
    assert ec.drone
    q, v, tb, last, info = solve_drag2(ec, sig_a=lv["sig_a"], sig_g=lv["sig_g"], **VARS[var])
    tgt = info["target"]
    return to_cache(q, v, tb, True, scale=info["scale"], target_k=np.stack([np.interp(np.arange(tb[0], tb[-1], DT), tb, tgt[:, i]) for i in range(3)], 1))


def one(key):
    n, k, lvl, var = key
    f = f"{D9}/{n}_{k[0]}_{k[1]}_{k[2]}_{lvl}_{var}.pkl"
    if not os.path.exists(f):
        with open(f, "wb") as fh:
            pickle.dump(solve_file(n, k, lvl, var), fh)
    return key


def loader_for(lvl, var):
    def load(n, k):
        return pickle.load(open(f"{D9}/{n}_{k[0]}_{k[1]}_{k[2]}_{lvl}_{var}.pkl", "rb"))
    return load


if __name__ == "__main__":
    vs = sys.argv[1].split(",")
    lvls = sys.argv[2].split(",")
    keys = [(n, k, l, v) for l in lvls for v in vs for n in SEQS for k in ALLK]
    t0 = time.time()
    with mp.get_context("spawn").Pool(16) as p:
        for _ in p.imap_unordered(one, keys, chunksize=2):
            pass
    print(f"solved {len(keys)} in {time.time()-t0:.0f}s", flush=True)
    base = evaluate(SEQS, EU_FOLDS, ALLK, lambda n, k: m57.load(n, *k, LVL))
    table(base, "EuRoC frozen L15 (reference)")
    allr = {"base": base}
    for l in lvls:
        for v in vs:
            r = evaluate(SEQS, EU_FOLDS, ALLK, loader_for(l, v))
            allr[(l, v)] = r
            table(r, f"EuRoC coupled target, {l}, {v}")
            sc = [loader_for(l, v)(n, ("full", "zero", 1))["scale"] for n in SEQS]
            print("  drag scale per file on full:", np.round(sc, 3))
    pickle.dump(allr, open(f"{RES}/dragbias.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
