"""Trials for the priors/model-selection notes. Does not touch the team's caches: writes only under argv[1].

Reimplements the LM core of DragCal/EnergyCal so that H, the rows and the prior are visible, with options:
  - evidence numbers per level (BIC, Laplace, block CV) for drift-model selection
  - rw penalty with evidence (Fellner-Schall / MacKay) updates per file
  - IRLS Student-t weights on velocity rows
  - multi-scale (octave) velocity energy with a learned table
  - Laplace sampling of the posterior for a posterior-mean output

Folds and scorer are the frozen entry's (ev.py). Ground truth is only used for scoring, for band gains on other environments
and for the learned motion table on other environments.
"""
import os, pickle, sys, time, multiprocessing as mp
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import ev
from ev import *
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.ndimage import uniform_filter1d
from energy2 import cross_mat

m46 = m48.m46
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
KEYS_KNOWN = [("clean", "delete", 1), ("mild", "zero", 1), ("full", "zero", 1)]
KEYS_NEW = [("burst", "x", 1), ("steps", "x", 1), ("randomwalk", "x", 1), ("sine", "x", 1)]
KEYS = KEYS_KNOWN + KEYS_NEW
ev.KNOWN = KEYS_KNOWN
LV = {"L60": (60.0, 0.1, 0.01), "L15": (15.0, 0.3, 0.03), "L8": (8.0, 0.5, 0.05)}
NAMES_ALL = list(SEQS) + list(DEV_ROOMS)


def core(ec, shift, iters, sig_a, sig_g, sig_v, sig_p, sig_off=None, sig_dv=1.0, smooth_s=5.0, row_s=1.0, pen="amp", robust=None, bands=None, kappa=1.0, eb_passes=0, nu=4.0, amp_g_sig=0.1):
    fs_b = 1.0 / np.median(ec.dtb)
    smooth = max(1, int(smooth_s * fs_b))
    stride = max(1, int(row_s * fs_b))
    ec.sw = 1.0 / stride
    x = {k: np.array(v, float) for k, v in ec.x.items()}
    Ka, Kg, G, K = ec.Ka, ec.Kg, ec.G, ec.K
    npar = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
    rows = np.arange(smooth // 2, len(ec.idx), stride)
    m = len(rows)
    dtb = ec.dtb
    Id = np.tile(np.eye(3), (len(dtb), 1, 1))
    sp = 1.0 / sig_p
    sv_b = np.where(ec.on_b, 1.0 / sig_v, 1.0 / (sig_off if sig_off else sig_v))
    w_b = np.ones(len(dtb))
    ia = 3 + 2 * K
    ig = ia + 3 * Ka

    def diff_pen(n):
        D = np.zeros((n - 1, n))
        for i in range(n - 1):
            D[i, i], D[i, i + 1] = -1.0, 1.0
        return np.kron(D.T @ D, np.eye(3))

    S_a = diff_pen(Ka) if pen == "rw" else np.eye(3 * Ka)
    S_g = diff_pen(Kg) if pen == "rw" else np.eye(3 * Kg)
    alpha = dict(a=1.0 / sig_a ** 2, g=1.0 / sig_g ** 2)
    amp_a, amp_g = (1 / 1.0 ** 2, 1 / amp_g_sig ** 2) if pen == "rw" else (0.0, 0.0)  # weak amplitude anchor on the rw form

    def prior():
        P = np.zeros((npar, npar))
        P[:3, :3] = np.eye(3) / 25.0
        P[ia:ia + 3 * Ka, ia:ia + 3 * Ka] = alpha["a"] * S_a + amp_a * np.eye(3 * Ka)
        P[ig:ig + 3 * Kg, ig:ig + 3 * Kg] = alpha["g"] * S_g + amp_g * np.eye(3 * Kg)
        if G:
            P[ig + 3 * Kg:, ig + 3 * Kg:] = np.eye(3 * G) / sig_dv ** 2
        return P

    def vec(x):
        return np.concatenate([x["v0"], np.zeros(2 * K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()])

    def vel_rows(X):
        """X (N,3,...) -> weighted stacked velocity rows."""
        if bands is None:
            sm = uniform_filter1d(X, smooth, axis=0, mode="nearest")[rows]
            wt = np.repeat(sv_b[rows] * np.sqrt(w_b[rows]), 3)
            s = (sm.shape[0] * 3,) + sm.shape[2:]
            r = sm.reshape(s)
            return r * wt.reshape((-1,) + (1,) * (r.ndim - 1))
        out = []
        for T_s, wgt in bands:
            n1 = max(1, int(T_s * fs_b))
            a = uniform_filter1d(X, n1, axis=0, mode="nearest") - uniform_filter1d(X, 2 * n1, axis=0, mode="nearest")
            rr = np.arange(n1, len(dtb) - n1, n1)
            s = (len(rr) * 3,) + a.shape[2:]
            out.append(wgt * a[rr].reshape(s))
        return np.concatenate(out, 0)

    def pos_rows(X):
        P = np.cumsum(X * (dtb if X.ndim == 2 else dtb)[(slice(None),) + (None,) * (X.ndim - 1)], axis=0)
        P = P - P.mean(0, keepdims=True)
        s = (m * 3,) + P.shape[2:]
        return sp * P[rows].reshape(s)

    def obj(v, x):
        P = prior()
        c = vec(x)
        if bands is None:
            d = uniform_filter1d(v - shift, smooth, axis=0, mode="nearest")
            jv = (((d * sv_b[:, None]) ** 2).sum(1) * w_b).sum() * ec.sw
        else:
            jv = (vel_rows(v - shift) ** 2).sum()
        p = np.cumsum(v * dtb[:, None], axis=0)
        p = p - p.mean(0)
        return jv + (p ** 2).sum() * sp ** 2 * ec.sw + c @ P @ c - (x["v0"] ** 2).sum() / 25.0

    v, Rg, Rm, fw, Sm = ec.forward(x)
    best = obj(v, x)
    lam = 1e-3
    A = H = y = P = None
    log_alpha = []
    nit = 0
    for it in range(iters):
        nit += 1
        Fx = cross_mat(fw)
        cols = []
        cols.append((vel_rows(Id), pos_rows(Id)))
        for k in range(K):
            mk = (ec.seg_of == k).astype(float)
            M = np.cumsum(-Fx * (dtb * mk)[:, None, None], axis=0)[:, :, :2]
            cols.append((vel_rows(M), pos_rows(M)))
        wdt = dtb * ec.okb
        for j in range(Ka):
            M = np.cumsum(-Rm * (ec.Ba[:, j] * wdt)[:, None, None], axis=0)
            cols.append((vel_rows(M), pos_rows(M)))
        Sb = np.zeros_like(Rm)
        Sb[:] = np.eye(3)
        for k in range(K):
            Sb[ec.seg_of == k] = Sm[k]
        for j in range(Kg):
            Phi = Sb @ np.cumsum(-Rg * (ec.Bg[:, j] * dtb)[:, None, None], axis=0)
            M = np.cumsum(-np.einsum("nij,njk->nik", Fx, Phi) * dtb[:, None, None], axis=0)
            cols.append((vel_rows(M), pos_rows(M)))
        for j, (s, e) in enumerate(ec.gaps):
            M = Id * ec._ramp(s, e)[:, None, None]
            cols.append((vel_rows(M), pos_rows(M)))
        A = np.concatenate([np.concatenate([c[0] for c in cols], 1), np.concatenate([c[1] for c in cols], 1)], 0)
        y = -np.concatenate([vel_rows(v - shift), pos_rows(v)])
        P = prior()
        cur = vec(x)
        H = A.T @ A + P
        rhs = A.T @ y - P @ cur
        D = np.diag(np.maximum(np.diag(H), 1e-9))
        ok = False
        for _ in range(6):
            dx = np.linalg.solve(H + lam * D, rhs)
            xn = ec._apply(x, dx)
            r = ec.forward(xn)
            o = obj(r[0], xn)
            if o < best:
                x, best = xn, o
                v, Rg, Rm, fw, Sm = r
                lam = max(lam / 3.0, 1e-6)
                ok = True
                break
            lam *= 8.0
        # robust: reweight velocity rows from the current residual, after a first plain pass (graduated start)
        if robust and it >= 2 and bands is None:
            d = uniform_filter1d(v - shift, smooth, axis=0, mode="nearest") * sv_b[:, None]
            r2 = (d ** 2).sum(1)
            s2 = max(np.median(r2) / 2.366, 1e-12)
            if robust == "t":
                w_b = (nu + 3.0) / (nu + r2 / s2) * (nu / (nu + 3.0)) * 1.0
            elif robust == "gm":
                mu = max(1.0, 2.0 * r2.max() / (9.0 * s2) / 1.4 ** (it - 2))
                w_b = (mu * 9.0 * s2 / (r2 + mu * 9.0 * s2)) ** 2
            best = obj(v, x)
            ok = True
        # evidence update of the drift stiffness (Fellner-Schall form), kappa = rows per independent sample
        if eb_passes and it >= 2 and (it % 2 == 0):
            Hi = np.linalg.inv(A.T @ A + P + 1e-9 * np.diag(np.diag(H)))
            for nm, S_, i0, n_ in (("a", S_a, ia, 3 * Ka), ("g", S_g, ig, 3 * Kg)):
                c = vec(x)[i0:i0 + n_]
                rankS = np.linalg.matrix_rank(S_)
                gam = rankS - alpha[nm] * np.trace(Hi[i0:i0 + n_, i0:i0 + n_] @ S_)
                new = kappa * max(gam, 0.5) / max(c @ S_ @ c, 1e-12)
                alpha[nm] = float(np.clip(new, alpha[nm] / 8.0, alpha[nm] * 8.0))
            alpha["a"] = float(np.clip(alpha["a"], 1 / 2.0 ** 2, 1 / 0.01 ** 2))
            alpha["g"] = float(np.clip(alpha["g"], 1 / 0.3 ** 2, 1 / 0.001 ** 2))
            log_alpha.append((1 / np.sqrt(alpha["a"]), 1 / np.sqrt(alpha["g"])))
            best = obj(v, x)
            ok = True
        if not ok:
            break
    ec.x = x
    q, vout, tb, last, info = ec.finish(x, v, float(best))
    info.update(A=A, H=H, y=y, P=P, m=m, x=x, sig_a_final=1 / np.sqrt(alpha["a"]), sig_g_final=1 / np.sqrt(alpha["g"]), log_alpha=log_alpha)
    return q, vout, tb, last, info


def run_solver(P, b0, knot, sig_a, sig_g, **kw):
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=b0, knot_a=knot, knot_g=knot)
    ec.set_target()
    if ec.drone:
        kw2 = dict(kw)
        sig_p = 5.0
        for outer in range(4):
            v, Rg, Rm, fw, Sm = ec.forward(ec.x)
            shift = np.einsum("nij,nj->ni", Rm, ec.vb_t) * ec.on_b[:, None]
            if outer > 0:
                pos = np.cumsum(v * ec.dtb[:, None], axis=0)
                sig_p = float(np.clip(1.5 * np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()), 1.0, 20.0))
            q, vs, tb, last, info = core(ec, shift, 8, sig_a, sig_g, 0.5, sig_p, **kw2)
            if kw2.get("eb_passes"):
                sig_a, sig_g = info["sig_a_final"], info["sig_g_final"]
    else:
        shift = np.zeros((len(ec.idx), 3))
        q, vs, tb, last, info = core(ec, shift, 12, sig_a, sig_g, 3.0, 5.0, **kw)
    info["drone"] = bool(ec.drone)
    return q, vs, tb, last, info, ec, shift


def crit(info, v_res_rows):
    """Selection numbers from the final linearisation."""
    A, H, y, P, m = info["A"], info["H"], info["y"], info["P"], info["m"]
    npar = A.shape[1]
    nrows = A.shape[0]
    Jd = float(y @ y)
    pd = np.diag(P)
    keep = pd > 0
    sign, ld_h = np.linalg.slogdet(H + 1e-9 * np.diag(np.diag(H)))
    w, _ = np.linalg.eigh(P)
    ld_p = float(np.log(w[w > 1e-12]).sum())
    k_eff = int(keep.sum())
    # row correlation time from the velocity rows
    vr = y[: 3 * m].reshape(m, 3) if y[: 3 * m].size == 3 * m else None
    kap = 1.0
    if vr is not None:
        z = vr - vr.mean(0)
        ac = [1.0]
        den = (z ** 2).sum()
        for lag in range(1, min(60, m // 2)):
            c = (z[lag:] * z[:-lag]).sum() / max(den, 1e-12)
            if c < 0.05:
                break
            ac.append(c)
        kap = 1.0 + 2.0 * sum(ac[1:])
    x_ = info["x"]
    c = np.concatenate([x_["v0"], np.zeros(2 * len(x_["S"])), x_["ca"].ravel(), x_["cg"].ravel(), x_["dv"].ravel()])
    Jp = float(c @ P @ c)
    out = dict(J=Jd, Jp=Jp, npar=npar, nrows=nrows, kap=kap, ld_h=float(ld_h), ld_p=ld_p, k_eff=k_eff)
    out["bic_white"] = Jd + Jp + npar * np.log(nrows)
    neff = nrows / kap
    out["bic_eff"] = neff * np.log(max(Jd, 1e-9) / neff) + npar * np.log(neff)
    out["lap_white"] = 0.5 * (Jd + Jp) + 0.5 * float(ld_h) - 0.5 * ld_p
    # tempered Laplace: rows count kappa times too many
    H_t = A.T @ A / kap + P
    s2, ld_t = np.linalg.slogdet(H_t + 1e-9 * np.diag(np.diag(H_t)))
    out["lap_temp"] = 0.5 * (Jd / kap + Jp) + 0.5 * float(ld_t) - 0.5 * ld_p
    # block cross-validation at the final linearisation: 20 s blocks every 4th block, 5 s guard, velocity rows
    nb = 4
    blk = 20
    guard = 5
    fold = (np.arange(m) // blk) % nb
    err_v, err_p, cnt = 0.0, 0.0, 0
    c0 = c
    for f in range(nb):
        te = fold == f
        near = np.convolve(te.astype(float), np.ones(2 * guard + 1), mode="same") > 0
        tr = ~near
        idx_tr = np.concatenate([3 * np.nonzero(tr)[0] + i for i in range(3)] + [3 * m + 3 * np.nonzero(tr)[0] + i for i in range(3)])
        At, yt = A[idx_tr], y[idx_tr]
        Ht = At.T @ At + P
        Ht = Ht + 1e-8 * np.diag(np.diag(Ht) + 1e-9)
        dx = np.linalg.solve(Ht, At.T @ yt - P @ c0)
        idv = np.concatenate([3 * np.nonzero(te)[0] + i for i in range(3)])
        idp = np.concatenate([3 * m + 3 * np.nonzero(te)[0] + i for i in range(3)])
        err_v += ((y[idv] - A[idv] @ dx) ** 2).sum()
        err_p += ((y[idp] - A[idp] @ dx) ** 2).sum()
        cnt += len(idv)
    out["cv_v"] = float(err_v / max(cnt, 1))
    out["cv_p"] = float(err_p / max(cnt, 1))
    out["cv"] = out["cv_v"] + out["cv_p"]
    return out


VARIANTS = {
    "L60": dict(level="L60"),
    "L15": dict(level="L15"),
    "L8": dict(level="L8"),
    "L15_t": dict(level="L15", robust="t"),
    "L8_t": dict(level="L8", robust="t"),
    "L15_gm": dict(level="L15", robust="gm"),
    "L8rw_eb1": dict(level="L8", pen="rw", eb_passes=1, kappa=1.0),
    "L8rw_eb5": dict(level="L8", pen="rw", eb_passes=1, kappa=5.0),
    "L15rw_eb5": dict(level="L15", pen="rw", eb_passes=1, kappa=5.0),
    "L15_ms": dict(level="L15", bands="table"),
}


def motion_table(n):
    """Octave velocity band variance from GT on other environments (handheld only)."""
    rooms = [r for r in DEV_ROOMS if r != n]
    out = {}
    for T_s in (2.5, 5.0, 10.0, 20.0, 40.0):
        acc = []
        for r in rooms:
            room = load_room(r)
            t = (room["gt_t_ns"] - room["gt_t_ns"][0]) * 1e-9
            tk = np.arange(t[0] + 0.1, t[-1] - 0.1, 0.05)
            pi = lambda x: np.stack([np.interp(x, t, room["gt_p"][:, i]) for i in range(3)], 1)
            v = (pi(tk + 0.05) - pi(tk - 0.05)) / 0.1
            n1 = int(T_s / 0.05)
            a = uniform_filter1d(v, n1, axis=0, mode="nearest") - uniform_filter1d(v, 2 * n1, axis=0, mode="nearest")
            acc.append((a[n1:-n1] ** 2).mean())
        out[T_s] = float(np.mean(acc))
    return out


def fname(n, k, var):
    return f"{OUT}/{n}_{k[0]}_{k[1]}_{k[2]}_{var}.pkl"


def work(job):
    n, k, var = job
    f = fname(n, k, var)
    if os.path.exists(f):
        return job, 0.0
    t0 = time.time()
    np.random.seed(0)
    room = load_room(n)
    t, imu, room = file_for(n, k)
    P = prepare(t, imu)
    cfg = dict(VARIANTS[var])
    lv = cfg.pop("level")
    knot, sa, sg = LV[lv]
    bands = None
    if cfg.get("bands") == "table":
        if isinstance(n, str):
            # handheld study only: drones keep the plain rows
            cfg.pop("bands")
        else:
            tb = motion_table(n)
            bands = [(T_s, 1.0 / np.sqrt(tb[T_s] * 3.0 / 2.0)) for T_s in sorted(tb)]
            cfg["bands"] = bands
    b0 = m41.calib_excl(n)
    q, v, tb_, last, info, ec, shift = run_solver(P, b0, knot, sa, sg, **cfg)
    d = to_cache(q, v, tb_, info["drone"])
    d["crit"] = crit(info, None)
    d["sig_a_final"], d["sig_g_final"] = info["sig_a_final"], info["sig_g_final"]
    if var == "L15":
        # Laplace samples for the posterior-mean test: scale 1 (rows white) and scale kappa (rows correlated)
        rng = np.random.default_rng(7)
        H = info["H"]
        Hj = H + 1e-9 * np.diag(np.diag(H))
        L = np.linalg.cholesky(Hj)
        M = 12
        z = rng.standard_normal((H.shape[0], M))
        dxs = solve_triangular(L.T, z, lower=False)
        x0 = info["x"]
        tk = d["tk"]
        for scale_nm, sc_ in (("s1", 1.0), ("kap", np.sqrt(max(d["crit"]["kap"], 1.0)))):
            ps = []
            for i in range(M):
                # antithetic pair keeps the mean centred on the MAP in the linear case
                sgn = 1.0 if i % 2 == 0 else -1.0
                zz = dxs[:, i // 2] * sgn
                xs = ec._apply(x0, zz * sc_)
                vs_ = ec.forward(xs)[0]
                vk = np.stack([np.interp(tk, tb_, vs_[:, a]) for a in range(3)], 1)
                ps.append(np.cumsum(vk, axis=0) * DT)
            ps = np.array(ps)
            # plain mean in the file's frame
            d[f"vm_{scale_nm}"] = np.gradient(ps.mean(0), DT, axis=0).astype(np.float32)
            # Procrustes mean (rotation + translation, no scale): what -a alignment rewards
            mu = np.cumsum(d["v"], axis=0) * DT
            for _ in range(3):
                al = []
                for p in ps:
                    r, tr = umeyama(p, mu)
                    al.append((r @ p.T).T + tr)
                mu = np.mean(al, 0)
            d[f"vp_{scale_nm}"] = np.gradient(mu, DT, axis=0).astype(np.float32)
            d[f"spread_{scale_nm}"] = float(np.sqrt(((np.array(al) - mu) ** 2).sum(2).mean()))
    with open(f, "wb") as fh:
        pickle.dump(d, fh)
    return job, time.time() - t0


def loader_for(var):
    def fn(n, k):
        return pickle.load(open(fname(n, k, var), "rb"))
    return fn


def loader_alt(field):
    def fn(n, k):
        d = pickle.load(open(fname(n, k, "L15"), "rb"))
        d = dict(d)
        d["v"] = d[field].astype(np.float64)
        return d
    return fn


if __name__ == "__main__":
    stage = sys.argv[2]
    names = NAMES_ALL
    if stage == "time":
        for var in sys.argv[3:]:
            for n in ("MH03", 1):
                print(var, n, work((n, ("full", "zero", 1), var)), flush=True)
    elif stage == "solve":
        vars_ = sys.argv[3:] or list(VARIANTS)
        jobs = [(n, k, var) for var in vars_ for n in names for k in KEYS]
        t0 = time.time()
        with mp.get_context("spawn").Pool(12) as p:
            for i, (job, dt) in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
                if i % 50 == 0:
                    print(i, len(jobs), job, f"{dt:.1f}s", f"elapsed {time.time()-t0:.0f}", flush=True)
        print("solved", len(jobs), time.time() - t0, flush=True)
    elif stage == "score":
        res = {}
        for var in sys.argv[3:] or list(VARIANTS):
            r = {}
            r["eu"] = evaluate(SEQS, EU_FOLDS, KEYS, loader_for(var), procs=8)
            r["tu"] = evaluate(DEV_ROOMS, TU_FOLDS, KEYS, loader_for(var), procs=8)
            res[var] = r
            pickle.dump(r, open(f"{OUT}/score_{var}.pkl", "wb"))
            for pl in ("eu", "tu"):
                print(var, pl, " ".join(f"{kk}:{np.nanmean([r[pl][k]['final']['ate_se3'] for k in r[pl] if k[1]==kk]):.2f}/{np.nanmean([r[pl][k]['nocheck']['ate_se3'] for k in r[pl] if k[1]==kk]):.2f}" for kk in ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine")), flush=True)
    elif stage == "scorepm":
        for fld in ("vm_s1", "vp_s1", "vm_kap", "vp_kap"):
            r = {}
            r["eu"] = evaluate(SEQS, EU_FOLDS, KEYS, loader_alt(fld), procs=8)
            r["tu"] = evaluate(DEV_ROOMS, TU_FOLDS, KEYS, loader_alt(fld), procs=8)
            pickle.dump(r, open(f"{OUT}/score_{fld}.pkl", "wb"))
            for pl in ("eu", "tu"):
                print(fld, pl, " ".join(f"{kk}:{np.nanmean([r[pl][k]['final']['ate_se3'] for k in r[pl] if k[1]==kk]):.3f}/{np.nanmean([r[pl][k]['nocheck']['ate_se3'] for k in r[pl] if k[1]==kk]):.3f}" for kk in ("clean", "mild", "full", "burst", "steps", "randomwalk", "sine")), flush=True)
