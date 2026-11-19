"""Second-generation solver: same physics, but the cost is the correct statistical one.

Old cost: |5 s average velocity|^2 / s_v^2 + |position|^2 / s_p^2, with knot spacing chosen by hand and a threshold as a safety net.
New cost: real motion is treated as a Gaussian process whose power spectrum is measured on other environments, so velocity rows are
whitened by that spectrum (DCT rows, each with unit variance when the motion is typical). The file's own scale of motion is read from
bands drift cannot reach. Drift is a random walk on a fine grid, and its size is chosen per file by Laplace evidence over a small grid
of drift rates. Since the rows are approximately white, the evidence is meaningful. No thresholds, no knot choice.

Nothing here reads ground truth at solve time. Spectra come from training environments only.
"""
import importlib, os, pickle, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import numpy as np
from scipy.fft import dct
from energy2 import cross_mat
m46 = importlib.import_module("46_drag_energy")
from phys import *

EDG = np.array([0.0] + [2.0 ** k / 512 for k in range(0, 10)])   # top edge 1 Hz at row_dt 0.5 s
NB = len(EDG) - 1
RDT = 0.5
SIG_A = (0.03, 0.1, 0.3, 1.0)     # accel drift: rms change over 60 s, m/s^2
SIG_G = (0.003, 0.01, 0.03, 0.1)  # gyro drift: rms change over 60 s, rad/s


def band_of(f):
    return np.clip(np.digitize(f, EDG) - 1, 0, NB - 1)


def centres():
    c = np.sqrt(np.maximum(EDG[:-1], 1 / 1024) * EDG[1:])
    return c


def dct_rows(x):
    return dct(x, type=2, norm="ortho", axis=0)


def bands_power(Y, f):
    """Mean squared DCT value per band; Y is (N, ...)."""
    b = band_of(f)
    out = np.full(NB, np.nan)
    for i in range(NB):
        if (b == i).any():
            out[i] = (Y[b == i] ** 2).mean()
    return out


def gt_vel_rows(room, t0_ns, N=None):
    gt_t = (room["gt_t_ns"] - t0_ns) * 1e-9
    te = np.arange(max(gt_t[0], 0.0) + RDT / 2, gt_t[-1] - RDT / 2, RDT)
    f = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    return te, (f(te + RDT / 2) - f(te - RDT / 2)) / RDT


def spec_one(n):
    room = load_room(n)
    te, v = gt_vel_rows(room, room["t_ns"][0])
    Y = dct_rows(v)
    fr = np.arange(len(v)) / (2 * len(v) * RDT)
    return dict(xy=bands_power(Y[:, :2], fr), z=bands_power(Y[:, 2], fr))


def drag_err_one(n):
    """Spectrum of (drag velocity in the true attitude - true velocity), clean rows, horizontal axes. Training stage only."""
    room = load_room(n)
    t, imu = room["t_ns"], room["imu"]
    P = prepare(t, imu)
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=np.zeros(3))
    ec.set_target()
    gt_t = (room["gt_t_ns"] - t[0]) * 1e-9
    from scipy.spatial.transform import Slerp
    Rg = Slerp(gt_t, R.from_quat(room["gt_q"]))(np.clip(ec.tb, gt_t[0], gt_t[-1])).as_matrix()
    tgt = np.einsum("nij,nj->ni", Rg, ec.vb_t) * ec.on_b[:, None]
    bl = int(round(RDT / np.median(ec.dtb)))
    N = len(tgt) // bl
    tr = tgt[: N * bl].reshape(N, bl, 3).mean(1)
    te = ec.tb[: N * bl].reshape(N, bl).mean(1)
    f = lambda x: np.stack([np.interp(x, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    vg = (f(te + RDT / 2) - f(te - RDT / 2)) / RDT
    Y = dct_rows(tr - vg)
    fr = np.arange(N) / (2 * N * RDT)
    return dict(xy=bands_power(Y[:, :2], fr))


def spec_tables(names, cache):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = {}
    for n in names:
        o = spec_one(n)
        if isinstance(n, str):
            o["e"] = drag_err_one(n)["xy"]
        out[n] = o
    pickle.dump(out, open(cache, "wb"))
    return out


def fold_spec(tables, train):
    """Geometric mean of the band powers of the training files."""
    S = dict(xy=gm_key(tables, train, "xy"), z=gm_key(tables, train, "z"))
    if isinstance(train[0], str):
        S["e"] = np.exp(np.nanmean([np.log(np.maximum(tables[n]["e"], 1e-9)) for n in train], axis=0))
    return S


def gm_key(tables, train, key):
    return np.exp(np.nanmean([np.log(np.maximum(tables[n][key], 1e-9)) for n in train], axis=0))


def spec_at(P, fr):
    """Band table to a value per DCT row. DC keeps its own band."""
    c = centres()
    p = np.array(P, float)
    good = ~np.isnan(p)
    lf = np.log(np.maximum(fr, 1e-4))
    s = np.exp(np.interp(lf, np.log(c[good])[1:], np.log(p[good])[1:]))
    s[0] = p[0] if good[0] else s[1]
    return s


class Solver:
    def __init__(self, ec, spec, knot=10.0, couple=True):
        self.ec, self.spec, self.couple = ec, spec, couple
        self.bl = int(round(RDT / np.median(ec.dtb)))
        self.N = len(ec.dtb) // self.bl
        self.fr = np.arange(self.N) / (2 * self.N * RDT)
        self.Wxy0 = 1.0 / np.sqrt(spec_at(spec["xy"], self.fr))
        self.Wz0 = 1.0 / np.sqrt(spec_at(spec["z"], self.fr))
        self.drone = bool(ec.drone)
        if self.drone:
            self.We = 1.0 / np.sqrt(spec_at(spec["e"], self.fr))
        self.ax, self.az = 1.0, 1.0     # file scale of motion
        self.dk = (ec.t_end - ec.t[0]) / max(1, ec.Ka - 1)
        self.sig_a, self.sig_g = 0.3, 0.03
        self.k = m46.K_DRAG
        self.Rfix = None
        self.nrows = 3 * self.N * (2 if self.drone else 1)

    # ------------------------------------------------------------- row operators
    def down(self, M):
        sh = M.shape
        return M[: self.N * self.bl].reshape((self.N, self.bl) + sh[1:]).mean(1)

    def wrows(self, M, kind="v"):
        """M: (n, 3, ...) at block rate. Whitened DCT rows."""
        Y = dct_rows(self.down(M))
        if kind == "v":
            Y[:, :2] *= (self.Wxy0 / self.ax).reshape((-1, 1) + (1,) * (Y.ndim - 2))
            Y[:, 2] *= (self.Wz0 / self.az).reshape((-1,) + (1,) * (Y.ndim - 2))
        else:
            Y[:, :2] *= self.We.reshape((-1, 1) + (1,) * (Y.ndim - 2))
            Y[:, 2] *= 0.0
        return Y

    def prior_mats(self, sa, sg):
        ec = self.ec

        def rw(K, s60, s0):
            D = np.diff(np.eye(K), axis=0)
            q = s60 ** 2 / 60.0 * self.dk
            L = D.T @ D / q
            L[0, 0] += 1.0 / s0 ** 2
            return L

        return rw(ec.Ka, sa, 0.5), rw(ec.Kg, sg, 0.05)

    def full_prior(self, sa, sg):
        ec = self.ec
        Ka, Kg, G, K = ec.Ka, ec.Kg, ec.G, ec.K
        n = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
        Lam = np.zeros((n, n))
        Lam[np.arange(3), np.arange(3)] = 1 / 25.0
        La, Lg = self.prior_mats(sa, sg)
        p = 3 + 2 * K
        for i in range(3):
            Lam[p + i * Ka : p + (i + 1) * Ka, p + i * Ka : p + (i + 1) * Ka] = La
        # coefficient order: ca.ravel() is knot-major, so reorder below
        idx_a = np.arange(3 * Ka).reshape(Ka, 3)
        block = np.zeros((3 * Ka, 3 * Ka))
        for i in range(3):
            block[np.ix_(idx_a[:, i], idx_a[:, i])] = La
        Lam[p : p + 3 * Ka, p : p + 3 * Ka] = block
        p += 3 * Ka
        idx_g = np.arange(3 * Kg).reshape(Kg, 3)
        block = np.zeros((3 * Kg, 3 * Kg))
        for i in range(3):
            block[np.ix_(idx_g[:, i], idx_g[:, i])] = Lg
        Lam[p : p + 3 * Kg, p : p + 3 * Kg] = block
        p += 3 * Kg
        Lam[np.arange(p, p + 3 * G), np.arange(p, p + 3 * G)] = 1.0
        return Lam, np.linalg.slogdet(La)[1] * 3 + np.linalg.slogdet(Lg)[1] * 3

    # ------------------------------------------------------------- cost
    def shift(self, x):
        ec = self.ec
        base = np.einsum("nij,nj->ni", self.Rfix, ec.vb_t)
        if self.couple:
            base = base + np.einsum("nij,nj->ni", self.Rfix @ (ec.E @ ec.E.T), ec.Ba @ x["ca"]) / self.k
        return base * ec.on_b[:, None]

    def res(self, x, v):
        r = self.wrows(v, "v").ravel()
        if self.drone:
            r = np.concatenate([r, self.wrows(v - self.shift(x), "e").ravel()])
        return r

    def vec(self, x):
        ec = self.ec
        return np.concatenate([x["v0"], np.zeros(2 * ec.K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()])

    def jac(self, x, v, Rg, Rm, fw, Sm):
        ec = self.ec
        Ka, Kg, G, K = ec.Ka, ec.Kg, ec.G, ec.K
        np_total = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
        rows = 3 * self.N
        A = np.zeros((self.nrows, np_total))
        dtb = ec.dtb
        n = len(dtb)
        Fx = cross_mat(fw)
        Id = np.tile(np.eye(3), (n, 1, 1))
        if self.drone:
            RP = self.Rfix @ (ec.E @ ec.E.T)
            on = ec.on_b.astype(float)

        def put(col, M, Ms=None):
            p = M.shape[2]
            A[:rows, col : col + p] = self.wrows(M, "v").reshape(rows, p)
            if self.drone:
                A[rows:, col : col + p] = self.wrows(M if Ms is None else M - Ms, "e").reshape(rows, p)

        put(0, Id)
        col = 3
        for k in range(K):
            mk = (ec.seg_of == k).astype(float)
            put(col, np.cumsum(-Fx * (dtb * mk)[:, None, None], axis=0)[:, :, :2])
            col += 2
        wdt = dtb * ec.okb
        for j in range(Ka):
            M = np.cumsum(-Rm * (ec.Ba[:, j] * wdt)[:, None, None], axis=0)
            put(col, M, RP * (ec.Ba[:, j] * on)[:, None, None] / self.k if (self.drone and self.couple) else None)
            col += 3
        Sb = np.zeros_like(Rm)
        Sb[:] = np.eye(3)
        for k in range(K):
            Sb[ec.seg_of == k] = Sm[k]
        for j in range(Kg):
            Phi = Sb @ np.cumsum(-Rg * (ec.Bg[:, j] * dtb)[:, None, None], axis=0)
            put(col, np.cumsum(-np.einsum("nij,njk->nik", Fx, Phi) * dtb[:, None, None], axis=0))
            col += 3
        for j, (s, e) in enumerate(ec.gaps):
            put(col, Id * ec._ramp(s, e)[:, None, None])
            col += 3
        return A

    def cost(self, x, v, Lam):
        r = self.res(x, v)
        c = self.vec(x)
        return float(r @ r + c @ Lam @ c), r

    def lm(self, x, sa, sg, iters=5):
        ec = self.ec
        Lam, _ = self.full_prior(sa, sg)
        v, Rg, Rm, fw, Sm = ec.forward(x)
        best, r = self.cost(x, v, Lam)
        lam = 1e-3
        for it in range(iters):
            A = self.jac(x, v, Rg, Rm, fw, Sm)
            H = A.T @ A + Lam
            c = self.vec(x)
            g = A.T @ r + Lam @ c
            D = np.diag(np.maximum(np.diag(H), 1e-9))
            ok = False
            for _ in range(6):
                dx = np.linalg.solve(H + lam * D, -g)
                xn = ec._apply(x, dx)
                rr = ec.forward(xn)
                o, rn = self.cost(xn, rr[0], Lam)
                if o < best:
                    x, best, r = xn, o, rn
                    v, Rg, Rm, fw, Sm = rr
                    lam = max(lam / 3.0, 1e-6)
                    ok = True
                    break
                lam *= 8.0
            if not ok:
                break
        return x, v, (Rg, Rm, fw, Sm), best, r

    def update_scale(self, v, fmin=1 / 24.0):
        """The file's own motion scale, read from bands drift cannot reach."""
        Y = dct_rows(self.down(v))
        m = self.fr > fmin
        self.ax = float(np.clip(np.sqrt(np.mean((Y[m][:, :2] * self.Wxy0[m, None]) ** 2)), 0.25, 4.0))
        self.az = float(np.clip(np.sqrt(np.mean((Y[m][:, 2] * self.Wz0[m]) ** 2)), 0.25, 4.0))

    def evidence_grid(self, x, v, rest):
        Rg, Rm, fw, Sm = rest
        A = self.jac(x, v, Rg, Rm, fw, Sm)
        r = self.res(x, v)
        AtA, Atr = A.T @ A, A.T @ r
        c = self.vec(x)
        rr = r @ r
        tab = {}
        for sa in SIG_A:
            for sg in SIG_G:
                Lam, ld = self.full_prior(sa, sg)
                H = AtA + Lam
                g = Atr + Lam @ c
                dx = np.linalg.solve(H, -g)
                cst = rr + c @ Lam @ c + g @ dx
                tab[(sa, sg)] = float(-0.5 * cst - 0.5 * np.linalg.slogdet(H)[1] + 0.5 * ld)
        return tab

    def solve(self, passes=3, select=True, sa=0.3, sg=0.03):
        ec = self.ec
        x = {k: np.array(v, float) for k, v in ec.x.items()}
        hist = []
        for p in range(passes):
            v, Rg, Rm, fw, Sm = ec.forward(x)
            self.Rfix = Rm
            x, v, rest, best, r = self.lm(x, sa, sg, iters=6 if p == 0 else 4)
            self.update_scale(v)
            self.Rfix = rest[1]
            if select:
                tab = self.evidence_grid(x, v, rest)
                sa, sg = max(tab, key=tab.get)
                hist.append((sa, sg))
        x, v, rest, best, r = self.lm(x, sa, sg, iters=4)
        ec.x = x
        q, vout, tb, last, info = ec.finish(x, v, float(best))
        info.update(sa=sa, sg=sg, hist=hist, ax=self.ax, az=self.az, chi2=float(r @ r / len(r)))
        if select:
            self.Rfix = rest[1]
            info["tab"] = self.evidence_grid(x, v, rest)
        info["drone"] = self.drone
        return q, vout, tb, last, info


def solve_file(P, b0, spec, knot=10.0, couple=True, select=True, sa=0.3, sg=0.03):
    ec = m46.DragCal(P["t"], P["imu"], P["ok"], b0=b0, knot_a=knot, knot_g=knot)
    ec.set_target()
    s = Solver(ec, spec, knot=knot, couple=couple)
    return s.solve(select=select, sa=sa, sg=sg)
