"""New idea 5: energy-minimising self-calibration.

Double integration explodes because of a few slow sensor errors: accel bias, tilt, gyro bias. A velocity that runs away
to hundreds of m/s is an extremely sensitive measurement of those errors. So instead of high-passing the drift away (which
also deletes real slow motion), solve for the errors: pick the bias curves, start tilt and per-gap jumps that make the
slow part of the velocity over the whole file as small as physically reasonable. Linear after linearising, so a small
least-squares solve, guarded by a line search. Needs only "velocity stays bounded", with no map or room size.
Nothing here uses ground truth.
"""
import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R

from attitude import coast_rates, find_gaps, integrate_rows, offline_attitude, spline_basis
from common import _qexp, _qmul, quiet_init


@njit(cache=True)
def _integrate3(t, gyro, ok, bias, coast, q0, step3):
    n = len(t)
    out = np.zeros((n, 4))
    q = q0.copy()
    out[0] = q
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        w = gyro[k] - bias[k] if ok[k] else coast[k]
        q = _qmul(q, _qexp(w * dt))
        if step3[k, 0] != 0.0 or step3[k, 1] != 0.0 or step3[k, 2] != 0.0:
            q = _qmul(_qexp(step3[k]), q)
        q /= np.sqrt(np.sum(q * q))
        out[k] = q
    return out


def _cross_mat(f):
    """[f x] for an array (T,3): shape (T,3,3)."""
    m = np.zeros((len(f), 3, 3))
    m[:, 0, 1], m[:, 0, 2] = -f[:, 2], f[:, 1]
    m[:, 1, 0], m[:, 1, 2] = f[:, 2], -f[:, 0]
    m[:, 2, 0], m[:, 2, 1] = -f[:, 1], f[:, 0]
    return m


class EnergyCal:
    def __init__(self, t, imu, ok, b0=None, knot_a=60.0, knot_g=60.0, hop=4, seed_fit=True, seed_w=5.0):
        self.t, self.imu, self.ok = t, imu, ok
        self.gyro, self.acc = imu[:, :3], imu[:, 3:]
        self.n = len(t)
        self.b0 = np.zeros(3) if b0 is None else np.asarray(b0, float)
        self.last = int(np.nonzero(ok)[0][-1])
        self.t_end = t[self.last]
        self.g = float(np.median(np.linalg.norm(self.acc[ok], axis=1)))
        self.hop = hop
        im2 = imu.copy()
        im2[ok, :3] -= self.b0
        self.q0, _ = quiet_init(t, im2, ok)
        self.Ka = max(2, int(round((self.t_end - t[0]) / knot_a)) + 1)
        self.Kg = max(2, int(round((self.t_end - t[0]) / knot_g)) + 1)
        ka = np.linspace(t[0], self.t_end + 1e-3, self.Ka)
        kg = np.linspace(t[0], self.t_end + 1e-3, self.Kg)
        tc = np.minimum(t, self.t_end)
        self.Ba_r, self.Bg_r = spline_basis(tc, ka), spline_basis(tc, kg)
        self.gaps_all = find_gaps(ok)
        self.gaps = [(s, e) for s, e in self.gaps_all if e < self.n and s > 0]
        self.G = len(self.gaps)
        self.idx = np.arange(0, self.last + 1 - hop + 1, hop)
        i = self.idx
        self.tb = t[i + hop // 2]
        self.okb = ok[i + hop // 2].astype(float)
        self.fb = np.add.reduceat(self.acc[: i[-1] + hop] * ok[: i[-1] + hop, None], i, axis=0) / np.maximum(
            np.add.reduceat(ok[: i[-1] + hop].astype(float), i), 1)[:, None]
        self.dtb = np.diff(t[np.append(i, min(i[-1] + hop, self.last))])
        self.Ba, self.Bg = self.Ba_r[i + hop // 2], self.Bg_r[i + hop // 2]
        self.x = dict(v0=np.zeros(3), q0=np.array(self.q0, float), ca=np.zeros((self.Ka, 3)), cg=np.zeros((self.Kg, 3)),
                      theta=np.zeros((self.G, 3)), dv=np.zeros((self.G, 3)))
        if seed_fit:
            # gravity fit gets gyro bias and tilt inside the range where linearising is valid
            _, info = offline_attitude(t, imu, ok, b0=self.b0, knot_s=knot_g, s_r=0.3, sigma_b=0.003, residual_s=0.0, return_info=True)
            B = spline_basis(np.linspace(t[0], self.t_end, self.Kg), info["knots"])
            self.x["cg"] = B @ info["bias"]
            self._seed_gaps(seed_w)

    def _seed_gaps(self, W):
        """Tilt after each gap, read straight from gravity. Walks forward: each gap fixes the up direction of the next
        segment using the mean accel over its first W seconds, carried through the gyro. Large rotations are fine because
        this step is exact, not linearised, so the solver starts inside its valid range."""
        fs = 1.0 / np.median(np.diff(self.t))
        wn = int(W * fs)
        for j, (s, e) in enumerate(self.gaps):
            next_gap = self.gaps[j + 1][0] if j + 1 < self.G else self.last + 1
            stop = min(e + wn, next_gap)
            if stop - e < 20:
                continue
            bias = self.Bg_r @ self.x["cg"] + self.b0
            coast = coast_rates(self.t, self.gyro, self.ok, bias, self.gaps_all, "lerp")
            ts = np.zeros((self.n, 3))
            for k, (a, b) in enumerate(self.gaps):
                ts[a:b] = self.x["theta"][k] / max(1, b - a)
            q = _integrate3(self.t, self.gyro, self.ok, bias, coast, self.x["q0"], ts)
            rr = np.arange(e, stop)
            rr = rr[self.ok[rr]]
            Rm = R.from_quat(q[rr][:, [1, 2, 3, 0]]).as_matrix()
            u = np.einsum("nij,nj->ni", Rm, self.acc[rr]).mean(0)
            u = u / np.linalg.norm(u)
            ax = np.cross(u, [0.0, 0.0, 1.0])
            sn = np.linalg.norm(ax)
            ang = np.arctan2(sn, u[2])
            if sn < 1e-9:
                continue
            new = R.from_rotvec(ax / sn * ang) * R.from_rotvec(self.x["theta"][j])
            self.x["theta"][j] = new.as_rotvec()

    def forward(self, x):
        bias = self.Bg_r @ x["cg"] + self.b0
        coast = coast_rates(self.t, self.gyro, self.ok, bias, self.gaps_all, "lerp")
        ts = np.zeros((self.n, 3))
        for j, (s, e) in enumerate(self.gaps):
            ts[s:e] = x["theta"][j] / max(1, e - s)
        q = _integrate3(self.t, self.gyro, self.ok, bias, coast, x["q0"], ts)
        i = self.idx
        Rm = R.from_quat(q[i + self.hop // 2][:, [1, 2, 3, 0]]).as_matrix()
        fw = np.einsum("nij,nj->ni", Rm, self.fb) * self.okb[:, None]
        aw = (fw - np.einsum("nij,nj->ni", Rm, self.Ba @ x["ca"]) * self.okb[:, None]) - np.array([0, 0, self.g]) * self.okb[:, None]
        v = np.cumsum(aw * self.dtb[:, None], axis=0) + x["v0"]
        for j, (s, e) in enumerate(self.gaps):
            v += x["dv"][j][None] * self._ramp(s, e)[:, None]
        return q, v, Rm, fw

    def _ramp(self, s, e):
        return np.clip((self.idx - s) / max(1, e - s), 0.0, 1.0)

    def _position(self, v):
        return np.cumsum(v * self.dtb[:, None], axis=0)

    def objective(self, v, x, pr, smooth, sig_v, sig_p):
        vs = uniform_filter1d(v, smooth, axis=0, mode="nearest")
        p = self._position(v)
        p = p - p.mean(0)
        return (vs ** 2).sum() / sig_v ** 2 * self.stride_w + (p ** 2).sum() / sig_p ** 2 * self.stride_w + (x["ca"] ** 2).sum() * pr["a"] + (
            x["cg"] ** 2).sum() * pr["g"] + (x["dv"] ** 2).sum() * pr["dv"] + (x["theta"][:, :2] ** 2).sum() * pr["th"] + (x["theta"][:, 2] ** 2).sum() * pr["psi"]

    def solve(self, iters=12, sig_a=0.3, sig_g=0.01, sig_dv=1.0, sig_th=2.0, sig_psi=0.01, sig_phi=0.5, sig_v=3.0, sig_p=5.0, smooth_s=5.0, row_s=1.0):
        fs_b = 1.0 / np.median(self.dtb)
        smooth = max(1, int(smooth_s * fs_b))
        stride = max(1, int(row_s * fs_b))
        self.stride_w = 1.0 / stride
        pr = dict(a=1 / sig_a ** 2, g=1 / sig_g ** 2, dv=1 / sig_dv ** 2, th=1 / sig_th ** 2, psi=1 / sig_psi ** 2)
        x = {k: np.array(v, float) for k, v in self.x.items()}
        Ka, Kg, G = self.Ka, self.Kg, self.G
        np_total = 3 + 2 + 3 * Ka + 3 * Kg + 3 * G + 3 * G
        rows = np.arange(smooth // 2, len(self.idx), stride)
        m = len(rows)
        q, v, Rm, fw = self.forward(x)
        best = self.objective(v, x, pr, smooth, sig_v, sig_p)
        lam = 1e-3
        for it in range(iters):
            Fx = _cross_mat(fw)
            dtb, Ba, Bg = self.dtb, self.Ba, self.Bg
            A = np.zeros((6 * m, np_total))
            sv, sp = 1.0 / sig_v, 1.0 / sig_p
            def put(col, M):
                sm = uniform_filter1d(M, smooth, axis=0, mode="nearest")
                A[: 3 * m, col : col + M.shape[2]] = sv * sm[rows].reshape(3 * m, M.shape[2])
                P = np.cumsum(M * dtb[:, None, None], axis=0)
                P = P - P.mean(0, keepdims=True)
                A[3 * m :, col : col + M.shape[2]] = sp * P[rows].reshape(3 * m, M.shape[2])
            Id = np.tile(np.eye(3), (len(dtb), 1, 1))
            put(0, Id)
            col = 3
            N = np.cumsum(-Fx * dtb[:, None, None], axis=0)
            put(col, N[:, :, :2])
            col += 2
            wdt = dtb * self.okb
            for j in range(Ka):
                put(col, np.cumsum(-Rm * (Ba[:, j] * wdt)[:, None, None], axis=0))
                col += 3
            for j in range(Kg):
                Phi = np.cumsum(-Rm * (Bg[:, j] * dtb)[:, None, None], axis=0)
                put(col, np.cumsum(-np.einsum("nij,njk->nik", Fx, Phi) * dtb[:, None, None], axis=0))
                col += 3
            for j, (s, e) in enumerate(self.gaps):
                put(col, Id * self._ramp(s, e)[:, None, None])
                col += 3
            for j, (s, e) in enumerate(self.gaps):
                mask = (self.idx >= e).astype(float)
                put(col, np.cumsum(-Fx * (dtb * mask)[:, None, None], axis=0))
                col += 3
            vs = uniform_filter1d(v, smooth, axis=0, mode="nearest")
            pc = self._position(v)
            pc = pc - pc.mean(0)
            y = np.concatenate([-sv * vs[rows].ravel(), -sp * pc[rows].ravel()])
            pri = np.concatenate([np.full(3, 1 / 25.0), np.full(2, 1 / sig_phi ** 2), np.full(3 * Ka, pr["a"]), np.full(3 * Kg, pr["g"]),
                                  np.full(3 * G, pr["dv"]), np.tile([pr["th"], pr["th"], pr["psi"]], G)])
            cur = np.concatenate([x["v0"], np.zeros(2), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel(), x["theta"].ravel()])
            H = A.T @ A + np.diag(pri)
            rhs = A.T @ y - pri * cur
            improved = False
            D = np.diag(np.maximum(np.diag(H), 1e-9))
            for _try in range(6):
                dx = np.linalg.solve(H + lam * D, rhs)
                xn = self._step(x, dx, 1.0)
                qn, vn, Rn, fn = self.forward(xn)
                o = self.objective(vn, xn, pr, smooth, sig_v, sig_p)
                if o < best:
                    x, q, v, Rm, fw, best = xn, qn, vn, Rn, fn, o
                    lam = max(lam / 3.0, 1e-6)
                    improved = True
                    break
                lam *= 8.0
            if not improved:
                break
        self.x = x
        return q, v, self.tb, self.last, dict(ca=x["ca"], cg=x["cg"], rms_v=float(np.sqrt((v ** 2).sum(1).mean())), obj=float(best))

    def _step(self, x, dx, al):
        Ka, Kg, G = self.Ka, self.Kg, self.G
        xn = {k: np.array(v, float) for k, v in x.items()}
        p = 0
        xn["v0"] += al * dx[p : p + 3]
        p += 3
        dphi = al * np.array([dx[p], dx[p + 1], 0.0])
        q = _qmul(_qexp(dphi), xn["q0"])
        xn["q0"] = q / np.linalg.norm(q)
        p += 2
        xn["ca"] += al * dx[p : p + 3 * Ka].reshape(Ka, 3)
        p += 3 * Ka
        xn["cg"] += al * dx[p : p + 3 * Kg].reshape(Kg, 3)
        p += 3 * Kg
        if G:
            xn["dv"] += al * dx[p : p + 3 * G].reshape(G, 3)
            p += 3 * G
            xn["theta"] += al * dx[p : p + 3 * G].reshape(G, 3)
        return xn


def energy_cal(t, imu, ok, b0=None, knot_a=60.0, knot_g=60.0, return_info=False, **kw):
    ec = EnergyCal(t, imu, ok, b0=b0, knot_a=knot_a, knot_g=knot_g)
    q, v, tb, last, info = ec.solve(**kw)
    return (q, v, tb, last, info) if return_info else (q, v, tb, last)
