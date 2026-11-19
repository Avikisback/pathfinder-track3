"""Energy-minimising self-calibration, second form: each stretch between gaps has its own tilt.

Double integration explodes because of a few slow sensor errors. A velocity that runs away is an extremely sensitive
measurement of those errors, so solve for them: bias curves, a tilt per segment and a velocity change per gap, chosen so
that the slow part of velocity and the position stay small. Needs only "stays bounded", with no map or room size. Nothing
here uses ground truth.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R, Slerp

from attitude import coast_rates, find_gaps, integrate_rows, offline_attitude, spline_basis
from common import quiet_init


def cross_mat(f):
    m = np.zeros((len(f), 3, 3))
    m[:, 0, 1], m[:, 0, 2] = -f[:, 2], f[:, 1]
    m[:, 1, 0], m[:, 1, 2] = f[:, 2], -f[:, 0]
    m[:, 2, 0], m[:, 2, 1] = -f[:, 1], f[:, 0]
    return m


class EnergyCal:
    def __init__(self, t, imu, ok, b0=None, knot_a=60.0, knot_g=60.0, hop=4, seed_w=20.0):
        self.t, self.imu, self.ok = t, imu, ok
        self.gyro, self.acc = imu[:, :3], imu[:, 3:]
        self.n = len(t)
        self.hop = hop
        self.b0 = np.zeros(3) if b0 is None else np.asarray(b0, float)
        self.last = int(np.nonzero(ok)[0][-1])
        self.t_end = t[self.last]
        self.g = float(np.median(np.linalg.norm(self.acc[ok], axis=1)))
        self.Ka = max(2, int(round((self.t_end - t[0]) / knot_a)) + 1)
        self.Kg = max(2, int(round((self.t_end - t[0]) / knot_g)) + 1)
        tc = np.minimum(t, self.t_end)
        self.Ba_r = spline_basis(tc, np.linspace(t[0], self.t_end + 1e-3, self.Ka))
        self.Bg_r = spline_basis(tc, np.linspace(t[0], self.t_end + 1e-3, self.Kg))
        self.gaps_all = find_gaps(ok)
        self.gaps = [(s, e) for s, e in self.gaps_all if e < self.n and s > 0]
        self.G = len(self.gaps)
        # segments: row ranges with data
        edges = [0] + [e for s, e in self.gaps]
        ends = [s for s, e in self.gaps] + [self.last + 1]
        self.segs = list(zip(edges, ends))
        self.K = len(self.segs)
        i = np.arange(0, self.last + 1 - hop + 1, hop)
        self.idx = i
        self.mid = i + hop // 2
        self.tb = t[self.mid]
        self.okb = ok[self.mid].astype(float)
        self.fb = np.add.reduceat(self.acc[: i[-1] + hop] * ok[: i[-1] + hop, None], i, axis=0) / np.maximum(
            np.add.reduceat(ok[: i[-1] + hop].astype(float), i), 1)[:, None]
        self.dtb = np.diff(t[np.append(i, min(i[-1] + hop, self.last))])
        self.Ba, self.Bg = self.Ba_r[self.mid], self.Bg_r[self.mid]
        self.seg_of = np.full(len(i), -1)
        for k, (a, b) in enumerate(self.segs):
            self.seg_of[(self.mid >= a) & (self.mid < b)] = k
        self.x = dict(v0=np.zeros(3), S=np.zeros((self.K, 3)), ca=np.zeros((self.Ka, 3)), cg=np.zeros((self.Kg, 3)), dv=np.zeros((self.G, 3)))
        im2 = imu.copy()
        im2[ok, :3] -= self.b0
        self.q0, _ = quiet_init(t, im2, ok)
        # gravity fit gets gyro bias into the valid range. Its gap handling is not used.
        _, info = offline_attitude(t, imu, ok, b0=self.b0, knot_s=knot_g, s_r=0.3, sigma_b=0.003, residual_s=0.0, return_info=True)
        self.x["cg"] = spline_basis(np.linspace(t[0], self.t_end, self.Kg), info["knots"]) @ info["bias"]
        self._seed_tilts(seed_w)

    def gyro_q(self, x):
        bias = self.Bg_r @ x["cg"] + self.b0
        coast = coast_rates(self.t, self.gyro, self.ok, bias, self.gaps_all, "lerp")
        return integrate_rows(self.t, self.gyro, self.ok, bias, coast, self.q0, np.zeros((self.n, 3)))

    def _seed_tilts(self, W):
        """Up direction per segment from gravity over the segment's first W seconds. Exact, so any size of rotation is fine."""
        fs = 1.0 / np.median(np.diff(self.t))
        qg = self.gyro_q(self.x)
        for k, (a, b) in enumerate(self.segs):
            stop = min(a + int(W * fs), b)
            rr = np.arange(a, stop)
            rr = rr[self.ok[rr]]
            if len(rr) < 50:
                continue
            Rm = R.from_quat(qg[rr][:, [1, 2, 3, 0]]).as_matrix()
            u = np.einsum("nij,nj->ni", Rm, self.acc[rr]).mean(0)
            u /= np.linalg.norm(u)
            ax = np.cross(u, [0.0, 0.0, 1.0])
            sn = np.linalg.norm(ax)
            if sn > 1e-9:
                self.x["S"][k] = ax / sn * np.arctan2(sn, u[2])

    def _ramp(self, s, e):
        return np.clip((self.idx - s) / max(1, e - s), 0.0, 1.0)

    def forward(self, x):
        qg = self.gyro_q(x)
        Rg = R.from_quat(qg[self.mid][:, [1, 2, 3, 0]]).as_matrix()
        Sm = R.from_rotvec(x["S"]).as_matrix()
        Rm = Rg.copy()
        for k in range(self.K):
            m = self.seg_of == k
            Rm[m] = Sm[k] @ Rg[m]
        fw = np.einsum("nij,nj->ni", Rm, self.fb) * self.okb[:, None]
        aw = fw - np.einsum("nij,nj->ni", Rm, self.Ba @ x["ca"]) * self.okb[:, None] - np.array([0, 0, self.g]) * self.okb[:, None]
        v = np.cumsum(aw * self.dtb[:, None], axis=0) + x["v0"]
        for j, (s, e) in enumerate(self.gaps):
            v += x["dv"][j][None] * self._ramp(s, e)[:, None]
        return v, Rg, Rm, fw, Sm

    def objective(self, v, x, smooth, sig_v, sig_p, pr):
        vs = uniform_filter1d(v, smooth, axis=0, mode="nearest")
        p = np.cumsum(v * self.dtb[:, None], axis=0)
        p = p - p.mean(0)
        return ((vs ** 2).sum() / sig_v ** 2 + (p ** 2).sum() / sig_p ** 2) * self.sw + (x["ca"] ** 2).sum() * pr["a"] + (x["cg"] ** 2).sum() * pr["g"] + (x["dv"] ** 2).sum() * pr["dv"]

    def solve(self, iters=12, sig_a=0.3, sig_g=0.01, sig_dv=1.0, sig_v=3.0, sig_p=5.0, smooth_s=5.0, row_s=1.0):
        fs_b = 1.0 / np.median(self.dtb)
        smooth = max(1, int(smooth_s * fs_b))
        stride = max(1, int(row_s * fs_b))
        self.sw = 1.0 / stride
        pr = dict(a=1 / sig_a ** 2, g=1 / sig_g ** 2, dv=1 / sig_dv ** 2)
        x = {k: np.array(v, float) for k, v in self.x.items()}
        Ka, Kg, G, K = self.Ka, self.Kg, self.G, self.K
        np_total = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
        rows = np.arange(smooth // 2, len(self.idx), stride)
        m = len(rows)
        dtb = self.dtb
        v, Rg, Rm, fw, Sm = self.forward(x)
        best = self.objective(v, x, smooth, sig_v, sig_p, pr)
        lam = 1e-3
        Id = np.tile(np.eye(3), (len(dtb), 1, 1))
        for it in range(iters):
            Fx = cross_mat(fw)
            A = np.zeros((6 * m, np_total))
            sv, sp = 1.0 / sig_v, 1.0 / sig_p

            def put(col, M):
                sm = uniform_filter1d(M, smooth, axis=0, mode="nearest")
                A[: 3 * m, col : col + M.shape[2]] = sv * sm[rows].reshape(3 * m, M.shape[2])
                P = np.cumsum(M * dtb[:, None, None], axis=0)
                P = P - P.mean(0, keepdims=True)
                A[3 * m :, col : col + M.shape[2]] = sp * P[rows].reshape(3 * m, M.shape[2])

            put(0, Id)
            col = 3
            for k in range(K):
                mk = (self.seg_of == k).astype(float)
                put(col, np.cumsum(-Fx * (dtb * mk)[:, None, None], axis=0)[:, :, :2])
                col += 2
            wdt = dtb * self.okb
            for j in range(Ka):
                put(col, np.cumsum(-Rm * (self.Ba[:, j] * wdt)[:, None, None], axis=0))
                col += 3
            # a body-frame gyro bias change rotates the frame of later rows; seen through each segment's own tilt
            Sb = np.zeros_like(Rm)
            Sb[:] = np.eye(3)
            for k in range(K):
                Sb[self.seg_of == k] = Sm[k]
            for j in range(Kg):
                Phi = np.cumsum(-Rg * (self.Bg[:, j] * dtb)[:, None, None], axis=0)
                Phi = Sb @ Phi
                put(col, np.cumsum(-np.einsum("nij,njk->nik", Fx, Phi) * dtb[:, None, None], axis=0))
                col += 3
            for j, (s, e) in enumerate(self.gaps):
                put(col, Id * self._ramp(s, e)[:, None, None])
                col += 3
            vs = uniform_filter1d(v, smooth, axis=0, mode="nearest")
            pc = np.cumsum(v * dtb[:, None], axis=0)
            pc = pc - pc.mean(0)
            y = np.concatenate([-sv * vs[rows].ravel(), -sp * pc[rows].ravel()])
            pri = np.concatenate([np.full(3, 1 / 25.0), np.zeros(2 * K), np.full(3 * Ka, pr["a"]), np.full(3 * Kg, pr["g"]), np.full(3 * G, pr["dv"])])
            cur = np.concatenate([x["v0"], np.zeros(2 * K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()])
            H = A.T @ A + np.diag(pri)
            rhs = A.T @ y - pri * cur
            D = np.diag(np.maximum(np.diag(H), 1e-9))
            improved = False
            for _ in range(6):
                dx = np.linalg.solve(H + lam * D, rhs)
                xn = self._apply(x, dx)
                r = self.forward(xn)
                o = self.objective(r[0], xn, smooth, sig_v, sig_p, pr)
                if o < best:
                    x, best = xn, o
                    v, Rg, Rm, fw, Sm = r
                    lam = max(lam / 3.0, 1e-6)
                    improved = True
                    break
                lam *= 8.0
            if not improved:
                break
        self.x = x
        return self.finish(x, v, float(best))

    def _apply(self, x, dx):
        Ka, Kg, G, K = self.Ka, self.Kg, self.G, self.K
        xn = {k: np.array(v, float) for k, v in x.items()}
        p = 0
        xn["v0"] += dx[p : p + 3]
        p += 3
        for k in range(K):
            xn["S"][k] = (R.from_rotvec([dx[p], dx[p + 1], 0.0]) * R.from_rotvec(xn["S"][k])).as_rotvec()
            p += 2
        xn["ca"] += dx[p : p + 3 * Ka].reshape(Ka, 3)
        p += 3 * Ka
        xn["cg"] += dx[p : p + 3 * Kg].reshape(Kg, 3)
        p += 3 * Kg
        if G:
            xn["dv"] += dx[p : p + 3 * G].reshape(G, 3)
        return xn

    def finish(self, x, v, obj):
        """Orientation per row. Tilt jumps are spread across each gap, so the output stays continuous."""
        qg = self.gyro_q(x)
        rot = np.zeros((self.n, 3))
        S = x["S"]
        for k, (a, b) in enumerate(self.segs):
            rot[a:b] = S[k]
        for j, (s, e) in enumerate(self.gaps):
            sl = Slerp([0.0, 1.0], R.from_rotvec([S[j], S[j + 1]]))
            rot[s:e] = sl((np.arange(s, e) - s + 1.0) / (e - s + 1.0)).as_rotvec()
        rot[self.last + 1 :] = S[-1]
        q = (R.from_rotvec(rot) * R.from_quat(qg[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        return q, v, self.tb, self.last, dict(rms_v=float(np.sqrt((v ** 2).sum(1).mean())), obj=obj, ca=x["ca"], cg=x["cg"], dv=x["dv"], S=x["S"])


def energy_cal(t, imu, ok, b0=None, knot_a=60.0, knot_g=60.0, return_info=False, **kw):
    ec = EnergyCal(t, imu, ok, b0=b0, knot_a=knot_a, knot_g=knot_g)
    q, v, tb, last, info = ec.solve(**kw)
    return (q, v, tb, last, info) if return_info else (q, v, tb, last)
