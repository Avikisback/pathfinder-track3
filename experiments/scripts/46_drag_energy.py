"""New idea 6: on drones, replace "velocity is near zero" with "velocity is near what drag says".

Multirotor drag gives body-plane velocity directly from the accelerometer at low frequency. Feeding it to the energy solve
as a velocity measurement gives a smoother that needs no bounded-room assumption for the two horizontal axes, and heading
drift starts to matter through R * v_body. Measured through the same harness. k comes from other sequences only.
"""
import importlib, multiprocessing as mp, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
from phys import *
from phys import _lpf, _hpf
from energy2 import EnergyCal, cross_mat
from scipy.ndimage import uniform_filter1d

K_DRAG = 0.187


class DragCal(EnergyCal):
    """Same solver. The velocity rows now compare with a target velocity instead of zero."""

    def set_target(self, k=K_DRAG, fc=0.1):
        t, acc, ok = self.t, self.acc, self.ok
        fs = round(1.0 / np.median(np.diff(t)))
        idx = np.arange(self.n)
        a = np.stack([np.interp(idx, idx[ok], acc[ok, i]) for i in range(3)], 1)
        on = rotors_on(a, fs)
        last = self.last
        self.drone = on[: last + 1].mean() > 0.3
        if not self.drone:
            self.vb_t = np.zeros((len(self.idx), 3))
            self.on_b = np.zeros(len(self.idx), bool)
            return
        ta = a[: last + 1][on[: last + 1]].mean(0)
        ta /= np.linalg.norm(ta)
        e1 = np.cross(ta, [0, 1, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ta, e1)
        E = np.stack([e1, e2], 1)
        fp = _lpf(a, fc, fs) @ E
        off = fp[: last + 1][on[: last + 1]].mean(0)
        vp = -(fp - off) / k
        vb = vp @ E.T
        self.vb_t = vb[self.mid]
        self.on_b = on[self.mid].astype(bool)
        self.E = E


def solve_drag(ec, sig_d=1.0, auto_p=None, **kw):
    """Fixed-point on the solver: compare with R*v_drag at the current attitude, iterate.
    auto_p: position prior width follows this file's own estimated spread, times this factor."""
    for outer in range(4 if auto_p else 3):
        v, Rg, Rm, fw, Sm = ec.forward(ec.x)
        shift = np.einsum("nij,nj->ni", Rm, ec.vb_t) * ec.on_b[:, None]
        if auto_p and outer > 0:
            pos = np.cumsum(v * ec.dtb[:, None], axis=0)
            kw["sig_p"] = float(np.clip(auto_p * np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean()), 1.0, 20.0))
        q, vs, tb, last, info = _solve_shift(ec, shift, **kw)
    info["sig_p"] = kw.get("sig_p")
    return q, vs, tb, last, info


def _solve_shift(ec, shift, iters=8, sig_a=0.1, sig_g=0.01, sig_dv=1.0, sig_v=1.0, sig_off=None, sig_p=5.0, smooth_s=5.0, row_s=1.0):
    fs_b = 1.0 / np.median(ec.dtb)
    smooth = max(1, int(smooth_s * fs_b))
    stride = max(1, int(row_s * fs_b))
    ec.sw = 1.0 / stride
    pr = dict(a=1 / sig_a ** 2, g=1 / sig_g ** 2, dv=1 / sig_dv ** 2)
    x = {k: np.array(v, float) for k, v in ec.x.items()}
    Ka, Kg, G, K = ec.Ka, ec.Kg, ec.G, ec.K
    np_total = 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G
    rows = np.arange(smooth // 2, len(ec.idx), stride)
    m = len(rows)
    dtb = ec.dtb
    Id = np.tile(np.eye(3), (len(dtb), 1, 1))
    sp = 1.0 / sig_p
    # velocity rows are trusted according to whether a drag measurement exists there
    sv_b = np.where(ec.on_b, 1.0 / sig_v, 1.0 / (sig_off if sig_off else sig_v))
    sv = sv_b[rows]
    sv3 = np.repeat(sv, 3)

    def obj(v, x):
        d = uniform_filter1d(v - shift, smooth, axis=0, mode="nearest")
        p = np.cumsum(v * dtb[:, None], axis=0)
        p = p - p.mean(0)
        return (((d * sv_b[:, None]) ** 2).sum() + (p ** 2).sum() * sp ** 2) * ec.sw + (x["ca"] ** 2).sum() * pr["a"] + (x["cg"] ** 2).sum() * pr["g"] + (x["dv"] ** 2).sum() * pr["dv"]

    v, Rg, Rm, fw, Sm = ec.forward(x)
    best = obj(v, x)
    lam = 1e-3
    for it in range(iters):
        Fx = cross_mat(fw)
        A = np.zeros((6 * m, np_total))

        def put(col, M):
            sm = uniform_filter1d(M, smooth, axis=0, mode="nearest")
            A[: 3 * m, col : col + M.shape[2]] = sv3[:, None] * sm[rows].reshape(3 * m, M.shape[2])
            P = np.cumsum(M * dtb[:, None, None], axis=0)
            P = P - P.mean(0, keepdims=True)
            A[3 * m :, col : col + M.shape[2]] = sp * P[rows].reshape(3 * m, M.shape[2])

        put(0, Id)
        col = 3
        for k in range(K):
            mk = (ec.seg_of == k).astype(float)
            put(col, np.cumsum(-Fx * (dtb * mk)[:, None, None], axis=0)[:, :, :2])
            col += 2
        wdt = dtb * ec.okb
        for j in range(Ka):
            put(col, np.cumsum(-Rm * (ec.Ba[:, j] * wdt)[:, None, None], axis=0))
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
        d = uniform_filter1d(v - shift, smooth, axis=0, mode="nearest")
        pc = np.cumsum(v * dtb[:, None], axis=0)
        pc = pc - pc.mean(0)
        y = np.concatenate([-sv3 * d[rows].ravel(), -sp * pc[rows].ravel()])
        pri = np.concatenate([np.full(3, 1 / 25.0), np.zeros(2 * K), np.full(3 * Ka, pr["a"]), np.full(3 * Kg, pr["g"]), np.full(3 * G, pr["dv"])])
        cur = np.concatenate([x["v0"], np.zeros(2 * K), x["ca"].ravel(), x["cg"].ravel(), x["dv"].ravel()])
        H = A.T @ A + np.diag(pri)
        rhs = A.T @ y - pri * cur
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
        if not ok:
            break
    ec.x = x
    return ec.finish(x, v, float(best))


VARS = {"au15": dict(sig_v=0.5, sig_p=5.0, auto_p=1.5), "off3": dict(sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_off=3.0), "off10": dict(sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_off=10.0),
        "au2off3": dict(sig_v=0.5, sig_p=5.0, auto_p=2.0, sig_off=3.0), "v03off3": dict(sig_v=0.3, sig_p=5.0, auto_p=1.5, sig_off=3.0)}


def one(key):
    n, s, m, sd = key
    t, imu, _ = eval_file(n, s, m, sd)
    P = prepare(t, imu)
    room = load_room(n)
    b0 = m41.calib_excl(n)
    out = {}
    tk = P["t"][np.arange(0, P["last_good"] + 1, m41.DECV)]
    for nm, kw in VARS.items():
        ec = DragCal(P["t"], P["imu"], P["ok"], b0=b0)
        ec.set_target()
        if kw is None:
            q, v, tb, last, info = ec.solve(sig_a=0.1, sig_p=5.0)
        else:
            q, v, tb, last, info = solve_drag(ec, **kw)
        v20 = np.stack([np.interp(tk, tb, v[:, i]) for i in range(3)], 1)
        p = np.cumsum(v20, axis=0) / m41.FV
        a, st = m41.ate(p, tk, room)
        pre = P["ids"] != -1
        pre[last + 1 :] = False
        d = score(P["t_ns"][pre] * 1e-9, np.zeros((int(pre.sum()), 3)), q[pre][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        out[nm] = dict(a=a, st=st, att=d["att_org"], tilt=d["tilt"], rpe_r=d["rpe_r"])
    return key, out


if __name__ == "__main__":
    files = [k for k in eval_set(m41.SEQS, seeds=(1,), modes=("zero",))]
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(10) as p:
        for k, o in p.imap_unordered(one, files):
            res[k] = o
    print(f"{len(res)} files {time.time()-t0:.0f}s")
    for sc in ("clean", "mild", "full"):
        ks = [k for k in res if k[1] == sc]
        for v in VARS:
            print(f"{sc:5s} {v:7s} raw ATE {np.mean([res[k][v]['a'] for k in ks]):.3f}/{np.max([res[k][v]['a'] for k in ks]):.3f} att {np.mean([res[k][v]['att'] for k in ks]):6.1f} tilt {np.mean([res[k][v]['tilt'] for k in ks]):5.2f}  by seq " + " ".join(f"{res[k][v]['a']:.2f}" for k in sorted(ks, key=lambda x: x[0])), flush=True)
    print("stationary", " ".join(f"{res[k]['au15']['st']:.2f}" for k in sorted([k for k in res if k[1] == 'full'], key=lambda x: x[0])))


def solve_any(t, imu, ok, b0=None, **kw):
    """Main entry. Drones: drag-aided solve. Anything else: plain bounded-motion solve. The choice comes from the rotor
    signature in the accelerometer, which identifies a sensor rig, not a sequence."""
    ec = DragCal(t, imu, ok, b0=b0)
    ec.set_target()
    if ec.drone:
        q, v, tb, last, info = solve_drag(ec, sig_v=0.5, sig_p=5.0, auto_p=1.5, sig_off=kw.get("sig_off", None))
    else:
        q, v, tb, last, info = ec.solve(sig_a=0.1, sig_p=5.0)
    info["drone"] = bool(ec.drone)
    return q, v, tb, last, info
