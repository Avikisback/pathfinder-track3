"""H3: whole-file attitude. The file is offline, so gyro bias can be fitted against gravity over the full sequence.

Idea: if attitude is right, mean accel over a second, rotated to the world, points straight up. A small world
rotation phi changes that horizontally, and phi caused by a body-frame bias error is a running sum of R(t) b(t) dt.
So tilt residuals are LINEAR in spline coefficients of b(t). Bias along any body axis that moved relative to gravity
becomes observable. Fitted with a few Gauss-Newton passes. Nothing here uses ground truth.
"""
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

from common import _qexp, _qmul, baseline_attitude, quiet_init


@njit(cache=True)
def integrate_rows(t, gyro, ok, bias, coast, q0, tilt_step):
    """q per row (w x y z). Valid rows use gyro-bias, invalid rows use coast. tilt_step is a world-frame rotation
    per row, used to spread a correction across a gap."""
    n = len(t)
    out = np.zeros((n, 4))
    q = q0.copy()
    out[0] = q
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        w = gyro[k] - bias[k] if ok[k] else coast[k]
        q = _qmul(q, _qexp(w * dt))
        if tilt_step[k, 0] != 0.0 or tilt_step[k, 1] != 0.0:
            q = _qmul(_qexp(tilt_step[k]), q)
        q /= np.sqrt(np.sum(q * q))
        out[k] = q
    return out


def find_gaps(ok):
    """Runs of invalid rows that have valid data on both sides, plus any final run."""
    gaps = []
    n = len(ok)
    k = 0
    while k < n:
        if not ok[k]:
            s = k
            while k < n and not ok[k]:
                k += 1
            gaps.append((s, k))
        else:
            k += 1
    return gaps


def coast_rates(t, gyro, ok, bias, gaps, mode, tau=0.5, side_s=0.25):
    """Body rate to use on invalid rows. 'lerp' uses both sides of the gap, 'decay' only the past, as the baseline."""
    coast = np.zeros_like(gyro)
    fs = int(round(1.0 / np.median(np.diff(t))))
    side = max(1, int(side_s * fs))
    for s, e in gaps:
        before = gyro[max(0, s - side):s] - bias[max(0, s - side):s]
        wb = before.mean(0) if len(before) else np.zeros(3)
        after = gyro[e:e + side] - bias[e:e + side] if e < len(t) else np.zeros((0, 3))
        n = e - s
        x = (np.arange(n) + 1.0) / (n + 1.0)
        if mode == "lerp" and len(after):
            coast[s:e] = wb[None] * (1 - x)[:, None] + after.mean(0)[None] * x[:, None]
        elif mode == "zero":
            coast[s:e] = 0.0
        else:  # past only: decays to zero (also what happens after the final failure)
            coast[s:e] = wb[None] * np.exp(-(t[s:e] - t[s - 1] if s > 0 else t[s:e] - t[s])[:, None] / tau)
    return coast


def spline_basis(tt, knots):
    """Linear spline: returns matrix (len(tt), len(knots))."""
    B = np.zeros((len(tt), len(knots)))
    idx = np.clip(np.searchsorted(knots, tt) - 1, 0, len(knots) - 2)
    x = np.clip((tt - knots[idx]) / (knots[idx + 1] - knots[idx]), 0, 1)
    B[np.arange(len(tt)), idx] = 1 - x
    B[np.arange(len(tt)), idx + 1] = x
    return B


def offline_attitude(t, imu, ok, knot_s=30.0, W=1.0, iters=3, gap_mode="lerp", sigma_b=0.05, sigma_gap=0.5,
                     residual_s=0.0, fit_bias=True, return_info=False, s_r=0.03, b0=None):
    gyro, acc = imu[:, :3], imu[:, 3:]
    n = len(t)
    b0 = np.zeros(3) if b0 is None else np.asarray(b0)
    imu2 = imu.copy()
    imu2[ok, :3] -= b0
    q0, g = quiet_init(t, imu2, ok)
    an = np.linalg.norm(acc[ok], axis=1)
    g = np.median(an)
    T = t[-1] - t[0]
    last_good = np.nonzero(ok)[0][-1]
    t_end = t[last_good]
    J = max(2, int(round((t_end - t[0]) / knot_s)) + 1)
    knots = np.linspace(t[0], t_end, J)
    knots[-1] += 1e-3
    gaps_all = find_gaps(ok)
    gaps = [(s, e) for s, e in gaps_all if e < n and s > 0]    # bracketed gaps only
    G = len(gaps)
    c = np.zeros((J, 3))
    theta = np.zeros((G, 2))
    Bk = spline_basis(np.minimum(t, t_end), knots)

    # one-second windows of mean accel
    fs = int(round(1.0 / np.median(np.diff(t))))
    wn = max(1, int(W * fs))
    starts = np.arange(0, last_good - wn, wn)
    cs = np.cumsum(np.vstack([np.zeros((1, 3)), acc * ok[:, None]]), 0)
    cn = np.cumsum(np.concatenate([[0], ok.astype(float)]))
    cnt = cn[starts + wn] - cn[starts]
    keep = cnt > 0.5 * wn
    starts, cnt = starts[keep], cnt[keep]
    am = (cs[starts + wn] - cs[starts]) / cnt[:, None]
    an_w = np.linalg.norm(am, axis=1)
    centre = starts + wn // 2
    w0 = (cnt / wn) * np.exp(-((an_w - g) / 0.5) ** 2)
    u = am / an_w[:, None]

    q = None
    for it in range(iters + 1):
        bias = Bk @ c + b0
        coast = coast_rates(t, gyro, ok, bias, gaps_all, gap_mode)
        ts = np.zeros((n, 3))
        for g_i, (s, e) in enumerate(gaps):
            ts[s:e, 0:2] = theta[g_i] / max(1, e - s)
        q = integrate_rows(t, gyro, ok, bias, coast, q0, ts)
        if it == iters or not fit_bias:
            break
        # blocks of 10 rows for the running sums
        hop = 10
        bi = np.arange(hop // 2, n, hop)
        Rm = R.from_quat(q[bi][:, [1, 2, 3, 0]]).as_matrix()
        dtb = t[np.minimum(bi + hop // 2, n - 1)] - t[bi - hop // 2]
        okb = ok[bi].astype(float)
        Bb = Bk[bi]
        # cumulative -sum R B dt, shape (blocks, J, 3 world, 3 body)
        inc = -(Rm[:, None, :, :] * (Bb * (dtb * okb)[:, None])[:, :, None, None])
        Cum = np.cumsum(inc, axis=0)
        m = R.from_quat(q[centre][:, [1, 2, 3, 0]]).apply(u)   # measured up in the estimate world
        kb = np.minimum(np.searchsorted(bi, centre), len(bi) - 1)
        rows = 2 * len(centre)
        ncol = 2 + 3 * J + 2 * G
        A = np.zeros((rows, ncol))
        # target: phi_x = m_y, phi_y = -m_x  (see header); weighted later
        y = np.concatenate([m[:, 1], -m[:, 0]])
        K = len(centre)
        A[:K, 0] = 1.0
        A[K:, 1] = 1.0
        A[:K, 2:2 + 3 * J] = Cum[kb, :, 0, :].reshape(K, -1)
        A[K:, 2:2 + 3 * J] = Cum[kb, :, 1, :].reshape(K, -1)
        for g_i, (s, e) in enumerate(gaps):
            aft = centre >= e
            A[:K, 2 + 3 * J + 2 * g_i] = aft
            A[K:, 2 + 3 * J + 2 * g_i + 1] = aft
        w = np.concatenate([w0, w0])
        # robust pass: downweight windows with large residual from this iteration's starting point
        ATA = (A.T * w) @ A / s_r ** 2
        ATy = (A.T * w) @ y / s_r ** 2
        pri = np.zeros(ncol)
        pri[2:2 + 3 * J] = 1.0 / sigma_b ** 2
        pri[2 + 3 * J:] = 1.0 / sigma_gap ** 2
        # Priors act on the total value, so the step is pulled back toward zero total
        x0 = np.concatenate([[0, 0], c.ravel(), theta.ravel()])
        dx = np.linalg.solve(ATA + np.diag(pri + 1e-9), ATy - pri * x0)
        # apply: start tilt as world rotation
        dphi = np.array([dx[0], dx[1], 0.0])
        qd = _qexp(dphi)
        q0 = _qmul(qd, q0)
        q0 /= np.linalg.norm(q0)
        c = c + dx[2:2 + 3 * J].reshape(J, 3)
        if G:
            theta = theta + dx[2 + 3 * J:].reshape(G, 2)
    if residual_s > 0:
        # zero-phase smoothing of what is still wrong in tilt, applied as a slowly varying world rotation
        m = R.from_quat(q[centre][:, [1, 2, 3, 0]]).apply(u)
        tc = t[centre]
        tg = np.arange(t[0], t[-1] + 0.5, 0.5)
        ker = np.exp(-0.5 * ((tg[:, None] - tc[None]) / residual_s) ** 2) * w0[None]
        den = ker.sum(1) + 1e-3
        px = (ker * m[:, 1]).sum(1) / den
        py = (ker * -m[:, 0]).sum(1) / den
        phi = np.stack([np.interp(t, tg, px), np.interp(t, tg, py), np.zeros(n)], 1)
        q = (R.from_rotvec(phi) * R.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    if return_info:
        return q, dict(bias=c, knots=knots, theta=theta)
    return q


def still_bias(t, imu, ok, std_thr=0.01):
    """Mean gyro over truly still seconds, or None. Same test as main.cpp: gyro std per axis under 0.01 rad/s."""
    fs = int(round(1.0 / np.median(np.diff(t))))
    means, first = [], None
    for s in range(0, len(t) - fs, fs):
        m = ok[s:s + fs]
        if m.sum() < 0.8 * fs:
            continue
        w = imu[s:s + fs, :3][m]
        if np.sqrt(w.var(0).mean()) < std_thr:
            means.append(w.mean(0))
            if first is None:
                first = float(t[s])
    if not means:
        return None, 0.0, None
    return np.median(np.array(means), 0), float(len(means)), first


def gravity_bias(t, imu, ok, sig_thr=0.004, W=1.0, return_all=False):
    """Closed-form constant gyro bias from how gravity moves in the body frame. Needs no starting attitude.

    Mean accel over a window gives up-in-body u. If the gyro were right, the gyro rotation between two windows
    maps u[k+L] onto u[k]. The mismatch is linear in bias: dR u[k+L] - u[k] = L dR (b x u[k+L]). Directions of b
    that gravity does not constrain stay at zero, by truncating on the estimated standard deviation.
    """
    fs = int(round(1.0 / np.median(np.diff(t))))
    wn = int(W * fs)
    last = np.nonzero(ok)[0][-1]
    starts = np.arange(0, last - wn, wn)
    cs = np.cumsum(np.vstack([np.zeros((1, 3)), imu[:, 3:] * ok[:, None]]), 0)
    cn = np.cumsum(np.concatenate([[0], ok.astype(float)]))
    cnt_w = cn[starts + wn] - cn[starts]
    good = cnt_w > 0.9 * wn
    am = (cs[starts + wn] - cs[starts]) / np.maximum(cnt_w, 1)[:, None]
    an = np.linalg.norm(am, axis=1)
    g = np.median(an[good])
    u = am / np.maximum(an, 1e-6)[:, None]
    good &= np.abs(an - g) < 1.0
    centre = starts + wn // 2
    gap_id = np.cumsum(~ok)
    b = np.zeros(3)
    sig = np.full(3, 9.0)
    V = np.eye(3)
    for lags in ((1, 2), (1, 2, 4, 8), (1, 2, 4, 8, 16)):
        coast = np.zeros_like(imu[:, :3])
        q = integrate_rows(t, imu[:, :3], ok, np.tile(b, (len(t), 1)), coast, np.array([1.0, 0, 0, 0]), np.zeros((len(t), 3)))
        Rq = R.from_quat(q[centre][:, [1, 2, 3, 0]])
        H, y, ss, cnt = np.zeros((3, 3)), np.zeros(3), 0.0, 0
        for L in lags:
            i = np.arange(0, len(centre) - L)
            i = i[good[i] & good[i + L] & (gap_id[centre[i]] == gap_id[centre[i + L]])]
            if len(i) < 10:
                continue
            dR = Rq[i].inv() * Rq[i + L]
            uj = u[i + L]
            r = dR.apply(uj) - u[i]
            dt = t[centre[i + L]] - t[centre[i]]
            Dm = dR.as_matrix()
            cx = np.zeros((len(i), 3, 3))   # matrix of b -> b x u
            cx[:, 0, 1], cx[:, 0, 2] = uj[:, 2], -uj[:, 1]
            cx[:, 1, 0], cx[:, 1, 2] = -uj[:, 2], uj[:, 0]
            cx[:, 2, 0], cx[:, 2, 1] = uj[:, 1], -uj[:, 0]
            A = dt[:, None, None] * (Dm @ cx)
            H += np.einsum("nij,nik->jk", A, A)
            y += np.einsum("nij,ni->j", A, r)
            ss += (r ** 2).sum()
            cnt += 3 * len(i)
        if cnt == 0:
            break
        s2 = max(ss / cnt, 1e-6)
        ev, V = np.linalg.eigh(H)
        sig = np.sqrt(s2 / np.maximum(ev, 1e-12))
        b = b + V @ np.where(sig < sig_thr, (V.T @ y) / np.maximum(ev, 1e-12), 0.0)
        # strong directions only: never keep a component that gravity does not constrain
        b = V @ np.where(sig < sig_thr, V.T @ b, 0.0)
    if return_all:
        return b, sig, V
    return b


STRONG = dict(knot_s=60, s_r=0.3, sigma_b=0.003, residual_s=5.0)


def auto_bias(t, imu, ok):
    sb = still_bias(t, imu, ok)[0]
    return sb if sb is not None else gravity_bias(t, imu, ok)


def attitude_auto(t, imu, ok):
    """Best no-calibration attitude: still-second bias if there is one, else gravity fit, then the offline fit."""
    return offline_attitude(t, imu, ok, b0=auto_bias(t, imu, ok), **STRONG)


def attitude_fast(t, imu, ok):
    """Cheaper version for training pools: same bias, forward tilt filter."""
    i2 = imu.copy()
    i2[ok, :3] -= auto_bias(t, imu, ok)
    return baseline_attitude(t, i2, ok)
