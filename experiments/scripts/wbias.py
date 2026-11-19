"""Gyro bias from the file itself, without a rest period and from any starting error (15 Nov).

The whole-file gravity fit (offline_bias) linearises around one long integration, so it cannot recover from a starting
bias that is wrong by 0.08 rad/s: the attitude is already lost after 20 s. Here the file is cut into short windows. In each
one the up direction seen by the accelerometer at the start is carried to the end by the gyro and compared with the up
direction seen there. The mismatch is linear in the bias, so all windows together give a small least-squares problem.
Only the bias components across the up direction can be seen; the component along it stays at zero.
On a drone the accelerometer shows the thrust axis, not gravity. Real tilt changes then act as noise with zero mean.

wbias(t, gyro, acc, ok) -> b (3,), N (3x3 information matrix), number of windows used
agree(b, N, cal) -> distance between b and a calibration, counted only in the directions the file can see (rad/s)
"""
import numpy as np

W_S, EDGE_S, ITERS, RIDGE = 4.0, 0.5, 3, 1e-2


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def _exp(r):
    th = float(np.sqrt(r @ r))
    K = _skew(r)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / th ** 2) * (K @ K)


def wbias(t, gyro, acc, ok, w_s=W_S, edge_s=EDGE_S, iters=ITERS, ridge=RIDGE):
    fs = int(round(1.0 / np.median(np.diff(t))))
    wn, en = int(round(w_s * fs)), max(1, int(round(edge_s * fs)))
    last = np.nonzero(ok)[0][-1]
    bad = np.concatenate([[0], np.cumsum(~ok)])
    wins = [k for k in range(0, last + 1 - wn, wn) if bad[k + wn] - bad[k] == 0]
    b = np.zeros(3)
    N = np.zeros((3, 3))
    if not wins:
        return b, N, 0
    dt = np.diff(t, append=t[-1] + 1.0 / fs)
    for it in range(iters):
        rows = []
        for k0 in wins:
            k1 = k0 + wn
            m0 = acc[k0:k0 + en].mean(0)
            m1 = acc[k1 - en:k1].mean(0)
            n0, n1 = np.linalg.norm(m0), np.linalg.norm(m1)
            if n0 < 1e-6 or n1 < 1e-6:
                continue
            m0, m1 = m0 / n0, m1 / n1
            C = np.eye(3)
            IC = np.zeros((3, 3))
            for i in range(k0 + en // 2, k1 - en // 2):
                IC += C * dt[i]
                C = C @ _exp((gyro[i] - b) * dt[i])
            r = m1 - C.T @ m0
            S = -C.T @ _skew(m0) @ IC
            rows.append((r, S))
        if not rows:
            return b, N, 0
        res = np.array([np.linalg.norm(r) for r, S in rows])
        sc = 1.4826 * np.median(res) + 1e-6
        wts = np.minimum(1.0, 1.5 * sc / np.maximum(res, 1e-12))      # Huber weights on the window mismatch
        N = sum(w * S.T @ S for w, (r, S) in zip(wts, rows))
        g = sum(w * S.T @ r for w, (r, S) in zip(wts, rows))
        lam = ridge * np.trace(N) / 3.0
        # ridge pulls the total estimate (not the step) to zero, so the unseen component stays at zero
        b = np.linalg.solve(N + lam * np.eye(3), g + N @ b)
    return b, N, len(rows)


def agree(b, N, cal):
    d = np.asarray(b) - np.asarray(cal)
    tr = np.trace(N)
    return float(np.sqrt(max(0.0, 2.0 * d @ N @ d / tr))) if tr > 0 else 0.0
