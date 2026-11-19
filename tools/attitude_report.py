#!/usr/bin/env python3
"""Diagnose orientation: error over time, split into tilt and heading, plus the gyro bias implied
by ground truth. The bias printed here is for DIAGNOSIS ONLY. It must never go into the estimator,
because that would be using the answers.

Usage: attitude_report.py <gt.tum> <est.txt> [imu.csv]
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def load(path):
    a = np.array([[float(x) for x in l.split()] for l in open(path) if l[0] != "#"])
    return a[:, 0], R.from_quat(a[:, 4:8])


def main():
    tg, qg = load(sys.argv[1])
    te, qe = load(sys.argv[2])
    keep = (tg >= te[0]) & (tg <= te[-1])
    tg, qg = tg[keep], qg[keep]
    idx = np.clip(np.searchsorted(te, tg), 0, len(te) - 1)
    qe_m = qe[idx]
    # Line the two world frames up using the first 2 s only (heading is arbitrary at the start).
    n0 = max(1, int(np.sum(tg < tg[0] + 2)))
    align = (qg[:n0] * qe_m[:n0].inv()).mean()
    err = qg.inv() * (align * qe_m)
    ang = np.degrees(err.magnitude())
    up_g = qg.apply([0, 0, 1], inverse=True)
    up_e = (align * qe_m).apply([0, 0, 1], inverse=True)
    tilt = np.degrees(np.arccos(np.clip(np.sum(up_g * up_e, axis=1), -1, 1)))
    print(f"overall: rmse {np.sqrt(np.mean(ang**2)):.1f} deg, tilt rmse {np.sqrt(np.mean(tilt**2)):.1f} deg")
    for t in (5, 10, 20, 40, 80, 120, 140):
        i = np.searchsorted(tg, tg[0] + t)
        if i < len(tg):
            print(f"  t={t:>3}s total {ang[i]:6.1f} deg  tilt {tilt[i]:5.1f} deg")

    if len(sys.argv) > 3:
        imu = np.array([[float(x) for x in l.split(",")] for l in open(sys.argv[3]) if l[0] != "#"])
        ti, w = imu[:, 0] * 1e-9, imu[:, 1:4]
        # True rotation over 1 s windows, against the gyro's mean over the same window.
        diffs = []
        for t0 in np.arange(tg[0], tg[-1] - 1, 1.0):
            a, b = np.searchsorted(tg, [t0, t0 + 1])
            m = (ti >= tg[a]) & (ti < tg[b])
            if m.sum() < 150:
                continue
            true_rate = (qg[a].inv() * qg[b]).as_rotvec() / (tg[b] - tg[a])
            diffs.append(w[m].mean(axis=0) - true_rate)
        d = np.array(diffs)
        print("gyro bias implied by ground truth (median):", np.round(np.median(d, axis=0), 5))
        print("  spread (std of 1 s windows):", np.round(d.std(axis=0), 5))
        dt = np.diff(ti, prepend=ti[0])
        for name, bias in (("zero", np.zeros(3)), ("gt-implied", np.median(d, axis=0))):
            q = R.identity()
            out = []
            j = 0
            step = R.from_rotvec((w - bias) * dt[:, None])
            for k in range(len(ti)):
                q = q * step[k]
                while j < len(tg) and tg[j] <= ti[k]:
                    out.append(q.as_quat())
                    j += 1
            qq = R.from_quat(out)
            al = (qg[:n0] * qq[:n0].inv()).mean()
            a2 = np.degrees((qg[: len(qq)].inv() * (al * qq)).magnitude())
            print(f"  pure gyro, {name} bias: rmse {np.sqrt(np.mean(a2**2)):.1f} deg, end {a2[-1]:.1f} deg")


if __name__ == "__main__":
    main()
