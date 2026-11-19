#!/usr/bin/env python3
"""Measure the blind forecast: position error in the 5 s after the IMU stops.

Aligns estimate to ground truth using only poses BEFORE the failure, then measures the blind tail.
Also scores "just hold the last position" as a yardstick.

Usage: blind_error.py <gt.tum> <est.txt> <degraded_imu.csv>
"""
import sys

import numpy as np


def load(path):
    return np.array([[float(x) for x in l.split()[:4]] for l in open(path) if l[0] != "#"])


def main():
    gt, est = load(sys.argv[1]), load(sys.argv[2])
    t_fail = None
    with open(sys.argv[3]) as f:
        for line in f:
            if line[0] != "#" and line.strip():
                c = line.split(",")
                if any(float(x) != 0 for x in c[1:7]):
                    t_fail = int(c[0]) * 1e-9
    idx = np.clip(np.searchsorted(est[:, 0], gt[:, 0]), 0, len(est) - 1)
    close = np.abs(est[idx, 0] - gt[:, 0]) < 0.01
    g, e, t = gt[close, 1:4], est[idx[close], 1:4], gt[close, 0]
    pre, tail = t <= t_fail, t > t_fail
    print(f"failure at {t_fail - gt[0, 0]:.1f} s; blind window has {tail.sum()} ground-truth poses "
          f"over {t[tail][-1] - t_fail:.2f} s" if tail.any() else "no ground truth in blind window")
    if not tail.any():
        return
    # Umeyama rigid alignment fitted on the pre-failure part only.
    mg, me = g[pre].mean(0), e[pre].mean(0)
    u, _, vt = np.linalg.svd((g[pre] - mg).T @ (e[pre] - me))
    s = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rot = u @ s @ vt
    aligned = (rot @ (e - me).T).T + mg
    err = np.linalg.norm(aligned[tail] - g[tail], axis=1)
    last = g[pre][-1]  # yardstick: hold the true last position
    hold = np.linalg.norm(g[tail] - last, axis=1)
    print(f"ours: blind RMSE {np.sqrt(np.mean(err**2)):.3f} m, endpoint {err[-1]:.3f} m")
    print(f"hold TRUE last position (not achievable, yardstick): RMSE {np.sqrt(np.mean(hold**2)):.3f} m, "
          f"endpoint {hold[-1]:.3f} m")


if __name__ == "__main__":
    main()
