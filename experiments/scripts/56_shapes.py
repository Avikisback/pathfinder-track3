"""Drift shapes kept apart from the known schedules, for robustness checks."""
import numpy as np


def degrade_custom(room, seed, kind):
    """Drift shapes that no training pool or setting has seen."""
    rng = np.random.default_rng(seed + 100)
    t = room["t_ns"]
    t1 = int(t[-1]) - 5 * 10 ** 9
    keep = t <= t1
    t = t[keep]
    v = room["imu"][keep].copy()
    ts = (t - t[0]) * 1e-9
    n, T = len(t), ts[-1]
    fs = 1.0 / np.median(np.diff(ts))

    def gaps(num, length, lo=0.2):
        mask = np.zeros(n, bool)
        for _ in range(num):
            s = rng.uniform(lo * T, T - length - 1)
            mask |= (ts >= s) & (ts < s + length)
        return mask

    if kind == "randomwalk":
        bg = np.cumsum(rng.standard_normal((n, 3)), 0)
        bg = bg - bg.mean(0)
        bg *= 0.012 / bg.std(0)
        ba = np.cumsum(rng.standard_normal((n, 3)), 0)
        ba = ba - ba.mean(0)
        ba *= 0.12 / ba.std(0)
        sg, sa = np.full(n, 0.006), np.full(n, 0.06)
        g, mode = gaps(3, 2.0), "delete"
    elif kind == "sine":
        ph = rng.uniform(0, 6.28, 6)
        per = rng.uniform(40, 120, 6)
        bg = 0.02 * np.stack([np.sin(2 * np.pi * ts / per[i] + ph[i]) for i in range(3)], 1)
        ba = 0.2 * np.stack([np.sin(2 * np.pi * ts / per[i + 3] + ph[i + 3]) for i in range(3)], 1)
        sg, sa = np.full(n, 0.004), np.full(n, 0.04)
        g, mode = gaps(3, 2.0), "zero"
    elif kind == "burst":
        x = (ts / T)[:, None]
        bg = 0.01 * x * rng.uniform(-1, 1, 3)
        ba = 0.1 * x * rng.uniform(-1, 1, 3)
        sg, sa = np.full(n, 0.004), np.full(n, 0.04)
        for _ in range(4):
            s = rng.uniform(0.1 * T, T - 4)
            m = (ts >= s) & (ts < s + 3)
            sg[m], sa[m] = 0.04, 0.4
        g = gaps(6, 0.5) | gaps(1, 2.0)
        mode = "zero"
    else:
        raise ValueError(kind)
    v[:, :3] += bg + rng.standard_normal((n, 3)) * sg[:, None]
    v[:, 3:] += ba + rng.standard_normal((n, 3)) * sa[:, None]
    if mode == "delete":
        return t[~g], v[~g]
    v[g] = 0.0
    return t, v


