"""Degradations that no setting has been chosen on, for a last robustness check. Each tests one way the organisers' script could
differ from our guess. Fixed before any result on them was seen.

hold     : the 'full' schedule, but dropouts repeat the last row (a frozen sensor) instead of zero rows or missing rows
rot      : the 'full' schedule after a random fixed rotation of the sensor axes ("unknown gravity alignment")
warmup   : a new drift shape. Exponential warm-up bias plus a slow random walk, noise growing, four gaps of 1.5 s
longgap  : the 'full' schedule with two 4 s gaps instead of three 2 s gaps
noisy    : the 'full' schedule with three times the noise
"""
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

KINDS = ("hold", "rot", "warmup", "longgap", "noisy")


def _full_like(room, seed, gaps, gap_s, noise=1.0, rot=None, mode="zero"):
    rnd = random.Random(seed + 900)
    rng = np.random.default_rng(seed + 900)
    t, imu = room["t_ns"], room["imu"].copy()
    if rot is not None:
        imu[:, :3] = imu[:, :3] @ rot.T
        imu[:, 3:] = imu[:, 3:] @ rot.T
    t0, t1 = int(t[0]), int(t[-1]) - 5 * 10 ** 9
    keep = t <= t1
    t, imu = t[keep], imu[keep]
    span = t1 - t0
    x = (t - t0) / span
    gd = np.array([rnd.uniform(-1, 1) for _ in range(3)])
    ad = np.array([rnd.uniform(-1, 1) for _ in range(3)])
    sg = (0.002 + 0.008 * x) * noise
    sa = (0.02 + 0.08 * x) * noise
    out = imu.copy()
    out[:, :3] += rng.standard_normal((len(t), 3)) * sg[:, None] + 0.02 * x[:, None] * gd
    out[:, 3:] += rng.standard_normal((len(t), 3)) * sa[:, None] + 0.2 * x[:, None] * ad
    out = np.round(out, 6)
    starts = sorted(t0 + int(span * (0.3 + 0.7 * (i + rnd.random() * 0.8) / gaps)) for i in range(gaps))
    g = np.zeros(len(t), bool)
    for s in starts:
        g |= (t >= s) & (t < s + int(gap_s * 1e9))
    if mode == "zero":
        out[g] = 0.0
    elif mode == "hold":
        for i in np.nonzero(g)[0]:
            out[i] = out[i - 1]
    return t, out


def degrade_robust(room, seed, kind):
    if kind == "hold":
        return _full_like(room, seed, 3, 2.0, mode="hold")
    if kind == "rot":
        rot = R.random(random_state=seed + 77).as_matrix()
        return _full_like(room, seed, 3, 2.0, rot=rot)
    if kind == "longgap":
        return _full_like(room, seed, 2, 4.0)
    if kind == "noisy":
        return _full_like(room, seed, 3, 2.0, noise=3.0)
    if kind == "warmup":
        rng = np.random.default_rng(seed + 1300)
        t, imu = room["t_ns"], room["imu"].copy()
        t1 = int(t[-1]) - 5 * 10 ** 9
        keep = t <= t1
        t, imu = t[keep], imu[keep]
        ts = (t - t[0]) * 1e-9
        n, T = len(t), ts[-1]
        tau = rng.uniform(20, 60)
        wg = (1 - np.exp(-ts / tau))[:, None] * rng.uniform(-0.02, 0.02, 3)
        wa = (1 - np.exp(-ts / tau))[:, None] * rng.uniform(-0.2, 0.2, 3)
        rw_g = np.cumsum(rng.standard_normal((n, 3)), 0) * 0.004 / np.sqrt(n)
        rw_a = np.cumsum(rng.standard_normal((n, 3)), 0) * 0.04 / np.sqrt(n)
        x = (ts / T)[:, None]
        imu[:, :3] += wg + rw_g + rng.standard_normal((n, 3)) * (0.002 + 0.006 * x)
        imu[:, 3:] += wa + rw_a + rng.standard_normal((n, 3)) * (0.02 + 0.06 * x)
        imu = np.round(imu, 6)
        g = np.zeros(n, bool)
        for _ in range(4):
            s = rng.uniform(0.2 * T, T - 3)
            g |= (ts >= s) & (ts < s + 1.5)
        return t[~g], imu[~g]
    raise ValueError(kind)
