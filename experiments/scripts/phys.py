"""Attitude v2 (hardened rest search) and physics velocity channels, adapted from the robust-estimation thread's
experiments: rotor detector, drag complementary velocity, high-pass. Every parameter that was fitted there is fitted
here inside training folds. Nothing in this file reads ground truth."""
import numpy as np
from scipy.signal import butter, filtfilt

from pipeline import *
from attitude import *

K0 = 0.187           # scale only; folds refit a multiplier on it from training sequences
FB = 0.1             # Hz, crossover between drag (low) and integrated accel (high)


def _noise3(x):
    """Noise sigma per axis from the third difference (robust to smooth motion)."""
    d = np.diff(x, 3, axis=0)
    return np.median(np.abs(d), axis=0) / (0.6745 * np.sqrt(20.0))


def rest_blocks(t, imu, ok, win_s=0.5):
    """Scores each block by how white the gyro is. Fixed thresholds fail when degradation raises noise 5x, so the
    test is relative: std of gyro against the third-difference noise of the SAME block. Returns list of
    (centre_t, mean_gyro, ratio, gyro_std)."""
    fs = int(round(1.0 / np.median(np.diff(t))))
    win = int(win_s * fs)
    out = []
    an = np.linalg.norm(imu[ok, 3:], axis=1)
    g = np.median(an)
    for s in range(0, len(t) - win, win):
        if not ok[s:s + win].all():
            continue
        w, a = imu[s:s + win, :3], imu[s:s + win, 3:]
        n3 = _noise3(w)
        sd = w.std(0)
        ratio = float((sd / np.maximum(n3, 1e-4)).max())
        aw = a.std(0).max()
        aN3 = _noise3(a).max()
        if abs(np.linalg.norm(a.mean(0)) - g) < 0.2 and sd.max() < 0.03:
            out.append((float(t[s + win // 2]), w.mean(0), ratio, float(sd.max()), float(aw / max(aN3, 1e-4)), float(aN3)))
    return out


def hardened_rest(t, imu, ok, ratio_thr=1.8, acc_ratio=3.0, rotor_a3=0.3, min_blocks=2, abs_floor=0.006):
    """Blocks with gyro std near this block's own noise (or under an absolute floor), rotors off by the accel third
    difference, accel norm near g. The noise is read locally, so a 5x noise increase does not hide rest."""
    bl = [b for b in rest_blocks(t, imu, ok) if (b[2] < ratio_thr or b[3] < abs_floor) and b[4] < acc_ratio and b[5] < rotor_a3]
    if len(bl) < min_blocks:
        return None, None
    return np.array([b[0] for b in bl]), np.array([b[1] for b in bl])


def bias_from_rests(tb, mb):
    """Median in clusters: returns knots and values for piecewise linear bias(t) over the file. Clamped outside."""
    # group blocks that are close in time
    groups = [[0]]
    for i in range(1, len(tb)):
        if tb[i] - tb[groups[-1][-1]] < 5.0:
            groups[-1].append(i)
        else:
            groups.append([i])
    groups = [g for g in groups if len(g) >= 2] or groups
    tk = np.array([tb[g].mean() for g in groups])
    vk = np.array([np.median(mb[g], 0) for g in groups])
    return tk, vk


def attitude_v2(t, imu, ok, calib=None, drift_prior=0.003, mode="rest"):
    """Bias b0: hardened rest if found, else gravity for the observable directions, else zero.
    With calib (a training-set mean bias, a sensor calibration declared as such) that replaces zero as prior mean."""
    tb, mb = hardened_rest(t, imu, ok)
    drone = is_drone(t, imu, ok)
    info = {"rest": tb is not None, "drone": drone}
    im2 = imu.copy()
    b0 = np.asarray(calib) if (calib is not None and drone) else np.zeros(3)
    if tb is not None:
        tk, vk = bias_from_rests(tb, mb)
        if len(tk) >= 2 and mode == "rest_ramp":
            bt = np.stack([np.interp(t, tk, vk[:, i]) for i in range(3)], 1)
            im2[ok, :3] -= bt[ok]
            b0 = np.zeros(3)
        else:
            b0 = np.median(vk, 0)
    elif mode == "gravity":
        b0 = gravity_bias(t, imu, ok)
    q = offline_attitude(t, im2, ok, b0=b0, knot_s=60, s_r=0.3, sigma_b=drift_prior, residual_s=5.0)
    return q, info


# ------------------------------------------------------------------ velocity channels
def _lpf(x, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def _hpf(x, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2), btype="high")
    return filtfilt(b, a, x, axis=0)


def rotors_on(a_filled, fs, win_s=1.0, th=0.35):
    win = int(win_s * fs)
    n = len(a_filled)
    s = np.zeros(n)
    for i in range(0, n, win):
        seg = a_filled[i:i + win]
        s[i:i + win] = np.median(np.abs(np.diff(seg, 3, axis=0))) / (0.6745 * np.sqrt(20.0)) if len(seg) > 5 else 0.0
    return s > th


def is_drone(t, imu, ok):
    fs = round(1.0 / np.median(np.diff(t)))
    idx = np.arange(len(t))
    a = np.stack([np.interp(idx, idx[ok], imu[ok, 3 + i]) for i in range(3)], 1)
    last = np.nonzero(ok)[0][-1]
    return bool(rotors_on(a, fs)[: last + 1].mean() > 0.3)


def phys_channels(t, imu, ok, q):
    """All at row rate, in the estimate world frame:
    v_int: integrated gravity-free accel, mean removed, high-passed at 0.06 Hz (band filter's backbone).
    v_drag: in-plane velocity from rotor drag at 1/K0 scale, complementary with v_int above FB, zero with rotors off.
    on: rotor flag, per row. Tail and gaps: accel is held by interpolation, and marked by the mask channel."""
    fs = round(1.0 / np.median(np.diff(t)))
    n = len(t)
    idx = np.arange(n)
    a = np.stack([np.interp(idx, idx[ok], imu[ok, 3 + i]) for i in range(3)], 1)
    Rm = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    on = rotors_on(a, fs)
    an = np.linalg.norm(imu[ok, 3:], axis=1)
    g = np.median(an)
    aw = np.einsum("nij,nj->ni", Rm, a) - np.array([0, 0, g])
    last = np.nonzero(ok)[0][-1]
    aw[last + 1:] = 0
    ref = on if on.sum() > 100 else np.ones(n, bool)
    aw0 = aw - aw[: last + 1][ref[: last + 1]].mean(0)
    aw0[last + 1:] = 0
    vi = np.cumsum(aw0, axis=0) / fs
    v_int = _hpf(vi, 0.06, fs)
    frac_on = on[: last + 1].mean()
    v_drag = np.zeros((n, 3))
    v_comp = np.zeros((n, 3))
    if frac_on > 0.3:
        ta = a[: last + 1][on[: last + 1]].mean(0)
        ta /= np.linalg.norm(ta)
        e1 = np.cross(ta, [0, 1, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ta, e1)
        E = np.stack([e1, e2], 1)
        fp = _lpf(a, 1.0, fs) @ E
        off = fp[: last + 1][on[: last + 1]].mean(0)
        vd = -(fp - off) / K0
        vh = np.einsum("nji,nj->ni", Rm, _hpf(vi, FB, fs))
        vc = _lpf(vd, FB, fs) + vh @ E
        vb = (vc @ E.T) * on[:, None]
        v_drag = np.einsum("nij,nj->ni", Rm, vb)
        v_comp = v_drag
    return dict(v_int=v_int, v_drag=v_drag, on=on, last=last, drone=frac_on > 0.3)


def decay_tail(v, last, t, tau=1.0):
    out = v.copy()
    if last + 1 < len(v):
        dt = t[last + 1:] - t[last]
        out[last + 1:] = v[last] * np.exp(-dt / tau)[:, None]
    return out


def hp_position(v, t, hp, fs=None):
    """Integrate, then remove the slow part with a zero-phase filter. hp in Hz."""
    fs = round(1.0 / np.median(np.diff(t))) if fs is None else fs
    p = np.cumsum(v, axis=0) / fs
    return p - _lpf(p, hp, fs)


# band velocity filter, gains on 9 octave bands of velocity (Wiener style), zero phase
EDGES = np.array([0, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 1e9])


def _ev(x, dt):
    return np.fft.rfft(np.concatenate([x, x[::-1]], 0), axis=0), np.fft.rfftfreq(2 * len(x), dt)


def band_fit(pairs, dt):
    num, den = np.zeros(9), np.zeros(9)
    for vh, v in pairs:
        X, f = _ev(vh, dt)
        Y, _ = _ev(v, dt)
        b = np.digitize(f, EDGES) - 1
        num += np.bincount(b, (np.conj(X) * Y).real.sum(1), minlength=9)[:9]
        den += np.bincount(b, (np.abs(X) ** 2).sum(1), minlength=9)[:9]
    return num / np.maximum(den, 1e-12)


def band_apply(vh, g, dt):
    X, f = _ev(vh, dt)
    cen = np.sqrt(np.maximum(EDGES[:-1], 1 / 512) * np.minimum(EDGES[1:], 4))
    Gn = np.interp(np.log(np.maximum(f, 1e-6)), np.log(cen), g)
    return np.fft.irfft(X * Gn[:, None], n=2 * len(vh), axis=0)[: len(vh)]
