"""Features, randomised degradation for training, the velocity net, and turning its output into positions."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

DEC = 4          # 200 Hz rows averaged into 50 Hz feature rows
STRIDE = 5       # one velocity output per 5 feature rows, so 10 Hz
FS = 50.0
RES = EXP + "/results"
WGT = EXP + "/weights"
os.makedirs(RES, exist_ok=True)
os.makedirs(WGT, exist_ok=True)


# ------------------------------------------------------------------ labels (ground truth, training and scoring only)
def gt_labels(room, h=0.05):
    t = room["t_ns"] * 1e-9
    gt_t = room["gt_t_ns"] * 1e-9
    p = room["gt_p"]
    rot = R.from_quat(room["gt_q"])
    tc = np.clip(t, gt_t[0], gt_t[-1])
    q = Slerp(gt_t, rot)(tc).as_quat()
    interp = lambda tt: np.stack([np.interp(tt, gt_t, p[:, i]) for i in range(3)], 1)
    v = (interp(t + h) - interp(t - h)) / (2 * h)
    yaw = 2 * np.arctan2(q[:, 2], q[:, 3])
    # a label is good only inside the mocap span and away from mocap holes
    ok = (t - h >= gt_t[0]) & (t + h <= gt_t[-1])
    idx = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1)
    ok &= (gt_t[idx] - gt_t[idx - 1]) < 0.1
    return dict(t=t, v=v, yaw=yaw, ok=ok, q=q)


# ------------------------------------------------------------------ randomised degradation for training
def random_degrade(t, imu, rng):
    """Deliberately broader than tools/degrade.py, since the real schedule is unknown."""
    n = len(t)
    x = (t - t[0]) / (t[-1] - t[0])
    sev = 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 1.3)
    shape = x ** rng.uniform(0.5, 2.0)
    sg = 0.002 * rng.uniform(0.2, 1.5) + 0.008 * sev * x
    sa = 0.02 * rng.uniform(0.2, 1.5) + 0.08 * sev * x
    out = imu.copy()
    out[:, :3] += rng.standard_normal((n, 3)) * sg[:, None] + (sev * 0.02 * shape)[:, None] * rng.uniform(-1, 1, 3) \
        + rng.normal(0, 0.005, 3)
    out[:, 3:] += rng.standard_normal((n, 3)) * sa[:, None] + (sev * 0.2 * shape)[:, None] * rng.uniform(-1, 1, 3) \
        + rng.normal(0, 0.1, 3)
    valid = np.ones(n, dtype=bool)
    for _ in range(rng.integers(0, 6)):
        s = rng.uniform(t[0], t[-1])
        valid[(t >= s) & (t < s + rng.uniform(0.2, 3.0))] = False
    if rng.random() < 0.5:
        valid[t > t[-1] - 5.0] = False
    out[~valid] = 0.0
    return out, valid


# ------------------------------------------------------------------ features
def features(t, imu, valid, q_wxyz):
    """50 Hz channels in the gravity-aligned estimate frame: linear accel 3, rate 3, up-in-body 3, mask 1."""
    n = len(t) // DEC
    m = valid[: n * DEC].reshape(n, DEC, 1).astype(np.float64)
    cnt = m.sum(1)
    w = (imu[: n * DEC, :3].reshape(n, DEC, 3) * m).sum(1) / np.maximum(cnt, 1)
    a = (imu[: n * DEC, 3:].reshape(n, DEC, 3) * m).sum(1) / np.maximum(cnt, 1)
    q = q_wxyz[DEC // 2:: DEC][:n]
    rot = R.from_quat(q[:, [1, 2, 3, 0]])
    Rm = rot.as_matrix()
    an = np.linalg.norm(imu[valid, 3:], axis=1)
    g = np.median(an) if len(an) else 9.81
    aw = np.einsum("nij,nj->ni", Rm, a) - np.array([0, 0, g])
    ww = np.einsum("nij,nj->ni", Rm, w)
    up = Rm[:, 2, :]
    good = (cnt > 0)
    aw, ww = aw * good, ww * good
    X = np.concatenate([aw / 4.0, ww / 2.0, up, cnt / DEC], 1).astype(np.float32)
    yaw = 2 * np.arctan2(q[:, 3], q[:, 0])
    tc = t[DEC // 2:: DEC][:n]
    return X, yaw.astype(np.float32), tc


def pool_item(room, lab, rng, attitude=None):
    t = lab["t"]
    imu, valid = random_degrade(t, room["imu"], rng)
    if attitude == "auto":
        from attitude import attitude_fast
        q = attitude_fast(t, imu, valid)
    else:
        q = baseline_attitude(t, imu, valid)
    X, yawh, tc = features(t, imu, valid, q)
    n = len(X)
    idx = DEC // 2 + np.arange(n) * DEC
    return dict(X=X, yawh=yawh, v=lab["v"][idx].astype(np.float32), yaw=lab["yaw"][idx].astype(np.float32),
                ok=lab["ok"][idx])


_POOLS = {}


def pool_cached(seqs, rng, per):
    items = []
    for n in seqs:
        if n not in _POOLS:
            d = np.load(f"{CACHE}/pool_{n}.npz")
            _POOLS[n] = {k: d[k] for k in d.files}
        d = _POOLS[n]
        for i in rng.integers(0, len(d["X"]), per):
            items.append(dict(X=d["X"][i], yawh=d["yawh"][i], v=d["v"], yaw=d["yaw"], ok=d["ok"]))
    return items


class Pool:
    """Concatenates sequences on the GPU, padded so windows at the ends see mask=0."""

    def __init__(self, items, W, device):
        pad = W + 2
        Xs, yh, vs, ys, oks = [], [], [], [], []
        for it in items:
            n = len(it["X"])
            Xs += [np.zeros((pad, 10), np.float32), it["X"]]
            yh += [np.zeros(pad, np.float32), it["yawh"]]
            vs += [np.zeros((pad, 3), np.float32), it["v"]]
            ys += [np.zeros(pad, np.float32), it["yaw"]]
            oks += [np.zeros(pad, bool), it["ok"]]
        Xs.append(np.zeros((pad, 10), np.float32)); yh.append(np.zeros(pad, np.float32))
        vs.append(np.zeros((pad, 3), np.float32)); ys.append(np.zeros(pad, np.float32)); oks.append(np.zeros(pad, bool))
        self.X = torch.from_numpy(np.concatenate(Xs)).to(device)
        self.yawh = torch.from_numpy(np.concatenate(yh)).to(device)
        self.v = torch.from_numpy(np.concatenate(vs)).to(device)
        self.yaw = torch.from_numpy(np.concatenate(ys)).to(device)
        self.good = torch.from_numpy(np.nonzero(np.concatenate(oks))[0]).to(device)
        self.device = device


def rot_xy(x, c, s):
    """Rotates the xy part of the last dim by the angle with cos c, sin s."""
    return torch.stack([c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1], x[..., 2]], -1)


def window_batch(X, yawh, k, offs):
    """k: LongTensor of centre rows. Returns (B, C, T), rotated to the heading at k."""
    win = X[k[:, None] + offs]
    c, s = torch.cos(-yawh[k])[:, None], torch.sin(-yawh[k])[:, None]
    aw = rot_xy(win[..., :3], c, s)
    ww = rot_xy(win[..., 3:6], c, s)
    return torch.cat([aw, ww, win[..., 6:]], -1).permute(0, 2, 1)


# ------------------------------------------------------------------ net
class Block(nn.Module):
    def __init__(self, a, b, k, drop):
        super().__init__()
        self.down = nn.Conv1d(a, b, k, stride=2, padding=k // 2)
        self.res = nn.Conv1d(b, b, 3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.down(x))
        return self.drop(x + self.act(self.res(x)))


class VelNet(nn.Module):
    """Small 1D conv net: window of 50 Hz features in, 3 velocities and 3 log variances out."""

    def __init__(self, T=100, ch=(24, 32, 48, 64), k=5, drop=0.05, c_in=10, out=6, fc=64):
        super().__init__()
        self.cfg = dict(T=T, ch=list(ch), k=k, drop=drop, c_in=c_in, out=out, fc=fc)
        layers, a = [], c_in
        L = T
        for b in ch:
            layers.append(Block(a, b, k, drop))
            a = b
            L = (L + 1) // 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(a * L, fc), nn.GELU(), nn.Linear(fc, out))

    def forward(self, x):
        return self.head(self.body(x))


def macs(cfg):
    """Multiply-adds per forward pass."""
    T, a, tot = cfg["T"], cfg["c_in"], 0
    L = T
    for b in cfg["ch"]:
        L = (L + 1) // 2
        tot += a * b * cfg["k"] * L + b * b * 3 * L
        a = b
    tot += a * L * cfg["fc"] + cfg["fc"] * cfg["out"]
    return tot


def offsets(T, center, device):
    if center:
        o = np.arange(T) - T // 2
    else:
        o = np.arange(T) - (T - 1)
    return torch.from_numpy(o).long().to(device)


def train_fold(train_rooms, cfg, seed, device="cuda", log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if not cfg.get("cached"):
        rooms = {n: load_room(n) for n in train_rooms}
        labs = {n: gt_labels(rooms[n]) for n in train_rooms}
    T = cfg["T"]
    net = VelNet(T=T, ch=cfg.get("ch", (24, 32, 48, 64)), drop=cfg.get("drop", 0.05)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 2e-3), weight_decay=cfg.get("wd", 1e-4))
    steps = cfg.get("steps", 4000)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 2e-3), total_steps=steps, pct_start=0.1)
    offs = offsets(T, cfg["center"], device)
    bs = cfg.get("bs", 256)
    seg = cfg.get("seg", 0)
    pool = None
    for step in range(steps):
        if step % cfg.get("refresh", 250) == 0:
            if cfg.get("cached"):
                items = pool_cached(train_rooms, rng, cfg.get("pool_per_room", 2))
            else:
                items = [pool_item(rooms[n], labs[n], rng, cfg.get("attitude")) for n in train_rooms for _ in range(cfg.get("pool_per_room", 4))]
            pool = Pool(items, T, device)
        net.train()
        if seg:
            nb = max(1, bs // seg)
            start = pool.good[torch.randint(len(pool.good), (nb,), device=device)]
            kk = start[:, None] + torch.arange(seg, device=device)[None] * STRIDE
            k = kk.flatten().clamp(max=len(pool.X) - T - 3)
        else:
            k = pool.good[torch.randint(len(pool.good), (bs,), device=device)]
        x = window_batch(pool.X, pool.yawh, k, offs)
        c, s = torch.cos(-pool.yaw[k])[:, None], torch.sin(-pool.yaw[k])[:, None]
        y = rot_xy(pool.v[k], c[:, 0], s[:, 0])
        if cfg.get("yaw_aug"):
            # random heading for input and label together, so body-forward cannot be relied on
            ang = torch.rand(len(k), device=device) * 6.2831853
            ca, sa = torch.cos(ang), torch.sin(ang)
            y = rot_xy(y, ca, sa)
            for i in (0, 3):
                x0, x1 = x[:, i], x[:, i + 1]
                x = torch.cat([x[:, :i], (ca[:, None] * x0 - sa[:, None] * x1)[:, None], (sa[:, None] * x0 + ca[:, None] * x1)[:, None], x[:, i + 2:]], 1)
        if cfg.get("no_up"):
            x = torch.cat([x[:, :6], torch.zeros_like(x[:, 6:9]), x[:, 9:]], 1)
        out = net(x)
        mu, lv = out[:, :3], out[:, 3:].clamp(-7, 3)
        err = (mu - y)
        mse = (err ** 2).mean()
        nll = 0.5 * (lv + err.detach() ** 2 * torch.exp(-lv)).mean()
        loss = mse + 0.1 * nll
        if seg:
            # low-frequency term: error of displacement summed over the segment, in the true world frame
            cw, sw = torch.cos(pool.yaw[k]), torch.sin(pool.yaw[k])
            ew = rot_xy(err, cw, sw).reshape(nb, seg, 3)
            d = torch.cumsum(ew, 1) * 0.1
            hor = torch.arange(1, seg + 1, device=device, dtype=torch.float32)[None, :, None] * 0.1
            loss = loss + cfg.get("seg_w", 1.0) * ((d / hor) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if log and step % 1000 == 0:
            print(f"{log} step {step} mse {mse.item():.4f}", flush=True)
    net.eval()
    return net


# ------------------------------------------------------------------ evaluation files
def eval_file(n, sched_name, mode, seed):
    os.makedirs(CACHE, exist_ok=True)
    c = f"{CACHE}/eval_r{n}_{sched_name}_{mode}_{seed}.npz"
    if os.path.exists(c):
        d = np.load(c)
        return d["t"], d["imu"], d["gaps"]
    t, imu, gaps = degrade(load_room(n), seed, mode, sched_name)
    np.savez(c, t=t, imu=imu, gaps=gaps)
    return t, imu, gaps


def eval_set(rooms, schedules=("full", "mild"), seeds=(1, 2, 3), modes=("delete", "zero")):
    out = [(n, s, m, sd) for n in rooms for s in schedules for m in modes for sd in seeds]
    out += [(n, "clean", "delete", 1) for n in rooms]
    return out


def prepare(t_ns, imu, blind_s=5.0):
    """What the estimator knows: rows only. Appends the blind tail as invalid rows."""
    tg, vg, ok, ids = to_grid(t_ns, imu)
    last_good = np.nonzero(ok)[0][-1]
    med = int(np.median(np.diff(t_ns)))
    n_tail = int(round(blind_s * 1e9 / med))
    # if the file already ends in dead rows they count towards the blind time
    already = len(tg) - 1 - last_good
    n_add = max(0, n_tail - already)
    tail_t = tg[-1] + med * (1 + np.arange(n_add))
    t_all = np.concatenate([tg, tail_t])
    v_all = np.vstack([vg, np.zeros((n_add, 6))])
    ok_all = np.concatenate([ok, np.zeros(n_add, bool)])
    ids_all = np.concatenate([ids, -2 * np.ones(n_add, dtype=int)])
    t_s = (t_all - t_all[0]) * 1e-9
    return dict(t_ns=t_all, t=t_s, imu=v_all, ok=ok_all, ids=ids_all, last_good=last_good, t_fail_ns=tg[last_good])


@torch.no_grad()
def infer(net, P, q=None, device="cuda", batch=1024):
    """Whole sequence, 10 Hz outputs. Returns output times, world velocity, per-axis variance, headings."""
    if q is None:
        q = baseline_attitude(P["t"], P["imu"], P["ok"])
    X, yawh, tc = features(P["t"], P["imu"], P["ok"], q)
    T = net.cfg["T"]
    Xp = torch.from_numpy(np.concatenate([np.zeros((T + 2, 10), np.float32), X, np.zeros((T + 2, 10), np.float32)])).to(device)
    yp = torch.from_numpy(np.concatenate([np.zeros(T + 2, np.float32), yawh, np.zeros(T + 2, np.float32)])).to(device)
    ks = np.arange(0, len(X), STRIDE)
    kt = torch.from_numpy(ks + T + 2).long().to(device)
    center = net.center if hasattr(net, "center") else True
    offs = offsets(T, center, device)
    outs = []
    for i in range(0, len(kt), batch):
        xb = window_batch(Xp, yp, kt[i: i + batch], offs)
        if getattr(net, "no_up", False):
            xb = torch.cat([xb[:, :6], torch.zeros_like(xb[:, 6:9]), xb[:, 9:]], 1)
        outs.append(net(xb))
    out = torch.cat(outs).cpu().numpy().astype(np.float64)
    yh = yawh[ks].astype(np.float64)
    c, s = np.cos(yh), np.sin(yh)
    vw = np.stack([c * out[:, 0] - s * out[:, 1], s * out[:, 0] + c * out[:, 1], out[:, 2]], 1)
    return dict(tk=tc[ks], v=vw, var=np.exp(out[:, 3:].clip(-7, 3)), q=q)


# ------------------------------------------------------------------ velocity to position
def integrate(v, dt=0.1):
    return np.vstack([np.zeros((1, 3)), np.cumsum(v[:-1] * dt, axis=0)])


def leak_causal(v, tau, dt=0.1):
    p = np.zeros_like(v)
    for k in range(1, len(v)):
        p[k] = p[k - 1] + dt * (v[k - 1] - p[k - 1] / tau)
    return p


def smooth_prior(v, var, tau0, alpha=0.0, dt=0.1, tau_z=None):
    """Two-sided shrinkage. Minimises sum (x[k+1]-x[k]-v[k]dt)^2/q[k] + sum x[k]^2 dt/r.

    Solved exactly as a tridiagonal system per axis. tau is the time scale at which integrated velocity is
    trusted: tau[k]^2 = r/q_rate[k]. alpha=0 gives one fixed tau, a zero-phase high-pass. alpha>0 pulls harder
    where the net says it is unsure. The solution is automatically centred, no centroid has to be chosen.
    """
    n = len(v)
    out = np.zeros_like(v)
    for ax in range(3):
        t0 = tau_z if (ax == 2 and tau_z is not None) else tau0
        s = var[:, ax] / np.median(var[:, ax]) if alpha > 0 else np.ones(n)
        w_d = 1.0 / (dt * s[:-1] ** alpha * dt)     # 1/q in units where r=t0^2 q_rate_ref
        w_p = dt / (t0 ** 2) * np.ones(n) * (1.0 / dt)
        ab = np.zeros((3, n))
        ab[1, :] = w_p
        ab[1, :-1] += w_d
        ab[1, 1:] += w_d
        ab[0, 1:] = -w_d
        ab[2, :-1] = -w_d
        b = np.zeros(n)
        d = v[:-1, ax] * dt * w_d
        b[:-1] -= d
        b[1:] += d
        out[:, ax] = solve_banded((1, 1), ab, b)
    return out


def to_rows(P, tk, x):
    return np.stack([np.interp(P["t"], tk, x[:, i]) for i in range(3)], 1)


def finish(P, p_grid, q_wxyz):
    """Keeps input rows and the blind tail, drops rows that the estimator inserted inside deleted gaps."""
    keep = P["ids"] != -1
    return P["t_ns"][keep], p_grid[keep], q_wxyz[keep][:, [1, 2, 3, 0]]


def score_all(P, room, p_grid, q_wxyz):
    et, ep, eq = finish(P, p_grid, q_wxyz)
    gt_t = room["gt_t_ns"] * 1e-9
    d = score(et * 1e-9, ep, eq, gt_t, room["gt_p"], room["gt_q"])
    d.update(score_tail(et * 1e-9, ep, gt_t, room["gt_p"], P["t_fail_ns"] * 1e-9))
    # same scores with the blind tail left out, so that 5 s of forecast does not mask the rest of the file
    pre = et <= P["t_fail_ns"]
    d2 = score(et[pre] * 1e-9, ep[pre], eq[pre], gt_t, room["gt_p"], room["gt_q"])
    d.update({k + "_pre": d2[k] for k in ("ate_se3", "ate_org", "att_org", "tilt", "rpe_t", "rpe_r")})
    return d


class PosScorer:
    """Precomputes association and rotations for one file, then scores many position tracks in about a millisecond."""

    def __init__(self, P, room, q_wxyz):
        keep = P["ids"] != -1
        et = P["t_ns"][keep] * 1e-9
        gt_t = room["gt_t_ns"] * 1e-9
        self.keep = keep
        self.gi, self.ei = associate(gt_t, et)
        self.g = room["gt_p"][self.gi]
        self.t = gt_t[self.gi]
        q = q_wxyz[keep][:, [1, 2, 3, 0]][self.ei]
        Rg, Re = R.from_quat(room["gt_q"][self.gi]), R.from_quat(q)
        self.to = (Rg[0] * Re[0].inv()).as_matrix()
        d = int(round(1.0 / np.median(np.diff(gt_t))))
        self.i = np.arange(0, len(self.g) - d, d)
        self.j = self.i + d
        self.Rg_i = Rg[self.i].as_matrix()
        self.Re_i = Re[self.i].as_matrix()
        self.Qrel = (Rg[self.i].inv() * Rg[self.j]).as_matrix()
        self.t_fail = P["t_fail_ns"] * 1e-9
        self.pre = self.t <= self.t_fail
        self.tail = ~self.pre

    def __call__(self, p_grid):
        e = p_grid[self.keep][self.ei]
        g = self.g
        if np.ptp(e, axis=0).max() < 1e-9:
            a = float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))
        else:
            r, tr = umeyama(e, g)
            a = float(np.sqrt((((r @ e.T).T + tr - g) ** 2).sum(1).mean()))
        eo = (self.to @ (e - e[0]).T).T + g[0]
        o = float(np.sqrt(((eo - g) ** 2).sum(1).mean()))
        qt = np.einsum("nji,nj->ni", self.Rg_i, g[self.j] - g[self.i])
        pt = np.einsum("nji,nj->ni", self.Re_i, e[self.j] - e[self.i])
        et = np.einsum("nji,nj->ni", self.Qrel, pt - qt)
        rp = float(np.sqrt((et ** 2).sum(1).mean()))
        out = dict(ate_se3=a, ate_org=o, rpe_t=rp)
        if self.tail.sum() >= 10 and self.pre.sum() >= 10:
            ep, gp = e[self.pre], g[self.pre]
            if np.ptp(ep, axis=0).max() < 1e-9:
                r2, tr2 = np.eye(3), gp.mean(0) - ep.mean(0)
            else:
                r2, tr2 = umeyama(ep, gp)
            al = (r2 @ e.T).T + tr2
            k = np.nonzero(self.pre)[0][-1]
            dd = np.sqrt((((al[self.tail] - al[k]) - (g[self.tail] - g[k])) ** 2).sum(1))
            out["blind_rel"] = float(np.sqrt((dd ** 2).mean()))
            out["ate_se3_pre"] = float(np.sqrt(((al[self.pre] - gp) ** 2).sum(1).mean()))
        return out


# ------------------------------------------------------------------ reference numbers
def references(files, path):
    """Stationary and the real baseline binary, on every file. Cached."""
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for (n, s, m, sd) in files:
        key = f"{n}|{s}|{m}|{sd}"
        if key in cache:
            continue
        room = load_room(n)
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        q = baseline_attitude(P["t"], P["imu"], P["ok"])
        st = score_all(P, room, np.zeros((len(P["t"]), 3)), q)
        et, ep, eq = run_baseline(t, imu)
        gt_t = room["gt_t_ns"] * 1e-9
        b = score(et * 1e-9, ep, eq, gt_t, room["gt_p"], room["gt_q"])
        b.update(score_tail(et * 1e-9, ep, gt_t, room["gt_p"], P["t_fail_ns"] * 1e-9))
        cache[key] = dict(stationary=st, baseline=b)
        json.dump(cache, open(path, "w"))
    return cache


def summarise(rows):
    """rows: list of (key, dict). Returns per schedule mean and worst for each metric."""
    out = {}
    for s in ("clean", "mild", "full"):
        sel = [d for k, d in rows if k.split("|")[1] == s]
        if not sel:
            continue
        o = {}
        for m in sel[0]:
            v = np.array([d[m] for d in sel if d.get(m) is not None], dtype=float)
            o[m] = (float(np.nanmean(v)), float(np.nanmax(v)))
        out[s] = o
    return out


def show(name, summ, keys=("ate_se3", "ate_org", "rpe_t", "att_org", "blind_rel")):
    for s, o in summ.items():
        txt = " ".join(f"{k} {o[k][0]:.3f}/{o[k][1]:.3f}" for k in keys if k in o)
        print(f"  {name:34s} {s:5s} {txt}", flush=True)
