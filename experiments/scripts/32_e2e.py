"""Coordinator items 1 and 3: physics baselines and a residual velocity net, all parameters re-tuned inside folds.

Methods on every held-out file:
  physics   drag complementary velocity (drones) or high-passed integrated accel (handheld), then a position high-pass.
            k scale and high-pass come from training sequences only.
  band      9 velocity band gains on the same velocity, fitted on randomised degraded training realisations.
  net       velocity = a*v_int + b*v_drag + small net(window), a net fed gravity direction, rotor flag and drag channel.
            Two-sided shrink tau chosen from inner folds that never see the test sequence.

Usage: 32_e2e.py make_pool | train --scheme loeo [--no_up 0] [--inner 1]
"""
import argparse, glob, json, multiprocessing as mp, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *

SEQS = list(EUROC)
NR = 8
C_IN = 17
ROT_CH = (0, 3, 10, 13)   # channel groups that are world vectors and turn with heading


def calib_for(seqs):
    """Sensor calibration: mean ground-truth bias over the given EuRoC training sequences."""
    use = [x for x in seqs if isinstance(x, str)]
    return np.mean([np.median(load_room(x)["gt_bw"], 0) for x in use], 0) if use else None


def build_X(t, imu, ok, q):
    X, yaw, tc = features(t, imu, ok, q)
    ph = phys_channels(t, imu, ok, q)
    n = len(X)
    idx = DEC // 2 + np.arange(n) * DEC
    extra = np.concatenate([ph["v_int"][idx] / 2.0, ph["v_drag"][idx] / 2.0, ph["on"][idx, None].astype(float)], 1).astype(np.float32)
    return np.concatenate([X, extra], 1), yaw, tc, ph


def pool_one(n):
    f = f"{CACHE}/pool2_{n}.npz"
    if os.path.exists(f):
        return n
    room = load_room(n)
    lab = gt_labels(room)
    t = lab["t"] - lab["t"][0]
    rng = np.random.default_rng(5000 + sum(ord(c) for c in str(n)))
    Xs, ys = [], []
    for _ in range(NR):
        imu, ok = random_degrade(t, room["imu"], rng)
        q, _i = attitude_v2(t, imu, ok, calib=calib_for([x for x in SEQS if x != n]))
        X, yaw, tc, ph = build_X(t, imu, ok, q)
        Xs.append(X)
        ys.append(yaw)
    k = len(Xs[0])
    idx = DEC // 2 + np.arange(k) * DEC
    np.savez(f, X=np.stack(Xs), yawh=np.stack(ys), v=lab["v"][idx].astype(np.float32), yaw=lab["yaw"][idx].astype(np.float32), ok=lab["ok"][idx])
    return n


_POOL = {}


def get_pool(n):
    if n not in _POOL:
        d = np.load(f"{CACHE}/pool2_{n}.npz")
        _POOL[n] = {k: d[k] for k in d.files}
    return _POOL[n]


def batch_tensors(seqs, rng, per, T, device):
    pad = T + 2
    Xs, yh, vs, ys, oks = [], [], [], [], []
    for n in seqs:
        d = get_pool(n)
        for i in rng.integers(0, len(d["X"]), per):
            Xs += [np.zeros((pad, C_IN), np.float32), d["X"][i]]
            yh += [np.zeros(pad, np.float32), d["yawh"][i]]
            vs += [np.zeros((pad, 3), np.float32), d["v"]]
            ys += [np.zeros(pad, np.float32), d["yaw"]]
            oks += [np.zeros(pad, bool), d["ok"]]
    return (torch.from_numpy(np.concatenate(Xs + [np.zeros((pad, C_IN), np.float32)])).to(device),
            torch.from_numpy(np.concatenate(yh + [np.zeros(pad, np.float32)])).to(device),
            torch.from_numpy(np.concatenate(vs + [np.zeros((pad, 3), np.float32)])).to(device),
            torch.from_numpy(np.concatenate(ys + [np.zeros(pad, np.float32)])).to(device),
            torch.from_numpy(np.nonzero(np.concatenate(oks + [np.zeros(pad, bool)]))[0]).to(device))


def windows(X, yawh, k, offs):
    win = X[k[:, None] + offs]
    c, s = torch.cos(-yawh[k])[:, None], torch.sin(-yawh[k])[:, None]
    parts = []
    for g in ROT_CH[:2]:
        parts.append(rot_xy(win[..., g:g + 3], c, s))
    parts.append(win[..., 6:10])
    for g in ROT_CH[2:]:
        parts.append(rot_xy(win[..., g:g + 3], c, s))
    parts.append(win[..., 16:17])
    return torch.cat(parts, -1).permute(0, 2, 1)


class ResNet(nn.Module):
    def __init__(self, T=100, ch=(24, 32, 48, 64), drop=0.05):
        super().__init__()
        self.net = VelNet(T=T, ch=ch, drop=drop, c_in=C_IN, out=6)
        nn.init.zeros_(self.net.head[-1].weight)
        nn.init.zeros_(self.net.head[-1].bias)
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.T = T

    def forward(self, x):
        o = self.net(x)
        mid = self.T // 2
        base = 2.0 * (self.a * x[:, 10:13, mid] + self.b * x[:, 13:16, mid])
        return torch.cat([o[:, :3] + base, o[:, 3:]], 1)


def train_net(train_seqs, seed, no_up=False, steps=3000, T=100, device="cuda", yaw_aug=False, per=2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ResNet(T=T).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    offs = offsets(T, True, device)
    for step in range(steps):
        if step % 250 == 0:
            X, YH, V, Y, G = batch_tensors(train_seqs, rng, per, T, device)
        net.train()
        k = G[torch.randint(len(G), (256,), device=device)]
        x = windows(X, YH, k, offs)
        y = rot_xy(V[k], torch.cos(-Y[k]), torch.sin(-Y[k]))
        if yaw_aug:
            ang = torch.rand(len(k), device=device) * 6.2831853
            ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
            y = rot_xy(y, ca[:, 0], sa[:, 0])
            for g in ROT_CH:
                a0, a1 = x[:, g].clone(), x[:, g + 1].clone()
                x[:, g], x[:, g + 1] = ca * a0 - sa * a1, sa * a0 + ca * a1
        if no_up:
            x[:, 6:9] = 0
        out = net(x)
        err = out[:, :3] - y
        lv = out[:, 3:].clamp(-7, 3)
        loss = (err ** 2).mean() + 0.1 * (0.5 * (lv + err.detach() ** 2 * torch.exp(-lv))).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
    net.eval()
    net.no_up = no_up
    return net


@torch.no_grad()
def infer17(net, X, yawh, tc, device="cuda"):
    T = net.T
    pad = np.zeros((T + 2, C_IN), np.float32)
    Xp = torch.from_numpy(np.concatenate([pad, X, pad])).to(device)
    yp = torch.from_numpy(np.concatenate([np.zeros(T + 2, np.float32), yawh, np.zeros(T + 2, np.float32)])).to(device)
    ks = np.arange(0, len(X), STRIDE)
    kt = torch.from_numpy(ks + T + 2).long().to(device)
    offs = offsets(T, True, device)
    outs = []
    for i in range(0, len(kt), 1024):
        xb = windows(Xp, yp, kt[i:i + 1024], offs)
        if net.no_up:
            xb[:, 6:9] = 0
        outs.append(net(xb))
    o = torch.cat(outs).cpu().numpy().astype(np.float64)
    yh = yawh[ks].astype(np.float64)
    c, s = np.cos(yh), np.sin(yh)
    vw = np.stack([c * o[:, 0] - s * o[:, 1], s * o[:, 0] + c * o[:, 1], o[:, 2]], 1)
    return tc[ks], vw, np.exp(o[:, 3:].clip(-7, 3))


_F = {}
CALIB = [None]     # set per fold, from training sequences only


def file_ctx(n, s, m, sd):
    key = (n, s, m, sd, None if CALIB[0] is None else tuple(CALIB[0]))
    if key not in _F:
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        q, info = attitude_v2(P["t"], P["imu"], P["ok"], calib=CALIB[0])
        X, yaw, tc, ph = build_X(P["t"], P["imu"], P["ok"], q)
        room = load_room(n)
        _F[key] = dict(P=P, q=q, X=X, yaw=yaw, tc=tc, ph=ph, room=room, sc=PosScorer(P, room, q), rest=info["rest"])
    return _F[key]


def src_velocity(c):
    return c["ph"]["v_drag"] if c["ph"]["drone"] else c["ph"]["v_int"]


def yaw_gauge(vh, v):
    cov = vh[:, :2].T @ v[:, :2]
    th = np.arctan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
    return rz_mat(th)


def rz_mat(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fit_physics(train_seqs):
    """Per training sequence on realisations: velocity scale (k fold) and band gains, each by platform class."""
    pairs_d, pairs_h = [], []
    for n in train_seqs:
        d = get_pool(n)
        lab_ok = d["ok"]
        for i in range(min(4, len(d["X"]))):
            X = d["X"][i]
            drone = X[:, 16].mean() > 0.3
            src = X[:, 13:16] * 2.0 if drone else X[:, 10:13] * 2.0
            v = np.nan_to_num(d["v"].astype(np.float64))
            m = lab_ok
            Rg = yaw_gauge(src[m], v[m])
            vs = (Rg @ src.T).T.astype(np.float64)
            v2 = v.copy()
            v2[~m] = vs[~m]        # no information where mocap is missing
            (pairs_d if drone else pairs_h).append((vs, v2))
    out = {}
    for nm, pairs in (("drone", pairs_d), ("hand", pairs_h)):
        if pairs:
            out[nm] = dict(gains=band_fit(pairs, 1 / FS).tolist(),
                           scale=float(sum((a * b).sum() for a, b in pairs) / sum((a * a).sum() for a, b in pairs)))
    return out


def phys_track(c, cal, hp):
    P, ph = c["P"], c["ph"]
    nm = "drone" if ph["drone"] else "hand"
    if nm not in cal:         # the platform was never seen in this fold's training: scale 1
        s = 1.0
    else:
        s = cal[nm]["scale"]
    v = decay_tail(src_velocity(c) * s, ph["last"], P["t"])
    return hp_position(v, P["t"], hp)


def band_track(c, cal):
    P, ph = c["P"], c["ph"]
    nm = "drone" if ph["drone"] else "hand"
    v0 = src_velocity(c)
    n = len(v0)
    idx = np.arange(0, n, DEC)
    if nm in cal:
        vb = band_apply(v0[idx], np.array(cal[nm]["gains"]), DEC / round(1 / np.median(np.diff(P["t"]))))
    else:
        vb = v0[idx]
    v = np.stack([np.interp(np.arange(n), idx, vb[:, i]) for i in range(3)], 1)
    v = decay_tail(v, ph["last"], P["t"])
    fs = round(1 / np.median(np.diff(P["t"])))
    return np.cumsum(v, axis=0) / fs


def net_tracks(c, net):
    return infer17(net, c["X"], c["yaw"], c["tc"])


HPS = [1 / 15, 1 / 30, 1 / 45, 1 / 60, 1 / 90, 1 / 120]
TAUS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 40, 80]
ALPHAS = [0.0, 0.5, 1.0]
INNER_FILES = [("clean", "delete", 1), ("mild", "zero", 1), ("full", "zero", 1), ("full", "delete", 2)]


def split2(train):
    a, b = train[0::2], train[1::2]
    return a, b


def choose(train, no_up, seed, inner, yaw_aug):
    """Picks shrinkage from models that see only part of the training set, plus physics settings from training files."""
    cal = fit_physics(train)
    # high-pass for physics, separately per platform class
    best_hp = {}
    for nm in ("drone", "hand"):
        best, bv = HPS[3], 1e18
        files = [(n, *f) for n in train for f in INNER_FILES if ("drone" if isinstance(n, str) else "hand") == nm]
        if files:
            for hp in HPS:
                v = np.mean([file_ctx(*f)["sc"](phys_track(file_ctx(*f), cal, hp))["ate_se3"] for f in files])
                if v < bv:
                    best, bv = hp, v
        best_hp[nm] = best
    tau = {}
    if inner:
        a, b = split2(train)
        vals = {}
        for seen, unseen in ((a, b), (b, a)):
            netI = train_net(seen, seed + 7, no_up=no_up, yaw_aug=yaw_aug)
            for n in unseen:
                nm = "drone" if isinstance(n, str) else "hand"
                for f in INNER_FILES:
                    c = file_ctx(n, *f)
                    tk, v, var = net_tracks(c, netI)
                    for t0 in TAUS:
                        for al in ALPHAS:
                            x = to_rows(c["P"], tk, smooth_prior(v, var, t0, al))
                            vals.setdefault((nm, t0, al), []).append(c["sc"](x)["ate_se3"])
        for nm in ("drone", "hand"):
            sel = {k[1:]: np.mean(v) for k, v in vals.items() if k[0] == nm}
            if sel:
                tau[nm] = min(sel, key=sel.get)
    return cal, best_hp, tau


def run_fold(test, train, args, fi):
    seed = 900 + fi
    t0 = time.time()
    CALIB[0] = calib_for(train)
    _F.clear()
    cal, best_hp, tau = choose(train, args.no_up, seed, args.inner, args.yaw_aug)
    net = train_net(train, seed, no_up=bool(args.no_up), yaw_aug=bool(args.yaw_aug))
    torch.save(dict(state=net.state_dict(), cfg=net.net.cfg, a=float(net.a), b=float(net.b), train=train, cal=cal, best_hp=best_hp, tau=tau),
               f"{WGT}/e2e_{args.tag}_f{fi}.pt")
    rows = {}
    for (n, s, m, sd) in eval_set(test):
        c = file_ctx(n, s, m, sd)
        nm = "drone" if c["ph"]["drone"] else "hand"
        tk, v, var = net_tracks(c, net)
        hp = best_hp.get(nm, 1 / 60)
        t_, al_ = tau.get(nm, (4, 0.0))
        tracks = {
            "physics": phys_track(c, cal, hp),
            "band": band_track(c, cal),
            "net": to_rows(c["P"], tk, smooth_prior(v, var, t_, al_)),
        }
        rows[f"{n}|{s}|{m}|{sd}"] = {k: score_all(c["P"], c["room"], x, c["q"]) for k, x in tracks.items()}
        rows[f"{n}|{s}|{m}|{sd}"]["rest"] = bool(c["rest"])
        rows[f"{n}|{s}|{m}|{sd}"]["drone"] = bool(c["ph"]["drone"])
    pickle.dump(dict(rows=rows, cal=cal, hp=best_hp, tau=tau, a=float(net.a), b=float(net.b)), open(f"{RES}/e2e_{args.tag}_f{fi}.pkl", "wb"))
    print(f"{args.tag} fold {fi} test {test[:3]} {time.time()-t0:.0f}s hp {best_hp} tau {tau} a {float(net.a):.2f} b {float(net.b):.2f} scale { {k: round(v['scale'],2) for k,v in cal.items()} }", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make_pool":
        with mp.get_context("spawn").Pool(16) as p:
            for n in p.imap_unordered(pool_one, list(DEV_ROOMS) + SEQS):
                print("pool2", n, flush=True)
        print("POOL2_DONE")
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_up", type=int, default=0)
    ap.add_argument("--yaw_aug", type=int, default=0)
    ap.add_argument("--inner", type=int, default=1)
    a = ap.parse_args()
    a.tag = a.tag or a.scheme
    if a.scheme == "loeo":
        folds = [(v, [x for x in SEQS if x not in v]) for v in ENV.values()]
    elif a.scheme == "loso":
        folds = [([s], [x for x in SEQS if x != s]) for s in SEQS]
    elif a.scheme == "loro":
        folds = [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
    elif a.scheme == "tum2euroc":
        folds = [(SEQS, list(DEV_ROOMS))]
    elif a.scheme == "euroc2tum":
        folds = [(list(DEV_ROOMS), SEQS)]
    elif a.scheme == "both":
        folds = [(v, [x for x in SEQS if x not in v] + list(DEV_ROOMS)) for v in ENV.values()]
        folds += [([r], SEQS + [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
    for fi, (test, train) in enumerate(folds):
        if os.path.exists(f"{RES}/e2e_{a.tag}_f{fi}.pkl"):
            continue
        run_fold(test, train, a, fi)
    print(f"{a.tag} E2E_DONE", flush=True)
