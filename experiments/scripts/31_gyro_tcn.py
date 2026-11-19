"""Coordinator item 2: gyro TCN under injected drift, both datasets, held-out folds. Brossard-style in spirit: a window of
gyro and accel in the body frame gives a slowly varying rate correction. Input is bias-corrected gyro (rest or calib).
Loss is on rate over 0.2 s. The output is low-passed at test time, since real bias is slow.

Compared with the same attitude pipeline without the net. Reads only IMU at test time."""
import pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from phys import *
from scipy.spatial.transform import Slerp

SEQS = list(EUROC)
T = 400
CH = (16, 24, 32, 48, 64)


def calib_for(seqs):
    use = [x for x in seqs if isinstance(x, str)]
    return np.mean([np.median(load_room(x)["gt_bw"], 0) for x in use], 0) if use else np.zeros(3)


def body_channels(t, imu, ok):
    n = len(t) // DEC
    m = ok[: n * DEC].reshape(n, DEC, 1).astype(np.float64)
    c = np.maximum(m.sum(1), 1)
    w = (imu[: n * DEC, :3].reshape(n, DEC, 3) * m).sum(1) / c
    a = (imu[: n * DEC, 3:].reshape(n, DEC, 3) * m).sum(1) / c
    good = m.sum(1) > 0
    return np.concatenate([w / 2.0 * good, a / 10.0 * good, m.mean(1)], 1).astype(np.float32)


def seq_labels(n):
    room = load_room(n)
    t = (room["t_ns"] - room["t_ns"][0]) * 1e-9
    gt = (room["gt_t_ns"] - room["t_ns"][0]) * 1e-9
    sl = Slerp(gt, R.from_quat(room["gt_q"]))
    k = len(t) // DEC
    tc = t[DEC // 2:: DEC][:k]
    h = 0.1
    good = (tc - h > gt[0]) & (tc + h < gt[-1])
    # mocap holes
    idx = np.clip(np.searchsorted(gt, tc), 1, len(gt) - 1)
    good &= (gt[idx] - gt[idx - 1]) < 0.1
    rate = np.zeros((k, 3))
    r0, r1 = sl(tc[good] - h), sl(tc[good] + h)
    rate[good] = (r0.inv() * r1).as_rotvec() / (2 * h)
    return dict(t=t, imu=room["imu"], rate=rate.astype(np.float32), good=good, tc=tc)


def train(train_seqs, seed, steps=2500, device="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labs = {n: seq_labels(n) for n in train_seqs}
    calib = calib_for(train_seqs)
    net = VelNet(T=T, ch=CH, out=3, fc=48, c_in=7, drop=0.05).to(device)
    nn.init.zeros_(net.head[-1].weight)
    nn.init.zeros_(net.head[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    offs = torch.from_numpy(np.arange(T) - T // 2).long().to(device)
    pad = T + 2
    for step in range(steps):
        if step % 200 == 0:
            Xs, Ys, Gs = [], [], []
            base = 0
            for n in train_seqs:
                L = labs[n]
                drone = isinstance(n, str)
                for _ in range(2):
                    imu, ok = random_degrade(L["t"], L["imu"], rng)
                    im2 = imu.copy()
                    if drone:
                        im2[ok, :3] -= calib
                    X = body_channels(L["t"], im2, ok)
                    k = len(X)
                    # target: what must be subtracted from this realisation's gyro
                    cnt = np.maximum(ok[: k * DEC].reshape(k, DEC).sum(1, keepdims=True), 1)
                    w_avg = (im2[: k * DEC, :3].reshape(k, DEC, 3) * ok[: k * DEC].reshape(k, DEC, 1)).sum(1) / cnt
                    y = (w_avg - L["rate"]).astype(np.float32)
                    g = L["good"] & (ok[DEC // 2:: DEC][:k])
                    Xs += [np.zeros((pad, 7), np.float32), X]
                    Ys += [np.zeros((pad, 3), np.float32), y]
                    Gs.append(np.nonzero(g)[0] + base + pad)
                    base += pad + k
            X_t = torch.from_numpy(np.concatenate(Xs + [np.zeros((pad, 7), np.float32)])).to(device)
            Y_t = torch.from_numpy(np.concatenate(Ys + [np.zeros((pad, 3), np.float32)])).to(device)
            G_t = torch.from_numpy(np.concatenate(Gs)).to(device)
        k = G_t[torch.randint(len(G_t), (256,), device=device)]
        x = X_t[k[:, None] + offs].permute(0, 2, 1)
        loss = ((net(x) - Y_t[k]) ** 2).mean() * 100
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return net, calib


@torch.no_grad()
def correction(net, t, im2, ok, device="cuda"):
    X = body_channels(t, im2, ok)
    pad = np.zeros((T + 2, 7), np.float32)
    Xp = torch.from_numpy(np.concatenate([pad, X, pad])).to(device)
    offs = torch.from_numpy(np.arange(T) - T // 2).long().to(device)
    ks = np.arange(0, len(X), 25)
    kt = torch.from_numpy(ks + T + 2).long().to(device)
    o = torch.cat([net(Xp[kt[i:i + 512, None] + offs].permute(0, 2, 1)) for i in range(0, len(kt), 512)]).cpu().numpy().astype(np.float64)
    tc = t[DEC // 2:: DEC][: len(X)][ks]
    return tc, o


KEYS = ("att_org_pre", "tilt_pre", "rpe_r_pre")
for label, folds in (("EuRoC LOEO", [(v, [x for x in SEQS if x not in v]) for v in ENV.values()]),
                     ("TUM dev LORO", [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS])):
    base_rows, tcn_rows, tcn_slow = [], [], []
    for i, (test, tr) in enumerate(folds):
        t0 = time.time()
        net, calib = train(tr, 500 + i)
        for (n, s, m, sd) in eval_set(test, seeds=(1, 2)):
            room = load_room(n)
            t, imu, _ = eval_file(n, s, m, sd)
            P = prepare(t, imu)
            ok = P["ok"]
            drone = isinstance(n, str)
            b0 = calib if drone else np.zeros(3)
            im2 = P["imu"].copy()
            im2[ok, :3] -= b0
            q0 = offline_attitude(P["t"], im2, ok, b0=np.zeros(3), **STRONG)
            tc, o = correction(net, P["t"], im2, ok)
            im3 = im2.copy()
            for ax in range(3):
                im3[ok, ax] -= np.interp(P["t"][ok], tc, o[:, ax])
            q1 = offline_attitude(P["t"], im3, ok, b0=np.zeros(3), **STRONG)
            # bias is slow: smooth the net output over about 20 s before using it
            ker = np.exp(-0.5 * ((tc[:, None] - tc[None]) / 10.0) ** 2)
            os = (ker @ o) / ker.sum(1, keepdims=True)
            im4 = im2.copy()
            for ax in range(3):
                im4[ok, ax] -= np.interp(P["t"][ok], tc, os[:, ax])
            q2 = offline_attitude(P["t"], im4, ok, b0=np.zeros(3), **STRONG)
            for q, rows in ((q0, base_rows), (q1, tcn_rows), (q2, tcn_slow)):
                et, _, eq = finish(P, np.zeros((len(P["t"]), 3)), q)
                pre = et <= P["t_fail_ns"]
                d = score(et[pre] * 1e-9, np.zeros((pre.sum(), 3)), eq[pre], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
                rows.append((f"{n}|{s}|{m}|{sd}", dict(att_org_pre=d["att_org"], tilt_pre=d["tilt"], rpe_r_pre=d["rpe_r"])))
        print(label, "fold", i, f"{time.time()-t0:.0f}s params {sum(p.numel() for p in net.parameters())} macs {macs(net.cfg)}", flush=True)
    print(f"== {label}")
    show("no TCN, calib or zero bias", summarise(base_rows), KEYS)
    show("gyro TCN, per 0.5 s", summarise(tcn_rows), KEYS)
    show("gyro TCN, slowly smoothed", summarise(tcn_slow), KEYS)
print("TCN_DONE")
