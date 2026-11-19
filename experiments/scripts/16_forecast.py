"""H6: tiny learned 5 s forecast. Causal 2 s window in, displacement at 1..5 s out, in the heading frame at failure.

Trains held-out folds and stores predictions on the clean-schedule files, for 15_gap_blind.py to score next to
hold and damped velocity. Failure anywhere is trained, not only at file ends.
"""
import pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *

HOR = np.array([1, 2, 3, 4, 5.0])
CH = (16, 24, 32)


def labels(room):
    lab = gt_labels(room)
    gt_t = room["gt_t_ns"] * 1e-9
    t = lab["t"]
    p0 = np.stack([np.interp(t, gt_t, room["gt_p"][:, i]) for i in range(3)], 1)
    d = np.zeros((len(t), 5, 3))
    for j, h in enumerate(HOR):
        d[:, j] = np.stack([np.interp(t + h, gt_t, room["gt_p"][:, i]) for i in range(3)], 1) - p0
    lab["d"] = d
    lab["ok"] = lab["ok"] & (t + 5.0 <= gt_t[-1])
    return lab


def train(train_seqs, seed, steps=3000, device="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rooms = {n: load_room(n) for n in train_seqs}
    labs = {n: labels(rooms[n]) for n in train_seqs}
    T = 100
    net = VelNet(T=T, ch=CH, out=15, fc=48, drop=0.1).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    offs = offsets(T, False, device)
    for step in range(steps):
        if step % 300 == 0:
            Xs, yh, ds, ys, goods = [], [], [], [], []
            base = 0
            pad = T + 2
            for n in train_seqs:
                for _ in range(2):
                    it = pool_item(rooms[n], labs[n], rng, "auto")
                    k = len(it["X"])
                    idx = DEC // 2 + np.arange(k) * DEC
                    Xs += [np.zeros((pad, 10), np.float32), it["X"]]
                    yh += [np.zeros(pad, np.float32), it["yawh"]]
                    ys += [np.zeros(pad, np.float32), it["yaw"]]
                    ds += [np.zeros((pad, 15), np.float32), labs[n]["d"][idx].reshape(k, 15).astype(np.float32)]
                    goods.append(np.nonzero(it["ok"])[0] + base + pad)
                    base += pad + k
            X = torch.from_numpy(np.concatenate(Xs + [np.zeros((pad, 10), np.float32)])).to(device)
            YH = torch.from_numpy(np.concatenate(yh + [np.zeros(pad, np.float32)])).to(device)
            Y = torch.from_numpy(np.concatenate(ys + [np.zeros(pad, np.float32)])).to(device)
            D = torch.from_numpy(np.concatenate(ds + [np.zeros((pad, 15), np.float32)])).to(device)
            G = torch.from_numpy(np.concatenate(goods)).to(device)
        k = G[torch.randint(len(G), (256,), device=device)]
        x = window_batch(X, YH, k, offs)
        c, s = torch.cos(-Y[k]), torch.sin(-Y[k])
        d = D[k].reshape(-1, 5, 3)
        y = rot_xy(d, c[:, None], s[:, None]).reshape(-1, 15)
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, P, q, device="cuda"):
    X, yawh, tc = features(P["t"], P["imu"], P["ok"], q)
    T = net.cfg["T"]
    pad = np.zeros((T + 2, 10), np.float32)
    Xp = torch.from_numpy(np.concatenate([pad, X, pad])).to(device)
    yp = torch.from_numpy(np.concatenate([np.zeros(T + 2, np.float32), yawh, np.zeros(T + 2, np.float32)])).to(device)
    ks = np.arange(0, len(X), STRIDE)
    kt = torch.from_numpy(ks + T + 2).long().to(device)
    offs = offsets(T, False, device)
    out = torch.cat([net(window_batch(Xp, yp, kt[i:i + 1024], offs)) for i in range(0, len(kt), 1024)]).cpu().numpy()
    return dict(tk=tc[ks], d=out)


SEQS = list(EUROC)
for label, folds in (("fc_euroc", [([s], [x for x in SEQS if x != s]) for s in SEQS]),
                     ("fc_tum", [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS])):
    for i, (test, tr) in enumerate(folds):
        f = f"{RES}/pred_{label}_f{i}.pkl"
        if os.path.exists(f):
            continue
        t0 = time.time()
        net = train(tr, 300 + i)
        preds = {}
        for n in test:
            t, imu, _ = eval_file(n, "clean", "delete", 1)
            P = prepare(t, imu)
            q = attitude_auto(P["t"], P["imu"], P["ok"]) if isinstance(n, str) else baseline_attitude(P["t"], P["imu"], P["ok"])
            preds[f"{n}|clean|delete|1"] = predict(net, P, q)
        pickle.dump(preds, open(f, "wb"))
        print(f"{label} fold {i} done {time.time()-t0:.0f}s params {sum(p.numel() for p in net.parameters())} macs {macs(net.cfg)}", flush=True)
print("FC_DONE")
