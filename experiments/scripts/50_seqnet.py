"""New idea 2: train on the scoring metric itself, over whole trajectories.

The window net sees 2 s and is trained on velocity. Position error lives at low frequency, where a 2 s window has no
information, and the loss never sees the metric. Here: a convolution front end, a bidirectional GRU over the whole file
(offline is allowed), and a loss that integrates velocity to position, aligns it the way evo does and penalises ATE and
RPE. No ground truth at test time, no absolute position as input, no sequence identity.

Usage: 50_seqnet.py --scheme loeo|loso|loro [--tag x] [--hidden 48] [--steps 2500] [--both 0]
"""
import argparse, importlib, json, os, pickle, sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
e2e = importlib.import_module("32_e2e")
from phys import *

STEP = 5           # feature rows per output, so 10 Hz
ROT_CH = (0, 3, 10, 13)


class SeqNet(nn.Module):
    def __init__(self, hidden=48, ch=32, c_in=17, drop=0.1):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(c_in, ch, 7, padding=3), nn.GELU(), nn.Conv1d(ch, ch, 5, stride=STEP, padding=2), nn.GELU())
        self.gru = nn.GRU(ch, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.a = nn.Parameter(torch.tensor(0.4))
        self.b = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """x: (B, T, 17) at 50 Hz. Returns velocity at 10 Hz in the frame of the input."""
        f = self.front(x.permute(0, 2, 1)).permute(0, 2, 1)
        h, _ = self.gru(f)
        r = self.head(h)
        n = r.shape[1]
        mid = STEP // 2 + torch.arange(n, device=x.device) * STEP
        mid = mid.clamp(max=x.shape[1] - 1)
        base = 2.0 * (self.a * x[:, mid, 10:13] + self.b * x[:, mid, 13:16])
        return r + base


def kabsch(p, g, w):
    """Best rotation and centres, weighted by w (B, T). Rotation is detached, like a fixed gauge for this step."""
    ws = w.sum(1, keepdim=True).clamp(min=1.0)
    mp = (p * w[..., None]).sum(1, keepdim=True) / ws[..., None]
    mg = (g * w[..., None]).sum(1, keepdim=True) / ws[..., None]
    H = ((g - mg) * w[..., None]).permute(0, 2, 1) @ (p - mp) / ws[..., None]
    u, _, vh = torch.linalg.svd(H.detach())
    d = torch.sign(torch.det(u @ vh))
    s = torch.eye(3, device=p.device)[None].repeat(len(p), 1, 1)
    s[:, 2, 2] = d
    return u @ s @ vh, mp, mg


def metric_loss(v, g, w):
    """v (B,T,3) velocity at 10 Hz, g true position at 10 Hz, w valid. Each term relative to standing still."""
    p = torch.cumsum(v, 1) * 0.1
    Rot, mp, mg = kabsch(p, g, w)
    pa = (p - mp) @ Rot.permute(0, 2, 1)
    gc = g - mg
    e2 = ((pa - gc) ** 2).sum(-1)
    ws = w.sum(1).clamp(min=1.0)
    r2 = ((gc ** 2).sum(-1) * w).sum(1) / ws
    ate = torch.sqrt((e2 * w).sum(1) / ws + 1e-9) / torch.sqrt(r2 + 1e-6)
    tot = ate.mean()
    terms = [ate.mean()]
    for lag, wt in ((10, 0.5), (50, 0.25)):
        dp = pa[:, lag:] - pa[:, :-lag]
        dg = gc[:, lag:] - gc[:, :-lag]
        ww = w[:, lag:] * w[:, :-lag]
        e = (((dp - dg) ** 2).sum(-1) * ww).sum(1) / ww.sum(1).clamp(min=1.0)
        s = ((dg ** 2).sum(-1) * ww).sum(1) / ww.sum(1).clamp(min=1.0)
        term = (torch.sqrt(e + 1e-9) / torch.sqrt(s + 1e-6)).mean()
        tot = tot + wt * term
        terms.append(term)
    return tot, terms


_LAB = {}


def labels(n):
    if n not in _LAB:
        room = load_room(n)
        lab = gt_labels(room)
        d = e2e.get_pool(n)
        k = d["X"].shape[1]
        out = k // STEP
        idx = DEC // 2 + (STEP // 2 + np.arange(out) * STEP) * DEC
        gt_t = room["gt_t_ns"] * 1e-9
        t = lab["t"][idx]
        p = np.stack([np.interp(t, gt_t, room["gt_p"][:, i]) for i in range(3)], 1).astype(np.float32)
        _LAB[n] = dict(p=p, ok=lab["ok"][idx])
    return _LAB[n]


def batch(train, rng, B, min_s, device):
    Ls = []
    for n in train:
        Ls.append(e2e.get_pool(n)["X"].shape[1])
    # crop length: between min_s and the shortest file in this draw
    seqs = [train[i] for i in rng.integers(0, len(train), B)]
    Lmax = min(e2e.get_pool(n)["X"].shape[1] for n in seqs)
    L = int(rng.uniform(min(min_s * 50, Lmax), Lmax)) // STEP * STEP
    X, P, W = [], [], []
    for n in seqs:
        d = e2e.get_pool(n)
        i = rng.integers(0, len(d["X"]))
        s = rng.integers(0, d["X"].shape[1] - L + 1) // STEP * STEP
        x = d["X"][i][s : s + L].copy()
        th = rng.uniform(0, 2 * np.pi)
        c, sn = np.cos(th), np.sin(th)
        for g in ROT_CH:
            a0, a1 = x[:, g].copy(), x[:, g + 1].copy()
            x[:, g], x[:, g + 1] = c * a0 - sn * a1, sn * a0 + c * a1
        lab = labels(n)
        a = s // STEP
        P.append(lab["p"][a : a + L // STEP])
        W.append(lab["ok"][a : a + L // STEP])
        X.append(x)
    return (torch.from_numpy(np.stack(X)).to(device), torch.from_numpy(np.stack(P)).to(device),
            torch.from_numpy(np.stack(W)).float().to(device))


def train_seq(train, seed, hidden, steps, B=10, device="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SeqNet(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    for st in range(steps):
        net.train()
        x, g, w = batch(train, rng, B, 60, device)
        v = net(x)
        n = min(v.shape[1], g.shape[1])
        loss, terms = metric_loss(v[:, :n], g[:, :n], w[:, :n])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        sched.step()
        if st % 500 == 0:
            print(f"  step {st} loss {loss.item():.3f} ate {terms[0].item():.3f} rpe1 {terms[1].item():.3f} rpe5 {terms[2].item():.3f} a {net.a.item():.2f} b {net.b.item():.2f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def infer_seq(nets, c, device="cuda"):
    x = torch.from_numpy(c["X"][None]).to(device)
    outs = [net(x)[0].cpu().numpy().astype(np.float64) for net in nets]
    v = np.mean(outs, 0)
    k = len(v)
    tk = c["tc"][np.minimum(STEP // 2 + np.arange(k) * STEP, len(c["tc"]) - 1)]
    return tk, v


def track(c, tk, v, tau_hp=None, tail_tau=1.0):
    P = c["P"]
    t_last = P["t"][c["ph"]["last"]]
    v = v.copy()
    tail = tk > t_last
    if tail.any() and (~tail).any():
        v0 = v[~tail][-5:].mean(0)
        v[tail] = v0[None] * np.exp(-(tk[tail] - t_last) / tail_tau)[:, None]
    if tau_hp:
        x = smooth_prior(v, np.ones_like(v), tau_hp, 0.0)
    else:
        x = np.cumsum(v, axis=0) * 0.1
    return to_rows(P, tk, x)


def folds_for(scheme):
    S = e2e.SEQS
    if scheme == "loeo":
        return [(v, [x for x in S if x not in v]) for v in ENV.values()]
    if scheme == "loso":
        return [([s], [x for x in S if x != s]) for s in S]
    if scheme == "loro":
        return [([r], [x for x in DEV_ROOMS if x != r]) for r in DEV_ROOMS]
    raise ValueError(scheme)


TAUS_S = (None, 80, 40, 24, 16, 8, 4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--tag", default="seq")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--members", type=int, default=1)
    a = ap.parse_args()
    for fi, (test, train) in enumerate(folds_for(a.scheme)):
        out = f"{RES}/{a.tag}_{a.scheme}_f{fi}.pkl"
        if os.path.exists(out):
            continue
        t0 = time.time()
        e2e.CALIB[0] = e2e.calib_for(train)
        e2e._F.clear()
        nets = [train_seq(train, 700 + fi + 31 * m, a.hidden, a.steps) for m in range(a.members)]
        torch.save(dict(states=[n.state_dict() for n in nets], hidden=a.hidden, train=train), f"{WGT}/{a.tag}_{a.scheme}_f{fi}.pt")
        rows = {}
        for key in eval_set(test):
            c = e2e.file_ctx(*key)
            tk, v = infer_seq(nets, c)
            o = {}
            for tau in TAUS_S:
                o[f"tau{tau}"] = c["sc"](track(c, tk, v, tau))
            rows["|".join(map(str, key))] = o
        pickle.dump(dict(rows=rows), open(out, "wb"))
        print(f"{a.tag} {a.scheme} fold {fi} test {test[:3]} {time.time()-t0:.0f}s", flush=True)
    print("SEQ_DONE", flush=True)
