"""One-off edit: fast position-only scorer for grids, same numbers as score_all (checked in the patched eval)."""
s = open("pipeline.py").read()
if "class PosScorer" not in s:
    s = s.replace("# ------------------------------------------------------------------ reference numbers", '''class PosScorer:
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


# ------------------------------------------------------------------ reference numbers''')
    open("pipeline.py", "w").write(s)

s = open("21_eval_cross.py").read()
if "PosScorer" not in s:
    s = s.replace('ap.add_argument("--tau_from", default="")', 'ap.add_argument("--att", default="auto")\nap.add_argument("--rfold", type=int, default=0)')
    s = s.replace('''        q = attitude_auto(P["t"], P["imu"], P["ok"])
        _P[key] = (P, q, load_room(n))''', '''        q = attitude_auto(P["t"], P["imu"], P["ok"]) if a.att == "auto" else baseline_attitude(P["t"], P["imu"], P["ok"])
        room = load_room(n)
        _P[key] = (P, q, room, PosScorer(P, room, q))''')
    s = s.replace('''        P, q, room = getP(key)
        for (tau, al) in grid:
            x = smooth_prior(o["v"], o["var"], tau, al)
            grid[(tau, al)][key] = score_all(P, room, to_rows(P, o["tk"], x), q)
        raw[key] = score_all(P, room, to_rows(P, o["tk"], integrate(o["v"])), q)''', '''        P, q, room, sc = getP(key)
        for (tau, al) in grid:
            x = smooth_prior(o["v"], o["var"], tau, al)
            grid[(tau, al)][key] = sc(to_rows(P, o["tk"], x))
        raw[key] = sc(to_rows(P, o["tk"], integrate(o["v"])))''')
    s = s.replace('glob.glob(f"{RES}/pred_{tag}_f*.pkl")', 'glob.glob(f"{RES}/pred_{tag}_f*.pkl") + glob.glob(f"{RES}/pred_{tag}_r?.pkl")')
    s = s.replace('cross = scheme in ("loso", "loeo", "both")', 'cross = scheme in ("loso", "loeo", "both") or a.att == "baseline"')
    open("21_eval_cross.py", "w").write(s)
print("patched")
