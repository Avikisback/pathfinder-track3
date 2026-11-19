"""Why does energy calibration blow up once gaps are present? One file, several cases."""
import importlib, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
from phys import *
from energy import EnergyCal
import common

CASES = {
    "clean": dict(sg=(0.0, 0.0), sa=(0.0, 0.0), bg=0.0, ba=0.0, gaps=0, gap_s=0.0),
    "gaps_only": dict(sg=(0.0, 0.0), sa=(0.0, 0.0), bg=0.0, ba=0.0, gaps=3, gap_s=2.0),
    "noise_only": dict(sg=(0.002, 0.008), sa=(0.02, 0.08), bg=0.0, ba=0.0, gaps=0, gap_s=0.0),
    "bias_only": dict(sg=(1e-9, 0.0), sa=(1e-9, 0.0), bg=0.02, ba=0.2, gaps=0, gap_s=0.0),
    "gbias_only": dict(sg=(1e-9, 0.0), sa=(1e-9, 0.0), bg=0.02, ba=0.0, gaps=0, gap_s=0.0),
    "abias_only": dict(sg=(1e-9, 0.0), sa=(1e-9, 0.0), bg=0.0, ba=0.2, gaps=0, gap_s=0.0),
}
for name, sch in CASES.items():
    common.SCHEDULES["dbg"] = sch
    for n in (1, "MH03"):
        room = load_room(n)
        t, imu, gaps = degrade(room, 1, "zero", "dbg")
        P = prepare(t, imu)
        b0 = m41.calib_excl(n)
        ec = EnergyCal(P["t"], P["imu"], P["ok"], b0=b0)
        q0, v0, *_ = ec.forward(ec.x)
        q, v, tb, last, info = ec.solve()
        tk = tb
        vs = np.cumsum(v - v.mean(0), axis=0) * np.median(np.diff(tb))
        a, st = m41.ate(vs, tk, room)
        pre = P["ids"] != -1
        pre[last + 1 :] = False
        d = score(P["t_ns"][pre] * 1e-9, np.zeros((int(pre.sum()), 3)), q[pre][:, [1, 2, 3, 0]], room["gt_t_ns"] * 1e-9, room["gt_p"], room["gt_q"])
        print(f"{name:11s} {n!s:5s} seed rmsv {np.sqrt((v0**2).sum(1).mean()):8.2f} -> {info['rms_v']:6.2f}  raw ATE {a:7.2f} stat {st:.2f} att {d['att_org']:5.1f} tilt {d['tilt']:5.2f} ca {np.abs(ec.x['ca']).max():.3f} cg {np.abs(ec.x['cg']).max():.4f} th {np.round(np.degrees(ec.x['theta']).ravel(),0)}", flush=True)
