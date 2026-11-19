"""Trace the solver on a file that blows up: mild schedule with gaps."""
import importlib, sys
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
m41 = importlib.import_module("41_att_band")
from phys import *
from energy import EnergyCal

for n in (1, 3, 5, "MH03", "V201"):
    for sc in ("mild",):
        t, imu, gaps = eval_file(n, sc, "zero", 1)
        P = prepare(t, imu)
        room = load_room(n)
        b0 = m41.calib_excl(n)
        ec = EnergyCal(P["t"], P["imu"], P["ok"], b0=b0)
        print(n, sc, "gaps at", [round(P["t"][s], 1) for s, e in ec.gaps], "len", round(P["t"][ec.last], 1))
        q, v, Rm, fw = ec.forward(ec.x)
        print("  seed rms v", np.round(np.sqrt((v ** 2).sum(1).mean()), 2), "theta deg", np.round(np.degrees(ec.x["theta"]), 1).tolist())
        # velocity by segment
        for it in (12,):
            ec2 = EnergyCal(P["t"], P["imu"], P["ok"], b0=b0)
            q, v, tb, last, info = ec2.solve(iters=it)
            seg = [np.sqrt((v[(tb > a) & (tb < b)] ** 2).sum(1).mean()) for a, b in zip([0] + [P["t"][e] for s, e in ec2.gaps], [P["t"][s] for s, e in ec2.gaps] + [tb[-1]])]
            print(f"  iters {it}: rms v {info['rms_v']:.2f} obj {info['obj']:.1f} seg rms v {np.round(seg,2)} dv {np.round(ec2.x['dv'],1).tolist()} th {np.round(np.degrees(ec2.x['theta']),0).tolist()} cgmax {np.abs(ec2.x['cg']).max():.4f} camax {np.abs(ec2.x['ca']).max():.3f}", flush=True)
