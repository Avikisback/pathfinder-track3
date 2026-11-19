"""Harness checks: degrade() equals tools/degrade.py, score() equals evo, frames and clocks of ground truth."""
import subprocess, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from common import *

check_degrade(1, 2)

# ground truth frame and clock against the IMU, dev rooms only (room 6 stays locked)
for n in DEV_ROOMS:
    d = load_room(n)
    ti, imu = d["t_ns"] * 1e-9, d["imu"]
    gt, q = d["gt_t_ns"] * 1e-9, R.from_quat(d["gt_q"])
    t0 = max(ti[0], gt[0]); 
    grid = np.arange(t0 + 1, min(ti[-1], gt[-1]) - 1, 0.02)
    from scipy.spatial.transform import Slerp
    sl = Slerp(gt, q)
    Rg = sl(grid)
    w_gt = (Rg[:-1].inv() * Rg[1:]).as_rotvec() / 0.02   # body rate from ground truth
    tm = grid[:-1] + 0.01
    w_imu = np.stack([np.interp(tm, ti, imu[:, i]) for i in range(3)], 1)
    # best time offset, then best fixed rotation between the two frames (Kabsch on rate vectors)
    best = None
    for off in np.arange(-0.05, 0.0501, 0.0025):
        w2 = np.stack([np.interp(tm + off, ti, imu[:, i]) for i in range(3)], 1)
        c = (w2 * w_gt).sum() / np.sqrt((w2 ** 2).sum() * (w_gt ** 2).sum())
        if best is None or c > best[0]:
            best = (c, off)
    u, _, vt = np.linalg.svd(w_gt.T @ w_imu)
    Rx = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    dg = np.diff(gt)
    a_w = Rg[:-1].apply(np.stack([np.interp(tm, ti, imu[:, i + 3]) for i in range(3)], 1))
    print(f"room{n}: imu {ti[-1]-ti[0]:.1f}s rows {len(ti)} | gt {gt[-1]-gt[0]:.1f}s rate {1/np.median(dg):.1f}Hz gaps>0.05s {(dg>0.05).sum()} max {dg.max():.3f}s | "
          f"clock offset {best[1]*1e3:+.1f} ms corr {best[0]:.4f} | R_gt_imu angle {np.degrees(R.from_matrix(Rx).magnitude()):.2f} deg | "
          f"mean world accel {np.round(a_w.mean(0),3)} | RMS radius {np.sqrt(((d['gt_p']-d['gt_p'].mean(0))**2).sum(1).mean()):.3f} m "
          f"speed {np.linalg.norm(np.diff(d['gt_p'],axis=0),axis=1).sum()/(gt[-1]-gt[0]):.2f} m/s", flush=True)

# score() against evo on real baseline output
EVO = ROOT + "/.venv-wsl/bin"
def evo(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("rmse"):
            return float(line.split()[1])
    return float("nan")

for n, seed, mode in [(1, 1, "delete"), (3, 2, "zero")]:
    d = load_room(n)
    t, imu, _ = degrade(d, seed, mode)
    et, ep, eq = run_baseline(t, imu)
    mine = score(et * 1e-9, ep, eq, d["gt_t_ns"] * 1e-9, d["gt_p"], d["gt_q"])
    f = f"{TMP}/chk.txt"; write_tum(f, et, ep, eq)
    g = f"{ROOT}/data/tumvi_room{n}/gt.tum"
    base = [g, f, "--t_max_diff", "0.01", "--no_warnings"]
    e = dict(
        ate_se3=evo([EVO + "/evo_ape", "tum", *base, "-a"]),
        ate_org=evo([EVO + "/evo_ape", "tum", *base, "--align_origin"]),
        att_org=evo([EVO + "/evo_ape", "tum", *base, "--align_origin", "-r", "angle_deg"]),
        rpe_t=evo([EVO + "/evo_rpe", "tum", *base, "-a", "-r", "trans_part", "--delta", "120", "--delta_unit", "f"]),
        rpe_r=evo([EVO + "/evo_rpe", "tum", *base, "-a", "-r", "angle_deg", "--delta", "120", "--delta_unit", "f"]),
    )
    print(f"room{n} seed{seed} {mode} baseline")
    for k in e:
        print(f"  {k:8s} mine {mine[k]:.4f}  evo {e[k]:.4f}  diff {mine[k]-e[k]:+.4f}", flush=True)
