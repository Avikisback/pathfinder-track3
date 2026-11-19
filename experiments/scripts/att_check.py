"""Attitude of the entry's output against ground truth: first-pose alignment (evo --align_origin), best constant rotation,
tilt only (angle between true up and estimated up), and heading drift over time. Usage: att_check.py <out.txt> <room>"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import common

out, rn = sys.argv[1], sys.argv[2]
room = common.load_room(int(rn) if rn.isdigit() else rn)
a = np.loadtxt(out, comments="#")
te = a[:, 0]
qe = R.from_quat(a[:, 4:8])
gt_t = room["gt_t_ns"] * 1e-9
gi = np.clip(np.searchsorted(te, gt_t), 1, len(te) - 1)
gi = gi - (gt_t - te[gi - 1] < te[gi] - gt_t)
m = np.abs(te[gi] - gt_t) < 0.01
qg = R.from_quat(room["gt_q"][m])
qe = qe[gi[m]]
t = gt_t[m] - gt_t[m][0]
# first pose alignment: world correction W = Rg0 Re0^-1
W = qg[0] * qe[0].inv()
e_org = (qg.inv() * (W * qe)).magnitude() * 180 / np.pi
# best constant world rotation (chordal mean of Rg Re^-1)
Wb = (qg * qe.inv()).mean()
e_best = (qg.inv() * (Wb * qe)).magnitude() * 180 / np.pi
# tilt: angle between up vectors (world z seen in body)
up_g = qg.inv().apply([0, 0, 1.0])
up_e = qe.inv().apply([0, 0, 1.0])
tilt = np.degrees(np.arccos(np.clip((up_g * up_e).sum(1), -1, 1)))
# heading drift: yaw of (Rg Re^-1) over time relative to its start
D = (qg * qe.inv()).as_matrix()
yaw = np.degrees(np.unwrap(np.arctan2(D[:, 1, 0] - D[:, 0, 1], D[:, 0, 0] + D[:, 1, 1])))
print(f"origin-aligned RMS {np.sqrt((e_org**2).mean()):.2f} deg | best constant rotation RMS {np.sqrt((e_best**2).mean()):.2f} deg | "
      f"tilt RMS {np.sqrt((tilt**2).mean()):.2f} deg, first 2 s {tilt[t < 2].mean():.2f} | heading change over file {yaw[-1]-yaw[0]:+.1f} deg, "
      f"range {yaw.max()-yaw.min():.1f} deg")
