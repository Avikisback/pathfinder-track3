"""Top-down picture of one run for the demo video: ground truth against the entry's path, aligned the way the scorer
aligns (SE(3) Umeyama over all rows), with the blind 5 s window dashed. 1920x1080 PNG.
Usage: demo_plot.py <degraded csv> <estimated_trajectory.txt> <out.png> "<title>"
Ground truth comes from <csv>.gt.npz next to the CSV if present, else from the dataset cache (common.load_room)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
import common  # noqa: E402

csv, est, out, title = sys.argv[1:5]
name = os.path.basename(csv)[:-4]
n = name.split("_")[0]
if os.path.exists(csv[:-4] + ".gt.npz"):
    room = dict(np.load(csv[:-4] + ".gt.npz"))
else:
    room = common.load_room(int(n[4:]) if n.startswith("room") else n)

tin = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
a = np.loadtxt(est, comments="#", usecols=(1, 2, 3))
gt_t = room["gt_t_ns"]
j = np.clip(np.searchsorted(gt_t, tin), 1, len(gt_t) - 1)
j = j - ((tin - gt_t[j - 1]) < (gt_t[j] - tin))
m = np.abs(gt_t[j] - tin) < 5_000_000
t = (tin[m] - tin[0]) * 1e-9
e, g = a[m], room["gt_p"][j[m]]
r, tr = common.umeyama(e, g)
al = (r @ e.T).T + tr
ate = float(np.sqrt(((al - g) ** 2).sum(1).mean()))
still = float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))
blind = t >= (tin[-1] - tin[0]) * 1e-9 - 5.0

INK, MUTED, GRID, SURF = "#1B2638", "#5A6678", "#DAD6CC", "#F6F5F1"
BLUE, ORANGE = "#2F6FD1", "#F28C28"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
live = ~blind
ax.plot(g[live, 0], g[live, 1], color=BLUE, linewidth=2.5, solid_capstyle="round", label="Ground truth")
ax.plot(al[live, 0], al[live, 1], color=ORANGE, linewidth=2.5, solid_capstyle="round", label="Path Finder")
if blind.any():
    ax.plot(g[blind, 0], g[blind, 1], color=BLUE, linewidth=2.5, linestyle=(0, (4, 3)))
    ax.plot(al[blind, 0], al[blind, 1], color=ORANGE, linewidth=2.5, linestyle=(0, (4, 3)), label="Blind 5 s (dashed)")
    k = np.nonzero(blind)[0][0]
    ax.plot([al[k, 0]], [al[k, 1]], "o", markersize=12, color=ORANGE, markeredgecolor=SURF, markeredgewidth=2)
ax.plot([g[0, 0]], [g[0, 1]], "o", markersize=12, color=BLUE, markeredgecolor=SURF, markeredgewidth=2)
box = dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.9)
ax.annotate("start", (g[0, 0], g[0, 1]), xytext=(-70, -44), textcoords="offset points", color=INK, fontsize=22, bbox=box,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
# direct labels where each path's live part ends (just before the blind window), pushed apart
for p, lab, off in ((g[live][-1], "Ground truth", (40, 48)), (al[live][-1], "Path Finder", (40, -60))):
    ax.annotate(lab, (p[0], p[1]), xytext=off, textcoords="offset points", fontsize=22, color=INK, weight="bold",
                bbox=box, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
leg = ax.legend(loc="upper left", frameon=False, fontsize=22)
fig.suptitle(title, x=0.06, ha="left", fontsize=34, weight="bold", color=INK)
ax.set_title("Position error (ATE) %.2f m   ·   standing still %.2f m" % (ate, still), loc="left", fontsize=24,
             color=MUTED, pad=16)
fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.95))
fig.savefig(out, facecolor=SURF)
print("%s: ATE %.3f m, still %.3f m -> %s" % (name, ate, still, out))
