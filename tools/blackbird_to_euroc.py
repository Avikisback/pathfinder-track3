"""Convert the Air-IO Blackbird copy to EuRoC layout: <out>/<split>_<shape>_<speed>/imu0/data.csv and gt.csv (t_ns,x,y,z,qw,qx,qy,qz).
imu_data.csv: seconds, gyro xyz, accel xyz (checked: column 6 reads about -9.8 at the start, so accel is last). Ground truth:
microseconds, position, quaternion w first (first value about 0.98)."""
import glob, os, sys
import numpy as np
src, out = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{src}/Blackbird/*/*/yawForward/*/imu_data.csv")):
    d = os.path.dirname(f)
    parts = d.replace("\\", "/").split("/")
    name = f"{parts[-4]}_{parts[-3]}_{parts[-1]}"
    imu = np.loadtxt(f, delimiter=",")
    gt = np.loadtxt(f"{d}/groundTruthPoses.csv", delimiter=",")
    od = f"{out}/{name}"
    os.makedirs(f"{od}/imu0", exist_ok=True)
    t = np.round(imu[:, 0] * 1e9).astype(np.int64)
    with open(f"{od}/imu0/data.csv", "w") as fh:
        fh.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for tk, r in zip(t, imu[:, 1:7]):
            fh.write(f"{tk}," + ",".join(f"{x:.9f}" for x in r) + "\n")
    tg = np.round(gt[:, 0] * 1e3).astype(np.int64)
    with open(f"{od}/gt.csv", "w") as fh:
        fh.write("# timestamp[ns],tx,ty,tz,qw,qx,qy,qz\n")
        for tk, r in zip(tg, gt[:, 1:8]):
            fh.write(f"{tk}," + ",".join(f"{x:.9f}" for x in r) + "\n")
    print(name, len(t), "imu rows", f"{(t[-1]-t[0])/1e9:.1f}s", "gt", f"{(tg[0]-t[0])/1e9:+.2f}..{(tg[-1]-t[-1])/1e9:+.2f}s vs imu")
