import sys, time
t00 = time.time()
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *
print("import", time.time() - t00, flush=True)
cfg = dict(T=100, center=True, steps=200, ch=(24, 32, 48, 64), attitude="auto", cached=True, pool_per_room=2)
t0 = time.time(); net = train_fold(["MH02", "MH03", "V101", "V102"], cfg, 1); print("train 200 steps", time.time() - t0, flush=True)
net.center = True
t, imu, _ = eval_file("MH01", "full", "zero", 1)
t0 = time.time(); P = prepare(t, imu); print("prepare", time.time() - t0, flush=True)
t0 = time.time(); b = auto_bias(P["t"], P["imu"], P["ok"]); print("auto_bias", time.time() - t0, flush=True)
t0 = time.time(); gb = gravity_bias(P["t"], P["imu"], P["ok"]); print("gravity_bias", time.time() - t0, flush=True)
t0 = time.time(); q = attitude_auto(P["t"], P["imu"], P["ok"]); print("attitude_auto", time.time() - t0, flush=True)
t0 = time.time(); o = infer(net, P, q=q); print("infer", time.time() - t0, flush=True)
t0 = time.time(); X = features(P["t"], P["imu"], P["ok"], q); print("features", time.time() - t0, flush=True)
