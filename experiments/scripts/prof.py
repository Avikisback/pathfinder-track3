import sys, time
sys.path.insert(0, "/mnt/e/clipping-pipeline/xr_track3/experiments/scripts")
from pipeline import *
from attitude import *
room = load_room("MH01"); lab = gt_labels(room); t = lab["t"]; rng = np.random.default_rng(1)
imu, ok = random_degrade(t, room["imu"], rng)
for name, f in [("degrade", lambda: random_degrade(t, room["imu"], rng)), ("still", lambda: still_bias(t, imu, ok)),
                ("gravity", lambda: gravity_bias(t, imu, ok)), ("fast", lambda: attitude_fast(t, imu, ok)),
                ("auto", lambda: attitude_auto(t, imu, ok)), ("features", lambda: features(t, imu, ok, attitude_fast(t, imu, ok))),
                ("pool_item", lambda: pool_item(room, lab, rng, "auto"))]:
    for _ in range(2):
        t0 = time.time(); f(); dt = time.time() - t0
    print(name, f"{dt*1e3:.0f} ms", flush=True)
