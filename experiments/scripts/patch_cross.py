"""One-off edit: gravity_bias final projection; attitude_auto; yaw augmentation and no_up options in training."""
s = open("attitude.py").read()
if "strong directions only" not in s:
    s = s.replace("        thr = sig_thr * (1 if len(lags) > 2 else 4)\n        b = b + V @ np.where(sig < thr, (V.T @ y) / np.maximum(ev, 1e-12), 0.0)\n",
                  "        b = b + V @ np.where(sig < sig_thr, (V.T @ y) / np.maximum(ev, 1e-12), 0.0)\n        # strong directions only: never keep a component that gravity does not constrain\n        b = V @ np.where(sig < sig_thr, V.T @ b, 0.0)\n")
    s += '''

STRONG = dict(knot_s=60, s_r=0.3, sigma_b=0.003, residual_s=5.0)


def auto_bias(t, imu, ok):
    sb = still_bias(t, imu, ok)[0]
    return sb if sb is not None else gravity_bias(t, imu, ok)


def attitude_auto(t, imu, ok):
    """Best no-calibration attitude: still-second bias if there is one, else gravity fit, then the offline fit."""
    return offline_attitude(t, imu, ok, b0=auto_bias(t, imu, ok), **STRONG)


def attitude_fast(t, imu, ok):
    """Cheaper version for training pools: same bias, forward tilt filter."""
    i2 = imu.copy()
    i2[ok, :3] -= auto_bias(t, imu, ok)
    return baseline_attitude(t, i2, ok)
'''
    open("attitude.py", "w").write(s)

s = open("pipeline.py").read()
if "no_up" not in s:
    s = s.replace('''def pool_item(room, lab, rng, decimate_labels=True):
    t = lab["t"]
    imu, valid = random_degrade(t, room["imu"], rng)
    q = baseline_attitude(t, imu, valid)
''', '''def pool_item(room, lab, rng, attitude=None):
    t = lab["t"]
    imu, valid = random_degrade(t, room["imu"], rng)
    if attitude == "auto":
        from attitude import attitude_fast
        q = attitude_fast(t, imu, valid)
    else:
        q = baseline_attitude(t, imu, valid)
''')
    s = s.replace('''items = [pool_item(rooms[n], labs[n], rng) for n in train_rooms for _ in range(cfg.get("pool_per_room", 4))]''',
                  '''items = [pool_item(rooms[n], labs[n], rng, cfg.get("attitude")) for n in train_rooms for _ in range(cfg.get("pool_per_room", 4))]''')
    s = s.replace('''        x = window_batch(pool.X, pool.yawh, k, offs)
        out = net(x)
        mu, lv = out[:, :3], out[:, 3:].clamp(-7, 3)
        c, s = torch.cos(-pool.yaw[k])[:, None], torch.sin(-pool.yaw[k])[:, None]
        y = rot_xy(pool.v[k], c[:, 0], s[:, 0])
''', '''        x = window_batch(pool.X, pool.yawh, k, offs)
        c, s = torch.cos(-pool.yaw[k])[:, None], torch.sin(-pool.yaw[k])[:, None]
        y = rot_xy(pool.v[k], c[:, 0], s[:, 0])
        if cfg.get("yaw_aug"):
            # random heading for input and label together, so body-forward cannot be relied on
            ang = torch.rand(len(k), device=device) * 6.2831853
            ca, sa = torch.cos(ang), torch.sin(ang)
            y = rot_xy(y, ca, sa)
            for i in (0, 3):
                x0, x1 = x[:, i], x[:, i + 1]
                x = torch.cat([x[:, :i], (ca[:, None] * x0 - sa[:, None] * x1)[:, None], (sa[:, None] * x0 + ca[:, None] * x1)[:, None], x[:, i + 2:]], 1)
        if cfg.get("no_up"):
            x = torch.cat([x[:, :6], torch.zeros_like(x[:, 6:9]), x[:, 9:]], 1)
        out = net(x)
        mu, lv = out[:, :3], out[:, 3:].clamp(-7, 3)
''')
    s = s.replace('''    for i in range(0, len(kt), batch):
        outs.append(net(window_batch(Xp, yp, kt[i: i + batch], offs)))
''', '''    for i in range(0, len(kt), batch):
        xb = window_batch(Xp, yp, kt[i: i + batch], offs)
        if getattr(net, "no_up", False):
            xb = torch.cat([xb[:, :6], torch.zeros_like(xb[:, 6:9]), xb[:, 9:]], 1)
        outs.append(net(xb))
''')
    open("pipeline.py", "w").write(s)
print("patched")
