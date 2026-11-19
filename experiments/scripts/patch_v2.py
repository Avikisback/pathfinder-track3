"""One-off edit of phys.py, 30_attitude_v2.py and 32_e2e.py after the first look at rest detection and gravity fits.

Gravity alone cannot see bias along the mean up axis, and on EuRoC that axis is tilted from body x, so a truncated fit
loses about 0.029 rad/s of real yaw bias. On handheld TUM-VI the fit is pulled by walking accel (errors up to 0.05 rad/s).
So: rest if found, then platform calibration if the rotor signature says drone, otherwise zero with drift prior."""
s = open("phys.py").read()
if "def is_drone" not in s:
    s = s.replace('''def hardened_rest(t, imu, ok, ratio_thr=1.25, acc_ratio=1.6, rotor_a3=0.3, min_blocks=2):
    """Blocks whose gyro AND accel are white, rotors off by the accel third difference. Returns (bias samples at times)."""
    bl = [b for b in rest_blocks(t, imu, ok) if b[2] < ratio_thr and b[4] < acc_ratio and b[5] < rotor_a3]
''', '''def hardened_rest(t, imu, ok, ratio_thr=1.8, acc_ratio=3.0, rotor_a3=0.3, min_blocks=2, abs_floor=0.006):
    """Blocks with gyro std near this block's own noise (or under an absolute floor), rotors off by the accel third
    difference, accel norm near g. The noise is read locally, so a 5x noise increase does not hide rest."""
    bl = [b for b in rest_blocks(t, imu, ok) if (b[2] < ratio_thr or b[3] < abs_floor) and b[4] < acc_ratio and b[5] < rotor_a3]
''')
    s = s.replace('''    tb, mb = hardened_rest(t, imu, ok)
    info = {"rest": tb is not None}
    im2 = imu.copy()
    b0 = np.zeros(3) if calib is None else np.asarray(calib)
''', '''    tb, mb = hardened_rest(t, imu, ok)
    drone = is_drone(t, imu, ok)
    info = {"rest": tb is not None, "drone": drone}
    im2 = imu.copy()
    b0 = np.asarray(calib) if (calib is not None and drone) else np.zeros(3)
''')
    s = s.replace('''    elif calib is None:
        b0 = gravity_bias(t, imu, ok)
    q = offline_attitude''', '''    elif mode == "gravity":
        b0 = gravity_bias(t, imu, ok)
    q = offline_attitude''')
    s = s.replace('''def phys_channels(t, imu, ok, q):''', '''def is_drone(t, imu, ok):
    fs = round(1.0 / np.median(np.diff(t)))
    idx = np.arange(len(t))
    a = np.stack([np.interp(idx, idx[ok], imu[ok, 3 + i]) for i in range(3)], 1)
    last = np.nonzero(ok)[0][-1]
    return bool(rotors_on(a, fs)[: last + 1].mean() > 0.3)


def phys_channels(t, imu, ok, q):''')
    open("phys.py", "w").write(s)

s = open("30_attitude_v2.py").read()
if '"hardened rest, else zero' not in s:
    a = s.index("M = {")
    b = s.index('KEYS = ("att_org_pre"')
    s = s[:a] + '''M = {
    "main.cpp-style: still test, tilt filter": still_tilt,
    "gravity fit (truncated), offline": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], mode="gravity")[0],
    "hardened rest, else zero, offline": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"])[0],
    "hardened rest with ramp, else zero": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], mode="rest_ramp")[0],
    "hardened rest, else calib other envs": lambda P, s: attitude_v2(P["t"], P["imu"], P["ok"], calib=others(s, True))[0],
    "calib other envs only, offline": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=others(s, True) if isinstance(s, str) else np.zeros(3), **STRONG),
    "true gt bias, offline (oracle bound)": lambda P, s: offline_attitude(P["t"], P["imu"], P["ok"], b0=gtb[s] if isinstance(s, str) else np.zeros(3), **STRONG),
}
''' + s[b:]
    open("30_attitude_v2.py", "w").write(s)

s = open("32_e2e.py").read()
if "def calib_for" not in s:
    s = s.replace('''def build_X(t, imu, ok, q):''', '''def calib_for(seqs):
    """Sensor calibration: mean ground-truth bias over the given EuRoC training sequences."""
    use = [x for x in seqs if isinstance(x, str)]
    return np.mean([np.median(load_room(x)["gt_bw"], 0) for x in use], 0) if use else None


def build_X(t, imu, ok, q):''')
    s = s.replace('''        q, _i = attitude_v2(t, imu, ok)
        X, yaw, tc, ph = build_X(t, imu, ok, q)
        Xs.append(X)''', '''        q, _i = attitude_v2(t, imu, ok, calib=calib_for([x for x in SEQS if x != n]))
        X, yaw, tc, ph = build_X(t, imu, ok, q)
        Xs.append(X)''')
    s = s.replace('''_F = {}


def file_ctx(n, s, m, sd):
    key = (n, s, m, sd)
    if key not in _F:
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        q, info = attitude_v2(P["t"], P["imu"], P["ok"])
''', '''_F = {}
CALIB = [None]     # set per fold, from training sequences only


def file_ctx(n, s, m, sd):
    key = (n, s, m, sd, None if CALIB[0] is None else tuple(CALIB[0]))
    if key not in _F:
        t, imu, _ = eval_file(n, s, m, sd)
        P = prepare(t, imu)
        q, info = attitude_v2(P["t"], P["imu"], P["ok"], calib=CALIB[0])
''')
    s = s.replace('''    seed = 900 + fi
    t0 = time.time()
''', '''    seed = 900 + fi
    t0 = time.time()
    CALIB[0] = calib_for(train)
    _F.clear()
''')
    open("32_e2e.py", "w").write(s)
print("patched")
