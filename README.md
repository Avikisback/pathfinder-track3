# Path Finder (Track 3 entry)

Our entry for the Samsung XR Hackathon 2026, Track 3. It reads an IMU CSV (EuRoC format) and writes a 6-DoF pose for
every input row, including the rows of the 5 s blind window at the end. No network, no learned weights: one standalone
Python file, `src/main.py`, that needs only NumPy (the organisers' FAQ of 17 Nov asks for Python and a `main.py`).

## Run it

```bash
pip install "numpy>=1.24,<3"
python3 src/main.py data/tumvi_room3/imu0/data.csv out/estimated_trajectory.txt
python3 tools/check_output.py data/tumvi_room3/imu0/data.csv out/estimated_trajectory.txt
```

- The input may also be a dataset folder holding `imu0/data.csv`, or a zip of one.
- `bash tools/make_submission.sh <evaluation dataset> <CollegeName_TeamName>` (WSL or Linux) builds the bundle:
  `estimated_trajectory.txt`, `metadata.json`, `src/main.py`, `src/README.md`, `Dockerfile`, `requirements.txt`. It
  runs `tools/measure.py` (time and heap) and the format check.
- `Dockerfile` runs the same `main.py` on `python:3.12-slim` with NumPy. Built and run on 19 Nov (Docker Desktop 29.5,
  Python 3.12.16, NumPy 2.5.5, `--network none`): CSV, folder and zip inputs give the local output to 0.003 mm with
  identical timestamps.

Output lines are `timestamp tx ty tz qx qy qz qw`: exactly one per input row, in the input order, with the input's
own timestamp (duplicated rows keep their duplicated stamp). Nothing is appended: the blind window is already in the
file as rows with unusable values, and those rows carry the forecast.

## Rules we work to (FAQ, 17 Nov; `tools/Track3_FAQ_Responses.txt`)

- Python only, standalone `main.py`. Batch (non-causal) processing is allowed.
- 8 MB peak memory at the algorithm / heap level; 20 ms per second of IMU; one thread; offline.
- Dropouts are physically missing rows. Scoring: evo, SE(3) Umeyama alignment before ATE.
- Degradation: the organisers' `tools/degrade_imu.py` (seed 42). It expects the columns timestamp, accel, gyro, while
  EuRoC files are timestamp, gyro, accel, and we do not know which way they ran it. We test both:
  `_sw` (columns swapped for the script and back, so each noise lands on the sensor it was designed for) and `_og`
  (the script run on the EuRoC file as it is).

## How it works

The whole file is read before any pose is written, so slow sensor errors are solved for instead of filtered.

1. **Repair.** Rows go on a regular time grid. Spikes (far from the median of 5 rows), frozen repeats and invalid rows
   are dropped; holes up to 0.08 s (handheld) or 0.25 s (drone) are interpolated; longer ones stay dropouts.
2. **Platform.** Rotor vibration in the accelerometer tells a drone from a handheld device.
3. **Gyro calibration.** A rest period in the file is used when there is one. A drone without one starts from the
   declared calibration only if the file agrees: the gyro bias across the up direction is estimated from 4 s windows
   of the file itself, and if it is more than 0.03 rad/s away the file's own estimate is used.
4. **Attitude.** Gyro integration with a bias curve (linear spline, 8 s knots), first fitted against gravity.
5. **Self-calibration.** One Levenberg-Marquardt solve finds the gyro and accel bias curves, a tilt for each stretch
   between dropouts and a velocity step at each dropout, such that the 5 s average velocity and the position stay
   small. On drones the target velocity comes from rotor drag instead of zero. The stiffness of the bias curves is
   chosen per file by Bayesian evidence (Fellner-Schall updates).
6. **Post-filter.** A self-check, then a Wiener gain per octave band (drones) or fixed band gains (handheld). If the
   check fails, a plain high-pass keeps only fast motion. It fails when the 5 s average speed is implausible (over
   1.0 m/s on drones, 0.5 m/s handheld), or on a drone when the evidence chose a fast accel drift (over 0.3 m/s²)
   while that speed is over 0.6 m/s.
7. **Blind 5 s.** Hold, plus a damped share of the last velocity. A handheld file fits that share on its own path (how
   well did the velocity at every earlier second predict the next 5 s?). A drone uses 0.5, and descends 0.25 m over
   3 s if its rotors are still on at the cut, because logged flights end by landing; if the self-check failed, it
   holds. A 5 mm curve keeps the tail from being a straight line, which would break evo's alignment of the tail alone.

**Memory and time are bounded by design**, for any file length or rate:
- Rows are worked through in pieces of 2048; the solver works on 8-row blocks with a float32 Jacobian that is freed as
  soon as the normal equations hold it; poses are written as they are made; timestamps are re-read at write time.
- A file faster or longer than 40 000 rows is averaged into cells as it is read (spikes, frozen and invalid rows left
  out), and every input row gets its cell's pose. Nothing is ever held at the full rate.
- The one solve has size caps that only bind beyond about 3 minutes: at most 24 knots per bias curve, 200 residual
  rows, and 20 dropouts with their own velocity step (shorter extra ones are bridged). Cutting long files into
  separately solved windows was tried and was worse than standing still (see below).
- Robust to malformed input: NaN, empty or text values, missing header, CRLF, extra columns, out-of-order or duplicated
  stamps, ns/µs/ms/s units, all-zero files, empty files. Any failure inside the solver still yields one finite pose
  per row (`experiments/scripts/odd_inputs.py`, 22 cases).

## Declared constants (top of `src/main.py`)

All come from training data only; none names a sequence or a place.
- Drone gyro calibration: the mean gyro bias of the EuRoC flights, used only if the file confirms it (rest within
  0.02 rad/s, or the windowed estimate within 0.03 rad/s).
- Rotor drag k = 0.195 /s, IMU lever arm (0.019, 0.016, 0.103) m, rotor-rate term (-0.013, 0.014, 0.030), fitted
  jointly on the EuRoC flights (lever arm within 6 mm across leave-one-out folds).
- Post-filter: error power per band for drones (EuRoC), gain per band for handheld (TUM-VI rooms 1-5).
- Thresholds and the descent prior: 1.0 / 0.5 m/s, 0.3 m/s², 0.25 m over 3 s.

## Results with the organisers' degradation script

Scored by `experiments/scripts/score_off.py`, which follows the FAQ: one pose per input row with matching timestamps
(0 mismatches on every file), ATE after SE(3) Umeyama alignment over all rows including the blind window, RPE over
1 s, and the blind-window displacement error. "Still" is a constant pose. EuRoC = 11 drone flights, rooms = TUM-VI
rooms 1-5 (handheld).

| Set (seed 42 unless noted) | ATE, m | still | better than still | RPE 1 s, m (still) | blind 5 s, m (hold) |
|---|---|---|---|---|---|
| EuRoC `_sw` | 2.247 | 3.449 | 11 / 11 | 0.315 (0.777) | 0.307 (0.335) |
| EuRoC `_og` | 2.448 | 3.449 | 11 / 11 | 0.385 (0.777) | 0.323 (0.335) |
| Rooms `_sw` | 0.515 | 1.096 | 5 / 5 | 0.223 (0.859) | 0.498 (0.490) |
| Rooms `_og` | 0.530 | 1.096 | 5 / 5 | 0.233 (0.859) | 0.496 (0.490) |
| EuRoC `_sw`, seeds 1 / 2 / 3 | 2.332 / 2.369 / 2.372 | 3.448 | 11 / 11 each | | |
| Rooms `_sw`, seeds 1 / 2 / 3 | 0.561 / 0.486 / 0.508 | 1.096 | 5 / 5 each | | |
| TUM-VI walks, 22 files of 1.5-24.5 min (never tuned on) | 0.790 | 0.878 | 20 / 22 | 0.467 (0.712) | 0.889 (0.891) | | |
| Long rooms, 4.7-9.3 min (forward + reversed copies) | 0.721 | 1.096 | 5 / 5 | 0.397 (0.853) | 0.299 (0.301) | | |

RPE, the other metric the deck names, has a margin of 2.5 to 4 times over standing still. In rooms a blind-window hold
is the floor: with the true positions, moving home or to the centre of the path scores no better (`86_endings.py`).

What the official degradation costs (ablation, `make_official_abl.py`): on handheld files only the burst dropouts
matter (rooms 0.765 m with everything, 0.476 m without the bursts); white noise, bias and scale walks, spikes, stuck
values, single drops and duplicates are each nearly harmless after the repair and self-calibration.

**Earlier unseen-data checks** (C++ reference of the same method, our own degradation):
- Blackbird, a different quadrotor at 100 Hz: 3.02 m clean and 3.26 m degraded, against 3.40 m standing still.
- **Samsung Odyssey+ headset** (Monado SLAM dataset, 10 recordings of 20-180 s, real 1 kHz IMU;
  `tools/get_monado.py`): with the dataset's calibrated IMU, ATE 0.76 m against 0.86 m standing still (better on 9 of
  10), RPE over 1 s 0.24 m against 0.45 m. Degraded or raw, ATE is level with standing still while RPE stays ahead.

## Resources

Measured with `tools/measure.py` / `experiments/scripts/resources.py` (WSL 2, one thread, input on the Linux disk).
Heap: `tracemalloc` peak of the estimation after Python and NumPy are loaded (NumPy reports its array buffers to it).
Time: the whole `python3 main.py` process, interpreter start included, best of three.

| Input | Peak heap (limit 8 MB) | Time per second of IMU (limit 20 ms) |
|---|---|---|
| EuRoC flights and TUM-VI rooms, 200 Hz, 1.4-3 min | 2.5-4.8 MB | 3.5-6.1 ms |
| 1 kHz IMU, 2.4 min | 5.4 MB | 5.8 ms |
| Real walks 5-24.5 min; synthetic 59 min | 6.1-7.3 MB | 0.6-2.8 ms |

The laptop these were measured on slows down up to 3x when hot (after half an hour of 8-process scoring the same
files took 17-22 ms per s); the figures above are from a cool machine. Ask the organisers whether start-up counts. Python with NumPy loaded holds about 10 MB before the estimator starts and the process peaks near
45-55 MB resident; the FAQ measures at the heap level, which is the first column.

## Tested and not adopted

- **Independent windows for long files** (17 Nov). Solving 3-minute stretches separately and joining them on heading
  and position was worse than standing still on long files (1.21 m against 1.10), and a warm-up overlap made it worse
  still: a short window cannot pin the gyro bias, and its heading drifts tens of degrees. One capped solve: 0.72 m (0.66 m uncapped, which does not fit in 8 MB).
- **Tighter gyro anchor** (0.003 rad/s). Attitude on clean files improves (6.0° to 3.0° drones, 6.6° to 1.6° handheld)
  but rotation RPE does not change, mean ATE gets slightly worse, and it is worse when the starting calibration is
  wrong. No scored gain.
- **Other rotation bridges in dropouts.** Zero turn rate, or each side fading out, are all worse than the straight line
  between both sides (room ATE 0.69 to 0.79-0.88 m).
- **Fitted tail on drones.** Worse (0.32-0.34 m against 0.28-0.29 m): these flights end by landing.
- **Drift-rate self-check on handheld files.** Long real walks also look like fast drift; falling back there loses the
  fitted tail. Drones only.
- **IMU scale and axis-alignment correction.** Makes rooms worse (0.44 to 0.52 m): the fit finds the mocap-to-IMU frame
  rotation, not a sensor error.
- **True turn across every dropout** (an oracle). No help (rooms 0.68 to 0.78 m): what a gap costs is the velocity lost
  inside it, not the heading.
- **Homing or centre-seeking blind tail, restoring the drone's mean forward speed, a per-file walking-direction axis.**
  All rejected with numbers in `research_notes/`.

## Known weaknesses

- **Heading.** An IMU cannot see drift about the vertical. Attitude error is about 7-9° on the official `_sw` files and
  27° on `_og` EuRoC, nearly all heading. Position scores already include this.
- **Column order of the official script.** If the organisers ran it on EuRoC files as they are (`_og`), the gyro gets
  noise designed for an accelerometer; EuRoC ATE is 2.45 m instead of 2.25 m, still 29% ahead of standing still.
- **Burst dropouts on handheld files** are what the official degradation costs us; the velocity lost in a burst cannot
  be recovered from the IMU.
- **Slow flights.** On MH01, MH02, V101, V103 and V201 the gain over standing still is smaller than on fast flights.
- **Very short files.** Under about 20 s of live data the entry outputs a constant pose; Python start-up alone is about
  0.15 s, which is 15 ms per second on a 10 s file.

## Integrity rules for this code

Never identify a sequence from its timestamps, embed a trajectory, or use per-sequence constants.
`tools/attitude_report.py` prints a bias derived from ground truth for diagnosis only, and it must never feed the
estimator. Report held-out numbers, with standing still beside them. `src/pathfinder.cpp` is the C++ reference of the
same method (no longer submittable), kept for comparison; `src/main.cpp` is the earlier baseline.
