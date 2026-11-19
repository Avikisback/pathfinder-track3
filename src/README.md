# Path Finder (Track 3): how to run

The estimator is one standalone file, `main.py`: Python 3 and NumPy, nothing else. It runs on one thread
(BLAS threads are pinned to 1 before NumPy loads), offline, with no learned weights.

## Run

```bash
pip install "numpy>=1.24,<3"
python3 main.py <imu0/data.csv | dataset folder | dataset.zip> estimated_trajectory.txt
```

A dataset folder (or zip) is searched for `imu0/data.csv`. Without a second argument the output is
`estimated_trajectory.txt` in the current folder.

With Docker, from the folder that holds the `Dockerfile`:

```bash
docker build -t pathfinder .
docker run --rm --network none -v /path/to/dataset:/data -v "$PWD/out":/out pathfinder /data /out/estimated_trajectory.txt
```

Tested with Python 3.12 and NumPy 1.26 and 2.x on Ubuntu 24.04 (WSL 2). The Docker image was built and run on 19 Nov
(Python 3.12.16, NumPy 2.5.5, `--network none`): CSV, folder and zip inputs match the local output to 0.003 mm.

## Input and output

- Input: EuRoC-format IMU CSV, `timestamp [ns], w_x, w_y, w_z, a_x, a_y, a_z`, header optional. The two sensor
  triplets are told apart by which one carries gravity, so accel-first files work too. Timestamps in ns, us, ms or s.
- Output: `timestamp tx ty tz qx qy qz qw`, space separated, timestamp in seconds with every nanosecond digit of the
  input. **Exactly one pose per input row, in the input order; nothing is appended** (FAQ, 17 Nov): the blind 5 s
  window is already in the input as rows with unusable values, and those rows get the forecast. Duplicated input
  rows keep their duplicated timestamp.
- Robust to: missing rows (dropouts), duplicated or out-of-order timestamps, NaN / empty / non-numeric values, all-zero
  or frozen rows, spikes, Windows line ends, extra columns, a missing header. Whatever happens inside the solver, the
  output always has one finite pose per input row.

## Resources

Limits: 8 MB peak memory (FAQ: measured at the algorithm / heap level), 20 ms per second of IMU, single thread.

| Input | Peak heap of the algorithm | Time per second of IMU (whole process) |
|---|---|---|
| EuRoC flights, TUM-VI rooms (200 Hz, 1.4-3 min), organisers' degradation | 2.5-4.8 MB | 3.5-6.1 ms |
| Long walks, 5-24.5 min, and a synthetic 59 min file | 6.1-7.3 MB | 0.6-2.8 ms |
| 1 kHz IMU | 5.4 MB | 5.8 ms |

Peak heap: `tracemalloc` peak while the estimator reads, solves and writes, after Python and NumPy are loaded
(NumPy reports its array buffers to `tracemalloc`). Time: wall time of the whole `python3 main.py` run, interpreter
start included. Both are recorded in `metadata.json` by `tools/measure.py`.

How memory stays bounded: rows are processed in pieces of 2048, the solver works on 8-row blocks with a float32
Jacobian, poses are written as they are made, and timestamps are re-read from the file at write time. A file faster
than about 40 000 rows per solve is averaged into cells as it is read (spikes, frozen and invalid rows left out). The
one least-squares solve has caps that only bind on files longer than about 3 minutes: at most 24 knots per bias curve,
200 residual rows and 20 dropouts with their own velocity step (shorter extra dropouts are bridged).

## What the estimator does

The whole file is read before any pose is written, so slow sensor errors are solved for instead of filtered out.

1. **Repair.** Rows go on a regular time grid. Spikes (far from the median of 5 rows) are removed, frozen and invalid
   rows dropped, short holes interpolated (0.08 s handheld, 0.25 s drone); longer holes stay dropouts.
2. **Platform.** Rotor vibration in the accelerometer tells a drone from a handheld device.
3. **Gyro calibration.** A rest period in the file if there is one. A drone without one keeps a declared calibration
   only if a bias estimate from the file itself (4 s windows, across the up direction) agrees within 0.03 rad/s.
4. **Self-calibration.** One Levenberg-Marquardt solve over the whole file finds gyro and accelerometer bias curves,
   a tilt for each stretch between dropouts and a velocity step at each dropout, such that the 5 s average velocity
   and the position stay small (drones: such that velocity agrees with rotor drag). How fast the bias may change is
   chosen per file by Bayesian evidence.
5. **Shrinkage.** A self-check on the slow speed (and, for drones, on the evidence-chosen drift rate), then a gain per
   octave band of velocity.
6. **Blind 5 s.** Hold, plus a damped share of the last velocity. A handheld file fits that share on its own earlier
   motion; a drone whose rotors are still on descends 0.25 m over 3 s.

No network, no learned weights, no sequence-specific values. Method, constants, held-out results and known
weaknesses are in the repository README.
