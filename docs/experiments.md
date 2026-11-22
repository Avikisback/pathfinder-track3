# Track 3 "Path Finder": what beats a stationary output on IMU-only position (experiments)

Status: finished. Earlier sections came from the interrupted thread; the last section supersedes them.

How to read this file:
- "Measured" means run here, by scripts in `E:/clipping-pipeline/xr_track3/experiments/scripts/`, with logs in `experiments/logs/`. Source tags below name the script or log.
- Table cells are mean/worst over rooms, degradation seeds 1-3, and both gap modes, unless stated otherwise.
- Dev rooms are 1-5 (leave-one-room-out). Room 6 is locked away until the end, chosen before any of its data was looked at. Only its row counts and duration were read, for the integrity check.
- "full" is `tools/degrade.py` as written. "mild" is our gentler guess: noise x2 rather than x5, bias at a quarter, same gaps. "clean" is only the 5 s failure. All of these are guesses, not the organisers' script.

## Setup and harness

### Takeaway
All six rooms and the 11 EuRoC sequences are on disk. The scoring code reproduces evo, and the degradation code reproduces `tools/degrade.py` exactly.

### Cited Findings
- IMU rows with header: room 1 28,123; room 2 28,733; room 3 28,125; room 4 22,213; room 5 28,382; room 6 26,282. Curl and tar exit codes were 0 for rooms 2, 4, 5, 6 — `logs/download.log`.
- In-memory degradation equals `tools/degrade.py` in both gap modes on room 1 seed 2, max absolute difference 0 — `logs/00_check.log`.
- Own ATE matches evo to 4 decimals for SE(3) and origin alignment, and attitude likewise, on two real baseline outputs. RPE then differed by 0.004 m and 0.27 deg because own code used all pairs and evo uses consecutive pairs. Own code was changed to consecutive pairs afterwards — `logs/00_check.log`.
- TUM-VI ground truth is already in the IMU frame: fixed rotation 0.08-0.22 deg, clock offset 0 ms, gyro correlation 0.957-0.993, on rooms 1-5. It has holes of up to 2.49 s (room 2), which labels skip — `logs/00_check.log`.
- RMS radius (the stationary SE(3) ATE before the tail) rooms 1-5: 1.00, 1.22, 1.25, 1.08, 0.93 m. Mean speed 1.04, 0.98, 0.96, 0.62, 0.93 m/s — `logs/00_check.log`.
- EuRoC copy: IMU rows MH01 36,821; MH02 30,401; MH03 27,009; MH04 20,321; MH05 22,722; V101 29,121; V102 17,101; V103 21,501; V201 22,801; V202 23,491; V203 23,371. Ground truth rows are similar except V101 at 2,896 (20 Hz). It is a third-party copy of the data (Air-IO release), checked by the coordinator against the deck's sample timestamp and ORB_SLAM3's public ground truth.

### Inferences
- WSL shuts its VM down when no `wsl.exe` client stays attached, which killed background jobs: uptime had reset to 35 s after one loss. Long jobs must stay attached to a client. Six concurrent GPU processes also ran out of CUDA memory; three at a time is fine.

### Gaps
- None for the harness.

## H3: can attitude be recovered offline under injected bias drift? (dev rooms, no training)

### Takeaway
Partly, and this is the biggest attitude result so far. A whole-file gyro-bias fit against gravity with a strong prior brings degraded attitude (tail excluded) from 58.6 deg to about 35 deg on the full schedule, and tilt from 9.8 to 5.2 deg. With a loose prior it is much worse than doing nothing, because body-z bias is barely observable on a nearly level handheld rig. Heading lost across gaps remains the main residual, since `rpe_r` stays near 9 deg.

### Cited Findings
Attitude under origin alignment, deg, blind tail excluded, 75 files on dev rooms — `logs/12b.log`:

| Method | clean: attitude, tilt, RPE 1 s rot | mild: attitude, tilt | full: attitude, tilt, RPE 1 s rot |
|---|---|---|---|
| Baseline tilt filter | 2.15/2.75, 1.33/2.08, 0.99/1.13 | 61.7/92.9, 9.6/29.5 | 58.6/87.1, 9.8/30.2, 11.7/20.6 |
| No bias fit, two-sided gap rate, no tilt correction | 1.60/2.52, 0.68/1.07, 0.67/0.86 | 54.7/98.8, 39.4/97.1 | 56.9/98.1, 39.1/96.5, 11.2/21.3 |
| Bias fit, prior 0.003 rad/s, noise 0.3, 60 s knots, tilt residual 5 s | 2.96/4.20, 2.36/4.15, 0.80/0.97 | 34.3/80.0, 5.3/25.6 | 35.0/84.0, 5.2/22.2, 9.0/20.7 |
| Bias fit, prior 0.01, noise 0.1 | 6.63/9.28, 2.77/4.19, 1.01/1.58 | 35.6/83.0, 5.8/23.3 | 35.8/87.5, 5.6/20.7, 9.1/20.2 |
| Bias fit, prior 0.03, noise 0.03 | 74.8/99.1, 3.66/5.14, 3.13/5.07 | 73.9/143.1, 11.1/36.0 | 76.6/130.9, 11.1/47.7, 10.1/22.6 |

- First attempt, prior 0.05 rad/s and residual noise 0.03 rad: 83 deg on CLEAN data including the tail, versus 9 deg for the baseline — `logs/12_attitude.log`. Negative result, kept as a warning.
- With the 5 s tail included, clean attitude is 9.1/19.7 deg and 1 s rotation RPE is 6.4/12.8 deg, so the tail alone dominates those RMS numbers on clean data — `logs/12b.log`.
- Other drift shapes used the first prior settings and are not valid for the strong-prior method — `logs/12_attitude.log`. They need rerunning.

### Inferences
- The strong prior costs about 0.8 deg on clean data (2.15 to 2.96) and gains about 24 deg on the full schedule.
- Prior and noise settings here were picked on the same five rooms, so these numbers are development numbers, not a clean test.
- On TUM-VI, a body-z bias error of 0.01 rad/s integrates to 80 deg in 140 s and gravity barely constrains it. That matches the failure above. EuRoC behaviour needs its own test.

### Gaps
- Tested on TUM-VI only so far. The EuRoC gyro bias (about 4.6 deg/s) is a different regime.
- Brossard-style gyro denoiser: not yet run.

## H1 and H2 on TUM-VI: does a velocity net beat stationary? (dev rooms, leave-one-room-out)

### Takeaway
Yes on the mean, narrowly under degradation and clearly on clean data, but not in the worst case. The non-causal 2 s net gives 1.03 m against stationary 1.09 m on the full schedule, and its worst file is worse (1.42 against 1.25 m). On clean data it reaches 0.84 m against 1.10 m. RPE over 1 s is the solid win, 0.70 against 0.92 m, on every file. Shrinkage is essential, since the unshrunk net scores 1.97 m. The learned uncertainty (H2) adds essentially nothing to ATE here.

### Cited Findings
Centred 2 s net, 81,542 parameters, baseline tilt attitude; shrinkage settings for each room chosen on the other rooms. Mean/worst, m — `logs/21_tum.log`:

| Schedule | Stationary ATE SE3 | Baseline ATE SE3 | Net no shrink | Net fixed tau | Net sigma-driven | Stationary ATE origin | Net ATE origin | Stationary RPE 1 s | Net RPE 1 s |
|---|---|---|---|---|---|---|---|---|---|
| clean | 1.096/1.250 | 0.987/1.197 | 2.078/3.622 | 0.835/1.049 | 0.837/1.060 | 1.238/1.569 | 1.071/1.412 | 0.904/1.122 | 0.664/0.953 |
| mild | 1.093/1.250 | 1.693/3.652 | 1.934/3.923 | 1.028/1.427 | 1.027/1.417 | 1.234/1.573 | 1.192/1.776 | 0.919/1.158 | 0.703/1.044 |
| full | 1.093/1.250 | 1.706/3.668 | 1.970/3.953 | 1.034/1.422 | 1.034/1.432 | 1.234/1.573 | 1.198/1.769 | 0.919/1.158 | 0.702/1.049 |

- Beats stationary on 85% of files for ATE SE3, 82% for ATE origin, 100% for RPE 1 s — `logs/21_tum.log`.
- Full schedule by room, ATE SE3 net against stationary: 0.98/1.00, 1.10/1.21, 1.24/1.25, 1.01/1.07, 0.84/0.93 m — `logs/21_tum.log`.
- Selected time scale is tau about 2 s in almost every fold, a high-pass that leaves position changes only over a few seconds. ATE SE3 by tau (alpha 0, all schedules, sensitivity only): 1 s 1.013, 2 s 1.002, 4 s 1.039, 8 s 1.117, 16 s 1.240, 40 s 1.513, 80 s 1.760 m — `logs/21_tum.log`.
- Multiply-adds are 731,488 per forward pass at a 2 s window, so 7.3M per second of IMU at 10 Hz output — `logs/center2s.log`.

### Inferences
- The gap between clean (0.84 m) and degraded (1.03 m) suggests attitude damage is costing about as much as the net gains. Better attitude is the next lever on TUM-VI.
- Tau near 2 s depends on the room: it discards all position information beyond a few seconds, and works partly because a person in a 4 m room keeps turning around. This result is marked as dependent on the bounded room.
- Selection of settings used the five dev rooms, so these are development numbers until room 6 is opened.

### Gaps
- Causal net and segment-loss comparisons are still being scored. EuRoC is the next section.

## EuRoC: references, still periods and gyro bias

### Takeaway
Stationary scores 3.44 m mean and 6.89 m worst under SE(3) alignment, and the current baseline is worse on the mean (4.10 m) on the full schedule. Bias is large and very stable across sequences, and 9 of 11 sequences do contain a still second, but only MH05 and V103 start still, and V102 and V203 have none.

### Cited Findings
Measured on 11 sequences, seeds 1-3, both gap modes, mean/worst, with `tools/degrade.py` confirmed identical on MH02 — `logs/13_euroc_att.log`:

| Schedule | Stationary ATE SE3 | Baseline ATE SE3 | Stationary ATE origin | Baseline ATE origin | Stationary RPE 1 s | Baseline RPE 1 s | Baseline attitude origin | Stationary blind rel. to failure | Baseline blind rel. to failure |
|---|---|---|---|---|---|---|---|---|---|
| clean | 3.446/6.879 | 4.244/6.795 | 4.415/8.077 | 5.681/8.182 | 0.763/1.188 | 1.051/1.404 | 48.9/119.0 deg | 0.334/0.719 | 0.930/2.169 |
| mild | 3.439/6.885 | 4.134/6.763 | 4.396/8.077 | 5.590/8.247 | 0.793/1.419 | 1.078/1.530 | 54.1/135.6 deg | 0.334/0.719 | 0.877/1.809 |
| full | 3.439/6.885 | 4.097/6.735 | 4.396/8.077 | 5.599/8.407 | 0.793/1.419 | 1.078/1.529 | 56.1/132.9 deg | 0.334/0.719 | 0.877/1.940 |

- Still seconds by the `main.cpp` test (gyro std under 0.01 rad/s): MH01 13 s from 22 s, MH02 2 s from 27 s, MH03 3 s from 11 s, MH04 2 s from 12 s, MH05 4 s from 0 s, V101 1 s at 144 s, V102 none, V103 4 s from 0 s, V201 2 s at 111 s, V202 1 s at 116 s, V203 none. — `logs/13_euroc_att.log`.
- Ground-truth gyro bias is (-0.0022, 0.0223, 0.0778) rad/s mean, std (0.0005, 0.0018, 0.0018) over sequences, V2 slightly higher. The mean of still seconds is within 0.0022 rad/s of it on every axis where a still second exists. — `logs/13_euroc_att.log`.
- Body x points up on EuRoC, and it carries the smallest bias, about -0.002 rad/s. — `logs/13_euroc_att.log`.
- Ground truth frame agrees with the IMU to 0.04-0.91 deg, and the clock to within 15 ms. — `logs/13_euroc_att.log`.
- The first version of the gravity-only bias fit got the two horizontal body axes to about 0.01 rad/s or better but left a consistent +0.026 rad/s error on the up axis, because a weak direction leaked in during early passes. It was fixed afterwards by keeping only well-constrained directions. — `logs/13_euroc_att.log`.

### Inferences
- `main.cpp` searches the whole file and will find a still second on 9 of 11 sequences, which is why its attitude is much better than the stationary row's no-bias filter. V102 and V203 fall through to zero bias. Degradation noise could also push a marginal second over the 0.01 threshold.
- A training-set sensor calibration is a credible fallback, since bias varies by only about 0.002 rad/s across sequences. It is a dataset constant rather than a per-sequence one, so it is within the integrity rules if declared, but it only holds for that sensor.

### Gaps
- Velocity net on EuRoC and cross-domain results follow below when scored.

## EuRoC attitude under drift, first round

### Takeaway
A declared sensor calibration (mean bias from other sequences or other environments) is by far the best single fix: 3 deg clean and 28 deg on the full schedule, against 48 and 59 deg for a `main.cpp`-style still test. Gravity alone cannot recover the bias. Most remaining error comes from injected drift and gap heading.

### Cited Findings
Attitude under origin alignment, deg, blind tail excluded, 143 files, mean/worst — `logs/13_euroc_att.log`:

| Method | clean: attitude, tilt | mild: attitude, tilt | full: attitude, tilt, RPE 1 s rot |
|---|---|---|---|
| Gyro only, no bias | 118.5/135.6, 85.4/104.2 | 119.2/137.2, 87.2/107.3 | 119.2/138.3, 86.5/108.7, 6.4/13.6 |
| Tilt filter, no bias | 111.5/142.4, 38.8/96.1 | 108.8/144.5, 37.3/98.5 | 108.4/144.7, 37.2/103.8, 5.8/13.4 |
| Tilt filter, calib other seqs | 3.2/8.0, 1.7/3.1 | 22.2/75.3, 3.6/11.3 | 30.8/84.0, 4.4/11.8, 4.0/13.0 |
| Tilt filter, calib other envs | 3.5/7.9, 2.2/4.5 | 22.3/75.8, 3.9/11.3 | 30.9/84.4, 4.6/11.8, 4.0/13.0 |
| Offline fit, calib other seqs | 3.0/7.3, 1.3/2.2 | 19.2/72.0, 2.2/7.4 | 27.9/79.3, 2.5/7.3, 3.3/10.3 |
| Offline fit, calib other envs | 3.0/6.6, 1.4/2.2 | 19.3/73.0, 2.2/7.6 | 28.0/80.0, 2.5/7.2, 3.4/10.3 |

- Gravity-fit runs in that log used a faulty first version and are not quoted. With the corrected version, the body-x error stays near +0.026 rad/s and body-z near -0.01 rad/s on most files, which is exactly the component of true bias along the mean thrust axis, about 0.029 rad/s. On handheld TUM-VI the same fit errs by up to 0.05 rad/s on clean data. — `logs/30_att_v2.log`
- The first hardened rest test (gyro std within 1.25x of its own third-difference noise) was too strict. On clean files it found rest only on MH05, and found fewer blocks than the `main.cpp` threshold. — `logs/30_att_v2.log`

### Inferences
- At 0.029 rad/s the missing up-axis bias is about 1.7 deg/s, which is several full turns' worth of yaw in a file. So with no rest period, IMU alone cannot give EuRoC yaw without a calibration prior.
- A platform calibration is legitimate only if the rotor signature is what selects it and the entry declares it. It will fail on any different airframe or sensor.

### Gaps
- Second round with a relaxed rest test is running. See the next sections.

## H4: small learned linear filter on integrated accel (position-domain version)

### Takeaway
It never loses to stationary on TUM-VI, but by very little, and on EuRoC it is effectively level with stationary. This is a position-domain NNLS on nine bands. A velocity-domain band filter in the sibling notes reports much larger gains on clean data, so this implementation is not the bar.

### Cited Findings
Mean/worst, m, against stationary and baseline, attitude from the then-current automatic bias — `logs/14_linear.log`:

| Fold | Schedule | Stationary ATE SE3 | Linear ATE SE3 | Linear ATE origin (stationary) | Linear RPE 1 s (stationary) |
|---|---|---|---|---|---|
| TUM rooms LORO | clean | 1.096/1.250 | 1.053/1.246 | 1.210/1.572 (1.238/1.569) | 0.876/1.121 (0.904/1.122) |
| TUM rooms LORO | full | 1.093/1.250 | 1.034/1.247 | 1.213/1.573 (1.234/1.573) | 0.856/1.157 (0.919/1.158) |
| EuRoC LOSO | clean | 3.446/6.879 | 3.367/6.828 | 4.444/8.092 (4.415/8.077) | 0.625/1.016 (0.763/1.188) |
| EuRoC LOSO | full | 3.439/6.885 | 3.480/6.858 | 4.695/10.072 (4.396/8.077) | 0.695/1.299 (0.793/1.419) |
| EuRoC LOEO | full | 3.439/6.885 | 3.573/6.958 | 4.860/12.793 (4.396/8.077) | 0.746/1.477 (0.793/1.419) |
| Train TUM, test EuRoC | full | 3.439/6.885 | 3.427/6.878 | 4.401/8.079 (4.396/8.077) | 0.791/1.417 (0.793/1.419) |
| Train EuRoC, test TUM | full | 1.093/1.250 | 1.473/6.716 | 2.280/12.010 (1.234/1.573) | 0.714/1.377 (0.919/1.158) |

- Share of files beating stationary, ATE SE3: TUM 1.00, EuRoC LOSO 0.87, LOEO 0.76, TUM to EuRoC 1.00, EuRoC to TUM 0.62. ATE origin: 0.51, 0.18, 0.16, 0.36, 0.38. — `logs/14_linear.log`

### Inferences
- A filter trained on EuRoC can blow up on TUM-VI (6.7 m worst), because the fitted gain on the top band grows very large on EuRoC (39-61) where vibration dominates. Gains should never cross platforms unchecked.
- The TUM-trained filter on EuRoC stays safe only because its gains collapse towards zero output.

### Gaps
- Velocity-domain band gains inside folds are in the end-to-end section.


## Coordinator trials, 7 Nov: new ideas tested directly, and the frozen entry

Status: finished. Everything below was run on this machine, scripts in `E:/clipping-pipeline/xr_track3/experiments/scripts/`, numbered 40-58, logs in `experiments/logs/`, stored results in `experiments/results/`. All numbers are mean/worst unless stated. The fast scorer was checked against real evo on MH04 and room 3 and matched to 4 decimals for ATE `-a`, ATE `--align_origin` and RPE at 1 s (`55_final.py evo`).

### Takeaway
The best entry found needs no neural network. A self-calibrating least-squares solve plus a drag model on drones plus a very small filter beats standing still on every known schedule, on all 11 EuRoC sequences, on all rooms, and on a drift shape that played no part in any choice. A whole-trajectory network on top adds only about 0-10% on unseen environments, and more when the test environment was in training. An earlier version looked better still on known schedules, then failed badly on new drift shapes. That failure is why the final version looks the way it does.

### The method (new as far as this research found)
Script `energy2.py`, `46_drag_energy.py`, then `57_nonet.py`, `58_freeze.py`.
1. **Energy-minimising self-calibration.** Plain double integration explodes because of a few slow errors. A velocity that runs away is a very sensitive measurement of them. Solve for: accel bias spline, gyro bias spline, one tilt per stretch between gaps, a velocity change per gap, by least squares so that 5 s-averaged velocity and position stay small. Levenberg-Marquardt, about 12 iterations. It needs only "stays bounded", with no map or room size. Tilt per segment is seeded exactly from gravity, so large rotations across a gap are fine.
2. **Drag aiding on drones.** The velocity rows compare with drag velocity (body-plane accel, low-passed at 0.1 Hz, divided by k=0.187, rotated by current attitude) instead of zero. Rotor signature in the accelerometer switches this on, which identifies the sensor rig and not a sequence.
3. **Position prior that follows the file.** Width = 1.5 x this file's own estimated spread, clipped to 1-20 m, so big and small rooms both work.
4. **Filter.** Drones: self-adjusting Wiener per octave band (`gain = 1 - error power / measured power`), error power from other environments. Handheld: fixed median band gains from other rooms.
5. **Self-check and fallback.** 5 s-averaged velocity RMS above 0.9 m/s on drones or 0.4 m/s on handheld means drift won. Fall back to a short high-pass (8 s, 1 s) that keeps only fast motion.
6. **Blind 5 s.** Near hold: `0.5 * v * 0.35 * (1 - exp(-s/0.35))`.

### Results for the frozen entry
Configuration frozen before the `steps` column was run. "Other shapes" were used to choose the fallback thresholds, so only `steps` is a clean unseen-shape test. EuRoC is leave-one-environment-out, TUM is leave-one-room-out. `tools/degrade.py` is still the team's guess, not the organisers' script. Source: `results/nonet_frozen.pkl`, `58_freeze.py`.

| Drift shape | EuRoC ATE SE3 (stationary) | EuRoC RPE 1 s (stationary) | TUM ATE SE3 (stationary) | TUM RPE 1 s (stationary) | Room 6 ATE (stationary) |
|---|---|---|---|---|---|
| clean (5 s failure only) | 2.51 (3.45), worst 4.44 | 0.39 (0.76) | 0.50 (1.10), worst 0.91 | 0.23 (0.90) | 0.57 (0.99) |
| mild | 2.62 (3.44), worst 4.42 | 0.47 (0.79) | 0.73 (1.09), worst 0.99 | 0.37 (0.92) | 0.73 (0.99) |
| full | 2.59 (3.44), worst 4.51 | 0.46 (0.78) | 0.75 (1.09), worst 1.02 | 0.40 (0.91) | 0.74 (0.99) |
| bursts of noise, many short gaps | 2.46 (3.45), worst 4.44 | 0.46 (0.76) | 0.73 (1.10), worst 1.05 | 0.35 (0.90) | 0.71 (0.99) |
| **steps (new shape, clean test)** | **2.84 (3.45), worst 5.32** | 0.52 (0.82) | **0.97 (1.09), worst 1.16** | 0.59 (0.93) | 0.79 (0.99) |
| random walk | 3.72 (3.45), worst 5.48 | 0.80 (0.83) | 0.98 (1.09), worst 1.10 | 0.65 (0.93) | 0.91 (0.99) |
| sine | 4.05 (3.45), worst 5.85 | 0.78 (0.76) | 0.95 (1.10), worst 1.07 | 0.61 (0.90) | 0.90 (0.99) |

- Share of files beating stationary on ATE: EuRoC 1.00, 1.00, 0.97, 1.00, 0.91, 0.36, 0.18 in table order. TUM 1.00, 1.00, 0.93, 0.90, 0.80, 0.90, 1.00. Room 6 all 1.00.
- ATE origin alignment, EuRoC: 3.50, 3.65, 3.73, 3.56, 4.41, 5.92, 5.58 against stationary 4.40. TUM: 0.74, 0.98, 1.00, 0.98, 1.25, 1.21, 1.14 against 1.24.
- Blind 5 s relative to failure, EuRoC about 0.33-0.39 against stationary 0.33; TUM about 0.49-0.52 against 0.49. So the blind score is about level with holding position. It is not a win.
- Room 6: an earlier version of the pipeline had looked at it once, so these are not a pristine first look.
- Attitude: 26-34 deg RMS on `full` remains, nearly all heading. No method found recovers heading from an IMU alone under drift plus gaps.
- Honest weakness: on random-walk and sine drift at this severity (gyro about 0.012-0.02 rad/s, accel 0.12-0.2 m/s^2), drones end 8-17% worse than standing still on ATE. Accel drift enters the drag velocity directly (0.2 m/s^2 is about 1 m/s of error), and accel bias cannot be separated from tilt there.

### Network comparison
Whole-trajectory bidirectional GRU (77k parameters, trained directly on aligned ATE and RPE, `50_seqnet.py`, `54_seq4.py`). Uses solver velocity, drag and gravity as inputs, ends in the self-adjusting filter. Known schedules only, ATE SE3 clean/mild/full:
- Solver with the stiff 60 s drift model, no net, EuRoC unseen environment: 2.27/2.42/2.51.
- Same plus net, EuRoC unseen environment: 2.23/2.41/2.46.
- Same plus net, EuRoC leave-one-sequence-out, so same building in training: 1.98/2.21/2.27, worst 3.32/3.80/3.92. Best number measured, but it relies on the environment being seen.
- Earlier window TCN net: 2.79/2.96/3.04 unseen environment, 2.18/2.50/2.65 leave-one-sequence-out. TUM 0.68/0.82/0.82.
- TUM unseen room, net stack 0.60/0.73/0.74, against no-net frozen entry 0.50/0.73/0.75. RPE on TUM is worse with the net: 0.42-0.45 against 0.23-0.40.
- Ensembling three nets did not help (2.72 against 2.69 at that stage). Gyro TCN made attitude worse on TUM (33 to 48 deg) and did not help on EuRoC (26.2 to 25.5 deg).
- The stack was built on the stiff drift model, so it inherits the failure on random-walk and sine drift (6.92 and 9.24 m on EuRoC, 2.75 and 4.27 m on TUM). It has not been retrained on the flexible model.

### Tried and rejected, with numbers
- Yaw-rate mode tracking (most-common turn rate as gyro bias): attitude on clean TUM went from 3.0 to 95 deg. Rejected.
- Velocity-aided tilt: TUM clean 0.39-0.44 against plain fit 0.34, no gain under drift. Rejected.
- Two-sided low-gain tilt filter: tilt 13-38 deg under drift, worse than gravity fit at 3-7 deg. Rejected.
- Heading jump across gaps from velocity continuity: attitude 34.2 against 36.4 deg TUM, 19.4 against 17.4 deg EuRoC, mixed. Dropped.
- Relaxing the velocity constraint with rotors off: 2.36 against 2.35 m, no gain.
- Stiff drift model alone: excellent on known schedules (EuRoC 2.20-2.34), but 10.99 and 14.74 m on random-walk and sine. Replaced by the 15 s knot model, 2.25-2.39 on known, 4.28 and 8.65 raw on those.
- Plausibility clamp on band power: helped hostile shapes on handheld, cost about 10% on known shapes on drones. Replaced by the self-check.

### Inferences
- The structure of the winning idea is that drift is solved for rather than filtered, so slow real motion survives. Band gains on the solver's velocity stay near 1 down to about 1/32 Hz on clean EuRoC, against a corner near 1/16 Hz before.
- Settings chosen on the same data: k=0.187 and the sensor bias calibration use other sequences or all sequences, position prior factor, knot spacing, thresholds. Expect a little optimism except on `steps`.
- The organisers' script is promised. When it arrives, rerun `57_nonet.py` and `58_freeze.py` on it. If its drift is gentle, use the stiffer model and consider the network.

### Gaps
- Not ported to C++. Memory and time are estimates: about 80-130 parameters, 900 rows, dense normal equations, 12 iterations, so about 2 ms per 1 s of IMU and 1-2 MB on a 150 s file. Long TUM-VI files (up to about 30 min) must be solved in overlapping chunks, because dense memory grows with the square of knot count. Measure this before relying on it.
- Frozen-row encoding of gaps (repeated samples) is handled in `main.cpp` but not in this Python harness.
- Mid-file IMU failure was only checked for forecast shape, not end to end.
- Non-room TUM-VI sequences not tested: corridors, magistrale, outdoors, slides, where motion is not bounded the same way. If the hidden set includes them, position prior and self-check must be re-checked. A long corridor walk could break "stays bounded".
- EuRoC data came from the Air-IO authors' release, checked against the deck's sample timestamp and ORB_SLAM3's public ground truth on MH01 only.

## Round 2, 7 Nov (scripts 60-73, logs in experiments/logs, research notes in "Samsung XR Track 3 new methods")
- Error budget: drones' error is horizontal velocity below 1/16 Hz; the true attitude does not help (65_oracles). Handheld error sits at 4-16 s periods.
- Rejected: coupling the bias into the drag target (63), the whitened learned-spectrum cost (gls.py, 64), two-input band fusion (69), compactness yaw at gaps (66, 69), gait PCA direction (66), drag k other than 0.187 (68), b0 from the file's own rest (73).
- Best new: random-walk drift on 8 s knots with the drift rate per file from evidence (rw_evidence.py, variant L8rw_eb1), self-check thresholds 1.0/0.5 (72_freeze2). On 5 unseen degradations (robust_shapes.py) it beats the old frozen entry in 8 of 10 cells.
- Open bug: rotated sensor axes ("rot") make drones worse than standing still (3.79 vs 3.45), probably because the declared b0 is in the sensor frame. Fix not found yet.
- Harness: frozen, NaN and zero rows are now invalid in common.to_grid.
- Rotation bug FIXED (73_rotfix.py): keep the declared drone gyro calibration only if the rest on the ground agrees within 0.02 rad/s, else use the rest reading. EuRoC rot 3.79 -> 3.02 (stationary 3.45), hold 2.53, known shapes unchanged (2.22/2.35/2.32/2.49/2.95), randomwalk 3.07 -> 3.30, sine 4.19 -> 4.36.
- UNSEEN DATA (8 Nov, approved downloads; tools/get_unseen.py, tools/blackbird_to_euroc.py; 74_unseen.py, 75_blackbird.py):
  - TUM-VI corridor 1-5, magistrale 1-6, slides 1-3, outdoors 1-8 (91-1465 s, ground truth only at start/end in the mocap room). Mean ATE clean: stationary 0.88, old 0.68, new 0.63; full: 0.88 / 0.72 / 0.72. Beats stationary in 95-100% of files. "Stays bounded" did not break on long walks.
  - Blackbird (different quadrotor, 100 Hz, 10 flights 16-95 s, band gains from EuRoC, k unchanged): clean stationary 3.40, old 2.86, new 2.95; full 3.41 / 3.20 / 3.26. Old beats stationary on all files, new on 100% clean and 90% full.
  - Python solver time on long files reaches 170 ms per s of IMU (outdoors6): dense normal equations grow with knot count. The C++ port must use a banded or chunked solve.

## Round 3, 8 Nov: two drone upgrades, then the C++ port (scripts 76-77, ref_entry.py, compare_cpp.py, compare_long.py)
- Descent prior for the drone's blind 5 s (76_descent.py, on the entry's saved outputs, no re-solve). z(h) = -0.25 min(h/3, 1) m on
  drones whose rotors are still on at the cut. EuRoC blind_rel: known 0.334 -> 0.288, hostile 0.378 -> 0.349, robust 0.343 -> 0.309,
  all 132 files 0.352 -> 0.317; whole-file ATE unchanged. The declared value (0.25 m / 3 s, chosen by the researcher before this run)
  is the best or tied best in the sweep (0.15-0.35 m, 2-3 s). TUM rooms: never flagged as drone. Blackbird (76b): only the two fast
  'egg' flights are flagged as drones; effect there +-1% (they are cut mid-air). SHIPPED.
- Lever-arm + rotor-rate correction of the drag target (77_lever.py), constants k, r, d fitted per fold on the other EuRoC environments
  with the researcher's code: r = (0.015-0.021, 0.015-0.017, 0.102-0.104) m in every fold. Mean ATE over all 12 degradations:
  entry 2.296 known / 2.780 all; lever 2.292 / 2.760; lever + drag low-pass 0.2 Hz 2.252 / 2.710; 0.2 Hz alone 2.249 / 2.755.
  The 0.2 Hz cut-off (suggested by the notes in advance) gives the clean gain, the lever term the hostile one (randomwalk 3.15 -> 3.01,
  sine 4.62 -> 4.27). blind_rel 0.351 -> 0.343, RPE 0.504 -> 0.485. Blackbird drone files with EuRoC constants: 6.34 -> 6.29 (no harm).
  SHIPPED as lever + 0.2 Hz, constants refitted on all 11 flights: k 0.195, r (0.019, 0.016, 0.103) m, d (-0.013, 0.014, 0.030).
- Shipped entry = ref_entry.py (Python reference) = src/pathfinder.cpp (C++). Constants in src/constants.h (ref_entry.py export).
- C++ parity on 64 test files (EuRoC 11 x clean/full/rot/hold, TUM rooms 1-5 x clean/full/rot/noisy): worst position difference
  0.22 mm, mean ATE 1.9554 both. Static build: peak memory 2.3-4.0 MB, 1.6-5.3 ms per s of IMU (limits 8 MB, 20 ms).
- Memory design: the solver accumulates its normal equations block by block (the Jacobian is never stored); the drag target is
  filtered in pieces with margins (exact); files over 5 min are solved in 4 min windows every 3 min, aligned on heading in the
  middle of each 60 s overlap and crossfaded over 30 s.
- Long files (compare_long.py, 22 unseen TUM-VI walks): C++ windows 0.599 clean / 0.737 full against Python whole-file
  0.629 / 0.722 and stationary 0.881. Beats stationary on 22/22 clean, 19/22 full. Peak 6.2 MB, worst 10.3 ms per s (24 min).
- Edge cases (edge_cases.py): NaN rows, frozen block, 3 s timestamp hole, dead from the middle, 8 s and 25 s files, 100 Hz
  Blackbird: all run, one pose per row plus tail, finite, unit quaternions. Header-only input writes an empty trajectory.
- High rate (highrate.py, EuRoC resampled): 400 Hz 5.1 MB 6.7 ms/s; 1 kHz 4.2 MB but 43.6 ms/s, OVER the 20 ms limit. Fix not
  built: solve on a 200 Hz average, integrate the final attitude at full rate. (Resampling also hid the rotor vibration, so
  those files were read as handheld; that is an artefact of linear resampling.)
- Heading drift of the output attitude (att_check.py): tilt about 1 deg, but heading wanders 40-150 deg over a TUM room
  (origin-aligned attitude 25 deg TUM, 17 deg EuRoC on the known schedules). 78_gyroprior.py: anchoring the gyro-bias amplitude
  at 0.01 or 0.003 rad/s instead of 0.1 barely helps (TUM 25.1 -> 24.4 / 24.2 deg, EuRoC 16.8 -> 15.9 / 14.8 deg; ATE within
  +-0.02 m except randomwalk at 0.003, 3.01 -> 3.14). Even 0.003 rad/s of bias is 24 deg over 140 s, and the degraded files
  carry real injected gyro drift, so the heading cannot be pinned this way. Not shipped. If rotation is scored, give the output
  a separate attitude; check RPE, which uses the output attitude to express displacement in the body frame.
- FAST IMU FIX (scan.dec in pathfinder.cpp): above 300 Hz the solver runs on group means of D = round(fs/200) rows (group time
  = its last row); output rows integrate the full-rate gyro minus the solved bias from the previous group's solved attitude
  (slerp inside a group with an invalid row). Check (highrate2.py, highrate_long.py): each 200 Hz sample held D times with
  zero-sum jitter. ATE equals the 200 Hz run to 3 decimals (V101 1.611, MH03 full 2.228, room1 0.282/0.283, room3 full
  0.690/0.689, windowed corridor2 0.638/0.638 and 0.681/0.675). Drones are still detected. 1 kHz: 3.1-3.9 ms per s, 3.1-5.5 MB.
- Timing note: reading CSVs from /mnt/e (Windows drive over WSL 9P) dominates time at high rate (17 ms/s at 1 kHz vs 4.6 ms/s from
  /tmp). Native-disk timings: 200 Hz 1.3-2.4 ms/s, 24-min walk 2.8 ms/s, 6.2 MB peak.
- 200 Hz path unchanged by the fix: compare_cpp SUMMARY identical, outputs byte-identical (cmp) on a short and a windowed file.

## Round 4, 15 Nov: audit probes (scripts 79-81, logs 80_orc.log and 81_carryaxis.log)

- Mean body-frame velocity of the drone (79_dragoffset.py). The drag target subtracts the file mean of the in-plane force,
  which also removes the mean in-plane BODY velocity c. Truth: c = 0.04-0.37 m/s on EuRoC (mean 0.18, 6-54% of rms body
  speed). In an ideal drag path with true attitude, the missing c alone gives 2.46 m of horizontal error (true spread 3.33 m).
  One Blackbird flight (egg) has c = 5.3 m/s, all of its speed.
- Declared hover reading (79b_hover.py): h = mean accel + k c is NOT a stable platform constant. Its sd across flights is
  0.06 / 0.02 / 0.07 m/s2, above the k c signal of 0.035 m/s2. Leave-one-environment-out error of c: 0.52 m/s against 0.18
  for assuming zero, worse on 11 of 11 flights. Rejected.
- Oracle (80_orc.py): true c added to the drag target inside the entry solver. Clean 2.16 -> 2.09 m, known shapes
  2.252 -> 2.193, all 12 shapes 2.710 -> 2.675. Ceiling of about 3%, so this error source is closed.
- Per flight against standing still (cache v18, lev2): never worse on clean, mild, full, burst, hold, longgap, noisy.
  Worse on 8 of 11 for sine, 6 for random walk, 5 for rot, 2 each for steps and warmup. Per-file minimum of entry and
  standing still: sine 3.05 (entry 4.27), rot 2.65 (2.95), random walk 2.83 (3.01). Clean gains come mostly from MH03-05
  and V102; MH01, MH02, V101, V103, V201 are only 5-15% better than standing still.
- Attitude split by condition (78_gyroprior outputs, first-pose aligned RMS, tail excluded). Entry (anchor 0.1 rad/s):
  EuRoC clean 6.5, mild 19.1, full 24.8 deg; TUM clean 6.6, mild 34.5, full 34.2 deg. Anchor 0.01: EuRoC 4.9 / 18.4 / 24.5,
  TUM 2.9 / 35.6 / 34.7. Anchor 0.003: EuRoC 3.1 / 17.5 / 23.8, TUM 1.6 / 36.0 / 35.1. The earlier "barely helps" verdict
  averaged the three conditions. On clean files the tight anchor is 2-4 times better at an ATE cost within 0.02 m on the
  known shapes (random walk: EuRoC 3.01 -> 3.14, TUM 0.97 -> 1.01). Recommended if rotation is scored.
- Handheld carry axis (81_carryaxis.py): walking direction as a fixed device axis found per file, times gait speed. The file
  finds its own axis (0.3-12 deg from the truth-fitted axis, mean 4.7 clean), and the speedometer has RMSE 0.22 m/s. But the
  resultant length of body-frame velocity is only 0.37-0.83 (mean 0.52) and the direction error is 32 deg median, 77 deg at
  the 75th percentile (11 deg median only in room 2). With true speed, true attitude and the truth-fitted axis, the
  integrated path scores 6.8 m (standing still 1.10, entry 0.45). Blending its band below 1/32 to 1/8 Hz into the entry
  output gives 5.6-6.2 m. Rejected on this rig; untested on head-mounted data.

## Round 5, 15 Nov (later): deck V3 and the three pre-deadline fixes, tested on the real C++ entry

Tools: score_bin.py (any binary over the 64 test files, Blackbird or the long walks), score_diff.py, score_gaps.py.
Build flags in pathfinder.cpp: PF_GYRO_AMP, PF_GAP_BRIDGE, PF_GAP_TAU, PF_TAIL_GAIN_HAND. Log: logs/score_variants_15nov.log.
- Deck V3: Track 3 slides unchanged word for word. Dates moved: submission 22 Nov, shortlist 29 Nov, final demo 9 Dec,
  results 12 Dec.
- Default flags reproduce the 8 Nov binary byte for byte on all 64 files.
- SHIPPED, handheld blind tail = hold (PF_TAIL_GAIN_HAND 0): rooms blind 0.523 / 0.513 / 0.515 / 0.521 -> 0.491 / 0.491 /
  0.491 / 0.489 (hold 0.490), ATE unchanged, drones untouched. Python reference updated (ref_entry.py), room references
  regenerated. Parity: worst position difference 0.22 mm, mean ATE 1.9553 both, 3.80 MB peak.
- NOT SHIPPED, gyro anchor 0.003 rad/s (PF_GYRO_AMP=111111). Attitude on clean files 6.0 -> 3.0 deg (EuRoC) and 6.6 -> 1.6
  (rooms), but rotation RPE at 1 s is unchanged (0.18 -> 0.16, 0.68 -> 0.67 deg), mean ATE over 64 files 1.955 -> 1.967,
  and the drone blind tail gets worse on hold (0.290 -> 0.313) and rot (0.334 -> 0.373). The deck scores ATE and RPE, so
  there is no scored gain, and it is risky when the starting calibration is wrong.
- Dropouts measured for the first time (score_gaps.py). Displacement error since the start of the gap, entry / hold /
  straight line: EuRoC full 0.776 / 1.069 / 0.788 m, room full 0.794 / 0.984 / 0.843, room noisy 0.873 / 0.987 / 0.896.
  Rotation across a gap, entry against no rotation: EuRoC full 18.5 against 38.6 deg rms, room full 51.5 against 69.5.
- NOT SHIPPED, other rotation bridges in gaps: zero rate and two-sided fade (0.2, 0.35, 0.5 s) are all worse than the
  current straight line (room full ATE 0.687 -> 0.879 / 0.835 / 0.809 / 0.787; attitude 34 -> 59 / 48 / 42 / 39 deg).
- NOT DONE: rerun of the long walks and Blackbird with the new build, resource timing on a native disk, a test with
  bias-compensated EuRoC gyro (the deck calls the input compensated), README and WALKTHROUGH text.

## Round 6, 16 Nov: calibration check, self-fitted handheld tail, submission bundle

Scripts: wbias.py, 84_wbias_survey.py, make_comp.py, make_kinds.py, 85_tail_lab.py, score_bin.py (sets tests, comp, kinds,
long, bb), score_gaps.py. Logs: score_unseen_15nov.log, score_v3_16nov.log, final_parity_16nov.log.
- CORRECTION to Round 5: the hold tail is not a clean win. Rooms (5 endings): hold better on 5 of 5 (0.523 -> 0.491).
  Unseen walks (21 endings with truth in the tail): the old damped tail is better on 14 (0.849 against 0.891 for hold).
- SHIPPED, self-fitted handheld tail (Track::fit_tail, ref_entry.fit_tail). Tail = p_last + G tau (1 - exp(-s / tau)) v0.
  G in [0, 1] and tau in {0.35, 0.7, 1.5, 3} are fitted on the file's own estimated path: cuts every second, velocity of
  the last 0.5 s against the displacement over the next 5 s. Rooms 0.495 (hold 0.490, old 0.518), walks 0.840 (hold
  0.891, old 0.843). Fitted G: median 0.07 in rooms, 0.48 on walks. Lab variants with a doubled gain or a fixed tau of
  0.7 do better on walks (0.79-0.81) and worse in rooms (0.50-0.53); not used, they would be tuned on the walks.
- NOT SHIPPED, the same fit on drones (PF_TAIL_DRONE=2): EuRoC blind 0.323 / 0.342 / 0.333 against 0.284 / 0.288 / 0.290.
  Flights end by landing. A plain hold plus descent (PF_TAIL_DRONE=0) is equal to the current tail (0.285 / 0.290 / 0.286).
- FOUND: bias-compensated gyro input breaks drones without a rest block. The deck calls the input compensated; public
  EuRoC is not. With the true bias subtracted (make_comp.py), clean ATE 2.160 -> 2.633 and full 2.318 -> 3.129, attitude
  51 deg, V101 3.43 m against 1.85 for standing still. Cause: hardened_rest finds no rest on V101, V102, V201, V202,
  V203, so the declared calibration (0.08 rad/s) is trusted, and the whole-file gravity fit cannot recover from that.
- SHIPPED, windowed calibration check (window_bias / wbias.py), drones without a rest block only. 4 s windows, up
  direction carried by the gyro from start to end, linear least squares in the bias, 3 passes, Huber weights; the
  component along the mean accel direction is dropped. Distance of that estimate from the declared calibration, across
  the up direction: normal files 0.000-0.022 rad/s (clean, full, hold), rotated axes 0.035-0.064, compensated
  0.061-0.083. Threshold 0.03. Estimate error on compensated files without rest: under 0.005 rad/s, V203 0.016.
  Results: compensated clean 2.633 -> 2.234, full 3.129 -> 2.350; rot 3.044 -> 2.672; all other eleven shapes unchanged
  to three decimals (make_kinds.py, 88 files). The estimator is poor on handheld files (0.02-0.10 rad/s error, the hand
  accelerates while it turns), so it is not used there.
- Final parity (compare_cpp.py): 64 files, worst position difference 0.22 mm, mean ATE 1.8915 for both. Fast-IMU check
  and edge cases pass. Native disk: rooms and flights 0.7-1.7 ms per s and 3.1-4.1 MB, 24-minute walk 1.9 ms per s and
  6.4 MB, 1 kHz 3.7-4.1 ms per s.
- Shipped C++ on all twelve EuRoC shapes: clean 2.160, mild 2.334, full 2.318, burst 2.481, steps 2.758, random walk
  2.988, sine 4.220, hold 2.522, rot 2.672, warmup 2.817, longgap 2.430, noisy 2.526 (standing still 3.446).
- Submission: src/main.py (launcher; compiles and runs the C++, reads a folder or a zip, writes metadata.json),
  Dockerfile (not built here, no Docker in WSL), tools/make_submission.sh (bundle of slide 17, tested end to end on a
  zipped MH03 file: format check PASS, 3.4 MB, 1.46 ms per s).

## Round 7, 16 Nov (later): search for margin. Two things shipped, five rejected, Samsung headset run

Scripts: 86_endings.py, 87_selfcheck.py, 87b_rule.py, 87c_blend.py, 88_intrinsics.py, 89_gapyaw_oracle.py, make_monado.py,
tools/get_monado.py. Logs: score_v7_16nov.log, final_v8_16nov.log, monado_16nov.log, parity_v7_16nov.log.
- REJECTED, blind tail that moves home or to the centre (86_endings.py, ground truth positions). Rooms: hold 0.492,
  home 0.494, centre 0.492. EuRoC: home +7%, centre -6% (the entry already has -15% from the descent). Walks: home -22%,
  centre -32% with TRUE positions, but that needs the position after a long walk to 0.3 m. In rooms a hold is the floor.
- SHIPPED, drones only: drift-rate self-check. Diagnostics per file (87_selfcheck.py): evidence-chosen accel drift
  rate 0.02-0.11 on clean, mild, full, burst, hold, longgap, noisy, rot; 0.16 steps, 0.34 warmup, 0.55 random walk,
  0.87 sine. Rule: fall back to the high-pass if slow speed > 1.0, or rate > 0.3 and slow speed > 0.6. Picked on the
  seven design shapes, read off on the five untuned ones: design 2.751 -> 2.659, untuned 2.593 -> 2.595. Sine 4.220
  -> 3.505 (standing still 3.446), random walk 2.988 -> 3.054, warmup 2.817 -> 2.823. A blend of the two outputs or a
  threshold of 0.2 scores a little better on design shapes and worse on untuned ones; not used.
- SHIPPED, drones only: when the self-check falls back, the blind tail carries no velocity. Blind under random walk
  0.343 -> 0.302 and sine 0.379 -> 0.320 (hold 0.334).
- REJECTED for handheld: the same rule. Rooms 0.844 -> 0.839 over 11 shapes. On the long walks under full degradation
  nearly every file has a rate of 0.37-1.05, because real unbounded walking looks like fast drift to the evidence; ATE
  0.737 -> 0.743 and blind 0.834 -> 0.887. Handheld output is byte-identical to the build before this round.
- REJECTED, IMU scale and misalignment (88_intrinsics.py). Gyro matrix against mocap: diagonal -0.3%, one off-diagonal
  +0.8%, consistent across rooms (sd 0.1-0.2%). Applied leave-one-room-out: ATE clean 0.437 -> 0.524, full 0.686 ->
  0.733, while attitude against mocap improves 6.6 -> 3.6 deg. So the fit found the mocap-to-IMU frame rotation, not a
  sensor error. The accel fit is not identifiable this way (scale trades against gravity; -6 to -12% is an artefact).
- REJECTED, heading across gaps as the cause of the handheld dropout loss (89_gapyaw_oracle.py). With the TRUE turn
  across every gap applied to the rest of the path: rooms mild 0.681 -> 0.778, full 0.686 -> 0.789, longgap 0.802 ->
  0.832. Confirms the earlier oracle. The loss (clean 0.44 against 0.68 with three 2 s gaps, the same for mild and
  full) is velocity lost in the gap.
- Samsung Odyssey+ (Monado, 10 recordings, 1 kHz, never tuned on). cal = dataset-calibrated IMU, raw = uncalibrated.
  ATE entry / standing still: cal clean 0.763 / 0.858, cal full 0.838 / 0.856, raw clean 0.837 / 0.858, raw full
  0.871 / 0.856. RPE 1 s: 0.244-0.314 against 0.448. Attitude cal clean 3.1 deg; raw clean 56 deg, because the raw
  gyro has about 0.035 rad/s of bias about the vertical and only MOO05 has a detected rest. 3.8 MB peak.
- Margins vs standing still on other metrics (shipped build): first-pose ATE EuRoC 3.04-3.68 against 4.42, rooms
  0.68-0.94 against 1.24. RPE 1 s position EuRoC 0.26-0.35 against 0.76, rooms 0.18-0.30 against 0.90. RPE 1 s
  rotation EuRoC 0.2-3.1 against 17.9 deg, rooms 0.7-10.8 against 60.7 deg.

## Round 8, 17 Nov: organisers' script and FAQ, Python port, robustness and limits

Inputs from the organisers: `tools/degrade_imu.py` (seed 42) and the FAQ (`tools/Track3_FAQ_Responses.txt`): Python
only (standalone `main.py`), 8 MB at the heap level, one pose per input row and nothing appended (the blind rows are in
the file), dropouts are missing rows, batch allowed, evo SE(3) Umeyama. Test sets: `make_official.py` (`_sw` columns
swapped for the script and back, `_og` as is), `make_official_seeds.py` (seeds 1-3), `make_long.py` (rooms forward +
reversed + forward, 4.7-9.3 min, exact ground truth; reversal is real motion for handheld: specific force unchanged,
gyro negated), `make_tv_off.py` (the 22 TUM-VI walks, 1.5-24.5 min). Scorer: `score_off.py` (FAQ rules).
- Ablation of the official degradation (`make_official_abl*.py`): on handheld only the burst drops matter (rooms 0.765
  all, 0.476 without bursts); drones prefer bridging every hole, handheld at most 0.08-0.1 s. SHIPPED: platform-
  specific bridging (0.08 s handheld, 0.25 s drone) and row cleaning before the solve (C++ v10, then the port).
- PORT: `src/main.py`, Python + NumPy only, same method as C++ v10. Same aggregate scores (drone trajectories differ by
  20-26 cm rms from solver sensitivity; aggregates match). NumPy 1.26 and 2.5 give the same output to 1 um.
- Scores (seed 42): EuRoC `_sw` 2.247 / `_og` 2.448 (still 3.449, 11/11 better); rooms `_sw` 0.515 / `_og` 0.530
  (still 1.096, 5/5). Seeds 1-3 EuRoC 2.332 / 2.369 / 2.372, rooms 0.561 / 0.486 / 0.508. 0 timestamp mismatches.
- FIXED: singular normal matrix when a segment had no block of its own (room4 seed 1 wrote only a header): weak prior
  on tilts, guarded inverse, and a global safety net (any exception -> remaining rows hold the last pose).
- REJECTED, independent windows for long files (190 s windows): long rooms 1.21 m against 0.66 m for one solve and
  1.10 still; a 30 s warm-up overlap made it worse (2.9 m): short windows cannot pin the gyro bias (heading drifts
  21 deg in 48 s against the whole-file solve). Joins themselves were exact.
- SHIPPED, one capped solve for long files: <= 24 knots per bias curve, <= 200 residual rows, <= 20 dropouts with a
  velocity step (the shortest extra ones bridged; ties broken explicitly). None binds below about 3 min, so normal
  files are unchanged. Long rooms 0.70 m (uncapped 0.66, still 1.10).
- Memory fixes for long/fast files: gravity_fit at most 600 windows; Jacobian freed before trial steps (MH01 peak
  5.55 -> 4.5 MB); slabs of 96 rows; post_filter by real FFT per axis (59 min file 18.4 -> 8.1 MB at that stage); slow_rms
  per axis; fit_tail at most 600 cuts. Averaging reader counts rows per cell (no second pass), no full scan pass
  (`probe` reads the head and the end), batched output formatting: 1 kHz 3.19 -> 0.77 s, identical output.
- Final caps 20 dropouts / 200 residual rows: TUM-VI walks 0.790 (still 0.878, 20/22 better), long rooms 0.721 (still
  1.096, 5/5). Looser 30/300 scored 0.771 / 0.700 but peaks at 8.4-8.8 MB on 24-59 min files; 30/200 and 20/300 at
  7.5-8.1 MB. Kept 20/200 (worst 7.28 MB on a synthetic 59 min file, 6.9-7.1 MB on real 13-24 min walks).
- Resources, cool machine, local disk, whole process: rooms 3.5-3.7 ms/s, EuRoC 5.3-6.1, 1 kHz 5.8, long 0.6-2.8; heap
  2.5-4.8 MB normal files, 5.4 MB at 1 kHz. Hot laptop (after 30 min of 8-process scoring): EuRoC 17-22 ms/s.

## Round 9, 19 Nov: independent review (Codex, read-only) and fixes

Codex reviewed a clean clone and wrote 50 inputs (`experiments/tmp/codex`, harness `scripts/codex_cases.py`). 14 failed
on our side before the fixes, 0 after. Normal official files stay byte-identical except where noted.
- FIXED, no change to normal outputs: a rate change across a dropout expanded the grid (23.6 MB -> 5.2 MB; grid size now
  checked before the raw path); one timestamp shared by 65 000 rows was spread in one batch (38.3 -> 2.3 MB); 1 607
  columns per line were parsed in full (18.2 -> 0.7 MB); a 3 h, 2 Hz file built a 10 Hz track of 3 h (15.0 -> 5.9 MB;
  track coarsened beyond an hour, tail fit skipped there); non-ns stamps lost precision (now exact per row, header
  unit first); a UTF-8 BOM dropped row 1; a 10^309 stamp crashed before the output opened; BLAS threads followed the
  caller's environment; sensor limits were tested by column, not sensor; thrust along body y gave a NaN drag target;
  a sub-1 Hz file gave a zero row stride; a steady 1 rad/s turn was taken as a gyro bias at rest (rest gate 0.2
  rad/s; the official script's largest bias is 0.008). Averaging reader's spike test made NaN-safe (long rooms 0.721 ->
  0.722, TUM-VI walks 0.790 -> 0.787). Rate cap 400 Hz added (1 kHz drone 30 s: 20.6 -> 11.9 ms/s; headset ATE 0.671
  unchanged).
- SHIPPED, rotor windows: one-second windows straddling a 2048-row boundary were never judged. Fixed: EuRoC mean over
  the 5 official runs 2.354 -> 2.278 m (seed 42 sw 2.247 -> 2.328, og 2.448 -> 2.476; seeds 1-3 2.332/2.369/2.372 ->
  2.191/2.210/2.186). The seed-42 loss is MH04 alone (2.19 -> 4.17): its self-check moved 0.959 -> 1.011, over 1.0.
- REJECTED: forecast anchored at the last live row (blind worse on every set, e.g. 0.307 -> 0.317); tilt Jacobian from
  bias-corrected acceleration (mean -0.2%, rooms slightly worse, one og flight behind still).
- Not changed: keeping the declared drone calibration when there are fewer than 5 usable windows (no evidence either
  way; README wording fixed); the v0 prior mismatch between normal equations and objective (no measured effect).
- Checker rewritten to exact decimal comparison with BOM handling; the old checker passed the BOM bug.
