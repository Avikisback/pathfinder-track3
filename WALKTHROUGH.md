# Read this, then explain it back

Whoever presents Track 3 should read `src/main.py` with this page open. Allow about two hours. If you can answer the
questions at the end without looking, the team can defend the entry in front of a jury.
`WALKTHROUGH_baseline.md` covers the first, simpler program (`src/main.cpp`). Read it first if double integration is
new to you. `src/pathfinder.cpp` is the same method in C++ (our earlier entry; the FAQ now asks for Python only).

## The idea in one paragraph

Integrating an IMU twice turns tiny sensor errors (gyro bias, accel bias, a tilt that leaks gravity) into position
drift that grows with time squared. A filter fights this one sample at a time. We have the whole file before writing
any pose, so we can do better: write the slow sensor errors as a few dozen unknown numbers, and pick the numbers that
make the drift go away. "Drift goes away" means the 5 s average velocity is small and the position stays near its
centre. That is true for a person in a room; for a drone the velocity is instead close to what rotor drag says. The
same sensor errors cause the drift, so solving for them removes it without removing real motion, which is fast and
keeps changing.

## Read the code in this order

1. **`probe`, `read_rows`, `make_grid` (input).** `probe` looks at the first 4096 rows and the end of the file (step,
   units, first and last stamp) without reading everything. `read_rows` reads values in chunks of 1024 lines; the
   timestamps themselves are not kept, only time since the first row. `make_grid` puts rows on a regular grid: a
   missing row becomes a dead grid row, a duplicated stamp maps to the same grid row, and a row is invalid if it is
   NaN, all zero, an exact repeat of the row before (a frozen sensor) or impossibly large. The 5 s blind window is
   already in the file as dead rows (FAQ), so nothing is appended. Every input row keeps its grid row (`gi`) and gets
   exactly one output pose.
2. **`repair`, `bridge` (cleaning).** A value far from the median of its 5 neighbours (10 robust sigmas plus a floor)
   is a spike and is dropped. Short dead runs are interpolated; longer ones stay dropouts, which the solver handles.
3. **`rotor_windows` (what platform is this?).** Rotors shake the accelerometer at high frequency. The third
   difference of the accel cancels slow motion and keeps that vibration. If more than 30% of 1 s windows have it,
   the file is a drone. This identifies a sensor rig, never a sequence. Drones then bridge holes up to 0.25 s.
4. **`rest_bias`, `window_bias` (gyro calibration).** Drones start from the declared calibration, unless the file
   contradicts it.
   - If the file has a still moment and the gyro there disagrees by more than 0.02 rad/s, we trust the file.
   - If there is no still moment, `window_bias` estimates the bias from the file. In each 4 s window the up direction
     at the start is carried to the end by the gyro and compared with the up direction seen there. The mismatch is
     linear in the bias, so all windows give one small least-squares problem. It sees only the bias across the up
     direction. If that is more than 0.03 rad/s from the declared value, the estimate is used.
   - This is what makes the entry safe if the organisers' gyro data is already bias-compensated, and it fixed
     rotated sensor axes on flights without a rest.
5. **`Solver.rates`, `chain_chunks`, `qchain` (attitude).** Gyro minus a bias curve, integrated into quaternions in
   pieces. Inside a gap the turn rate coasts on what was seen at its edges.
6. **`Solver.quiet_init`, `Solver.gravity_fit` (first guess).** The quietest second gives the start attitude (its mean
   accel is up). Gravity points down, so the averaged accel, turned into the world, should point straight up; tilt
   errors are linear in the bias curve's values, so a small least-squares fit gives a good starting bias.
7. **`Solver.forward` (the model).** For given unknowns (start velocity, a tilt per stretch between gaps, accel and gyro
   bias curves, a velocity step per gap) it turns the accel into the world, removes gravity and integrates to
   velocity, once per 8-row block.
8. **`Solver.core` (what "no drift" means, and how to fix it).**
   - The residual rows are the 5 s box average of velocity (minus the drag target on drones), once per second, and
     the centred position.
   - A Levenberg-Marquardt step needs the derivative of every residual with respect to every unknown (the Jacobian
     `A`). A bias knot only acts inside its two knot intervals; after them its effect is constant (accel bias) or
     follows the summed specific force (gyro bias), so each column is built from that short stretch plus a closed
     form tail from prefix sums (`put_local`). `A` is float32, added into the normal equations in slabs of 96 rows in
     double precision, and freed before the trial steps.
   - Each step is accepted only if the full objective drops.
   - Every second iteration the stiffness of the bias curves is re-chosen by Bayesian evidence (Fellner-Schall /
     MacKay update). A file with jumpy bias gets a flexible curve, a calm file a stiff one, chosen per file from its
     own data.
9. **`Solver.set_target` (drones only).**
   - Rotor drag pushes back in proportion to velocity, so the in-plane accel, low-passed at 0.2 Hz, gives velocity
     directly (k = 0.195 /s).
   - The IMU sits 10 cm from the centre of rotation, so turning adds accel that is not drag. We subtract it using the
     lever arm and the gyro, then remove a rotor-rate term.
10. **`slow_rms`, `post_filter` (self-check and post-filter, in `estimate`).**
    - If the 5 s average speed is still large (above 1.0 m/s on drones, 0.5 m/s handheld), the solve has failed, and
      a high-pass keeps only fast motion.
    - A drone has a second warning sign: the drift rate the evidence chose in step 8. Ordinary flights settle at
      0.01-0.1 m/s². If it is over 0.3 while the slow speed is over 0.6 m/s, the drift is of a kind the solve handles
      badly (oscillating, random walk), and the high-pass is used. Handheld files do not use this, because a long real
      walk raises the drift rate just as hostile drift does.
    - Otherwise each octave band of velocity is scaled (real FFT of the mirrored track, one axis at a time). Drones
      get a Wiener gain from their own band power; handheld files get fixed gains learned on other rooms.
11. **`fit_tail` and the end of `estimate` (blind 5 s).**
    - The forecast is hold plus a damped share of the last velocity. Drones use a share of 0.5, and with rotors on
      also descend 0.25 m over 3 s.
    - A handheld file picks its own share in `fit_tail`: at every earlier second, how well did the velocity then
      predict the next 5 s of this same path? A room gives a share near zero, a steady walk more.
    - A 5 mm quarter-circle keeps the tail from being a straight line, because a straight tail breaks evo's alignment
      when the tail is scored alone.
12. **`main`, `text` (output).** Poses are written as they are made, in pieces, with each row's own timestamp read
    again from the file (`stamps`, digit for digit). If anything raises inside the solver, the remaining rows hold the
    last pose: the output always has one finite pose per input row.
13. **`poses_of`, `read_cells`, `spread` (fast or long files).** Above 40 000 rows, rows are averaged into cells as
    they are read (spikes, frozen and invalid rows left out), the solve runs on the cells, and each input row gets
    its cell's pose (`np.repeat` of the poses by the row count of each cell). Long files also hit the caps of the
    solve: 25 knots per bias curve, about 200 residual sites and 20 dropouts with their own velocity step. An IMU
    faster than 400 Hz is averaged too.

## How we know it is right

- `experiments/scripts/score_off.py` scores the entry by the FAQ rules on files made with the organisers' own
  degradation script (both column orders, four seeds), on long files and on 22 unseen TUM-VI walks. Standing still is
  always printed beside the result. `experiments/scripts/odd_inputs.py` runs 22 malformed, tiny, long and fast inputs.
- The Python port was checked against the C++ reference on the same files (same scores, solver-level differences of
  a few centimetres on drones), and NumPy 1.26 and 2.x give the same output to a micrometre.
- `tools/measure.py` gives time and heap; `tools/check_output.py` checks one pose per row and matching timestamps.

## Questions to answer without looking

1. Why does double integration drift with time squared, and which three sensor errors cause most of it?
2. What does the solver assume about motion that a person in a room satisfies? What replaces it on a drone?
3. Why are the bias curves solved for, not filtered out? What real motion survives that a high-pass would kill?
4. How does the program tell a drone from a handheld file? Why is that allowed when naming a sequence is not?
5. What does the evidence update choose, and why per file instead of one fixed value?
6. The IMU is 10 cm from the drone's centre. What does turning do to the accelerometer, and why does it matter for drag?
7. Why does the drone descend in the blind 5 s, and when does the program not do it? Why does a handheld file fit its
   own tail while a drone does not?
8. Why is the tail bent by 5 mm?
9. How does memory stay below 8 MB on a 24-minute file, and time below 20 ms per second on a 1 kHz IMU?
10. A juror asks: "Your heading drifts on handheld files. Why?" What do you say, and what would you change?
11. Why are the blind-window rows not appended, and how does the program know where the blind window starts?
12. Why was cutting a long file into separately solved windows worse than one solve with caps?
