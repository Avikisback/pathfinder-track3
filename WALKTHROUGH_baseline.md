# Read this, then explain it back

Whoever owns Track 3 should read `src/main.cpp` with this page. It takes about an hour with no
background in state estimation. If you can answer the questions at the end without looking, the
team can defend this code in front of a jury.

## The problem in one paragraph

A gyroscope reports how fast you are rotating. An accelerometer reports acceleration, including
gravity. Integrate rotation once to get orientation. Rotate acceleration into the world, subtract
gravity, and integrate twice to get position. Any small error in acceleration is also integrated
twice, so position error grows with time squared. On room3, doing only that misses by 217 m in a room
about 4 m wide. The whole job is stopping this while keeping the real motion.

## Read the code in this order

1. **`next_row`: reading data.** It reads one row at a time and never stores the file, which is why
   memory stays at 576 KB whatever the file length. Bad rows (NaN, all zero, frozen, impossible values)
   are kept but marked invalid, because the deck wants one pose per IMU frame.

2. **`survey`: first pass over the file.** It finds the normal time between samples and the quietest
   one-second block. The mean accel there tells us which way is up.

3. **`kStillGyroStd`: the bug that taught us the most.** The first version always used the quiet
   block's mean gyro as sensor bias. On room1 the "quiet" second was someone holding a camera, so that
   mean was real motion. Subtracting it caused about 119 deg of attitude error. The fix is to accept it
   as bias only if the block is truly still. Otherwise bias stays zero. Error fell to about 2 deg.

4. **`initial_attitude`: second pass.** Gravity gives the attitude during the quiet block, not at
   row 0. So it integrates the gyro from row 0 to the quiet block, then undoes that rotation.

5. **`step_valid`: the estimator, for a normal row.**
   - **Tilt correction.** Compare measured "up" with predicted "up" and nudge the orientation by
     `kp`. A gyro bias `b` settles at a tilt error of about `b / kp`. Heading gets no correction,
     because gravity carries no information about it.
   - **`a_mean`.** Removes the slow average of world acceleration, which is mostly tilt error and accel
     bias, so steady errors never reach position.
   - **`tau_v`, `tau_p`.** Leaks on velocity and position. They bound the output and keep fast motion.
     This is the design idea from the report: RPE rewards fast motion, and ATE rewards restraint.

6. **`step_coast`: no data.** During gaps and the blind 5 s, the turn rate and velocity both fade.
   Earlier research found constant velocity is the worst forecast tested, so we do not hold it. After
   a gap, `boost` raises the tilt gain for a moment so "up" recovers quickly.

7. **`write_pose`, `main`.** Timestamps are printed from integer nanoseconds so no precision is lost.
   `main` runs the passes, handles holes in the timestamps, then appends the blind tail.

## Run the experiments yourself

```bash
./pathfinder data/tumvi_room3/imu0/data.csv out/t.txt --tau-a 1e9 --tau-v 1e9 --tau-p 1e9 --kp 0 --boost 0
bash tools/score.sh data/tumvi_room3/gt.tum out/t.txt
```

That is naive integration. Now turn on one thing at a time and watch position error change.

## Questions to answer without looking

1. Why does the program read the whole file before writing any pose?
2. Why did subtracting the quiet block's mean gyro wreck orientation on room1?
3. With a gyro bias of 0.02 rad/s and `kp` of 0.2, what steady tilt error do you expect, in degrees?
4. Without `a_mean`, what position offset would a steady 0.2 m/s² accel error cause at `tau_v` 1 and
   `tau_p` 4?
5. After a 2 s gap, why can tilt recover but heading cannot?
6. Why are timestamps never converted to a floating-point number inside the program?
7. Why is attitude scored with first-pose alignment in `score.sh`, instead of evo's `-a`?
8. A juror asks: "Standing still scores about the same. Why not submit that?" What do you say?
9. Why might this code score worse on EuRoC than on TUM-VI?
10. Name the next two things you would add to beat stationary, and how you would prove they work.
