#!/usr/bin/env bash
# Streams only imu0 and mocap0 out of each official TUM-VI room tarball. Nothing else touches disk.
cd /mnt/e/clipping-pipeline/xr_track3/data || exit 1
for n in 2 4 5 6; do
  (
    d=tumvi_room$n
    mkdir -p "$d"
    if [ -s "$d/imu0/data.csv" ] && [ -s "$d/mocap0/data.csv" ]; then echo "$d present"; exit 0; fi
    cd "$d" || exit 1
    curl -sL --fail --retry 3 "https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room${n}_512_16.tar" \
      | tar -x --wildcards --strip-components=2 '*/mav0/imu0/*' '*/mav0/mocap0/*'
    echo "$d done rc=${PIPESTATUS[*]} $(wc -l imu0/data.csv mocap0/data.csv | tr '\n' ' ')"
  ) &
done
wait
echo ALL_DONE
