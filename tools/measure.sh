#!/usr/bin/env bash
# Peak memory (max RSS) and time of the entry on IMU files: measure.sh <binary> <csv>...
# Prints one line per file: name, seconds of IMU, wall seconds, ms per second of IMU, peak KB.
bin="$1"
shift
for f in "$@"; do
    dur=$(awk -F, '!/^#/ && NF>1 {if (!s) s=$1; e=$1} END {printf "%.1f", (e-s)/1e9}' "$f")
    out=$(/usr/bin/time -f "%e %M" "$bin" "$f" /tmp/measure_out.txt 2>&1 >/dev/null | tail -1)
    set -- $out
    printf "%-28s %7ss of IMU  %6.2f s  %6.2f ms/s  %6d KB\n" "$(basename "$f")" "$dur" "$1" "$(echo "$1 $dur" | awk '{print 1000*$1/$2}')" "$2"
done
