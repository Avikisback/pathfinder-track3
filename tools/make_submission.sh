#!/usr/bin/env bash
# Builds the Track 3 submission bundle from the organisers' evaluation dataset.
#
#   bash tools/make_submission.sh <dataset.zip | dataset folder | imu csv> <CollegeName_TeamName>
#
# Writes submission/<name>/ and submission/<name>.zip:
#   estimated_trajectory.txt   the trajectory for the evaluation dataset (one pose per input row)
#   metadata.json              runtime, peak heap of the algorithm, machine
#   src/main.py                the estimator: standalone, Python 3 + NumPy (FAQ: Python only, main.py)
#   src/README.md, Dockerfile, requirements.txt
# Run it from the xr_track3 folder, inside WSL or Linux. It stops if the output fails the format check.
set -euo pipefail
if [ $# -lt 2 ]; then
    sed -n '2,11p' "$0"
    exit 1
fi
data="$1"
name="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/submission/$name"
rm -rf "$dst" "$root/submission/$name.zip"
mkdir -p "$dst/src"

cp "$root/src/main.py" "$root/src/README.md" "$dst/src/"
cp "$root/Dockerfile" "$root/requirements.txt" "$dst/"

# work on a copy on the local disk: reading from a Windows drive through WSL inflates the runtime several times
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
if [ -d "$data" ]; then cp -r "$data" "$work/data"; in="$work/data"; else cp "$data" "$work/"; in="$work/$(basename "$data")"; fi

# the run that makes the trajectory, from the bundle's own main.py, timed and measured
python3 "$root/tools/measure.py" "$dst/src/main.py" "$in" "$dst/estimated_trajectory.txt" "$dst/metadata.json" > /dev/null

# the format check needs the CSV itself
csv="$in"
if [ -f "$in" ] && python3 -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$in"; then
    python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$in" "$work/unz"
    in="$work/unz"
fi
if [ -d "$in" ]; then csv="$(find "$in" -path '*imu0/data.csv' | sort | head -1)"; fi
python3 "$root/tools/check_output.py" "$csv" "$dst/estimated_trajectory.txt"

rm -rf "$dst/src/__pycache__"
( cd "$root/submission" && python3 -c "
import os, sys, zipfile
name = sys.argv[1]
with zipfile.ZipFile(name + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for d, _, files in os.walk(name):
        for f in sorted(files):
            z.write(os.path.join(d, f))
print('wrote submission/' + name + '.zip')
" "$name" )
echo "--- metadata.json"
cat "$dst/metadata.json"
echo "--- bundle"
( cd "$root/submission" && find "$name" -type f | sort )
