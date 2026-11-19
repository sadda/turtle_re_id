#!/bin/sh
set -eu
cd "$(dirname "$0")"

LANGS="en de ar"
SIZES="1920x1080 1440x1080"
JOBS=$(nproc)

for size in $SIZES; do
    for lang in $LANGS; do
        echo "$lang ${size%x*} ${size#*x}"
    done
done | xargs -P "$JOBS" -L 1 sh -c 'python make_video.py "$0" --width "$1" --height "$2" '"$*"
