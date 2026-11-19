#!/bin/sh
set -eu
cd "$(dirname "$0")"

python make_video.py --width 1920 --height 1080 "$@"
python make_video.py --width 1440 --height 1080 "$@"
