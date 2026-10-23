#!/usr/bin/env bash
# Download OpenScene v1.1 trainval camera shards (the NAVSIM sensor blobs).
#
# After extraction the tree is
#   <out>/openscene-v1.1/sensor_blobs/trainval/<log>/CAM_F0/<frame>.jpg
# so pass <out>/openscene-v1.1/sensor_blobs/trainval as --data_root for
#   <data_root>/<log>/CAM_F0/<frame>.jpg
#
# Usage:
#   bash tools/pickle_gen/download_openscene_camera.sh --out /mnt/sfs_turbo/lmc/navsim_openscene
#   bash tools/pickle_gen/download_openscene_camera.sh --out DIR --start 0 --end 9
#
# Full trainval camera set is 200 shards, ~4.7 GB each (~940 GB total).
set -euo pipefail

OUT=""
START=0
END=199
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)    OUT="$2";    shift 2 ;;
    --start)  START="$2";  shift 2 ;;
    --end)    END="$2";    shift 2 ;;
    --mirror) MIRROR="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 1; }
mkdir -p "$OUT"

BASE="$MIRROR/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera"
for i in $(seq "$START" "$END"); do
  TAR="$OUT/openscene_sensor_trainval_camera_${i}.tgz"
  echo "[shard $i] downloading"
  wget -c -q --show-progress "$BASE/openscene_sensor_trainval_camera_${i}.tgz" -O "$TAR"
  echo "[shard $i] extracting"
  tar -xzf "$TAR" -C "$OUT"
  rm -f "$TAR"
done

echo "done: $OUT/openscene-v1.1/sensor_blobs/trainval"
