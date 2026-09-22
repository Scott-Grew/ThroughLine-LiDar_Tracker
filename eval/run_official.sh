#!/bin/zsh
# Prints Waymo's official 3D tracking metrics for one track export. It
# converts both inputs to Objects protos and runs Waymo's binary in Docker.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: run_official.sh <segment_name> <tracks.csv>" >&2
  exit 1
fi

SEGMENT_NAME="$1"
TRACKS_CSV="$2"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG="tracker-official-eval"

if [ ! -f "$TRACKS_CSV" ]; then
  echo "run_official.sh: no such tracks file: $TRACKS_CSV" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tracker-official-eval.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "run_official.sh: building the evaluator image (cached once built)..." >&2
docker build --platform linux/amd64 -t "$IMAGE_TAG" "$EVAL_DIR"

echo "run_official.sh: converting $TRACKS_CSV to Waymo's Objects proto..." >&2
"${PYTHON:-python3}" "$EVAL_DIR/to_waymo_objects.py" \
  --parquet-root "${PARQUET_ROOT:-data/parquet}" \
  --segment "$SEGMENT_NAME" \
  --tracks "$TRACKS_CSV" \
  --out-predictions "$WORK_DIR/predictions.bin" \
  --out-ground-truth "$WORK_DIR/ground_truth.bin"

echo "run_official.sh: scoring with compute_tracking_metrics_main..." >&2
docker run --rm --platform linux/amd64 \
  -v "$WORK_DIR:/data" \
  "$IMAGE_TAG" \
  /data/predictions.bin /data/ground_truth.bin
