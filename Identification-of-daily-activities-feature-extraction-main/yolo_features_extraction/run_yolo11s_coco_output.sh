#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-object_detection}"
MODEL="${MODEL:-yolo11s.pt}"
INPUT_ROOT="${INPUT_ROOT:-/home/julius/datasets/seminar_object_detection/iAssistADL_dataset_export_2026-05-18/world_camera_videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv}"
CONF="${CONF:-0.25}"
LIMIT_ARG=()
OVERWRITE_ARG=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OVERWRITE_ARG=(--overwrite)
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

conda run --no-capture-output -n "$CONDA_ENV" python -m yolo_features_extraction.object_detection_csv_output \
  --model "$MODEL" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --conf "$CONF" \
  "${LIMIT_ARG[@]}" \
  "${OVERWRITE_ARG[@]}"
