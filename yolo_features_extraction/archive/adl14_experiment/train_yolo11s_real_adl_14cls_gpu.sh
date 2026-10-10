#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-object_detection}"
MODEL="${MODEL:-yolo11s.pt}"
DATA_YAML="${DATA_YAML:-datasets/real_adl_hands_yolo_labeled/real_adl_hands_14cls.yaml}"
PROJECT="${PROJECT:-$REPO_ROOT/runs/real_adl_hands}"
NAME="${NAME:-yolo11s_real_adl_14cls_gpu}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-25}"
SEED="${SEED:-42}"

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "Preparing YOLO dataset view from labeled images only..."
python3 yolo_features_extraction/archive/adl14_experiment/prepare_real_adl_hands_yolo_dataset.py

echo "Checking CUDA in conda env: $CONDA_ENV"
if ! conda run --no-capture-output -n "$CONDA_ENV" python -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device visible'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
then
  echo
  echo "CUDA is not visible inside conda env '$CONDA_ENV'."
  echo "Check these before training on GPU:"
  echo "  nvidia-smi"
  echo "  conda run --no-capture-output -n $CONDA_ENV python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\""
  echo
  echo "If you are in WSL, make sure the Windows NVIDIA driver supports WSL CUDA and that /dev/dxg is available."
  exit 1
fi

echo "Starting YOLO11s GPU training..."
conda run --no-capture-output -n "$CONDA_ENV" yolo detect train \
  model="$MODEL" \
  data="$DATA_YAML" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME" \
  workers="$WORKERS" \
  patience="$PATIENCE" \
  seed="$SEED"

echo "Done. Results:"
echo "$PROJECT/$NAME"
