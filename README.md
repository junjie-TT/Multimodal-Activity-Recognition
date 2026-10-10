# Identification of Daily Activities: Feature Extraction

This repository contains feature extraction, cleaning, interaction modeling, and
baseline training workflows for daily activity recognition using the iAssistADL
dataset. The code is organized around a staged pipeline: raw aligned sensor data
is cleaned, synchronized to video time, converted into lower- and upper-level
features, and then used for activity-recognition experiments.

The repository is intended for collaborative research. Source code and
lightweight documentation are version-controlled; raw datasets, trained model
weights, and generated result folders should remain outside the repository.

## Repository Structure

```text
.
├── lower_feature_cleaning_extraction/
│                                      Lower-level cleaning, alignment, and sensor export
├── yolo_features_extraction/         YOLO object-detection feature extraction
├── upper_features_extraction/        Upper-level motion and interaction features
├── baseline_svm_current_action/      Baseline current-action modeling scripts
├── train/                           Unified training and evaluation workflows
├── train_YOLO_hands/                 YOLO hand-detection training utilities
├── notebooks/                       Exploration, validation, and visualization notebooks
└── README.md
```

## Main Pipeline

The project currently supports the following workflow:

```text
Aligned HDF5 recordings
  -> clean recording manifest construction
  -> sensor CSV export
  -> video-time alignment
  -> YOLO object and hand detection outputs
  -> upper-level motion and interaction features
  -> model training and evaluation
```

The exact workflow depends on the experiment. The most commonly used entry
points are documented in the README files inside each module.

## Key Components

`lower_feature_cleaning_extraction/`

Builds clean recording manifests, exports aligned HDF5 modalities to CSV,
applies video alignment offsets, fills missing gaze coordinates, and extracts
lower-level IMU and detection-related features.

`yolo_features_extraction/`

Runs YOLO object detection on world-camera videos and exports frame-level object
detections as CSV files. Experimental ADL14 scripts are archived under
`yolo_features_extraction/archive/`.

`upper_features_extraction/`

Builds upper-level features from cleaned lower-level outputs, including head
motion, skeleton kinematics, depth, gaze-object interaction, and hand-object
interaction features.

`baseline_svm_current_action/`

Contains baseline modeling scripts for frame-level and window-level activity
recognition experiments.

`train/`

Provides a unified runner for training and evaluation workflows. It supports
baseline current-action experiments and GOI/HOI feature-based experiments.

`train_YOLO_hands/`

Contains utilities for preparing, checking, training, and running YOLO hand
detection workflows.

`notebooks/`

Stores exploratory analysis, visualization, validation, and demonstration
notebooks. Keep notebooks lightweight and avoid embedding large outputs.

## Environment

Create the shared environment from the repository root:

```bash
conda env create -f environment.yml
conda activate object_detection
python -m pip check
```

`environment.yml` keeps the project dependencies and selected versions without
local paths or platform-specific Conda build identifiers. Pip resolves supporting
libraries, including the CUDA runtime required by PyTorch. It is a shared setup
file, not an exact lockfile of the original machine.

On Linux/Windows x86_64, the file preserves PyTorch 2.5.1 and torchvision 0.20.1
with CUDA 12.1 using the official PyTorch wheel source. GPU use requires a
compatible NVIDIA driver. On macOS, the file selects the standard PyTorch wheels;
CUDA is unavailable. Other architectures may require different PyTorch versions
and are not covered by this configuration.

For a CPU-only Linux/Windows installation, replace `+cu121` with `+cpu` and
`https://download.pytorch.org/whl/cu121` with
`https://download.pytorch.org/whl/cpu` in a copy of the YAML before creating the
environment. See the [official PyTorch 2.5.1 installation options](https://pytorch.org/get-started/previous-versions/#v251).
The optional RAPIDS SVM backend (`--backend cuml`) needs a separate compatible
RAPIDS installation; the default sklearn backend is included.

Check the installation:

```bash
python -c "import cv2, h5py, numpy, pandas, pyarrow, sklearn, torch, torchvision, ultralytics; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

Several workflows use the environment name `object_detection`. For example:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_clean_manifest
```

Some scripts may also run directly with Python:

```bash
python train/run_training_workflow.py --list
```

Run commands from the repository root. Review each module README before running
a workflow, because dataset paths and
output roots are intentionally local-machine specific.

## Data and Output Policy

Do not commit raw datasets, generated feature tables, model weights, or large
training outputs. These files should be stored under local dataset or experiment
directories, such as:

```text
/home/julius/datasets/
/home/julius/models/
```

Keep the repository focused on:

```text
source code
configuration templates
documentation
small metadata files
lightweight analysis summaries
```

Avoid committing:

```text
raw video files
HDF5 datasets
generated CSV/Parquet/NPZ feature outputs
model checkpoints
training runs
Python cache files
large notebook outputs
```

## Common Commands

Build the lower-feature clean manifest:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_clean_manifest
```

Export clean sensor CSV files:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.export_clean_sensor_csv
```

See [the lower-feature workflow](lower_feature_cleaning_extraction/README.md)
for alignment and subsequent preprocessing steps.

Run YOLO object detection:

```bash
./yolo_features_extraction/run_yolo11s_coco_output.sh
```

List available training workflows:

```bash
python train/run_training_workflow.py --list
```

Run a standard baseline workflow:

```bash
python train/run_training_workflow.py baseline_classical --skip-existing
```

## Collaboration Workflow

Use feature branches for development:

```bash
git switch main
git pull origin main
git switch -c feature/<name>/<task>
```

Commit focused changes with clear messages:

```bash
git add <paths>
git commit -m "Add concise description of the change"
```

Push the feature branch and open a pull request into `main`:

```bash
git push -u origin feature/<name>/<task>
```

Before merging, verify that the pull request only contains the intended files.
Avoid mixing generated data, unrelated refactors, and experimental outputs in
the same change.

## Notes for Contributors

Keep module-specific instructions in the corresponding module README. When a
workflow changes, update the local README near the code as well as this project
overview if the top-level structure or recommended entry points change.

The central rule for this repository is simple: code and documentation are
shared; raw data and generated results stay local.
