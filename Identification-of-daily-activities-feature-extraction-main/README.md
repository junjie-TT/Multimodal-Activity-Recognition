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
├── scripts/                         General dataset utilities
├── feature_extraction/              Core multi-modal feature extraction package
├── lower_feature_cleaning_extraction/
│                                      Lower-level cleaning, alignment, and sensor export
├── yolo_features_extraction/         YOLO object-detection feature extraction
├── upper_features_extraction/        Upper-level motion and interaction features
├── baseline_svm_current_action/      Baseline current-action modeling scripts
├── train/                           Unified training and evaluation workflows
├── train_YOLO_hands/                YOLO hand-detection training utilities
├── notebooks/                       Exploration, validation, and visualization notebooks
├── reports/                         Lightweight analysis outputs and summaries
└── README.md
```

## Main Pipeline

The project currently supports the following workflow:

```text
Aligned HDF5 recordings
  -> dataset manifest construction
  -> lower-level cleaning and sensor CSV export
  -> video-time alignment
  -> YOLO object and hand detection outputs
  -> upper-level motion and interaction features
  -> model training and evaluation
```

The exact workflow depends on the experiment. The most commonly used entry
points are documented in the README files inside each module.

## Key Components

`scripts/`

Contains shared utilities, including dataset manifest construction. The manifest
serves as the recording index used by downstream feature extraction scripts.

`feature_extraction/`

Implements the core multi-modal feature extraction package for HDF5 recordings,
including IMU, gaze, skeleton, metadata, and supporting utility functions.

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

Several workflows assume a local Conda environment named `object_detection`.
For example:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_clean_manifest
```

Some scripts may also run directly with Python:

```bash
python -m feature_extraction.main
python train/run_training_workflow.py --list
```

Review each module README before running a workflow, because dataset paths and
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

Build a dataset manifest:

```bash
python scripts/build_manifest.py
```

Run the core feature extraction package:

```bash
python -m feature_extraction.main
```

Build the lower-feature clean manifest:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_clean_manifest
```

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
