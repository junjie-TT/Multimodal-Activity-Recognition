import json
from pathlib import Path
import sys

import h5py
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    ALIGNED_H5_ROOT,
    EXCLUDED_OUTPUT,
    MANIFEST_OUTPUT,
    OUTPUT_ROOT,
    SKIP_TASKS,
    SUMMARY_OUTPUT,
)


def parse_h5_name(path):
    participant_id = path.parent.name
    stem = path.stem.replace("_aligned_dataset", "")
    prefix = f"{participant_id}_"
    task_name = stem[len(prefix):] if stem.startswith(prefix) else stem
    return participant_id, task_name, stem


def inspect_h5(path):
    row = {
        "h5_path": str(path),
        "opens": False,
        "samples": 0,
        "duration_s": 0.0,
        "modality_count": 0,
        "has_gaze": False,
        "has_eye_tracker_imu": False,
        "movella_count": 0,
        "skeleton_body_count": 0,
        "has_depth": False,
        "modality_names": "",
        "error": "",
    }

    try:
        with h5py.File(path, "r") as f:
            row["opens"] = True

            if "metadata/global_time_seconds" in f:
                t = f["metadata/global_time_seconds"][:]
                row["samples"] = int(len(t))
                if len(t) > 1:
                    row["duration_s"] = float(t[-1] - t[0])

            modalities = list(f.get("modalities", {}).keys())
            row["modality_count"] = len(modalities)
            row["has_gaze"] = "eye_tracker__gaze" in modalities
            row["has_eye_tracker_imu"] = "eye_tracker__imu" in modalities
            row["movella_count"] = sum(name.startswith("movella__DOT") for name in modalities)
            row["skeleton_body_count"] = sum(name.startswith("zed__skeleton__") for name in modalities)
            row["has_depth"] = "zed__depth" in modalities
            row["modality_names"] = ";".join(modalities)
    except Exception as exc:
        row["error"] = repr(exc)

    return row


def build_manifest():
    rows = []

    for h5_path in sorted(ALIGNED_H5_ROOT.glob("*/*.h5")):
        participant_id, task_name, full_id = parse_h5_name(h5_path)
        is_calibration = task_name in SKIP_TASKS

        row = {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "is_calibration": is_calibration,
            "exclude_reason": "calibration_task" if is_calibration else "",
        }
        row.update(inspect_h5(h5_path))
        rows.append(row)

    all_df = pd.DataFrame(rows)
    clean_df = all_df[(~all_df["is_calibration"]) & (all_df["opens"])].copy()
    excluded_df = all_df[all_df["is_calibration"] | (~all_df["opens"])].copy()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(MANIFEST_OUTPUT, index=False)
    excluded_df.to_csv(EXCLUDED_OUTPUT, index=False)

    summary = {
        "aligned_h5_root": str(ALIGNED_H5_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "total_h5_files": int(len(all_df)),
        "clean_h5_files": int(len(clean_df)),
        "excluded_files": int(len(excluded_df)),
        "calibration_excluded": int(all_df["is_calibration"].sum()),
        "open_failed": int((~all_df["opens"]).sum()),
        "participant_count_clean": int(clean_df["participant_id"].nunique()),
        "skip_tasks": sorted(SKIP_TASKS),
        "clean_manifest": str(MANIFEST_OUTPUT),
        "excluded_manifest": str(EXCLUDED_OUTPUT),
    }

    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    return clean_df, excluded_df, summary


def main():
    clean_df, excluded_df, summary = build_manifest()

    print("Lower feature cleaning manifest built.")
    print(json.dumps(summary, indent=2))
    print("\nClean task counts:")
    print(clean_df["task_name"].value_counts().sort_index().to_string())
    print("\nExcluded task counts:")
    print(excluded_df["task_name"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
