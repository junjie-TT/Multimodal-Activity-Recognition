import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    GAZE_XY_FILLED_INDEX_OUTPUT,
    GAZE_XY_FILLED_OUTPUT_ROOT,
    VIDEO_ALIGNED_CSV_INDEX_OUTPUT,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def nearest_valid_indices(valid_mask):
    valid_idx = np.flatnonzero(valid_mask)
    n = len(valid_mask)

    if len(valid_idx) == 0:
        return np.full(n, -1, dtype=int)

    all_idx = np.arange(n)
    pos = np.searchsorted(valid_idx, all_idx)

    left_pos = np.clip(pos - 1, 0, len(valid_idx) - 1)
    right_pos = np.clip(pos, 0, len(valid_idx) - 1)

    left_idx = valid_idx[left_pos]
    right_idx = valid_idx[right_pos]

    left_dist = np.abs(all_idx - left_idx)
    right_dist = np.abs(right_idx - all_idx)

    # Tie-break toward the previous valid point, which keeps short gaps stable.
    choose_right = right_dist < left_dist
    nearest = np.where(choose_right, right_idx, left_idx)
    return nearest.astype(int)


def fill_gaze_xy_nearest(df):
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("gaze CSV must contain x and y columns.")

    out = df.copy()
    x = pd.to_numeric(out["x"], errors="coerce")
    y = pd.to_numeric(out["y"], errors="coerce")

    valid_xy = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    missing_xy = ~valid_xy

    out["gaze_xy_was_missing"] = missing_xy.astype(int)

    nearest = nearest_valid_indices(valid_xy)
    can_fill = missing_xy & (nearest >= 0)

    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    x_values[can_fill] = x_values[nearest[can_fill]]
    y_values[can_fill] = y_values[nearest[can_fill]]

    out["x"] = x_values
    out["y"] = y_values
    out["gaze_xy_filled_nearest"] = can_fill.astype(int)

    return out, {
        "rows": int(len(out)),
        "valid_xy_before": int(valid_xy.sum()),
        "missing_xy_before": int(missing_xy.sum()),
        "filled_xy": int(can_fill.sum()),
        "missing_xy_after": int((~np.isfinite(x_values) | ~np.isfinite(y_values)).sum()),
    }


def relative_output_path(csv_path):
    csv_path = Path(csv_path)
    return Path(*csv_path.parts[-3:])


def fill_gaze_files(
    aligned_index_path=VIDEO_ALIGNED_CSV_INDEX_OUTPUT,
    output_root=GAZE_XY_FILLED_OUTPUT_ROOT,
):
    aligned_index_path = Path(aligned_index_path)
    output_root = Path(output_root)

    if not aligned_index_path.exists():
        raise FileNotFoundError(
            f"Missing aligned sensor CSV index: {aligned_index_path}. "
            "Run apply_video_alignment_to_csv.py first."
        )

    index_df = pd.read_csv(aligned_index_path)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    records = list(index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Filling gaze x/y with nearest valid point", unit="csv")

    for record in iterator:
        input_path = Path(record["aligned_csv_path"])
        output_path = output_root / relative_output_path(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if record["modality"] == "eye_tracker__gaze":
            df = pd.read_csv(input_path)
            out_df, stats = fill_gaze_xy_nearest(df)
            out_df.to_csv(output_path, index=False)
            action = "filled_gaze_xy"
        else:
            shutil.copy2(input_path, output_path)
            stats = {
                "rows": int(record["rows"]),
                "valid_xy_before": np.nan,
                "missing_xy_before": np.nan,
                "filled_xy": np.nan,
                "missing_xy_after": np.nan,
            }
            action = "copied"

        rows.append(
            {
                "participant_id": record["participant_id"],
                "task_name": record["task_name"],
                "full_id": record["full_id"],
                "modality": record["modality"],
                "action": action,
                "source_csv_path": str(input_path),
                "output_csv_path": str(output_path),
                **stats,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(GAZE_XY_FILLED_INDEX_OUTPUT, index=False)

    summary = {
        "aligned_index": str(aligned_index_path),
        "output_root": str(output_root),
        "index_csv": str(GAZE_XY_FILLED_INDEX_OUTPUT),
        "csv_file_count": int(len(result_df)),
        "gaze_file_count": int((result_df["modality"] == "eye_tracker__gaze").sum()),
        "filled_xy_total": int(result_df["filled_xy"].fillna(0).sum()),
        "policy": "For eye_tracker__gaze, rows where x or y is missing are filled from the nearest row with both x and y valid. Other modalities are copied unchanged.",
    }

    summary_path = output_root / "gaze_xy_filled_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return result_df, summary


def main():
    result_df, summary = fill_gaze_files()
    print("Gaze x/y nearest-fill completed.")
    print(json.dumps(summary, indent=2))
    gaze_rows = result_df[result_df["modality"] == "eye_tracker__gaze"]
    if not gaze_rows.empty:
        print("\nFilled gaze x/y by sample:")
        print(gaze_rows[["full_id", "missing_xy_before", "filled_xy", "missing_xy_after"]].to_string(index=False))


if __name__ == "__main__":
    main()
