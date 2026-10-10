import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    ALIGNMENT_MANIFEST_OUTPUT,
    CLEAN_CSV_INDEX_OUTPUT,
    MISSING_ALIGNMENT_OUTPUT,
    VIDEO_ALIGNED_CSV_INDEX_OUTPUT,
    VIDEO_ALIGNED_CSV_OUTPUT_ROOT,
    VIDEO_FPS,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def load_alignment_lookup(path=ALIGNMENT_MANIFEST_OUTPUT):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing alignment manifest: {path}. "
            "Run build_alignment_manifest.py first."
        )

    df = pd.read_csv(path)
    return {row["full_id"]: row for _, row in df.iterrows()}


def relative_output_path(csv_path):
    csv_path = Path(csv_path)
    # clean_sensor_csv/{participant}/{full_id}/{modality}.csv
    return Path(*csv_path.parts[-3:])


def apply_alignment_to_dataframe(df, offset_s, video_fps):
    out = df.copy()
    if "time_s" not in out.columns:
        raise ValueError("CSV is missing required time_s column.")

    out.insert(2, "video_time_s", out["time_s"].astype(float) - float(offset_s))
    out.insert(3, "video_frame_idx", np.round(out["video_time_s"] * video_fps).astype("Int64"))
    return out


def apply_video_alignment(
    clean_csv_index_path=CLEAN_CSV_INDEX_OUTPUT,
    output_root=VIDEO_ALIGNED_CSV_OUTPUT_ROOT,
    video_fps=VIDEO_FPS,
):
    clean_csv_index_path = Path(clean_csv_index_path)
    output_root = Path(output_root)

    if not clean_csv_index_path.exists():
        raise FileNotFoundError(
            f"Missing clean sensor CSV index: {clean_csv_index_path}. "
            "Run export_clean_sensor_csv.py first."
        )

    alignment_lookup = load_alignment_lookup()
    clean_index = pd.read_csv(clean_csv_index_path)

    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_rows = []
    records = list(clean_index.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Applying video alignment", unit="csv")

    for record in iterator:
        full_id = record["full_id"]
        alignment = alignment_lookup.get(full_id)

        alignment_status = "from_alignment_json"
        alignment_method = ""

        if alignment is None or pd.isna(alignment.get("offset_s")):
            offset_s = 0.0
            alignment_status = "assumed_zero_offset"
            alignment_method = "missing_alignment_assume_no_offset"
            missing_rows.append(
                {
                    "participant_id": record["participant_id"],
                    "task_name": record["task_name"],
                    "full_id": full_id,
                    "modality": record["modality"],
                    "csv_path": record["csv_path"],
                    "assumed_offset_s": offset_s,
                    "reason": "missing_alignment_assumed_zero_offset",
                }
            )
        else:
            offset_s = float(alignment["offset_s"])
            alignment_method = alignment.get("method", "")

        input_path = Path(record["csv_path"])
        output_path = output_root / relative_output_path(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_path)
        aligned = apply_alignment_to_dataframe(df, offset_s, video_fps)
        aligned.to_csv(output_path, index=False)

        rows.append(
            {
                "participant_id": record["participant_id"],
                "task_name": record["task_name"],
                "full_id": full_id,
                "modality": record["modality"],
                "rows": int(len(aligned)),
                "columns": int(len(aligned.columns)),
                "offset_s": offset_s,
                "video_fps": video_fps,
                "alignment_status": alignment_status,
                "alignment_method": alignment_method,
                "aligned_csv_path": str(output_path),
                "source_csv_path": record["csv_path"],
            }
        )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(VIDEO_ALIGNED_CSV_INDEX_OUTPUT, index=False)

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(MISSING_ALIGNMENT_OUTPUT, index=False)

    summary = {
        "clean_csv_index": str(clean_csv_index_path),
        "alignment_manifest": str(ALIGNMENT_MANIFEST_OUTPUT),
        "output_root": str(output_root),
        "video_fps": video_fps,
        "aligned_csv_count": int(len(index_df)),
        "assumed_zero_offset_csv_count": int(len(missing_df)),
        "index_csv": str(VIDEO_ALIGNED_CSV_INDEX_OUTPUT),
        "assumed_zero_offset_csv": str(MISSING_ALIGNMENT_OUTPUT),
        "time_transform": "video_time_s = sensor time_s - offset_s; video_frame_idx = round(video_time_s * video_fps)",
        "missing_alignment_policy": "If a recording has no alignment JSON, offset_s is assumed to be 0.0 and the CSV is still exported.",
    }

    summary_path = output_root / "video_aligned_sensor_csv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, missing_df, summary


def main():
    index_df, missing_df, summary = apply_video_alignment()
    print("Video alignment applied to clean sensor CSV files.")
    print(json.dumps(summary, indent=2))
    if not missing_df.empty:
        print("\nNo alignment JSON; assumed zero offset by task:")
        print(missing_df[["full_id"]].drop_duplicates()["full_id"].to_string(index=False))


if __name__ == "__main__":
    main()
