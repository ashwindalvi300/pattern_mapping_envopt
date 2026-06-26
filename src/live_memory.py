# src/live_memory.py

import json
import uuid
from pathlib import Path

import pandas as pd

LIVE_WINDOWS_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "live_windows.parquet"
)


def save_live_windows(window_results):
    if not window_results:
        return pd.DataFrame()

    LIVE_WINDOWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_df = pd.DataFrame(window_results)

    for col in ["spike_features", "top_sensors"]:
        if col in new_df.columns:
            new_df[col] = new_df[col].apply(json.dumps)

    new_df["eligible_for_history"] = (
        new_df["alert_type"].eq("NORMAL")
        &
        (new_df["point_ratio"] < 0.01)
        &
        (new_df["if_score"] > -0.05)
    )

    if LIVE_WINDOWS_PATH.exists():
        try:
            existing_df = pd.read_parquet(LIVE_WINDOWS_PATH)
        except (OSError, ValueError):
            existing_df = pd.DataFrame()

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )
    else:
        combined_df = new_df

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=["time", "window_start", "window_end"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    temp_path = LIVE_WINDOWS_PATH.with_name(
        f"{LIVE_WINDOWS_PATH.stem}.{uuid.uuid4().hex}.tmp.parquet"
    )

    combined_df.to_parquet(
        temp_path,
        index=False
    )
    temp_path.replace(LIVE_WINDOWS_PATH)

    return combined_df
