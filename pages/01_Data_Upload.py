# pages/01_DATA_UPLOAD.py

import pandas as pd
import streamlit as st

from src.app_state import dataset_summary, ensure_dataset_loaded
from src.config import RAW_NUMERIC_COLUMNS


st.set_page_config(
    page_title="Dataset",
    layout="wide"
)

st.title("Dataset")

try:
    ensure_dataset_loaded()
    summary = dataset_summary()

except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

full_df = st.session_state["full_raw"]
historical_df = st.session_state["historical_raw"]
live_df = st.session_state["live_raw"]

st.caption(summary["path"])

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Rows", summary["full_rows"])

with c2:
    st.metric("Historical Rows", summary["historical_rows"])

with c3:
    st.metric("Live Rows", summary["live_rows"])

with c4:
    st.metric("Columns", summary["columns"])

st.info(
    "The app uses one dataset. The first 60% is used for training "
    "and the remaining 40% is used for live monitoring."
)

missing_columns = [
    col
    for col in RAW_NUMERIC_COLUMNS
    if col not in full_df.columns
]

if missing_columns:
    st.error(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

st.success(
    f"Dataset ready from {summary['start_time']} to {summary['end_time']}"
)

tabs = st.tabs(
    [
        "Full Dataset",
        "Historical Split",
        "Live Split",
        "Missing Values",
    ]
)

with tabs[0]:
    st.dataframe(
        full_df.head(100),
        use_container_width=True
    )

with tabs[1]:
    st.dataframe(
        historical_df.head(100),
        use_container_width=True
    )

with tabs[2]:
    st.dataframe(
        live_df.head(100),
        use_container_width=True
    )

with tabs[3]:
    missing_df = pd.DataFrame({
        "Column": full_df.columns,
        "Missing Values": full_df.isnull().sum().values,
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )
