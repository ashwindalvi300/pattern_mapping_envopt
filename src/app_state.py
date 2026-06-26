# src/app_state.py

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import prepare_dataset, split_historical_live


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = APP_ROOT.parent
ARTIFACTS_DIR = APP_ROOT / "artifacts"
DEFAULT_DATASET_PATH = DATA_ROOT / "PowerOptimus_DigitalTwin_30Sec_90Day.csv"


@st.cache_data(show_spinner="Loading dataset...")
def load_default_dataset(path):

    df = pd.read_csv(path)

    return prepare_dataset(df)


def ensure_dataset_loaded():

    if (
        "full_raw" in st.session_state
        and "historical_raw" in st.session_state
        and "live_raw" in st.session_state
    ):
        return

    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DEFAULT_DATASET_PATH}"
        )

    full_df = load_default_dataset(
        str(DEFAULT_DATASET_PATH)
    )

    historical_df, live_df = split_historical_live(
        full_df
    )

    st.session_state["dataset_path"] = str(DEFAULT_DATASET_PATH)
    st.session_state["full_raw"] = full_df
    st.session_state["historical_raw"] = historical_df
    st.session_state["live_raw"] = live_df


def dataset_summary():

    ensure_dataset_loaded()

    full_df = st.session_state["full_raw"]
    historical_df = st.session_state["historical_raw"]
    live_df = st.session_state["live_raw"]

    return {
        "path": st.session_state.get(
            "dataset_path",
            str(DEFAULT_DATASET_PATH)
        ),
        "full_rows": len(full_df),
        "historical_rows": len(historical_df),
        "live_rows": len(live_df),
        "columns": len(full_df.columns),
        "start_time": full_df["Timestamp"].iloc[0],
        "end_time": full_df["Timestamp"].iloc[-1],
    }
