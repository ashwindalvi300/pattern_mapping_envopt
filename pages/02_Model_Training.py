# pages/02_MODEL_TRAINING.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detection import (
    apply_iforest,
    fit_scaler,
    train_isolation_forest,
    transform_data,
)
from src.anomaly_rules import compute_flags
from src.app_state import ensure_dataset_loaded
from src.artifact_manager import save_faiss, save_pickle
from src.baseline_statistics import build_historical_reference
from src.config import *
from src.feature_selection import build_correlation_matrix, select_features
from src.lag_analysis import calculate_cross_lag_correlations
from src.logger import StreamlitLogger
from src.pattern_matching import build_faiss_repository


st.set_page_config(
    page_title="Model Training",
    layout="wide"
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

st.title("Model Training")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    ensure_dataset_loaded()

except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()


def render_training_summary():

    summary = st.session_state.get("training_summary")
    logs = st.session_state.get("training_logs")

    if not summary:
        return

    st.success("Training Completed")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Selected Features",
            summary["selected_features"]
        )

    with c2:
        st.metric(
            "Historical Frames",
            summary["historical_frames"]
        )

    with c3:
        st.metric(
            "Training Rows",
            summary["training_rows"]
        )

    if logs:
        st.subheader("Training Logs")
        st.dataframe(
            pd.DataFrame(logs),
            use_container_width=True
        )


if st.button("Start / Re-run Training"):

    logger = StreamlitLogger()
    progress = st.progress(0)
    log_placeholder = st.empty()

    def update_logs():
        logs = logger.get_logs()
        st.session_state["training_logs"] = logs
        log_placeholder.dataframe(
            pd.DataFrame(logs),
            use_container_width=True
        )

    try:

        df = st.session_state["historical_raw"]

        missing_columns = [
            col
            for col in RAW_NUMERIC_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Historical dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        logger.info("Training started")
        update_logs()

        progress.progress(10)
        logger.info("Generating correlation matrix...")
        update_logs()

        corr_matrix = build_correlation_matrix(
            df,
            RAW_NUMERIC_COLUMNS
        )

        save_pickle(
            corr_matrix,
            "artifacts/corr_matrix.pkl"
        )

        logger.success("Correlation matrix completed")
        update_logs()

        progress.progress(20)
        logger.info("Running cross-lag analysis...")
        update_logs()

        lag_df = calculate_cross_lag_correlations(
            df,
            RAW_NUMERIC_COLUMNS,
            MAX_LAG
        )

        save_pickle(
            lag_df,
            "artifacts/lag_df.pkl"
        )

        logger.success("Cross-lag analysis completed")
        update_logs()

        progress.progress(35)

        selected_features = select_features(
            corr_matrix,
            lag_df
        )

        logger.success(f"{len(selected_features)} features selected")
        update_logs()

        save_pickle(
            selected_features,
            "artifacts/selected_features.pkl"
        )

        train_full = df[
            RAW_NUMERIC_COLUMNS
            +
            ["Timestamp"]
        ].copy()

        train_df = df[
            selected_features
            +
            ["Timestamp"]
        ].copy()

        progress.progress(45)
        logger.info("Fitting scaler...")
        update_logs()

        scaler = fit_scaler(
            train_df,
            selected_features
        )

        historical_df = transform_data(
            train_df,
            scaler,
            selected_features
        )

        save_pickle(
            scaler,
            "artifacts/scaler.pkl"
        )

        logger.success("Scaling completed")
        update_logs()

        progress.progress(60)
        logger.info("Training Isolation Forest...")
        update_logs()

        iso = train_isolation_forest(
            historical_df,
            selected_features
        )

        historical_df = apply_iforest(
            historical_df,
            iso,
            selected_features
        )

        save_pickle(
            iso,
            "artifacts/isolation_forest.pkl"
        )

        logger.success("Isolation Forest completed")
        update_logs()

        progress.progress(75)
        logger.info("Building historical references...")
        update_logs()

        (
            hist_means,
            hist_stds,
            diff_thresholds,
            value_envelopes
        ) = build_historical_reference(
            historical_df,
            train_full,
            selected_features
        )

        save_pickle(
            hist_means,
            "artifacts/hist_means.pkl"
        )

        save_pickle(
            hist_stds,
            "artifacts/hist_stds.pkl"
        )

        save_pickle(
            diff_thresholds,
            "artifacts/diff_thresholds.pkl"
        )

        save_pickle(
            value_envelopes,
            "artifacts/value_envelopes.pkl"
        )

        logger.success("Historical references built")
        update_logs()

        progress.progress(85)
        logger.info("Computing anomaly flags...")
        update_logs()

        historical_df["point_anomaly"] = compute_flags(
            historical_df,
            train_full,
            hist_means,
            hist_stds,
            diff_thresholds,
            value_envelopes,
            selected_features,
            PRIMARY_FEATURES,
            Z_POINT,
            Z_ROLLING,
            WINDOW_60
        )

        logger.success("Three-layer anomaly engine completed")
        update_logs()

        progress.progress(95)
        logger.info("Building FAISS repository...")
        update_logs()

        index, vectors, metadata = build_faiss_repository(
            historical_df,
            selected_features,
            PRIMARY_FEATURES,
            PRIMARY_WEIGHT,
            SECONDARY_WEIGHT,
            WINDOW_SIZE,
            STEP
        )

        np.save(
            ARTIFACTS_DIR / "historical_vectors.npy",
            vectors
        )

        save_pickle(
            metadata,
            "artifacts/frame_metadata.pkl"
        )

        save_faiss(
            index,
            "artifacts/gas_turbine_weighted_machine_state_memory.index"
        )

        logger.success("FAISS repository created")
        update_logs()

        st.session_state["selected_features"] = selected_features
        st.session_state["corr_matrix"] = corr_matrix
        st.session_state["lag_df"] = lag_df
        st.session_state["historical_df"] = historical_df
        st.session_state["training_summary"] = {
            "selected_features": len(selected_features),
            "historical_frames": len(metadata),
            "training_rows": len(df),
        }

        progress.progress(100)

        logger.success("Training completed successfully")
        update_logs()

        render_training_summary()

    except Exception as exc:

        st.session_state["training_logs"] = logger.get_logs()
        st.error(f"Training Failed: {exc}")

else:

    render_training_summary()
