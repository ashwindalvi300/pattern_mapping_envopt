# pages/04_LIVE_MONITORING.py

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.anomaly_detection import (
    apply_iforest,
    transform_data,
)
from src.anomaly_rules import compute_flags
from src.artifact_manager import (
    load_faiss,
    load_pickle,
)
from src.app_state import ensure_dataset_loaded
from src.config import *
from src.live_memory import save_live_windows
from src.vector_utils import (
    build_weighted_vector,
    unpack_interleaved_vector,
)


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def dataset_signature(df):
    return (
        len(df),
        str(df["Timestamp"].iloc[0]),
        str(df["Timestamp"].iloc[-1]),
    )


def reset_stream_state():
    st.session_state["live_stream_cursor"] = 0
    st.session_state["live_stream_running"] = False
    st.session_state["live_stream_alerts"] = []
    st.session_state["live_stream_windows"] = []
    st.session_state["live_stream_last_event"] = None


def ensure_stream_state(signature):
    if st.session_state.get("live_stream_signature") != signature:
        st.session_state["live_stream_signature"] = signature
        reset_stream_state()

    st.session_state.setdefault("live_stream_cursor", 0)
    st.session_state.setdefault("live_stream_running", False)
    st.session_state.setdefault("live_stream_alerts", [])
    st.session_state.setdefault("live_stream_windows", [])
    st.session_state.setdefault("live_stream_last_event", None)


def format_display_cell(value):
    if isinstance(value, (list, tuple, dict)):
        return str(value)

    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False

    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""

    return str(value)


def prepare_dataframe_for_display(df):
    display_df = df.copy()

    for col in ["spike_features", "top_sensors"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_display_cell)

    return display_df


def classify_window(
    current_chunk,
    current_chunk_raw,
    current_time,
    window_start,
    window_end,
    selected_features,
    frame_metadata,
    index,
    hist_means,
    hist_stds,
    diff_thresholds,
    value_envelopes,
):
    live_vector = build_weighted_vector(
        current_chunk,
        selected_features,
        PRIMARY_FEATURES,
        PRIMARY_WEIGHT,
        SECONDARY_WEIGHT
    )

    distances, indices = index.search(
        live_vector.reshape(1, -1),
        k=3
    )

    similarity = float(distances[0][0])
    matched_frame = int(indices[0][0])
    matched_meta = frame_metadata[matched_frame]

    current_point_ratio = current_chunk["point_anomaly"].mean()
    current_if_score = current_chunk["if_score"].mean()

    spike_features = []

    # for col in selected_features:
    #     raw_diff = current_chunk_raw[col].diff().abs().fillna(0)
    #     v_min, v_max = value_envelopes[col]

    #     outside_env = (
    #         (current_chunk_raw[col] < v_min)
    #         |
    #         (current_chunk_raw[col] > v_max)
    #     )

    #     spike_rows = raw_diff[
    #         (raw_diff > diff_thresholds[col])
    #         &
    #         outside_env
    #     ]

    #     if len(spike_rows) > 0:
    #         spike_features.append(
    #             (
    #                 col,
    #                 round(float(spike_rows.max()), 4)
    #             )
    #         )

    for col in selected_features:
        raw_diff = current_chunk_raw[col].diff().abs().fillna(0)
        v_min, v_max = value_envelopes[col]

        outside_env = (
            (current_chunk_raw[col] < v_min)
            |
            (current_chunk_raw[col] > v_max)
        )

        spike_rows = raw_diff[
            raw_diff > diff_thresholds[col]
        ]

        if len(spike_rows) > 0:
            spike_features.append(
                (
                    col,
                    round(float(spike_rows.max()), 4)
                )
            )

    window_z = {
        col: round(
            abs(
                (
                    current_chunk[col].mean()
                    -
                    hist_means[col]
                )
                /
                hist_stds[col]
            ),
            3
        )
        for col in selected_features
        if not pd.isna(hist_stds[col]) and hist_stds[col] != 0
    }

    top_sensors = sorted(
        window_z,
        key=window_z.get,
        reverse=True
    )[:3]

    alert_type = None

    if current_point_ratio >= POINT_RATIO_CRITICAL:
        alert_type = "CRITICAL"

    elif spike_features:
        alert_type = "SPIKE ALERT"

    elif (
        similarity > SIMILARITY_THRESHOLD
        and matched_meta["anomaly_ratio"] > HIST_ANOM_RATIO
        and matched_meta["point_anomaly_ratio"] > HIST_POINT_RATIO
        and current_if_score < IF_SCORE_THRESHOLD
    ):
        alert_type = "HIGH RISK"

    elif (
        current_point_ratio >= POINT_RATIO_WARNING
        or (
            similarity > SIMILARITY_THRESHOLD
            and matched_meta["point_anomaly_ratio"] > HIST_POINT_RATIO
        )
    ):
        alert_type = "EARLY WARNING"

    return {
        "window_start": window_start,
        "window_end": window_end,
        "time": str(current_time),
        "alert_type": alert_type or "NORMAL",
        "similarity": round(similarity, 4),
        "point_ratio": round(current_point_ratio, 4),
        "if_score": round(current_if_score, 4),
        "matched_frame": matched_frame,
        "spike_features": spike_features,
        "top_sensors": top_sensors,
    }


st.set_page_config(
    page_title="Live Monitoring",
    layout="wide"
)

st.title("Live Monitoring")

try:
    ensure_dataset_loaded()

except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

try:
    scaler = load_pickle("artifacts/scaler.pkl")
    iso = load_pickle("artifacts/isolation_forest.pkl")
    selected_features = load_pickle("artifacts/selected_features.pkl")
    hist_means = load_pickle("artifacts/hist_means.pkl")
    hist_stds = load_pickle("artifacts/hist_stds.pkl")
    diff_thresholds = load_pickle("artifacts/diff_thresholds.pkl")
    value_envelopes = load_pickle("artifacts/value_envelopes.pkl")
    frame_metadata = load_pickle("artifacts/frame_metadata.pkl")
    index = load_faiss(
        "artifacts/gas_turbine_weighted_machine_state_memory.index"
    )
    historical_vectors = np.load(
        ARTIFACTS_DIR / "historical_vectors.npy"
    )

except Exception as exc:
    st.warning(
        f"Please train the model first. Missing artifact: {exc}"
    )
    st.stop()

live_raw_full = st.session_state["live_raw"].copy()

st.info(
    f"Using live split from configured dataset: {live_raw_full.shape}"
)

missing_features = [
    col
    for col in selected_features
    if col not in live_raw_full.columns
]

if missing_features:
    st.error(
        "Live dataset is missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()

if len(live_raw_full) < WINDOW_SIZE:
    st.warning(
        f"Live dataset needs at least {WINDOW_SIZE} rows."
    )
    st.stop()

ensure_stream_state(
    dataset_signature(live_raw_full)
)

st.subheader("Streaming Controls")

control_cols = st.columns([1, 1, 1, 1, 2])

with control_cols[0]:
    if st.button("Start Stream", use_container_width=True):
        st.session_state["live_stream_running"] = True

with control_cols[1]:
    if st.button("Pause", use_container_width=True):
        st.session_state["live_stream_running"] = False

with control_cols[2]:
    step_once = st.button("Ingest One Row", use_container_width=True)

with control_cols[3]:
    if st.button("Reset", use_container_width=True):
        reset_stream_state()
        st.rerun()

with control_cols[4]:
    stream_delay = st.slider(
        "Seconds per row",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05
    )

cursor = st.session_state["live_stream_cursor"]
should_ingest = (
    step_once
    or st.session_state["live_stream_running"]
)

if should_ingest and cursor < len(live_raw_full):
    cursor += 1
    st.session_state["live_stream_cursor"] = cursor
    st.session_state["live_stream_last_event"] = {
        "type": "row",
        "message": f"Ingested row {cursor}"
    }

elif cursor >= len(live_raw_full):
    st.session_state["live_stream_running"] = False

ingested_raw_full = live_raw_full.iloc[:cursor].copy()

if cursor == 0:
    st.info("Press Start Stream or Ingest One Row to begin.")
    st.stop()

live_raw = ingested_raw_full[
    selected_features + ["Timestamp"]
].copy()

live_df = transform_data(
    live_raw,
    scaler,
    selected_features
)

live_df = apply_iforest(
    live_df,
    iso,
    selected_features
)

live_df["point_anomaly"] = compute_flags(
    live_df,
    ingested_raw_full,
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

chunk_completed = (
    cursor >= WINDOW_SIZE
    and (cursor - WINDOW_SIZE) % STEP == 0
)

already_processed = {
    item["window_end"]
    for item in st.session_state["live_stream_windows"]
}

if chunk_completed and cursor not in already_processed:
    window_start = cursor - WINDOW_SIZE
    window_end = cursor
    current_chunk = live_df.iloc[window_start:window_end]
    current_chunk_raw = ingested_raw_full.iloc[window_start:window_end]
    current_time = current_chunk["Timestamp"].iloc[-1]

    result = classify_window(
        current_chunk,
        current_chunk_raw,
        current_time,
        window_start,
        window_end,
        selected_features,
        frame_metadata,
        index,
        hist_means,
        hist_stds,
        diff_thresholds,
        value_envelopes,
    )

    st.session_state["live_stream_windows"].append(result)
    save_live_windows([result])

    if result["alert_type"] != "NORMAL":
        st.session_state["live_stream_alerts"].append(result)
        st.session_state["live_stream_last_event"] = {
            "type": "alert",
            "message": (
                f"{result['alert_type']} generated at "
                f"{result['time']}"
            )
        }
    else:
        st.session_state["live_stream_last_event"] = {
            "type": "window",
            "message": (
                "Chunk completed and mapped as NORMAL at "
                f"{result['time']}"
            )
        }

alerts_df = pd.DataFrame(
    st.session_state["live_stream_alerts"]
)

window_results_df = pd.DataFrame(
    st.session_state["live_stream_windows"]
)

latest_row = ingested_raw_full.iloc[-1]
rows_in_current_chunk = (
    min(cursor, WINDOW_SIZE)
    if cursor < WINDOW_SIZE
    else WINDOW_SIZE
)

next_mapping_in = (
    WINDOW_SIZE - cursor
    if cursor < WINDOW_SIZE
    else STEP - ((cursor - WINDOW_SIZE) % STEP)
)

if next_mapping_in == STEP and chunk_completed:
    next_mapping_in = STEP

status_cols = st.columns(5)

with status_cols[0]:
    st.metric("Rows Ingested", f"{cursor} / {len(live_raw_full)}")

with status_cols[1]:
    st.metric("Rows In Current Chunk", f"{rows_in_current_chunk} / {WINDOW_SIZE}")

with status_cols[2]:
    st.metric("Next Mapping In", next_mapping_in)

with status_cols[3]:
    st.metric("Chunks Mapped", len(window_results_df))

with status_cols[4]:
    st.metric("Alerts Generated", len(alerts_df))

progress_value = min(cursor / len(live_raw_full), 1.0)
st.progress(progress_value)

last_event = st.session_state["live_stream_last_event"]

if last_event:
    if last_event["type"] == "alert":
        st.error(last_event["message"])
    elif last_event["type"] == "window":
        st.success(last_event["message"])
    else:
        st.info(last_event["message"])

st.subheader("Current Ingested Row")

current_row_df = pd.DataFrame([latest_row])

st.dataframe(
    current_row_df,
    use_container_width=True
)

st.subheader("Latest Rolling Buffer")

buffer_preview = ingested_raw_full.tail(
    min(WINDOW_SIZE, cursor)
)

st.dataframe(
    buffer_preview.tail(20),
    use_container_width=True
)

st.subheader("Alert Stream")

if len(alerts_df):
    st.dataframe(
        prepare_dataframe_for_display(alerts_df.tail(50)),
        use_container_width=True
    )
else:
    st.caption("No alerts yet. Alerts are generated only when a chunk is mapped.")

st.subheader("Chunk Mapping History")

if len(window_results_df):
    st.dataframe(
        prepare_dataframe_for_display(window_results_df.tail(50)),
        use_container_width=True
    )
else:
    st.caption(
        f"Waiting for the first {WINDOW_SIZE}-row chunk before pattern mapping."
    )

st.subheader("Live Signal Viewer")

chart_features = st.multiselect(
    "Features",
    selected_features,
    default=selected_features[: min(4, len(selected_features))]
)

chart_mode = st.radio(
    "Chart Values",
    ["Raw sensor values", "Scaled model values"],
    horizontal=True
)

if chart_mode == "Raw sensor values":
    plot_source_df = ingested_raw_full
    y_axis_note = "raw"
else:
    plot_source_df = live_df
    y_axis_note = "scaled"

latest_plot_time = plot_source_df["Timestamp"].max()
plot_start_time = latest_plot_time - pd.Timedelta(hours=12)
plot_df = plot_source_df[
    plot_source_df["Timestamp"] >= plot_start_time
]

st.caption(
    "Showing live data for the last 12 hours: "
    f"{plot_start_time} to {latest_plot_time}"
)

if chart_features:
    fig = make_subplots(
        rows=len(chart_features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[
            f"{feature} ({y_axis_note})"
            for feature in chart_features
        ]
    )

    for idx, col in enumerate(chart_features):
        fig.add_trace(
            go.Scattergl(
                x=plot_df["Timestamp"],
                y=plot_df[col],
                mode="lines",
                showlegend=False
            ),
            row=idx + 1,
            col=1
        )

    fig.update_layout(
        height=min(900, 260 * len(chart_features)),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

if len(window_results_df):
    st.subheader("Latest Pattern Match")
    st.caption(
        "Pattern comparison is min-max normalized to compare shape. "
        "Use Live Signal Viewer for raw sensor units."
    )

    latest_window = st.session_state["live_stream_windows"][-1]
    latest_chunk = live_df.iloc[
        latest_window["window_start"]:
        latest_window["window_end"]
    ]

    latest_vector = build_weighted_vector(
        latest_chunk,
        selected_features,
        PRIMARY_FEATURES,
        PRIMARY_WEIGHT,
        SECONDARY_WEIGHT
    )

    distances, indices = index.search(
        latest_vector.reshape(1, -1),
        k=3
    )

    matches = []

    for rank in range(3):
        frame_id = int(indices[0][rank])
        meta = frame_metadata[frame_id]

        matches.append({
            "Rank": rank + 1,
            "Similarity %": round(float(distances[0][rank]) * 100, 2),
            "Frame ID": frame_id,
            "Historical IF Ratio %": round(meta["anomaly_ratio"] * 100, 2),
            "Historical Point Ratio %": round(
                meta["point_anomaly_ratio"] * 100,
                2
            )
        })

    st.dataframe(
        pd.DataFrame(matches),
        use_container_width=True
    )

    best_idx = int(indices[0][0])

    historical_series = unpack_interleaved_vector(
        historical_vectors[best_idx],
        selected_features,
        WINDOW_SIZE
    )

    compare_features = st.multiselect(
        "Compare Features",
        selected_features,
        default=selected_features[: min(3, len(selected_features))]
    )

    def minmax(values):
        values = np.array(values)
        value_range = np.max(values) - np.min(values)

        if value_range == 0:
            return values

        return (values - np.min(values)) / value_range

    if compare_features:
        compare_fig = make_subplots(
            rows=len(compare_features),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.04,
            subplot_titles=compare_features
        )

        for idx, feature in enumerate(compare_features):
            compare_fig.add_trace(
                go.Scatter(
                    y=minmax(latest_chunk[feature].values),
                    mode="lines",
                    name="Live Chunk",
                    line=dict(color="blue"),
                    showlegend=(idx == 0)
                ),
                row=idx + 1,
                col=1
            )

            compare_fig.add_trace(
                go.Scatter(
                    y=minmax(historical_series[feature]),
                    mode="lines",
                    name="Historical Match",
                    line=dict(color="green"),
                    showlegend=(idx == 0)
                ),
                row=idx + 1,
                col=1
            )

        compare_fig.update_layout(
            height=min(900, 260 * len(compare_features)),
            hovermode="x unified"
        )

        st.plotly_chart(
            compare_fig,
            use_container_width=True
        )

if st.session_state["live_stream_running"] and cursor < len(live_raw_full):
    time.sleep(stream_delay)
    st.rerun()
