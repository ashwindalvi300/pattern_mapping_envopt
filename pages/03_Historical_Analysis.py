# pages/03_HISTORICAL_ANALYSIS.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_state import ensure_dataset_loaded
from src.artifact_manager import load_pickle
from src.config import *


st.set_page_config(
    page_title="Historical Analysis",
    layout="wide"
)

st.title("Historical Analysis")

try:
    ensure_dataset_loaded()

except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

try:
    corr_matrix = load_pickle(
        "artifacts/corr_matrix.pkl"
    )

    lag_df = load_pickle(
        "artifacts/lag_df.pkl"
    )

except Exception:
    st.warning("Please train the model first.")
    st.stop()

historical_raw = st.session_state["historical_raw"]

st.header("Correlation Matrix")

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

st.header("Cross-Lag Relationships")

top_lag = lag_df.sort_values(
    by="Best_Lag_Correlation",
    ascending=False
)

st.dataframe(
    top_lag.head(50),
    use_container_width=True
)

st.header("Feature Statistics")

stats_df = (
    historical_raw[
        RAW_NUMERIC_COLUMNS
    ]
    .describe()
    .T
)

st.dataframe(
    stats_df,
    use_container_width=True
)

st.header("Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    RAW_NUMERIC_COLUMNS
)

fig_hist = px.histogram(
    historical_raw,
    x=feature,
    nbins=50,
    title=f"{feature} Distribution"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

st.header("Historical Time Series")

ts_feature = st.selectbox(
    "Time Series Feature",
    RAW_NUMERIC_COLUMNS,
    key="timeseries_feature"
)

fig_ts = go.Figure()

fig_ts.add_trace(
    go.Scatter(
        x=historical_raw["Timestamp"],
        y=historical_raw[ts_feature],
        mode="lines",
        name=ts_feature
    )
)

fig_ts.update_layout(
    title=f"{ts_feature} Over Time",
    hovermode="x unified"
)

st.plotly_chart(
    fig_ts,
    use_container_width=True
)

st.header("Missing Values")

missing_df = pd.DataFrame({
    "Column": historical_raw.columns,
    "Missing Values": historical_raw.isnull().sum().values
})

st.dataframe(
    missing_df,
    use_container_width=True
)

st.header("Dataset Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Rows",
        historical_raw.shape[0]
    )

with c2:
    st.metric(
        "Columns",
        historical_raw.shape[1]
    )

with c3:
    st.metric(
        "Features",
        len(RAW_NUMERIC_COLUMNS)
    )
