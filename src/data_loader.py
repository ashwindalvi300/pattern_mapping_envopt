# src/data_loader.py

import pandas as pd
import numpy as np


def load_file(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file type")

    return df


def prepare_timestamp(df):

    if "Timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Timestamp"})

    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column not found")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = (
        df.sort_values("Timestamp")
        .reset_index(drop=True)
    )

    return df


def add_engineered_features(df):

    output = df.copy()

    output["delta_t_chw"] = (
        output["chw_return_temp_c"]
        -
        output["chw_supply_temp_c"]
    )

    output["delta_t_cw"] = (
        output["cw_outlet_temp_c"]
        -
        output["cw_inlet_temp_c"]
    )

    output["compression_ratio"] = (
        output["discharge_pressure_bar"]
        /
        output["suction_pressure_bar"]
    )

    output["power_per_flow"] = (
        output["power_kw"]
        /
        (output["chw_flow_m3h"] + 1e-6)
    )

    output["vibration_total"] = np.sqrt(
        output["vibration_x"] ** 2
        +
        output["vibration_y"] ** 2
        +
        output["vibration_z"] ** 2
    )

    return output


def prepare_dataset(df):

    return add_engineered_features(
        prepare_timestamp(df)
    )


def split_historical_live(df, historical_ratio=0.60):

    split_index = int(len(df) * historical_ratio)

    historical_raw_full = (
        df.iloc[:split_index]
        .reset_index(drop=True)
    )

    live_raw_full = (
        df.iloc[split_index:]
        .reset_index(drop=True)
    )

    return historical_raw_full, live_raw_full
