# src/baseline_statistics.py

import numpy as np


def build_historical_reference(
    historical_df,
    hist_raw_full,
    selected_features
):

    hist_means = (
        historical_df[selected_features]
        .mean()
    )

    hist_stds = (
        historical_df[selected_features]
        .std()
        .replace(0, np.nan)
    )

    diff_thresholds = {}
    value_envelopes = {}

    for col in selected_features:

        raw_diffs = (
            hist_raw_full[col]
            .diff()
            .abs()
            .dropna()
        )

        diff_thresholds[col] = raw_diffs.quantile(0.9999)

        value_envelopes[col] = (
            hist_raw_full[col].min(),
            hist_raw_full[col].max()
        )

    return (
        hist_means,
        hist_stds,
        diff_thresholds,
        value_envelopes
    )
