# src/feature_selection.py

import pandas as pd


def build_correlation_matrix(df, feature_columns):

    return df[feature_columns].corr()


def select_features(
    corr_matrix,
    lag_df,
    corr_threshold=0.40,
    lag_threshold=0.40
):

    selected_features = set()

    for col in corr_matrix.columns:

        correlated = (
            abs(corr_matrix[col]) > corr_threshold
        )

        if correlated.sum() > 1:
            selected_features.add(col)

    strong_lag = lag_df[
        abs(
            lag_df["Best_Lag_Correlation"]
        ) > lag_threshold
    ]

    selected_features.update(
        strong_lag["Feature_1"]
    )

    selected_features.update(
        strong_lag["Feature_2"]
    )

    return list(selected_features)
