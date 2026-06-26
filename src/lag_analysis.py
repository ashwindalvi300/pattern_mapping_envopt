# src/lag_analysis.py

import numpy as np
import pandas as pd


def calculate_cross_lag_correlations(
    df,
    feature_columns,
    max_lag=1800
):

    lag_results = []

    for col1 in feature_columns:

        for col2 in feature_columns:

            if col1 == col2:
                continue

            best_corr = 0
            best_lag = 0

            s1 = df[col1].values
            s2 = df[col2].values

            for lag in range(1, max_lag + 1):

                lc = np.corrcoef(
                    s1[lag:],
                    s2[:-lag]
                )[0, 1]

                if (
                    np.isfinite(lc)
                    and abs(lc) > abs(best_corr)
                ):
                    best_corr = lc
                    best_lag = lag

            lag_results.append({
                "Feature_1": col1,
                "Feature_2": col2,
                "Best_Lag_Correlation": best_corr,
                "Best_Lag": best_lag
            })

    return pd.DataFrame(lag_results)