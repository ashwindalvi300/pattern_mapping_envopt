# src/anomaly_rules.py

import pandas as pd


def compute_flags(
    df_scaled,
    df_raw,
    hist_means,
    hist_stds,
    diff_thresholds,
    value_envelopes,
    selected_features,
    primary_features,
    z_point,
    z_rolling,
    rolling_window
):

    flags = pd.Series(
        0,
        index=df_scaled.index
    )

    for col in selected_features:

        mu = hist_means[col]
        sig = hist_stds[col]

        if pd.isna(sig) or sig == 0:
            continue

        z_pt = abs(
            (df_scaled[col] - mu) / sig
        )

        layer_a = (
            z_pt > z_point
        ).astype(int)

        if col in primary_features:

            roll_mean = (
                df_scaled[col]
                .rolling(
                    rolling_window,
                    min_periods=1
                )
                .mean()
            )

            z_rl = abs(
                (roll_mean - mu) / sig
            )

            layer_b = (
                z_rl > z_rolling
            ).astype(int)

        else:

            layer_b = pd.Series(
                0,
                index=df_scaled.index
            )

        # raw_diff = (
        #     df_raw[col]
        #     .diff()
        #     .abs()
        #     .fillna(0)
        # )

        # v_min, v_max = (
        #     value_envelopes[col]
        # )

        # outside_env = (
        #     (df_raw[col] < v_min)
        #     |
        #     (df_raw[col] > v_max)
        # )

        # layer_c = (
        #     (
        #         raw_diff
        #         > diff_thresholds[col]
        #     )
        #     &
        #     outside_env
        # ).astype(int)

        raw_diff = (
            df_raw[col]
            .diff()
            .abs()
            .fillna(0)
        )

        v_min, v_max = (
            value_envelopes[col]
        )

        outside_env = (
            (df_raw[col] < v_min)
            |
            (df_raw[col] > v_max)
        )

        layer_c = (
            raw_diff > diff_thresholds[col]
        ).astype(int)


        flags = (
            flags
            | layer_a
            | layer_b
            | layer_c
        )

    return flags
