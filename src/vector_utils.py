# src/vector_utlis.py

import numpy as np


def build_weighted_vector(
    frame,
    selected_features,
    primary_features,
    primary_weight,
    secondary_weight,
):
    weighted = []

    for col in selected_features:
        data = frame[col].values.copy()

        data *= (
            primary_weight
            if col in primary_features
            else secondary_weight
        )

        weighted.append(data)

    vector = (
        np.column_stack(weighted)
        .flatten()
        .astype(np.float32)
    )

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def unpack_interleaved_vector(
    vector,
    selected_features,
    window_size,
):
    matrix = vector.reshape(
        window_size,
        len(selected_features)
    )

    return {
        feature: matrix[:, idx]
        for idx, feature in enumerate(selected_features)
    }
