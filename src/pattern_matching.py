# src/pattern_matching.py

import faiss
import numpy as np

from src.vector_utils import build_weighted_vector


def build_faiss_repository(
    historical_df,
    selected_features,
    primary_features,
    primary_weight,
    secondary_weight,
    window_size,
    step
):

    frames = []
    metadata = []

    for start in range(
        0,
        len(historical_df) - window_size,
        step
    ):

        frame = historical_df.iloc[
            start:start + window_size
        ]

        frames.append(frame)

    vectors = []

    for idx, frame in enumerate(frames):

        vector = build_weighted_vector(
            frame,
            selected_features,
            primary_features,
            primary_weight,
            secondary_weight,
        )

        vectors.append(vector)

        item = {
            "frame_id": idx,
            "anomaly_ratio":
                frame["anomaly"]
                .value_counts(normalize=True)
                .get(-1, 0),
            "point_anomaly_ratio":
                frame["point_anomaly"]
                .mean()
        }

        if "health_index" in frame.columns:
            item["avg_health"] = frame["health_index"].mean()

        if "power_kw" in frame.columns:
            item["avg_power"] = frame["power_kw"].mean()

        if "kw_per_tr" in frame.columns:
            item["avg_efficiency"] = frame["kw_per_tr"].mean()

        metadata.append(item)

    vectors = np.array(
        vectors,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    return index, vectors, metadata
