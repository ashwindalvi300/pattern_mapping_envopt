# src/anomaly_detection.py

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def fit_scaler(df, selected_features):

    scaler = StandardScaler()

    scaler.fit(
        df[selected_features]
    )

    return scaler


def transform_data(
    df,
    scaler,
    selected_features
):

    output = df.copy()

    output[selected_features] = scaler.transform(
        output[selected_features]
    )

    return output


def train_isolation_forest(
    historical_df,
    selected_features
):

    model = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=200
    )

    model.fit(
        historical_df[selected_features]
    )

    return model


def apply_iforest(
    df,
    model,
    selected_features
):

    output = df.copy()

    output["if_score"] = model.decision_function(
        output[selected_features]
    )

    output["anomaly"] = model.predict(
        output[selected_features]
    )

    return output
