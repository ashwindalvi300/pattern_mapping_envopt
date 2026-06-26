# src/artifact_manager.py

from pathlib import Path

import faiss
import joblib

APP_ROOT = Path(__file__).resolve().parents[1]


def resolve_artifact_path(path):

    artifact_path = Path(path)

    if not artifact_path.is_absolute():
        artifact_path = APP_ROOT / artifact_path

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return artifact_path


def save_pickle(obj, path):
    joblib.dump(
        obj,
        resolve_artifact_path(path)
    )


def load_pickle(path):
    return joblib.load(
        resolve_artifact_path(path)
    )


def save_faiss(index, path):
    faiss.write_index(
        index,
        str(resolve_artifact_path(path))
    )


def load_faiss(path):
    return faiss.read_index(
        str(resolve_artifact_path(path))
    )
