"""Reusable Adult Census preprocessing and feature export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


TARGET_MAPPING = {"<=50K": 0, ">50K": 1}


def prepare_xy(frame: pd.DataFrame, *, feature_columns=None):
    """Separate labels and normalize missing markers without fitting anything.

    Pass training feature_columns when preparing the held-out test partition.
    The loader must already have removed whitespace and test-label periods.
    """
    if "income" not in frame.columns or frame.empty:
        raise ValueError("Expected a nonempty DataFrame with an income column.")
    if frame["income"].isna().any():
        raise ValueError("Income contains missing labels.")
    if not set(frame["income"].unique()).issubset(TARGET_MAPPING):
        raise ValueError("Income must contain only <=50K and >50K.")

    X = frame.drop(columns="income").copy()
    y = frame["income"].map(TARGET_MAPPING).astype(int)
    if feature_columns is not None:
        if set(X.columns) != set(feature_columns):
            raise ValueError("Train and test feature columns do not match.")
        X = X.loc[:, list(feature_columns)]

    categorical = X.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    return X, y


def build_preprocessor(
    X: pd.DataFrame,
    *,
    scaler="standard",
    feature_range=(0, 1),
    numeric_strategy="median",
    categorical_strategy="most_frequent",
):
    """Create an unfitted transformer; fit it only inside a model Pipeline."""
    if "income" in X.columns:
        raise ValueError("Remove the target income before building preprocessing.")
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if len(numeric_cols) + len(categorical_cols) != X.shape[1]:
        raise ValueError("Unsupported feature dtype detected.")
    if numeric_strategy not in {"mean", "median"}:
        raise ValueError("numeric_strategy must be mean or median.")
    if categorical_strategy not in {"most_frequent", "constant"}:
        raise ValueError("categorical_strategy must be most_frequent or constant.")

    if scaler == "standard":
        scaling = StandardScaler()
    elif scaler == "minmax":
        scaling = MinMaxScaler(feature_range=feature_range)
    elif scaler == "none":
        scaling = "passthrough"
    else:
        raise ValueError("scaler must be standard, minmax or none.")

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", scaling),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(
            strategy=categorical_strategy, fill_value="Unknown"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])


def export_features(
    transformer, X_train, X_test, y_train, y_test,
    *, output_dir="features", metadata=None,
):
    """Transform with a fitted transformer, save arrays, and verify round trips.

    No fit is performed. Output filenames match the original notebook's cfg1.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_features = np.asarray(transformer.transform(X_train))
    test_features = np.asarray(transformer.transform(X_test))
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Expected dense two-dimensional feature arrays.")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Feature dimensions do not match.")
    if len(train_features) != len(y_train) or len(test_features) != len(y_test):
        raise ValueError("Feature and label counts do not match.")
    if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
        raise ValueError("Transformed features contain NaN or infinity.")

    arrays = {
        "X_train_cfg1.npy": train_features,
        "X_test_cfg1.npy": test_features,
        "y_train.npy": np.asarray(y_train),
        "y_test.npy": np.asarray(y_test),
    }
    rows = []
    for filename, array in arrays.items():
        path = destination / filename
        np.save(path, array, allow_pickle=False)
        restored = np.load(path, allow_pickle=False)
        if not np.array_equal(restored, array):
            raise RuntimeError(f"Round-trip verification failed: {filename}")
        rows.append({
            "file": filename, "shape": array.shape,
            "size_MiB": path.stat().st_size / 1024**2,
        })

    details = {
        **(metadata or {}),
        "configuration_id": "cfg1",
        "target_mapping": TARGET_MAPPING,
        "feature_names": transformer.get_feature_names_out().tolist(),
        "row_order": "Preserved from supplied X and y; no filtering or sorting",
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (destination / "metadata_cfg1.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)
    return pd.DataFrame(rows)
