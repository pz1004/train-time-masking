from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TabularDatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    categorical_columns: list[str]
    numerical_columns: list[str]
    metadata: dict[str, Any]


def load_dataset(dataset_config: dict[str, Any]) -> TabularDatasetBundle:
    dataset_info = dataset_config["dataset"]
    source = str(dataset_info.get("source", "openml"))
    if source != "openml":
        raise ValueError(f"Unsupported dataset source: {source}")
    return _load_openml_dataset(dataset_config)


def make_seed_splits(
    target: pd.Series,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seeds: list[int],
) -> dict[int, dict[str, Any]]:
    total_fraction = train_fraction + validation_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    row_ids = target.index.to_numpy()
    split_map: dict[int, dict[str, Any]] = {}
    temp_fraction = validation_fraction + test_fraction
    validation_share_of_temp = validation_fraction / temp_fraction

    for seed in seeds:
        train_ids, temp_ids, train_target, temp_target = train_test_split(
            row_ids,
            target.to_numpy(),
            test_size=temp_fraction,
            stratify=target.to_numpy(),
            random_state=seed,
        )
        validation_ids, test_ids, validation_target, test_target = train_test_split(
            temp_ids,
            temp_target,
            test_size=test_fraction / temp_fraction,
            stratify=temp_target,
            random_state=seed,
        )
        split_map[int(seed)] = {
            "seed": int(seed),
            "train_row_ids": [int(row_id) for row_id in train_ids.tolist()],
            "validation_row_ids": [int(row_id) for row_id in validation_ids.tolist()],
            "test_row_ids": [int(row_id) for row_id in test_ids.tolist()],
            "train_size": int(len(train_ids)),
            "validation_size": int(len(validation_ids)),
            "test_size": int(len(test_ids)),
            "train_positive_rate": float(train_target.mean()),
            "validation_positive_rate": float(validation_target.mean()),
            "test_positive_rate": float(test_target.mean()),
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_share_of_temp * temp_fraction),
            "test_fraction": float(test_fraction),
        }

    return split_map


def _load_openml_dataset(dataset_config: dict[str, Any]) -> TabularDatasetBundle:
    dataset_info = dataset_config["dataset"]
    data_home = str(dataset_info.get("data_home") or environ.get("SKLEARN_DATA", "/tmp/skdata"))

    fetch_kwargs: dict[str, Any] = {
        "as_frame": True,
        "parser": str(dataset_info.get("parser", "auto")),
        "data_home": data_home,
    }
    if "openml_id" in dataset_info:
        fetch_kwargs["data_id"] = int(dataset_info["openml_id"])
    else:
        fetch_kwargs["name"] = str(dataset_info["fetch_name"])
        if "version" in dataset_info:
            fetch_kwargs["version"] = int(dataset_info["version"])

    openml_dataset = fetch_openml(**fetch_kwargs)
    features = openml_dataset.data.copy()
    target = _extract_target(openml_dataset, dataset_info, features)

    missing_tokens = _missing_tokens(dataset_info)
    if missing_tokens:
        features = features.replace(missing_tokens, pd.NA)

    original_row_count = int(len(features))
    deduplicate = bool(dataset_info.get("deduplicate_feature_rows", True))
    duplicate_mask = features.astype("string").duplicated(keep="first") if deduplicate else pd.Series(False, index=features.index)
    duplicates_removed = int(duplicate_mask.sum())
    if duplicates_removed:
        features = features.loc[~duplicate_mask].copy()
        target = target.loc[features.index].copy()

    categorical_columns = _categorical_columns(features, dataset_info)
    numerical_columns = [column_name for column_name in features.columns if column_name not in categorical_columns]
    target = _binary_target(target, dataset_info)

    metadata = {
        "dataset_name": str(dataset_info.get("primary_dataset", dataset_info.get("fetch_name", "openml_dataset"))),
        "openml_id": int(dataset_info["openml_id"]) if "openml_id" in dataset_info else None,
        "openml_version": int(dataset_info["version"]) if "version" in dataset_info else None,
        "target_column": str(dataset_info["label_column"]),
        "positive_label": str(dataset_info["positive_label"]),
        "negative_label": str(dataset_info.get("negative_label", "not_positive_label")),
        "missing_tokens": missing_tokens,
        "categorical_inference": str(dataset_info.get("categorical_inference", "dtype_category_or_object")),
        "original_row_count": original_row_count,
        "deduplicated_row_count": int(len(features)),
        "duplicate_feature_rows_removed": duplicates_removed,
        "feature_count": int(features.shape[1]),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "missing_value_count": int(features.isna().sum().sum()),
        "missing_per_feature": {column_name: int(count) for column_name, count in features.isna().sum().items()},
        "class_balance": {str(label): float(value) for label, value in target.value_counts(normalize=True).sort_index().items()},
        "data_home": str(Path(data_home)),
    }

    return TabularDatasetBundle(
        features=features,
        target=target,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        metadata=metadata,
    )


def _extract_target(openml_dataset: Any, dataset_info: dict[str, Any], features: pd.DataFrame) -> pd.Series:
    label_column = str(dataset_info["label_column"])
    if openml_dataset.target is not None:
        target = openml_dataset.target.copy()
        target.name = label_column
        return target
    if label_column not in openml_dataset.frame:
        raise ValueError(f"OpenML dataset does not expose target column '{label_column}'.")
    target = openml_dataset.frame[label_column].copy()
    if label_column in features.columns:
        features.drop(columns=[label_column], inplace=True)
    return target


def _missing_tokens(dataset_info: dict[str, Any]) -> list[str]:
    if "missing_tokens" in dataset_info:
        return [str(token) for token in dataset_info.get("missing_tokens", [])]
    if "missing_token" in dataset_info:
        return [str(dataset_info["missing_token"])]
    return []


def _categorical_columns(features: pd.DataFrame, dataset_info: dict[str, Any]) -> list[str]:
    if "categorical_columns" in dataset_info:
        return [str(column_name) for column_name in dataset_info.get("categorical_columns", [])]
    return [
        column_name
        for column_name in features.columns
        if str(features[column_name].dtype) in {"category", "object", "string"}
        or str(features[column_name].dtype).startswith("string")
    ]


def _binary_target(target: pd.Series, dataset_info: dict[str, Any]) -> pd.Series:
    positive_label = str(dataset_info["positive_label"])
    binary_target = (target.astype("string") == positive_label).astype(int)
    binary_target.name = str(dataset_info["label_column"])
    return binary_target
