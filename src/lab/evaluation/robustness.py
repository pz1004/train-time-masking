from __future__ import annotations

from typing import Any
import hashlib

import numpy as np
import pandas as pd


def apply_missingness_overlay(
    features: pd.DataFrame,
    robustness_config: dict[str, Any],
    slice_config: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay = features.copy()
    columns = [str(column_name) for column_name in robustness_config["robustness"]["columns"]]
    additional_mask_rate = float(slice_config["additional_mask_rate"])
    slice_name = str(slice_config["name"])
    mask_only_observed = bool(robustness_config["robustness"].get("mask_only_observed_values", True))
    rng = np.random.default_rng(_overlay_seed(seed, slice_name))

    column_stats: dict[str, dict[str, Any]] = {}
    touched_rows: set[int] = set()

    for column_name in columns:
        eligible_mask = overlay[column_name].notna() if mask_only_observed else pd.Series(True, index=overlay.index)
        eligible_rows = overlay.index[eligible_mask].to_numpy(dtype=int)
        eligible_count = int(len(eligible_rows))
        target_mask_count = int(round(eligible_count * additional_mask_rate))
        if target_mask_count > eligible_count:
            target_mask_count = eligible_count

        if target_mask_count:
            masked_rows = rng.choice(eligible_rows, size=target_mask_count, replace=False)
            overlay.loc[masked_rows, column_name] = pd.NA
            touched_rows.update(int(row_id) for row_id in masked_rows.tolist())
        else:
            masked_rows = np.asarray([], dtype=int)

        column_stats[column_name] = {
            "eligible_count": eligible_count,
            "masked_count": int(len(masked_rows)),
            "requested_mask_rate": additional_mask_rate,
            "realized_mask_rate": 0.0 if eligible_count == 0 else round(len(masked_rows) / eligible_count, 6),
        }

    slice_metadata = {
        "slice_name": slice_name,
        "kind": str(slice_config["kind"]),
        "severity": str(slice_config["severity"]),
        "additional_mask_rate": additional_mask_rate,
        "columns": columns,
        "mask_only_observed_values": mask_only_observed,
        "overlay_seed": _overlay_seed(seed, slice_name),
        "row_count": int(len(overlay)),
        "rows_touched": int(len(touched_rows)),
        "column_stats": column_stats,
    }
    return overlay, slice_metadata


def apply_mar_overlay(
    features: pd.DataFrame,
    *,
    target_columns: list[str],
    driver_column: str,
    seed: int,
    low_rate: float = 0.05,
    high_rate: float = 0.35,
    threshold_quantile: float = 0.5,
    severity: str = "mar",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_mar_columns(driver_column, target_columns)
    overlay = features.copy()
    rng = np.random.default_rng(_overlay_seed(seed, f"mar:{driver_column}:{','.join(target_columns)}"))
    driver_series = overlay[driver_column]
    driver_mask = driver_series.notna()

    if pd.api.types.is_numeric_dtype(driver_series):
        threshold = float(pd.to_numeric(driver_series[driver_mask], errors="coerce").quantile(threshold_quantile))
        high_probability_rows = pd.to_numeric(driver_series, errors="coerce") >= threshold
        driver_metadata: dict[str, Any] = {
            "driver_type": "numeric",
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
        }
    else:
        dominant_category = (
            driver_series.astype("string").loc[driver_mask].value_counts().sort_values(ascending=False).index[0]
            if driver_mask.any()
            else "__missing__"
        )
        high_probability_rows = driver_series.astype("string") == dominant_category
        driver_metadata = {
            "driver_type": "categorical",
            "dominant_category": str(dominant_category),
        }

    column_stats: dict[str, dict[str, Any]] = {}
    touched_rows: set[int] = set()
    row_probabilities = np.where(high_probability_rows.to_numpy(dtype=bool), high_rate, low_rate)

    for column_name in target_columns:
        eligible_mask = overlay[column_name].notna()
        eligible_positions = np.flatnonzero(eligible_mask.to_numpy(dtype=bool))
        sampled_positions = []
        for position in eligible_positions:
            if rng.random() < float(row_probabilities[position]):
                sampled_positions.append(position)
        if sampled_positions:
            row_ids = overlay.index.to_numpy()[np.asarray(sampled_positions, dtype=int)]
            overlay.loc[row_ids, column_name] = pd.NA
            touched_rows.update(int(row_id) for row_id in row_ids.tolist())
        column_stats[column_name] = {
            "eligible_count": int(len(eligible_positions)),
            "masked_count": int(len(sampled_positions)),
            "low_rate": float(low_rate),
            "high_rate": float(high_rate),
            "realized_mask_rate": round(0.0 if not eligible_positions.size else len(sampled_positions) / len(eligible_positions), 6),
        }

    metadata = {
        "slice_name": "missingness_mar",
        "kind": "mar_missingness",
        "severity": severity,
        "driver_column": driver_column,
        "driver_metadata": driver_metadata,
        "target_columns": list(target_columns),
        "low_rate": float(low_rate),
        "high_rate": float(high_rate),
        "overlay_seed": _overlay_seed(seed, f"mar:{driver_column}:{','.join(target_columns)}"),
        "row_count": int(len(overlay)),
        "rows_touched": int(len(touched_rows)),
        "column_stats": column_stats,
    }
    return overlay, metadata


def apply_mnar_overlay(
    features: pd.DataFrame,
    *,
    target_columns: list[str],
    seed: int,
    low_rate: float = 0.05,
    high_rate: float = 0.35,
    threshold_quantile: float = 0.75,
    tail: str = "upper",
    configured_categories: dict[str, list[str]] | None = None,
    severity: str = "mnar",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay = features.copy()
    rng = np.random.default_rng(_overlay_seed(seed, f"mnar:{tail}:{','.join(target_columns)}"))
    configured_categories = configured_categories or {}
    column_stats: dict[str, dict[str, Any]] = {}
    touched_rows: set[int] = set()

    for column_name in target_columns:
        column_series = overlay[column_name]
        eligible_mask = column_series.notna()
        high_probability_rows, mechanism_metadata = _mnar_high_probability_rows(
            column_series,
            eligible_mask=eligible_mask,
            low_rate=float(low_rate),
            high_rate=float(high_rate),
            threshold_quantile=float(threshold_quantile),
            tail=tail,
            configured_categories=configured_categories.get(column_name),
        )
        row_probabilities = np.where(high_probability_rows.to_numpy(dtype=bool), high_rate, low_rate)
        eligible_positions = np.flatnonzero(eligible_mask.to_numpy(dtype=bool))
        sampled_positions = [
            int(position)
            for position in eligible_positions
            if rng.random() < float(row_probabilities[int(position)])
        ]
        if sampled_positions:
            row_ids = overlay.index.to_numpy()[np.asarray(sampled_positions, dtype=int)]
            overlay.loc[row_ids, column_name] = pd.NA
            touched_rows.update(int(row_id) for row_id in row_ids.tolist())
        column_stats[column_name] = {
            "eligible_count": int(len(eligible_positions)),
            "masked_count": int(len(sampled_positions)),
            "low_rate": float(low_rate),
            "high_rate": float(high_rate),
            "realized_mask_rate": round(0.0 if not eligible_positions.size else len(sampled_positions) / len(eligible_positions), 6),
            "mechanism": mechanism_metadata,
        }

    metadata = {
        "slice_name": "missingness_mnar",
        "kind": "mnar_missingness",
        "severity": severity,
        "target_columns": list(target_columns),
        "low_rate": float(low_rate),
        "high_rate": float(high_rate),
        "threshold_quantile": float(threshold_quantile),
        "tail": str(tail),
        "overlay_seed": _overlay_seed(seed, f"mnar:{tail}:{','.join(target_columns)}"),
        "row_count": int(len(overlay)),
        "rows_touched": int(len(touched_rows)),
        "column_stats": column_stats,
    }
    return overlay, metadata


def apply_structured_missingness_overlay(
    features: pd.DataFrame,
    overlay_config: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    kind = str(overlay_config["kind"])
    if kind == "mar":
        overlay, metadata = apply_mar_overlay(
            features,
            target_columns=[str(column_name) for column_name in overlay_config["target_columns"]],
            driver_column=str(overlay_config["driver_column"]),
            seed=seed,
            low_rate=float(overlay_config.get("low_rate", 0.05)),
            high_rate=float(overlay_config.get("high_rate", 0.35)),
            threshold_quantile=float(overlay_config.get("threshold_quantile", 0.5)),
            severity=str(overlay_config.get("name", "mar")),
        )
    elif kind == "mnar":
        overlay, metadata = apply_mnar_overlay(
            features,
            target_columns=[str(column_name) for column_name in overlay_config["target_columns"]],
            seed=seed,
            low_rate=float(overlay_config.get("low_rate", 0.05)),
            high_rate=float(overlay_config.get("high_rate", 0.35)),
            threshold_quantile=float(overlay_config.get("threshold_quantile", 0.75)),
            tail=str(overlay_config.get("tail", "upper")),
            configured_categories={
                str(column_name): [str(value) for value in values]
                for column_name, values in dict(overlay_config.get("configured_categories", {})).items()
            },
            severity=str(overlay_config.get("name", "mnar")),
        )
    else:
        raise ValueError(f"Unsupported structured missingness kind: {kind}")

    slice_name = str(overlay_config.get("name", metadata["slice_name"]))
    metadata = {**metadata, "slice_name": slice_name, "configured_kind": kind}
    if "additional_mask_rate" not in metadata:
        metadata["additional_mask_rate"] = float(overlay_config.get("high_rate", metadata.get("high_rate", 0.0)))
    return overlay, metadata


def default_structured_overlay_configs(robustness_config: dict[str, Any]) -> list[dict[str, Any]]:
    robustness_columns = [str(column_name) for column_name in robustness_config["robustness"]["columns"]]
    configured_mar = dict(robustness_config.get("mar", {}))
    if configured_mar:
        mar_driver = str(configured_mar["driver_column"])
        mar_targets = [str(column_name) for column_name in configured_mar["target_columns"]]
    else:
        if len(robustness_columns) < 2:
            raise ValueError(
                "Structured MAR overlays require either a [mar] config or at least two robustness columns "
                "so the driver column is distinct from the target column."
            )
        mar_driver = robustness_columns[1]
        mar_targets = [robustness_columns[0]]
    _validate_mar_columns(mar_driver, mar_targets)

    mnar_targets = list(robustness_columns)
    return [
        {
            "name": "mar_primary",
            "kind": "mar",
            "driver_column": mar_driver,
            "target_columns": mar_targets,
            "low_rate": float(configured_mar.get("low_rate", 0.05)),
            "high_rate": float(configured_mar.get("high_rate", 0.35)),
            "threshold_quantile": float(configured_mar.get("threshold_quantile", 0.5)),
        },
        {
            "name": "mar_stress",
            "kind": "mar",
            "driver_column": mar_driver,
            "target_columns": mar_targets,
            "low_rate": 0.10,
            "high_rate": 0.50,
            "threshold_quantile": float(configured_mar.get("threshold_quantile", 0.5)),
        },
        {
            "name": "mnar_primary",
            "kind": "mnar",
            "target_columns": mnar_targets,
            "low_rate": 0.05,
            "high_rate": 0.35,
            "threshold_quantile": 0.75,
            "tail": "upper",
        },
        {
            "name": "mnar_stress",
            "kind": "mnar",
            "target_columns": mnar_targets,
            "low_rate": 0.10,
            "high_rate": 0.50,
            "threshold_quantile": 0.75,
            "tail": "upper",
        },
    ]


def _validate_mar_columns(driver_column: str, target_columns: list[str]) -> None:
    if str(driver_column) in {str(column_name) for column_name in target_columns}:
        raise ValueError(
            "MAR overlays require a driver column distinct from all target columns; "
            f"got driver={driver_column!r}, targets={target_columns!r}."
        )


def _mnar_high_probability_rows(
    column_series: pd.Series,
    *,
    eligible_mask: pd.Series,
    low_rate: float,
    high_rate: float,
    threshold_quantile: float,
    tail: str,
    configured_categories: list[str] | None,
) -> tuple[pd.Series, dict[str, Any]]:
    if pd.api.types.is_numeric_dtype(column_series):
        numeric_values = pd.to_numeric(column_series[eligible_mask], errors="coerce")
        threshold = float(numeric_values.quantile(threshold_quantile))
        all_numeric = pd.to_numeric(column_series, errors="coerce")
        if tail == "lower":
            high_probability_rows = all_numeric <= threshold
        else:
            high_probability_rows = all_numeric >= threshold
        return high_probability_rows.fillna(False), {
            "mechanism_type": "numeric_self_dependent",
            "tail": str(tail),
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
            "low_rate": float(low_rate),
            "high_rate": float(high_rate),
        }

    as_string = column_series.astype("string")
    if configured_categories:
        rare_categories = [str(value) for value in configured_categories]
    else:
        counts = as_string.loc[eligible_mask].value_counts(normalize=True)
        if counts.empty:
            rare_categories = []
        else:
            cutoff = float(counts.quantile(0.25))
            rare_categories = counts[counts <= cutoff].index.astype(str).tolist()
            if not rare_categories:
                rare_categories = [str(counts.sort_values(ascending=True).index[0])]
    high_probability_rows = as_string.isin(rare_categories)
    return high_probability_rows.fillna(False), {
        "mechanism_type": "categorical_self_dependent",
        "selected_categories": rare_categories,
        "low_rate": float(low_rate),
        "high_rate": float(high_rate),
    }


def _overlay_seed(seed: int, slice_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{slice_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)
