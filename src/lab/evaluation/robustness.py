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


def _overlay_seed(seed: int, slice_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{slice_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)
