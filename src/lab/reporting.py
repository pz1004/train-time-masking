from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_severity_series(summary: dict[str, Any], *, metric_key: str) -> list[dict[str, Any]]:
    nominal_metric_key = "mean_auroc" if metric_key == "mean_auroc" else "mean_ece"
    nominal_lookup: dict[str, float] = {}
    nominal_lookup.update({name: float(metrics[nominal_metric_key]) for name, metrics in summary["baseline_summary"].items()})
    nominal_lookup.update({name: float(metrics[nominal_metric_key]) for name, metrics in summary["method_summary"].items()})
    nominal_lookup.update({name: float(metrics[nominal_metric_key]) for name, metrics in summary["ablation_summary"].items()})

    series = []
    for model_name, slice_map in summary["robustness_summary"].items():
        points = [{"x": 0.0, "y": nominal_lookup.get(model_name, float("nan"))}]
        ordered_slices = sorted(
            slice_map.items(),
            key=lambda item: float(item[1]["additional_mask_rate"]),
        )
        for _, metrics in ordered_slices:
            points.append({"x": float(metrics["additional_mask_rate"]) * 100.0, "y": float(metrics[metric_key])})
        series.append({"label": model_name, "points": points})
    return series


def build_prediction_frame(
    baseline_name: str,
    seed: int,
    split_name: str,
    target: pd.Series,
    probabilities: Any,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "baseline_name": baseline_name,
        "seed": seed,
        "split": split_name,
        "row_id": target.index.astype(int),
        "target": target.to_numpy(dtype=int),
        "predicted_probability": np.asarray(probabilities, dtype=float),
    }
    if extra_columns:
        for key, value in extra_columns.items():
            payload[key] = value
    return pd.DataFrame(payload)
