from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import shutil

import numpy as np
import pandas as pd

from lab.artifacts import read_json, reset_directory, stage_manifest_path, stage_seed_manifest_path, write_json, write_text
from lab.data import TabularDatasetBundle, make_seed_splits
from lab.evaluation.metrics import binary_classification_metrics
from lab.evaluation.robustness import apply_missingness_overlay
from lab.study import StudySpec


def split_map_from_protocol(spec: StudySpec, dataset_bundle: TabularDatasetBundle) -> dict[int, dict[str, Any]]:
    split_config = spec.configs["protocol"]["split"]
    return make_seed_splits(
        dataset_bundle.target,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        seeds=[int(seed) for seed in split_config["seeds"]],
    )


def prediction_frame(
    model_name: str,
    seed: int,
    split_name: str,
    target: pd.Series,
    probabilities: np.ndarray,
    *,
    raw_probabilities: np.ndarray,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "model_name": model_name,
        "seed": seed,
        "split": split_name,
        "row_id": target.index.astype(int),
        "target": target.to_numpy(dtype=int),
        "predicted_probability": np.asarray(probabilities, dtype=float),
        "raw_probability": np.asarray(raw_probabilities, dtype=float),
    }
    if extra_columns:
        payload.update(extra_columns)
    return pd.DataFrame(payload)


def robustness_artifacts(
    spec: StudySpec,
    stage_name: str,
    *,
    model_name: str,
    split_metadata: dict[str, Any],
    dataset_bundle: TabularDatasetBundle,
    predict_probabilities: Callable[[pd.DataFrame], np.ndarray],
    slice_config: dict[str, Any],
    robustness_config: dict[str, Any],
    nominal_reference_kind: str,
    model_metadata: dict[str, Any],
    software_versions: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()
    X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
    overlay_features, slice_metadata = apply_missingness_overlay(
        X_test,
        robustness_config,
        slice_config,
        seed=int(split_metadata["seed"]),
    )
    predict_start = perf_counter()
    probabilities = predict_probabilities(overlay_features)
    predict_seconds = round(perf_counter() - predict_start, 6)
    predictions = prediction_frame(
        model_name,
        int(split_metadata["seed"]),
        "test",
        y_test,
        probabilities,
        raw_probabilities=probabilities,
        extra_columns={"evaluation_slice": slice_metadata["slice_name"], "result_kind": "robustness_overlay"},
    )
    payload = {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "stage": stage_name,
        "result_kind": "robustness_overlay",
        "model_name": model_name,
        "baseline_name": model_name,
        "nominal_reference_name": model_name,
        "nominal_reference_kind": nominal_reference_kind,
        "evaluation_slice": slice_metadata["slice_name"],
        "seed": int(split_metadata["seed"]),
        "row_counts": {"test": int(split_metadata["test_size"])},
        "test_metrics": binary_classification_metrics(y_test.to_numpy(), probabilities),
        "fit_seconds": 0.0,
        "predict_seconds": predict_seconds,
        "model_metadata": model_metadata,
        "software_versions": software_versions,
    }
    return payload, predictions, slice_metadata


def write_run_artifacts(
    spec: StudySpec,
    stage_name: str,
    *,
    run_directory: Path,
    metrics_payload: dict[str, Any],
    predictions: pd.DataFrame,
    split_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
    slice_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.csv.gz"
    split_metadata_path = run_directory / "split_metadata.json"
    dataset_metadata_path = run_directory / "dataset_metadata.json"
    manifest_reference_path = run_directory / "manifest_reference.json"

    write_json(metrics_path, metrics_payload)
    predictions.sort_values(["split", "row_id"]).to_csv(predictions_path, index=False, compression="gzip")
    write_json(split_metadata_path, split_metadata)
    write_json(dataset_metadata_path, dataset_metadata)
    write_json(
        manifest_reference_path,
        {
            "stage_manifest": str(stage_manifest_path(spec, stage_name).relative_to(spec.root)),
            "seed_manifest": str(stage_seed_manifest_path(spec, stage_name).relative_to(spec.root)),
            "study_config": str(spec.study_config_path.relative_to(spec.root)),
        },
    )

    artifact_paths = [
        metrics_path,
        predictions_path,
        split_metadata_path,
        dataset_metadata_path,
        manifest_reference_path,
    ]
    if slice_metadata is not None:
        slice_metadata_path = run_directory / "slice_metadata.json"
        write_json(slice_metadata_path, slice_metadata)
        artifact_paths.append(slice_metadata_path)
    return artifact_paths


def read_metrics(run_directory: Path) -> dict[str, Any]:
    metrics_path = run_directory / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing required artifact: {metrics_path}")
    return read_json(metrics_path)


def clear_existing_robustness_runs(
    robustness_root: Path,
    model_name: str,
    slice_names: list[str],
    seed_list: list[int],
) -> None:
    robustness_root.mkdir(parents=True, exist_ok=True)
    for slice_name in slice_names:
        for seed in seed_list:
            run_directory = robustness_root / f"{model_name}__{slice_name}__seed_{seed}"
            if run_directory.exists():
                shutil.rmtree(run_directory)


def collect_variant_records(
    root: Path,
    model_name: str,
    seed_list: list[int],
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seed_list:
        run_directory = root / f"{model_name}__seed_{seed}"
        if allow_missing and not run_directory.exists():
            return []
        metrics = read_metrics(run_directory)
        predictions = pd.read_csv(run_directory / "predictions.csv.gz")
        records.append({"metrics": metrics, "predictions": predictions})
    return records


def summarize_variant_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    validation_metrics = [record["metrics"]["validation_metrics"] for record in records]
    test_metrics = [record["metrics"]["test_metrics"] for record in records]
    pre_validation_metrics = [record["metrics"]["pre_calibration_validation_metrics"] for record in records]
    pre_test_metrics = [record["metrics"]["pre_calibration_test_metrics"] for record in records]
    return {
        "n_runs": len(records),
        "mean_validation_auroc": round(float(np.mean([item["auroc"] for item in validation_metrics])), 6),
        "mean_validation_brier": round(float(np.mean([item["brier"] for item in validation_metrics])), 6),
        "mean_validation_ece": round(float(np.mean([item["ece"] for item in validation_metrics])), 6),
        "mean_test_auroc": round(float(np.mean([item["auroc"] for item in test_metrics])), 6),
        "mean_test_brier": round(float(np.mean([item["brier"] for item in test_metrics])), 6),
        "mean_test_ece": round(float(np.mean([item["ece"] for item in test_metrics])), 6),
        "mean_pre_validation_auroc": round(float(np.mean([item["auroc"] for item in pre_validation_metrics])), 6),
        "mean_pre_validation_brier": round(float(np.mean([item["brier"] for item in pre_validation_metrics])), 6),
        "mean_pre_validation_ece": round(float(np.mean([item["ece"] for item in pre_validation_metrics])), 6),
        "mean_pre_test_auroc": round(float(np.mean([item["auroc"] for item in pre_test_metrics])), 6),
        "mean_pre_test_brier": round(float(np.mean([item["brier"] for item in pre_test_metrics])), 6),
        "mean_pre_test_ece": round(float(np.mean([item["ece"] for item in pre_test_metrics])), 6),
        "mean_test_brier_delta": round(
            float(np.mean([post["brier"] - pre["brier"] for pre, post in zip(pre_test_metrics, test_metrics)])),
            6,
        ),
        "mean_test_ece_delta": round(
            float(np.mean([post["ece"] - pre["ece"] for pre, post in zip(pre_test_metrics, test_metrics)])),
            6,
        ),
        "mean_test_auroc_delta": round(
            float(np.mean([post["auroc"] - pre["auroc"] for pre, post in zip(pre_test_metrics, test_metrics)])),
            6,
        ),
    }


def calibration_summary_markdown(summary_payload: dict[str, Any]) -> str:
    lines = [
        "# Method Calibration Summary",
        "",
        f"Base model: `{summary_payload['base_model_name']}`. "
        f"Calibration method: `{summary_payload['calibration_method']}`.",
        "",
        "| variant | mean_pre_test_auroc | mean_test_auroc | mean_pre_test_brier | mean_test_brier | mean_pre_test_ece | mean_test_ece |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant_name, metrics in summary_payload["variants"].items():
        lines.append(
            f"| {variant_name} | {metrics['mean_pre_test_auroc']:.6f} | {metrics['mean_test_auroc']:.6f} | "
            f"{metrics['mean_pre_test_brier']:.6f} | {metrics['mean_test_brier']:.6f} | "
            f"{metrics['mean_pre_test_ece']:.6f} | {metrics['mean_test_ece']:.6f} |"
        )
    if summary_payload["missing_variants"]:
        lines.extend(
            [
                "",
                "Missing variants: " + ", ".join(summary_payload["missing_variants"]),
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "calibration_summary_markdown",
    "clear_existing_robustness_runs",
    "collect_variant_records",
    "prediction_frame",
    "read_metrics",
    "reset_directory",
    "robustness_artifacts",
    "split_map_from_protocol",
    "summarize_variant_calibration",
    "write_json",
    "write_run_artifacts",
    "write_text",
]
