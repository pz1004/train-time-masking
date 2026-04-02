from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from .artifacts import read_json, write_json, write_text
from .evaluation.calibration import fit_probability_calibrator
from .evaluation.metrics import binary_classification_metrics
from .evaluation.robustness import apply_missingness_overlay
from .reporting import build_prediction_frame, build_severity_series
from .study import StudySpec


BASELINE_ARTIFACT_FILES = (
    "metrics.json",
    "predictions.csv.gz",
    "split_metadata.json",
    "dataset_metadata.json",
    "manifest_reference.json",
)

ROBUSTNESS_ARTIFACT_FILES = BASELINE_ARTIFACT_FILES + ("slice_metadata.json",)


def _calibrated_baseline_artifacts(
    spec: StudySpec,
    stage_name: str,
    *,
    base_baseline_name: str,
    calibration_method: str,
    split_metadata: dict[str, Any],
    y_validation: pd.Series,
    y_test: pd.Series,
    base_run: Any,
    fit_split: str,
) -> tuple[dict[str, Any], pd.DataFrame, Callable[[pd.DataFrame], Any]]:
    calibrator = fit_probability_calibrator(
        calibration_method,
        y_validation.to_numpy(),
        base_run.validation_probabilities,
        seed=int(split_metadata["seed"]),
    )
    predict_start = perf_counter()
    calibrated_validation = calibrator.predict(base_run.validation_probabilities)
    calibrated_test = calibrator.predict(base_run.test_probabilities)
    calibration_predict_seconds = round(perf_counter() - predict_start, 6)
    model_name = _calibrated_model_name(base_baseline_name, calibration_method)

    predictions = pd.concat(
        [
            build_prediction_frame(
                model_name,
                int(split_metadata["seed"]),
                "validation",
                y_validation,
                calibrated_validation,
                extra_columns={
                    "base_baseline_name": base_baseline_name,
                    "calibration_method": calibration_method,
                },
            ),
            build_prediction_frame(
                model_name,
                int(split_metadata["seed"]),
                "test",
                y_test,
                calibrated_test,
                extra_columns={
                    "base_baseline_name": base_baseline_name,
                    "calibration_method": calibration_method,
                },
            ),
        ],
        ignore_index=True,
    )

    metrics_payload = {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "stage": stage_name,
        "result_kind": "calibrated_variant",
        "model_name": model_name,
        "baseline_name": model_name,
        "baseline_family": "calibrated_variant",
        "base_baseline_name": base_baseline_name,
        "calibration_method": calibration_method,
        "seed": int(split_metadata["seed"]),
        "row_counts": {
            "train": int(split_metadata["train_size"]),
            "validation": int(split_metadata["validation_size"]),
            "test": int(split_metadata["test_size"]),
        },
        "pre_calibration_validation_metrics": base_run.validation_metrics,
        "pre_calibration_test_metrics": base_run.test_metrics,
        "validation_metrics": binary_classification_metrics(y_validation.to_numpy(), calibrated_validation),
        "test_metrics": binary_classification_metrics(y_test.to_numpy(), calibrated_test),
        "fit_seconds": round(base_run.fit_seconds + calibrator.fit_seconds, 6),
        "predict_seconds": round(base_run.predict_seconds + calibration_predict_seconds, 6),
        "model_metadata": {
            "implementation": "posthoc_calibration",
            "base_baseline_name": base_baseline_name,
            "calibration_method": calibration_method,
            "fit_split": fit_split,
            "base_model_metadata": base_run.model_metadata,
            "calibrator_metadata": calibrator.metadata,
        },
        "software_versions": {**base_run.software_versions, **calibrator.software_versions},
    }

    def calibrated_predictor(features: pd.DataFrame) -> Any:
        return calibrator.predict(base_run.predict_probabilities(features))

    return metrics_payload, predictions, calibrated_predictor


def _robustness_artifacts(
    spec: StudySpec,
    stage_name: str,
    *,
    model_name: str,
    split_metadata: dict[str, Any],
    dataset_bundle: Any,
    predict_probabilities: Callable[[pd.DataFrame], Any],
    slice_config: dict[str, Any],
    robustness_config: dict[str, Any],
    source_result_kind: str,
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
    robustness_probabilities = predict_probabilities(overlay_features)
    predict_seconds = round(perf_counter() - predict_start, 6)
    predictions = build_prediction_frame(
        model_name,
        int(split_metadata["seed"]),
        "test",
        y_test,
        robustness_probabilities,
        extra_columns={"evaluation_slice": slice_metadata["slice_name"]},
    )
    metrics_payload = {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "stage": stage_name,
        "result_kind": "robustness_overlay",
        "model_name": model_name,
        "baseline_name": model_name,
        "nominal_reference_name": model_name,
        "nominal_reference_kind": source_result_kind,
        "evaluation_slice": slice_metadata["slice_name"],
        "seed": int(split_metadata["seed"]),
        "row_counts": {"test": int(split_metadata["test_size"])},
        "test_metrics": binary_classification_metrics(y_test.to_numpy(), robustness_probabilities),
        "fit_seconds": 0.0,
        "predict_seconds": predict_seconds,
        "model_metadata": model_metadata,
        "software_versions": software_versions,
    }
    return metrics_payload, predictions, slice_metadata


def _write_run_artifacts(
    run_directory: Path,
    metrics_payload: dict[str, Any],
    predictions: pd.DataFrame,
    split_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
    manifest_reference: dict[str, Any],
    *,
    slice_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.json"
    predictions_path = run_directory / "predictions.csv.gz"
    split_metadata_path = run_directory / "split_metadata.json"
    dataset_metadata_path = run_directory / "dataset_metadata.json"
    manifest_reference_path = run_directory / "manifest_reference.json"

    write_json(metrics_path, metrics_payload)
    sort_columns = [column_name for column_name in ("split", "evaluation_slice", "row_id") if column_name in predictions.columns]
    predictions.sort_values(sort_columns).to_csv(predictions_path, index=False, compression="gzip")
    write_json(split_metadata_path, split_metadata)
    write_json(dataset_metadata_path, dataset_metadata)
    write_json(manifest_reference_path, manifest_reference)

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


def _collect_nominal_baseline_run_records(spec: StudySpec) -> list[dict[str, Any]]:
    baseline_root = spec.raw_dir / "baselines"
    records: list[dict[str, Any]] = []

    for baseline_config in spec.configs["baselines"]["baseline"]:
        baseline_name = str(baseline_config["name"])
        for seed in spec.seed_list:
            run_directory = baseline_root / f"{baseline_name}__seed_{seed}"
            records.append(_read_required_metrics(run_directory, BASELINE_ARTIFACT_FILES, expected_kind="nominal_baseline"))

    return records


def _collect_calibrated_run_records(
    spec: StudySpec,
    selected_calibration_bases: list[str],
    calibration_config: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_root = spec.raw_dir / "baselines"
    records: list[dict[str, Any]] = []

    for base_baseline_name in selected_calibration_bases:
        for calibration_method in calibration_config["methods"]:
            model_name = _calibrated_model_name(base_baseline_name, str(calibration_method))
            for seed in spec.seed_list:
                run_directory = baseline_root / f"{model_name}__seed_{seed}"
                records.append(_read_required_metrics(run_directory, BASELINE_ARTIFACT_FILES, expected_kind="calibrated_variant"))

    return records


def _collect_robustness_run_records(spec: StudySpec, model_names: list[str]) -> list[dict[str, Any]]:
    robustness_root = spec.raw_dir / "robustness"
    records: list[dict[str, Any]] = []

    for model_name in model_names:
        for slice_config in spec.configs["robustness"]["slice"]:
            slice_name = str(slice_config["name"])
            for seed in spec.seed_list:
                run_directory = robustness_root / f"{model_name}__{slice_name}__seed_{seed}"
                record = _read_required_metrics(run_directory, ROBUSTNESS_ARTIFACT_FILES, expected_kind="robustness_overlay")
                record["slice_metadata"] = read_json(run_directory / "slice_metadata.json")
                records.append(record)

    return records


def _collect_all_robustness_run_records(spec: StudySpec) -> list[dict[str, Any]]:
    robustness_root = spec.raw_dir / "robustness"
    return _collect_optional_run_records(
        robustness_root,
        ROBUSTNESS_ARTIFACT_FILES,
        expected_kind="robustness_overlay",
        include_slice_metadata=True,
    )


def _collect_optional_run_records(
    root: Path,
    required_files: tuple[str, ...],
    *,
    expected_kind: str,
    include_slice_metadata: bool = False,
) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for run_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        record = _read_required_metrics(run_directory, required_files, expected_kind=expected_kind)
        if include_slice_metadata and (run_directory / "slice_metadata.json").exists():
            record["slice_metadata"] = read_json(run_directory / "slice_metadata.json")
        records.append(record)
    return records


def _read_required_metrics(run_directory: Path, required_files: tuple[str, ...], *, expected_kind: str) -> dict[str, Any]:
    if not run_directory.exists():
        raise RuntimeError(f"Missing run directory: {run_directory}")
    for filename in required_files:
        required_path = run_directory / filename
        if not required_path.exists():
            raise RuntimeError(f"Missing required artifact: {required_path}")
    metrics_payload = read_json(run_directory / "metrics.json")
    if str(metrics_payload["result_kind"]) != expected_kind:
        raise RuntimeError(f"Unexpected artifact kind in {run_directory}: {metrics_payload['result_kind']} != {expected_kind}")
    return metrics_payload


def _artifacts_complete(run_directory: Path, required_files: tuple[str, ...], *, expected_kind: str) -> bool:
    try:
        _read_required_metrics(run_directory, required_files, expected_kind=expected_kind)
    except RuntimeError:
        return False
    return True


def _summarize_records_by_key(
    records: list[dict[str, Any]],
    key_name: str,
    metric_section: str,
) -> dict[str, dict[str, float]]:
    grouped_records: dict[str, list[dict[str, float]]] = {}
    for record in records:
        grouped_records.setdefault(str(record[key_name]), []).append(record[metric_section])

    metric_summary: dict[str, dict[str, float]] = {}
    for item_name, metrics_list in grouped_records.items():
        metric_summary[item_name] = _summarize_metric_list(metrics_list)
    return metric_summary


def _summarize_calibration_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped_records.setdefault(str(record["model_name"]), []).append(record)

    calibration_summary: dict[str, dict[str, Any]] = {}
    for model_name, model_records in grouped_records.items():
        summary_payload: dict[str, Any] = {
            "n_runs": float(len(model_records)),
            "base_baseline_name": str(model_records[0]["base_baseline_name"]),
            "calibration_method": str(model_records[0]["calibration_method"]),
        }
        pre_metrics = [record["pre_calibration_test_metrics"] for record in model_records]
        post_metrics = [record["test_metrics"] for record in model_records]
        for metric_name in pre_metrics[0].keys():
            pre_values = [float(metrics[metric_name]) for metrics in pre_metrics]
            post_values = [float(metrics[metric_name]) for metrics in post_metrics]
            deltas = [round(post - pre, 6) for pre, post in zip(pre_values, post_values)]
            summary_payload[f"mean_pre_calibration_{metric_name}"] = round(mean(pre_values), 6)
            summary_payload[f"std_pre_calibration_{metric_name}"] = round(pstdev(pre_values), 6)
            summary_payload[f"mean_post_calibration_{metric_name}"] = round(mean(post_values), 6)
            summary_payload[f"std_post_calibration_{metric_name}"] = round(pstdev(post_values), 6)
            summary_payload[f"mean_{metric_name}_delta"] = round(mean(deltas), 6)
            summary_payload[f"std_{metric_name}_delta"] = round(pstdev(deltas), 6)
        calibration_summary[model_name] = summary_payload
    return calibration_summary


def _summarize_robustness_records(
    robustness_records: list[dict[str, Any]],
    nominal_records: list[dict[str, Any]],
    calibrated_records: list[dict[str, Any]],
    method_records: list[dict[str, Any]] | None = None,
    ablation_records: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    reference_lookup = {
        (str(record["model_name"]), int(record["seed"])): record["test_metrics"]
        for record in [*nominal_records, *calibrated_records, *(method_records or []), *(ablation_records or [])]
    }
    grouped_records: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for record in robustness_records:
        model_name = str(record["model_name"])
        slice_name = str(record["evaluation_slice"])
        reference_metrics = reference_lookup[(model_name, int(record["seed"]))]
        delta_metrics = {
            f"{metric_name}_delta": round(float(record["test_metrics"][metric_name]) - float(reference_metrics[metric_name]), 6)
            for metric_name in record["test_metrics"].keys()
        }
        grouped_records.setdefault(model_name, {}).setdefault(slice_name, []).append(
            {
                "test_metrics": record["test_metrics"],
                "delta_metrics": delta_metrics,
                "slice_metadata": record["slice_metadata"],
            }
        )

    robustness_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, slice_map in grouped_records.items():
        robustness_summary[model_name] = {}
        for slice_name, entries in slice_map.items():
            summary_payload: dict[str, Any] = {
                "n_runs": float(len(entries)),
                "kind": str(entries[0]["slice_metadata"]["kind"]),
                "severity": str(entries[0]["slice_metadata"]["severity"]),
                "additional_mask_rate": float(entries[0]["slice_metadata"]["additional_mask_rate"]),
            }
            summary_payload.update(_summarize_metric_list([entry["test_metrics"] for entry in entries]))
            delta_summary = _summarize_metric_list([entry["delta_metrics"] for entry in entries])
            summary_payload.update(delta_summary)
            robustness_summary[model_name][slice_name] = summary_payload
    return robustness_summary


def _summarize_metric_list(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        raise ValueError("metrics_list must not be empty")
    metric_names = metrics_list[0].keys()
    summary_payload: dict[str, float] = {"n_runs": float(len(metrics_list))}
    for metric_name in metric_names:
        values = [float(metrics[metric_name]) for metrics in metrics_list]
        summary_payload[f"mean_{metric_name}"] = round(mean(values), 6)
        summary_payload[f"std_{metric_name}"] = round(pstdev(values), 6)
    return summary_payload


def _write_main_summary_csv(
    path: Path,
    baseline_summary: dict[str, dict[str, float]],
    method_summary: dict[str, dict[str, float]],
    ablation_summary: dict[str, dict[str, float]],
) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, metrics in baseline_summary.items():
        rows.append({"kind": "baseline", "name": model_name, **metrics})
    for model_name, metrics in method_summary.items():
        rows.append({"kind": "method", "name": model_name, **metrics})
    for model_name, metrics in ablation_summary.items():
        rows.append({"kind": "ablation", "name": model_name, **metrics})

    fieldnames = [
        "kind",
        "name",
        "n_runs",
        "mean_auroc",
        "std_auroc",
        "mean_brier",
        "std_brier",
        "mean_log_loss",
        "std_log_loss",
        "mean_ece",
        "std_ece",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _main_results_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Main Results",
        "",
        "Model results aggregated from raw per-seed artifacts.",
        "",
        "| kind | name | mean_auroc | std_auroc | mean_brier | mean_log_loss | mean_ece |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, metrics in summary["baseline_summary"].items():
        lines.append(
            f"| baseline | {name} | {metrics['mean_auroc']:.6f} | {metrics['std_auroc']:.6f} | "
            f"{metrics['mean_brier']:.6f} | {metrics['mean_log_loss']:.6f} | {metrics['mean_ece']:.6f} |"
        )
    for name, metrics in summary["method_summary"].items():
        lines.append(
            f"| method | {name} | {metrics['mean_auroc']:.6f} | {metrics['std_auroc']:.6f} | "
            f"{metrics['mean_brier']:.6f} | {metrics['mean_log_loss']:.6f} | {metrics['mean_ece']:.6f} |"
        )
    for name, metrics in summary["ablation_summary"].items():
        lines.append(
            f"| ablation | {name} | {metrics['mean_auroc']:.6f} | {metrics['std_auroc']:.6f} | "
            f"{metrics['mean_brier']:.6f} | {metrics['mean_log_loss']:.6f} | {metrics['mean_ece']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _robustness_markdown(summary: dict[str, dict[str, dict[str, Any]]]) -> str:
    if not summary:
        return "\n".join(
            [
                "# Robustness Results",
                "",
                "No robustness overlays were aggregated.",
            ]
        ) + "\n"

    lines = [
        "# Robustness Results",
        "",
        "| model | slice | severity | mean_auroc | mean_auroc_delta | mean_ece | mean_ece_delta |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, slice_map in summary.items():
        for slice_name, metrics in slice_map.items():
            lines.append(
                f"| {model_name} | {slice_name} | {metrics['severity']} | {metrics['mean_auroc']:.6f} | "
                f"{metrics['mean_auroc_delta']:.6f} | {metrics['mean_ece']:.6f} | {metrics['mean_ece_delta']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def _calibration_markdown(summary: dict[str, Any]) -> str:
    calibration_summary = summary["calibration_summary"]
    if not calibration_summary:
        lines = [
            "# Calibration Results",
            "",
            "No calibrated variants were aggregated.",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "# Calibration Results",
        "",
        "| model | base_baseline | mean_pre_calibration_auroc | mean_post_calibration_auroc | mean_pre_calibration_ece | mean_post_calibration_ece |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, metrics in calibration_summary.items():
        lines.append(
            f"| {model_name} | {metrics['base_baseline_name']} | "
            f"{metrics['mean_pre_calibration_auroc']:.6f} | {metrics['mean_post_calibration_auroc']:.6f} | "
            f"{metrics['mean_pre_calibration_ece']:.6f} | {metrics['mean_post_calibration_ece']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _performance_figure_series(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return _severity_series(summary, metric_key="mean_auroc")


def _calibration_figure_series(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return _severity_series(summary, metric_key="mean_ece")


def _delta_auroc_figure_series(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return _severity_series(summary, metric_key="mean_auroc_delta")


def _severity_series(summary: dict[str, Any], *, metric_key: str) -> list[dict[str, Any]]:
    return build_severity_series(summary, metric_key=metric_key)


def _bar_chart_svg(title: str, series: list[tuple[str, float]], reverse: bool = False) -> str:
    width = 760
    bar_height = 28
    left_margin = 240
    top_margin = 40
    chart_width = 460
    chart_height = max(1, len(series)) * (bar_height + 10)
    total_height = top_margin + chart_height + 30
    max_value = max(value for _, value in series) if series else 1.0
    max_value = max(max_value, 1e-12)

    rows = []
    for index, (label, value) in enumerate(series):
        y = top_margin + index * (bar_height + 10)
        normalized = (1.0 - (value / max_value)) if reverse else (value / max_value)
        bar_width = max(12, int(chart_width * normalized))
        rows.append(f'<text x="20" y="{y + 19}" font-size="14">{label}</text>')
        rows.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#4a7c59" rx="4" ry="4" />'
        )
        rows.append(f'<text x="{left_margin + bar_width + 10}" y="{y + 19}" font-size="13">{value:.4f}</text>')

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
            '<rect width="100%" height="100%" fill="#f8f5ef" />',
            f'<text x="20" y="24" font-size="18" font-weight="bold">{title}</text>',
            *rows,
            "</svg>",
        ]
    ) + "\n"


def _message_svg(title: str, message: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120" viewBox="0 0 720 120">',
            '<rect width="100%" height="100%" fill="#f8f5ef" />',
            f'<text x="20" y="30" font-size="18" font-weight="bold">{title}</text>',
            f'<text x="20" y="68" font-size="15">{message}</text>',
            "</svg>",
        ]
    ) + "\n"


def _multi_line_chart_svg(title: str, series: list[dict[str, Any]]) -> str:
    if not series:
        return _message_svg(title, "No data available.")

    width = 840
    height = 440
    left = 80
    right = 30
    top = 50
    bottom = 60
    colors = ["#2f6690", "#3a7d44", "#9d4edd", "#c1121f", "#fb8500", "#6d597a", "#1b998b", "#4f772d"]

    x_values = [point["x"] for item in series for point in item["points"]]
    y_values = [point["y"] for item in series for point in item["points"] if point["y"] == point["y"]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(value: float) -> float:
        return left + ((value - x_min) / max(x_max - x_min, 1e-9)) * plot_width

    def y_pos(value: float) -> float:
        return top + (1.0 - ((value - y_min) / max(y_max - y_min, 1e-9))) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{left}" y="28" font-size="20" font-weight="bold">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#444" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#444" stroke-width="1.5"/>',
    ]

    for tick in [0, 10, 20, 30]:
        x_tick = x_pos(float(tick))
        elements.append(f'<line x1="{x_tick}" y1="{top + plot_height}" x2="{x_tick}" y2="{top + plot_height + 6}" stroke="#444" stroke-width="1"/>')
        elements.append(f'<text x="{x_tick - 10}" y="{top + plot_height + 24}" font-size="12">{tick}%</text>')

    for tick_index in range(5):
        y_value = y_min + ((y_max - y_min) * tick_index / 4.0)
        y_tick = y_pos(y_value)
        elements.append(f'<line x1="{left - 6}" y1="{y_tick}" x2="{left}" y2="{y_tick}" stroke="#444" stroke-width="1"/>')
        elements.append(f'<text x="12" y="{y_tick + 4}" font-size="12">{y_value:.3f}</text>')

    legend_y = top
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        point_commands = [f"{x_pos(point['x']):.2f},{y_pos(point['y']):.2f}" for point in item["points"]]
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(point_commands)}"/>')
        for point in item["points"]:
            elements.append(f'<circle cx="{x_pos(point["x"]):.2f}" cy="{y_pos(point["y"]):.2f}" r="3.2" fill="{color}"/>')
        legend_x = left + plot_width + 10
        elements.append(f'<line x1="{legend_x}" y1="{legend_y + (index * 18)}" x2="{legend_x + 16}" y2="{legend_y + (index * 18)}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 22}" y="{legend_y + 4 + (index * 18)}" font-size="12">{item["label"]}</text>')

    elements.append(f'<text x="{left + plot_width / 2 - 70}" y="{height - 16}" font-size="13">Overlay severity</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _mask_sweep_heatmap_svg(payload: dict[str, Any]) -> str:
    rows = payload.get("results", [])
    if not rows:
        return _message_svg("Mask Sweep Heatmap", "No mask sweep data available.")

    severities = ["missingness_10", "missingness_20", "missingness_30"]
    width = 560
    height = 260
    cell_w = 110
    cell_h = 42
    start_x = 120
    start_y = 60

    values = [
        float(row["robustness"].get(severity, {}).get("mean_auroc", float("nan")))
        for row in rows
        for severity in severities
        if row["robustness"].get(severity, {}).get("mean_auroc") is not None
    ]
    min_value = min(values)
    max_value = max(values)
    scale = max(max_value - min_value, 1e-9)

    def color(value: float) -> str:
        ratio = (value - min_value) / scale
        red = int(245 - (90 * ratio))
        green = int(235 - (30 * ratio))
        blue = int(220 - (120 * ratio))
        return f"rgb({red},{green},{blue})"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        '<text x="20" y="28" font-size="20" font-weight="bold">Mask Sweep Heatmap</text>',
    ]
    for col_index, severity in enumerate(severities):
        elements.append(f'<text x="{start_x + col_index * cell_w + 18}" y="{start_y - 16}" font-size="12">{severity}</text>')
    for row_index, row in enumerate(rows):
        y = start_y + row_index * cell_h
        elements.append(f'<text x="20" y="{y + 24}" font-size="12">{row["mask_rate"]:.2f}</text>')
        for col_index, severity in enumerate(severities):
            value = float(row["robustness"].get(severity, {}).get("mean_auroc", float("nan")))
            x = start_x + col_index * cell_w
            elements.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" fill="{color(value)}" stroke="#d0d0d0"/>')
            elements.append(f'<text x="{x + 18}" y="{y + 24}" font-size="12">{value:.4f}</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _select_calibration_bases(records: list[dict[str, Any]], calibration_config: dict[str, Any]) -> list[str]:
    selection_metric_name = str(calibration_config["selection_metric"])
    tie_break_name = str(calibration_config["tie_break"])
    metric_section, selection_metric = _selection_metric_parts(selection_metric_name)
    tie_section, tie_metric = _selection_metric_parts(tie_break_name)

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped_records.setdefault(str(record["model_name"]), []).append(record)

    ranking_payload: list[tuple[str, float, float]] = []
    for model_name, model_records in grouped_records.items():
        selection_values = [float(record[metric_section][selection_metric]) for record in model_records]
        tie_values = [float(record[tie_section][tie_metric]) for record in model_records]
        ranking_payload.append(
            (
                model_name,
                round(mean(selection_values), 6),
                round(mean(tie_values), 6),
            )
        )

    ranking_payload.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [model_name for model_name, _, _ in ranking_payload[: int(calibration_config["top_k"])]]


def _selection_metric_parts(metric_name: str) -> tuple[str, str]:
    if metric_name.startswith("validation_"):
        return "validation_metrics", metric_name.removeprefix("validation_")
    if metric_name.startswith("test_"):
        return "test_metrics", metric_name.removeprefix("test_")
    raise ValueError(f"Unsupported selection metric name: {metric_name}")


def _calibrated_model_name(base_baseline_name: str, calibration_method: str) -> str:
    return f"{base_baseline_name}_calibrated_{calibration_method}"


def _all_robustness_model_names(
    spec: StudySpec,
    selected_calibration_bases: list[str],
    calibration_config: dict[str, Any],
) -> list[str]:
    model_names = [str(config["name"]) for config in spec.configs["baselines"]["baseline"]]
    for base_baseline_name in selected_calibration_bases:
        for calibration_method in calibration_config["methods"]:
            model_names.append(_calibrated_model_name(base_baseline_name, str(calibration_method)))
    return model_names


def _configured_robustness_model_names(
    spec: StudySpec,
    selected_calibration_bases: list[str],
    calibration_config: dict[str, Any],
    method_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
) -> list[str]:
    model_names = _all_robustness_model_names(spec, selected_calibration_bases, calibration_config)
    model_names.extend(sorted({str(record["model_name"]) for record in method_records}))
    model_names.extend(sorted({str(record["model_name"]) for record in ablation_records}))
    return model_names
