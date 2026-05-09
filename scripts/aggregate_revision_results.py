from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import platform
from glob import glob
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.evaluation.metrics import select_threshold_by_validation_f1, thresholded_classification_metrics
from lab.study import load_study_spec


DATASET_LABELS = {
    "adult": "Adult",
    "credit-g": "German Credit",
    "bank-marketing": "Bank Marketing",
    "credit": "Give Me Some Credit",
    "give-me-some-credit": "Give Me Some Credit",
    "covertype": "Covertype",
}

MODEL_LABELS = {
    "mask_augmented_imputation_training": "MAIT",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "LR",
    "ft_transformer": "FT-Transformer",
    "tabpfn": "TabPFN",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate revision-stage artifacts into manuscript-ready tables.")
    parser.add_argument("--study-glob", required=True, help="Glob for study configs.")
    args = parser.parse_args(argv)
    config_paths = sorted(glob(args.study_glob))
    if not config_paths:
        raise SystemExit(f"No study configs matched: {args.study_glob}")

    specs = [_with_latest_revision_results(load_study_spec(path)) for path in config_paths]
    output_dir = ROOT / "paper" / "revision_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    modern_baseline_rows = _modern_baseline_rows(specs)
    payload = {
        "dataset_regime": _dataset_regime_rows(specs),
        "modern_baselines": modern_baseline_rows,
        "all_model_significance": _all_model_significance_rows(specs),
        "all_model_ranks": _primary_model_rank_rows(modern_baseline_rows),
        "runtime": _runtime_rows(specs),
        "confusion": _confusion_rows(specs),
        "leakage": _leakage_rows(specs),
        "feature_stability": _feature_stability_rows(specs),
        "threshold_sensitivity": _threshold_sensitivity_rows(),
        "hardware": _hardware_payload(),
    }
    (output_dir / "revision_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dataset_regime_table(output_dir / "dataset_regime_table.tex", payload["dataset_regime"])
    _write_modern_baseline_table(output_dir / "modern_baseline_table.tex", payload["modern_baselines"])
    _write_all_model_significance_table(output_dir / "all_model_significance_table.tex", payload["all_model_significance"])
    _write_all_model_rank_table(output_dir / "all_model_rank_table.tex", payload["all_model_ranks"])
    _write_runtime_table(output_dir / "runtime_table.tex", payload["runtime"], payload["hardware"])
    _write_confusion_table(output_dir / "confusion_table.tex", payload["confusion"])
    _write_leakage_table(output_dir / "leakage_table.tex", payload["leakage"])
    _write_feature_stability_table(output_dir / "feature_stability_table.tex", payload["feature_stability"])
    _write_threshold_sensitivity_table(output_dir / "threshold_sensitivity_table.tex", payload["threshold_sensitivity"])
    return 0


def _dataset_regime_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        metadata = _first_dataset_metadata(spec)
        dataset_name = str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id))
        class_balance = metadata.get("class_balance", {})
        positive_rate = class_balance.get("1", class_balance.get(1, ""))
        rows.append(
            {
                "dataset": DATASET_LABELS.get(dataset_name, dataset_name),
                "rows": metadata.get("deduplicated_row_count", ""),
                "features": metadata.get("feature_count", ""),
                "categorical": len(metadata.get("categorical_columns", [])),
                "numerical": len(metadata.get("numerical_columns", [])),
                "positive_rate": positive_rate,
                "native_missing": metadata.get("missing_value_count", ""),
                "preprocessing": "train-only stats/vocabs",
                "overlay_columns": ", ".join(str(column) for column in spec.configs["robustness"]["robustness"]["columns"]),
            }
        )
    return rows


def _modern_baseline_rows(specs) -> list[dict[str, Any]]:
    rows = []
    model_order = [
        "mask_augmented_imputation_training",
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
        "ft_transformer",
        "tabpfn",
    ]
    for spec in specs:
        summary = _read_optional_json(spec.aggregated_dir / "performance_summary.json")
        structured = _read_optional_json(spec.results_dir / "structured_missingness" / "structured_missingness_results.json")
        if not summary:
            continue
        dataset_label = _dataset_label(spec)
        nominal_lookup = {}
        nominal_lookup.update(summary.get("baseline_summary", {}))
        nominal_lookup.update(summary.get("method_summary", {}))
        robustness_summary = summary.get("robustness_summary", {})
        for model_name in model_order:
            if model_name not in nominal_lookup:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "nominal_auroc": _maybe_float(nominal_lookup.get(model_name, {}).get("mean_auroc")),
                    "mcar30_auroc": _maybe_float(robustness_summary.get(model_name, {}).get("missingness_30", {}).get("mean_auroc")),
                    "mar_primary_auroc": _structured_metric(structured, "mar_primary", model_name),
                    "mar_stress_auroc": _structured_metric(structured, "mar_stress", model_name),
                    "mnar_primary_auroc": _structured_metric(structured, "mnar_primary", model_name),
                    "mnar_stress_auroc": _structured_metric(structured, "mnar_stress", model_name),
                }
            )
    return rows


def _all_model_significance_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        payload = _read_optional_json(spec.audits_dir / "all_model_significance_results.json")
        if not payload:
            continue
        dataset_label = _dataset_label(spec)
        for test in payload.get("tests", []):
            if str(test.get("slice")) not in {"nominal", "missingness_30"}:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "slice": str(test.get("slice", "")),
                    "comparator": MODEL_LABELS.get(str(test.get("comparator", "")), str(test.get("comparator", ""))),
                    "mean_diff_auroc": _maybe_float(test.get("mean_diff_auroc")),
                    "holm_p": _maybe_float(test.get("holm_corrected_p_value")),
                    "ci": _ci_cell(test),
                    "n_runs": test.get("n_runs", ""),
                }
            )
        for skipped in payload.get("skipped_comparators", []):
            rows.append(
                {
                    "dataset": dataset_label,
                    "slice": "skipped",
                    "comparator": MODEL_LABELS.get(str(skipped.get("comparator", "")), str(skipped.get("comparator", ""))),
                    "mean_diff_auroc": "",
                    "holm_p": "",
                    "ci": str(skipped.get("reason", "")),
                    "n_runs": "",
                }
            )
    return rows


def _primary_model_rank_rows(modern_baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_specs = [
        ("Nominal", "nominal_auroc"),
        ("MCAR-30", "mcar30_auroc"),
        ("MAR-primary", "mar_primary_auroc"),
        ("MNAR-primary", "mnar_primary_auroc"),
    ]
    rows = []
    datasets = sorted({str(row["dataset"]) for row in modern_baseline_rows})
    models = [MODEL_LABELS[name] for name in [
        "mask_augmented_imputation_training",
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
        "ft_transformer",
        "tabpfn",
    ]]
    for slice_label, metric_key in rank_specs:
        per_model_ranks: dict[str, list[float]] = {model: [] for model in models}
        for dataset in datasets:
            values = [
                (str(row["model"]), row.get(metric_key))
                for row in modern_baseline_rows
                if str(row["dataset"]) == dataset and row.get(metric_key) is not None
            ]
            values.sort(key=lambda item: float(item[1]), reverse=True)
            for rank, (model, _) in enumerate(values, start=1):
                if model in per_model_ranks:
                    per_model_ranks[model].append(float(rank))
        for model, ranks in per_model_ranks.items():
            if not ranks:
                continue
            rows.append(
                {
                    "slice": slice_label,
                    "model": model,
                    "mean_rank": round(float(np.mean(ranks)), 4),
                    "n_datasets": len(ranks),
                }
            )
    rows.sort(key=lambda row: (str(row["slice"]), float(row["mean_rank"]), str(row["model"])))
    return rows


def _runtime_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        dataset_label = _dataset_label(spec)
        for model_name, run_root in _nominal_run_roots(spec):
            fit_values = []
            predict_values = []
            for metrics_path in sorted(run_root.glob(f"{model_name}__seed_*/metrics.json")):
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                fit_values.append(float(payload.get("fit_seconds", 0.0)))
                predict_values.append(float(payload.get("predict_seconds", 0.0)))
            if not fit_values:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "n_runs": len(fit_values),
                    "mean_fit_seconds": round(float(np.mean(fit_values)), 4),
                    "total_fit_minutes": round(float(np.sum(fit_values) / 60.0), 2),
                    "mean_predict_seconds": round(float(np.mean(predict_values)), 4),
                }
            )
    return rows


def _confusion_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        dataset_label = _dataset_label(spec)
        for model_name, run_root in _nominal_run_roots(spec):
            metric_rows = []
            for prediction_path in sorted(run_root.glob(f"{model_name}__seed_*/predictions.csv.gz")):
                predictions = pd.read_csv(prediction_path)
                validation = predictions[predictions["split"] == "validation"]
                test = predictions[predictions["split"] == "test"]
                if validation.empty or test.empty:
                    continue
                threshold = select_threshold_by_validation_f1(validation["target"].to_numpy(), validation["predicted_probability"].to_numpy())
                metric_rows.append(
                    thresholded_classification_metrics(
                        test["target"].to_numpy(),
                        test["predicted_probability"].to_numpy(),
                        threshold=threshold,
                    )
                )
            if not metric_rows:
                continue
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "mean_threshold": _mean_metric(metric_rows, "threshold"),
                    "mean_f1": _mean_metric(metric_rows, "f1"),
                    "mean_recall": _mean_metric(metric_rows, "recall"),
                    "mean_specificity": _mean_metric(metric_rows, "specificity"),
                    "mean_balanced_accuracy": _mean_metric(metric_rows, "balanced_accuracy"),
                }
            )
    return rows


def _leakage_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        payload = _read_optional_json(spec.results_dir / "leakage_ablation" / "leakage_ablation_results.json")
        if not payload:
            continue
        standard_summary = _read_optional_json(spec.aggregated_dir / "performance_summary.json") or {}
        dataset_label = _dataset_label(spec)
        for model_name, result in payload.get("results", {}).items():
            if result.get("status") != "completed":
                rows.append({"dataset": dataset_label, "model": MODEL_LABELS.get(model_name, model_name), "status": result.get("status", ""), "nominal_delta": "", "overlay_delta": ""})
                continue
            standard_nominal = standard_summary.get("baseline_summary", {}).get(model_name, {}).get("mean_auroc")
            standard_overlay = standard_summary.get("robustness_summary", {}).get(model_name, {}).get("missingness_30", {}).get("mean_auroc")
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "status": "completed",
                    "nominal_delta": _delta(result["nominal"]["mean_auroc"], standard_nominal),
                    "overlay_delta": _delta(result["overlay"]["mean_auroc"], standard_overlay),
                }
            )
    return rows


def _feature_stability_rows(specs) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        payload = _read_optional_json(spec.results_dir / "feature_stability" / "feature_stability_results.json")
        if not payload:
            continue
        dataset_label = _dataset_label(spec)
        for model_name, result in payload.get("results", {}).items():
            rows.append(
                {
                    "dataset": dataset_label,
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "spearman": result.get("mean_spearman_rank_correlation"),
                    "shift": result.get("mean_absolute_importance_shift"),
                }
            )
    return rows


def _threshold_sensitivity_rows() -> list[dict[str, Any]]:
    payload = _read_optional_json(ROOT / "paper" / "submission_summary" / "robustness_advantages.json")
    if not payload:
        return []
    thresholds = [0.001, 0.0025, 0.005, 0.01]
    rows = []
    for threshold in thresholds:
        passing = [row for row in payload if float(row["mean_advantage"]) > threshold]
        rows.append(
            {
                "threshold": threshold,
                "n_passing": len(passing),
                "datasets": ", ".join(sorted({str(row["dataset_label"]) for row in passing})),
            }
        )
    return rows


def _hardware_payload() -> dict[str, Any]:
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        payload["torch"] = "not available"
        payload["cuda_available"] = False
        payload["cuda_device"] = ""
    return payload


def _with_latest_revision_results(spec):
    if os.environ.get("LAB_RESULTS_DIR_PREFIX"):
        return spec
    candidates = sorted(spec.results_dir.parent.glob(f"*_revision_full_{spec.results_dir.name}"))
    for candidate in reversed(candidates):
        required_paths = [
            candidate / "aggregated" / "performance_summary.json",
            candidate / "structured_missingness" / "structured_missingness_results.json",
            candidate / "audits" / "all_model_significance_results.json",
            candidate / "feature_stability" / "feature_stability_results.json",
            candidate / "leakage_ablation" / "leakage_ablation_results.json",
        ]
        if all(path.exists() for path in required_paths):
            return replace(spec, results_dir=candidate)
    return spec


def _write_dataset_regime_table(path: Path, rows: list[dict[str, Any]]) -> None:
    body = [
        r"\begin{table*}[t]",
        r"\caption{Dataset regimes and missingness-audit design. Overlay columns are fixed before evaluation.}",
        r"\label{tab:dataset-regimes}",
        r"\centering\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrclll}",
        r"\toprule",
        r"Dataset & Rows & Feat. & Cat. & Num. & Pos. rate & Native miss. & Preprocessing & Overlay columns \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(
            f"{_tex(row['dataset'])} & {_tex(row['rows'])} & {_tex(row['features'])} & {_tex(row['categorical'])} & "
            f"{_tex(row['numerical'])} & {_format_float(row['positive_rate'])} & {_tex(row['native_missing'])} & "
            f"{_tex(row['preprocessing'])} & {_tex(row['overlay_columns'])} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _write_modern_baseline_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Nominal and missingness-regime AUROC for classical, neural, and foundation-model baselines. Structured overlays use the completed MAR/MNAR revision-full artifacts.",
        label="tab:modern-structured-results",
        columns=[
            ("dataset", "Dataset"),
            ("model", "Model"),
            ("nominal_auroc", "Nom."),
            ("mcar30_auroc", "MCAR-30"),
            ("mar_primary_auroc", "MAR-p"),
            ("mar_stress_auroc", "MAR-s"),
            ("mnar_primary_auroc", "MNAR-p"),
            ("mnar_stress_auroc", "MNAR-s"),
        ],
        rows=rows,
    )


def _write_all_model_significance_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="MAIT-versus-all-comparator paired AUROC tests after Holm correction. Positive deltas favor MAIT.",
        label="tab:all-model-significance",
        columns=[
            ("dataset", "Dataset"),
            ("slice", "Slice"),
            ("comparator", "Comparator"),
            ("mean_diff_auroc", r"$\Delta$AUROC"),
            ("holm_p", "Holm p"),
            ("ci", "95\\% CI / status"),
            ("n_runs", "Runs"),
        ],
        rows=rows,
    )


def _write_all_model_rank_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Cross-dataset mean ranks for the primary executable models only. Lower ranks indicate higher mean AUROC.",
        label="tab:all-model-ranks",
        columns=[("slice", "Slice"), ("model", "Model"), ("mean_rank", "Mean rank"), ("n_datasets", "Datasets")],
        rows=rows,
    )


def _write_runtime_table(path: Path, rows: list[dict[str, Any]], hardware: dict[str, Any]) -> None:
    note = f"Hardware snapshot: {hardware.get('cpu_count')} CPU threads; CUDA available={hardware.get('cuda_available')}."
    _write_simple_table(
        path,
        caption="Training and prediction wall-clock summaries from per-seed artifacts. " + note,
        label="tab:runtime-expanded",
        columns=[("dataset", "Dataset"), ("model", "Model"), ("mean_fit_seconds", "Fit s/seed"), ("total_fit_minutes", "Fit min/20"), ("mean_predict_seconds", "Pred. s/seed")],
        rows=rows,
    )


def _write_confusion_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Thresholded test-set behavior using validation-selected F1 thresholds.",
        label="tab:thresholded-confusion",
        columns=[("dataset", "Dataset"), ("model", "Model"), ("mean_threshold", "Thr."), ("mean_f1", "F1"), ("mean_recall", "Recall"), ("mean_specificity", "Spec."), ("mean_balanced_accuracy", "Bal. acc.")],
        rows=rows,
    )


def _write_leakage_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Intentional leakage ablation: AUROC deltas from fitting preprocessing on train+validation+test rows rather than train rows only.",
        label="tab:leakage-ablation",
        columns=[("dataset", "Dataset"), ("model", "Model"), ("status", "Status"), ("nominal_delta", "Nom. delta"), ("overlay_delta", "MCAR-30 delta")],
        rows=rows,
    )


def _write_feature_stability_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Permutation-importance stability between nominal and MCAR-30 test copies.",
        label="tab:feature-stability",
        columns=[("dataset", "Dataset"), ("model", "Model"), ("spearman", "Spearman"), ("shift", "Mean abs. shift")],
        rows=rows,
    )


def _write_threshold_sensitivity_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_simple_table(
        path,
        caption="Sensitivity of the practical-robustness decision to the AUROC advantage threshold.",
        label="tab:threshold-sensitivity",
        columns=[("threshold", r"$\delta$"), ("n_passing", "Passing comparisons"), ("datasets", "Datasets represented")],
        rows=rows,
    )


def _write_simple_table(path: Path, *, caption: str, label: str, columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> None:
    body = [
        r"\begin{table*}[t]",
        rf"\caption{{{_tex(caption)}}}",
        rf"\label{{{label}}}",
        r"\centering\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
    ]
    if rows:
        for row in rows:
            body.append(" & ".join(_format_cell(row.get(key, "")) for key, _ in columns) + r" \\")
    else:
        body.append(r"\multicolumn{" + str(len(columns)) + r"}{c}{Run \texttt{python scripts/run\_all\_studies.py --include-extras --revision-full} to populate this table.} \\")
    body.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _nominal_run_roots(spec) -> list[tuple[str, Path]]:
    model_roots = [(str(config["name"]), spec.raw_dir / "baselines") for config in spec.configs["baselines"]["baseline"]]
    model_roots.append((str(spec.configs["method"]["method"]["name"]), spec.raw_dir / "methods"))
    return model_roots


def _first_dataset_metadata(spec) -> dict[str, Any]:
    for root in (spec.raw_dir / "methods", spec.raw_dir / "baselines"):
        for path in sorted(root.glob("*__seed_*/dataset_metadata.json")):
            return json.loads(path.read_text(encoding="utf-8"))
    return {
        "deduplicated_row_count": "",
        "feature_count": "",
        "categorical_columns": spec.configs["dataset"]["dataset"].get("categorical_columns", []),
        "numerical_columns": [],
        "class_balance": {},
        "missing_value_count": "",
    }


def _dataset_label(spec) -> str:
    dataset_name = str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id))
    return DATASET_LABELS.get(dataset_name, dataset_name)


def _structured_metric(payload: dict[str, Any] | None, overlay_name: str, model_name: str) -> float | None:
    if not payload:
        return None
    return _maybe_float(payload.get("results", {}).get(overlay_name, {}).get(model_name, {}).get("mean_auroc"))


def _mean_metric(rows: list[dict[str, float]], metric_name: str) -> float:
    return round(float(np.mean([float(row[metric_name]) for row in rows])), 6)


def _delta(value: Any, reference: Any) -> float | str:
    if value is None or reference is None:
        return ""
    return round(float(value) - float(reference), 6)


def _read_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _ci_cell(row: dict[str, Any]) -> str:
    low = row.get("bootstrap_ci_low")
    high = row.get("bootstrap_ci_high")
    if low is None or high is None:
        return ""
    return f"[{float(low):.4f}, {float(high):.4f}]"


def _format_cell(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return _tex(value)


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def _tex(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    raise SystemExit(main())
