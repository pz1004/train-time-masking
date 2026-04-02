from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.study import load_study_spec


MODEL_ORDER = (
    "mask_augmented_imputation_training",
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "logistic_regression",
)

MODEL_LABELS = {
    "mask_augmented_imputation_training": "MAIT",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}

CORE_MODELS = (
    "mask_augmented_imputation_training",
    "lightgbm",
    "xgboost",
)

DATASET_LABELS = {
    "adult": "Adult",
    "credit-g": "German Credit",
    "bank-marketing": "Bank Marketing",
    "credit": "Give Me Some Credit",
    "give-me-some-credit": "Give Me Some Credit",
    "covertype": "Covertype",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-study MAR overlay results into one cross-study table.")
    parser.add_argument(
        "--study-glob",
        default="configs/studies/*missingness_robustness*.toml",
        help="Glob for the study configs to aggregate.",
    )
    args = parser.parse_args(argv)

    config_paths = sorted(glob(args.study_glob))
    if not config_paths:
        raise SystemExit(f"No study configs matched: {args.study_glob}")

    rows = []
    missing = []
    for config_path in config_paths:
        spec = load_study_spec(config_path)
        mar_path = spec.results_dir / "mar_overlay" / "mar_overlay_results.json"
        summary_path = spec.aggregated_dir / "performance_summary.json"
        if not mar_path.exists():
            missing.append(str(mar_path))
            continue
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        payload = json.loads(mar_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset_key = str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id))
        rows.append(
            {
                "study_id": spec.study_id,
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
                "driver_column": str(payload["mar_overlay"]["driver_column"]),
                "target_columns": [str(column_name) for column_name in payload["mar_overlay"]["target_columns"]],
                "low_rate": float(payload["mar_overlay"]["low_rate"]),
                "high_rate": float(payload["mar_overlay"]["high_rate"]),
                "threshold_quantile": float(payload["mar_overlay"]["threshold_quantile"]),
                "summary": _build_cross_regime_summary(summary, payload["results"]),
                "results": payload["results"],
            }
        )

    if missing:
        missing_text = "\n".join(missing)
        raise SystemExit(f"Missing MAR overlay artifacts:\n{missing_text}")

    rows.sort(key=lambda item: item["dataset_label"])
    output_dir = ROOT / "paper" / "submission_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mar_summary.json"
    md_path = output_dir / "mar_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return 0


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# MAR Overlay Summary",
        "",
        "## Cross-Regime AUROC (Core Comparators)",
        "",
        "| dataset | probe | MAIT nominal | MAIT MCAR-30 | MAIT MAR | LightGBM nominal | LightGBM MCAR-30 | LightGBM MAR | XGBoost nominal | XGBoost MCAR-30 | XGBoost MAR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {_probe_label(row)} | "
            f"{_format_cross_regime_triplet(row['summary'], 'mask_augmented_imputation_training')} | "
            f"{_format_cross_regime_triplet(row['summary'], 'lightgbm')} | "
            f"{_format_cross_regime_triplet(row['summary'], 'xgboost')} |"
        )
    lines.extend(
        [
            "",
            "## MAR AUROC (All Reported Models)",
            "",
            "| dataset | MAIT | LightGBM | XGBoost | CatBoost | Random Forest | Logistic Regression |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        metrics = row["results"]
        lines.append(
            f"| {row['dataset_label']} | "
            f"{_format_mean_std(metrics, 'mask_augmented_imputation_training', 'mean_auroc', 'std_auroc')} | "
            f"{_format_mean_std(metrics, 'lightgbm', 'mean_auroc', 'std_auroc')} | "
            f"{_format_mean_std(metrics, 'xgboost', 'mean_auroc', 'std_auroc')} | "
            f"{_format_mean_std(metrics, 'catboost', 'mean_auroc', 'std_auroc')} | "
            f"{_format_mean_std(metrics, 'random_forest', 'mean_auroc', 'std_auroc')} | "
            f"{_format_mean_std(metrics, 'logistic_regression', 'mean_auroc', 'std_auroc')} |"
        )
    lines.extend(
        [
            "",
            "## MAR ECE (All Reported Models)",
            "",
            "| dataset | MAIT | LightGBM | XGBoost | CatBoost | Random Forest | Logistic Regression |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        metrics = row["results"]
        lines.append(
            f"| {row['dataset_label']} | "
            f"{_format_mean_std(metrics, 'mask_augmented_imputation_training', 'mean_ece', 'std_ece')} | "
            f"{_format_mean_std(metrics, 'lightgbm', 'mean_ece', 'std_ece')} | "
            f"{_format_mean_std(metrics, 'xgboost', 'mean_ece', 'std_ece')} | "
            f"{_format_mean_std(metrics, 'catboost', 'mean_ece', 'std_ece')} | "
            f"{_format_mean_std(metrics, 'random_forest', 'mean_ece', 'std_ece')} | "
            f"{_format_mean_std(metrics, 'logistic_regression', 'mean_ece', 'std_ece')} |"
        )
    lines.extend(["", "## Overlay Definitions", ""])
    for row in rows:
        lines.append(
            f"- {row['dataset_label']}: driver `{row['driver_column']}`, targets {', '.join(f'`{name}`' for name in row['target_columns'])}, "
            f"low/high rates {row['low_rate']:.2f}/{row['high_rate']:.2f}, threshold quantile {row['threshold_quantile']:.2f}."
        )
    return "\n".join(lines) + "\n"


def _build_cross_regime_summary(
    performance_summary: dict[str, object],
    mar_results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    method_summary = performance_summary["method_summary"]
    baseline_summary = performance_summary["baseline_summary"]
    robustness_summary = performance_summary["robustness_summary"]
    combined: dict[str, dict[str, float]] = {}
    for model_name in CORE_MODELS:
        nominal_row = method_summary.get(model_name) or baseline_summary.get(model_name)
        if nominal_row is None:
            raise KeyError(f"Missing nominal metrics for model: {model_name}")
        mcar_row = robustness_summary[model_name]["missingness_30"]
        mar_row = mar_results[model_name]
        combined[model_name] = {
            "nominal_mean_auroc": float(nominal_row["mean_auroc"]),
            "nominal_std_auroc": float(nominal_row["std_auroc"]),
            "mcar30_mean_auroc": float(mcar_row["mean_auroc"]),
            "mcar30_std_auroc": float(mcar_row["std_auroc"]),
            "mar_mean_auroc": float(mar_row["mean_auroc"]),
            "mar_std_auroc": float(mar_row["std_auroc"]),
        }
    return combined


def _probe_label(row: dict[str, object]) -> str:
    return (
        f"{row['driver_column']} → "
        f"{', '.join(str(name) for name in row['target_columns'])} "
        f"({row['low_rate']:.2f}/{row['high_rate']:.2f})"
    )


def _format_cross_regime_triplet(summary: dict[str, dict[str, float]], model_name: str) -> str:
    row = summary[model_name]
    return (
        f"{row['nominal_mean_auroc']:.4f} | "
        f"{row['mcar30_mean_auroc']:.4f} | "
        f"{row['mar_mean_auroc']:.4f}"
    )


def _format_mean_std(metrics: dict[str, dict[str, float]], model_name: str, mean_key: str, std_key: str) -> str:
    row = metrics[model_name]
    return f"{row[mean_key]:.4f} ± {row[std_key]:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
