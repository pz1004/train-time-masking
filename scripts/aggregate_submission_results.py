from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.study import load_study_spec


DATASET_LABELS = {
    "adult": "Adult",
    "credit-g": "German Credit",
    "bank-marketing": "Bank Marketing",
    "credit": "Give Me Some Credit",
    "give-me-some-credit": "Give Me Some Credit",
    "covertype": "Covertype",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-study submission tables.")
    parser.add_argument("--study-glob", required=True, help="Glob for study config files.")
    args = parser.parse_args(argv)

    configs = sorted(glob(args.study_glob))
    if not configs:
        raise SystemExit(f"No study configs matched: {args.study_glob}")

    rows = []
    mlp_control_rows = []
    for config_path in configs:
        spec = load_study_spec(config_path)
        summary_path = spec.aggregated_dir / "performance_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        method_name = str(spec.configs["method"]["method"]["name"])
        method_summary = summary.get("method_summary", {}).get(method_name, {})
        baseline_summary = summary.get("baseline_summary", {})
        ablation_summary = summary.get("ablation_summary", {})
        robustness_summary = summary.get("robustness_summary", {})
        best_baseline_name = None
        best_baseline_auroc = None
        if baseline_summary:
            best_baseline_name, best_baseline_metrics = max(
                baseline_summary.items(),
                key=lambda item: float(item[1]["mean_auroc"]),
            )
            best_baseline_auroc = float(best_baseline_metrics["mean_auroc"])
        significance_path = spec.audits_dir / "significance_results.json"
        significance = json.loads(significance_path.read_text(encoding="utf-8")) if significance_path.exists() else {}
        rows.append(
            {
                "study_id": spec.study_id,
                "dataset_name": spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id),
                "dataset_label": DATASET_LABELS.get(
                    str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id)),
                    str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id)),
                ),
                "method_name": method_name,
                "method_mean_auroc": float(method_summary.get("mean_auroc", float("nan"))),
                "best_baseline_name": best_baseline_name,
                "best_baseline_mean_auroc": best_baseline_auroc,
                "significance_tests": significance.get("tests", []),
            }
        )
        mlp_row = _build_mlp_control_row(
            spec.study_id,
            DATASET_LABELS.get(
                str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id)),
                str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id)),
            ),
            method_name,
            method_summary,
            ablation_summary,
            robustness_summary,
        )
        if mlp_row is not None:
            mlp_control_rows.append(mlp_row)

    output_dir = ROOT / "paper" / "submission_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "submission_summary.json"
    md_path = output_dir / "submission_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(rows), encoding="utf-8")
    mlp_json_path = output_dir / "mlp_control_summary.json"
    mlp_md_path = output_dir / "mlp_control_summary.md"
    mlp_json_path.write_text(json.dumps(mlp_control_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mlp_md_path.write_text(_mlp_control_markdown(mlp_control_rows), encoding="utf-8")
    return 0


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Submission Summary",
        "",
        "| study_id | dataset | method | method_mean_auroc | best_baseline | best_baseline_mean_auroc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['study_id']} | {row['dataset_label']} | {row['method_name']} | "
            f"{float(row['method_mean_auroc']):.6f} | {row['best_baseline_name']} | "
            f"{float(row['best_baseline_mean_auroc']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _build_mlp_control_row(
    study_id: str,
    dataset_label: str,
    method_name: str,
    method_summary: dict[str, object],
    ablation_summary: dict[str, object],
    robustness_summary: dict[str, object],
) -> dict[str, object] | None:
    mlp_nominal = ablation_summary.get("mlp_only")
    method_robustness = robustness_summary.get(method_name, {}).get("missingness_30")
    mlp_robustness = robustness_summary.get("mlp_only", {}).get("missingness_30")
    if mlp_nominal is None or method_robustness is None or mlp_robustness is None:
        return None
    return {
        "study_id": study_id,
        "dataset_label": dataset_label,
        "mlp_only_nominal_mean_auroc": float(mlp_nominal["mean_auroc"]),
        "mlp_only_nominal_std_auroc": float(mlp_nominal["std_auroc"]),
        "mait_nominal_mean_auroc": float(method_summary["mean_auroc"]),
        "mait_nominal_std_auroc": float(method_summary["std_auroc"]),
        "mlp_only_mcar30_mean_auroc": float(mlp_robustness["mean_auroc"]),
        "mlp_only_mcar30_std_auroc": float(mlp_robustness["std_auroc"]),
        "mait_mcar30_mean_auroc": float(method_robustness["mean_auroc"]),
        "mait_mcar30_std_auroc": float(method_robustness["std_auroc"]),
        "mlp_only_mcar30_mean_delta": float(mlp_robustness["mean_auroc_delta"]),
        "mlp_only_mcar30_std_delta": float(mlp_robustness["std_auroc_delta"]),
        "mait_mcar30_mean_delta": float(method_robustness["mean_auroc_delta"]),
        "mait_mcar30_std_delta": float(method_robustness["std_auroc_delta"]),
    }


def _mlp_control_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# MLP Control Summary",
        "",
        "| dataset | MLP-only nominal | MAIT nominal | MLP-only MCAR-30 | MAIT MCAR-30 | MLP-only ΔAUROC @30 | MAIT ΔAUROC @30 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | "
            f"{_format_pair(row['mlp_only_nominal_mean_auroc'], row['mlp_only_nominal_std_auroc'])} | "
            f"{_format_pair(row['mait_nominal_mean_auroc'], row['mait_nominal_std_auroc'])} | "
            f"{_format_pair(row['mlp_only_mcar30_mean_auroc'], row['mlp_only_mcar30_std_auroc'])} | "
            f"{_format_pair(row['mait_mcar30_mean_auroc'], row['mait_mcar30_std_auroc'])} | "
            f"{_format_pair(row['mlp_only_mcar30_mean_delta'], row['mlp_only_mcar30_std_delta'])} | "
            f"{_format_pair(row['mait_mcar30_mean_delta'], row['mait_mcar30_std_delta'])} |"
        )
    return "\n".join(lines) + "\n"


def _format_pair(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
