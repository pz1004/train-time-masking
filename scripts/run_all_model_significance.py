from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.study import load_study_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute MAIT-vs-all-model paired significance and rank summaries.")
    parser.add_argument("--study-config", default="", help="Path to one study config.")
    parser.add_argument("--study-glob", default="", help="Glob for multiple study configs; also writes cross-dataset ranks.")
    args = parser.parse_args(argv)

    config_paths = _config_paths(args.study_config, args.study_glob)
    per_study_payloads = []
    for config_path in config_paths:
        spec = load_study_spec(config_path)
        payload = _compute_study_significance(spec)
        per_study_payloads.append(payload)
        output_dir = spec.audits_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "all_model_significance_results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "all_model_significance_results.md").write_text(_study_markdown(payload), encoding="utf-8")

    if args.study_glob:
        rank_payload = _cross_dataset_ranks(config_paths)
        output_dir = ROOT / "paper" / "submission_summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "all_model_rank_summary.json").write_text(
            json.dumps(rank_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "all_model_rank_summary.md").write_text(_rank_markdown(rank_payload), encoding="utf-8")
    return 0


def _config_paths(study_config: str, study_glob: str) -> list[str]:
    if study_config and study_glob:
        raise ValueError("Use either --study-config or --study-glob, not both.")
    if study_config:
        return [study_config]
    if study_glob:
        paths = sorted(glob(study_glob))
        if not paths:
            raise SystemExit(f"No study configs matched: {study_glob}")
        return paths
    raise SystemExit("Either --study-config or --study-glob is required.")


def _compute_study_significance(spec) -> dict[str, Any]:
    method_name = str(spec.configs["method"]["method"]["name"])
    slice_names = ["nominal", *[str(item["name"]) for item in spec.configs["robustness"]["slice"]]]
    method_vectors = {slice_name: _metric_vector(spec, method_name, slice_name) for slice_name in slice_names}
    comparators: list[str] = []
    skipped_comparators: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for baseline_config in spec.configs["baselines"]["baseline"]:
        comparator = str(baseline_config["name"])
        comparator_nominal = _metric_vector(spec, comparator, "nominal")
        if comparator_nominal is None:
            skipped_comparators.append(
                {
                    "comparator": comparator,
                    "reason": "missing_nominal_seed_artifacts",
                    "missing_seed_count": _missing_seed_count(spec, comparator, "nominal"),
                }
            )
            continue
        comparator_tests: list[dict[str, Any]] = []
        for slice_name in slice_names:
            method_values = method_vectors[slice_name]
            comparator_values = comparator_nominal if slice_name == "nominal" else _metric_vector(spec, comparator, slice_name)
            if method_values is None or comparator_values is None:
                continue
            comparator_tests.append(_paired_test(slice_name, method_name, comparator, method_values, comparator_values))
        if comparator_tests:
            comparators.append(comparator)
            tests.extend(comparator_tests)
        else:
            skipped_comparators.append(
                {
                    "comparator": comparator,
                    "reason": "missing_overlay_seed_artifacts",
                    "missing_seed_count": _missing_seed_count(spec, comparator, slice_names[-1]),
                }
            )
    _apply_holm_correction(tests)
    return {
        "study_id": spec.study_id,
        "method_name": method_name,
        "seed_list": [int(seed) for seed in spec.seed_list],
        "comparators": comparators,
        "skipped_comparators": skipped_comparators,
        "tests": tests,
    }


def _metric_vector(spec, model_name: str, slice_name: str) -> np.ndarray | None:
    values = []
    for seed in spec.seed_list:
        metrics_path = _nominal_metrics_path(spec, model_name, int(seed)) if slice_name == "nominal" else _robustness_metrics_path(spec, model_name, slice_name, int(seed))
        if not metrics_path.exists():
            return None
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = float(payload["test_metrics"]["auroc"])
        if np.isnan(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=float)


def _nominal_metrics_path(spec, model_name: str, seed: int) -> Path:
    if model_name == str(spec.configs["method"]["method"]["name"]):
        return spec.raw_dir / "methods" / f"{model_name}__seed_{seed}" / "metrics.json"
    return spec.raw_dir / "baselines" / f"{model_name}__seed_{seed}" / "metrics.json"


def _robustness_metrics_path(spec, model_name: str, slice_name: str, seed: int) -> Path:
    return spec.raw_dir / "robustness" / f"{model_name}__{slice_name}__seed_{seed}" / "metrics.json"


def _missing_seed_count(spec, model_name: str, slice_name: str) -> int:
    count = 0
    for seed in spec.seed_list:
        metrics_path = _nominal_metrics_path(spec, model_name, int(seed)) if slice_name == "nominal" else _robustness_metrics_path(spec, model_name, slice_name, int(seed))
        if not metrics_path.exists():
            count += 1
    return count


def _paired_test(slice_name: str, method_name: str, comparator: str, method_values: np.ndarray, comparator_values: np.ndarray) -> dict[str, Any]:
    diffs = method_values - comparator_values
    ci_low, ci_high = _bootstrap_ci(diffs)
    if len(diffs) >= 2 and np.any(np.abs(diffs) > 1e-12):
        statistic, p_value = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
    else:
        statistic, p_value = 0.0, 1.0
    return {
        "slice": slice_name,
        "method_name": method_name,
        "comparator": comparator,
        "n_runs": int(len(diffs)),
        "mean_method_auroc": round(float(np.mean(method_values)), 6),
        "mean_comparator_auroc": round(float(np.mean(comparator_values)), 6),
        "mean_diff_auroc": round(float(np.mean(diffs)), 6),
        "wilcoxon_statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "holm_corrected_p_value": None,
        "bootstrap_ci_low": round(ci_low, 6),
        "bootstrap_ci_high": round(ci_high, 6),
    }


def _bootstrap_ci(diffs: np.ndarray, *, rounds: int = 5000, seed: int = 20260502) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray([float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(rounds)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _apply_holm_correction(tests: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(tests), key=lambda item: float(item[1]["p_value"]))
    running_max = 0.0
    m = len(ordered)
    for rank, (original_index, test) in enumerate(ordered, start=1):
        adjusted = min(1.0, float(test["p_value"]) * (m - rank + 1))
        running_max = max(running_max, adjusted)
        tests[original_index]["holm_corrected_p_value"] = round(running_max, 6)


def _cross_dataset_ranks(config_paths: list[str]) -> dict[str, Any]:
    rows = []
    for config_path in config_paths:
        spec = load_study_spec(config_path)
        summary_path = spec.aggregated_dir / "performance_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        nominal_models: dict[str, float] = {}
        for section in ("baseline_summary", "method_summary"):
            for model_name, metrics in summary.get(section, {}).items():
                value = float(metrics["mean_auroc"])
                if not np.isnan(value):
                    nominal_models[str(model_name)] = value
        rows.extend(_rank_rows(spec.study_id, "nominal", nominal_models))

        mcar30_models: dict[str, float] = {}
        for model_name, slice_map in summary.get("robustness_summary", {}).items():
            if "missingness_30" in slice_map:
                value = float(slice_map["missingness_30"]["mean_auroc"])
                if not np.isnan(value):
                    mcar30_models[str(model_name)] = value
        rows.extend(_rank_rows(spec.study_id, "missingness_30", mcar30_models))

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["slice"]), []).append(float(row["rank"]))
    mean_ranks = [
        {
            "model_name": model_name,
            "slice": slice_name,
            "mean_rank": round(float(np.mean(ranks)), 6),
            "n_datasets": len(ranks),
        }
        for (model_name, slice_name), ranks in sorted(grouped.items())
    ]
    return {"rows": rows, "mean_ranks": mean_ranks}


def _rank_rows(study_id: str, slice_name: str, model_scores: dict[str, float]) -> list[dict[str, Any]]:
    sorted_models = sorted(model_scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "study_id": study_id,
            "slice": slice_name,
            "model_name": model_name,
            "mean_auroc": round(float(score), 6),
            "rank": float(index + 1),
        }
        for index, (model_name, score) in enumerate(sorted_models)
    ]


def _study_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# All-Model Significance Results",
        "",
        f"Study: `{payload['study_id']}`. Method: `{payload['method_name']}`.",
        "",
        "| slice | comparator | mean_diff_auroc | holm_p | 95% bootstrap CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for test in payload["tests"]:
        lines.append(
            f"| {test['slice']} | {test['comparator']} | {test['mean_diff_auroc']:.6f} | "
            f"{float(test['holm_corrected_p_value'] or 1.0):.6f} | "
            f"[{test['bootstrap_ci_low']:.6f}, {test['bootstrap_ci_high']:.6f}] |"
        )
    if payload.get("skipped_comparators"):
        lines.extend(
            [
                "",
                "| skipped comparator | reason | missing_seed_count |",
                "| --- | --- | --- |",
            ]
        )
        for row in payload["skipped_comparators"]:
            lines.append(f"| {row['comparator']} | {row['reason']} | {row['missing_seed_count']} |")
    return "\n".join(lines) + "\n"


def _rank_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# All-Model Cross-Dataset Rank Summary",
        "",
        "| slice | model | mean_rank | n_datasets |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["mean_ranks"]:
        lines.append(f"| {row['slice']} | {row['model_name']} | {row['mean_rank']:.6f} | {row['n_datasets']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
