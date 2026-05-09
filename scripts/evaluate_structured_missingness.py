from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.baselines.tabular import ensure_baseline_environment, run_tabular_baseline
from lab.data import load_dataset
from lab.evaluation.metrics import binary_classification_metrics
from lab.evaluation.robustness import apply_structured_missingness_overlay, default_structured_overlay_configs
from lab.methods import mait_missingness_robustness as mait_impl
from lab.methods import support as method_support
from lab.study import load_study_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MAR/MNAR structured missingness overlays for a study.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated subset of seeds.")
    parser.add_argument("--combine-only", action="store_true", help="Combine existing partials without training.")
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    if args.combine_only:
        _combine_partial_results(spec)
        return 0

    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    overlay_configs = _structured_overlay_configs(spec)
    selected_seeds = _selected_seeds(args.seeds, spec.seed_list)
    baseline_configs = list(spec.configs["baselines"]["baseline"])
    ensure_baseline_environment(baseline_configs)
    blueprint = mait_impl._method_blueprint(spec)
    ablation_plans = mait_impl._study_ablation_plans(spec)

    for seed in selected_seeds:
        partial_path = _partial_path(spec, seed)
        if partial_path.exists():
            continue
        print(f"Running structured overlays for seed {seed}...")
        split_metadata = split_map[int(seed)]
        y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()
        X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
        predictors = _train_predictors(
            spec,
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            baseline_configs=baseline_configs,
            blueprint=blueprint,
            ablation_plans=ablation_plans,
        )

        seed_results: dict[str, dict[str, dict[str, float]]] = {}
        overlay_metadata: dict[str, dict[str, Any]] = {}
        for overlay_config in overlay_configs:
            overlay_features, metadata = apply_structured_missingness_overlay(X_test, overlay_config, seed=int(seed))
            overlay_name = str(metadata["slice_name"])
            overlay_metadata[overlay_name] = metadata
            seed_results[overlay_name] = {}
            for model_name, predictor in predictors.items():
                probabilities = predictor(overlay_features)
                seed_results[overlay_name][model_name] = binary_classification_metrics(y_test.to_numpy(), probabilities)
        _write_partial_result(
            spec,
            {
                "study_id": spec.study_id,
                "seed": int(seed),
                "overlay_configs": overlay_configs,
                "overlay_metadata": overlay_metadata,
                "results": seed_results,
            },
        )
        print(f"  -> overlays={', '.join(seed_results)} models={len(predictors)}")

    _combine_partial_results(spec)
    return 0


def _train_predictors(
    spec,
    *,
    dataset_bundle,
    split_metadata: dict[str, Any],
    baseline_configs: list[dict[str, Any]],
    blueprint,
    ablation_plans: list[dict[str, Any]],
) -> dict[str, Callable[[Any], np.ndarray]]:
    predictors: dict[str, Callable[[Any], np.ndarray]] = {}
    for baseline_config in baseline_configs:
        baseline_result = run_tabular_baseline(baseline_config, dataset_bundle, split_metadata)
        predictors[str(baseline_config["name"])] = baseline_result.predict_probabilities

    training_config = spec.configs["method"].get("training", {})
    method_variant = mait_impl._train_variant(
        spec,
        "evaluate_structured_missingness",
        dataset_bundle=dataset_bundle,
        split_metadata=split_metadata,
        blueprint=blueprint,
        result_kind="method",
        model_name=blueprint.method_name,
        use_missingness_indicators=bool(training_config.get("uses_missingness_indicators", True)),
        use_calibration=bool(training_config.get("uses_validation_calibration", False)),
    )
    predictors[blueprint.method_name] = method_variant["predict_probabilities"]

    for plan in ablation_plans:
        ablation_variant = mait_impl._train_variant(
            spec,
            "evaluate_structured_missingness",
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            blueprint=blueprint,
            result_kind="ablation",
            model_name=str(plan["name"]),
            use_missingness_indicators=bool(plan["use_missingness_indicators"]),
            use_calibration=bool(plan["use_calibration"]),
            training_overrides=dict(plan.get("training_overrides", {})),
            augmentation_overrides=dict(plan.get("augmentation_overrides", {})),
        )
        predictors[str(plan["name"])] = ablation_variant["predict_probabilities"]
    return predictors


def _structured_overlay_configs(spec) -> list[dict[str, Any]]:
    configured = list(spec.configs["robustness"].get("structured_overlay", []))
    if configured:
        return [dict(item) for item in configured]
    return default_structured_overlay_configs(spec.configs["robustness"])


def _selected_seeds(seed_argument: str, study_seeds: list[int]) -> list[int]:
    if not seed_argument.strip():
        return [int(seed) for seed in study_seeds]
    requested = [int(token.strip()) for token in seed_argument.split(",") if token.strip()]
    invalid = sorted(set(requested) - set(int(seed) for seed in study_seeds))
    if invalid:
        raise ValueError("Requested seeds are not in the study seed roster: " + ", ".join(str(seed) for seed in invalid))
    return requested


def _partial_path(spec, seed: int) -> Path:
    return spec.results_dir / "structured_missingness" / "partials" / f"seed_{int(seed)}.json"


def _write_partial_result(spec, payload: dict[str, Any]) -> None:
    partial_path = _partial_path(spec, int(payload["seed"]))
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combine_partial_results(spec) -> None:
    output_dir = spec.results_dir / "structured_missingness"
    partial_dir = output_dir / "partials"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(partial_dir.glob("seed_*.json"))]
    if not partial_payloads:
        return

    overlay_names = sorted(partial_payloads[0]["results"])
    model_names = sorted(next(iter(partial_payloads[0]["results"].values())).keys())
    aggregated_results: dict[str, dict[str, dict[str, float]]] = {}
    for overlay_name in overlay_names:
        aggregated_results[overlay_name] = {}
        for model_name in model_names:
            metrics_by_seed = [payload["results"][overlay_name][model_name] for payload in partial_payloads]
            aggregated_results[overlay_name][model_name] = _summarize_metrics(metrics_by_seed)

    payload = {
        "study_id": spec.study_id,
        "n_runs": len(partial_payloads),
        "overlay_configs": partial_payloads[0]["overlay_configs"],
        "overlay_metadata": partial_payloads[0]["overlay_metadata"],
        "results": aggregated_results,
    }
    json_path = output_dir / "structured_missingness_results.json"
    md_path = output_dir / "structured_missingness_results.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _summarize_metrics(metrics_by_seed: list[dict[str, float]]) -> dict[str, float]:
    metric_names = metrics_by_seed[0].keys()
    summary = {"n_runs": float(len(metrics_by_seed))}
    for metric_name in metric_names:
        values = [float(metrics[metric_name]) for metrics in metrics_by_seed]
        summary[f"mean_{metric_name}"] = round(float(np.mean(values)), 6)
        summary[f"std_{metric_name}"] = round(float(np.std(values, ddof=0)), 6)
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Structured Missingness Results",
        "",
        f"Study: `{payload['study_id']}`. Runs: {payload['n_runs']}.",
        "",
        "| overlay | model | mean_auroc | std_auroc | mean_ece | std_ece |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for overlay_name, model_map in payload["results"].items():
        for model_name, metrics in model_map.items():
            lines.append(
                f"| {overlay_name} | {model_name} | {metrics['mean_auroc']:.6f} | {metrics['std_auroc']:.6f} | "
                f"{metrics['mean_ece']:.6f} | {metrics['std_ece']:.6f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
