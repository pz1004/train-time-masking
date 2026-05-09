from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.baselines.tabular import ensure_baseline_environment, run_tabular_baseline
from lab.data import load_dataset
from lab.evaluation.metrics import binary_classification_metrics
from lab.evaluation.robustness import apply_missingness_overlay
from lab.methods import mait_missingness_robustness as mait_impl
from lab.methods import support as method_support
from lab.study import load_study_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate permutation-importance stability under missingness.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated subset of seeds.")
    parser.add_argument("--models", default="all", help="Comma-separated model names or `all`.")
    parser.add_argument("--combine-only", action="store_true", help="Combine existing partials without training.")
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    if args.combine_only:
        _combine_partial_results(spec)
        return 0

    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    selected_seeds = _selected_seeds(args.seeds, spec.seed_list)
    selected_models = _selected_models(args.models, spec)
    baseline_configs = [config for config in spec.configs["baselines"]["baseline"] if str(config["name"]) in selected_models]
    ensure_baseline_environment(baseline_configs)
    blueprint = mait_impl._method_blueprint(spec)
    hardest_slice = spec.configs["robustness"]["slice"][-1]

    for seed in selected_seeds:
        partial_path = _partial_path(spec, seed)
        if partial_path.exists():
            continue
        print(f"Running feature stability for seed {seed}...")
        split_metadata = split_map[int(seed)]
        y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()
        X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
        overlay_features, overlay_metadata = apply_missingness_overlay(
            X_test,
            spec.configs["robustness"],
            hardest_slice,
            seed=int(seed),
        )
        predictors = _train_predictors(
            spec,
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            baseline_configs=baseline_configs,
            blueprint=blueprint,
            include_method=blueprint.method_name in selected_models,
        )
        results = {}
        for model_name, predictor in predictors.items():
            nominal_importance = _permutation_importance(
                predictor,
                X_test,
                y_test.to_numpy(),
                seed=int(seed),
                namespace=f"{model_name}:nominal",
            )
            overlay_importance = _permutation_importance(
                predictor,
                overlay_features,
                y_test.to_numpy(),
                seed=int(seed),
                namespace=f"{model_name}:{overlay_metadata['slice_name']}",
            )
            nominal_vector = np.asarray([nominal_importance[column] for column in X_test.columns], dtype=float)
            overlay_vector = np.asarray([overlay_importance[column] for column in X_test.columns], dtype=float)
            correlation = spearmanr(nominal_vector, overlay_vector, nan_policy="omit").correlation
            results[model_name] = {
                "nominal_importance": nominal_importance,
                "overlay_importance": overlay_importance,
                "spearman_rank_correlation": None if np.isnan(correlation) else round(float(correlation), 6),
                "mean_absolute_importance_shift": round(float(np.mean(np.abs(nominal_vector - overlay_vector))), 6),
            }
        _write_partial_result(
            spec,
            {
                "study_id": spec.study_id,
                "seed": int(seed),
                "slice_name": str(overlay_metadata["slice_name"]),
                "results": results,
            },
        )
    _combine_partial_results(spec)
    return 0


def _train_predictors(
    spec,
    *,
    dataset_bundle,
    split_metadata: dict[str, Any],
    baseline_configs: list[dict[str, Any]],
    blueprint,
    include_method: bool,
) -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    predictors: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {}
    for baseline_config in baseline_configs:
        baseline_result = run_tabular_baseline(baseline_config, dataset_bundle, split_metadata)
        predictors[str(baseline_config["name"])] = baseline_result.predict_probabilities
    if include_method:
        training_config = spec.configs["method"].get("training", {})
        method_variant = mait_impl._train_variant(
            spec,
            "run_feature_stability",
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            blueprint=blueprint,
            result_kind="method",
            model_name=blueprint.method_name,
            use_missingness_indicators=bool(training_config.get("uses_missingness_indicators", True)),
            use_calibration=bool(training_config.get("uses_validation_calibration", False)),
        )
        predictors[blueprint.method_name] = method_variant["predict_probabilities"]
    return predictors


def _permutation_importance(
    predictor: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    namespace: str,
) -> dict[str, float]:
    base_probabilities = predictor(X)
    base_auroc = float(binary_classification_metrics(y, base_probabilities)["auroc"])
    importances: dict[str, float] = {}
    rng = np.random.default_rng(_stable_seed(seed, namespace))
    for column_name in X.columns:
        permuted = X.copy()
        values = permuted[column_name].to_numpy(copy=True)
        permuted[column_name] = rng.permutation(values)
        permuted_probabilities = predictor(permuted)
        permuted_auroc = float(binary_classification_metrics(y, permuted_probabilities)["auroc"])
        importances[str(column_name)] = round(base_auroc - permuted_auroc, 6)
    return importances


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _selected_models(model_argument: str, spec) -> set[str]:
    method_name = str(spec.configs["method"]["method"]["name"])
    all_models = {str(config["name"]) for config in spec.configs["baselines"]["baseline"]} | {method_name}
    if model_argument.strip() == "all":
        return all_models
    requested = {token.strip() for token in model_argument.split(",") if token.strip()}
    unknown = sorted(requested - all_models)
    if unknown:
        raise ValueError("Unknown model(s): " + ", ".join(unknown))
    return requested


def _selected_seeds(seed_argument: str, study_seeds: list[int]) -> list[int]:
    if not seed_argument.strip():
        return [int(seed) for seed in study_seeds]
    requested = [int(token.strip()) for token in seed_argument.split(",") if token.strip()]
    invalid = sorted(set(requested) - set(int(seed) for seed in study_seeds))
    if invalid:
        raise ValueError("Requested seeds are not in the study seed roster: " + ", ".join(str(seed) for seed in invalid))
    return requested


def _partial_path(spec, seed: int) -> Path:
    return spec.results_dir / "feature_stability" / "partials" / f"seed_{int(seed)}.json"


def _write_partial_result(spec, payload: dict[str, Any]) -> None:
    partial_path = _partial_path(spec, int(payload["seed"]))
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combine_partial_results(spec) -> None:
    output_dir = spec.results_dir / "feature_stability"
    partial_dir = output_dir / "partials"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(partial_dir.glob("seed_*.json"))]
    if not partial_payloads:
        return
    model_names = sorted(partial_payloads[0]["results"])
    aggregated = {}
    for model_name in model_names:
        rows = [payload["results"][model_name] for payload in partial_payloads]
        correlations = [float(row["spearman_rank_correlation"]) for row in rows if row["spearman_rank_correlation"] is not None]
        shifts = [float(row["mean_absolute_importance_shift"]) for row in rows]
        aggregated[model_name] = {
            "n_runs": len(rows),
            "mean_spearman_rank_correlation": round(float(np.mean(correlations)), 6) if correlations else None,
            "std_spearman_rank_correlation": round(float(np.std(correlations, ddof=0)), 6) if correlations else None,
            "mean_absolute_importance_shift": round(float(np.mean(shifts)), 6),
            "std_absolute_importance_shift": round(float(np.std(shifts, ddof=0)), 6),
        }
    payload = {"study_id": spec.study_id, "slice_name": partial_payloads[0]["slice_name"], "results": aggregated}
    (output_dir / "feature_stability_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "feature_stability_results.md").write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Feature-Importance Stability Results",
        "",
        f"Study: `{payload['study_id']}`. Slice: `{payload['slice_name']}`.",
        "",
        "| model | mean_spearman_rank_correlation | mean_absolute_importance_shift |",
        "| --- | --- | --- |",
    ]
    for model_name, result in payload["results"].items():
        corr = "" if result["mean_spearman_rank_correlation"] is None else f"{result['mean_spearman_rank_correlation']:.6f}"
        lines.append(f"| {model_name} | {corr} | {result['mean_absolute_importance_shift']:.6f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
