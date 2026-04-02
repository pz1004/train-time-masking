from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.baselines.tabular import ensure_baseline_environment, run_tabular_baseline
from lab.data import load_dataset
from lab.evaluation.metrics import binary_classification_metrics
from lab.evaluation.robustness import apply_mar_overlay
from lab.methods import mait_missingness_robustness as mait_impl
from lab.methods import support as method_support
from lab.study import load_study_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one MAR missingness overlay for a study.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated subset of seeds to process. Defaults to the full study seed roster.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine existing per-seed MAR partials without running new experiments.",
    )
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    if args.combine_only:
        _combine_partial_results(spec)
        return 0
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    mar_config = _mar_config(spec, dataset_bundle)
    selected_seeds = _selected_seeds(args.seeds, spec.seed_list)

    baseline_configs = list(spec.configs["baselines"]["baseline"])
    available_baselines = [config for config in baseline_configs if str(config["name"]) in {"lightgbm", "catboost", "logistic_regression", "random_forest", "xgboost"}]
    ensure_baseline_environment(available_baselines)

    blueprint = mait_impl._method_blueprint(spec)
    for seed in selected_seeds:
        print(f"Running MAR overlay for seed {seed}...")
        split_metadata = split_map[int(seed)]
        y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()
        X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
        overlay_features, overlay_metadata = apply_mar_overlay(
            X_test,
            target_columns=mar_config["target_columns"],
            driver_column=mar_config["driver_column"],
            seed=int(seed),
            low_rate=float(mar_config["low_rate"]),
            high_rate=float(mar_config["high_rate"]),
            threshold_quantile=float(mar_config["threshold_quantile"]),
        )

        seed_results: dict[str, dict[str, float]] = {}
        trained_method = mait_impl._train_variant(
            spec,
            "evaluate_mar",
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            blueprint=blueprint,
            result_kind="method",
            model_name=blueprint.method_name,
            use_missingness_indicators=True,
            use_calibration=False,
        )
        seed_results[blueprint.method_name] = binary_classification_metrics(
            y_test.to_numpy(), trained_method["predict_probabilities"](overlay_features)
        )

        for baseline_config in available_baselines:
            baseline_name = str(baseline_config["name"])
            trained_baseline = run_tabular_baseline(baseline_config, dataset_bundle, split_metadata)
            seed_results[baseline_name] = binary_classification_metrics(
                y_test.to_numpy(), trained_baseline.predict_probabilities(overlay_features)
            )
        _write_partial_result(
            spec,
            {
                "study_id": spec.study_id,
                "seed": int(seed),
                "mar_overlay": mar_config,
                "overlay_metadata": overlay_metadata,
                "results": seed_results,
            },
        )
        print(f"  -> MAIT AUROC={seed_results[blueprint.method_name]['auroc']:.6f}")

    _combine_partial_results(spec)
    return 0


def _mar_config(spec, dataset_bundle) -> dict[str, object]:
    configured = dict(spec.configs["robustness"].get("mar", {}))
    if configured:
        return {
            "driver_column": str(configured["driver_column"]),
            "target_columns": [str(column_name) for column_name in configured["target_columns"]],
            "low_rate": float(configured.get("low_rate", 0.05)),
            "high_rate": float(configured.get("high_rate", 0.35)),
            "threshold_quantile": float(configured.get("threshold_quantile", 0.5)),
        }
    driver_column = dataset_bundle.numerical_columns[0] if dataset_bundle.numerical_columns else dataset_bundle.categorical_columns[0]
    target_columns = [str(spec.configs["robustness"]["robustness"]["columns"][0])]
    return {
        "driver_column": driver_column,
        "target_columns": target_columns,
        "low_rate": 0.05,
        "high_rate": 0.35,
        "threshold_quantile": 0.5,
    }


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# MAR Overlay Results",
        "",
        f"Study: `{payload['study_id']}`.",
        "",
        f"Driver column: `{payload['mar_overlay']['driver_column']}`. Target columns: {', '.join(f'`{name}`' for name in payload['mar_overlay']['target_columns'])}.",
        "",
        "| model | mean_auroc | std_auroc | mean_ece | std_ece |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_name, metrics in payload["results"].items():
        lines.append(
            f"| {model_name} | {metrics['mean_auroc']:.6f} | {metrics['std_auroc']:.6f} | "
            f"{metrics['mean_ece']:.6f} | {metrics['std_ece']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _selected_seeds(seed_argument: str, study_seeds: list[int]) -> list[int]:
    if not seed_argument.strip():
        return [int(seed) for seed in study_seeds]
    requested = [int(token.strip()) for token in seed_argument.split(",") if token.strip()]
    invalid = sorted(set(requested) - set(int(seed) for seed in study_seeds))
    if invalid:
        invalid_text = ", ".join(str(seed) for seed in invalid)
        raise ValueError(f"Requested seeds are not in the study seed roster: {invalid_text}")
    return requested


def _write_partial_result(spec, payload: dict[str, object]) -> None:
    output_dir = spec.results_dir / "mar_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"seed_{int(payload['seed'])}.json"
    partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combine_partial_results(spec) -> None:
    output_dir = spec.results_dir / "mar_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(partial_dir.glob("seed_*.json"))
    ]
    if not partial_payloads:
        return
    mar_overlay = partial_payloads[0]["mar_overlay"]
    overlay_metadata = partial_payloads[0]["overlay_metadata"]
    model_names = sorted(partial_payloads[0]["results"])
    aggregated_results = {}
    for model_name in model_names:
        aurocs = [float(payload["results"][model_name]["auroc"]) for payload in partial_payloads]
        eces = [float(payload["results"][model_name]["ece"]) for payload in partial_payloads]
        aggregated_results[model_name] = {
            "mean_auroc": round(float(np.mean(aurocs)), 6),
            "std_auroc": round(float(np.std(aurocs, ddof=0)), 6),
            "mean_ece": round(float(np.mean(eces)), 6),
            "std_ece": round(float(np.std(eces, ddof=0)), 6),
            "n_runs": len(partial_payloads),
        }
    payload = {
        "study_id": spec.study_id,
        "mar_overlay": mar_overlay,
        "results": aggregated_results,
        "overlay_metadata": overlay_metadata,
    }
    json_path = output_dir / "mar_overlay_results.json"
    md_path = output_dir / "mar_overlay_results.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
