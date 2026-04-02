from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.data import load_dataset
from lab.methods import mait_missingness_robustness as mait_impl
from lab.methods import support as method_support
from lab.study import load_study_spec


DEFAULT_LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reconstruction-weight sweep for one study.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    parser.add_argument(
        "--lambda-grid",
        default=",".join(f"{value:.1f}" for value in DEFAULT_LAMBDA_GRID),
        help="Comma-separated reconstruction weights to evaluate.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine existing partial lambda-sweep outputs without running new experiments.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated subset of seeds to process. Defaults to the full study seed roster.",
    )
    args = parser.parse_args(argv)

    lambda_grid = [float(token) for token in args.lambda_grid.split(",") if token.strip()]
    spec = load_study_spec(args.study_config)
    if args.combine_only:
        _combine_partial_results(spec)
        return 0
    selected_seeds = _selected_seeds(args.seeds, spec.seed_list)
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)

    original_method_config = copy.deepcopy(spec.configs["method"])

    for reconstruction_weight in lambda_grid:
        print(f"Running lambda={reconstruction_weight:g}...")
        spec.configs["method"]["training"]["reconstruction_weight"] = float(reconstruction_weight)
        spec.configs["method"]["training"]["enable_reconstruction"] = bool(reconstruction_weight > 0.0)
        blueprint = mait_impl._method_blueprint(spec)
        for seed in selected_seeds:
            print(f"  seed {seed}...")
            split_metadata = split_map[int(seed)]
            trained_variant = mait_impl._train_variant(
                spec,
                "lambda_sweep",
                dataset_bundle=dataset_bundle,
                split_metadata=split_metadata,
                blueprint=blueprint,
                result_kind="ablation",
                model_name=f"lambda_{reconstruction_weight:g}",
                use_missingness_indicators=bool(spec.configs["method"]["training"].get("uses_missingness_indicators", True)),
                use_calibration=False,
            )
            nominal_metrics = trained_variant["metrics_payload"]["test_metrics"]
            robustness_summary: dict[str, dict[str, float]] = {}

            for slice_config in spec.configs["robustness"]["slice"]:
                payload, _, _ = method_support.robustness_artifacts(
                    spec,
                    "lambda_sweep",
                    model_name=f"lambda_{reconstruction_weight:g}",
                    split_metadata=split_metadata,
                    dataset_bundle=dataset_bundle,
                    predict_probabilities=trained_variant["predict_probabilities"],
                    slice_config=slice_config,
                    robustness_config=spec.configs["robustness"],
                    nominal_reference_kind="ablation",
                    model_metadata=trained_variant["metrics_payload"]["model_metadata"],
                    software_versions=trained_variant["metrics_payload"]["software_versions"],
                )
                robustness_summary[str(slice_config["name"])] = {
                    "auroc": float(payload["test_metrics"]["auroc"]),
                    "ece": float(payload["test_metrics"]["ece"]),
                }

            row = {
                "reconstruction_weight": float(reconstruction_weight),
                "seed": int(seed),
                "nominal_auroc": float(nominal_metrics["auroc"]),
                "nominal_ece": float(nominal_metrics["ece"]),
                "robustness": robustness_summary,
            }
            _write_partial_result(spec, row)
            print(
                "    -> nominal AUROC="
                f"{row['nominal_auroc']:.6f}, "
                "AUROC@30="
                f"{row['robustness'].get('missingness_30', {}).get('auroc', float('nan')):.6f}"
            )

    spec.configs["method"] = original_method_config
    _combine_partial_results(spec)
    return 0


def _write_partial_result(spec, row: dict[str, object]) -> None:
    output_dir = spec.results_dir / "lambda_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"lambda_{float(row['reconstruction_weight']):g}__seed_{int(row['seed'])}.json"
    payload = {"study_id": spec.study_id, "result": row}
    partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combine_partial_results(spec) -> None:
    output_dir = spec.results_dir / "lambda_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_paths = sorted(partial_dir.glob("lambda_*.json"))
    json_path = output_dir / "lambda_sweep.json"
    md_path = output_dir / "lambda_sweep.md"
    if not partial_paths:
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not md_path.exists():
                md_path.write_text(_markdown(payload), encoding="utf-8")
            print(f"No lambda-sweep partials found for {spec.study_id}; preserving existing combined outputs.")
            return
        raise SystemExit(f"No lambda-sweep partials found for {spec.study_id}: {partial_dir}")
    per_weight: dict[float, list[dict[str, object]]] = {}
    for partial_path in partial_paths:
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        row = payload["result"]
        per_weight.setdefault(float(row["reconstruction_weight"]), []).append(row)
    sweep_results = []
    for reconstruction_weight, rows in per_weight.items():
        nominal_aurocs = [float(row["nominal_auroc"]) for row in rows]
        nominal_eces = [float(row["nominal_ece"]) for row in rows]
        slice_names = sorted({slice_name for row in rows for slice_name in row["robustness"]})
        sweep_results.append(
            {
                "reconstruction_weight": float(reconstruction_weight),
                "n_runs": len(rows),
                "mean_nominal_auroc": round(float(np.mean(nominal_aurocs)), 6),
                "std_nominal_auroc": round(float(np.std(nominal_aurocs, ddof=0)), 6),
                "mean_nominal_ece": round(float(np.mean(nominal_eces)), 6),
                "std_nominal_ece": round(float(np.std(nominal_eces, ddof=0)), 6),
                "robustness": {
                    slice_name: {
                        "mean_auroc": round(
                            float(np.mean([float(row["robustness"][slice_name]["auroc"]) for row in rows])),
                            6,
                        ),
                        "std_auroc": round(
                            float(np.std([float(row["robustness"][slice_name]["auroc"]) for row in rows], ddof=0)),
                            6,
                        ),
                        "mean_ece": round(
                            float(np.mean([float(row["robustness"][slice_name]["ece"]) for row in rows])),
                            6,
                        ),
                        "std_ece": round(
                            float(np.std([float(row["robustness"][slice_name]["ece"]) for row in rows], ddof=0)),
                            6,
                        ),
                    }
                    for slice_name in slice_names
                },
            }
        )
    sweep_results.sort(key=lambda row: float(row["reconstruction_weight"]))
    payload = {"study_id": spec.study_id, "results": sweep_results}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Reconstruction-Weight Sweep",
        "",
        f"Study: `{payload['study_id']}`.",
        "",
        "| lambda | mean_nominal_auroc | mean_nominal_ece | missingness_10 | missingness_20 | missingness_30 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['reconstruction_weight']:.1f} | {row['mean_nominal_auroc']:.6f} | "
            f"{row['mean_nominal_ece']:.6f} | "
            f"{row['robustness'].get('missingness_10', {}).get('mean_auroc', float('nan')):.6f} | "
            f"{row['robustness'].get('missingness_20', {}).get('mean_auroc', float('nan')):.6f} | "
            f"{row['robustness'].get('missingness_30', {}).get('mean_auroc', float('nan')):.6f} |"
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


if __name__ == "__main__":
    raise SystemExit(main())
