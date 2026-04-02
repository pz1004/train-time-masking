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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MAIT training-mask sweep for one study.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)

    original_method_config = copy.deepcopy(spec.configs["method"])
    mask_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_results = []

    for rate in mask_grid:
        spec.configs["method"]["augmentation"]["mask_rates"] = [float(rate)]
        blueprint = mait_impl._method_blueprint(spec)
        nominal_aurocs = []
        robustness_summary = {}

        for seed in spec.seed_list:
            split_metadata = split_map[int(seed)]
            trained_variant = mait_impl._train_variant(
                spec,
                "mask_sweep",
                dataset_bundle=dataset_bundle,
                split_metadata=split_metadata,
                blueprint=blueprint,
                result_kind="method",
                model_name=blueprint.method_name,
                use_missingness_indicators=True,
                use_calibration=False,
            )
            nominal_aurocs.append(float(trained_variant["metrics_payload"]["test_metrics"]["auroc"]))
            for slice_config in spec.configs["robustness"]["slice"]:
                payload, _, _ = method_support.robustness_artifacts(
                    spec,
                    "mask_sweep",
                    model_name=blueprint.method_name,
                    split_metadata=split_metadata,
                    dataset_bundle=dataset_bundle,
                    predict_probabilities=trained_variant["predict_probabilities"],
                    slice_config=slice_config,
                    robustness_config=spec.configs["robustness"],
                    nominal_reference_kind="method",
                    model_metadata=trained_variant["metrics_payload"]["model_metadata"],
                    software_versions=trained_variant["metrics_payload"]["software_versions"],
                )
                robustness_summary.setdefault(str(slice_config["name"]), []).append(float(payload["test_metrics"]["auroc"]))

        sweep_results.append(
            {
                "mask_rate": float(rate),
                "mean_nominal_auroc": round(float(np.mean(nominal_aurocs)), 6),
                "std_nominal_auroc": round(float(np.std(nominal_aurocs, ddof=0)), 6),
                "robustness": {
                    slice_name: {
                        "mean_auroc": round(float(np.mean(values)), 6),
                        "std_auroc": round(float(np.std(values, ddof=0)), 6),
                    }
                    for slice_name, values in robustness_summary.items()
                },
            }
        )

    spec.configs["method"] = original_method_config
    output_dir = spec.results_dir / "mask_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mask_sweep.json"
    md_path = output_dir / "mask_sweep.md"
    payload = {"study_id": spec.study_id, "results": sweep_results}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return 0


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Mask Sweep",
        "",
        f"Study: `{payload['study_id']}`.",
        "",
        "| mask_rate | mean_nominal_auroc | missingness_10 | missingness_20 | missingness_30 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['mask_rate']:.2f} | {row['mean_nominal_auroc']:.6f} | "
            f"{row['robustness'].get('missingness_10', {}).get('mean_auroc', float('nan')):.6f} | "
            f"{row['robustness'].get('missingness_20', {}).get('mean_auroc', float('nan')):.6f} | "
            f"{row['robustness'].get('missingness_30', {}).get('mean_auroc', float('nan')):.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
