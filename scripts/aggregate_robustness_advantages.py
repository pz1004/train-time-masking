from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.study import load_study_spec


COMPARATORS = ("lightgbm", "xgboost", "catboost", "random_forest")
DATASET_LABELS = {
    "adult": "Adult",
    "credit-g": "German Credit",
    "bank-marketing": "Bank Marketing",
    "credit": "Give Me Some Credit",
    "give-me-some-credit": "Give Me Some Credit",
    "covertype": "Covertype",
}
COMPARATOR_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate 30% MCAR robustness advantages with confidence intervals.")
    parser.add_argument(
        "--study-glob",
        default="configs/studies/*missingness_robustness*.toml",
        help="Glob for the study configs to aggregate.",
    )
    parser.add_argument(
        "--slice-name",
        default="missingness_30",
        help="Robustness slice to compare against each model's nominal performance.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.005,
        help="Practical-significance threshold for the mean robustness advantage.",
    )
    args = parser.parse_args(argv)

    config_paths = sorted(glob(args.study_glob))
    if not config_paths:
        raise SystemExit(f"No study configs matched: {args.study_glob}")

    rows = []
    for config_path in config_paths:
        spec = load_study_spec(config_path)
        method_name = str(spec.configs["method"]["method"]["name"])
        dataset_key = str(spec.configs["dataset"]["dataset"].get("primary_dataset", spec.study_id))
        for comparator in COMPARATORS:
            diffs = _advantage_vector(spec, method_name=method_name, comparator=comparator, slice_name=args.slice_name)
            ci_low, ci_high = _bootstrap_ci(diffs)
            rows.append(
                {
                    "study_id": spec.study_id,
                    "dataset_key": dataset_key,
                    "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
                    "slice_name": args.slice_name,
                    "method_name": method_name,
                    "comparator": comparator,
                    "comparator_label": COMPARATOR_LABELS.get(comparator, comparator),
                    "n_runs": int(len(diffs)),
                    "mean_advantage": round(float(np.mean(diffs)), 6),
                    "ci_low": round(ci_low, 6),
                    "ci_high": round(ci_high, 6),
                    "practical_threshold": float(args.delta_threshold),
                    "practical_at_threshold": bool(float(np.mean(diffs)) > float(args.delta_threshold)),
                }
            )

    rows.sort(key=lambda item: (item["dataset_label"], item["comparator_label"]))
    output_dir = ROOT / "paper" / "submission_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robustness_advantages.json"
    md_path = output_dir / "robustness_advantages.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return 0


def _advantage_vector(spec, *, method_name: str, comparator: str, slice_name: str) -> np.ndarray:
    method_nominal = _auroc_vector(spec.raw_dir / "methods", method_name, spec.seed_list)
    method_slice = _auroc_vector(spec.raw_dir / "robustness", f"{method_name}__{slice_name}", spec.seed_list)
    comparator_nominal = _auroc_vector(spec.raw_dir / "baselines", comparator, spec.seed_list)
    comparator_slice = _auroc_vector(spec.raw_dir / "robustness", f"{comparator}__{slice_name}", spec.seed_list)
    method_delta = method_slice - method_nominal
    comparator_delta = comparator_slice - comparator_nominal
    return np.abs(comparator_delta) - np.abs(method_delta)


def _auroc_vector(root: Path, run_name: str, seed_list: list[int]) -> np.ndarray:
    values = []
    for seed in seed_list:
        metrics_path = root / f"{run_name}__seed_{seed}" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        values.append(float(payload["test_metrics"]["auroc"]))
    return np.asarray(values, dtype=float)


def _bootstrap_ci(values: np.ndarray, *, rounds: int = 5000, seed: int = 20260331) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(rounds)],
        dtype=float,
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Robustness Advantage Summary",
        "",
        "| dataset | comparator | mean_advantage | 95% CI | practical_threshold | exceeds_threshold |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {row['comparator_label']} | {row['mean_advantage']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['practical_threshold']:.3f} | "
            f"{'yes' if row['practical_at_threshold'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
