from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.study import load_study_spec


# The manuscript pre-specifies boosted-tree significance comparisons against these two baselines only.
DEFAULT_COMPARATORS = ("lightgbm", "xgboost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute paired significance tests for a robustness study.")
    parser.add_argument("--study-config", required=True, help="Path to configs/studies/<study_id>.toml")
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    summary_path = spec.aggregated_dir / "performance_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing aggregated summary: {summary_path}")

    nominal_method_dir = spec.raw_dir / "methods"
    if not nominal_method_dir.exists():
        raise SystemExit(f"Missing method artifacts: {nominal_method_dir}")

    method_name = str(spec.configs["method"]["method"]["name"])
    results = _compute_significance(spec, method_name=method_name)

    output_dir = spec.audits_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "significance_results.json"
    md_path = output_dir / "significance_results.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(results), encoding="utf-8")
    return 0


def _compute_significance(spec, *, method_name: str) -> dict[str, object]:
    baseline_root = spec.raw_dir / "baselines"
    robustness_root = spec.raw_dir / "robustness"
    comparators = []
    for candidate in DEFAULT_COMPARATORS:
        if all((baseline_root / f"{candidate}__seed_{seed}" / "metrics.json").exists() for seed in spec.seed_list):
            comparators.append(candidate)

    slice_names = [str(item["name"]) for item in spec.configs["robustness"]["slice"]]
    tests: list[dict[str, object]] = []

    for comparator in comparators:
        method_nominal = _metric_vector(spec.raw_dir / "methods", method_name, spec.seed_list)
        comparator_nominal = _metric_vector(baseline_root, comparator, spec.seed_list)
        tests.append(_paired_test("nominal", method_name, comparator, method_nominal, comparator_nominal))
        for slice_name in slice_names:
            method_slice = _metric_vector(robustness_root, f"{method_name}__{slice_name}", spec.seed_list, robustness=True)
            comparator_slice = _metric_vector(robustness_root, f"{comparator}__{slice_name}", spec.seed_list, robustness=True)
            tests.append(_paired_test(slice_name, method_name, comparator, method_slice, comparator_slice))

    _apply_holm_correction(tests)
    return {
        "study_id": spec.study_id,
        "method_name": method_name,
        "seed_list": list(spec.seed_list),
        "comparators": comparators,
        "tests": tests,
    }


def _metric_vector(root: Path, run_name: str, seed_list: list[int], *, robustness: bool = False) -> np.ndarray:
    values = []
    for seed in seed_list:
        if robustness:
            metrics_path = root / f"{run_name}__seed_{seed}" / "metrics.json"
        else:
            metrics_path = root / f"{run_name}__seed_{seed}" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        values.append(float(payload["test_metrics"]["auroc"]))
    return np.asarray(values, dtype=float)


def _paired_test(slice_name: str, method_name: str, comparator: str, method_values: np.ndarray, comparator_values: np.ndarray) -> dict[str, object]:
    diffs = method_values - comparator_values
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = _bootstrap_ci(diffs)
    non_zero = np.any(np.abs(diffs) > 1e-12)
    if len(diffs) >= 2 and non_zero:
        statistic, p_value = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic_value = float(statistic)
        p_value_value = float(p_value)
    else:
        statistic_value = 0.0
        p_value_value = 1.0
    return {
        "slice": slice_name,
        "method_name": method_name,
        "comparator": comparator,
        "n_runs": int(len(diffs)),
        "mean_method_auroc": round(float(np.mean(method_values)), 6),
        "mean_comparator_auroc": round(float(np.mean(comparator_values)), 6),
        "mean_diff_auroc": round(mean_diff, 6),
        "wilcoxon_statistic": round(statistic_value, 6),
        "p_value": round(p_value_value, 6),
        "holm_corrected_p_value": None,
        "bootstrap_ci_low": round(ci_low, 6),
        "bootstrap_ci_high": round(ci_high, 6),
    }


def _bootstrap_ci(diffs: np.ndarray, *, rounds: int = 5000, seed: int = 20260327) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(rounds)],
        dtype=float,
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _apply_holm_correction(tests: list[dict[str, object]]) -> None:
    ordered = sorted(enumerate(tests), key=lambda item: float(item[1]["p_value"]))
    m = len(ordered)
    corrected = [0.0] * m
    running_max = 0.0
    for rank, (original_index, test) in enumerate(ordered, start=1):
        adjusted = min(1.0, float(test["p_value"]) * (m - rank + 1))
        running_max = max(running_max, adjusted)
        corrected[rank - 1] = running_max
        tests[original_index]["holm_corrected_p_value"] = round(running_max, 6)


def _markdown(results: dict[str, object]) -> str:
    lines = [
        "# Significance Results",
        "",
        f"Study: `{results['study_id']}`. Method: `{results['method_name']}`.",
        "",
        "| slice | comparator | n_runs | mean_method_auroc | mean_comparator_auroc | mean_diff_auroc | p_value | holm_corrected_p_value | 95% bootstrap CI |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for test in results["tests"]:
        lines.append(
            f"| {test['slice']} | {test['comparator']} | {test['n_runs']} | {test['mean_method_auroc']:.6f} | "
            f"{test['mean_comparator_auroc']:.6f} | {test['mean_diff_auroc']:.6f} | {test['p_value']:.6f} | "
            f"{float(test['holm_corrected_p_value'] or 1.0):.6f} | [{test['bootstrap_ci_low']:.6f}, {test['bootstrap_ci_high']:.6f}] |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
