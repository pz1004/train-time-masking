from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
import tomllib
import unittest
from pathlib import Path

import torch

sklearn_module = types.ModuleType("sklearn")
metrics_module = types.ModuleType("sklearn.metrics")
metrics_module.roc_auc_score = lambda y_true, y_prob: 0.5
sklearn_module.metrics = metrics_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules["sklearn.metrics"] = metrics_module

from lab.models import neural_tabular
from lab.reporting import build_severity_series


ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIG = ROOT / "configs" / "ablations" / "lineage_d9_06_adult_missingness_robustness.toml"


class DraftCodeConsistencyTests(unittest.TestCase):
    def test_masking_respects_augmentation_columns_and_observed_only(self) -> None:
        batch = {
            "numerical": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "numerical_missing": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
            "categorical": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "categorical_missing": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        }
        masked = neural_tabular._apply_random_mask(
            batch,
            rng=neural_tabular.np.random.default_rng(7),
            rate=1.0,
            use_missingness_indicators=True,
            numerical_augmentation_mask=torch.tensor([True, False], dtype=torch.bool),
            categorical_augmentation_mask=torch.tensor([False, True], dtype=torch.bool),
            mask_only_observed_values=True,
        )

        expected_numerical = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
        expected_categorical = torch.tensor([[False, True], [False, True]], dtype=torch.bool)
        self.assertTrue(torch.equal(masked["artificial_numerical_mask"].cpu(), expected_numerical))
        self.assertTrue(torch.equal(masked["artificial_categorical_mask"].cpu(), expected_categorical))
        self.assertEqual(float(masked["categorical_indicator"][0, 0].item()), 1.0)
        self.assertEqual(float(masked["categorical_indicator"][0, 1].item()), 1.0)

    def test_mask_only_observed_values_false_changes_mask_behavior(self) -> None:
        batch = {
            "numerical": torch.tensor([[1.0]], dtype=torch.float32),
            "numerical_missing": torch.tensor([[1.0]], dtype=torch.float32),
            "categorical": torch.zeros((1, 0), dtype=torch.long),
            "categorical_missing": torch.zeros((1, 0), dtype=torch.float32),
        }
        masked = neural_tabular._apply_random_mask(
            batch,
            rng=neural_tabular.np.random.default_rng(11),
            rate=1.0,
            use_missingness_indicators=True,
            numerical_augmentation_mask=torch.tensor([True], dtype=torch.bool),
            categorical_augmentation_mask=torch.zeros((0,), dtype=torch.bool),
            mask_only_observed_values=False,
        )

        self.assertTrue(bool(masked["artificial_numerical_mask"][0, 0].item()))
        self.assertFalse(bool(masked["reconstruction_numerical_mask"][0, 0].item()))

    def test_reconstruction_loss_averages_over_masked_entries(self) -> None:
        numeric_reconstruction = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
        categorical_reconstruction = [
            torch.zeros((1, 3), dtype=torch.float32),
            torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        ]
        original_batch = {
            "numerical": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "categorical": torch.tensor([[1, 2]], dtype=torch.long),
        }
        masked_batch = {
            "reconstruction_numerical_mask": torch.tensor([[True, False]], dtype=torch.bool),
            "reconstruction_categorical_mask": torch.tensor([[False, True]], dtype=torch.bool),
        }

        loss = neural_tabular._reconstruction_loss(
            numeric_reconstruction=numeric_reconstruction,
            categorical_reconstruction=categorical_reconstruction,
            original_batch=original_batch,
            masked_batch=masked_batch,
        )
        numeric_loss = 9.0
        categorical_loss = torch.nn.functional.cross_entropy(
            categorical_reconstruction[1],
            torch.tensor([2], dtype=torch.long),
            reduction="sum",
        ).item()
        expected = (numeric_loss + categorical_loss) / 2.0
        self.assertAlmostEqual(float(loss.item()), float(expected), places=6)

    def test_identity_mask_batch_disables_artificial_masks(self) -> None:
        batch = {
            "numerical": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "numerical_missing": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "categorical": torch.tensor([[3, 4]], dtype=torch.long),
            "categorical_missing": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        }
        masked = neural_tabular._identity_mask_batch(batch, use_missingness_indicators=False)

        self.assertTrue(torch.equal(masked["numerical_input"], batch["numerical"]))
        self.assertTrue(torch.equal(masked["categorical_input"], batch["categorical"]))
        self.assertFalse(bool(masked["artificial_numerical_mask"].any().item()))
        self.assertFalse(bool(masked["artificial_categorical_mask"].any().item()))
        self.assertFalse(bool(masked["numerical_indicator"].any().item()))
        self.assertFalse(bool(masked["categorical_indicator"].any().item()))

    def test_ece_curve_uses_nominal_ece_anchor(self) -> None:
        summary = {
            "baseline_summary": {
                "lightgbm": {"mean_auroc": 0.91, "mean_ece": 0.012},
            },
            "method_summary": {},
            "ablation_summary": {},
            "robustness_summary": {
                "lightgbm": {
                    "missingness_10": {
                        "additional_mask_rate": 0.1,
                        "mean_auroc": 0.9,
                        "mean_auroc_delta": -0.01,
                        "mean_ece": 0.02,
                    }
                }
            },
        }
        series = build_severity_series(summary, metric_key="mean_ece")
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(float(series[0]["points"][0]["y"]), 0.012, places=9)

    def test_manuscript_does_not_hardcode_calibration_bases_or_stale_repo_claims(self) -> None:
        text = (ROOT / "access.tex").read_text(encoding="utf-8")
        self.assertNotIn("LightGBM & Nominal", text)
        self.assertNotIn("CatBoost & Nominal", text)
        self.assertNotIn("still require dedicated configs, study docs", text)
        self.assertNotIn("figure set must be upgraded", text)
        self.assertIn("dynamically selected top-2 baselines", text)
        self.assertIn("Paired significance summary", text)

    def test_significance_scope_is_limited_to_prespecified_boosted_tree_comparators(self) -> None:
        spec = importlib.util.spec_from_file_location("run_significance", ROOT / "scripts" / "run_significance.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.DEFAULT_COMPARATORS, ("lightgbm", "xgboost"))

        for path in sorted((ROOT / "results").glob("*/audits/significance_results.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["comparators"], ["lightgbm", "xgboost"], msg=str(path))

    def test_baseline_calibration_counts_match_artifacts_and_manuscript(self) -> None:
        sigmoid_cases = 0
        sigmoid_improved_ece = 0
        isotonic_cases = 0
        isotonic_reduced_auroc = 0

        for path in sorted((ROOT / "results").glob("*/aggregated/performance_summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for model_name, summary in payload["calibration_summary"].items():
                if model_name.endswith("_calibrated_sigmoid"):
                    sigmoid_cases += 1
                    if float(summary["mean_ece_delta"]) < 0.0:
                        sigmoid_improved_ece += 1
                    self.assertEqual(float(summary["mean_auroc_delta"]), 0.0, msg=model_name)
                if model_name.endswith("_calibrated_isotonic"):
                    isotonic_cases += 1
                    self.assertLess(float(summary["mean_auroc_delta"]), 0.0, msg=model_name)
                    isotonic_reduced_auroc += 1

        self.assertEqual(sigmoid_cases, 10)
        self.assertEqual(sigmoid_improved_ece, 8)
        self.assertEqual(isotonic_cases, 10)
        self.assertEqual(isotonic_reduced_auroc, 10)

        text = (ROOT / "access.tex").read_text(encoding="utf-8")
        self.assertIn("Sigmoid calibration preserves AUROC in all ten baseline cases and improves ECE in eight", text)
        self.assertIn("isotonic calibration reduces AUROC in all ten cases", text)

    def test_mlp_only_ablation_and_manuscript_use_missingness_indicator_language(self) -> None:
        payload = tomllib.loads(ABLATION_CONFIG.read_text(encoding="utf-8"))
        ablations = {entry["name"]: entry for entry in payload["ablation"]}
        mlp_only = ablations["mlp_only"]

        self.assertFalse(bool(mlp_only["use_missingness_indicators"]))
        self.assertFalse(bool(mlp_only["use_stochastic_masking"]))
        self.assertFalse(bool(mlp_only["enable_reconstruction"]))
        self.assertEqual(list(mlp_only["mask_rates"]), [0.0])

        text = (ROOT / "access.tex").read_text(encoding="utf-8")
        self.assertNotIn("artificial-mask indicators", text)
        self.assertIn("stochastic masking, reconstruction, and missingness indicators", text)

    def test_manuscript_uses_post_hoc_calibration_terminology(self) -> None:
        text = (ROOT / "access.tex").read_text(encoding="utf-8")
        self.assertNotIn("calibration-head ablation", text)
        self.assertNotIn("sigmoid calibration head", text)
        self.assertIn("post-hoc sigmoid calibrator", text)
        self.assertIn("legacy ablation key \\texttt{enable\\_calibration\\_head}", text)

    def test_run_all_studies_dry_run_covers_manuscript_regeneration_steps(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_all_studies.py"), "--include-extras", "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = completed.stdout
        expected_fragments = [
            "run_lambda_sweep.py --study-config configs/studies/lineage_d9_06_adult_missingness_robustness.toml",
            "run_lambda_sweep.py --study-config configs/studies/lineage_d9_10_covertype_missingness_robustness.toml",
            "aggregate_mar_results.py --study-glob configs/studies/*missingness_robustness*.toml",
            "aggregate_robustness_advantages.py --study-glob configs/studies/*missingness_robustness*.toml",
            "aggregate_submission_results.py --study-glob configs/studies/*missingness_robustness*.toml",
            "make_composite_figures.py",
        ]
        positions = [output.index(fragment) for fragment in expected_fragments]
        self.assertEqual(positions, sorted(positions))

    def test_confidence_band_script_no_longer_targets_canonical_manuscript_pdfs(self) -> None:
        text = (ROOT / "scripts" / "make_confidence_band_figures.py").read_text(encoding="utf-8")
        self.assertIn("degradation_curves_confidence_band_absolute.pdf", text)
        self.assertIn("degradation_curves_confidence_band_delta.pdf", text)
        self.assertNotIn('os.path.join(FIGURES_DIR, "degradation_curves_absolute.pdf")', text)
        self.assertNotIn('os.path.join(FIGURES_DIR, "degradation_curves_delta.pdf")', text)

    def test_manuscript_dataset_regime_matches_current_gmsc_artifacts(self) -> None:
        text = (ROOT / "access.tex").read_text(encoding="utf-8")
        metadata = json.loads(
            (
                ROOT
                / "results"
                / "lineage_d9_09_give_me_some_credit_missingness_robustness"
                / "raw"
                / "methods"
                / "mask_augmented_imputation_training__seed_7"
                / "dataset_metadata.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["categorical_columns"], [])
        self.assertIn("remaining four benchmarks", text)
        self.assertIn("purely numerical (Covertype and Give Me Some Credit)", text)
        self.assertIn("Give Me Some Credit, approximately 50\\%", text)
        self.assertNotIn("few categorical columns", text)
        self.assertNotIn("6.7\\% positive", text)

    def test_manuscript_quoted_numbers_match_current_artifact_summaries(self) -> None:
        text = (ROOT / "access.tex").read_text(encoding="utf-8")

        def read_json(path: Path) -> object:
            return json.loads(path.read_text(encoding="utf-8"))

        performance = {
            "adult": read_json(ROOT / "results" / "lineage_d9_06_adult_missingness_robustness" / "aggregated" / "performance_summary.json"),
            "credit-g": read_json(ROOT / "results" / "lineage_d9_07_credit_g_missingness_robustness" / "aggregated" / "performance_summary.json"),
            "bank-marketing": read_json(ROOT / "results" / "lineage_d9_08_bank_marketing_missingness_robustness" / "aggregated" / "performance_summary.json"),
            "give-me-some-credit": read_json(ROOT / "results" / "lineage_d9_09_give_me_some_credit_missingness_robustness" / "aggregated" / "performance_summary.json"),
            "covertype": read_json(ROOT / "results" / "lineage_d9_10_covertype_missingness_robustness" / "aggregated" / "performance_summary.json"),
        }
        mlp_summary = {
            row["dataset_label"]: row
            for row in read_json(ROOT / "paper" / "submission_summary" / "mlp_control_summary.json")
        }
        robustness_rows = {
            (row["dataset_label"], row["comparator_label"]): row
            for row in read_json(ROOT / "paper" / "submission_summary" / "robustness_advantages.json")
        }
        mar_rows = {
            row["dataset_label"]: row
            for row in read_json(ROOT / "paper" / "submission_summary" / "mar_summary.json")
        }
        method_calibration = {
            "Adult": read_json(ROOT / "results" / "lineage_d9_06_adult_missingness_robustness" / "aggregated" / "method_calibration_summary.json"),
            "German Credit": read_json(ROOT / "results" / "lineage_d9_07_credit_g_missingness_robustness" / "aggregated" / "method_calibration_summary.json"),
            "Bank Marketing": read_json(ROOT / "results" / "lineage_d9_08_bank_marketing_missingness_robustness" / "aggregated" / "method_calibration_summary.json"),
            "Give Me Some Credit": read_json(ROOT / "results" / "lineage_d9_09_give_me_some_credit_missingness_robustness" / "aggregated" / "method_calibration_summary.json"),
            "Covertype": read_json(ROOT / "results" / "lineage_d9_10_covertype_missingness_robustness" / "aggregated" / "method_calibration_summary.json"),
        }
        adult_lambda = read_json(
            ROOT / "results" / "lineage_d9_06_adult_missingness_robustness" / "lambda_sweep" / "lambda_sweep.json"
        )["results"]
        covertype_lambda = read_json(
            ROOT / "results" / "lineage_d9_10_covertype_missingness_robustness" / "lambda_sweep" / "lambda_sweep.json"
        )["results"]

        adult_perf = performance["adult"]
        german_perf = performance["credit-g"]
        bank_perf = performance["bank-marketing"]
        gmsc_perf = performance["give-me-some-credit"]
        covertype_perf = performance["covertype"]
        method_name = "mask_augmented_imputation_training"

        self.assertIn(f"LightGBM leads Adult ({adult_perf['baseline_summary']['lightgbm']['mean_auroc']:.4f})", text)
        self.assertIn(f"Random Forest leads German Credit ({german_perf['baseline_summary']['random_forest']['mean_auroc']:.4f})", text)
        self.assertIn(f"CatBoost leads both Bank Marketing ({bank_perf['baseline_summary']['catboost']['mean_auroc']:.4f}) and Give Me Some Credit ({gmsc_perf['baseline_summary']['catboost']['mean_auroc']:.4f})", text)
        self.assertIn(f"MAIT achieves {covertype_perf['method_summary'][method_name]['mean_auroc']:.4f}", text)
        self.assertIn(f"Random Forest ({covertype_perf['baseline_summary']['random_forest']['mean_auroc']:.4f})", text)
        self.assertIn(f"LightGBM ({covertype_perf['baseline_summary']['lightgbm']['mean_auroc']:.4f})", text)
        self.assertIn(
            f"({german_perf['method_summary'][method_name]['mean_auroc']:.4f} versus "
            f"{german_perf['baseline_summary']['lightgbm']['mean_auroc']:.4f} and "
            f"{german_perf['baseline_summary']['xgboost']['mean_auroc']:.4f})",
            text,
        )
        self.assertIn(
            f"trails LightGBM by "
            f"{adult_perf['baseline_summary']['lightgbm']['mean_auroc'] - adult_perf['method_summary'][method_name]['mean_auroc']:.4f} on Adult",
            text,
        )
        self.assertIn(
            f"trails CatBoost by "
            f"{bank_perf['baseline_summary']['catboost']['mean_auroc'] - bank_perf['method_summary'][method_name]['mean_auroc']:.4f} on Bank Marketing",
            text,
        )
        self.assertIn(
            f"trails CatBoost by "
            f"{gmsc_perf['baseline_summary']['catboost']['mean_auroc'] - gmsc_perf['method_summary'][method_name]['mean_auroc']:.4f} on Give Me Some Credit",
            text,
        )
        self.assertIn(
            f"MAIT achieves the lowest ECE ({covertype_perf['method_summary'][method_name]['mean_ece']:.4f})",
            text,
        )

        adult_mlp = mlp_summary["Adult"]
        german_mlp = mlp_summary["German Credit"]
        bank_mlp = mlp_summary["Bank Marketing"]
        covertype_mlp = mlp_summary["Covertype"]
        self.assertIn(f"MLP-only {covertype_mlp['mlp_only_nominal_mean_auroc']:.4f}", text)
        self.assertIn(f"MAIT {covertype_mlp['mait_nominal_mean_auroc']:.4f}", text)
        self.assertIn(f"MAIT {covertype_mlp['mait_mcar30_mean_auroc']:.4f}", text)
        self.assertIn(f"MLP-only {covertype_mlp['mlp_only_mcar30_mean_auroc']:.4f}", text)
        self.assertIn(
            f"from {abs(adult_mlp['mlp_only_mcar30_mean_delta']):.4f} to {abs(adult_mlp['mait_mcar30_mean_delta']):.4f} on Adult",
            text,
        )
        self.assertIn(
            f"from {abs(german_mlp['mlp_only_mcar30_mean_delta']):.4f} to {abs(german_mlp['mait_mcar30_mean_delta']):.4f} on German Credit",
            text,
        )
        self.assertIn(
            f"from {abs(bank_mlp['mlp_only_mcar30_mean_delta']):.4f} to {abs(bank_mlp['mait_mcar30_mean_delta']):.4f} on Bank Marketing",
            text,
        )

        for dataset_label, payload in method_calibration.items():
            delta = abs(float(payload["variants"]["enable_calibration_head"]["mean_test_ece_delta"]))
            self.assertIn(f"{delta:.4f} on {dataset_label}", text)

        max_nominal_indicator_delta = max(
            abs(
                float(summary["ablation_summary"]["remove_missingness_indicators"]["mean_auroc"])
                - float(summary["method_summary"][method_name]["mean_auroc"])
            )
            for summary in performance.values()
        )
        max_overlay_indicator_delta = max(
            abs(
                float(summary["robustness_summary"]["remove_missingness_indicators"][slice_name]["mean_auroc"])
                - float(summary["robustness_summary"][method_name][slice_name]["mean_auroc"])
            )
            for summary in performance.values()
            for slice_name in ("missingness_10", "missingness_20", "missingness_30")
        )
        self.assertIn(f"at most {max_nominal_indicator_delta:.4f}", text)
        self.assertIn(f"at most {max_overlay_indicator_delta:.4f}", text)

        adult_lambda_by_weight = {float(row["reconstruction_weight"]): row for row in adult_lambda}
        covertype_lambda_by_weight = {float(row["reconstruction_weight"]): row for row in covertype_lambda}
        adult_0 = adult_lambda_by_weight[0.0]["robustness"]["missingness_30"]["mean_auroc"]
        adult_1 = adult_lambda_by_weight[1.0]["robustness"]["missingness_30"]["mean_auroc"]
        covertype_0 = covertype_lambda_by_weight[0.0]["robustness"]["missingness_30"]["mean_auroc"]
        covertype_1 = covertype_lambda_by_weight[1.0]["robustness"]["missingness_30"]["mean_auroc"]
        displayed_covertype_gap = abs(float(f"{covertype_0:.4f}") - float(f"{covertype_1:.4f}"))
        self.assertIn(f"{covertype_0:.4f}", text)
        self.assertIn(f"{covertype_1:.4f}", text)
        self.assertIn(f"{displayed_covertype_gap:.4f}", text)
        self.assertIn(f"{adult_0:.4f}", text)
        self.assertIn(f"{adult_1:.4f}", text)

        self.assertIn(
            f"{robustness_rows[('Covertype', 'LightGBM')]['mean_advantage']:.4f} over LightGBM",
            text,
        )
        self.assertIn(
            f"{robustness_rows[('Covertype', 'XGBoost')]['mean_advantage']:.4f} over XGBoost",
            text,
        )
        self.assertIn(
            f"{robustness_rows[('Covertype', 'CatBoost')]['mean_advantage']:.4f} over CatBoost",
            text,
        )
        self.assertIn(
            f"{robustness_rows[('Covertype', 'Random Forest')]['mean_advantage']:.4f} over Random Forest",
            text,
        )
        self.assertIn(
            f"XGBoost ({robustness_rows[('German Credit', 'XGBoost')]['mean_advantage']:.4f})",
            text,
        )
        self.assertIn(
            f"CatBoost ({robustness_rows[('German Credit', 'CatBoost')]['mean_advantage']:.4f})",
            text,
        )
        self.assertIn(
            f"Random Forest ({robustness_rows[('German Credit', 'Random Forest')]['mean_advantage']:.4f})",
            text,
        )
        self.assertIn(
            f"LightGBM ({robustness_rows[('German Credit', 'LightGBM')]['mean_advantage']:.4f})",
            text,
        )
        self.assertIn(
            f"largest advantage is {max(float(row['mean_advantage']) for key, row in robustness_rows.items() if key[0] == 'Adult'):.4f}",
            text,
        )
        self.assertIn(
            f"{max(float(row['mean_advantage']) for key, row in robustness_rows.items() if key[0] == 'Bank Marketing'):.4f}",
            text,
        )
        self.assertIn(
            f"{max(float(row['mean_advantage']) for key, row in robustness_rows.items() if key[0] == 'Give Me Some Credit'):.4f}",
            text,
        )

        self.assertIn("0.05/0.35", text)
        for row in mar_rows.values():
            self.assertIn(str(row["driver_column"]), text)
            for column_name in row["target_columns"]:
                self.assertIn(str(column_name).replace("_", "\\_"), text)


if __name__ == "__main__":
    unittest.main()
