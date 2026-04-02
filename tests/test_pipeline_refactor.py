from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from lab import pipeline
from lab.pipeline_audit import _audit_markdown
from lab.pipeline_outputs import (
    BASELINE_ARTIFACT_FILES,
    _main_results_markdown,
    _multi_line_chart_svg,
    _read_required_metrics,
    _select_calibration_bases,
    _write_main_summary_csv,
    _write_run_artifacts,
)
from lab.reporting import build_prediction_frame
from lab.study import StudySpec


class PipelineRefactorTests(unittest.TestCase):
    def test_build_prediction_frame_preserves_layout(self) -> None:
        target = pd.Series([1, 0], index=pd.Index([8, 3]))
        frame = build_prediction_frame(
            "lightgbm",
            7,
            "test",
            target,
            [0.2, 0.8],
            extra_columns={"evaluation_slice": ["missingness_10", "missingness_10"]},
        )

        expected = pd.DataFrame(
            {
                "baseline_name": ["lightgbm", "lightgbm"],
                "seed": [7, 7],
                "split": ["test", "test"],
                "row_id": [8, 3],
                "target": [1, 0],
                "predicted_probability": [0.2, 0.8],
                "evaluation_slice": ["missingness_10", "missingness_10"],
            }
        )
        pd.testing.assert_frame_equal(frame, expected)

    def test_select_calibration_bases_keeps_metric_ranking_and_tie_breaks(self) -> None:
        records = [
            {
                "model_name": "xgboost",
                "validation_metrics": {"auroc": 0.93, "ece": 0.03},
                "test_metrics": {"auroc": 0.91, "ece": 0.04},
            },
            {
                "model_name": "lightgbm",
                "validation_metrics": {"auroc": 0.93, "ece": 0.02},
                "test_metrics": {"auroc": 0.92, "ece": 0.05},
            },
            {
                "model_name": "catboost",
                "validation_metrics": {"auroc": 0.90, "ece": 0.01},
                "test_metrics": {"auroc": 0.90, "ece": 0.03},
            },
        ]
        calibration_config = {
            "selection_metric": "validation_auroc",
            "tie_break": "validation_ece",
            "top_k": 2,
        }

        self.assertEqual(_select_calibration_bases(records, calibration_config), ["lightgbm", "xgboost"])

    def test_write_run_artifacts_preserves_required_files_and_sorted_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_directory = Path(tmp) / "lightgbm__seed_7"
            predictions = pd.DataFrame(
                {
                    "baseline_name": ["lightgbm", "lightgbm", "lightgbm"],
                    "seed": [7, 7, 7],
                    "split": ["validation", "test", "validation"],
                    "row_id": [4, 2, 1],
                    "target": [0, 1, 1],
                    "predicted_probability": [0.7, 0.8, 0.6],
                }
            )
            metrics_payload = {
                "study_id": "demo",
                "artifact_status": "draft",
                "stage": "run_baselines",
                "result_kind": "nominal_baseline",
                "model_name": "lightgbm",
                "baseline_name": "lightgbm",
                "baseline_family": "tree",
                "seed": 7,
                "row_counts": {"train": 10, "validation": 2, "test": 1},
                "validation_metrics": {"auroc": 0.9},
                "test_metrics": {"auroc": 0.8},
                "fit_seconds": 1.2,
                "predict_seconds": 0.3,
                "model_metadata": {"implementation": "demo"},
                "software_versions": {"python": "3.12"},
            }

            artifact_paths = _write_run_artifacts(
                run_directory,
                metrics_payload,
                predictions,
                {"seed": 7, "train_size": 10, "validation_size": 2, "test_size": 1},
                {"dataset_name": "demo"},
                {"stage_manifest": "results/manifests/run_baselines_resolved_config.toml"},
            )

            self.assertEqual(
                [path.name for path in artifact_paths],
                [
                    "metrics.json",
                    "predictions.csv.gz",
                    "split_metadata.json",
                    "dataset_metadata.json",
                    "manifest_reference.json",
                ],
            )
            self.assertEqual(
                _read_required_metrics(run_directory, BASELINE_ARTIFACT_FILES, expected_kind="nominal_baseline"),
                metrics_payload,
            )

            written_predictions = pd.read_csv(run_directory / "predictions.csv.gz")
            self.assertEqual(written_predictions["split"].tolist(), ["test", "validation", "validation"])
            self.assertEqual(written_predictions["row_id"].tolist(), [2, 1, 4])

    def test_write_main_summary_csv_preserves_field_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "performance_summary.csv"
            _write_main_summary_csv(
                output_path,
                baseline_summary={
                    "lightgbm": {
                        "n_runs": 2.0,
                        "mean_auroc": 0.91,
                        "std_auroc": 0.01,
                        "mean_brier": 0.08,
                        "std_brier": 0.002,
                        "mean_log_loss": 0.30,
                        "std_log_loss": 0.004,
                        "mean_ece": 0.02,
                        "std_ece": 0.001,
                    }
                },
                method_summary={},
                ablation_summary={},
            )

            header = output_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header,
                "kind,name,n_runs,mean_auroc,std_auroc,mean_brier,std_brier,mean_log_loss,std_log_loss,mean_ece,std_ece",
            )

    def test_main_results_markdown_is_byte_stable_for_fixed_fixture(self) -> None:
        summary = {
            "baseline_summary": {
                "lightgbm": {
                    "mean_auroc": 0.91,
                    "std_auroc": 0.01,
                    "mean_brier": 0.08,
                    "mean_log_loss": 0.30,
                    "mean_ece": 0.02,
                    "std_ece": 0.001,
                    "n_runs": 2.0,
                }
            },
            "method_summary": {
                "mait": {
                    "mean_auroc": 0.905,
                    "std_auroc": 0.02,
                    "mean_brier": 0.085,
                    "mean_log_loss": 0.31,
                    "mean_ece": 0.021,
                    "std_ece": 0.002,
                    "n_runs": 2.0,
                }
            },
            "ablation_summary": {
                "mlp_only": {
                    "mean_auroc": 0.89,
                    "std_auroc": 0.03,
                    "mean_brier": 0.09,
                    "mean_log_loss": 0.33,
                    "mean_ece": 0.025,
                    "std_ece": 0.003,
                    "n_runs": 2.0,
                }
            },
        }

        expected = (
            "# Main Results\n"
            "\n"
            "Model results aggregated from raw per-seed artifacts.\n"
            "\n"
            "| kind | name | mean_auroc | std_auroc | mean_brier | mean_log_loss | mean_ece |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| baseline | lightgbm | 0.910000 | 0.010000 | 0.080000 | 0.300000 | 0.020000 |\n"
            "| method | mait | 0.905000 | 0.020000 | 0.085000 | 0.310000 | 0.021000 |\n"
            "| ablation | mlp_only | 0.890000 | 0.030000 | 0.090000 | 0.330000 | 0.025000 |\n"
        )
        self.assertEqual(_main_results_markdown(summary), expected)

    def test_multi_line_chart_svg_is_byte_stable_for_fixed_fixture(self) -> None:
        series = [{"label": "lightgbm", "points": [{"x": 0.0, "y": 0.91}, {"x": 10.0, "y": 0.9}]}]
        expected = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="440" viewBox="0 0 840 440">\n'
            '<rect width="100%" height="100%" fill="#fbfaf7" />\n'
            '<text x="80" y="28" font-size="20" font-weight="bold">AUROC vs Overlay Severity</text>\n'
            '<line x1="80" y1="50" x2="80" y2="380" stroke="#444" stroke-width="1.5"/>\n'
            '<line x1="80" y1="380" x2="810" y2="380" stroke="#444" stroke-width="1.5"/>\n'
            '<line x1="80.0" y1="380" x2="80.0" y2="386" stroke="#444" stroke-width="1"/>\n'
            '<text x="70.0" y="404" font-size="12">0%</text>\n'
            '<line x1="810.0" y1="380" x2="810.0" y2="386" stroke="#444" stroke-width="1"/>\n'
            '<text x="800.0" y="404" font-size="12">10%</text>\n'
            '<line x1="1540.0" y1="380" x2="1540.0" y2="386" stroke="#444" stroke-width="1"/>\n'
            '<text x="1530.0" y="404" font-size="12">20%</text>\n'
            '<line x1="2270.0" y1="380" x2="2270.0" y2="386" stroke="#444" stroke-width="1"/>\n'
            '<text x="2260.0" y="404" font-size="12">30%</text>\n'
            '<line x1="74" y1="380.0" x2="80" y2="380.0" stroke="#444" stroke-width="1"/>\n'
            '<text x="12" y="384.0" font-size="12">0.900</text>\n'
            '<line x1="74" y1="297.4999999999982" x2="80" y2="297.4999999999982" stroke="#444" stroke-width="1"/>\n'
            '<text x="12" y="301.4999999999982" font-size="12">0.903</text>\n'
            '<line x1="74" y1="215.0" x2="80" y2="215.0" stroke="#444" stroke-width="1"/>\n'
            '<text x="12" y="219.0" font-size="12">0.905</text>\n'
            '<line x1="74" y1="132.50000000000182" x2="80" y2="132.50000000000182" stroke="#444" stroke-width="1"/>\n'
            '<text x="12" y="136.50000000000182" font-size="12">0.907</text>\n'
            '<line x1="74" y1="50.0" x2="80" y2="50.0" stroke="#444" stroke-width="1"/>\n'
            '<text x="12" y="54.0" font-size="12">0.910</text>\n'
            '<polyline fill="none" stroke="#2f6690" stroke-width="2.5" points="80.00,50.00 810.00,380.00"/>\n'
            '<circle cx="80.00" cy="50.00" r="3.2" fill="#2f6690"/>\n'
            '<circle cx="810.00" cy="380.00" r="3.2" fill="#2f6690"/>\n'
            '<line x1="820" y1="50" x2="836" y2="50" stroke="#2f6690" stroke-width="3"/>\n'
            '<text x="842" y="54" font-size="12">lightgbm</text>\n'
            '<text x="375.0" y="424" font-size="13">Overlay severity</text>\n'
            "</svg>\n"
        )
        self.assertEqual(_multi_line_chart_svg("AUROC vs Overlay Severity", series), expected)

    def test_audit_markdown_is_byte_stable_for_fixed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._make_spec(Path(tmp), active_stages=["audit_results"])
            summary = {
                "baseline_summary": {
                    "lightgbm": {
                        "mean_auroc": 0.91,
                        "std_auroc": 0.01,
                        "mean_ece": 0.02,
                        "n_runs": 1.0,
                    }
                },
                "selected_calibration_bases": [],
                "calibration_summary": {},
                "robustness_summary": {},
                "method_summary": {},
                "ablation_summary": {},
            }
            nominal_records = [{"seed": 7, "model_name": "lightgbm", "test_metrics": {"auroc": 0.91}}]
            expected = (
                "# Results Audit\n"
                "\n"
                "Artifact status: `draft`. This audit covers nominal baselines generated under the frozen Adult protocol.\n"
                "\n"
                "## Variance Across Seeds\n"
                "- `lightgbm` nominal mean test AUROC is 0.910000 with seed spread 0.010000 across 1 runs.\n"
                "- The top nominal baseline is stable across seeds: `lightgbm` leads all three splits.\n"
                "\n"
                "## Regime-Specific Robustness Assessment\n"
                "- No robustness overlays were audited, so robustness evidence is incomplete.\n"
                "\n"
                "## Calibration Assessment\n"
                "- No calibrated variants are available, so calibration evidence remains nominal-only.\n"
                "\n"
                "## Loss-Case Summary\n"
                "- No calibrated loss cases are available because no calibrated variants were aggregated.\n"
                "\n"
                "## Suspicious-Gain Warnings\n"
                "- The study still covers one dataset only, so any cross-dataset generalization claim would be unverified.\n"
                "- No proposed-method comparison is audited in this phase, so no superiority claim is supported.\n"
                "- Calibration gains in this audit strengthen the baseline table only and do not justify a method claim by themselves.\n"
                "- Robustness artifacts are missing, so robustness-driven motivation for a method would be unverified.\n"
                "\n"
                "## Exact Failure Regime to Motivate a Method\n"
                "- No calibrated models were aggregated, so no exact failure regime can be named.\n"
                "\n"
                "## Remaining Gaps to Claim-Ready Evidence\n"
                "- No proposed method artifacts exist yet.\n"
                "- No ablation artifacts exist yet.\n"
                "- No multi-dataset evidence exists yet.\n"
                "- Robustness evidence is limited to the scripted Adult missingness overlay grid.\n"
                "- `claim_evidence_table.md` has not been updated in this phase.\n"
            )
            self.assertEqual(_audit_markdown(spec, summary, nominal_records, [], []), expected)

    def test_run_stage_preserves_dependency_order_and_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._make_spec(
                Path(tmp),
                active_stages=[
                    "run_baselines",
                    "run_method",
                    "run_ablations",
                    "evaluate_robustness",
                    "evaluate_calibration",
                    "aggregate_results",
                ],
            )
            handler = mock.Mock()
            with (
                mock.patch.object(pipeline, "ensure_layout"),
                mock.patch.object(pipeline, "ensure_required_docs"),
                mock.patch.object(pipeline, "reset_stage_outputs"),
                mock.patch.object(pipeline, "write_stage_manifest"),
                mock.patch.object(pipeline, "write_seed_manifest"),
                mock.patch.object(pipeline, "stage_complete", return_value=True) as stage_complete,
                mock.patch.dict(pipeline.STAGE_HANDLERS, {"aggregate_results": handler}, clear=False),
            ):
                pipeline.run_stage(spec, "aggregate_results")

            self.assertEqual(
                [call.args for call in stage_complete.call_args_list],
                [
                    (spec, "run_baselines"),
                    (spec, "run_method"),
                    (spec, "run_ablations"),
                    (spec, "evaluate_robustness"),
                    (spec, "evaluate_calibration"),
                ],
            )
            handler.assert_called_once_with(spec, "aggregate_results")

    def test_run_stage_fails_on_first_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._make_spec(
                Path(tmp),
                active_stages=["run_baselines", "run_method", "aggregate_results"],
            )
            handler = mock.Mock()
            with (
                mock.patch.object(pipeline, "ensure_layout"),
                mock.patch.object(pipeline, "ensure_required_docs"),
                mock.patch.object(pipeline, "reset_stage_outputs"),
                mock.patch.object(pipeline, "write_stage_manifest"),
                mock.patch.object(pipeline, "write_seed_manifest"),
                mock.patch.object(pipeline, "stage_complete", side_effect=[True, False]) as stage_complete,
                mock.patch.dict(pipeline.STAGE_HANDLERS, {"aggregate_results": handler}, clear=False),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Stage 'aggregate_results' requires completed stage 'run_method'.",
                ):
                    pipeline.run_stage(spec, "aggregate_results")

            self.assertEqual(
                [call.args for call in stage_complete.call_args_list],
                [(spec, "run_baselines"), (spec, "run_method")],
            )
            handler.assert_not_called()

    def _make_spec(self, root: Path, *, active_stages: list[str]) -> StudySpec:
        return StudySpec(
            root=root,
            study_config_path=root / "configs" / "study.toml",
            study={"id": "demo", "task_family": "classification", "task_type": "binary", "status": "draft"},
            execution={"active_stages": active_stages, "primary_metric": "auroc"},
            config_refs={},
            configs={},
            research_dir=root / "research",
            results_dir=root / "results",
            paper_dir=root / "paper",
            seed_list=[7],
        )


if __name__ == "__main__":
    unittest.main()
