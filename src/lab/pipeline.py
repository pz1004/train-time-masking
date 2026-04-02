from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from .artifacts import (
    ensure_layout,
    ensure_required_docs,
    read_json,
    reset_stage_outputs,
    stage_complete,
    stage_manifest_path,
    stage_seed_manifest_path,
    write_completion,
    write_json,
    write_seed_manifest,
    write_stage_manifest,
    write_text,
)
from .baselines.tabular import ensure_baseline_environment, run_tabular_baseline
from .data import load_dataset, make_seed_splits
from .pipeline_audit import _audit_markdown
from .pipeline_outputs import (
    BASELINE_ARTIFACT_FILES,
    ROBUSTNESS_ARTIFACT_FILES,
    _artifacts_complete,
    _calibrated_baseline_artifacts,
    _calibrated_model_name,
    _calibration_figure_series,
    _calibration_markdown,
    _collect_calibrated_run_records,
    _collect_nominal_baseline_run_records,
    _collect_optional_run_records,
    _collect_robustness_run_records,
    _configured_robustness_model_names,
    _delta_auroc_figure_series,
    _main_results_markdown,
    _mask_sweep_heatmap_svg,
    _multi_line_chart_svg,
    _performance_figure_series,
    _read_required_metrics,
    _robustness_artifacts,
    _robustness_markdown,
    _select_calibration_bases,
    _summarize_calibration_records,
    _summarize_records_by_key,
    _summarize_robustness_records,
    _write_main_summary_csv,
    _write_run_artifacts,
)
from .study import StudySpec, load_study_spec


PIPELINE_STAGES = (
    "run_baselines",
    "run_method",
    "run_ablations",
    "evaluate_robustness",
    "evaluate_calibration",
    "aggregate_results",
    "make_tables",
    "make_figures",
    "audit_results",
)

STAGE_DEPENDENCIES = {
    "run_baselines": (),
    "run_method": ("run_baselines",),
    "run_ablations": ("run_method",),
    "evaluate_robustness": ("run_method",),
    "evaluate_calibration": ("run_method", "run_ablations"),
    "aggregate_results": (
        "run_baselines",
        "run_method",
        "run_ablations",
        "evaluate_robustness",
        "evaluate_calibration",
    ),
    "make_tables": ("aggregate_results",),
    "make_figures": ("aggregate_results",),
    "audit_results": ("aggregate_results",),
}


def run_stage_cli(stage_name: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run the {stage_name} stage for a manifest-driven study.")
    parser.add_argument("--study-config", required=True, help="Path to configs/studies/<study_id>.toml")
    args = parser.parse_args(argv)
    spec = load_study_spec(args.study_config)
    run_stage(spec, stage_name)
    return 0


def run_full_pipeline_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the active pilot workflow for a manifest-driven study.")
    parser.add_argument("--study-config", required=True, help="Path to configs/studies/<study_id>.toml")
    args = parser.parse_args(argv)
    spec = load_study_spec(args.study_config)
    for stage_name in _configured_active_stages(spec):
        run_stage(spec, stage_name)
    return 0


def run_stage(spec: StudySpec, stage_name: str) -> None:
    ensure_layout(spec)
    ensure_required_docs(spec)
    _ensure_stage_is_active(spec, stage_name)

    for dependency in _active_dependencies(spec, stage_name):
        if not stage_complete(spec, dependency):
            raise RuntimeError(f"Stage '{stage_name}' requires completed stage '{dependency}'.")

    reset_stage_outputs(spec, stage_name)
    write_stage_manifest(spec, stage_name)
    write_seed_manifest(spec, stage_name)
    STAGE_HANDLERS[stage_name](spec, stage_name)


def _configured_active_stages(spec: StudySpec) -> list[str]:
    return spec.active_stages or list(PIPELINE_STAGES)


def _ensure_stage_is_active(spec: StudySpec, stage_name: str) -> None:
    if stage_name not in _configured_active_stages(spec):
        raise RuntimeError(f"Stage '{stage_name}' is not active for study '{spec.study_id}'.")


def _active_dependencies(spec: StudySpec, stage_name: str) -> tuple[str, ...]:
    active_stages = set(_configured_active_stages(spec))
    return tuple(dependency for dependency in STAGE_DEPENDENCIES.get(stage_name, ()) if dependency in active_stages)


def _run_baselines(spec: StudySpec, stage_name: str) -> None:
    artifact_paths = _stage_artifact_paths(spec, stage_name)
    baseline_configs = list(spec.configs["baselines"]["baseline"])
    ensure_baseline_environment(baseline_configs)

    baseline_root = spec.raw_dir / "baselines"
    robustness_root = spec.raw_dir / "robustness"
    baseline_root.mkdir(parents=True, exist_ok=True)
    robustness_root.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = _split_map_from_protocol(spec, dataset_bundle)
    manifest_reference = _manifest_reference(spec, stage_name)
    robustness_slices = list(spec.configs["robustness"]["slice"])

    nominal_records: list[dict[str, Any]] = []
    for baseline_config in baseline_configs:
        nominal_records.extend(
            _materialize_nominal_baseline_runs(
                spec,
                stage_name,
                baseline_config,
                dataset_bundle=dataset_bundle,
                split_map=split_map,
                manifest_reference=manifest_reference,
                robustness_slices=robustness_slices,
                baseline_root=baseline_root,
                robustness_root=robustness_root,
                artifact_paths=artifact_paths,
            )
        )

    calibration_config = spec.configs["baselines"]["calibration"]
    selected_calibration_bases = _select_calibration_bases(nominal_records, calibration_config)
    baseline_config_map = {str(config["name"]): config for config in baseline_configs}
    _materialize_calibrated_baseline_runs(
        spec,
        stage_name,
        dataset_bundle=dataset_bundle,
        split_map=split_map,
        manifest_reference=manifest_reference,
        robustness_slices=robustness_slices,
        baseline_root=baseline_root,
        robustness_root=robustness_root,
        calibration_config=calibration_config,
        selected_calibration_bases=selected_calibration_bases,
        baseline_config_map=baseline_config_map,
        artifact_paths=artifact_paths,
    )

    write_completion(
        spec,
        stage_name,
        artifact_paths,
        "Generated baseline artifacts, calibrated variants, and robustness overlays.",
    )


def _run_method(spec: StudySpec, stage_name: str) -> None:
    _run_custom_method_stage(spec, stage_name, function_name="run_method_stage")


def _run_ablations(spec: StudySpec, stage_name: str) -> None:
    _run_custom_method_stage(spec, stage_name, function_name="run_ablations_stage")


def _evaluate_robustness(spec: StudySpec, stage_name: str) -> None:
    _run_custom_method_stage(spec, stage_name, function_name="evaluate_robustness_stage")


def _evaluate_calibration(spec: StudySpec, stage_name: str) -> None:
    _run_custom_method_stage(spec, stage_name, function_name="evaluate_calibration_stage")


def _aggregate_results(spec: StudySpec, stage_name: str) -> None:
    artifact_paths = _stage_artifact_paths(spec, stage_name)
    nominal_records = _collect_nominal_baseline_run_records(spec)
    calibration_config = spec.configs["baselines"]["calibration"]
    selected_calibration_bases = _select_calibration_bases(nominal_records, calibration_config)
    calibrated_records = _collect_calibrated_run_records(spec, selected_calibration_bases, calibration_config)
    method_records = _collect_optional_run_records(spec.raw_dir / "methods", BASELINE_ARTIFACT_FILES, expected_kind="method")
    ablation_records = _collect_optional_run_records(spec.raw_dir / "ablations", BASELINE_ARTIFACT_FILES, expected_kind="ablation")
    robustness_model_names = _configured_robustness_model_names(
        spec,
        selected_calibration_bases,
        calibration_config,
        method_records,
        ablation_records,
    )
    robustness_records = _collect_robustness_run_records(spec, robustness_model_names)

    baseline_summary = _summarize_records_by_key(nominal_records, "model_name", "test_metrics")
    calibration_summary = _summarize_calibration_records(calibrated_records)
    method_summary = _summarize_records_by_key(method_records, "model_name", "test_metrics") if method_records else {}
    ablation_summary = _summarize_records_by_key(ablation_records, "model_name", "test_metrics") if ablation_records else {}
    robustness_summary = _summarize_robustness_records(
        robustness_records,
        nominal_records,
        calibrated_records,
        method_records,
        ablation_records,
    )

    performance_summary = {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "active_stages": _configured_active_stages(spec),
        "primary_metric": spec.execution["primary_metric"],
        "baseline_summary": baseline_summary,
        "selected_calibration_bases": selected_calibration_bases,
        "calibration_summary": calibration_summary,
        "robustness_summary": robustness_summary,
        "method_summary": method_summary,
        "ablation_summary": ablation_summary,
    }

    summary_path = spec.aggregated_dir / "performance_summary.json"
    csv_path = spec.aggregated_dir / "performance_summary.csv"
    write_json(summary_path, performance_summary)
    _write_main_summary_csv(csv_path, baseline_summary, method_summary, ablation_summary)
    artifact_paths.extend([summary_path, csv_path])
    write_completion(
        spec,
        stage_name,
        artifact_paths,
        "Aggregated baseline, method, ablation, calibration, and robustness metrics.",
    )


def _make_tables(spec: StudySpec, stage_name: str) -> None:
    artifact_paths = _stage_artifact_paths(spec, stage_name)
    reporting = spec.configs["reporting"]["reporting"]
    performance_summary = read_json(spec.aggregated_dir / "performance_summary.json")

    main_md = spec.tables_dir / f"{reporting['primary_table']}.md"
    main_csv = spec.tables_dir / f"{reporting['primary_table']}.csv"
    robustness_md = spec.tables_dir / f"{reporting['robustness_table']}.md"
    calibration_md = spec.tables_dir / f"{reporting['calibration_table']}.md"

    write_text(main_md, _main_results_markdown(performance_summary))
    _write_main_summary_csv(
        main_csv,
        performance_summary["baseline_summary"],
        performance_summary["method_summary"],
        performance_summary["ablation_summary"],
    )
    write_text(robustness_md, _robustness_markdown(performance_summary["robustness_summary"]))
    write_text(calibration_md, _calibration_markdown(performance_summary))
    artifact_paths.extend([main_md, main_csv, robustness_md, calibration_md])
    write_completion(spec, stage_name, artifact_paths, "Built tables from aggregated artifacts.")


def _make_figures(spec: StudySpec, stage_name: str) -> None:
    artifact_paths = _stage_artifact_paths(spec, stage_name)
    reporting = spec.configs["reporting"]["reporting"]
    performance_summary = read_json(spec.aggregated_dir / "performance_summary.json")

    performance_path = spec.figures_dir / f"{reporting['performance_figure']}.svg"
    calibration_path = spec.figures_dir / f"{reporting['calibration_figure']}.svg"
    robustness_path = spec.figures_dir / f"{reporting['robustness_figure']}.svg"

    write_text(performance_path, _multi_line_chart_svg("AUROC vs Overlay Severity", _performance_figure_series(performance_summary)))
    write_text(calibration_path, _multi_line_chart_svg("ECE vs Overlay Severity", _calibration_figure_series(performance_summary)))
    write_text(robustness_path, _multi_line_chart_svg("Delta AUROC vs Overlay Severity", _delta_auroc_figure_series(performance_summary)))
    artifact_paths.extend([performance_path, calibration_path, robustness_path])

    mask_sweep_path = spec.results_dir / "mask_sweep" / "mask_sweep.json"
    if mask_sweep_path.exists():
        mask_sweep_figure_path = spec.figures_dir / "mask_sweep_heatmap.svg"
        write_text(mask_sweep_figure_path, _mask_sweep_heatmap_svg(read_json(mask_sweep_path)))
        artifact_paths.append(mask_sweep_figure_path)
    write_completion(spec, stage_name, artifact_paths, "Built nominal and robustness line plots plus the optional mask-sweep heatmap.")


def _audit_results(spec: StudySpec, stage_name: str) -> None:
    artifact_paths = _stage_artifact_paths(spec, stage_name)
    performance_summary = read_json(spec.aggregated_dir / "performance_summary.json")
    nominal_records = _collect_nominal_baseline_run_records(spec)
    calibration_config = spec.configs["baselines"]["calibration"]
    calibrated_records = _collect_calibrated_run_records(
        spec,
        performance_summary["selected_calibration_bases"],
        calibration_config,
    )
    method_records = _collect_optional_run_records(spec.raw_dir / "methods", BASELINE_ARTIFACT_FILES, expected_kind="method")
    ablation_records = _collect_optional_run_records(spec.raw_dir / "ablations", BASELINE_ARTIFACT_FILES, expected_kind="ablation")
    robustness_model_names = _configured_robustness_model_names(
        spec,
        performance_summary["selected_calibration_bases"],
        calibration_config,
        method_records,
        ablation_records,
    )
    robustness_records = _collect_robustness_run_records(spec, robustness_model_names)
    method_calibration_summary_path = spec.aggregated_dir / "method_calibration_summary.json"
    method_calibration_summary = read_json(method_calibration_summary_path) if method_calibration_summary_path.exists() else {}
    audit_path = spec.audits_dir / f"{spec.configs['reporting']['reporting']['audit_report']}.md"
    write_text(
        audit_path,
        _audit_markdown(
            spec,
            performance_summary,
            nominal_records,
            calibrated_records,
            robustness_records,
            method_calibration_summary=method_calibration_summary,
        ),
    )
    artifact_paths.append(audit_path)
    write_completion(spec, stage_name, artifact_paths, "Produced the study audit summary from aggregated artifacts.")


def _stage_artifact_paths(spec: StudySpec, stage_name: str) -> list[Path]:
    return [stage_manifest_path(spec, stage_name), stage_seed_manifest_path(spec, stage_name)]


def _run_custom_method_stage(spec: StudySpec, stage_name: str, *, function_name: str) -> None:
    method_config = spec.configs["method"]["method"]
    module_name = str(method_config["implementation_module"])
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            f"Stage '{stage_name}' for study '{spec.study_id}' requires implementation module '{module_name}'."
        ) from exc

    stage_function = getattr(module, function_name, None)
    if stage_function is None:
        raise NotImplementedError(
            f"Implementation module '{module_name}' does not define '{function_name}' for study '{spec.study_id}'."
        )

    result = stage_function(spec, stage_name)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Custom stage '{stage_name}' for study '{spec.study_id}' must return a dict with artifact_paths and summary."
        )
    custom_artifacts = [Path(path) for path in result.get("artifact_paths", [])]
    artifact_paths = _stage_artifact_paths(spec, stage_name) + custom_artifacts
    write_completion(spec, stage_name, artifact_paths, str(result.get("summary", f"Completed {stage_name}.")))


def _split_map_from_protocol(spec: StudySpec, dataset_bundle: Any) -> dict[int, dict[str, Any]]:
    split_config = spec.configs["protocol"]["split"]
    return make_seed_splits(
        dataset_bundle.target,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        seeds=[int(seed) for seed in split_config["seeds"]],
    )


def _manifest_reference(spec: StudySpec, stage_name: str) -> dict[str, str]:
    return {
        "stage_manifest": str(stage_manifest_path(spec, stage_name).relative_to(spec.root)),
        "seed_manifest": str(stage_seed_manifest_path(spec, stage_name).relative_to(spec.root)),
        "study_config": str(spec.study_config_path.relative_to(spec.root)),
    }


def _nominal_metrics_payload(
    spec: StudySpec,
    stage_name: str,
    baseline_config: dict[str, Any],
    split_metadata: dict[str, Any],
    run_result: Any,
) -> dict[str, Any]:
    baseline_name = str(baseline_config["name"])
    return {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "stage": stage_name,
        "result_kind": "nominal_baseline",
        "model_name": baseline_name,
        "baseline_name": baseline_name,
        "baseline_family": str(baseline_config["family"]),
        "seed": int(split_metadata["seed"]),
        "row_counts": {
            "train": int(split_metadata["train_size"]),
            "validation": int(split_metadata["validation_size"]),
            "test": int(split_metadata["test_size"]),
        },
        "validation_metrics": run_result.validation_metrics,
        "test_metrics": run_result.test_metrics,
        "fit_seconds": run_result.fit_seconds,
        "predict_seconds": run_result.predict_seconds,
        "model_metadata": run_result.model_metadata,
        "software_versions": run_result.software_versions,
    }


def _materialize_nominal_baseline_runs(
    spec: StudySpec,
    stage_name: str,
    baseline_config: dict[str, Any],
    *,
    dataset_bundle: Any,
    split_map: dict[int, dict[str, Any]],
    manifest_reference: dict[str, Any],
    robustness_slices: list[dict[str, Any]],
    baseline_root: Path,
    robustness_root: Path,
    artifact_paths: list[Path],
) -> list[dict[str, Any]]:
    baseline_name = str(baseline_config["name"])
    nominal_records: list[dict[str, Any]] = []

    for seed in spec.seed_list:
        split_metadata = split_map[int(seed)]
        nominal_directory = baseline_root / f"{baseline_name}__seed_{seed}"
        robustness_directories = _robustness_run_directories(robustness_root, baseline_name, seed, robustness_slices)
        if _nominal_robustness_complete(nominal_directory, robustness_directories):
            nominal_records.append(
                _read_required_metrics(
                    nominal_directory,
                    BASELINE_ARTIFACT_FILES,
                    expected_kind="nominal_baseline",
                )
            )
            continue

        run_result = run_tabular_baseline(baseline_config, dataset_bundle, split_metadata)
        metrics_payload = _nominal_metrics_payload(spec, stage_name, baseline_config, split_metadata, run_result)
        artifact_paths.extend(
            _write_run_artifacts(
                nominal_directory,
                metrics_payload,
                run_result.predictions,
                split_metadata,
                dataset_bundle.metadata,
                manifest_reference,
            )
        )
        nominal_records.append(metrics_payload)
        _materialize_robustness_overlays(
            spec,
            stage_name,
            model_name=baseline_name,
            split_metadata=split_metadata,
            dataset_bundle=dataset_bundle,
            predict_probabilities=run_result.predict_probabilities,
            robustness_slices=robustness_slices,
            robustness_root=robustness_root,
            manifest_reference=manifest_reference,
            model_metadata=run_result.model_metadata,
            software_versions=run_result.software_versions,
            source_result_kind="nominal_baseline",
            artifact_paths=artifact_paths,
        )

    return nominal_records


def _materialize_calibrated_baseline_runs(
    spec: StudySpec,
    stage_name: str,
    *,
    dataset_bundle: Any,
    split_map: dict[int, dict[str, Any]],
    manifest_reference: dict[str, Any],
    robustness_slices: list[dict[str, Any]],
    baseline_root: Path,
    robustness_root: Path,
    calibration_config: dict[str, Any],
    selected_calibration_bases: list[str],
    baseline_config_map: dict[str, dict[str, Any]],
    artifact_paths: list[Path],
) -> None:
    for base_baseline_name in selected_calibration_bases:
        baseline_config = baseline_config_map[base_baseline_name]
        for seed in spec.seed_list:
            split_metadata = split_map[int(seed)]
            pending_methods = _pending_calibration_methods(
                base_baseline_name,
                seed,
                robustness_slices=robustness_slices,
                robustness_root=robustness_root,
                baseline_root=baseline_root,
                calibration_config=calibration_config,
            )
            if not pending_methods:
                continue

            base_run = run_tabular_baseline(baseline_config, dataset_bundle, split_metadata)
            y_validation = dataset_bundle.target.loc[split_metadata["validation_row_ids"]].copy()
            y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()

            for calibration_method in pending_methods:
                calibrated_payload, calibrated_predictions, calibrated_predictor = _calibrated_baseline_artifacts(
                    spec,
                    stage_name,
                    base_baseline_name=base_baseline_name,
                    calibration_method=calibration_method,
                    split_metadata=split_metadata,
                    y_validation=y_validation,
                    y_test=y_test,
                    base_run=base_run,
                    fit_split=str(calibration_config["fit_split"]),
                )
                calibrated_name = str(calibrated_payload["model_name"])
                run_directory = baseline_root / f"{calibrated_name}__seed_{seed}"
                artifact_paths.extend(
                    _write_run_artifacts(
                        run_directory,
                        calibrated_payload,
                        calibrated_predictions,
                        split_metadata,
                        dataset_bundle.metadata,
                        manifest_reference,
                    )
                )
                _materialize_robustness_overlays(
                    spec,
                    stage_name,
                    model_name=calibrated_name,
                    split_metadata=split_metadata,
                    dataset_bundle=dataset_bundle,
                    predict_probabilities=calibrated_predictor,
                    robustness_slices=robustness_slices,
                    robustness_root=robustness_root,
                    manifest_reference=manifest_reference,
                    model_metadata=calibrated_payload["model_metadata"],
                    software_versions=calibrated_payload["software_versions"],
                    source_result_kind="calibrated_variant",
                    artifact_paths=artifact_paths,
                )


def _pending_calibration_methods(
    base_baseline_name: str,
    seed: int,
    *,
    robustness_slices: list[dict[str, Any]],
    robustness_root: Path,
    baseline_root: Path,
    calibration_config: dict[str, Any],
) -> list[str]:
    pending_methods: list[str] = []
    for calibration_method in calibration_config["methods"]:
        calibrated_name = _calibrated_model_name(base_baseline_name, str(calibration_method))
        calibrated_directory = baseline_root / f"{calibrated_name}__seed_{seed}"
        robustness_directories = _robustness_run_directories(robustness_root, calibrated_name, seed, robustness_slices)
        if not _calibrated_robustness_complete(calibrated_directory, robustness_directories):
            pending_methods.append(str(calibration_method))
    return pending_methods


def _materialize_robustness_overlays(
    spec: StudySpec,
    stage_name: str,
    *,
    model_name: str,
    split_metadata: dict[str, Any],
    dataset_bundle: Any,
    predict_probabilities: Any,
    robustness_slices: list[dict[str, Any]],
    robustness_root: Path,
    manifest_reference: dict[str, Any],
    model_metadata: dict[str, Any],
    software_versions: dict[str, str],
    source_result_kind: str,
    artifact_paths: list[Path],
) -> None:
    robustness_directories = _robustness_run_directories(
        robustness_root,
        model_name,
        int(split_metadata["seed"]),
        robustness_slices,
    )
    for slice_config in robustness_slices:
        robustness_payload, robustness_predictions, slice_metadata = _robustness_artifacts(
            spec,
            stage_name,
            model_name=model_name,
            split_metadata=split_metadata,
            dataset_bundle=dataset_bundle,
            predict_probabilities=predict_probabilities,
            slice_config=slice_config,
            robustness_config=spec.configs["robustness"],
            source_result_kind=source_result_kind,
            model_metadata=model_metadata,
            software_versions=software_versions,
        )
        artifact_paths.extend(
            _write_run_artifacts(
                robustness_directories[slice_metadata["slice_name"]],
                robustness_payload,
                robustness_predictions,
                split_metadata,
                dataset_bundle.metadata,
                manifest_reference,
                slice_metadata=slice_metadata,
            )
        )


def _robustness_run_directories(
    robustness_root: Path,
    model_name: str,
    seed: int,
    robustness_slices: list[dict[str, Any]],
) -> dict[str, Path]:
    return {
        str(slice_config["name"]): robustness_root / f"{model_name}__{slice_config['name']}__seed_{seed}"
        for slice_config in robustness_slices
    }


def _nominal_robustness_complete(nominal_directory: Path, robustness_directories: dict[str, Path]) -> bool:
    return _artifacts_complete(
        nominal_directory,
        BASELINE_ARTIFACT_FILES,
        expected_kind="nominal_baseline",
    ) and _robustness_runs_complete(robustness_directories)


def _calibrated_robustness_complete(calibrated_directory: Path, robustness_directories: dict[str, Path]) -> bool:
    return _artifacts_complete(
        calibrated_directory,
        BASELINE_ARTIFACT_FILES,
        expected_kind="calibrated_variant",
    ) and _robustness_runs_complete(robustness_directories)


def _robustness_runs_complete(robustness_directories: dict[str, Path]) -> bool:
    return all(
        _artifacts_complete(
            run_directory,
            ROBUSTNESS_ARTIFACT_FILES,
            expected_kind="robustness_overlay",
        )
        for run_directory in robustness_directories.values()
    )


STAGE_HANDLERS = {
    "run_baselines": _run_baselines,
    "run_method": _run_method,
    "run_ablations": _run_ablations,
    "evaluate_robustness": _evaluate_robustness,
    "evaluate_calibration": _evaluate_calibration,
    "aggregate_results": _aggregate_results,
    "make_tables": _make_tables,
    "make_figures": _make_figures,
    "audit_results": _audit_results,
}
