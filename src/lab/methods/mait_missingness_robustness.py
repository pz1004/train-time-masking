from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from lab.data import TabularDatasetBundle, load_dataset
from lab.evaluation.calibration import fit_probability_calibrator
from lab.evaluation.metrics import binary_classification_metrics
from lab.models import train_mait_classifier
from lab.study import StudySpec

from . import support as method_support


@dataclass(frozen=True)
class NeuralMethodBlueprint:
    method_name: str
    architecture_name: str
    calibration_method: str
    selection_summary: dict[str, Any]


def run_method_stage(spec: StudySpec, stage_name: str) -> dict[str, Any]:
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    blueprint = _method_blueprint(spec)
    method_root = spec.raw_dir / "methods"
    method_support.reset_directory(method_root)

    training_config = spec.configs["method"].get("training", {})
    use_missingness_indicators = bool(training_config.get("uses_missingness_indicators", True))
    use_calibration = bool(training_config.get("uses_validation_calibration", False))

    artifact_paths = []
    for seed in spec.seed_list:
        split_metadata = split_map[int(seed)]
        trained_variant = _train_variant(
            spec,
            stage_name,
            dataset_bundle=dataset_bundle,
            split_metadata=split_metadata,
            blueprint=blueprint,
            result_kind="method",
            model_name=blueprint.method_name,
            use_missingness_indicators=use_missingness_indicators,
            use_calibration=use_calibration,
        )
        run_directory = method_root / f"{blueprint.method_name}__seed_{seed}"
        artifact_paths.extend(
            method_support.write_run_artifacts(
                spec,
                stage_name,
                run_directory=run_directory,
                metrics_payload=trained_variant["metrics_payload"],
                predictions=trained_variant["predictions"],
                split_metadata=split_metadata,
                dataset_metadata=dataset_bundle.metadata,
            )
        )

    summary = (
        "Implemented the study method as neural mask-augmented imputation training "
        f"with `{blueprint.architecture_name}` and "
        f"{'validation-fit sigmoid calibration enabled' if use_calibration else 'validation calibration disabled'}."
    )
    return {"artifact_paths": [str(path) for path in artifact_paths], "summary": summary}


def run_ablations_stage(spec: StudySpec, stage_name: str) -> dict[str, Any]:
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    blueprint = _method_blueprint(spec)
    ablation_root = spec.raw_dir / "ablations"
    ablation_root.mkdir(parents=True, exist_ok=True)
    robustness_root = spec.raw_dir / "robustness"
    robustness_root.mkdir(parents=True, exist_ok=True)
    robustness_config = spec.configs["robustness"]
    slices = list(robustness_config["slice"])
    ablation_plans = _study_ablation_plans(spec)

    artifact_paths = []
    for plan in ablation_plans:
        for seed in spec.seed_list:
            run_directory = ablation_root / f"{plan['name']}__seed_{seed}"
            if _variant_artifacts_complete(run_directory, expected_kind="ablation"):
                continue
            split_metadata = split_map[int(seed)]
            trained_variant = _train_variant(
                spec,
                stage_name,
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
            artifact_paths.extend(
                method_support.write_run_artifacts(
                    spec,
                    stage_name,
                    run_directory=run_directory,
                    metrics_payload=trained_variant["metrics_payload"],
                    predictions=trained_variant["predictions"],
                    split_metadata=split_metadata,
                    dataset_metadata=dataset_bundle.metadata,
                )
            )
            for slice_config in slices:
                robustness_directory = robustness_root / f"{plan['name']}__{slice_config['name']}__seed_{seed}"
                if _variant_artifacts_complete(
                    robustness_directory,
                    expected_kind="robustness_overlay",
                    requires_slice_metadata=True,
                ):
                    continue
                metrics_payload, predictions, slice_metadata = method_support.robustness_artifacts(
                    spec,
                    stage_name,
                    model_name=str(plan["name"]),
                    split_metadata=split_metadata,
                    dataset_bundle=dataset_bundle,
                    predict_probabilities=trained_variant["predict_probabilities"],
                    slice_config=slice_config,
                    robustness_config=robustness_config,
                    nominal_reference_kind="ablation",
                    model_metadata=trained_variant["metrics_payload"]["model_metadata"],
                    software_versions=trained_variant["metrics_payload"]["software_versions"],
                )
                artifact_paths.extend(
                    method_support.write_run_artifacts(
                        spec,
                        stage_name,
                        run_directory=robustness_directory,
                        metrics_payload=metrics_payload,
                        predictions=predictions,
                        split_metadata=split_metadata,
                        dataset_metadata=dataset_bundle.metadata,
                        slice_metadata=slice_metadata,
                    )
                )

    summary = (
        "Ran MAIT ablations for explicit missingness indicators, the validation-fit post-hoc sigmoid "
        f"calibrator, and the same-backbone MLP-only control around `{blueprint.architecture_name}`."
    )
    return {"artifact_paths": [str(path) for path in artifact_paths], "summary": summary}


def evaluate_robustness_stage(spec: StudySpec, stage_name: str) -> dict[str, Any]:
    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    blueprint = _method_blueprint(spec)
    robustness_root = spec.raw_dir / "robustness"
    robustness_config = spec.configs["robustness"]
    slices = list(robustness_config["slice"])
    training_config = spec.configs["method"].get("training", {})
    variant_plans = [
        {
            "result_kind": "method",
            "model_name": blueprint.method_name,
            "use_missingness_indicators": bool(training_config.get("uses_missingness_indicators", True)),
            "use_calibration": bool(training_config.get("uses_validation_calibration", False)),
            "training_overrides": {},
            "augmentation_overrides": {},
        },
        *[
            {
                "result_kind": "ablation",
                "model_name": str(plan["name"]),
                "use_missingness_indicators": bool(plan["use_missingness_indicators"]),
                "use_calibration": bool(plan["use_calibration"]),
                "training_overrides": dict(plan.get("training_overrides", {})),
                "augmentation_overrides": dict(plan.get("augmentation_overrides", {})),
            }
            for plan in _study_ablation_plans(spec)
        ],
    ]
    current_variant_names = [str(plan["model_name"]) for plan in variant_plans]

    artifact_paths = []
    for legacy_name in _legacy_robustness_variant_names(current_variant_names):
        method_support.clear_existing_robustness_runs(
            robustness_root,
            legacy_name,
            [str(item["name"]) for item in slices],
            spec.seed_list,
        )
    for plan in variant_plans:
        for seed in spec.seed_list:
            pending_slices = []
            for slice_config in slices:
                run_directory = robustness_root / f"{plan['model_name']}__{slice_config['name']}__seed_{seed}"
                if not _variant_artifacts_complete(run_directory, expected_kind="robustness_overlay", requires_slice_metadata=True):
                    pending_slices.append(slice_config)
            if not pending_slices:
                continue
            split_metadata = split_map[int(seed)]
            trained_variant = _train_variant(
                spec,
                stage_name,
                dataset_bundle=dataset_bundle,
                split_metadata=split_metadata,
                blueprint=blueprint,
                result_kind=str(plan["result_kind"]),
                model_name=str(plan["model_name"]),
                use_missingness_indicators=bool(plan["use_missingness_indicators"]),
                use_calibration=bool(plan["use_calibration"]),
                training_overrides=dict(plan.get("training_overrides", {})),
                augmentation_overrides=dict(plan.get("augmentation_overrides", {})),
            )
            for slice_config in pending_slices:
                metrics_payload, predictions, slice_metadata = method_support.robustness_artifacts(
                    spec,
                    stage_name,
                    model_name=str(plan["model_name"]),
                    split_metadata=split_metadata,
                    dataset_bundle=dataset_bundle,
                    predict_probabilities=trained_variant["predict_probabilities"],
                    slice_config=slice_config,
                    robustness_config=robustness_config,
                    nominal_reference_kind=str(plan["result_kind"]),
                    model_metadata=trained_variant["metrics_payload"]["model_metadata"],
                    software_versions=trained_variant["metrics_payload"]["software_versions"],
                )
                run_directory = robustness_root / f"{plan['model_name']}__{slice_metadata['slice_name']}__seed_{seed}"
                artifact_paths.extend(
                    method_support.write_run_artifacts(
                        spec,
                        stage_name,
                        run_directory=run_directory,
                        metrics_payload=metrics_payload,
                        predictions=predictions,
                        split_metadata=split_metadata,
                        dataset_metadata=dataset_bundle.metadata,
                        slice_metadata=slice_metadata,
                    )
                )

    summary = "Generated robustness overlays for the neural MAIT method and all configured ablations on frozen test copies."
    return {"artifact_paths": [str(path) for path in artifact_paths], "summary": summary}


def evaluate_calibration_stage(spec: StudySpec, stage_name: str) -> dict[str, Any]:
    blueprint = _method_blueprint(spec)
    method_name = blueprint.method_name
    records = {
        method_name: method_support.collect_variant_records(spec.raw_dir / "methods", method_name, spec.seed_list),
    }
    for plan in _study_ablation_plans(spec):
        records[str(plan["name"])] = method_support.collect_variant_records(
            spec.raw_dir / "ablations",
            str(plan["name"]),
            spec.seed_list,
        )

    summary_payload = {
        "study_id": spec.study_id,
        "stage": stage_name,
        "method_name": method_name,
        "base_model_name": blueprint.architecture_name,
        "calibration_method": blueprint.calibration_method,
        "selection_summary": blueprint.selection_summary,
        "variants": {},
        "missing_variants": [],
    }
    for variant_name, variant_records in records.items():
        if not variant_records:
            raise RuntimeError(
                f"Calibration evaluation requires completed artifacts for variant '{variant_name}' "
                f"under study '{spec.study_id}'."
            )
        summary_payload["variants"][variant_name] = method_support.summarize_variant_calibration(variant_records)

    summary_path = spec.aggregated_dir / "method_calibration_summary.json"
    markdown_path = spec.tables_dir / "method_calibration_summary.md"
    method_support.write_json(summary_path, summary_payload)
    method_support.write_text(markdown_path, method_support.calibration_summary_markdown(summary_payload))
    summary = f"Summarized calibration behavior for neural MAIT variants anchored to `{blueprint.architecture_name}`."
    return {"artifact_paths": [str(summary_path), str(markdown_path)], "summary": summary}


def _train_variant(
    spec: StudySpec,
    stage_name: str,
    *,
    dataset_bundle: TabularDatasetBundle,
    split_metadata: dict[str, Any],
    blueprint: NeuralMethodBlueprint,
    result_kind: str,
    model_name: str,
    use_missingness_indicators: bool,
    use_calibration: bool,
    training_overrides: dict[str, Any] | None = None,
    augmentation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = int(split_metadata["seed"])
    X_train = dataset_bundle.features.loc[split_metadata["train_row_ids"]].copy()
    X_validation = dataset_bundle.features.loc[split_metadata["validation_row_ids"]].copy()
    X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
    y_train = dataset_bundle.target.loc[split_metadata["train_row_ids"]].copy()
    y_validation = dataset_bundle.target.loc[split_metadata["validation_row_ids"]].copy()
    y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()

    augmentation_columns = _augmentation_columns(spec, X_train)
    training_config = dict(spec.configs["method"].get("training", {}))
    if training_overrides:
        training_config.update(training_overrides)
    augmentation_config = dict(spec.configs["method"].get("augmentation", {}))
    if augmentation_overrides:
        augmentation_config.update(augmentation_overrides)
    augmentation_metadata = _augmentation_metadata(
        X_train,
        augmentation_columns=augmentation_columns,
        augmentation_config=augmentation_config,
    )
    trained_model = train_mait_classifier(
        train_features=X_train,
        validation_features=X_validation,
        test_features=X_test,
        y_train=y_train,
        y_validation=y_validation,
        categorical_columns=list(dataset_bundle.categorical_columns),
        numerical_columns=list(dataset_bundle.numerical_columns),
        seed=seed,
        use_missingness_indicators=use_missingness_indicators,
        augmentation_columns=augmentation_columns,
        mask_only_observed_values=bool(augmentation_config.get("mask_only_observed_values", True)),
        training_config=training_config,
        augmentation_config=augmentation_config,
    )

    raw_validation_probabilities = np.asarray(trained_model.validation_probabilities, dtype=float)
    raw_test_probabilities = np.asarray(trained_model.test_probabilities, dtype=float)

    if use_calibration:
        calibrator = fit_probability_calibrator(
            blueprint.calibration_method,
            y_validation.to_numpy(),
            raw_validation_probabilities,
            seed=seed,
        )
        predict_start = perf_counter()
        validation_probabilities = calibrator.predict(raw_validation_probabilities)
        test_probabilities = calibrator.predict(raw_test_probabilities)
        calibration_predict_seconds = round(perf_counter() - predict_start, 6)

        def predict_probabilities(features: pd.DataFrame) -> np.ndarray:
            return calibrator.predict(trained_model.predict_probabilities(features))

        fit_seconds = round(trained_model.fit_seconds + calibrator.fit_seconds, 6)
        predict_seconds = round(trained_model.predict_seconds + calibration_predict_seconds, 6)
        calibrator_metadata: dict[str, Any] | None = calibrator.metadata
        calibrator_versions = calibrator.software_versions
    else:
        validation_probabilities = raw_validation_probabilities
        test_probabilities = raw_test_probabilities

        def predict_probabilities(features: pd.DataFrame) -> np.ndarray:
            return trained_model.predict_probabilities(features)

        fit_seconds = trained_model.fit_seconds
        predict_seconds = trained_model.predict_seconds
        calibrator_metadata = None
        calibrator_versions = {}

    predictions = pd.concat(
        [
            method_support.prediction_frame(
                model_name,
                seed,
                "validation",
                y_validation,
                validation_probabilities,
                raw_probabilities=raw_validation_probabilities,
                extra_columns={
                    "result_kind": result_kind,
                    "base_model_name": blueprint.architecture_name,
                    "calibration_method": blueprint.calibration_method if use_calibration else "disabled",
                    "uses_missingness_indicators": use_missingness_indicators,
                    "uses_calibration": use_calibration,
                },
            ),
            method_support.prediction_frame(
                model_name,
                seed,
                "test",
                y_test,
                test_probabilities,
                raw_probabilities=raw_test_probabilities,
                extra_columns={
                    "result_kind": result_kind,
                    "base_model_name": blueprint.architecture_name,
                    "calibration_method": blueprint.calibration_method if use_calibration else "disabled",
                    "uses_missingness_indicators": use_missingness_indicators,
                    "uses_calibration": use_calibration,
                },
            ),
        ],
        ignore_index=True,
    )

    metrics_payload = {
        "study_id": spec.study_id,
        "artifact_status": spec.artifact_status,
        "stage": stage_name,
        "result_kind": result_kind,
        "model_name": model_name,
        "method_name": blueprint.method_name,
        "base_model_name": blueprint.architecture_name,
        "calibration_method": blueprint.calibration_method if use_calibration else "disabled",
        "seed": seed,
        "row_counts": {
            "train": int(split_metadata["train_size"]),
            "validation": int(split_metadata["validation_size"]),
            "test": int(split_metadata["test_size"]),
        },
        "validation_metrics": binary_classification_metrics(y_validation.to_numpy(), validation_probabilities),
        "test_metrics": binary_classification_metrics(y_test.to_numpy(), test_probabilities),
        "pre_calibration_validation_metrics": binary_classification_metrics(y_validation.to_numpy(), raw_validation_probabilities),
        "pre_calibration_test_metrics": binary_classification_metrics(y_test.to_numpy(), raw_test_probabilities),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "model_metadata": {
            "implementation": "neural_mait",
            "base_model_name": blueprint.architecture_name,
            "selection_summary": blueprint.selection_summary,
            "applied_training_config": training_config,
            "augmentation": augmentation_metadata,
            "applied_augmentation_config": augmentation_config,
            "uses_missingness_indicators": use_missingness_indicators,
            "uses_validation_calibration": use_calibration,
            "neural_model_metadata": trained_model.model_metadata,
            "calibrator_metadata": calibrator_metadata,
        },
        "software_versions": {**trained_model.software_versions, **calibrator_versions},
    }
    return {
        "metrics_payload": metrics_payload,
        "predictions": predictions,
        "predict_probabilities": predict_probabilities,
    }


def _augmentation_metadata(
    features: pd.DataFrame,
    *,
    augmentation_columns: list[str],
    augmentation_config: dict[str, Any],
) -> dict[str, Any]:
    mask_rates = [float(rate) for rate in augmentation_config.get("mask_rates", [0.1, 0.2, 0.3])]
    use_stochastic_masking = bool(augmentation_config.get("use_stochastic_masking", True))
    return {
        "enabled": use_stochastic_masking,
        "copy_mode": "stochastic_masking" if use_stochastic_masking else "disabled",
        "columns_source": str(augmentation_config.get("columns_source", "robustness_config")),
        "mask_rates": mask_rates,
        "source_row_count": int(len(features)),
        "columns": augmentation_columns,
        "mask_only_observed_values": bool(augmentation_config.get("mask_only_observed_values", True)),
    }


def _augmentation_columns(spec: StudySpec, features: pd.DataFrame) -> list[str]:
    robustness_columns = [str(column_name) for column_name in spec.configs["robustness"]["robustness"]["columns"]]
    missing_columns = [column_name for column_name in robustness_columns if column_name not in features.columns]
    if missing_columns:
        raise ValueError(
            f"Robustness augmentation columns are missing from the training frame for study '{spec.study_id}': "
            + ", ".join(sorted(missing_columns))
        )
    return robustness_columns


def _method_blueprint(spec: StudySpec) -> NeuralMethodBlueprint:
    method_config = spec.configs["method"]["method"]
    training_config = spec.configs["method"].get("training", {})
    augmentation_config = spec.configs["method"].get("augmentation", {})
    calibration_config = spec.configs["method"].get("calibration", {})
    return NeuralMethodBlueprint(
        method_name=str(method_config["name"]),
        architecture_name=str(training_config.get("architecture", "mait_mlp")),
        calibration_method=str(calibration_config.get("method", "sigmoid")),
        selection_summary={
            "training": {
                "architecture": str(training_config.get("architecture", "mait_mlp")),
                "hidden_dim_1": int(training_config.get("hidden_dim_1", 256)),
                "hidden_dim_2": int(training_config.get("hidden_dim_2", 128)),
                "dropout": float(training_config.get("dropout", 0.1)),
                "learning_rate": float(training_config.get("learning_rate", 1e-3)),
                "weight_decay": float(training_config.get("weight_decay", 1e-4)),
                "batch_size": int(training_config.get("batch_size", 512)),
                "max_epochs": int(training_config.get("max_epochs", 60)),
                "reconstruction_weight": float(training_config.get("reconstruction_weight", 1.0)),
            },
            "augmentation": {
                "columns_source": str(augmentation_config.get("columns_source", "robustness_config")),
                "mask_rates": [float(rate) for rate in augmentation_config.get("mask_rates", [0.1, 0.2, 0.3])],
            },
            "calibration_method": str(calibration_config.get("method", "sigmoid")),
        },
    )


def _study_ablation_plans(spec: StudySpec) -> list[dict[str, Any]]:
    ablation_rows = list(spec.configs.get("ablations", {}).get("ablation", []))
    supported_plans = {
        "remove_missingness_indicators": {
            "name": "remove_missingness_indicators",
            "use_missingness_indicators": False,
            "use_calibration": False,
            "training_overrides": {},
            "augmentation_overrides": {},
        },
        "enable_calibration_head": {
            "name": "enable_calibration_head",
            "use_missingness_indicators": True,
            "use_calibration": True,
            "training_overrides": {},
            "augmentation_overrides": {},
        },
        "mlp_only": {
            "name": "mlp_only",
            "use_missingness_indicators": False,
            "use_calibration": False,
            "training_overrides": {
                "enable_reconstruction": False,
                "reconstruction_weight": 0.0,
            },
            "augmentation_overrides": {
                "use_stochastic_masking": False,
                "mask_rates": [0.0],
            },
        },
    }

    configured_names = [str(row["name"]) for row in ablation_rows]
    unsupported_names = [name for name in configured_names if name not in supported_plans]
    if unsupported_names:
        raise ValueError(
            f"Unsupported ablation variant(s) for study '{spec.study_id}': {', '.join(sorted(unsupported_names))}"
        )

    missing_required = [name for name in supported_plans if name not in configured_names]
    if missing_required:
        raise ValueError(
            f"Missing required ablation variant(s) for study '{spec.study_id}': {', '.join(sorted(missing_required))}"
        )

    plans: list[dict[str, Any]] = []
    for row in ablation_rows:
        row_name = str(row["name"])
        base_plan = supported_plans[row_name]
        plan = {
            "name": row_name,
            "use_missingness_indicators": bool(row.get("use_missingness_indicators", base_plan["use_missingness_indicators"])),
            "use_calibration": bool(row.get("use_calibration", base_plan["use_calibration"])),
            "training_overrides": dict(base_plan.get("training_overrides", {})),
            "augmentation_overrides": dict(base_plan.get("augmentation_overrides", {})),
        }
        if "enable_reconstruction" in row:
            plan["training_overrides"]["enable_reconstruction"] = bool(row["enable_reconstruction"])
        if "reconstruction_weight" in row:
            plan["training_overrides"]["reconstruction_weight"] = float(row["reconstruction_weight"])
        if "use_stochastic_masking" in row:
            plan["augmentation_overrides"]["use_stochastic_masking"] = bool(row["use_stochastic_masking"])
        if "mask_rates" in row:
            plan["augmentation_overrides"]["mask_rates"] = [float(rate) for rate in row["mask_rates"]]
        if not plan["augmentation_overrides"].get("use_stochastic_masking", True):
            plan["augmentation_overrides"].setdefault("mask_rates", [0.0])
        plans.append(plan)
    return plans


def _variant_artifacts_complete(
    run_directory: Path,
    *,
    expected_kind: str,
    requires_slice_metadata: bool = False,
) -> bool:
    required_paths = [
        run_directory / "metrics.json",
        run_directory / "predictions.csv.gz",
        run_directory / "split_metadata.json",
        run_directory / "dataset_metadata.json",
        run_directory / "manifest_reference.json",
    ]
    if requires_slice_metadata:
        required_paths.append(run_directory / "slice_metadata.json")
    if not all(path.exists() for path in required_paths):
        return False
    try:
        metrics = method_support.read_metrics(run_directory)
    except RuntimeError:
        return False
    return str(metrics.get("result_kind")) == expected_kind


def _legacy_robustness_variant_names(current_variant_names: list[str]) -> list[str]:
    historical_variant_names = ["disable_calibration_head"]
    return [name for name in historical_variant_names if name not in current_variant_names]
