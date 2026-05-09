from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.data import load_dataset
from lab.evaluation.metrics import binary_classification_metrics
from lab.evaluation.robustness import apply_missingness_overlay
from lab.methods import support as method_support
from lab.study import load_study_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantify intentionally leaky all-split preprocessing effects.")
    parser.add_argument("--study-config", required=True, help="Path to the study config TOML.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated subset of seeds.")
    parser.add_argument("--combine-only", action="store_true", help="Combine existing partials without training.")
    args = parser.parse_args(argv)

    spec = load_study_spec(args.study_config)
    if args.combine_only:
        _combine_partial_results(spec)
        return 0

    dataset_bundle = load_dataset(spec.configs["dataset"])
    split_map = method_support.split_map_from_protocol(spec, dataset_bundle)
    selected_seeds = _selected_seeds(args.seeds, spec.seed_list)
    supported_baselines = [
        config
        for config in spec.configs["baselines"]["baseline"]
        if str(config["name"]) in {"logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"}
    ]
    hardest_slice = spec.configs["robustness"]["slice"][-1]

    for seed in selected_seeds:
        partial_path = _partial_path(spec, seed)
        if partial_path.exists():
            continue
        print(f"Running leakage ablation for seed {seed}...")
        split_metadata = split_map[int(seed)]
        seed_payload = {
            "study_id": spec.study_id,
            "seed": int(seed),
            "slice_name": str(hardest_slice["name"]),
            "results": {},
        }
        for baseline_config in supported_baselines:
            model_name = str(baseline_config["name"])
            result = _run_leaky_baseline(
                baseline_config,
                dataset_bundle,
                split_metadata,
                hardest_slice,
                spec.configs["robustness"],
            )
            seed_payload["results"][model_name] = result
        _write_partial_result(spec, seed_payload)
    _combine_partial_results(spec)
    return 0


def _run_leaky_baseline(
    baseline_config: dict[str, Any],
    dataset_bundle,
    split_metadata: dict[str, Any],
    slice_config: dict[str, Any],
    robustness_config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(split_metadata["seed"])
    X_train = dataset_bundle.features.loc[split_metadata["train_row_ids"]].copy()
    X_validation = dataset_bundle.features.loc[split_metadata["validation_row_ids"]].copy()
    X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
    X_all = dataset_bundle.features.loc[
        [*split_metadata["train_row_ids"], *split_metadata["validation_row_ids"], *split_metadata["test_row_ids"]]
    ].copy()
    y_train = dataset_bundle.target.loc[split_metadata["train_row_ids"]].copy()
    y_validation = dataset_bundle.target.loc[split_metadata["validation_row_ids"]].copy()
    y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()
    baseline_name = str(baseline_config["name"])

    if baseline_name in {"logistic_regression", "random_forest", "xgboost"}:
        model = _fit_leaky_sklearn_like(
            baseline_config,
            baseline_name=baseline_name,
            X_all=X_all,
            X_train=X_train,
            y_train=y_train,
            categorical_columns=dataset_bundle.categorical_columns,
            numerical_columns=dataset_bundle.numerical_columns,
            seed=seed,
        )
        predict_probabilities = lambda features: np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    elif baseline_name == "lightgbm":
        model, category_levels = _fit_leaky_lightgbm(
            baseline_config,
            X_all=X_all,
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            categorical_columns=dataset_bundle.categorical_columns,
            numerical_columns=dataset_bundle.numerical_columns,
            seed=seed,
        )
        predict_probabilities = lambda features: np.asarray(
            model.predict_proba(_prepare_lightgbm_frame(features, category_levels, dataset_bundle.categorical_columns, dataset_bundle.numerical_columns))[:, 1],
            dtype=float,
        )
    elif baseline_name == "catboost":
        return {
            "status": "not_applicable",
            "reason": "CatBoost uses native missing/categorical handling; no separate fitted imputer/scaler/category vocabulary is ablated here.",
        }
    else:
        return {"status": "unsupported", "reason": f"Unsupported leakage baseline: {baseline_name}"}

    nominal_probabilities = predict_probabilities(X_test)
    overlay_features, overlay_metadata = apply_missingness_overlay(X_test, robustness_config, slice_config, seed=seed)
    overlay_probabilities = predict_probabilities(overlay_features)
    return {
        "status": "completed",
        "nominal_metrics": binary_classification_metrics(y_test.to_numpy(), nominal_probabilities),
        "overlay_metrics": binary_classification_metrics(y_test.to_numpy(), overlay_probabilities),
        "overlay_metadata": overlay_metadata,
    }


def _fit_leaky_sklearn_like(
    baseline_config: dict[str, Any],
    *,
    baseline_name: str,
    X_all: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _dense_one_hot_encoder())]),
                categorical_columns,
            ),
        ]
    )
    preprocessor.fit(X_all)
    if baseline_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=int(baseline_config["max_iter"]),
            solver=str(baseline_config["solver"]),
            C=float(baseline_config["C"]),
            random_state=seed,
        )
    elif baseline_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(baseline_config["n_estimators"]),
            max_depth=int(baseline_config["max_depth"]),
            min_samples_leaf=int(baseline_config["min_samples_leaf"]),
            max_features=str(baseline_config["max_features"]),
            n_jobs=int(baseline_config.get("n_jobs", -1)),
            random_state=seed,
        )
    else:
        xgb_classifier = getattr(__import__("xgboost"), "XGBClassifier")
        classifier = xgb_classifier(
            n_estimators=int(baseline_config.get("n_estimators", 400)),
            max_depth=int(baseline_config.get("max_depth", 6)),
            learning_rate=float(baseline_config.get("learning_rate", 0.05)),
            subsample=float(baseline_config.get("subsample", 0.8)),
            colsample_bytree=float(baseline_config.get("colsample_bytree", 0.8)),
            random_state=seed,
            n_jobs=int(baseline_config.get("n_jobs", -1)),
            eval_metric="logloss",
        )
    train_transformed = preprocessor.transform(X_train)
    classifier.fit(train_transformed, y_train)
    return Pipeline(steps=[("preprocessor", _PrefitTransformer(preprocessor)), ("classifier", classifier)])


def _fit_leaky_lightgbm(
    baseline_config: dict[str, Any],
    *,
    X_all: pd.DataFrame,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> tuple[LGBMClassifier, dict[str, list[str]]]:
    category_levels = _category_levels(X_all, categorical_columns)
    prepared_train = _prepare_lightgbm_frame(X_train, category_levels, categorical_columns, numerical_columns)
    prepared_validation = _prepare_lightgbm_frame(X_validation, category_levels, categorical_columns, numerical_columns)
    model = LGBMClassifier(
        n_estimators=int(baseline_config["n_estimators"]),
        learning_rate=float(baseline_config["learning_rate"]),
        num_leaves=int(baseline_config["num_leaves"]),
        subsample=float(baseline_config["subsample"]),
        colsample_bytree=float(baseline_config["colsample_bytree"]),
        random_state=seed,
        n_jobs=int(baseline_config.get("n_jobs", -1)),
        verbosity=-1,
    )
    model.fit(
        prepared_train,
        y_train,
        eval_set=[(prepared_validation, y_validation)],
        eval_metric="auc",
        categorical_feature=categorical_columns,
        callbacks=[early_stopping(int(baseline_config["early_stopping_rounds"]), verbose=False)],
    )
    return model, category_levels


class _PrefitTransformer:
    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_PrefitTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        return self.transformer.transform(X)


def _category_levels(features: pd.DataFrame, categorical_columns: list[str]) -> dict[str, list[str]]:
    return {
        column_name: features[column_name].astype("string").dropna().drop_duplicates().sort_values().tolist()
        for column_name in categorical_columns
    }


def _prepare_lightgbm_frame(
    features: pd.DataFrame,
    category_levels: dict[str, list[str]],
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    prepared = features.copy()
    for column_name in numerical_columns:
        prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")
    for column_name in categorical_columns:
        prepared[column_name] = pd.Categorical(prepared[column_name].astype("string"), categories=category_levels[column_name])
    return prepared


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _selected_seeds(seed_argument: str, study_seeds: list[int]) -> list[int]:
    if not seed_argument.strip():
        return [int(seed) for seed in study_seeds]
    requested = [int(token.strip()) for token in seed_argument.split(",") if token.strip()]
    invalid = sorted(set(requested) - set(int(seed) for seed in study_seeds))
    if invalid:
        raise ValueError("Requested seeds are not in the study seed roster: " + ", ".join(str(seed) for seed in invalid))
    return requested


def _partial_path(spec, seed: int) -> Path:
    return spec.results_dir / "leakage_ablation" / "partials" / f"seed_{int(seed)}.json"


def _write_partial_result(spec, payload: dict[str, Any]) -> None:
    partial_path = _partial_path(spec, int(payload["seed"]))
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combine_partial_results(spec) -> None:
    output_dir = spec.results_dir / "leakage_ablation"
    partial_dir = output_dir / "partials"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(partial_dir.glob("seed_*.json"))]
    if not partial_payloads:
        return
    model_names = sorted(partial_payloads[0]["results"])
    aggregated = {}
    for model_name in model_names:
        completed = [payload["results"][model_name] for payload in partial_payloads if payload["results"][model_name].get("status") == "completed"]
        if not completed:
            aggregated[model_name] = partial_payloads[0]["results"][model_name]
            continue
        aggregated[model_name] = {
            "status": "completed",
            "n_runs": len(completed),
            "nominal": _summarize([item["nominal_metrics"] for item in completed]),
            "overlay": _summarize([item["overlay_metrics"] for item in completed]),
        }
    payload = {"study_id": spec.study_id, "results": aggregated}
    (output_dir / "leakage_ablation_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "leakage_ablation_results.md").write_text(_markdown(payload), encoding="utf-8")


def _summarize(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for metric_name in metrics_list[0]:
        values = [float(metrics[metric_name]) for metrics in metrics_list]
        summary[f"mean_{metric_name}"] = round(float(np.mean(values)), 6)
        summary[f"std_{metric_name}"] = round(float(np.std(values, ddof=0)), 6)
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Leakage Ablation Results",
        "",
        f"Study: `{payload['study_id']}`.",
        "",
        "| model | status | leaky nominal AUROC | leaky overlay AUROC |",
        "| --- | --- | --- | --- |",
    ]
    for model_name, result in payload["results"].items():
        if result.get("status") != "completed":
            lines.append(f"| {model_name} | {result.get('status')} |  |  |")
        else:
            lines.append(
                f"| {model_name} | completed | {result['nominal']['mean_auroc']:.6f} | {result['overlay']['mean_auroc']:.6f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
