from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
import importlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lab.data import TabularDatasetBundle
from lab.evaluation.metrics import binary_classification_metrics
from lab.reporting import build_prediction_frame


class BaselineEnvironmentError(RuntimeError):
    """Raised when a required baseline dependency is missing."""


@dataclass(frozen=True)
class TrainedBaselineModel:
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    model_metadata: dict[str, Any]
    software_versions: dict[str, str]
    fit_seconds: float
    predict_seconds: float
    predict_probabilities: Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class BaselineRunResult:
    baseline_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    predictions: pd.DataFrame
    model_metadata: dict[str, Any]
    software_versions: dict[str, str]
    fit_seconds: float
    predict_seconds: float
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    predict_probabilities: Callable[[pd.DataFrame], np.ndarray]


def ensure_baseline_environment(baseline_configs: list[dict[str, Any]]) -> None:
    baseline_names = {str(config["name"]) for config in baseline_configs}
    if "catboost" in baseline_names:
        try:
            importlib.import_module("catboost")
        except ModuleNotFoundError as exc:
            raise BaselineEnvironmentError(
                "CatBoost is required for this study phase but is not installed. "
                "Install requirements-research.txt before running run_baselines."
            ) from exc
    if "xgboost" in baseline_names:
        try:
            importlib.import_module("xgboost")
        except ModuleNotFoundError as exc:
            raise BaselineEnvironmentError(
                "XGBoost is required for this study phase but is not installed. "
                "Install requirements-research.txt before running run_baselines."
            ) from exc


def run_tabular_baseline(
    baseline_config: dict[str, Any],
    dataset_bundle: TabularDatasetBundle,
    split_metadata: dict[str, Any],
) -> BaselineRunResult:
    baseline_name = str(baseline_config["name"])
    seed = int(split_metadata["seed"])

    X_train = dataset_bundle.features.loc[split_metadata["train_row_ids"]].copy()
    X_validation = dataset_bundle.features.loc[split_metadata["validation_row_ids"]].copy()
    X_test = dataset_bundle.features.loc[split_metadata["test_row_ids"]].copy()
    y_train = dataset_bundle.target.loc[split_metadata["train_row_ids"]].copy()
    y_validation = dataset_bundle.target.loc[split_metadata["validation_row_ids"]].copy()
    y_test = dataset_bundle.target.loc[split_metadata["test_row_ids"]].copy()

    if baseline_name == "lightgbm":
        trained_model = _run_lightgbm(
            baseline_config,
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            dataset_bundle.categorical_columns,
            dataset_bundle.numerical_columns,
            seed,
        )
    elif baseline_name == "catboost":
        trained_model = _run_catboost(
            baseline_config,
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            dataset_bundle.categorical_columns,
            dataset_bundle.numerical_columns,
            seed,
        )
    elif baseline_name == "logistic_regression":
        trained_model = _run_logistic_regression(
            baseline_config,
            X_train,
            X_validation,
            X_test,
            y_train,
            dataset_bundle.categorical_columns,
            dataset_bundle.numerical_columns,
            seed,
        )
    elif baseline_name == "random_forest":
        trained_model = _run_random_forest(
            baseline_config,
            X_train,
            X_validation,
            X_test,
            y_train,
            dataset_bundle.categorical_columns,
            dataset_bundle.numerical_columns,
            seed,
        )
    elif baseline_name == "xgboost":
        trained_model = _run_xgboost(
            baseline_config,
            X_train,
            X_validation,
            X_test,
            y_train,
            dataset_bundle.categorical_columns,
            dataset_bundle.numerical_columns,
            seed,
        )
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    validation_metrics = binary_classification_metrics(y_validation.to_numpy(), trained_model.validation_probabilities)
    test_metrics = binary_classification_metrics(y_test.to_numpy(), trained_model.test_probabilities)
    predictions = pd.concat(
        [
            build_prediction_frame(baseline_name, seed, "validation", y_validation, trained_model.validation_probabilities),
            build_prediction_frame(baseline_name, seed, "test", y_test, trained_model.test_probabilities),
        ],
        ignore_index=True,
    )

    return BaselineRunResult(
        baseline_name=baseline_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
        model_metadata=trained_model.model_metadata,
        software_versions=trained_model.software_versions,
        fit_seconds=trained_model.fit_seconds,
        predict_seconds=trained_model.predict_seconds,
        validation_probabilities=trained_model.validation_probabilities,
        test_probabilities=trained_model.test_probabilities,
        predict_probabilities=trained_model.predict_probabilities,
    )


def _run_logistic_regression(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    fit_start = perf_counter()
    encoder = _dense_one_hot_encoder()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=int(baseline_config["max_iter"]),
                    solver=str(baseline_config["solver"]),
                    C=float(baseline_config["C"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predict_seconds = round(perf_counter() - predict_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "max_iter": int(baseline_config["max_iter"]),
        "solver": str(baseline_config["solver"]),
        "C": float(baseline_config["C"]),
    }
    versions = {
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=lambda features: np.asarray(model.predict_proba(features)[:, 1], dtype=float),
    )


def _run_random_forest(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    fit_start = perf_counter()
    encoder = _dense_one_hot_encoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", SimpleImputer(strategy="median"), numerical_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(baseline_config["n_estimators"]),
                    max_depth=int(baseline_config["max_depth"]),
                    min_samples_leaf=int(baseline_config["min_samples_leaf"]),
                    max_features=str(baseline_config["max_features"]),
                    n_jobs=int(baseline_config.get("n_jobs", -1)),
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predict_seconds = round(perf_counter() - predict_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "n_estimators": int(baseline_config["n_estimators"]),
        "max_depth": int(baseline_config["max_depth"]),
        "min_samples_leaf": int(baseline_config["min_samples_leaf"]),
        "max_features": str(baseline_config["max_features"]),
        "n_jobs": int(baseline_config.get("n_jobs", -1)),
    }
    versions = {
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=lambda features: np.asarray(model.predict_proba(features)[:, 1], dtype=float),
    )


def _run_xgboost(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    xgboost_module = importlib.import_module("xgboost")
    xgb_classifier = getattr(xgboost_module, "XGBClassifier")

    fit_start = perf_counter()
    encoder = _dense_one_hot_encoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", SimpleImputer(strategy="median"), numerical_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb_classifier(
                    n_estimators=int(baseline_config.get("n_estimators", 400)),
                    max_depth=int(baseline_config.get("max_depth", 6)),
                    learning_rate=float(baseline_config.get("learning_rate", 0.05)),
                    subsample=float(baseline_config.get("subsample", 0.8)),
                    colsample_bytree=float(baseline_config.get("colsample_bytree", 0.8)),
                    random_state=seed,
                    n_jobs=int(baseline_config.get("n_jobs", -1)),
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predict_seconds = round(perf_counter() - predict_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "n_estimators": int(baseline_config.get("n_estimators", 400)),
        "max_depth": int(baseline_config.get("max_depth", 6)),
        "learning_rate": float(baseline_config.get("learning_rate", 0.05)),
        "subsample": float(baseline_config.get("subsample", 0.8)),
        "colsample_bytree": float(baseline_config.get("colsample_bytree", 0.8)),
        "n_jobs": int(baseline_config.get("n_jobs", -1)),
    }
    versions = {
        "xgboost": str(xgboost_module.__version__),
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=lambda features: np.asarray(model.predict_proba(features)[:, 1], dtype=float),
    )


def _run_lightgbm(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    fit_start = perf_counter()
    category_levels = _train_category_levels(X_train, categorical_columns)
    prepared_train = _prepare_lightgbm_frame(X_train, category_levels, categorical_columns, numerical_columns)
    prepared_validation = _prepare_lightgbm_frame(X_validation, category_levels, categorical_columns, numerical_columns)
    prepared_test = _prepare_lightgbm_frame(X_test, category_levels, categorical_columns, numerical_columns)
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
    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = model.predict_proba(prepared_validation)[:, 1]
    test_probabilities = model.predict_proba(prepared_test)[:, 1]
    predict_seconds = round(perf_counter() - predict_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "n_estimators": int(baseline_config["n_estimators"]),
        "learning_rate": float(baseline_config["learning_rate"]),
        "num_leaves": int(baseline_config["num_leaves"]),
        "subsample": float(baseline_config["subsample"]),
        "colsample_bytree": float(baseline_config["colsample_bytree"]),
        "early_stopping_rounds": int(baseline_config["early_stopping_rounds"]),
        "best_iteration": int(model.best_iteration_ or int(baseline_config["n_estimators"])),
        "n_jobs": int(baseline_config.get("n_jobs", -1)),
    }
    versions = {
        "lightgbm": __import__("lightgbm").__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=lambda features: np.asarray(
            model.predict_proba(_prepare_lightgbm_frame(features, category_levels, categorical_columns, numerical_columns))[:, 1],
            dtype=float,
        ),
    )


def _run_catboost(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    catboost_module = importlib.import_module("catboost")
    catboost_classifier = getattr(catboost_module, "CatBoostClassifier")

    fit_start = perf_counter()
    prepared_train = _prepare_catboost_frame(X_train, categorical_columns, numerical_columns)
    prepared_validation = _prepare_catboost_frame(X_validation, categorical_columns, numerical_columns)
    prepared_test = _prepare_catboost_frame(X_test, categorical_columns, numerical_columns)
    categorical_indices = [prepared_train.columns.get_loc(column_name) for column_name in categorical_columns]
    model = catboost_classifier(
        iterations=int(baseline_config["iterations"]),
        learning_rate=float(baseline_config["learning_rate"]),
        depth=int(baseline_config["depth"]),
        l2_leaf_reg=float(baseline_config["l2_leaf_reg"]),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=int(baseline_config.get("thread_count", -1)),
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        prepared_train,
        y_train,
        eval_set=(prepared_validation, y_validation),
        cat_features=categorical_indices,
        use_best_model=True,
        verbose=False,
        early_stopping_rounds=int(baseline_config["early_stopping_rounds"]),
    )
    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = model.predict_proba(prepared_validation)[:, 1]
    test_probabilities = model.predict_proba(prepared_test)[:, 1]
    predict_seconds = round(perf_counter() - predict_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "iterations": int(baseline_config["iterations"]),
        "learning_rate": float(baseline_config["learning_rate"]),
        "depth": int(baseline_config["depth"]),
        "l2_leaf_reg": float(baseline_config["l2_leaf_reg"]),
        "early_stopping_rounds": int(baseline_config["early_stopping_rounds"]),
        "thread_count": int(baseline_config.get("thread_count", -1)),
        "best_iteration": int(model.get_best_iteration()),
    }
    versions = {
        "catboost": str(catboost_module.__version__),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=lambda features: np.asarray(
            model.predict_proba(_prepare_catboost_frame(features, categorical_columns, numerical_columns))[:, 1],
            dtype=float,
        ),
    )


def _train_category_levels(X_train: pd.DataFrame, categorical_columns: list[str]) -> dict[str, list[str]]:
    return {
        column_name: (
            X_train[column_name]
            .astype("string")
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
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
        prepared[column_name] = pd.Categorical(
            prepared[column_name].astype("string"),
            categories=category_levels[column_name],
        )
    return prepared


def _prepare_catboost_frame(
    features: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    prepared = features.copy()
    for column_name in numerical_columns:
        prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")
    for column_name in categorical_columns:
        prepared[column_name] = (
            prepared[column_name]
            .astype("string")
            .fillna("__missing__")
            .astype(str)
        )
    return prepared


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
