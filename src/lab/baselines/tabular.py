from __future__ import annotations

from dataclasses import dataclass
import inspect
from time import perf_counter
from typing import Any, Callable
import copy
import importlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    if "ft_transformer" in baseline_names:
        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise BaselineEnvironmentError(
                "The FT-Transformer baseline requires PyTorch. Install requirements-research.txt "
                "or remove `ft_transformer` from the baseline config."
            ) from exc
    if "tabpfn" in baseline_names:
        try:
            importlib.import_module("tabpfn")
        except ModuleNotFoundError as exc:
            raise BaselineEnvironmentError(
                "The TabPFN baseline requires the optional `tabpfn` package. Install requirements-research.txt "
                "or remove `tabpfn` from the baseline config."
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
    elif baseline_name == "ft_transformer":
        trained_model = _run_ft_transformer(
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
    elif baseline_name == "tabpfn":
        trained_model = _run_tabpfn(
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


def _run_ft_transformer(
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
    torch = importlib.import_module("torch")
    nn = torch.nn
    data_module = torch.utils.data

    fit_start = perf_counter()
    preprocessor = _dense_numeric_frame_preprocessor(categorical_columns, numerical_columns)
    train_array = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    validation_array = np.asarray(preprocessor.transform(X_validation), dtype=np.float32)
    test_array = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    train_target = y_train.to_numpy(dtype=np.float32)
    validation_target = y_validation.to_numpy(dtype=np.float32)

    torch.manual_seed(seed)
    device_name = str(baseline_config.get("device", "cuda_if_available"))
    if device_name == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = _FTTransformerModule(
        input_dim=int(train_array.shape[1]),
        token_dim=int(baseline_config.get("token_dim", 32)),
        n_heads=int(baseline_config.get("n_heads", 4)),
        n_layers=int(baseline_config.get("n_layers", 2)),
        dropout=float(baseline_config.get("dropout", 0.1)),
        nn_module=nn,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(baseline_config.get("learning_rate", 1e-3)),
        weight_decay=float(baseline_config.get("weight_decay", 1e-4)),
    )
    loss_function = nn.BCEWithLogitsLoss()

    train_dataset = data_module.TensorDataset(
        torch.as_tensor(train_array, dtype=torch.float32),
        torch.as_tensor(train_target, dtype=torch.float32),
    )
    train_loader = data_module.DataLoader(
        train_dataset,
        batch_size=int(baseline_config.get("batch_size", 512)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state: dict[str, Any] | None = None
    best_validation_auroc = float("-inf")
    patience = int(baseline_config.get("early_stopping_patience", 8))
    stale_epochs = 0
    max_epochs = int(baseline_config.get("max_epochs", 60))
    for _epoch in range(max_epochs):
        model.train()
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = loss_function(logits, batch_target)
            loss.backward()
            optimizer.step()

        validation_probabilities_epoch = _torch_predict_probabilities(
            model,
            validation_array,
            torch_module=torch,
            device=device,
            batch_size=int(baseline_config.get("batch_size", 512)),
        )
        validation_auroc = _safe_roc_auc(validation_target, validation_probabilities_epoch)
        if validation_auroc > best_validation_auroc + 1e-6:
            best_validation_auroc = validation_auroc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = _torch_predict_probabilities(
        model,
        validation_array,
        torch_module=torch,
        device=device,
        batch_size=int(baseline_config.get("batch_size", 512)),
    )
    test_probabilities = _torch_predict_probabilities(
        model,
        test_array,
        torch_module=torch,
        device=device,
        batch_size=int(baseline_config.get("batch_size", 512)),
    )
    predict_seconds = round(perf_counter() - predict_start, 6)

    def predict_probabilities(features: pd.DataFrame) -> np.ndarray:
        feature_array = np.asarray(preprocessor.transform(features), dtype=np.float32)
        return _torch_predict_probabilities(
            model,
            feature_array,
            torch_module=torch,
            device=device,
            batch_size=int(baseline_config.get("batch_size", 512)),
        )

    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "input_dim": int(train_array.shape[1]),
        "token_dim": int(baseline_config.get("token_dim", 32)),
        "n_heads": int(baseline_config.get("n_heads", 4)),
        "n_layers": int(baseline_config.get("n_layers", 2)),
        "dropout": float(baseline_config.get("dropout", 0.1)),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "best_validation_auroc": round(best_validation_auroc, 6),
        "device": str(device),
        "note": "Self-contained FT-Transformer-style baseline using per-dimension tokenization.",
    }
    versions = {
        "torch": str(torch.__version__),
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
        predict_probabilities=predict_probabilities,
    )


def _run_tabpfn(
    baseline_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
) -> TrainedBaselineModel:
    tabpfn_module = importlib.import_module("tabpfn")
    tabpfn_classifier = getattr(tabpfn_module, "TabPFNClassifier")

    fit_start = perf_counter()
    preprocessor = _TabPFNFramePreprocessor(categorical_columns, numerical_columns)
    train_array_full = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    validation_array = np.asarray(preprocessor.transform(X_validation), dtype=np.float32)
    test_array = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    train_target_full = y_train.to_numpy(dtype=int)
    categorical_feature_indices = preprocessor.categorical_feature_indices
    max_train_rows = int(baseline_config.get("max_train_rows", 10000))
    subsampled_for_resource_limit = False
    if len(train_array_full) > max_train_rows:
        rng = np.random.default_rng(seed)
        selected_indices = _stratified_subsample_indices(train_target_full, max_train_rows=max_train_rows, rng=rng)
        train_array = train_array_full[selected_indices]
        train_target = train_target_full[selected_indices]
        subsampled_for_resource_limit = True
    else:
        train_array = train_array_full
        train_target = train_target_full

    device_name = str(baseline_config.get("device", "auto"))
    constructor_kwargs: dict[str, Any] = {}
    if device_name != "auto":
        constructor_kwargs["device"] = device_name
    if "ignore_pretraining_limits" in baseline_config:
        constructor_kwargs["ignore_pretraining_limits"] = bool(baseline_config["ignore_pretraining_limits"])
    categorical_constructor_kwargs = _categorical_feature_kwargs(tabpfn_classifier, categorical_feature_indices)
    used_constructor_categorical_indices = bool(categorical_constructor_kwargs)
    try:
        model = tabpfn_classifier(random_state=seed, **constructor_kwargs, **categorical_constructor_kwargs)
    except TypeError:
        used_constructor_categorical_indices = False
        try:
            model = tabpfn_classifier(random_state=seed, **constructor_kwargs)
        except TypeError:
            model = tabpfn_classifier(**constructor_kwargs)

    fit_kwargs = (
        {}
        if used_constructor_categorical_indices
        else _categorical_feature_kwargs(model.fit, categorical_feature_indices)
    )
    try:
        try:
            model.fit(train_array, train_target, **fit_kwargs)
        except TypeError:
            if fit_kwargs:
                model.fit(train_array, train_target)
            else:
                raise
    except Exception as exc:
        if not bool(baseline_config.get("allow_constant_fallback", True)):
            raise
        return _constant_probability_model(
            baseline_config,
            X_validation,
            X_test,
            y_train,
            fallback_reason=f"tabpfn_fit_failed:{type(exc).__name__}:{exc}",
            fit_start=fit_start,
            extra_metadata={
                "native_missing_values": True,
                "categorical_feature_indices": categorical_feature_indices,
                "categorical_features_argument": list(categorical_constructor_kwargs or fit_kwargs),
                "subsampled_for_resource_limit": bool(subsampled_for_resource_limit),
                "train_rows_seen": int(len(train_array)),
                "original_train_rows": int(len(train_array_full)),
                "max_train_rows": max_train_rows,
            },
        )

    fit_seconds = round(perf_counter() - fit_start, 6)
    predict_start = perf_counter()
    validation_probabilities = _positive_class_probabilities(model.predict_proba(validation_array))
    test_probabilities = _positive_class_probabilities(model.predict_proba(test_array))
    predict_seconds = round(perf_counter() - predict_start, 6)

    def predict_probabilities(features: pd.DataFrame) -> np.ndarray:
        feature_array = np.asarray(preprocessor.transform(features), dtype=np.float32)
        return _positive_class_probabilities(model.predict_proba(feature_array))

    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "input_dim": int(train_array_full.shape[1]),
        "train_rows_seen": int(len(train_array)),
        "original_train_rows": int(len(train_array_full)),
        "max_train_rows": max_train_rows,
        "subsampled_for_resource_limit": bool(subsampled_for_resource_limit),
        "native_missing_values": True,
        "categorical_feature_indices": categorical_feature_indices,
        "categorical_features_argument": list(categorical_constructor_kwargs or fit_kwargs),
        "device": device_name,
        "note": "TabPFN uses train-only ordinal categorical encoding, preserves missing values as NaN, and stratified subsampling when configured limits are exceeded.",
    }
    versions = {
        "tabpfn": str(getattr(tabpfn_module, "__version__", "unknown")),
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
        predict_probabilities=predict_probabilities,
    )


class _FTTransformerModule:
    def __new__(
        cls,
        *,
        input_dim: int,
        token_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        nn_module: Any,
    ) -> Any:
        class Module(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn_module.Parameter(importlib.import_module("torch").empty(input_dim, token_dim))
                self.bias = nn_module.Parameter(importlib.import_module("torch").zeros(input_dim, token_dim))
                importlib.import_module("torch").nn.init.xavier_uniform_(self.weight)
                encoder_layer = nn_module.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=n_heads,
                    dim_feedforward=token_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn_module.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.normalization = nn_module.LayerNorm(token_dim)
                self.head = nn_module.Sequential(
                    nn_module.Linear(token_dim, token_dim),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                    nn_module.Linear(token_dim, 1),
                )

            def forward(self, features: Any) -> Any:
                tokens = features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
                encoded = self.encoder(tokens)
                pooled = self.normalization(encoded.mean(dim=1))
                return self.head(pooled).squeeze(-1)

        return Module()


def _torch_predict_probabilities(
    model: Any,
    feature_array: np.ndarray,
    *,
    torch_module: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch_module.no_grad():
        for start in range(0, len(feature_array), batch_size):
            batch = torch_module.as_tensor(feature_array[start:start + batch_size], dtype=torch_module.float32, device=device)
            logits = model(batch)
            probabilities.append(torch_module.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def _dense_numeric_frame_preprocessor(categorical_columns: list[str], numerical_columns: list[str]) -> ColumnTransformer:
    encoder = _dense_one_hot_encoder()
    return ColumnTransformer(
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


class _TabPFNFramePreprocessor:
    def __init__(self, categorical_columns: list[str], numerical_columns: list[str]) -> None:
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns)
        self._category_maps: dict[str, dict[str, int]] = {}

    def fit(self, features: pd.DataFrame) -> "_TabPFNFramePreprocessor":
        for column_name in self.categorical_columns:
            categories = (
                features[column_name]
                .astype("string")
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            self._category_maps[column_name] = {str(category): index for index, category in enumerate(categories)}
        return self

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column_name in self.numerical_columns:
            columns.append(pd.to_numeric(features[column_name], errors="coerce").to_numpy(dtype=np.float32))
        for column_name in self.categorical_columns:
            encoded = features[column_name].astype("string").map(self._category_maps[column_name])
            columns.append(encoded.to_numpy(dtype=np.float32, na_value=np.nan))
        if not columns:
            return np.zeros((len(features), 0), dtype=np.float32)
        return np.column_stack(columns).astype(np.float32, copy=False)

    @property
    def categorical_feature_indices(self) -> list[int]:
        start = len(self.numerical_columns)
        return list(range(start, start + len(self.categorical_columns)))


def _categorical_feature_kwargs(callable_object: Any, categorical_feature_indices: list[int]) -> dict[str, list[int]]:
    if not categorical_feature_indices:
        return {}
    preferred_names = ("categorical_features_indices", "categorical_feature_indices", "categorical_features")
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return {"categorical_features_indices": categorical_feature_indices}
    parameters = signature.parameters
    for name in preferred_names:
        if name in parameters:
            return {name: categorical_feature_indices}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {"categorical_features_indices": categorical_feature_indices}
    return {}


def _stratified_subsample_indices(y: np.ndarray, *, max_train_rows: int, rng: np.random.Generator) -> np.ndarray:
    selected: list[int] = []
    for label in sorted(np.unique(y).tolist()):
        label_indices = np.flatnonzero(y == label)
        target_count = max(1, int(round(max_train_rows * len(label_indices) / len(y))))
        target_count = min(target_count, len(label_indices))
        selected.extend(rng.choice(label_indices, size=target_count, replace=False).tolist())
    if len(selected) > max_train_rows:
        selected = rng.choice(np.asarray(selected, dtype=int), size=max_train_rows, replace=False).tolist()
    elif len(selected) < max_train_rows:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(selected, dtype=int), assume_unique=False)
        if len(remaining):
            fill_count = min(max_train_rows - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=fill_count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _positive_class_probabilities(probability_matrix: Any) -> np.ndarray:
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]


def _constant_probability_model(
    baseline_config: dict[str, Any],
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    fallback_reason: str,
    fit_start: float,
    extra_metadata: dict[str, Any] | None = None,
) -> TrainedBaselineModel:
    prior = float(y_train.mean())
    validation_probabilities = np.full(len(X_validation), prior, dtype=float)
    test_probabilities = np.full(len(X_test), prior, dtype=float)
    fit_seconds = round(perf_counter() - fit_start, 6)
    metadata = {
        "implementation": str(baseline_config["implementation"]),
        "resource_limited_fallback": True,
        "fallback_reason": fallback_reason,
        "constant_probability": prior,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    versions = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return TrainedBaselineModel(
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=0.0,
        predict_probabilities=lambda features: np.full(len(features), prior, dtype=float),
    )


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


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
