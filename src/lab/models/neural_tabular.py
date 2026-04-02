from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


@dataclass(frozen=True)
class TrainedNeuralTabularModel:
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    model_metadata: dict[str, Any]
    software_versions: dict[str, str]
    fit_seconds: float
    predict_seconds: float
    predict_probabilities: Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class PreparedTabularFrame:
    numerical: np.ndarray
    numerical_missing: np.ndarray
    categorical: np.ndarray
    categorical_missing: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.numerical.shape[0] if self.numerical.size else self.categorical.shape[0])


class NeuralTabularPreprocessor:
    def __init__(self, categorical_columns: list[str], numerical_columns: list[str]) -> None:
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns)
        self._category_maps: dict[str, dict[str, int]] = {}
        self._medians: dict[str, float] = {}
        self._scales: dict[str, float] = {}

    def fit(self, features: pd.DataFrame) -> None:
        for column_name in self.categorical_columns:
            categories = (
                features[column_name]
                .astype("string")
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            self._category_maps[column_name] = {category: index + 1 for index, category in enumerate(categories)}

        for column_name in self.numerical_columns:
            numeric_series = pd.to_numeric(features[column_name], errors="coerce")
            median = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0
            filled = numeric_series.fillna(median)
            scale = float(filled.std(ddof=0))
            self._medians[column_name] = median
            self._scales[column_name] = scale if scale > 1e-6 else 1.0

    def transform(self, features: pd.DataFrame) -> PreparedTabularFrame:
        if self.numerical_columns:
            numerical_missing = np.column_stack(
                [
                    pd.to_numeric(features[column_name], errors="coerce").isna().to_numpy(dtype=bool)
                    for column_name in self.numerical_columns
                ]
            )
            numerical = np.column_stack(
                [
                    (
                        pd.to_numeric(features[column_name], errors="coerce")
                        .fillna(self._medians[column_name])
                        .sub(self._medians[column_name])
                        .div(self._scales[column_name])
                    ).to_numpy(dtype=np.float32)
                    for column_name in self.numerical_columns
                ]
            )
        else:
            numerical = np.zeros((len(features), 0), dtype=np.float32)
            numerical_missing = np.zeros((len(features), 0), dtype=bool)

        if self.categorical_columns:
            categorical_missing = np.column_stack(
                [features[column_name].isna().to_numpy(dtype=bool) for column_name in self.categorical_columns]
            )
            categorical = np.column_stack(
                [
                    features[column_name]
                    .astype("string")
                    .map(self._category_maps[column_name])
                    .fillna(0)
                    .astype(int)
                    .to_numpy(dtype=np.int64)
                    for column_name in self.categorical_columns
                ]
            )
        else:
            categorical = np.zeros((len(features), 0), dtype=np.int64)
            categorical_missing = np.zeros((len(features), 0), dtype=bool)

        return PreparedTabularFrame(
            numerical=numerical,
            numerical_missing=numerical_missing,
            categorical=categorical,
            categorical_missing=categorical_missing,
        )

    @property
    def category_cardinalities(self) -> list[int]:
        return [len(self._category_maps[column_name]) + 1 for column_name in self.categorical_columns]


class MLPMAITModel(nn.Module):
    def __init__(
        self,
        *,
        category_cardinalities: list[int],
        numerical_count: int,
        use_missingness_indicators: bool,
        enable_reconstruction: bool = True,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_missingness_indicators = use_missingness_indicators
        self.enable_reconstruction = bool(enable_reconstruction)
        self.numerical_count = numerical_count
        self.category_cardinalities = list(category_cardinalities)

        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, _embedding_dim(cardinality), padding_idx=0)
                for cardinality in self.category_cardinalities
            ]
        )
        cat_embedding_dim = sum(embedding.embedding_dim for embedding in self.category_embeddings)
        indicator_dim = (numerical_count + len(self.category_cardinalities)) if use_missingness_indicators else 0
        input_dim = numerical_count + cat_embedding_dim + indicator_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(hidden_dims[1], 1)
        if self.enable_reconstruction:
            self.numeric_reconstruction = nn.Linear(hidden_dims[1], numerical_count) if numerical_count else None
            self.categorical_reconstruction = nn.ModuleList(
                [nn.Linear(hidden_dims[1], cardinality) for cardinality in self.category_cardinalities]
            )
        else:
            self.numeric_reconstruction = None
            self.categorical_reconstruction = nn.ModuleList([])

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        numerical_indicator: torch.Tensor,
        categorical_indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        feature_blocks: list[torch.Tensor] = [numerical]
        if self.category_embeddings:
            feature_blocks.extend(
                [embedding(categorical[:, index]) for index, embedding in enumerate(self.category_embeddings)]
            )
        if self.use_missingness_indicators:
            feature_blocks.append(numerical_indicator)
            feature_blocks.append(categorical_indicator)
        encoded = self.encoder(torch.cat(feature_blocks, dim=1))
        logits = self.classifier_head(encoded).squeeze(1)
        numeric_reconstruction = self.numeric_reconstruction(encoded) if self.numeric_reconstruction is not None else None
        categorical_reconstruction = [head(encoded) for head in self.categorical_reconstruction]
        return logits, numeric_reconstruction, categorical_reconstruction


def train_mait_classifier(
    *,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    categorical_columns: list[str],
    numerical_columns: list[str],
    seed: int,
    use_missingness_indicators: bool,
    augmentation_columns: list[str] | None = None,
    mask_only_observed_values: bool = True,
    training_config: dict[str, Any] | None = None,
    augmentation_config: dict[str, Any] | None = None,
) -> TrainedNeuralTabularModel:
    training_config = training_config or {}
    augmentation_config = augmentation_config or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(seed)

    preprocessor = NeuralTabularPreprocessor(categorical_columns, numerical_columns)
    preprocessor.fit(train_features)
    prepared_train = preprocessor.transform(train_features)
    prepared_validation = preprocessor.transform(validation_features)
    prepared_test = preprocessor.transform(test_features)
    numerical_augmentation_mask, categorical_augmentation_mask = _augmentation_column_masks(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        augmentation_columns=augmentation_columns,
        device=device,
    )

    model = MLPMAITModel(
        category_cardinalities=preprocessor.category_cardinalities,
        numerical_count=len(numerical_columns),
        use_missingness_indicators=use_missingness_indicators,
        enable_reconstruction=bool(
            training_config.get(
                "enable_reconstruction",
                float(training_config.get("reconstruction_weight", 1.0)) > 0.0,
            )
        ),
        hidden_dims=(
            int(training_config.get("hidden_dim_1", 256)),
            int(training_config.get("hidden_dim_2", 128)),
        ),
        dropout=float(training_config.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    classification_loss = nn.BCEWithLogitsLoss()
    max_epochs = int(training_config.get("max_epochs", 60))
    batch_size = int(training_config.get("batch_size", 512))
    patience = int(training_config.get("early_stopping_patience", 8))
    reconstruction_weight = float(training_config.get("reconstruction_weight", 1.0))
    enable_reconstruction = bool(training_config.get("enable_reconstruction", reconstruction_weight > 0.0))
    use_stochastic_masking = bool(augmentation_config.get("use_stochastic_masking", True))
    mask_rates = [float(rate) for rate in augmentation_config.get("mask_rates", [0.1, 0.2, 0.3])]
    if not mask_rates:
        mask_rates = [0.0]

    train_targets = y_train.to_numpy(dtype=np.float32)
    validation_targets = y_validation.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_auroc = float("-inf")
    best_epoch = 0
    best_history: list[dict[str, float]] = []

    fit_start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_indices = rng.permutation(prepared_train.row_count)
        running_loss = 0.0
        running_batches = 0

        for start in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[start : start + batch_size]
            batch = _batch_tensors(prepared_train, batch_indices, device)
            if use_stochastic_masking:
                rate = float(rng.choice(mask_rates))
                masked_batch = _apply_random_mask(
                    batch,
                    rng=rng,
                    rate=rate,
                    use_missingness_indicators=use_missingness_indicators,
                    numerical_augmentation_mask=numerical_augmentation_mask,
                    categorical_augmentation_mask=categorical_augmentation_mask,
                    mask_only_observed_values=mask_only_observed_values,
                )
            else:
                masked_batch = _identity_mask_batch(
                    batch,
                    use_missingness_indicators=use_missingness_indicators,
                )

            optimizer.zero_grad(set_to_none=True)
            logits, numeric_reconstruction, categorical_reconstruction = model(
                masked_batch["numerical_input"],
                masked_batch["categorical_input"],
                masked_batch["numerical_indicator"],
                masked_batch["categorical_indicator"],
            )
            targets = torch.as_tensor(train_targets[batch_indices], dtype=torch.float32, device=device)
            loss = classification_loss(logits, targets)
            reconstruction_loss = (
                _reconstruction_loss(
                    numeric_reconstruction=numeric_reconstruction,
                    categorical_reconstruction=categorical_reconstruction,
                    original_batch=batch,
                    masked_batch=masked_batch,
                )
                if enable_reconstruction
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            total_loss = loss + (reconstruction_weight * reconstruction_loss)
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.detach().cpu().item())
            running_batches += 1

        validation_probabilities = _predict_probabilities(model, prepared_validation, batch_size=batch_size, device=device)
        validation_auroc = _safe_auroc(validation_targets, validation_probabilities)
        best_history.append(
            {
                "epoch": float(epoch),
                "mean_total_loss": round(running_loss / max(running_batches, 1), 6),
                "validation_auroc": round(validation_auroc, 6),
            }
        )

        if validation_auroc > best_validation_auroc + 1e-6:
            best_validation_auroc = validation_auroc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("MAIT training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    fit_seconds = round(perf_counter() - fit_start, 6)

    predict_start = perf_counter()
    validation_probabilities = _predict_probabilities(model, prepared_validation, batch_size=batch_size, device=device)
    test_probabilities = _predict_probabilities(model, prepared_test, batch_size=batch_size, device=device)
    predict_seconds = round(perf_counter() - predict_start, 6)

    metadata = {
        "implementation": "torch_mait_mlp",
        "device": str(device),
        "hidden_dims": [int(training_config.get("hidden_dim_1", 256)), int(training_config.get("hidden_dim_2", 128))],
        "dropout": float(training_config.get("dropout", 0.1)),
        "learning_rate": float(training_config.get("learning_rate", 1e-3)),
        "weight_decay": float(training_config.get("weight_decay", 1e-4)),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "best_epoch": best_epoch,
        "best_validation_auroc": round(best_validation_auroc, 6),
        "mask_rates": mask_rates,
        "augmentation_columns": list(augmentation_columns or categorical_columns + numerical_columns),
        "mask_only_observed_values": bool(mask_only_observed_values),
        "uses_stochastic_masking": use_stochastic_masking,
        "reconstruction_weight": reconstruction_weight,
        "enable_reconstruction": enable_reconstruction,
        "uses_missingness_indicators": use_missingness_indicators,
        "category_cardinalities": preprocessor.category_cardinalities,
        "categorical_columns": list(categorical_columns),
        "numerical_columns": list(numerical_columns),
        "training_history": best_history,
    }
    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    def predict_probabilities(features: pd.DataFrame) -> np.ndarray:
        prepared_features = preprocessor.transform(features)
        return _predict_probabilities(model, prepared_features, batch_size=batch_size, device=device)

    return TrainedNeuralTabularModel(
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        model_metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        predict_probabilities=predict_probabilities,
    )


def _batch_tensors(prepared: PreparedTabularFrame, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "numerical": torch.as_tensor(prepared.numerical[indices], dtype=torch.float32, device=device),
        "numerical_missing": torch.as_tensor(prepared.numerical_missing[indices], dtype=torch.float32, device=device),
        "categorical": torch.as_tensor(prepared.categorical[indices], dtype=torch.long, device=device),
        "categorical_missing": torch.as_tensor(prepared.categorical_missing[indices], dtype=torch.float32, device=device),
    }


def _apply_random_mask(
    batch: dict[str, torch.Tensor],
    *,
    rng: np.random.Generator,
    rate: float,
    use_missingness_indicators: bool,
    numerical_augmentation_mask: torch.Tensor,
    categorical_augmentation_mask: torch.Tensor,
    mask_only_observed_values: bool,
) -> dict[str, torch.Tensor]:
    numerical_missing_mask = batch["numerical_missing"].bool()
    categorical_missing_mask = batch["categorical_missing"].bool()
    numerical_mask = _sample_mask(
        batch["numerical"].shape,
        numerical_missing_mask,
        augmentation_mask=numerical_augmentation_mask,
        rng=rng,
        rate=rate,
        device=batch["numerical"].device,
        mask_only_observed_values=mask_only_observed_values,
    )
    categorical_mask = _sample_mask(
        batch["categorical"].shape,
        categorical_missing_mask,
        augmentation_mask=categorical_augmentation_mask,
        rng=rng,
        rate=rate,
        device=batch["categorical"].device,
        mask_only_observed_values=mask_only_observed_values,
    )
    reconstruction_numerical_mask = numerical_mask & (~numerical_missing_mask)
    reconstruction_categorical_mask = categorical_mask & (~categorical_missing_mask)

    numerical_input = batch["numerical"].clone()
    numerical_input[numerical_mask] = 0.0
    categorical_input = batch["categorical"].clone()
    categorical_input[categorical_mask] = 0

    numerical_indicator = batch["numerical_missing"].clone()
    categorical_indicator = batch["categorical_missing"].clone()
    if use_missingness_indicators:
        numerical_indicator = torch.maximum(numerical_indicator, numerical_mask.float())
        categorical_indicator = torch.maximum(categorical_indicator, categorical_mask.float())

    return {
        "numerical_input": numerical_input,
        "categorical_input": categorical_input,
        "numerical_indicator": numerical_indicator,
        "categorical_indicator": categorical_indicator,
        "artificial_numerical_mask": numerical_mask,
        "artificial_categorical_mask": categorical_mask,
        "reconstruction_numerical_mask": reconstruction_numerical_mask,
        "reconstruction_categorical_mask": reconstruction_categorical_mask,
    }


def _identity_mask_batch(
    batch: dict[str, torch.Tensor],
    *,
    use_missingness_indicators: bool,
) -> dict[str, torch.Tensor]:
    numerical_indicator = batch["numerical_missing"].clone()
    categorical_indicator = batch["categorical_missing"].clone()
    if not use_missingness_indicators:
        numerical_indicator = torch.zeros_like(numerical_indicator)
        categorical_indicator = torch.zeros_like(categorical_indicator)
    return {
        "numerical_input": batch["numerical"].clone(),
        "categorical_input": batch["categorical"].clone(),
        "numerical_indicator": numerical_indicator,
        "categorical_indicator": categorical_indicator,
        "artificial_numerical_mask": torch.zeros_like(batch["numerical"], dtype=torch.bool),
        "artificial_categorical_mask": torch.zeros_like(batch["categorical"], dtype=torch.bool),
        "reconstruction_numerical_mask": torch.zeros_like(batch["numerical"], dtype=torch.bool),
        "reconstruction_categorical_mask": torch.zeros_like(batch["categorical"], dtype=torch.bool),
    }


def _sample_mask(
    shape: torch.Size,
    natural_missing: torch.Tensor,
    *,
    augmentation_mask: torch.Tensor,
    rng: np.random.Generator,
    rate: float,
    device: torch.device,
    mask_only_observed_values: bool,
) -> torch.Tensor:
    if np.prod(shape) == 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    sampled = rng.random(shape) < rate
    sampled_mask = torch.as_tensor(sampled, dtype=torch.bool, device=device)
    if augmentation_mask.numel():
        sampled_mask = sampled_mask & augmentation_mask.unsqueeze(0).expand(shape)
    if mask_only_observed_values:
        sampled_mask = sampled_mask & (~natural_missing)
    return sampled_mask


def _reconstruction_loss(
    *,
    numeric_reconstruction: torch.Tensor | None,
    categorical_reconstruction: list[torch.Tensor],
    original_batch: dict[str, torch.Tensor],
    masked_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = original_batch["numerical"].device
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    masked_entries = 0

    if numeric_reconstruction is not None and masked_batch["reconstruction_numerical_mask"].any():
        mask = masked_batch["reconstruction_numerical_mask"]
        target = original_batch["numerical"][mask]
        prediction = numeric_reconstruction[mask]
        total_loss = total_loss + torch.sum((prediction - target) ** 2)
        masked_entries += int(mask.sum().item())

    for column_index, logits in enumerate(categorical_reconstruction):
        mask = masked_batch["reconstruction_categorical_mask"][:, column_index]
        if mask.any():
            targets = original_batch["categorical"][mask, column_index]
            total_loss = total_loss + nn.functional.cross_entropy(logits[mask], targets, reduction="sum")
            masked_entries += int(mask.sum().item())

    if masked_entries == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return total_loss / float(masked_entries)


def _augmentation_column_masks(
    *,
    categorical_columns: list[str],
    numerical_columns: list[str],
    augmentation_columns: list[str] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_columns = set(augmentation_columns or categorical_columns + numerical_columns)
    numerical_mask = torch.as_tensor(
        [column_name in active_columns for column_name in numerical_columns],
        dtype=torch.bool,
        device=device,
    )
    categorical_mask = torch.as_tensor(
        [column_name in active_columns for column_name in categorical_columns],
        dtype=torch.bool,
        device=device,
    )
    return numerical_mask, categorical_mask


def _predict_probabilities(
    model: MLPMAITModel,
    prepared: PreparedTabularFrame,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, prepared.row_count, batch_size):
            stop = min(start + batch_size, prepared.row_count)
            batch_indices = np.arange(start, stop)
            batch = _batch_tensors(prepared, batch_indices, device)
            logits, _, _ = model(
                batch["numerical"],
                batch["categorical"],
                batch["numerical_missing"],
                batch["categorical_missing"],
            )
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probabilities, axis=0) if probabilities else np.asarray([], dtype=float)


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _embedding_dim(cardinality: int) -> int:
    return max(4, min(32, int(round(1.6 * (cardinality ** 0.56)))))


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
