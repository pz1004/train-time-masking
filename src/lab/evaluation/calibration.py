from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str
    metadata: dict[str, Any]
    software_versions: dict[str, str]
    fit_seconds: float
    _model: Any

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = _clip_probabilities(probabilities)
        if self.method == "sigmoid":
            logits = _probabilities_to_logits(clipped).reshape(-1, 1)
            return self._model.predict_proba(logits)[:, 1]
        if self.method == "isotonic":
            return np.clip(np.asarray(self._model.predict(clipped), dtype=float), 0.0, 1.0)
        raise ValueError(f"Unsupported calibration method: {self.method}")


def fit_probability_calibrator(
    method: str,
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    seed: int,
) -> ProbabilityCalibrator:
    fit_start = perf_counter()
    validation_targets = np.asarray(y_validation, dtype=int)
    clipped_probabilities = _clip_probabilities(validation_probabilities)

    if method == "sigmoid":
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        model.fit(_probabilities_to_logits(clipped_probabilities).reshape(-1, 1), validation_targets)
        metadata = {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "fit_representation": "clipped_logit",
            "clip_epsilon": 1e-6,
        }
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(clipped_probabilities, validation_targets)
        metadata = {
            "implementation": "sklearn.isotonic.IsotonicRegression",
            "fit_representation": "probability",
            "out_of_bounds": "clip",
            "clip_epsilon": 1e-6,
        }
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    fit_seconds = round(perf_counter() - fit_start, 6)
    versions = {
        "sklearn": __import__("sklearn").__version__,
        "numpy": np.__version__,
    }
    return ProbabilityCalibrator(
        method=method,
        metadata=metadata,
        software_versions=versions,
        fit_seconds=fit_seconds,
        _model=model,
    )


def _clip_probabilities(probabilities: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(probabilities)
    return np.log(clipped / (1.0 - clipped))
