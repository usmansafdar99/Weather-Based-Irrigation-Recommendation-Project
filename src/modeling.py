from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR
from .preprocessing import build_preprocessor


def build_model(preprocessor) -> Pipeline:
    """Create a full pipeline: preprocessing + regression model."""
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return model


def train_model(model: Pipeline, X_train, y_train) -> Pipeline:
    """Fit the model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test, y_test) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, and R² on the test set."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2


def save_model(model: Pipeline, filename: str = "water_requirement_model.joblib") -> Path:
    """Persist the trained model to disk."""
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(path: Path | str | None = None) -> Pipeline:
    """Load a trained model from disk."""
    if path is None:
        path = MODELS_DIR / "water_requirement_model.joblib"
    return joblib.load(path)

