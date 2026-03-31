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


DEFAULT_MODEL_FILENAME = "water_requirement_model.joblib"


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
        path = MODELS_DIR / DEFAULT_MODEL_FILENAME
    return joblib.load(path)


def train_and_save_default_model(data_path: Path | str | None = None) -> Pipeline:
    """
    Train a model from the dataset and save it to MODELS_DIR.

    This is mainly used for Streamlit Community Cloud deployments where
    the trained artifact is not committed to the repository.
    """
    from . import config
    from .data import add_numeric_target, load_dataset, train_test_split_xy
    from .preprocessing import engineer_features, get_feature_lists

    df = load_dataset(data_path)
    df = add_numeric_target(df)
    df = engineer_features(df)

    numeric_features, categorical_features = get_feature_lists()
    feature_cols = numeric_features + categorical_features + [
        "Soil_Moisture_Deficit",
        "Rainfall_per_ha",
        "Temp_Humidity_Index",
        "Rainfall_Anomaly",
    ]

    X_train, X_test, y_train, y_test = train_test_split_xy(
        df,
        feature_cols=feature_cols,
        target_col=config.TARGET_COL_NUMERIC,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(preprocessor)
    model = train_model(model, X_train, y_train)

    # Save for reuse (even if ephemeral on Streamlit Cloud)
    save_model(model, filename=DEFAULT_MODEL_FILENAME)
    return model

