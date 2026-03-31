from __future__ import annotations

import pprint

import pandas as pd

from src import config
from src.data import add_numeric_target, load_dataset, train_test_split_xy
from src.metrics import RegressionMetrics
from src.modeling import build_model, evaluate_model, save_model, train_model
from src.preprocessing import engineer_features, get_feature_lists, build_preprocessor


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    # Create numeric target for regression
    df = add_numeric_target(df)

    # Feature engineering
    df = engineer_features(df)

    # Features and target
    numeric_features, categorical_features = get_feature_lists()
    feature_cols = numeric_features + categorical_features + [
        "Soil_Moisture_Deficit",
        "Rainfall_per_ha",
        "Temp_Humidity_Index",
        "Rainfall_Anomaly",
    ]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(
        df,
        feature_cols=feature_cols,
        target_col=config.TARGET_COL_NUMERIC,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Build preprocessing + model pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(preprocessor)

    # Train
    print("Training model...")
    model = train_model(model, X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    metrics = RegressionMetrics(mae=mae, rmse=rmse, r2=r2)
    print("Test metrics:")
    pprint.pprint(metrics)

    # Save
    path = save_model(model)
    print(f"Saved trained model to: {path}")


if __name__ == "__main__":
    main()

