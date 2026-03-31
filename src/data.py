from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import DATA_PATH, IRRIGATION_NEED_TO_WATER_SCORE, TARGET_COL_CATEGORICAL, TARGET_COL_NUMERIC


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the irrigation dataset from CSV."""
    csv_path = Path(path) if path is not None else DATA_PATH
    df = pd.read_csv(csv_path)
    return df


def add_numeric_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric water requirement score column from categorical irrigation need."""
    df = df.copy()
    if TARGET_COL_CATEGORICAL not in df.columns:
        raise ValueError(f"Expected column '{TARGET_COL_CATEGORICAL}' not found in dataset.")

    df[TARGET_COL_NUMERIC] = df[TARGET_COL_CATEGORICAL].map(IRRIGATION_NEED_TO_WATER_SCORE)
    if df[TARGET_COL_NUMERIC].isna().any():
        unknown = df.loc[df[TARGET_COL_NUMERIC].isna(), TARGET_COL_CATEGORICAL].unique()
        raise ValueError(
            f"Found unknown irrigation need labels with no mapping: {unknown}. "
            "Please update IRRIGATION_NEED_TO_WATER_SCORE in config."
        )
    return df


def train_test_split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test sets."""
    from sklearn.model_selection import train_test_split

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

