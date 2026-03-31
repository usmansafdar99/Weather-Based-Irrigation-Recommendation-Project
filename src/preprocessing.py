from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SOIL_FIELD_CAPACITY


NUMERIC_FEATURES = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
    # engineered numeric features will be added later
]

CATEGORY_FEATURES = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired engineered features."""
    df = df.copy()

    # Soil moisture deficit relative to assumed field capacity
    df["Soil_Moisture_Deficit"] = np.maximum(0.0, SOIL_FIELD_CAPACITY - df["Soil_Moisture"])

    # Simple rainfall intensity per hectare
    df["Rainfall_per_ha"] = df["Rainfall_mm"] / df["Field_Area_hectare"].replace(0, np.nan)
    df["Rainfall_per_ha"] = df["Rainfall_per_ha"].fillna(0.0)

    # Temperature-humidity index proxy
    df["Temp_Humidity_Index"] = df["Temperature_C"] * (df["Humidity"] / 100.0)

    # Proxy for seasonal/region water stress: group-wise mean rainfall difference
    if {"Region", "Season", "Rainfall_mm"}.issubset(df.columns):
        group_means = (
            df.groupby(["Region", "Season"])["Rainfall_mm"].transform("mean")
        )
        df["Rainfall_Anomaly"] = df["Rainfall_mm"] - group_means
    else:
        df["Rainfall_Anomaly"] = 0.0

    return df


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric and categorical preprocessing."""
    # Extend numeric features with engineered ones
    extended_numeric = list(numeric_features) + [
        "Soil_Moisture_Deficit",
        "Rainfall_per_ha",
        "Temp_Humidity_Index",
        "Rainfall_Anomaly",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, extended_numeric),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical feature names used in the model."""
    return NUMERIC_FEATURES, CATEGORY_FEATURES

