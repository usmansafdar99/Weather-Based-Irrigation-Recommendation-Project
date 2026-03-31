import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_PATH = BASE_DIR / "Dataset" / "irrigation_prediction.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Target and feature configuration
TARGET_COL_CATEGORICAL = "Irrigation_Need"

# Derived numeric target for regression (water requirement score, in mm-equivalent units)
IRRIGATION_NEED_TO_WATER_SCORE = {
    "Low": 2.0,
    "Medium": 6.0,
    "High": 12.0,
}
TARGET_COL_NUMERIC = "Water_Requirement_Score"

# Assumed soil field capacity (percentage or volumetric units consistent with Soil_Moisture)
SOIL_FIELD_CAPACITY = 60.0

# Irrigation decision thresholds based on predicted water requirement score (mm/day)
LOW_THRESHOLD = 3.0
HIGH_THRESHOLD = 8.0

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

