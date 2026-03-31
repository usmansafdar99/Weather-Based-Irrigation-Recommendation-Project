from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.data import add_numeric_target, load_dataset
from src.preprocessing import engineer_features


def main() -> None:
    df = load_dataset()
    df = add_numeric_target(df)
    df = engineer_features(df)

    # Weather vs soil moisture
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Rainfall_mm", y="Soil_Moisture", hue="Irrigation_Need", alpha=0.5)
    plt.title("Rainfall vs Soil Moisture")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rainfall_vs_soil_moisture.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Temperature_C", y="Soil_Moisture", hue="Irrigation_Need", alpha=0.5)
    plt.title("Temperature vs Soil Moisture")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature_vs_soil_moisture.png")
    plt.close()

    # Correlation heatmap for numeric features
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()

    # Seasonal trends: average water requirement per season
    season_stats = df.groupby("Season")["Water_Requirement_Score"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=season_stats, x="Season", y="Water_Requirement_Score")
    plt.title("Average Water Requirement by Season")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "seasonal_water_requirement.png")
    plt.close()

    print(f"EDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

