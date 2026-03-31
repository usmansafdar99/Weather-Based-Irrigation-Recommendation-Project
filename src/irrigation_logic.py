from __future__ import annotations

from dataclasses import dataclass

from .config import HIGH_THRESHOLD, LOW_THRESHOLD


@dataclass
class IrrigationRecommendation:
    water_requirement_daily: float
    water_requirement_weekly: float
    level: str
    message: str
    color: str


def classify_water_requirement(water_requirement_daily: float) -> IrrigationRecommendation:
    """Convert numeric water requirement into human-readable irrigation advice."""
    weekly = water_requirement_daily * 7.0

    if water_requirement_daily < LOW_THRESHOLD:
        level = "Low"
        message = "No irrigation needed today."
        color = "green"
    elif water_requirement_daily < HIGH_THRESHOLD:
        level = "Medium"
        message = "Irrigate partially to maintain optimal moisture."
        color = "yellow"
    else:
        level = "High"
        message = "Irrigate fully to meet crop water demand."
        color = "red"

    return IrrigationRecommendation(
        water_requirement_daily=float(water_requirement_daily),
        water_requirement_weekly=float(weekly),
        level=level,
        message=message,
        color=color,
    )

