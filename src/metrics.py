from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


