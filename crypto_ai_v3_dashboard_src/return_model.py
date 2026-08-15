from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from features_v2 import BASE_FEATURES


@dataclass
class HorizonPrediction:
    horizon: int
    expected_return: float
    validation_mae: float


class MultiHorizonReturnModel:
    """Learns forward returns from market-state features.

    This is intentionally not a hard-coded trend strategy. Each horizon has a
    separate nonlinear model. The decision agent can abstain when predictions
    disagree or are too small relative to costs.
    """

    def __init__(self, horizons=(1, 3, 12)):
        self.horizons = tuple(horizons)
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.validation_mae: dict[int, float] = {}

    def fit(self, df: pd.DataFrame):
        n = len(df)
        if n < 500:
            raise ValueError("Need at least 500 rows to fit return models")

        split = max(int(n * 0.80), n - 300)
        train = df.iloc[:split]
        valid = df.iloc[split:]

        X_train = train[BASE_FEATURES]
        X_valid = valid[BASE_FEATURES]

        for h in self.horizons:
            y_train = train[f"future_return_{h}"]
            y_valid = valid[f"future_return_{h}"]

            model = HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=180,
                max_depth=4,
                l2_regularization=2.0,
                random_state=100 + h,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            self.models[h] = model
            self.validation_mae[h] = float(mean_absolute_error(y_valid, pred))
        return self

    def predict(self, row: pd.DataFrame) -> list[HorizonPrediction]:
        out = []
        for h in self.horizons:
            model = self.models[h]
            pred = float(model.predict(row[BASE_FEATURES])[0])
            out.append(
                HorizonPrediction(
                    horizon=h,
                    expected_return=pred,
                    validation_mae=self.validation_mae[h],
                )
            )
        return out
