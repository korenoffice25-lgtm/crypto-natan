from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


REGIME_FEATURES = [
    "realized_vol_12",
    "realized_vol_36",
    "abs_ret_1",
    "range_pct",
    "volume_z_24",
]


@dataclass
class RegimeReading:
    cluster: int
    confidence: float
    volatility_rank: float
    activity_rank: float


class RegimeModel:
    """Unsupervised market-state detector.

    It does not encode 'bull market' or 'bear market' rules. It clusters states
    from volatility, range and activity, then exposes the current state's
    confidence and relative intensity.
    """

    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
        )
        self._means = None
        self._stds = None

    def fit(self, df: pd.DataFrame):
        X = df[REGIME_FEATURES].astype(float).to_numpy()
        self._means = X.mean(axis=0)
        self._stds = X.std(axis=0)
        Z = (X - self._means) / np.where(self._stds == 0, 1, self._stds)
        self.model.fit(Z)
        return self

    def read(self, row: pd.DataFrame) -> RegimeReading:
        if self._means is None:
            raise RuntimeError("RegimeModel is not fitted")
        X = row[REGIME_FEATURES].astype(float).to_numpy()
        Z = (X - self._means) / np.where(self._stds == 0, 1, self._stds)
        probs = self.model.predict_proba(Z)[0]
        cluster = int(np.argmax(probs))

        vol = float(row["realized_vol_36"].iloc[0])
        activity = float(abs(row["volume_z_24"].iloc[0]))

        # Squashed intensity indicators for agent context.
        volatility_rank = float(np.tanh(max(vol, 0.0) * 100))
        activity_rank = float(np.tanh(max(activity, 0.0) / 2))

        return RegimeReading(
            cluster=cluster,
            confidence=float(probs[cluster]),
            volatility_rank=volatility_rank,
            activity_rank=activity_rank,
        )
