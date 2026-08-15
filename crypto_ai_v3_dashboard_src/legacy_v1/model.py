from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from features import FEATURE_COLUMNS

@dataclass
class SignalModel:
    model: HistGradientBoostingClassifier | None = None

    def train(self, df: pd.DataFrame):
        if len(df) < 300:
            raise ValueError("Need at least 300 feature rows to train.")
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_depth=4,
            l2_regularization=1.0,
            random_state=42,
        )
        self.model.fit(df[FEATURE_COLUMNS], df["target"])
        return self

    def probability_up(self, row: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        return float(self.model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])

def walk_forward_probabilities(df: pd.DataFrame, min_train: int = 500, retrain_every: int = 72):
    probs = np.full(len(df), np.nan)
    model = None

    for i in range(min_train, len(df)):
        if model is None or (i - min_train) % retrain_every == 0:
            train = df.iloc[:i].copy()
            model = SignalModel().train(train)
        probs[i] = model.probability_up(df.iloc[[i]])

    out = df.copy()
    out["prob_up"] = probs
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)
