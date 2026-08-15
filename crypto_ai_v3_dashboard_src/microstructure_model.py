from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


MICRO_MODEL_FEATURES = [
    "ob_spread_bps",
    "ob_imbalance_5",
    "ob_imbalance_20",
    "ob_microprice_edge_bps",
    "tf_flow_imbalance",
    "log_tf_avg_trade_notional",
    "log_ob_bid_depth_5",
    "log_ob_ask_depth_5",
    "log_ob_bid_depth_20",
    "log_ob_ask_depth_20",
    "short_return",
    "realized_vol",
]


@dataclass
class MicroPrediction:
    horizon_seconds: int
    expected_return: float
    validation_mae: float


def load_snapshots(db_path: str, symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT timestamp_ms, payload_json FROM market_snapshots "
            "WHERE symbol=? ORDER BY timestamp_ms",
            (symbol,),
        ).fetchall()
    finally:
        conn.close()

    parsed = []
    for ts, payload in rows:
        d = json.loads(payload)
        d["timestamp_ms"] = int(ts)
        parsed.append(d)

    if not parsed:
        return pd.DataFrame()
    return pd.DataFrame(parsed).drop_duplicates("timestamp_ms", keep="last")


def engineer_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp_ms").reset_index(drop=True)
    for c in [
        "tf_avg_trade_notional",
        "ob_bid_depth_5", "ob_ask_depth_5",
        "ob_bid_depth_20", "ob_ask_depth_20",
    ]:
        if c not in x:
            x[c] = 0.0

    x["log_tf_avg_trade_notional"] = np.log1p(x["tf_avg_trade_notional"].clip(lower=0))
    x["log_ob_bid_depth_5"] = np.log1p(x["ob_bid_depth_5"].clip(lower=0))
    x["log_ob_ask_depth_5"] = np.log1p(x["ob_ask_depth_5"].clip(lower=0))
    x["log_ob_bid_depth_20"] = np.log1p(x["ob_bid_depth_20"].clip(lower=0))
    x["log_ob_ask_depth_20"] = np.log1p(x["ob_ask_depth_20"].clip(lower=0))

    return x.replace([np.inf, -np.inf], np.nan)


def add_time_targets(df: pd.DataFrame, horizons_seconds=(30, 120, 300)) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp_ms").reset_index(drop=True)
    ts = x["timestamp_ms"].to_numpy(dtype=np.int64)
    prices = x["last"].to_numpy(dtype=float)

    for seconds in horizons_seconds:
        target_ts = ts + int(seconds * 1000)
        idx = np.searchsorted(ts, target_ts, side="left")
        valid = idx < len(x)
        future = np.full(len(x), np.nan)
        future[valid] = prices[idx[valid]]
        x[f"future_return_{seconds}s"] = future / prices - 1.0

    return x


class MicrostructureReturnModel:
    def __init__(self, horizons_seconds=(30, 120, 300)):
        self.horizons_seconds = tuple(horizons_seconds)
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.validation_mae: dict[int, float] = {}
        self.directional_accuracy: dict[int, float] = {}

    def fit(self, df: pd.DataFrame):
        required_targets = [f"future_return_{s}s" for s in self.horizons_seconds]
        clean = df.dropna(subset=MICRO_MODEL_FEATURES + required_targets).reset_index(drop=True)
        if len(clean) < 1000:
            raise ValueError(
                f"Need at least 1000 labeled microstructure snapshots; found {len(clean)}"
            )

        split = int(len(clean) * 0.8)
        train, valid = clean.iloc[:split], clean.iloc[split:]

        for seconds in self.horizons_seconds:
            target = f"future_return_{seconds}s"
            model = HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=220,
                max_depth=5,
                l2_regularization=3.0,
                random_state=200 + seconds,
            )
            model.fit(train[MICRO_MODEL_FEATURES], train[target])
            pred = model.predict(valid[MICRO_MODEL_FEATURES])
            actual = valid[target].to_numpy()
            self.models[seconds] = model
            self.validation_mae[seconds] = float(mean_absolute_error(actual, pred))
            self.directional_accuracy[seconds] = float(np.mean(np.sign(pred) == np.sign(actual)))
        return self

    def predict(self, row: pd.DataFrame) -> list[MicroPrediction]:
        out = []
        for seconds in self.horizons_seconds:
            out.append(MicroPrediction(
                horizon_seconds=seconds,
                expected_return=float(self.models[seconds].predict(row[MICRO_MODEL_FEATURES])[0]),
                validation_mae=self.validation_mae[seconds],
            ))
        return out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


def train_from_sqlite(db_path: str, symbol: str, output_path: str, horizons_seconds=(30, 120, 300)):
    snapshots = load_snapshots(db_path, symbol)
    if snapshots.empty:
        raise ValueError(f"No snapshots found for {symbol}")
    features = engineer_micro_features(snapshots)
    labeled = add_time_targets(features, horizons_seconds)
    model = MicrostructureReturnModel(horizons_seconds).fit(labeled)
    model.save(output_path)
    return model, len(labeled)
