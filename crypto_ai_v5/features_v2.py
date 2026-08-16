from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_12",
    "ret_36",
    "abs_ret_1",
    "realized_vol_12",
    "realized_vol_36",
    "range_pct",
    "body_pct",
    "volume_z_24",
    "volume_change",
    "atr_pct",
    "rsi_centered",
    "ema_distance_fast",
    "ema_distance_slow",
]

MICRO_FEATURES = [
    "ob_spread_bps",
    "ob_imbalance_5",
    "ob_imbalance_20",
    "ob_microprice_edge_bps",
    "tf_flow_imbalance",
    "tf_avg_trade_notional",
    "realized_vol",
    "short_return",
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_market_features(
    df: pd.DataFrame,
    horizons=(1, 3, 12),
    include_targets: bool = True,
) -> pd.DataFrame:
    """Create market-state features.

    Training mode adds forward-return targets and therefore drops the final
    horizon rows. Inference mode keeps the newest fully formed feature row.
    """
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)

    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_12"] = x["close"].pct_change(12)
    x["ret_36"] = x["close"].pct_change(36)
    x["abs_ret_1"] = x["ret_1"].abs()

    x["realized_vol_12"] = x["ret_1"].rolling(12).std()
    x["realized_vol_36"] = x["ret_1"].rolling(36).std()
    x["range_pct"] = (x["high"] - x["low"]) / x["close"]
    x["body_pct"] = (x["close"] - x["open"]) / x["open"]

    rolling_vol = x["volume"].rolling(24)
    x["volume_z_24"] = (x["volume"] - rolling_vol.mean()) / rolling_vol.std().replace(0, np.nan)
    x["volume_change"] = x["volume"].pct_change().clip(-10, 10)

    x["atr_pct"] = _atr(x, 14) / x["close"]
    x["rsi_centered"] = (_rsi(x["close"], 14) - 50.0) / 50.0

    ema_fast = x["close"].ewm(span=10, adjust=False).mean()
    ema_slow = x["close"].ewm(span=40, adjust=False).mean()
    x["ema_distance_fast"] = x["close"] / ema_fast - 1
    x["ema_distance_slow"] = x["close"] / ema_slow - 1

    if include_targets:
        for h in horizons:
            x[f"future_return_{h}"] = x["close"].shift(-h) / x["close"] - 1.0

    x = x.replace([np.inf, -np.inf], np.nan)

    required = BASE_FEATURES.copy()
    if include_targets:
        required += [f"future_return_{h}" for h in horizons]

    return x.dropna(subset=required).reset_index(drop=True)
