import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret_1"] = x["close"].pct_change()
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_12"] = x["close"].pct_change(12)

    x["ema_12"] = x["close"].ewm(span=12, adjust=False).mean()
    x["ema_26"] = x["close"].ewm(span=26, adjust=False).mean()
    x["ema_gap"] = (x["ema_12"] / x["ema_26"]) - 1

    x["rsi_14"] = rsi(x["close"], 14) / 100.0
    x["atr_14"] = atr(x, 14)
    x["atr_pct"] = x["atr_14"] / x["close"]

    vol_mean = x["volume"].rolling(24).mean()
    vol_std = x["volume"].rolling(24).std()
    x["volume_z"] = (x["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    x["volatility_24"] = x["ret_1"].rolling(24).std()
    x["hour"] = x["timestamp"].dt.hour / 23.0

    # Prediction target: whether next candle closes higher.
    x["target"] = (x["close"].shift(-1) > x["close"]).astype(int)

    return x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_12", "ema_gap", "rsi_14",
    "atr_pct", "volume_z", "volatility_24", "hour",
]
