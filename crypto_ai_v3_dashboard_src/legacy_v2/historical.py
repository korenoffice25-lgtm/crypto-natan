from __future__ import annotations

import ccxt
import pandas as pd


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def exchange_instance(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True})


def fetch_history(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex = exchange_instance(exchange_id)
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
