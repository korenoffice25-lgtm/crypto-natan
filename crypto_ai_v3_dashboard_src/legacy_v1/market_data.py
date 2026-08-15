import ccxt
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    return cls({"enableRateLimit": True})

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    exchange = make_exchange(exchange_id)
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
