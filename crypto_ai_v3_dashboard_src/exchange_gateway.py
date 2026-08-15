from __future__ import annotations

import ccxt.async_support as ccxt
import pandas as pd


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class ExchangeGateway:
    """Public/read-only gateway for paper mode.

    This class deliberately exposes no create_order / cancel_order methods.
    """

    def __init__(self, exchange_id: str):
        cls = getattr(ccxt, exchange_id)
        self.exchange = cls({"enableRateLimit": True})

    async def load_markets(self):
        return await self.exchange.load_markets()

    async def fetch_tickers(self):
        return await self.exchange.fetch_tickers()

    async def fetch_order_book(self, symbol: str, limit: int = 20):
        return await self.exchange.fetch_order_book(symbol, limit=limit)

    async def fetch_trades(self, symbol: str, limit: int = 100):
        return await self.exchange.fetch_trades(symbol, limit=limit)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        return await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = await self.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def close(self):
        await self.exchange.close()
