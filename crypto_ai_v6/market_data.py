from __future__ import annotations

import asyncio
from typing import Any

import ccxt.async_support as ccxt
import pandas as pd


class MarketDataGateway:
    """Public read-only market data gateway. No authenticated trading methods exist in V6."""

    def __init__(self, exchange_id: str = "binance"):
        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls({"enableRateLimit": True})
        self.markets_loaded = False

    async def load_markets(self):
        if not self.markets_loaded:
            await self.exchange.load_markets()
            self.markets_loaded = True

    async def fetch_tickers(self) -> dict[str, dict[str, Any]]:
        await self.load_markets()
        return await self.exchange.fetch_tickers()

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return await self.exchange.fetch_order_book(symbol, limit=limit)

    async def fetch_trades(self, symbol: str, limit: int = 160) -> list[dict[str, Any]]:
        return await self.exchange.fetch_trades(symbol, limit=limit)

    async def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna().reset_index(drop=True)

    async def close(self):
        await self.exchange.close()


async def gather_limited(coros, concurrency: int):
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(one(c) for c in coros), return_exceptions=True)
