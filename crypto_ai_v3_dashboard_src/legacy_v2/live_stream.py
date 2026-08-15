from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import time

import ccxt.pro as ccxtpro

from market_state import MarketSnapshot, orderbook_state, trade_flow_state


@dataclass
class StreamUpdate:
    snapshot: MarketSnapshot
    raw_orderbook: dict
    raw_trades: list[dict]
    raw_ohlcv: list[list]


class LiveMarketStream:
    """Public/read-only real-time market stream.

    It watches L2 order book, public trades and OHLCV. There are deliberately
    no authenticated methods and no order-creation methods in this class.
    """

    def __init__(self, exchange_id: str, symbol: str, timeframe: str = "1m", levels: int = 20, trade_window: int = 200):
        cls = getattr(ccxtpro, exchange_id)
        self.exchange = cls({"enableRateLimit": True})
        self.symbol = symbol
        self.timeframe = timeframe
        self.levels = levels
        self.trade_buffer = deque(maxlen=trade_window)
        self.last_prices = deque(maxlen=60)

    async def next(self) -> StreamUpdate:
        orderbook_task = asyncio.create_task(
            self.exchange.watch_order_book(self.symbol, limit=self.levels)
        )
        trades_task = asyncio.create_task(
            self.exchange.watch_trades(self.symbol)
        )
        ohlcv_task = asyncio.create_task(
            self.exchange.watch_ohlcv(self.symbol, timeframe=self.timeframe)
        )

        orderbook, trades, ohlcv = await asyncio.gather(
            orderbook_task, trades_task, ohlcv_task
        )

        for t in trades[-100:]:
            self.trade_buffer.append(t)

        ob = orderbook_state(orderbook, self.levels)
        tf = trade_flow_state(list(self.trade_buffer))

        # Prefer recent public-trade VWAP when available; fall back to mid.
        last = tf.vwap if tf.vwap > 0 else ob.mid
        self.last_prices.append(last)

        short_return = 0.0
        realized_vol = 0.0
        if len(self.last_prices) >= 2:
            prices = list(self.last_prices)
            short_return = prices[-1] / prices[-2] - 1
            rets = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
            if len(rets) > 2:
                mean = sum(rets) / len(rets)
                realized_vol = (
                    sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
                ) ** 0.5

        ts = int(orderbook.get("timestamp") or time.time() * 1000)
        snap = MarketSnapshot(
            timestamp_ms=ts,
            symbol=self.symbol,
            last=last,
            orderbook=ob,
            trades=tf,
            short_return=short_return,
            realized_vol=realized_vol,
        )
        return StreamUpdate(snap, orderbook, trades, ohlcv)

    async def close(self):
        await self.exchange.close()
