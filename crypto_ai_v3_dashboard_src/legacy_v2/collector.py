from __future__ import annotations

import asyncio
from config import SETTINGS
from live_stream import LiveMarketStream
from storage import Storage


async def collect_symbol(symbol: str, storage: Storage):
    stream = LiveMarketStream(
        SETTINGS.exchange_id,
        symbol,
        timeframe=SETTINGS.timeframe,
        levels=SETTINGS.orderbook_levels,
    )
    try:
        while True:
            update = await stream.next()
            snap = update.snapshot
            storage.add_snapshot(snap.timestamp_ms, symbol, snap.flatten())
            print(
                f"{symbol} last={snap.last:.2f} "
                f"spread={snap.orderbook.spread_bps:.2f}bps "
                f"imbalance={snap.orderbook.imbalance_20:+.3f} "
                f"flow={snap.trades.flow_imbalance:+.3f}"
            )
            await asyncio.sleep(SETTINGS.snapshot_interval_seconds)
    finally:
        await stream.close()


async def main():
    storage = Storage(SETTINGS.db_path)
    try:
        await asyncio.gather(*(collect_symbol(s, storage) for s in SETTINGS.symbols))
    finally:
        storage.close()


if __name__ == "__main__":
    asyncio.run(main())
