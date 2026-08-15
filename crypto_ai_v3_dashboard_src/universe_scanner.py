from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP", "PYUSD", "EUR", "USD"
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")


@dataclass
class Candidate:
    symbol: str
    base: str
    quote: str
    quote_volume: float
    intraday_range_pct: float
    spread_bps: float
    depth_notional: float
    opportunity_score: float
    risk_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value, default=0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def eligible_market(market: dict[str, Any], quote_currency: str) -> bool:
    if not market.get("spot", False):
        return False
    if market.get("active") is False:
        return False
    if market.get("quote") != quote_currency:
        return False

    base = str(market.get("base") or "").upper()
    if not base or base in STABLE_BASES:
        return False
    if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
        return False
    return True


def ticker_activity(ticker: dict[str, Any]) -> tuple[float, float]:
    last = _f(ticker.get("last"))
    high = _f(ticker.get("high"))
    low = _f(ticker.get("low"))
    quote_volume = _f(ticker.get("quoteVolume"))

    if quote_volume <= 0:
        # Some exchanges expose only baseVolume. Estimate quote notional.
        quote_volume = _f(ticker.get("baseVolume")) * max(last, 0.0)

    range_pct = ((high - low) / last) if last > 0 and high > 0 and low > 0 else 0.0
    return quote_volume, max(0.0, range_pct)


def prefilter_score(quote_volume: float, intraday_range_pct: float) -> float:
    # Direction-neutral: reward liquidity and movement, not whether price rose or fell.
    liquidity = math.log10(max(quote_volume, 1.0))
    activity = min(intraday_range_pct, 0.30) * 12.0
    return liquidity + activity


def book_quality(orderbook: dict[str, Any], levels: int = 10) -> tuple[float, float]:
    bids = (orderbook.get("bids") or [])[:levels]
    asks = (orderbook.get("asks") or [])[:levels]
    if not bids or not asks:
        return 10_000.0, 0.0

    best_bid = _f(bids[0][0])
    best_ask = _f(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 10_000.0

    bid_depth = sum(_f(r[0]) * _f(r[1]) for r in bids)
    ask_depth = sum(_f(r[0]) * _f(r[1]) for r in asks)
    return max(0.0, spread_bps), bid_depth + ask_depth


def final_score(
    quote_volume: float,
    intraday_range_pct: float,
    spread_bps: float,
    depth_notional: float,
) -> float:
    liquidity = math.log10(max(quote_volume, 1.0))
    movement = min(intraday_range_pct, 0.30) * 10.0
    depth = math.log10(max(depth_notional, 1.0)) * 0.45
    spread_penalty = min(spread_bps, 100.0) * 0.035
    return liquidity + movement + depth - spread_penalty


def risk_multiplier(quote_volume: float, spread_bps: float, depth_notional: float) -> float:
    # Smaller/less-liquid markets automatically receive less capital.
    volume_component = min(1.0, max(0.25, math.log10(max(quote_volume, 1.0)) / 9.0))
    depth_component = min(1.0, max(0.25, math.log10(max(depth_notional, 1.0)) / 7.0))
    spread_component = max(0.20, 1.0 - min(spread_bps, 50.0) / 60.0)
    return max(0.15, min(1.0, volume_component * depth_component * spread_component))


class UniverseScanner:
    def __init__(self, gateway, settings):
        self.gateway = gateway
        self.cfg = settings

    async def scan(self) -> list[Candidate]:
        markets = await self.gateway.load_markets()
        tickers = await self.gateway.fetch_tickers()

        stage1 = []
        for symbol, market in markets.items():
            if not eligible_market(market, self.cfg.quote_currency):
                continue
            ticker = tickers.get(symbol) or {}
            quote_volume, range_pct = ticker_activity(ticker)
            if quote_volume < self.cfg.min_quote_volume:
                continue
            stage1.append((
                prefilter_score(quote_volume, range_pct),
                symbol,
                market,
                quote_volume,
                range_pct,
            ))

        stage1.sort(reverse=True, key=lambda x: x[0])
        stage1 = stage1[: self.cfg.scanner_prefilter_size]

        candidates: list[Candidate] = []
        for _, symbol, market, quote_volume, range_pct in stage1:
            try:
                orderbook = await self.gateway.fetch_order_book(symbol, self.cfg.orderbook_levels)
                spread_bps, depth_notional = book_quality(orderbook, min(self.cfg.orderbook_levels, 10))
            except Exception:
                continue

            if spread_bps > self.cfg.max_spread_bps:
                continue
            if depth_notional < self.cfg.min_depth_notional:
                continue

            score = final_score(quote_volume, range_pct, spread_bps, depth_notional)
            candidates.append(Candidate(
                symbol=symbol,
                base=str(market.get("base") or ""),
                quote=str(market.get("quote") or ""),
                quote_volume=quote_volume,
                intraday_range_pct=range_pct,
                spread_bps=spread_bps,
                depth_notional=depth_notional,
                opportunity_score=score,
                risk_multiplier=risk_multiplier(quote_volume, spread_bps, depth_notional),
            ))

        candidates.sort(reverse=True, key=lambda c: c.opportunity_score)
        return candidates[: self.cfg.active_universe_size]
