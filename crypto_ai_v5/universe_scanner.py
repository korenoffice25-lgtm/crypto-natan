from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
import math
from typing import Any


STABLE_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURC",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "5L", "5S")


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
    radar_rank: int = 0
    deep_rank: int = 0
    is_major: bool = False
    fast_cabinet: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RadarItem:
    symbol: str
    base: str
    quote_volume: float
    intraday_range_pct: float
    change_pct: float
    radar_score: float
    radar_rank: int
    is_major: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanBundle:
    radar: list[RadarItem]
    deep: list[Candidate]
    fast_cabinet: list[Candidate]
    breadth_positive_pct: float
    median_change_pct: float
    median_range_pct: float


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
    if str(market.get("quote") or "").upper() != quote_currency.upper():
        return False
    base = str(market.get("base") or "").upper()
    if not base or base in STABLE_BASES:
        return False
    if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
        return False
    return True


def ticker_activity(ticker: dict[str, Any]) -> tuple[float, float, float]:
    last = _f(ticker.get("last"))
    high = _f(ticker.get("high"))
    low = _f(ticker.get("low"))
    open_ = _f(ticker.get("open"))
    quote_volume = _f(ticker.get("quoteVolume"))
    if quote_volume <= 0:
        quote_volume = _f(ticker.get("baseVolume")) * max(last, 0.0)
    range_pct = ((high - low) / last) if last > 0 and high > 0 and low > 0 else 0.0
    pct = ticker.get("percentage")
    if pct is not None:
        change_pct = _f(pct) / 100.0
    else:
        change_pct = (last / open_ - 1.0) if open_ > 0 and last > 0 else 0.0
    return quote_volume, max(0.0, range_pct), change_pct


def radar_score(quote_volume: float, intraday_range_pct: float, change_pct: float) -> float:
    liquidity = math.log10(max(quote_volume, 1.0))
    movement = min(intraday_range_pct, 0.40) * 14.0
    directional_activity = min(abs(change_pct), 0.30) * 3.0
    return liquidity + movement + directional_activity


def book_quality(orderbook: dict[str, Any], levels: int = 10) -> tuple[float, float]:
    bids = (orderbook.get("bids") or [])[:levels]
    asks = (orderbook.get("asks") or [])[:levels]
    if not bids or not asks:
        return 10_000.0, 0.0
    best_bid, best_ask = _f(bids[0][0]), _f(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 10_000.0
    bid_depth = sum(_f(r[0]) * _f(r[1]) for r in bids)
    ask_depth = sum(_f(r[0]) * _f(r[1]) for r in asks)
    return max(0.0, spread_bps), bid_depth + ask_depth


def final_score(quote_volume: float, intraday_range_pct: float, spread_bps: float, depth_notional: float) -> float:
    liquidity = math.log10(max(quote_volume, 1.0))
    movement = min(intraday_range_pct, 0.35) * 11.0
    depth = math.log10(max(depth_notional, 1.0)) * 0.50
    spread_penalty = min(spread_bps, 100.0) * 0.035
    return liquidity + movement + depth - spread_penalty


def risk_multiplier(quote_volume: float, spread_bps: float, depth_notional: float) -> float:
    volume_component = min(1.0, max(0.25, math.log10(max(quote_volume, 1.0)) / 9.0))
    depth_component = min(1.0, max(0.25, math.log10(max(depth_notional, 1.0)) / 7.0))
    spread_component = max(0.20, 1.0 - min(spread_bps, 50.0) / 60.0)
    return max(0.15, min(1.0, volume_component * depth_component * spread_component))


class UniverseScanner:
    """Three-stage scanner: broad Radar, deep liquidity analysis, Fast Cabinet."""

    def __init__(self, gateway, settings):
        self.gateway = gateway
        self.cfg = settings

    async def _deep_candidate(self, entry: tuple, sem: asyncio.Semaphore) -> Candidate | None:
        _, symbol, market, quote_volume, range_pct, radar_rank, is_major = entry
        async with sem:
            try:
                orderbook = await self.gateway.fetch_order_book(symbol, self.cfg.orderbook_levels)
                spread_bps, depth_notional = book_quality(orderbook, min(self.cfg.orderbook_levels, 10))
            except Exception:
                return None
        # Majors are retained for monitoring with a low risk multiplier even if temporarily thin/wide.
        if not is_major and spread_bps > self.cfg.max_spread_bps:
            return None
        if not is_major and depth_notional < self.cfg.min_depth_notional:
            return None
        score = final_score(quote_volume, range_pct, spread_bps, depth_notional)
        return Candidate(
            symbol=symbol,
            base=str(market.get("base") or ""),
            quote=str(market.get("quote") or ""),
            quote_volume=quote_volume,
            intraday_range_pct=range_pct,
            spread_bps=spread_bps,
            depth_notional=depth_notional,
            opportunity_score=score,
            risk_multiplier=risk_multiplier(quote_volume, spread_bps, depth_notional),
            radar_rank=radar_rank,
            is_major=is_major,
        )

    async def scan(self) -> ScanBundle:
        markets = await self.gateway.load_markets()
        tickers = await self.gateway.fetch_tickers()
        majors = set(self.cfg.major_bases)

        stage1: list[tuple] = []
        changes: list[float] = []
        ranges: list[float] = []
        for symbol, market in markets.items():
            if not eligible_market(market, self.cfg.quote_currency):
                continue
            ticker = tickers.get(symbol) or {}
            quote_volume, range_pct, change_pct = ticker_activity(ticker)
            base = str(market.get("base") or "").upper()
            is_major = base in majors
            if quote_volume < self.cfg.min_quote_volume and not is_major:
                continue
            score = radar_score(quote_volume, range_pct, change_pct)
            stage1.append((score, symbol, market, quote_volume, range_pct, change_pct, is_major))
            changes.append(change_pct)
            ranges.append(range_pct)

        stage1.sort(key=lambda x: x[0], reverse=True)
        stage1 = stage1[: self.cfg.radar_size]
        radar: list[RadarItem] = []
        for rank, (score, symbol, market, quote_volume, range_pct, change_pct, is_major) in enumerate(stage1, 1):
            radar.append(RadarItem(
                symbol=symbol, base=str(market.get("base") or ""), quote_volume=quote_volume,
                intraday_range_pct=range_pct, change_pct=change_pct, radar_score=score,
                radar_rank=rank, is_major=is_major,
            ))

        # Start with the strongest radar names, then inject any missing majors.
        predeep = stage1[: self.cfg.deep_analysis_size]
        pre_symbols = {x[1] for x in predeep}
        for item in stage1:
            if item[6] and item[1] not in pre_symbols:
                predeep.append(item)
                pre_symbols.add(item[1])

        sem = asyncio.Semaphore(max(1, self.cfg.scanner_concurrency))
        jobs = []
        radar_rank_by_symbol = {r.symbol: r.radar_rank for r in radar}
        for score, symbol, market, quote_volume, range_pct, change_pct, is_major in predeep:
            jobs.append(self._deep_candidate(
                (score, symbol, market, quote_volume, range_pct, radar_rank_by_symbol.get(symbol, 9999), is_major), sem
            ))
        deep_results = await asyncio.gather(*jobs, return_exceptions=False)
        deep = [x for x in deep_results if x is not None]
        deep.sort(key=lambda c: c.opportunity_score, reverse=True)
        for rank, c in enumerate(deep, 1):
            c.deep_rank = rank
            c.fast_cabinet = rank <= self.cfg.fast_cabinet_size or c.is_major
        fast = [c for c in deep if c.fast_cabinet]

        # Breadth is based on all eligible radar names, not only deep candidates.
        radar_changes = [r.change_pct for r in radar]
        radar_ranges = [r.intraday_range_pct for r in radar]
        positive = sum(1 for x in radar_changes if x > 0)
        breadth_positive_pct = positive / len(radar_changes) if radar_changes else 0.5
        def median(vals: list[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        return ScanBundle(
            radar=radar,
            deep=deep,
            fast_cabinet=fast,
            breadth_positive_pct=breadth_positive_pct,
            median_change_pct=median(radar_changes),
            median_range_pct=median(radar_ranges),
        )
