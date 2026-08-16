from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Candidate:
    symbol: str
    base: str
    quote: str
    quote_volume: float
    change_pct: float
    range_pct: float
    spread_bps: float
    depth_notional: float
    market_score: float
    risk_multiplier: float
    radar_rank: int = 0
    deep_rank: int = 0
    is_major: bool = False
    fast_cabinet: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeaturePacket:
    symbol: str
    price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    volatility: float
    ret_1: float
    ret_3: float
    ret_12: float
    ret_36: float
    volume_z: float
    volume_change: float
    rsi: float
    ema_fast: float
    ema_slow: float
    ema_fast_distance: float
    ema_slow_distance: float
    orderbook_imbalance: float
    trade_flow_imbalance: float
    microprice_edge_bps: float
    return_vector: list[float] = field(default_factory=list)


@dataclass
class Signal:
    symbol: str
    brain: str
    score: float
    raw_score: float
    confidence: float
    expected_edge_bps: float
    target_exposure_pct: float
    stop_distance_pct: float
    reason: str
    setup_key: str
    candidate_risk_multiplier: float
    volatility: float
    best_bid: float
    best_ask: float
    spread_bps: float
    return_vector: list[float]
    context: dict[str, Any] = field(default_factory=dict)
    meta_score: float = 0.0
    regime_multiplier: float = 1.0
    utility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("return_vector", None)
        d["engine"] = self.brain
        return d


@dataclass
class MarketRegime:
    name: str
    confidence: float
    breadth_positive_pct: float
    median_change_pct: float
    median_range_pct: float
    target_utilization_pct: float
    max_utilization_pct: float
    risk_multiplier: float
    brain_multipliers: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioAction:
    action: str  # OPEN / ADD / HOLD / REDUCE / CLOSE / ROTATE / CASH
    symbol: str = ""
    brain: str = ""
    target_exposure_pct: float = 0.0
    fraction: float = 0.0
    reason: str = ""
    signal: Signal | None = None
    displaced_symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.signal is not None:
            d["signal"] = self.signal.to_dict()
        return d
