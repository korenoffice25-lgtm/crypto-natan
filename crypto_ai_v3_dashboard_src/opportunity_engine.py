from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
import pandas as pd

from decision_agent import Decision, Action
from regime_model import RegimeReading
from universe_scanner import Candidate


@dataclass
class Opportunity:
    symbol: str
    engine: str
    score: float
    confidence: float
    target_exposure_pct: float
    expected_edge_bps: float
    reason: str
    setup_key: str
    memory_multiplier: float
    repeat_penalty: float
    candidate_risk_multiplier: float
    volatility: float
    best_bid: float
    best_ask: float
    spread_bps: float
    return_vector: list[float]
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("return_vector", None)
        return d


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def trader_opportunity(
    *, candidate: Candidate, decision: Decision, row: pd.DataFrame,
    best_bid: float, best_ask: float, spread_bps: float,
    orderbook_imbalance: float, trade_flow_imbalance: float,
    regime: RegimeReading, memory_multiplier: float, repeat_penalty: float,
    returns: np.ndarray,
) -> Opportunity | None:
    if decision.action != Action.BUY:
        return None

    edge_component = _clip01(decision.expected_edge_bps / 80.0)
    liquidity_component = _clip01((candidate.opportunity_score - 5.0) / 8.0)
    micro_component = _clip01((orderbook_imbalance + trade_flow_imbalance + 2.0) / 4.0)
    base = 100.0 * (
        0.52 * decision.confidence + 0.24 * edge_component +
        0.14 * liquidity_component + 0.10 * micro_component
    )
    score = base * memory_multiplier * repeat_penalty
    target = decision.target_exposure_pct * (0.75 + 0.25 * min(memory_multiplier, 1.25))
    setup_key = f"trader:r{regime.cluster}:vol{int(regime.volatility_rank*3)}"
    ctx = {
        "engine": "TRADER",
        "decision": {
            "confidence": decision.confidence,
            "edge_bps": decision.expected_edge_bps,
            "uncertainty_bps": decision.uncertainty_bps,
            "reason": decision.reason,
        },
        "market": {
            "universe_score": candidate.opportunity_score,
            "quote_volume": candidate.quote_volume,
            "intraday_range_pct": candidate.intraday_range_pct,
            "spread_bps": spread_bps,
            "risk_multiplier": candidate.risk_multiplier,
        },
        "regime": {
            "cluster": regime.cluster,
            "confidence": regime.confidence,
            "volatility_rank": regime.volatility_rank,
            "activity_rank": regime.activity_rank,
        },
        "features": {k: float(row[k].iloc[0]) for k in row.columns if k != "timestamp"},
        "micro": {
            "orderbook_imbalance": orderbook_imbalance,
            "trade_flow_imbalance": trade_flow_imbalance,
        },
    }
    return Opportunity(
        symbol=candidate.symbol, engine="TRADER", score=float(score),
        confidence=float(decision.confidence), target_exposure_pct=float(target),
        expected_edge_bps=float(decision.expected_edge_bps),
        reason="Learned-return Trader signal ranked against the full universe",
        setup_key=setup_key, memory_multiplier=memory_multiplier,
        repeat_penalty=repeat_penalty, candidate_risk_multiplier=candidate.risk_multiplier,
        volatility=max(float(row["realized_vol_12"].iloc[0]), 0.001),
        best_bid=best_bid, best_ask=best_ask, spread_bps=spread_bps,
        return_vector=[float(x) for x in returns.tolist()], context=ctx,
    )


def hunter_opportunity(
    *, candidate: Candidate, row: pd.DataFrame, best_bid: float, best_ask: float,
    spread_bps: float, orderbook_imbalance: float, microprice_edge_bps: float,
    trade_flow_imbalance: float, regime: RegimeReading,
    memory_multiplier: float, repeat_penalty: float, returns: np.ndarray,
    min_volume_z: float, min_momentum_pct: float, micro_model_edge_bps: float = 0.0,
) -> Opportunity | None:
    r3 = float(row["ret_3"].iloc[0])
    r12 = float(row["ret_12"].iloc[0])
    vol_z = float(row["volume_z_24"].iloc[0])
    vol_change = float(row["volume_change"].iloc[0])
    ema_fast = float(row["ema_distance_fast"].iloc[0])
    body = float(row["body_pct"].iloc[0])

    # Hunter looks for early positive expansion, not merely a coin that already pumped.
    momentum = 0.65 * r3 + 0.35 * max(r12, 0.0)
    if vol_z < min_volume_z or momentum < min_momentum_pct or r3 <= 0:
        return None

    volume_score = _clip01((vol_z - min_volume_z) / 3.0 + 0.35)
    momentum_score = _clip01(momentum / 0.02)
    acceleration_score = _clip01((r3 - r12 / 4.0) / 0.012 + 0.5)
    micro_score = _clip01((0.55 * orderbook_imbalance + 0.45 * trade_flow_imbalance + 1.0) / 2.0)
    tape_score = _clip01(microprice_edge_bps / 8.0 + 0.5)
    breakout_shape = _clip01((ema_fast + max(body, 0.0)) / 0.02)

    learned_micro_score = _clip01(micro_model_edge_bps / 20.0 + 0.5)
    raw = 100.0 * (
        0.25 * volume_score + 0.22 * momentum_score + 0.16 * acceleration_score +
        0.13 * micro_score + 0.07 * tape_score + 0.09 * breakout_shape + 0.08 * learned_micro_score
    )
    # Very late vertical moves are penalized; V4 wants the middle of the move.
    late_penalty = 0.78 if r12 > 0.12 else (0.88 if r12 > 0.07 else 1.0)
    score = raw * memory_multiplier * repeat_penalty * late_penalty
    confidence = _clip01(_sigmoid((score - 60.0) / 9.0))
    target = 0.04 + 0.11 * _clip01((score - 60.0) / 35.0)
    setup_key = f"hunter:r{regime.cluster}:v{int(min(max(vol_z,0),5))}:m{int(min(max(momentum*100,0),5))}"

    ctx = {
        "engine": "HUNTER",
        "hunter": {
            "volume_z": vol_z,
            "volume_change": vol_change,
            "ret_3": r3,
            "ret_12": r12,
            "momentum": momentum,
            "acceleration_score": acceleration_score,
            "late_penalty": late_penalty,
        },
        "market": {
            "universe_score": candidate.opportunity_score,
            "quote_volume": candidate.quote_volume,
            "intraday_range_pct": candidate.intraday_range_pct,
            "spread_bps": spread_bps,
            "risk_multiplier": candidate.risk_multiplier,
        },
        "regime": {
            "cluster": regime.cluster,
            "confidence": regime.confidence,
            "volatility_rank": regime.volatility_rank,
            "activity_rank": regime.activity_rank,
        },
        "features": {k: float(row[k].iloc[0]) for k in row.columns if k != "timestamp"},
        "micro": {
            "orderbook_imbalance": orderbook_imbalance,
            "trade_flow_imbalance": trade_flow_imbalance,
            "microprice_edge_bps": microprice_edge_bps,
            "micro_model_edge_bps": micro_model_edge_bps,
        },
    }
    return Opportunity(
        symbol=candidate.symbol, engine="HUNTER", score=float(score),
        confidence=float(confidence), target_exposure_pct=float(target),
        expected_edge_bps=float(momentum * 10_000),
        reason="Volume + momentum expansion with microstructure confirmation",
        setup_key=setup_key, memory_multiplier=memory_multiplier,
        repeat_penalty=repeat_penalty, candidate_risk_multiplier=candidate.risk_multiplier,
        volatility=max(float(row["realized_vol_12"].iloc[0]), 0.001),
        best_bid=best_bid, best_ask=best_ask, spread_bps=spread_bps,
        return_vector=[float(x) for x in returns.tolist()], context=ctx,
    )
