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
    brain: str
    score: float
    raw_score: float
    confidence: float
    target_exposure_pct: float
    expected_edge_bps: float
    reason: str
    setup_key: str
    memory_multiplier: float
    model_weight: float
    repeat_penalty: float
    candidate_risk_multiplier: float
    volatility: float
    best_bid: float
    best_ask: float
    spread_bps: float
    return_vector: list[float]
    context: dict[str, Any]
    meta_score: float = 0.0
    regime_multiplier: float = 1.0

    @property
    def engine(self) -> str:  # compatibility with V4-style display code
        return self.brain

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("return_vector", None)
        d["engine"] = self.brain
        return d


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def _safe(row: pd.DataFrame, key: str, default=0.0) -> float:
    try:
        v = float(row[key].iloc[0])
        return v if math.isfinite(v) else default
    except Exception:
        return default


def trader_opportunity(*, candidate: Candidate, decision: Decision, row: pd.DataFrame,
                       best_bid: float, best_ask: float, spread_bps: float,
                       orderbook_imbalance: float, trade_flow_imbalance: float,
                       regime: RegimeReading, memory_multiplier: float, model_weight: float,
                       repeat_penalty: float, returns: np.ndarray) -> Opportunity | None:
    if decision.action != Action.BUY:
        return None
    edge_component = _clip01(decision.expected_edge_bps / 80.0)
    liquidity_component = _clip01((candidate.opportunity_score - 5.0) / 8.0)
    micro_component = _clip01((orderbook_imbalance + trade_flow_imbalance + 2.0) / 4.0)
    raw = 100.0 * (0.52 * decision.confidence + 0.24 * edge_component + 0.14 * liquidity_component + 0.10 * micro_component)
    score = raw * memory_multiplier * model_weight * repeat_penalty
    target = 0.03 + 0.10 * _clip01((score - 55.0) / 40.0)
    setup_key = f"trader:r{regime.cluster}:vol{int(regime.volatility_rank*3)}"
    return Opportunity(
        candidate.symbol, "TRADER", float(score), float(raw), float(decision.confidence), float(target),
        float(decision.expected_edge_bps), "Learned short-horizon edge + current microstructure", setup_key,
        memory_multiplier, model_weight, repeat_penalty, candidate.risk_multiplier,
        max(_safe(row, "realized_vol_12", 0.001), 0.001), best_bid, best_ask, spread_bps,
        [float(x) for x in returns.tolist()],
        {"decision": asdict(decision), "regime": asdict(regime), "micro": {"orderbook_imbalance": orderbook_imbalance,
         "trade_flow_imbalance": trade_flow_imbalance}, "candidate": candidate.to_dict()},
    )


def hunter_opportunity(*, candidate: Candidate, row: pd.DataFrame, best_bid: float, best_ask: float,
                       spread_bps: float, orderbook_imbalance: float, microprice_edge_bps: float,
                       trade_flow_imbalance: float, regime: RegimeReading, memory_multiplier: float,
                       model_weight: float, repeat_penalty: float, returns: np.ndarray,
                       min_volume_z: float, min_momentum_pct: float, micro_model_edge_bps: float = 0.0) -> Opportunity | None:
    r1, r3, r12 = _safe(row, "ret_1"), _safe(row, "ret_3"), _safe(row, "ret_12")
    vol_z, vol_change = _safe(row, "volume_z_24"), _safe(row, "volume_change")
    ema_fast, body = _safe(row, "ema_distance_fast"), _safe(row, "body_pct")
    momentum = 0.55 * r3 + 0.30 * max(r12, 0.0) + 0.15 * max(r1, 0.0)
    if vol_z < min_volume_z or momentum < min_momentum_pct or r3 <= 0:
        return None
    volume_score = _clip01((vol_z - min_volume_z) / 3.0 + 0.35)
    momentum_score = _clip01(momentum / 0.018)
    acceleration_score = _clip01((r3 - r12 / 4.0) / 0.010 + 0.5)
    micro_score = _clip01((0.55 * orderbook_imbalance + 0.45 * trade_flow_imbalance + 1.0) / 2.0)
    tape_score = _clip01(microprice_edge_bps / 8.0 + 0.5)
    shape_score = _clip01((ema_fast + max(body, 0.0)) / 0.018)
    learned_micro = _clip01(micro_model_edge_bps / 18.0 + 0.5)
    raw = 100.0 * (0.24*volume_score + 0.23*momentum_score + 0.17*acceleration_score + 0.13*micro_score + 0.07*tape_score + 0.09*shape_score + 0.07*learned_micro)
    late_penalty = 0.72 if r12 > 0.14 else (0.86 if r12 > 0.08 else 1.0)
    score = raw * memory_multiplier * model_weight * repeat_penalty * late_penalty
    confidence = _clip01(_sigmoid((score - 60.0) / 8.0))
    target = 0.04 + 0.11 * _clip01((score - 60.0) / 35.0)
    setup_key = f"hunter:r{regime.cluster}:v{int(min(max(vol_z,0),5))}:m{int(min(max(momentum*100,0),5))}"
    return Opportunity(
        candidate.symbol, "HUNTER", float(score), float(raw), confidence, target, momentum*10_000,
        "Early volume/momentum expansion with tape confirmation", setup_key, memory_multiplier, model_weight,
        repeat_penalty, candidate.risk_multiplier, max(_safe(row, "realized_vol_12", 0.001), 0.001),
        best_bid, best_ask, spread_bps, [float(x) for x in returns.tolist()],
        {"hunter": {"ret_1": r1, "ret_3": r3, "ret_12": r12, "momentum": momentum, "volume_z": vol_z,
         "volume_change": vol_change, "late_penalty": late_penalty, "micro_model_edge_bps": micro_model_edge_bps},
         "regime": asdict(regime), "candidate": candidate.to_dict()},
    )


def reversal_opportunity(*, candidate: Candidate, row: pd.DataFrame, best_bid: float, best_ask: float,
                         spread_bps: float, orderbook_imbalance: float, trade_flow_imbalance: float,
                         regime: RegimeReading, memory_multiplier: float, model_weight: float,
                         repeat_penalty: float, returns: np.ndarray, min_drop_pct: float) -> Opportunity | None:
    r1, r3, r12 = _safe(row, "ret_1"), _safe(row, "ret_3"), _safe(row, "ret_12")
    rsi = _safe(row, "rsi_centered")
    vol_z = _safe(row, "volume_z_24")
    body = _safe(row, "body_pct")
    atr = max(_safe(row, "atr_pct", 0.005), 0.001)
    drop = -min(r12, 0.0)
    # Long-only rebound: meaningful pullback + exhaustion + first sign of demand returning.
    if drop < min_drop_pct or rsi > -0.05:
        return None
    micro_turn = 0.55 * orderbook_imbalance + 0.45 * trade_flow_imbalance
    confirmation = max(r1, 0.0) + max(body, 0.0) + max(micro_turn, 0.0) * 0.004
    if confirmation <= 0.0005:
        return None
    stretch_score = _clip01(drop / max(atr * 4.0, 0.01))
    rsi_score = _clip01((-rsi - 0.05) / 0.70)
    turn_score = _clip01(confirmation / 0.008)
    volume_score = _clip01((vol_z + 1.0) / 3.0)
    micro_score = _clip01((micro_turn + 1.0) / 2.0)
    raw = 100.0 * (0.30*stretch_score + 0.22*rsi_score + 0.22*turn_score + 0.12*volume_score + 0.14*micro_score)
    score = raw * memory_multiplier * model_weight * repeat_penalty
    confidence = _clip01(_sigmoid((score - 59.0) / 9.0))
    target = 0.03 + 0.09 * _clip01((score - 58.0) / 40.0)
    setup_key = f"reversal:r{regime.cluster}:d{int(min(drop*100,9))}:rsi{int((rsi+1)*5)}"
    return Opportunity(
        candidate.symbol, "REVERSAL", float(score), float(raw), confidence, target,
        confirmation*10_000, "Oversold pullback showing demand/reversal confirmation", setup_key,
        memory_multiplier, model_weight, repeat_penalty, candidate.risk_multiplier,
        max(_safe(row, "realized_vol_12", 0.001), 0.001), best_bid, best_ask, spread_bps,
        [float(x) for x in returns.tolist()],
        {"reversal": {"drop_12": drop, "rsi_centered": rsi, "confirmation": confirmation, "volume_z": vol_z,
         "micro_turn": micro_turn}, "regime": asdict(regime), "candidate": candidate.to_dict()},
    )


def _tf_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df is None or len(df) < 60:
        return {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "pullback": 0.0, "vol": 0.01}
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    trend = (ema_fast.iloc[-1] / ema_slow.iloc[-1] - 1.0) if ema_slow.iloc[-1] else 0.0
    momentum = close.iloc[-1] / close.iloc[-12] - 1.0 if len(close) >= 12 else 0.0
    pullback = close.iloc[-1] / ema_fast.iloc[-1] - 1.0 if ema_fast.iloc[-1] else 0.0
    rv = close.pct_change().tail(24).std()
    roll = volume.tail(30)
    vz = (volume.iloc[-1] - roll.mean()) / roll.std() if roll.std() and math.isfinite(float(roll.std())) else 0.0
    return {"trend": float(trend), "momentum": float(momentum), "volume": float(vz), "pullback": float(pullback), "vol": max(float(rv or 0.0), 0.001)}


def swing_opportunity(*, candidate: Candidate, frames: dict[str, pd.DataFrame], best_bid: float, best_ask: float,
                      spread_bps: float, regime: RegimeReading, memory_multiplier: float, model_weight: float,
                      repeat_penalty: float, returns: np.ndarray) -> Opportunity | None:
    m15 = _tf_metrics(frames.get("15m")); h1 = _tf_metrics(frames.get("1h")); h4 = _tf_metrics(frames.get("4h"))
    alignment = sum(1 for m in (m15, h1, h4) if m["trend"] > 0) / 3.0
    if alignment < 2/3 or h1["momentum"] <= -0.005:
        return None
    trend_score = _clip01((m15["trend"]*12 + h1["trend"]*8 + h4["trend"]*5) / 0.16 + 0.35)
    momentum_score = _clip01((m15["momentum"] + h1["momentum"] + max(h4["momentum"], 0.0)) / 0.12 + 0.25)
    # Healthy pullback is slightly below/near fast EMA, not a vertical chase.
    pullback_dist = abs(min(m15["pullback"], 0.02))
    pullback_score = _clip01(1.0 - pullback_dist / 0.04)
    volume_score = _clip01((m15["volume"] + 1.5) / 4.0)
    alignment_score = alignment
    raw = 100.0 * (0.29*trend_score + 0.26*momentum_score + 0.18*alignment_score + 0.15*pullback_score + 0.12*volume_score)
    late_penalty = 0.82 if m15["momentum"] > 0.08 else 1.0
    score = raw * memory_multiplier * model_weight * repeat_penalty * late_penalty
    confidence = _clip01(_sigmoid((score - 60.0) / 9.0))
    target = 0.05 + 0.10 * _clip01((score - 60.0) / 35.0)
    setup_key = f"swing:r{regime.cluster}:a{int(alignment*3)}:h1{int(min(max(h1['momentum']*100+3,0),9))}"
    edge = (0.35*m15["momentum"] + 0.40*h1["momentum"] + 0.25*h4["momentum"]) * 10_000
    return Opportunity(
        candidate.symbol, "SWING", float(score), float(raw), confidence, target, float(edge),
        "15m/1h/4h trend alignment with healthy continuation structure", setup_key,
        memory_multiplier, model_weight, repeat_penalty, candidate.risk_multiplier,
        max(m15["vol"], 0.001), best_bid, best_ask, spread_bps,
        [float(x) for x in returns.tolist()],
        {"swing": {"15m": m15, "1h": h1, "4h": h4, "alignment": alignment, "late_penalty": late_penalty},
         "regime": asdict(regime), "candidate": candidate.to_dict()},
    )
