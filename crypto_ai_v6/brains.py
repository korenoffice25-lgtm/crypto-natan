from __future__ import annotations

import math
from typing import Any

from domain import Candidate, FeaturePacket, Signal
from features import timeframe_metrics


def _clip(x: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def trader_signal(candidate: Candidate, f: FeaturePacket) -> Signal | None:
    trend = _clip((f.ema_fast_distance + max(f.ema_slow_distance, 0.0)) / 0.018 + 0.35)
    momentum = _clip((0.55*f.ret_3 + 0.30*f.ret_12 + 0.15*f.ret_1) / 0.018 + 0.35)
    tape = _clip((0.55*f.orderbook_imbalance + 0.45*f.trade_flow_imbalance + 1.0) / 2.0)
    volume = _clip((f.volume_z + 1.2) / 4.0)
    spread = _clip(1.0 - f.spread_bps / 35.0)
    if f.ema_fast <= f.ema_slow or f.ret_3 <= -0.0025:
        return None
    raw = 100*(0.31*trend + 0.27*momentum + 0.20*tape + 0.12*volume + 0.10*spread)
    score = raw
    confidence = _clip(_sigmoid((score-60)/8))
    edge = (0.50*f.ret_3 + 0.35*f.ret_12 + 0.15*f.ret_1)*10_000
    return Signal(candidate.symbol, "TRADER", score, raw, confidence, edge,
                  0.03 + 0.10*_clip((score-58)/34), max(0.018,min(0.045,f.volatility*2.6)),
                  "Short-horizon trend/momentum with live tape confirmation",
                  f"trader:t{int(trend*4)}:m{int(momentum*4)}", candidate.risk_multiplier,
                  f.volatility, f.best_bid, f.best_ask, f.spread_bps, f.return_vector,
                  {"trend": trend, "momentum": momentum, "tape": tape, "volume": volume})


def hunter_signal(candidate: Candidate, f: FeaturePacket) -> Signal | None:
    momentum_raw = 0.55*f.ret_3 + 0.30*max(f.ret_12, 0.0) + 0.15*max(f.ret_1, 0.0)
    acceleration = f.ret_3 - f.ret_12/4.0
    if f.volume_z < 1.05 or momentum_raw < 0.0023 or f.ret_3 <= 0:
        return None
    volume = _clip((f.volume_z-0.8)/3.2)
    momentum = _clip(momentum_raw/0.020)
    accel = _clip(acceleration/0.011 + 0.45)
    tape = _clip((0.5*f.orderbook_imbalance + 0.5*f.trade_flow_imbalance + 1)/2)
    micro = _clip(f.microprice_edge_bps/8 + 0.5)
    shape = _clip((max(f.ema_fast_distance,0)+max(f.ret_1,0))/0.015)
    raw = 100*(0.24*volume + 0.25*momentum + 0.18*accel + 0.15*tape + 0.08*micro + 0.10*shape)
    late_penalty = 0.74 if f.ret_12 > 0.14 else (0.87 if f.ret_12 > 0.08 else 1.0)
    score = raw*late_penalty
    confidence = _clip(_sigmoid((score-62)/7.5))
    return Signal(candidate.symbol, "HUNTER", score, raw, confidence, momentum_raw*10_000,
                  0.04 + 0.12*_clip((score-62)/32), max(0.026,min(0.065,f.volatility*3.2)),
                  "Early price/volume acceleration with order-flow confirmation",
                  f"hunter:v{int(_clip(f.volume_z/4)*4)}:a{int(accel*4)}", candidate.risk_multiplier,
                  f.volatility, f.best_bid, f.best_ask, f.spread_bps, f.return_vector,
                  {"volume_z": f.volume_z, "momentum": momentum_raw, "acceleration": acceleration,
                   "late_penalty": late_penalty, "tape": tape})


def reversal_signal(candidate: Candidate, f: FeaturePacket) -> Signal | None:
    drop = -min(f.ret_12, 0.0)
    turn = max(f.ret_1, 0.0) + max(0.55*f.orderbook_imbalance + 0.45*f.trade_flow_imbalance, 0.0)*0.004
    if drop < 0.006 or f.rsi > 46 or turn <= 0.0005:
        return None
    stretch = _clip(drop/0.045)
    oversold = _clip((48-f.rsi)/28)
    confirmation = _clip(turn/0.008)
    tape = _clip((0.55*f.orderbook_imbalance + 0.45*f.trade_flow_imbalance + 1)/2)
    volume = _clip((f.volume_z+1)/3.5)
    raw = 100*(0.31*stretch + 0.24*oversold + 0.23*confirmation + 0.14*tape + 0.08*volume)
    score = raw
    confidence = _clip(_sigmoid((score-62)/8))
    return Signal(candidate.symbol, "REVERSAL", score, raw, confidence, turn*10_000,
                  0.03 + 0.09*_clip((score-61)/34), max(0.022,min(0.055,f.volatility*2.5)),
                  "Oversold pullback showing first confirmed demand turn",
                  f"reversal:d{int(_clip(drop/.06)*5)}:r{int(_clip((50-f.rsi)/30)*5)}", candidate.risk_multiplier,
                  f.volatility, f.best_bid, f.best_ask, f.spread_bps, f.return_vector,
                  {"drop_12": drop, "rsi": f.rsi, "turn": turn, "tape": tape})


def swing_signal(candidate: Candidate, f: FeaturePacket, frames: dict[str, Any] | None) -> Signal | None:
    if not frames:
        return None
    m15 = timeframe_metrics(frames.get("15m")); h1 = timeframe_metrics(frames.get("1h")); h4 = timeframe_metrics(frames.get("4h"))
    alignment = sum(1 for m in (m15,h1,h4) if m["trend"] > 0)/3.0
    if alignment < 2/3 or h1["momentum"] < -0.005:
        return None
    trend = _clip((m15["trend"]*12 + h1["trend"]*8 + h4["trend"]*5)/0.16 + 0.35)
    momentum = _clip((m15["momentum"] + h1["momentum"] + max(h4["momentum"],0))/0.12 + 0.25)
    pullback = _clip(1-abs(min(m15["pullback"], 0.025))/0.045)
    volume = _clip((m15["volume_z"]+1.5)/4)
    raw = 100*(0.30*trend + 0.27*momentum + 0.18*alignment + 0.14*pullback + 0.11*volume)
    late_penalty = 0.82 if m15["momentum"] > 0.085 else 1.0
    score = raw*late_penalty
    confidence = _clip(_sigmoid((score-62)/8.5))
    edge = (0.35*m15["momentum"] + 0.40*h1["momentum"] + 0.25*h4["momentum"])*10_000
    return Signal(candidate.symbol, "SWING", score, raw, confidence, edge,
                  0.05 + 0.11*_clip((score-62)/32), max(0.045,min(0.085,max(m15["vol"],f.volatility)*3.5)),
                  "15m/1h/4h trend alignment with continuation structure",
                  f"swing:a{int(alignment*3)}:m{int(momentum*4)}", candidate.risk_multiplier,
                  max(m15["vol"], f.volatility), f.best_bid, f.best_ask, f.spread_bps, f.return_vector,
                  {"15m": m15, "1h": h1, "4h": h4, "alignment": alignment, "late_penalty": late_penalty})


def all_signals(candidate: Candidate, f: FeaturePacket, swing_frames: dict[str, Any] | None = None) -> list[Signal]:
    out = []
    for fn, args in [
        (trader_signal, (candidate, f)),
        (hunter_signal, (candidate, f)),
        (reversal_signal, (candidate, f)),
    ]:
        s = fn(*args)
        if s:
            out.append(s)
    sw = swing_signal(candidate, f, swing_frames)
    if sw:
        out.append(sw)
    return out
