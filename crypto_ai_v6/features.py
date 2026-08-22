from __future__ import annotations

import math
import numpy as np
import pandas as pd

from domain import FeaturePacket


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 2:
        return 50.0
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    v = 100 - 100 / (1 + rs.iloc[-1])
    return float(v) if math.isfinite(float(v)) else 50.0


def _ret(close: pd.Series, n: int) -> float:
    if len(close) <= n or close.iloc[-n-1] <= 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-n-1] - 1.0)


def _volume_z(volume: pd.Series, window: int = 30) -> float:
    x = volume.tail(window)
    if len(x) < 8:
        return 0.0
    sd = float(x.std())
    return float((x.iloc[-1] - x.mean()) / sd) if sd > 1e-12 else 0.0


def book_metrics(orderbook: dict, levels: int = 20) -> tuple[float, float, float, float, float]:
    bids = (orderbook.get("bids") or [])[:levels]
    asks = (orderbook.get("asks") or [])[:levels]
    if not bids or not asks:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    bid = float(bids[0][0]); ask = float(asks[0][0]); mid = (bid + ask) / 2.0
    spread_bps = (ask / bid - 1.0) * 10_000 if bid > 0 else 999.0
    bid_notional = sum(float(p) * float(q) for p, q, *_ in bids)
    ask_notional = sum(float(p) * float(q) for p, q, *_ in asks)
    denom = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / denom if denom > 0 else 0.0
    top_bid_qty = float(bids[0][1]); top_ask_qty = float(asks[0][1])
    micro = (ask * top_bid_qty + bid * top_ask_qty) / max(top_bid_qty + top_ask_qty, 1e-12)
    micro_edge_bps = (micro / mid - 1.0) * 10_000 if mid > 0 else 0.0
    return bid, ask, spread_bps, imbalance, micro_edge_bps


def trade_flow_metrics(trades: list[dict]) -> float:
    buy = 0.0; sell = 0.0
    for t in trades or []:
        amount = float(t.get("amount") or 0.0)
        price = float(t.get("price") or 0.0)
        notion = amount * price
        side = str(t.get("side") or "").lower()
        if side == "buy":
            buy += notion
        elif side == "sell":
            sell += notion
    denom = buy + sell
    return (buy - sell) / denom if denom > 0 else 0.0


def build_feature_packet(symbol: str, candles: pd.DataFrame, orderbook: dict, trades: list[dict], correlation_lookback: int = 96) -> FeaturePacket | None:
    if candles is None or len(candles) < 60:
        return None
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)
    bid, ask, spread_bps, ob_imb, micro_edge = book_metrics(orderbook)
    if bid <= 0 or ask <= 0:
        return None
    price = (bid + ask) / 2.0
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(returns.tail(24).std()) if len(returns) > 2 else 0.001
    vol = max(vol, 0.001)
    ema_fast = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    vz = _volume_z(volume)
    vc = float(volume.iloc[-1] / max(volume.tail(20).median(), 1e-12) - 1.0)
    return FeaturePacket(
        symbol=symbol,
        price=price,
        best_bid=bid,
        best_ask=ask,
        spread_bps=spread_bps,
        volatility=vol,
        ret_1=_ret(close, 1),
        ret_3=_ret(close, 3),
        ret_12=_ret(close, 12),
        ret_36=_ret(close, 36),
        volume_z=vz,
        volume_change=vc,
        rsi=_rsi(close),
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_fast_distance=price / ema_fast - 1.0 if ema_fast > 0 else 0.0,
        ema_slow_distance=price / ema_slow - 1.0 if ema_slow > 0 else 0.0,
        orderbook_imbalance=ob_imb,
        trade_flow_imbalance=trade_flow_metrics(trades),
        microprice_edge_bps=micro_edge,
        return_vector=[float(x) for x in returns.tail(correlation_lookback).tolist()],
    )


def timeframe_metrics(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or len(df) < 60:
        return {"trend": 0.0, "momentum": 0.0, "pullback": 0.0, "volume_z": 0.0, "vol": 0.01}
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    rv = close.pct_change().tail(24).std()
    return {
        "trend": float(fast.iloc[-1] / slow.iloc[-1] - 1.0) if slow.iloc[-1] else 0.0,
        "momentum": float(close.iloc[-1] / close.iloc[-12] - 1.0) if len(close) >= 12 else 0.0,
        "pullback": float(close.iloc[-1] / fast.iloc[-1] - 1.0) if fast.iloc[-1] else 0.0,
        "volume_z": _volume_z(vol),
        "vol": max(float(rv or 0.0), 0.001),
    }
