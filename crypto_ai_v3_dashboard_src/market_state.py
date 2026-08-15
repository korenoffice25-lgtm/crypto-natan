from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) and math.isfinite(b) else default


@dataclass
class OrderBookState:
    best_bid: float
    best_ask: float
    mid: float
    spread_bps: float
    bid_depth_5: float
    ask_depth_5: float
    bid_depth_20: float
    ask_depth_20: float
    imbalance_5: float
    imbalance_20: float
    microprice: float
    microprice_edge_bps: float


@dataclass
class TradeFlowState:
    trade_count: int
    buy_notional: float
    sell_notional: float
    flow_imbalance: float
    vwap: float
    avg_trade_notional: float


@dataclass
class MarketSnapshot:
    timestamp_ms: int
    symbol: str
    last: float
    orderbook: OrderBookState
    trades: TradeFlowState
    short_return: float = 0.0
    realized_vol: float = 0.0

    def flatten(self) -> dict[str, float | int | str]:
        data: dict[str, float | int | str] = {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "last": self.last,
            "short_return": self.short_return,
            "realized_vol": self.realized_vol,
        }
        data.update({f"ob_{k}": v for k, v in asdict(self.orderbook).items()})
        data.update({f"tf_{k}": v for k, v in asdict(self.trades).items()})
        return data


def orderbook_state(orderbook: dict[str, Any], levels: int = 20) -> OrderBookState:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        raise ValueError("Order book is empty")

    bids = bids[:levels]
    asks = asks[:levels]
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    spread_bps = _safe_div(best_ask - best_bid, mid) * 10_000

    def depth(rows, n):
        return float(sum(float(p) * float(q) for p, q, *_ in rows[:n]))

    bid5, ask5 = depth(bids, 5), depth(asks, 5)
    bid20, ask20 = depth(bids, min(20, levels)), depth(asks, min(20, levels))

    imbalance5 = _safe_div(bid5 - ask5, bid5 + ask5)
    imbalance20 = _safe_div(bid20 - ask20, bid20 + ask20)

    bid_qty = float(bids[0][1])
    ask_qty = float(asks[0][1])
    denom = bid_qty + ask_qty
    # Microprice leans toward the side with less resting top-of-book liquidity.
    microprice = _safe_div(best_ask * bid_qty + best_bid * ask_qty, denom, mid)
    edge_bps = _safe_div(microprice - mid, mid) * 10_000

    return OrderBookState(
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        spread_bps=spread_bps,
        bid_depth_5=bid5,
        ask_depth_5=ask5,
        bid_depth_20=bid20,
        ask_depth_20=ask20,
        imbalance_5=imbalance5,
        imbalance_20=imbalance20,
        microprice=microprice,
        microprice_edge_bps=edge_bps,
    )


def trade_flow_state(trades: list[dict[str, Any]]) -> TradeFlowState:
    if not trades:
        return TradeFlowState(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    buy_notional = 0.0
    sell_notional = 0.0
    total_notional = 0.0
    total_amount = 0.0

    for t in trades:
        price = float(t.get("price") or 0.0)
        amount = float(t.get("amount") or 0.0)
        notional = price * amount
        total_notional += notional
        total_amount += amount
        side = (t.get("side") or "").lower()
        if side == "buy":
            buy_notional += notional
        elif side == "sell":
            sell_notional += notional

    signed = buy_notional - sell_notional
    flow_imbalance = _safe_div(signed, buy_notional + sell_notional)
    vwap = _safe_div(total_notional, total_amount)
    avg_notional = _safe_div(total_notional, len(trades))

    return TradeFlowState(
        trade_count=len(trades),
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        flow_imbalance=flow_imbalance,
        vwap=vwap,
        avg_trade_notional=avg_notional,
    )
