from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any
import hashlib


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    symbol: str
    brain: str
    qty: float
    entry_price: float
    entry_time: str
    entry_fee: float
    initial_stop_price: float
    stop_price: float
    highest_price: float
    lowest_price: float
    trailing_stop_price: float = 0.0
    trailing_active: bool = False
    partial_taken: bool = False
    scale_count: int = 0
    realized_partial_pnl: float = 0.0
    realized_partial_fees: float = 0.0
    capital_invested: float = 0.0
    lifecycle: str = "ENTRY"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperBroker:
    """Long-only paper broker with fees, slippage, partial exits and action idempotency."""

    def __init__(self, starting_cash: float, fee_rate: float, slippage_bps: float, processed_action_ids: list[str] | None = None):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.processed_action_ids = set(processed_action_ids or [])

    @classmethod
    def from_state(cls, state: dict[str, Any], fee_rate: float, slippage_bps: float):
        obj = cls(float(state.get("starting_cash", 10000)), fee_rate, slippage_bps, state.get("processed_action_ids", []))
        obj.cash = float(state.get("cash", obj.starting_cash))
        obj.realized_pnl = float(state.get("realized_pnl", 0.0))
        for raw in state.get("positions", []):
            x = dict(raw)
            x.setdefault("brain", str((x.get("meta") or {}).get("brain") or "TRADER"))
            x.setdefault("initial_stop_price", float(x.get("stop_price", 0.0)))
            x.setdefault("highest_price", float(x.get("entry_price", 0.0)))
            x.setdefault("lowest_price", float(x.get("entry_price", 0.0)))
            x.setdefault("trailing_stop_price", 0.0)
            x.setdefault("trailing_active", False)
            x.setdefault("partial_taken", False)
            x.setdefault("scale_count", 0)
            x.setdefault("realized_partial_pnl", 0.0)
            x.setdefault("realized_partial_fees", 0.0)
            x.setdefault("capital_invested", float(x.get("entry_price",0))*float(x.get("qty",0))+float(x.get("entry_fee",0)))
            x.setdefault("lifecycle", "ENTRY")
            x.setdefault("meta", {})
            p = Position(**x)
            obj.positions[p.symbol] = p
        return obj

    def snapshot(self) -> dict[str, Any]:
        ids = list(self.processed_action_ids)[-500:]
        return {"starting_cash": self.starting_cash, "cash": self.cash, "realized_pnl": self.realized_pnl,
                "positions": [p.to_dict() for p in self.positions.values()], "processed_action_ids": ids}

    def _done(self, action_id: str) -> bool:
        return bool(action_id and action_id in self.processed_action_ids)

    def _mark_done(self, action_id: str):
        if action_id:
            self.processed_action_ids.add(action_id)
            if len(self.processed_action_ids) > 1000:
                self.processed_action_ids = set(list(self.processed_action_ids)[-700:])

    @staticmethod
    def make_action_id(cycle_id: str, action: str, symbol: str, extra: str = "") -> str:
        raw = f"{cycle_id}|{action}|{symbol}|{extra}".encode()
        return hashlib.sha1(raw).hexdigest()[:20]

    def _buy_fill(self, ask: float, extra_slippage_bps: float = 0.0) -> float:
        return float(ask)*(1+(self.slippage_bps+max(0.0,extra_slippage_bps))/10000)

    def _sell_fill(self, bid: float, extra_slippage_bps: float = 0.0) -> float:
        return float(bid)*(1-(self.slippage_bps+max(0.0,extra_slippage_bps))/10000)

    def open(self, *, symbol: str, brain: str, qty: float, ask: float, stop_price: float,
             meta: dict[str, Any], action_id: str, extra_slippage_bps: float = 0.0) -> dict[str, Any]:
        if self._done(action_id):
            return {"ok": False, "reason": "duplicate_action", "action_id": action_id}
        if symbol in self.positions:
            return {"ok": False, "reason": "position_exists", "action_id": action_id}
        fill = self._buy_fill(ask, extra_slippage_bps)
        notional = float(qty)*fill
        fee = notional*self.fee_rate
        if qty <= 0 or notional+fee > self.cash:
            return {"ok": False, "reason": "insufficient_cash_or_invalid_qty", "action_id": action_id}
        self.cash -= notional+fee
        p = Position(symbol, brain, float(qty), fill, _utc(), fee, float(stop_price), float(stop_price), fill, fill,
                     capital_invested=notional+fee, meta=dict(meta))
        self.positions[symbol] = p
        self._mark_done(action_id)
        return {"ok": True, "side": "buy", "symbol": symbol, "brain": brain, "qty": qty, "fill": fill,
                "notional": notional, "fee": fee, "stop_price": stop_price, "entry_time": p.entry_time,
                "action_id": action_id, "meta": p.meta}

    def add(self, *, symbol: str, qty: float, ask: float, action_id: str, reason: str = "ADD",
            extra_slippage_bps: float = 0.0) -> dict[str, Any]:
        if self._done(action_id):
            return {"ok": False, "reason": "duplicate_action", "action_id": action_id}
        p = self.positions.get(symbol)
        if not p:
            return {"ok": False, "reason": "no_position", "action_id": action_id}
        fill = self._buy_fill(ask, extra_slippage_bps)
        notional = float(qty)*fill
        fee = notional*self.fee_rate
        if qty <= 0 or notional+fee > self.cash:
            return {"ok": False, "reason": "insufficient_cash_or_invalid_qty", "action_id": action_id}
        old_qty = p.qty
        p.entry_price = (p.entry_price*old_qty + fill*qty)/(old_qty+qty)
        p.qty += qty
        p.entry_fee += fee
        p.capital_invested += notional+fee
        p.scale_count += 1
        p.highest_price = max(p.highest_price, fill)
        p.lowest_price = min(p.lowest_price, fill)
        self.cash -= notional+fee
        self._mark_done(action_id)
        return {"ok": True, "side": "buy_add", "symbol": symbol, "qty": qty, "fill": fill, "fee": fee,
                "new_qty": p.qty, "new_entry_price": p.entry_price, "reason": reason, "action_id": action_id, "meta": p.meta}

    def reduce(self, *, symbol: str, bid: float, fraction: float, action_id: str, reason: str = "REDUCE",
               extra_slippage_bps: float = 0.0) -> dict[str, Any]:
        if self._done(action_id):
            return {"ok": False, "reason": "duplicate_action", "action_id": action_id}
        p = self.positions.get(symbol)
        if not p:
            return {"ok": False, "reason": "no_position", "action_id": action_id}
        fraction = max(0.0, min(float(fraction), 0.95))
        qty = p.qty*fraction
        if qty <= 0:
            return {"ok": False, "reason": "zero_reduce_qty", "action_id": action_id}
        fill = self._sell_fill(bid, extra_slippage_bps)
        proceeds = qty*fill
        exit_fee = proceeds*self.fee_rate
        entry_fee_alloc = p.entry_fee*(qty/p.qty)
        pnl_net = (fill-p.entry_price)*qty-entry_fee_alloc-exit_fee
        self.cash += proceeds-exit_fee
        self.realized_pnl += pnl_net
        p.realized_partial_pnl += pnl_net
        p.realized_partial_fees += entry_fee_alloc+exit_fee
        p.entry_fee -= entry_fee_alloc
        p.qty -= qty
        if reason == "PARTIAL_TAKE_PROFIT":
            p.partial_taken = True
        self._mark_done(action_id)
        return {"ok": True, "side": "sell_partial", "symbol": symbol, "qty": qty, "fill": fill, "pnl_net": pnl_net,
                "exit_fee": exit_fee, "remaining_qty": p.qty, "reason": reason, "action_id": action_id, "meta": p.meta}

    def close(self, *, symbol: str, bid: float, action_id: str, reason: str = "CLOSE",
              extra_slippage_bps: float = 0.0) -> dict[str, Any]:
        if self._done(action_id):
            return {"ok": False, "reason": "duplicate_action", "action_id": action_id}
        p = self.positions.pop(symbol, None)
        if not p:
            return {"ok": False, "reason": "no_position", "action_id": action_id}
        fill = self._sell_fill(bid, extra_slippage_bps)
        proceeds = p.qty*fill
        exit_fee = proceeds*self.fee_rate
        final_leg = (fill-p.entry_price)*p.qty-p.entry_fee-exit_fee
        total_pnl = p.realized_partial_pnl+final_leg
        fees = p.realized_partial_fees+p.entry_fee+exit_fee
        self.cash += proceeds-exit_fee
        self.realized_pnl += final_leg
        ret = total_pnl/max(p.capital_invested, 1e-9)*100
        mfe = (p.highest_price/p.entry_price-1)*100 if p.entry_price else 0.0
        mae = (p.lowest_price/p.entry_price-1)*100 if p.entry_price else 0.0
        self._mark_done(action_id)
        return {"ok": True, "side": "sell", "symbol": symbol, "brain": p.brain, "qty": p.qty,
                "entry_price": p.entry_price, "exit_price": fill, "fill": fill, "pnl_net": total_pnl,
                "final_leg_pnl_net": final_leg, "fees_total": fees, "return_pct": ret, "mfe_pct": mfe, "mae_pct": mae,
                "entry_time": p.entry_time, "exit_time": _utc(), "reason": reason, "scale_count": p.scale_count,
                "partial_realized_pnl": p.realized_partial_pnl, "action_id": action_id, "meta": p.meta}

    def update_mark(self, symbol: str, mark: float):
        p = self.positions.get(symbol)
        if not p:
            return
        p.highest_price = max(p.highest_price, mark)
        p.lowest_price = min(p.lowest_price, mark)

    def update_trailing(self, symbol: str, mark: float, activation_pct: float, distance_pct: float):
        p = self.positions.get(symbol)
        if not p:
            return
        self.update_mark(symbol, mark)
        if p.highest_price/p.entry_price-1 >= activation_pct:
            p.trailing_active = True
            p.trailing_stop_price = max(p.trailing_stop_price, p.highest_price*(1-distance_pct), p.stop_price)

    def stop_hit(self, symbol: str, bid: float) -> tuple[bool, str]:
        p = self.positions.get(symbol)
        if not p:
            return False, ""
        if bid <= p.stop_price:
            return True, "HARD_STOP"
        if p.trailing_active and p.trailing_stop_price > 0 and bid <= p.trailing_stop_price:
            return True, "TRAILING_STOP"
        return False, ""

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(p.qty*float(marks.get(s, p.entry_price)) for s,p in self.positions.items())

    def total_exposure_pct(self, marks: dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return sum(p.qty*float(marks.get(s,p.entry_price)) for s,p in self.positions.items())/equity

    def symbol_exposure_pct(self, symbol: str, marks: dict[str, float], equity: float) -> float:
        p = self.positions.get(symbol)
        return p.qty*float(marks.get(symbol,p.entry_price))/equity if p and equity > 0 else 0.0

    def open_risk_pct(self, marks: dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        risk = 0.0
        for s,p in self.positions.items():
            mark = float(marks.get(s,p.entry_price))
            loss_to_stop = max(0.0, mark-p.stop_price)*p.qty
            risk += loss_to_stop
        return risk/equity

    def report(self, marks: dict[str, float]) -> list[dict[str, Any]]:
        out=[]
        for s,p in self.positions.items():
            mark=float(marks.get(s,p.entry_price)); self.update_mark(s,mark)
            value=p.qty*mark
            est_exit_fee=value*self.fee_rate
            unreal=(mark-p.entry_price)*p.qty-p.entry_fee-est_exit_fee
            cost=p.entry_price*p.qty+p.entry_fee
            out.append({**p.to_dict(), "current_price":mark,"market_value":value,"unrealized_pnl":unreal,
                        "return_pct":unreal/cost*100 if cost>0 else 0.0, "estimated_exit_fee":est_exit_fee})
        return out
