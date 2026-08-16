from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperPosition:
    symbol: str
    qty: float
    entry_price: float
    entry_fee: float
    entry_time: str
    stop_price: float
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_active: bool = False
    realized_partial_pnl: float = 0.0
    realized_partial_fees: float = 0.0
    partial_taken: bool = False
    scale_count: int = 0
    capital_invested: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class PaperExchange:
    def __init__(self, starting_cash: float, fee_rate: float, slippage_bps: float):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.positions: dict[str, PaperPosition] = {}
        self.realized_pnl = 0.0

    @classmethod
    def from_state(cls, state: dict[str, Any], fee_rate: float, slippage_bps: float):
        obj = cls(float(state.get("starting_cash", 10_000)), fee_rate, slippage_bps)
        obj.cash = float(state.get("cash", obj.starting_cash))
        obj.realized_pnl = float(state.get("realized_pnl", 0.0))
        for raw in state.get("positions", []):
            raw = dict(raw)
            raw.setdefault("highest_price", float(raw.get("entry_price", 0.0)))
            raw.setdefault("lowest_price", float(raw.get("entry_price", 0.0)))
            raw.setdefault("trailing_stop_price", 0.0)
            raw.setdefault("trailing_active", False)
            raw.setdefault("realized_partial_pnl", 0.0)
            raw.setdefault("realized_partial_fees", 0.0)
            raw.setdefault("partial_taken", False)
            raw.setdefault("scale_count", 0)
            raw.setdefault("capital_invested", float(raw.get("entry_price", 0.0)) * float(raw.get("qty", 0.0)) + float(raw.get("entry_fee", 0.0)))
            raw.setdefault("meta", {})
            pos = PaperPosition(**raw)
            obj.positions[pos.symbol] = pos
        return obj

    def snapshot(self) -> dict[str, Any]:
        return {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "positions": [asdict(p) for p in self.positions.values()],
        }

    def _buy_fill(self, ask: float, extra_slippage_bps: float = 0.0) -> float:
        return ask * (1 + (self.slippage_bps + extra_slippage_bps) / 10_000)

    def _sell_fill(self, bid: float, extra_slippage_bps: float = 0.0) -> float:
        return bid * (1 - (self.slippage_bps + extra_slippage_bps) / 10_000)

    def buy(self, symbol: str, qty: float, ask: float, stop_price: float,
            extra_slippage_bps: float = 0.0, meta: dict[str, Any] | None = None) -> dict:
        if symbol in self.positions:
            return {"ok": False, "reason": "position_exists"}
        fill = self._buy_fill(ask, extra_slippage_bps)
        notional = qty * fill
        fee = notional * self.fee_rate
        if notional + fee > self.cash or qty <= 0:
            return {"ok": False, "reason": "insufficient_cash_or_invalid_qty"}
        self.cash -= notional + fee
        entry_time = _utc_now()
        position = PaperPosition(
            symbol=symbol, qty=qty, entry_price=fill, entry_fee=fee,
            entry_time=entry_time, stop_price=float(stop_price), highest_price=fill,
            lowest_price=fill, capital_invested=notional + fee, meta=dict(meta or {}),
        )
        self.positions[symbol] = position
        return {
            "ok": True, "side": "buy", "symbol": symbol, "qty": qty, "fill": fill,
            "notional": notional, "fee": fee, "entry_time": entry_time,
            "stop_price": stop_price, "meta": position.meta,
        }

    def add_to_position(self, symbol: str, qty: float, ask: float, stop_price: float | None = None,
                        extra_slippage_bps: float = 0.0, reason: str = "SCALE_IN") -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "reason": "no_position"}
        fill = self._buy_fill(ask, extra_slippage_bps)
        notional = qty * fill
        fee = notional * self.fee_rate
        if qty <= 0 or notional + fee > self.cash:
            return {"ok": False, "reason": "insufficient_cash_or_invalid_qty"}
        old_notional = pos.qty * pos.entry_price
        new_qty = pos.qty + qty
        pos.entry_price = (old_notional + notional) / new_qty
        pos.qty = new_qty
        pos.entry_fee += fee
        pos.highest_price = max(pos.highest_price, fill)
        pos.lowest_price = min(pos.lowest_price or fill, fill)
        if stop_price is not None:
            pos.stop_price = max(pos.stop_price, float(stop_price))
        pos.scale_count += 1
        pos.capital_invested += notional + fee
        self.cash -= notional + fee
        return {
            "ok": True, "side": "buy_add", "symbol": symbol, "qty": qty, "fill": fill,
            "notional": notional, "fee": fee, "reason": reason, "new_qty": pos.qty,
            "new_entry_price": pos.entry_price, "meta": pos.meta,
        }

    def update_extremes(self, symbol: str, mark: float):
        pos = self.positions.get(symbol)
        if not pos:
            return
        pos.highest_price = max(float(pos.highest_price or pos.entry_price), float(mark))
        low = float(pos.lowest_price or pos.entry_price)
        pos.lowest_price = min(low, float(mark))

    def update_trailing(self, symbol: str, mark: float, activation_pct: float, trail_distance_pct: float) -> dict[str, Any] | None:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        self.update_extremes(symbol, mark)
        gain = pos.highest_price / pos.entry_price - 1.0 if pos.entry_price > 0 else 0.0
        if gain >= activation_pct:
            pos.trailing_active = True
            candidate = pos.highest_price * (1 - max(trail_distance_pct, 0.001))
            pos.trailing_stop_price = max(float(pos.trailing_stop_price or 0.0), candidate, pos.stop_price)
        return {
            "highest_price": pos.highest_price,
            "lowest_price": pos.lowest_price,
            "trailing_active": pos.trailing_active,
            "trailing_stop_price": pos.trailing_stop_price,
        }

    def stop_hit(self, symbol: str, bid: float) -> tuple[bool, str]:
        pos = self.positions.get(symbol)
        if not pos:
            return False, ""
        if bid <= pos.stop_price:
            return True, "HARD_STOP"
        if pos.trailing_active and pos.trailing_stop_price > 0 and bid <= pos.trailing_stop_price:
            return True, "TRAILING_STOP"
        return False, ""

    def reduce(self, symbol: str, bid: float, fraction: float, extra_slippage_bps: float = 0.0,
               reason: str = "PARTIAL_EXIT") -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "reason": "no_position"}
        fraction = max(0.0, min(float(fraction), 0.95))
        qty = pos.qty * fraction
        if qty <= 0:
            return {"ok": False, "reason": "zero_reduce_qty"}
        fill = self._sell_fill(bid, extra_slippage_bps)
        proceeds = qty * fill
        exit_fee = proceeds * self.fee_rate
        entry_fee_alloc = pos.entry_fee * (qty / pos.qty)
        gross_pnl = (fill - pos.entry_price) * qty
        pnl_net = gross_pnl - entry_fee_alloc - exit_fee
        self.cash += proceeds - exit_fee
        self.realized_pnl += pnl_net
        pos.realized_partial_pnl += pnl_net
        pos.realized_partial_fees += entry_fee_alloc + exit_fee
        pos.entry_fee -= entry_fee_alloc
        pos.qty -= qty
        if reason == "PARTIAL_TAKE_PROFIT":
            pos.partial_taken = True
        return {
            "ok": True, "side": "sell_partial", "symbol": symbol, "qty": qty, "fill": fill,
            "exit_fee": exit_fee, "entry_fee_alloc": entry_fee_alloc, "gross_pnl": gross_pnl,
            "pnl_net": pnl_net, "reason": reason, "remaining_qty": pos.qty, "meta": pos.meta,
        }

    def exit(self, symbol: str, bid: float, extra_slippage_bps: float = 0.0, reason: str = "MODEL_EXIT") -> dict:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return {"ok": False, "reason": "no_position"}
        fill = self._sell_fill(bid, extra_slippage_bps)
        proceeds = pos.qty * fill
        exit_fee = proceeds * self.fee_rate
        self.cash += proceeds - exit_fee
        gross_pnl = (fill - pos.entry_price) * pos.qty
        final_leg_pnl = gross_pnl - pos.entry_fee - exit_fee
        total_trade_pnl = pos.realized_partial_pnl + final_leg_pnl
        total_fees = pos.realized_partial_fees + pos.entry_fee + exit_fee
        self.realized_pnl += final_leg_pnl
        capital_basis = max(pos.capital_invested, 1e-9)
        return_pct_total = total_trade_pnl / capital_basis * 100.0
        mfe_pct = (pos.highest_price / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
        mae_pct = (pos.lowest_price / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
        exit_time = _utc_now()
        return {
            "ok": True, "side": "sell", "symbol": symbol, "qty": pos.qty,
            "entry_price": pos.entry_price, "exit_price": fill, "fill": fill,
            "entry_fee": pos.entry_fee, "exit_fee": exit_fee, "fees_total": total_fees,
            "gross_pnl": gross_pnl, "final_leg_pnl_net": final_leg_pnl,
            "pnl_net": total_trade_pnl, "return_pct": return_pct_total,
            "entry_time": pos.entry_time, "exit_time": exit_time, "reason": reason,
            "meta": pos.meta, "highest_price": pos.highest_price, "lowest_price": pos.lowest_price,
            "mfe_pct": mfe_pct, "mae_pct": mae_pct, "partial_realized_pnl": pos.realized_partial_pnl,
            "trailing_stop_price": pos.trailing_stop_price, "scale_count": pos.scale_count,
        }

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(pos.qty * marks.get(symbol, pos.entry_price) for symbol, pos in self.positions.items())

    def exposure_pct(self, symbol: str, mark: float, equity: float) -> float:
        pos = self.positions.get(symbol)
        return (pos.qty * mark / equity) if pos and equity > 0 else 0.0

    def total_exposure_pct(self, marks: dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = sum(pos.qty * marks.get(symbol, pos.entry_price) for symbol, pos in self.positions.items())
        return total / equity

    def open_position_report(self, marks: dict[str, float]) -> list[dict[str, Any]]:
        rows = []
        for symbol, pos in self.positions.items():
            mark = float(marks.get(symbol, pos.entry_price))
            self.update_extremes(symbol, mark)
            market_value = pos.qty * mark
            gross = (mark - pos.entry_price) * pos.qty
            estimated_exit_fee = market_value * self.fee_rate
            unrealized_net = gross - pos.entry_fee - estimated_exit_fee
            cost = pos.entry_price * pos.qty + pos.entry_fee
            return_pct = (unrealized_net / cost * 100.0) if cost > 0 else 0.0
            rows.append({
                "symbol": symbol, "qty": pos.qty, "entry_price": pos.entry_price,
                "current_price": mark, "market_value": market_value, "entry_fee": pos.entry_fee,
                "estimated_exit_fee": estimated_exit_fee, "unrealized_pnl": unrealized_net,
                "return_pct": return_pct, "entry_time": pos.entry_time,
                "stop_price": pos.stop_price, "highest_price": pos.highest_price,
                "lowest_price": pos.lowest_price, "trailing_stop_price": pos.trailing_stop_price,
                "trailing_active": pos.trailing_active, "partial_taken": pos.partial_taken,
                "scale_count": pos.scale_count, "capital_invested": pos.capital_invested, "realized_partial_pnl": pos.realized_partial_pnl,
                "meta": pos.meta,
            })
        return rows
