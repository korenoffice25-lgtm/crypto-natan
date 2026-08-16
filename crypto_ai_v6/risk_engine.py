from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RiskState:
    equity: float
    cash: float
    total_exposure_pct: float
    open_risk_pct: float
    day_start_equity: float
    peak_equity: float


@dataclass
class RiskApproval:
    allowed: bool
    approved_exposure_pct: float = 0.0
    quantity: float = 0.0
    reason: str = ""


class RiskEngine:
    """Hard risk layer. No AI/brain can override it."""

    def __init__(self, settings):
        self.cfg = settings

    def halt_reason(self, state: RiskState) -> str:
        if state.equity <= 0:
            return "Equity is non-positive"
        if state.day_start_equity > 0 and state.equity <= state.day_start_equity*(1-self.cfg.max_daily_loss_pct):
            return "Daily loss guard active"
        if state.peak_equity > 0 and state.equity <= state.peak_equity*(1-self.cfg.max_drawdown_pct):
            return "Portfolio drawdown kill-switch active"
        return ""

    def risk_per_trade(self, brain: str) -> float:
        return {
            "TRADER": self.cfg.trader_risk_per_trade,
            "HUNTER": self.cfg.hunter_risk_per_trade,
            "SWING": self.cfg.swing_risk_per_trade,
            "REVERSAL": self.cfg.reversal_risk_per_trade,
        }.get(brain.upper(), self.cfg.trader_risk_per_trade)

    def approve_open(self, *, brain: str, target_exposure_pct: float, stop_distance_pct: float,
                     price: float, state: RiskState, regime_max_exposure_pct: float,
                     symbol_current_exposure_pct: float = 0.0, candidate_risk_multiplier: float = 1.0) -> RiskApproval:
        halt = self.halt_reason(state)
        if halt:
            return RiskApproval(False, reason=halt)
        hard_total_cap = min(self.cfg.absolute_max_exposure_pct, regime_max_exposure_pct)
        remaining_total = max(0.0, hard_total_cap-state.total_exposure_pct)
        remaining_symbol = max(0.0, self.cfg.max_position_pct-symbol_current_exposure_pct)
        risk_room = max(0.0, self.cfg.max_open_risk_pct-state.open_risk_pct)
        stop = max(0.008, float(stop_distance_pct))
        trade_risk = min(self.risk_per_trade(brain), risk_room)
        if trade_risk <= 0:
            return RiskApproval(False, reason="Open-risk budget exhausted")
        by_stop = trade_risk/stop
        target = min(float(target_exposure_pct), remaining_total, remaining_symbol, by_stop)
        target *= max(0.30, min(1.0, candidate_risk_multiplier))
        max_cash_exposure = max(0.0, state.cash/state.equity-self.cfg.min_cash_reserve_pct) if state.equity > 0 else 0.0
        target = min(target, max_cash_exposure)
        if target < self.cfg.min_position_pct:
            return RiskApproval(False, reason="Approved sizing fell below minimum useful allocation")
        qty = state.equity*target/price if price > 0 else 0.0
        if qty <= 0 or not math.isfinite(qty):
            return RiskApproval(False, reason="Invalid quantity")
        return RiskApproval(True, target, qty, "Approved by V6 hard risk engine")
