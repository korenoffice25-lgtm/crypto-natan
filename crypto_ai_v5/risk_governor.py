from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RiskState:
    equity: float
    cash: float
    current_exposure_pct: float
    total_exposure_pct: float
    day_start_equity: float
    peak_equity: float
    open_positions: int


@dataclass
class RiskApproval:
    allowed: bool
    approved_exposure_pct: float
    quantity: float
    reason: str


class RiskGovernor:
    """Non-negotiable hard limits. No brain/meta/coprocessor can override this layer."""

    def __init__(
        self,
        risk_per_trade: float,
        max_position_pct: float,
        max_total_exposure_pct: float,
        min_position_pct: float,
        max_daily_loss_pct: float,
        max_drawdown_pct: float,
    ):
        self.risk_per_trade = float(risk_per_trade)
        self.max_position_pct = float(max_position_pct)
        self.max_total_exposure_pct = float(max_total_exposure_pct)
        self.min_position_pct = float(min_position_pct)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)

    def portfolio_halt_reason(self, state: RiskState) -> str:
        if state.equity <= state.day_start_equity * (1 - self.max_daily_loss_pct):
            return "Daily loss limit reached"
        if state.equity <= state.peak_equity * (1 - self.max_drawdown_pct):
            return "Portfolio drawdown kill-switch reached"
        return ""

    def approve_buy(
        self,
        *,
        target_exposure_pct: float,
        state: RiskState,
        price: float,
        volatility: float,
        market_risk_multiplier: float = 1.0,
        existing_symbol_exposure_pct: float = 0.0,
    ) -> RiskApproval:
        halt = self.portfolio_halt_reason(state)
        if halt:
            return RiskApproval(False, 0.0, 0.0, halt)

        remaining_total = max(0.0, self.max_total_exposure_pct - state.total_exposure_pct)
        symbol_remaining = max(0.0, self.max_position_pct - existing_symbol_exposure_pct)
        market_cap = self.max_position_pct * max(0.0, min(market_risk_multiplier, 1.0))
        target = min(float(target_exposure_pct), market_cap, symbol_remaining, remaining_total)
        if target < self.min_position_pct:
            return RiskApproval(False, 0.0, 0.0, "Insufficient risk/exposure capacity for minimum useful allocation")

        vol = max(float(volatility), 0.001)
        max_notional_by_risk = state.equity * self.risk_per_trade / vol
        target_notional = min(state.equity * target, max_notional_by_risk, state.cash * 0.995)
        approved_exposure = target_notional / state.equity if state.equity > 0 else 0.0
        qty = target_notional / price if price > 0 else 0.0
        if approved_exposure < self.min_position_pct or qty <= 0 or not math.isfinite(qty):
            return RiskApproval(False, 0.0, 0.0, "Risk sizing resolved below minimum position")
        return RiskApproval(True, approved_exposure, qty, "Approved by V5 hard risk layer")
