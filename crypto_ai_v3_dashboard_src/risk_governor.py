from __future__ import annotations

from dataclasses import dataclass
import math

from decision_agent import Decision, Action


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
    action: Action
    approved_exposure_pct: float
    quantity: float
    reason: str


class RiskGovernor:
    def __init__(
        self,
        risk_per_trade: float,
        max_position_pct: float,
        max_total_exposure_pct: float,
        max_daily_loss_pct: float,
        max_drawdown_pct: float,
        max_positions: int,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_positions = max_positions

    def approve(
        self,
        decision: Decision,
        state: RiskState,
        price: float,
        volatility: float,
        market_risk_multiplier: float = 1.0,
    ) -> RiskApproval:
        if decision.action in (Action.DO_NOTHING, Action.HOLD):
            return RiskApproval(True, decision.action, state.current_exposure_pct, 0.0, "No new risk")

        if decision.action == Action.EXIT:
            return RiskApproval(True, Action.EXIT, 0.0, 0.0, "Risk-reducing action always allowed")

        if state.equity <= state.day_start_equity * (1 - self.max_daily_loss_pct):
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "Daily loss limit reached")

        if state.equity <= state.peak_equity * (1 - self.max_drawdown_pct):
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "Portfolio drawdown kill-switch reached")

        if state.open_positions >= self.max_positions:
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "Maximum open positions reached")

        if decision.action != Action.BUY:
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "Unsupported risk-increasing action")

        remaining_total = max(0.0, self.max_total_exposure_pct - state.total_exposure_pct)
        market_cap = self.max_position_pct * max(0.0, min(market_risk_multiplier, 1.0))
        target = min(decision.target_exposure_pct, market_cap, remaining_total)

        if target <= 0:
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "No portfolio exposure capacity remaining")

        # Approximate risk sizing from short-horizon volatility.
        vol = max(float(volatility), 0.001)
        max_notional_by_risk = state.equity * self.risk_per_trade / vol
        target_notional = min(state.equity * target, max_notional_by_risk, state.cash * 0.995)
        qty = target_notional / price if price > 0 else 0.0
        approved_exposure = target_notional / state.equity if state.equity > 0 else 0.0

        if qty <= 0 or not math.isfinite(qty):
            return RiskApproval(False, Action.DO_NOTHING, 0.0, 0.0, "Position size resolved to zero")

        return RiskApproval(True, Action.BUY, approved_exposure, qty, "Approved within hard limits")
