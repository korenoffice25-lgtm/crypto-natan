from dataclasses import dataclass

@dataclass
class RiskDecision:
    allowed: bool
    quantity: float = 0.0
    reason: str = ""

def size_position(
    equity: float,
    price: float,
    atr_value: float,
    risk_per_trade: float,
    stop_loss_atr: float,
    max_position_pct: float,
) -> RiskDecision:
    if equity <= 0 or price <= 0 or atr_value <= 0:
        return RiskDecision(False, reason="Invalid market/account values")

    stop_distance = atr_value * stop_loss_atr
    risk_budget = equity * risk_per_trade

    qty_by_risk = risk_budget / stop_distance
    qty_by_cap = (equity * max_position_pct) / price
    qty = min(qty_by_risk, qty_by_cap)

    if qty <= 0:
        return RiskDecision(False, reason="Calculated position is zero")

    return RiskDecision(True, quantity=qty, reason="OK")
