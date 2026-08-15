from __future__ import annotations

import numpy as np
import pandas as pd

from decision_agent import Decision, Action
from opportunity_engine import trader_opportunity, hunter_opportunity
from paper_exchange import PaperExchange
from regime_model import RegimeReading
from risk_governor import RiskGovernor, RiskState
from universe_scanner import Candidate


def row():
    return pd.DataFrame([{
        "ret_1": 0.002, "ret_3": 0.012, "ret_12": 0.018, "ret_36": 0.02,
        "abs_ret_1": 0.002, "realized_vol_12": 0.004, "realized_vol_36": 0.005,
        "range_pct": 0.015, "body_pct": 0.008, "volume_z_24": 2.8,
        "volume_change": 1.4, "atr_pct": 0.006, "rsi_centered": 0.3,
        "ema_distance_fast": 0.008, "ema_distance_slow": 0.012,
    }])


def main():
    cand = Candidate("SOL/USDT", "SOL", "USDT", 150_000_000, 0.08, 2.0, 2_000_000, 12.0, 0.9)
    reg = RegimeReading(1, 0.85, 0.45, 0.7)
    dec = Decision(Action.BUY, 0.72, 52.0, 20.0, 0.12, "test")
    returns = np.linspace(-0.002, 0.003, 60)

    trader = trader_opportunity(
        candidate=cand, decision=dec, row=row(), best_bid=99.99, best_ask=100.01,
        spread_bps=2.0, orderbook_imbalance=0.25, trade_flow_imbalance=0.35,
        regime=reg, memory_multiplier=1.05, repeat_penalty=1.0, returns=returns,
    )
    assert trader and trader.engine == "TRADER" and trader.score > 50

    hunter = hunter_opportunity(
        candidate=cand, row=row(), best_bid=99.99, best_ask=100.01, spread_bps=2.0,
        orderbook_imbalance=0.35, microprice_edge_bps=2.0, trade_flow_imbalance=0.45,
        regime=reg, memory_multiplier=1.0, repeat_penalty=1.0, returns=returns,
        min_volume_z=1.15, min_momentum_pct=0.003, micro_model_edge_bps=4.0,
    )
    assert hunter and hunter.engine == "HUNTER" and hunter.score > 60

    broker = PaperExchange(10_000, 0.001, 2)
    risk = RiskGovernor(0.0045, 0.15, 0.60, 0.025, 0.02, 0.08)
    state = RiskState(10_000, 10_000, 0, 0, 10_000, 10_000, 0)
    approval = risk.approve_buy(
        target_exposure_pct=0.12, state=state, price=100,
        volatility=0.01, market_risk_multiplier=1.0,
    )
    assert approval.allowed and 0.025 <= approval.approved_exposure_pct <= 0.15
    buy = broker.buy("SOL/USDT", approval.quantity, 100, 97, meta={"engine": "HUNTER", "score": 80})
    assert buy["ok"]
    broker.update_trailing("SOL/USDT", 104, activation_pct=0.012, trail_distance_pct=0.015)
    pos = broker.positions["SOL/USDT"]
    assert pos.trailing_active and pos.trailing_stop_price > pos.stop_price
    hit, reason = broker.stop_hit("SOL/USDT", pos.trailing_stop_price - 0.01)
    assert hit and reason == "TRAILING_STOP"
    sell = broker.exit("SOL/USDT", 103.0, reason=reason)
    assert sell["ok"] and sell["meta"]["engine"] == "HUNTER"

    print("V4 smoke test passed")
    print(f"Trader score={trader.score:.1f} Hunter score={hunter.score:.1f}")
    print(f"Approved exposure={approval.approved_exposure_pct:.1%}")


if __name__ == "__main__":
    main()
