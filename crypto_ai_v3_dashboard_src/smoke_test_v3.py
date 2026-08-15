from __future__ import annotations

from universe_scanner import (
    eligible_market, ticker_activity, prefilter_score, book_quality,
    final_score, risk_multiplier,
)
from paper_exchange import PaperExchange
from risk_governor import RiskGovernor, RiskState
from decision_agent import Decision, Action


def main():
    market = {"spot": True, "active": True, "quote": "USDT", "base": "SOL"}
    assert eligible_market(market, "USDT")
    assert not eligible_market({"spot": True, "active": True, "quote": "USDT", "base": "USDC"}, "USDT")

    ticker = {"last": 100, "high": 110, "low": 92, "quoteVolume": 25_000_000}
    qv, rp = ticker_activity(ticker)
    assert qv == 25_000_000
    assert 0.17 < rp < 0.19
    assert prefilter_score(qv, rp) > 0

    ob = {
        "bids": [[99.99, 100], [99.98, 150]],
        "asks": [[100.01, 100], [100.02, 150]],
    }
    spread, depth = book_quality(ob, 2)
    assert 0 < spread < 5
    assert depth > 40_000
    assert final_score(qv, rp, spread, depth) > 0
    rm = risk_multiplier(qv, spread, depth)
    assert 0.15 <= rm <= 1.0

    broker = PaperExchange(10_000, fee_rate=0.001, slippage_bps=2)
    buy = broker.buy("SOL/USDT", qty=5, ask=100, stop_price=96)
    assert buy["ok"]
    assert broker.emergency_stop_hit("SOL/USDT", 95.5)
    open_report = broker.open_position_report({"SOL/USDT": 103.0})
    assert len(open_report) == 1
    assert "unrealized_pnl" in open_report[0]
    assert "return_pct" in open_report[0]

    state = broker.snapshot()
    broker2 = PaperExchange.from_state(state, 0.001, 2)
    assert "SOL/USDT" in broker2.positions
    sell = broker2.exit("SOL/USDT", bid=105.0, reason="TEST_EXIT")
    assert sell["ok"]
    assert sell["pnl_net"] > 0
    assert sell["return_pct"] > 0
    assert sell["entry_price"] > 0 and sell["exit_price"] > 0

    risk = RiskGovernor(0.0035, 0.10, 0.25, 0.015, 0.08, max_positions=4)
    decision = Decision(Action.BUY, 0.75, 30.0, 10.0, 0.10, "test")
    approval = risk.approve(
        decision,
        RiskState(
            equity=10_000, cash=9_000, current_exposure_pct=0,
            total_exposure_pct=0.20, day_start_equity=10_000,
            peak_equity=10_000, open_positions=2,
        ),
        price=100,
        volatility=0.02,
        market_risk_multiplier=0.5,
    )
    assert approval.allowed
    assert approval.approved_exposure_pct <= 0.05 + 1e-9

    print("V3 smoke test passed")
    print(f"scanner: range={rp:.2%}, spread={spread:.2f}bps, depth=${depth:,.0f}, risk_mult={rm:.2f}")
    print(f"paper persistence: cash=${broker2.cash:,.2f}, positions={len(broker2.positions)}")
    print(f"risk approval exposure={approval.approved_exposure_pct:.2%}")


if __name__ == "__main__":
    main()
