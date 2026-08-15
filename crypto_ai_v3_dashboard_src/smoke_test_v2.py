from __future__ import annotations

import numpy as np
import pandas as pd

from decision_agent import DecisionAgent, Action
from features_v2 import add_market_features
from market_state import orderbook_state, trade_flow_state
from regime_model import RegimeModel
from return_model import MultiHorizonReturnModel
from risk_governor import RiskGovernor, RiskState


def synthetic_candles(n=1300, seed=9):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")

    # Several changing statistical regimes; not a single permanent trend.
    vol = np.where((np.arange(n) // 180) % 2 == 0, 0.0025, 0.006)
    drift = np.where((np.arange(n) // 120) % 3 == 0, 0.00025, -0.00005)
    returns = drift + rng.normal(0, vol, n)

    close = 50_000 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0.0015, 0.001, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(8.5, 0.45, n) * (1 + vol * 60)

    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def main():
    raw = synthetic_candles()
    feat = add_market_features(raw, (1, 3, 12), include_targets=True)
    inference_feat = add_market_features(raw, (1, 3, 12), include_targets=False)
    assert inference_feat["timestamp"].iloc[-1] == raw["timestamp"].iloc[-1]

    model = MultiHorizonReturnModel((1, 3, 12)).fit(feat)
    regime = RegimeModel().fit(feat)
    row = feat.iloc[[-1]]

    preds = model.predict(row)
    reg = regime.read(row)

    ob = orderbook_state({
        "bids": [[49999, 3.0], [49998, 4.0], [49997, 5.0], [49996, 4.0], [49995, 5.0]],
        "asks": [[50001, 1.5], [50002, 2.0], [50003, 2.0], [50004, 3.0], [50005, 3.0]],
    }, levels=5)

    tf = trade_flow_state([
        {"price": 50000, "amount": 0.5, "side": "buy"},
        {"price": 50001, "amount": 0.4, "side": "buy"},
        {"price": 49999, "amount": 0.2, "side": "sell"},
    ])

    agent = DecisionAgent()
    decision = agent.decide(
        preds,
        reg,
        spread_bps=ob.spread_bps,
        orderbook_imbalance=ob.imbalance_5,
        trade_flow_imbalance=tf.flow_imbalance,
        has_position=False,
    )

    risk = RiskGovernor(0.004, 0.20, 0.35, 0.015, 0.08)
    approval = risk.approve(
        decision,
        RiskState(
            equity=10_000,
            cash=10_000,
            current_exposure_pct=0.0,
            day_start_equity=10_000,
            peak_equity=10_000,
        ),
        price=50_000,
        volatility=max(float(row["realized_vol_12"].iloc[0]), 0.001),
    )

    print("Predictions:")
    for p in preds:
        print(f"  h={p.horizon}: exp_return={p.expected_return:+.6f}, validation_mae={p.validation_mae:.6f}")
    print("Regime:", reg)
    print("Order book imbalance:", round(ob.imbalance_5, 4))
    print("Trade flow imbalance:", round(tf.flow_imbalance, 4))
    print("Decision:", decision)
    print("Risk approval:", approval)

    assert 0 <= decision.confidence <= 1
    assert ob.best_bid < ob.best_ask
    assert -1 <= ob.imbalance_5 <= 1
    assert -1 <= tf.flow_imbalance <= 1
    print("\nV2 smoke test passed.")


if __name__ == "__main__":
    main()
