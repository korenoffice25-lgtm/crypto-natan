from __future__ import annotations

from types import SimpleNamespace
import tempfile

import numpy as np
import pandas as pd

from brains import trader_signal, hunter_signal, reversal_signal, swing_signal
from domain import Candidate, FeaturePacket
from meta_brain import MetaBrain
from regime import RegimeRouter
from risk_engine import RiskEngine, RiskState
from paper_broker import PaperBroker
from portfolio_brain import PortfolioBrain
from storage import Storage


def feature(**kw):
    d=dict(symbol="SOL/USDT",price=100,best_bid=99.98,best_ask=100.02,spread_bps=4,volatility=.006,
           ret_1=.003,ret_3=.012,ret_12=.028,ret_36=.045,volume_z=2.8,volume_change=1.2,rsi=61,
           ema_fast=99,ema_slow=97,ema_fast_distance=.010,ema_slow_distance=.030,
           orderbook_imbalance=.35,trade_flow_imbalance=.42,microprice_edge_bps=2.2,
           return_vector=list(np.linspace(-.003,.004,96)))
    d.update(kw); return FeaturePacket(**d)


def frame(seed=1, drift=.001):
    rng=np.random.default_rng(seed); r=drift+rng.normal(0,.003,220); close=100*np.exp(np.cumsum(r))
    return pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=220,freq="15min",tz="UTC"),"open":np.r_[close[0],close[:-1]],"high":close*1.003,"low":close*.997,"close":close,"volume":rng.lognormal(8,.3,220)})


def settings():
    return SimpleNamespace(
        absolute_max_exposure_pct=.80,max_position_pct=.16,min_position_pct=.018,min_cash_reserve_pct=.12,max_open_risk_pct=.05,max_daily_loss_pct=.025,max_drawdown_pct=.085,
        trader_risk_per_trade=.0045,hunter_risk_per_trade=.0055,swing_risk_per_trade=.005,reversal_risk_per_trade=.0038,
        exceptional_score=84,rotation_score_advantage=10,rotation_reduce_fraction=.45,rotation_min_age_seconds=0,utilization_tolerance_pct=.04,
        scale_score_threshold=82,scale_min_profit_pct=.008,scale_fraction=.35,max_scales_per_position=2,scale_cooldown_seconds=0
    )


def main():
    c=Candidate("SOL/USDT","SOL","USDT",150_000_000,.08,.12,4,1_000_000,88,1.0,is_major=True,fast_cabinet=True)
    f=feature()
    tr=trader_signal(c,f); hu=hunter_signal(c,f)
    rev=reversal_signal(c,feature(ret_12=-.035,ret_3=-.004,ret_1=.003,rsi=30,orderbook_imbalance=.5,trade_flow_imbalance=.45))
    sw=swing_signal(c,f,{"15m":frame(1),"1h":frame(2),"4h":frame(3)})
    assert tr and hu and rev and sw, "all four brains must produce valid synthetic signals"
    regime=RegimeRouter().classify(.76,.018,.07)
    ranked=MetaBrain().apply([tr,hu,sw],regime,{})
    assert ranked[0].utility>=ranked[-1].utility and all(0<=x.meta_score<=100 for x in ranked)

    cfg=settings(); risk=RiskEngine(cfg); broker=PaperBroker(10_000,.001,2)
    rs=RiskState(10_000,10_000,0,0,10_000,10_000)
    ap=risk.approve_open(brain="HUNTER",target_exposure_pct=.12,stop_distance_pct=.026,price=100,state=rs,regime_max_exposure_pct=.80,candidate_risk_multiplier=1)
    assert ap.allowed and ap.approved_exposure_pct>=cfg.min_position_pct
    buy=broker.open(symbol="SOL/USDT",brain="HUNTER",qty=ap.quantity,ask=100,stop_price=97.4,meta={"live_meta_score":88},action_id="a1")
    assert buy["ok"]
    assert not broker.open(symbol="SOL/USDT",brain="HUNTER",qty=1,ask=100,stop_price=97,meta={},action_id="a1")["ok"]
    add=broker.add(symbol="SOL/USDT",qty=.5,ask=102,action_id="a2"); assert add["ok"]
    broker.update_trailing("SOL/USDT",106,.02,.015)
    part=broker.reduce(symbol="SOL/USDT",bid=105,fraction=.35,action_id="a3",reason="PARTIAL_TAKE_PROFIT"); assert part["ok"]
    sell=broker.close(symbol="SOL/USDT",bid=104,action_id="a4",reason="TEST"); assert sell["ok"] and sell["pnl_net"]!=0

    pb=PortfolioBrain(cfg)
    plan=pb.plan(ranked,[],0,regime)
    assert len([x for x in plan if x.action=="OPEN"])>=2, "ranked signals must be allowed to compete for real execution-time capacity"

    with tempfile.TemporaryDirectory() as td:
        st=Storage(td+"/v6.sqlite3")
        st.add_journal("TEST","ok","SOL/USDT","HUNTER",{})
        st.add_fill("SOL/USDT",sell)
        st.add_equity(10100,5000,50,100)
        st.add_shadow_event("ETH/USDT",100,"REJECTED_OPPORTUNITY","SWING",72,"test")
        st.update_shadow("ETH/USDT",103)
        assert st.recent_journal(1)[0]["event"]=="TEST"
        assert st.performance_summary()["closed_trades"]==1
        assert st.missed_opportunities(5)[0]["symbol"]=="ETH/USDT"
        assert isinstance(st.score_calibration(),list)
        st.close()

    print("V6 smoke test passed")
    print("4 independent brains: OK")
    print("Meta/Regime + utility ranking: OK")
    print("Risk sizing + open-risk cap: OK")
    print("paper open/add/partial/runner/close + idempotency: OK")
    print("Portfolio capital competition: OK")
    print("persistence analytics: OK")


if __name__=="__main__":
    main()
