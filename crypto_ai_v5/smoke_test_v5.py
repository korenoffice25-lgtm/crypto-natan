from __future__ import annotations

import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from decision_agent import Decision, Action
from meta_brain import RegimeRouter, MetaBrain
from opportunity_engine import trader_opportunity, hunter_opportunity, reversal_opportunity, swing_opportunity
from paper_exchange import PaperExchange
from portfolio_brain import PortfolioBrain
from regime_model import RegimeReading
from risk_governor import RiskGovernor, RiskState
from storage import Storage
from universe_scanner import Candidate


def row(**overrides):
    x = {
        "ret_1": 0.003, "ret_3": 0.012, "ret_12": 0.018, "ret_36": 0.025,
        "abs_ret_1": 0.003, "realized_vol_12": 0.004, "realized_vol_36": 0.005,
        "range_pct": 0.014, "body_pct": 0.006, "volume_z_24": 2.8,
        "volume_change": 1.2, "atr_pct": 0.006, "rsi_centered": 0.15,
        "ema_distance_fast": 0.008, "ema_distance_slow": 0.012,
    }
    x.update(overrides)
    return pd.DataFrame([x])


def frame(n=220, drift=.001, seed=1):
    rng=np.random.default_rng(seed); ret=drift+rng.normal(0,.003,n); close=100*np.exp(np.cumsum(ret));
    return pd.DataFrame({"timestamp":pd.date_range("2025-01-01",periods=n,freq="15min",tz="UTC"),
                         "open":np.r_[close[0],close[:-1]],"high":close*1.003,"low":close*.997,"close":close,
                         "volume":rng.lognormal(8,0.3,n)})


def main():
    cand=Candidate("SOL/USDT","SOL","USDT",150_000_000,.08,2,2_000_000,12,.9,1,1,True,True)
    reg=RegimeReading(1,.85,.45,.7); returns=np.linspace(-.002,.003,72)
    dec=Decision(Action.BUY,.74,55,20,.12,"test")
    tr=trader_opportunity(candidate=cand,decision=dec,row=row(),best_bid=99.99,best_ask=100.01,spread_bps=2,
                          orderbook_imbalance=.25,trade_flow_imbalance=.35,regime=reg,memory_multiplier=1,model_weight=1,
                          repeat_penalty=1,returns=returns)
    assert tr and tr.brain=="TRADER"
    hu=hunter_opportunity(candidate=cand,row=row(),best_bid=99.99,best_ask=100.01,spread_bps=2,orderbook_imbalance=.35,
                          microprice_edge_bps=2,trade_flow_imbalance=.45,regime=reg,memory_multiplier=1,model_weight=1,
                          repeat_penalty=1,returns=returns,min_volume_z=1.1,min_momentum_pct=.0025,micro_model_edge_bps=4)
    assert hu and hu.brain=="HUNTER"
    rev=reversal_opportunity(candidate=cand,row=row(ret_1=.003,ret_3=-.004,ret_12=-.035,rsi_centered=-.55,body_pct=.004,volume_z_24=1.0),
                             best_bid=99,best_ask=99.1,spread_bps=3,orderbook_imbalance=.4,trade_flow_imbalance=.5,regime=reg,
                             memory_multiplier=1,model_weight=1,repeat_penalty=1,returns=returns,min_drop_pct=.006)
    assert rev and rev.brain=="REVERSAL"
    sw=swing_opportunity(candidate=cand,frames={"15m":frame(seed=1),"1h":frame(seed=2),"4h":frame(seed=3)},
                         best_bid=100,best_ask=100.1,spread_bps=2,regime=reg,memory_multiplier=1,model_weight=1,
                         repeat_penalty=1,returns=returns)
    assert sw and sw.brain=="SWING"

    router=RegimeRouter(); mr=router.classify(.72,.012,.06); ranked=MetaBrain().apply([tr,hu,sw],mr,{"TRADER":1,"HUNTER":1,"SWING":1})
    assert ranked and all(0 <= x.meta_score <= 100 for x in ranked)

    broker=PaperExchange(10_000,.001,2); risk=RiskGovernor(.0045,.15,.60,.02,.02,.08)
    st=RiskState(10_000,10_000,0,0,10_000,10_000,0)
    approval=risk.approve_buy(target_exposure_pct=.12,state=st,price=100,volatility=.01,market_risk_multiplier=1)
    assert approval.allowed
    buy=broker.buy("SOL/USDT",approval.quantity,100,97,meta={"brain":"HUNTER","score":80,"live_meta_score":85})
    assert buy["ok"]
    add=broker.add_to_position("SOL/USDT",1,102,98,reason="TEST_SCALE")
    assert add["ok"]
    broker.update_trailing("SOL/USDT",106,.018,.015)
    part=broker.reduce("SOL/USDT",105,.4,reason="PARTIAL_TAKE_PROFIT")
    assert part["ok"] and "SOL/USDT" in broker.positions
    sell=broker.exit("SOL/USDT",104,reason="TEST_EXIT")
    assert sell["ok"] and sell["pnl_net"] != 0

    cfg=SimpleNamespace(min_position_pct=.02,max_position_pct=.15,max_total_exposure_pct=.60,
                        rotation_min_age_seconds=0,rotation_score_advantage=10,rotation_reduce_fraction=.5,
                        scale_score_threshold=80,scale_min_profit_pct=.005,max_scales_per_position=2,
                        scale_cooldown_seconds=0,scale_fraction=.35)
    pb=PortfolioBrain(cfg)
    open_positions=[]
    plan=pb.plan_new_allocations(ranked,open_positions,.0,1.0)
    assert sum(1 for a in plan if a.action=="BUY") >= 2, "V5 must support multiple simultaneous allocations"

    with tempfile.TemporaryDirectory() as td:
        s=Storage(td+"/v5.sqlite3")
        s.add_journal(1,"TEST","ok","SOL/USDT","HUNTER",{})
        s.add_shadow_event(1,"SOL/USDT",100,"REJECTED_OPPORTUNITY","HUNTER",80,"test",{})
        s.update_shadow("SOL/USDT",7_300_000,105)
        assert s.recent_journal(1)[0]["event"]=="TEST"
        s.close()

    print("V5 smoke test passed")
    print("brains: Trader/Hunter/Swing/Reversal OK")
    print("Meta Brain + Regime Router OK")
    print("multi-position Portfolio Brain OK")
    print("scale-in + partial profit + runner + full exit OK")
    print("journal + shadow logging OK")


if __name__=="__main__":
    main()
