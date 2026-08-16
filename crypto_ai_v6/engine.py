from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from brains import all_signals
from calibration import PerformanceCalibrator
from config import SETTINGS
from copilot import TradingCopilot
from domain import Candidate, FeaturePacket, MarketRegime, PortfolioAction, Signal
from features import build_feature_packet
from market_data import MarketDataGateway
from meta_brain import MetaBrain
from paper_broker import PaperBroker
from portfolio_brain import PortfolioBrain
from regime import RegimeRouter
from risk_engine import RiskEngine, RiskState
from scanner import UniverseScanner
from state_store import StateStore
from storage import Storage


class V6Engine:
    def __init__(self, settings=SETTINGS):
        self.cfg = settings
        self._prepare_run()
        self.gateway = MarketDataGateway(settings.exchange_id)
        self.scanner = UniverseScanner(self.gateway, settings)
        self.storage = Storage(settings.db_path)
        self.state_store = StateStore(settings.state_path)
        saved = self.state_store.load()
        broker_state = saved.get("broker") if saved else None
        self.broker = PaperBroker.from_state(broker_state, settings.fee_rate, settings.slippage_bps) if broker_state else PaperBroker(settings.starting_cash, settings.fee_rate, settings.slippage_bps)
        self.risk = RiskEngine(settings)
        self.regime_router = RegimeRouter()
        self.meta_brain = MetaBrain()
        self.portfolio_brain = PortfolioBrain(settings)
        self.calibrator = PerformanceCalibrator(self.storage)

        self.radar: list[Candidate] = []
        self.universe: list[Candidate] = []
        self.fast_cabinet: list[Candidate] = []
        self.candidate_cache: dict[str, Candidate] = {}
        self.last_scan_at = 0.0
        self.rotation_cursor = int(saved.get("rotation_cursor", 0) or 0) if saved else 0
        self.regime: MarketRegime = self.regime_router.classify(0.5, 0.0, 0.0)
        self.swing_cache: dict[str, tuple[float, dict[str, pd.DataFrame]]] = {}

        self.latest_marks: dict[str, float] = dict(saved.get("latest_marks", {})) if saved else {}
        self.latest_quotes: dict[str, dict[str, float]] = {}
        self.latest_return_vectors: dict[str, list[float]] = {}
        self.last_features: dict[str, FeaturePacket] = {}
        self.cooldowns: dict[str, int] = {k:int(v) for k,v in (saved.get("cooldowns", {}) if saved else {}).items()}
        risk_saved = saved.get("risk", {}) if saved else {}
        self.current_day = risk_saved.get("current_day")
        self.day_start_equity = float(risk_saved.get("day_start_equity", self.broker.starting_cash))
        self.peak_equity = float(risk_saved.get("peak_equity", self.broker.starting_cash))

        self.top_opportunities: list[dict[str, Any]] = []
        self.fast_cabinet_view: list[dict[str, Any]] = []
        self.next_actions: list[dict[str, Any]] = []
        self.last_rejections: list[dict[str, Any]] = []
        self.latest_status: dict[str, Any] = {"state":"starting","version":6,"last_update_ms":None}
        self.health_state: dict[str, Any] = {"ok":False,"state":"starting","last_good_cycle_ms":None,"cycle_errors":[]}
        self.running = True
        self.copilot = TradingCopilot(self, settings)

    def _prepare_run(self):
        data = Path(self.cfg.data_dir)
        data.mkdir(parents=True, exist_ok=True)
        marker = Path(self.cfg.run_marker_path)
        previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if previous != self.cfg.paper_run_id:
            for p in (Path(self.cfg.state_path), Path(self.cfg.db_path)):
                if p.exists():
                    p.unlink()
            marker.write_text(self.cfg.paper_run_id, encoding="utf-8")

    def _save_state(self):
        self.state_store.save({
            "version":6,
            "paper_run_id":self.cfg.paper_run_id,
            "broker":self.broker.snapshot(),
            "latest_marks":self.latest_marks,
            "cooldowns":self.cooldowns,
            "rotation_cursor":self.rotation_cursor,
            "risk":{"current_day":self.current_day,"day_start_equity":self.day_start_equity,"peak_equity":self.peak_equity},
        })

    async def refresh_universe(self):
        bundle = await self.scanner.scan()
        self.radar = bundle["radar"]
        self.universe = bundle["deep"]
        self.fast_cabinet = bundle["fast"]
        self.candidate_cache.update({c.symbol:c for c in self.radar})
        self.regime = self.regime_router.classify(bundle["breadth_positive_pct"],bundle["median_change_pct"],bundle["median_range_pct"])
        self.last_scan_at = time.time()
        self.storage.add_journal("REGIME", self.regime.reason, payload=self.regime.to_dict())

    def _cycle_candidates(self) -> list[Candidate]:
        seen=set(); out=[]
        # 1) Open positions must always be managed.
        for symbol,p in self.broker.positions.items():
            c=self.candidate_cache.get(symbol)
            if c is None:
                c=Candidate(symbol,symbol.split("/")[0],self.cfg.quote_currency,0,0,0,0,0,0,0.35,is_major=symbol.split("/")[0] in self.cfg.major_bases)
            out.append(c); seen.add(symbol)
        # 2) Hunter Fast Lane.
        for c in self.fast_cabinet:
            if c.symbol not in seen:
                out.append(c); seen.add(c.symbol)
        # 3) Majors.
        for c in self.universe:
            if c.is_major and c.symbol not in seen:
                out.append(c); seen.add(c.symbol)
        # 4) Rotating deep batch.
        rem=[c for c in self.universe if c.symbol not in seen]
        if rem:
            n=min(self.cfg.rotation_batch_size,len(rem)); start=self.rotation_cursor%len(rem)
            rot=rem[start:]+rem[:start]
            out.extend(rot[:n]); self.rotation_cursor=(start+n)%len(rem)
        return out

    async def _market_packet(self, symbol: str):
        return await asyncio.gather(
            self.gateway.fetch_order_book(symbol,self.cfg.orderbook_levels),
            self.gateway.fetch_trades(symbol,self.cfg.trade_limit),
            self.gateway.fetch_ohlcv_df(symbol,self.cfg.live_timeframe,self.cfg.live_candles),
        )

    async def _swing_frames(self, symbol: str) -> dict[str,pd.DataFrame] | None:
        cached=self.swing_cache.get(symbol)
        if cached and time.time()-cached[0]<self.cfg.swing_refresh_seconds:
            return cached[1]
        try:
            vals=await asyncio.gather(*[self.gateway.fetch_ohlcv_df(symbol,tf,self.cfg.swing_candles) for tf in self.cfg.swing_timeframes])
            frames={tf:df for tf,df in zip(self.cfg.swing_timeframes,vals)}
            self.swing_cache[symbol]=(time.time(),frames)
            return frames
        except Exception:
            return cached[1] if cached else None

    def _on_cooldown(self,symbol:str,now_ms:int)->bool:
        until=int(self.cooldowns.get(symbol,0) or 0)
        if until<=now_ms:
            self.cooldowns.pop(symbol,None); return False
        return True

    def _set_cooldown(self,symbol:str,now_ms:int):
        self.cooldowns[symbol]=now_ms+self.cfg.cooldown_seconds*1000

    def _risk_state(self)->RiskState:
        eq=self.broker.equity(self.latest_marks)
        now_ms=int(time.time()*1000); day=now_ms//86_400_000
        if self.current_day!=day:
            self.current_day=day; self.day_start_equity=eq
        self.peak_equity=max(self.peak_equity,eq)
        return RiskState(eq,self.broker.cash,self.broker.total_exposure_pct(self.latest_marks,eq),self.broker.open_risk_pct(self.latest_marks,eq),self.day_start_equity,self.peak_equity)

    def _life(self,brain:str)->dict[str,float]:
        return {
            "TRADER":{"maturity":self.cfg.trader_min_maturity_seconds,"confirm":self.cfg.trader_exit_confirmations,"partial":self.cfg.trader_partial_profit_pct,"trail_activation":self.cfg.trader_trail_activation_pct,"trail_distance":self.cfg.trader_trail_distance_pct,"exit_score":43,"severe_score":24,"stop":self.cfg.trader_stop_pct},
            "HUNTER":{"maturity":self.cfg.hunter_min_maturity_seconds,"confirm":self.cfg.hunter_exit_confirmations,"partial":self.cfg.hunter_partial_profit_pct,"trail_activation":self.cfg.hunter_trail_activation_pct,"trail_distance":self.cfg.hunter_trail_distance_pct,"exit_score":45,"severe_score":25,"stop":self.cfg.hunter_stop_pct},
            "SWING":{"maturity":self.cfg.swing_min_maturity_seconds,"confirm":self.cfg.swing_exit_confirmations,"partial":self.cfg.swing_partial_profit_pct,"trail_activation":self.cfg.swing_trail_activation_pct,"trail_distance":self.cfg.swing_trail_distance_pct,"exit_score":48,"severe_score":27,"stop":self.cfg.swing_stop_pct},
            "REVERSAL":{"maturity":self.cfg.reversal_min_maturity_seconds,"confirm":self.cfg.reversal_exit_confirmations,"partial":self.cfg.reversal_partial_profit_pct,"trail_activation":self.cfg.reversal_trail_activation_pct,"trail_distance":self.cfg.reversal_trail_distance_pct,"exit_score":44,"severe_score":24,"stop":self.cfg.reversal_stop_pct},
        }.get(brain.upper(),{"maturity":180,"confirm":2,"partial":.02,"trail_activation":.015,"trail_distance":.015,"exit_score":44,"severe_score":24,"stop":.025})

    @staticmethod
    def _age(entry_time:str)->float:
        try:
            d=datetime.fromisoformat(entry_time.replace("Z","+00:00"))
            return max(0.0,(datetime.now(timezone.utc)-d).total_seconds())
        except Exception:
            return 0.0

    @staticmethod
    def _corr(a:list[float],b:list[float])->float:
        if len(a)<12 or len(b)<12:return 0.0
        n=min(len(a),len(b)); x=np.asarray(a[-n:],float); y=np.asarray(b[-n:],float)
        if np.std(x)<=1e-12 or np.std(y)<=1e-12:return 0.0
        c=float(np.corrcoef(x,y)[0,1]); return c if np.isfinite(c) else 0.0

    def _generic_live_score(self,f:FeaturePacket)->float:
        trend=1.0 if f.ema_fast>f.ema_slow else -1.0
        raw=50 + trend*7 + max(-15,min(15,f.ret_3*600)) + max(-10,min(10,f.ret_12*220)) + 8*f.orderbook_imbalance + 8*f.trade_flow_imbalance
        return max(0.0,min(100.0,raw))

    def _update_lifecycle(self,p,mark_return:float,age:float,life:dict[str,float]):
        if p.partial_taken:
            p.lifecycle="RUNNER"
        elif mark_return>=life["partial"]:
            p.lifecycle="WINNER"
        elif age>=life["maturity"]:
            p.lifecycle="MATURE"
        elif age>=life["maturity"]*0.25:
            p.lifecycle="BUILDING"
        else:
            p.lifecycle="ENTRY"

    def _record_exit(self,symbol:str,fill:dict[str,Any],now_ms:int):
        self.storage.add_fill(symbol,fill,now_ms)
        brain=str((fill.get("meta") or {}).get("brain") or fill.get("brain") or "UNKNOWN")
        self.storage.add_journal("EXIT",str(fill.get("reason") or "EXIT"),symbol,brain,fill,now_ms)
        self.storage.add_shadow_event(symbol,float(fill.get("exit_price") or fill.get("fill") or 0),"POST_EXIT",brain,float((fill.get("meta") or {}).get("entry_meta_score",0) or 0),str(fill.get("reason") or "EXIT"),fill,now_ms)
        self._set_cooldown(symbol,now_ms)
        self._save_state()

    async def _manage_position(self,candidate:Candidate,f:FeaturePacket,signals:list[Signal],now_ms:int)->bool:
        p=self.broker.positions.get(candidate.symbol)
        if not p:return False
        life=self._life(p.brain)
        vol_mult={"TRADER":2.0,"HUNTER":2.3,"SWING":3.0,"REVERSAL":1.9}.get(p.brain,2.0)
        dynamic_trail=max(life["trail_distance"],min(0.07,f.volatility*vol_mult))
        self.broker.update_trailing(p.symbol,f.price,life["trail_activation"],dynamic_trail)
        hit,why=self.broker.stop_hit(p.symbol,f.best_bid)
        if hit:
            aid=self.broker.make_action_id(str(now_ms),"CLOSE",p.symbol,why)
            fill=self.broker.close(symbol=p.symbol,bid=f.best_bid,action_id=aid,reason=why,extra_slippage_bps=f.spread_bps*0.15)
            if fill.get("ok"):self._record_exit(p.symbol,fill,now_ms)
            return True

        same=next((s for s in signals if s.brain==p.brain),None)
        live=float(same.meta_score if same else self._generic_live_score(f))
        p.meta["live_meta_score"]=live
        p.meta["last_evaluated_ms"]=now_ms
        p.meta["regime"]=self.regime.name
        if same:
            p.meta["live_confidence"]=same.confidence; p.meta["live_edge_bps"]=same.expected_edge_bps
        mark_return=f.price/p.entry_price-1 if p.entry_price else 0.0
        age=self._age(p.entry_time)
        self._update_lifecycle(p,mark_return,age,life)

        if not p.partial_taken and mark_return>=life["partial"]:
            aid=self.broker.make_action_id(str(now_ms),"PARTIAL",p.symbol,p.lifecycle)
            part=self.broker.reduce(symbol=p.symbol,bid=f.best_bid,fraction=self.cfg.partial_profit_fraction,action_id=aid,reason="PARTIAL_TAKE_PROFIT",extra_slippage_bps=f.spread_bps*0.12)
            if part.get("ok"):
                p.lifecycle="RUNNER"; self.storage.add_fill(p.symbol,part,now_ms); self.storage.add_journal("PARTIAL_TAKE_PROFIT",f"{p.brain} locked profit; runner remains",p.symbol,p.brain,part,now_ms)

        severe=live<=life["severe_score"]
        if p.brain=="HUNTER" and f.ret_3<-0.018 and f.trade_flow_imbalance<-0.45:severe=True
        if p.brain=="REVERSAL" and f.ret_3<-0.022:severe=True
        exit_signal=live<life["exit_score"]
        count=int(p.meta.get("exit_signal_count",0) or 0)
        if exit_signal:
            count+=1; p.meta["exit_signal_count"]=count
            if severe or (age>=life["maturity"] and count>=int(life["confirm"])):
                reason="THESIS_INVALIDATED_SEVERE" if severe else f"THESIS_INVALIDATED_CONFIRMED_{count}"
                aid=self.broker.make_action_id(str(now_ms),"CLOSE",p.symbol,reason)
                fill=self.broker.close(symbol=p.symbol,bid=f.best_bid,action_id=aid,reason=reason,extra_slippage_bps=f.spread_bps*0.15)
                if fill.get("ok"):self._record_exit(p.symbol,fill,now_ms)
                return True
            self.storage.add_journal("EXIT_PENDING",f"{p.brain} invalidation {count}/{int(life['confirm'])}; lifecycle={p.lifecycle}",p.symbol,p.brain,{"live_score":live,"age":age},now_ms)
        else:
            if count:p.meta["exit_signal_count"]=0
            self.storage.add_journal("HOLD",f"{p.brain} thesis valid; lifecycle={p.lifecycle}",p.symbol,p.brain,{"live_score":live,"return_pct":mark_return*100},now_ms)
        return False

    async def analyze_candidate(self,candidate:Candidate,performance_weights:dict[str,float])->list[Signal]:
        ob,trades,candles=await self._market_packet(candidate.symbol)
        f=build_feature_packet(candidate.symbol,candles,ob,trades,self.cfg.correlation_lookback)
        if not f:return []
        try:
            candle_ts = pd.Timestamp(candles["timestamp"].iloc[-1]).timestamp()
            if time.time()-candle_ts > self.cfg.max_data_age_seconds:
                raise RuntimeError(f"stale market data: {time.time()-candle_ts:.0f}s old")
        except KeyError:
            pass
        self.last_features[candidate.symbol]=f; self.latest_marks[candidate.symbol]=f.price
        self.storage.update_shadow(candidate.symbol,f.price,int(time.time()*1000))
        self.latest_quotes[candidate.symbol]={"bid":f.best_bid,"ask":f.best_ask,"spread_bps":f.spread_bps,"volatility":f.volatility}
        self.latest_return_vectors[candidate.symbol]=f.return_vector
        need_swing=candidate.fast_cabinet or candidate.is_major or (candidate.symbol in self.broker.positions and self.broker.positions[candidate.symbol].brain=="SWING")
        frames=await self._swing_frames(candidate.symbol) if need_swing else None
        signals=self.meta_brain.apply(all_signals(candidate,f,frames),self.regime,performance_weights)
        now_ms=int(time.time()*1000)
        if candidate.symbol in self.broker.positions:
            await self._manage_position(candidate,f,signals,now_ms)
            return []
        if self._on_cooldown(candidate.symbol,now_ms):return []
        thresholds={"TRADER":self.cfg.trader_min_score,"HUNTER":self.cfg.hunter_min_score,"SWING":self.cfg.swing_min_score,"REVERSAL":self.cfg.reversal_min_score}
        qualified=[]
        for sig in signals:
            threshold=thresholds.get(sig.brain,999)
            if sig.meta_score>=threshold:
                qualified.append(sig)
            else:
                reason=f"Meta score {sig.meta_score:.1f} below {sig.brain} threshold {threshold:.1f}"
                self.storage.add_opportunity(sig.to_dict(),"REJECTED_THRESHOLD",reason,now_ms)
                if sig.meta_score>=max(58.0,threshold-5):
                    self.storage.add_shadow_event(sig.symbol,sig.best_ask,"REJECTED_OPPORTUNITY",sig.brain,sig.meta_score,reason,sig.to_dict(),now_ms)
        return qualified

    def _filter_correlation(self,ranked:list[Signal])->list[Signal]:
        accepted=[]
        for s in ranked:
            bad=""
            for a in accepted:
                c=self._corr(s.return_vector,a.return_vector)
                if c>=self.cfg.max_pair_correlation:
                    bad=f"Correlation {c:.2f} with selected {a.symbol}"; break
            if not bad:
                for symbol in self.broker.positions:
                    if symbol==s.symbol:continue
                    vec=self.latest_return_vectors.get(symbol)
                    if vec:
                        c=self._corr(s.return_vector,vec)
                        if c>=self.cfg.max_pair_correlation:
                            bad=f"Correlation {c:.2f} with open {symbol}"; break
            if bad:
                self.storage.add_opportunity(s.to_dict(),"REJECTED",bad)
                self.storage.add_shadow_event(s.symbol,s.best_ask,"REJECTED_OPPORTUNITY",s.brain,s.meta_score,bad,s.to_dict())
                self.last_rejections.insert(0,{"symbol":s.symbol,"brain":s.brain,"score":s.meta_score,"reason":bad})
            else:accepted.append(s)
        return accepted

    def _liquidity_slippage(self,s:Signal)->float:
        return max(0.0,s.spread_bps*0.15 + self.cfg.liquidity_slippage_scale_bps*(1-s.candidate_risk_multiplier))

    def _annotated_positions(self,equity:float)->list[dict[str,Any]]:
        rows=self.broker.report(self.latest_marks)
        for p in rows:p["portfolio_equity"]=equity
        return rows

    async def execute_portfolio(self,signals:list[Signal],cycle_id:str):
        weights=self.calibrator.weights()
        ranked=self.meta_brain.apply(signals,self.regime,weights)
        strongest={}
        for s in ranked:
            if s.symbol not in strongest or s.utility>strongest[s.symbol].utility:strongest[s.symbol]=s
        ranked=sorted(strongest.values(),key=lambda x:(x.utility,x.meta_score),reverse=True)
        ranked=self._filter_correlation(ranked)
        self.top_opportunities=[s.to_dict() for s in ranked[:25]]
        state=self._risk_state(); positions=self._annotated_positions(state.equity)
        plan=self.portfolio_brain.plan(ranked,positions,state.total_exposure_pct,self.regime)
        self.next_actions=[a.to_dict() for a in plan]

        for idx,a in enumerate(plan):
            if a.action=="CASH":
                self.storage.add_journal("CASH",a.reason,payload={"regime":self.regime.to_dict()}); continue
            if a.action in {"REDUCE","ROTATE"}:
                p=self.broker.positions.get(a.symbol); q=self.latest_quotes.get(a.symbol) or {}
                bid=float(q.get("bid") or self.latest_marks.get(a.symbol,0))
                if p and bid>0:
                    aid=self.broker.make_action_id(cycle_id,a.action,a.symbol,str(idx))
                    red=self.broker.reduce(symbol=a.symbol,bid=bid,fraction=a.fraction,action_id=aid,reason="CAPITAL_ROTATION" if a.action=="ROTATE" else "REGIME_REDUCE",extra_slippage_bps=float(q.get("spread_bps",0))*0.12)
                    if red.get("ok"):
                        self.storage.add_fill(a.symbol,red); self.storage.add_journal(a.action,a.reason,a.symbol,p.brain,red)
                continue
            if a.action!="OPEN" or not a.signal:continue
            s=a.signal
            if s.symbol in self.broker.positions:continue
            state=self._risk_state()
            approval=self.risk.approve_open(brain=s.brain,target_exposure_pct=a.target_exposure_pct,stop_distance_pct=s.stop_distance_pct,
                                            price=s.best_ask,state=state,regime_max_exposure_pct=self.regime.max_utilization_pct,
                                            candidate_risk_multiplier=s.candidate_risk_multiplier)
            if not approval.allowed:
                self.storage.add_opportunity(s.to_dict(),"REJECTED",approval.reason)
                self.storage.add_shadow_event(s.symbol,s.best_ask,"REJECTED_OPPORTUNITY",s.brain,s.meta_score,approval.reason,s.to_dict())
                self.last_rejections.insert(0,{"symbol":s.symbol,"brain":s.brain,"score":s.meta_score,"reason":approval.reason})
                continue
            stop=s.best_ask*(1-s.stop_distance_pct)
            meta={"brain":s.brain,"entry_score":s.score,"entry_meta_score":s.meta_score,"live_meta_score":s.meta_score,
                  "confidence":s.confidence,"expected_edge_bps":s.expected_edge_bps,"setup_key":s.setup_key,"entry_reason":s.reason,
                  "regime":self.regime.name,"utility":s.utility,"exit_signal_count":0,"last_scale_ms":0,"context":s.context}
            aid=self.broker.make_action_id(cycle_id,"OPEN",s.symbol,str(idx))
            fill=self.broker.open(symbol=s.symbol,brain=s.brain,qty=approval.quantity,ask=s.best_ask,stop_price=stop,meta=meta,action_id=aid,extra_slippage_bps=self._liquidity_slippage(s))
            if fill.get("ok"):
                self.storage.add_fill(s.symbol,fill); self.storage.add_opportunity(s.to_dict(),"ENTERED",approval.reason)
                self.storage.add_journal("ENTER",f"{s.brain}: {s.reason}; allocation {approval.approved_exposure_pct:.1%}",s.symbol,s.brain,{"signal":s.to_dict(),"approval":approval.__dict__})
                self._save_state()
            else:
                self.storage.add_opportunity(s.to_dict(),"REJECTED",str(fill.get("reason")))

        await self._scale_winners(cycle_id)

    async def _scale_winners(self,cycle_id:str):
        state=self._risk_state(); free=max(0.0,min(self.regime.max_utilization_pct,self.cfg.absolute_max_exposure_pct)-state.total_exposure_pct)
        if free<self.cfg.min_position_pct:return
        rows=sorted(self._annotated_positions(state.equity),key=lambda p:float((p.get("meta") or {}).get("live_meta_score",0) or 0),reverse=True)
        for idx,p in enumerate(rows):
            if free<self.cfg.min_position_pct:break
            symbol=p["symbol"]; current=float(p["market_value"])/state.equity if state.equity else 0
            ok,target,reason=self.portfolio_brain.scale_decision(p,current,free)
            if not ok:continue
            q=self.latest_quotes.get(symbol) or {}; ask=float(q.get("ask") or self.latest_marks.get(symbol,0)); pos=self.broker.positions.get(symbol)
            if ask<=0 or not pos:continue
            rs=self._risk_state()
            approval=self.risk.approve_open(brain=pos.brain,target_exposure_pct=target,stop_distance_pct=max(0.008,1-pos.stop_price/ask),price=ask,state=rs,
                                            regime_max_exposure_pct=self.regime.max_utilization_pct,symbol_current_exposure_pct=current,candidate_risk_multiplier=1.0)
            if not approval.allowed:continue
            aid=self.broker.make_action_id(cycle_id,"ADD",symbol,str(idx))
            add=self.broker.add(symbol=symbol,qty=approval.quantity,ask=ask,action_id=aid,reason="SCALE_WINNER",extra_slippage_bps=float(q.get("spread_bps",0))*0.12)
            if add.get("ok"):
                pos.meta["last_scale_ms"]=int(time.time()*1000); self.storage.add_fill(symbol,add); self.storage.add_journal("ADD",reason,symbol,pos.brain,add)
                free=max(0.0,free-approval.approved_exposure_pct); self._save_state()

    def _fast_view(self,signals:list[Signal]):
        strongest={}
        for s in signals:
            if s.symbol not in strongest or s.meta_score>strongest[s.symbol].meta_score:strongest[s.symbol]=s
        rows=[]
        for c in self.fast_cabinet[:25]:
            s=strongest.get(c.symbol)
            rows.append({"symbol":c.symbol,"radar_rank":c.radar_rank,"deep_rank":c.deep_rank,"market_score":c.market_score,
                         "brain":s.brain if s else "WATCH","score":s.meta_score if s else 0.0,"confidence":s.confidence if s else 0.0,
                         "decision":"QUALIFIED" if s else "WATCH","reason":s.reason if s else "Fast Cabinet: no brain currently clears qualification"})
        self.fast_cabinet_view=rows

    async def run_forever(self):
        await self.gateway.load_markets()
        try:
            while self.running:
                started=time.time(); cycle_id=str(int(started*1000)); errors=[]; all_signals_live=[]
                try:
                    if not self.universe or time.time()-self.last_scan_at>=self.cfg.scanner_refresh_seconds:
                        await self.refresh_universe()
                except Exception as e:
                    errors.append({"stage":"scanner","error":str(e)})
                weights=self.calibrator.weights(); candidates=self._cycle_candidates()
                sem=asyncio.Semaphore(max(1,self.cfg.scanner_concurrency))
                async def analyze_safe(c):
                    async with sem:
                        try:
                            return c, await self.analyze_candidate(c,weights), None
                        except Exception as e:
                            return c, [], str(e)
                results=await asyncio.gather(*(analyze_safe(c) for c in candidates)) if candidates else []
                for c,sigs,err in results:
                    all_signals_live.extend(sigs)
                    if err: errors.append({"stage":"symbol","symbol":c.symbol,"error":err})
                try:
                    await self.execute_portfolio(all_signals_live,cycle_id)
                except Exception as e:
                    errors.append({"stage":"portfolio","error":str(e),"trace":traceback.format_exc(limit=3)})
                self._fast_view(all_signals_live)
                rs=self._risk_state(); now_ms=int(time.time()*1000)
                state_name="running" if not errors else "degraded"
                self.latest_status={"state":state_name,"version":6,"mode":"PAPER_ONLY","exchange":self.cfg.exchange_id,"equity":rs.equity,"cash":rs.cash,
                                    "realized_pnl":self.broker.realized_pnl,"open_positions":len(self.broker.positions),"capital_utilization_pct":rs.total_exposure_pct*100,
                                    "open_risk_pct":rs.open_risk_pct*100,"radar_size":len(self.radar),"deep_size":len(self.universe),"fast_cabinet_size":len(self.fast_cabinet),
                                    "analyzed_this_cycle":len(candidates),"opportunities_found":len(all_signals_live),"regime":self.regime.name,
                                    "cycle_duration_seconds":time.time()-started,"last_update_ms":now_ms,"errors":errors}
                self.health_state={"ok":not errors,"state":state_name,"last_good_cycle_ms":now_ms if not errors else self.health_state.get("last_good_cycle_ms"),"cycle_errors":errors}
                self.storage.add_equity(rs.equity,rs.cash,rs.total_exposure_pct*100,self.broker.realized_pnl,now_ms)
                self._save_state()
                await asyncio.sleep(max(1.0,self.cfg.cycle_seconds-(time.time()-started)))
        finally:
            self._save_state(); await self.gateway.close()

    async def shutdown(self):
        self.running=False

    def report(self)->dict[str,Any]:
        rs=self._risk_state(); positions=self._annotated_positions(rs.equity); unreal=sum(float(p.get("unrealized_pnl",0)) for p in positions)
        perf=self.storage.performance_summary()
        allocation=[{"label":"Cash","symbol":"CASH","value":self.broker.cash}]+[{"label":p["symbol"].split("/")[0],"symbol":p["symbol"],"value":p["market_value"]} for p in positions]
        return {"version":6,"mode":"PAPER_ONLY","paper_run_id":self.cfg.paper_run_id,"asset_class":"CRYPTO_PAPER","starting_cash":self.broker.starting_cash,
                "equity":rs.equity,"cash":rs.cash,"realized_pnl":self.broker.realized_pnl,"unrealized_pnl":unreal,"total_pnl":rs.equity-self.broker.starting_cash,
                "total_return_pct":((rs.equity/self.broker.starting_cash)-1)*100 if self.broker.starting_cash else 0.0,
                "capital_utilization_pct":rs.total_exposure_pct*100,"open_risk_pct":rs.open_risk_pct*100,
                "absolute_max_capital_utilization_pct":self.cfg.absolute_max_exposure_pct*100,"open_position_count":len(positions),"open_positions":positions,
                "allocation":allocation,"regime":self.regime.to_dict(),"top_opportunities":self.top_opportunities,"fast_cabinet":self.fast_cabinet_view,
                "next_actions":self.next_actions,"last_rejections":self.last_rejections[:20],"performance":perf,"by_regime":self.storage.performance_by_regime(),
                "score_calibration":self.storage.score_calibration(),"shadow_analysis":self.storage.shadow_summary(100),"missed_opportunities":self.storage.missed_opportunities(30),
                "model_weights":self.calibrator.weights(),"radar_count":len(self.radar),"deep_analysis_count":len(self.universe),"health":self.health_state,"last_update_ms":self.latest_status.get("last_update_ms")}

    def status(self)->dict[str,Any]:
        r=self.report()
        return {**self.latest_status,"positions":r["open_positions"],"regime_detail":r["regime"],"top_opportunities":r["top_opportunities"],
                "fast_cabinet":r["fast_cabinet"],"next_actions":r["next_actions"],"health":r["health"]}

    def chat(self,message:str)->dict[str,Any]:
        return self.copilot.answer(message)
