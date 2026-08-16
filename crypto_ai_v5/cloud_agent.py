from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from config import SETTINGS
from copilot import TradingCopilot
from decision_agent import DecisionAgent, Action
from exchange_gateway import ExchangeGateway
from features_v2 import add_market_features
from market_state import orderbook_state, trade_flow_state, MarketSnapshot
from memory_engine import TradeMemory, ModelCompetition
from meta_brain import RegimeRouter, MetaBrain, MarketRegime
from microstructure_model import MicrostructureReturnModel, add_time_targets, engineer_micro_features, load_snapshots
from model_registry import LearnedModelRegistry
from opportunity_engine import Opportunity, trader_opportunity, hunter_opportunity, reversal_opportunity, swing_opportunity
from paper_exchange import PaperExchange, PaperPosition
from portfolio_brain import PortfolioBrain
from regime_model import RegimeReading
from risk_governor import RiskGovernor, RiskState
from state_store import StateStore
from storage import Storage
from universe_scanner import UniverseScanner, Candidate, ScanBundle


class CloudPaperAgent:
    def __init__(self, settings=SETTINGS):
        self.cfg = settings
        self._prepare_run()
        self.gateway = ExchangeGateway(settings.exchange_id)
        self.scanner = UniverseScanner(self.gateway, settings)
        self.storage = Storage(settings.db_path)
        self.state_store = StateStore(settings.state_path)

        legacy = []
        if settings.learn_from_v4:
            legacy.append(settings.legacy_v4_db_path)
        if settings.learn_from_v3:
            legacy.append(settings.legacy_v3_db_path)
        self.memory = TradeMemory(settings.db_path, legacy, settings.min_memory_trades)
        self.model_competition = ModelCompetition(self.memory, settings.model_promotion_min_trades)
        self.models = LearnedModelRegistry(self.gateway, settings)

        saved = self.state_store.load() or {}
        broker_state = saved.get("broker")
        self.broker = PaperExchange.from_state(broker_state, settings.fee_rate, settings.slippage_bps) if broker_state else PaperExchange(settings.starting_cash, settings.fee_rate, settings.slippage_bps)
        self.decision_agent = DecisionAgent(settings.round_trip_cost_buffer_bps, settings.min_confidence, settings.exit_confidence)
        self.risk = RiskGovernor(settings.risk_per_trade, settings.max_position_pct, settings.max_total_exposure_pct,
                                 settings.min_position_pct, settings.max_daily_loss_pct, settings.max_drawdown_pct)
        self.regime_router = RegimeRouter()
        self.meta_brain = MetaBrain()
        self.portfolio_brain = PortfolioBrain(settings)

        self.micro_models: dict[str, MicrostructureReturnModel] = {}
        self.micro_trained_at: dict[str, float] = {}
        self.micro_training: set[str] = set()
        self.model_tasks: dict[str, asyncio.Task] = {}
        self.swing_cache: dict[str, tuple[float, dict[str, pd.DataFrame]]] = {}

        self.scan_bundle: ScanBundle | None = None
        self.universe: list[Candidate] = []
        self.fast_cabinet: list[Candidate] = []
        self.radar: list[dict[str, Any]] = []
        self.rotation_cursor = int(saved.get("rotation_cursor", 0) or 0)
        self.last_scan_at = 0.0
        self.market_regime: MarketRegime = self.regime_router.classify(0.5, 0.0, 0.0)

        self.latest_marks: dict[str, float] = dict(saved.get("latest_marks", {}))
        self.latest_quotes: dict[str, dict[str, float]] = {}
        self.latest_return_vectors: dict[str, list[float]] = {}
        self.latest_status: dict[str, Any] = {"state": "starting"}
        self.top_opportunities: list[dict[str, Any]] = []
        self.fast_cabinet_view: list[dict[str, Any]] = []
        self.cooldowns: dict[str, int] = {k: int(v) for k, v in saved.get("cooldowns", {}).items()}
        risk_saved = saved.get("risk", {})
        self.current_day = risk_saved.get("current_day")
        self.day_start_equity = float(risk_saved.get("day_start_equity", self.broker.starting_cash))
        self.peak_equity = float(risk_saved.get("peak_equity", self.broker.starting_cash))
        self.running = True
        self.copilot = TradingCopilot(self)

    def _prepare_run(self):
        data = Path(self.cfg.data_dir)
        data.mkdir(parents=True, exist_ok=True)
        marker = Path(self.cfg.run_marker_path)
        previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if previous != self.cfg.paper_run_id:
            # New PAPER_RUN_ID means a deliberate clean V5 forward-test. Legacy V3/V4 data is never touched.
            for path in (Path(self.cfg.db_path), Path(self.cfg.state_path)):
                if path.exists():
                    path.unlink()
            marker.write_text(self.cfg.paper_run_id, encoding="utf-8")

    def _save_state(self):
        self.state_store.save({
            "version": 5, "paper_run_id": self.cfg.paper_run_id, "broker": self.broker.snapshot(),
            "risk": {"current_day": self.current_day, "day_start_equity": self.day_start_equity, "peak_equity": self.peak_equity},
            "latest_marks": self.latest_marks, "cooldowns": self.cooldowns, "rotation_cursor": self.rotation_cursor,
        })

    def _schedule_model(self, symbol: str):
        task = self.model_tasks.get(symbol)
        if task and not task.done():
            return
        async def train():
            try:
                await self.models.ensure(symbol)
            except Exception:
                pass
        self.model_tasks[symbol] = asyncio.create_task(train())

    async def _train_micro_model_if_ready(self, symbol: str):
        if symbol in self.micro_training or self.storage.market_snapshot_count(symbol) < 1000:
            return
        age = time.time() - self.micro_trained_at.get(symbol, 0.0)
        if symbol in self.micro_models and age < self.cfg.model_retrain_seconds:
            return
        self.micro_training.add(symbol)
        try:
            def fit():
                snapshots = load_snapshots(self.cfg.db_path, symbol)
                features = engineer_micro_features(snapshots)
                labeled = add_time_targets(features, (30, 120, 300))
                return MicrostructureReturnModel((30, 120, 300)).fit(labeled)
            self.micro_models[symbol] = await asyncio.to_thread(fit)
            self.micro_trained_at[symbol] = time.time()
        except Exception:
            pass
        finally:
            self.micro_training.discard(symbol)

    async def refresh_universe(self):
        bundle = await self.scanner.scan()
        symbols = {c.symbol for c in bundle.deep}
        # Always monitor open positions even if they leave the deep list.
        for symbol in self.broker.positions:
            if symbol not in symbols:
                bundle.deep.append(Candidate(
                    symbol=symbol, base=symbol.split("/")[0], quote=self.cfg.quote_currency,
                    quote_volume=0.0, intraday_range_pct=0.0, spread_bps=0.0, depth_notional=0.0,
                    opportunity_score=-999.0, risk_multiplier=0.20, is_major=False,
                ))
        self.scan_bundle = bundle
        self.universe = bundle.deep
        self.fast_cabinet = bundle.fast_cabinet
        self.radar = [r.to_dict() for r in bundle.radar]
        self.market_regime = self.regime_router.classify(bundle.breadth_positive_pct, bundle.median_change_pct, bundle.median_range_pct)
        self.last_scan_at = time.time()
        self.storage.add_universe(int(time.time()*1000), {
            "radar_count": len(bundle.radar), "deep_count": len(bundle.deep),
            "fast_cabinet": [c.to_dict() for c in bundle.fast_cabinet], "regime": self.market_regime.to_dict(),
        })
        self.storage.add_journal(int(time.time()*1000), "REGIME", self.market_regime.reason, payload=self.market_regime.to_dict())

    def _cycle_candidates(self) -> list[Candidate]:
        if not self.universe:
            return []
        priority: list[Candidate] = []
        seen = set()
        for c in self.fast_cabinet:
            if c.symbol not in seen:
                priority.append(c); seen.add(c.symbol)
        for c in self.universe:
            if c.is_major and c.symbol not in seen:
                priority.append(c); seen.add(c.symbol)
        for symbol in self.broker.positions:
            c = next((x for x in self.universe if x.symbol == symbol), None)
            if c and c.symbol not in seen:
                priority.append(c); seen.add(c.symbol)
        remainder = [c for c in self.universe if c.symbol not in seen]
        if remainder:
            n = min(self.cfg.rotation_batch_size, len(remainder))
            start = self.rotation_cursor % len(remainder)
            rotated = remainder[start:] + remainder[:start]
            priority.extend(rotated[:n])
            self.rotation_cursor = (start + n) % len(remainder)
        return priority

    async def _market_packet(self, symbol: str):
        return await asyncio.gather(
            self.gateway.fetch_order_book(symbol, self.cfg.orderbook_levels),
            self.gateway.fetch_trades(symbol, self.cfg.trade_limit),
            self.gateway.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.live_feature_candles),
        )

    async def _swing_frames(self, symbol: str) -> dict[str, pd.DataFrame] | None:
        cached = self.swing_cache.get(symbol)
        if cached and time.time() - cached[0] < self.cfg.swing_refresh_seconds:
            return cached[1]
        try:
            frames_list = await asyncio.gather(*[
                self.gateway.fetch_ohlcv_df(symbol, tf, self.cfg.swing_candle_limit) for tf in self.cfg.swing_timeframes
            ])
            frames = {tf: df for tf, df in zip(self.cfg.swing_timeframes, frames_list)}
            self.swing_cache[symbol] = (time.time(), frames)
            return frames
        except Exception:
            return cached[1] if cached else None

    def _fallback_regime(self, row: pd.DataFrame) -> RegimeReading:
        vol = max(float(row["realized_vol_36"].iloc[0]), 0.0)
        activity = abs(float(row["volume_z_24"].iloc[0]))
        return RegimeReading(cluster=0, confidence=0.5,
                             volatility_rank=float(np.tanh(vol*100)), activity_rank=float(np.tanh(activity/2)))

    def _on_cooldown(self, symbol: str, now_ms: int) -> bool:
        until = int(self.cooldowns.get(symbol, 0))
        if until <= now_ms:
            self.cooldowns.pop(symbol, None); return False
        return True

    def _set_cooldown(self, symbol: str, now_ms: int):
        self.cooldowns[symbol] = now_ms + self.cfg.cooldown_seconds * 1000

    @staticmethod
    def _corr(a: list[float], b: list[float]) -> float:
        if len(a) < 12 or len(b) < 12:
            return 0.0
        n = min(len(a), len(b)); x = np.asarray(a[-n:], float); y = np.asarray(b[-n:], float)
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return 0.0
        c = float(np.corrcoef(x, y)[0,1])
        return c if np.isfinite(c) else 0.0

    def _risk_state(self, symbol: str, mark: float) -> RiskState:
        equity = self.broker.equity(self.latest_marks)
        now_ms = int(time.time()*1000); day = now_ms // 86_400_000
        if self.current_day != day:
            self.current_day = day; self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        return RiskState(equity, self.broker.cash, self.broker.exposure_pct(symbol, mark, equity),
                         self.broker.total_exposure_pct(self.latest_marks, equity), self.day_start_equity,
                         self.peak_equity, len(self.broker.positions))

    def _brain_lifecycle(self, brain: str) -> dict[str, float]:
        b = brain.upper()
        return {
            "TRADER": {"maturity": self.cfg.trader_min_maturity_seconds, "confirm": self.cfg.trader_exit_confirmations,
                       "trail_activation": self.cfg.trader_trailing_activation_pct, "trail_mult": self.cfg.trader_trailing_vol_mult,
                       "partial": self.cfg.trader_partial_profit_pct, "stop_mult": self.cfg.trader_stop_vol_mult},
            "HUNTER": {"maturity": self.cfg.hunter_min_maturity_seconds, "confirm": self.cfg.hunter_exit_confirmations,
                       "trail_activation": self.cfg.hunter_trailing_activation_pct, "trail_mult": self.cfg.hunter_trailing_vol_mult,
                       "partial": self.cfg.hunter_partial_profit_pct, "stop_mult": self.cfg.hunter_stop_vol_mult},
            "SWING": {"maturity": self.cfg.swing_min_maturity_seconds, "confirm": self.cfg.swing_exit_confirmations,
                      "trail_activation": self.cfg.swing_trailing_activation_pct, "trail_mult": self.cfg.swing_trailing_vol_mult,
                      "partial": self.cfg.swing_partial_profit_pct, "stop_mult": self.cfg.swing_stop_vol_mult},
            "REVERSAL": {"maturity": self.cfg.reversal_min_maturity_seconds, "confirm": self.cfg.reversal_exit_confirmations,
                         "trail_activation": self.cfg.reversal_trailing_activation_pct, "trail_mult": self.cfg.reversal_trailing_vol_mult,
                         "partial": self.cfg.reversal_partial_profit_pct, "stop_mult": self.cfg.reversal_stop_vol_mult},
        }.get(b, {"maturity": 120, "confirm": 2, "trail_activation": .015, "trail_mult": 1.8, "partial": .02, "stop_mult": 2.5})

    @staticmethod
    def _age_seconds(entry_time: str) -> float:
        try:
            dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            return max(0.0, (datetime.now(timezone.utc)-dt).total_seconds())
        except Exception:
            return 0.0

    def _record_exit(self, now_ms: int, symbol: str, fill: dict[str, Any]):
        self.storage.add_fill(now_ms, symbol, fill)
        self.memory.invalidate()
        meta = fill.get("meta") or {}; brain = str(meta.get("brain") or "UNKNOWN")
        self.storage.add_journal(now_ms, "EXIT", str(fill.get("reason") or "EXIT"), symbol, brain, fill)
        pnl = float(fill.get("pnl_net") or 0.0); mfe = float(fill.get("mfe_pct") or 0.0)
        if pnl < 0 and mfe > 1.5:
            classification = "EARLY_PROFIT_NOT_CAPTURED_OR_REVERSAL"
        elif fill.get("reason") == "HARD_STOP":
            classification = "PROTECTIVE_STOP"
        elif pnl > 0 and mfe > max(2.0, float(fill.get("return_pct") or 0.0) + 1.0):
            classification = "WINNER_WITH_MISSED_UPSIDE"
        elif pnl > 0:
            classification = "PROFITABLE_EXIT"
        else:
            classification = "LOSING_EXIT"
        self.storage.add_journal(now_ms, "TRADE_CLASSIFIED", classification, symbol, brain, fill)
        self.storage.add_shadow_event(now_ms, symbol, float(fill.get("exit_price") or fill.get("fill") or 0.0),
                                      "POST_EXIT", brain, float(meta.get("score") or 0.0), fill.get("reason", ""), fill)
        self._set_cooldown(symbol, now_ms)
        self._save_state()

    async def _manage_open_position(self, candidate: Candidate, row: pd.DataFrame, regime: RegimeReading,
                                    decision, brain_signals: dict[str, Opportunity], ob, tf, now_ms: int) -> bool:
        symbol = candidate.symbol
        pos = self.broker.positions.get(symbol)
        if not pos:
            return False
        brain = str((pos.meta or {}).get("brain") or (pos.meta or {}).get("engine") or "TRADER").upper()
        life = self._brain_lifecycle(brain)
        vol = max(float(row["realized_vol_12"].iloc[0]), 0.001)
        trail_distance = max(0.006, min(0.07, vol * life["trail_mult"]))
        self.broker.update_trailing(symbol, ob.mid, life["trail_activation"], trail_distance)
        hit, stop_reason = self.broker.stop_hit(symbol, ob.best_bid)
        if hit:
            fill = self.broker.exit(symbol, ob.best_bid, max(0.0, ob.spread_bps*0.25), stop_reason)
            if fill.get("ok"):
                self._record_exit(now_ms, symbol, fill)
            return True

        # Mark-to-market metadata used by Portfolio Brain and Copilot.
        signal = brain_signals.get(brain)
        if brain == "TRADER":
            live_score = max(0.0, min(100.0, 50 + decision.expected_edge_bps*0.35 + (decision.confidence-0.5)*70))
            live_conf = decision.confidence; live_edge = decision.expected_edge_bps
            exit_signal = decision.action == Action.EXIT or live_score < 45
            severe = decision.expected_edge_bps < -35 and decision.confidence < 0.30
        else:
            live_score = float((signal.meta_score or signal.score) if signal else 35.0)
            live_conf = float(signal.confidence if signal else 0.30)
            live_edge = float(signal.expected_edge_bps if signal else 0.0)
            threshold = 46 if brain == "HUNTER" else (50 if brain == "SWING" else 47)
            exit_signal = live_score < threshold
            severe = live_score < 25
            if brain == "HUNTER":
                severe = severe or (float(row["ret_3"].iloc[0]) < -0.018 and tf.flow_imbalance < -0.45)
            elif brain == "REVERSAL":
                severe = severe or float(row["ret_3"].iloc[0]) < -0.022
        pos.meta["live_meta_score"] = live_score
        pos.meta["live_confidence"] = live_conf
        pos.meta["live_edge_bps"] = live_edge
        pos.meta["last_evaluated_ms"] = now_ms
        pos.meta["regime"] = self.market_regime.name

        # Partial take profit creates a runner rather than closing the whole winner.
        mark_return = ob.mid / pos.entry_price - 1.0 if pos.entry_price else 0.0
        if self.cfg.enable_partial_profit and not pos.partial_taken and mark_return >= life["partial"]:
            part = self.broker.reduce(symbol, ob.best_bid, self.cfg.partial_profit_fraction,
                                      max(0.0, ob.spread_bps*0.15), "PARTIAL_TAKE_PROFIT")
            if part.get("ok"):
                self.storage.add_fill(now_ms, symbol, part)
                self.storage.add_journal(now_ms, "PARTIAL_TAKE_PROFIT", f"Locked part of {brain} winner; runner remains", symbol, brain, part)

        count = int(pos.meta.get("exit_signal_count", 0) or 0)
        age = self._age_seconds(pos.entry_time)
        if exit_signal:
            count += 1; pos.meta["exit_signal_count"] = count
            if severe or (age >= life["maturity"] and count >= int(life["confirm"])):
                reason = "AI_EXIT_SEVERE" if severe else f"AI_EXIT_CONFIRMED_{count}"
                fill = self.broker.exit(symbol, ob.best_bid, max(0.0, ob.spread_bps*0.15), reason)
                if fill.get("ok"):
                    self._record_exit(now_ms, symbol, fill)
                return True
            self.storage.add_journal(now_ms, "EXIT_PENDING", f"{brain} exit signal {count}/{int(life['confirm'])}; age {age:.0f}s", symbol, brain,
                                     {"live_score": live_score, "live_confidence": live_conf, "edge_bps": live_edge, "severe": severe})
        else:
            if count:
                pos.meta["exit_signal_count"] = 0
            self.storage.add_journal(now_ms, "HOLD", f"{brain} thesis remains valid", symbol, brain,
                                     {"live_score": live_score, "live_confidence": live_conf, "edge_bps": live_edge})
        return False

    async def analyze_candidate(self, candidate: Candidate, model_weights: dict[str, float]) -> list[Opportunity]:
        symbol = candidate.symbol
        orderbook, trades, candles = await self._market_packet(symbol)
        ob = orderbook_state(orderbook, self.cfg.orderbook_levels)
        tf = trade_flow_state(trades)
        feat = add_market_features(candles, self.cfg.horizons, include_targets=False)
        if feat.empty:
            return []
        row = feat.iloc[[-1]]; now_ms = int(time.time()*1000); last = ob.mid
        self.latest_marks[symbol] = last
        self.latest_quotes[symbol] = {"bid": ob.best_bid, "ask": ob.best_ask, "spread_bps": ob.spread_bps,
                                      "volatility": max(float(row["realized_vol_12"].iloc[0]), 0.001)}
        self.storage.update_shadow(symbol, now_ms, last)

        raw_returns = candles["close"].pct_change().replace([np.inf,-np.inf],np.nan).dropna()
        return_vector = raw_returns.tail(self.cfg.correlation_lookback).to_numpy(float)
        self.latest_return_vectors[symbol] = [float(x) for x in return_vector.tolist()]
        snap = MarketSnapshot(now_ms, symbol, last, ob, tf,
                              float(raw_returns.tail(3).sum()) if len(raw_returns) else 0.0,
                              float(raw_returns.tail(12).std()) if len(raw_returns)>=2 else 0.0)
        self.storage.add_market_snapshot(now_ms, symbol, snap.flatten())
        asyncio.create_task(self._train_micro_model_if_ready(symbol))

        # Learned model trains/persists in background. Other brains operate immediately.
        model_age = time.time() - self.models.trained_at.get(symbol, 0.0)
        if symbol not in self.models.return_models:
            self._schedule_model(symbol)
            preds = None; regime = self._fallback_regime(row); trader_decision = None
        else:
            if model_age >= self.cfg.model_retrain_seconds:
                self._schedule_model(symbol)

            try:
                preds, regime = self.models.predict(symbol, row)
                trader_decision = self.decision_agent.decide(preds, regime, ob.spread_bps, ob.imbalance_20, tf.flow_imbalance,
                                                              has_position=symbol in self.broker.positions)
            except Exception:
                regime = self._fallback_regime(row); trader_decision = None

        repeat = self.memory.repeat_penalty(symbol, self.cfg.repeat_window_seconds, self.cfg.repeat_penalty_per_trade)
        brain_signals: dict[str, Opportunity] = {}

        if trader_decision is not None:
            setup = f"trader:r{regime.cluster}:vol{int(regime.volatility_rank*3)}"
            mm = self.memory.memory_multiplier(symbol, "TRADER", setup)
            # For an open Trader, create a synthetic BUY-view decision only for live-score context if positive.
            if symbol not in self.broker.positions:
                tr = trader_opportunity(candidate=candidate, decision=trader_decision, row=row, best_bid=ob.best_bid, best_ask=ob.best_ask,
                                        spread_bps=ob.spread_bps, orderbook_imbalance=ob.imbalance_20, trade_flow_imbalance=tf.flow_imbalance,
                                        regime=regime, memory_multiplier=mm, model_weight=model_weights.get("TRADER",1.0), repeat_penalty=repeat,
                                        returns=return_vector)
                if tr: brain_signals["TRADER"] = tr
            self.storage.add_decision(now_ms, symbol, "TRADER", trader_decision.action.value, trader_decision.confidence,
                                      trader_decision.expected_edge_bps, candidate.opportunity_score,
                                      {**asdict(trader_decision), "regime": asdict(regime)})
        else:
            trader_decision = type("Fallback", (), {"action": Action.DO_NOTHING, "confidence": 0.5, "expected_edge_bps": 0.0})()

        micro_edge_bps = 0.0
        micro_model = self.micro_models.get(symbol)
        if micro_model is not None:
            try:
                engineered = engineer_micro_features(pd.DataFrame([snap.flatten()]))
                mp = micro_model.predict(engineered)
                weights = np.array([1/max(p.validation_mae,1e-7) for p in mp]); vals = np.array([p.expected_return for p in mp])
                micro_edge_bps = float(np.average(vals,weights=weights)*10_000)
            except Exception:
                pass

        # Hunter
        vol_z = float(row["volume_z_24"].iloc[0]); momentum = 0.55*float(row["ret_3"].iloc[0])+0.30*max(float(row["ret_12"].iloc[0]),0)+0.15*max(float(row["ret_1"].iloc[0]),0)
        hsetup = f"hunter:r{regime.cluster}:v{int(min(max(vol_z,0),5))}:m{int(min(max(momentum*100,0),5))}"
        h = hunter_opportunity(candidate=candidate,row=row,best_bid=ob.best_bid,best_ask=ob.best_ask,spread_bps=ob.spread_bps,
                               orderbook_imbalance=ob.imbalance_20,microprice_edge_bps=ob.microprice_edge_bps,trade_flow_imbalance=tf.flow_imbalance,
                               regime=regime,memory_multiplier=self.memory.memory_multiplier(symbol,"HUNTER",hsetup),model_weight=model_weights.get("HUNTER",1.0),
                               repeat_penalty=repeat,returns=return_vector,min_volume_z=self.cfg.hunter_min_volume_z,
                               min_momentum_pct=self.cfg.hunter_min_momentum_pct,micro_model_edge_bps=micro_edge_bps)
        if h: brain_signals["HUNTER"] = h

        # Reversal
        rsetup = f"reversal:r{regime.cluster}"
        rev = reversal_opportunity(candidate=candidate,row=row,best_bid=ob.best_bid,best_ask=ob.best_ask,spread_bps=ob.spread_bps,
                                   orderbook_imbalance=ob.imbalance_20,trade_flow_imbalance=tf.flow_imbalance,regime=regime,
                                   memory_multiplier=self.memory.memory_multiplier(symbol,"REVERSAL",rsetup),model_weight=model_weights.get("REVERSAL",1.0),
                                   repeat_penalty=repeat,returns=return_vector,min_drop_pct=self.cfg.reversal_min_drop_pct)
        if rev: brain_signals["REVERSAL"] = rev

        # Swing is evaluated on Fast Cabinet, majors and existing Swing positions only.
        is_fast_or_swing = candidate.fast_cabinet or candidate.is_major or str((self.broker.positions.get(symbol).meta if symbol in self.broker.positions else {}).get("brain", "")) == "SWING"
        frames = await self._swing_frames(symbol) if is_fast_or_swing else None
        if frames:
            ssetup = f"swing:r{regime.cluster}"
            sw = swing_opportunity(candidate=candidate,frames=frames,best_bid=ob.best_bid,best_ask=ob.best_ask,spread_bps=ob.spread_bps,
                                   regime=regime,memory_multiplier=self.memory.memory_multiplier(symbol,"SWING",ssetup),model_weight=model_weights.get("SWING",1.0),
                                   repeat_penalty=repeat,returns=return_vector)
            if sw: brain_signals["SWING"] = sw

        # Apply current Meta/Regime multipliers to all live brain signals before position management.
        signals = self.meta_brain.apply(list(brain_signals.values()), self.market_regime, model_weights)
        brain_signals = {s.brain: s for s in signals}

        if symbol in self.broker.positions:
            closed = await self._manage_open_position(candidate,row,regime,trader_decision,brain_signals,ob,tf,now_ms)
            return []
        if self._on_cooldown(symbol, now_ms):
            return []

        thresholds = {"TRADER":self.cfg.trader_min_score,"HUNTER":self.cfg.hunter_min_score,"SWING":self.cfg.swing_min_score,"REVERSAL":self.cfg.reversal_min_score}
        out = []
        for s in signals:
            if (s.meta_score or s.score) >= thresholds.get(s.brain, 999):
                out.append(s)
        return out

    def _correlation_ok(self, opp: Opportunity, accepted: list[Opportunity]) -> tuple[bool, str]:
        for other in accepted:
            c = self._corr(opp.return_vector, other.return_vector)
            if c >= self.cfg.max_pair_correlation:
                return False, f"Correlation guard: {c:.2f} with newly selected {other.symbol}"
        for symbol in self.broker.positions:
            if symbol == opp.symbol: continue
            vec = self.latest_return_vectors.get(symbol)
            if vec:
                c = self._corr(opp.return_vector, vec)
                if c >= self.cfg.max_pair_correlation:
                    return False, f"Correlation guard: {c:.2f} with open {symbol}"
        return True, ""

    def _stop_price(self, opp: Opportunity) -> float:
        life = self._brain_lifecycle(opp.brain)
        dist = min(0.09, max(0.010, opp.volatility*life["stop_mult"]))
        return opp.best_ask * (1-dist)

    def _reject(self, now_ms: int, opp: Opportunity, reason: str):
        self.storage.add_opportunity(now_ms, opp.to_dict(), False, reason)
        self.storage.add_journal(now_ms, "REJECT", reason, opp.symbol, opp.brain, opp.to_dict())
        if (opp.meta_score or opp.score) >= self.cfg.shadow_min_score:
            self.storage.add_shadow_event(now_ms, opp.symbol, opp.best_ask, "REJECTED_OPPORTUNITY", opp.brain,
                                          opp.meta_score or opp.score, reason, opp.to_dict())

    async def execute_portfolio_plan(self, opportunities: list[Opportunity]):
        now_ms = int(time.time()*1000)
        model_weights = self.model_competition.weights()
        ranked_all = self.meta_brain.apply(opportunities, self.market_regime, model_weights)
        # Keep strongest brain thesis per symbol for capital allocation; dashboard retains all thesis rows.
        self.top_opportunities = [o.to_dict() for o in ranked_all[:20]]
        strongest: dict[str, Opportunity] = {}
        for o in ranked_all:
            if o.symbol not in strongest or (o.meta_score or o.score) > (strongest[o.symbol].meta_score or strongest[o.symbol].score):
                strongest[o.symbol] = o
        ranked = sorted(strongest.values(), key=lambda o:o.meta_score or o.score, reverse=True)

        accepted_corr: list[Opportunity] = []
        corr_filtered: list[Opportunity] = []
        for opp in ranked:
            ok, reason = self._correlation_ok(opp, accepted_corr)
            if not ok:
                self._reject(now_ms, opp, reason)
            else:
                corr_filtered.append(opp); accepted_corr.append(opp)

        equity = self.broker.equity(self.latest_marks)
        exposure = self.broker.total_exposure_pct(self.latest_marks,equity)
        open_report = self.broker.open_position_report(self.latest_marks)
        plan = self.portfolio_brain.plan_new_allocations(corr_filtered, open_report, exposure, self.market_regime.risk_multiplier)

        pending_rotation_freed: dict[str,float] = {}
        for action in plan:
            opp = action.opportunity
            if action.action == "REJECT" and opp:
                self._reject(now_ms,opp,action.reason); continue
            if action.action == "REDUCE_ROTATE":
                q = self.latest_quotes.get(action.symbol) or {}
                bid = float(q.get("bid") or self.latest_marks.get(action.symbol,0.0))
                if bid > 0 and action.symbol in self.broker.positions:
                    red = self.broker.reduce(action.symbol,bid,action.fraction,float(q.get("spread_bps",0))*0.15,"CAPITAL_ROTATION")
                    if red.get("ok"):
                        self.storage.add_fill(now_ms,action.symbol,red)
                        self.storage.add_journal(now_ms,"CAPITAL_ROTATION",action.reason,action.symbol,action.brain,red)
                        self.storage.add_shadow_event(now_ms,action.symbol,bid,"ROTATED_OUT",action.brain,0.0,action.reason,red)
                continue
            if action.action != "BUY" or not opp:
                continue
            if opp.symbol in self.broker.positions:
                continue
            state = self._risk_state(opp.symbol,opp.best_ask)
            approval = self.risk.approve_buy(target_exposure_pct=action.target_exposure_pct,state=state,price=opp.best_ask,
                                             volatility=opp.volatility,market_risk_multiplier=opp.candidate_risk_multiplier)
            if not approval.allowed:
                self._reject(now_ms,opp,approval.reason); continue
            meta = {
                "brain":opp.brain,"engine":opp.brain,"score":opp.score,"meta_score":opp.meta_score,"confidence":opp.confidence,
                "setup_key":opp.setup_key,"expected_edge_bps":opp.expected_edge_bps,"memory_multiplier":opp.memory_multiplier,
                "model_weight":opp.model_weight,"entry_reason":opp.reason,"context":opp.context,"regime":self.market_regime.name,
                "live_meta_score":opp.meta_score or opp.score,"exit_signal_count":0,"last_scale_ms":0,
            }
            fill = self.broker.buy(opp.symbol,approval.quantity,opp.best_ask,self._stop_price(opp),max(0.0,opp.spread_bps*0.15),meta)
            if fill.get("ok"):
                self.storage.add_fill(now_ms,opp.symbol,fill)
                self.storage.add_opportunity(now_ms,opp.to_dict(),True,approval.reason)
                self.storage.add_journal(now_ms,"ENTER",f"{opp.brain}: {opp.reason}; allocation {approval.approved_exposure_pct:.1%}",opp.symbol,opp.brain,
                                         {"opportunity":opp.to_dict(),"approval":asdict(approval)})
                self._save_state()
            else:
                self._reject(now_ms,opp,fill.get("reason","buy_failed"))

        # Scale only after new globally-ranked allocations had first claim on free capital.
        await self._scale_winners()

    async def _scale_winners(self):
        now_ms = int(time.time()*1000)
        equity = self.broker.equity(self.latest_marks)
        if equity <= 0: return
        exposure = self.broker.total_exposure_pct(self.latest_marks,equity)
        available = max(0.0,self.cfg.max_total_exposure_pct-exposure)
        positions = sorted(self.broker.open_position_report(self.latest_marks),
                           key=lambda p:float((p.get("meta") or {}).get("live_meta_score",0) or 0), reverse=True)
        for p in positions:
            if available < self.cfg.min_position_pct: break
            symbol=p["symbol"]; current=p["market_value"]/equity
            ok,target,reason=self.portfolio_brain.should_scale(p,current,available)
            if not ok: continue
            q=self.latest_quotes.get(symbol) or {}; ask=float(q.get("ask") or self.latest_marks.get(symbol,0))
            if ask<=0: continue
            state=self._risk_state(symbol,ask)
            approval=self.risk.approve_buy(target_exposure_pct=target,state=state,price=ask,
                                           volatility=float(q.get("volatility",0.01)),market_risk_multiplier=1.0,
                                           existing_symbol_exposure_pct=current)
            if not approval.allowed: continue
            pos=self.broker.positions.get(symbol)
            if not pos: continue
            add=self.broker.add_to_position(symbol,approval.quantity,ask,pos.stop_price,float(q.get("spread_bps",0))*0.15,"SCALE_WINNER")
            if add.get("ok"):
                pos.meta["last_scale_ms"]=now_ms
                self.storage.add_fill(now_ms,symbol,add)
                self.storage.add_journal(now_ms,"SCALE_IN",reason,symbol,str(pos.meta.get("brain","")),add)
                available=max(0.0,available-approval.approved_exposure_pct)
                self._save_state()

    def _update_fast_cabinet_view(self, opportunities: list[Opportunity]):
        by_symbol: dict[str,Opportunity] = {}
        for o in opportunities:
            if o.symbol not in by_symbol or (o.meta_score or o.score) > (by_symbol[o.symbol].meta_score or by_symbol[o.symbol].score):
                by_symbol[o.symbol]=o
        rows=[]
        for c in self.fast_cabinet[:20]:
            o=by_symbol.get(c.symbol)
            rows.append({
                "symbol":c.symbol,"radar_rank":c.radar_rank,"deep_rank":c.deep_rank,"market_score":c.opportunity_score,
                "brain":o.brain if o else "WATCH","score":(o.meta_score or o.score) if o else 0.0,
                "confidence":o.confidence if o else 0.0,"decision":"QUALIFIED" if o else "WATCH",
                "reason":o.reason if o else "Fast Cabinet: awaiting/failed brain qualification",
            })
        self.fast_cabinet_view=rows

    async def run_forever(self):
        try:
            await self.gateway.load_markets()
            while self.running:
                cycle_start=time.time()
                if not self.universe or time.time()-self.last_scan_at>=self.cfg.scanner_refresh_seconds:
                    try:
                        await self.refresh_universe()
                    except Exception as e:
                        self.latest_status={"state":"scanner_error","error":str(e)}
                model_weights=self.model_competition.weights()
                opportunities: list[Opportunity]=[]
                candidates=self._cycle_candidates()
                for c in candidates:
                    if not self.running: break
                    try:
                        opportunities.extend(await self.analyze_candidate(c,model_weights))
                    except Exception as e:
                        self.latest_status={"state":"symbol_error","symbol":c.symbol,"error":str(e),"trace":traceback.format_exc(limit=2)}
                try:
                    await self.execute_portfolio_plan(opportunities)
                except Exception as e:
                    self.latest_status={"state":"portfolio_error","error":str(e),"trace":traceback.format_exc(limit=2)}
                self._update_fast_cabinet_view(opportunities)
                equity=self.broker.equity(self.latest_marks); exposure=self.broker.total_exposure_pct(self.latest_marks,equity)
                self.latest_status={
                    "state":"running","version":5,"exchange":self.cfg.exchange_id,"equity":equity,"cash":self.broker.cash,
                    "realized_pnl":self.broker.realized_pnl,"open_positions":len(self.broker.positions),
                    "capital_utilization_pct":exposure*100,"radar_size":len(self.radar),"deep_size":len(self.universe),
                    "fast_cabinet_size":len(self.fast_cabinet),"analyzed_this_cycle":len(candidates),"opportunities_found":len(opportunities),
                    "regime":self.market_regime.name,"cycle_duration_seconds":time.time()-cycle_start,"last_update_ms":int(time.time()*1000),
                }
                self._save_state()
                sleep=max(1.0,self.cfg.cycle_seconds-(time.time()-cycle_start))
                await asyncio.sleep(sleep)
        finally:
            self._save_state()
            await self.gateway.close()

    async def shutdown(self):
        self.running=False
        for t in self.model_tasks.values():
            if not t.done(): t.cancel()

    def report(self) -> dict[str,Any]:
        equity=self.broker.equity(self.latest_marks); open_positions=self.broker.open_position_report(self.latest_marks)
        unrealized=sum(float(p["unrealized_pnl"]) for p in open_positions)
        exposure=self.broker.total_exposure_pct(self.latest_marks,equity)*100 if equity else 0.0
        allocation=[{"label":"Cash","symbol":"CASH","value":self.broker.cash}]+[
            {"label":p["symbol"].split("/")[0],"symbol":p["symbol"],"value":p["market_value"]} for p in open_positions]
        perf=self.storage.performance_summary()
        return {
            "version":5,"mode":"PAPER_ONLY","paper_run_id":self.cfg.paper_run_id,"currency":"USD-equivalent / USDT",
            "starting_cash":self.broker.starting_cash,"equity":equity,"cash":self.broker.cash,"realized_pnl":perf["realized_pnl"],
            "unrealized_pnl":unrealized,"total_pnl":equity-self.broker.starting_cash,
            "total_return_pct":((equity/self.broker.starting_cash)-1)*100 if self.broker.starting_cash else 0.0,
            "capital_utilization_pct":exposure,"max_capital_utilization_pct":self.cfg.max_total_exposure_pct*100,
            "available_exposure_pct":max(0.0,self.cfg.max_total_exposure_pct*100-exposure),
            "open_position_count":len(open_positions),"open_positions":open_positions,"closed_trades":self.storage.closed_trades(300),
            "performance":perf,"by_brain":perf.get("by_brain",{}),"by_regime":perf.get("by_regime",{}),"allocation":allocation,"top_opportunities":self.top_opportunities,
            "fast_cabinet":self.fast_cabinet_view,"regime":self.market_regime.to_dict(),"model_weights":self.model_competition.weights(),
            "score_calibration":self.storage.score_calibration(),"shadow_analysis":self.storage.shadow_summary(),
            "radar_count":len(self.radar),"deep_analysis_count":len(self.universe),
            "cooldowns":{k:v for k,v in self.cooldowns.items() if v>int(time.time()*1000)},
            "last_update_ms":self.latest_status.get("last_update_ms"),
        }

    def status(self) -> dict[str,Any]:
        r=self.report()
        return {**self.latest_status,"version":5,"equity":r["equity"],"cash":r["cash"],"positions":r["open_positions"],
                "regime":r["regime"],"top_opportunities":r["top_opportunities"],"fast_cabinet":r["fast_cabinet"]}

    def chat(self, message: str) -> dict[str,Any]:
        return self.copilot.answer(message)
