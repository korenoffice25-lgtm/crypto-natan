from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from config import SETTINGS
from decision_agent import DecisionAgent, Action
from exchange_gateway import ExchangeGateway
from features_v2 import add_market_features
from market_state import orderbook_state, trade_flow_state, MarketSnapshot
from memory_engine import TradeMemory
from microstructure_model import (
    MicrostructureReturnModel, add_time_targets, engineer_micro_features, load_snapshots,
)
from opportunity_engine import Opportunity, trader_opportunity, hunter_opportunity
from paper_exchange import PaperExchange
from regime_model import RegimeModel
from return_model import MultiHorizonReturnModel
from risk_governor import RiskGovernor, RiskState
from state_store import StateStore
from storage import Storage
from universe_scanner import UniverseScanner, Candidate


class CloudPaperAgent:
    def __init__(self, settings=SETTINGS):
        self.cfg = settings
        self.gateway = ExchangeGateway(settings.exchange_id)
        self.scanner = UniverseScanner(self.gateway, settings)
        self.storage = Storage(settings.db_path)
        self.state_store = StateStore(settings.state_path)
        legacy = settings.legacy_v3_db_path if settings.learn_from_v3 else None
        self.memory = TradeMemory(settings.db_path, legacy, settings.min_memory_trades)

        saved = self.state_store.load() or {}
        broker_state = saved.get("broker")
        self.broker = (
            PaperExchange.from_state(broker_state, settings.fee_rate, settings.slippage_bps)
            if broker_state else PaperExchange(settings.starting_cash, settings.fee_rate, settings.slippage_bps)
        )

        self.decision_agent = DecisionAgent(
            settings.round_trip_cost_buffer_bps,
            settings.min_confidence,
            settings.exit_confidence,
        )
        self.risk = RiskGovernor(
            settings.risk_per_trade,
            settings.max_position_pct,
            settings.max_total_exposure_pct,
            settings.min_position_pct,
            settings.max_daily_loss_pct,
            settings.max_drawdown_pct,
        )

        self.return_models: dict[str, MultiHorizonReturnModel] = {}
        self.regime_models: dict[str, RegimeModel] = {}
        self.model_trained_at: dict[str, float] = {}
        self.micro_models: dict[str, MicrostructureReturnModel] = {}
        self.micro_trained_at: dict[str, float] = {}
        self.micro_training: set[str] = set()
        self.universe: list[Candidate] = []
        self.latest_marks: dict[str, float] = dict(saved.get("latest_marks", {}))
        self.latest_status: dict[str, Any] = {"state": "starting"}
        self.last_scan_at = 0.0
        risk_saved = saved.get("risk", {})
        self.current_day = risk_saved.get("current_day")
        self.day_start_equity = float(risk_saved.get("day_start_equity", self.broker.starting_cash))
        self.peak_equity = float(risk_saved.get("peak_equity", self.broker.starting_cash))
        self.cooldowns: dict[str, int] = {k: int(v) for k, v in saved.get("cooldowns", {}).items()}
        self.top_opportunities: list[dict[str, Any]] = []
        self.latest_return_vectors: dict[str, list[float]] = {}
        self.running = True

    def _save_state(self):
        self.state_store.save({
            "version": 4,
            "broker": self.broker.snapshot(),
            "risk": {
                "current_day": self.current_day,
                "day_start_equity": self.day_start_equity,
                "peak_equity": self.peak_equity,
            },
            "latest_marks": self.latest_marks,
            "cooldowns": self.cooldowns,
        })

    async def _train_models(self, symbol: str):
        raw = await self.gateway.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.history_limit)
        feat = add_market_features(raw, self.cfg.horizons, include_targets=True)
        if len(feat) < self.cfg.min_training_rows:
            raise RuntimeError(f"Not enough training rows for {symbol}: {len(feat)}")

        def fit_models():
            return (
                MultiHorizonReturnModel(self.cfg.horizons).fit(feat),
                RegimeModel().fit(feat),
            )

        ret_model, regime_model = await asyncio.to_thread(fit_models)
        self.return_models[symbol] = ret_model
        self.regime_models[symbol] = regime_model
        self.model_trained_at[symbol] = time.time()

    async def _ensure_models(self, symbol: str):
        age = time.time() - self.model_trained_at.get(symbol, 0.0)
        if symbol not in self.return_models or age >= self.cfg.model_retrain_seconds:
            await self._train_models(symbol)

    async def _train_micro_model_if_ready(self, symbol: str):
        if symbol in self.micro_training:
            return
        count = self.storage.market_snapshot_count(symbol)
        if count < 1000:
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
            # Micro model is additive; V4 continues safely without it.
            pass
        finally:
            self.micro_training.discard(symbol)

    async def refresh_universe(self):
        universe = await self.scanner.scan()
        symbols = {c.symbol for c in universe}
        # Open positions are always monitored even if they drop out of the active scan.
        for symbol in self.broker.positions:
            if symbol not in symbols:
                universe.append(Candidate(
                    symbol=symbol, base=symbol.split("/")[0], quote=symbol.split("/")[-1],
                    quote_volume=0.0, intraday_range_pct=0.0, spread_bps=0.0,
                    depth_notional=0.0, opportunity_score=-999.0, risk_multiplier=0.20,
                ))
        self.universe = universe
        self.last_scan_at = time.time()
        self.storage.add_universe(int(time.time() * 1000), [c.to_dict() for c in universe])

    async def _market_packet(self, symbol: str):
        return await asyncio.gather(
            self.gateway.fetch_order_book(symbol, self.cfg.orderbook_levels),
            self.gateway.fetch_trades(symbol, self.cfg.trade_limit),
            self.gateway.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.live_feature_candles),
        )

    def _on_cooldown(self, symbol: str, now_ms: int) -> bool:
        until = int(self.cooldowns.get(symbol, 0))
        if until <= now_ms:
            self.cooldowns.pop(symbol, None)
            return False
        return True

    def _set_cooldown(self, symbol: str, now_ms: int):
        self.cooldowns[symbol] = now_ms + self.cfg.cooldown_seconds * 1000

    @staticmethod
    def _corr(a: list[float], b: list[float]) -> float:
        if len(a) < 12 or len(b) < 12:
            return 0.0
        n = min(len(a), len(b))
        x, y = np.asarray(a[-n:], dtype=float), np.asarray(b[-n:], dtype=float)
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return c if np.isfinite(c) else 0.0

    async def analyze_candidate(self, candidate: Candidate) -> list[Opportunity]:
        symbol = candidate.symbol
        await self._ensure_models(symbol)
        orderbook, trades, candles = await self._market_packet(symbol)
        ob = orderbook_state(orderbook, self.cfg.orderbook_levels)
        tf = trade_flow_state(trades)
        feat = add_market_features(candles, self.cfg.horizons, include_targets=False)
        if feat.empty:
            return []
        row = feat.iloc[[-1]]
        now_ms = int(time.time() * 1000)
        last = ob.mid
        self.latest_marks[symbol] = last

        raw_returns = candles["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return_vector = raw_returns.tail(self.cfg.correlation_lookback).to_numpy(dtype=float)
        self.latest_return_vectors[symbol] = [float(x) for x in return_vector.tolist()]
        short_return = float(raw_returns.tail(3).sum()) if len(raw_returns) else 0.0
        realized_vol = float(raw_returns.tail(12).std()) if len(raw_returns) >= 2 else 0.0
        snap = MarketSnapshot(
            timestamp_ms=now_ms, symbol=symbol, last=last, orderbook=ob, trades=tf,
            short_return=short_return, realized_vol=realized_vol,
        )
        self.storage.add_market_snapshot(now_ms, symbol, snap.flatten())
        asyncio.create_task(self._train_micro_model_if_ready(symbol))

        # Position management first: trailing/hard stops always take priority.
        if symbol in self.broker.positions:
            pos = self.broker.positions[symbol]
            engine = str((pos.meta or {}).get("engine", "TRADER")).upper()
            vol = max(float(row["realized_vol_12"].iloc[0]), 0.001)
            trail_mult = self.cfg.hunter_trailing_vol_mult if engine == "HUNTER" else self.cfg.trader_trailing_vol_mult
            trail_distance = max(0.006, min(0.05, vol * trail_mult))
            self.broker.update_trailing(symbol, last, self.cfg.trailing_activation_pct, trail_distance)
            hit, reason = self.broker.stop_hit(symbol, ob.best_bid)
            if hit:
                fill = self.broker.exit(symbol, ob.best_bid, max(0.0, ob.spread_bps * 0.25), reason)
                if fill.get("ok"):
                    self.storage.add_fill(now_ms, symbol, fill)
                    self._set_cooldown(symbol, now_ms)
                    self._save_state()
                return []

        preds = self.return_models[symbol].predict(row)
        regime = self.regime_models[symbol].read(row)
        has_position = symbol in self.broker.positions
        decision = self.decision_agent.decide(
            predictions=preds, regime=regime, spread_bps=ob.spread_bps,
            orderbook_imbalance=ob.imbalance_20, trade_flow_imbalance=tf.flow_imbalance,
            has_position=has_position,
        )
        self.storage.add_decision(
            now_ms, symbol, decision.action.value, decision.confidence,
            decision.expected_edge_bps, candidate.opportunity_score,
            {**asdict(decision), "regime": asdict(regime)},
        )

        # For open positions, only manage exits; do not create a second opportunity.
        if has_position:
            pos = self.broker.positions[symbol]
            engine = str((pos.meta or {}).get("engine", "TRADER")).upper()
            should_exit = decision.action == Action.EXIT
            # Hunter gets more room to ride; the trailing stop is its main profit capture.
            if engine == "HUNTER" and should_exit:
                should_exit = decision.expected_edge_bps < -5.0 or decision.confidence < 0.38
            if should_exit:
                fill = self.broker.exit(
                    symbol, ob.best_bid, max(0.0, ob.spread_bps * 0.15), reason="AI_EXIT",
                )
                if fill.get("ok"):
                    self.storage.add_fill(now_ms, symbol, fill)
                    self._set_cooldown(symbol, now_ms)
                    self._save_state()
            return []

        if self._on_cooldown(symbol, now_ms):
            return []

        repeat_penalty = self.memory.repeat_penalty(
            symbol, self.cfg.repeat_window_seconds, self.cfg.repeat_penalty_per_trade,
        )
        opportunities: list[Opportunity] = []

        trader_setup = f"trader:r{regime.cluster}:vol{int(regime.volatility_rank*3)}"
        trader_memory = self.memory.memory_multiplier(symbol, "TRADER", trader_setup)
        trader = trader_opportunity(
            candidate=candidate, decision=decision, row=row, best_bid=ob.best_bid, best_ask=ob.best_ask,
            spread_bps=ob.spread_bps, orderbook_imbalance=ob.imbalance_20,
            trade_flow_imbalance=tf.flow_imbalance, regime=regime,
            memory_multiplier=trader_memory, repeat_penalty=repeat_penalty,
            returns=return_vector,
        )
        if trader and trader.score >= self.cfg.trader_min_score:
            opportunities.append(trader)

        micro_edge_bps = 0.0
        micro_model = self.micro_models.get(symbol)
        if micro_model is not None:
            try:
                micro_df = pd.DataFrame([snap.flatten()])
                engineered = engineer_micro_features(micro_df)
                mp = micro_model.predict(engineered)
                weights = np.array([1 / max(p.validation_mae, 1e-7) for p in mp], dtype=float)
                vals = np.array([p.expected_return for p in mp], dtype=float)
                micro_edge_bps = float(np.average(vals, weights=weights) * 10_000)
            except Exception:
                micro_edge_bps = 0.0

        # setup key is reproduced here so memory can weight Hunter before Opportunity creation.
        vol_z = float(row["volume_z_24"].iloc[0])
        momentum = 0.65 * float(row["ret_3"].iloc[0]) + 0.35 * max(float(row["ret_12"].iloc[0]), 0.0)
        hunter_setup = f"hunter:r{regime.cluster}:v{int(min(max(vol_z,0),5))}:m{int(min(max(momentum*100,0),5))}"
        hunter_memory = self.memory.memory_multiplier(symbol, "HUNTER", hunter_setup)
        hunter = hunter_opportunity(
            candidate=candidate, row=row, best_bid=ob.best_bid, best_ask=ob.best_ask,
            spread_bps=ob.spread_bps, orderbook_imbalance=ob.imbalance_20,
            microprice_edge_bps=ob.microprice_edge_bps, trade_flow_imbalance=tf.flow_imbalance,
            regime=regime, memory_multiplier=hunter_memory, repeat_penalty=repeat_penalty,
            returns=return_vector, min_volume_z=self.cfg.hunter_min_volume_z,
            min_momentum_pct=self.cfg.hunter_min_momentum_pct, micro_model_edge_bps=micro_edge_bps,
        )
        if hunter and hunter.score >= self.cfg.hunter_min_score:
            opportunities.append(hunter)

        # One symbol can qualify for both brains; global ranking keeps only its stronger thesis.
        if len(opportunities) > 1:
            opportunities = [max(opportunities, key=lambda x: x.score)]
        return opportunities

    def _risk_state(self, symbol: str, mark: float) -> RiskState:
        equity = self.broker.equity(self.latest_marks)
        now_ms = int(time.time() * 1000)
        day = now_ms // 86_400_000
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        return RiskState(
            equity=equity, cash=self.broker.cash,
            current_exposure_pct=self.broker.exposure_pct(symbol, mark, equity),
            total_exposure_pct=self.broker.total_exposure_pct(self.latest_marks, equity),
            day_start_equity=self.day_start_equity, peak_equity=self.peak_equity,
            open_positions=len(self.broker.positions),
        )

    def _correlation_ok(self, opp: Opportunity, accepted: list[Opportunity]) -> tuple[bool, str]:
        # Reject only extreme same-direction redundancy. No hard asset-count cap.
        for other in accepted:
            c = self._corr(opp.return_vector, other.return_vector)
            if c >= self.cfg.max_pair_correlation:
                return False, f"Highly correlated with {other.symbol} ({c:.2f})"
        for symbol in self.broker.positions:
            if symbol == opp.symbol:
                continue
            vec = self.latest_return_vectors.get(symbol)
            if not vec:
                continue
            c = self._corr(opp.return_vector, vec)
            if c >= self.cfg.max_pair_correlation:
                return False, f"Highly correlated with open {symbol} ({c:.2f})"
        return True, ""

    async def execute_ranked(self, opportunities: list[Opportunity]):
        ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        accepted: list[Opportunity] = []
        self.top_opportunities = [x.to_dict() for x in ranked[:12]]
        now_ms = int(time.time() * 1000)

        for opp in ranked:
            if opp.symbol in self.broker.positions:
                continue
            corr_ok, corr_reason = self._correlation_ok(opp, accepted)
            if not corr_ok:
                self.storage.add_opportunity(now_ms, opp.to_dict(), False, corr_reason)
                continue

            state = self._risk_state(opp.symbol, opp.best_ask)
            approval = self.risk.approve_buy(
                target_exposure_pct=opp.target_exposure_pct,
                state=state, price=opp.best_ask, volatility=opp.volatility,
                market_risk_multiplier=opp.candidate_risk_multiplier,
            )
            if not approval.allowed:
                self.storage.add_opportunity(now_ms, opp.to_dict(), False, approval.reason)
                # Once total portfolio capacity is gone, lower-ranked trades cannot fit either.
                if state.total_exposure_pct >= self.cfg.max_total_exposure_pct - self.cfg.min_position_pct:
                    break
                continue

            stop_mult = self.cfg.hunter_stop_vol_mult if opp.engine == "HUNTER" else self.cfg.trader_stop_vol_mult
            hard_stop_distance = min(0.07, max(0.010, opp.volatility * stop_mult))
            stop_price = opp.best_ask * (1 - hard_stop_distance)
            meta = {
                "engine": opp.engine, "score": opp.score, "confidence": opp.confidence,
                "setup_key": opp.setup_key, "expected_edge_bps": opp.expected_edge_bps,
                "memory_multiplier": opp.memory_multiplier, "repeat_penalty": opp.repeat_penalty,
                "entry_reason": opp.reason, "context": opp.context,
            }
            fill = self.broker.buy(
                opp.symbol, approval.quantity, opp.best_ask, stop_price,
                extra_slippage_bps=max(0.0, opp.spread_bps * 0.15), meta=meta,
            )
            if fill.get("ok"):
                self.storage.add_fill(now_ms, opp.symbol, fill)
                self.storage.add_opportunity(now_ms, opp.to_dict(), True, approval.reason)
                accepted.append(opp)
                self._save_state()
            else:
                self.storage.add_opportunity(now_ms, opp.to_dict(), False, fill.get("reason", "buy_failed"))

    async def run_forever(self):
        try:
            await self.gateway.load_markets()
            while self.running:
                if not self.universe or time.time() - self.last_scan_at >= self.cfg.scanner_refresh_seconds:
                    try:
                        await self.refresh_universe()
                    except Exception as e:
                        self.latest_status = {"state": "scanner_error", "error": str(e)}

                opportunities: list[Opportunity] = []
                for candidate in list(self.universe):
                    if not self.running:
                        break
                    try:
                        opportunities.extend(await self.analyze_candidate(candidate))
                    except Exception as e:
                        self.latest_status = {
                            "state": "symbol_error", "symbol": candidate.symbol,
                            "error": str(e), "trace": traceback.format_exc(limit=2),
                        }

                try:
                    await self.execute_ranked(opportunities)
                except Exception as e:
                    self.latest_status = {"state": "allocation_error", "error": str(e)}

                equity = self.broker.equity(self.latest_marks)
                exposure = self.broker.total_exposure_pct(self.latest_marks, equity)
                self.latest_status = {
                    "state": "running", "exchange": self.cfg.exchange_id,
                    "equity": equity, "cash": self.broker.cash,
                    "realized_pnl": self.broker.realized_pnl,
                    "open_positions": len(self.broker.positions),
                    "capital_utilization_pct": exposure * 100.0,
                    "universe_size": len(self.universe),
                    "opportunities_found": len(opportunities),
                    "last_update_ms": int(time.time() * 1000),
                }
                self._save_state()
                await asyncio.sleep(self.cfg.cycle_seconds)
        finally:
            self._save_state()
            await self.gateway.close()

    async def shutdown(self):
        self.running = False

    def report(self) -> dict[str, Any]:
        equity = self.broker.equity(self.latest_marks)
        open_positions = self.broker.open_position_report(self.latest_marks)
        unrealized_pnl = sum(float(p["unrealized_pnl"]) for p in open_positions)
        exposure_pct = self.broker.total_exposure_pct(self.latest_marks, equity) * 100 if equity else 0.0
        allocation = [{"label": "Cash", "symbol": "CASH", "value": self.broker.cash}]
        for p in open_positions:
            allocation.append({"label": p["symbol"].split("/")[0], "symbol": p["symbol"], "value": p["market_value"]})
        perf = self.storage.performance_summary()
        return {
            "version": 4, "mode": "PAPER_ONLY", "currency": "USD-equivalent / USDT",
            "starting_cash": self.broker.starting_cash, "equity": equity, "cash": self.broker.cash,
            "realized_pnl": perf["realized_pnl"], "unrealized_pnl": unrealized_pnl,
            "total_pnl": equity - self.broker.starting_cash,
            "total_return_pct": ((equity / self.broker.starting_cash) - 1) * 100 if self.broker.starting_cash else 0.0,
            "capital_utilization_pct": exposure_pct,
            "max_capital_utilization_pct": self.cfg.max_total_exposure_pct * 100,
            "open_position_count": len(open_positions), "open_positions": open_positions,
            "closed_trades": self.storage.closed_trades(200), "performance": perf,
            "allocation": allocation, "top_opportunities": self.top_opportunities,
            "cooldowns": {k: v for k, v in self.cooldowns.items() if v > int(time.time() * 1000)},
            "last_update_ms": self.latest_status.get("last_update_ms"),
        }

    def status(self) -> dict[str, Any]:
        equity = self.broker.equity(self.latest_marks)
        return {
            **self.latest_status, "version": 4, "equity": equity, "cash": self.broker.cash,
            "realized_pnl": self.broker.realized_pnl,
            "total_return_pct": ((equity / self.broker.starting_cash) - 1) * 100 if self.broker.starting_cash else 0.0,
            "positions": self.broker.open_position_report(self.latest_marks),
            "universe": [c.to_dict() for c in self.universe],
            "top_opportunities": self.top_opportunities,
        }
