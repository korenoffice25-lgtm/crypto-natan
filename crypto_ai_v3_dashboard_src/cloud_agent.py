from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import time
import traceback

from config import SETTINGS
from decision_agent import DecisionAgent, Action
from exchange_gateway import ExchangeGateway
from features_v2 import add_market_features
from market_state import orderbook_state, trade_flow_state
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

        saved = self.state_store.load()
        broker_state = None
        if saved:
            broker_state = saved.get("broker", saved)
        if broker_state:
            self.broker = PaperExchange.from_state(broker_state, settings.fee_rate, settings.slippage_bps)
        else:
            self.broker = PaperExchange(settings.starting_cash, settings.fee_rate, settings.slippage_bps)

        self.decision_agent = DecisionAgent(
            settings.round_trip_cost_buffer_bps,
            settings.min_confidence,
            settings.exit_confidence,
        )
        self.risk = RiskGovernor(
            settings.risk_per_trade,
            settings.max_position_pct,
            settings.max_total_exposure_pct,
            settings.max_daily_loss_pct,
            settings.max_drawdown_pct,
            settings.max_positions,
        )

        self.return_models: dict[str, MultiHorizonReturnModel] = {}
        self.regime_models: dict[str, RegimeModel] = {}
        self.model_trained_at: dict[str, float] = {}
        self.universe: list[Candidate] = []
        risk_saved = (saved or {}).get("risk", {}) if saved else {}
        self.latest_marks: dict[str, float] = dict((saved or {}).get("latest_marks", {})) if saved else {}
        self.latest_status: dict = {"state": "starting"}
        self.last_scan_at = 0.0
        self.current_day = risk_saved.get("current_day")
        self.day_start_equity = float(risk_saved.get("day_start_equity", self.broker.starting_cash))
        self.peak_equity = float(risk_saved.get("peak_equity", self.broker.starting_cash))
        self.running = True

    def _save_state(self):
        self.state_store.save({
            "version": 3,
            "broker": self.broker.snapshot(),
            "risk": {
                "current_day": self.current_day,
                "day_start_equity": self.day_start_equity,
                "peak_equity": self.peak_equity,
            },
            "latest_marks": self.latest_marks,
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

    async def refresh_universe(self):
        universe = await self.scanner.scan()

        # Never drop an already-open paper position from observation.
        symbols = {c.symbol for c in universe}
        for symbol in self.broker.positions:
            if symbol not in symbols:
                universe.append(Candidate(
                    symbol=symbol, base=symbol.split("/")[0], quote=symbol.split("/")[-1],
                    quote_volume=0.0, intraday_range_pct=0.0, spread_bps=0.0,
                    depth_notional=0.0, opportunity_score=-999.0, risk_multiplier=0.20,
                ))

        self.universe = universe
        self.last_scan_at = time.time()
        now_ms = int(time.time() * 1000)
        self.storage.add_universe(now_ms, [c.to_dict() for c in universe])

    async def _market_packet(self, symbol: str):
        book_task = self.gateway.fetch_order_book(symbol, self.cfg.orderbook_levels)
        trades_task = self.gateway.fetch_trades(symbol, self.cfg.trade_limit)
        candles_task = self.gateway.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.live_feature_candles)
        return await asyncio.gather(book_task, trades_task, candles_task)

    async def evaluate_candidate(self, candidate: Candidate):
        symbol = candidate.symbol
        await self._ensure_models(symbol)
        orderbook, trades, candles = await self._market_packet(symbol)

        ob = orderbook_state(orderbook, self.cfg.orderbook_levels)
        tf = trade_flow_state(trades)
        feat = add_market_features(candles, self.cfg.horizons, include_targets=False)
        if len(feat) == 0:
            return
        row = feat.iloc[[-1]]

        last = ob.mid
        self.latest_marks[symbol] = last

        # Emergency hard stop is outside the AI and always wins.
        if self.broker.emergency_stop_hit(symbol, ob.best_bid):
            fill = self.broker.exit(symbol, ob.best_bid, extra_slippage_bps=max(0.0, ob.spread_bps * 0.25), reason="HARD_STOP")
            if fill.get("ok"):
                self.storage.add_fill(int(time.time() * 1000), symbol, fill)
                self._save_state()
            return

        preds = self.return_models[symbol].predict(row)
        regime = self.regime_models[symbol].read(row)
        has_position = symbol in self.broker.positions
        decision = self.decision_agent.decide(
            predictions=preds,
            regime=regime,
            spread_bps=ob.spread_bps,
            orderbook_imbalance=ob.imbalance_20,
            trade_flow_imbalance=tf.flow_imbalance,
            has_position=has_position,
        )

        # The universe scanner controls size, never direction.
        if decision.action == Action.BUY:
            decision = replace(
                decision,
                target_exposure_pct=decision.target_exposure_pct * candidate.risk_multiplier,
            )

        now_ms = int(time.time() * 1000)
        self.storage.add_decision(
            now_ms, symbol, decision.action.value, decision.confidence,
            decision.expected_edge_bps, candidate.opportunity_score, asdict(decision),
        )

        equity = self.broker.equity(self.latest_marks)
        day = now_ms // 86_400_000
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        state = RiskState(
            equity=equity,
            cash=self.broker.cash,
            current_exposure_pct=self.broker.exposure_pct(symbol, last, equity),
            total_exposure_pct=self.broker.total_exposure_pct(self.latest_marks, equity),
            day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity,
            open_positions=len(self.broker.positions),
        )

        volatility = max(float(row["realized_vol_12"].iloc[0]), 0.001)
        approval = self.risk.approve(
            decision=decision,
            state=state,
            price=last,
            volatility=volatility,
            market_risk_multiplier=candidate.risk_multiplier,
        )

        fill = None
        if approval.allowed and approval.action == Action.BUY:
            hard_stop_distance = min(0.08, max(0.012, volatility * 3.0))
            stop_price = ob.best_ask * (1 - hard_stop_distance)
            fill = self.broker.buy(
                symbol,
                approval.quantity,
                ob.best_ask,
                stop_price=stop_price,
                extra_slippage_bps=max(0.0, ob.spread_bps * 0.15),
            )
        elif approval.allowed and approval.action == Action.EXIT:
            fill = self.broker.exit(
                symbol,
                ob.best_bid,
                extra_slippage_bps=max(0.0, ob.spread_bps * 0.15),
                reason="AI_EXIT",
            )

        if fill and fill.get("ok"):
            self.storage.add_fill(now_ms, symbol, fill)
            self._save_state()

        self.latest_status = {
            "state": "running",
            "exchange": self.cfg.exchange_id,
            "symbol": symbol,
            "last_action": decision.action.value,
            "last_confidence": decision.confidence,
            "last_edge_bps": decision.expected_edge_bps,
            "equity": self.broker.equity(self.latest_marks),
            "cash": self.broker.cash,
            "realized_pnl": self.broker.realized_pnl,
            "open_positions": len(self.broker.positions),
            "universe_size": len(self.universe),
            "last_update_ms": now_ms,
        }

    async def run_forever(self):
        try:
            await self.gateway.load_markets()
            while self.running:
                if not self.universe or time.time() - self.last_scan_at >= self.cfg.scanner_refresh_seconds:
                    try:
                        await self.refresh_universe()
                    except Exception as e:
                        self.latest_status = {"state": "scanner_error", "error": str(e)}

                for candidate in list(self.universe):
                    if not self.running:
                        break
                    try:
                        await self.evaluate_candidate(candidate)
                    except Exception as e:
                        self.latest_status = {
                            "state": "symbol_error",
                            "symbol": candidate.symbol,
                            "error": str(e),
                            "trace": traceback.format_exc(limit=2),
                        }

                self._save_state()
                await asyncio.sleep(self.cfg.cycle_seconds)
        finally:
            self._save_state()
            await self.gateway.close()

    async def shutdown(self):
        self.running = False

    def report(self) -> dict:
        equity = self.broker.equity(self.latest_marks)
        open_positions = self.broker.open_position_report(self.latest_marks)
        unrealized_pnl = sum(float(p["unrealized_pnl"]) for p in open_positions)
        allocation = [{"label": "Cash", "symbol": "CASH", "value": self.broker.cash}]
        for p in open_positions:
            allocation.append({
                "label": p["symbol"].split("/")[0],
                "symbol": p["symbol"],
                "value": p["market_value"],
            })

        perf = self.storage.performance_summary()
        return {
            "mode": "PAPER_ONLY",
            "currency": "USD-equivalent / USDT",
            "starting_cash": self.broker.starting_cash,
            "equity": equity,
            "cash": self.broker.cash,
            "realized_pnl": perf["realized_pnl"],
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": equity - self.broker.starting_cash,
            "total_return_pct": ((equity / self.broker.starting_cash) - 1) * 100 if self.broker.starting_cash else 0.0,
            "open_positions": open_positions,
            "closed_trades": self.storage.closed_trades(200),
            "performance": perf,
            "allocation": allocation,
            "last_update_ms": self.latest_status.get("last_update_ms"),
        }

    def status(self) -> dict:
        equity = self.broker.equity(self.latest_marks)
        return {
            **self.latest_status,
            "equity": equity,
            "cash": self.broker.cash,
            "realized_pnl": self.broker.realized_pnl,
            "total_return_pct": ((equity / self.broker.starting_cash) - 1) * 100 if self.broker.starting_cash else 0.0,
            "positions": self.broker.open_position_report(self.latest_marks),
            "universe": [c.to_dict() for c in self.universe],
        }
