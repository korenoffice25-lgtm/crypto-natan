from __future__ import annotations

import asyncio
from dataclasses import asdict

import pandas as pd

from config import SETTINGS
from decision_agent import DecisionAgent, Action
from features_v2 import add_market_features
from historical import fetch_history
from live_stream import LiveMarketStream
from paper_exchange import PaperExchange
from regime_model import RegimeModel
from return_model import MultiHorizonReturnModel
from risk_governor import RiskGovernor, RiskState
from storage import Storage


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class PaperTradingAgent:
    def __init__(self):
        cfg = SETTINGS
        self.cfg = cfg
        self.storage = Storage(cfg.db_path)
        self.broker = PaperExchange(cfg.starting_cash, cfg.fee_rate, cfg.slippage_bps)
        self.decision_agent = DecisionAgent(
            cfg.round_trip_cost_buffer_bps,
            cfg.min_confidence,
            cfg.exit_confidence,
        )
        self.risk = RiskGovernor(
            cfg.risk_per_trade,
            cfg.max_position_pct,
            cfg.max_total_exposure_pct,
            cfg.max_daily_loss_pct,
            cfg.max_drawdown_pct,
        )
        self.models = {}
        self.regimes = {}
        self.raw_history: dict[str, pd.DataFrame] = {}
        self.last_rows = {}
        self.last_candle_ts = {}
        self.day_start_equity = cfg.starting_cash
        self.peak_equity = cfg.starting_cash
        self.current_day = None
        self.latest_marks: dict[str, float] = {}

    def train_symbol(self, symbol: str):
        raw = fetch_history(
            self.cfg.exchange_id,
            symbol,
            self.cfg.timeframe,
            self.cfg.history_limit,
        )
        train_feat = add_market_features(
            raw, self.cfg.horizons, include_targets=True
        )
        if len(train_feat) < self.cfg.min_training_rows:
            raise RuntimeError(f"Not enough history for {symbol}: {len(train_feat)} rows")

        model = MultiHorizonReturnModel(self.cfg.horizons).fit(train_feat)
        regime = RegimeModel().fit(train_feat)

        inference_feat = add_market_features(
            raw, self.cfg.horizons, include_targets=False
        )
        self.models[symbol] = model
        self.regimes[symbol] = regime
        self.raw_history[symbol] = raw.copy()
        self.last_rows[symbol] = inference_feat.iloc[[-1]].copy()
        self.last_candle_ts[symbol] = int(raw["timestamp"].iloc[-1].timestamp() * 1000)

    def update_live_candles(self, symbol: str, ohlcv: list[list]):
        if not ohlcv:
            return

        live = pd.DataFrame(ohlcv, columns=OHLCV_COLUMNS)
        live["timestamp"] = pd.to_datetime(live["timestamp"], unit="ms", utc=True)

        history = pd.concat([self.raw_history[symbol], live], ignore_index=True)
        history = (
            history.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(self.cfg.history_limit)
            .reset_index(drop=True)
        )
        self.raw_history[symbol] = history

        inference_feat = add_market_features(
            history, self.cfg.horizons, include_targets=False
        )
        if len(inference_feat):
            self.last_rows[symbol] = inference_feat.iloc[[-1]].copy()
            self.last_candle_ts[symbol] = int(
                inference_feat["timestamp"].iloc[-1].timestamp() * 1000
            )

    async def run_symbol(self, symbol: str):
        self.train_symbol(symbol)
        stream = LiveMarketStream(
            self.cfg.exchange_id,
            symbol,
            timeframe=self.cfg.timeframe,
            levels=self.cfg.orderbook_levels,
        )

        try:
            while True:
                update = await stream.next()
                snap = update.snapshot
                self.update_live_candles(symbol, update.raw_ohlcv)

                flat = snap.flatten()
                flat["model_candle_timestamp_ms"] = self.last_candle_ts[symbol]
                self.storage.add_snapshot(snap.timestamp_ms, symbol, flat)

                row = self.last_rows[symbol]
                predictions = self.models[symbol].predict(row)
                regime = self.regimes[symbol].read(row)

                has_position = symbol in self.broker.positions
                decision = self.decision_agent.decide(
                    predictions=predictions,
                    regime=regime,
                    spread_bps=snap.orderbook.spread_bps,
                    orderbook_imbalance=snap.orderbook.imbalance_20,
                    trade_flow_imbalance=snap.trades.flow_imbalance,
                    has_position=has_position,
                )

                self.storage.add_decision(
                    snap.timestamp_ms,
                    symbol,
                    decision.action.value,
                    decision.confidence,
                    decision.expected_edge_bps,
                    asdict(decision),
                )

                self.latest_marks[symbol] = snap.last
                equity = self.broker.equity(self.latest_marks)
                day = snap.timestamp_ms // 86_400_000
                if self.current_day != day:
                    self.current_day = day
                    self.day_start_equity = equity
                self.peak_equity = max(self.peak_equity, equity)

                risk_state = RiskState(
                    equity=equity,
                    cash=self.broker.cash,
                    current_exposure_pct=self.broker.exposure_pct(
                        symbol, snap.last, equity
                    ),
                    day_start_equity=self.day_start_equity,
                    peak_equity=self.peak_equity,
                )

                approval = self.risk.approve(
                    decision=decision,
                    state=risk_state,
                    price=snap.last,
                    volatility=max(
                        snap.realized_vol,
                        float(row["realized_vol_12"].iloc[0]),
                    ),
                )

                fill = None
                if approval.allowed and approval.action == Action.BUY:
                    fill = self.broker.buy(
                        symbol, approval.quantity, snap.orderbook.best_ask
                    )
                elif approval.allowed and approval.action == Action.EXIT:
                    fill = self.broker.exit(
                        symbol, snap.orderbook.best_bid
                    )

                if fill and fill.get("ok"):
                    self.storage.add_fill(snap.timestamp_ms, symbol, fill)

                print(
                    f"{symbol} | {decision.action.value:<10} "
                    f"conf={decision.confidence:.2f} "
                    f"edge={decision.expected_edge_bps:+.1f}bps "
                    f"spread={snap.orderbook.spread_bps:.2f}bps "
                    f"ob={snap.orderbook.imbalance_20:+.2f} "
                    f"flow={snap.trades.flow_imbalance:+.2f} "
                    f"equity=${equity:,.2f}"
                )

                await asyncio.sleep(self.cfg.snapshot_interval_seconds)
        finally:
            await stream.close()

    async def run(self):
        await asyncio.gather(*(self.run_symbol(s) for s in self.cfg.symbols))


async def main():
    agent = PaperTradingAgent()
    try:
        await agent.run()
    finally:
        agent.storage.close()


if __name__ == "__main__":
    asyncio.run(main())
