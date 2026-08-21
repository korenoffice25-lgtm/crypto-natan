from __future__ import annotations

import asyncio
import time
from typing import Any

from engine import V6Engine
from domain import Signal
from market_data import MarketDataGateway
from scanner import UniverseScanner


class V7Engine(V6Engine):
    """V7 overlay on V6: resilient runtime + stricter final entry-quality gate.
    PAPER ONLY. No real-order adapter.
    """
    VERSION = 7

    def __init__(self, settings=None):
        super().__init__() if settings is None else super().__init__(settings)
        self.latest_status.update({"version": 7, "state": "starting_v7"})
        self.health_state.update({
            "ok": False, "state": "starting_v7", "engine_version": 7,
            "startup_attempt": 0, "last_engine_error": None,
            "last_engine_restart_ms": None, "restart_count": 0,
        })
        self._engine_restart_count = 0

    def _round_trip_cost_bps(self, signal: Signal) -> float:
        fees = float(self.cfg.fee_rate) * 2.0 * 10_000.0
        slippage = float(self.cfg.slippage_bps) * 2.0
        return fees + slippage + max(0.0, float(signal.spread_bps))

    def _drawdown_pct(self) -> float:
        state = self._risk_state()
        if state.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - state.equity / state.peak_equity)

    def _quality_threshold(self, brain: str) -> float:
        base = {
            "TRADER": float(self.cfg.trader_min_score),
            "HUNTER": float(self.cfg.hunter_min_score),
            "SWING": float(self.cfg.swing_min_score),
            "REVERSAL": float(self.cfg.reversal_min_score),
        }.get(brain.upper(), 70.0)
        dd = self._drawdown_pct()
        if dd >= 0.05:
            base += 8.0
        elif dd >= 0.03:
            base += 5.0
        elif dd >= 0.015:
            base += 2.5
        return min(92.0, base)

    def _reject_v7(self, signal: Signal, reason: str) -> None:
        payload = signal.to_dict()
        payload["v7_gate"] = reason
        try:
            self.storage.add_opportunity(payload, "REJECTED", reason)
            self.storage.add_shadow_event(
                signal.symbol, signal.best_ask, "V7_GATE_REJECTED",
                signal.brain, signal.meta_score, reason, payload,
            )
        except Exception:
            pass
        self.last_rejections.insert(0, {
            "symbol": signal.symbol, "brain": signal.brain,
            "score": signal.meta_score, "reason": f"V7_GATE: {reason}",
        })
        del self.last_rejections[30:]

    def _passes_v7_gate(self, signal: Signal) -> tuple[bool, str]:
        brain = signal.brain.upper()
        score = float(signal.meta_score or signal.score)
        threshold = self._quality_threshold(brain)
        if score < threshold:
            return False, f"score {score:.1f} below V7 threshold {threshold:.1f}"

        confidence_floor = {
            "TRADER": 0.60, "HUNTER": 0.67,
            "SWING": 0.61, "REVERSAL": 0.65,
        }.get(brain, 0.62)
        if float(signal.confidence) < confidence_floor:
            return False, f"confidence {signal.confidence:.2f} below {confidence_floor:.2f}"

        costs = self._round_trip_cost_bps(signal)
        net_edge = float(signal.expected_edge_bps) - costs
        min_net = {
            "TRADER": 7.0, "HUNTER": 14.0,
            "SWING": 20.0, "REVERSAL": 10.0,
        }.get(brain, 10.0)
        if net_edge < min_net:
            return False, f"net edge {net_edge:.1f}bps after {costs:.1f}bps costs is too small"

        brain_mult = float(self.regime.brain_multipliers.get(brain, 1.0))
        if brain_mult < 0.72 and score < float(self.cfg.exceptional_score):
            return False, f"{brain} suppressed in {self.regime.name} regime"

        f = self.last_features.get(signal.symbol)
        if f is not None:
            if float(f.spread_bps) > min(float(self.cfg.max_spread_bps), 28.0):
                return False, f"spread {f.spread_bps:.1f}bps too wide"

            if brain == "HUNTER":
                late = float(f.ret_12) > 0.10 or float(f.ret_3) > 0.045
                weak_tape = float(f.trade_flow_imbalance) < 0.12 or float(f.orderbook_imbalance) < -0.10
                if late and (weak_tape or score < 89.0):
                    return False, "anti-FOMO: move already extended"
                if float(f.volume_z) < 1.0:
                    return False, "Hunter lacks abnormal-volume confirmation"

            elif brain == "REVERSAL":
                if float(f.ret_1) <= 0:
                    return False, "Reversal has no positive first-turn candle"
                if float(f.trade_flow_imbalance) <= 0.05 and float(f.orderbook_imbalance) <= 0.05:
                    return False, "Reversal lacks returning demand"

            elif brain == "SWING":
                if float(f.ema_fast) <= float(f.ema_slow) and float(f.ret_3) < -0.006:
                    return False, "Swing conflicts with live short-term trend"

            elif brain == "TRADER":
                if float(f.trade_flow_imbalance) < -0.35 and float(f.orderbook_imbalance) < -0.25:
                    return False, "Trader conflicts with strongly negative microstructure"

        return True, f"qualified score={score:.1f}, net_edge={net_edge:.1f}bps"

    async def execute_portfolio(self, signals: list[Signal], cycle_id: str):
        qualified = []
        for signal in signals:
            if signal.symbol in self.broker.positions:
                qualified.append(signal)
                continue
            ok, reason = self._passes_v7_gate(signal)
            if ok:
                signal.context = dict(signal.context or {})
                signal.context["v7_gate"] = reason
                qualified.append(signal)
            else:
                self._reject_v7(signal, reason)
        return await super().execute_portfolio(qualified, cycle_id)

    async def run_forever(self):
        attempt = 0
        while self.running:
            try:
                attempt += 1
                self.health_state.update({
                    "ok": False, "state": "connecting", "engine_version": 7,
                    "startup_attempt": attempt, "restart_count": self._engine_restart_count,
                })
                self.latest_status.update({"state": "connecting", "version": 7})
                await super().run_forever()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._engine_restart_count += 1
                now_ms = int(time.time() * 1000)
                self.health_state.update({
                    "ok": False, "state": "recovering", "engine_version": 7,
                    "last_engine_error": str(exc), "last_engine_restart_ms": now_ms,
                    "restart_count": self._engine_restart_count,
                })
                self.latest_status.update({
                    "state": "recovering", "version": 7,
                    "error": str(exc), "last_update_ms": now_ms,
                })
                try:
                    self.storage.add_journal(
                        "ENGINE_RECOVERY",
                        f"V7 restarting after engine failure: {exc}",
                        payload={"attempt": attempt, "restart_count": self._engine_restart_count},
                        now_ms=now_ms,
                    )
                except Exception:
                    pass
                try:
                    await self.gateway.close()
                except Exception:
                    pass
                if not self.running:
                    break
                await asyncio.sleep(min(60.0, 2.0 ** min(attempt, 5)))
                self.gateway = MarketDataGateway(self.cfg.exchange_id)
                self.scanner = UniverseScanner(self.gateway, self.cfg)

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["version"] = 7
        report["engine"] = "V7_RESILIENT_EDGE"
        report["objective"] = "maximize risk-adjusted paper returns; cash beats weak edge"
        report["health"] = self.health_state
        return report

    def status(self) -> dict[str, Any]:
        status = super().status()
        status["version"] = 7
        status["engine"] = "V7_RESILIENT_EDGE"
        return status
