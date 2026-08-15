from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path


class TradeMemory:
    """Conservative performance memory.

    V4 does not rewrite its own code. It only applies bounded performance weights to
    symbols/engines/setups after enough closed trades, so one lucky/awful trade cannot
    hijack the strategy.
    """

    def __init__(self, db_path: str, legacy_v3_db_path: str | None = None, min_trades: int = 5):
        self.db_path = db_path
        self.legacy_v3_db_path = legacy_v3_db_path
        self.min_trades = min_trades

    @staticmethod
    def _read_sells(path: str, limit: int = 2000) -> list[dict]:
        if not path or not Path(path).exists():
            return []
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT timestamp_ms, symbol, pnl_net, payload_json FROM paper_fills "
                "WHERE side='sell' AND pnl_net IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()
        out = []
        for ts, symbol, pnl, payload in rows:
            try:
                d = json.loads(payload or "{}")
            except Exception:
                d = {}
            out.append({"timestamp_ms": int(ts), "symbol": symbol, "pnl": float(pnl), "details": d})
        return out

    def trades(self) -> list[dict]:
        out = self._read_sells(self.db_path)
        if self.legacy_v3_db_path and self.legacy_v3_db_path != self.db_path:
            out.extend(self._read_sells(self.legacy_v3_db_path, 1200))
        return out

    def repeat_penalty(self, symbol: str, window_seconds: int, penalty_per_trade: float) -> float:
        cutoff = int((time.time() - window_seconds) * 1000)
        n = sum(1 for t in self.trades() if t["symbol"] == symbol and t["timestamp_ms"] >= cutoff)
        return max(0.55, 1.0 - penalty_per_trade * max(0, n - 1))

    def memory_multiplier(self, symbol: str, engine: str, setup_key: str) -> float:
        trades = self.trades()
        selected = []
        for t in trades:
            d = t["details"]
            meta = d.get("meta") or {}
            # V3 had no engine/setup metadata, but its symbol-level outcomes still count softly.
            same_symbol = t["symbol"] == symbol
            same_engine = str(meta.get("engine", "")).upper() == engine.upper()
            same_setup = meta.get("setup_key") == setup_key
            if same_setup:
                selected.append((t["pnl"], 1.0))
            elif same_engine:
                selected.append((t["pnl"], 0.65))
            elif same_symbol:
                selected.append((t["pnl"], 0.35))

        effective_n = sum(w for _, w in selected)
        if effective_n < self.min_trades:
            return 1.0

        weighted_pnl = sum(p * w for p, w in selected)
        wins = sum(w for p, w in selected if p > 0)
        win_rate = wins / max(effective_n, 1e-9)
        avg_pnl = weighted_pnl / max(effective_n, 1e-9)

        # Bounded: memory can influence ranking but cannot dominate it.
        wr_term = (win_rate - 0.50) * 0.45
        pnl_term = math.tanh(avg_pnl / 8.0) * 0.18
        return max(0.75, min(1.25, 1.0 + wr_term + pnl_term))
