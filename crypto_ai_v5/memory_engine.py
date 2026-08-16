from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path


class TradeMemory:
    """Bounded learning from V5 + weak evidence from V4/V3. Never self-rewrites code."""

    def __init__(self, db_path: str, legacy_paths: list[str] | None = None, min_trades: int = 5):
        self.db_path = db_path
        self.legacy_paths = [p for p in (legacy_paths or []) if p]
        self.min_trades = min_trades
        self._cache: list[dict] = []
        self._cache_at: float = 0.0

    @staticmethod
    def _read_sells(path: str, limit: int = 3000) -> list[dict]:
        if not path or not Path(path).exists():
            return []
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT timestamp_ms,symbol,pnl_net,payload_json FROM paper_fills WHERE side='sell' AND pnl_net IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()
        out = []
        for ts, symbol, pnl, payload in rows:
            try: d = json.loads(payload or "{}")
            except Exception: d = {}
            out.append({"timestamp_ms": int(ts), "symbol": symbol, "pnl": float(pnl), "details": d})
        return out

    def trades(self) -> list[dict]:
        now = time.time()
        if self._cache and now - self._cache_at < 20.0:
            return self._cache
        out = self._read_sells(self.db_path)
        for p in self.legacy_paths:
            if p != self.db_path:
                out.extend(self._read_sells(p, 1500))
        self._cache = out
        self._cache_at = now
        return out

    def invalidate(self):
        self._cache_at = 0.0

    def repeat_penalty(self, symbol: str, window_seconds: int, penalty_per_trade: float) -> float:
        cutoff = int((time.time() - window_seconds) * 1000)
        n = sum(1 for t in self.trades() if t["symbol"] == symbol and t["timestamp_ms"] >= cutoff)
        return max(0.55, 1.0 - penalty_per_trade * max(0, n - 1))

    def memory_multiplier(self, symbol: str, brain: str, setup_key: str) -> float:
        selected = []
        for t in self.trades():
            d = t["details"]; meta = d.get("meta") or {}
            old_brain = str(meta.get("brain") or meta.get("engine") or "").upper()
            same_symbol = t["symbol"] == symbol
            same_brain = old_brain == brain.upper()
            same_setup = meta.get("setup_key") == setup_key
            if same_setup:
                selected.append((t["pnl"], 1.0))
            elif same_brain:
                selected.append((t["pnl"], 0.65))
            elif same_symbol:
                selected.append((t["pnl"], 0.30))
        effective_n = sum(w for _, w in selected)
        if effective_n < self.min_trades:
            return 1.0
        weighted_pnl = sum(p * w for p, w in selected)
        wins = sum(w for p, w in selected if p > 0)
        win_rate = wins / max(effective_n, 1e-9)
        avg_pnl = weighted_pnl / max(effective_n, 1e-9)
        wr_term = (win_rate - 0.50) * 0.40
        pnl_term = math.tanh(avg_pnl / 8.0) * 0.16
        return max(0.78, min(1.22, 1.0 + wr_term + pnl_term))


class ModelCompetition:
    """Forward-result promotion weighting for the four active brain families."""

    BRAINS = ("TRADER", "HUNTER", "SWING", "REVERSAL")

    def __init__(self, memory: TradeMemory, min_trades: int = 20):
        self.memory = memory
        self.min_trades = min_trades
        self._cache: list[dict] = []
        self._cache_at: float = 0.0

    def weights(self) -> dict[str, float]:
        trades = self.memory.trades()
        groups: dict[str, list[float]] = {b: [] for b in self.BRAINS}
        for t in trades:
            meta = (t.get("details") or {}).get("meta") or {}
            brain = str(meta.get("brain") or meta.get("engine") or "").upper()
            if brain in groups:
                groups[brain].append(float(t.get("pnl") or 0.0))
        out: dict[str, float] = {}
        for brain, vals in groups.items():
            if len(vals) < self.min_trades:
                out[brain] = 1.0
                continue
            wins = [x for x in vals if x > 0]; losses = [x for x in vals if x < 0]
            wr = len(wins) / len(vals)
            avg = sum(vals) / len(vals)
            pf = sum(wins) / abs(sum(losses)) if losses else (2.0 if wins else 0.0)
            evidence = min(1.0, len(vals) / 100.0)
            quality = 0.45 * (wr - 0.5) * 2 + 0.35 * math.tanh((pf - 1.0) / 1.2) + 0.20 * math.tanh(avg / 8.0)
            out[brain] = max(0.75, min(1.25, 1.0 + quality * 0.20 * evidence))
        return out
