from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    edge_bps REAL NOT NULL,
    universe_score REAL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    fill REAL NOT NULL,
    qty REAL NOT NULL,
    pnl_net REAL,
    reason TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_symbol_ts ON market_snapshots(symbol, timestamp_ms);

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    engine TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT,
    payload_json TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_decision(self, timestamp_ms: int, symbol: str, action: str, confidence: float, edge_bps: float, universe_score: float, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO decisions(timestamp_ms, symbol, action, confidence, edge_bps, universe_score, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp_ms, symbol, action, confidence, edge_bps, universe_score, json.dumps(payload)),
        )
        self.conn.commit()

    def add_fill(self, timestamp_ms: int, symbol: str, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO paper_fills(timestamp_ms, symbol, side, fill, qty, pnl_net, reason, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp_ms, symbol, payload.get("side", ""), float(payload.get("fill", 0.0)),
                float(payload.get("qty", 0.0)), payload.get("pnl_net"), payload.get("reason"), json.dumps(payload),
            ),
        )
        self.conn.commit()

    def add_universe(self, timestamp_ms: int, payload: list[dict[str, Any]]):
        self.conn.execute("INSERT INTO universe_scans(timestamp_ms, payload_json) VALUES (?, ?)", (timestamp_ms, json.dumps(payload)))
        self.conn.commit()

    def add_market_snapshot(self, timestamp_ms: int, symbol: str, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO market_snapshots(timestamp_ms, symbol, payload_json) VALUES (?, ?, ?)",
            (timestamp_ms, symbol, json.dumps(payload)),
        )
        self.conn.commit()

    def add_opportunity(self, timestamp_ms: int, payload: dict[str, Any], approved: bool, reason: str = ""):
        self.conn.execute(
            "INSERT INTO opportunities(timestamp_ms, symbol, engine, score, confidence, approved, reason, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp_ms, payload.get("symbol", ""), payload.get("engine", ""),
                float(payload.get("score", 0.0)), float(payload.get("confidence", 0.0)), int(bool(approved)),
                reason, json.dumps(payload),
            ),
        )
        self.conn.commit()


    def market_snapshot_count(self, symbol: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE symbol=?", (symbol,))
        return int(cur.fetchone()[0])

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT timestamp_ms, symbol, action, confidence, edge_bps, universe_score, payload_json FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "action": r[2], "confidence": r[3], "edge_bps": r[4], "universe_score": r[5], "details": json.loads(r[6])}
            for r in cur.fetchall()
        ]

    def recent_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT timestamp_ms, symbol, side, fill, qty, pnl_net, reason, payload_json FROM paper_fills ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "side": r[2], "fill": r[3], "qty": r[4], "pnl_net": r[5], "reason": r[6], "details": json.loads(r[7])}
            for r in cur.fetchall()
        ]

    def recent_opportunities(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT timestamp_ms, symbol, engine, score, confidence, approved, reason, payload_json FROM opportunities ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "engine": r[2], "score": r[3], "confidence": r[4], "approved": bool(r[5]), "reason": r[6], "details": json.loads(r[7])}
            for r in cur.fetchall()
        ]

    def closed_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT timestamp_ms, symbol, fill, qty, pnl_net, reason, payload_json FROM paper_fills WHERE side='sell' ORDER BY id DESC LIMIT ?", (limit,)
        )
        out = []
        for r in cur.fetchall():
            details = json.loads(r[6])
            meta = details.get("meta") or {}
            out.append({
                "timestamp_ms": r[0], "symbol": r[1], "entry_price": details.get("entry_price"),
                "exit_price": details.get("exit_price", r[2]), "qty": r[3], "gross_pnl": details.get("gross_pnl"),
                "fees_total": details.get("fees_total"), "pnl_net": r[4], "return_pct": details.get("return_pct"),
                "entry_time": details.get("entry_time"), "exit_time": details.get("exit_time"), "reason": r[5],
                "engine": meta.get("engine"), "score": meta.get("score"), "confidence": meta.get("confidence"),
                "setup_key": meta.get("setup_key"),
            })
        return out

    def performance_summary(self) -> dict[str, Any]:
        cur = self.conn.execute("SELECT pnl_net, payload_json FROM paper_fills WHERE side='sell' AND pnl_net IS NOT NULL")
        rows = cur.fetchall()
        pnls = [float(r[0] or 0.0) for r in rows]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gross_profit, gross_loss = sum(wins), sum(losses)
        by_engine: dict[str, list[float]] = {}
        for pnl, payload in rows:
            try:
                engine = (json.loads(payload or "{}").get("meta") or {}).get("engine") or "UNKNOWN"
            except Exception:
                engine = "UNKNOWN"
            by_engine.setdefault(engine, []).append(float(pnl or 0.0))
        engine_stats = {}
        for engine, vals in by_engine.items():
            ew = [x for x in vals if x > 0]
            el = [x for x in vals if x < 0]
            engine_stats[engine] = {
                "trades": len(vals), "pnl": sum(vals), "win_rate_pct": len(ew) / len(vals) * 100 if vals else 0.0,
                "profit_factor": (sum(ew) / abs(sum(el))) if el else (float("inf") if ew else 0.0),
            }
        return {
            "closed_trades": len(pnls), "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0.0,
            "realized_pnl": sum(pnls), "gross_profit": gross_profit, "gross_loss": gross_loss,
            "average_trade_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "best_trade_pnl": max(pnls) if pnls else 0.0, "worst_trade_pnl": min(pnls) if pnls else 0.0,
            "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0),
            "by_engine": engine_stats,
        }

    def close(self):
        self.conn.close()
