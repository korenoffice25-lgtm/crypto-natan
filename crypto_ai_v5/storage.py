from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    brain TEXT,
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
    brain TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT,
    brain TEXT,
    event TEXT NOT NULL,
    reason TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_journal_symbol_ts ON journal(symbol, timestamp_ms);
CREATE TABLE IF NOT EXISTS shadow_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    brain TEXT,
    event_type TEXT NOT NULL,
    reference_price REAL NOT NULL,
    score REAL,
    reason TEXT,
    observations_json TEXT NOT NULL DEFAULT '{}',
    completed INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_shadow_symbol_completed ON shadow_events(symbol, completed, timestamp_ms);
"""


class Storage:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_decision(self, timestamp_ms: int, symbol: str, brain: str, action: str,
                     confidence: float, edge_bps: float, universe_score: float, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO decisions(timestamp_ms,symbol,brain,action,confidence,edge_bps,universe_score,payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (timestamp_ms, symbol, brain, action, confidence, edge_bps, universe_score, json.dumps(payload)),
        )
        self.conn.commit()

    def add_fill(self, timestamp_ms: int, symbol: str, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO paper_fills(timestamp_ms,symbol,side,fill,qty,pnl_net,reason,payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (timestamp_ms, symbol, payload.get("side", ""), float(payload.get("fill", 0.0)),
             float(payload.get("qty", 0.0)), payload.get("pnl_net"), payload.get("reason"), json.dumps(payload)),
        )
        self.conn.commit()

    def add_universe(self, timestamp_ms: int, payload: dict[str, Any]):
        self.conn.execute("INSERT INTO universe_scans(timestamp_ms,payload_json) VALUES(?,?)", (timestamp_ms, json.dumps(payload)))
        self.conn.commit()

    def add_market_snapshot(self, timestamp_ms: int, symbol: str, payload: dict[str, Any]):
        self.conn.execute(
            "INSERT INTO market_snapshots(timestamp_ms,symbol,payload_json) VALUES(?,?,?)",
            (timestamp_ms, symbol, json.dumps(payload)),
        )
        self.conn.commit()

    def add_opportunity(self, timestamp_ms: int, payload: dict[str, Any], approved: bool, reason: str = ""):
        brain = payload.get("brain") or payload.get("engine") or "UNKNOWN"
        self.conn.execute(
            "INSERT INTO opportunities(timestamp_ms,symbol,brain,score,confidence,approved,reason,payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (timestamp_ms, payload.get("symbol", ""), brain, float(payload.get("score", 0.0)),
             float(payload.get("confidence", 0.0)), int(bool(approved)), reason, json.dumps(payload)),
        )
        self.conn.commit()

    def add_journal(self, timestamp_ms: int, event: str, reason: str = "", symbol: str = "",
                    brain: str = "", payload: dict[str, Any] | None = None):
        self.conn.execute(
            "INSERT INTO journal(timestamp_ms,symbol,brain,event,reason,payload_json) VALUES(?,?,?,?,?,?)",
            (timestamp_ms, symbol, brain, event, reason, json.dumps(payload or {})),
        )
        self.conn.commit()

    def add_shadow_event(self, timestamp_ms: int, symbol: str, reference_price: float, event_type: str,
                         brain: str = "", score: float = 0.0, reason: str = "", payload: dict[str, Any] | None = None):
        if reference_price <= 0:
            return
        self.conn.execute(
            "INSERT INTO shadow_events(timestamp_ms,symbol,brain,event_type,reference_price,score,reason,observations_json,completed,payload_json) VALUES(?,?,?,?,?,?,?,'{}',0,?)",
            (timestamp_ms, symbol, brain, event_type, reference_price, score, reason, json.dumps(payload or {})),
        )
        self.conn.commit()

    def update_shadow(self, symbol: str, now_ms: int, current_price: float, horizons_seconds=(300, 1800, 7200)):
        cur = self.conn.execute(
            "SELECT id,timestamp_ms,reference_price,observations_json FROM shadow_events WHERE symbol=? AND completed=0 ORDER BY id ASC LIMIT 100",
            (symbol,),
        )
        changed = False
        max_h = max(horizons_seconds)
        for event_id, ts, ref, obs_json in cur.fetchall():
            try:
                obs = json.loads(obs_json or "{}")
            except Exception:
                obs = {}
            age_s = (now_ms - int(ts)) / 1000.0
            for h in horizons_seconds:
                key = f"{h}s_return_pct"
                if key not in obs and age_s >= h:
                    obs[key] = ((current_price / float(ref)) - 1.0) * 100.0 if ref else 0.0
                    changed = True
            completed = int(age_s >= max_h and all(f"{h}s_return_pct" in obs for h in horizons_seconds))
            self.conn.execute(
                "UPDATE shadow_events SET observations_json=?, completed=? WHERE id=?",
                (json.dumps(obs), completed, event_id),
            )
        if changed:
            self.conn.commit()

    def market_snapshot_count(self, symbol: str) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE symbol=?", (symbol,)).fetchone()[0])

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp_ms,symbol,brain,action,confidence,edge_bps,universe_score,payload_json FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "brain": r[2], "action": r[3], "confidence": r[4],
             "edge_bps": r[5], "universe_score": r[6], "details": json.loads(r[7])}
            for r in rows
        ]

    def recent_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp_ms,symbol,side,fill,qty,pnl_net,reason,payload_json FROM paper_fills ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "side": r[2], "fill": r[3], "qty": r[4],
             "pnl_net": r[5], "reason": r[6], "details": json.loads(r[7])}
            for r in rows
        ]

    def recent_opportunities(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp_ms,symbol,brain,score,confidence,approved,reason,payload_json FROM opportunities ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "brain": r[2], "score": r[3], "confidence": r[4],
             "approved": bool(r[5]), "reason": r[6], "details": json.loads(r[7])}
            for r in rows
        ]

    def recent_journal(self, limit: int = 100, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            rows = self.conn.execute(
                "SELECT timestamp_ms,symbol,brain,event,reason,payload_json FROM journal WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT timestamp_ms,symbol,brain,event,reason,payload_json FROM journal ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"timestamp_ms": r[0], "symbol": r[1], "brain": r[2], "event": r[3], "reason": r[4], "details": json.loads(r[5])}
            for r in rows
        ]

    def closed_trades(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT timestamp_ms,symbol,fill,qty,pnl_net,reason,payload_json FROM paper_fills WHERE side='sell' ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = json.loads(r[6]); meta = d.get("meta") or {}
            out.append({
                "timestamp_ms": r[0], "symbol": r[1], "entry_price": d.get("entry_price"),
                "exit_price": d.get("exit_price", r[2]), "qty": r[3], "gross_pnl": d.get("gross_pnl"),
                "fees_total": d.get("fees_total"), "pnl_net": r[4], "return_pct": d.get("return_pct"),
                "entry_time": d.get("entry_time"), "exit_time": d.get("exit_time"), "reason": r[5],
                "brain": meta.get("brain") or meta.get("engine"), "score": meta.get("score"),
                "confidence": meta.get("confidence"), "setup_key": meta.get("setup_key"),
                "mfe_pct": d.get("mfe_pct"), "mae_pct": d.get("mae_pct"), "scale_count": d.get("scale_count", 0),
                "partial_realized_pnl": d.get("partial_realized_pnl", 0.0), "regime": meta.get("regime"),
            })
        return out

    @staticmethod
    def _holding_minutes(entry_time: str | None, exit_time: str | None) -> float | None:
        if not entry_time or not exit_time:
            return None
        try:
            a = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            b = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            return (b - a).total_seconds() / 60.0
        except Exception:
            return None

    def performance_summary(self) -> dict[str, Any]:
        trades = self.closed_trades(100000)
        pnls = [float(t.get("pnl_net") or 0.0) for t in trades]
        wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x < 0]
        by_brain: dict[str, list[dict[str, Any]]] = {}
        for t in trades:
            by_brain.setdefault(str(t.get("brain") or "UNKNOWN"), []).append(t)
        brain_stats: dict[str, Any] = {}
        for brain, vals in by_brain.items():
            bp = [float(v.get("pnl_net") or 0.0) for v in vals]
            bw = [x for x in bp if x > 0]; bl = [x for x in bp if x < 0]
            holds = [self._holding_minutes(v.get("entry_time"), v.get("exit_time")) for v in vals]
            holds = [x for x in holds if x is not None]
            mfes = [float(v.get("mfe_pct") or 0.0) for v in vals]
            maes = [float(v.get("mae_pct") or 0.0) for v in vals]
            brain_stats[brain] = {
                "trades": len(vals), "pnl": sum(bp), "win_rate_pct": len(bw) / len(vals) * 100 if vals else 0.0,
                "profit_factor": (sum(bw) / abs(sum(bl))) if bl else (99.0 if bw else 0.0),
                "avg_winner": sum(bw) / len(bw) if bw else 0.0, "avg_loser": sum(bl) / len(bl) if bl else 0.0,
                "avg_holding_minutes": sum(holds) / len(holds) if holds else 0.0,
                "avg_mfe_pct": sum(mfes) / len(mfes) if mfes else 0.0,
                "avg_mae_pct": sum(maes) / len(maes) if maes else 0.0,
            }
        by_regime: dict[str, list[float]] = {}
        for t in trades:
            reg = str(t.get("regime") or "UNKNOWN")
            by_regime.setdefault(reg, []).append(float(t.get("pnl_net") or 0.0))
        regime_stats = {}
        for reg, vals in by_regime.items():
            rw = [x for x in vals if x > 0]; rl = [x for x in vals if x < 0]
            regime_stats[reg] = {
                "trades": len(vals), "pnl": sum(vals),
                "win_rate_pct": len(rw)/len(vals)*100 if vals else 0.0,
                "profit_factor": (sum(rw)/abs(sum(rl))) if rl else (99.0 if rw else 0.0),
            }
        gp, gl = sum(wins), sum(losses)
        return {
            "closed_trades": len(pnls), "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0.0,
            "realized_pnl": sum(pnls), "gross_profit": gp, "gross_loss": gl,
            "average_trade_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "best_trade_pnl": max(pnls) if pnls else 0.0, "worst_trade_pnl": min(pnls) if pnls else 0.0,
            "profit_factor": (gp / abs(gl)) if gl < 0 else (99.0 if gp > 0 else 0.0),
            "by_brain": brain_stats, "by_regime": regime_stats,
        }

    def score_calibration(self) -> list[dict[str, Any]]:
        bins: dict[tuple[str, int], list[float]] = {}
        for t in self.closed_trades(100000):
            score = t.get("score")
            if score is None:
                continue
            brain = str(t.get("brain") or "UNKNOWN")
            lo = int(float(score) // 10) * 10
            bins.setdefault((brain, lo), []).append(float(t.get("pnl_net") or 0.0))
        out = []
        for (brain, lo), vals in sorted(bins.items()):
            wins = [x for x in vals if x > 0]
            out.append({
                "brain": brain, "score_band": f"{lo}-{lo+9}", "trades": len(vals),
                "win_rate_pct": len(wins) / len(vals) * 100 if vals else 0.0,
                "avg_pnl": sum(vals) / len(vals) if vals else 0.0,
                "expectancy_positive": (sum(vals) / len(vals)) > 0 if vals else False,
            })
        return out

    def shadow_summary(self) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT event_type,brain,observations_json FROM shadow_events WHERE completed=1 ORDER BY id DESC LIMIT 5000"
        ).fetchall()
        groups: dict[str, list[float]] = {}
        for event_type, brain, obs_json in rows:
            try: obs = json.loads(obs_json or "{}")
            except Exception: obs = {}
            key = f"{event_type}:{brain or 'UNKNOWN'}"
            ret = obs.get("1800s_return_pct")
            if ret is not None:
                groups.setdefault(key, []).append(float(ret))
        return {
            k: {"events": len(v), "avg_30m_return_pct": sum(v)/len(v) if v else 0.0,
                "positive_30m_pct": sum(1 for x in v if x > 0)/len(v)*100 if v else 0.0}
            for k, v in groups.items()
        }

    def close(self):
        self.conn.close()
