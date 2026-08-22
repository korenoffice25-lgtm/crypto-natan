from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Storage:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS journal(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    symbol TEXT,
                    brain TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts_ms DESC);
                CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal(symbol, ts_ms DESC);

                CREATE TABLE IF NOT EXISTS fills(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts_ms DESC);

                CREATE TABLE IF NOT EXISTS opportunities(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    brain TEXT NOT NULL,
                    score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_opp_ts ON opportunities(ts_ms DESC);

                CREATE TABLE IF NOT EXISTS equity_curve(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    exposure_pct REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts_ms DESC);

                CREATE TABLE IF NOT EXISTS shadow_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    reference_price REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    brain TEXT,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    max_price REAL NOT NULL,
                    min_price REAL NOT NULL,
                    last_price REAL NOT NULL,
                    last_update_ms INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_symbol ON shadow_events(symbol, ts_ms DESC);
                """
            )

    def _insert(self, sql: str, args: tuple):
        with self.lock, self.conn:
            self.conn.execute(sql, args)

    def add_journal(self, event: str, reason: str, symbol: str = "", brain: str = "", payload: dict[str, Any] | None = None, ts_ms: int | None = None):
        ts_ms = int(ts_ms or time.time() * 1000)
        self._insert("INSERT INTO journal(ts_ms,event,reason,symbol,brain,payload) VALUES(?,?,?,?,?,?)",
                     (ts_ms, event, reason, symbol, brain, json.dumps(payload or {}, ensure_ascii=False)))

    def add_fill(self, symbol: str, fill: dict[str, Any], ts_ms: int | None = None):
        ts_ms = int(ts_ms or time.time() * 1000)
        self._insert("INSERT INTO fills(ts_ms,symbol,side,payload) VALUES(?,?,?,?)",
                     (ts_ms, symbol, str(fill.get("side", "")), json.dumps(fill, ensure_ascii=False)))

    def add_opportunity(self, signal: dict[str, Any], decision: str, reason: str, ts_ms: int | None = None):
        ts_ms = int(ts_ms or time.time() * 1000)
        self._insert("INSERT INTO opportunities(ts_ms,symbol,brain,score,decision,reason,payload) VALUES(?,?,?,?,?,?,?)",
                     (ts_ms, signal.get("symbol", ""), signal.get("brain", ""), float(signal.get("meta_score") or signal.get("score") or 0), decision, reason, json.dumps(signal, ensure_ascii=False)))

    def add_equity(self, equity: float, cash: float, exposure_pct: float, realized_pnl: float, ts_ms: int | None = None):
        ts_ms = int(ts_ms or time.time() * 1000)
        self._insert("INSERT INTO equity_curve(ts_ms,equity,cash,exposure_pct,realized_pnl) VALUES(?,?,?,?,?)",
                     (ts_ms, equity, cash, exposure_pct, realized_pnl))

    def _rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def recent_journal(self, limit: int = 100, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            rows = self._rows("SELECT * FROM journal WHERE symbol=? ORDER BY ts_ms DESC LIMIT ?", (symbol, limit))
        else:
            rows = self._rows("SELECT * FROM journal ORDER BY ts_ms DESC LIMIT ?", (limit,))
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    def recent_fills(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM fills ORDER BY ts_ms DESC LIMIT ?", (limit,))
        for r in rows:
            r.update(json.loads(r.pop("payload") or "{}"))
        return rows

    def recent_opportunities(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM opportunities ORDER BY ts_ms DESC LIMIT ?", (limit,))
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    def equity_curve(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._rows("SELECT ts_ms,equity,cash,exposure_pct,realized_pnl FROM equity_curve ORDER BY ts_ms DESC LIMIT ?", (limit,))
        return list(reversed(rows))

    def closed_trades(self, limit: int = 500) -> list[dict[str, Any]]:
        fills = self.recent_fills(max(limit * 5, 500))
        return [f for f in fills if f.get("side") == "sell" and "pnl_net" in f][:limit]

    def performance_summary(self) -> dict[str, Any]:
        trades = self.closed_trades(10000)
        pnls = [float(t.get("pnl_net", 0) or 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        by_brain: dict[str, dict[str, Any]] = {}
        for t in trades:
            brain = str((t.get("meta") or {}).get("brain") or "UNKNOWN")
            x = by_brain.setdefault(brain, {"trades": 0, "pnl": 0.0, "wins": 0})
            x["trades"] += 1
            x["pnl"] += float(t.get("pnl_net", 0) or 0)
            x["wins"] += int(float(t.get("pnl_net", 0) or 0) > 0)
        for x in by_brain.values():
            x["win_rate_pct"] = 100 * x["wins"] / x["trades"] if x["trades"] else 0.0
        return {
            "closed_trades": len(trades),
            "realized_pnl": sum(pnls),
            "win_rate_pct": 100 * len(wins) / len(trades) if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 1e-9 else (999.0 if gross_profit > 0 else 0.0),
            "average_win": sum(wins) / len(wins) if wins else 0.0,
            "average_loss": sum(losses) / len(losses) if losses else 0.0,
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "by_brain": by_brain,
        }


    def add_shadow_event(self, symbol: str, reference_price: float, event_type: str, brain: str, score: float, reason: str, payload: dict[str, Any] | None = None, ts_ms: int | None = None):
        ts_ms = int(ts_ms or time.time() * 1000)
        price = float(reference_price or 0.0)
        if price <= 0:
            return
        self._insert("INSERT INTO shadow_events(ts_ms,symbol,reference_price,event_type,brain,score,reason,max_price,min_price,last_price,last_update_ms,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (ts_ms, symbol, price, event_type, brain, float(score or 0), reason, price, price, price, ts_ms, json.dumps(payload or {}, ensure_ascii=False)))

    def update_shadow(self, symbol: str, price: float, ts_ms: int | None = None, horizon_ms: int = 86_400_000):
        ts_ms = int(ts_ms or time.time() * 1000)
        price = float(price or 0.0)
        if price <= 0:
            return
        cutoff = ts_ms - horizon_ms
        with self.lock, self.conn:
            rows = self.conn.execute("SELECT id,max_price,min_price FROM shadow_events WHERE symbol=? AND ts_ms>=?", (symbol, cutoff)).fetchall()
            for r in rows:
                self.conn.execute("UPDATE shadow_events SET max_price=?,min_price=?,last_price=?,last_update_ms=? WHERE id=?",
                                  (max(float(r['max_price']),price), min(float(r['min_price']),price), price, ts_ms, int(r['id'])))

    def shadow_summary(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM shadow_events ORDER BY ts_ms DESC LIMIT ?", (limit,))
        out=[]
        for r in rows:
            ref=float(r.get('reference_price') or 0)
            r['max_move_pct']=(float(r.get('max_price') or ref)/ref-1)*100 if ref else 0.0
            r['min_move_pct']=(float(r.get('min_price') or ref)/ref-1)*100 if ref else 0.0
            r['last_move_pct']=(float(r.get('last_price') or ref)/ref-1)*100 if ref else 0.0
            r['payload']=json.loads(r.get('payload') or '{}')
            out.append(r)
        return out

    def missed_opportunities(self, limit: int = 50) -> list[dict[str, Any]]:
        rows=self.shadow_summary(500)
        missed=[]
        for r in rows:
            if r.get('event_type')!='REJECTED_OPPORTUNITY':
                continue
            # Meaningful missed upside after rejection; purely diagnostic, not hindsight execution.
            if float(r.get('max_move_pct') or 0) >= 2.0:
                missed.append(r)
        return sorted(missed,key=lambda x:float(x.get('max_move_pct') or 0),reverse=True)[:limit]

    def score_calibration(self) -> list[dict[str, Any]]:
        trades=self.closed_trades(10000)
        buckets={}
        for t in trades:
            meta=t.get('meta') or {}
            score=float(meta.get('entry_meta_score',meta.get('entry_score',0)) or 0)
            if score<=0: continue
            lo=int(score//10)*10; key=f"{lo}-{lo+9}"
            x=buckets.setdefault(key,{'bucket':key,'trades':0,'wins':0,'pnl':0.0,'return_sum':0.0})
            pnl=float(t.get('pnl_net',0) or 0); ret=float(t.get('return_pct',0) or 0)
            x['trades']+=1; x['wins']+=int(pnl>0); x['pnl']+=pnl; x['return_sum']+=ret
        out=[]
        for _,x in sorted(buckets.items()):
            x['win_rate_pct']=100*x['wins']/x['trades'] if x['trades'] else 0.0
            x['avg_return_pct']=x['return_sum']/x['trades'] if x['trades'] else 0.0
            out.append(x)
        return out

    def performance_by_regime(self) -> dict[str, dict[str, Any]]:
        out={}
        for t in self.closed_trades(10000):
            regime=str((t.get('meta') or {}).get('regime') or 'UNKNOWN')
            x=out.setdefault(regime,{'trades':0,'wins':0,'pnl':0.0})
            pnl=float(t.get('pnl_net',0) or 0); x['trades']+=1; x['wins']+=int(pnl>0); x['pnl']+=pnl
        for x in out.values(): x['win_rate_pct']=100*x['wins']/x['trades'] if x['trades'] else 0.0
        return out

    def close(self):
        with self.lock:
            self.conn.close()
