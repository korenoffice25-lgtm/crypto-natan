from __future__ import annotations

import re
from typing import Any


class TradingCopilot:
    """Grounded, zero-LLM-cost chat over live V5 state and journals."""

    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _hebrew(text: str) -> bool:
        return bool(re.search(r"[\u0590-\u05FF]", text))

    @staticmethod
    def _symbol(text: str) -> str | None:
        m = re.search(r"\b([A-Z0-9]{2,12})(?:/USDT)?\b", text.upper())
        if not m:
            return None
        bad = {"WHY","WHAT","HOW","CASH","AI","USDT","V5","THE","AND","NOW"}
        return None if m.group(1) in bad else f"{m.group(1)}/USDT"

    def answer(self, text: str) -> dict[str, Any]:
        q = (text or "").strip()
        he = self._hebrew(q)
        low = q.lower()
        report = self.agent.report()
        symbol = self._symbol(q)

        if symbol and ("למה" in q or "why" in low):
            journal = self.agent.storage.recent_journal(8, symbol=symbol)
            if journal:
                j = journal[0]
                if he:
                    ans = f"האירוע האחרון של {symbol} היה {j['event']}: {j['reason']}."
                else:
                    ans = f"The latest {symbol} event was {j['event']}: {j['reason']}."
                return {"answer": ans, "evidence": journal[:4]}

        if any(x in low for x in ["best", "strongest", "opportunity"]) or any(x in q for x in ["הכי טובה", "הזדמנות", "הכי חזק"]):
            top = report.get("top_opportunities") or []
            if not top:
                ans = "כרגע אין הזדמנות שעברה את כל הסינונים." if he else "No opportunity currently passes all filters."
                return {"answer": ans, "evidence": []}
            x = top[0]
            score = x.get("meta_score") or x.get("score") or 0
            if he:
                ans = f"ההזדמנות המדורגת ראשונה כרגע היא {x['symbol']} דרך {x.get('brain')} עם Meta Score {score:.1f}. {x.get('reason','')}"
            else:
                ans = f"The top-ranked opportunity is {x['symbol']} via {x.get('brain')} with Meta Score {score:.1f}. {x.get('reason','')}"
            return {"answer": ans, "evidence": [x]}

        if any(x in low for x in ["cash", "exposure", "utilization", "capital"]) or any(x in q for x in ["מזומן", "חשיפה", "הון", "מושקע"]):
            u = report.get("capital_utilization_pct", 0.0); maxu = report.get("max_capital_utilization_pct", 0.0)
            cash = report.get("cash", 0.0)
            regime = report.get("regime") or {}
            if he:
                ans = f"כרגע {u:.1f}% מהתיק בחשיפה מתוך תקרה של {maxu:.1f}%, ו-${cash:,.2f} במזומן. משטר השוק: {regime.get('name','UNKNOWN')}. המזומן נשאר פנוי אם אין הזדמנויות עם עדיפות מספקת."
            else:
                ans = f"Current exposure is {u:.1f}% of a {maxu:.1f}% cap, with ${cash:,.2f} in cash. Regime: {regime.get('name','UNKNOWN')}. Cash remains available when ranked opportunities are not strong enough."
            return {"answer": ans, "evidence": [{"regime": regime, "utilization": u, "cash": cash}]}

        if any(x in low for x in ["weakest", "replace", "rotate"]) or any(x in q for x in ["חלשה", "להחליף", "רוטציה"]):
            positions = report.get("open_positions") or []
            if not positions:
                return {"answer": "אין כרגע פוזיציות פתוחות." if he else "There are no open positions.", "evidence": []}
            positions = sorted(positions, key=lambda p: float((p.get("meta") or {}).get("live_meta_score", (p.get("meta") or {}).get("score", 0)) or 0))
            p = positions[0]; s = float((p.get("meta") or {}).get("live_meta_score", 0) or 0)
            ans = (f"הפוזיציה החלשה ביותר לפי הדירוג החי היא {p['symbol']} עם Live Score {s:.1f}. Portfolio Brain יבצע רוטציה רק אם הזדמנות חדשה עוברת אותה בפער שהוגדר וגם תנאי הסיכון מאפשרים זאת."
                   if he else f"The weakest live-ranked position is {p['symbol']} at Live Score {s:.1f}. Portfolio Brain rotates only when a new opportunity clears the configured score advantage and risk rules.")
            return {"answer": ans, "evidence": [p]}

        if any(x in low for x in ["performance", "profit", "loss", "today", "win rate"]) or any(x in q for x in ["ביצועים", "רווח", "הפסד", "היום", "אחוז הצלחה"]):
            perf = report.get("performance") or {}
            if he:
                ans = f"נסגרו {perf.get('closed_trades',0)} עסקאות. P&L ממומש ${perf.get('realized_pnl',0):,.2f}, Win Rate {perf.get('win_rate_pct',0):.1f}%, Profit Factor {perf.get('profit_factor',0):.2f}."
            else:
                ans = f"Closed trades: {perf.get('closed_trades',0)}. Realized P&L ${perf.get('realized_pnl',0):,.2f}, win rate {perf.get('win_rate_pct',0):.1f}%, profit factor {perf.get('profit_factor',0):.2f}."
            return {"answer": ans, "evidence": [perf]}

        regime = report.get("regime") or {}
        top = (report.get("top_opportunities") or [])[:3]
        if he:
            ans = f"V5 פעילה במצב {regime.get('name','UNKNOWN')}. יש {report.get('open_position_count',0)} פוזיציות פתוחות, חשיפה {report.get('capital_utilization_pct',0):.1f}%, ו-{len(top)} הזדמנויות מובילות שמוצגות כרגע. אפשר לשאול אותי למה נכנסנו למטבע מסוים, מה ההזדמנות הכי חזקה, למה נשאר מזומן, איזו פוזיציה חלשה או מה הביצועים."
        else:
            ans = f"V5 is running in {regime.get('name','UNKNOWN')} regime with {report.get('open_position_count',0)} open positions and {report.get('capital_utilization_pct',0):.1f}% exposure. Ask why a symbol was traded, the best opportunity, why cash is idle, the weakest position, or performance."
        return {"answer": ans, "evidence": top}
