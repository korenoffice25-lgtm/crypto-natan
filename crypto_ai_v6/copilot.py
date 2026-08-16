from __future__ import annotations

import json
import re
from typing import Any

try:
    from openai import OpenAI
except Exception:  # optional dependency/fallback
    OpenAI = None


class TradingCopilot:
    """Grounded V6 assistant. With OPENAI_API_KEY it can also research public stocks/crypto via web search."""

    def __init__(self, agent, settings):
        self.agent = agent
        self.cfg = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if (OpenAI and settings.openai_api_key) else None

    @staticmethod
    def _hebrew(text: str) -> bool:
        return bool(re.search(r"[\u0590-\u05FF]", text or ""))

    def _live_context(self) -> dict[str, Any]:
        r = self.agent.report()
        return {
            "version": 6,
            "mode": "PAPER_ONLY",
            "equity": r.get("equity"),
            "cash": r.get("cash"),
            "capital_utilization_pct": r.get("capital_utilization_pct"),
            "open_risk_pct": r.get("open_risk_pct"),
            "regime": r.get("regime"),
            "positions": (r.get("open_positions") or [])[:15],
            "top_opportunities": (r.get("top_opportunities") or [])[:12],
            "next_actions": (r.get("next_actions") or [])[:15],
            "performance": r.get("performance"),
            "health": r.get("health"),
        }

    def _llm_answer(self, q: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        context = self._live_context()
        instructions = (
            "You are Crypto AI V6 Trading Copilot. You are embedded inside a PAPER-TRADING portfolio manager. "
            "Answer in the user's language. Ground any statement about the portfolio/system strictly in LIVE_STATE. "
            "You can explain stocks and crypto generally and, when web search is available, research current public market information. "
            "Clearly distinguish external market research from the live V6 portfolio. Never claim guaranteed returns. "
            "Never say a real order was placed; V6 has no authenticated real-order methods. "
            "Be concise but useful. When asked why the system did or did not trade, explain the actual visible signals, risk, regime and actions."
        )
        payload = f"LIVE_STATE:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\nUSER_QUESTION:\n{q}"
        kwargs: dict[str, Any] = {"model": self.cfg.copilot_model, "instructions": instructions, "input": payload}
        if self.cfg.copilot_web_search:
            kwargs["tools"] = [{"type":"web_search"}]
        try:
            response = self.client.responses.create(**kwargs)
            text = (getattr(response, "output_text", "") or "").strip()
            if text:
                return {"answer": text, "mode": "llm_grounded", "evidence": context}
        except Exception as e:
            return {"answer": self._fallback(q)["answer"], "mode": "fallback", "warning": f"Copilot API unavailable: {type(e).__name__}"}
        return None

    def _fallback(self, q: str) -> dict[str, Any]:
        he = self._hebrew(q)
        low = (q or "").lower()
        r = self.agent.report()
        top = r.get("top_opportunities") or []
        positions = r.get("open_positions") or []
        if any(x in low for x in ["cash","capital","exposure","utilization"]) or any(x in q for x in ["מזומן","הון","חשיפה","מושקע"]):
            ans = (f"כרגע החשיפה היא {r.get('capital_utilization_pct',0):.1f}% והקאש ${r.get('cash',0):,.2f}. יעד המשטר הוא {r.get('regime',{}).get('target_utilization_pct',0)*100:.0f}% ותקרת המשטר {r.get('regime',{}).get('max_utilization_pct',0)*100:.0f}%." if he
                   else f"Exposure is {r.get('capital_utilization_pct',0):.1f}% with ${r.get('cash',0):,.2f} cash. Regime target is {r.get('regime',{}).get('target_utilization_pct',0)*100:.0f}% and regime max {r.get('regime',{}).get('max_utilization_pct',0)*100:.0f}%.")
            return {"answer":ans,"mode":"local","evidence":[r.get("regime"),r.get("next_actions")]}
        if any(x in low for x in ["best","strongest","opportunity"]) or any(x in q for x in ["הזדמנות","הכי טובה","הכי חזק"]):
            if not top:
                return {"answer":"כרגע אין הזדמנות שעברה את ספי V6." if he else "No opportunity currently clears V6 thresholds.","mode":"local","evidence":[]}
            x=top[0]
            return {"answer":(f"המדורגת ראשונה היא {x['symbol']} דרך {x['brain']} עם Meta {x.get('meta_score',0):.1f} ו-Utility {x.get('utility',0):.1f}. {x.get('reason','')}" if he else f"Top-ranked is {x['symbol']} via {x['brain']} with Meta {x.get('meta_score',0):.1f}, utility {x.get('utility',0):.1f}. {x.get('reason','')}"),"mode":"local","evidence":[x]}
        if any(x in low for x in ["performance","profit","loss","win rate"]) or any(x in q for x in ["ביצועים","רווח","הפסד","אחוז הצלחה"]):
            p=r.get("performance") or {}
            return {"answer":(f"נסגרו {p.get('closed_trades',0)} עסקאות, P&L ממומש ${p.get('realized_pnl',0):,.2f}, Win Rate {p.get('win_rate_pct',0):.1f}%, Profit Factor {p.get('profit_factor',0):.2f}." if he else f"Closed trades {p.get('closed_trades',0)}, realized P&L ${p.get('realized_pnl',0):,.2f}, win rate {p.get('win_rate_pct',0):.1f}%, profit factor {p.get('profit_factor',0):.2f}."),"mode":"local","evidence":[p]}
        ans = (f"V6 פעילה ב-{r.get('regime',{}).get('name','UNKNOWN')}, עם {len(positions)} פוזיציות, חשיפה {r.get('capital_utilization_pct',0):.1f}% וסיכון פתוח {r.get('open_risk_pct',0):.2f}%. אפשר לשאול אותי על התיק, פוזיציות, הזדמנויות, למה לא נכנסנו, או מידע על מניות וקריפטו. חיפוש שוק חיצוני דורש OPENAI_API_KEY." if he else
               f"V6 is in {r.get('regime',{}).get('name','UNKNOWN')} with {len(positions)} positions, {r.get('capital_utilization_pct',0):.1f}% exposure and {r.get('open_risk_pct',0):.2f}% open risk. Ask about positions, opportunities, rejected trades, or stocks/crypto. External market research requires OPENAI_API_KEY.")
        return {"answer":ans,"mode":"local","evidence":top[:3]}

    def answer(self, q: str) -> dict[str, Any]:
        q=(q or "").strip()
        if not q:
            return {"answer":"שאל אותי משהו על V6, קריפטו או מניות.","mode":"local","evidence":[]}
        result=self._llm_answer(q)
        return result if result else self._fallback(q)
