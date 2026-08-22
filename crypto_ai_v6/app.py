from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from engine import V6Engine


engine = V6Engine()
bot_task: asyncio.Task | None = None
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task
    bot_task = asyncio.create_task(engine.run_forever())
    yield
    await engine.shutdown()
    if bot_task:
        try:
            await asyncio.wait_for(bot_task, timeout=15)
        except Exception:
            bot_task.cancel()
    engine.storage.close()


app = FastAPI(title="Crypto AI V6 — Autonomous Paper Portfolio Manager", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name":"Crypto AI V6",
        "mode":"PAPER_ONLY",
        "mission":"capital growth through regime-aware portfolio allocation with hard drawdown controls",
        "brains":["TRADER","HUNTER","SWING","REVERSAL"],
        "dashboard_url":"/dashboard","status_url":"/status","report_url":"/report","health_url":"/health",
        "real_order_methods":False,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"version":6,"mode":"PAPER_ONLY",**engine.health_state}


@app.get("/status")
def status():
    return engine.status()


@app.get("/report")
def report():
    return engine.report()


@app.get("/radar")
def radar(limit: int = 200):
    return [c.to_dict() for c in engine.radar[:max(1,min(limit,600))]]


@app.get("/universe")
def universe():
    return [c.to_dict() for c in engine.universe]


@app.get("/fast-cabinet")
def fast_cabinet():
    return engine.fast_cabinet_view


@app.get("/opportunities")
def opportunities(limit: int = 200):
    return engine.storage.recent_opportunities(max(1,min(limit,2000)))


@app.get("/journal")
def journal(limit: int = 200, symbol: str | None = None):
    return engine.storage.recent_journal(max(1,min(limit,2000)),symbol=symbol)


@app.get("/trades")
def trades(limit: int = 200):
    return engine.storage.recent_fills(max(1,min(limit,2000)))


@app.get("/equity-curve")
def equity_curve(limit: int = 500):
    return engine.storage.equity_curve(max(10,min(limit,5000)))


@app.post("/chat")
def chat(req: ChatRequest):
    return engine.chat(req.message)


@app.get("/export/trades.csv")
def export_trades_csv():
    rows=engine.storage.closed_trades(100000)
    fields=["symbol","brain","entry_price","exit_price","qty","pnl_net","return_pct","mfe_pct","mae_pct","scale_count","entry_time","exit_time","reason"]
    buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=fields); w.writeheader()
    for row in rows:w.writerow({k:row.get(k) for k in fields})
    return StreamingResponse(iter([buf.getvalue().encode("utf-8")]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=crypto_ai_v6_closed_trades.csv"})
