from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from cloud_agent import CloudPaperAgent


agent = CloudPaperAgent()
bot_task: asyncio.Task | None = None
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task
    bot_task = asyncio.create_task(agent.run_forever())
    yield
    await agent.shutdown()
    if bot_task:
        try:
            await asyncio.wait_for(bot_task, timeout=15)
        except Exception:
            bot_task.cancel()
    agent.storage.close()


app = FastAPI(title="Crypto AI V5 — Autonomous Paper Portfolio Manager", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name": "Crypto AI V5",
        "mode": "PAPER_ONLY",
        "architecture": "Trader + Hunter + Swing + Reversal + Meta Brain + Portfolio Brain + Copilot",
        "dashboard_url": "/dashboard",
        "status_url": "/status",
        "report_url": "/report",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"ok": True, "mode": "paper", "version": 5, "state": agent.latest_status.get("state")}


@app.get("/status")
def status():
    return agent.status()


@app.get("/report")
def report():
    return agent.report()


@app.get("/radar")
def radar(limit: int = 600):
    return agent.radar[: max(1, min(limit, 600))]


@app.get("/universe")
def universe():
    return [c.to_dict() for c in agent.universe]


@app.get("/fast-cabinet")
def fast_cabinet():
    return agent.fast_cabinet_view


@app.get("/opportunities")
def opportunities(limit: int = 100):
    return agent.storage.recent_opportunities(max(1, min(limit, 1000)))


@app.get("/decisions")
def decisions(limit: int = 100):
    return agent.storage.recent_decisions(max(1, min(limit, 1000)))


@app.get("/journal")
def journal(limit: int = 100, symbol: str | None = None):
    return agent.storage.recent_journal(max(1, min(limit, 1000)), symbol=symbol)


@app.get("/trades")
def trades(limit: int = 100):
    return agent.storage.recent_fills(max(1, min(limit, 1000)))


@app.get("/closed-trades")
def closed_trades(limit: int = 300):
    return agent.storage.closed_trades(max(1, min(limit, 5000)))


@app.get("/calibration")
def calibration():
    return agent.storage.score_calibration()


@app.get("/shadow-analysis")
def shadow_analysis():
    return agent.storage.shadow_summary()


@app.post("/chat")
def chat(req: ChatRequest):
    return agent.chat(req.message)


@app.get("/export/trades.csv")
def export_trades_csv():
    rows = agent.storage.closed_trades(100000)
    fields = [
        "timestamp_ms", "symbol", "brain", "score", "confidence", "entry_price", "exit_price", "qty",
        "gross_pnl", "fees_total", "pnl_net", "return_pct", "mfe_pct", "mae_pct", "scale_count",
        "entry_time", "exit_time", "reason", "setup_key", "regime",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fields})
    return StreamingResponse(
        iter([buffer.getvalue().encode("utf-8")]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crypto_ai_v5_closed_trades.csv"},
    )
