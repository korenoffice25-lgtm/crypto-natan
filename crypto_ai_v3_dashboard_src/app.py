from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from cloud_agent import CloudPaperAgent


agent = CloudPaperAgent()
bot_task: asyncio.Task | None = None
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


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


app = FastAPI(title="Crypto AI V3 — Paper Trading", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name": "Crypto AI V3",
        "mode": "PAPER_ONLY",
        "dashboard_url": "/dashboard",
        "report_url": "/report",
        "status_url": "/status",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"ok": True, "mode": "paper"}


@app.get("/status")
def status():
    return agent.status()


@app.get("/report")
def report():
    return agent.report()


@app.get("/universe")
def universe():
    return [c.to_dict() for c in agent.universe]


@app.get("/decisions")
def decisions(limit: int = 50):
    return agent.storage.recent_decisions(max(1, min(limit, 200)))


@app.get("/trades")
def trades(limit: int = 50):
    return agent.storage.recent_fills(max(1, min(limit, 200)))


@app.get("/closed-trades")
def closed_trades(limit: int = 100):
    return agent.storage.closed_trades(max(1, min(limit, 1000)))


@app.get("/export/trades.csv")
def export_trades_csv():
    rows = agent.storage.closed_trades(100000)
    buffer = io.StringIO()
    fields = [
        "timestamp_ms", "symbol", "entry_price", "exit_price", "qty",
        "gross_pnl", "fees_total", "pnl_net", "return_pct",
        "entry_time", "exit_time", "reason",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fields})
    return StreamingResponse(
        iter([buffer.getvalue().encode("utf-8")]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crypto_ai_closed_trades.csv"},
    )
