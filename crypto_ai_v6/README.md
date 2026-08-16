# Crypto AI V6 — Autonomous Paper Portfolio Manager

V6 is a **paper-trading only** crypto portfolio manager. It uses public market data, four independent brains (Trader, Hunter, Swing, Reversal), a neutral Meta Brain, regime-aware capital allocation, hard portfolio risk controls, lifecycle-aware exits, partial profit + runner logic, scale-ins, capital rotation, decision logging, and a conversational Copilot.

## What changed from V5

- Portfolio errors are no longer silently overwritten by `state=running`; `/health` and `/status` expose degraded cycles and the actual error stage.
- Capital utilization is regime-aware instead of a fixed 60% ceiling. Cash is explicit, but strong qualified opportunities can compete for available capital up to the regime cap.
- Planner does **not pre-reserve requested target size**; the Risk Engine sizes each actual fill, avoiding the under-utilization pattern where a 12% request can be risk-sized to 3% but 12% capacity was treated as consumed.
- Trader no longer depends on a learned model being trained before it can generate opportunities.
- Hunter gets first-priority Fast Cabinet analysis each cycle.
- Swing has separate 15m/1h/4h analysis and longer maturity/exit confirmation.
- Position lifecycle: ENTRY → BUILDING → MATURE → WINNER → RUNNER.
- Portfolio actions: OPEN / ADD / HOLD / REDUCE / CLOSE / ROTATE / CASH.
- Paper execution is idempotent and models fee + spread/slippage.
- Copilot can use live V6 state; if `OPENAI_API_KEY` is set it can also research current public stocks/crypto using web search. It never executes real orders.

## Deploy on Railway

Deploy the `crypto_ai_v6` directory as the service root. Recommended variables:

```text
PAPER_RUN_ID=v6-paper-1
STARTING_CASH=10000
DATA_DIR=/data
COPILOT_MODEL=gpt-5.6
OPENAI_API_KEY=...      # optional; needed for full conversational + web-search Copilot
```

For multi-day persistence, attach a Railway volume at `/data` and set `DATA_DIR=/data`.

## Endpoints

- `/dashboard` — control room
- `/health` — real engine health, including degraded-cycle errors
- `/status` — live engine/portfolio state
- `/report` — full portfolio report
- `/radar`, `/universe`, `/fast-cabinet`
- `/opportunities`, `/journal`, `/trades`, `/equity-curve`
- `POST /chat` with `{"message":"..."}`

## Safety / scope

V6 has **no authenticated exchange trading methods**. It cannot place real orders. It is intended for forward paper testing and calibration. Profitability is not guaranteed; promote to any live execution layer only after a multi-day forward test demonstrates acceptable return, drawdown, execution behavior, and stability.

## Test

```bash
python smoke_test_v6.py
```
