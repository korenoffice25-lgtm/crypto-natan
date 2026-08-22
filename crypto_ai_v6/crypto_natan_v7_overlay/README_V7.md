# Crypto Natan V7 overlay

These files are designed to be placed inside the existing `crypto_ai_v6/` directory
on the `v6` branch.

## Files
- `engine_v7.py`: resilient V7 overlay on the existing V6 engine.
- `app_v7.py`: V7 FastAPI entrypoint.

## What V7 changes
1. Restarts automatically after transient market-data / CCXT startup failures.
2. Exposes recovering/connecting state instead of silently dying.
3. Final entry gate calculates expected edge after estimated round-trip costs.
4. Raises quality thresholds automatically during drawdown.
5. Hunter anti-FOMO filter blocks late vertical chases.
6. Reversal requires actual returning demand.
7. Swing and Trader get live microstructure/trend conflict filters.
8. Existing V6 PortfolioBrain, RiskEngine, correlation controls, position lifecycle,
   partials, runners and scaling remain intact.

## Deployment
Keep this PAPER ONLY.
Copy `engine_v7.py` into `crypto_ai_v6/`.
Either replace `app.py` with the contents of `app_v7.py`, or change Railway start command
to `uvicorn app_v7:app --host 0.0.0.0 --port ${PORT:-8080}`.

For persistence, set Railway `DATA_DIR` to a mounted persistent Volume path.
Do not change PAPER_RUN_ID unless you intentionally want a clean paper run because
the existing V6 `_prepare_run()` deletes state/database when the run id changes.

## Performance target
A 10%+ monthly return is an evaluation target, not a guarantee. V7 deliberately does
not force capital utilization or trade frequency to hit a monthly number. Judge it over
a meaningful paper sample using return, max drawdown, profit factor, expectancy,
win/loss asymmetry, brain/regime breakdown, and stability across restarts.
