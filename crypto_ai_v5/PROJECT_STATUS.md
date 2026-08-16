# Crypto AI V5 — Release Status

## Release goal
V5 is a complete **paper-only** autonomous portfolio research system intended to be deployed and then left unchanged for a multi-day forward test.

## Market intelligence
- [x] Broad USDT radar (up to 600 eligible liquid spot markets)
- [x] ~50-name deep analysis layer
- [x] 10–20-name Fast Cabinet
- [x] Major coins always monitored
- [x] 1m OHLCV + L2 order book + public trade flow
- [x] 15m / 1h / 4h cached Swing analysis
- [x] Market breadth Regime Router

## Four brains
- [x] Trader — learned short-horizon return model
- [x] Hunter 2.0 — volume/momentum/tape expansion
- [x] Swing — 15m/1h/4h continuation structure
- [x] Reversal — long-only exhaustion/rebound setups
- [x] Neutral Meta Brain ranks all qualified brain signals

## Portfolio Brain
- [x] True multi-position allocations
- [x] No arbitrary fixed position-count target
- [x] 60% default portfolio exposure ceiling (not a target)
- [x] Dynamic position sizing
- [x] Cash treated as a valid allocation
- [x] Correlation guard
- [x] Opportunity-cost rejection
- [x] Capital rotation from weaker to materially stronger opportunities
- [x] Scaling into strong winners after new opportunities get first claim on capital
- [x] Regime-aware portfolio risk multiplier

## Position lifecycle
- [x] Brain-specific minimum maturity
- [x] Consecutive Exit Stability confirmations
- [x] Severe deterioration can exit before maturity
- [x] Hard stop always has priority
- [x] Brain-specific smart trailing stop
- [x] Partial take-profit + runner
- [x] Reduce without full exit
- [x] Scale-in within hard caps

## Learning / diagnostics
- [x] Reads V3 and V4 closed-trade evidence without altering legacy databases
- [x] Bounded symbol/brain/setup memory
- [x] Forward-result brain promotion weights
- [x] Score calibration by brain and score band
- [x] Full rejection reasons
- [x] Decision journal
- [x] Trade classification after exit
- [x] Shadow tracking after rejected opportunities and exits
- [x] Post-exit / missed-opportunity 5m, 30m and 2h observations
- [x] Microstructure snapshot collection + optional 30s/120s/300s model once enough data exists

## Dashboard / Copilot
- [x] Portfolio capacity panel
- [x] Four-brain analytics
- [x] Fast Cabinet
- [x] Open positions with live score / trailing / scale count
- [x] Global opportunity ranking
- [x] Model promotion weights
- [x] Score calibration
- [x] Closed-trade audit trail
- [x] CSV export
- [x] In-dashboard grounded Trading Copilot chat
- [x] Copilot answers from live portfolio, journal, opportunity and performance data
- [x] Copilot uses no paid LLM calls by default

## Safety / execution
- [x] Read-only public exchange gateway
- [x] No create_order method
- [x] No cancel_order method
- [x] Paper fills only
- [x] Fees + slippage simulation
- [x] Daily loss limit
- [x] Portfolio drawdown kill switch
- [x] Max total exposure
- [x] Max per-symbol exposure
- [x] Persistent Railway state
- [x] PAPER_RUN_ID clean-run mechanism

## Verification completed here
- [x] Python syntax compilation
- [x] Offline V5 smoke test
- [x] Four brain signal construction
- [x] Meta Brain / Regime Router
- [x] Multi-position Portfolio Brain
- [x] Scale-in
- [x] Partial take profit + runner
- [x] Journal and shadow logging

## Verification still required after deployment
- [ ] Railway dependency installation
- [ ] Binance public endpoint connectivity from the deployed region
- [ ] `/health` returns OK
- [ ] Dashboard loads live data
- [ ] Multi-day forward paper run

A multi-day profitable paper result is not proof of future or live profitability.
