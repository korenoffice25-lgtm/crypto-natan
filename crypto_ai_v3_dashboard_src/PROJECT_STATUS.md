# Crypto AI V3 — Build Status

## Runtime
- [x] Docker container
- [x] FastAPI health/status service
- [x] Railway config
- [x] Persistent virtual portfolio state
- [x] SQLite decision/trade journal

## Market selection
- [x] Dynamic multi-coin scanner
- [x] Liquid spot-market filtering
- [x] Stablecoin/leveraged-token exclusions
- [x] Spread filter
- [x] Order-book depth filter
- [x] Direction-neutral opportunity score
- [x] Per-market risk multiplier

## AI
- [x] Multi-horizon learned return model
- [x] Unsupervised regime model
- [x] L2 order-book state
- [x] Public trade-flow state
- [x] BUY / HOLD / EXIT / DO_NOTHING
- [x] Model uncertainty/cost penalty

## Portfolio safety
- [x] Max positions
- [x] Max position exposure
- [x] Max total exposure
- [x] Daily loss stop
- [x] Drawdown kill switch
- [x] Emergency hard stop outside AI
- [x] Fees + slippage simulation

## Not yet enabled
- [ ] Authenticated exchange testnet execution
- [ ] Real-money execution
- [ ] Automatic model promotion based on forward results
- [ ] Full historical L2 microstructure training

## Reporting dashboard
- [x] Live portfolio value in dollars
- [x] Realized and unrealized P&L in dollars
- [x] Return percentages
- [x] Open-position table
- [x] Closed-trade audit trail
- [x] Win/loss statistics and profit factor
- [x] Portfolio allocation pie
- [x] CSV export for closed trades
