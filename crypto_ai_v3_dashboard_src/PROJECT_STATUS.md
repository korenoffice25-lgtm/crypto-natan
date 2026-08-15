# Crypto AI V4 — Build Status

## Core architecture
- [x] Global market ranking before execution
- [x] Learned-return Trader brain
- [x] Breakout/Momentum Hunter brain
- [x] Dynamic capital allocation up to configured portfolio exposure cap
- [x] No hard maximum number of positions; portfolio capacity is exposure/risk driven
- [x] Correlation guard between newly accepted opportunities
- [x] Per-symbol cooldown and repeat-entry penalty

## Learning / memory
- [x] Full market snapshots stored in SQLite
- [x] Entry context saved inside each paper position and propagated to the closed trade
- [x] Bounded performance memory by symbol / engine / setup
- [x] Optional soft learning from the existing V3 SQLite history
- [x] Microstructure dataset collection
- [x] Auto-train 30s/120s/300s microstructure model after enough snapshots
- [ ] Automatic promotion of entirely new self-generated models (intentionally disabled)

## Risk / exits
- [x] 60% default max total exposure
- [x] 15% default max single-position exposure
- [x] Risk-per-trade sizing
- [x] Daily loss stop
- [x] Drawdown kill switch
- [x] Dynamic hard stop
- [x] Trailing stop with separate Trader/Hunter behavior
- [x] Fees + slippage simulation

## Execution
- [x] Real public market data
- [x] Internal paper exchange only
- [ ] Live-order adapter (intentionally absent)

## Dashboard
- [x] Capital utilization
- [x] Open position count
- [x] Trader/Hunter label
- [x] Score + confidence
- [x] Top ranked opportunities
- [x] Trailing stop visibility
- [x] Closed-trade audit trail + CSV export
