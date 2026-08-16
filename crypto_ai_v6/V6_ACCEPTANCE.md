# V6 forward-test acceptance criteria

V6 is not considered ready because the server says `running`. It is ready for a longer paper test only when the mechanics below are observed in forward data.

## Mechanical pass

1. `/health` stays `running` and does not hide scanner/symbol/portfolio exceptions.
2. State survives Railway restarts when a persistent volume is mounted at `DATA_DIR`.
3. At least Trader, Hunter and Swing generate forward qualified signals when their market conditions occur; Reversal must generate when oversold-turn conditions occur.
4. Qualified signals produce one of: ENTERED, REJECTED_THRESHOLD, REJECTED by risk/correlation, or a logged portfolio CASH/rotation decision. No unexplained disappearance.
5. Multiple simultaneous positions can exist; there is no fixed position-count target.
6. Exposure never exceeds the regime max or absolute max; open risk never exceeds the hard cap.
7. Hunter fast-cabinet symbols are analyzed every cycle before the rotating deep universe.
8. Swing positions survive short-term noise through maturity + confirmation logic unless a hard stop/severe invalidation occurs.
9. Partial take-profit leaves a runner. Winners can be added to only within hard caps.
10. Capital rotation reduces an inferior position rather than requiring a full liquidation.
11. Duplicate action IDs cannot create duplicate fills after retry/restart.
12. Equity, cash and position market values reconcile within fee/slippage accounting tolerance.

## Multi-day paper evaluation

Do not judge V6 on win rate alone. Review:

- Net return
- Max drawdown
- Profit factor
- Expectancy
- Average win / average loss
- Capital utilization by regime
- Open-risk utilization
- Performance by brain
- Performance by regime
- Score calibration
- Missed opportunities after rejected signals
- Post-exit shadow moves
- Slippage/fee drag
- Percentage of cycles with degraded health

A profitable short run with an unacceptable drawdown is not a pass. A safe run that remains almost entirely cash despite repeated high-quality qualified opportunities is also not a pass.
