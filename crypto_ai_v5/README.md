# Crypto AI V5 — Autonomous Multi-Brain Paper Portfolio Manager

V5 is a release-ready **paper trading / research** system built to run continuously on public crypto market data for several days without manual intervention. It contains no authenticated live-order adapter.

## Architecture

```text
Eligible USDT spot markets (up to ~600)
                |
                v
          MARKET RADAR
                |
                v
       ~50 Deep Analysis
                |
                +---------------------+
                |                     |
                v                     v
        Fast Cabinet             Rotating Deep Batch
       (~10–20 names)             + open positions
                |                     |
                +----------+----------+
                           v
          OHLCV + L2 + public trade flow
                           |
       +-------------------+-------------------+-------------------+
       |                   |                   |                   |
       v                   v                   v                   v
   TRADER               HUNTER 2.0           SWING              REVERSAL
learned returns      momentum/volume       15m/1h/4h        exhaustion/rebound
       |                   |                   |                   |
       +-------------------+-------------------+-------------------+
                           v
                    REGIME ROUTER
                           v
                     META BRAIN
             neutral cross-brain ranking
                           v
                   PORTFOLIO BRAIN
 dynamic sizing / correlation / opportunity cost / capital rotation
                           v
                    HARD RISK LAYER
                           v
              INTERNAL PAPER EXCHANGE
                           v
       position lifecycle / partials / runner / trailing / scale-in
                           v
          journal + shadow tracking + score calibration
                           v
                   TRADING COPILOT
```

## Brains

### Trader
Uses the existing nonlinear multi-horizon return model plus uncertainty, spread and current microstructure. Models are persisted under `/data/models_v5` so Railway restarts do not always start training from zero.

### Hunter 2.0
Looks for early volume/momentum expansion with acceleration, order-book/trade-flow confirmation, breakout shape and optional learned microstructure forecast. Late vertical moves are penalized.

### Swing
Runs only on priority names, majors and existing Swing positions. It uses cached 15m, 1h and 4h data and looks for multi-timeframe continuation rather than reacting to one-minute noise.

### Reversal
Long-only rebound brain. It requires a meaningful pullback/exhaustion state plus initial demand confirmation; it is not a blind “buy because RSI is low” rule.

## Meta Brain + Regime Router
The Regime Router uses broad-market breadth, median change and median range to classify `RISK_ON`, `RISK_OFF`, `HIGH_VOL`, `CHOP` or `NEUTRAL`. It changes brain/risk multipliers but does not create trades itself.

The Meta Brain ranks signals across all brains. No brain is permanently favored.

## Portfolio Brain

- True multi-position portfolio.
- No arbitrary hard target for number of positions.
- Default maximum total exposure: **60%**.
- Default maximum exposure per symbol: **15%**.
- Cash is a valid position.
- Dynamic sizing uses signal quality, confidence, liquidity and current regime.
- Correlation guard prevents redundant positions.
- Capital Rotation can reduce a materially weaker position to fund a stronger one.
- New globally ranked opportunities get first claim on spare capital; strong existing winners may then scale in if capacity remains.

## Exit Stability / brain-specific lifecycle
A weak snapshot is not automatically a full exit.

- Hard/trailing stops always execute immediately.
- Normal AI exits require brain-specific consecutive confirmations after minimum maturity.
- Severe deterioration can still force an earlier exit.
- Hunter and Swing have longer maturity / confirmation windows than Trader.
- Partial profit can lock a portion while leaving a runner.
- Positions can be reduced without full exit or scaled up while winning.

These defaults are research settings, not claims of optimal parameters.

## Learning from V3 and V4
If these files exist on the same Railway volume:

```text
/data/crypto_ai_v3.sqlite3
/data/crypto_ai_v4.sqlite3
```

V5 reads their closed-trade outcomes as **weak bounded evidence**. It never modifies those legacy databases. New V5 data is stored separately:

```text
/data/crypto_ai_v5.sqlite3
/data/paper_state_v5.json
/data/models_v5/
```

Learning mechanisms include:

- symbol / brain / setup memory multipliers,
- forward-result brain promotion weights,
- score calibration by score band,
- post-trade classification,
- rejection logging,
- shadow tracking after rejected opportunities and exits,
- 5m / 30m / 2h missed-upside observations,
- microstructure model training after enough snapshots.

V5 does **not** rewrite its own source code.

## Trading Copilot
The dashboard includes a working chat. It answers directly from V5’s live report and journal, for example:

- “Why did you enter COW?”
- “למה יצאת מ-COW?”
- “What is the strongest opportunity right now?”
- “Why is so much capital in cash?”
- “Which open position is weakest?”
- “How are Hunter and Swing performing?”

The default Copilot is deterministic and grounded, so it adds **no paid LLM cost** and cannot invent an exchange action. A paid language model is deliberately not called on every trading decision.

## Clean forward-test runs
`PAPER_RUN_ID` identifies a paper-test generation.

Default:

```text
PAPER_RUN_ID=v5-paper-1
```

When V5 sees a new `PAPER_RUN_ID`, it clears only the **V5** paper database/state once and starts a clean run. Restarts with the same ID preserve the portfolio. V3/V4 data is never deleted.

To start another clean forward test later, change it to e.g.:

```text
PAPER_RUN_ID=v5-paper-2
```

## Railway deployment

1. Replace the repository contents with this V5 folder.
2. Keep/add the persistent Railway Volume mounted at `/data`.
3. In Railway Variables set at minimum:

```text
DATA_DIR=/data
PAPER_RUN_ID=v5-paper-1
```

4. Recommended initial research variables are already the code defaults, but you can set them explicitly:

```text
STARTING_CASH=10000
MAX_TOTAL_EXPOSURE_PCT=0.60
MAX_POSITION_PCT=0.15
RISK_PER_TRADE=0.0045
RADAR_SIZE=600
DEEP_ANALYSIS_SIZE=50
FAST_CABINET_SIZE=15
CYCLE_SECONDS=30
LEARN_FROM_V3=true
LEARN_FROM_V4=true
```

5. Deploy.
6. Verify `/health`.
7. Open `/dashboard`.
8. Once the clean paper run starts, avoid changing parameters during the chosen multi-day evaluation period.

The included `Dockerfile` and `railway.toml` are ready for this flow.

## Main endpoints

- `/dashboard`
- `/health`
- `/status`
- `/report`
- `/radar`
- `/universe`
- `/fast-cabinet`
- `/opportunities`
- `/decisions`
- `/journal`
- `/trades`
- `/closed-trades`
- `/calibration`
- `/shadow-analysis`
- `POST /chat`
- `/export/trades.csv`

## Local verification

```bash
python smoke_test_v5.py
```

The smoke test is offline and validates all four brains, Meta/Regime logic, multi-position planning, scale-in, partial-profit runner behavior, full exit, journaling and shadow logging.

## Cost profile
The design avoids an LLM call per tick/decision. Compute is mostly exchange reads + lightweight feature/model inference. Actual Railway cost depends on the selected instance, request rate and runtime; this code does not itself impose a paid AI dependency.

## Important
This is a research system. Paper performance can differ materially from live trading because of liquidity, fills, latency, market impact and regime change. V5 should remain paper-only until a sufficiently large forward sample is evaluated after fees/slippage and across multiple market conditions.
