# Crypto AI V4 — Adaptive Multi-Brain Paper Trader

V4 is a paper-only autonomous crypto research system. It consumes live public market data but does **not** expose any authenticated order methods.

## What V4 changes

V3 evaluated markets one at a time. V4 first builds opportunities across the active universe, then globally ranks them and allocates capital to the best available set.

```text
Liquid spot markets
      |
      v
Universe Scanner (broad activity/liquidity filter)
      |
      v
Market snapshots: OHLCV + L2 + public trades
      |
      +------------------------+
      |                        |
      v                        v
TRADER brain              HUNTER brain
learned returns           volume/momentum breakout
      |                        |
      +-----------+------------+
                  v
         Trade Memory weighting
                  v
           Global Ranking
                  v
     Correlation + Risk Governor
                  v
      Dynamic capital allocation
                  v
           Paper Exchange
       hard stop + trailing stop
                  v
         Outcome -> memory
```

## Default research guardrails

- Starting paper capital: `$10,000`
- Maximum total exposure: `60%`
- Maximum single-position exposure: `15%`
- No hard maximum count of positions
- Minimum useful position: `2.5%`
- Daily loss stop: `2%`
- Portfolio drawdown kill switch: `8%`
- Cooldown after exit: `30 minutes`

The 60% value is a **cap, not a target**. V4 can hold 100% cash when no opportunity meets its standards.

## Learning behavior

V4 stores the market state around decisions and each entry. Closed trades retain their engine, score, confidence, setup key and entry context. The memory layer applies only bounded ranking multipliers (0.75x–1.25x) after enough observations. It does not rewrite or deploy its own code.

If `LEARN_FROM_V3=true` and `/data/crypto_ai_v3.sqlite3` exists, V4 also uses old symbol-level outcomes as weak evidence. V4 itself writes to `/data/crypto_ai_v4.sqlite3` and `/data/paper_state_v4.json`, so the V3 dataset remains separate.

The system also stores market snapshots. Once a symbol has enough snapshots, V4 can train the included 30s/120s/300s microstructure model and use its forecast as one component of Hunter scoring.

## Railway

Use the existing Dockerfile/Railway setup. Mount the Railway Volume at `/data` and set:

```text
DATA_DIR=/data
```

Useful optional variables:

```text
MAX_TOTAL_EXPOSURE_PCT=0.60
MAX_POSITION_PCT=0.15
RISK_PER_TRADE=0.0045
ACTIVE_UNIVERSE_SIZE=12
SCANNER_PREFILTER_SIZE=60
COOLDOWN_SECONDS=1800
TRADER_MIN_SCORE=58
HUNTER_MIN_SCORE=68
LEARN_FROM_V3=true
```

## Endpoints

- `/dashboard`
- `/health`
- `/status`
- `/report`
- `/universe`
- `/opportunities`
- `/decisions`
- `/trades`
- `/closed-trades`
- `/export/trades.csv`

## Important

This build is for forward paper testing and research. A profitable paper result is not proof that the same strategy will remain profitable with real execution, larger size or changing market conditions.
