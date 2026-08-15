# Crypto AI V3 — Multi-Coin Autonomous Paper Trader

V3 runs continuously against **real public crypto market data** while all capital and fills are virtual.
There is intentionally **no live-order adapter** in this build.

## What changed from V2

- Dynamic multi-coin universe scanner.
- Scans liquid spot markets rather than being limited to BTC/ETH.
- Direction-neutral candidate ranking: liquidity, market activity, order-book depth and spread.
- Filters stablecoins, leveraged-token variants, illiquid books and wide spreads.
- Smaller/less-liquid markets automatically get a lower risk multiplier.
- Multi-position portfolio limits.
- Persistent virtual wallet/positions across restarts.
- Emergency hard stop outside the AI.
- Live portfolio dashboard with allocation pie, open/closed P&L and trade history.\n- Downloadable closed-trade CSV report.\n- Cloud-ready FastAPI status/report endpoints.
- Dockerfile + Railway deployment config.

## Architecture

```text
All spot markets
      |
      v
Universe Scanner
(volume / activity / depth / spread)
      |
      v
Top 8 active markets
      |
      v
Market Reader
OHLCV + L2 book + public trades
      |
      v
Learned return models + regime model
      |
      v
Decision Agent
BUY / HOLD / EXIT / DO NOTHING
      |
      v
Risk Governor
position / portfolio / daily loss / drawdown / liquidity caps
      |
      v
Virtual Paper Exchange
fees + slippage + hard stop
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8080
```

Then open:

- `/status`
- `/universe`
- `/decisions`
- `/trades`

## Railway deployment

1. Put this folder in a GitHub repository.
2. Create a Railway project from the repository.
3. Railway detects the root `Dockerfile` automatically.
4. Add a persistent Railway Volume mounted at `/data`.
5. Add the environment variables from `.env.example` in Railway Variables.
6. Make sure `DATA_DIR=/data`.
7. Generate a Railway public domain.
8. Open `/health` and then `/status`.

The service command comes from the Dockerfile and launches FastAPI plus the background paper-trading agent in the same process.

## Starting settings

- Virtual capital: `$10,000`
- Active universe: `8` markets
- Maximum simultaneous positions: `4`
- Max single-position exposure: `10%`
- Max total exposure: `25%`
- Risk budget per new trade: `0.35%`
- Daily loss stop: `1.5%`
- Portfolio drawdown kill-switch: `8%`

These are research settings, not claims of optimality.

## Why internal paper money instead of exchange testnet first?

The system consumes the live public market and simulates execution locally. That lets us evaluate the strategy against real prices/order books without giving it trading credentials. Exchange testnet is useful later for validating authenticated order-routing code.

## Before real money

Do not add live execution until we have enough forward data to evaluate:

- total return after estimated fees/slippage,
- maximum drawdown,
- profit factor,
- turnover,
- results by market-cap/liquidity tier,
- results by market regime,
- out-of-sample stability,
- performance after multiple service restarts.
