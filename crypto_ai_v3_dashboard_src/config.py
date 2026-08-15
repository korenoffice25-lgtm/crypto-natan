from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class Settings:
    # Public market data. No API keys are required for paper mode.
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    quote_currency: str = os.getenv("QUOTE_CURRENCY", "USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1m")

    # Universe scanner
    active_universe_size: int = int(os.getenv("ACTIVE_UNIVERSE_SIZE", "8"))
    scanner_prefilter_size: int = int(os.getenv("SCANNER_PREFILTER_SIZE", "40"))
    scanner_refresh_seconds: int = int(os.getenv("SCANNER_REFRESH_SECONDS", "600"))
    min_quote_volume: float = float(os.getenv("MIN_QUOTE_VOLUME", "5000000"))
    max_spread_bps: float = float(os.getenv("MAX_SPREAD_BPS", "25"))
    min_depth_notional: float = float(os.getenv("MIN_DEPTH_NOTIONAL", "25000"))

    # Agent cadence/model
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "20"))
    history_limit: int = int(os.getenv("HISTORY_LIMIT", "1000"))
    live_feature_candles: int = int(os.getenv("LIVE_FEATURE_CANDLES", "120"))
    orderbook_levels: int = int(os.getenv("ORDERBOOK_LEVELS", "20"))
    trade_limit: int = int(os.getenv("TRADE_LIMIT", "100"))
    model_retrain_seconds: int = int(os.getenv("MODEL_RETRAIN_SECONDS", "21600"))
    horizons: tuple[int, ...] = (1, 3, 12)
    min_training_rows: int = 550

    # Virtual portfolio
    starting_cash: float = float(os.getenv("STARTING_CASH", "10000"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "4"))

    # Hard Risk Governor. The AI cannot override these.
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.0035"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    max_total_exposure_pct: float = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.25"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.015"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.08"))

    # Decision threshold
    round_trip_cost_buffer_bps: float = float(os.getenv("ROUND_TRIP_COST_BUFFER_BPS", "18"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.58"))
    exit_confidence: float = float(os.getenv("EXIT_CONFIDENCE", "0.47"))

    # Persistence. On Railway mount a volume at /data and set DATA_DIR=/data.
    data_dir: str = os.getenv("DATA_DIR", ".")

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v3.sqlite3")

    @property
    def state_path(self) -> str:
        return str(Path(self.data_dir) / "paper_state.json")


SETTINGS = Settings()
