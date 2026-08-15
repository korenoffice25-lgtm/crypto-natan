from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _b(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Public market data. No API keys are required for paper mode.
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    quote_currency: str = os.getenv("QUOTE_CURRENCY", "USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1m")

    # Universe scanner: broad prefilter, then deeper analysis on the top markets.
    active_universe_size: int = int(os.getenv("ACTIVE_UNIVERSE_SIZE", "12"))
    scanner_prefilter_size: int = int(os.getenv("SCANNER_PREFILTER_SIZE", "60"))
    scanner_refresh_seconds: int = int(os.getenv("SCANNER_REFRESH_SECONDS", "300"))
    min_quote_volume: float = float(os.getenv("MIN_QUOTE_VOLUME", "5000000"))
    max_spread_bps: float = float(os.getenv("MAX_SPREAD_BPS", "25"))
    min_depth_notional: float = float(os.getenv("MIN_DEPTH_NOTIONAL", "25000"))

    # Agent cadence/model.
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "20"))
    history_limit: int = int(os.getenv("HISTORY_LIMIT", "1000"))
    live_feature_candles: int = int(os.getenv("LIVE_FEATURE_CANDLES", "140"))
    orderbook_levels: int = int(os.getenv("ORDERBOOK_LEVELS", "20"))
    trade_limit: int = int(os.getenv("TRADE_LIMIT", "100"))
    model_retrain_seconds: int = int(os.getenv("MODEL_RETRAIN_SECONDS", "21600"))
    horizons: tuple[int, ...] = (1, 3, 12)
    min_training_rows: int = int(os.getenv("MIN_TRAINING_ROWS", "550"))

    # Virtual portfolio.
    starting_cash: float = float(os.getenv("STARTING_CASH", "10000"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))

    # V4 portfolio guardrails. No hard max-position count: capacity is risk/exposure driven.
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.0045"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.15"))
    max_total_exposure_pct: float = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.60"))
    min_position_pct: float = float(os.getenv("MIN_POSITION_PCT", "0.025"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.08"))

    # Decision threshold.
    round_trip_cost_buffer_bps: float = float(os.getenv("ROUND_TRIP_COST_BUFFER_BPS", "18"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.58"))
    exit_confidence: float = float(os.getenv("EXIT_CONFIDENCE", "0.47"))

    # Global ranking / Trader + Hunter.
    trader_min_score: float = float(os.getenv("TRADER_MIN_SCORE", "58"))
    hunter_min_score: float = float(os.getenv("HUNTER_MIN_SCORE", "68"))
    hunter_min_volume_z: float = float(os.getenv("HUNTER_MIN_VOLUME_Z", "1.15"))
    hunter_min_momentum_pct: float = float(os.getenv("HUNTER_MIN_MOMENTUM_PCT", "0.003"))
    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", "1800"))
    repeat_window_seconds: int = int(os.getenv("REPEAT_WINDOW_SECONDS", "10800"))
    repeat_penalty_per_trade: float = float(os.getenv("REPEAT_PENALTY_PER_TRADE", "0.08"))
    min_memory_trades: int = int(os.getenv("MIN_MEMORY_TRADES", "5"))
    correlation_lookback: int = int(os.getenv("CORRELATION_LOOKBACK", "60"))
    max_pair_correlation: float = float(os.getenv("MAX_PAIR_CORRELATION", "0.88"))

    # Dynamic stop/trailing behavior.
    trader_stop_vol_mult: float = float(os.getenv("TRADER_STOP_VOL_MULT", "2.5"))
    hunter_stop_vol_mult: float = float(os.getenv("HUNTER_STOP_VOL_MULT", "2.0"))
    trailing_activation_pct: float = float(os.getenv("TRAILING_ACTIVATION_PCT", "0.012"))
    trader_trailing_vol_mult: float = float(os.getenv("TRADER_TRAILING_VOL_MULT", "1.8"))
    hunter_trailing_vol_mult: float = float(os.getenv("HUNTER_TRAILING_VOL_MULT", "1.4"))

    # Persistence. Railway volume should be mounted at /data.
    data_dir: str = os.getenv("DATA_DIR", ".")
    learn_from_v3: bool = _b("LEARN_FROM_V3", "true")

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v4.sqlite3")

    @property
    def state_path(self) -> str:
        return str(Path(self.data_dir) / "paper_state_v4.json")

    @property
    def legacy_v3_db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v3.sqlite3")


SETTINGS = Settings()
