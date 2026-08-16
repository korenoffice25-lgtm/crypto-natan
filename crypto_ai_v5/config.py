from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _b(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass
class Settings:
    # Public market data only. V5 intentionally has no authenticated exchange methods.
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    quote_currency: str = os.getenv("QUOTE_CURRENCY", "USDT")

    # Radar -> deep analysis -> Fast Cabinet.
    radar_size: int = int(os.getenv("RADAR_SIZE", "600"))
    deep_analysis_size: int = int(os.getenv("DEEP_ANALYSIS_SIZE", "50"))
    fast_cabinet_size: int = int(os.getenv("FAST_CABINET_SIZE", "15"))
    rotation_batch_size: int = int(os.getenv("ROTATION_BATCH_SIZE", "10"))
    scanner_refresh_seconds: int = int(os.getenv("SCANNER_REFRESH_SECONDS", "300"))
    min_quote_volume: float = float(os.getenv("MIN_QUOTE_VOLUME", "3000000"))
    max_spread_bps: float = float(os.getenv("MAX_SPREAD_BPS", "30"))
    min_depth_notional: float = float(os.getenv("MIN_DEPTH_NOTIONAL", "20000"))
    scanner_concurrency: int = int(os.getenv("SCANNER_CONCURRENCY", "8"))
    major_bases: tuple[str, ...] = _csv("MAJOR_BASES", "BTC,ETH,SOL,BNB,XRP,DOGE,ADA")

    # Agent cadence.
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "30"))
    timeframe: str = os.getenv("TIMEFRAME", "1m")
    history_limit: int = int(os.getenv("HISTORY_LIMIT", "1100"))
    live_feature_candles: int = int(os.getenv("LIVE_FEATURE_CANDLES", "160"))
    orderbook_levels: int = int(os.getenv("ORDERBOOK_LEVELS", "20"))
    trade_limit: int = int(os.getenv("TRADE_LIMIT", "120"))
    model_retrain_seconds: int = int(os.getenv("MODEL_RETRAIN_SECONDS", "21600"))
    horizons: tuple[int, ...] = (1, 3, 12)
    min_training_rows: int = int(os.getenv("MIN_TRAINING_ROWS", "550"))

    # Swing cache / multi-timeframe inputs.
    swing_refresh_seconds: int = int(os.getenv("SWING_REFRESH_SECONDS", "300"))
    swing_timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    swing_candle_limit: int = int(os.getenv("SWING_CANDLE_LIMIT", "220"))

    # Paper portfolio.
    starting_cash: float = float(os.getenv("STARTING_CASH", "10000"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))

    # Hard risk layer. 60% is a ceiling, not a utilization target.
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.0045"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.15"))
    max_total_exposure_pct: float = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.60"))
    min_position_pct: float = float(os.getenv("MIN_POSITION_PCT", "0.02"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.08"))
    max_pair_correlation: float = float(os.getenv("MAX_PAIR_CORRELATION", "0.88"))
    correlation_lookback: int = int(os.getenv("CORRELATION_LOOKBACK", "72"))

    # Decision thresholds.
    round_trip_cost_buffer_bps: float = float(os.getenv("ROUND_TRIP_COST_BUFFER_BPS", "18"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.58"))
    exit_confidence: float = float(os.getenv("EXIT_CONFIDENCE", "0.47"))
    trader_min_score: float = float(os.getenv("TRADER_MIN_SCORE", "58"))
    hunter_min_score: float = float(os.getenv("HUNTER_MIN_SCORE", "66"))
    swing_min_score: float = float(os.getenv("SWING_MIN_SCORE", "64"))
    reversal_min_score: float = float(os.getenv("REVERSAL_MIN_SCORE", "64"))

    # Hunter / Reversal inputs.
    hunter_min_volume_z: float = float(os.getenv("HUNTER_MIN_VOLUME_Z", "1.10"))
    hunter_min_momentum_pct: float = float(os.getenv("HUNTER_MIN_MOMENTUM_PCT", "0.0025"))
    reversal_min_drop_pct: float = float(os.getenv("REVERSAL_MIN_DROP_PCT", "0.006"))

    # Portfolio Brain.
    rotation_score_advantage: float = float(os.getenv("ROTATION_SCORE_ADVANTAGE", "14"))
    rotation_min_age_seconds: int = int(os.getenv("ROTATION_MIN_AGE_SECONDS", "300"))
    rotation_reduce_fraction: float = float(os.getenv("ROTATION_REDUCE_FRACTION", "0.50"))
    scale_score_threshold: float = float(os.getenv("SCALE_SCORE_THRESHOLD", "82"))
    scale_min_profit_pct: float = float(os.getenv("SCALE_MIN_PROFIT_PCT", "0.006"))
    scale_fraction: float = float(os.getenv("SCALE_FRACTION", "0.35"))
    scale_cooldown_seconds: int = int(os.getenv("SCALE_COOLDOWN_SECONDS", "900"))
    max_scales_per_position: int = int(os.getenv("MAX_SCALES_PER_POSITION", "2"))

    # Trade lifecycle by brain.
    trader_min_maturity_seconds: int = int(os.getenv("TRADER_MIN_MATURITY_SECONDS", "60"))
    hunter_min_maturity_seconds: int = int(os.getenv("HUNTER_MIN_MATURITY_SECONDS", "300"))
    swing_min_maturity_seconds: int = int(os.getenv("SWING_MIN_MATURITY_SECONDS", "1800"))
    reversal_min_maturity_seconds: int = int(os.getenv("REVERSAL_MIN_MATURITY_SECONDS", "180"))
    trader_exit_confirmations: int = int(os.getenv("TRADER_EXIT_CONFIRMATIONS", "2"))
    hunter_exit_confirmations: int = int(os.getenv("HUNTER_EXIT_CONFIRMATIONS", "3"))
    swing_exit_confirmations: int = int(os.getenv("SWING_EXIT_CONFIRMATIONS", "3"))
    reversal_exit_confirmations: int = int(os.getenv("REVERSAL_EXIT_CONFIRMATIONS", "2"))

    # Stops / trailing.
    trader_stop_vol_mult: float = float(os.getenv("TRADER_STOP_VOL_MULT", "2.5"))
    hunter_stop_vol_mult: float = float(os.getenv("HUNTER_STOP_VOL_MULT", "2.2"))
    swing_stop_vol_mult: float = float(os.getenv("SWING_STOP_VOL_MULT", "3.2"))
    reversal_stop_vol_mult: float = float(os.getenv("REVERSAL_STOP_VOL_MULT", "2.0"))
    trader_trailing_activation_pct: float = float(os.getenv("TRADER_TRAILING_ACTIVATION_PCT", "0.010"))
    hunter_trailing_activation_pct: float = float(os.getenv("HUNTER_TRAILING_ACTIVATION_PCT", "0.018"))
    swing_trailing_activation_pct: float = float(os.getenv("SWING_TRAILING_ACTIVATION_PCT", "0.030"))
    reversal_trailing_activation_pct: float = float(os.getenv("REVERSAL_TRAILING_ACTIVATION_PCT", "0.012"))
    trader_trailing_vol_mult: float = float(os.getenv("TRADER_TRAILING_VOL_MULT", "1.8"))
    hunter_trailing_vol_mult: float = float(os.getenv("HUNTER_TRAILING_VOL_MULT", "1.6"))
    swing_trailing_vol_mult: float = float(os.getenv("SWING_TRAILING_VOL_MULT", "2.4"))
    reversal_trailing_vol_mult: float = float(os.getenv("REVERSAL_TRAILING_VOL_MULT", "1.5"))

    # Partial profit. A runner remains unless disabled.
    enable_partial_profit: bool = _b("ENABLE_PARTIAL_PROFIT", "true")
    trader_partial_profit_pct: float = float(os.getenv("TRADER_PARTIAL_PROFIT_PCT", "0.015"))
    hunter_partial_profit_pct: float = float(os.getenv("HUNTER_PARTIAL_PROFIT_PCT", "0.030"))
    swing_partial_profit_pct: float = float(os.getenv("SWING_PARTIAL_PROFIT_PCT", "0.050"))
    reversal_partial_profit_pct: float = float(os.getenv("REVERSAL_PARTIAL_PROFIT_PCT", "0.020"))
    partial_profit_fraction: float = float(os.getenv("PARTIAL_PROFIT_FRACTION", "0.40"))

    # Cooldown / memory / score learning.
    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", "1200"))
    repeat_window_seconds: int = int(os.getenv("REPEAT_WINDOW_SECONDS", "10800"))
    repeat_penalty_per_trade: float = float(os.getenv("REPEAT_PENALTY_PER_TRADE", "0.08"))
    min_memory_trades: int = int(os.getenv("MIN_MEMORY_TRADES", "5"))
    model_promotion_min_trades: int = int(os.getenv("MODEL_PROMOTION_MIN_TRADES", "20"))
    shadow_min_score: float = float(os.getenv("SHADOW_MIN_SCORE", "65"))

    # Persistence / migration. V3 and V4 are read-only evidence; V5 has separate files.
    data_dir: str = os.getenv("DATA_DIR", ".")
    learn_from_v3: bool = _b("LEARN_FROM_V3", "true")
    learn_from_v4: bool = _b("LEARN_FROM_V4", "true")
    paper_run_id: str = os.getenv("PAPER_RUN_ID", "v5-paper-1")

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v5.sqlite3")

    @property
    def state_path(self) -> str:
        return str(Path(self.data_dir) / "paper_state_v5.json")

    @property
    def model_dir(self) -> str:
        return str(Path(self.data_dir) / "models_v5")

    @property
    def run_marker_path(self) -> str:
        return str(Path(self.data_dir) / "v5_paper_run_id.txt")

    @property
    def legacy_v3_db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v3.sqlite3")

    @property
    def legacy_v4_db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v4.sqlite3")


SETTINGS = Settings()
