from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _b(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # V6 is intentionally PAPER ONLY. Market data is public and unauthenticated.
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    quote_currency: str = os.getenv("QUOTE_CURRENCY", "USDT")
    major_bases: tuple[str, ...] = _csv("MAJOR_BASES", "BTC,ETH,SOL,BNB,XRP,DOGE,ADA,LINK,AVAX")

    # Scanner / cadence
    radar_size: int = int(os.getenv("RADAR_SIZE", "600"))
    deep_analysis_size: int = int(os.getenv("DEEP_ANALYSIS_SIZE", "60"))
    fast_cabinet_size: int = int(os.getenv("FAST_CABINET_SIZE", "20"))
    rotation_batch_size: int = int(os.getenv("ROTATION_BATCH_SIZE", "14"))
    scanner_refresh_seconds: int = int(os.getenv("SCANNER_REFRESH_SECONDS", "240"))
    cycle_seconds: int = int(os.getenv("CYCLE_SECONDS", "25"))
    scanner_concurrency: int = int(os.getenv("SCANNER_CONCURRENCY", "8"))
    min_quote_volume: float = float(os.getenv("MIN_QUOTE_VOLUME", "2500000"))
    max_spread_bps: float = float(os.getenv("MAX_SPREAD_BPS", "35"))
    min_depth_notional: float = float(os.getenv("MIN_DEPTH_NOTIONAL", "15000"))

    # Market packets
    live_timeframe: str = os.getenv("LIVE_TIMEFRAME", "1m")
    live_candles: int = int(os.getenv("LIVE_CANDLES", "220"))
    orderbook_levels: int = int(os.getenv("ORDERBOOK_LEVELS", "20"))
    trade_limit: int = int(os.getenv("TRADE_LIMIT", "160"))
    max_data_age_seconds: int = int(os.getenv("MAX_DATA_AGE_SECONDS", "180"))
    swing_timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    swing_candles: int = int(os.getenv("SWING_CANDLES", "220"))
    swing_refresh_seconds: int = int(os.getenv("SWING_REFRESH_SECONDS", "240"))
    correlation_lookback: int = int(os.getenv("CORRELATION_LOOKBACK", "96"))

    # Paper execution
    starting_cash: float = float(os.getenv("STARTING_CASH", "10000"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.001"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))
    liquidity_slippage_scale_bps: float = float(os.getenv("LIQUIDITY_SLIPPAGE_SCALE_BPS", "8"))

    # Hard risk. These are ceilings, never utilization targets.
    absolute_max_exposure_pct: float = float(os.getenv("ABSOLUTE_MAX_EXPOSURE_PCT", "0.80"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.16"))
    min_position_pct: float = float(os.getenv("MIN_POSITION_PCT", "0.018"))
    max_open_risk_pct: float = float(os.getenv("MAX_OPEN_RISK_PCT", "0.050"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.025"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.085"))
    max_pair_correlation: float = float(os.getenv("MAX_PAIR_CORRELATION", "0.90"))

    # Per-brain risk budget per new trade (fraction of equity at stop).
    trader_risk_per_trade: float = float(os.getenv("TRADER_RISK_PER_TRADE", "0.0045"))
    hunter_risk_per_trade: float = float(os.getenv("HUNTER_RISK_PER_TRADE", "0.0055"))
    swing_risk_per_trade: float = float(os.getenv("SWING_RISK_PER_TRADE", "0.0050"))
    reversal_risk_per_trade: float = float(os.getenv("REVERSAL_RISK_PER_TRADE", "0.0038"))

    # Qualification thresholds. Meta score is 0..100.
    trader_min_score: float = float(os.getenv("TRADER_MIN_SCORE", "61"))
    hunter_min_score: float = float(os.getenv("HUNTER_MIN_SCORE", "66"))
    swing_min_score: float = float(os.getenv("SWING_MIN_SCORE", "65"))
    reversal_min_score: float = float(os.getenv("REVERSAL_MIN_SCORE", "65"))
    exceptional_score: float = float(os.getenv("EXCEPTIONAL_SCORE", "84"))

    # Portfolio allocation / rotation
    rotation_score_advantage: float = float(os.getenv("ROTATION_SCORE_ADVANTAGE", "10"))
    rotation_reduce_fraction: float = float(os.getenv("ROTATION_REDUCE_FRACTION", "0.45"))
    rotation_min_age_seconds: int = int(os.getenv("ROTATION_MIN_AGE_SECONDS", "300"))
    utilization_tolerance_pct: float = float(os.getenv("UTILIZATION_TOLERANCE_PCT", "0.04"))
    min_cash_reserve_pct: float = float(os.getenv("MIN_CASH_RESERVE_PCT", "0.12"))

    # Scaling / partials
    scale_score_threshold: float = float(os.getenv("SCALE_SCORE_THRESHOLD", "82"))
    scale_min_profit_pct: float = float(os.getenv("SCALE_MIN_PROFIT_PCT", "0.008"))
    scale_fraction: float = float(os.getenv("SCALE_FRACTION", "0.35"))
    max_scales_per_position: int = int(os.getenv("MAX_SCALES_PER_POSITION", "2"))
    scale_cooldown_seconds: int = int(os.getenv("SCALE_COOLDOWN_SECONDS", "600"))
    partial_profit_fraction: float = float(os.getenv("PARTIAL_PROFIT_FRACTION", "0.35"))

    # Lifecycle
    trader_min_maturity_seconds: int = int(os.getenv("TRADER_MIN_MATURITY_SECONDS", "90"))
    hunter_min_maturity_seconds: int = int(os.getenv("HUNTER_MIN_MATURITY_SECONDS", "240"))
    swing_min_maturity_seconds: int = int(os.getenv("SWING_MIN_MATURITY_SECONDS", "1800"))
    reversal_min_maturity_seconds: int = int(os.getenv("REVERSAL_MIN_MATURITY_SECONDS", "180"))
    trader_exit_confirmations: int = int(os.getenv("TRADER_EXIT_CONFIRMATIONS", "2"))
    hunter_exit_confirmations: int = int(os.getenv("HUNTER_EXIT_CONFIRMATIONS", "3"))
    swing_exit_confirmations: int = int(os.getenv("SWING_EXIT_CONFIRMATIONS", "4"))
    reversal_exit_confirmations: int = int(os.getenv("REVERSAL_EXIT_CONFIRMATIONS", "2"))

    # Stops and profit management
    trader_stop_pct: float = float(os.getenv("TRADER_STOP_PCT", "0.018"))
    hunter_stop_pct: float = float(os.getenv("HUNTER_STOP_PCT", "0.026"))
    swing_stop_pct: float = float(os.getenv("SWING_STOP_PCT", "0.045"))
    reversal_stop_pct: float = float(os.getenv("REVERSAL_STOP_PCT", "0.022"))
    trader_partial_profit_pct: float = float(os.getenv("TRADER_PARTIAL_PROFIT_PCT", "0.018"))
    hunter_partial_profit_pct: float = float(os.getenv("HUNTER_PARTIAL_PROFIT_PCT", "0.035"))
    swing_partial_profit_pct: float = float(os.getenv("SWING_PARTIAL_PROFIT_PCT", "0.055"))
    reversal_partial_profit_pct: float = float(os.getenv("REVERSAL_PARTIAL_PROFIT_PCT", "0.024"))
    trader_trail_activation_pct: float = float(os.getenv("TRADER_TRAIL_ACTIVATION_PCT", "0.013"))
    hunter_trail_activation_pct: float = float(os.getenv("HUNTER_TRAIL_ACTIVATION_PCT", "0.024"))
    swing_trail_activation_pct: float = float(os.getenv("SWING_TRAIL_ACTIVATION_PCT", "0.038"))
    reversal_trail_activation_pct: float = float(os.getenv("REVERSAL_TRAIL_ACTIVATION_PCT", "0.017"))
    trader_trail_distance_pct: float = float(os.getenv("TRADER_TRAIL_DISTANCE_PCT", "0.012"))
    hunter_trail_distance_pct: float = float(os.getenv("HUNTER_TRAIL_DISTANCE_PCT", "0.018"))
    swing_trail_distance_pct: float = float(os.getenv("SWING_TRAIL_DISTANCE_PCT", "0.028"))
    reversal_trail_distance_pct: float = float(os.getenv("REVERSAL_TRAIL_DISTANCE_PCT", "0.014"))

    cooldown_seconds: int = int(os.getenv("COOLDOWN_SECONDS", "900"))

    # Persistence
    data_dir: str = os.getenv("DATA_DIR", ".")
    paper_run_id: str = os.getenv("PAPER_RUN_ID", "v6-paper-1")

    # Copilot. If no API key is configured, V6 falls back to a grounded local assistant.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    copilot_model: str = os.getenv("COPILOT_MODEL", "gpt-5.6")
    copilot_web_search: bool = _b("COPILOT_WEB_SEARCH", "true")

    @property
    def state_path(self) -> str:
        return str(Path(self.data_dir) / "paper_state_v6.json")

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "crypto_ai_v6.sqlite3")

    @property
    def run_marker_path(self) -> str:
        return str(Path(self.data_dir) / "v6_paper_run_id.txt")


SETTINGS = Settings()
