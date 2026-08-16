from __future__ import annotations

import asyncio
from pathlib import Path
import time

import joblib

from features_v2 import add_market_features
from regime_model import RegimeModel
from return_model import MultiHorizonReturnModel


class LearnedModelRegistry:
    """Persistent Trader/regime models so Railway restarts do not retrain every market from zero."""

    def __init__(self, gateway, settings):
        self.gateway = gateway
        self.cfg = settings
        self.root = Path(settings.model_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.return_models: dict[str, MultiHorizonReturnModel] = {}
        self.regime_models: dict[str, RegimeModel] = {}
        self.trained_at: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._train_sem = asyncio.Semaphore(2)

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.joblib"

    def _try_load(self, symbol: str) -> bool:
        p = self._path(symbol)
        if not p.exists():
            return False
        try:
            payload = joblib.load(p)
            trained = float(payload.get("trained_at", 0.0))
            if time.time() - trained > self.cfg.model_retrain_seconds * 2:
                return False
            self.return_models[symbol] = payload["return_model"]
            self.regime_models[symbol] = payload["regime_model"]
            self.trained_at[symbol] = trained
            return True
        except Exception:
            return False

    async def ensure(self, symbol: str):
        age = time.time() - self.trained_at.get(symbol, 0.0)
        if symbol in self.return_models and age < self.cfg.model_retrain_seconds:
            return
        if symbol not in self.return_models and self._try_load(symbol):
            return
        lock = self._locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            age = time.time() - self.trained_at.get(symbol, 0.0)
            if symbol in self.return_models and age < self.cfg.model_retrain_seconds:
                return
            async with self._train_sem:
                raw = await self.gateway.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.history_limit)
                feat = add_market_features(raw, self.cfg.horizons, include_targets=True)
                if len(feat) < self.cfg.min_training_rows:
                    raise RuntimeError(f"Not enough training rows for {symbol}: {len(feat)}")

                def fit():
                    return MultiHorizonReturnModel(self.cfg.horizons).fit(feat), RegimeModel().fit(feat)
                ret, reg = await asyncio.to_thread(fit)
                trained = time.time()
                self.return_models[symbol] = ret; self.regime_models[symbol] = reg; self.trained_at[symbol] = trained
                try:
                    await asyncio.to_thread(joblib.dump, {"return_model": ret, "regime_model": reg, "trained_at": trained}, self._path(symbol))
                except Exception:
                    pass

    def predict(self, symbol: str, row):
        return self.return_models[symbol].predict(row), self.regime_models[symbol].read(row)
