from __future__ import annotations

import math
from statistics import median
from typing import Any

from domain import Candidate


class UniverseScanner:
    def __init__(self, gateway, settings):
        self.gateway = gateway
        self.cfg = settings

    @staticmethod
    def _num(x, default=0.0) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else default
        except Exception:
            return default

    async def scan(self) -> dict[str, Any]:
        tickers = await self.gateway.fetch_tickers()
        rows: list[Candidate] = []
        changes: list[float] = []
        ranges: list[float] = []

        for symbol, t in tickers.items():
            if not symbol.endswith(f"/{self.cfg.quote_currency}"):
                continue
            if ":" in symbol:  # skip derivatives / settlement symbols
                continue
            base = symbol.split("/")[0].upper()
            if base in {self.cfg.quote_currency, "USDC", "FDUSD", "TUSD", "DAI", "EUR", "TRY"}:
                continue
            qv = self._num(t.get("quoteVolume"))
            if qv < self.cfg.min_quote_volume and base not in self.cfg.major_bases:
                continue
            bid = self._num(t.get("bid")); ask = self._num(t.get("ask")); last = self._num(t.get("last"))
            if bid <= 0 or ask <= 0 or last <= 0:
                continue
            spread = (ask / bid - 1.0) * 10_000
            if spread > self.cfg.max_spread_bps and base not in self.cfg.major_bases:
                continue
            pct = self._num(t.get("percentage")) / 100.0
            high = self._num(t.get("high")); low = self._num(t.get("low"))
            rng = (high / low - 1.0) if high > 0 and low > 0 else abs(pct)
            # Radar score rewards liquidity, movement and tradable spread; it does not make entry decisions.
            liq = min(1.0, math.log10(max(qv, 1.0)) / 9.0)
            motion = min(1.0, abs(pct) / 0.12 + rng / 0.20)
            spread_quality = max(0.0, 1.0 - spread / max(self.cfg.max_spread_bps, 1.0))
            major_bonus = 0.10 if base in self.cfg.major_bases else 0.0
            score = 100.0 * (0.48 * liq + 0.34 * motion + 0.18 * spread_quality + major_bonus)
            risk_multiplier = max(0.35, min(1.0, 0.45 + 0.40 * liq + 0.15 * spread_quality))
            rows.append(Candidate(symbol, base, self.cfg.quote_currency, qv, pct, rng, spread, 0.0, score, risk_multiplier, is_major=base in self.cfg.major_bases))
            changes.append(pct); ranges.append(rng)

        rows.sort(key=lambda c: c.market_score, reverse=True)
        rows = rows[: self.cfg.radar_size]
        for i, c in enumerate(rows, 1):
            c.radar_rank = i
        deep = list(rows[: self.cfg.deep_analysis_size])
        # Ensure majors are always monitored.
        known = {c.symbol for c in deep}
        for c in rows:
            if c.is_major and c.symbol not in known:
                deep.append(c); known.add(c.symbol)
        for i, c in enumerate(deep, 1):
            c.deep_rank = i
            c.fast_cabinet = i <= self.cfg.fast_cabinet_size

        pos_breadth = sum(1 for x in changes if x > 0) / len(changes) if changes else 0.5
        return {
            "radar": rows,
            "deep": deep,
            "fast": [c for c in deep if c.fast_cabinet],
            "breadth_positive_pct": pos_breadth,
            "median_change_pct": median(changes) if changes else 0.0,
            "median_range_pct": median(ranges) if ranges else 0.0,
        }
