from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

from opportunity_engine import Opportunity


@dataclass
class MarketRegime:
    name: str
    confidence: float
    breadth_positive_pct: float
    median_change_pct: float
    median_range_pct: float
    risk_multiplier: float
    brain_multipliers: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeRouter:
    """Portfolio-level market-state router. It never generates trades itself."""

    def classify(self, breadth_positive_pct: float, median_change_pct: float, median_range_pct: float) -> MarketRegime:
        b = float(breadth_positive_pct)
        ch = float(median_change_pct)
        rng = float(median_range_pct)

        if rng >= 0.10:
            name = "HIGH_VOL"
            confidence = min(1.0, 0.55 + (rng - 0.10) * 3.0)
            risk = 0.72
            mult = {"TRADER": 0.90, "HUNTER": 1.06, "SWING": 0.82, "REVERSAL": 1.05}
            reason = "Broad market volatility is elevated; reduce portfolio aggression and favor fast confirmation."
        elif b >= 0.64 and ch > 0.006:
            name = "RISK_ON"
            confidence = min(1.0, 0.55 + (b - 0.64) * 1.8 + ch * 8)
            risk = 1.00
            mult = {"TRADER": 1.04, "HUNTER": 1.10, "SWING": 1.10, "REVERSAL": 0.88}
            reason = "Positive breadth and median market appreciation favor continuation/momentum."
        elif b <= 0.36 and ch < -0.006:
            name = "RISK_OFF"
            confidence = min(1.0, 0.55 + (0.36 - b) * 1.8 + abs(ch) * 8)
            risk = 0.55
            mult = {"TRADER": 0.82, "HUNTER": 0.76, "SWING": 0.68, "REVERSAL": 1.04}
            reason = "Weak breadth and negative median performance call for defensive cash-heavy allocation."
        elif abs(ch) < 0.004 and 0.42 <= b <= 0.58:
            name = "CHOP"
            confidence = 0.70
            risk = 0.72
            mult = {"TRADER": 0.92, "HUNTER": 0.80, "SWING": 0.78, "REVERSAL": 1.08}
            reason = "Mixed breadth and flat median movement suggest a choppy/mean-reverting environment."
        else:
            name = "NEUTRAL"
            confidence = 0.55
            risk = 0.85
            mult = {"TRADER": 1.00, "HUNTER": 1.00, "SWING": 1.00, "REVERSAL": 1.00}
            reason = "No dominant portfolio-level regime signal."

        return MarketRegime(name, confidence, b, ch, rng, risk, mult, reason)


class MetaBrain:
    """Neutral arbiter: adjusts rankings using regime + forward model weights, never hard-wires a favorite brain."""

    def apply(self, opportunities: list[Opportunity], regime: MarketRegime,
              model_weights: dict[str, float]) -> list[Opportunity]:
        out = []
        for opp in opportunities:
            regime_mult = float(regime.brain_multipliers.get(opp.brain, 1.0))
            perf_mult = float(model_weights.get(opp.brain, 1.0))
            # Opportunity already includes its model weight from generation; this refreshes current promotion state.
            quality = max(0.0, min(1.0, (opp.score - 50.0) / 50.0))
            cost_penalty = min(0.08, max(0.0, opp.spread_bps / 1000.0))
            meta = opp.score * regime_mult * perf_mult * (1.0 - cost_penalty)
            # Keep extreme scores bounded so one feature cannot monopolize capital.
            meta = max(0.0, min(100.0, meta))
            opp.meta_score = float(meta)
            opp.regime_multiplier = regime_mult
            opp.context["meta_brain"] = {
                "regime": regime.name,
                "regime_multiplier": regime_mult,
                "model_promotion_weight": perf_mult,
                "meta_score": meta,
                "portfolio_risk_multiplier": regime.risk_multiplier,
                "quality": quality,
            }
            out.append(opp)
        return sorted(out, key=lambda x: x.meta_score, reverse=True)
