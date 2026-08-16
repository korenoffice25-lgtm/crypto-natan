from __future__ import annotations

from domain import MarketRegime


class RegimeRouter:
    """Classifies portfolio-level conditions and provides dynamic utilization guidance."""

    def classify(self, breadth_positive_pct: float, median_change_pct: float, median_range_pct: float) -> MarketRegime:
        b = float(breadth_positive_pct)
        ch = float(median_change_pct)
        rng = float(median_range_pct)

        if rng >= 0.13 and ch < -0.015:
            return MarketRegime(
                "PANIC", 0.88, b, ch, rng, 0.05, 0.18, 0.38,
                {"TRADER": 0.65, "HUNTER": 0.55, "SWING": 0.45, "REVERSAL": 0.92},
                "High volatility with broad downside: preserve capital and require exceptional confirmation.",
            )
        if b <= 0.34 and ch < -0.007:
            return MarketRegime(
                "RISK_OFF", min(0.95, 0.62 + (0.34-b)*1.3 + abs(ch)*6), b, ch, rng,
                0.15, 0.28, 0.58,
                {"TRADER": 0.78, "HUNTER": 0.70, "SWING": 0.60, "REVERSAL": 1.04},
                "Negative breadth and median returns favor defensive exposure.",
            )
        if b >= 0.72 and ch > 0.012 and rng >= 0.045:
            return MarketRegime(
                "EXPANSION", min(0.98, 0.68 + (b-0.72)*1.2 + ch*5), b, ch, rng,
                0.72, 0.80, 1.00,
                {"TRADER": 1.08, "HUNTER": 1.14, "SWING": 1.10, "REVERSAL": 0.82},
                "Strong breadth plus expansion favors momentum, breakout and trend continuation.",
            )
        if b >= 0.62 and ch > 0.005:
            return MarketRegime(
                "TREND", min(0.92, 0.60 + (b-0.62)*1.2 + ch*5), b, ch, rng,
                0.60, 0.72, 0.94,
                {"TRADER": 1.05, "HUNTER": 1.08, "SWING": 1.10, "REVERSAL": 0.88},
                "Positive breadth and trend conditions justify meaningful but controlled deployment.",
            )
        if abs(ch) < 0.0045 and 0.40 <= b <= 0.60:
            return MarketRegime(
                "CHOP", 0.74, b, ch, rng, 0.28, 0.42, 0.72,
                {"TRADER": 0.90, "HUNTER": 0.78, "SWING": 0.76, "REVERSAL": 1.08},
                "Mixed breadth and flat median returns suggest selective mean reversion and lower utilization.",
            )
        return MarketRegime(
            "NEUTRAL", 0.58, b, ch, rng, 0.44, 0.58, 0.85,
            {"TRADER": 1.00, "HUNTER": 1.00, "SWING": 1.00, "REVERSAL": 1.00},
            "No dominant regime signal; deploy only into ranked setups with positive utility.",
        )
