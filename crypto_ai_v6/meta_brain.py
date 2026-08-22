from __future__ import annotations

from domain import Signal, MarketRegime


class MetaBrain:
    """Neutral score normalizer. It ranks brains; it never places orders."""

    def apply(self, signals: list[Signal], regime: MarketRegime, performance_weights: dict[str, float] | None = None) -> list[Signal]:
        weights = performance_weights or {}
        for s in signals:
            regime_mult = float(regime.brain_multipliers.get(s.brain, 1.0))
            perf_mult = max(0.82, min(1.18, float(weights.get(s.brain, 1.0))))
            spread_penalty = max(0.88, 1.0 - max(0.0, s.spread_bps-3.0)/250.0)
            edge_quality = max(0.55, min(1.12, 0.72 + max(s.expected_edge_bps, 0.0)/450.0))
            meta = s.score * regime_mult * perf_mult * spread_penalty * edge_quality
            s.meta_score = max(0.0, min(100.0, meta))
            s.regime_multiplier = regime_mult
            # Utility measures how attractive a dollar of capital is, not just signal score.
            s.utility = max(0.0, min(100.0,
                0.55*s.meta_score + 24.0*s.confidence + 12.0*s.candidate_risk_multiplier + min(9.0, max(0.0, s.expected_edge_bps)/35.0)
            ))
            s.context["meta"] = {
                "regime": regime.name,
                "regime_multiplier": regime_mult,
                "performance_weight": perf_mult,
                "spread_penalty": spread_penalty,
                "edge_quality": edge_quality,
                "utility": s.utility,
            }
        return sorted(signals, key=lambda x: (x.utility, x.meta_score), reverse=True)
