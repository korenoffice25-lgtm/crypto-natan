from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
from typing import Any

from opportunity_engine import Opportunity


@dataclass
class PortfolioAction:
    action: str  # BUY / SCALE / REDUCE_ROTATE / HOLD / REJECT
    symbol: str
    brain: str
    target_exposure_pct: float = 0.0
    fraction: float = 0.0
    reason: str = ""
    opportunity: Opportunity | None = None
    displaced_symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.opportunity is not None:
            d["opportunity"] = self.opportunity.to_dict()
        return d


def _age_seconds(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return 0.0


class PortfolioBrain:
    """Capital allocator. Cash competes with every opportunity and every open position."""

    def __init__(self, settings):
        self.cfg = settings

    def dynamic_target(self, opp: Opportunity, portfolio_regime_risk: float) -> float:
        score = opp.meta_score or opp.score
        quality = max(0.0, min(1.0, (score - 55.0) / 40.0))
        confidence = max(0.0, min(1.0, opp.confidence))
        liquidity = max(0.2, min(1.0, opp.candidate_risk_multiplier))
        # Start near minimum; exceptional opportunities can approach the single-position cap.
        target = self.cfg.min_position_pct + (self.cfg.max_position_pct - self.cfg.min_position_pct) * (
            0.52 * quality + 0.28 * confidence + 0.20 * liquidity
        )
        target *= max(0.35, min(1.05, portfolio_regime_risk))
        target = min(target, opp.target_exposure_pct, self.cfg.max_position_pct)
        return max(0.0, target)

    @staticmethod
    def weakest_position(open_positions: list[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = []
        for p in open_positions:
            meta = p.get("meta") or {}
            live_score = float(meta.get("live_meta_score", meta.get("score", 0.0)) or 0.0)
            age = _age_seconds(p.get("entry_time"))
            eligible.append((live_score, age, p))
        if not eligible:
            return None
        eligible.sort(key=lambda x: (x[0], -x[1]))
        return eligible[0][2]

    def plan_new_allocations(self, opportunities: list[Opportunity], open_positions: list[dict[str, Any]],
                             current_exposure_pct: float, portfolio_regime_risk: float) -> list[PortfolioAction]:
        """Rank opportunities without pretending target size equals actual approved size.

        The hard RiskGovernor recomputes real capacity after every fill. This is deliberate:
        a volatile 12% target may be risk-sized to 3%, so pre-reserving the full 12% would
        reproduce the V4 under-utilization bug.
        """
        actions: list[PortfolioAction] = []
        open_symbols = {p["symbol"] for p in open_positions}
        initial_capacity = max(0.0, self.cfg.max_total_exposure_pct - current_exposure_pct)
        positions = list(open_positions)

        for opp in sorted(opportunities, key=lambda x: x.meta_score or x.score, reverse=True):
            if opp.symbol in open_symbols:
                continue
            score = opp.meta_score or opp.score
            target = self.dynamic_target(opp, portfolio_regime_risk)
            if target < self.cfg.min_position_pct:
                actions.append(PortfolioAction("REJECT", opp.symbol, opp.brain, reason="Dynamic allocation below useful minimum", opportunity=opp))
                continue

            # If there was useful free capacity at planning time, let every ranked opportunity
            # compete. Execution-time risk sizing decides how many actually fit.
            if initial_capacity >= self.cfg.min_position_pct:
                actions.append(PortfolioAction("BUY", opp.symbol, opp.brain, target_exposure_pct=target,
                                               reason="Ranked opportunity competes for live risk-sized capacity", opportunity=opp))
                continue

            # Portfolio was already full before this planning pass: only a material score upgrade
            # can displace a weaker existing use of capital.
            weakest = self.weakest_position(positions)
            if weakest:
                meta = weakest.get("meta") or {}
                weak_score = float(meta.get("live_meta_score", meta.get("score", 0.0)) or 0.0)
                age = _age_seconds(weakest.get("entry_time"))
                if age >= self.cfg.rotation_min_age_seconds and score >= weak_score + self.cfg.rotation_score_advantage:
                    actions.append(PortfolioAction(
                        "REDUCE_ROTATE", weakest["symbol"], str(meta.get("brain") or "UNKNOWN"),
                        fraction=self.cfg.rotation_reduce_fraction,
                        reason=f"Capital rotation: {opp.symbol} score {score:.1f} exceeds {weakest['symbol']} {weak_score:.1f}",
                        opportunity=opp, displaced_symbol=weakest["symbol"],
                    ))
                    actions.append(PortfolioAction(
                        "BUY", opp.symbol, opp.brain, target_exposure_pct=target,
                        reason=f"Funded by rotation from {weakest['symbol']}", opportunity=opp,
                    ))
                    positions = [p for p in positions if p["symbol"] != weakest["symbol"]]
                    continue
            actions.append(PortfolioAction("REJECT", opp.symbol, opp.brain,
                                           reason="Portfolio is full and this opportunity does not justify rotation", opportunity=opp))
        return actions

    def should_scale(self, position: dict[str, Any], current_symbol_exposure_pct: float, available_portfolio_pct: float) -> tuple[bool, float, str]:
        meta = position.get("meta") or {}
        live_score = float(meta.get("live_meta_score", 0.0) or 0.0)
        return_pct = float(position.get("return_pct", 0.0) or 0.0) / 100.0
        scale_count = int(position.get("scale_count", 0) or 0)
        last_scale_ms = int(meta.get("last_scale_ms", 0) or 0)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if live_score < self.cfg.scale_score_threshold:
            return False, 0.0, "Live score below scale threshold"
        if return_pct < self.cfg.scale_min_profit_pct:
            return False, 0.0, "Winner has not earned a scale-in yet"
        if scale_count >= self.cfg.max_scales_per_position:
            return False, 0.0, "Maximum scale-ins reached"
        if last_scale_ms and now_ms - last_scale_ms < self.cfg.scale_cooldown_seconds * 1000:
            return False, 0.0, "Scale cooldown active"
        room_symbol = max(0.0, self.cfg.max_position_pct - current_symbol_exposure_pct)
        room = min(room_symbol, max(0.0, available_portfolio_pct))
        desired = min(room, max(self.cfg.min_position_pct, current_symbol_exposure_pct * self.cfg.scale_fraction))
        if desired < self.cfg.min_position_pct:
            return False, 0.0, "Insufficient room to scale"
        return True, desired, "Strong winner retained high edge; add within hard caps"
