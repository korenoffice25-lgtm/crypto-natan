from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from domain import Signal, MarketRegime, PortfolioAction


def _age(entry_time: str) -> float:
    try:
        d=datetime.fromisoformat(entry_time.replace("Z","+00:00"))
        return max(0.0,(datetime.now(timezone.utc)-d).total_seconds())
    except Exception:
        return 0.0


class PortfolioBrain:
    """V6 capital allocator: every dollar competes between open positions, new signals and cash."""

    def __init__(self, settings):
        self.cfg=settings

    def target_size(self, s: Signal, regime: MarketRegime) -> float:
        q=max(0.0,min(1.0,(s.meta_score-58)/34))
        conf=max(0.0,min(1.0,s.confidence))
        liq=max(0.3,min(1.0,s.candidate_risk_multiplier))
        utility=max(0.0,min(1.0,(s.utility-55)/40))
        target=self.cfg.min_position_pct+(self.cfg.max_position_pct-self.cfg.min_position_pct)*(
            0.34*q+0.24*conf+0.20*liq+0.22*utility)
        # Signal's own recommended target remains a cap, but exceptional signals can earn modest headroom.
        signal_cap=s.target_exposure_pct*(1.10 if s.meta_score>=self.cfg.exceptional_score else 1.0)
        return max(0.0,min(target,signal_cap,self.cfg.max_position_pct))

    @staticmethod
    def live_score(p: dict[str,Any]) -> float:
        m=p.get("meta") or {}
        return float(m.get("live_meta_score",m.get("entry_meta_score",m.get("score",0))) or 0)

    def plan(self, signals: list[Signal], positions: list[dict[str,Any]], current_exposure: float,
             regime: MarketRegime) -> list[PortfolioAction]:
        actions: list[PortfolioAction]=[]
        open_symbols={p["symbol"] for p in positions}
        scheduled_displaced: set[str] = set()
        target_util=min(regime.target_utilization_pct,self.cfg.absolute_max_exposure_pct)
        max_util=min(regime.max_utilization_pct,self.cfg.absolute_max_exposure_pct)
        free_to_target=max(0.0,target_util-current_exposure)
        free_to_max=max(0.0,max_util-current_exposure)

        # Risk-off de-risking is explicit. It does not wait for each individual brain to notice.
        if current_exposure > max_util + self.cfg.utilization_tolerance_pct:
            excess=current_exposure-max_util
            weak=sorted(positions,key=self.live_score)
            for p in weak:
                if excess <= 0: break
                pexp=float(p.get("market_value",0))/max(float(p.get("portfolio_equity",1)),1e-9) if p.get("portfolio_equity") else 0.0
                fraction=min(0.55,max(0.20,excess/max(pexp,0.001)))
                actions.append(PortfolioAction("REDUCE",p["symbol"],str(p.get("brain") or (p.get("meta") or {}).get("brain") or ""),fraction=fraction,
                                               reason=f"Regime {regime.name} exposure compression"))
                scheduled_displaced.add(p["symbol"])
                excess-=pexp*fraction

        ranked=[s for s in signals if s.symbol not in open_symbols]
        for s in ranked:
            target=self.target_size(s,regime)
            if target < self.cfg.min_position_pct:
                continue
            # New capital is allowed up to regime max, but target-utilization gets first preference.
            if free_to_max >= self.cfg.min_position_pct:
                # Do not pre-reserve the requested target here. The hard RiskEngine may size
                # a 12% request to 3%; reserving 12% would recreate V5 under-utilization.
                # Let every ranked signal compete and recompute capacity after each actual fill.
                desired=min(target,free_to_max)
                if desired >= self.cfg.min_position_pct:
                    reason="High-ranked opportunity earns available portfolio capacity"
                    if free_to_target > 0:
                        reason="Deploying toward regime-aware target utilization into qualified edge"
                    actions.append(PortfolioAction("OPEN",s.symbol,s.brain,desired,reason=reason,signal=s))
                    continue

            # Full portfolio: rotate only when opportunity cost is clearly favorable.
            eligible=[p for p in positions if p["symbol"] not in scheduled_displaced and _age(p.get("entry_time",""))>=self.cfg.rotation_min_age_seconds]
            if eligible:
                weakest=min(eligible,key=self.live_score)
                weak_score=self.live_score(weakest)
                if s.meta_score >= weak_score+self.cfg.rotation_score_advantage:
                    actions.append(PortfolioAction("ROTATE",weakest["symbol"],str(weakest.get("brain") or (weakest.get("meta") or {}).get("brain") or ""),
                                                   fraction=self.cfg.rotation_reduce_fraction,
                                                   reason=f"Opportunity cost: {s.symbol} {s.meta_score:.1f} > {weakest['symbol']} {weak_score:.1f}",
                                                   signal=s,displaced_symbol=weakest["symbol"]))
                    actions.append(PortfolioAction("OPEN",s.symbol,s.brain,target,
                                                   reason=f"Funded by rotation from {weakest['symbol']}",signal=s))
                    scheduled_displaced.add(weakest["symbol"])
                    continue

        if not any(a.action in {"OPEN","ROTATE","REDUCE"} for a in actions):
            actions.append(PortfolioAction("CASH",reason="No portfolio change has positive risk-adjusted opportunity cost"))
        return actions

    def scale_decision(self, p: dict[str,Any], symbol_exposure: float, free_capacity: float) -> tuple[bool,float,str]:
        m=p.get("meta") or {}
        score=float(m.get("live_meta_score",0) or 0)
        ret=float(p.get("return_pct",0) or 0)/100
        if score < self.cfg.scale_score_threshold:
            return False,0.0,"Live score below add threshold"
        if ret < self.cfg.scale_min_profit_pct:
            return False,0.0,"Position has not earned an add"
        if int(p.get("scale_count",0) or 0)>=self.cfg.max_scales_per_position:
            return False,0.0,"Maximum adds reached"
        last=int(m.get("last_scale_ms",0) or 0)
        now=int(datetime.now(timezone.utc).timestamp()*1000)
        if last and now-last<self.cfg.scale_cooldown_seconds*1000:
            return False,0.0,"Add cooldown active"
        room=min(max(0.0,self.cfg.max_position_pct-symbol_exposure),max(0.0,free_capacity))
        desired=min(room,max(self.cfg.min_position_pct,symbol_exposure*self.cfg.scale_fraction))
        if desired<self.cfg.min_position_pct:
            return False,0.0,"Insufficient capacity"
        return True,desired,"Winner retains high edge; scale within hard caps"
