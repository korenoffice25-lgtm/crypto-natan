from __future__ import annotations


class PerformanceCalibrator:
    """Bounded promotion weights from closed paper trades. Never changes code or hard risk rules."""

    def __init__(self, storage, min_trades: int = 12):
        self.storage = storage
        self.min_trades = min_trades

    def weights(self) -> dict[str, float]:
        perf = self.storage.performance_summary().get("by_brain", {})
        out = {"TRADER":1.0,"HUNTER":1.0,"SWING":1.0,"REVERSAL":1.0}
        for brain, x in perf.items():
            n = int(x.get("trades",0) or 0)
            if n < self.min_trades:
                continue
            pnl = float(x.get("pnl",0) or 0)
            wr = float(x.get("win_rate_pct",0) or 0)/100.0
            # Conservative bounded adjustment; no self-modifying behavior.
            edge = (wr-0.50)*0.22 + max(-0.08,min(0.08,pnl/5000.0))
            out[brain] = max(0.86,min(1.14,1.0+edge))
        return out
