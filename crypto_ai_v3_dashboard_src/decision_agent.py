from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from return_model import HorizonPrediction
from regime_model import RegimeReading


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    EXIT = "EXIT"
    DO_NOTHING = "DO_NOTHING"


@dataclass
class Decision:
    action: Action
    confidence: float
    expected_edge_bps: float
    uncertainty_bps: float
    target_exposure_pct: float
    reason: str


class DecisionAgent:
    """Converts learned return forecasts + current market microstructure into action.

    No EMA crossover / RSI threshold / named trend strategy is used here.
    The agent acts only when the learned edge is large enough relative to model
    error, estimated trading costs and current microstructure quality.
    """

    def __init__(
        self,
        round_trip_cost_buffer_bps: float = 18.0,
        min_confidence: float = 0.58,
        exit_confidence: float = 0.47,
    ):
        self.cost_bps = round_trip_cost_buffer_bps
        self.min_confidence = min_confidence
        self.exit_confidence = exit_confidence

    @staticmethod
    def _weighted_forecast(predictions: list[HorizonPrediction]) -> tuple[float, float]:
        # More reliable models get more weight; horizon normalization avoids
        # letting a long horizon dominate only because its raw return is larger.
        values = []
        weights = []
        for p in predictions:
            per_step = p.expected_return / max(p.horizon, 1)
            err_per_step = max(p.validation_mae / max(p.horizon, 1), 1e-7)
            weight = 1.0 / err_per_step
            values.append(per_step)
            weights.append(weight)

        values = np.asarray(values)
        weights = np.asarray(weights)
        mean = float(np.average(values, weights=weights))
        disagreement = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        model_error = float(np.average(
            [p.validation_mae / max(p.horizon, 1) for p in predictions],
            weights=weights,
        ))
        uncertainty = disagreement + model_error
        return mean, uncertainty

    def decide(
        self,
        predictions: list[HorizonPrediction],
        regime: RegimeReading,
        spread_bps: float,
        orderbook_imbalance: float = 0.0,
        trade_flow_imbalance: float = 0.0,
        has_position: bool = False,
    ) -> Decision:
        mean_return, uncertainty = self._weighted_forecast(predictions)

        raw_edge_bps = mean_return * 10_000
        uncertainty_bps = uncertainty * 10_000

        # Microstructure is used as corroborating evidence, not as a fixed buy rule.
        # This bounded adjustment can only modestly change the learned forecast.
        micro_confirmation = math.tanh(
            0.55 * orderbook_imbalance + 0.45 * trade_flow_imbalance
        )
        adjusted_edge_bps = raw_edge_bps + 4.0 * micro_confirmation

        effective_cost = self.cost_bps + max(spread_bps, 0.0)
        signal_to_noise = adjusted_edge_bps / max(uncertainty_bps + effective_cost, 1e-6)
        confidence = float(1 / (1 + math.exp(-signal_to_noise)))

        # High regime uncertainty reduces willingness to deploy capital.
        confidence *= 0.75 + 0.25 * regime.confidence
        confidence = max(0.0, min(confidence, 1.0))

        net_edge = adjusted_edge_bps - effective_cost

        if has_position:
            if net_edge <= 0 or confidence < self.exit_confidence:
                return Decision(
                    Action.EXIT, confidence, net_edge, uncertainty_bps, 0.0,
                    "Learned edge no longer compensates for costs/uncertainty",
                )
            exposure = min(0.20, max(0.03, net_edge / 250.0))
            return Decision(
                Action.HOLD, confidence, net_edge, uncertainty_bps, exposure,
                "Existing position still has positive learned edge",
            )

        if net_edge > 0 and confidence >= self.min_confidence:
            exposure = min(0.20, max(0.03, net_edge / 250.0))
            return Decision(
                Action.BUY, confidence, net_edge, uncertainty_bps, exposure,
                "Forecasted edge exceeds costs and uncertainty threshold",
            )

        return Decision(
            Action.DO_NOTHING, confidence, net_edge, uncertainty_bps, 0.0,
            "No sufficiently strong edge",
        )
