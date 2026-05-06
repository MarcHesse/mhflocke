"""
MH-FLOCKE — StrategyAdapter (Meta-Learning Loop Phase B)
==========================================================
Converts EpisodeAnalyzer insights into parameter adjustments.

Design reference: docs/DESIGN_AUTONOMOUS_LOOP.md, Verbindung 2
Biology: Prefrontal cortex → basal ganglia strategy selection (Doya 2002)

Author: Marc Hesse
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterAdjustment:
    """A single parameter adjustment with reason."""
    parameter: str
    old_value: float
    new_value: float
    reason: str
    confidence: float
    step: int = 0


class StrategyAdapter:
    """Converts insights into parameter adjustments.

    Phase B of the Meta-Learning Loop (DESIGN_AUTONOMOUS_LOOP.md).
    Conservative: only adjusts when confidence > threshold,
    changes are small and bounded.
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 max_adjustment_pct: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.max_pct = max_adjustment_pct

        self.params: Dict[str, float] = {
            'rt_run_duration': 40.0,
            'rt_tumble_duration': 12.0,
            'pid_kp_scale': 1.0,
            'exploration_bias': 0.0,
        }

        self.bounds: Dict[str, tuple] = {
            'rt_run_duration': (20.0, 120.0),
            'rt_tumble_duration': (6.0, 24.0),
            'pid_kp_scale': (0.5, 2.0),
            'exploration_bias': (-1.0, 1.0),
        }

        self.adjustments: List[ParameterAdjustment] = []
        self._pending_adjustments: List[ParameterAdjustment] = []

        self._insight_to_param = {
            'heading_error': {
                'lower': ('pid_kp_scale', +0.1),
                'higher': ('pid_kp_scale', -0.05),
            },
            'velocity': {
                'higher': ('rt_run_duration', +5.0),
                'lower': ('rt_run_duration', -3.0),
            },
            'cumulative_turn': {
                'higher': ('exploration_bias', +0.1),
                'lower': ('exploration_bias', -0.1),
            },
            'steps_since_last': {
                'lower': ('rt_run_duration', -3.0),
                'higher': ('rt_run_duration', +5.0),
            },
            'steering_offset': {
                'higher': ('pid_kp_scale', +0.1),
                'lower': ('pid_kp_scale', -0.05),
            },
            'gait_quality': {
                'higher': ('rt_run_duration', +3.0),
                'lower': ('rt_tumble_duration', +2.0),
            },
        }

    def process_insights(self, insights: list, step: int = 0):
        """Process insights from EpisodeAnalyzer and adjust parameters."""
        for insight in insights:
            if insight.confidence < self.confidence_threshold:
                continue

            factor = insight.factor
            direction = 'lower' if 'lower' in insight.description else 'higher'

            mapping = self._insight_to_param.get(factor, {}).get(direction)
            if mapping is None:
                continue

            param_name, delta = mapping
            scaled_delta = delta * min(1.0, insight.confidence)

            old_val = self.params[param_name]
            lo, hi = self.bounds[param_name]
            new_val = max(lo, min(hi, old_val + scaled_delta))

            if abs(old_val) > 0.01:
                pct_change = abs(new_val - old_val) / abs(old_val)
                if pct_change > self.max_pct:
                    max_delta = abs(old_val) * self.max_pct
                    new_val = old_val + max_delta * (1.0 if scaled_delta > 0 else -1.0)
                    new_val = max(lo, min(hi, new_val))

            if abs(new_val - old_val) < 0.001:
                continue

            self.params[param_name] = new_val

            adj = ParameterAdjustment(
                parameter=param_name, old_value=old_val,
                new_value=new_val,
                reason=f"{factor} {direction} (conf={insight.confidence:.2f})",
                confidence=insight.confidence, step=step,
            )
            self.adjustments.append(adj)
            self._pending_adjustments.append(adj)
            logger.info(
                f"[STRATEGY] {param_name}: {old_val:.1f} → {new_val:.1f} "
                f"({factor} {direction}, conf={insight.confidence:.2f})"
            )

    def get_rt_run_duration(self) -> int:
        return int(self.params['rt_run_duration'])

    def get_rt_tumble_duration(self) -> int:
        return int(self.params['rt_tumble_duration'])

    def get_pid_kp_scale(self) -> float:
        return self.params['pid_kp_scale']

    def get_exploration_bias(self) -> float:
        return self.params['exploration_bias']

    def get_pending_adjustments(self) -> List[ParameterAdjustment]:
        result = list(self._pending_adjustments)
        self._pending_adjustments.clear()
        return result

    def stats(self) -> Dict[str, Any]:
        return {
            'strategy_rt_run': self.params['rt_run_duration'],
            'strategy_rt_tumble': self.params['rt_tumble_duration'],
            'strategy_pid_scale': self.params['pid_kp_scale'],
            'strategy_exploration': self.params['exploration_bias'],
            'strategy_adjustments': len(self.adjustments),
        }

    def save_state(self) -> Dict[str, Any]:
        return {
            'params': dict(self.params),
            'adjustments': [
                {
                    'parameter': a.parameter, 'old_value': a.old_value,
                    'new_value': a.new_value, 'reason': a.reason,
                    'confidence': a.confidence, 'step': a.step,
                }
                for a in self.adjustments[-20:]
            ],
        }

    def load_state(self, state: Dict[str, Any]):
        if 'params' in state:
            for k, v in state['params'].items():
                if k in self.params:
                    self.params[k] = v
        if 'adjustments' in state:
            self.adjustments = [
                ParameterAdjustment(**a) for a in state['adjustments']
            ]
        logger.info(
            f"[StrategyAdapter] Restored: rt_run={self.params['rt_run_duration']:.0f} "
            f"pid_scale={self.params['pid_kp_scale']:.2f} "
            f"({len(self.adjustments)} prior adjustments)"
        )
