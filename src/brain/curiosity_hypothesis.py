"""
MH-FLOCKE — CuriosityExplorer + HypothesisGenerator
=====================================================
Phase C: World Model PE → Exploration Drive
Phase D: Episodic Memory correlations → testable hypotheses

Design reference: docs/DESIGN_AUTONOMOUS_LOOP.md, Verbindungen 3+4
Biology:
  C: Active Inference / Free Energy (Friston 2010)
  D: Prefrontal hypothesis testing (Doya 2002, Wang et al. 2018)

Author: Marc Hesse
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CuriosityExplorer:
    """Phase C: Modulates exploration based on World Model prediction error."""

    def __init__(self, pe_ema_decay: float = 0.995,
                 novelty_threshold: float = 0.15,
                 familiar_threshold: float = 0.05):
        self.pe_ema_decay = pe_ema_decay
        self.novelty_threshold = novelty_threshold
        self.familiar_threshold = familiar_threshold
        self._pe_ema = 0.0
        self._exploration_drive = 0.5
        self._grid_coverage = 0.0
        self._pe_history: List[float] = []
        self._max_history = 100

    def update(self, prediction_error: float, grid_coverage: float = 0.0):
        self._pe_ema = self._pe_ema * self.pe_ema_decay + prediction_error * (1 - self.pe_ema_decay)
        self._pe_history.append(prediction_error)
        if len(self._pe_history) > self._max_history:
            self._pe_history = self._pe_history[-self._max_history:]
        self._grid_coverage = grid_coverage

        if self._pe_ema > self.novelty_threshold:
            self._exploration_drive = min(1.0, 0.5 + (self._pe_ema - self.novelty_threshold) * 3.0)
        elif self._pe_ema < self.familiar_threshold:
            self._exploration_drive = max(0.0, 0.5 - (self.familiar_threshold - self._pe_ema) * 5.0)
        else:
            self._exploration_drive = self._exploration_drive * 0.99 + 0.5 * 0.01

        if grid_coverage < 0.1:
            self._exploration_drive = min(1.0, self._exploration_drive + 0.1)

    def get_rt_modulation(self) -> Dict[str, float]:
        drive = self._exploration_drive
        return {
            'rt_run_scale': 1.0 + (0.5 - drive) * 0.6,
            'rt_tumble_scale': 1.0 + (drive - 0.5) * 0.4,
        }

    def get_exploration_drive(self) -> float:
        return self._exploration_drive

    def stats(self) -> Dict[str, Any]:
        return {
            'curiosity_pe_ema': self._pe_ema,
            'curiosity_drive': self._exploration_drive,
            'curiosity_grid_coverage': self._grid_coverage,
        }

    def save_state(self) -> Dict[str, Any]:
        return {
            'pe_ema': self._pe_ema,
            'exploration_drive': self._exploration_drive,
            'pe_history': self._pe_history[-50:],
        }

    def load_state(self, state: Dict[str, Any]):
        self._pe_ema = state.get('pe_ema', 0.0)
        self._exploration_drive = state.get('exploration_drive', 0.5)
        self._pe_history = state.get('pe_history', [])
        logger.info(f"[CuriosityExplorer] Restored: drive={self._exploration_drive:.2f}")


@dataclass
class AutoHypothesis:
    """An autonomously generated hypothesis to test."""
    description: str
    parameter: str
    value: float
    source_insight: str
    confidence: float
    tested: bool = False
    result_gq_delta: float = 0.0
    confirmed: bool = False
    step_generated: int = 0


class HypothesisGenerator:
    """Phase D: Generates testable hypotheses from EpisodeAnalyzer insights."""

    def __init__(self, max_hypotheses: int = 10,
                 min_insight_confidence: float = 0.5):
        self.max_hypotheses = max_hypotheses
        self.min_confidence = min_insight_confidence
        self.hypotheses: List[AutoHypothesis] = []
        self._pending: List[AutoHypothesis] = []

        self._insight_to_hypothesis = {
            'velocity': {
                'higher': ('frequency', 1.12, "Increase CPG frequency to 112%"),
                'lower': ('frequency', 0.90, "Decrease CPG frequency to 90%"),
            },
            'gait_quality': {
                'higher': ('frequency', 1.05, "Slightly increase frequency for better gait"),
                'lower': ('amplitude', 0.90, "Reduce amplitude for stability"),
            },
            'heading_error': {
                'lower': ('frequency', 0.95, "Slower gait for better steering precision"),
            },
            'cumulative_turn': {
                'higher': ('amplitude', 1.10, "Higher amplitude for wider turns"),
                'lower': ('frequency', 1.08, "Faster straight-line walking"),
            },
            'steering_offset': {
                'higher': ('amplitude', 0.95, "Reduce amplitude for more steering range"),
            },
        }

    def generate_from_insights(self, insights: list, step: int = 0) -> List[AutoHypothesis]:
        new_hyps = []
        for insight in insights:
            if insight.confidence < self.min_confidence:
                continue
            factor = insight.factor
            direction = 'lower' if 'lower' in insight.description else 'higher'
            mapping = self._insight_to_hypothesis.get(factor, {}).get(direction)
            if mapping is None:
                continue
            param, value, desc = mapping
            existing = [h for h in self.hypotheses
                       if h.parameter == param and abs(h.value - value) < 0.01]
            if existing:
                continue
            hyp = AutoHypothesis(
                description=desc, parameter=param, value=value,
                source_insight=f"{factor} {direction}",
                confidence=insight.confidence, step_generated=step,
            )
            self.hypotheses.append(hyp)
            self._pending.append(hyp)
            new_hyps.append(hyp)
            logger.info(f"[HYPOTHESIS] Generated: {desc} (from {factor} {direction})")

        if len(self.hypotheses) > self.max_hypotheses:
            untested = [h for h in self.hypotheses if not h.tested]
            tested = sorted([h for h in self.hypotheses if h.tested],
                          key=lambda h: h.step_generated, reverse=True)
            self.hypotheses = untested + tested[:self.max_hypotheses - len(untested)]
        return new_hyps

    def get_next_untested(self) -> Optional[AutoHypothesis]:
        untested = [h for h in self.hypotheses if not h.tested]
        if not untested:
            return None
        return max(untested, key=lambda h: h.confidence)

    def record_result(self, hypothesis: AutoHypothesis,
                      gq_delta: float, confirmed: bool):
        hypothesis.tested = True
        hypothesis.result_gq_delta = gq_delta
        hypothesis.confirmed = confirmed
        logger.info(f"[HYPOTHESIS] {'CONFIRMED' if confirmed else 'REJECTED'}: {hypothesis.description} (GQ delta={gq_delta:+.3f})")

    def get_pending(self) -> List[AutoHypothesis]:
        result = list(self._pending)
        self._pending.clear()
        return result

    def stats(self) -> Dict[str, Any]:
        tested = [h for h in self.hypotheses if h.tested]
        confirmed = [h for h in tested if h.confirmed]
        return {
            'hypothesis_total': len(self.hypotheses),
            'hypothesis_untested': len(self.hypotheses) - len(tested),
            'hypothesis_confirmed': len(confirmed),
            'hypothesis_rejected': len(tested) - len(confirmed),
            'hypothesis_confirmation_rate': len(confirmed) / len(tested) if tested else 0.0,
        }

    def save_state(self) -> Dict[str, Any]:
        return {
            'hypotheses': [
                {
                    'description': h.description, 'parameter': h.parameter,
                    'value': h.value, 'source_insight': h.source_insight,
                    'confidence': h.confidence, 'tested': h.tested,
                    'result_gq_delta': h.result_gq_delta,
                    'confirmed': h.confirmed, 'step_generated': h.step_generated,
                }
                for h in self.hypotheses[-10:]
            ],
        }

    def load_state(self, state: Dict[str, Any]):
        if 'hypotheses' in state:
            self.hypotheses = [AutoHypothesis(**h) for h in state['hypotheses']]
        logger.info(f"[HypothesisGenerator] Restored {len(self.hypotheses)} hypotheses")
