"""
MH-FLOCKE — Directed Learning v0.1.0
========================================

The missing piece: Drive → Goal → Hypothesis → Test → Evaluate → Remember.

When the dog feels "fearful" (dead limb, bad gait), this module generates
concrete motor hypotheses, tests them by modifying CPG parameters, evaluates
the result via Gait Quality, and stores successful adaptations in memory.

This is NOT reinforcement learning. This is systematic self-experimentation:
1. Detect problem (Body Awareness + Gait Quality → Emotion)
2. Generate hypothesis ("maybe increase left amplitude by 20%?")
3. Test hypothesis (modify CPG for N steps)
4. Evaluate result (did Gait Quality improve?)
5. Remember outcome (store in adaptation memory)
6. Apply best known solution

Biology:
    - Motor exploration in infant development (Thelen & Smith 1994)
    - Trial-and-error learning in cerebellar patients (Martin et al. 1996)
    - Hypothesis testing in corvids (Taylor et al. 2009)
    - Motor babbling → schema formation (Piaget sensorimotor stage)

RPi cost: ~0.1ms per evaluation (runs every eval_interval steps, not every step).
Memory: ~1KB for adaptation history.

Author: MH-FLOCKE Level 15 v0.7.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

DIRECTED_LEARNING_VERSION = "v0.1.0"


@dataclass
class Adaptation:
    """A remembered motor adaptation and its outcome."""
    hypothesis: str                     # What we tried ("FL_amp+20%")
    parameter: str                      # Which parameter was changed
    value: float                        # What value was used
    gait_quality_before: float          # GQ before the test
    gait_quality_after: float           # GQ after the test
    upright_before: float               # Upright before
    upright_after: float                # Upright after
    improvement: float                  # GQ_after - GQ_before
    step_applied: int                   # When it was tested
    duration_steps: int                 # How long the test ran
    successful: bool                    # Did it improve?
    context: str = ''                   # e.g. "FR_dead", "4_legs"


@dataclass
class Hypothesis:
    """A motor hypothesis to test."""
    description: str
    parameter: str                      # e.g. 'leg_amplitude_FL'
    value: float                        # e.g. 1.4 (140%)
    priority: float = 0.5              # Higher = try first
    tested: bool = False
    result: Optional[Adaptation] = None


class DirectedLearning:
    """
    Systematic self-experimentation for motor adaptation.

    The dog detects a problem (via Emotion/Drive), generates hypotheses
    about what to change, tests them one at a time, and remembers results.

    Currently supports:
      - Per-leg CPG amplitude scaling (for leg-loss compensation)
      - CPG frequency adjustment (for gait quality)

    Usage:
        dl = DirectedLearning(cpg=spinal_cpg)
        # In training loop:
        if dl.should_evaluate(step):
            dl.evaluate_and_adapt(step, gait_quality, upright, body_awareness)
    """

    VERSION = DIRECTED_LEARNING_VERSION

    def __init__(self, eval_interval: int = 2000, test_duration: int = 1000,
                 max_memory: int = 50):
        """
        Args:
            eval_interval: Steps between evaluations.
            test_duration: How long to test each hypothesis (steps).
            max_memory: Max stored adaptations.
        """
        self.eval_interval = eval_interval
        self.test_duration = test_duration
        self.max_memory = max_memory

        # Adaptation memory: what worked, what didn't
        self.memory: List[Adaptation] = []

        # Current hypothesis being tested
        self._current_hypothesis: Optional[Hypothesis] = None
        self._test_start_step: int = 0
        self._baseline_gq: float = 0.0
        self._baseline_upright: float = 0.0

        # Hypothesis queue
        self._hypotheses: List[Hypothesis] = []

        # Current best known adaptation per context
        self._best_adaptations: Dict[str, Adaptation] = {}

        # Output: CPG modifications to apply
        self.leg_amplitude_scales: Dict[str, float] = {
            'FL': 1.0, 'FR': 1.0, 'RL': 1.0, 'RR': 1.0
        }
        self.freq_scale_modifier: float = 1.0
        self.amp_scale_modifier: float = 1.0

        # State
        self._step_count = 0
        self._adaptations_tried = 0
        self._adaptations_successful = 0

    def should_evaluate(self, step: int) -> bool:
        """Check if it's time to evaluate/adapt."""
        return step > 0 and step % self.eval_interval == 0

    def evaluate_and_adapt(self, step: int, gait_quality: float,
                           upright: float, dead_limbs: List[str],
                           degraded_limbs: List[str],
                           emotion: str = '',
                           obstacle_hits: int = 0,
                           obstacle_distance: float = -1.0,
                           cpg=None) -> Dict:
        """
        Main entry point. Called every eval_interval steps.

        1. If testing a hypothesis → evaluate result
        2. If problem detected → generate new hypotheses
        3. If hypotheses queued → start testing next one
        4. If no problems → apply best known adaptation

        Args:
            step: Current training step.
            gait_quality: Current GQ score (0-1).
            upright: Current upright value (-1 to 1).
            dead_limbs: List of dead limb names from Body Awareness.
            degraded_limbs: List of degraded limb names.
            emotion: Current dominant emotion string.
            obstacle_hits: Number of wall collisions in current episode.
            obstacle_distance: Current obstacle distance (-1 = none).
            cpg: Reference to CPG for applying amplitude changes.

        Returns:
            Dict with action taken and current state.
        """
        self._step_count = step
        result = {'action': 'none', 'hypothesis': None, 'adaptation': None}

        # --- Phase 1: Evaluate running hypothesis ---
        if self._current_hypothesis and (step - self._test_start_step) >= self.test_duration:
            adaptation = Adaptation(
                hypothesis=self._current_hypothesis.description,
                parameter=self._current_hypothesis.parameter,
                value=self._current_hypothesis.value,
                gait_quality_before=self._baseline_gq,
                gait_quality_after=gait_quality,
                upright_before=self._baseline_upright,
                upright_after=upright,
                improvement=gait_quality - self._baseline_gq,
                step_applied=self._test_start_step,
                duration_steps=step - self._test_start_step,
                successful=(gait_quality > self._baseline_gq + 0.02),
                context=self._get_context(dead_limbs),
            )
            self._current_hypothesis.tested = True
            self._current_hypothesis.result = adaptation
            self._remember(adaptation)
            self._adaptations_tried += 1

            if adaptation.successful:
                self._adaptations_successful += 1
                # Keep this adaptation active
                ctx = adaptation.context
                if ctx not in self._best_adaptations or \
                   adaptation.improvement > self._best_adaptations[ctx].improvement:
                    self._best_adaptations[ctx] = adaptation
                result['action'] = 'hypothesis_succeeded'
                print(f'  [LEARN] Hypothesis SUCCEEDED: {adaptation.hypothesis} '
                      f'(GQ {adaptation.gait_quality_before:.2f}→{adaptation.gait_quality_after:.2f}, '
                      f'+{adaptation.improvement:.3f})')
            else:
                # Revert to defaults or best known
                self._apply_best_known(dead_limbs, cpg)
                result['action'] = 'hypothesis_failed'
                print(f'  [LEARN] Hypothesis FAILED: {adaptation.hypothesis} '
                      f'(GQ {adaptation.gait_quality_before:.2f}→{adaptation.gait_quality_after:.2f})')

            result['adaptation'] = adaptation
            self._current_hypothesis = None

        # --- Phase 2: Generate hypotheses if problem detected ---
        if not self._current_hypothesis and not self._hypotheses:
            if dead_limbs or obstacle_hits > 0 or emotion in ('fearful', 'sad') or gait_quality < 0.4:
                self._generate_hypotheses(dead_limbs, degraded_limbs, gait_quality,
                                         obstacle_hits=obstacle_hits)

        # --- Phase 3: Start testing next hypothesis ---
        if not self._current_hypothesis and self._hypotheses:
            # Pick highest priority untested hypothesis
            untested = [h for h in self._hypotheses if not h.tested]
            if untested:
                untested.sort(key=lambda h: h.priority, reverse=True)
                hyp = untested[0]
                self._current_hypothesis = hyp
                self._test_start_step = step
                self._baseline_gq = gait_quality
                self._baseline_upright = upright
                self._apply_hypothesis(hyp, cpg)
                result['action'] = 'testing_hypothesis'
                result['hypothesis'] = hyp.description
                print(f'  [LEARN] Testing: {hyp.description} '
                      f'(baseline GQ={gait_quality:.2f}, upright={upright:.2f})')

        # --- Phase 4: No problems → apply best known ---
        if not self._current_hypothesis and not self._hypotheses:
            ctx = self._get_context(dead_limbs)
            if ctx in self._best_adaptations:
                best = self._best_adaptations[ctx]
                self._apply_adaptation(best, cpg)

        return result

    def _generate_hypotheses(self, dead_limbs: List[str],
                             degraded_limbs: List[str],
                             gait_quality: float,
                             obstacle_hits: int = 0) -> None:
        """Generate motor hypotheses based on current problems.
        
        v0.1.1: Hypotheses are generated through RANDOM PERTURBATION
        around the current best-known values, not from a fixed catalog.
        
        Biology: Motor babbling + reward-modulated selection.
        An infant doesn't have a catalog of "try 120%, try 140%".
        It makes small random variations in its motor output and
        keeps the ones that feel better (proprioceptive reward).
        
        The perturbation magnitude starts large (exploration) and
        shrinks as successful adaptations are found (exploitation).
        """
        self._hypotheses.clear()
        ctx = self._get_context(dead_limbs, obstacle_hits)

        # Perturbation magnitude: large when no successful adaptations,
        # smaller as we find what works (explore → exploit)
        n_successes = len([m for m in self.memory if m.context == ctx and m.successful])
        perturbation = max(0.05, 0.3 - n_successes * 0.05)  # 0.30 → 0.05

        if dead_limbs:
            # Leg-loss: perturb amplitude of healthy legs
            healthy = sorted({'FL', 'FR', 'RL', 'RR'} - set(dead_limbs))
            for leg in healthy:
                # Get current best value or default
                base = self.leg_amplitude_scales.get(leg, 1.0)
                # Random perturbation around current best
                delta = np.random.uniform(-perturbation, perturbation * 2)  # Bias toward increase
                value = round(max(0.5, min(2.0, base + delta)), 2)
                self._hypotheses.append(Hypothesis(
                    description=f'{leg} amplitude {value:.0%}',
                    parameter=f'leg_amplitude_{leg}',
                    value=value,
                    priority=0.5 + abs(delta),  # Bigger changes = try first
                ))

            # Also perturb frequency
            freq_base = self.freq_scale_modifier
            freq_delta = np.random.uniform(-perturbation, perturbation)
            freq_val = round(max(0.3, min(1.5, freq_base + freq_delta)), 2)
            self._hypotheses.append(Hypothesis(
                description=f'Frequency {freq_val:.0%}',
                parameter='freq_scale',
                value=freq_val,
                priority=0.4,
            ))

        elif obstacle_hits > 0:
            # Wall collision: perturb frequency and amplitude downward
            # Biology: an animal that bumps into things slows down
            freq_base = self.freq_scale_modifier
            freq_delta = np.random.uniform(-perturbation * 2, 0)  # Bias toward slower
            freq_val = round(max(0.3, freq_base + freq_delta), 2)
            self._hypotheses.append(Hypothesis(
                description=f'Slower approach {freq_val:.0%}',
                parameter='freq_scale',
                value=freq_val,
                priority=0.6,
            ))
            amp_base = self.amp_scale_modifier
            amp_delta = np.random.uniform(-perturbation * 2, 0)  # Bias toward smaller
            amp_val = round(max(0.2, amp_base + amp_delta), 2)
            self._hypotheses.append(Hypothesis(
                description=f'Smaller steps {amp_val:.0%}',
                parameter='amp_scale',
                value=amp_val,
                priority=0.5,
            ))

        elif gait_quality < 0.4:
            # General gait: random perturbation on frequency
            freq_base = self.freq_scale_modifier
            freq_delta = np.random.uniform(-perturbation, perturbation)
            freq_val = round(max(0.3, min(1.5, freq_base + freq_delta)), 2)
            self._hypotheses.append(Hypothesis(
                description=f'Frequency {freq_val:.0%}',
                parameter='freq_scale',
                value=freq_val,
                priority=0.5,
            ))

    def _apply_hypothesis(self, hyp: Hypothesis, cpg=None) -> None:
        """Apply a hypothesis by modifying CPG parameters."""
        if hyp.parameter.startswith('leg_amplitude_'):
            leg = hyp.parameter.replace('leg_amplitude_', '')
            self.leg_amplitude_scales[leg] = hyp.value
            if cpg and hasattr(cpg, '_leg_amplitude_scale'):
                leg_idx = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}.get(leg)
                if leg_idx is not None:
                    cpg._leg_amplitude_scale[leg_idx] = hyp.value
        elif hyp.parameter == 'freq_scale':
            self.freq_scale_modifier = hyp.value
        elif hyp.parameter == 'amp_scale':
            self.amp_scale_modifier = hyp.value

    def _apply_adaptation(self, adaptation: Adaptation, cpg=None) -> None:
        """Apply a known-good adaptation."""
        hyp = Hypothesis(
            description=adaptation.hypothesis,
            parameter=adaptation.parameter,
            value=adaptation.value,
        )
        self._apply_hypothesis(hyp, cpg)

    def _apply_best_known(self, dead_limbs: List[str], cpg=None) -> None:
        """Revert to best known adaptation or defaults."""
        ctx = self._get_context(dead_limbs)
        if ctx in self._best_adaptations:
            self._apply_adaptation(self._best_adaptations[ctx], cpg)
        else:
            # Reset to defaults
            self.leg_amplitude_scales = {
                'FL': 1.0, 'FR': 1.0, 'RL': 1.0, 'RR': 1.0
            }
            self.freq_scale_modifier = 1.0
            self.amp_scale_modifier = 1.0
            if cpg and hasattr(cpg, '_leg_amplitude_scale'):
                cpg._leg_amplitude_scale = [1.0, 1.0, 1.0, 1.0]

    def _remember(self, adaptation: Adaptation) -> None:
        """Store adaptation in memory."""
        self.memory.append(adaptation)
        if len(self.memory) > self.max_memory:
            # Remove oldest unsuccessful adaptation
            for i, m in enumerate(self.memory):
                if not m.successful:
                    self.memory.pop(i)
                    break
            else:
                self.memory.pop(0)

    def _get_context(self, dead_limbs: List[str], obstacle_hits: int = 0) -> str:
        """Context string for grouping adaptations."""
        if dead_limbs:
            return 'dead_' + '_'.join(sorted(dead_limbs))
        if obstacle_hits > 0:
            return 'wall_collision'
        return '4_legs'

    def get_successful_adaptations(self, context: str = None) -> List[Adaptation]:
        """Get all successful adaptations, optionally filtered by context."""
        return [m for m in self.memory
                if m.successful and (context is None or m.context == context)]

    def stats(self) -> Dict:
        """Compact stats for FLOG logging."""
        return {
            'dl_tried': self._adaptations_tried,
            'dl_successful': self._adaptations_successful,
            'dl_success_rate': (self._adaptations_successful /
                                max(self._adaptations_tried, 1)),
            'dl_memory_size': len(self.memory),
            'dl_testing': self._current_hypothesis.description if self._current_hypothesis else '',
            'dl_best_known': len(self._best_adaptations),
            'dl_freq_mod': self.freq_scale_modifier,
        }

    def __repr__(self) -> str:
        testing = self._current_hypothesis.description if self._current_hypothesis else 'idle'
        return (f'DirectedLearning(tried={self._adaptations_tried}, '
                f'success={self._adaptations_successful}, '
                f'testing="{testing}")')
