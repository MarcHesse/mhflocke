"""
MH-FLOCKE — Evolved Plasticity v0.4.1
========================================
Genome-encoded plasticity rules for SNN learning.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PlasticityGenome:
    """Evolvierbare Parameter einer Plastizitätsregel."""
    # STDP Timing
    pre_factor: float = 1.0         # A+ (LTP bei pre-before-post)
    post_factor: float = -0.5       # A- (LTD bei post-before-pre)
    tau_plus: float = 20.0          # ms, Zeitkonstante LTP
    tau_minus: float = 20.0         # ms, Zeitkonstante LTD

    # Reward Modulation
    reward_factor: float = 1.0      # Wie stark Reward die Änderung skaliert
    reward_baseline: float = 0.0    # Subtrahiert von Reward

    # Eligibility Trace
    eligibility_tau: float = 100.0  # ms

    # Hebbian vs Anti-Hebbian
    hebbian_weight: float = 0.1     # Reines Hebb-Learning (ohne Reward)
    anti_hebbian_weight: float = 0.0

    # Homeostatic Component
    homeostatic_target_rate: float = 0.05
    homeostatic_strength: float = 0.01

    # Weight Bounds
    w_min: float = -2.0
    w_max: float = 2.0
    decay_rate: float = 0.0001

    def copy(self) -> 'PlasticityGenome':
        """Tiefe Kopie."""
        return PlasticityGenome(**self.__dict__)


class PlasticityMutator:
    """Mutiert Plastizitätsregeln."""

    PARAM_RANGES = {
        'pre_factor': (0.0, 3.0),
        'post_factor': (-3.0, 0.0),
        'tau_plus': (5.0, 100.0),
        'tau_minus': (5.0, 100.0),
        'reward_factor': (0.0, 5.0),
        'reward_baseline': (-1.0, 1.0),
        'eligibility_tau': (10.0, 500.0),
        'hebbian_weight': (0.0, 1.0),
        'anti_hebbian_weight': (0.0, 0.5),
        'homeostatic_target_rate': (0.01, 0.2),
        'homeostatic_strength': (0.001, 0.1),
        'decay_rate': (0.0, 0.01),
    }

    def mutate(self, pg: PlasticityGenome,
               mutation_rate: float = 0.3,
               mutation_strength: float = 0.2) -> PlasticityGenome:
        """
        Mutiere Plastizitätsparameter.

        Args:
            pg: Zu mutierende PlasticityGenome
            mutation_rate: Wahrscheinlichkeit pro Parameter
            mutation_strength: Stärke der Mutation (relativ zu Range)

        Returns:
            Mutierte Kopie
        """
        child = pg.copy()
        for param, (lo, hi) in self.PARAM_RANGES.items():
            if np.random.random() < mutation_rate:
                val = getattr(child, param)
                delta = np.random.normal(0, mutation_strength * (hi - lo))
                val = np.clip(val + delta, lo, hi)
                setattr(child, param, float(val))
        return child

    def crossover(self, parent1: PlasticityGenome,
                  parent2: PlasticityGenome) -> PlasticityGenome:
        """Uniform Crossover der Plastizitätsparameter."""
        child = parent1.copy()
        for param in self.PARAM_RANGES:
            if np.random.random() < 0.5:
                setattr(child, param, getattr(parent2, param))
        return child


class EvolvedPlasticityRule:
    """
    Wendet eine evolvierte Plastizitätsregel auf ein SNN an.

    Nutzt SNNController-API:
    - snn.apply_rstdp(reward_signal: float)
    - snn._weight_values: Tensor
    - snn.config.stdp_lr: float
    """

    def __init__(self, genome: PlasticityGenome):
        """
        Args:
            genome: PlasticityGenome mit evolvierten Parametern
        """
        self.genome = genome
        self._reward_baseline = genome.reward_baseline

    def apply(self, snn, reward: float = 0.0):
        """
        Wende evolvierte Plastizität auf SNN an.

        Args:
            snn: SNNController-Instanz
            reward: Reward-Signal (float)
        """
        g = self.genome

        # 1. Reward-Modulation mit evolviertem Factor
        effective_reward = (reward - self._reward_baseline) * g.reward_factor
        self._reward_baseline = 0.99 * self._reward_baseline + 0.01 * reward

        # R-STDP mit moduliertem Reward
        if hasattr(snn, 'apply_rstdp'):
            # Temporarily adjust stdp_lr
            original_lr = snn.config.stdp_lr
            snn.config.stdp_lr = original_lr * abs(g.pre_factor)
            snn.apply_rstdp(reward_signal=effective_reward)
            snn.config.stdp_lr = original_lr

        # 2. Weight Decay
        if g.decay_rate > 0 and hasattr(snn, '_weight_values') and snn._weight_values is not None:
            snn._weight_values = snn._weight_values * (1.0 - g.decay_rate)

        # 3. Weight Bounds
        if hasattr(snn, '_weight_values') and snn._weight_values is not None:
            snn._weight_values = snn._weight_values.clamp(g.w_min, g.w_max)

    def to_dict(self) -> dict:
        """Für Logging/Dashboard."""
        return {
            'pre_factor': self.genome.pre_factor,
            'post_factor': self.genome.post_factor,
            'reward_factor': self.genome.reward_factor,
            'hebbian_weight': self.genome.hebbian_weight,
            'decay_rate': self.genome.decay_rate,
        }
