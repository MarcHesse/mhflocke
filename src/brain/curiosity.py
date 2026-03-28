"""
MH-FLOCKE — Curiosity Drive v0.4.1
========================================
Intrinsic motivation from world model prediction error.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CuriosityConfig:
    """Konfiguration für Curiosity Drive."""
    novelty_threshold: float = 0.1
    max_reward: float = 1.0
    boredom_steps: int = 200
    boredom_reward: float = 0.5
    alpha: float = 0.5                # Balance: 0=nur extrinsisch, 1=nur intrinsisch
    running_mean_decay: float = 0.99


class CuriosityDrive:
    """
    Intrinsische Motivation durch Neugierde.

    Berechnet intrinsischen Reward aus Prediction Error.
    Integration: CognitiveBrain Step 10 (REWARD).
    """

    def __init__(self, config: CuriosityConfig = None):
        self.config = config or CuriosityConfig()
        self.boredom_counter = 0
        self._running_mean = 0.0
        self._running_var = 1.0
        self._step_count = 0
        self.last_prediction_error = 0.0

    def compute_intrinsic_reward(self, prediction_error: float) -> float:
        """
        Berechne intrinsischen Reward aus Prediction Error.

        Normalisiert den Error via Running-Mean/Var.

        Args:
            prediction_error: MSE zwischen vorhergesagtem und tatsächlichem State

        Returns:
            Intrinsischer Reward (0.0 bis max_reward)
        """
        self._step_count += 1
        self._running_mean = (self.config.running_mean_decay * self._running_mean
                              + (1 - self.config.running_mean_decay) * prediction_error)
        self._running_var = (self.config.running_mean_decay * self._running_var
                             + (1 - self.config.running_mean_decay)
                             * (prediction_error - self._running_mean) ** 2)
        normalized_error = (prediction_error - self._running_mean) / (
            np.sqrt(self._running_var) + 1e-8)

        self.last_prediction_error = prediction_error

        if normalized_error > self.config.novelty_threshold:
            intrinsic = min(float(normalized_error), self.config.max_reward)
            self.boredom_counter = 0
        else:
            intrinsic = 0.0
            self.boredom_counter += 1

        if self.boredom_counter > self.config.boredom_steps:
            intrinsic = self.config.boredom_reward
            self.boredom_counter = 0

        return intrinsic

    def total_reward(self, extrinsic: float, intrinsic: float) -> float:
        """Kombiniere externen + internen Reward."""
        alpha = self.config.alpha
        return (1 - alpha) * extrinsic + alpha * intrinsic

    def get_neuromodulator_signals(self) -> dict:
        """Curiosity → Neuromodulator-Signale für Step 15."""
        return {
            'novelty': min(self.last_prediction_error, 1.0),
            'boredom': self.boredom_counter / max(self.config.boredom_steps, 1),
        }

    def reset(self):
        """Reset für neue Kreatur/Generation."""
        self.boredom_counter = 0
        self._running_mean = 0.0
        self._running_var = 1.0
        self._step_count = 0
        self.last_prediction_error = 0.0
