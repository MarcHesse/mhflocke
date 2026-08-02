"""
MH-FLOCKE — Curiosity Drive v0.4.1
========================================
Intrinsic motivation from world model prediction error.
"""

__version__ = "0.4.1"
__logbook__ = 94

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
    # Lever C (Task #84): learning-progress curiosity. When True, reward the
    # DECREASE of prediction error (slow baseline minus fast mean) instead of the
    # surprise (error above its own recent mean). Unlearnable chaos -- a wall the
    # world-model can't predict -- keeps error flat-high -> no progress -> ~0 reward
    # -> the wall stops being "interesting". Default False = unchanged behaviour.
    learning_progress_mode: bool = False
    lp_slow_decay: float = 0.999      # slow baseline EMA horizon (~1000 steps)
    lp_scale: float = 1.0             # scales the progress signal into reward


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
        self._lp_fast = 0.0              # Lever C: own fast EMA (seeded), decoupled from novelty state
        self._lp_slow = 0.0             # Lever C: slow baseline EMA for learning-progress
        self._lp_signal = 0.0           # Lever C: smoothed SIGNED progress (rectified after) -> noise robust
        self._lp_seeded = False
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

        if self.config.learning_progress_mode:
            # Lever C (Task #84): reward error DECREASE, not error level/surprise.
            # Two own EMAs, seeded to the first error so they start equal (decoupled
            # from the novelty-mode running_mean): a fast one and a slower baseline.
            # progress = baseline - fast, >0 only when recent error has dropped below
            # the older baseline (the model is learning). A chaotic/unlearnable wall
            # keeps error flat-high -> fast ~= slow -> ~0 reward -> the wall gets
            # boring. No boredom_reward branch (that one actively pushed to the wall).
            if not self._lp_seeded:
                self._lp_fast = prediction_error
                self._lp_slow = prediction_error
                self._lp_signal = 0.0
                self._lp_seeded = True
            else:
                self._lp_fast = (self.config.running_mean_decay * self._lp_fast
                                 + (1 - self.config.running_mean_decay) * prediction_error)
                self._lp_slow = (self.config.lp_slow_decay * self._lp_slow
                                 + (1 - self.config.lp_slow_decay) * prediction_error)
                # smooth the SIGNED progress (slow-fast) and rectify AFTER: symmetric
                # noise at a chaotic wall averages to ~0 (no false reward), while a
                # sustained downward trend (real learning) stays positive.
                self._lp_signal = (self.config.running_mean_decay * self._lp_signal
                                   + (1 - self.config.running_mean_decay) * (self._lp_slow - self._lp_fast))
            return min(self.config.max_reward, max(0.0, self._lp_signal) * self.config.lp_scale)

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
        self._lp_fast = 0.0
        self._lp_slow = 0.0
        self._lp_signal = 0.0
        self._lp_seeded = False
        self._running_var = 1.0
        self._step_count = 0
        self.last_prediction_error = 0.0
