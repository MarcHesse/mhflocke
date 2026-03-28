"""
MH-FLOCKE — Empowerment Drive v0.4.1
========================================
Intrinsic motivation from action-state mutual information.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from collections import deque


@dataclass
class EmpowermentConfig:
    history_size: int = 100
    n_action_samples: int = 5
    max_reward: float = 1.0
    weight: float = 0.3
    state_dim_limit: int = 20


class EmpowermentDrive:
    """
    Intrinsische Motivation durch Handlungsfähigkeit.
    Kreatur maximiert ihren Einfluss auf die Welt.
    """

    def __init__(self, config: EmpowermentConfig = None):
        self.config = config or EmpowermentConfig()
        self._action_history: deque = deque(maxlen=self.config.history_size)
        self._state_history: deque = deque(maxlen=self.config.history_size)
        self.last_empowerment: float = 0.0

    def record(self, action: np.ndarray, next_state: np.ndarray):
        """Action-State Paar aufzeichnen."""
        a = action[:self.config.state_dim_limit].copy()
        s = next_state[:self.config.state_dim_limit].copy()
        self._action_history.append(a)
        self._state_history.append(s)

    def compute_empowerment(self) -> float:
        """
        Approximation der Mutual Information Action→State.
        Hohe Varianz der States bei verschiedenen Actions = hohe Empowerment.
        """
        if len(self._action_history) < 20:
            return 0.0

        actions = np.array(list(self._action_history))
        states = np.array(list(self._state_history))

        try:
            action_magnitudes = np.linalg.norm(actions, axis=1)
            median_mag = np.median(action_magnitudes)

            low_mask = action_magnitudes <= median_mag
            high_mask = action_magnitudes > median_mag

            if np.sum(low_mask) < 5 or np.sum(high_mask) < 5:
                return 0.0

            mean_diff = np.linalg.norm(
                np.mean(states[high_mask], axis=0) - np.mean(states[low_mask], axis=0))
            var_total = np.mean(np.var(states, axis=0)) + 1e-8

            empowerment = float(np.clip(mean_diff / var_total, 0, self.config.max_reward))
            self.last_empowerment = empowerment
            return empowerment
        except Exception:
            return 0.0

    def compute_reward(self) -> float:
        """Empowerment als gewichteter Reward."""
        return self.compute_empowerment() * self.config.weight

    def reset(self):
        self._action_history.clear()
        self._state_history.clear()
        self.last_empowerment = 0.0

    def to_dict(self) -> dict:
        return {
            'empowerment': self.last_empowerment,
            'history_size': len(self._action_history),
        }
