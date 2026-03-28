"""
MH-FLOCKE — Theory of Mind v0.4.1
========================================
Multi-compartment SNN module for modeling other agents.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from src.brain.multi_compartment import MultiCompartmentLayer, MultiCompartmentConfig


@dataclass
class ToMConfig:
    """Konfiguration für Theory of Mind Modul."""
    n_mirror_neurons: int = 200
    n_decision_neurons: int = 100
    n_observed_features: int = 10
    n_action_features: int = 5

    prediction_horizon: int = 5
    prediction_lr: float = 0.02

    cooperation_threshold: float = 0.5

    device: str = 'cpu'
    dtype: torch.dtype = torch.float32


class MirrorSNN:
    """
    Mirror Neuron System.

    Basal: beobachtete Aktion/Position der anderen Kreatur
    Apikal: eigener Zustand (Kontext)
    Output: vorhergesagte nächste Aktion
    """

    def __init__(self, config: ToMConfig):
        self.config = config
        self.device = config.device

        # Multi-Compartment Layer als Mirror-Netzwerk
        mc_config = MultiCompartmentConfig(
            n_neurons=config.n_mirror_neurons,
            tau_basal=10.0,
            tau_apical=30.0,
            apical_gain_factor=0.5,
            device=config.device,
            dtype=config.dtype,
        )
        self.layer = MultiCompartmentLayer(mc_config)

        # Output-Gewichte: Mirror-Neuronen → Aktions-Vorhersage
        self._output_weights = torch.randn(
            config.n_action_features, config.n_mirror_neurons,
            device=config.device, dtype=config.dtype,
        ) * 0.1

        # Prediction Error History
        self._error_history: List[float] = []
        self._confidence = 0.0

    def observe(self, other_state: torch.Tensor,
                own_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Beobachte andere Kreatur, sage nächste Aktion vorher."""
        n = self.config.n_mirror_neurons
        n_obs = self.config.n_observed_features

        # Basal-Input: beobachtete Features → Mirror-Neuronen
        basal = torch.zeros(n, device=self.device, dtype=self.config.dtype)
        s_len = min(len(other_state), n_obs, n)
        basal[:s_len] = other_state[:s_len] * 3.0

        # Apical input: own state (context)
        apical = None
        if own_state is not None:
            apical = torch.zeros(n, device=self.device, dtype=self.config.dtype)
            o_len = min(len(own_state), n_obs, n)
            apical[:o_len] = own_state[:o_len] * 2.0

        # Multi-Compartment Step
        spikes = self.layer.step(basal, apical)

        # Output: Membranpotential → Aktions-Vorhersage
        V = self.layer.V_soma
        predicted = torch.tanh(self._output_weights @ V)

        return predicted

    def train_step(self, other_state: torch.Tensor,
                   actual_action: torch.Tensor,
                   own_state: Optional[torch.Tensor] = None) -> float:
        """Trainiere mit tatsächlicher Aktion. Returns prediction_error."""
        predicted = self.observe(other_state, own_state)

        n_act = self.config.n_action_features
        target = actual_action[:n_act]

        # Prediction Error
        error = torch.mean((predicted - target) ** 2).item()

        # Einfaches Gradient-Update auf Output-Gewichte
        lr = self.config.prediction_lr
        diff = (predicted - target).unsqueeze(1)  # [n_act, 1]
        V = self.layer.V_soma.unsqueeze(0)  # [1, n_mirror]
        grad = diff @ V  # [n_act, n_mirror]
        self._output_weights -= lr * grad

        # Tracking
        self._error_history.append(error)
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-100:]

        # Confidence: inverse des mittleren Fehlers
        mean_err = np.mean(self._error_history)
        self._confidence = float(np.clip(1.0 / (1.0 + mean_err * 5.0), 0.0, 1.0))

        return error

    def get_prediction_confidence(self) -> float:
        """0=unsicher, 1=sehr sicher."""
        return self._confidence


class CooperationDecider:
    """Entscheidet Kooperation vs. Kompetition."""

    def __init__(self, config: ToMConfig):
        self.config = config
        self.device = config.device

        n_in = config.n_observed_features + config.n_action_features
        self._weights = torch.randn(
            1, n_in, device=config.device, dtype=config.dtype,
        ) * 0.1
        self._bias = torch.zeros(1, device=config.device, dtype=config.dtype)

    def decide(self, own_state: torch.Tensor,
               predicted_other_action: torch.Tensor,
               confidence: float) -> float:
        """Returns cooperation_score: -1 (Kompetition) bis +1 (Kooperation)."""
        n_obs = self.config.n_observed_features
        n_act = self.config.n_action_features

        # Input zusammenbauen
        inp = torch.zeros(n_obs + n_act, device=self.device, dtype=self.config.dtype)
        o_len = min(len(own_state), n_obs)
        a_len = min(len(predicted_other_action), n_act)
        inp[:o_len] = own_state[:o_len]
        inp[n_obs:n_obs + a_len] = predicted_other_action[:a_len]

        # Linear → tanh
        raw = (self._weights @ inp + self._bias).item()
        score = float(np.tanh(raw))

        # Confidence scales the decision strength
        score = score * confidence

        return float(np.clip(score, -1.0, 1.0))

    def train_step(self, own_state: torch.Tensor,
                   other_prediction: torch.Tensor,
                   cooperation_outcome: float):
        """Trainiere basierend auf Ergebnis."""
        n_obs = self.config.n_observed_features
        n_act = self.config.n_action_features

        inp = torch.zeros(n_obs + n_act, device=self.device, dtype=self.config.dtype)
        o_len = min(len(own_state), n_obs)
        a_len = min(len(other_prediction), n_act)
        inp[:o_len] = own_state[:o_len]
        inp[n_obs:n_obs + a_len] = other_prediction[:a_len]

        # Einfaches Update: Gewichte in Richtung Outcome schieben
        lr = 0.01
        current = (self._weights @ inp + self._bias).item()
        error = cooperation_outcome - np.tanh(current)
        self._weights += lr * error * inp.unsqueeze(0)
        self._bias += lr * error


class TheoryOfMind:
    """Gesamt ToM-Modul: Mirror-SNN + Cooperation-Decider."""

    def __init__(self, config: ToMConfig):
        self.config = config
        self.mirror = MirrorSNN(config)
        self.decider = CooperationDecider(config)

    def step(self, other_state: torch.Tensor,
             own_state: torch.Tensor) -> Dict:
        """Vollständiger ToM-Schritt."""
        predicted_action = self.mirror.observe(other_state, own_state)
        confidence = self.mirror.get_prediction_confidence()
        cooperation = self.decider.decide(own_state, predicted_action, confidence)

        return {
            'predicted_action': predicted_action,
            'prediction_confidence': confidence,
            'cooperation_signal': cooperation,
        }

    def train(self, other_state: torch.Tensor,
              actual_action: torch.Tensor,
              own_state: torch.Tensor,
              cooperation_outcome: float):
        """Trainiere alle Komponenten."""
        self.mirror.train_step(other_state, actual_action, own_state)

        predicted = self.mirror.observe(other_state, own_state)
        self.decider.train_step(own_state, predicted, cooperation_outcome)

    def get_state(self) -> Dict:
        return {
            'prediction_confidence': self.mirror.get_prediction_confidence(),
            'n_mirror_neurons': self.config.n_mirror_neurons,
            'n_decision_neurons': self.config.n_decision_neurons,
        }

    def reset_model(self):
        """Reset für neue Beobachtung."""
        self.mirror = MirrorSNN(self.config)
        self.decider = CooperationDecider(self.config)
