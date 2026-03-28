"""
MH-FLOCKE — Global Workspace v0.4.1
========================================
Module competition for attentional broadcast (Baars, 1988).
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GWTBridgeConfig:
    """Konfiguration für GWT-SNN Brücke."""
    n_neurons: int = 500_000
    n_modules: int = 4
    broadcast_strength: float = 0.5
    competition_method: str = 'softmax'  # 'softmax' oder 'winner_takes_all'
    broadcast_duration: int = 10
    attention_decay: float = 0.95
    device: str = 'cpu'


class GWTModule:
    """Ein Modul das um den Global Workspace Broadcast konkurriert."""

    def __init__(self, name: str, n_neurons: int, device: str = 'cpu'):
        self.name = name
        self.n_neurons = n_neurons
        self.device = device

        self._signal = torch.zeros(n_neurons, device=device)
        self._salience = 0.0
        self._input_history: List[float] = []

    def compute_salience(self, input_signal: torch.Tensor) -> float:
        """
        Salienz: Kombination aus Signalstärke und Neuheit.
        Starke oder plötzlich veränderte Signale = hohe Salienz.
        """
        strength = input_signal.abs().mean().item()

        # Neuheit: Wie anders ist das Signal vs. letztes?
        novelty = 0.0
        if self._input_history:
            novelty = abs(strength - self._input_history[-1])

        self._input_history.append(strength)
        if len(self._input_history) > 50:
            self._input_history = self._input_history[-50:]

        self._salience = 0.7 * strength + 0.3 * novelty
        return self._salience

    def get_broadcast_signal(self) -> torch.Tensor:
        """Das Signal das broadcastet wird wenn dieses Modul gewinnt."""
        return self._signal.clone()

    def update(self, input_signal: torch.Tensor):
        """Aktualisiert internen Zustand."""
        n = min(len(input_signal), self.n_neurons)
        self._signal = torch.zeros(self.n_neurons, device=self.device)
        self._signal[:n] = input_signal[:n].to(self.device)


class GlobalWorkspaceBridge:
    """
    Verbindet GWT mit Multi-Compartment SNN.

    Pro Step:
    1. Module aktualisieren
    2. Salienz-Wettbewerb
    3. Broadcast → Apikaler Kontext
    """

    def __init__(self, config: GWTBridgeConfig):
        self.config = config
        self.device = config.device

        self._modules: Dict[str, GWTModule] = {}
        self._broadcast_signal = torch.zeros(config.n_neurons, device=self.device)
        self._winning_module = ''
        self._broadcast_age = 0
        self._attention_map = torch.zeros(config.n_neurons, device=self.device)
        self._history: List[Dict] = []

        # Override
        self._override_module: Optional[str] = None
        self._override_remaining: int = 0

    def register_module(self, module: GWTModule):
        """Registriert ein GWT-Modul."""
        self._modules[module.name] = module

    def step(self, module_inputs: Dict[str, torch.Tensor]) -> Dict:
        """Ein GWT-Schritt."""
        salience_scores = {}

        # 1. Module aktualisieren + Salienz berechnen
        for name, module in self._modules.items():
            if name in module_inputs:
                module.update(module_inputs[name])
                salience_scores[name] = module.compute_salience(module_inputs[name])
            else:
                salience_scores[name] = 0.0

        # 2. Wettbewerb
        if self._override_module and self._override_remaining > 0:
            winner = self._override_module
            self._override_remaining -= 1
        elif salience_scores:
            if self.config.competition_method == 'winner_takes_all':
                winner = max(salience_scores, key=salience_scores.get)
            else:  # softmax
                names = list(salience_scores.keys())
                scores = torch.tensor([salience_scores[n] for n in names])
                if scores.sum() > 1e-8:
                    probs = torch.softmax(scores * 5.0, dim=0)
                    idx = torch.multinomial(probs, 1).item()
                    winner = names[idx]
                else:
                    winner = names[0] if names else ''
        else:
            winner = ''

        # 3. Broadcast-Signal
        if winner and winner in self._modules:
            new_signal = self._modules[winner].get_broadcast_signal()
            self._broadcast_signal = new_signal * self.config.broadcast_strength
            self._winning_module = winner
            self._broadcast_age = 0
        else:
            self._broadcast_age += 1

        # 4. Attention-Map: Broadcast mit Decay
        self._attention_map = (self.config.attention_decay * self._attention_map +
                               (1 - self.config.attention_decay) * self._broadcast_signal.abs())

        # 5. Broadcast decay over time
        if self._broadcast_age > self.config.broadcast_duration:
            self._broadcast_signal *= self.config.attention_decay

        # Historie
        entry = {
            'winning_module': winner,
            'salience_scores': dict(salience_scores),
            'broadcast_strength': self._broadcast_signal.abs().mean().item(),
        }
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        return {
            'broadcast_signal': self._broadcast_signal.clone(),
            'winning_module': winner,
            'salience_scores': salience_scores,
            'attention_map': self._attention_map.clone(),
        }

    def get_apical_context(self) -> torch.Tensor:
        """Aktueller apikaler Kontext für Multi-Compartment Layer."""
        return self._broadcast_signal.clone()

    def get_attention_focus(self) -> str:
        """Worauf ist die Aufmerksamkeit gerichtet?"""
        return self._winning_module

    def get_broadcast_history(self, n_steps: int = 50) -> List[Dict]:
        """Letzte N Broadcast-Entscheidungen."""
        return self._history[-n_steps:]

    def override_attention(self, module_name: str, duration: int = 10):
        """Erzwingt Aufmerksamkeit (willentliche Kontrolle)."""
        if module_name in self._modules:
            self._override_module = module_name
            self._override_remaining = duration
