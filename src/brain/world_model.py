"""
MH-FLOCKE — World Model v0.4.1
========================================
Spiking world model with prediction error and dream-based consolidation.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from src.brain.snn_controller import SNNController, SNNConfig


@dataclass
class WorldModelConfig:
    """Konfiguration für das World Model."""
    n_input: int = 20
    n_hidden: int = 500
    n_output: int = 10
    tau_base: float = 30.0
    learning_rate: float = 0.02
    prediction_window: int = 5
    device: str = 'cpu'


class SpikingWorldModel:
    """
    Spiking World Model mit LTC-Dynamik.
    Lernt: (sensor_t, action_t) → sensor_{t+1}
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.device = config.device

        # Dimensionen
        n_sensors = config.n_output  # Sensor-Dimensionen
        n_motors = config.n_input - config.n_output  # Motor-Dimensionen
        self.n_sensors = n_sensors
        self.n_motors = max(n_motors, 1)

        # SNN als World Model
        total = n_sensors + self.n_motors + config.n_hidden + config.n_output
        snn_config = SNNConfig(
            n_neurons=total,
            connectivity_prob=0.05,
            device=config.device,
        )
        self.snn = SNNController(snn_config)

        # Populationen
        idx = 0
        self._input_start = idx
        self._input_end = idx + n_sensors + self.n_motors
        idx = self._input_end

        self._hidden_start = idx
        self._hidden_end = idx + config.n_hidden
        idx = self._hidden_end

        self._output_start = idx
        self._output_end = idx + config.n_output

        # Verbindungen
        input_ids = torch.arange(self._input_start, self._input_end)
        hidden_ids = torch.arange(self._hidden_start, self._hidden_end)
        output_ids = torch.arange(self._output_start, self._output_end)

        self.snn.define_population('wm_input', input_ids)
        self.snn.define_population('wm_hidden', hidden_ids)
        self.snn.define_population('wm_output', output_ids)

        self.snn.connect_populations('wm_input', 'wm_hidden', prob=0.1, weight_range=(0.2, 0.8))
        self.snn.connect_populations('wm_hidden', 'wm_output', prob=0.1, weight_range=(0.2, 0.8))
        self.snn.connect_populations('wm_hidden', 'wm_hidden', prob=0.02, weight_range=(0.05, 0.3))

        # Output-Membranpotential als analoges Signal
        self._prediction_history: List[float] = []

    def predict(self, sensor_input: torch.Tensor,
                motor_command: torch.Tensor) -> torch.Tensor:
        """Vorhersage des nächsten Sensor-Zustands."""
        n = self.snn.config.n_neurons

        # Input zusammenbauen
        snn_input = torch.zeros(n, device=self.device)
        s_len = min(len(sensor_input), self.n_sensors)
        m_len = min(len(motor_command), self.n_motors)
        snn_input[self._input_start:self._input_start + s_len] = sensor_input[:s_len] * 3.0
        snn_input[self._input_start + self.n_sensors:self._input_start + self.n_sensors + m_len] = motor_command[:m_len] * 3.0

        # SNN step
        self.snn.step(snn_input)

        # Output: Membranpotential der Output-Neuronen, normalisiert
        V_out = self.snn.V[self._output_start:self._output_end]
        predicted = torch.sigmoid(V_out)

        return predicted

    def train_step(self, sensor_input: torch.Tensor,
                   motor_command: torch.Tensor,
                   actual_next_sensors: torch.Tensor) -> float:
        """Ein Trainingsschritt. Returns prediction_error."""
        predicted = self.predict(sensor_input, motor_command)

        # Prediction Error
        error = torch.mean((predicted - actual_next_sensors[:self.config.n_output]) ** 2).item()

        # R-STDP: Negative error as reward (less error = more reward)
        reward = -error * self.config.learning_rate * 10.0
        self.snn.apply_rstdp(reward_signal=reward)

        self._prediction_history.append(error)
        return error

    def get_prediction_error(self, sensor_input: torch.Tensor,
                              motor_command: torch.Tensor,
                              actual_next_sensors: torch.Tensor) -> float:
        """Prediction Error ohne Training."""
        predicted = self.predict(sensor_input, motor_command)
        error = torch.mean((predicted - actual_next_sensors[:self.config.n_output]) ** 2).item()
        return error

    def get_state(self) -> Dict:
        """Interner Zustand."""
        return {
            'n_neurons': self.snn.config.n_neurons,
            'n_input': self.config.n_input,
            'n_hidden': self.config.n_hidden,
            'n_output': self.config.n_output,
            'mean_prediction_error': np.mean(self._prediction_history[-50:]) if self._prediction_history else 0.0,
        }


class DreamEngine:
    """
    Dream-Modus: Offline-Lernen durch Replay und Halluzination.
    """

    def __init__(self, world_model: SpikingWorldModel,
                 creature_snn: SNNController):
        self.world_model = world_model
        self.creature_snn = creature_snn
        self._buffer = deque(maxlen=10000)

    def record_experience(self, sensor: torch.Tensor,
                           action: torch.Tensor,
                           reward: float):
        """Speichert Erfahrung im Replay-Buffer."""
        self._buffer.append((
            sensor.detach().cpu().clone(),
            action.detach().cpu().clone(),
            reward,
        ))

    def dream(self, n_steps: int = 100,
              replay_ratio: float = 0.7) -> Dict:
        """
        Träum-Phase: Offline-Lernen.
        replay_ratio der Steps: Replay, Rest: Halluzination.
        """
        n_replay = 0
        n_hallucination = 0
        errors = []
        rewards = []

        buffer_size = len(self._buffer)
        n_sensors = self.world_model.n_sensors
        n_motors = self.world_model.n_motors

        for i in range(n_steps):
            use_replay = (buffer_size > 0 and
                          np.random.random() < replay_ratio)

            if use_replay:
                # Replay: repeat random experience
                idx = np.random.randint(0, buffer_size)
                sensor, action, reward = self._buffer[idx]
                sensor = sensor.to(self.world_model.device)
                action = action.to(self.world_model.device)

                # World model predicts next state
                predicted = self.world_model.predict(sensor[:n_sensors], action[:n_motors])

                # Creature-SNN lernt auf dem Replay
                snn_input = torch.zeros(self.creature_snn.config.n_neurons,
                                        device=self.creature_snn.config.device)
                s_len = min(len(sensor), snn_input.shape[0])
                snn_input[:s_len] = sensor[:s_len] * 3.0
                self.creature_snn.step(snn_input)
                self.creature_snn.apply_rstdp(reward_signal=reward * 0.5)

                n_replay += 1
                rewards.append(reward)

            else:
                # Halluzination: World Model generiert
                fake_sensor = torch.randn(n_sensors, device=self.world_model.device) * 0.5
                fake_action = torch.randn(n_motors, device=self.world_model.device) * 0.5
                predicted = self.world_model.predict(fake_sensor, fake_action)

                # Creature-SNN lernt auf halluzinierten Daten
                snn_input = torch.zeros(self.creature_snn.config.n_neurons,
                                        device=self.creature_snn.config.device)
                p_len = min(len(predicted), snn_input.shape[0])
                snn_input[:p_len] = predicted[:p_len] * 3.0
                self.creature_snn.step(snn_input)

                n_hallucination += 1

            errors.append(0.0)  # Placeholder

        return {
            'n_replay_steps': n_replay,
            'n_hallucination_steps': n_hallucination,
            'mean_prediction_error': float(np.mean(errors)) if errors else 0.0,
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
        }

    def get_replay_buffer_size(self) -> int:
        """Anzahl gespeicherter Erfahrungen."""
        return len(self._buffer)

    def clear_buffer(self):
        """Löscht Replay-Buffer."""
        self._buffer = deque(maxlen=10000)
