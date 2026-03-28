"""
MH-FLOCKE — Body Schema v0.4.1
========================================
Efference copy comparison for anomaly detection.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class BodySchemaConfig:
    """Konfiguration für Körperschema."""
    learning_rate: float = 0.05     # Forward-Model Lernrate
    anomaly_threshold: float = 0.3  # Ab wann ist eine Diskrepanz "anomal"?
    confidence_decay: float = 0.999 # Langsamer Vertrauensverfall
    max_history: int = 100          # Motor-Sensor Paare für Statistik


class BodySchema:
    """
    Körperschema — die Kreatur versteht ihren Körper.
    
    Lernt ein Forward Model: Aktion → erwarteter Sensor-Effekt.
    Diskrepanz zwischen Erwartung und Realität = Anomalie.
    """

    def __init__(self, n_joints: int, n_sensors: int,
                 config: BodySchemaConfig = None):
        """
        Args:
            n_joints: Anzahl Gelenke (Motor-Dimensionen)
            n_sensors: Anzahl Sensor-Kanäle
            config: Optionale Konfiguration
        """
        self.n_joints = n_joints
        self.n_sensors = n_sensors
        self.config = config or BodySchemaConfig()
        
        # Forward Model: Motor → erwarteter Sensor-Effekt (lineares Modell)
        # W[i,j] = "wie stark beeinflusst Motor j den Sensor i"
        self.forward_weights = np.zeros((n_sensors, n_joints), dtype=np.float32)
        self.forward_bias = np.zeros(n_sensors, dtype=np.float32)
        
        # Confidence per joint: how well do I understand this joint?
        self.joint_confidence = np.zeros(n_joints, dtype=np.float32)
        
        # Global body confidence
        self._body_confidence = 0.0
        
        # Last state
        self._last_sensors: Optional[np.ndarray] = None
        self._last_motor: Optional[np.ndarray] = None
        self._last_anomaly = 0.0
        
        # Statistiken
        self._update_count = 0
        self._total_error = 0.0

    def update(self, motor_command: list, current_sensors: list,
               previous_sensors: Optional[list] = None) -> dict:
        """
        Update Forward Model mit beobachtetem Motor→Sensor Paar.
        
        Args:
            motor_command: Aktuelle Motor-Befehle [-1..1]
            current_sensors: Aktuelle Sensor-Werte
            previous_sensors: Vorherige Sensor-Werte (falls extern gegeben)
            
        Returns:
            Dict mit prediction_error, anomaly, body_confidence
        """
        motor = np.array(motor_command[:self.n_joints], dtype=np.float32)
        sensors = np.array(current_sensors[:self.n_sensors], dtype=np.float32)
        
        # If we have previous sensors -> compute sensor difference
        if previous_sensors is not None:
            prev = np.array(previous_sensors[:self.n_sensors], dtype=np.float32)
        elif self._last_sensors is not None:
            prev = self._last_sensors
        else:
            # Erster Step — nur speichern
            self._last_sensors = sensors.copy()
            self._last_motor = motor.copy()
            return {'prediction_error': 0.0, 'anomaly': 0.0, 
                    'body_confidence': 0.0}
        
        # Sensor change (reafference)
        sensor_change = sensors - prev
        
        # Forward Model Vorhersage
        if self._last_motor is not None:
            predicted_change = self.forward_weights @ self._last_motor + self.forward_bias
        else:
            predicted_change = self.forward_bias.copy()
        
        # Prediction Error
        error = sensor_change - predicted_change
        prediction_error = float(np.mean(error ** 2))
        
        # Forward Model Update (Online Learning)
        lr = self.config.learning_rate
        if self._last_motor is not None:
            # Gradient: dE/dW = -2 * error * motor^T
            self.forward_weights += lr * np.outer(error, self._last_motor)
            self.forward_bias += lr * error * 0.1  # Bias langsamer
        
        # Joint confidence: based on per-joint prediction error
        for j in range(min(self.n_joints, len(motor))):
            if abs(motor[j]) > 0.05:  # Nur wenn Motor aktiv war
                # Which sensors did this joint affect?
                joint_error = float(np.mean(np.abs(error) * np.abs(self.forward_weights[:, j])))
                # Confidence increases when error is small
                self.joint_confidence[j] = (
                    0.95 * self.joint_confidence[j] +
                    0.05 * max(0, 1.0 - joint_error * 5)
                )
        
        # Confidence Decay (vergesse langsam)
        self.joint_confidence *= self.config.confidence_decay
        
        # Globale Body Confidence
        self._body_confidence = float(np.mean(self.joint_confidence))
        
        # Anomalie-Detektion
        anomaly = float(np.max(np.abs(error)))
        self._last_anomaly = anomaly
        
        # State Update
        self._last_sensors = sensors.copy()
        self._last_motor = motor.copy()
        self._update_count += 1
        self._total_error += prediction_error
        
        return {
            'prediction_error': prediction_error,
            'anomaly': anomaly,
            'body_confidence': self._body_confidence,
            'is_anomalous': anomaly > self.config.anomaly_threshold,
        }

    def get_body_confidence(self) -> float:
        """Wie gut versteht die Kreatur ihren Körper? 0..1."""
        return self._body_confidence

    def detect_anomaly(self, motor: list, sensor_change: list) -> float:
        """
        "Das hätte nicht passieren sollen" → Integrity-Signal.
        
        Returns:
            Anomalie-Score (0 = normal, 1+ = stark anomal)
        """
        m = np.array(motor[:self.n_joints], dtype=np.float32)
        sc = np.array(sensor_change[:self.n_sensors], dtype=np.float32)
        
        predicted = self.forward_weights @ m + self.forward_bias
        error = sc - predicted
        return float(np.sqrt(np.mean(error ** 2)))

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        return {
            'body_confidence': round(self._body_confidence, 3),
            'joint_confidences': [round(c, 3) for c in self.joint_confidence.tolist()],
            'last_anomaly': round(self._last_anomaly, 3),
            'update_count': self._update_count,
            'avg_error': round(self._total_error / max(self._update_count, 1), 4),
        }
