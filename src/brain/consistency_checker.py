"""
MH-FLOCKE — Consistency Checker v0.4.1
========================================
Anterior cingulate-inspired conflict monitoring.
"""

import numpy as np
from typing import Dict, Optional


class ConsistencyChecker:
    """
    Integrity-Check: Stimmt mein Weltmodell mit der Realität überein?
    
    Hohe Dissonanz → Neuromodulator-Reset (Lernen erzwingen).
    """

    # Weights for the three consistency channels
    WEIGHT_WORLD_MODEL = 0.5
    WEIGHT_BODY_SCHEMA = 0.3
    WEIGHT_MEMORY = 0.2
    
    # Dissonanz-Schwellen
    MILD_THRESHOLD = 0.3     # Leichte Inkonsistenz
    SEVERE_THRESHOLD = 0.7   # Schwere Inkonsistenz → Alarm

    def __init__(self):
        self._last_dissonance = 0.0
        self._dissonance_history = []
        self._max_history = 200
        self._step_count = 0
        self._alert_count = 0

    def check(self, prediction_error: float,
              body_anomaly: float,
              memory_mismatch: float = 0.0) -> dict:
        """
        Konsistenz-Check über alle Kanäle.
        
        Args:
            prediction_error: World Model PE (0..1+)
            body_anomaly: Body Schema Anomalie (0..1+)
            memory_mismatch: Episodic Memory Abweichung (0..1+, optional)
            
        Returns:
            Dict mit:
            - dissonance: Gesamt-Dissonanz (0..1)
            - severity: 'none'|'mild'|'severe'
            - neuromod_reset: Dict mit empfohlenen Neuromodulator-Änderungen
            - channel_scores: Pro-Kanal Scores
        """
        # Normalisiere Inputs auf 0..1
        pe_score = min(prediction_error * 2, 1.0)
        body_score = min(body_anomaly * 2, 1.0)
        mem_score = min(memory_mismatch * 2, 1.0)
        
        # Gewichtete Gesamt-Dissonanz
        dissonance = (
            self.WEIGHT_WORLD_MODEL * pe_score +
            self.WEIGHT_BODY_SCHEMA * body_score +
            self.WEIGHT_MEMORY * mem_score
        )
        
        # Maximum across channels as additional signal
        max_channel = max(pe_score, body_score, mem_score)
        # Dissonance is the mean of weighted average and max
        dissonance = 0.6 * dissonance + 0.4 * max_channel
        dissonance = float(np.clip(dissonance, 0, 1))
        
        self._last_dissonance = dissonance
        
        # Severity
        if dissonance >= self.SEVERE_THRESHOLD:
            severity = 'severe'
            self._alert_count += 1
        elif dissonance >= self.MILD_THRESHOLD:
            severity = 'mild'
        else:
            severity = 'none'
        
        # Neuromodulator-Reset bei hoher Dissonanz
        neuromod_reset = self._compute_neuromod_reset(dissonance, severity)
        
        # History
        if len(self._dissonance_history) >= self._max_history:
            self._dissonance_history.pop(0)
        self._dissonance_history.append(dissonance)
        
        self._step_count += 1
        
        return {
            'dissonance': round(dissonance, 3),
            'severity': severity,
            'neuromod_reset': neuromod_reset,
            'channel_scores': {
                'world_model': round(pe_score, 3),
                'body_schema': round(body_score, 3),
                'memory': round(mem_score, 3),
            },
        }

    def _compute_neuromod_reset(self, dissonance: float, severity: str) -> dict:
        """
        Neuromodulator-Anpassung basierend auf Dissonanz.
        
        Hohe Dissonanz →
          ACh HOCH (Lernen!)
          DA RUNTER (alte Strategien verwerfen)
          NE HOCH (Aufmerksamkeit!)
          5-HT RUNTER (Unruhe)
        """
        if severity == 'none':
            return {}  # Kein Reset nötig
        
        if severity == 'severe':
            return {
                'ach': min(0.9, 0.5 + dissonance * 0.5),   # Maximales Lernen
                'da':  max(0.1, 0.5 - dissonance * 0.4),   # Alte Strategien verwerfen
                'ne':  min(0.9, 0.4 + dissonance * 0.5),   # Maximale Aufmerksamkeit
                '5ht': max(0.1, 0.4 - dissonance * 0.3),   # Unruhe
            }
        else:  # mild
            return {
                'ach': min(0.7, 0.4 + dissonance * 0.3),
                'ne':  min(0.7, 0.3 + dissonance * 0.3),
            }

    def get_avg_dissonance(self) -> float:
        """Durchschnittliche Dissonanz der letzten Steps."""
        if not self._dissonance_history:
            return 0.0
        return float(np.mean(self._dissonance_history))

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        return {
            'last_dissonance': round(self._last_dissonance, 3),
            'avg_dissonance': round(self.get_avg_dissonance(), 3),
            'alert_count': self._alert_count,
            'step_count': self._step_count,
        }
