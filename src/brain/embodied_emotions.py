"""
MH-FLOCKE — Embodied Emotions v0.4.1
========================================
Valence-arousal emotional system derived from body state signals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class EmotionalState:
    """Emotionaler Zustand nach Valence-Arousal Modell."""
    valence: float = 0.0       # -1.0 (Schmerz/Angst) bis +1.0 (Freude)
    arousal: float = 0.5       # 0.0 (ruhig) bis 1.0 (erregt)
    dominant_emotion: str = "neutral"
    timestamp: float = field(default_factory=time.time)


# Emotion-Klassifikation basierend auf Valence-Arousal Quadranten
EMOTION_MAP = {
    # (valence_sign, arousal_level) → emotion
    (+1, 'high'): 'excited',     # Hohe Valence + Hoher Arousal = Aufregung
    (+1, 'low'):  'content',     # Hohe Valence + Niedriger Arousal = Zufriedenheit
    (-1, 'high'): 'fearful',    # Niedrige Valence + Hoher Arousal = Angst
    (-1, 'low'):  'sad',         # Niedrige Valence + Niedriger Arousal = Trauer
    (0,  'mid'):  'neutral',     # Mitte = neutral
}


class EmbodiedEmotions:
    """
    Valence-Arousal aus Körperzustand — somatische Marker für Kreaturen.
    
    Keine Text-Analyse, sondern direkte Ableitung aus:
    - Sensor-Daten (Höhe, Aufrichtung, Geschwindigkeit)
    - Prediction Error (Überraschung)
    - Reward (Fortschritt)
    - Sturz-Zustand
    """

    # Exponential Moving Average Decay
    VALENCE_DECAY = 0.9
    AROUSAL_DECAY = 0.85

    def __init__(self):
        self.state = EmotionalState()
        self.history: List[EmotionalState] = []
        self.max_history: int = 500
        self._step_count = 0

    def update(self, sensor_data: dict, prediction_error: float,
               reward: float, is_fallen: bool) -> EmotionalState:
        """
        Berechne emotionalen Zustand aus Körpersignalen.
        
        Args:
            sensor_data: Dict mit height, upright, forward_velocity, joint_angles etc.
            prediction_error: World Model Prediction Error (0..1+)
            reward: Externer Reward (-1..1+)
            is_fallen: Ob die Kreatur gestürzt ist
            
        Returns:
            Aktueller EmotionalState
        """
        # === VALENCE (how am I doing?) ===
        valence_signals = []
        
        # Aufrecht sein = gut
        upright = sensor_data.get('upright', 0.5)
        valence_signals.append(upright * 0.3)  # 0..0.3
        
        # Fortschritt = gut
        valence_signals.append(np.clip(reward * 0.4, -0.4, 0.4))
        
        # Niedriger Prediction Error = stabile Welt = gut
        pe_valence = (1.0 - min(prediction_error * 3, 1.0)) * 0.15
        valence_signals.append(pe_valence)
        
        # Sturz = sehr schlecht
        if is_fallen:
            valence_signals.append(-0.5)
        
        # Joint at limit = pain (joint_angles near +/-1)
        joint_angles = sensor_data.get('joint_angles', [])
        if joint_angles:
            pain = sum(1.0 for a in joint_angles if abs(a) > 0.9) / max(len(joint_angles), 1)
            valence_signals.append(-pain * 0.2)
        
        raw_valence = sum(valence_signals)
        
        # === AROUSAL (wie erregt bin ich?) ===
        arousal_signals = []
        
        # High prediction error = surprise
        arousal_signals.append(min(prediction_error * 2, 0.5))
        
        # Hohe Geschwindigkeit = Aufregung
        fwd_vel = abs(sensor_data.get('forward_velocity', 0))
        arousal_signals.append(min(fwd_vel * 0.1, 0.2))
        
        # Sturz-Gefahr (niedrige Aufrichtung) = Alarmstufe
        if upright < 0.5:
            arousal_signals.append((0.5 - upright) * 0.6)
        
        # Fallen state = initial high arousal, then resignation
        if is_fallen:
            arousal_signals.append(0.3)
        
        raw_arousal = sum(arousal_signals)
        
        # === EMA Smoothing ===
        self.state.valence = (
            self.VALENCE_DECAY * self.state.valence +
            (1 - self.VALENCE_DECAY) * np.clip(raw_valence, -1.0, 1.0)
        )
        self.state.arousal = (
            self.AROUSAL_DECAY * self.state.arousal +
            (1 - self.AROUSAL_DECAY) * np.clip(raw_arousal, 0.0, 1.0)
        )
        
        # Emotion klassifizieren
        self.state.dominant_emotion = self._classify_emotion(
            self.state.valence, self.state.arousal)
        self.state.timestamp = time.time()
        
        # History
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(EmotionalState(
            valence=self.state.valence,
            arousal=self.state.arousal,
            dominant_emotion=self.state.dominant_emotion,
        ))
        
        self._step_count += 1
        return self.state

    def get_somatic_markers(self) -> dict:
        """
        Neuromodulator-Modulation basierend auf Emotionen (Damasio).
        
        Returns:
            Dict mit Neuromodulator-Levels:
            - da: Dopamin (Belohnung, positive Valence)
            - 5ht: Serotonin (Wohlbefinden, stabile Valence)
            - ne: Noradrenalin (Arousal, Aufmerksamkeit)
            - ach: Acetylcholin (Lernen, bei negativer Überraschung)
        """
        v = self.state.valence
        a = self.state.arousal
        
        return {
            'da':  np.clip(0.3 + v * 0.4, 0.05, 0.9),      # Positiv → DA hoch
            '5ht': np.clip(0.5 + v * 0.2 - a * 0.1, 0.1, 0.8),  # Valence↑, Arousal↓ → 5-HT
            'ne':  np.clip(0.2 + a * 0.5, 0.1, 0.9),        # Arousal → NE
            'ach': np.clip(0.3 + a * 0.3 - v * 0.2, 0.1, 0.9),  # Überraschung → ACh
        }

    def get_gwt_salience_modulation(self) -> dict:
        """
        Emotion moduliert GWT-Salience.
        
        Ängstliche Kreatur → Error-Modul hat mehr Gewicht
        Zufriedene Kreatur → Motor-Modul hat mehr Gewicht (Exploitation)
        """
        v = self.state.valence
        a = self.state.arousal
        
        return {
            'sensory':    1.0 + a * 0.3,                    # Arousal → mehr Sensor-Fokus
            'motor':      1.0 + max(v, 0) * 0.3,            # Positive → Motor (handeln!)
            'predictive': 1.0 + max(v, 0) * 0.2 - a * 0.1,  # Ruhe → besser vorhersagen
            'error':      1.0 + a * 0.4 - v * 0.2,          # Angst → Error-Fokus!
            'memory':     1.0 + a * 0.2,                     # Arousal → Memory-Abruf
            'social':     1.0 + max(v, 0) * 0.2,            # Positive → Soziales
        }

    def _classify_emotion(self, valence: float, arousal: float) -> str:
        """Emotion aus Valence-Arousal Quadranten bestimmen.
        
        Thresholds widened (v0.4.1) — original thresholds were too narrow,
        causing perpetual 'neutral' in Go2 training runs.
        """
        # Only neutral if both valence AND arousal are truly flat
        if abs(valence) < 0.02 and abs(arousal - 0.5) < 0.1:
            return 'neutral'
        
        v_sign = 1 if valence > 0.02 else (-1 if valence < -0.02 else 0)
        a_level = 'high' if arousal > 0.55 else ('low' if arousal < 0.35 else 'mid')
        
        return EMOTION_MAP.get((v_sign, a_level), 'neutral')

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        return {
            'valence': round(self.state.valence, 3),
            'arousal': round(self.state.arousal, 3),
            'dominant_emotion': self.state.dominant_emotion,
            'somatic_markers': self.get_somatic_markers(),
        }
