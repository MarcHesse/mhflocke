"""
MH-FLOCKE — Motivational Drives v0.4.1
========================================
Survival, exploration, comfort, and social drives for behavior selection.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DriveState:
    """Zustand aller Drives."""
    survival: float = 0.5       # Aufrecht bleiben, nicht fallen
    exploration: float = 0.3    # Neue Orte entdecken
    comfort: float = 0.2       # Energie sparen, stabiler Zustand
    social: float = 0.0        # Nähe zu anderen (Phase 10-12)
    play: float = 0.1          # Spieltrieb — Objekte untersuchen, Ball jagen (Panksepp PLAY)
    dominant: str = 'survival'


class MotivationalDrives:
    """
    Biologische Antriebe — Präfrontaler Cortex für Kreaturen.
    
    Fünf primäre Drives (angeboren):
    - SURVIVAL: Aufrecht bleiben, nicht fallen
    - EXPLORATION: Neue Orte entdecken (Curiosity)
    - COMFORT: Energie sparen, stabiler Zustand
    - SOCIAL: Nähe zu anderen Kreaturen
    - PLAY: Spieltrieb — Objekte untersuchen, Ball jagen (Panksepp 1998)
    
    Der dominante Drive moduliert Reward und GWT-Bias.
    """

    # Drive Decay (pro Step tendiert jeder Drive zum Baseline)
    DECAY = 0.98
    
    # Baselines (wohin Drives konvergieren ohne Input)
    BASELINES = {
        'survival': 0.4,
        'exploration': 0.3,
        'comfort': 0.2,
        'social': 0.1,
        'play': 0.15,       # Low baseline, rises with object salience
    }

    def __init__(self):
        self.state = DriveState()
        self._step_count = 0

    def compute_drive_strengths(self, state: dict) -> DriveState:
        """
        Berechne Drive-Stärken aus Kreatur-Zustand.
        
        Args:
            state: Dict mit:
                - upright: float (0..1, Aufrichtung)
                - height: float (Höhe)
                - prediction_error: float (World Model Error)
                - energy_spent: float (kumulierte Energie)
                - is_fallen: bool
                - learning_progress: float (Fitness-Trend)
                - other_creature_visible: bool (für Social)
                
        Returns:
            Aktualisierter DriveState
        """
        # === SURVIVAL: High when danger ===
        upright = state.get('upright', 0.5)
        is_fallen = state.get('is_fallen', False)
        height = state.get('height', 0.5)
        
        survival_signal = 0.0
        if is_fallen:
            survival_signal = 1.0  # Maximale Priorität
        elif upright < 0.5:
            survival_signal = (0.5 - upright) * 2.0  # Steigt wenn Aufrichtung sinkt
        elif height < 0.2:
            survival_signal = 0.5
        
        # v0.7.0: Bad gait quality → survival concern (body not functioning well)
        gait_q = state.get('gait_quality', 0.5)
        if gait_q < 0.4:
            survival_signal += (0.4 - gait_q) * 1.5  # GQ=0.2 → +0.3
        
        # v0.7.0: Dead limb → strong survival/adaptation signal
        limb_dead = state.get('limb_dead', [])
        if limb_dead:
            survival_signal += 0.4 * len(limb_dead)
        
        self.state.survival = (
            self.DECAY * self.state.survival +
            (1 - self.DECAY) * np.clip(self.BASELINES['survival'] + survival_signal, 0, 1)
        )
        
        # === EXPLORATION: High when bored (low PE = nothing new) ===
        pe = state.get('prediction_error', 0.1)
        learning_progress = state.get('learning_progress', 0.0)
        
        # Exploration rises when: PE low (everything known) + still learning progress
        boredom = max(0, 0.5 - pe * 5)  # Niedrige PE → Langeweile
        # v0.7.0: Low explored ratio → more exploration drive (terra incognita!)
        explored = state.get('spatial_explored', 0.0)
        terra_incognita = max(0, 0.3 - explored * 0.5)  # 0% explored → +0.3
        self.state.exploration = (
            self.DECAY * self.state.exploration +
            (1 - self.DECAY) * np.clip(self.BASELINES['exploration'] + boredom + terra_incognita, 0, 1)
        )
        
        # === COMFORT: High when energy spent ===
        energy = state.get('energy_spent', 0.0)
        # Comfort rises with energy expenditure (fatigue)
        fatigue = min(energy / 500.0, 0.5)  # Normalisiert
        self.state.comfort = (
            self.DECAY * self.state.comfort +
            (1 - self.DECAY) * np.clip(self.BASELINES['comfort'] + fatigue, 0, 1)
        )
        
        # === SOCIAL: High when other creature visible ===
        other_visible = state.get('other_creature_visible', False)
        social_signal = 0.5 if other_visible else 0.0
        self.state.social = (
            self.DECAY * self.state.social +
            (1 - self.DECAY) * np.clip(self.BASELINES['social'] + social_signal, 0, 1)
        )
        
        # === PLAY: High when object salience present (ball, toy) ===
        # Biology: Panksepp 1998 — PLAY is a primary emotional system.
        # Play drive rises when: salient object detected (smell/visual),
        # creature is safe (not fallen), and has energy.
        # NOTE: smell_strength is 0.0 on Freenove (no olfactory sensor).
        # Kept for Go2/future hardware with chemical sensors.
        smell_strength = state.get('smell_strength', 0.0)
        ball_salience = state.get('ball_salience', 0.0)
        play_signal = max(smell_strength, ball_salience) * 0.6  # Vision or smell
        if is_fallen:
            play_signal = 0.0  # No play when in danger
        if limb_dead:
            play_signal *= 0.3  # Injured animals don't play much
        self.state.play = (
            self.DECAY * self.state.play +
            (1 - self.DECAY) * np.clip(self.BASELINES['play'] + play_signal, 0, 1)
        )
        
        # === DOMINANT DRIVE ===
        drives = {
            'survival': self.state.survival,
            'exploration': self.state.exploration,
            'comfort': self.state.comfort,
            'social': self.state.social,
            'play': self.state.play,
        }
        self.state.dominant = max(drives, key=drives.get)
        
        self._step_count += 1
        return self.state

    def get_dominant_drive(self) -> str:
        """Winner-take-all → bestimmt GWT-Bias."""
        return self.state.dominant

    def modulate_reward(self, base_reward: float, upright_bonus: float = 0.0,
                        curiosity_bonus: float = 0.0) -> float:
        """
        Reward wird durch aktiven Drive moduliert.
        
        - Survival aktiv → Upright-Bonus verdoppelt
        - Exploration aktiv → Curiosity-Bonus verdoppelt
        - Comfort aktiv → Energiespar-Bonus
        
        Args:
            base_reward: Basis-Reward aus Szenario
            upright_bonus: Aufrecht-Bonus
            curiosity_bonus: Curiosity/Prediction Error Reward
            
        Returns:
            Modulierter Reward
        """
        d = self.state.dominant
        
        modulated = base_reward
        
        if d == 'survival':
            modulated += upright_bonus * 1.5  # Doppelter Upright-Bonus
        elif d == 'exploration':
            modulated += curiosity_bonus * 1.5  # Doppelter Curiosity-Bonus
        elif d == 'comfort':
            # Reward for low motor activity (energy saving)
            modulated += 0.01  # Kleiner Comfort-Bonus
        elif d == 'social':
            # Reward for proximity (becomes relevant in Phase 10-12)
            pass
        
        return modulated

    def get_gwt_bias(self) -> dict:
        """
        Drive moduliert GWT-Competition Bias.
        
        Returns:
            Dict mit Bias pro GWT-Modul
        """
        d = self.state.dominant
        
        # Base: all equal
        bias = {
            'sensory': 1.0,
            'motor': 1.0,
            'predictive': 1.0,
            'error': 1.0,
            'memory': 1.0,
            'social': 1.0,
        }
        
        if d == 'survival':
            bias['sensory'] = 1.5    # Mehr Sensor-Fokus
            bias['error'] = 1.3      # Fehler erkennen
        elif d == 'exploration':
            bias['predictive'] = 1.3  # World Model nutzen
            bias['memory'] = 1.3     # Erinnerungen abrufen
        elif d == 'comfort':
            bias['motor'] = 0.7      # Weniger Motor-Aktivität
        elif d == 'social':
            bias['social'] = 2.0     # Soziale Signale priorisieren
        
        return bias

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        return {
            'survival': round(self.state.survival, 3),
            'exploration': round(self.state.exploration, 3),
            'comfort': round(self.state.comfort, 3),
            'social': round(self.state.social, 3),
            'play': round(self.state.play, 3),
            'dominant': self.state.dominant,
        }
