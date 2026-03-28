"""
MH-FLOCKE — Embodied Metacognition v0.4.1
=========================================
Self-assessment of confidence, learning progress, and consciousness level (PCI).
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque


class EmbodiedMetacognition:
    """
    Metacognition — die Kreatur überwacht ihr eigenes Lernen.
    
    Keine Text-basierte Wissensassessment, sondern:
    - Wie gut sind meine Vorhersagen? (World Model Accuracy)
    - Wie gut verstehe ich meinen Körper? (Body Schema)
    - Lerne ich noch? (Learning Progress = Fitness-Trend)
    - Wie integriert ist mein Erleben? (Consciousness Level)
    """

    # Window sizes for running averages
    ACCURACY_WINDOW = 50
    PROGRESS_WINDOW = 100

    def __init__(self):
        # Tracking-Metriken
        self.world_model_accuracy = 0.5   # 0..1, wie gut Vorhersagen sind
        self.body_schema_confidence = 0.0  # 0..1, Körperverständnis
        self.learning_progress = 0.0       # -1..1, Fitness-Trend
        self.consciousness_level = 0       # 0..5
        
        # History for trend calculation
        self._pe_history = deque(maxlen=self.ACCURACY_WINDOW)
        self._fitness_history = deque(maxlen=self.PROGRESS_WINDOW)
        self._confidence_history = deque(maxlen=50)
        
        # Module aktiv?
        self._modules_active = {
            'world_model': False,
            'body_schema': False,
            'emotions': False,
            'memory': False,
            'drives': False,
            'synaptogenesis': False,
            'theory_of_mind': False,
            'consistency': False,
        }
        
        self._step_count = 0
        
        # Level 8-12 Tracking
        self._behavior_count = 0      # Anzahl verschiedener Behaviors
        self._skill_count = 0         # Anzahl gelernter Skills

    def assess_situation(self, prediction_error: float,
                         body_anomaly: float,
                         body_confidence: float,
                         fitness: float = 0.0,
                         n_episodes: int = 0,
                         modules_active: Optional[dict] = None,
                         behavior_count: int = 0,
                         skill_count: int = 0) -> dict:
        """
        Situationsassessment — wie steht es um die Kreatur?
        
        Args:
            prediction_error: World Model PE (0..1+)
            body_anomaly: Body Schema Anomalie (0..1+)
            body_confidence: Body Schema Confidence (0..1)
            fitness: Aktuelle Fitness
            n_episodes: Episoden im Memory
            modules_active: Welche Module aktiv sind
            
        Returns:
            Dict mit confidence, completeness, learning_progress, consciousness_level
        """
        # Update Module-Status
        if modules_active:
            self._modules_active.update(modules_active)
        
        # Level 8-12 Tracking
        self._behavior_count = behavior_count
        self._skill_count = skill_count
        
        # === World Model Accuracy ===
        self._pe_history.append(prediction_error)
        avg_pe = np.mean(self._pe_history) if self._pe_history else 0.5
        self.world_model_accuracy = max(0, 1.0 - avg_pe * 3)  # PE→Accuracy invertieren
        
        # === Body Schema Confidence ===
        self.body_schema_confidence = body_confidence
        
        # === Learning Progress (Fitness-Trend) ===
        self._fitness_history.append(fitness)
        if len(self._fitness_history) >= 10:
            recent = list(self._fitness_history)
            half = len(recent) // 2
            old_avg = np.mean(recent[:half])
            new_avg = np.mean(recent[half:])
            self.learning_progress = float(np.clip(new_avg - old_avg, -1, 1))
        
        # === Confidence (situation-dependent) ===
        # "I understand this situation"
        confidence = (
            self.world_model_accuracy * 0.4 +
            self.body_schema_confidence * 0.3 +
            min(n_episodes / 100, 1.0) * 0.3  # Erfahrung
        )
        
        # === Completeness ===
        # "Ich habe genug Module aktiv"
        active_count = sum(1 for v in self._modules_active.values() if v)
        completeness = active_count / max(len(self._modules_active), 1)
        
        # === Consciousness Level ===
        self.consciousness_level = self._compute_consciousness_level()
        
        self._step_count += 1
        
        return {
            'confidence': round(float(confidence), 3),
            'completeness': round(float(completeness), 3),
            'learning_progress': round(self.learning_progress, 3),
            'consciousness_level': self.consciousness_level,
            'world_model_accuracy': round(self.world_model_accuracy, 3),
            'body_schema_confidence': round(self.body_schema_confidence, 3),
        }

    def should_explore(self) -> bool:
        """
        Soll die Kreatur explorieren?
        
        Ja, wenn: hohe Confidence + niedriger Learning Progress
        (Schmidhuber: Neugier sinkt wenn nichts Neues zu lernen)
        """
        return (self.world_model_accuracy > 0.6 and 
                abs(self.learning_progress) < 0.05)

    def _compute_consciousness_level(self) -> int:
        """
        Emergentes Bewusstseinslevel basierend auf aktiven Modulen.
        
        Level  0: Aus — Keine Aktivität
        Level  1: Reaktiv — SNN + Reflexe
        Level  2: Aufmerksam — World Model aktiv
        Level  3: Emotional — Emotionen modulieren Verhalten
        Level  4: Zielgerichtet — Goals + Drives aktiv
        Level  5: Selbst-bewusst — Body Schema + Metacognition
        Level  6: Antizipierend — Predictive Coding, niedrige PE
        Level  7: Integriert — Alle Module interagieren
        Level  8: Verhaltensflex — Mehrere Verhaltensweisen
        Level  9: Lernfähig — Skills erlernt + eingefroren
        Level 10: Träumend — Dream-Konsolidierung aktiv
        Level 11: Sozial — Theory of Mind aktiv
        Level 12: Sprachfähig — LLM-Bridge verbindet Erfahrung ↔ Sprache
        """
        m = self._modules_active
        
        if not any(m.values()):
            return 0
        
        level = 1  # Mindestens reaktiv
        
        # Level 2: World Model → versteht Ursache/Wirkung
        if m.get('world_model') and self.world_model_accuracy > 0.3:
            level = 2
        else:
            return level
        
        # Level 3: Emotions modulate behavior
        if m.get('emotions'):
            level = 3
        else:
            return level
        
        # Level 4: Drives + Goals aktiv
        if m.get('drives'):
            level = 4
        else:
            return level
        
        # Level 5: Body Schema + Metacognition → Selbstbewusstsein
        if m.get('body_schema') and self.body_schema_confidence > 0.2:
            level = 5
        else:
            return level
        
        # Level 6: Predictive Coding funktioniert
        if self.world_model_accuracy > 0.5:
            level = 6
        else:
            return level
        
        # Level 7: Everything integrated + coherent
        active_count = sum(1 for v in m.values() if v)
        if active_count >= 6 and m.get('consistency'):
            level = 7
        else:
            return level
        
        # Level 8: Behavior-Repertoire (>= 3 verschiedene Behaviors)
        if m.get('behavior_planner') and self._behavior_count >= 3:
            level = 8
        else:
            return level
        
        # Level 9: Mindestens 1 Skill erlernt
        if self._skill_count >= 1:
            level = 9
        else:
            return level
        
        # Level 10: Dream-Konsolidierung aktiv
        if m.get('dream_engine'):
            level = 10
        else:
            return level
        
        # Level 11: Theory of Mind aktiv
        if m.get('theory_of_mind'):
            level = 11
        else:
            return level
        
        # Level 12: LLM-Bridge — can translate experience to language
        if m.get('language_bridge'):
            level = 12
        
        return level

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        return {
            'world_model_accuracy': round(self.world_model_accuracy, 3),
            'body_schema_confidence': round(self.body_schema_confidence, 3),
            'learning_progress': round(self.learning_progress, 3),
            'consciousness_level': self.consciousness_level,
            'should_explore': self.should_explore(),
            'modules_active': dict(self._modules_active),
        }
