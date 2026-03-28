"""
MH-FLOCKE — Sensorimotor Memory v0.4.1
========================================
Episodic memory for sensorimotor sequences.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class Episode:
    """Ein episodisches Trajektorien-Fragment."""
    sensor_seq: np.ndarray       # (T, n_sensors) — Sensor-Verlauf
    motor_seq: np.ndarray        # (T, n_motors) — Motor-Verlauf
    reward_seq: np.ndarray       # (T,) — Reward-Verlauf
    valence: float               # Emotionale Markierung (Damasio)
    arousal: float               # Erregungslevel
    total_reward: float          # Summe Rewards
    start_step: int              # Globaler Step-Counter
    pattern_hash: int = 0        # Für Deduplizierung


class SensomotorMemory:
    """
    Episodisches Gedächtnis aus Trajektorien-Fragmenten.
    
    Keine externen Dependencies (kein LanceDB, keine Sentence Transformers).
    Speichert kurze Sensor-Motor-Reward Sequenzen mit emotionaler Indexierung.
    """

    def __init__(self, max_episodes: int = 500, fragment_length: int = 20,
                 n_sensors: int = 16, n_motors: int = 5):
        """
        Args:
            max_episodes: Maximale Episoden im Speicher
            fragment_length: Steps pro Fragment
            n_sensors: Sensor-Dimensionen
            n_motors: Motor-Dimensionen
        """
        self.max_episodes = max_episodes
        self.fragment_length = fragment_length
        self.n_sensors = n_sensors
        self.n_motors = n_motors
        
        # Episode storage
        self.episodes: List[Episode] = []
        
        # Aktuelles Recording-Buffer
        self._current_sensors: List[np.ndarray] = []
        self._current_motors: List[np.ndarray] = []
        self._current_rewards: List[float] = []
        self._current_valence = 0.0
        self._current_arousal = 0.0
        
        # Statistiken
        self._total_recorded = 0
        self._total_recalled = 0

    def record_step(self, sensors: list, motors: list, reward: float,
                    valence: float = 0.0, arousal: float = 0.5):
        """
        Einen Step aufzeichnen.
        
        Args:
            sensors: Aktuelle Sensor-Werte
            motors: Aktuelle Motor-Befehle
            reward: Aktueller Reward
            valence: Emotionale Valence (-1..1)
            arousal: Erregungslevel (0..1)
        """
        self._current_sensors.append(np.array(sensors[:self.n_sensors], dtype=np.float32))
        self._current_motors.append(np.array(motors[:self.n_motors], dtype=np.float32))
        self._current_rewards.append(float(reward))
        
        # Running average der Emotion
        alpha = 0.2
        self._current_valence = (1 - alpha) * self._current_valence + alpha * valence
        self._current_arousal = (1 - alpha) * self._current_arousal + alpha * arousal
        
        # Fragment voll? → speichern
        if len(self._current_sensors) >= self.fragment_length:
            self._store_fragment()

    def _store_fragment(self):
        """Aktuelles Fragment als Episode speichern."""
        if len(self._current_sensors) < min(3, self.fragment_length):  # Mindestlänge
            self._clear_buffer()
            return
        
        sensor_seq = np.stack(self._current_sensors)
        motor_seq = np.stack(self._current_motors)
        reward_seq = np.array(self._current_rewards, dtype=np.float32)
        
        episode = Episode(
            sensor_seq=sensor_seq,
            motor_seq=motor_seq,
            reward_seq=reward_seq,
            valence=self._current_valence,
            arousal=self._current_arousal,
            total_reward=float(reward_seq.sum()),
            start_step=self._total_recorded,
            pattern_hash=hash(sensor_seq[:3].tobytes()),  # Grober Hash
        )
        
        self.episodes.append(episode)
        self._total_recorded += len(self._current_sensors)
        
        # Storage management: remove emotionally unimportant episodes
        if len(self.episodes) > self.max_episodes:
            self._evict_least_important()
        
        self._clear_buffer()

    def _evict_least_important(self):
        """Entferne die emotional unwichtigste Episode (behalte mind. 1)."""
        if len(self.episodes) <= 1:
            return
        
        # Importance = |valence| + arousal + |total_reward|
        importances = [
            abs(ep.valence) + ep.arousal + abs(ep.total_reward) * 0.5
            for ep in self.episodes
        ]
        least_idx = int(np.argmin(importances))
        self.episodes.pop(least_idx)

    def _clear_buffer(self):
        """Recording-Buffer leeren."""
        self._current_sensors.clear()
        self._current_motors.clear()
        self._current_rewards.clear()
        self._current_valence = 0.0
        self._current_arousal = 0.5

    def recall_similar(self, current_sensors: np.ndarray, k: int = 3) -> List[Episode]:
        """
        Finde ähnliche Erfahrungen via Cosine Similarity auf Sensor-Patterns.
        
        Args:
            current_sensors: Aktuelle Sensor-Werte (1D)
            k: Anzahl zurückgegebener Episoden
            
        Returns:
            Liste der k ähnlichsten Episoden
        """
        if not self.episodes:
            return []
        
        query = np.array(current_sensors[:self.n_sensors], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query = query / query_norm
        
        similarities = []
        for ep in self.episodes:
            # Vergleiche mit dem Mittelwert der Sensor-Sequenz
            ep_mean = ep.sensor_seq.mean(axis=0)
            ep_norm = np.linalg.norm(ep_mean)
            if ep_norm < 1e-8:
                similarities.append(-1.0)
                continue
            sim = float(np.dot(query, ep_mean / ep_norm))
            # Emotional wichtige Episoden haben leichten Bonus
            emotional_bonus = (abs(ep.valence) + ep.arousal) * 0.1
            similarities.append(sim + emotional_bonus)
        
        # Top-K
        top_indices = np.argsort(similarities)[-k:][::-1]
        self._total_recalled += 1
        
        return [self.episodes[i] for i in top_indices if similarities[i] > 0]

    def recall_by_emotion(self, target_valence: float, k: int = 3) -> List[Episode]:
        """
        Finde Episoden mit ähnlicher emotionaler Signatur.
        
        Nützlich für: "Wann war ich schon mal so ängstlich?"
        """
        if not self.episodes:
            return []
        
        distances = [abs(ep.valence - target_valence) for ep in self.episodes]
        top_indices = np.argsort(distances)[:k]
        return [self.episodes[i] for i in top_indices]

    def consolidate(self) -> Dict:
        """
        Dream-Phase Konsolidierung.
        
        - Häufige Sensor-Patterns → können zu Konzepten werden (Synaptogenesis)
        - Seltene aber hochvalente Episoden bleiben
        - Redundante Episoden werden zusammengefasst
        
        Returns:
            Dict mit Konsolidierungs-Statistiken
        """
        if len(self.episodes) < 10:
            return {'consolidated': 0, 'kept': len(self.episodes)}
        
        # 1. Find redundant episodes (similar pattern hash)
        hash_groups: Dict[int, List[int]] = {}
        for i, ep in enumerate(self.episodes):
            h = ep.pattern_hash
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(i)
        
        # 2. From redundant groups keep only the emotionally strongest
        to_remove = set()
        for h, indices in hash_groups.items():
            if len(indices) > 3:
                # Sortiere nach Importance, behalte Top-2
                scored = [(i, abs(self.episodes[i].valence) + self.episodes[i].arousal)
                          for i in indices]
                scored.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in scored[2:]:
                    to_remove.add(idx)
        
        # 3. Entferne
        if to_remove:
            self.episodes = [ep for i, ep in enumerate(self.episodes) 
                           if i not in to_remove]
        
        # 4. Extract frequent patterns for synaptogenesis
        frequent_patterns = []
        if self.episodes:
            all_means = np.stack([ep.sensor_seq.mean(axis=0) for ep in self.episodes])
            # Simple K-means-like clustering (greedy)
            # TODO: Kann in Zukunft sophistizierter werden
            if len(all_means) > 5:
                centroid = all_means.mean(axis=0)
                frequent_patterns.append(centroid)
        
        return {
            'consolidated': len(to_remove),
            'kept': len(self.episodes),
            'frequent_patterns': len(frequent_patterns),
        }

    def get_state(self) -> dict:
        """Für Dashboard/Logging."""
        avg_valence = np.mean([ep.valence for ep in self.episodes]) if self.episodes else 0
        return {
            'n_episodes': len(self.episodes),
            'total_recorded_steps': self._total_recorded,
            'total_recalls': self._total_recalled,
            'avg_valence': round(float(avg_valence), 3),
            'buffer_size': len(self._current_sensors),
        }
