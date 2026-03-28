"""
MH-FLOCKE — Creature State v0.4.1
========================================
Runtime state tracking for creature instances.
"""

import torch
import numpy as np
import json
import copy
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class CreatureState:
    """Persistenter Zustand einer Kreatur über Generationen."""

    # SNN Weights (die gelernten synaptischen Gewichte)
    snn_weight_values: Optional[List[float]] = None
    snn_weight_indices: Optional[List[List[int]]] = None  # [src_indices, tgt_indices]
    snn_thresholds: Optional[List[float]] = None

    # Evolved Plasticity Parameters
    plasticity_genome: Optional[Dict] = None

    # Body schema (learned body representation)
    body_confidence: float = 0.5
    body_prediction_weights: Optional[List[float]] = None

    # Episodic Memory (komprimiert: nur Top-K Episoden)
    episodic_memories: List[Dict] = field(default_factory=list)
    max_episodes: int = 50

    # Fitness history (for inheritance: how good was this creature?)
    fitness_history: List[float] = field(default_factory=list)
    best_fitness: float = 0.0
    generation_born: int = 0
    n_evaluations: int = 0

    # Metadata
    parent_state_id: Optional[int] = None
    lineage_depth: int = 0

    def to_dict(self) -> dict:
        """Serialisierung für Checkpoint."""
        return {
            'snn_weight_values': self.snn_weight_values,
            'snn_weight_indices': self.snn_weight_indices,
            'snn_thresholds': self.snn_thresholds,
            'plasticity_genome': self.plasticity_genome,
            'body_confidence': float(self.body_confidence),
            'body_prediction_weights': self.body_prediction_weights,
            'episodic_memories': self.episodic_memories[:self.max_episodes],
            'fitness_history': [float(f) for f in self.fitness_history],
            'best_fitness': float(self.best_fitness),
            'generation_born': int(self.generation_born),
            'n_evaluations': int(self.n_evaluations),
            'parent_state_id': self.parent_state_id,
            'lineage_depth': int(self.lineage_depth),
        }

    @staticmethod
    def from_dict(d: dict) -> 'CreatureState':
        """Deserialisierung."""
        state = CreatureState()
        state.snn_weight_values = d.get('snn_weight_values')
        state.snn_weight_indices = d.get('snn_weight_indices')
        state.snn_thresholds = d.get('snn_thresholds')
        state.plasticity_genome = d.get('plasticity_genome')
        state.body_confidence = d.get('body_confidence', 0.5)
        state.body_prediction_weights = d.get('body_prediction_weights')
        state.episodic_memories = d.get('episodic_memories', [])
        state.fitness_history = d.get('fitness_history', [])
        state.best_fitness = d.get('best_fitness', 0.0)
        state.generation_born = d.get('generation_born', 0)
        state.n_evaluations = d.get('n_evaluations', 0)
        state.parent_state_id = d.get('parent_state_id')
        state.lineage_depth = d.get('lineage_depth', 0)
        return state


class CreatureStateManager:
    """Verwaltet persistente Zustände für eine Population."""

    def __init__(self, max_states: int = 100):
        self.states: Dict[int, CreatureState] = {}  # genome_id → State
        self.max_states = max_states

    def capture_state(self, genome_id: int, creature) -> CreatureState:
        """
        Extrahiert den aktuellen Zustand einer Kreatur nach Evaluation.

        Args:
            genome_id: ID des Genoms
            creature: MuJoCoCreature Instanz
        """
        state = CreatureState()

        # SNN Weights extrahieren
        snn = creature.snn
        if hasattr(snn, '_weight_values') and snn._weight_values is not None:
            state.snn_weight_values = snn._weight_values.cpu().tolist()
            if hasattr(snn, '_weight_indices') and snn._weight_indices is not None:
                state.snn_weight_indices = [
                    snn._weight_indices[0].cpu().tolist(),
                    snn._weight_indices[1].cpu().tolist(),
                ]
            if hasattr(snn, '_thresholds') and snn._thresholds is not None:
                state.snn_thresholds = snn._thresholds.cpu().tolist()

        # Plasticity Genome
        if creature.genome.plasticity_genome:
            state.plasticity_genome = copy.deepcopy(creature.genome.plasticity_genome)

        # Body Schema
        if creature.brain and hasattr(creature.brain, 'body_schema'):
            bs = creature.brain.body_schema
            state.body_confidence = getattr(bs, 'confidence', 0.5)

        # Episodic Memory (Top-K nach Relevanz)
        if creature.brain and hasattr(creature.brain, 'memory'):
            mem = creature.brain.memory
            if hasattr(mem, 'episodes') and mem.episodes:
                # Nur die wichtigsten Episoden behalten
                episodes = mem.episodes[-state.max_episodes:]
                state.episodic_memories = [
                    {'reward': float(ep.get('reward', 0)),
                     'step': int(ep.get('step', 0))}
                    for ep in episodes
                    if isinstance(ep, dict)
                ]

        self.states[genome_id] = state
        return state

    def restore_state(self, genome_id: int, creature) -> bool:
        """
        Stellt gespeicherten Zustand in einer Kreatur wieder her.

        Returns:
            True wenn State vorhanden und wiederhergestellt.
        """
        state = self.states.get(genome_id)
        if state is None:
            return False

        snn = creature.snn

        # SNN Weights wiederherstellen
        if state.snn_weight_values and state.snn_weight_indices:
            try:
                current_n = len(snn._weight_values)
                saved_n = len(state.snn_weight_values)

                if current_n == saved_n:
                    # Direct transfer
                    snn._weight_values = torch.tensor(
                        state.snn_weight_values,
                        device=snn.device, dtype=snn.dtype
                    )
                    snn._rebuild_sparse_weights()
                elif saved_n > 0:
                    # Partial transfer (morphology may have changed)
                    n_copy = min(current_n, saved_n)
                    snn._weight_values[:n_copy] = torch.tensor(
                        state.snn_weight_values[:n_copy],
                        device=snn.device, dtype=snn.dtype
                    )
                    snn._rebuild_sparse_weights()
            except Exception:
                pass  # Graceful fallback

        # Thresholds
        if state.snn_thresholds:
            try:
                n_neurons = snn.config.n_neurons
                saved_n = len(state.snn_thresholds)
                n_copy = min(n_neurons, saved_n)
                snn._thresholds[:n_copy] = torch.tensor(
                    state.snn_thresholds[:n_copy],
                    device=snn.device, dtype=snn.dtype
                )
            except Exception:
                pass

        return True

    def inherit_state(self, child_id: int, parent_id: int,
                       generation: int, mutation_noise: float = 0.05) -> CreatureState:
        """
        Kind erbt State vom Elternteil (mit leichter Perturbation).

        Args:
            child_id: Genome-ID des Kindes
            parent_id: Genome-ID des Elternteils
            generation: Aktuelle Generation
            mutation_noise: Standardabweichung der Gewichts-Perturbation
        """
        parent_state = self.states.get(parent_id)
        if parent_state is None:
            # No state to inherit -> new empty state
            child_state = CreatureState(generation_born=generation)
            self.states[child_id] = child_state
            return child_state

        child_state = CreatureState()
        child_state.generation_born = generation
        child_state.parent_state_id = parent_id
        child_state.lineage_depth = parent_state.lineage_depth + 1

        # SNN Weights mit Noise
        if parent_state.snn_weight_values:
            weights = np.array(parent_state.snn_weight_values)
            noise = np.random.normal(0, mutation_noise, weights.shape)
            child_state.snn_weight_values = (weights + noise).tolist()
            child_state.snn_weight_indices = copy.deepcopy(parent_state.snn_weight_indices)

        if parent_state.snn_thresholds:
            thresholds = np.array(parent_state.snn_thresholds)
            noise = np.random.normal(0, mutation_noise * 0.1, thresholds.shape)
            child_state.snn_thresholds = np.clip(thresholds + noise, 0.1, 5.0).tolist()

        # Plasticity
        child_state.plasticity_genome = copy.deepcopy(parent_state.plasticity_genome)

        # Body Schema (teilweise erben)
        child_state.body_confidence = parent_state.body_confidence * 0.8  # Leicht reduziert

        # Episodic Memory (Top-K erben)
        child_state.episodic_memories = copy.deepcopy(
            parent_state.episodic_memories[:25]  # Hälfte erben
        )

        self.states[child_id] = child_state
        return child_state

    def cleanup(self, active_ids: List[int]):
        """Entfernt States von Genomen die nicht mehr in der Population sind."""
        to_remove = [gid for gid in self.states if gid not in active_ids]
        for gid in to_remove:
            del self.states[gid]

    def save(self, path: str):
        """Speichert alle States."""
        data = {str(k): v.to_dict() for k, v in self.states.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Lädt gespeicherte States."""
        with open(path) as f:
            data = json.load(f)
        self.states = {int(k): CreatureState.from_dict(v) for k, v in data.items()}
