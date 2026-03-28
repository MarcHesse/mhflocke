"""
MH-FLOCKE — Synaptogenesis v0.4.1
========================================
Activity-dependent synapse formation with astrocyte gating.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.brain.snn_controller import SNNController


@dataclass
class SynaptogenesisConfig:
    """Konfiguration für SNN ↔ Graph Brücke."""
    consolidation_threshold: int = 5
    similarity_threshold: float = 0.7
    max_concepts: int = 1000
    pattern_window: int = 50
    pattern_dimensions: int = 64
    retrieval_strength: float = 0.5
    retrieval_top_k: int = 5
    device: str = 'cpu'


class ConceptGraph:
    """
    Symbolischer Konzept-Graph (leichtgewichtig, standalone).

    Knoten: Konzepte mit Pattern, Valence, Properties
    Kanten: Relationen zwischen Konzepten
    """

    def __init__(self, max_concepts: int = 1000):
        self.max_concepts = max_concepts
        self._concepts: Dict[int, Dict] = {}
        self._relations: List[Tuple[int, int, str, float]] = []
        self._next_id = 0

    def add_concept(self, label: str, pattern: torch.Tensor,
                    valence: float = 0.0,
                    properties: Optional[Dict] = None) -> int:
        """Fügt Konzept hinzu. Returns concept_id."""
        if len(self._concepts) >= self.max_concepts:
            # Remove oldest, least activated
            min_id = min(self._concepts, key=lambda k: self._concepts[k]['activation_count'])
            del self._concepts[min_id]

        cid = self._next_id
        self._next_id += 1

        self._concepts[cid] = {
            'id': cid,
            'label': label,
            'pattern': pattern.detach().cpu().clone(),
            'valence': valence,
            'properties': properties or {},
            'activation_count': 1,
        }
        return cid

    def add_relation(self, source_id: int, target_id: int,
                     relation_type: str, weight: float = 1.0):
        """Fügt Relation hinzu."""
        if source_id in self._concepts and target_id in self._concepts:
            self._relations.append((source_id, target_id, relation_type, weight))

    def find_similar(self, pattern: torch.Tensor,
                     top_k: int = 5) -> List[Tuple[int, float]]:
        """Findet ähnlichste Konzepte (Kosinus-Ähnlichkeit)."""
        if not self._concepts:
            return []

        pattern_flat = pattern.detach().cpu().float().flatten()
        p_norm = torch.norm(pattern_flat)
        if p_norm < 1e-8:
            return []

        similarities = []
        for cid, concept in self._concepts.items():
            c_flat = concept['pattern'].flatten()
            c_norm = torch.norm(c_flat)
            if c_norm < 1e-8:
                continue

            # Pad to equal length
            min_len = min(len(pattern_flat), len(c_flat))
            sim = torch.dot(pattern_flat[:min_len], c_flat[:min_len]) / (p_norm * c_norm + 1e-8)
            similarities.append((cid, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def get_concept(self, concept_id: int) -> Optional[Dict]:
        """Gibt Konzept-Details zurück."""
        c = self._concepts.get(concept_id)
        if c is None:
            return None
        return {k: v for k, v in c.items() if k != 'pattern'}

    def get_related(self, concept_id: int) -> List[Tuple[int, str, float]]:
        """Verwandte Konzepte."""
        result = []
        for src, tgt, rtype, w in self._relations:
            if src == concept_id:
                result.append((tgt, rtype, w))
            elif tgt == concept_id:
                result.append((src, rtype, w))
        return result

    def update_concept(self, concept_id: int, properties: Dict = None,
                       valence_delta: float = 0.0):
        """Aktualisiert ein Konzept."""
        if concept_id not in self._concepts:
            return
        c = self._concepts[concept_id]
        c['activation_count'] += 1
        c['valence'] = np.clip(c['valence'] + valence_delta, -1.0, 1.0)
        if properties:
            for k, v in properties.items():
                # Exponential moving average (only for numeric values)
                if isinstance(v, (int, float)):
                    if k in c['properties'] and isinstance(c['properties'][k], (int, float)):
                        c['properties'][k] = 0.7 * c['properties'][k] + 0.3 * v
                    else:
                        c['properties'][k] = v
                else:
                    c['properties'][k] = v

    def size(self) -> int:
        return len(self._concepts)

    def to_dict(self) -> Dict:
        """Serialisierung."""
        concepts = {}
        for cid, c in self._concepts.items():
            concepts[cid] = {
                'label': c['label'],
                'pattern': c['pattern'].tolist(),
                'valence': c['valence'],
                'properties': c['properties'],
                'activation_count': c['activation_count'],
            }
        return {
            'concepts': concepts,
            'relations': self._relations,
            'next_id': self._next_id,
        }

    def from_dict(self, data: Dict):
        """Deserialisierung."""
        self._concepts = {}
        for cid_str, c in data.get('concepts', {}).items():
            cid = int(cid_str)
            self._concepts[cid] = {
                'id': cid,
                'label': c['label'],
                'pattern': torch.tensor(c['pattern'], dtype=torch.float32),
                'valence': c['valence'],
                'properties': c['properties'],
                'activation_count': c['activation_count'],
            }
        self._relations = [tuple(r) for r in data.get('relations', [])]
        self._next_id = data.get('next_id', 0)


class PatternExtractor:
    """Extrahiert komprimierte Patterns aus SNN-Aktivität."""

    def __init__(self, n_neurons: int, pattern_dimensions: int = 64,
                 device: str = 'cpu'):
        self.n_neurons = n_neurons
        self.pattern_dimensions = pattern_dimensions
        self.device = device

        # Random projection: n_neurons -> pattern_dimensions (fixed)
        torch.manual_seed(42)
        self._projection = torch.randn(
            pattern_dimensions, min(n_neurons, 10000),
            device=device, dtype=torch.float32,
        ) * 0.01

    def extract(self, spike_window: torch.Tensor) -> torch.Tensor:
        """
        Extrahiert Pattern aus Spike-Fenster [timesteps, n_neurons].
        Returns: [pattern_dimensions] normalisiert.
        """
        # Mean firing rate per neuron
        rates = spike_window.float().mean(dim=0)  # [n_neurons]

        # Truncate to projection size
        n = self._projection.shape[1]
        rates_trimmed = rates[:n].to(self.device)

        # Projizieren
        pattern = self._projection @ rates_trimmed

        # Normalisieren
        norm = torch.norm(pattern)
        if norm > 1e-8:
            pattern = pattern / norm

        return pattern

    def similarity(self, pattern_a: torch.Tensor,
                   pattern_b: torch.Tensor) -> float:
        """Kosinus-Ähnlichkeit."""
        a = pattern_a.flatten().float()
        b = pattern_b.flatten().float()
        min_len = min(len(a), len(b))
        dot = torch.dot(a[:min_len], b[:min_len])
        na = torch.norm(a[:min_len])
        nb = torch.norm(b[:min_len])
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(dot / (na * nb))


class ExperienceBuffer:
    """Puffer für ungefilterte Erfahrungen."""

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._buffer: List[Dict] = []

    def record(self, pattern: torch.Tensor, context: Dict,
               valence: float):
        """Speichert Erfahrung."""
        self._buffer.append({
            'pattern': pattern.detach().cpu().clone(),
            'context': context,
            'valence': valence,
        })
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def get_frequent_patterns(self, min_count: int = 3,
                               similarity_threshold: float = 0.7) -> List[Dict]:
        """Findet häufig wiederkehrende Patterns durch Clustering."""
        if len(self._buffer) < min_count:
            return []

        # Einfaches Clustering: Greedy nearest-neighbor
        clusters: List[Dict] = []

        for exp in self._buffer:
            p = exp['pattern'].flatten().float()
            p_norm = torch.norm(p)
            if p_norm < 1e-8:
                continue

            matched = False
            for cluster in clusters:
                c = cluster['centroid'].flatten().float()
                c_norm = torch.norm(c)
                if c_norm < 1e-8:
                    continue
                min_len = min(len(p), len(c))
                sim = float(torch.dot(p[:min_len], c[:min_len]) / (p_norm * c_norm + 1e-8))
                if sim > similarity_threshold:
                    # Update centroid (running average)
                    n = cluster['count']
                    cluster['centroid'] = (cluster['centroid'] * n + exp['pattern']) / (n + 1)
                    cluster['count'] += 1
                    cluster['total_valence'] += exp['valence']
                    # Merge context
                    for k, v in exp['context'].items():
                        if k not in cluster['contexts']:
                            cluster['contexts'][k] = []
                        cluster['contexts'][k].append(v)
                    matched = True
                    break

            if not matched:
                clusters.append({
                    'centroid': exp['pattern'].clone(),
                    'count': 1,
                    'total_valence': exp['valence'],
                    'contexts': {k: [v] for k, v in exp['context'].items()},
                })

        # Filter to frequent ones
        frequent = [c for c in clusters if c['count'] >= min_count]

        # Bestimme dominante Eigenschaften
        for cluster in frequent:
            cluster['mean_valence'] = cluster['total_valence'] / cluster['count']
            # Most frequent context per key
            resolved = {}
            for key, values in cluster['contexts'].items():
                if isinstance(values[0], (int, float)):
                    resolved[key] = float(np.mean(values))
                else:
                    from collections import Counter
                    # Convert unhashable types (lists) to tuples for Counter
                    try:
                        hashable = [tuple(v) if isinstance(v, list) else v for v in values]
                        resolved[key] = Counter(hashable).most_common(1)[0][0]
                    except TypeError:
                        resolved[key] = str(values[0]) if values else 'unknown'
            cluster['dominant_context'] = resolved

        return frequent

    def clear(self):
        self._buffer = []

    def size(self) -> int:
        return len(self._buffer)


class Synaptogenesis:
    """
    Hauptklasse: SNN ↔ Knowledge Graph Brücke.

    1. consolidate(): Dream-Modus — SNN-Erfahrungen → Graph-Konzepte
    2. retrieve(): Echtzeit — Graph-Konzepte → SNN apikaler Kontext
    """

    def __init__(self, config: SynaptogenesisConfig,
                 snn: SNNController,
                 multi_compartment=None):
        self.config = config
        self.snn = snn
        self.mc = multi_compartment

        self.graph = ConceptGraph(max_concepts=config.max_concepts)
        self.extractor = PatternExtractor(
            n_neurons=snn.config.n_neurons,
            pattern_dimensions=config.pattern_dimensions,
            device=config.device,
        )
        self.buffer = ExperienceBuffer()

        # Spike window for pattern extraction
        self._spike_window: List[torch.Tensor] = []

    def record_experience(self, context: Dict, valence: float):
        """Zeichnet aktuelle SNN-Aktivität als Erfahrung auf."""
        # Spike-Fenster → Pattern
        if len(self._spike_window) >= 3:
            window = torch.stack(self._spike_window[-self.config.pattern_window:])
            pattern = self.extractor.extract(window)
            self.buffer.record(pattern, context, valence)

    def observe_spikes(self, spikes: torch.Tensor):
        """Fügt Spikes zum Beobachtungsfenster hinzu."""
        self._spike_window.append(spikes.detach().cpu())
        if len(self._spike_window) > self.config.pattern_window * 2:
            self._spike_window = self._spike_window[-self.config.pattern_window:]

    def consolidate(self) -> Dict:
        """Konsolidierung: Erfahrungen → Konzepte."""
        frequent = self.buffer.get_frequent_patterns(
            min_count=self.config.consolidation_threshold,
            similarity_threshold=self.config.similarity_threshold,
        )

        n_new = 0
        n_updated = 0
        n_relations = 0

        for cluster in frequent:
            pattern = cluster['centroid']
            valence = cluster['mean_valence']
            context = cluster.get('dominant_context', {})

            # Check if similar concept exists
            similar = self.graph.find_similar(pattern, top_k=1)
            if similar and similar[0][1] > self.config.similarity_threshold:
                # Update bestehendes Konzept
                cid = similar[0][0]
                self.graph.update_concept(cid, properties=context, valence_delta=valence * 0.1)
                n_updated += 1
            else:
                # Neues Konzept
                label = context.get('object_type', context.get('event', 'unknown'))
                if isinstance(label, list):
                    label = str(label[0]) if label else 'unknown'
                cid = self.graph.add_concept(str(label), pattern, valence, context)
                n_new += 1

                # Relationen zu ähnlichen Konzepten
                for other_id, sim in self.graph.find_similar(pattern, top_k=3):
                    if other_id != cid and sim > 0.3:
                        self.graph.add_relation(cid, other_id, 'similar', float(sim))
                        n_relations += 1

        return {
            'n_new_concepts': n_new,
            'n_updated': n_updated,
            'n_relations_added': n_relations,
            'graph_size': self.graph.size(),
        }

    def retrieve(self, current_context: Dict = None) -> torch.Tensor:
        """Abruf: Kontext → Apikaler Modulations-Vektor."""
        n = self.snn.config.n_neurons
        apical = torch.zeros(n, device=self.config.device)

        if len(self._spike_window) < 3:
            return apical

        # Aktuelles Pattern
        window = torch.stack(self._spike_window[-min(20, len(self._spike_window)):])
        current_pattern = self.extractor.extract(window)

        # Ähnliche Konzepte finden
        similar = self.graph.find_similar(current_pattern, top_k=self.config.retrieval_top_k)

        if not similar:
            return apical

        # Konzept-Patterns als apikalen Kontext mischen
        for cid, sim in similar:
            concept = self.graph._concepts.get(cid)
            if concept is None:
                continue

            c_pattern = concept['pattern'].to(self.config.device)
            valence = concept['valence']

            # Expandiere Pattern zu Neuronen-Dimension
            strength = self.config.retrieval_strength * sim
            n_fill = min(len(c_pattern), n)
            apical[:n_fill] += c_pattern[:n_fill] * strength * (1.0 + valence)

            # Aktivierungszähler erhöhen
            concept['activation_count'] += 1

        return apical

    def categorize_object(self, object_type: str,
                           interaction_history: List[Dict]) -> Dict:
        """Kategorisiert Objekt basierend auf Interaktions-Historie."""
        properties = {}

        for interaction in interaction_history:
            for key, value in interaction.items():
                if isinstance(value, (int, float)):
                    if key not in properties:
                        properties[key] = []
                    properties[key].append(float(value))

        # Mittelwerte
        result = {}
        for key, values in properties.items():
            result[key] = float(np.mean(values))

        result['object_type'] = object_type
        result['n_interactions'] = len(interaction_history)
        result['confidence'] = min(1.0, len(interaction_history) / 10.0)

        return result

    def get_stats(self) -> Dict:
        return {
            'graph_size': self.graph.size(),
            'buffer_size': self.buffer.size(),
            'spike_window_length': len(self._spike_window),
        }
