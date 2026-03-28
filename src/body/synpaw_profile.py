"""
MH-FLOCKE — Synpaw Profile v0.4.1
========================================
Morphology profile loader for creature configurations.
"""

import json
import copy
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SensorConfig:
    """Konfiguration eines Sensors am Synpaw."""
    name: str                       # z.B. 'nose', 'whiskers', 'ear_left'
    sensor_type: str                # 'chemical', 'touch', 'audio', 'visual'
    parent_segment: str             # An welchem Segment befestigt
    sensitivity: float = 0.5        # 0.0 - 1.0
    range: float = 1.0              # Reichweite in Metern
    active: bool = True             # Ein/Aus-schaltbar

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'sensor_type': self.sensor_type,
            'parent_segment': self.parent_segment,
            'sensitivity': self.sensitivity,
            'range': self.range,
            'active': self.active,
        }

    @staticmethod
    def from_dict(d: dict) -> 'SensorConfig':
        return SensorConfig(**d)


@dataclass
class SynpawProfile:
    """
    Persistentes Profil eines benannten Synpaw.
    
    Speichert alles was einen Synpaw ausmacht:
    - Name und Identität
    - Genome (Morphologie)
    - CreatureState (gelerntes Wissen)
    - Sensor-Konfiguration (erweiterbar)
    - Training-Historie
    """
    
    # Identity
    name: str                                   # Deutscher Hundename
    template: str = 'synpaw'                    # MJCF Template
    created_at: str = ''                        # ISO Timestamp
    description: str = ''                       # Freitext
    
    # Genome (Morphologie) — als dict serialisiert
    genome_data: Optional[Dict] = None
    
    # CreatureState (Lernstand) — als dict serialisiert
    state_data: Optional[Dict] = None
    
    # Sensors (extensible)
    sensors: List[SensorConfig] = field(default_factory=list)
    
    # Training-Historie
    training_log: List[Dict] = field(default_factory=list)
    total_generations: int = 0
    total_steps: int = 0
    best_fitness: float = 0.0
    
    # Abstammung
    parent_name: Optional[str] = None           # Von wem abstammend
    children: List[str] = field(default_factory=list)
    
    # Tags for organization
    tags: List[str] = field(default_factory=list)

    # ================================================================
    # CREATION
    # ================================================================

    @staticmethod
    def create(name: str, template: str = 'synpaw',
               description: str = '',
               genome=None) -> 'SynpawProfile':
        """
        Neuen Synpaw erstellen.
        
        Args:
            name: Deutscher Hundename (z.B. "Mogli", "Bella", "Rex")
            template: 'synpaw', 'mogli', 'quadruped', etc.
            description: Freitext-Beschreibung
            genome: Optional Genome-Objekt, sonst wird Template-Default verwendet
        """
        profile = SynpawProfile(
            name=name,
            template=template if template != 'mogli' else 'synpaw',
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            description=description or f"Synpaw '{name}'",
        )
        
        if genome is not None:
            # Lazy import to avoid circular dependencies
            from src.body.genome import GenomeSerializer
            profile.genome_data = GenomeSerializer.to_dict(genome)
        
        # Default sensors: eyes + base sensors
        profile.sensors = [
            SensorConfig('eye_left', 'visual', 'head', 0.7, 2.0),
            SensorConfig('eye_right', 'visual', 'head', 0.7, 2.0),
            SensorConfig('gyro', 'balance', 'torso', 1.0, 0.0),
            SensorConfig('accel', 'motion', 'torso', 1.0, 0.0),
        ]
        
        return profile

    # ================================================================
    # SENSOR MANAGEMENT
    # ================================================================

    def add_sensor(self, name: str, sensor_type: str,
                   parent_segment: str = 'head',
                   sensitivity: float = 0.5,
                   range: float = 1.0) -> SensorConfig:
        """
        Sensor hinzufügen.
        
        Beispiele:
            profile.add_sensor('nose', 'chemical', 'head', 0.8, 3.0)
            profile.add_sensor('whiskers', 'touch', 'head', 0.6, 0.1)
            profile.add_sensor('ear_left', 'audio', 'head', 0.7, 5.0)
        """
        # Duplikat-Check
        if any(s.name == name for s in self.sensors):
            raise ValueError(f"Sensor '{name}' existiert bereits bei {self.name}")
        
        sensor = SensorConfig(name, sensor_type, parent_segment,
                              sensitivity, range)
        self.sensors.append(sensor)
        return sensor

    def remove_sensor(self, name: str) -> bool:
        """Sensor entfernen."""
        before = len(self.sensors)
        self.sensors = [s for s in self.sensors if s.name != name]
        return len(self.sensors) < before

    def get_sensor(self, name: str) -> Optional[SensorConfig]:
        """Sensor by name."""
        return next((s for s in self.sensors if s.name == name), None)

    def get_active_sensors(self) -> List[SensorConfig]:
        """Nur aktive Sensoren."""
        return [s for s in self.sensors if s.active]

    # ================================================================
    # LEARNING STATE MANAGEMENT
    # ================================================================

    def set_state(self, creature_state) -> None:
        """CreatureState-Objekt speichern."""
        if hasattr(creature_state, 'to_dict'):
            self.state_data = creature_state.to_dict()
        elif isinstance(creature_state, dict):
            self.state_data = creature_state
    
    def get_state(self):
        """CreatureState-Objekt zurückgeben."""
        if self.state_data is None:
            return None
        from src.body.creature_state import CreatureState
        return CreatureState.from_dict(self.state_data)

    def reset_weights(self) -> None:
        """SNN-Gewichte löschen — Kreatur vergisst motorische Fähigkeiten."""
        if self.state_data:
            self.state_data['snn_weight_values'] = None
            self.state_data['snn_weight_indices'] = None
            self.state_data['snn_thresholds'] = None
            self._log('reset_weights', 'SNN weights cleared')

    def reset_memory(self) -> None:
        """Episodische Erinnerungen löschen."""
        if self.state_data:
            self.state_data['episodic_memories'] = []
            self._log('reset_memory', 'Episodic memories cleared')

    def reset_body(self) -> None:
        """Körper-Schema zurücksetzen."""
        if self.state_data:
            self.state_data['body_confidence'] = 0.5
            self.state_data['body_prediction_weights'] = None
            self._log('reset_body', 'Body schema reset')

    def reset_all(self) -> None:
        """Komplett-Reset — Kreatur startet bei Null."""
        self.state_data = None
        self._log('reset_all', 'Full learning state cleared')

    # ================================================================
    # TRAINING LOG
    # ================================================================

    def log_training(self, generations: int, best_fitness: float,
                     scenario: str = '', notes: str = '') -> None:
        """Training-Session dokumentieren."""
        self.total_generations += generations
        self.best_fitness = max(self.best_fitness, best_fitness)
        self._log('training', f"Gen +{generations}, Best: {best_fitness:.3f}",
                  extra={'scenario': scenario, 'generations': generations,
                         'best_fitness': best_fitness, 'notes': notes})

    def _log(self, event: str, message: str, extra: Dict = None) -> None:
        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'event': event,
            'message': message,
        }
        if extra:
            entry.update(extra)
        self.training_log.append(entry)

    # ================================================================
    # FAMILY / LINEAGE
    # ================================================================

    def create_child(self, child_name: str, 
                     mutate_genome: bool = True,
                     inherit_state: bool = True) -> 'SynpawProfile':
        """
        Kind-Synpaw erstellen.
        
        Args:
            child_name: Name des Kindes
            mutate_genome: Genome mutieren?
            inherit_state: Lernstand (teilweise) erben?
        """
        child = SynpawProfile(
            name=child_name,
            template=self.template,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            description=f"Kind von {self.name}",
            parent_name=self.name,
            sensors=copy.deepcopy(self.sensors),
        )
        
        # Inherit genome (and optionally mutate)
        if self.genome_data:
            child.genome_data = copy.deepcopy(self.genome_data)
            if mutate_genome:
                child._log('create', f'Genome inherited + mutated from {self.name}')
            else:
                child._log('create', f'Genome cloned from {self.name}')
        
        # State teilweise erben
        if inherit_state and self.state_data:
            child.state_data = copy.deepcopy(self.state_data)
            # Halve episodic memory (child inherits only strongest)
            if child.state_data.get('episodic_memories'):
                mems = child.state_data['episodic_memories']
                child.state_data['episodic_memories'] = mems[:len(mems) // 2]
            # Body confidence reduzieren (muss sich neu kalibrieren)
            child.state_data['body_confidence'] = (
                child.state_data.get('body_confidence', 0.5) * 0.7)
            child._log('create', f'State partially inherited from {self.name}')
        
        self.children.append(child_name)
        return child

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'template': self.template,
            'created_at': self.created_at,
            'description': self.description,
            'genome_data': self.genome_data,
            'state_data': self.state_data,
            'sensors': [s.to_dict() for s in self.sensors],
            'training_log': self.training_log,
            'total_generations': self.total_generations,
            'total_steps': self.total_steps,
            'best_fitness': self.best_fitness,
            'parent_name': self.parent_name,
            'children': self.children,
            'tags': self.tags,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'SynpawProfile':
        profile = SynpawProfile(
            name=d['name'],
            template=d.get('template', 'synpaw'),
            created_at=d.get('created_at', ''),
            description=d.get('description', ''),
            genome_data=d.get('genome_data'),
            state_data=d.get('state_data'),
            sensors=[SensorConfig.from_dict(s) for s in d.get('sensors', [])],
            training_log=d.get('training_log', []),
            total_generations=d.get('total_generations', 0),
            total_steps=d.get('total_steps', 0),
            best_fitness=d.get('best_fitness', 0.0),
            parent_name=d.get('parent_name'),
            children=d.get('children', []),
            tags=d.get('tags', []),
        )
        return profile

    def save(self, path: str) -> None:
        """Profil als JSON speichern."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> 'SynpawProfile':
        """Profil aus JSON laden."""
        with open(path, 'r', encoding='utf-8') as f:
            return SynpawProfile.from_dict(json.load(f))

    # ================================================================
    # DISPLAY
    # ================================================================

    def summary(self) -> str:
        """Kurze Zusammenfassung."""
        sensors = ', '.join(s.name for s in self.get_active_sensors())
        parent = f" (Kind von {self.parent_name})" if self.parent_name else ""
        state = "trained" if self.state_data else "untrained"
        return (f"🐾 {self.name}{parent} — {self.template}, "
                f"{state}, Gen: {self.total_generations}, "
                f"Best: {self.best_fitness:.2f}m, "
                f"Sensors: [{sensors}]")

    def __repr__(self):
        return f"SynpawProfile('{self.name}', gen={self.total_generations}, fit={self.best_fitness:.2f})"
