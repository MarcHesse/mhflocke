"""
MH-FLOCKE — Genome v0.4.1
========================================
Genetic encoding for creature morphology and neural parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import json
import copy
import random


# ========================================================================
# DATENSTRUKTUREN
# ========================================================================

@dataclass
class Segment:
    """Ein Körpersegment (Rigid Body)."""
    id: int
    length: float         # 0.1 - 2.0 Einheiten
    radius: float         # 0.05 - 0.5
    mass: float           # auto-berechnet aus Volumen × Dichte
    density: float = 1.0  # Relative Dichte
    segment_type: str = 'limb'  # 'limb', 'wheel', 'gripper'
    # Wheel-spezifisch
    wheel_friction: float = 0.9
    max_rpm: float = 500.0
    # Gripper-spezifisch
    grip_strength: float = 0.5
    gripper_open_angle: float = 0.8

    def recompute_mass(self):
        """Masse aus Zylindervolumen × Dichte."""
        volume = np.pi * self.radius ** 2 * self.length
        self.mass = volume * self.density


@dataclass
class Joint:
    """Gelenk zwischen zwei Segmenten."""
    parent_id: int
    child_id: int
    joint_type: str              # 'hinge' (1-DoF) oder 'ball' (3-DoF)
    axis: np.ndarray = None      # Rotationsachse(n)
    angle_min: float = -45.0     # Minimaler Winkel (Grad, MuJoCo Standard)
    angle_max: float = 45.0      # Maximaler Winkel (Grad)
    max_torque: float = 1.0      # Maximales Drehmoment (Motor-Stärke)

    def __post_init__(self):
        if self.axis is None:
            self.axis = np.array([1.0, 0.0, 0.0])


@dataclass
class Sensor:
    """Sensor an einem Segment."""
    segment_id: int
    sensor_type: str             # 'distance', 'ground', 'touch', 'proprioception', 'velocity'
    direction: np.ndarray = None # Richtungsvektor (für distance sensor)
    range: float = 5.0           # Reichweite (für distance)

    def __post_init__(self):
        if self.direction is None:
            self.direction = np.array([0.0, 0.0, -1.0])


@dataclass
class Genome:
    """Vollständiger Körperplan einer Kreatur."""
    segments: List[Segment] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    sensors: List[Sensor] = field(default_factory=list)

    # Metadaten
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    fitness: float = 0.0
    genome_id: int = 0

    # Evolvable plasticity rule (Phase 9W-07b)
    plasticity_genome: Optional[Dict] = None  # PlasticityGenome als Dict

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def n_motors(self) -> int:
        """Anzahl ansteuerbarer Gelenke = SNN Motor-Outputs."""
        return len(self.joints)

    @property
    def n_sensors(self) -> int:
        """Anzahl Sensor-Inputs = SNN Sensor-Inputs."""
        return len(self.sensors)

    @property
    def total_mass(self) -> float:
        return sum(s.mass for s in self.segments)


# ========================================================================
# FACTORY
# ========================================================================

class GenomeFactory:
    """Erzeugt Genome."""

    @staticmethod
    def create_random(n_segments: int = 5,
                      n_sensors_per_segment: int = 2) -> Genome:
        """
        Erzeugt ein zufälliges Genom.

        Strategie: Linearer Baum (Kette) mit zufälligen Verzweigungen.
        Segment 0 = Body (Torso), Rest = Limbs.
        """
        # Clamp to valid range
        n_segments = max(2, min(n_segments, GenomeValidator.MAX_SEGMENTS))

        segments = []
        joints = []
        sensors = []

        for i in range(n_segments):
            s = Segment(
                id=i,
                length=np.random.uniform(0.2, 1.5),
                radius=np.random.uniform(0.05, 0.3),
                mass=0.0,
                density=np.random.uniform(0.8, 1.2),
            )
            s.recompute_mass()
            segments.append(s)

        # Joints: tree structure — each segment (except 0) has exactly one parent
        for i in range(1, n_segments):
            # Parent is a random previous segment (usually the direct one)
            if np.random.random() < 0.7 or i == 1:
                parent = i - 1  # Kette
            else:
                parent = np.random.randint(0, i)  # Verzweigung

            axis_choices = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
            j = Joint(
                parent_id=parent,
                child_id=i,
                joint_type=np.random.choice(['hinge', 'ball']),
                axis=axis_choices[np.random.randint(0, 3)].copy(),
                angle_min=np.random.uniform(-90, -20),
                angle_max=np.random.uniform(20, 90),
                max_torque=np.random.uniform(0.5, 2.0),
            )
            joints.append(j)

        # Sensors
        for i in range(n_segments):
            n_sens = max(1, n_sensors_per_segment) if i == 0 else n_sensors_per_segment
            for _ in range(n_sens):
                sensor_type = np.random.choice(
                    ['distance', 'ground', 'touch', 'proprioception', 'velocity']
                )
                direction = np.random.randn(3)
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                sensors.append(Sensor(
                    segment_id=i,
                    sensor_type=sensor_type,
                    direction=direction,
                    range=np.random.uniform(2.0, 8.0),
                ))

        return Genome(segments=segments, joints=joints, sensors=sensors)

    @staticmethod
    def _make_limb(segments, joints, sensors, parent_id, next_id,
                   n_limb_segments=2, label='limb'):
        """Helper: Adds a limb chain to segments/joints/sensors.

        Bein-Gelenke rotieren um Y-Achse (Sagittalebene),
        damit Vorwärts-/Rückwärts-Schwung möglich ist.
        """
        for k in range(n_limb_segments):
            seg_id = next_id + k
            # Thigh thicker, lower leg thinner
            s = Segment(
                id=seg_id,
                length=0.35 - k * 0.05,   # 0.35 / 0.30
                radius=0.06 - k * 0.015,  # 0.06 / 0.045
                mass=0.0,
                density=6.0,
                segment_type='limb',
            )
            s.recompute_mass()
            segments.append(s)

            p = parent_id if k == 0 else seg_id - 1
            j = Joint(
                parent_id=p,
                child_id=seg_id,
                joint_type='hinge',
                axis=np.array([0.0, 1.0, 0.0]),  # Y-Achse: Bein schwingt vor/zurück (X-Z Ebene)
                angle_min=-60.0,  # Grad! MuJoCo erwartet Grad
                angle_max=60.0,
                max_torque=3.0,  # Stärker für Lokomotion
            )
            joints.append(j)

            # Proprioception on each limb segment, ground on tip
            sensors.append(Sensor(segment_id=seg_id, sensor_type='proprioception'))
            if k == n_limb_segments - 1:
                sensors.append(Sensor(segment_id=seg_id, sensor_type='ground',
                                      direction=np.array([0.0, -1.0, 0.0])))
                sensors.append(Sensor(segment_id=seg_id, sensor_type='touch'))

        return next_id + n_limb_segments

    @staticmethod
    def create_biped_template() -> Genome:
        """Erzeugt ein Bipedal-Starter-Genom.
        
        Aufbau:
          - Torso: Box, breit, flach
          - Kopf: Kugel oben auf Torso
          - 2 Beine: je Oberschenkel + Unterschenkel (Capsule)
        """
        segments = []
        joints = []
        sensors = []

        # --- Torso (id=0) --- Box body
        torso = Segment(id=0, length=0.4, radius=0.2, mass=0.0,
                         density=8.0, segment_type='torso')
        torso.recompute_mass()
        segments.append(torso)
        sensors.append(Sensor(segment_id=0, sensor_type='velocity'))
        sensors.append(Sensor(segment_id=0, sensor_type='distance',
                              direction=np.array([1.0, 0.0, 0.0])))

        # --- Kopf (id=1) --- Kugel oben
        head = Segment(id=1, length=0.12, radius=0.1, mass=0.0,
                        density=5.0, segment_type='head')
        head.recompute_mass()
        segments.append(head)
        joints.append(Joint(
            parent_id=0, child_id=1, joint_type='hinge',
            axis=np.array([0.0, 1.0, 0.0]),
            angle_min=-15.0, angle_max=15.0, max_torque=0.5,
        ))
        sensors.append(Sensor(segment_id=1, sensor_type='distance',
                              direction=np.array([1.0, 0.0, 0.0])))

        # --- Beine (id=2..5) ---
        next_id = 2
        for _ in range(2):
            next_id = GenomeFactory._make_limb(
                segments, joints, sensors,
                parent_id=0, next_id=next_id, n_limb_segments=2
            )

        return Genome(segments=segments, joints=joints, sensors=sensors)

    @staticmethod
    def create_quadruped_template() -> Genome:
        """Erzeugt ein Quadruped-Starter-Genom (4 Beine, Kopf, Schwanz).

        Aufbau — "Mogli" Design:
          - Torso: Capsule, länglich, warm-orange
          - Kopf: Kugel vorne, mit 2 Augen-Sensoren (distance, gerichtet)
          - 4 Beine: je Oberschenkel + Unterschenkel
          - Schwanz: 2 Segmente, aktiv für Balance

        Segment-IDs:
          0  = Torso
          1  = Kopf
          2  = Auge links (Sensor-Site, kein Gelenk — fixiert am Kopf)
          3  = Auge rechts
          4-5   = Vorderbein links (Ober/Unterschenkel)
          6-7   = Vorderbein rechts
          8-9   = Hinterbein links
          10-11 = Hinterbein rechts
          12-13 = Schwanz (2 Segmente)

        Unterscheidung zu Unity ML-Agents:
          - Augen sind Sensor-Punkte (distance sensors, sichtbar als kleine Kugeln)
          - Schwanz ist aktiv (motorisiert, hilft bei Balance)
          - Gelenke leuchten in Cyan (Flocke-Palette)
          - Kein Cartoon — wissenschaftliche Visualisierung
        """
        segments = []
        joints = []
        sensors = []

        # === TORSO (id=0) === Elongated body
        torso = Segment(id=0, length=0.6, radius=0.15, mass=0.0,
                         density=8.0, segment_type='torso')
        torso.recompute_mass()
        segments.append(torso)
        sensors.append(Sensor(segment_id=0, sensor_type='velocity'))
        sensors.append(Sensor(segment_id=0, sensor_type='proprioception'))

        # === KOPF (id=1) === Kugel vorne am Torso
        head = Segment(id=1, length=0.10, radius=0.09, mass=0.0,
                        density=5.0, segment_type='head')
        head.recompute_mass()
        segments.append(head)
        joints.append(Joint(
            parent_id=0, child_id=1, joint_type='hinge',
            axis=np.array([0.0, 1.0, 0.0]),   # Nicken
            angle_min=-20.0, angle_max=20.0, max_torque=0.5,
        ))

        # === AUGEN (id=2, 3) === Kleine Kugeln am Kopf, fixierte Sensor-Punkte
        # Left eye — distance sensor looks slightly left-forward
        eye_l = Segment(id=2, length=0.02, radius=0.025, mass=0.0,
                         density=1.0, segment_type='head')  # Typ 'head' für Kugel-Rendering
        eye_l.recompute_mass()
        segments.append(eye_l)
        joints.append(Joint(
            parent_id=1, child_id=2, joint_type='hinge',
            axis=np.array([0.0, 0.0, 1.0]),
            angle_min=-5.0, angle_max=5.0, max_torque=0.1,  # Minimal — fast fixiert
        ))
        sensors.append(Sensor(segment_id=2, sensor_type='distance',
                              direction=np.array([0.85, 0.0, 0.53]),  # Leicht nach oben-vorne-links
                              range=6.0))

        # Auge rechts
        eye_r = Segment(id=3, length=0.02, radius=0.025, mass=0.0,
                         density=1.0, segment_type='head')
        eye_r.recompute_mass()
        segments.append(eye_r)
        joints.append(Joint(
            parent_id=1, child_id=3, joint_type='hinge',
            axis=np.array([0.0, 0.0, 1.0]),
            angle_min=-5.0, angle_max=5.0, max_torque=0.1,
        ))
        sensors.append(Sensor(segment_id=3, sensor_type='distance',
                              direction=np.array([0.85, 0.0, -0.53]),  # Rechts-vorne
                              range=6.0))

        # Head sensors: frontal distance + ground sensor
        sensors.append(Sensor(segment_id=1, sensor_type='distance',
                              direction=np.array([1.0, 0.0, 0.0]), range=8.0))
        sensors.append(Sensor(segment_id=1, sensor_type='ground',
                              direction=np.array([0.0, -1.0, 0.0])))

        # === 4 BEINE (id=4..11) ===
        next_id = 4
        for _ in range(4):
            next_id = GenomeFactory._make_limb(
                segments, joints, sensors,
                parent_id=0, next_id=next_id, n_limb_segments=2
            )

        # === SCHWANZ (id=12, 13) === Aktiver Balance-Appendix
        # Schwanz-Basis
        tail_base = Segment(id=12, length=0.25, radius=0.04, mass=0.0,
                             density=4.0, segment_type='limb')
        tail_base.recompute_mass()
        segments.append(tail_base)
        joints.append(Joint(
            parent_id=0, child_id=12, joint_type='hinge',
            axis=np.array([0.0, 1.0, 0.0]),   # Schwanz schwingt hoch/runter
            angle_min=-45.0, angle_max=60.0, max_torque=1.5,
        ))
        sensors.append(Sensor(segment_id=12, sensor_type='proprioception'))

        # Schwanz-Spitze
        tail_tip = Segment(id=13, length=0.20, radius=0.025, mass=0.0,
                            density=3.0, segment_type='limb')
        tail_tip.recompute_mass()
        segments.append(tail_tip)
        joints.append(Joint(
            parent_id=12, child_id=13, joint_type='hinge',
            axis=np.array([0.0, 0.0, 1.0]),   # Schwanzspitze schwingt seitlich
            angle_min=-40.0, angle_max=40.0, max_torque=1.0,
        ))
        sensors.append(Sensor(segment_id=13, sensor_type='proprioception'))
        sensors.append(Sensor(segment_id=13, sensor_type='velocity'))

        return Genome(segments=segments, joints=joints, sensors=sensors)

    @staticmethod
    def create_mogli_template() -> Genome:
        """Alias für das Quadruped-Template — unser Flaggschiff-Design."""
        return GenomeFactory.create_quadruped_template()

    @staticmethod
    def create_worm_template() -> Genome:
        """Erzeugt ein Wurm-Genom (Kette ohne Verzweigungen)."""
        segments = []
        joints = []
        sensors = []

        n = 6  # Wurm-Segmente
        for i in range(n):
            s = Segment(
                id=i,
                length=0.4,
                radius=0.15 if i == 0 else 0.1,
                mass=0.0,
                density=1.0,
            )
            s.recompute_mass()
            segments.append(s)

            if i > 0:
                j = Joint(
                    parent_id=i - 1,
                    child_id=i,
                    joint_type='hinge',
                    axis=np.array([0.0, 0.0, 1.0]),  # Seitliche Biegung
                    angle_min=-0.8,
                    angle_max=0.8,
                    max_torque=1.0,
                )
                joints.append(j)

            sensors.append(Sensor(segment_id=i, sensor_type='ground',
                                  direction=np.array([0.0, 0.0, -1.0])))
            if i == 0:
                sensors.append(Sensor(segment_id=i, sensor_type='distance',
                                      direction=np.array([0.0, 1.0, 0.0])))
                sensors.append(Sensor(segment_id=i, sensor_type='velocity'))

        return Genome(segments=segments, joints=joints, sensors=sensors)


# ========================================================================
# MUTATION
# ========================================================================

class GenomeMutator:
    """Mutations-Operatoren für Genome."""

    def __init__(self,
                 segment_length_std: float = 0.1,
                 joint_angle_std: float = 0.2,
                 torque_std: float = 0.1,
                 add_segment_prob: float = 0.05,
                 remove_segment_prob: float = 0.03,
                 add_sensor_prob: float = 0.05):
        self.segment_length_std = segment_length_std
        self.joint_angle_std = joint_angle_std
        self.torque_std = torque_std
        self.add_segment_prob = add_segment_prob
        self.remove_segment_prob = remove_segment_prob
        self.add_sensor_prob = add_sensor_prob

    def mutate(self, genome: Genome) -> Genome:
        """Mutiert ein Genom (gibt Kopie zurück)."""
        g = copy.deepcopy(genome)

        # 1. Segment dimensions (symmetric for legs)
        # Find leg pairs: same depth from root, same type
        limb_segs = [s for s in g.segments if s.segment_type == 'limb']
        # Paarweise mutieren: je 2 Segmente (links/rechts) gleich
        mutated_ids = set()
        for i in range(0, len(limb_segs) - 1, 2):
            s1, s2 = limb_segs[i], limb_segs[i + 1]
            dl = np.random.randn() * self.segment_length_std * 0.5
            dr = np.random.randn() * self.segment_length_std * 0.2
            for s in (s1, s2):
                s.length = np.clip(s.length + dl, 0.1, 1.0)
                s.radius = np.clip(s.radius + dr, 0.03, 0.15)
                s.recompute_mass()
                mutated_ids.add(s.id)
        # Einzelne Limbs die übrig sind
        for seg in g.segments:
            if seg.id in mutated_ids or seg.segment_type in ('torso', 'head'):
                continue
            if seg.segment_type == 'limb':
                seg.length = np.clip(
                    seg.length + np.random.randn() * self.segment_length_std * 0.5,
                    0.1, 1.0
                )
                seg.radius = np.clip(
                    seg.radius + np.random.randn() * self.segment_length_std * 0.2,
                    0.03, 0.15
                )
                seg.recompute_mass()

        # 2. Gelenk-Parameter
        for j in g.joints:
            j.angle_min = np.clip(
                j.angle_min + np.random.randn() * self.joint_angle_std * 10,
                -120.0, 0.0
            )
            j.angle_max = np.clip(
                j.angle_max + np.random.randn() * self.joint_angle_std * 10,
                0.0, 120.0
            )
            j.max_torque = np.clip(
                j.max_torque + np.random.randn() * self.torque_std,
                0.1, 5.0
            )

        # 3. Segment hinzufügen
        if (np.random.random() < self.add_segment_prob and
                g.n_segments < GenomeValidator.MAX_SEGMENTS):
            self._add_segment(g)

        # 4. Segment entfernen
        if (np.random.random() < self.remove_segment_prob and
                g.n_segments > GenomeValidator.MIN_SEGMENTS):
            self._remove_segment(g)

        # 5. Segment-Typ mutieren (limb → wheel / gripper)
        if np.random.random() < 0.05 and g.n_segments > 1:
            candidates = [s for s in g.segments if s.id != 0]
            if candidates:
                seg = random.choice(candidates)
                seg.segment_type = random.choice(['limb', 'wheel', 'gripper'])
                if seg.segment_type == 'wheel':
                    seg.length = min(seg.length, 0.1)
                    seg.radius = max(seg.radius, 0.08)
                elif seg.segment_type == 'gripper':
                    seg.radius = min(seg.radius, 0.05)

        # 6. Sensor hinzufügen/entfernen
        if np.random.random() < self.add_sensor_prob and g.n_segments > 0:
            seg_id = np.random.choice([s.id for s in g.segments])
            sensor_type = np.random.choice(
                ['distance', 'ground', 'touch', 'proprioception', 'velocity']
            )
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            g.sensors.append(Sensor(
                segment_id=seg_id,
                sensor_type=sensor_type,
                direction=direction,
            ))

        return g

    def _add_segment(self, g: Genome):
        """Fügt ein neues Segment als Verzweigung an."""
        new_id = max(s.id for s in g.segments) + 1
        parent_id = np.random.choice([s.id for s in g.segments])

        seg = Segment(
            id=new_id,
            length=np.random.uniform(0.2, 1.0),
            radius=np.random.uniform(0.05, 0.2),
            mass=0.0,
            density=1.0,
        )
        seg.recompute_mass()
        g.segments.append(seg)

        j = Joint(
            parent_id=parent_id,
            child_id=new_id,
            joint_type=np.random.choice(['hinge', 'ball']),
            axis=np.array([1.0, 0.0, 0.0]),
            angle_min=np.random.uniform(-90, -20),
            angle_max=np.random.uniform(20, 90),
            max_torque=np.random.uniform(0.5, 2.0),
        )
        g.joints.append(j)

        # Sensor am neuen Segment
        g.sensors.append(Sensor(segment_id=new_id, sensor_type='proprioception'))

    def _remove_segment(self, g: Genome):
        """Entfernt ein Blatt-Segment (kein Child)."""
        # Finde Blätter (Segmente die kein Parent für andere sind)
        parent_ids = {j.parent_id for j in g.joints}
        leaves = [s for s in g.segments if s.id not in parent_ids and s.id != 0]

        if not leaves:
            return

        victim = np.random.choice(leaves)
        victim_id = victim.id

        g.segments = [s for s in g.segments if s.id != victim_id]
        g.joints = [j for j in g.joints if j.child_id != victim_id]
        g.sensors = [s for s in g.sensors if s.segment_id != victim_id]

    def crossover(self, parent_a: Genome, parent_b: Genome) -> Genome:
        """
        Crossover zweier Genome.

        Strategie: Nimm den Torso + Teil der Gliedmaßen von A,
        und ersetze einige Gliedmaßen durch welche von B.
        """
        child = copy.deepcopy(parent_a)

        # Finde Gliedmaßen (direkte Children von Segment 0) in beiden Eltern
        limbs_a = [j for j in parent_a.joints if j.parent_id == 0]
        limbs_b = [j for j in parent_b.joints if j.parent_id == 0]

        if not limbs_b:
            return child

        # Ersetze ~50% der Gliedmaßen von A durch welche von B
        n_replace = max(1, len(limbs_a) // 2)
        if len(limbs_a) > 0:
            replace_indices = random.sample(range(len(limbs_a)),
                                            min(n_replace, len(limbs_a)))

            for idx in sorted(replace_indices, reverse=True):
                limb_joint = limbs_a[idx]
                old_child_id = limb_joint.child_id

                # Entferne alten Subtree
                subtree_ids = self._get_subtree_ids(child, old_child_id)
                child.segments = [s for s in child.segments if s.id not in subtree_ids]
                child.joints = [j for j in child.joints
                                if j.child_id not in subtree_ids]
                child.sensors = [s for s in child.sensors
                                 if s.segment_id not in subtree_ids]

            # Füge Gliedmaßen von B hinzu
            donor_limb = random.choice(limbs_b)
            donor_subtree_ids = self._get_subtree_ids(parent_b, donor_limb.child_id)

            # Re-ID die Donor-Segmente
            max_id = max((s.id for s in child.segments), default=-1) + 1
            id_map = {}
            for i, old_id in enumerate(sorted(donor_subtree_ids)):
                id_map[old_id] = max_id + i

            for seg in parent_b.segments:
                if seg.id in donor_subtree_ids:
                    new_seg = copy.deepcopy(seg)
                    new_seg.id = id_map[seg.id]
                    child.segments.append(new_seg)

            # Root-Joint des Donors → Parent 0
            root_joint = copy.deepcopy(donor_limb)
            root_joint.parent_id = 0
            root_joint.child_id = id_map[donor_limb.child_id]
            child.joints.append(root_joint)

            # Interne Joints des Donors
            for j in parent_b.joints:
                if j.child_id in donor_subtree_ids and j.child_id != donor_limb.child_id:
                    new_j = copy.deepcopy(j)
                    new_j.parent_id = id_map.get(j.parent_id, j.parent_id)
                    new_j.child_id = id_map[j.child_id]
                    child.joints.append(new_j)

            # Sensoren des Donors
            for s in parent_b.sensors:
                if s.segment_id in donor_subtree_ids:
                    new_s = copy.deepcopy(s)
                    new_s.segment_id = id_map[s.segment_id]
                    child.sensors.append(new_s)

        child.generation = max(parent_a.generation, parent_b.generation) + 1
        child.parent_ids = [parent_a.genome_id, parent_b.genome_id]

        # Repair falls nötig
        valid, _ = GenomeValidator.validate(child)
        if not valid:
            child = GenomeValidator.repair(child)

        return child

    def _get_subtree_ids(self, genome: Genome, root_id: int) -> set:
        """Alle Segment-IDs im Subtree ab root_id (inklusive)."""
        ids = {root_id}
        changed = True
        while changed:
            changed = False
            for j in genome.joints:
                if j.parent_id in ids and j.child_id not in ids:
                    ids.add(j.child_id)
                    changed = True
        return ids


# ========================================================================
# VALIDIERUNG
# ========================================================================

class GenomeValidator:
    """Prüft ob ein Genom physikalisch gültig ist."""

    MAX_SEGMENTS = 20
    MIN_SEGMENTS = 2
    MAX_TOTAL_MASS = 50.0
    MIN_TOTAL_MASS = 0.01

    @staticmethod
    def validate(genome: Genome) -> Tuple[bool, List[str]]:
        """
        Prüft Genom auf Gültigkeit.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Segment count
        if genome.n_segments < GenomeValidator.MIN_SEGMENTS:
            errors.append(f"Too few segments: {genome.n_segments} < {GenomeValidator.MIN_SEGMENTS}")
        if genome.n_segments > GenomeValidator.MAX_SEGMENTS:
            errors.append(f"Too many segments: {genome.n_segments} > {GenomeValidator.MAX_SEGMENTS}")

        # All joints reference existing segments
        seg_ids = {s.id for s in genome.segments}
        for j in genome.joints:
            if j.parent_id not in seg_ids:
                errors.append(f"Joint references missing parent segment {j.parent_id}")
            if j.child_id not in seg_ids:
                errors.append(f"Joint references missing child segment {j.child_id}")

        # No cycles: each child has exactly one parent, tree structure
        children = set()
        for j in genome.joints:
            if j.child_id in children:
                errors.append(f"Segment {j.child_id} has multiple parents (cycle risk)")
            children.add(j.child_id)

        # Mass
        if genome.n_segments > 0:
            mass = genome.total_mass
            if mass > GenomeValidator.MAX_TOTAL_MASS:
                errors.append(f"Mass too high: {mass:.2f} > {GenomeValidator.MAX_TOTAL_MASS}")
            if mass < GenomeValidator.MIN_TOTAL_MASS:
                errors.append(f"Mass too low: {mass:.4f} < {GenomeValidator.MIN_TOTAL_MASS}")

        # At least 1 sensor
        if genome.n_sensors == 0:
            errors.append("No sensors")

        # At least 1 motor (joint)
        if genome.n_motors == 0:
            errors.append("No motors (joints)")

        # Sensors reference existing segments
        for s in genome.sensors:
            if s.segment_id not in seg_ids:
                errors.append(f"Sensor references missing segment {s.segment_id}")

        return (len(errors) == 0, errors)

    @staticmethod
    def repair(genome: Genome) -> Genome:
        """Versucht ein ungültiges Genom zu reparieren."""
        g = copy.deepcopy(genome)

        # Zu wenig Segmente → minimal viable erzeugen
        if g.n_segments < GenomeValidator.MIN_SEGMENTS:
            g = GenomeFactory.create_random(n_segments=GenomeValidator.MIN_SEGMENTS)
            return g

        # Zu viele Segmente → hinten abschneiden
        while g.n_segments > GenomeValidator.MAX_SEGMENTS:
            # Entferne Blätter
            parent_ids = {j.parent_id for j in g.joints}
            leaves = [s for s in g.segments if s.id not in parent_ids and s.id != 0]
            if not leaves:
                break
            victim = leaves[-1]
            g.segments = [s for s in g.segments if s.id != victim.id]
            g.joints = [j for j in g.joints if j.child_id != victim.id]
            g.sensors = [s for s in g.sensors if s.segment_id != victim.id]

        # Entferne Joints/Sensors die auf nicht-existente Segmente zeigen
        seg_ids = {s.id for s in g.segments}
        g.joints = [j for j in g.joints
                    if j.parent_id in seg_ids and j.child_id in seg_ids]
        g.sensors = [s for s in g.sensors if s.segment_id in seg_ids]

        # Kein Sensor? Füge einen hinzu
        if g.n_sensors == 0 and g.n_segments > 0:
            g.sensors.append(Sensor(
                segment_id=g.segments[0].id,
                sensor_type='velocity',
            ))

        # Kein Motor? Füge minimalen Joint hinzu
        if g.n_motors == 0 and g.n_segments >= 2:
            g.joints.append(Joint(
                parent_id=g.segments[0].id,
                child_id=g.segments[1].id,
                joint_type='hinge',
            ))

        # Masse reparieren
        for seg in g.segments:
            seg.recompute_mass()

        return g


# ========================================================================
# SERIALISIERUNG
# ========================================================================

class GenomeSerializer:
    """Serialisierung für Persistenz."""

    @staticmethod
    def to_dict(genome: Genome) -> Dict:
        """Genom → JSON-serialisierbares Dict."""
        return {
            'segments': [
                {
                    'id': int(s.id),
                    'length': float(s.length),
                    'radius': float(s.radius),
                    'mass': float(s.mass),
                    'density': float(s.density),
                    'segment_type': str(s.segment_type),
                    'wheel_friction': float(s.wheel_friction),
                    'grip_strength': float(s.grip_strength),
                }
                for s in genome.segments
            ],
            'joints': [
                {
                    'parent_id': int(j.parent_id),
                    'child_id': int(j.child_id),
                    'joint_type': str(j.joint_type),
                    'axis': [float(x) for x in j.axis],
                    'angle_min': float(j.angle_min),
                    'angle_max': float(j.angle_max),
                    'max_torque': float(j.max_torque),
                }
                for j in genome.joints
            ],
            'sensors': [
                {
                    'segment_id': int(s.segment_id),
                    'sensor_type': str(s.sensor_type),
                    'direction': [float(x) for x in s.direction],
                    'range': float(s.range),
                }
                for s in genome.sensors
            ],
            'generation': int(genome.generation),
            'parent_ids': [int(p) for p in genome.parent_ids],
            'fitness': float(genome.fitness),
            'genome_id': int(genome.genome_id),
            'plasticity_genome': {
                k: float(v) for k, v in genome.plasticity_genome.items()
            } if genome.plasticity_genome else None,
        }

    @staticmethod
    def from_dict(data: Dict) -> Genome:
        """Dict → Genom."""
        segments = [
            Segment(
                id=s['id'],
                length=s['length'],
                radius=s['radius'],
                mass=s['mass'],
                density=s.get('density', 1.0),
                segment_type=s.get('segment_type', 'limb'),
                wheel_friction=s.get('wheel_friction', 0.9),
                grip_strength=s.get('grip_strength', 0.5),
            )
            for s in data['segments']
        ]
        joints = [
            Joint(
                parent_id=j['parent_id'],
                child_id=j['child_id'],
                joint_type=j['joint_type'],
                axis=np.array(j['axis']),
                angle_min=j['angle_min'],
                angle_max=j['angle_max'],
                max_torque=j['max_torque'],
            )
            for j in data['joints']
        ]
        sensors = [
            Sensor(
                segment_id=s['segment_id'],
                sensor_type=s['sensor_type'],
                direction=np.array(s['direction']),
                range=s.get('range', 5.0),
            )
            for s in data['sensors']
        ]

        return Genome(
            segments=segments,
            joints=joints,
            sensors=sensors,
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            fitness=data.get('fitness', 0.0),
            genome_id=data.get('genome_id', 0),
            plasticity_genome=data.get('plasticity_genome', None),
        )

    @staticmethod
    def to_json(genome: Genome, path: str):
        """Genom → JSON-Datei."""
        data = GenomeSerializer.to_dict(genome)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def from_json(path: str) -> Genome:
        """JSON-Datei → Genom."""
        with open(path) as f:
            data = json.load(f)
        return GenomeSerializer.from_dict(data)
