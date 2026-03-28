"""
MH-FLOCKE — Behavior Knowledge v0.4.1
========================================
Built-in behavioral repertoire for creature types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MotorPattern:
    """Wie ein Behavior die Motorik beeinflusst."""
    # CPG-Modulation (Multiplikatoren auf Basis-CPG)
    cpg_frequency_scale: float = 1.0    # 0=stop, 0.3=langsam, 1=normal, 1.5=schnell
    cpg_amplitude_scale: float = 1.0    # 0=keine Beinbewegung, 1=normal

    # Cosmetic joints (target angle in degrees, None=unchanged)
    neck_angle: Optional[float] = None      # -30=runter, 0=neutral, +30=hoch
    jaw_angle: Optional[float] = None       # -10=leicht offen, 0=zu
    ear_angle: Optional[float] = None       # -15=angelegt, +15=aufgestellt
    tail_angle: Optional[float] = None      # -20=eingezogen, +40=hoch

    # Special overrides for individual legs (None=CPG controls)
    # Format: Dict[actuator_name, target_angle_normalized]
    leg_overrides: Dict[str, float] = field(default_factory=dict)

    # Interpolations-Geschwindigkeit (0.01=langsam, 0.1=schnell)
    blend_speed: float = 0.05


@dataclass
class BehaviorDef:
    """Definition eines Verhaltens."""
    name: str
    description: str = ''

    # Drive affinity: which drive triggers this behavior
    # Hoehere Werte = staerkere Assoziation
    drive_affinity: Dict[str, float] = field(default_factory=dict)

    # Motor-Pattern
    motor: MotorPattern = field(default_factory=MotorPattern)

    # Dauer in Simulation-Steps (bei 500 Hz ~ 2ms/step)
    min_duration: int = 500       # ~1 Sekunde
    max_duration: int = 3000      # ~6 Sekunden

    # Basis-Prioritaet (kann durch Drive-Staerke moduliert werden)
    priority: float = 1.0

    # Vorbedingungen
    requires_upright: bool = True   # Muss aufrecht stehen
    requires_still: bool = False    # Muss stillstehen
    min_steps_alive: int = 0        # Mindest-Lebenszeit

    # Allowed follow-up behaviors (empty = all allowed)
    allowed_transitions: List[str] = field(default_factory=list)

    # Cooldown in Steps nach Ausfuehrung
    cooldown: int = 500


# =====================================================================
# DOG BEHAVIORS (base knowledge, hardcoded)
# =====================================================================

DOG_BEHAVIORS: Dict[str, BehaviorDef] = {

    'walk': BehaviorDef(
        name='walk',
        description='Normal forward locomotion',
        drive_affinity={
            'exploration': 0.8,
            'survival': 0.5,
            'social': 0.3,
            'comfort': 0.1,
        },
        motor=MotorPattern(
            cpg_frequency_scale=1.0,
            cpg_amplitude_scale=1.0,
            neck_angle=0.0,
            tail_angle=20.0,
            ear_angle=0.0,
            blend_speed=0.05,
        ),
        min_duration=1000,
        max_duration=10000,
        priority=1.0,
        cooldown=200,
    ),

    'sniff': BehaviorDef(
        name='sniff',
        description='Sniff the ground — head down, slow movement, jaw slightly open',
        drive_affinity={
            'exploration': 0.9,
            'survival': 0.2,
            'social': 0.4,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.7,   # was 0.25 — dogs sniff WHILE walking
            cpg_amplitude_scale=0.8,   # was 0.4  — shorter strides, not stillstand
            neck_angle=-25.0,
            jaw_angle=-5.0,
            tail_angle=10.0,
            ear_angle=-5.0,
            blend_speed=0.03,
        ),
        min_duration=1500,
        max_duration=5000,
        priority=0.8,
        requires_upright=True,
        cooldown=1000,
    ),

    'look_around': BehaviorDef(
        name='look_around',
        description='Survey surroundings — head sweeps left/right, ears up, slow walk',
        drive_affinity={
            'exploration': 0.7,
            'survival': 0.6,
            'social': 0.5,
            'comfort': 0.1,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.5,   # was 0.05 — dogs look around while ambling
            cpg_amplitude_scale=0.6,   # was 0.1  — slow but moving
            neck_angle=15.0,       # Leicht erhoben
            ear_angle=15.0,        # Ohren aufgestellt
            tail_angle=15.0,
            blend_speed=0.04,
        ),
        min_duration=1000,
        max_duration=3000,
        priority=0.7,
        requires_upright=True,
        cooldown=800,
    ),

    'alert': BehaviorDef(
        name='alert',
        description='Alert attention — slow cautious walk, head high, ears forward',
        drive_affinity={
            'survival': 0.9,
            'exploration': 0.3,
            'social': 0.4,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.4,   # was 0.0 — alert ≠ freeze. Dogs stay mobile
            cpg_amplitude_scale=0.5,   # was 0.0 — cautious steps, ready to flee/chase
            neck_angle=30.0,
            ear_angle=18.0,
            tail_angle=35.0,       # Schwanz hoch
            jaw_angle=0.0,
            blend_speed=0.08,      # Schnelles Einfrieren
        ),
        min_duration=500,
        max_duration=2000,
        priority=1.2,              # Hohe Prioritaet
        requires_upright=True,
        cooldown=300,
    ),

    'rest': BehaviorDef(
        name='rest',
        description='Rest in place — stop moving, lower body, relax',
        drive_affinity={
            'comfort': 0.9,
            'survival': 0.1,
            'exploration': 0.0,
            'social': 0.2,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.0,
            cpg_amplitude_scale=0.0,
            neck_angle=-10.0,      # Kopf leicht gesenkt
            ear_angle=-10.0,       # Ohren angelegt
            tail_angle=-10.0,      # Schwanz ruhig
            jaw_angle=0.0,
            blend_speed=0.02,      # Langsames Ablegen
        ),
        min_duration=3000,
        max_duration=15000,
        priority=0.5,
        requires_upright=False,    # Kann auch liegend
        cooldown=2000,
    ),

    # NOTE on MLR (Mesencephalic Locomotor Region) principle:
    # Only 'rest' and 'mark' set CPG to zero. All other behaviors
    # maintain locomotion because biological quadrupeds move
    # continuously — they sniff WHILE walking, look around WHILE
    # ambling. The MLR provides tonic locomotor drive that
    # behaviors modulate but never fully suppress (except rest).
    # Ref: Shik & Orlovsky 1976, MLR stimulation in decerebrate cats

    'mark': BehaviorDef(
        name='mark',
        description='Territory marking — lift hind leg briefly',
        drive_affinity={
            'survival': 0.6,
            'exploration': 0.5,
            'social': 0.7,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.0,
            cpg_amplitude_scale=0.0,
            neck_angle=5.0,
            tail_angle=40.0,       # Schwanz hoch
            ear_angle=5.0,
            blend_speed=0.06,
            # Rechtes Hinterbein heben
            leg_overrides={
                'rr_hip_ab': 0.8,   # Bein abspreizen (normalized -1..1)
                'rr_hip': 0.3,      # Leicht nach vorn
                'rr_stifle': -0.5,  # Knie gebeugt
            },
        ),
        min_duration=800,
        max_duration=2000,
        priority=0.4,
        requires_upright=True,
        requires_still=True,
        min_steps_alive=5000,      # Erst nach Eingewoehnung
        cooldown=5000,
    ),

    # ── Neonatal Motor Babbling (Prechtl 1997) ──
    # Biology: spontaneous motor activity in neonates.
    # NOT random noise — it's structured exploration of the motor space.
    # "Fidgety movements" (Hadders-Algra 2004): small, variable movements
    # of all limbs, neck, trunk. They shift the center of mass, creating
    # vestibular + proprioceptive feedback that calibrates the cerebellum.
    # Without this, the CB has no training signal on flat terrain.
    #
    # Motor pattern: moderate frequency (0.6×), HIGH amplitude asymmetry.
    # The asymmetric amplitude causes the creature to wobble side to side,
    # generating roll/pitch errors that the vestibular system detects.
    # This is exactly what a newborn puppy does — uncoordinated but
    # purposeful limb movements that look clumsy but are essential.
    'motor_babbling': BehaviorDef(
        name='motor_babbling',
        description='Neonatal motor exploration — fidgety limb movements for cerebellar calibration',
        drive_affinity={
            'exploration': 1.0,
            'survival': 0.8,
            'social': 0.0,
            'comfort': 0.2,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.5,     # Slow, exploratory
            cpg_amplitude_scale=0.7,     # SMALLER than walk — gentle weight shifts
            neck_angle=0.0,
            tail_angle=10.0,
            ear_angle=0.0,
            blend_speed=0.03,            # Gradual onset
        ),
        min_duration=2000,
        max_duration=8000,
        priority=1.5,                    # Highest priority early on
        requires_upright=True,
        min_steps_alive=0,
        cooldown=500,
    ),

    'trot': BehaviorDef(
        name='trot',
        description='Fast trot — increased frequency and amplitude',
        drive_affinity={
            'exploration': 0.6,
            'survival': 0.7,
            'social': 0.5,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=1.4,
            cpg_amplitude_scale=1.2,
            neck_angle=5.0,
            tail_angle=30.0,
            ear_angle=5.0,
            blend_speed=0.06,
        ),
        min_duration=1000,
        max_duration=5000,
        priority=0.6,
        requires_upright=True,
        cooldown=1000,
    ),

    # === Ball/Object Interaction Behaviors (Issue #76) ===
    # Biology: dogs have innate object-interaction behaviors driven by
    # curiosity (investigate), play drive (play), and prey drive (chase).
    # Ref: Panksepp 1998 — PLAY system, Lorenz 1950 — prey drive

    'investigate': BehaviorDef(
        name='investigate',
        description='Approach and sniff an interesting object',
        drive_affinity={
            'exploration': 0.9,
            'play': 0.6,
            'survival': 0.3,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=0.8,   # Slow cautious approach
            cpg_amplitude_scale=0.9,
            neck_angle=-10.0,          # Head down, sniffing
            tail_angle=20.0,
            ear_angle=15.0,            # Ears forward
            blend_speed=0.04,
        ),
        min_duration=1000,
        max_duration=4000,
        priority=0.7,
        requires_upright=True,
        cooldown=500,
    ),

    'play': BehaviorDef(
        name='play',
        description='Playful interaction — nudge, paw, push object',
        drive_affinity={
            'play': 0.9,
            'exploration': 0.5,
            'social': 0.6,
            'survival': 0.0,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=1.2,   # Excited, bouncy
            cpg_amplitude_scale=1.1,
            neck_angle=-5.0,
            tail_angle=45.0,           # Tail wagging
            ear_angle=10.0,
            blend_speed=0.06,
        ),
        min_duration=500,
        max_duration=3000,
        priority=0.8,
        requires_upright=True,
        cooldown=300,
    ),

    'chase': BehaviorDef(
        name='chase',
        description='Chase a moving object — fast pursuit',
        drive_affinity={
            'play': 0.8,
            'survival': 0.7,   # Prey drive
            'exploration': 0.4,
            'comfort': 0.0,
        },
        motor=MotorPattern(
            cpg_frequency_scale=1.6,   # Fast run
            cpg_amplitude_scale=1.3,
            neck_angle=0.0,            # Head level, eyes on target
            tail_angle=15.0,           # Streaming back
            ear_angle=0.0,
            blend_speed=0.08,
        ),
        min_duration=500,
        max_duration=5000,
        priority=0.9,
        requires_upright=True,
        cooldown=500,
    ),
}


class BehaviorKnowledge:
    """
    Verwaltet Verhaltens-Wissen fuer eine Kreatur.
    Startet mit hardcoded Basis, kann durch ConceptGraph erweitert werden.
    """

    def __init__(self, creature_type: str = 'dog'):
        self.creature_type = creature_type
        self.behaviors: Dict[str, BehaviorDef] = {}
        self._load_base_knowledge(creature_type)

    def _load_base_knowledge(self, creature_type: str):
        """Laedt Basis-Verhalten fuer Kreatur-Typ."""
        if creature_type == 'dog':
            self.behaviors = dict(DOG_BEHAVIORS)
        else:
            # Fallback: nur walk
            self.behaviors = {
                'walk': DOG_BEHAVIORS['walk'],
                'rest': DOG_BEHAVIORS['rest'],
            }

    def get_behavior(self, name: str) -> Optional[BehaviorDef]:
        """Holt Behavior-Definition."""
        return self.behaviors.get(name)

    def get_behaviors_for_drive(self, drive: str,
                                 min_affinity: float = 0.3) -> List[BehaviorDef]:
        """Alle Behaviors die zu einem Drive passen."""
        result = []
        for beh in self.behaviors.values():
            if beh.drive_affinity.get(drive, 0) >= min_affinity:
                result.append(beh)
        result.sort(key=lambda b: b.drive_affinity.get(drive, 0), reverse=True)
        return result

    def get_all_names(self) -> List[str]:
        """Alle verfuegbaren Behavior-Namen."""
        return list(self.behaviors.keys())

    def add_behavior(self, behavior: BehaviorDef):
        """Fuegt neues Behavior hinzu (z.B. aus LLM/ConceptGraph gelernt)."""
        self.behaviors[behavior.name] = behavior

    def to_concept_graph_entries(self) -> List[Dict]:
        """Exportiert Wissen fuer ConceptGraph-Integration."""
        entries = []
        for name, beh in self.behaviors.items():
            entries.append({
                'label': f'behavior:{name}',
                'properties': {
                    'type': 'behavior',
                    'description': beh.description,
                    'drive_affinity': beh.drive_affinity,
                    'priority': beh.priority,
                    'duration_range': (beh.min_duration, beh.max_duration),
                },
            })
        return entries
