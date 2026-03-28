"""
MH-FLOCKE — Scene Instruction v0.4.1
========================================
Scene configuration: drive biases, behavior weights, terrain.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SceneInstruction:
    """Eine Szenen-Anweisung mit Drive- und Behavior-Biases."""
    text: str                                    # Freitext-Beschreibung
    drive_biases: Dict[str, float] = field(default_factory=dict)
    behavior_weights: Dict[str, float] = field(default_factory=dict)
    # Optionale Einschraenkungen
    forbidden_behaviors: List[str] = field(default_factory=list)
    forced_start_behavior: Optional[str] = None
    description: str = ''


# =====================================================================
# VORDEFINIERTE SZENEN
# =====================================================================

SCENE_INSTRUCTIONS: Dict[str, SceneInstruction] = {

    'meadow': SceneInstruction(
        text='Dog on a meadow — exploring, sniffing, relaxed',
        description='Wiese: Erkunden, Schnueffeln, entspannt',
        drive_biases={
            'exploration': 0.7,
            'comfort': 0.3,
            'survival': 0.2,
            'social': 0.0,
        },
        behavior_weights={
            'walk': 1.0,
            'sniff': 1.5,
            'look_around': 1.2,
            'rest': 0.8,
            'trot': 0.6,
            'mark': 0.4,
        },
        forced_start_behavior='look_around',
    ),

    'park': SceneInstruction(
        text='Dog in a park — other dogs nearby, alert and social',
        description='Park: Andere Hunde, aufmerksam und sozial',
        drive_biases={
            'social': 0.6,
            'exploration': 0.4,
            'survival': 0.4,
            'comfort': 0.1,
        },
        behavior_weights={
            'walk': 1.0,
            'alert': 1.5,
            'look_around': 1.3,
            'sniff': 1.0,
            'trot': 1.2,
            'mark': 0.8,
            'rest': 0.3,
        },
        forced_start_behavior='alert',
    ),

    'home_tired': SceneInstruction(
        text='Dog at home — tired, winding down',
        description='Zu Hause: Muede, kommt zur Ruhe',
        drive_biases={
            'comfort': 0.8,
            'survival': 0.1,
            'exploration': 0.1,
            'social': 0.2,
        },
        behavior_weights={
            'rest': 2.0,
            'walk': 0.3,
            'sniff': 0.4,
            'look_around': 0.5,
            'alert': 0.2,
        },
        forbidden_behaviors=['trot', 'mark'],
        forced_start_behavior='rest',
    ),

    'forest_walk': SceneInstruction(
        text='Dog on a forest trail — curious, many smells',
        description='Waldspaziergang: Neugierig, viele Gerueche',
        drive_biases={
            'exploration': 0.8,
            'survival': 0.3,
            'comfort': 0.1,
            'social': 0.0,
        },
        behavior_weights={
            'walk': 1.2,
            'sniff': 2.0,
            'look_around': 1.0,
            'alert': 0.8,
            'trot': 0.5,
            'mark': 0.7,
            'rest': 0.2,
        },
    ),

    'training_ground': SceneInstruction(
        text='Dog on training ground — focused locomotion',
        description='Trainingsgelaende: Fokus auf Laufen',
        drive_biases={
            'survival': 0.5,
            'exploration': 0.3,
            'comfort': 0.1,
            'social': 0.0,
        },
        behavior_weights={
            'walk': 2.0,
            'trot': 1.5,
            'sniff': 0.2,
            'look_around': 0.3,
            'rest': 0.3,
            'alert': 0.2,
        },
        forced_start_behavior='walk',
    ),

    'ball': SceneInstruction(
        text='Dog plays with ball on grass — playful, chasing, investigating',
        description='Ball-Spiel: Spieltrieb, Jagdtrieb, Untersuchen',
        drive_biases={
            'play': 0.9,
            'exploration': 0.5,
            'survival': 0.2,   # Prey-chase component
            'social': 0.3,
            'comfort': 0.0,
        },
        behavior_weights={
            'walk': 1.0,
            'investigate': 1.8,  # Approach and sniff the ball
            'play': 2.0,        # Playful interaction
            'chase': 1.5,       # Chase when ball rolls away
            'trot': 1.2,        # Energetic movement between play bouts
            'sniff': 1.0,
            'look_around': 0.5,
            'rest': 0.2,
            'alert': 0.3,
        },
        forced_start_behavior='walk',
    ),

    'guard': SceneInstruction(
        text='Dog guarding territory — vigilant, still, alert',
        description='Wache: Aufmerksam, ruhig, territorial',
        drive_biases={
            'survival': 0.7,
            'social': 0.3,
            'exploration': 0.1,
            'comfort': 0.1,
        },
        behavior_weights={
            'alert': 2.0,
            'look_around': 1.5,
            'mark': 1.0,
            'walk': 0.5,
            'rest': 0.3,
            'sniff': 0.3,
        },
        forced_start_behavior='alert',
    ),
}


def get_scene_instruction(name: str) -> Optional[SceneInstruction]:
    """Holt vordefinierte Szenen-Anweisung."""
    return SCENE_INSTRUCTIONS.get(name)


def list_scene_instructions() -> List[str]:
    """Alle verfuegbaren Szenen-Namen."""
    return list(SCENE_INSTRUCTIONS.keys())


def parse_free_text(text: str) -> SceneInstruction:
    """
    Einfacher Keyword-Parser fuer freie Szenen-Beschreibungen.
    Spaeter: LLM-basiert.

    Beispiel: "tired dog in a garden" -> comfort hoch, rest hoch
    """
    text_lower = text.lower()
    instruction = SceneInstruction(text=text)

    # Keyword-Matching (simpel, wird durch LLM ersetzt)
    keywords = {
        'tired': {'comfort': 0.3, '_beh': {'rest': 1.5}},
        'sleepy': {'comfort': 0.4, '_beh': {'rest': 2.0}},
        'excited': {'exploration': 0.3, '_beh': {'trot': 1.5, 'walk': 1.2}},
        'curious': {'exploration': 0.4, '_beh': {'sniff': 1.5, 'look_around': 1.3}},
        'scared': {'survival': 0.5, '_beh': {'alert': 2.0}},
        'alert': {'survival': 0.3, '_beh': {'alert': 1.5, 'look_around': 1.2}},
        'relaxed': {'comfort': 0.3, '_beh': {'rest': 1.0, 'sniff': 0.8}},
        'playful': {'play': 0.4, 'exploration': 0.3, 'social': 0.2,
                    '_beh': {'play': 2.0, 'chase': 1.5, 'trot': 1.3}},
        'ball': {'play': 0.5, 'exploration': 0.2,
                 '_beh': {'play': 2.0, 'chase': 1.5, 'investigate': 1.8}},
        'toy': {'play': 0.4, 'exploration': 0.3,
                '_beh': {'play': 1.8, 'investigate': 1.5}},
        'fetch': {'play': 0.5, 'survival': 0.2,
                  '_beh': {'chase': 2.0, 'play': 1.5, 'trot': 1.3}},
        'garden': {'exploration': 0.2, '_beh': {'sniff': 1.3, 'mark': 0.8}},
        'forest': {'exploration': 0.3, '_beh': {'sniff': 1.5, 'walk': 1.2}},
        'park': {'social': 0.2, 'exploration': 0.2, '_beh': {'alert': 1.0}},
        'home': {'comfort': 0.3, '_beh': {'rest': 1.2}},
        'walk': {'exploration': 0.2, '_beh': {'walk': 1.5, 'trot': 0.8}},
        'sniff': {'exploration': 0.2, '_beh': {'sniff': 2.0}},
    }

    for keyword, effects in keywords.items():
        if keyword in text_lower:
            for key, val in effects.items():
                if key == '_beh':
                    for beh_name, beh_weight in val.items():
                        current = instruction.behavior_weights.get(beh_name, 1.0)
                        instruction.behavior_weights[beh_name] = current * beh_weight
                else:
                    current = instruction.drive_biases.get(key, 0.0)
                    instruction.drive_biases[key] = min(1.0, current + val)

    return instruction
