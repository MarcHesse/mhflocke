"""
LLM Instruction → Fitness Pipeline — Phase 10/12
====================================================
Natürlichsprachliche Anweisung → parametrische Fitness-Funktion.

"Lerne laufen" → velocity + upright + energy_efficiency Fitness
"Hebe den Ball hoch" → approach + contact + lift Fitness

Nutzt den bestehenden LLM-Adapter für Parsing.
Fallback: Regelbasiertes Keyword-Matching wenn kein LLM verfügbar.

Migriert von: src/evolution/instruction_parser.py (Phase 10-14)
Neues Zuhause: src/bridge/ (Phase 12: LLM-Bridge)

Usage:
    from src.bridge.instruction_parser import InstructionParser, FitnessBuilder
    parser = InstructionParser()
    spec = parser.parse("lerne schnell zu laufen")
    fitness_fn = FitnessBuilder.from_spec(spec)
    # fitness_fn(creature) → float
"""

__version__ = "0.1.0"
__logbook__ = 144

import json
import re
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


# ================================================================
# INSTRUCTION SPEC
# ================================================================

@dataclass
class FitnessComponent:
    """Ein Baustein der Fitness-Funktion."""
    component_type: str       # 'velocity', 'upright', 'approach', 'contact', 'lift', etc.
    weight: float = 1.0
    target: Optional[str] = None   # Ziel-Objekt (z.B. 'ball')
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldObject:
    """Ein Objekt in der Welt."""
    name: str
    shape: str = 'sphere'
    position: List[float] = field(default_factory=lambda: [2.0, 0.0, 0.3])
    mass: float = 0.5
    size: List[float] = field(default_factory=lambda: [0.15])
    color: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.1, 1.0])


@dataclass
class InstructionSpec:
    """Vollständige Spezifikation aus einer Instruktion."""
    instruction: str = ''
    creature_template: str = 'biped'
    objects: List[WorldObject] = field(default_factory=list)
    fitness_components: List[FitnessComponent] = field(default_factory=list)
    eval_steps: int = 1000
    success_criterion: Optional[str] = None
    difficulty: int = 1


# ================================================================
# FITNESS COMPONENTS LIBRARY
# ================================================================

class FitnessComponents:
    """Vordefinierte, messbare Fitness-Bausteine."""

    @staticmethod
    def velocity(creature, direction: str = 'forward', **kwargs) -> float:
        state = creature.get_state()
        vel = state.get('velocity', 0.0)
        return max(0, float(vel))

    @staticmethod
    def upright(creature, **kwargs) -> float:
        return creature._standing_steps / max(creature._step_count, 1)

    @staticmethod
    def distance(creature, **kwargs) -> float:
        return creature.get_distance_traveled()

    @staticmethod
    def alive(creature, total_steps: int = 1, **kwargs) -> float:
        return creature._step_count / max(total_steps, 1)

    @staticmethod
    def energy_efficiency(creature, **kwargs) -> float:
        return -creature._energy_spent * 0.01

    @staticmethod
    def stability(creature, **kwargs) -> float:
        state = creature.get_state()
        height = state.get('height', 0)
        return max(0, 1.0 - abs(height - 0.5))

    @staticmethod
    def approach(creature, target_pos: List[float] = None, **kwargs) -> float:
        if target_pos is None:
            target_pos = [2.0, 0.0, 0.0]
        pos = creature.get_position()
        dist = np.linalg.norm(pos[:2] - np.array(target_pos[:2]))
        return max(0, 5.0 - dist) / 5.0

    @staticmethod
    def height_bonus(creature, min_height: float = 0.3, **kwargs) -> float:
        state = creature.get_state()
        height = state.get('height', 0)
        return max(0, height - min_height)


COMPONENT_REGISTRY: Dict[str, Callable] = {
    'velocity': FitnessComponents.velocity,
    'upright': FitnessComponents.upright,
    'distance': FitnessComponents.distance,
    'alive': FitnessComponents.alive,
    'energy_efficiency': FitnessComponents.energy_efficiency,
    'stability': FitnessComponents.stability,
    'approach': FitnessComponents.approach,
    'height_bonus': FitnessComponents.height_bonus,
}


# ================================================================
# KEYWORD-BASED PARSER
# ================================================================

KEYWORD_PATTERNS = {
    r'(lauf|walk|run|geh|renn)': [
        FitnessComponent('distance', weight=3.0),
        FitnessComponent('velocity', weight=2.0),
        FitnessComponent('upright', weight=1.0),
        FitnessComponent('energy_efficiency', weight=0.5),
    ],
    r'(schnell|fast|speed)': [
        FitnessComponent('velocity', weight=4.0),
        FitnessComponent('distance', weight=3.0),
    ],
    r'(steh|stand|balance|gleichgewicht)': [
        FitnessComponent('upright', weight=5.0),
        FitnessComponent('stability', weight=3.0),
        FitnessComponent('alive', weight=1.0),
    ],
    r'(ball|kick|schieb|push|heb|lift|trag|carry)': [
        FitnessComponent('approach', weight=2.0, target='ball'),
        FitnessComponent('distance', weight=1.0),
        FitnessComponent('upright', weight=0.5),
    ],
    r'(hindernis|obstacle|parcour|hürd)': [
        FitnessComponent('distance', weight=3.0),
        FitnessComponent('upright', weight=1.5),
        FitnessComponent('alive', weight=0.5),
    ],
    r'(aufsteh|get.?up|recov)': [
        FitnessComponent('upright', weight=4.0),
        FitnessComponent('height_bonus', weight=3.0),
        FitnessComponent('alive', weight=1.0),
    ],
    r'(effizien|efficien|spar|conserv)': [
        FitnessComponent('energy_efficiency', weight=4.0),
        FitnessComponent('distance', weight=1.0),
    ],
    r'(spring|jump|hüpf|hop|leap)': [
        FitnessComponent('height_bonus', weight=4.0),
        FitnessComponent('upright', weight=2.0),
        FitnessComponent('alive', weight=0.5),
    ],
    r'(kletter|climb|steig)': [
        FitnessComponent('height_bonus', weight=3.0),
        FitnessComponent('distance', weight=2.0),
        FitnessComponent('upright', weight=2.0),
    ],
    r'(explor|erkund|entdeck|neugier|curious)': [
        FitnessComponent('distance', weight=2.0),
        FitnessComponent('upright', weight=1.0),
        FitnessComponent('alive', weight=1.0),
    ],
}

DEFAULT_FITNESS = [
    FitnessComponent('distance', weight=2.0),
    FitnessComponent('upright', weight=1.0),
    FitnessComponent('alive', weight=0.5),
]


class KeywordParser:
    """Regelbasierter Fallback-Parser."""

    def parse(self, instruction: str) -> InstructionSpec:
        instruction_lower = instruction.lower()
        components = []
        matched = False

        for pattern, comps in KEYWORD_PATTERNS.items():
            if re.search(pattern, instruction_lower):
                components.extend(comps)
                matched = True

        if not matched:
            components = list(DEFAULT_FITNESS)

        # Deduplizieren
        seen = {}
        unique = []
        for c in components:
            key = (c.component_type, c.target)
            if key not in seen:
                seen[key] = c
                unique.append(c)
            else:
                if c.weight > seen[key].weight:
                    seen[key].weight = c.weight

        template = 'synpaw'  # Default für MH-FLOCKE
        if re.search(r'(wurm|worm|schlan|snake)', instruction_lower):
            template = 'worm'
        elif re.search(r'(zwei.?bein|biped|two.?leg)', instruction_lower):
            template = 'biped'

        eval_steps = 2000
        if re.search(r'(schnell|quick|fast)', instruction_lower):
            eval_steps = 1000

        return InstructionSpec(
            instruction=instruction,
            creature_template=template,
            fitness_components=unique,
            eval_steps=eval_steps,
        )


# ================================================================
# LLM-BASED PARSER
# ================================================================

LLM_SYSTEM_PROMPT = '''Du bist ein Fitness-Designer für MH-FLOCKE, ein Kreatur-Evolutions-System.
Gegeben eine natürlichsprachliche Instruktion, erzeuge eine JSON-Spezifikation.

Verfügbare fitness_components (type):
- velocity: Geschwindigkeit (weight: 1-5)
- upright: Aufrecht stehen (weight: 1-5)
- distance: Gelaufene Distanz (weight: 1-5)
- alive: Überlebenszeit (weight: 0.5-2)
- energy_efficiency: Wenig Energieverbrauch (weight: 0.5-3)
- stability: Wenig Wackeln (weight: 1-3)
- approach: Nähere dich einem Punkt (weight: 1-3, braucht target)
- height_bonus: Höher = besser (weight: 1-3)

Verfügbare creature_templates: synpaw, biped, quadruped, worm

Antworte NUR mit validem JSON:
{
  "creature_template": "synpaw",
  "fitness_components": [
    {"type": "distance", "weight": 3.0},
    {"type": "upright", "weight": 1.0}
  ],
  "eval_steps": 2000
}'''


class InstructionParser:
    """
    Übersetzt natürliche Sprache → InstructionSpec.
    Versucht erst LLM, fällt auf KeywordParser zurück.
    """

    def __init__(self, llm_adapter=None):
        self._llm = llm_adapter
        self._keyword_parser = KeywordParser()

    def parse(self, instruction: str) -> InstructionSpec:
        if self._llm is not None:
            try:
                spec = self._parse_with_llm(instruction)
                if spec and spec.fitness_components:
                    return spec
            except Exception:
                pass
        return self._keyword_parser.parse(instruction)

    def _parse_with_llm(self, instruction: str) -> Optional[InstructionSpec]:
        response = self._llm.generate(
            f"Instruktion: {instruction}",
            system_prompt=LLM_SYSTEM_PROMPT,
            max_tokens=500,
        )
        if not response.success:
            return None

        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        spec = InstructionSpec(instruction=instruction)
        spec.creature_template = data.get('creature_template', 'synpaw')
        spec.eval_steps = data.get('eval_steps', 2000)

        for comp_data in data.get('fitness_components', []):
            fc = FitnessComponent(
                component_type=comp_data.get('type', 'distance'),
                weight=float(comp_data.get('weight', 1.0)),
                target=comp_data.get('target'),
                params=comp_data.get('params', {}),
            )
            spec.fitness_components.append(fc)

        return spec


# ================================================================
# FITNESS BUILDER — Erzeugt callable Fitness-Funktion
# ================================================================

class FitnessBuilder:
    """Baut aufrufbare Fitness-Funktion aus InstructionSpec."""

    @staticmethod
    def from_spec(spec: InstructionSpec) -> Callable:
        """
        Erstellt Fitness-Funktion aus Spec.
        Returns: fn(creature, total_steps) → float
        """
        components = []
        for fc in spec.fitness_components:
            fn = COMPONENT_REGISTRY.get(fc.component_type)
            if fn is not None:
                components.append((fn, fc.weight, fc.params or {}))

        if not components:
            components = [
                (FitnessComponents.distance, 2.0, {}),
                (FitnessComponents.upright, 1.0, {}),
            ]

        def fitness_fn(creature, total_steps: int = 2000) -> float:
            total = 0.0
            for fn, weight, params in components:
                try:
                    val = fn(creature, total_steps=total_steps, **params)
                    total += weight * val
                except Exception:
                    pass
            return max(0.0, total)

        return fitness_fn

    @staticmethod
    def from_text(text: str, llm_adapter=None) -> Callable:
        """Shortcut: Text → Fitness-Funktion."""
        parser = InstructionParser(llm_adapter)
        spec = parser.parse(text)
        return FitnessBuilder.from_spec(spec)

    @staticmethod
    def describe(spec: InstructionSpec) -> str:
        """Menschenlesbare Beschreibung der Fitness."""
        parts = []
        for fc in spec.fitness_components:
            target_str = f" → {fc.target}" if fc.target else ""
            parts.append(f"  {fc.weight:.1f}× {fc.component_type}{target_str}")
        header = f'Instruction: "{spec.instruction}"'
        body = '\n'.join(parts)
        return f"{header}\nCreature: {spec.creature_template}\nFitness:\n{body}"
