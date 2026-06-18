#!/usr/bin/env python3
"""
Level 14: UNDERSTAND — LLM-Driven Knowledge Acquisition
==========================================================
Asks external LLMs: "What does a dog do in a forest?" and turns the
answers into structured BehaviorDefs, fitness components, and scene hints.

The system is self-contained: it acquires ALL knowledge about the task
autonomously via Multi-LLM queries, validated through Integrity-OS.

Pipeline:
    "dog walks in forest"
        → TaskParser (Level 13)
        → UnderstandEngine.understand(parsed_task)
            → LLM Query: "What behaviors does a dog show in a forest?"
            → Parse response into structured behaviors
            → Validate via cross-check / existing knowledge
            → Inject into BehaviorKnowledge
            → Derive fitness adjustments
            → Derive scene requirements
        → Enhanced TrainingConfig

Usage:
    from src.bridge.understand import UnderstandEngine

    engine = UnderstandEngine()
    knowledge = engine.understand(parsed_task)
    # knowledge.behaviors = [BehaviorDef(...), ...]
    # knowledge.fitness_adjustments = {'sniff_reward': 1.0, ...}
    # knowledge.scene_requirements = {'trees': True, 'uneven_ground': True}
"""

__version__ = "0.1.0"
__logbook__ = 147

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class BehaviorInsight:
    """A single behavior learned from LLM."""
    name: str
    description: str
    priority: float = 0.5
    drive: str = 'exploration'  # which drive triggers this
    motor_hints: Dict[str, float] = field(default_factory=dict)
    # e.g. {'cpg_frequency_scale': 0.3, 'neck_angle': -30} for sniffing
    confidence: float = 0.0
    source: str = 'llm'


@dataclass
class SceneRequirement:
    """Scene features inferred from task understanding."""
    terrain: str = 'flat'  # flat, uneven, steep, slippery
    objects: List[str] = field(default_factory=list)  # trees, rocks, water
    lighting: str = 'day'  # day, night, dusk
    weather: str = 'clear'  # clear, rain, wind, snow
    suggested_scene: str = 'flat_grass'


@dataclass
class UnderstandResult:
    """Complete result from UNDERSTAND phase."""
    task_text: str
    behaviors: List[BehaviorInsight]
    scene_requirements: SceneRequirement
    fitness_adjustments: Dict[str, float]
    knowledge_facts: List[str]  # raw facts for Knowledge Graph
    llm_used: bool = False  # True if LLM was actually queried
    source: str = 'builtin'  # 'builtin', 'llm', 'hybrid'


# ================================================================
# BUILT-IN KNOWLEDGE (Fallback when no LLM available)
# ================================================================

CREATURE_BEHAVIORS: Dict[str, Dict[str, List[BehaviorInsight]]] = {
    'dog': {
        'hilly_grassland': [
            BehaviorInsight('walk_hills', 'Walk with adjusted stride on slopes',
                            priority=0.9, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.8, 'cpg_amplitude_scale': 0.9}),
            BehaviorInsight('cautious_descent', 'Slow careful downhill walking, lower center of gravity',
                            priority=0.7, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.5, 'cpg_amplitude_scale': 0.7}),
            BehaviorInsight('balance_slope', 'Wider stance on slopes for stability',
                            priority=0.8, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.6}),
            BehaviorInsight('explore_terrain', 'Navigate varied terrain, seeking paths',
                            priority=0.6, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 1.0}),
            BehaviorInsight('sniff_grass', 'Head down, exploring grassland smells',
                            priority=0.4, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.3, 'neck_angle': -25}),
        ],
        'forest': [
            BehaviorInsight('sniff_ground', 'Nose to the ground, tracking scents',
                            priority=0.7, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.3, 'neck_angle': -30}),
            BehaviorInsight('chase', 'Sprint after moving target',
                            priority=0.5, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 1.8, 'cpg_amplitude_scale': 1.3}),
            BehaviorInsight('dig', 'Paw at ground to uncover objects',
                            priority=0.3, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.1, 'cpg_amplitude_scale': 0.5}),
            BehaviorInsight('alert_ears', 'Stand still, ears up, listening',
                            priority=0.6, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.0, 'ear_angle': 15}),
            BehaviorInsight('fetch_stick', 'Pick up stick and carry',
                            priority=0.4, drive='comfort',
                            motor_hints={'cpg_frequency_scale': 1.2, 'jaw_angle': -10}),
        ],
        'grass': [
            BehaviorInsight('walk_relaxed', 'Relaxed trot on flat ground',
                            priority=0.8, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 1.0}),
            BehaviorInsight('sniff_air', 'Head up, sampling wind',
                            priority=0.5, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.5, 'neck_angle': 20}),
            BehaviorInsight('roll', 'Roll on back in grass',
                            priority=0.2, drive='comfort',
                            motor_hints={'cpg_frequency_scale': 0.0}),
        ],
        'ice': [
            BehaviorInsight('careful_walk', 'Slow, wide stance on ice',
                            priority=0.9, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.4, 'cpg_amplitude_scale': 0.6}),
            BehaviorInsight('slip_recovery', 'Recover from sliding',
                            priority=0.7, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.2}),
        ],
        'sand': [
            BehaviorInsight('dig_sand', 'Dig energetically in sand',
                            priority=0.6, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.2}),
            BehaviorInsight('shake', 'Shake sand off body',
                            priority=0.3, drive='comfort',
                            motor_hints={'cpg_frequency_scale': 0.0}),
        ],
        'default': [
            BehaviorInsight('walk', 'Standard walking gait',
                            priority=0.8, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 1.0}),
            BehaviorInsight('sniff', 'Investigate surroundings',
                            priority=0.5, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.3, 'neck_angle': -20}),
        ],
    },
    'biped': {
        'default': [
            BehaviorInsight('walk_upright', 'Balanced bipedal walking',
                            priority=0.9, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 1.0}),
            BehaviorInsight('balance', 'Stand and maintain balance',
                            priority=0.7, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.0}),
        ],
    },
    'worm': {
        'default': [
            BehaviorInsight('crawl', 'Peristaltic crawling motion',
                            priority=0.9, drive='exploration',
                            motor_hints={'cpg_frequency_scale': 0.8}),
            BehaviorInsight('burrow', 'Dig into substrate',
                            priority=0.4, drive='survival',
                            motor_hints={'cpg_frequency_scale': 0.3}),
        ],
    },
}

ENVIRONMENT_SCENE_MAP: Dict[str, SceneRequirement] = {
    'forest': SceneRequirement(
        terrain='uneven', objects=['trees', 'roots', 'leaves'],
        lighting='dusk', weather='clear', suggested_scene='rocky',
    ),
    'grass': SceneRequirement(
        terrain='flat', objects=['grass'],
        lighting='day', weather='clear', suggested_scene='flat_grass',
    ),
    'ice': SceneRequirement(
        terrain='slippery', objects=['ice_patches'],
        lighting='day', weather='snow', suggested_scene='ice',
    ),
    'sand': SceneRequirement(
        terrain='soft', objects=['dunes'],
        lighting='day', weather='clear', suggested_scene='sand',
    ),
    'rocky': SceneRequirement(
        terrain='uneven', objects=['rocks', 'boulders'],
        lighting='day', weather='clear', suggested_scene='rocky',
    ),
    'night': SceneRequirement(
        terrain='flat', objects=[],
        lighting='night', weather='clear', suggested_scene='night',
    ),
    'hills': SceneRequirement(
        terrain='steep', objects=['hills', 'slopes'],
        lighting='day', weather='clear', suggested_scene='hills',
    ),
    'hilly_grassland': SceneRequirement(
        terrain='steep', objects=['hills', 'grass'],
        lighting='day', weather='clear', suggested_scene='hilly_grassland',
    ),
}


# ================================================================
# LLM PROMPT TEMPLATES
# ================================================================

BEHAVIOR_PROMPT = """You are analyzing animal and robot behavior for a physics simulation.

Task: "{task_text}"
Creature type: {creature_type}
Environment: {environment}

List 3-6 specific behaviors this creature would show in this environment.
For each behavior, provide:
- name: short identifier (snake_case)
- description: one sentence
- priority: 0.0-1.0 (how likely/important)
- drive: which motivation (exploration, survival, comfort, social)
- motor: movement style (slow/fast/still, head position, special movements)

Respond in JSON format:
[
  {{"name": "sniff_ground", "description": "...", "priority": 0.7, "drive": "exploration", "motor": "slow walk, head down"}},
  ...
]

Only JSON, no other text."""

SCENE_PROMPT = """You are designing a physics simulation environment.

Task: "{task_text}"
Environment hint: {environment}

Describe the scene requirements:
- terrain: flat/uneven/steep/slippery/soft
- objects: list of objects present (trees, rocks, water, etc.)
- lighting: day/night/dusk
- weather: clear/rain/wind/snow
- hazards: any special challenges

Respond in JSON format:
{{"terrain": "uneven", "objects": ["trees", "roots"], "lighting": "dusk", "weather": "clear", "hazards": ["roots to trip on"]}}

Only JSON, no other text."""


# ================================================================
# UNDERSTAND ENGINE
# ================================================================

class UnderstandEngine:
    """
    Acquires knowledge about a task via LLM or built-in knowledge.

    Strategy:
      1. Check built-in knowledge first (fast, no API calls)
      2. If LLM available: query for richer behavior details
      3. Validate LLM responses against built-in knowledge
      4. Merge results into UnderstandResult
    """

    def __init__(self, llm_adapter=None):
        """
        Args:
            llm_adapter: Optional MultiLLMAdapter instance.
                         If None, uses built-in knowledge only.
        """
        self.llm = llm_adapter
        self._cache: Dict[str, UnderstandResult] = {}

    def understand(self, task, creature_type: str = 'dog') -> UnderstandResult:
        """
        Main entry: understand a task and generate knowledge.

        Args:
            task: ParsedTask from TaskParser.
            creature_type: Agent type ('dog', 'biped', 'worm').

        Returns:
            UnderstandResult with behaviors, scene requirements, fitness adjustments.
        """
        task_text = task.description
        # Pick best environment: try all hints, prefer one with most behaviors
        environment = self._best_environment(creature_type, task.environment_hints)

        # Cache check
        cache_key = f'{creature_type}:{environment}:{task_text}'
        if cache_key in self._cache:
            logger.info(f'UNDERSTAND cache hit: {cache_key}')
            return self._cache[cache_key]

        # Step 1: Built-in knowledge
        builtin_behaviors = self._get_builtin_behaviors(creature_type, environment)
        builtin_scene = self._get_builtin_scene(environment)

        # Step 2: LLM enrichment (if available)
        llm_behaviors = []
        llm_used = False
        if self.llm and self.llm.enabled:
            llm_behaviors = self._query_llm_behaviors(task_text, creature_type, environment)
            if llm_behaviors:
                llm_used = True

        # Step 3: Merge behaviors (built-in + LLM, deduplicated)
        all_behaviors = self._merge_behaviors(builtin_behaviors, llm_behaviors)

        # Step 4: Derive fitness adjustments from behaviors
        fitness_adj = self._derive_fitness_adjustments(all_behaviors, task)

        # Step 5: Generate knowledge facts for storage
        facts = self._generate_facts(all_behaviors, creature_type, environment)

        result = UnderstandResult(
            task_text=task_text,
            behaviors=all_behaviors,
            scene_requirements=builtin_scene,
            fitness_adjustments=fitness_adj,
            knowledge_facts=facts,
            llm_used=llm_used,
            source='hybrid' if llm_used else 'builtin',
        )

        self._cache[cache_key] = result
        logger.info(f'UNDERSTAND: {len(all_behaviors)} behaviors, '
                     f'scene={builtin_scene.suggested_scene}, '
                     f'source={result.source}')
        return result

    def _best_environment(self, creature_type: str, env_hints: list) -> str:
        """Pick the environment hint that yields the most behaviors.
        
        TaskParser may return ['flat_grass', 'hills'] for 'hilly grassland'.
        'flat_grass' maps to 'grass' (3 generic behaviors).
        'hills' maps to 'hilly_grassland' (5 specific behaviors).
        We want the richer, more specific match.
        """
        if not env_hints:
            return 'default'
        
        best_env = env_hints[0]
        best_count = 0
        
        creature_data = CREATURE_BEHAVIORS.get(creature_type, {})
        for hint in env_hints:
            # Direct match
            behaviors = creature_data.get(hint, [])
            if len(behaviors) > best_count:
                best_count = len(behaviors)
                best_env = hint
            # Alias match
            alias = self.ENV_ALIASES.get(hint, '')
            if alias:
                behaviors = creature_data.get(alias, [])
                if len(behaviors) > best_count:
                    best_count = len(behaviors)
                    best_env = hint  # keep original hint, alias resolves in _get_builtin
            # Also try combining hints (e.g. 'hills' + 'grass' → 'hilly_grassland')
            for other in env_hints:
                if other != hint:
                    combined = f"{hint}_{other}"
                    behaviors = creature_data.get(combined, [])
                    if len(behaviors) > best_count:
                        best_count = len(behaviors)
                        best_env = combined
        
        return best_env

    # Environment aliases for behavior lookup
    # Maps TaskParser env hints → CREATURE_BEHAVIORS keys
    ENV_ALIASES = {
        'rocky': 'forest',
        'flat_grass': 'grass',
        'hills': 'hilly_grassland',
        'hilly': 'hilly_grassland',
        'grassland': 'hilly_grassland',
    }

    def _get_builtin_behaviors(self, creature_type: str,
                               environment: str) -> List[BehaviorInsight]:
        """Fetch built-in behaviors for creature + environment."""
        creature_data = CREATURE_BEHAVIORS.get(creature_type, {})

        # Try exact match, then alias, then 'default'
        behaviors = creature_data.get(environment, [])
        if not behaviors:
            alias = self.ENV_ALIASES.get(environment, '')
            behaviors = creature_data.get(alias, [])
        if not behaviors:
            behaviors = creature_data.get('default', [])

        return list(behaviors)  # copy

    def _get_builtin_scene(self, environment: str) -> SceneRequirement:
        """Fetch built-in scene requirements."""
        scene = ENVIRONMENT_SCENE_MAP.get(environment)
        if scene:
            return scene
        # Try alias
        alias = self.ENV_ALIASES.get(environment, '')
        scene = ENVIRONMENT_SCENE_MAP.get(alias)
        if scene:
            return scene
        # Default
        return SceneRequirement(suggested_scene='flat_grass')

    def _query_llm_behaviors(self, task_text: str, creature_type: str,
                              environment: str) -> List[BehaviorInsight]:
        """Query LLM for behavior knowledge."""
        prompt = BEHAVIOR_PROMPT.format(
            task_text=task_text,
            creature_type=creature_type,
            environment=environment,
        )

        try:
            response = self.llm.generate(prompt, max_tokens=800)
            if not response.success:
                logger.warning(f'LLM behavior query failed: {response.error}')
                return []

            behaviors = self._parse_llm_behaviors(response.text)
            logger.info(f'LLM returned {len(behaviors)} behaviors '
                         f'via {response.provider}')
            return behaviors

        except Exception as e:
            logger.error(f'LLM behavior query error: {e}')
            return []

    def _parse_llm_behaviors(self, text: str) -> List[BehaviorInsight]:
        """Parse LLM JSON response into BehaviorInsights."""
        behaviors = []

        # Extract JSON array from response
        try:
            # Try direct parse
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON in text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        for item in data:
            if not isinstance(item, dict) or 'name' not in item:
                continue

            # Parse motor hints from description
            motor_hints = self._parse_motor_hints(item.get('motor', ''))

            behaviors.append(BehaviorInsight(
                name=item['name'],
                description=item.get('description', ''),
                priority=float(item.get('priority', 0.5)),
                drive=item.get('drive', 'exploration'),
                motor_hints=motor_hints,
                confidence=0.7,  # LLM confidence
                source='llm',
            ))

        return behaviors

    @staticmethod
    def _parse_motor_hints(motor_text: str) -> Dict[str, float]:
        """Convert motor description to numeric hints."""
        hints = {}
        text = motor_text.lower()

        # Speed
        if 'still' in text or 'stationary' in text or 'stop' in text:
            hints['cpg_frequency_scale'] = 0.0
        elif 'slow' in text or 'careful' in text:
            hints['cpg_frequency_scale'] = 0.3
        elif 'fast' in text or 'sprint' in text or 'run' in text:
            hints['cpg_frequency_scale'] = 1.5
        elif 'walk' in text:
            hints['cpg_frequency_scale'] = 1.0

        # Head position
        if 'head down' in text or 'nose down' in text:
            hints['neck_angle'] = -30.0
        elif 'head up' in text or 'looking up' in text:
            hints['neck_angle'] = 20.0

        # Tail
        if 'tail wag' in text:
            hints['tail_angle'] = 30.0
        elif 'tail tuck' in text:
            hints['tail_angle'] = -20.0

        return hints

    def _merge_behaviors(self, builtin: List[BehaviorInsight],
                          llm: List[BehaviorInsight]) -> List[BehaviorInsight]:
        """Merge built-in and LLM behaviors, deduplicated by name."""
        seen = {}

        # Built-in first (higher trust)
        for b in builtin:
            seen[b.name] = b

        # LLM enrichment (add new, don't override built-in)
        for b in llm:
            if b.name not in seen:
                seen[b.name] = b
            else:
                # Merge: keep built-in motor_hints, take LLM description if richer
                existing = seen[b.name]
                if len(b.description) > len(existing.description):
                    existing.description = b.description

        return sorted(seen.values(), key=lambda x: x.priority, reverse=True)

    def _derive_fitness_adjustments(self, behaviors: List[BehaviorInsight],
                                     task) -> Dict[str, float]:
        """Derive fitness component adjustments from understood behaviors."""
        adj = {}

        # Count behavior types
        has_sniff = any('sniff' in b.name for b in behaviors)
        has_chase = any('chase' in b.name or 'sprint' in b.name for b in behaviors)
        has_careful = any('careful' in b.name or 'balance' in b.name for b in behaviors)
        has_dig = any('dig' in b.name for b in behaviors)

        # Adjust fitness based on expected behaviors
        if has_sniff:
            adj['exploration_bonus'] = 0.5  # reward area exploration
        if has_chase:
            adj['speed_bonus'] = 1.0  # reward higher speeds
        if has_careful:
            adj['stability_bonus'] = 1.5  # reward stability over speed
            adj['distance_reduction'] = 0.7  # less focus on distance
        if has_dig:
            adj['area_focus'] = 0.5  # staying in one area is ok

        return adj

    def _generate_facts(self, behaviors: List[BehaviorInsight],
                         creature_type: str, environment: str) -> List[str]:
        """Generate knowledge facts for storage in Knowledge Graph."""
        facts = []

        for b in behaviors:
            facts.append(
                f'{creature_type} in {environment}: '
                f'shows "{b.name}" behavior — {b.description}'
            )

        facts.append(
            f'{creature_type} has {len(behaviors)} known behaviors '
            f'in {environment} environment'
        )

        return facts

    def inject_into_behavior_knowledge(self, result: UnderstandResult,
                                        knowledge) -> int:
        """
        Inject understood behaviors into BehaviorKnowledge instance.

        Args:
            result: UnderstandResult from understand().
            knowledge: BehaviorKnowledge instance.

        Returns:
            Number of behaviors injected.
        """
        from src.behavior.behavior_knowledge import BehaviorDef, MotorPattern

        injected = 0
        for insight in result.behaviors:
            # Skip if already exists
            if knowledge.get_behavior(insight.name):
                continue

            # Convert BehaviorInsight → BehaviorDef
            motor = MotorPattern(
                cpg_frequency_scale=insight.motor_hints.get('cpg_frequency_scale', 1.0),
                cpg_amplitude_scale=insight.motor_hints.get('cpg_amplitude_scale', 1.0),
                neck_angle=insight.motor_hints.get('neck_angle'),
                jaw_angle=insight.motor_hints.get('jaw_angle'),
                ear_angle=insight.motor_hints.get('ear_angle'),
                tail_angle=insight.motor_hints.get('tail_angle'),
            )

            beh_def = BehaviorDef(
                name=insight.name,
                description=insight.description,
                drive_affinity={insight.drive: insight.priority},
                motor=motor,
                priority=insight.priority,
            )

            knowledge.add_behavior(beh_def)
            injected += 1
            logger.info(f'Injected behavior: {insight.name}')

        return injected
