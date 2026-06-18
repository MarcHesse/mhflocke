#!/usr/bin/env python3
"""
Level 13: LLM-Bridge — Language → Embodied Training
=======================================================
Bridge between natural language and SNN-based embodied training.

Kernidee: Ein Benutzer (oder das System selbst) kann in natürlicher
Sprache Aufgaben beschreiben, und die Bridge übersetzt das in:
  1. MJCF-Modifikationen (neuer Körper / neue Umgebung)
  2. Fitness-Funktionen (was optimiert wird)
  3. Training-Parameter (wie trainiert wird)
  4. Behavior-Wissen (was die Kreatur wissen sollte)

Consciousness Level 12: "Sprachfähig"
  — Die Kreatur kann über ihre eigene Erfahrung kommunizieren
  — Bidirektional: Sprache→Training UND Erfahrung→Sprache

Architektur:
  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │  Natürliche  │────▶│  TaskParser   │────▶│  Fitness-        │
  │  Sprache     │     │  (Intent →    │     │  Generator       │
  │  "Lerne zu   │     │   Task)       │     │  (Task → Reward) │
  │   springen"  │     └──────┬───────┘     └────────┬────────┘
  └─────────────┘            │                       │
                             ▼                       ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  AutoMJCF     │     │  Training-       │
                    │  (Task →      │     │  Orchestrator    │
                    │   Körper/Welt)│     │  (Config + Run)  │
                    └──────────────┘     └────────┬────────┘
                                                  │
                    ┌──────────────┐              ▼
                    │  Experience   │◀────  SNN + Evolution
                    │  Narrator     │     (Embodied Training)
                    │  (Erfahrung → │
                    │   Sprache)    │
                    └──────────────┘
"""

__version__ = "0.1.0"
__logbook__ = 145

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import json
import re
import os
import time
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# TASK TYPES
# ═══════════════════════════════════════════════════════

class TaskType(Enum):
    """Arten von Aufgaben die die Bridge versteht."""
    LOCOMOTION = "locomotion"       # Laufen, Springen, Klettern
    MANIPULATION = "manipulation"   # Greifen, Tragen, Schieben
    NAVIGATION = "navigation"       # Ziel finden, Hindernis vermeiden
    SOCIAL = "social"               # Andere Kreaturen, Kooperation
    EXPLORATION = "exploration"     # Neue Umgebung erkunden
    SURVIVAL = "survival"           # Balance halten, nicht fallen
    CUSTOM = "custom"               # Frei definiert


@dataclass
class ParsedTask:
    """Ergebnis des TaskParsers."""
    task_type: TaskType
    description: str                          # Original-Beschreibung
    objectives: List[str]                     # Zerlegte Ziele
    constraints: List[str]                    # Einschränkungen
    environment_hints: List[str]              # Umgebungs-Anforderungen
    difficulty: float = 0.5                   # 0=trivial, 1=extrem
    duration_hint: Optional[int] = None       # Geschätzte Steps
    requires_body_change: bool = False        # Braucht MJCF-Änderung
    body_requirements: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0                   # Parser-Konfidenz


@dataclass
class FitnessSpec:
    """Spezifikation einer Fitness-Funktion."""
    name: str
    components: Dict[str, float]              # Gewichte: distance, upright, alive, ...
    bonus_conditions: List[Dict[str, Any]]    # Bonus-Bedingungen
    penalty_conditions: List[Dict[str, Any]]  # Straf-Bedingungen
    curriculum: List[Dict[str, Any]]          # Stufenweiser Aufbau
    success_threshold: float = 0.0            # Ab wann "geschafft"


@dataclass 
class TrainingConfig:
    """Full configuration for a training run."""
    task: ParsedTask
    fitness: FitnessSpec
    environment: str = "flat_grass"
    neurons: int = 5000
    population: int = 50
    generations: int = 200
    eval_steps: int = 5000
    scene: str = "flat_grass"
    body_template: str = "synpaw"
    mjcf_modifications: Dict[str, Any] = field(default_factory=dict)
    # Level 14: Skill & Resume support
    skill_name: Optional[str] = None
    freeze_skill: bool = False
    base_skills: List[str] = field(default_factory=list)
    continue_from: bool = False  # Resume from checkpoint


# ═══════════════════════════════════════════════════════
# TASK PARSER
# ═══════════════════════════════════════════════════════

class TaskParser:
    """
    Natürliche Sprache → ParsedTask.
    
    Versteht Aufgaben wie:
      "Lerne auf der Wiese zu laufen"
      "Spring über Hindernisse"
      "Klettere den Hügel hoch ohne umzufallen"
      "Bewege dich auf eisigem Boden vorwärts"
      "Lerne auf zwei Beinen zu balancieren"
    
    Zweistufig:
      1. Keyword-basiertes Matching (schnell, kein LLM)
      2. Optional: LLM-Verfeinerung (wenn Adapter verfügbar)
    """

    # Keyword → TaskType Mapping
    TASK_KEYWORDS = {
        TaskType.LOCOMOTION: [
            'lauf', 'geh', 'renn', 'sprint', 'walk', 'run', 'move',
            'spring', 'jump', 'hop', 'hüpf', 'leap',
            'kletter', 'climb', 'steig',
            'schwimm', 'swim', 'kriech', 'crawl',
            'trab', 'trot', 'galoppier', 'gallop',
        ],
        TaskType.NAVIGATION: [
            'find', 'such', 'navigier', 'navigate', 'ziel', 'target',
            'vermeide', 'avoid', 'umgeh', 'hindernis', 'obstacle',
            'pfad', 'path', 'route', 'weg',
        ],
        TaskType.MANIPULATION: [
            'greif', 'grab', 'trag', 'carry', 'schieb', 'push',
            'zieh', 'pull', 'heb', 'lift', 'halt', 'hold',
        ],
        TaskType.SOCIAL: [
            'folge', 'follow', 'kooperier', 'cooperate',
            'kommunizier', 'communicate', 'zusammen', 'together',
            'gruppe', 'group', 'schwarm', 'swarm',
        ],
        TaskType.EXPLORATION: [
            'erkund', 'explore', 'entdeck', 'discover',
            'neugi', 'curious', 'umschau', 'look around',
        ],
        TaskType.SURVIVAL: [
            'überleb', 'survive', 'balanc', 'balance',
            'aufrecht', 'upright', 'steh', 'stand', 'stabil',
        ],
    }

    # Umgebungs-Keywords
    ENVIRONMENT_KEYWORDS = {
        'flat_grass': ['wiese', 'gras', 'flach', 'meadow', 'grass', 'flat'],
        'ice': ['eis', 'ice', 'glatt', 'slippery', 'rutsch'],
        'sand': ['sand', 'wüste', 'desert', 'strand', 'beach'],
        'rocky': ['fels', 'rock', 'stein', 'stone', 'berg', 'mountain',
                  'forest', 'wald', 'woods'],
        'hills': ['hügel', 'hill', 'steigung', 'slope', 'anstieg'],
        'night': ['nacht', 'night', 'dunkel', 'dark'],
        'windy': ['wind', 'sturm', 'storm'],
        'obstacle_course': ['hindernis', 'obstacle', 'parkour', 'parcour'],
    }

    # Schwierigkeits-Modifikatoren
    DIFFICULTY_KEYWORDS = {
        'einfach': -0.2, 'simple': -0.2, 'leicht': -0.2, 'easy': -0.2,
        'schnell': 0.1, 'fast': 0.1, 'weit': 0.1, 'far': 0.1,
        'schwer': 0.2, 'hard': 0.2, 'schwierig': 0.2, 'difficult': 0.2,
        'extrem': 0.3, 'extreme': 0.3, 'perfekt': 0.3, 'perfect': 0.3,
    }

    def parse(self, text: str) -> ParsedTask:
        """Parse natürliche Sprache in eine Task-Spezifikation."""
        text_lower = text.lower().strip()
        words = re.findall(r'\w+', text_lower)

        # 1. Task-Type bestimmen
        task_type, type_confidence = self._detect_task_type(words)

        # 2. Umgebung erkennen
        environment = self._detect_environment(words)

        # 3. Schwierigkeit schätzen
        difficulty = self._estimate_difficulty(words)

        # 4. Objectives extrahieren
        objectives = self._extract_objectives(text_lower, task_type)

        # 5. Constraints extrahieren
        constraints = self._extract_constraints(text_lower)

        # 6. Body Requirements
        body_reqs, needs_change = self._check_body_requirements(text_lower)

        return ParsedTask(
            task_type=task_type,
            description=text,
            objectives=objectives,
            constraints=constraints,
            environment_hints=environment,
            difficulty=difficulty,
            requires_body_change=needs_change,
            body_requirements=body_reqs,
            confidence=type_confidence,
        )

    def _detect_task_type(self, words: List[str]) -> Tuple[TaskType, float]:
        """Erkennt den Task-Typ aus Keywords."""
        scores = {tt: 0.0 for tt in TaskType}
        
        for tt, keywords in self.TASK_KEYWORDS.items():
            for word in words:
                for kw in keywords:
                    if word.startswith(kw) or kw.startswith(word):
                        scores[tt] += 1.0
                    elif kw in word:
                        scores[tt] += 0.5

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score == 0:
            return TaskType.CUSTOM, 0.1
        
        confidence = min(1.0, best_score / 3.0)
        return best_type, confidence

    def _detect_environment(self, words: List[str]) -> List[str]:
        """Erkennt Umgebungs-Hinweise."""
        hints = []
        for env, keywords in self.ENVIRONMENT_KEYWORDS.items():
            for word in words:
                if any(kw in word for kw in keywords):
                    hints.append(env)
                    break
        return hints

    def _estimate_difficulty(self, words: List[str]) -> float:
        """Schätzt Schwierigkeit (0-1)."""
        diff = 0.5
        for word in words:
            for kw, mod in self.DIFFICULTY_KEYWORDS.items():
                if kw in word:
                    diff += mod
        return max(0.0, min(1.0, diff))

    def _extract_objectives(self, text: str, task_type: TaskType) -> List[str]:
        """Extrahiert konkrete Ziele."""
        objectives = []
        
        if task_type == TaskType.LOCOMOTION:
            if any(w in text for w in ['lauf', 'walk', 'geh', 'run', 'renn']):
                objectives.append("maximize_forward_distance")
            if any(w in text for w in ['spring', 'jump', 'hüpf']):
                objectives.append("maximize_jump_height")
                objectives.append("land_upright")
            if any(w in text for w in ['kletter', 'climb']):
                objectives.append("maximize_vertical_distance")
            objectives.append("maintain_balance")
            
        elif task_type == TaskType.NAVIGATION:
            objectives.append("reach_target")
            if 'vermeid' in text or 'avoid' in text:
                objectives.append("avoid_obstacles")
                
        elif task_type == TaskType.SURVIVAL:
            objectives.append("maintain_balance")
            objectives.append("maximize_alive_time")
            
        elif task_type == TaskType.EXPLORATION:
            objectives.append("maximize_area_covered")
            objectives.append("maintain_balance")

        if not objectives:
            objectives.append("maximize_forward_distance")
            objectives.append("maintain_balance")
            
        return objectives

    def _extract_constraints(self, text: str) -> List[str]:
        """Extrahiert Einschränkungen."""
        constraints = []
        if any(w in text for w in ['ohne umzufallen', 'nicht fallen', 'aufrecht']):
            constraints.append("no_falling")
        if any(w in text for w in ['langsam', 'vorsichtig', 'slow', 'careful']):
            constraints.append("energy_efficient")
        if any(w in text for w in ['schnell', 'fast']):
            constraints.append("time_limited")
        return constraints

    def _check_body_requirements(self, text: str) -> Tuple[Dict, bool]:
        """Prüft ob Körper-Änderungen nötig sind."""
        reqs = {}
        needs_change = False
        
        if any(w in text for w in ['zwei bein', 'biped', 'two leg', 'aufrecht geh']):
            reqs['template'] = 'biped'
            needs_change = True
        if any(w in text for w in ['vier bein', 'quadruped', 'four leg']):
            reqs['template'] = 'quadruped'
        if any(w in text for w in ['groß', 'large', 'big', 'stark', 'strong']):
            reqs['scale'] = 1.5
            needs_change = True
        if any(w in text for w in ['klein', 'small', 'tiny']):
            reqs['scale'] = 0.5
            needs_change = True
        if any(w in text for w in ['flügel', 'wing', 'flieg', 'fly']):
            reqs['appendage'] = 'wings'
            needs_change = True
            
        return reqs, needs_change


# ═══════════════════════════════════════════════════════
# FITNESS GENERATOR
# ═══════════════════════════════════════════════════════

class FitnessGenerator:
    """
    ParsedTask → FitnessSpec.
    
    Übersetzt abstrakte Ziele in konkrete Fitness-Komponenten
    die der Evolution-Engine übergeben werden.
    """

    # Objective → Fitness-Mapping
    OBJECTIVE_WEIGHTS = {
        'maximize_forward_distance': {'distance': 3.0, 'upright': 1.0},
        'maximize_jump_height': {'height': 4.0, 'upright': 2.0, 'landing': 1.5},
        'maximize_vertical_distance': {'vertical_dist': 3.0, 'upright': 2.0},
        'maintain_balance': {'upright': 2.0, 'alive': 0.5},
        'maximize_alive_time': {'alive': 3.0, 'upright': 1.0},
        'reach_target': {'target_dist': 3.0, 'upright': 1.0},
        'avoid_obstacles': {'obstacle_penalty': -2.0, 'upright': 1.0},
        'maximize_area_covered': {'area': 2.0, 'distance': 1.0, 'upright': 1.0},
        'land_upright': {'upright': 3.0, 'landing_stability': 2.0},
        'energy_efficient': {'energy_penalty': -0.01},
    }

    def generate(self, task: ParsedTask) -> FitnessSpec:
        """Generiert Fitness-Spezifikation aus Task."""
        # Komponenten zusammenbauen
        components = {}
        for objective in task.objectives:
            weights = self.OBJECTIVE_WEIGHTS.get(objective, {})
            for comp, weight in weights.items():
                components[comp] = components.get(comp, 0.0) + weight

        # Schwierigkeits-Skalierung
        difficulty_scale = 0.5 + task.difficulty
        for comp in components:
            if components[comp] > 0:
                components[comp] *= difficulty_scale

        # Constraints als Bedingungen
        bonus = []
        penalty = []
        for constraint in task.constraints:
            if constraint == "no_falling":
                penalty.append({
                    'condition': 'is_fallen',
                    'penalty': -0.5 * difficulty_scale,
                })
            elif constraint == "energy_efficient":
                components['energy_penalty'] = components.get('energy_penalty', 0) - 0.01
            elif constraint == "time_limited":
                bonus.append({
                    'condition': 'early_finish',
                    'bonus': 1.0,
                })

        # Curriculum: stufenweise schwieriger
        curriculum = self._generate_curriculum(task, components)

        # Erfolgs-Schwelle
        threshold = 0.5 * difficulty_scale
        if 'distance' in components:
            threshold = 2.0 * difficulty_scale  # Meter

        return FitnessSpec(
            name=f"fitness_{task.task_type.value}",
            components=components,
            bonus_conditions=bonus,
            penalty_conditions=penalty,
            curriculum=curriculum,
            success_threshold=threshold,
        )

    def _generate_curriculum(self, task: ParsedTask, 
                             components: Dict[str, float]) -> List[Dict]:
        """Erzeugt stufenweisen Trainingsplan."""
        stages = []
        
        # Stage 1: Basisbalance (immer)
        stage1 = dict(components)
        stage1['upright'] = stage1.get('upright', 0) + 2.0
        stage1['distance'] = stage1.get('distance', 0) * 0.3
        stages.append({
            'generation_range': (0, 50),
            'weights': stage1,
            'description': 'Basisbalance',
        })

        # Stage 2: Hauptziel einführen
        stage2 = dict(components)
        stages.append({
            'generation_range': (50, 150),
            'weights': stage2,
            'description': 'Hauptziel',
        })

        # Stage 3: Verfeinerung
        stage3 = dict(components)
        for k in stage3:
            if stage3[k] > 0:
                stage3[k] *= 1.5
        stages.append({
            'generation_range': (150, None),
            'weights': stage3,
            'description': 'Verfeinerung',
        })

        return stages


# ═══════════════════════════════════════════════════════
# EXPERIENCE NARRATOR
# ═══════════════════════════════════════════════════════

class ExperienceNarrator:
    """
    Erfahrung → Sprache.
    
    Übersetzt den internen Zustand der Kreatur in
    verständliche Beschreibungen. Das Gegenstück zum TaskParser.
    
    Nutzt brain_state, training_stats und Episode-Daten um
    zu erzählen was die Kreatur erlebt/gelernt hat.
    """

    # Emotion → Beschreibung
    EMOTION_WORDS = {
        'joy': ['zufrieden', 'glücklich', 'freudig'],
        'fear': ['ängstlich', 'unsicher', 'vorsichtig'],
        'anger': ['frustriert', 'wütend', 'ungeduldig'],
        'sadness': ['traurig', 'entmutigt', 'erschöpft'],
        'surprise': ['überrascht', 'erstaunt', 'verwirrt'],
        'neutral': ['ruhig', 'ausgeglichen', 'konzentriert'],
        'curiosity': ['neugierig', 'aufmerksam', 'explorativ'],
    }

    # Drive → Beschreibung
    DRIVE_WORDS = {
        'survival': 'Sicherheit',
        'exploration': 'Entdeckung',
        'comfort': 'Komfort',
        'social': 'Gemeinschaft',
        'competence': 'Können',
    }

    def narrate_state(self, brain_state: Dict) -> str:
        """Erzählt den aktuellen Zustand."""
        parts = []

        # Emotion
        emo = brain_state.get('emotion', {})
        dominant = emo.get('dominant_emotion', 'neutral')
        valence = emo.get('valence', 0)
        words = self.EMOTION_WORDS.get(dominant, ['ruhig'])
        parts.append(f"Ich fühle mich {words[0]}")
        if valence > 0.3:
            parts.append("und die Dinge laufen gut")
        elif valence < -0.3:
            parts.append("aber etwas stimmt nicht")

        # Drive
        drives = brain_state.get('drives', {})
        dom_drive = drives.get('dominant', 'survival')
        drive_word = self.DRIVE_WORDS.get(dom_drive, dom_drive)
        parts.append(f"Mein stärkster Antrieb ist {drive_word}")

        # Consciousness
        meta = brain_state.get('metacognition', {})
        cl = meta.get('consciousness_level', 0)
        if cl >= 5:
            parts.append("Ich bin mir meiner selbst bewusst")
        elif cl >= 3:
            parts.append("Ich nehme meine Umgebung wahr")
        else:
            parts.append("Ich reagiere auf Reize")

        # Körper
        body = brain_state.get('body_schema', {})
        if body.get('avg_error', 0) > 0.3:
            parts.append("Mein Körpergefühl ist unsicher")
        else:
            parts.append("Ich spüre meinen Körper gut")

        return ". ".join(parts) + "."

    def narrate_training(self, stats: Dict) -> str:
        """Erzählt über den Trainingsfortschritt (skill-aware)."""
        parts = []

        distance = stats.get('best_distance', 0)
        gen = stats.get('generation', 0)
        falls = stats.get('falls', 0)
        skill = stats.get('skill', 'walk')
        steps = stats.get('steps', gen)

        # Skill-specific narration
        if skill == 'stand':
            if falls < 5:
                parts.append("Ich stehe stabil auf allen Vieren")
            elif distance < 0.2:
                parts.append("Ich lerne stillzustehen")
            else:
                parts.append("Ich versuche ruhig stehen zu bleiben")
        elif skill == 'jump':
            if distance > 1:
                parts.append(f"Ich kann schon {distance:.1f}m hoch springen")
            else:
                parts.append("Ich versuche vom Boden abzuheben")
        elif skill == 'sprint':
            if distance > 3:
                parts.append(f"Ich renne {distance:.1f}m weit")
            else:
                parts.append("Ich lerne schneller zu werden")
        else:  # walk and others
            if distance > 5:
                parts.append(f"Ich habe gelernt {distance:.1f} Meter weit zu laufen")
            elif distance > 1:
                parts.append(f"Ich schaffe schon {distance:.1f} Meter")
            else:
                parts.append("Ich lerne meine ersten Schritte")

        if steps > 10000:
            parts.append(f"nach {steps} Trainingsschritten")
        elif steps > 0:
            parts.append(f"nach {steps} Schritten Übung")

        if falls > 100:
            parts.append("Ich bin oft hingefallen, aber jedes Mal aufgestanden")
        elif falls > 10:
            parts.append("Ein paar Stürze waren dabei")
        elif falls < 3 and skill == 'stand':
            parts.append("Ohne umzufallen")

        return ". ".join(parts) + "."

    def narrate_episode(self, episode: Dict) -> str:
        """Erzählt eine einzelne Episode."""
        parts = []
        
        dist = episode.get('distance', 0)
        duration = episode.get('duration_steps', 0)
        fell = episode.get('fell', False)
        
        if dist > 3:
            parts.append(f"Ich bin {dist:.1f}m gelaufen")
        elif dist > 0.5:
            parts.append(f"Ich habe {dist:.1f}m geschafft")
        else:
            parts.append("Ich konnte mich kaum bewegen")
            
        if fell:
            parts.append("und bin dann hingefallen")
        elif duration > 1000:
            parts.append("und bin stabil geblieben")
            
        return ". ".join(parts) + "."


# ═══════════════════════════════════════════════════════
# TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════

class TrainingOrchestrator:
    """
    Level 14: Unified Orchestrator with UNDERSTAND + BUILD + TRAIN + GROW.
    
    Full Pipeline:
      Text → TaskParser → UNDERSTAND (LLM knowledge) → BUILD (scene + body)
           → FitnessGenerator → TrainingConfig → Evolution → GROW (next tasks)
    """

    def __init__(self, llm_adapter=None):
        self.parser = TaskParser()
        self.fitness_gen = FitnessGenerator()
        self.narrator = ExperienceNarrator()
        self.llm = llm_adapter

        # Level 14: Lazy-loaded engines
        self._understand_engine = None
        self._scene_generator = None
        self._grow_engine = None

    @property
    def understand(self):
        """Lazy-load UnderstandEngine."""
        if self._understand_engine is None:
            from src.bridge.understand import UnderstandEngine
            self._understand_engine = UnderstandEngine(llm_adapter=self.llm)
        return self._understand_engine

    @property
    def scene_gen(self):
        """Lazy-load SceneGenerator."""
        if self._scene_generator is None:
            from src.bridge.scene_generator import SceneGenerator
            self._scene_generator = SceneGenerator()
        return self._scene_generator

    @property
    def grow(self):
        """Lazy-load GrowEngine."""
        if self._grow_engine is None:
            from src.bridge.grow import GrowEngine
            self._grow_engine = GrowEngine(llm_adapter=self.llm)
        return self._grow_engine

    def plan(self, text: str, creature_name: str = "mogli",
             neurons: int = 5000, population: int = 50,
             generations: int = 200) -> TrainingConfig:
        """
        Plant einen Training-Run aus natürlicher Sprache.
        
        Args:
            text: Aufgaben-Beschreibung
            creature_name: Name der Kreatur
            neurons: SNN-Neuronen
            population: Populationsgröße
            generations: Generationen
            
        Returns:
            TrainingConfig mit allen Parametern
        """
        # 1. Parse task
        task = self.parser.parse(text)

        # 2. UNDERSTAND: acquire knowledge about the task
        creature_type = task.body_requirements.get('template', 'synpaw')
        # Resolve alias: synpaw → dog for knowledge lookup
        knowledge_type = {'synpaw': 'dog', 'quadruped': 'dog'}.get(
            creature_type, creature_type)
        understand_result = self.understand.understand(task, knowledge_type)

        # 3. Generate fitness (enriched by UNDERSTAND)
        fitness = self.fitness_gen.generate(task)

        # Apply UNDERSTAND fitness adjustments
        for key, value in understand_result.fitness_adjustments.items():
            if key.endswith('_reduction'):
                # Reduce an existing component
                base_key = key.replace('_reduction', '')
                if base_key in fitness.components:
                    fitness.components[base_key] *= value
            elif key.endswith('_bonus'):
                # Add bonus component
                base_key = key.replace('_bonus', '')
                fitness.components[base_key] = (
                    fitness.components.get(base_key, 0) + value)

        # 4. BUILD Scene
        scene_key = understand_result.scene_requirements.suggested_scene
        if not scene_key and task.environment_hints:
            scene_key = task.environment_hints[0]
        if not scene_key:
            scene_key = 'flat_grass'

        # Store generated scene for later use by run()
        self._last_understand = understand_result
        self._last_generated_scene = None
        if understand_result.scene_requirements.objects:
            self._last_generated_scene = self.scene_gen.generate(
                understand_result.scene_requirements,
                scene_key,
            )

        # Body Template
        template = task.body_requirements.get('template', 'synpaw')

        # Config
        config = TrainingConfig(
            task=task,
            fitness=fitness,
            environment=scene_key,
            neurons=neurons,
            population=population,
            generations=generations,
            scene=scene_key,
            body_template=template,
            mjcf_modifications=task.body_requirements,
        )

        return config

    def explain_plan(self, config: TrainingConfig) -> str:
        """Erklärt den Trainingsplan in natürlicher Sprache."""
        task = config.task
        fitness = config.fitness
        
        parts = [
            f"Aufgabe: {task.description}",
            f"Typ: {task.task_type.value} (Konfidenz: {task.confidence:.0%})",
            f"Schwierigkeit: {task.difficulty:.0%}",
            f"Umgebung: {config.scene}",
            f"Körper: {config.body_template}",
            f"",
            f"Fitness-Komponenten:",
        ]
        
        for comp, weight in sorted(fitness.components.items(), 
                                    key=lambda x: abs(x[1]), reverse=True):
            direction = "↑" if weight > 0 else "↓"
            parts.append(f"  {direction} {comp}: {weight:+.1f}")
        
        if fitness.curriculum:
            parts.append(f"")
            parts.append(f"Curriculum ({len(fitness.curriculum)} Stufen):")
            for stage in fitness.curriculum:
                gen_range = stage['generation_range']
                end = gen_range[1] or '∞'
                parts.append(f"  Gen {gen_range[0]}-{end}: {stage['description']}")
        
        parts.extend([
            f"",
            f"Training: {config.population} Pop × {config.generations} Gen",
            f"SNN: {config.neurons} Neurons (LIF-LTC + R-STDP)",
            f"Success threshold: {fitness.success_threshold:.1f}",
        ])

        # Level 14: Show UNDERSTAND results
        if hasattr(self, '_last_understand') and self._last_understand:
            u = self._last_understand
            if u.behaviors:
                parts.append(f"")
                parts.append(f"Understood Behaviors ({u.source}):")
                for b in u.behaviors[:6]:
                    parts.append(f"  • {b.name} (p={b.priority:.1f}): {b.description}")
            if u.fitness_adjustments:
                parts.append(f"Fitness adjustments: {u.fitness_adjustments}")
            if self._last_generated_scene:
                n_obj = len(self._last_generated_scene.objects)
                parts.append(f"Generated scene: {n_obj} objects")

        return "\n".join(parts)

    # Mapping from task keywords to CPG skill/fitness function names
    SKILL_MAP = {
        # Locomotion subtypes
        'walk': 'walk', 'lauf': 'walk', 'geh': 'walk', 'move': 'walk',
        'trab': 'walk', 'trot': 'walk',
        'run': 'sprint', 'renn': 'sprint', 'sprint': 'sprint',
        'gallop': 'sprint', 'galoppier': 'sprint',
        'jump': 'jump', 'spring': 'jump', 'hop': 'jump',
        'leap': 'jump', 'huepf': 'jump',
        'climb': 'climb', 'kletter': 'climb', 'steig': 'climb',
        'swim': 'swim', 'schwimm': 'swim',
        'crawl': 'walk', 'kriech': 'walk',
        # Survival subtypes
        'stand': 'stand', 'steh': 'stand', 'balance': 'stand',
        'sit': 'stand', 'sitz': 'stand',
    }

    def _detect_skill(self, text: str) -> str:
        """Detect CPG skill name from natural language input."""
        words = text.lower().split()
        for word in words:
            for keyword, skill in self.SKILL_MAP.items():
                if keyword in word:
                    return skill
        return 'walk'  # default

    def run(
        self,
        config: TrainingConfig,
        creature_name: str = 'mogli',
        checkpoint_dir: Optional[str] = None,
        on_generation: Optional[Callable] = None,
        narrate: bool = False,
        dashboard_ws=None,
        recorder=None,
        **kwargs,
    ) -> 'TrainingResult':
        """
        Full training pipeline: EVOLVE CPG -> TRAIN SNN.

        Phase 1 -- CPG Evolution (no SNN, lightweight):
          Finds optimal gait for the detected skill.
          Saves to cpg_{skill}.json (per-skill configs).

        Phase 2 -- SNN Training (single creature, step-by-step):
          Uses evolved CPG. SNN learns balance corrections.
        """
        import sys
        import json as _json
        import numpy as np
        import mujoco
        import torch
        from copy import deepcopy
        from dataclasses import asdict
        from src.body.synpaw_profile import SynpawProfile
        from src.body.genome import GenomeFactory
        from src.body.mujoco_creature import MuJoCoCreatureBuilder
        from src.brain.cpg import CentralPatternGenerator, CPGConfig

        # Lazy import CPG evolution components
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from scripts.evolve_cpg_params import (
            CPGGenome, evaluate_genome, mutate_genome, crossover_genomes,
            create_initial_population, genome_to_controls, PARAM_BOUNDS,
            FITNESS_FUNCTIONS,
        )

        # Setup
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join('checkpoints', creature_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        profile_dir = 'profiles'
        os.makedirs(profile_dir, exist_ok=True)
        brain_path = os.path.join(checkpoint_dir, 'brain.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Detect skill — use config override if provided
        skill = config.skill_name or self._detect_skill(config.task.description)
        cpg_config_path = os.path.join(checkpoint_dir, f'cpg_{skill}.json')

        # Fitness function for this skill
        fitness_fn = skill if skill in FITNESS_FUNCTIONS else 'walk'

        # Fast mode check
        try:
            from src.utils import config as _cfg
            fast_mode = getattr(_cfg, 'FAST_MODE', False)
        except Exception:
            fast_mode = False

        # Training steps (fast mode: tiny network for workflow testing)
        if fast_mode:
            total_steps = 1000
            log_every = 100
            save_every = 500
            config.neurons = min(config.neurons, 200)
            print(f'  *** FAST MODE: 200 neurons, 1000 steps ***')
        else:
            total_steps = config.generations * 250
            log_every = 500
            save_every = 5000

        # MJCF model
        xml_path = os.path.join('assets', 'meshes', 'mogli', 'mogli_mesh.xml')
        if not os.path.exists(xml_path):
            xml_path = None

        scene_xml = None
        if config.scene and xml_path:
            try:
                from src.body.scene_builder import SceneBuilder, SCENES
                if config.scene in SCENES:
                    scene = SCENES[config.scene]
                    scene_xml = SceneBuilder.build(xml_path, scene)
            except ImportError:
                pass

        # ============================================================
        # PHASE 1: CPG EVOLUTION (per-skill)
        # ============================================================
        if fast_mode:
            evo_pop = 10
            evo_gens = 5
            evo_steps = 500
        else:
            evo_pop = 30
            evo_gens = 20
            evo_steps = config.eval_steps  # default 5000, configurable

        if os.path.exists(cpg_config_path):
            print(f'\n  Found evolved CPG for skill "{skill}": {cpg_config_path}')
            print(f'  Skipping Phase 1 (delete file to re-evolve)')
            sys.stdout.flush()
        else:
            _line = "-" * 50
            print(f'\n  PHASE 1: CPG EVOLUTION')
            print(f'  {_line}')
            print(f'  Skill:       {skill} (fitness: {fitness_fn})')
            print(f'  Population:  {evo_pop}')
            print(f'  Generations: {evo_gens}')
            print(f'  Steps/Eval:  {evo_steps}')
            print(f'  Save to:     {cpg_config_path}')
            print(f'  {_line}')
            sys.stdout.flush()

            # Load MuJoCo model directly (no SNN)
            if scene_xml:
                model = mujoco.MjModel.from_xml_string(scene_xml)
            elif xml_path:
                model = mujoco.MjModel.from_xml_path(xml_path)
            else:
                logger.error("No MJCF model found")
                return TrainingResult(config=config, best_fitness=0.0)
            data = mujoco.MjData(model)

            pop = create_initial_population(evo_pop)
            n_elite = max(2, int(evo_pop * 0.2))
            best_ever_fit = 0.0
            best_genome_ever = None

            for gen in range(evo_gens):
                t_gen = time.time()
                for genome in pop:
                    result = evaluate_genome(genome, model, data,
                                            n_steps=evo_steps, fitness_fn=fitness_fn)
                    genome.fitness = result['fitness']
                    genome.generation = gen

                pop.sort(key=lambda g: g.fitness, reverse=True)
                best = pop[0]
                avg_fit = np.mean([g.fitness for g in pop])

                if best.fitness > best_ever_fit:
                    best_ever_fit = best.fitness
                    best_genome_ever = deepcopy(best)

                result = evaluate_genome(best, model, data,
                                        n_steps=evo_steps, fitness_fn=fitness_fn)
                dt_gen = time.time() - t_gen
                fell = f"@{result['fallen_step']}" if result['fallen_step'] else "STOOD"

                print(f'    Gen {gen+1:>2}/{evo_gens}'
                      f'  Best: {best.fitness:.3f}'
                      f'  Dist: {result["max_dist"]:.2f}m'
                      f'  Up: {result["avg_upright"]:.2f}'
                      f'  {fell}'
                      f'  Avg: {avg_fit:.3f}'
                      f'  {dt_gen:.1f}s')
                sys.stdout.flush()

                if recorder:
                    try:
                        recorder.log_frame({
                            'phase': 'cpg_evolution', 'skill': skill,
                            'generation': gen,
                            'best_fitness': best.fitness,
                            'avg_fitness': avg_fit,
                            'best_distance': result['max_dist'],
                        })
                    except Exception:
                        pass

                # Dashboard push (evolution progress)
                if dashboard_ws:
                    try:
                        dashboard_ws.send(json.dumps({
                            'type': 'training_update',
                            'generation': gen,
                            'stats': {
                                'best_distance': result['max_dist'],
                                'skill': skill,
                                'curriculum_stage': 'CPG EVOLUTION',
                            },
                        }))
                    except Exception:
                        pass

                # Selection + Reproduction
                elite = [deepcopy(g) for g in pop[:n_elite]]
                new_pop = list(elite)
                while len(new_pop) < evo_pop:
                    if np.random.random() < 0.7:
                        parent = elite[np.random.randint(0, n_elite)]
                        child = mutate_genome(parent, 0.3, 0.15)
                    else:
                        p1 = elite[np.random.randint(0, n_elite)]
                        p2 = elite[np.random.randint(0, n_elite)]
                        child = crossover_genomes(p1, p2)
                        child = mutate_genome(child, 0.15, 0.075)
                    child.genome_id = len(new_pop)
                    new_pop.append(child)
                pop = new_pop

            # Save evolved CPG config (skill-specific)
            if best_genome_ever:
                cpg_evolved = {
                    'skill': skill,
                    'fitness_fn': fitness_fn,
                    'base_frequency': best_genome_ever.frequency,
                    'frequency': best_genome_ever.frequency,
                    'base_amplitude': best_genome_ever.amplitude,
                    'amplitude': best_genome_ever.amplitude,
                    'shoulder_hip_amp': best_genome_ever.shoulder_hip_amp,
                    'elbow_stifle_amp': best_genome_ever.elbow_stifle_amp,
                    'carpus_hock_amp': best_genome_ever.carpus_hock_amp,
                    'abduction_amp': best_genome_ever.abduction_amp,
                    'stance_power': best_genome_ever.stance_power,
                    'swing_power': best_genome_ever.swing_power,
                    'phase_offsets': [
                        best_genome_ever.phase_fl, best_genome_ever.phase_fr,
                        best_genome_ever.phase_rl, best_genome_ever.phase_rr,
                    ],
                    'knee_phase_shift': best_genome_ever.knee_phase_shift,
                    'hock_phase_shift': best_genome_ever.hock_phase_shift,
                    'knee_swing_mult': best_genome_ever.knee_swing_mult,
                    'knee_stance_mult': best_genome_ever.knee_stance_mult,
                    'fitness': best_genome_ever.fitness,
                }
                with open(cpg_config_path, 'w') as f:
                    _json.dump(cpg_evolved, f, indent=2)
                print(f'\n  CPG Evolution done! Skill: {skill}, Best: {best_ever_fit:.3f}')
                print(f'  Saved: {cpg_config_path}')
            else:
                print(f'\n  CPG Evolution failed - using defaults')
            sys.stdout.flush()

        # ============================================================
        # PHASE 2: SNN TRAINING (single creature, evolved CPG)
        # ============================================================
        _line = "-" * 50
        print(f'\n  PHASE 2: SNN TRAINING')
        print(f'  {_line}')
        print(f'  Task:       {config.task.description}')
        print(f'  Skill:      {skill}')
        print(f'  Creature:   {creature_name}')
        print(f'  Steps:      {total_steps}')
        print(f'  Neurons:    {config.neurons}')
        print(f'  Device:     {device}')
        print(f'  Scene:      {config.scene}')
        print(f'  Mode:       Evolved CPG + SNN')
        print(f'  {_line}')
        sys.stdout.flush()

        # Build creature with SNN
        genome = GenomeFactory.create_mogli_template()
        print(f'  Building {creature_name}...', end=' ', flush=True)
        creature = MuJoCoCreatureBuilder.build(
            genome, n_hidden_neurons=config.neurons, device=device,
            xml_path=xml_path, xml_string=scene_xml)
        print(f'OK')
        print(f'  SNN: {creature.snn.config.n_neurons} neurons, '
              f'{creature.snn._n_synapses} synapses')
        print(f'  Actuators: {creature.world.n_actuators}')
        sys.stdout.flush()

        # CPG with evolved params (skill-aware)
        cpg_config = CPGConfig.auto_load(creature=creature_name.lower(), skill=skill)
        if hasattr(creature.world, '_model') and creature.world._model is not None:
            mujoco_dt = creature.world._model.opt.timestep
            if abs(cpg_config.dt - mujoco_dt) > 1e-6:
                cpg_config.dt = mujoco_dt
        cpg = CentralPatternGenerator(cpg_config, n_actuators=creature.world.n_actuators)
        print(f'  CPG: {cpg.config.base_frequency:.2f} Hz, '
              f'Amp {cpg.config.base_amplitude:.2f} (skill: {skill})')

        # Cerebellar learning setup — Marr-Albus architecture v0.3.0
        from src.brain.cerebellar_learning import (
            CerebellarLearning, CerebellarConfig,
        )
        cb_config = CerebellarConfig(
            snn_ramp_steps=total_steps // 3,
        )
        ac = CerebellarLearning(
            creature.snn, creature.world.n_actuators,
            config=cb_config, device=device,
        )
        # Connect cerebellar module to SNN populations
        snn = creature.snn
        ac.set_populations(
            mf_ids=snn.populations.get('mossy_fibers', snn.populations.get('input')),
            grc_ids=snn.populations['granule_cells'],
            goc_ids=snn.populations['golgi_cells'],
            pkc_ids=snn.populations['purkinje_cells'],
            dcn_ids=snn.populations['dcn'],
        )
        creature.actor_critic = ac
        print(f'  Cerebellar v0.3.0: GrC={cb_config.n_granule}, '
              f'GoC={cb_config.n_golgi}, PkC={cb_config.n_purkinje}, '
              f'DCN={cb_config.n_dcn}, ramp={cb_config.snn_ramp_steps}')
        print(f'    PF→PkC: {cb_config.pf_pkc_prob:.0%} conn, '
              f'LTD={cb_config.ltd_rate}, LTP={cb_config.ltp_rate}, '
              f'target sparseness={cb_config.target_grc_sparseness:.0%}')

        # Resume from checkpoint if requested
        if config.continue_from and os.path.exists(brain_path):
            try:
                from src.brain.brain_persistence import load_brain
                load_brain(creature.brain, creature.snn, brain_path)
                print(f'  ✅ Resumed brain from: {brain_path}')
            except Exception as e:
                logger.warning(f'Resume failed (starting fresh): {e}')

        # Start skill in brain
        if creature.brain:
            try:
                base = config.base_skills or []
                creature.brain.skills.begin_skill(
                    f'{skill}_v1',
                    description=f'{skill} with evolved CPG',
                    base_skills=base,
                )
                print(f'  Skill started: {skill}_v1 (base: {base})')
                if recorder:
                    recorder.record_event('skill_start', f'{skill}_v1')
            except Exception as e:
                logger.warning(f'Skill start failed: {e}')

        # Profile
        profile_path = os.path.join(profile_dir, f'{creature_name}.json')
        if os.path.exists(profile_path):
            profile = SynpawProfile.load(profile_path)
        else:
            profile = SynpawProfile.create(
                creature_name, template=config.body_template,
                description=f'Auto-created for: {config.task.description}')

        # Training Loop
        print(f'\n  {creature_name} begins {skill} training...\n')
        sys.stdout.flush()

        t_start = time.perf_counter()
        max_distance = 0.0
        total_reward = 0.0
        fall_count = 0
        fallen_counter = 0
        reset_count = 0
        prev_distance = 0.0
        prev_x = float(creature.world._data.qpos[0])
        step_times = []
        narration_log: List[str] = []
        episode_distances = []
        best_episode_dist = 0.0
        avg_episode_dist = 0.0

        for step in range(total_steps):
            t_step = time.perf_counter()
            distance = creature.get_distance_traveled()
            is_fallen = creature.is_fallen()

            # CPG Motor-Output
            if cpg is not None:
                # No modulation: let evolved CPG run unmodified.
                # The evolution found optimal params without freq/amp modulation.
                cpg.set_modulation(freq_mod=0.0, amp_mod=0.0)
                cpg_controls = cpg.step()
                creature._cpg_base = cpg_controls

            # Sensor data for cerebellar learning
            sensor_data = {}
            reward = 0.0
            if step > 0:
                try:
                    sensor_data = creature.world.get_sensor_data(creature.body_name)
                except Exception:
                    pass
                dx = float(creature.world._data.qpos[0])
                forward_vel = (dx - prev_x) if step > 1 else 0.0
                upright = sensor_data.get('upright', 1.0)
                reward = forward_vel * 5.0 + max(0, upright) * 0.1

            # Cerebellar learning update (sensor error → output weights)
            cb_loss = 0.0
            if ac is not None and step > 0 and sensor_data:
                cb_result = ac.update(creature, sensor_data)
                cb_loss = cb_result['loss']

            # Auto-Reset on fall
            if is_fallen:
                fallen_counter += 1
                if fallen_counter > 150:
                    import mujoco as mj_reset
                    ep_dist = creature.get_distance_traveled()
                    episode_distances.append(ep_dist)
                    if ep_dist > best_episode_dist:
                        best_episode_dist = ep_dist
                    recent = episode_distances[-10:]
                    avg_episode_dist = sum(recent) / len(recent)
                    creature.world.reset()
                    mj_reset.mj_forward(creature.world._model, creature.world._data)
                    creature._start_position = None
                    prev_distance = 0.0
                    prev_x = float(creature.world._data.qpos[0])
                    if cpg is not None:
                        cpg.reset()
                    if ac is not None:
                        ac.reset_episode()
                    fallen_counter = 0
                    reset_count += 1
            else:
                fallen_counter = 0

            # Step
            step_result = creature.step(reward_signal=reward)
            prev_distance = distance
            prev_x = float(creature.world._data.qpos[0])
            total_reward += reward
            if distance > max_distance:
                max_distance = distance
            if is_fallen:
                fall_count += 1
            step_dt = time.perf_counter() - t_step
            step_times.append(step_dt)

            # FLOG — Physics (every 10 steps for video replay)
            if recorder and step % 10 == 0:
                try:
                    recorder.record_creature(
                        joint_positions=creature.world._data.qpos.copy(),
                        joint_velocities=creature.world._data.qvel.copy(),
                        center_of_mass=creature.world._data.qpos[:3].copy(),
                        heading=float(creature.world._data.qpos[3]),
                        speed=float(distance - prev_distance) / (creature.world._model.opt.timestep * 10) if step > 0 else 0.0,
                        step=step, skill=skill,
                    )
                except Exception:
                    pass

            # FLOG — Stats (every log_every steps)
            if recorder and step % log_every == 0:
                try:
                    brain_st = creature.brain.get_state() if creature.brain else {}
                    # Cerebellar stats (v0.3.0)
                    cb_stats = {}
                    if hasattr(creature, 'actor_critic') and creature.actor_critic:
                        cb_stats = creature.actor_critic.get_stats()
                    recorder.log_frame({
                        'phase': 'snn_training', 'skill': skill,
                        'step': step, 'distance': distance,
                        'max_distance': max_distance,
                        'best_episode': best_episode_dist,
                        'avg_episode': avg_episode_dist,
                        'falls': fall_count, 'resets': reset_count,
                        'reward': reward,
                        'consciousness_level': brain_st.get('metacognition', {}).get('consciousness_level', 0),
                        'emotion': brain_st.get('emotion', {}).get('dominant_emotion', ''),
                        'valence': brain_st.get('emotion', {}).get('valence', 0.0),
                        # Cerebellar architecture data (v0.3.0)
                        'grc_sparseness': cb_stats.get('grc_sparseness', 0.0),
                        'cf_magnitude': cb_stats.get('cf_magnitude', 0.0),
                        'pf_pkc_weight': cb_stats.get('pf_pkc_mean_weight', 0.0),
                        'ltd_applied': cb_stats.get('ltd_applied', 0.0),
                        'ltp_applied': cb_stats.get('ltp_applied', 0.0),
                        'dcn_activity': cb_stats.get('dcn_activity', 0.0),
                        'correction_mag': cb_stats.get('correction_magnitude', 0.0),
                    })
                except Exception:
                    pass

            # Dashboard push (live training updates)
            if dashboard_ws and step % log_every == 0:
                try:
                    cb_ws = creature.actor_critic.get_stats() if hasattr(creature, 'actor_critic') and creature.actor_critic else {}
                    dashboard_ws.send(json.dumps({
                        'type': 'training_update',
                        'generation': step,
                        'stats': {
                            'best_distance': best_episode_dist,
                            'best_episode': best_episode_dist,
                            'falls': fall_count,
                            'skill': skill,
                            'curriculum_stage': 'PURE CPG' if step < total_steps * 0.2
                                else 'SNN ASSIST' if step < total_steps * 0.6
                                else 'SNN REFINE',
                            'valence': float(creature.brain.get_state().get('emotion', {}).get('valence', 0)) if creature.brain else 0,
                            'emotion': str(creature.brain.get_state().get('emotion', {}).get('dominant_emotion', '')) if creature.brain else '',
                            # Cerebellar data (v0.3.0)
                            'grc_sparseness': cb_ws.get('grc_sparseness', 0.0),
                            'cf_magnitude': cb_ws.get('cf_magnitude', 0.0),
                            'pf_pkc_weight': cb_ws.get('pf_pkc_mean_weight', 0.0),
                            'dcn_activity': cb_ws.get('dcn_activity', 0.0),
                            'correction_mag': cb_ws.get('correction_magnitude', 0.0),
                        },
                    }))
                except Exception:
                    pass

            # Console Logging
            if step > 0 and step % log_every == 0:
                avg_ms = np.mean(step_times[-log_every:]) * 1000
                eta_min = (total_steps - step) * (avg_ms / 1000) / 60
                cur_dist = creature.get_distance_traveled()
                if ac is not None:
                    mix_pct = ac.get_snn_mix()
                    if mix_pct < 0.01:
                        phase = 'CPG ONLY'
                    elif mix_pct < ac.config.snn_mix_end:
                        phase = f'AC RAMP {mix_pct:.0%}'
                    else:
                        phase = f'AC FULL {mix_pct:.0%}'
                else:
                    phase = 'PURE CPG'
                brain_state = creature.brain.get_state() if creature.brain else {}
                emo = brain_state.get('emotion', {}).get('dominant_emotion', '?')
                cl = brain_state.get('metacognition', {}).get('consciousness_level', 0)
                ac_info = ''
                if ac is not None:
                    acs = ac.get_metrics_summary(500)
                    if 'avg_loss' in acs:
                        ac_info = (f'  L:{acs["avg_loss"]:>.4f}'
                                  f'  corr:{acs.get("avg_correction", 0):>.3f}'
                                  f'  mix:{acs["snn_mix"]:.0%}')
                    else:
                        ac_info = (f'  δ:{acs.get("avg_td_error", 0):>+.3f}'
                                  f'  V:{acs.get("avg_value", 0):>.2f}'
                                  f'  mix:{acs["snn_mix"]:.0%}')
                print(
                    f'    Step {step:>6d}/{total_steps}'
                    f'  Best:{best_episode_dist:>4.2f}m'
                    f'  Now:{cur_dist:>4.2f}m'
                    f'{ac_info}'
                    f'  | L{cl} {emo:<8s}'
                    f'  | {avg_ms:>5.1f}ms'
                    f'  ETA:{eta_min:>5.1f}m'
                    f'  [{phase}] [{skill}] R:{reset_count}')
                sys.stdout.flush()
                if narrate and step % (log_every * 10) == 0:
                    stats = {'best_distance': best_episode_dist, 'steps': step,
                             'generation': step, 'falls': fall_count, 'skill': skill}
                    text = self.narrator.narrate_training(stats)
                    narration_log.append(f'[Step {step}] {text}')
                    print(f'    \U0001F4AC {text}')

            # Periodic Save
            if step > 0 and step % save_every == 0:
                try:
                    from src.brain.brain_persistence import save_brain
                    save_brain(creature.brain, creature.snn, brain_path,
                              metadata={'name': creature_name, 'step': step,
                                       'skill': skill,
                                       'distance': float(max_distance), 'falls': fall_count})
                except Exception as e:
                    logger.warning(f'Save failed: {e}')

        # End: freeze skill
        if creature.brain:
            try:
                creature.brain.skills.freeze_skill(f'{skill}_v1')
                print(f'  Skill frozen: {skill}_v1')
                if recorder:
                    recorder.record_event('skill_freeze', f'{skill}_v1',
                                        best_episode=best_episode_dist, falls=fall_count)
            except Exception:
                pass

        elapsed = time.perf_counter() - t_start
        creature.world.close()

        try:
            from src.brain.brain_persistence import save_brain
            save_brain(creature.brain, creature.snn, brain_path,
                      metadata={'name': creature_name, 'step': total_steps,
                               'skill': skill,
                               'distance': float(max_distance),
                               'best_episode': float(best_episode_dist),
                               'falls': fall_count, 'resets': reset_count})
        except Exception as e:
            logger.warning(f'Final save failed: {e}')

        # Save Actor-Critic state
        if ac is not None:
            ac_path = os.path.join(checkpoint_dir, 'actor_critic.pt')
            try:
                ac.save(ac_path)
                print(f'  Actor-Critic saved: {ac_path}')
            except Exception as e:
                logger.warning(f'AC save failed: {e}')

        profile.log_training(
            generations=total_steps, best_fitness=best_episode_dist,
            scenario=config.task.task_type.value,
            notes=f'Skill: {skill}, Time: {elapsed/60:.1f}min, '
                  f'Best ep: {best_episode_dist:.2f}m, Falls: {fall_count}')
        profile.save(profile_path)

        final_narration = ''
        if narrate:
            final_stats = {'best_distance': best_episode_dist, 'steps': total_steps,
                          'generation': total_steps, 'falls': fall_count, 'skill': skill}
            final_narration = self.narrator.narrate_training(final_stats)
            narration_log.append(f'[FINAL] {final_narration}')

        print(f'\n  \U0001F3C6 Training complete! ({skill})')
        print(f'  Best episode: {best_episode_dist:.2f}m')
        print(f'  Max distance: {max_distance:.2f}m')
        print(f'  Falls: {fall_count}, Resets: {reset_count}')
        print(f'  Time: {elapsed/60:.1f} min ({elapsed/total_steps*1000:.1f}ms/step)')
        print(f'  Brain saved: {brain_path}')
        print(f'  CPG config: {cpg_config_path}')
        sys.stdout.flush()

        return TrainingResult(
            config=config, best_fitness=best_episode_dist,
            best_genome=None, elapsed_seconds=elapsed,
            total_steps=total_steps, skill=skill,
            narration_log=narration_log, final_narration=final_narration,
            checkpoint_dir=checkpoint_dir, profile_path=profile_path,
            curriculum=None)



# ═══════════════════════════════════════════════════════════
# TRAINING RESULT
# ═══════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    """Result of a training run."""
    config: TrainingConfig
    best_fitness: float = 0.0
    best_genome: Optional[Any] = None
    elapsed_seconds: float = 0.0
    total_steps: int = 0
    skill: str = ''
    narration_log: List[str] = field(default_factory=list)
    final_narration: str = ''
    checkpoint_dir: str = ''
    profile_path: str = ''
    curriculum: Optional[Any] = None

    def summary(self) -> str:
        """Human-readable summary of training results."""
        mins = self.elapsed_seconds / 60
        parts = [
            f"Training: {self.config.task.description}",
            f"Best fitness: {self.best_fitness:.3f}",
            f"Duration: {mins:.1f} min",
        ]
        if self.final_narration:
            parts.append(f"Narrator: {self.final_narration}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
# BEHAVIOR KNOWLEDGE GENERATOR
# ═══════════════════════════════════════════════════════════

class BehaviorKnowledgeGenerator:
    """
    Generates behavior knowledge from task descriptions.
    Maps TaskType to expected behaviors with priorities.
    """

    TASK_BEHAVIORS = {
        TaskType.LOCOMOTION: [
            ('walk', 'Forward locomotion', 0.8),
            ('trot', 'Faster trot gait', 0.5),
            ('stand', 'Stand and balance', 0.3),
        ],
        TaskType.NAVIGATION: [
            ('walk', 'Walk to target', 0.8),
            ('turn', 'Change direction', 0.6),
            ('stop', 'Stop and orient', 0.3),
        ],
        TaskType.SURVIVAL: [
            ('stand', 'Stable standing', 0.9),
            ('balance', 'Maintain balance', 0.8),
            ('recover', 'Get up after fall', 0.5),
        ],
        TaskType.EXPLORATION: [
            ('walk', 'Explore', 0.6),
            ('sniff', 'Investigate', 0.5),
            ('turn', 'New direction', 0.4),
        ],
    }

    def generate(self, task: ParsedTask) -> List[Dict]:
        """Generate behavior knowledge for a task."""
        behaviors = []
        task_behaviors = self.TASK_BEHAVIORS.get(task.task_type, [])
        for name, desc, priority in task_behaviors:
            behaviors.append({
                'name': name,
                'description': desc,
                'priority': priority * (0.5 + task.difficulty),
                'task_context': task.task_type.value,
            })
        return behaviors


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def plan_training(text: str, **kwargs) -> TrainingConfig:
    """Shortcut: Text -> TrainingConfig."""
    orch = TrainingOrchestrator()
    return orch.plan(text, **kwargs)


def explain_training(text: str, **kwargs) -> str:
    """Shortcut: Text -> Plan explanation."""
    orch = TrainingOrchestrator()
    config = orch.plan(text, **kwargs)
    return orch.explain_plan(config)


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  LLM-Bridge — TaskParser Test")
    print("=" * 60)

    orch = TrainingOrchestrator()

    test_tasks = [
        "mogli walks on grass",
        "mogli jumps over obstacles",
        "mogli stands still on ice",
        "mogli runs fast on sand",
    ]

    for text in test_tasks:
        print(f"\n{'-'*60}")
        config = orch.plan(text)
        print(orch.explain_plan(config))
