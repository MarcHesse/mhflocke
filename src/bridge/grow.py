#!/usr/bin/env python3
"""
Level 14: GROW — Autonomous Post-Training Learning
======================================================
After training, the creature can discover new behaviors
and continue learning autonomously.

GROW Pipeline:
  1. EVALUATE: What did the creature learn? What can it do?
  2. SUGGEST: Ask LLM "What else could this creature learn?"
  3. EXTEND: Generate new training tasks from suggestions
  4. TRAIN: Run focused training on new capabilities
  5. CONSOLIDATE: Merge new skills into creature profile

Usage:
    from src.bridge.grow import GrowEngine

    grow = GrowEngine()
    suggestions = grow.suggest_next(training_result, understand_result)
    # → [GrowTask('learn to jump over obstacles'), ...]

    for task in suggestions:
        new_config = grow.plan_growth(task, creature_profile)
        # → Ready to train
"""

__version__ = "0.1.0"
__logbook__ = 142

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class GrowTask:
    """A suggested next learning task."""
    description: str
    rationale: str  # Why this task?
    difficulty: float = 0.5  # 0.0-1.0
    builds_on: List[str] = field(default_factory=list)  # skills it extends
    estimated_generations: int = 100
    priority: float = 0.5
    source: str = 'builtin'  # 'builtin', 'llm', 'self_analysis'


@dataclass
class GrowPlan:
    """A complete plan for autonomous growth."""
    tasks: List[GrowTask]
    current_skills: List[str]
    creature_name: str
    total_estimated_time_minutes: float = 0.0


# ================================================================
# BUILT-IN GROWTH PATHS
# ================================================================

# skill → what to learn next (curriculum-like progression)
GROWTH_PATHS: Dict[str, List[GrowTask]] = {
    'walk': [
        GrowTask('learn to trot faster',
                 rationale='Extend walking to faster gait',
                 difficulty=0.4, builds_on=['walk'],
                 estimated_generations=100, priority=0.8),
        GrowTask('learn to walk on uneven terrain',
                 rationale='Generalize walking to rough surfaces',
                 difficulty=0.6, builds_on=['walk'],
                 estimated_generations=150, priority=0.7),
        GrowTask('learn to turn while walking',
                 rationale='Add directional control',
                 difficulty=0.5, builds_on=['walk'],
                 estimated_generations=120, priority=0.6),
    ],
    'balance': [
        GrowTask('learn to recover from pushes',
                 rationale='Active balance recovery',
                 difficulty=0.5, builds_on=['balance'],
                 estimated_generations=100, priority=0.7),
        GrowTask('learn to balance on one leg',
                 rationale='Advanced balance challenge',
                 difficulty=0.8, builds_on=['balance'],
                 estimated_generations=200, priority=0.4),
    ],
    'walk_on_ice': [
        GrowTask('learn to walk on mixed terrain',
                 rationale='Transition between friction surfaces',
                 difficulty=0.7, builds_on=['walk_on_ice', 'walk'],
                 estimated_generations=150, priority=0.6),
    ],
    'jump': [
        GrowTask('learn to jump over obstacles',
                 rationale='Combine jumping with navigation',
                 difficulty=0.7, builds_on=['jump', 'walk'],
                 estimated_generations=200, priority=0.6),
    ],
    'sniff': [
        GrowTask('learn to track a scent trail',
                 rationale='Navigate using scent gradient',
                 difficulty=0.6, builds_on=['sniff', 'walk'],
                 estimated_generations=150, priority=0.5),
    ],
}

# Generic tasks for creatures with no specific skills yet
BEGINNER_TASKS = [
    GrowTask('learn basic walking',
             rationale='Fundamental locomotion skill',
             difficulty=0.3, builds_on=[],
             estimated_generations=200, priority=1.0),
    GrowTask('learn to stand upright',
             rationale='Balance is prerequisite for all movement',
             difficulty=0.2, builds_on=[],
             estimated_generations=100, priority=0.9),
]

# LLM prompt for growth suggestions
GROWTH_PROMPT = """You are a curriculum designer for an embodied AI creature.

Creature: {creature_type}
Current skills: {current_skills}
Last training: {last_task}
Best fitness achieved: {best_fitness}

Suggest 2-3 NEW skills this creature should learn next.
Each suggestion should build on existing skills.
Order from easiest to hardest.

Respond in JSON:
[
  {{"description": "learn to ...", "rationale": "Because ...", "difficulty": 0.5, "builds_on": ["walk"], "estimated_generations": 150}},
  ...
]

Only JSON, no other text."""


# ================================================================
# GROW ENGINE
# ================================================================

class GrowEngine:
    """
    Suggests and plans autonomous growth for trained creatures.

    After a creature completes initial training, GROW analyzes
    what it learned and suggests next steps for continued development.
    """

    def __init__(self, llm_adapter=None):
        """
        Args:
            llm_adapter: Optional MultiLLMAdapter for richer suggestions.
        """
        self.llm = llm_adapter

    def suggest_next(
        self,
        training_result=None,
        understand_result=None,
        creature_profile=None,
        creature_type: str = 'dog',
    ) -> List[GrowTask]:
        """
        Suggest next learning tasks after training.

        Args:
            training_result: TrainingResult from last run.
            understand_result: UnderstandResult from task understanding.
            creature_profile: SynpawProfile with skill history.
            creature_type: Type of creature.

        Returns:
            List of GrowTasks ordered by priority.
        """
        current_skills = self._extract_skills(training_result, creature_profile)

        # Built-in suggestions based on current skills
        suggestions = self._builtin_suggestions(current_skills)

        # LLM enrichment
        if self.llm and self.llm.enabled:
            llm_suggestions = self._llm_suggestions(
                creature_type, current_skills, training_result)
            suggestions.extend(llm_suggestions)

        # Filter out already-learned skills
        suggestions = [s for s in suggestions
                       if s.description not in current_skills]

        # Sort by priority × (1 - difficulty) → favor easy high-priority tasks
        suggestions.sort(
            key=lambda t: t.priority * (1.0 - t.difficulty * 0.3),
            reverse=True,
        )

        # Limit to top 5
        suggestions = suggestions[:5]

        logger.info(f'GROW: {len(suggestions)} tasks suggested '
                     f'(skills: {current_skills})')
        return suggestions

    def plan_growth(self, task: GrowTask, base_config=None) -> Dict:
        """
        Convert a GrowTask into a TrainingConfig-compatible dict.

        Args:
            task: The growth task to plan.
            base_config: Optional base TrainingConfig to modify.

        Returns:
            Dict with training parameters.
        """
        plan = {
            'task_text': task.description,
            'generations': task.estimated_generations,
            'population': 50 if task.difficulty < 0.5 else 80,
            'neurons': 5000 if task.difficulty < 0.7 else 8000,
            'builds_on': task.builds_on,
            'difficulty': task.difficulty,
            'continue_from_checkpoint': len(task.builds_on) > 0,
        }

        # If building on existing skills, reduce generations
        # (not starting from scratch)
        if task.builds_on:
            plan['generations'] = int(plan['generations'] * 0.7)

        return plan

    def create_growth_plan(
        self,
        creature_profile=None,
        max_tasks: int = 3,
        creature_type: str = 'dog',
    ) -> GrowPlan:
        """
        Create a complete multi-step growth plan.

        Args:
            creature_profile: SynpawProfile.
            max_tasks: Maximum number of tasks.
            creature_type: Creature type.

        Returns:
            GrowPlan with ordered tasks and time estimates.
        """
        suggestions = self.suggest_next(
            creature_profile=creature_profile,
            creature_type=creature_type,
        )[:max_tasks]

        # Estimate time (rough: 1 generation ≈ 10s on GPU)
        total_gens = sum(t.estimated_generations for t in suggestions)
        total_minutes = total_gens * 10 / 60  # seconds → minutes

        current_skills = self._extract_skills(creature_profile=creature_profile)

        return GrowPlan(
            tasks=suggestions,
            current_skills=current_skills,
            creature_name=getattr(creature_profile, 'name', 'Synpaw'),
            total_estimated_time_minutes=total_minutes,
        )

    # ── Private ──

    def _extract_skills(self, training_result=None,
                         creature_profile=None) -> List[str]:
        """Extract list of current skill names."""
        skills = []

        if creature_profile:
            frozen = getattr(creature_profile, 'skills', {})
            if isinstance(frozen, dict):
                skills.extend(frozen.keys())
            elif hasattr(frozen, 'get_frozen_skills'):
                skills.extend(frozen.get_frozen_skills())

        if training_result:
            config = getattr(training_result, 'config', None)
            if config:
                task = getattr(config, 'task', None)
                if task:
                    # Derive skill name from task
                    task_type = getattr(task, 'task_type', None)
                    if task_type:
                        skills.append(task_type.value)

        return list(set(skills))

    def _builtin_suggestions(self, current_skills: List[str]) -> List[GrowTask]:
        """Get built-in growth suggestions."""
        suggestions = []

        if not current_skills:
            return list(BEGINNER_TASKS)

        for skill in current_skills:
            path_tasks = GROWTH_PATHS.get(skill, [])
            suggestions.extend(path_tasks)

        # If no specific paths found, add generic progression
        if not suggestions:
            suggestions.extend(BEGINNER_TASKS)

        return suggestions

    def _llm_suggestions(self, creature_type: str,
                          current_skills: List[str],
                          training_result=None) -> List[GrowTask]:
        """Query LLM for growth suggestions."""
        last_task = 'initial training'
        best_fitness = 0.0
        if training_result:
            best_fitness = getattr(training_result, 'best_fitness', 0.0)
            config = getattr(training_result, 'config', None)
            if config and hasattr(config, 'task'):
                last_task = config.task.description

        prompt = GROWTH_PROMPT.format(
            creature_type=creature_type,
            current_skills=', '.join(current_skills) if current_skills else 'none',
            last_task=last_task,
            best_fitness=f'{best_fitness:.2f}',
        )

        try:
            response = self.llm.generate(prompt, max_tokens=600)
            if not response.success:
                return []

            return self._parse_llm_suggestions(response.text)
        except Exception as e:
            logger.error(f'GROW LLM query failed: {e}')
            return []

    def _parse_llm_suggestions(self, text: str) -> List[GrowTask]:
        """Parse LLM JSON response into GrowTasks."""
        import json
        import re

        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        tasks = []
        for item in data:
            if not isinstance(item, dict) or 'description' not in item:
                continue
            tasks.append(GrowTask(
                description=item['description'],
                rationale=item.get('rationale', ''),
                difficulty=float(item.get('difficulty', 0.5)),
                builds_on=item.get('builds_on', []),
                estimated_generations=int(item.get('estimated_generations', 150)),
                priority=0.6,  # LLM suggestions have moderate priority
                source='llm',
            ))

        return tasks
