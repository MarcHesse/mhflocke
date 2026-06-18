#!/usr/bin/env python3
"""
Level 13: Declarative Task & Reward System
=============================================
Combinable, composable reward modules that can be declared and mixed.

Replaces ad-hoc fitness functions with a structured system:
  - RewardComponent: single measurable objective
  - RewardComposer: combines components with weights
  - TaskSpec: full task declaration (components + curriculum + success)

Usage:
    from src.bridge.task_rewards import RewardComponent, RewardComposer, TaskSpec

    composer = RewardComposer()
    composer.add(RewardComponent.distance(weight=3.0))
    composer.add(RewardComponent.upright(weight=2.0))
    composer.add(RewardComponent.alive(weight=0.5))
    composer.add(RewardComponent.energy(weight=-0.003))

    fitness = composer.compute(trajectory)

    # Or from FitnessSpec:
    composer = RewardComposer.from_fitness_spec(fitness_spec)
"""

__version__ = "0.1.0"
__logbook__ = 143

import numpy as np
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ================================================================
# REWARD COMPONENTS
# ================================================================

@dataclass
class RewardComponent:
    """A single reward/penalty component."""
    name: str
    weight: float
    compute_fn: Callable[[Dict], float]  # trajectory_data -> value
    description: str = ''

    def compute(self, trajectory: Dict) -> float:
        """Compute this component's contribution."""
        return self.weight * self.compute_fn(trajectory)

    # ── Factory methods for common components ──

    @staticmethod
    def distance(weight: float = 3.0) -> 'RewardComponent':
        """Reward for forward distance traveled."""
        return RewardComponent(
            name='distance',
            weight=weight,
            compute_fn=lambda t: t.get('distance', 0.0),
            description='Forward distance traveled',
        )

    @staticmethod
    def upright(weight: float = 2.0) -> 'RewardComponent':
        """Reward for staying upright (fraction of time not fallen)."""
        return RewardComponent(
            name='upright',
            weight=weight,
            compute_fn=lambda t: t.get('alive_steps', 0) / max(t.get('max_steps', 1), 1),
            description='Fraction of time upright',
        )

    @staticmethod
    def alive(weight: float = 0.5) -> 'RewardComponent':
        """Reward for staying alive (absolute steps)."""
        return RewardComponent(
            name='alive',
            weight=weight,
            compute_fn=lambda t: t.get('alive_steps', 0) / 1000.0,
            description='Survival time (normalized)',
        )

    @staticmethod
    def energy(weight: float = -0.003) -> 'RewardComponent':
        """Penalty for energy expenditure."""
        return RewardComponent(
            name='energy',
            weight=weight,
            compute_fn=lambda t: t.get('total_energy', 0.0),
            description='Energy expenditure penalty',
        )

    @staticmethod
    def height(weight: float = 4.0) -> 'RewardComponent':
        """Reward for jump height."""
        return RewardComponent(
            name='height',
            weight=weight,
            compute_fn=lambda t: t.get('max_height', 0.0),
            description='Maximum jump height',
        )

    @staticmethod
    def target_distance(weight: float = 3.0) -> 'RewardComponent':
        """Reward for getting close to target (inverse distance)."""
        return RewardComponent(
            name='target_distance',
            weight=weight,
            compute_fn=lambda t: max(0, 10.0 - t.get('target_distance', 10.0)),
            description='Proximity to target',
        )

    @staticmethod
    def area_covered(weight: float = 2.0) -> 'RewardComponent':
        """Reward for exploration area covered."""
        return RewardComponent(
            name='area_covered',
            weight=weight,
            compute_fn=lambda t: t.get('area_covered', 0.0),
            description='Exploration area',
        )

    @staticmethod
    def stability(weight: float = 1.0) -> 'RewardComponent':
        """Reward for low body oscillation."""
        return RewardComponent(
            name='stability',
            weight=weight,
            compute_fn=lambda t: max(0, 1.0 - t.get('body_oscillation', 0.5)),
            description='Body stability (low oscillation)',
        )

    @staticmethod
    def custom(name: str, weight: float, fn: Callable[[Dict], float],
               description: str = '') -> 'RewardComponent':
        """Custom reward component."""
        return RewardComponent(name=name, weight=weight, compute_fn=fn,
                               description=description)


# ================================================================
# REWARD COMPOSER
# ================================================================

class RewardComposer:
    """
    Combines multiple RewardComponents into a single fitness function.

    Supports:
      - Weighted sum of components
      - Bonus conditions (if X then +Y)
      - Penalty conditions (if X then -Y)
      - Conversion to/from FitnessSpec
    """

    def __init__(self):
        self.components: List[RewardComponent] = []
        self.bonuses: List[Dict] = []
        self.penalties: List[Dict] = []

    def add(self, component: RewardComponent):
        """Add a reward component."""
        self.components.append(component)
        return self  # chainable

    def add_bonus(self, condition_fn: Callable[[Dict], bool],
                  bonus: float, description: str = ''):
        """Add a conditional bonus."""
        self.bonuses.append({
            'condition': condition_fn,
            'bonus': bonus,
            'description': description,
        })
        return self

    def add_penalty(self, condition_fn: Callable[[Dict], bool],
                    penalty: float, description: str = ''):
        """Add a conditional penalty."""
        self.penalties.append({
            'condition': condition_fn,
            'penalty': penalty,
            'description': description,
        })
        return self

    def compute(self, trajectory: Dict) -> float:
        """
        Compute total fitness from trajectory data.

        Args:
            trajectory: Dict with keys like 'distance', 'alive_steps', etc.

        Returns:
            Total fitness value.
        """
        total = 0.0

        # Sum components
        for comp in self.components:
            total += comp.compute(trajectory)

        # Apply bonuses
        for bonus in self.bonuses:
            if bonus['condition'](trajectory):
                total += bonus['bonus']

        # Apply penalties
        for penalty in self.penalties:
            if penalty['condition'](trajectory):
                total -= penalty['penalty']

        return total

    def describe(self) -> str:
        """Human-readable description of the reward function."""
        lines = ['Reward Components:']
        for c in self.components:
            sign = '+' if c.weight >= 0 else ''
            lines.append(f'  {sign}{c.weight:.2f} x {c.name}: {c.description}')
        if self.bonuses:
            lines.append('Bonuses:')
            for b in self.bonuses:
                lines.append(f'  +{b["bonus"]:.2f}: {b["description"]}')
        if self.penalties:
            lines.append('Penalties:')
            for p in self.penalties:
                lines.append(f'  -{p["penalty"]:.2f}: {p["description"]}')
        return '\n'.join(lines)

    @classmethod
    def from_fitness_spec(cls, spec) -> 'RewardComposer':
        """
        Create RewardComposer from a FitnessSpec (llm_bridge.py).

        Maps FitnessSpec.components to RewardComponents.
        """
        composer = cls()

        # Component name → factory method mapping
        factory_map = {
            'distance': RewardComponent.distance,
            'upright': RewardComponent.upright,
            'alive': RewardComponent.alive,
            'energy_penalty': RewardComponent.energy,
            'height': RewardComponent.height,
            'target_dist': RewardComponent.target_distance,
            'area': RewardComponent.area_covered,
        }

        for name, weight in spec.components.items():
            factory = factory_map.get(name)
            if factory:
                composer.add(factory(weight=weight))
            else:
                # Generic component
                composer.add(RewardComponent.custom(
                    name=name, weight=weight,
                    fn=lambda t, n=name: t.get(n, 0.0),
                    description=f'Auto-mapped: {name}',
                ))

        return composer

    def to_dict(self) -> Dict:
        """Serialize for checkpoints."""
        return {
            'components': [
                {'name': c.name, 'weight': c.weight, 'description': c.description}
                for c in self.components
            ],
        }


# ================================================================
# TASK SPEC (Declarative)
# ================================================================

@dataclass
class TaskSpec:
    """
    Complete declarative task specification.

    Combines reward function, curriculum, and success criteria
    into a single portable object.
    """
    name: str
    description: str
    reward: RewardComposer
    success_threshold: float = 2.0
    max_steps: int = 2000
    curriculum_stages: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def is_success(self, trajectory: Dict) -> bool:
        """Check if task was completed successfully."""
        fitness = self.reward.compute(trajectory)
        return fitness >= self.success_threshold

    @classmethod
    def walk(cls) -> 'TaskSpec':
        """Pre-built: Walk forward as far as possible."""
        reward = RewardComposer()
        reward.add(RewardComponent.distance(3.0))
        reward.add(RewardComponent.upright(2.0))
        reward.add(RewardComponent.alive(0.5))
        reward.add(RewardComponent.energy(-0.003))
        return cls(name='walk', description='Walk forward',
                   reward=reward, success_threshold=2.0)

    @classmethod
    def jump(cls) -> 'TaskSpec':
        """Pre-built: Jump as high as possible."""
        reward = RewardComposer()
        reward.add(RewardComponent.height(4.0))
        reward.add(RewardComponent.upright(2.0))
        reward.add(RewardComponent.stability(1.0))
        return cls(name='jump', description='Jump high',
                   reward=reward, success_threshold=0.5)

    @classmethod
    def explore(cls) -> 'TaskSpec':
        """Pre-built: Explore environment."""
        reward = RewardComposer()
        reward.add(RewardComponent.area_covered(2.0))
        reward.add(RewardComponent.distance(1.0))
        reward.add(RewardComponent.upright(1.5))
        return cls(name='explore', description='Explore area',
                   reward=reward, success_threshold=3.0)

    @classmethod
    def balance(cls) -> 'TaskSpec':
        """Pre-built: Stay upright as long as possible."""
        reward = RewardComposer()
        reward.add(RewardComponent.upright(4.0))
        reward.add(RewardComponent.alive(2.0))
        reward.add(RewardComponent.stability(1.5))
        return cls(name='balance', description='Stay balanced',
                   reward=reward, success_threshold=5.0)
