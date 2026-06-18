#!/usr/bin/env python3
"""
Level 13: Unified Curriculum System
======================================
Single source of truth for all training curricula.
Replaces: WalkScenario.CURRICULUM_STAGES, FitnessGenerator._generate_curriculum().

A Curriculum is an ordered list of Stages. Each stage defines:
  - fitness weight overrides
  - generation range (when it's active)
  - success criteria (when to advance)
  - optional scene/environment changes

Usage:
    curriculum = CurriculumBuilder.from_task(parsed_task, fitness_spec)
    stage = curriculum.active_stage(generation=75)
    weights = stage.effective_weights(base_weights)
"""

__version__ = "0.1.0"
__logbook__ = 141

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ================================================================
# STAGE
# ================================================================

@dataclass
class CurriculumStage:
    """A single training stage within a curriculum."""
    name: str
    description: str
    generation_start: int = 0
    generation_end: Optional[int] = None  # None = until curriculum ends
    weight_overrides: Dict[str, float] = field(default_factory=dict)
    weight_multipliers: Dict[str, float] = field(default_factory=dict)
    scene: Optional[str] = None  # override scene for this stage
    gravity_scale: float = 1.0   # 1.0 = normal, <1 = reduced gravity
    success_metric: Optional[str] = None  # e.g. 'distance > 2.0'
    success_threshold: float = 0.0

    def is_active(self, generation: int) -> bool:
        """Check if this stage is active at the given generation."""
        if generation < self.generation_start:
            return False
        if self.generation_end is not None and generation >= self.generation_end:
            return False
        return True

    def effective_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply this stage's overrides/multipliers to base weights."""
        result = dict(base_weights)
        # Apply overrides (absolute values)
        result.update(self.weight_overrides)
        # Apply multipliers
        for key, mult in self.weight_multipliers.items():
            if key in result:
                result[key] *= mult
        return result


# ================================================================
# CURRICULUM
# ================================================================

@dataclass
class Curriculum:
    """Ordered sequence of training stages."""
    name: str
    stages: List[CurriculumStage] = field(default_factory=list)
    total_generations: int = 200

    def active_stage(self, generation: int) -> Optional[CurriculumStage]:
        """Return the currently active stage for this generation."""
        for stage in reversed(self.stages):
            if stage.is_active(generation):
                return stage
        return self.stages[0] if self.stages else None

    def active_stage_index(self, generation: int) -> int:
        """Return the index of the active stage."""
        for i, stage in enumerate(self.stages):
            if stage.is_active(generation):
                last_active = i
        return last_active if 'last_active' in dir() else 0

    def add_stage(self, stage: CurriculumStage):
        """Add a stage to the curriculum."""
        self.stages.append(stage)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curriculum to dict (for checkpoints)."""
        return {
            'name': self.name,
            'total_generations': self.total_generations,
            'stages': [
                {
                    'name': s.name,
                    'description': s.description,
                    'generation_start': s.generation_start,
                    'generation_end': s.generation_end,
                    'weight_overrides': s.weight_overrides,
                    'weight_multipliers': s.weight_multipliers,
                    'scene': s.scene,
                    'gravity_scale': s.gravity_scale,
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Curriculum':
        """Deserialize curriculum from dict."""
        c = cls(name=d['name'], total_generations=d.get('total_generations', 200))
        for sd in d.get('stages', []):
            c.add_stage(CurriculumStage(**sd))
        return c


# ================================================================
# BUILDER
# ================================================================

class CurriculumBuilder:
    """Builds curricula from various sources."""

    @staticmethod
    def from_task(task, fitness_spec, total_generations: int = 200) -> Curriculum:
        """
        Build curriculum from a ParsedTask + FitnessSpec.
        Replaces FitnessGenerator._generate_curriculum().
        
        Strategy:
          Stage 1 (0-25%):  Balance focus, reduced primary objective
          Stage 2 (25-75%): Full objective weights
          Stage 3 (75%+):   Refinement with amplified weights
        """
        gen_25 = total_generations // 4
        gen_75 = (total_generations * 3) // 4

        curriculum = Curriculum(
            name=f"curriculum_{task.task_type.value}",
            total_generations=total_generations,
        )

        # Stage 1: Foundation — balance first
        curriculum.add_stage(CurriculumStage(
            name='foundation',
            description='Balance and basic stability',
            generation_start=0,
            generation_end=gen_25,
            weight_multipliers={'distance': 0.3, 'area': 0.3, 'height': 0.3},
            weight_overrides={'upright': max(fitness_spec.components.get('upright', 0), 2.0)},
        ))

        # Stage 2: Main objective
        curriculum.add_stage(CurriculumStage(
            name='main_objective',
            description=f'Primary: {task.task_type.value}',
            generation_start=gen_25,
            generation_end=gen_75,
        ))

        # Stage 3: Refinement
        amplify = {k: 1.5 for k, v in fitness_spec.components.items() if v > 0}
        curriculum.add_stage(CurriculumStage(
            name='refinement',
            description='Fine-tuning and optimization',
            generation_start=gen_75,
            generation_end=None,
            weight_multipliers=amplify,
        ))

        return curriculum

    @staticmethod
    def from_walk_scenario(config) -> Curriculum:
        """
        Build curriculum from WalkScenario config.
        Replaces WalkScenario.CURRICULUM_STAGES.
        """
        curriculum = Curriculum(
            name='walk_curriculum',
            total_generations=config.n_generations if hasattr(config, 'n_generations') else 200,
        )

        # Reduced gravity start
        curriculum.add_stage(CurriculumStage(
            name='low_gravity',
            description='Reduced gravity for initial balance',
            generation_start=0,
            generation_end=30,
            gravity_scale=0.5,
            weight_overrides={'upright': 3.0, 'alive': 1.0},
            weight_multipliers={'distance': 0.2},
        ))

        # Normal gravity
        curriculum.add_stage(CurriculumStage(
            name='normal_gravity',
            description='Full gravity, learning to walk',
            generation_start=30,
            generation_end=100,
            gravity_scale=1.0,
        ))

        # Slippery surface
        curriculum.add_stage(CurriculumStage(
            name='slippery',
            description='Reduced friction for robustness',
            generation_start=100,
            generation_end=None,
            gravity_scale=1.0,
            scene='ice',
            weight_multipliers={'upright': 1.5, 'distance': 1.3},
        ))

        return curriculum

    @staticmethod
    def simple(total_generations: int = 200) -> Curriculum:
        """A minimal single-stage curriculum (no progression)."""
        curriculum = Curriculum(name='simple', total_generations=total_generations)
        curriculum.add_stage(CurriculumStage(
            name='default',
            description='Single stage, constant weights',
            generation_start=0,
            generation_end=None,
        ))
        return curriculum
