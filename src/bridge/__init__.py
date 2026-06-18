"""
Level 13: Bridge — Language ↔ Embodied Training
==================================================
Connects natural language to creature training.

Modules:
  llm_bridge.py        — TaskParser, FitnessGenerator, Narrator, Orchestrator
  instruction_parser.py — Regex-based fitness parsing
  curriculum.py         — Unified Curriculum system
  auto_mjcf.py          — Task → Body/Scene templates
  task_rewards.py       — Declarative reward components
"""

__version__ = "0.8.0-dev"
__logbook__ = 159

from src.bridge.llm_bridge import (
    TaskParser,
    FitnessGenerator,
    ExperienceNarrator,
    TrainingOrchestrator,
    BehaviorKnowledgeGenerator,
    plan_training,
    explain_training,
    ParsedTask,
    FitnessSpec,
    TrainingConfig,
    TaskType,
)

from src.bridge.instruction_parser import (
    InstructionParser,
    FitnessBuilder,
    FitnessComponents,
    KeywordParser,
    InstructionSpec,
    FitnessComponent,
    COMPONENT_REGISTRY,
)

# Level 13 modules (lazy imports to avoid heavy deps at startup)
from src.bridge.curriculum import Curriculum, CurriculumStage, CurriculumBuilder
from src.bridge.auto_mjcf import AutoMJCF
from src.bridge.task_rewards import RewardComponent, RewardComposer, TaskSpec
from src.bridge.llm_bridge import TrainingResult

__all__ = [
    # LLM Bridge
    'TaskParser', 'FitnessGenerator', 'ExperienceNarrator',
    'TrainingOrchestrator', 'BehaviorKnowledgeGenerator',
    'plan_training', 'explain_training',
    'ParsedTask', 'FitnessSpec', 'TrainingConfig', 'TaskType',
    'TrainingResult',
    # Instruction Parser
    'InstructionParser', 'FitnessBuilder', 'FitnessComponents',
    'KeywordParser', 'InstructionSpec', 'FitnessComponent',
    'COMPONENT_REGISTRY',
    # Level 13
    'Curriculum', 'CurriculumStage', 'CurriculumBuilder',
    'AutoMJCF',
    'RewardComponent', 'RewardComposer', 'TaskSpec',
]
