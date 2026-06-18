#!/usr/bin/env python3
"""
Level 13: Auto-MJCF — Task → Body + Scene
=============================================
Automatically generates creature body and scene from a parsed task.

Connects TaskParser output (body_requirements, environment_hints)
to GenomeFactory templates and SceneBuilder scenes.

Architecture:
    ParsedTask.body_requirements → TemplateRegistry → Genome
    ParsedTask.environment_hints → SceneRegistry → Scene MJCF

The system is template-based and extensible:
    - Body templates map to GenomeFactory methods
    - Scene templates map to SceneBuilder SCENES dict
    - Modifiers apply scale, appendage, and physics changes

Usage:
    from src.bridge.auto_mjcf import AutoMJCF
    mjcf = AutoMJCF()

    # From a ParsedTask
    genome, scene_xml = mjcf.from_task(parsed_task)

    # Direct
    genome = mjcf.build_body('synpaw', scale=1.5)
    scene_xml = mjcf.build_scene('forest', gravity_scale=0.8)
"""

__version__ = "0.1.0"
__logbook__ = 140

import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ================================================================
# BODY TEMPLATE REGISTRY
# ================================================================

@dataclass
class BodyTemplate:
    """Metadata for a creature body template."""
    name: str
    description: str
    factory_method: str  # GenomeFactory method name
    default_neurons: int = 5000
    n_actuators_hint: int = 8  # approximate, for config estimation
    tags: list = field(default_factory=list)  # e.g. ['legged', 'quadruped']


# Built-in templates — maps to GenomeFactory.create_*_template()
BODY_TEMPLATES: Dict[str, BodyTemplate] = {
    'synpaw': BodyTemplate(
        name='synpaw',
        description='Mogli-style quadruped (dog)',
        factory_method='create_mogli_template',
        default_neurons=5000,
        n_actuators_hint=12,
        tags=['legged', 'quadruped', 'dog'],
    ),
    'quadruped': BodyTemplate(
        name='quadruped',
        description='Generic four-legged creature',
        factory_method='create_quadruped_template',
        default_neurons=5000,
        n_actuators_hint=12,
        tags=['legged', 'quadruped'],
    ),
    'biped': BodyTemplate(
        name='biped',
        description='Two-legged upright walker',
        factory_method='create_biped_template',
        default_neurons=3000,
        n_actuators_hint=8,
        tags=['legged', 'biped', 'humanoid'],
    ),
    'worm': BodyTemplate(
        name='worm',
        description='Limbless segmented creature',
        factory_method='create_worm_template',
        default_neurons=2000,
        n_actuators_hint=6,
        tags=['limbless', 'worm'],
    ),
}

# Agent-type aliases — natural language → template name
AGENT_ALIASES: Dict[str, str] = {
    'dog': 'synpaw', 'hund': 'synpaw', 'mogli': 'synpaw', 'puppy': 'synpaw',
    'cat': 'quadruped', 'katze': 'quadruped',
    'horse': 'quadruped', 'pferd': 'quadruped',
    'robot': 'biped', 'humanoid': 'biped', 'person': 'biped',
    'snake': 'worm', 'schlange': 'worm',
    'worm': 'worm', 'wurm': 'worm',
}


# ================================================================
# SCENE REGISTRY
# ================================================================

# Environment hint → SceneBuilder SCENES key
SCENE_ALIASES: Dict[str, str] = {
    'flat_grass': 'flat_grass', 'grass': 'flat_grass', 'meadow': 'flat_grass',
    'ice': 'ice', 'frozen': 'ice', 'slippery': 'ice',
    'sand': 'sand', 'desert': 'sand', 'beach': 'sand',
    'rocky': 'rocky', 'rocks': 'rocky', 'mountain': 'rocky',
    'hills': 'hills', 'hilly': 'hills',
    'night': 'night', 'dark': 'night',
    'windy': 'windy', 'storm': 'windy',
    'obstacle_course': 'obstacle_course', 'obstacle': 'obstacle_course',
    'forest': 'rocky',  # closest approximation — roots and uneven ground
    'snow': 'ice',      # closest approximation
    'water': 'flat_grass',  # placeholder until swim scene exists
}


# ================================================================
# AUTO-MJCF ENGINE
# ================================================================

class AutoMJCF:
    """
    Automatically generate body genome and scene from task description.

    Bridges TaskParser output to GenomeFactory + SceneBuilder.
    """

    def __init__(self):
        self.body_templates = dict(BODY_TEMPLATES)
        self.agent_aliases = dict(AGENT_ALIASES)
        self.scene_aliases = dict(SCENE_ALIASES)

    def register_body_template(self, template: BodyTemplate):
        """Register a custom body template."""
        self.body_templates[template.name] = template
        logger.info(f'Registered body template: {template.name}')

    def resolve_template(self, name: str) -> str:
        """Resolve agent name or alias to a template key."""
        name_lower = name.lower().strip()
        # Direct match
        if name_lower in self.body_templates:
            return name_lower
        # Alias match
        if name_lower in self.agent_aliases:
            return self.agent_aliases[name_lower]
        # Fuzzy: check if name contains a known alias
        for alias, template in self.agent_aliases.items():
            if alias in name_lower:
                return template
        # Default
        logger.warning(f'Unknown agent type "{name}", defaulting to synpaw')
        return 'synpaw'

    def resolve_scene(self, hint: str) -> str:
        """Resolve environment hint to a SceneBuilder SCENES key."""
        hint_lower = hint.lower().strip()
        if hint_lower in self.scene_aliases:
            return self.scene_aliases[hint_lower]
        # Fuzzy match
        for alias, scene_key in self.scene_aliases.items():
            if alias in hint_lower or hint_lower in alias:
                return scene_key
        logger.warning(f'Unknown scene "{hint}", defaulting to flat_grass')
        return 'flat_grass'

    def build_body(
        self,
        template_name: str = 'synpaw',
        scale: float = 1.0,
        extra_requirements: Optional[Dict[str, Any]] = None,
    ):
        """
        Build a creature genome from a template name.

        Args:
            template_name: Template key or agent alias.
            scale: Size multiplier (1.0 = default).
            extra_requirements: Additional body_requirements from TaskParser.

        Returns:
            Genome instance ready for MJCF generation.
        """
        from src.body.genome import GenomeFactory

        resolved = self.resolve_template(template_name)
        template = self.body_templates.get(resolved)

        if template is None:
            logger.error(f'Template "{resolved}" not found, using synpaw')
            template = self.body_templates['synpaw']

        # Call the appropriate factory method
        factory_method = getattr(GenomeFactory, template.factory_method, None)
        if factory_method is None:
            logger.error(f'Factory method {template.factory_method} not found')
            genome = GenomeFactory.create_quadruped_template()
        else:
            genome = factory_method()

        # Apply scale modifier
        if scale != 1.0:
            self._apply_scale(genome, scale)

        logger.info(f'Built body: {template.name} (scale={scale:.1f})')
        return genome

    def build_scene(
        self,
        scene_hint: str = 'flat_grass',
        gravity_scale: float = 1.0,
    ) -> str:
        """
        Build scene MJCF XML from a scene hint.

        Args:
            scene_hint: Environment name or alias.
            gravity_scale: Gravity multiplier (1.0 = normal, <1 = reduced).

        Returns:
            Scene key for use with SceneBuilder.build().
        """
        resolved = self.resolve_scene(scene_hint)
        logger.info(f'Resolved scene: {scene_hint} -> {resolved} (gravity={gravity_scale:.1f})')
        return resolved

    def from_task(self, task) -> Tuple:
        """
        Build body genome + scene from a ParsedTask.

        Args:
            task: ParsedTask from TaskParser.

        Returns:
            (genome, scene_key) tuple.
        """
        # Determine body template
        body_reqs = task.body_requirements or {}
        template_name = body_reqs.get('template', 'synpaw')
        scale = body_reqs.get('scale', 1.0)

        genome = self.build_body(template_name, scale=scale,
                                 extra_requirements=body_reqs)

        # Determine scene
        scene_key = 'flat_grass'
        if task.environment_hints:
            scene_key = self.build_scene(task.environment_hints[0])

        return genome, scene_key

    def get_template_info(self, name: str) -> Optional[BodyTemplate]:
        """Get metadata for a body template."""
        resolved = self.resolve_template(name)
        return self.body_templates.get(resolved)

    def list_templates(self) -> Dict[str, BodyTemplate]:
        """List all available body templates."""
        return dict(self.body_templates)

    def list_scenes(self) -> list:
        """List all available scene keys."""
        return sorted(set(self.scene_aliases.values()))

    # ── Private helpers ──

    @staticmethod
    def _apply_scale(genome, scale: float):
        """Scale all segment sizes in a genome."""
        if not hasattr(genome, 'segments'):
            return
        for segment in genome.segments:
            if hasattr(segment, 'size'):
                if isinstance(segment.size, (list, tuple)):
                    segment.size = [s * scale for s in segment.size]
                elif isinstance(segment.size, (int, float)):
                    segment.size = segment.size * scale
            if hasattr(segment, 'mass'):
                # Mass scales with volume (cube of linear scale)
                segment.mass = segment.mass * (scale ** 3)
