#!/usr/bin/env python3
"""
Level 14: Scene Generator — Auto-Generate MuJoCo Scenes
==========================================================
Creates new scenes dynamically from UNDERSTAND results.

Instead of only mapping to 7 pre-built scenes, this generates
custom Scene objects with appropriate terrain, objects, lighting,
and physics based on what UNDERSTAND learned about the task.

Pipeline:
    UnderstandResult.scene_requirements
      → SceneGenerator.generate()
      → Custom Scene with:
        - Terrain friction from environment type
        - Objects (trees, rocks, etc.) as SceneObjects
        - Lighting from time of day
        - Physics overrides (gravity, wind)

Usage:
    from src.bridge.scene_generator import SceneGenerator

    gen = SceneGenerator()
    scene = gen.generate(understand_result.scene_requirements, 'forest')
    # scene is a Scene object compatible with SceneBuilder.build()
"""

__version__ = "0.1.0"
__logbook__ = 146

import logging
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SceneGenerator:
    """
    Generates custom MuJoCo scenes from task understanding.

    Extends the 7 built-in scenes with procedurally generated
    objects, terrain, and atmospheric conditions.
    """

    # Terrain type → floor parameters
    TERRAIN_PARAMS = {
        'flat': {
            'floor_friction': '1.0 0.5 0.01',
            'floor_rgba': '',  # use texture
        },
        'uneven': {
            'floor_friction': '1.2 0.6 0.02',
            'floor_rgba': '',
        },
        'slippery': {
            'floor_friction': '0.15 0.02 0.001',
            'floor_rgba': '0.85 0.92 0.95 1',
        },
        'soft': {
            'floor_friction': '0.8 0.7 0.05',
            'floor_rgba': '0.85 0.78 0.55 1',
        },
        'steep': {
            'floor_friction': '1.4 0.8 0.03',
            'floor_rgba': '',
        },
    }

    # Lighting presets
    LIGHTING = {
        'day': {
            'light_diffuse': '0.7 0.65 0.6',
            'light_specular': '0.3 0.3 0.3',
            'skybox_rgb1': '0.4 0.6 0.8',
            'skybox_rgb2': '0.85 0.9 0.95',
            'haze_rgba': '0.04 0.055 0.10 1',
        },
        'night': {
            'light_diffuse': '0.08 0.08 0.15',
            'light_specular': '0.05 0.05 0.08',
            'skybox_rgb1': '0.02 0.02 0.06',
            'skybox_rgb2': '0.05 0.05 0.12',
            'haze_rgba': '0.01 0.01 0.03 1',
        },
        'dusk': {
            'light_diffuse': '0.5 0.35 0.25',
            'light_specular': '0.2 0.15 0.1',
            'skybox_rgb1': '0.6 0.3 0.15',
            'skybox_rgb2': '0.2 0.15 0.3',
            'haze_rgba': '0.06 0.04 0.03 1',
        },
    }

    # Object templates
    OBJECT_TEMPLATES = {
        'trees': {
            'geom_type': 'cylinder',
            'size': '0.15 0.15 1.5',
            'rgba': '0.35 0.25 0.15 1',
            'count': (3, 8),  # min, max
            'radius': (3.0, 10.0),  # spawn radius
            'z_offset': 1.5,
        },
        'rocks': {
            'geom_type': 'sphere',
            'size': '0.2 0.2 0.15',
            'rgba': '0.45 0.42 0.38 1',
            'count': (4, 12),
            'radius': (2.0, 8.0),
            'z_offset': 0.15,
        },
        'roots': {
            'geom_type': 'capsule',
            'size': '0.04 0.3',
            'rgba': '0.3 0.2 0.12 1',
            'count': (5, 15),
            'radius': (1.5, 6.0),
            'z_offset': 0.04,
        },
        'dunes': {
            'geom_type': 'ellipsoid',
            'size': '1.5 1.0 0.3',
            'rgba': '0.85 0.78 0.55 1',
            'count': (2, 5),
            'radius': (4.0, 12.0),
            'z_offset': 0.3,
        },
        'ice_patches': {
            'geom_type': 'box',
            'size': '0.8 0.8 0.02',
            'rgba': '0.9 0.95 1.0 0.7',
            'count': (3, 8),
            'radius': (2.0, 8.0),
            'z_offset': 0.02,
            'friction': '0.05 0.01 0.001',
        },
        'leaves': {
            'geom_type': 'box',
            'size': '0.3 0.3 0.01',
            'rgba': '0.4 0.5 0.2 0.8',
            'count': (10, 25),
            'radius': (1.0, 8.0),
            'z_offset': 0.01,
        },
        'grass': {
            # Grass is just floor texture, no objects
            'geom_type': None,
            'count': (0, 0),
        },
        'hills': {
            'geom_type': 'ellipsoid',
            'size': '3.0 2.0 0.6',
            'rgba': '0.3 0.5 0.2 1',
            'count': (2, 4),
            'radius': (5.0, 15.0),
            'z_offset': 0.6,
        },
        'slopes': {
            'geom_type': 'box',
            'size': '3.0 2.0 0.4',
            'rgba': '0.35 0.45 0.25 1',
            'count': (1, 3),
            'radius': (4.0, 10.0),
            'z_offset': 0.2,
            'euler': '10 0 0',
        },
        'boulders': {
            'geom_type': 'sphere',
            'size': '0.5 0.5 0.4',
            'rgba': '0.4 0.38 0.35 1',
            'count': (2, 6),
            'radius': (3.0, 10.0),
            'z_offset': 0.4,
        },
        'water': {
            'geom_type': 'box',
            'size': '2.0 2.0 0.02',
            'rgba': '0.2 0.4 0.7 0.5',
            'count': (1, 2),
            'radius': (3.0, 8.0),
            'z_offset': 0.02,
            'friction': '0.3 0.2 0.01',
        },
    }

    # Weather → physics
    WEATHER_PHYSICS = {
        'clear': {'wind': '', 'gravity': ''},
        'wind': {'wind': '2.0 0.5 0', 'gravity': ''},
        'rain': {'wind': '0.5 0.3 0', 'gravity': ''},
        'snow': {'wind': '1.0 0.5 0', 'gravity': ''},
    }

    def generate(self, scene_req, environment_name: str = 'custom') -> 'Scene':
        """
        Generate a custom Scene from SceneRequirement.

        Args:
            scene_req: SceneRequirement from UnderstandResult.
            environment_name: Name for the generated scene.

        Returns:
            Scene object ready for SceneBuilder.build().
        """
        from src.body.scene_builder import Scene, SceneObject

        # Terrain
        terrain_params = self.TERRAIN_PARAMS.get(
            scene_req.terrain, self.TERRAIN_PARAMS['flat'])

        # Lighting
        lighting = self.LIGHTING.get(
            scene_req.lighting, self.LIGHTING['day'])

        # Weather physics
        weather = self.WEATHER_PHYSICS.get(
            scene_req.weather, self.WEATHER_PHYSICS['clear'])

        # Generate objects
        objects = self._generate_objects(scene_req.objects)

        # Build Scene
        scene = Scene(
            name=f'{environment_name}_generated',
            description=f'Auto-generated: {scene_req.terrain} terrain, '
                        f'{scene_req.lighting}, {scene_req.weather}',
            floor_friction=terrain_params['floor_friction'],
            floor_rgba=terrain_params.get('floor_rgba', ''),
            light_diffuse=lighting['light_diffuse'],
            light_specular=lighting['light_specular'],
            skybox_rgb1=lighting.get('skybox_rgb1', ''),
            skybox_rgb2=lighting.get('skybox_rgb2', ''),
            haze_rgba=lighting.get('haze_rgba', '0.04 0.055 0.10 1'),
            wind=weather.get('wind', ''),
            gravity=weather.get('gravity', ''),
            objects=objects,
        )

        logger.info(f'Generated scene "{scene.name}": '
                     f'{len(objects)} objects, terrain={scene_req.terrain}')
        return scene

    def _generate_objects(self, object_names: List[str]) -> list:
        """Generate SceneObjects from object name list."""
        from src.body.scene_builder import SceneObject

        all_objects = []

        for obj_name in object_names:
            template = self.OBJECT_TEMPLATES.get(obj_name)
            if not template or template.get('geom_type') is None:
                continue

            min_count, max_count = template['count']
            count = random.randint(min_count, max_count)
            min_r, max_r = template['radius']

            for i in range(count):
                # Random position in ring around origin
                angle = random.uniform(0, 6.283)
                r = random.uniform(min_r, max_r)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = template.get('z_offset', 0)

                obj = SceneObject(
                    name=f'{obj_name}_{i}',
                    geom_type=template['geom_type'],
                    pos=f'{x:.2f} {y:.2f} {z:.2f}',
                    size=template['size'],
                    rgba=template['rgba'],
                    euler=template.get('euler', '0 0 0'),
                    friction=template.get('friction', '1.0 0.5 0.01'),
                )
                all_objects.append(obj)

        return all_objects

    def get_or_generate(self, scene_req, environment_name: str = 'custom') -> 'Scene':
        """
        Get existing scene if available, otherwise generate.

        Prefers built-in scenes (tested, reliable), falls back to generation.
        """
        from src.body.scene_builder import SCENES

        # Check if suggested_scene exists in built-in scenes
        suggested = scene_req.suggested_scene
        if suggested in SCENES:
            logger.info(f'Using built-in scene: {suggested}')
            return SCENES[suggested]

        # Generate custom scene
        return self.generate(scene_req, environment_name)

    @staticmethod
    def list_templates() -> List[str]:
        """List available object templates."""
        return [k for k, v in SceneGenerator.OBJECT_TEMPLATES.items()
                if v.get('geom_type') is not None]


# Need numpy for object placement
import numpy as np
