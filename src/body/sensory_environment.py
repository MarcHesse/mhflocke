"""
MH-FLOCKE — Sensory Environment v0.4.8
========================================
Olfactory gradients and acoustic events for stimulus-driven behavior.

v0.4.8: Heading-aware scent respawn (2026-04-22).
  Scent sources now spawn AHEAD of the creature's current heading,
  not in a fixed direction. This prevents the creature from walking
  past scents that spawned behind or beside it.
  Also stores creature heading for use in respawn logic.

v0.4.7: Fixed radius vs spawn distance mismatch (2026-04-22).
  radius=3.0 was larger than max_dist=2.0 → scents were instantly
  "reached" at spawn → respawned further away → sm=0.06 instead of 0.35.
  Fix: radius=0.5 (must walk TO the scent, not just spawn near it).

v0.4.6: Quadratic scent decay 1/(0.5+d)^2 for steeper gradient.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScentSource:
    """A scent source in the world."""
    position: np.ndarray    # [x, y, z]
    strength: float = 1.0   # Base intensity
    radius: float = 0.5     # v0.4.7: must be < min_dist to avoid instant reach
    name: str = 'scent'
    fixed: bool = False     # If True, never respawn (e.g. ball scent)


@dataclass
class SoundEvent:
    """A transient acoustic event."""
    direction: np.ndarray   # Unit vector [x, y, z]
    intensity: float = 1.0  # 0..1
    remaining_steps: int = 0  # Steps until event fades


class SensoryEnvironment:
    """
    Manages sensory stimuli in the world.

    Creates scent targets that the creature can navigate toward,
    and periodic sound events that trigger orienting behavior.
    """

    def __init__(self, world_size: float = 10.0, seed: int = 42,
                 sound_interval: int = 2000, sound_duration: int = 500):
        self._rng = np.random.RandomState(seed)
        self._world_size = world_size
        self._scents: List[ScentSource] = []
        self._sound: Optional[SoundEvent] = None
        self._sound_interval = sound_interval
        self._sound_duration = sound_duration
        self._next_sound_step = self._rng.randint(1000, sound_interval)
        self._scents_found = 0
        self._step = 0
        self._creature_pos = np.zeros(3)
        self._creature_heading = 0.0  # v0.4.8: track heading for forward-spawning
        self._initialized = False

    def spawn_scent(self, count: int = 2, min_dist: float = 2.0,
                    max_dist: float = 5.0):
        self._scents.clear()
        for i in range(count):
            self._spawn_new_scent(i, self._creature_pos, min_dist, max_dist)

    def _spawn_new_scent(self, idx: int, ref_pos: np.ndarray,
                         min_dist: float = 2.0, max_dist: float = 5.0,
                         heading: float = None):
        # v0.4.8: spawn AHEAD of creature's current heading
        # ±30° cone in front of creature, not fixed ±0.5 rad from +X axis
        base_angle = heading if heading is not None else self._creature_heading
        angle = base_angle + self._rng.uniform(-0.5, 0.5)
        dist = self._rng.uniform(min_dist, max_dist)
        x = ref_pos[0] + dist * np.cos(angle)
        y = ref_pos[1] + dist * np.sin(angle)
        pos = np.array([x, y, 0.0])

        scent = ScentSource(
            position=pos, strength=1.0, radius=0.8,  # v0.4.8: 0.5→0.8 for small robots
            name=f'scent_{idx}'
        )

        if idx < len(self._scents):
            self._scents[idx] = scent
        else:
            self._scents.append(scent)

        logger.info(f"Scent '{scent.name}' at ({pos[0]:.1f},{pos[1]:.1f}) "
                    f"dist={dist:.1f}m from ({ref_pos[0]:.1f},{ref_pos[1]:.1f})")

    def get_smell_gradient(self, creature_pos: np.ndarray
                           ) -> Tuple[float, np.ndarray]:
        """
        Compute olfactory gradient at creature position.

        v0.4.6: Quadratic decay 1/(0.5+d)^2 for steeper gradient.
        """
        self._creature_pos = creature_pos.copy()

        if not self._scents:
            return 0.0, np.zeros(3)

        best_strength = 0.0
        best_direction = np.zeros(3)

        for scent in self._scents:
            delta = scent.position - creature_pos
            delta[2] = 0.0
            dist = np.linalg.norm(delta)

            strength = scent.strength / (0.5 + dist) ** 2

            if strength > best_strength:
                best_strength = strength
                best_direction = delta / max(dist, 0.01)

        return min(1.0, best_strength), best_direction

    def get_olfactory_steering(self, creature_pos: np.ndarray,
                               heading: float) -> float:
        self._creature_heading = heading  # v0.4.8: store for heading-aware respawn
        smell_strength, smell_dir = self.get_smell_gradient(creature_pos)

        if smell_strength < 0.05:
            return 0.0

        scent_angle = np.arctan2(smell_dir[1], smell_dir[0])
        angle_diff = scent_angle - heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        steering = np.clip(angle_diff / np.pi, -1.0, 1.0) * smell_strength

        return float(steering)

    def check_scent_reached(self, creature_pos: np.ndarray) -> bool:
        reached = False
        for i, scent in enumerate(self._scents):
            delta = scent.position - creature_pos
            delta[2] = 0.0
            dist = np.linalg.norm(delta)

            if dist < scent.radius:
                self._scents_found += 1
                logger.info(f"Scent reached! ({self._scents_found} total) "
                            f"at dist={dist:.2f}m")
                if not scent.fixed:
                    self._spawn_new_scent(i, creature_pos)
                reached = True

            elif dist > 3.0 and not scent.fixed:
                # v0.4.8: reduced from 5.0 to 3.0m — don't let scents
                # linger far behind. Respawn ahead of creature's heading.
                logger.debug(f"Scent {i} faded (dist={dist:.1f}m), respawning ahead")
                self._spawn_new_scent(i, creature_pos)

        return reached

    def update_sound(self, step: int) -> Optional[SoundEvent]:
        self._step = step

        if step >= self._next_sound_step:
            if self._scents and self._rng.random() < 0.5:
                nearest = self._scents[0]
                min_dist = float('inf')
                for sc in self._scents:
                    d = np.linalg.norm(sc.position - self._creature_pos)
                    if d < min_dist:
                        min_dist = d
                        nearest = sc
                delta = nearest.position - self._creature_pos
                delta[2] = 0.0
                norm = np.linalg.norm(delta)
                if norm > 0.01:
                    direction = delta / norm
                    noise_angle = self._rng.uniform(-0.35, 0.35)
                    cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
                    direction = np.array([
                        direction[0] * cos_n - direction[1] * sin_n,
                        direction[0] * sin_n + direction[1] * cos_n,
                        0.0
                    ])
                else:
                    direction = self._random_direction()
            else:
                direction = self._random_direction()

            intensity = self._rng.uniform(0.3, 1.0)
            self._sound = SoundEvent(
                direction=direction,
                intensity=intensity,
                remaining_steps=self._sound_duration
            )
            self._next_sound_step = step + self._rng.randint(
                self._sound_interval // 2, self._sound_interval * 2)

        if self._sound and self._sound.remaining_steps > 0:
            self._sound.remaining_steps -= 1
            fade = self._sound.remaining_steps / self._sound_duration
            return SoundEvent(
                direction=self._sound.direction,
                intensity=self._sound.intensity * fade,
                remaining_steps=self._sound.remaining_steps
            )
        else:
            self._sound = None
            return None

    def _random_direction(self) -> np.ndarray:
        angle = self._rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    @property
    def scents_found(self) -> int:
        return self._scents_found

    def get_state(self) -> Dict:
        return {
            'scents_active': len(self._scents),
            'scents_found': self._scents_found,
            'sound_active': self._sound is not None and
                            self._sound.remaining_steps > 0,
        }
