"""
MH-FLOCKE — Sensory Environment v0.4.1
========================================
Olfactory gradients and acoustic events for stimulus-driven behavior.
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
    radius: float = 3.0     # Distance at which scent is "found"
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

    v0.4.0 changes:
    - Scents spawn relative to creature position (not origin)
    - Scents auto-respawn when >5m behind creature (was >8m)
    - Olfactory steering signal for motor modulation
    - Larger find radius (3.0m, was 2.5m)
    - Sound events from scent direction (biased, not purely random)
    """

    def __init__(self, world_size: float = 10.0, seed: int = 42,
                 sound_interval: int = 2000, sound_duration: int = 500):
        """
        Args:
            world_size: Radius of the usable world area
            seed: RNG seed for reproducible scent/sound placement
            sound_interval: Average steps between sound events
            sound_duration: Steps a sound event lasts
        """
        self._rng = np.random.RandomState(seed)
        self._world_size = world_size
        self._scents: List[ScentSource] = []
        self._sound: Optional[SoundEvent] = None
        self._sound_interval = sound_interval
        self._sound_duration = sound_duration
        self._next_sound_step = self._rng.randint(1000, sound_interval)
        self._scents_found = 0
        self._step = 0
        self._creature_pos = np.zeros(3)  # Track for relative spawning
        self._initialized = False

    # ── Scent Management ────────────────────────────────────

    def spawn_scent(self, count: int = 2, min_dist: float = 2.0,
                    max_dist: float = 5.0):
        """
        Place scent sources ahead of creature.

        Biology: food/territory scent carried by wind from the
        direction the animal is heading. Sources are forward-biased
        and close enough to reach within reasonable walking distance.

        Args:
            count: Number of scent sources
            min_dist: Minimum distance from creature
            max_dist: Maximum distance from creature
        """
        self._scents.clear()
        for i in range(count):
            self._spawn_new_scent(i, self._creature_pos, min_dist, max_dist)

    def _spawn_new_scent(self, idx: int, ref_pos: np.ndarray,
                         min_dist: float = 2.0, max_dist: float = 5.0):
        """
        Spawn a single scent source ahead of a reference position.

        Biology: wind-carried scent plumes drift from food sources.
        The animal detects the plume and follows it upwind to the source.
        Ref: Catania 2006 — star-nosed moles navigate via bilateral
        olfactory gradient comparison.

        Narrow forward cone (±30°) — scent arrives from ahead, not behind.
        """
        angle = self._rng.uniform(-0.5, 0.5)  # ±30° (was ±45°)
        dist = self._rng.uniform(min_dist, max_dist)
        x = ref_pos[0] + dist * np.cos(angle)
        y = ref_pos[1] + dist * np.sin(angle)
        pos = np.array([x, y, 0.0])

        scent = ScentSource(
            position=pos, strength=1.0, radius=3.0,
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

        Biology: olfactory receptor neurons detect concentration
        gradient via bilateral comparison (stereo olfaction).
        Strength decays linearly with distance:
            smell = strength / (1.0 + dist)
        Linear decay provides usable signal at moderate distances.

        Ref: Porter et al. 2007 — humans can track scent trails
             using cross-nostril comparison.

        Returns:
            (smell_strength, smell_direction)
            smell_strength: 0.0 (no scent) to 1.0 (at source)
            smell_direction: unit vector toward strongest scent [x, y, z]
        """
        self._creature_pos = creature_pos.copy()

        if not self._scents:
            return 0.0, np.zeros(3)

        best_strength = 0.0
        best_direction = np.zeros(3)

        for scent in self._scents:
            delta = scent.position - creature_pos
            delta[2] = 0.0  # Ignore height for smell
            dist = np.linalg.norm(delta)

            # Linear decay
            strength = scent.strength / (1.0 + dist)

            if strength > best_strength:
                best_strength = strength
                best_direction = delta / max(dist, 0.01)

        return min(1.0, best_strength), best_direction

    def get_olfactory_steering(self, creature_pos: np.ndarray,
                               heading: float) -> float:
        """
        Compute steering signal from olfactory gradient.

        Biology: bilateral olfactory comparison creates a turning
        tendency toward the stronger-smelling nostril. This is the
        fundamental mechanism of chemotaxis across species from
        C. elegans to mammals.

        Ref: Catania 2006 — moles alternate nostril sampling
             Duistermars et al. 2009 — Drosophila odor-guided turning

        Args:
            creature_pos: Current [x, y, z] position
            heading: Current heading angle (radians, 0 = forward/+x)

        Returns:
            steering: -1.0 (turn left) to +1.0 (turn right)
                      Magnitude proportional to smell strength and
                      angular offset to scent source.
        """
        smell_strength, smell_dir = self.get_smell_gradient(creature_pos)

        if smell_strength < 0.05:
            return 0.0  # Below detection threshold

        # Angle to scent source
        scent_angle = np.arctan2(smell_dir[1], smell_dir[0])
        angle_diff = scent_angle - heading

        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Steering proportional to angle offset and smell strength
        # Strong smell + large offset = strong turn
        # Weak smell = weak turn (uncertain direction)
        steering = np.clip(angle_diff / np.pi, -1.0, 1.0) * smell_strength

        return float(steering)

    def check_scent_reached(self, creature_pos: np.ndarray) -> bool:
        """
        Check if creature has reached a scent source.
        Also respawn scents that are too far away (faded/passed).

        Biology: scent plumes are transient. Wind shifts, thermal
        currents, and diffusion cause plumes to appear and disappear.
        When an animal can no longer detect a scent (strength below
        threshold), a new plume from a different source arrives.
        Ref: Murlis et al. 1992 — odor plumes in turbulent airflow

        Returns:
            True if a scent was reached (reward signal)
        """
        reached = False
        for i, scent in enumerate(self._scents):
            delta = scent.position - creature_pos
            delta[2] = 0.0
            dist = np.linalg.norm(delta)

            # Reached: close enough to "find" the source
            if dist < scent.radius:
                self._scents_found += 1
                logger.info(f"Scent reached! ({self._scents_found} total) "
                            f"at dist={dist:.2f}m")
                if not scent.fixed:
                    self._spawn_new_scent(i, creature_pos)
                reached = True

            # Faded: scent too weak to follow (moved away or wind shifted)
            # Fixed scents (e.g. ball) never fade — they stay at their position.
            elif dist > 5.0 and not scent.fixed:
                logger.debug(f"Scent {i} faded (dist={dist:.1f}m), respawning ahead")
                self._spawn_new_scent(i, creature_pos)

        return reached

    # ── Sound Events ────────────────────────────────────────

    def update_sound(self, step: int) -> Optional[SoundEvent]:
        """
        Generate periodic sound events.

        Biology: environmental sounds (wind, other animals, rustling)
        trigger the orienting reflex (Sokolov 1963). The animal
        freezes briefly, orients toward the sound, then resumes.

        50% of sounds come from scent direction (rustling at food
        source), 50% from random directions (environmental noise).
        This creates a biologically realistic soundscape where some
        sounds are informative and others are distractors.

        Returns:
            Active SoundEvent or None
        """
        self._step = step

        # Spawn new sound?
        if step >= self._next_sound_step:
            # 50% chance: sound from scent direction (informative)
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
                    # Add some noise to direction (±20°)
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
                # Random direction (environmental noise)
                direction = self._random_direction()

            intensity = self._rng.uniform(0.3, 1.0)
            self._sound = SoundEvent(
                direction=direction,
                intensity=intensity,
                remaining_steps=self._sound_duration
            )
            self._next_sound_step = step + self._rng.randint(
                self._sound_interval // 2, self._sound_interval * 2)
            logger.debug(f"Sound event: dir=({direction[0]:.2f},{direction[1]:.2f}), "
                         f"intensity={intensity:.2f}")

        # Decay active sound
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
        """Random unit vector in xy-plane."""
        angle = self._rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    # ── State ───────────────────────────────────────────────

    @property
    def scents_found(self) -> int:
        return self._scents_found

    def get_state(self) -> Dict:
        """For FLOG logging."""
        return {
            'scents_active': len(self._scents),
            'scents_found': self._scents_found,
            'sound_active': self._sound is not None and
                            self._sound.remaining_steps > 0,
        }
