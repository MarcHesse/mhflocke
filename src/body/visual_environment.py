"""
MH-FLOCKE — Visual Environment v0.1.0
========================================
Phototaxis navigation using onboard camera.

The robot navigates toward light sources using bilateral brightness
comparison — identical algorithm to Run-and-Tumble chemotaxis but
with camera pixels instead of scent gradients.

Sim: MuJoCo renders 64x48 from onboard camera, bright sphere = target.
Hardware: cv2.VideoCapture(0), flashlight on floor = target.

Biology: Phototaxis (moth navigation) uses the same mechanism as
chemotaxis (bacterial navigation). Berg & Brown 1972 applies to both.

API matches SensoryEnvironment so train_baby.py can swap with a flag.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MuJoCo import — optional, only needed in simulator
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@dataclass
class LightSource:
    """A light target in the world."""
    position: np.ndarray    # [x, y, z]
    intensity: float = 1.0  # Base brightness
    radius: float = 0.8     # Reach radius (same as scent)
    color: np.ndarray = None  # RGB for rendering
    name: str = 'light'

    def __post_init__(self):
        if self.color is None:
            self.color = np.array([1.0, 1.0, 0.8])  # Warm white


class VisualEnvironment:
    """
    Light-based navigation using onboard camera.

    Drop-in replacement for SensoryEnvironment with identical API:
      get_light_gradient()     ↔ get_smell_gradient()
      get_phototactic_steering() ↔ get_olfactory_steering()
      check_light_reached()    ↔ check_scent_reached()

    In simulator: renders MuJoCo onboard camera, compares left/right.
    On hardware: reads cv2 frame, same bilateral comparison.
    """

    def __init__(self, world_size: float = 10.0, seed: int = 42,
                 cam_width: int = 64, cam_height: int = 48):
        self._rng = np.random.RandomState(seed)
        self._world_size = world_size
        self._lights: List[LightSource] = []
        self._lights_found = 0
        self._step = 0
        self._creature_pos = np.zeros(3)
        self._creature_heading = 0.0

        # Camera settings
        self._cam_w = cam_width
        self._cam_h = cam_height
        self._last_frame = None  # Last rendered frame (H, W, 3) uint8
        self._last_brightness = 0.0
        self._last_direction = 0.0  # -1 left, +1 right

        # MuJoCo renderer (lazy init)
        self._renderer = None
        self._cam_id = -1

    def init_renderer(self, model, data):
        """Initialize MuJoCo offscreen renderer for onboard camera.

        Call this ONCE after model is loaded.
        """
        if not HAS_MUJOCO:
            logger.warning("MuJoCo not available — camera rendering disabled")
            return

        # Find the onboard camera
        self._cam_id = -1
        for i in range(model.ncam):
            name = model.camera(i).name
            if name == 'onboard':
                self._cam_id = i
                break

        if self._cam_id < 0:
            logger.warning("No 'onboard' camera found in model — "
                           "phototaxis will use geometric fallback")
            return

        # Create offscreen renderer
        self._renderer = mujoco.Renderer(model, height=self._cam_h,
                                         width=self._cam_w)
        logger.info(f"  Phototaxis: onboard camera initialized "
                    f"({self._cam_w}x{self._cam_h}, cam_id={self._cam_id})")

    # Fixed waypoints — spaced for Freenove turning radius (~5m)
    # 12% CPG steering asymmetry = wide arcs, points must be mostly ahead
    WAYPOINTS = [
        (3.5,  0.5),   # nearly straight, slight right
        (3.5, -0.8),   # slight left
        (3.5,  1.0),   # gentle right arc
        (3.5, -1.0),   # gentle left arc
        (3.5,  0.5),   # slight right
        (3.5, -0.5),   # slight left
        (3.5,  0.8),   # gentle right
        (3.5, -0.8),   # gentle left
    ]

    def spawn_lights(self, count: int = 1, min_dist: float = 1.5,
                     max_dist: float = 3.0):
        """Spawn first waypoint light source."""
        self._lights.clear()
        self._waypoint_idx = 0
        self._spawn_waypoint(self._creature_pos)

    def _spawn_waypoint(self, ref_pos: np.ndarray = None):
        """Spawn next waypoint RELATIVE to current creature position."""
        if ref_pos is None:
            ref_pos = self._creature_pos
        wp = self.WAYPOINTS[self._waypoint_idx % len(self.WAYPOINTS)]
        # Relative to current position — so waypoints always stay reachable
        x = ref_pos[0] + wp[0]
        y = ref_pos[1] + wp[1]
        pos = np.array([x, y, 0.02])

        light = LightSource(
            position=pos, intensity=3.0, radius=2.0,
            name=f'waypoint_{self._waypoint_idx}'
        )

        if len(self._lights) > 0:
            self._lights[0] = light
        else:
            self._lights.append(light)

        logger.info(f"Waypoint {self._waypoint_idx} at ({pos[0]:.1f},{pos[1]:.1f})")

    def _spawn_new_light(self, idx: int, ref_pos: np.ndarray,
                         min_dist: float = 1.5, max_dist: float = 3.0,
                         heading: float = None):
        """Advance to next waypoint (called when light is reached)."""
        self._waypoint_idx += 1
        self._spawn_waypoint(ref_pos)

    def render_onboard(self, model, data) -> Optional[np.ndarray]:
        """Render the onboard camera view.

        Returns (H, W, 3) uint8 array or None if renderer not available.
        """
        if self._renderer is None or self._cam_id < 0:
            return None

        try:
            self._renderer.update_scene(data, camera=self._cam_id)
            frame = self._renderer.render()
            self._last_frame = frame.copy()
            return frame
        except Exception as e:
            logger.debug(f"Camera render failed: {e}")
            return None

    def get_bilateral_brightness(self, frame: np.ndarray = None
                                  ) -> Tuple[float, float]:
        """Compute bilateral brightness from camera frame.

        Returns (brightness, direction) where:
          brightness: 0.0 (dark) to 1.0 (bright)
          direction: -1.0 (light on left) to +1.0 (light on right)
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return 0.0, 0.0

        # Convert to grayscale
        gray = np.mean(frame, axis=2).astype(np.float32)
        h, w = gray.shape

        # Split left/right
        left_mean = gray[:, :w // 2].mean()
        right_mean = gray[:, w // 2:].mean()
        total = left_mean + right_mean

        # Overall brightness (0-1)
        brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))

        # Direction: -1 (left brighter) to +1 (right brighter)
        if total < 5.0:  # Too dark, no signal
            direction = 0.0
        else:
            direction = float((right_mean - left_mean) / total)

        self._last_brightness = brightness
        self._last_direction = direction

        return brightness, direction

    def get_light_gradient(self, creature_pos: np.ndarray,
                           model=None, data=None
                           ) -> Tuple[float, np.ndarray]:
        """
        Get light gradient — drop-in for get_smell_gradient().

        If MuJoCo renderer is available, uses actual camera image.
        Otherwise falls back to geometric calculation (like scent).
        """
        self._creature_pos = creature_pos.copy()

        # Try camera-based sensing first
        if model is not None and data is not None and self._renderer is not None:
            frame = self.render_onboard(model, data)
            if frame is not None:
                brightness, direction = self.get_bilateral_brightness(frame)
                # Convert bilateral direction to 3D direction vector
                # direction>0 means light is to the right of the camera
                # In MuJoCo: camera Y-axis points right
                heading = self._creature_heading
                # Light direction in world frame
                light_angle = heading - direction * 0.5 * np.pi
                dir_vec = np.array([np.cos(light_angle),
                                    np.sin(light_angle), 0.0])
                return brightness, dir_vec

        # Geometric fallback (same as scent — for testing without renderer)
        if not self._lights:
            return 0.0, np.zeros(3)

        best_strength = 0.0
        best_direction = np.zeros(3)

        for light in self._lights:
            delta = light.position - creature_pos
            delta[2] = 0.0
            dist = np.linalg.norm(delta)
            strength = light.intensity / (0.5 + dist) ** 2

            if strength > best_strength:
                best_strength = strength
                best_direction = delta / max(dist, 0.01)

        return min(1.0, best_strength), best_direction

    def get_phototactic_steering(self, creature_pos: np.ndarray,
                                  heading: float,
                                  model=None, data=None) -> float:
        """
        Phototactic steering — drop-in for get_olfactory_steering().

        Uses camera if available, geometric fallback otherwise.
        """
        self._creature_heading = heading

        # Camera-based: direct bilateral comparison
        if model is not None and data is not None and self._renderer is not None:
            frame = self.render_onboard(model, data)
            if frame is not None:
                brightness, direction = self.get_bilateral_brightness(frame)
                if brightness < 0.03:
                    return 0.0
                # Direction is already -1..+1, scale by brightness
                return float(np.clip(direction * brightness * 2.0, -1.0, 1.0))

        # Geometric fallback (same logic as olfactory)
        light_strength, light_dir = self.get_light_gradient(creature_pos)
        if light_strength < 0.05:
            return 0.0

        light_angle = np.arctan2(light_dir[1], light_dir[0])
        angle_diff = light_angle - heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        steering = np.clip(angle_diff / np.pi, -1.0, 1.0) * light_strength
        return float(steering)

    def check_light_reached(self, creature_pos: np.ndarray) -> bool:
        """Check if creature reached a light source — like check_scent_reached()."""
        reached = False
        for i, light in enumerate(self._lights):
            delta = light.position - creature_pos
            delta[2] = 0.0
            dist = np.linalg.norm(delta)

            if dist < light.radius:
                self._lights_found += 1
                logger.info(f"Waypoint {self._waypoint_idx} reached! "
                            f"({self._lights_found} total) dist={dist:.2f}m")
                self._spawn_new_light(i, creature_pos)
                reached = True
            elif dist > 4.5:
                # Waypoint missed — respawn ahead of creature
                logger.info(f"Waypoint {self._waypoint_idx} missed (dist={dist:.1f}m), respawning")
                self._spawn_waypoint(creature_pos)

        return reached

    def inject_light_geom(self, model, data) -> int:
        """
        Inject a bright sphere into the MuJoCo scene as visual light source.

        Uses MuJoCo visualization geoms (mjv) — doesn't modify the physics model.
        Returns number of lights placed.

        The sphere is bright yellow/white, emissive, placed at the light
        source position. This is what the onboard camera will see.
        """
        # We inject visual-only geoms during rendering, not here.
        # The actual injection happens in the renderer's scene update.
        # This method returns the positions for the renderer to use.
        return len(self._lights)

    def get_light_positions(self) -> List[np.ndarray]:
        """Get positions of all light sources for scene injection."""
        return [l.position.copy() for l in self._lights]

    @property
    def lights_found(self) -> int:
        return self._lights_found

    # Aliases for compatibility with sensory_environment API
    @property
    def scents_found(self) -> int:
        """Alias so train_baby.py can use same field name."""
        return self._lights_found

    def get_state(self) -> Dict:
        return {
            'lights_active': len(self._lights),
            'lights_found': self._lights_found,
            'brightness': self._last_brightness,
            'light_direction': self._last_direction,
        }
