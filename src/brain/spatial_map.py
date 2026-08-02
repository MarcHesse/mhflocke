"""
MH-FLOCKE — Spatial Map v0.1.0
=================================

Internal 2D map built from proprioceptive path integration and
sensory observations. The dog knows where it is and where things are.

Architecture:
    The map uses EGOCENTRIC path integration (dead reckoning) with
    periodic landmark corrections. No GPS, no external positioning —
    just IMU integration and sensory observations. Exactly how a
    rat builds its cognitive map.

    Position tracking:
      - Forward velocity from CPG step counting / IMU integration
      - Heading from IMU yaw (gyroscope Z-axis)
      - Position = integral of velocity * heading

    Object memory:
      - Ball: last known position (updated when visible)
      - Wall: position where obstacle was detected
      - Light: direction and estimated position of light source
      - Home: starting position (always at origin)
      - Custom landmarks from sensory events

    The map drifts over time (IMU integration error accumulates).
    This is biologically correct — rats' place cells drift too.
    Landmark corrections reduce drift when recognizable objects
    are re-encountered.

RPi cost: ~0.02ms per step (2 multiplications + 1 addition for position update).
Memory: ~2KB for grid + landmark list.

Biology:
    - Hippocampal place cells: O'Keefe & Nadel 1978
    - Grid cells in entorhinal cortex: Hafting et al. 2005
    - Path integration in desert ants: Wehner & Srinivasan 2003
    - Dead reckoning in rats: McNaughton et al. 2006

Author: MH-FLOCKE Level 15 v0.7.0
"""

__version__ = "0.2.0"
__logbook__ = 98

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

SPATIAL_MAP_VERSION = "v0.1.0"


@dataclass
class Landmark:
    """A remembered object or location in the environment."""
    name: str                          # e.g. 'ball', 'wall_1', 'light_source'
    position: np.ndarray               # [x, y] in world coordinates
    category: str = 'object'           # 'object', 'obstacle', 'goal', 'danger', 'home'
    confidence: float = 1.0            # Decays over time if not re-observed
    last_seen_step: int = 0            # When was this last observed
    last_seen_distance: float = 0.0    # How far was it when last seen
    valence: float = 0.0               # Emotional association: -1 (danger) to +1 (reward)
    visit_count: int = 0               # How often has the dog been here


class SpatialMap:
    """
    2D cognitive map from path integration and sensory observations.

    The dog knows:
      - Where it is (position tracking via IMU)
      - Where it has been (visited cells in grid)
      - Where things are (landmark memory)
      - How to get back (path integration to home)

    Usage:
        spatial = SpatialMap()
        for step in loop:
            spatial.update_position(forward_vel, yaw, dt)
            if ball_visible:
                spatial.observe_landmark('ball', ball_relative_pos)
            goal_direction = spatial.direction_to('ball')
    """

    VERSION = SPATIAL_MAP_VERSION

    def __init__(self, world_size: float = 10.0, grid_resolution: int = 20,
                 position_decay: float = 0.9999):
        """
        Args:
            world_size: Size of the map in meters (world_size × world_size).
            grid_resolution: Number of grid cells per side.
            position_decay: Confidence decay per step for unobserved landmarks.
        """
        self.world_size = world_size
        self.grid_resolution = grid_resolution
        self.position_decay = position_decay

        # Current position and heading (egocentric dead reckoning)
        self.position = np.array([0.0, 0.0], dtype=np.float64)  # [x, y] meters
        self.heading = 0.0      # radians, 0 = forward (+x direction)

        # Visited grid: how many times each cell was visited
        # Grid covers [-world_size/2, +world_size/2] in both axes
        self.visit_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        # Trail: recent positions for path visualization
        self._trail: deque = deque(maxlen=500)
        self._trail_max = 500   # Keep last 500 positions
        self._trail_interval = 20  # Record every 20 steps

        # Landmark memory
        self.landmarks: Dict[str, Landmark] = {}

        # Home position (always known)
        self.landmarks['home'] = Landmark(
            name='home',
            position=np.array([0.0, 0.0]),
            category='home',
            confidence=1.0,
            valence=0.5,  # Home feels safe
        )

        # Path integration state
        self._total_distance = 0.0    # Odometer
        self._step_count = 0
        self._max_excursion = 0.0     # Furthest from home

        # IMU drift compensation
        self._heading_offset = 0.0    # Accumulated heading correction

    def update_position(self, forward_velocity: float, yaw: float,
                        dt: float = 0.005) -> None:
        """Update position from IMU/velocity data. Call every step.

        Args:
            forward_velocity: Forward speed in m/s (from CPG or IMU).
            yaw: Current heading in radians (from IMU quaternion).
                0 = +x direction, positive = counterclockwise.
            dt: Timestep in seconds.
        """
        self._step_count += 1
        self.heading = yaw

        # Dead reckoning: integrate velocity along heading
        dx = forward_velocity * dt * np.cos(yaw)
        dy = forward_velocity * dt * np.sin(yaw)
        self.position[0] += dx
        self.position[1] += dy

        self._total_distance += abs(forward_velocity * dt)

        # Update max excursion
        dist_from_home = float(np.linalg.norm(self.position))
        if dist_from_home > self._max_excursion:
            self._max_excursion = dist_from_home

        # Update visit grid
        gx, gy = self._world_to_grid(self.position[0], self.position[1])
        if 0 <= gx < self.grid_resolution and 0 <= gy < self.grid_resolution:
            self.visit_grid[gy, gx] += 1

        # Record trail
        if self._step_count % self._trail_interval == 0:
            self._trail.append(self.position.copy())
            # deque maxlen handles eviction automatically

        # Decay landmark confidence
        for lm in self.landmarks.values():
            if lm.category != 'home':  # Home never decays
                lm.confidence *= self.position_decay

    def observe_landmark(self, name: str, relative_position: np.ndarray,
                         category: str = 'object', valence: float = 0.0,
                         distance: float = 0.0) -> None:
        """Record or update a landmark observation.

        Args:
            name: Unique name for this landmark (e.g. 'ball', 'wall_north').
            relative_position: [dx, dy] relative to current position.
                Or absolute [x, y] if already in world coordinates.
            category: 'object', 'obstacle', 'goal', 'danger', 'home'.
            valence: Emotional association (-1 to +1).
            distance: Distance to landmark when observed.
        """
        # Convert relative to absolute position
        world_pos = self.position + np.asarray(relative_position, dtype=np.float64)

        if name in self.landmarks:
            # Update existing landmark
            lm = self.landmarks[name]
            # Weighted average: new observation has more weight
            alpha = 0.7  # Trust new observation more
            lm.position = alpha * world_pos + (1 - alpha) * lm.position
            lm.confidence = min(1.0, lm.confidence + 0.3)
            lm.last_seen_step = self._step_count
            lm.last_seen_distance = distance
            lm.visit_count += 1
            if valence != 0:
                lm.valence = 0.7 * valence + 0.3 * lm.valence
        else:
            # New landmark
            self.landmarks[name] = Landmark(
                name=name,
                position=world_pos.copy(),
                category=category,
                confidence=1.0,
                last_seen_step=self._step_count,
                last_seen_distance=distance,
                valence=valence,
            )

    def direction_to(self, landmark_name: str) -> Optional[Tuple[float, float]]:
        """Get direction and distance to a named landmark.

        Returns:
            (angle_radians, distance_meters) or None if landmark unknown.
            angle is relative to current heading:
              0 = straight ahead
              +pi/2 = 90° left
              -pi/2 = 90° right
        """
        if landmark_name not in self.landmarks:
            return None

        lm = self.landmarks[landmark_name]
        delta = lm.position - self.position
        distance = float(np.linalg.norm(delta))

        if distance < 0.01:
            return (0.0, distance)  # We're on top of it

        # Absolute angle to landmark
        abs_angle = float(np.arctan2(delta[1], delta[0]))

        # Relative to current heading
        rel_angle = abs_angle - self.heading
        # Normalize to [-pi, pi]
        while rel_angle > np.pi: rel_angle -= 2 * np.pi
        while rel_angle < -np.pi: rel_angle += 2 * np.pi

        return (rel_angle, distance)

    def direction_to_home(self) -> Tuple[float, float]:
        """Shortcut: direction and distance to home."""
        result = self.direction_to('home')
        return result if result else (0.0, 0.0)

    def direction_to_unexplored(self, radius: float = 2.0,
                                min_gap: float = 1e-3) -> Optional[Tuple[float, float]]:
        """Direction toward the least-visited nearby space (the curiosity vector).

        Task #96 / Weg 2: the intrinsic 'go where it's new' signal. Scans grid
        cells within `radius` meters of the current position and builds a vector
        that points away from well-visited cells and toward unvisited ones. This
        is the DIRECTION of curiosity; how strongly to act on it is scaled by the
        CuriosityExplorer's scalar drive in the loop, not here.

        A wall (or any explored dead-end) is self-avoiding under this signal: cells
        behind a wall never get visited, but cells the robot keeps scrubbing against
        DO accumulate visits, so the vector points away from the stuck spot toward
        open space -- avoidance falls out of exploration, no pain signal needed.

        Returns:
            (rel_angle, novelty) or None if there is no usable gradient.
            rel_angle: relative to current heading, SAME convention as
                       direction_to() -- 0 = straight ahead, +pi/2 = 90 deg left,
                       -pi/2 = 90 deg right.
            novelty:   0..1, how unexplored the chosen direction is (1 = fully
                       unvisited neighbourhood). Lets the caller fade the signal
                       out once the area is known.
        """
        gx0, gy0 = self._world_to_grid(self.position[0], self.position[1])
        cell_size = self.world_size / self.grid_resolution
        reach = max(1, int(radius / cell_size))

        # Weight each nearby cell by how UNvisited it is (1/(1+visits)) and by
        # inverse distance, then sum the unit vectors toward those cells. The result
        # points toward the least-visited, nearest open space.
        #
        # Bias correction: the cells around a position are not perfectly symmetric
        # (the position sits off-centre within its grid cell), so a UNIFORM field
        # still yields a small diagonal vector. We compute that geometric bias with
        # flat weights and subtract it, leaving only the true novelty gradient --
        # a uniform neighbourhood then cancels to zero, no spurious drift.
        vx = 0.0
        vy = 0.0
        wsum = 0.0
        bx = 0.0     # geometric bias accumulator (flat-weighted unit vectors)
        by = 0.0
        bw = 0.0
        for dgy in range(-reach, reach + 1):
            for dgx in range(-reach, reach + 1):
                if dgx == 0 and dgy == 0:
                    continue
                gx, gy = gx0 + dgx, gy0 + dgy
                if not (0 <= gx < self.grid_resolution and 0 <= gy < self.grid_resolution):
                    continue
                cell_dist = float(np.hypot(dgx, dgy))
                if cell_dist > reach:
                    continue
                visits = float(self.visit_grid[gy, gx])
                novelty_w = 1.0 / (1.0 + visits)      # 1 for unvisited, small for well-trodden
                # world-space direction of this cell from current position
                wx, wy = self._grid_to_world(gx, gy)
                ddx = wx - self.position[0]
                ddy = wy - self.position[1]
                dnorm = float(np.hypot(ddx, ddy))
                if dnorm < 1e-9:
                    continue
                ux, uy = ddx / dnorm, ddy / dnorm     # unit vector toward the cell
                dist_w = 1.0 / cell_dist              # nearer cells pull harder
                vx += novelty_w * dist_w * ux
                vy += novelty_w * dist_w * uy
                wsum += novelty_w * dist_w
                bx += dist_w * ux                     # same geometry, flat novelty
                by += dist_w * uy
                bw += dist_w

        if wsum < min_gap or bw < min_gap:
            return None

        # Subtract the geometric bias, scaled to the novelty weighting, so a uniform
        # field (novelty_w constant) cancels exactly.
        mean_nov = wsum / bw
        vx -= mean_nov * bx
        vy -= mean_nov * by

        mag = float(np.hypot(vx, vy))
        if mag < min_gap:
            return None                                 # uniform surroundings: no gradient

        abs_angle = float(np.arctan2(vy, vx))
        rel_angle = abs_angle - self.heading
        while rel_angle > np.pi:
            rel_angle -= 2 * np.pi
        while rel_angle < -np.pi:
            rel_angle += 2 * np.pi

        novelty = float(np.clip(mag / wsum, 0.0, 1.0))  # residual gradient strength, 0 = uniform
        return (rel_angle, novelty)

    def nearest_landmark(self, category: str = None,
                         max_distance: float = None) -> Optional[str]:
        """Find the nearest landmark, optionally filtered by category.

        Args:
            category: Filter by category (e.g. 'obstacle', 'goal').
            max_distance: Maximum search radius in meters.

        Returns:
            Name of nearest landmark, or None.
        """
        best_name = None
        best_dist = float('inf')

        for name, lm in self.landmarks.items():
            if category and lm.category != category:
                continue
            if lm.confidence < 0.1:
                continue  # Forgotten landmark

            dist = float(np.linalg.norm(lm.position - self.position))
            if max_distance and dist > max_distance:
                continue

            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name

    def get_explored_ratio(self) -> float:
        """What fraction of the map has been visited at least once."""
        visited = np.sum(self.visit_grid > 0)
        total = self.grid_resolution * self.grid_resolution
        return float(visited / total)

    def get_nearby_landmarks(self, radius: float = 2.0) -> List[Landmark]:
        """Get all landmarks within radius of current position."""
        nearby = []
        for lm in self.landmarks.values():
            dist = float(np.linalg.norm(lm.position - self.position))
            if dist <= radius and lm.confidence > 0.1:
                nearby.append(lm)
        return nearby

    def get_danger_nearby(self, radius: float = 1.0) -> Optional[Landmark]:
        """Check if any danger landmarks are nearby."""
        for lm in self.landmarks.values():
            if lm.category == 'danger' and lm.confidence > 0.1:
                dist = float(np.linalg.norm(lm.position - self.position))
                if dist <= radius:
                    return lm
        return None

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        half = self.world_size / 2.0
        gx = int((x + half) / self.world_size * self.grid_resolution)
        gy = int((y + half) / self.world_size * self.grid_resolution)
        return (gx, gy)

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        half = self.world_size / 2.0
        cell_size = self.world_size / self.grid_resolution
        x = -half + (gx + 0.5) * cell_size
        y = -half + (gy + 0.5) * cell_size
        return (x, y)

    def stats(self) -> Dict:
        """Compact stats for FLOG logging."""
        home_dir = self.direction_to_home()
        return {
            'spatial_x': float(self.position[0]),
            'spatial_y': float(self.position[1]),
            'spatial_heading': self.heading,
            'spatial_dist_home': home_dir[1] if home_dir else 0.0,
            'spatial_total_dist': self._total_distance,
            'spatial_max_excursion': self._max_excursion,
            'spatial_explored': self.get_explored_ratio(),
            'spatial_landmarks': len([l for l in self.landmarks.values() if l.confidence > 0.1]),
        }

    def get_map_state(self) -> Dict:
        """Full state for serialization / brain persistence."""
        return {
            'position': self.position.tolist(),
            'heading': self.heading,
            'visit_grid': self.visit_grid.tolist(),
            'landmarks': {
                name: {
                    'position': lm.position.tolist(),
                    'category': lm.category,
                    'confidence': lm.confidence,
                    'valence': lm.valence,
                    'visit_count': lm.visit_count,
                    'last_seen_step': lm.last_seen_step,
                }
                for name, lm in self.landmarks.items()
                if lm.confidence > 0.05
            },
            'total_distance': self._total_distance,
            'max_excursion': self._max_excursion,
            'step_count': self._step_count,
        }

    def __repr__(self) -> str:
        pos = self.position
        n_lm = len([l for l in self.landmarks.values() if l.confidence > 0.1])
        expl = self.get_explored_ratio()
        return (f'SpatialMap(pos=({pos[0]:.2f}, {pos[1]:.2f}), '
                f'heading={np.degrees(self.heading):.0f}°, '
                f'{n_lm} landmarks, {expl:.0%} explored)')

    # === Persistence (v4.2) ===

    def state_dict(self) -> dict:
        """Serialize all state for checkpoint/brain persistence."""
        landmarks_data = {}
        for name, lm in self.landmarks.items():
            landmarks_data[name] = {
                'position': lm.position.tolist(),
                'category': lm.category,
                'confidence': lm.confidence,
                'valence': lm.valence,
                'visit_count': lm.visit_count,
                'last_seen_step': lm.last_seen_step,
                'last_seen_distance': lm.last_seen_distance,
            }
        return {
            'version': SPATIAL_MAP_VERSION,
            'world_size': self.world_size,
            'grid_resolution': self.grid_resolution,
            'position': self.position.tolist(),
            'heading': self.heading,
            'visit_grid': self.visit_grid.tolist(),
            'landmarks': landmarks_data,
            'trail': [p.tolist() for p in self._trail],
            'total_distance': self._total_distance,
            'max_excursion': self._max_excursion,
            'step_count': self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint/brain persistence."""
        self.position = np.array(state['position'], dtype=np.float64)
        self.heading = state['heading']
        self._total_distance = state.get('total_distance', 0.0)
        self._max_excursion = state.get('max_excursion', 0.0)
        self._step_count = state.get('step_count', 0)

        # Grid
        grid = state.get('visit_grid')
        if grid is not None:
            loaded = np.array(grid, dtype=np.float32)
            if loaded.shape == self.visit_grid.shape:
                self.visit_grid = loaded

        # Trail
        trail = state.get('trail', [])
        self._trail.clear()
        for p in trail:
            self._trail.append(np.array(p, dtype=np.float64))

        # Landmarks
        self.landmarks.clear()
        for name, lm_data in state.get('landmarks', {}).items():
            self.landmarks[name] = Landmark(
                name=name,
                position=np.array(lm_data['position'], dtype=np.float64),
                category=lm_data.get('category', 'object'),
                confidence=lm_data.get('confidence', 1.0),
                valence=lm_data.get('valence', 0.0),
                visit_count=lm_data.get('visit_count', 0),
                last_seen_step=lm_data.get('last_seen_step', 0),
                last_seen_distance=lm_data.get('last_seen_distance', 0.0),
            )

        n_lm = len(self.landmarks)
        expl = self.get_explored_ratio()
        print(f'  Spatial Map restored: pos=({self.position[0]:.2f}, {self.position[1]:.2f}), '
              f'{n_lm} landmarks, {expl:.0%} explored, {self._total_distance:.1f}m traveled')
