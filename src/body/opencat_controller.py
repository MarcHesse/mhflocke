"""
OpenCat MuJoCo Controller for Petoi Bittle
============================================
High-level controller that translates OpenCat poses and gaits into MuJoCo
actuator commands. Provides the same interface as the real OpenCat firmware:
pose("sit"), gait("walk"), etc.

Usage:
    from src.body.opencat_controller import OpenCatController

    ctrl = OpenCatController()
    ctrl.set_gait('walk')

    # In simulation loop:
    mujoco_ctrl = ctrl.step(dt=0.005)  # returns 8-element np.array
    data.ctrl[:] = mujoco_ctrl

This module is simulator-agnostic: it returns numpy arrays, no MuJoCo imports.
Can be used with any physics engine that accepts joint angle targets in radians.

Architecture:
    opencat_data.py  — raw pose/gait tables from InstinctBittleESP.h
    opencat_controller.py  — this file: conversion, interpolation, state machine

Conversion chain:
    OpenCat degrees → MJCF radians
    Formula: mjcf_rad = (oc_deg - rest_deg) * axis_sign * pi / 180
    rest_deg: upper leg = 75, lower leg = -55 (from OpenCat currentAng[])
    axis_sign: per-actuator, from MJCF joint axis definitions

Reference: OpenCatEsp32/src/OpenCat.h, InstinctBittleESP.h
"""

__version__ = "1.1"      # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 65         # mh-logbuch module entry
__status__  = "active"    # active | veraltet | neu

import math
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.body.opencat_data import GAITS, POSES


# ============================================================
# Hardware Constants (from OpenCat.h, Bittle LL_LEG config)
# ============================================================

# OpenCat DOF layout (16 total):
#   0-3:  head pan, head tilt, extra, extra
#   4-7:  shoulders (LF, RF, RR, LR)
#   8-11: upper legs (LF, RF, RR, LR)  ← "shoulders" in gait context
#  12-15: lower legs (LF, RF, RR, LR)  ← "knees" in gait context
#
# Gait frames contain only indices 8-15 (8 walking DOF).

# Rest angles from OpenCat currentAng[] (LL_LEG boot position)
REST_UPPER = 75    # degrees, indices 8-11
REST_LOWER = -55   # degrees, indices 12-15

# OpenCat servo rotation directions (indices 8-15)
# Used at servo PWM level, NOT needed for gait conversion
# (already encoded in MJCF joint axes)
ROTATION_DIR = [1, -1, -1, 1, -1, 1, 1, -1]

# OpenCat angle limits (indices 8-15), degrees
ANGLE_LIMITS = [
    (-200, 80), (-200, 80), (-80, 200), (-80, 200),  # upper legs
    (-80, 200), (-80, 200), (-80, 200), (-80, 200),   # lower legs
]

# MJCF actuator order (8 position actuators):
#   0: RF_shoulder  1: RF_knee  2: LF_shoulder  3: LF_knee
#   4: RR_shoulder  5: RR_knee  6: LR_shoulder  7: LR_knee

# Mapping from OpenCat gait column → MJCF actuator
# (col_index, actuator_index, axis_sign, rest_angle_deg)
GAIT_COL_MAP = [
    (0, 2, +1, REST_UPPER),  # col 0: LF upper → act 2 LF_shoulder
    (1, 0, +1, REST_UPPER),  # col 1: RF upper → act 0 RF_shoulder
    (2, 4, -1, REST_UPPER),  # col 2: RR upper → act 4 RR_shoulder
    (3, 6, -1, REST_UPPER),  # col 3: LR upper → act 6 LR_shoulder
    (4, 3, +1, REST_LOWER),  # col 4: LF lower → act 3 LF_knee
    (5, 1, +1, REST_LOWER),  # col 5: RF lower → act 1 RF_knee
    (6, 5, +1, REST_LOWER),  # col 6: RR lower → act 5 RR_knee
    (7, 7, +1, REST_LOWER),  # col 7: LR lower → act 7 LR_knee
]

# Mapping from OpenCat pose column (indices 8-15) → MJCF actuator
# Same as GAIT_COL_MAP (poses store all 16 DOF, we extract 8-15)
POSE_DOF_OFFSET = 8  # pose[8:16] = walking DOF

# MJCF stand position (all actuators at rest)
STAND_CTRL = np.array([
    -0.7854, +1.4835,  # RF: shoulder, knee
    -0.7854, +1.4835,  # LF: shoulder, knee
    +0.7854, +1.4835,  # RR: shoulder, knee
    +0.7854, +1.4835,  # LR: shoulder, knee
])

# OpenCat frame rate: 50 Hz (20 ms per frame)
FRAME_DT = 0.020

# Inverse mapping: MJCF actuator index → (oc_index, axis_sign, rest_deg)
# Derived from GAIT_COL_MAP so sign/rest stay in sync automatically.
# OC walking DOF live at indices 8-15 (8 = LF_upper, ..., 15 = LR_lower).
_MJCF_TO_OC: list = [None] * 8
for _col, _act, _sign, _rest in GAIT_COL_MAP:
    _MJCF_TO_OC[_act] = (8 + _col, _sign, _rest)
del _col, _act, _sign, _rest  # keep module namespace clean


def _mjcf_rad_to_oc_deg(mjcf_rad: float, rest_deg: float,
                         axis_sign: int) -> int:
    """Inverse of _oc_deg_to_mjcf_rad. Returns rounded integer degrees."""
    return round(mjcf_rad * 180.0 / (axis_sign * math.pi) + rest_deg)


def mjcf_ctrl_to_oc_i_cmd(ctrl) -> str:
    """Convert 8-element MJCF ctrl array to an OpenCat 'i' command string.

    The 'i' token sends indexed joint targets simultaneously:
        i <oc_idx> <deg> <oc_idx> <deg> ...

    Angles are clamped to ANGLE_LIMITS before serialising.

    Args:
        ctrl: array-like of 8 floats, MJCF actuator positions in radians.
              Order: RF_sh, RF_kn, LF_sh, LF_kn, RR_sh, RR_kn, LR_sh, LR_kn.

    Returns:
        Command string ready to pass to BittleWS.send_commands(), e.g.
        "i 8 30 9 30 10 30 11 30 12 30 13 30 14 30 15 30".
    """
    parts = ["i"]
    for act_idx, mjcf_rad in enumerate(ctrl):
        oc_idx, axis_sign, rest_deg = _MJCF_TO_OC[act_idx]
        deg = _mjcf_rad_to_oc_deg(float(mjcf_rad), rest_deg, axis_sign)
        lo, hi = ANGLE_LIMITS[oc_idx - 8]
        deg = max(lo, min(hi, deg))
        parts.append(str(oc_idx))
        parts.append(str(deg))
    return " ".join(parts)


class Mode(Enum):
    """Controller operating mode."""
    IDLE = auto()     # holding current position
    POSE = auto()     # transitioning to a static pose
    GAIT = auto()     # cycling through gait frames


def _oc_deg_to_mjcf_rad(oc_degrees: float, rest_deg: float,
                         axis_sign: int) -> float:
    """Convert one OpenCat angle (degrees) to MJCF radians."""
    return (oc_degrees - rest_deg) * axis_sign * math.pi / 180.0


def _convert_gait_frame(frame: list) -> np.ndarray:
    """Convert one 8-element OpenCat gait frame to 8 MJCF ctrl values."""
    ctrl = np.zeros(8)
    for col_idx, act_idx, axis_sign, rest in GAIT_COL_MAP:
        ctrl[act_idx] = _oc_deg_to_mjcf_rad(frame[col_idx], rest, axis_sign)
    return ctrl


def _convert_pose(pose_16: list) -> np.ndarray:
    """Convert a 16-element OpenCat pose to 8 MJCF ctrl values.

    Extracts indices 8-15 (walking DOF) and converts using GAIT_COL_MAP.
    """
    walking_dof = pose_16[POSE_DOF_OFFSET:POSE_DOF_OFFSET + 8]
    return _convert_gait_frame(walking_dof)


def _precompute_gait(name: str) -> np.ndarray:
    """Precompute all frames of a gait as MJCF ctrl arrays (N x 8)."""
    raw = GAITS[name]
    return np.array([_convert_gait_frame(f) for f in raw])


class OpenCatController:
    """High-level OpenCat controller for MuJoCo simulation.

    Provides pose/gait commands with automatic frame interpolation
    and smooth transitions between states.

    Example:
        ctrl = OpenCatController()
        ctrl.set_gait('walk')

        for _ in range(10000):
            mujoco_ctrl = ctrl.step(dt=0.005)
            data.ctrl[:8] = mujoco_ctrl
            mujoco.mj_step(model, data)
    """

    def __init__(self, steering_mode: str = 'offset'):
        """OpenCat gait controller.

        Args:
            steering_mode: How a steering command turns the body.
                'offset'     -- legacy: static shoulder bias.  Measured at ~2 deg/s
                                at full lock, which is below the gait's own 2.84 deg/s
                                drift (knowledge #271).  Effectively inert, but it is
                                what every recorded run used, so it stays the default.
                'gait_blend' -- blend the stride toward OpenCat's own turning table
                                (trF -> trL, mirrored for right).  Measured at
                                7.41 deg/s.  Opt in explicitly.
        """
        if steering_mode not in ('offset', 'gait_blend'):
            raise ValueError(f"steering_mode must be 'offset' or 'gait_blend', got {steering_mode!r}")
        self._steering_mode = steering_mode
        self._mode = Mode.IDLE
        self._current_ctrl = STAND_CTRL.copy()

        # Gait state
        self._gait_name: Optional[str] = None
        self._gait_frames: Optional[np.ndarray] = None
        self._phase = 0.0           # 0..1 through gait cycle
        self._freq_scale = 1.0      # speed multiplier
        self._amp_scale = 1.0       # amplitude multiplier
        self._steering = 0.0        # left/right bias

        # Task #92/#94: the PURE turning component of this frame's output, captured
        # BEFORE amplitude scaling.  In gait_blend the stride is a mix of the straight
        # gait and the turning table; (turn - straight) * mag is the part that carries
        # the yaw.  The training loop damps the whole CPG command (cpg_weight * pd_scale
        # ~= 0.16), which shrinks a 7.41 deg/s turn to ~1.  apply_motor_output can add
        # THIS delta back at (near) full weight so the turn survives while the forward
        # gait stays damped for stability.  0 in offset mode and at zero steering, so
        # nothing changes unless a consumer reads it.  Joint deltas only -> HW-portable.
        self._last_steer_delta = np.zeros_like(STAND_CTRL)

        # Pose state
        self._target_pose: Optional[np.ndarray] = None
        self._pose_blend = 0.0      # 0..1 transition progress
        self._pose_speed = 2.0      # blend per second

        # Maturation (gradual amplitude ramp-up)
        self._maturation = 0.0
        self._maturation_rate = 1.0 / 500  # full amp after 500 steps
        self._step_count = 0

        # Precomputed gait cache
        self._gait_cache: dict[str, np.ndarray] = {}

        # Available gaits and poses (for introspection)
        self.available_gaits = sorted(GAITS.keys())
        self.available_poses = sorted(POSES.keys())

    # ---- Public API ----

    def set_gait(self, name: str, freq_scale: float = 1.0,
                 amp_scale: float = 1.0):
        """Start a gait cycle.

        Args:
            name: Gait name from OpenCat (e.g. 'wkF', 'trF', 'bkF').
                  Shortcuts: 'walk'='wkF', 'trot'='trF', 'back'='bkF',
                  'crawl'='crF', 'gallop'='gpF', 'bound'='bdF'.
            freq_scale: Speed multiplier (1.0 = normal).
            amp_scale: Amplitude multiplier (1.0 = normal).
        """
        name = self._resolve_gait_name(name)
        if name not in GAITS:
            raise ValueError(
                f"Unknown gait '{name}'. Available: {self.available_gaits}")

        if name not in self._gait_cache:
            self._gait_cache[name] = _precompute_gait(name)

        self._gait_name = name
        self._gait_frames = self._gait_cache[name]
        self._freq_scale = freq_scale
        self._amp_scale = amp_scale
        self._mode = Mode.GAIT
        # Keep current phase for smooth transitions between gaits

    def set_pose(self, name: str, speed: float = 2.0):
        """Transition to a static pose.

        Args:
            name: Pose name from OpenCat (e.g. 'sit', 'rest', 'balance').
            speed: Transition speed (blend units per second, 2.0 = 0.5s).
        """
        if name == 'stand':
            self._target_pose = STAND_CTRL.copy()
        elif name in POSES:
            self._target_pose = _convert_pose(POSES[name])
        else:
            raise ValueError(
                f"Unknown pose '{name}'. Available: {self.available_poses}")

        self._pose_blend = 0.0
        self._pose_speed = speed
        self._mode = Mode.POSE

    def set_steering(self, value: float):
        """Set left/right steering bias (-1.0 to 1.0)."""
        self._steering = max(-1.0, min(1.0, value))

    def stop(self):
        """Stop movement, hold current position."""
        self._mode = Mode.IDLE

    def step(self, dt: float) -> np.ndarray:
        """Advance one timestep and return 8-element MJCF ctrl array.

        Args:
            dt: Simulation timestep in seconds.

        Returns:
            np.ndarray of shape (8,) with absolute joint positions in radians.
        """
        self._step_count += 1

        if self._mode == Mode.GAIT:
            self._current_ctrl = self._step_gait(dt)
        elif self._mode == Mode.POSE:
            self._current_ctrl = self._step_pose(dt)
        # Mode.IDLE: keep self._current_ctrl unchanged

        return self._current_ctrl.copy()

    def step_delta(self, dt: float) -> np.ndarray:
        """Advance one timestep and return delta from STAND_CTRL.

        Useful for integration with CPG-based systems that add deltas
        to a standing pose.

        Returns:
            np.ndarray of shape (8,) with joint deltas in radians.
        """
        return self.step(dt) - STAND_CTRL

    def reset(self):
        """Reset to initial state (standing, no gait)."""
        self._mode = Mode.IDLE
        self._current_ctrl = STAND_CTRL.copy()
        self._phase = 0.0
        self._maturation = 0.0
        self._step_count = 0
        self._steering = 0.0

    # ---- Properties ----

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def gait_name(self) -> Optional[str]:
        return self._gait_name

    @property
    def phase(self) -> float:
        """Current gait phase (0.0 to 1.0)."""
        return self._phase

    @property
    def cycle_time(self) -> float:
        """Duration of one gait cycle in seconds."""
        if self._gait_frames is not None:
            return len(self._gait_frames) * FRAME_DT / self._freq_scale
        return 0.0

    @property
    def step_count(self) -> int:
        return self._step_count

    # ---- Checkpoint compatibility (for train_baby.py) ----

    @property
    def _step(self) -> int:
        return self._step_count

    @_step.setter
    def _step(self, value: int):
        self._step_count = value

    @property
    def _phases(self) -> np.ndarray:
        return np.array([self._phase * 2 * np.pi])

    @_phases.setter
    def _phases(self, value):
        if hasattr(value, '__len__') and len(value) > 0:
            self._phase = (float(value[0]) / (2 * np.pi)) % 1.0

    # ---- Stats interface (MogliCPG compatibility) ----

    def get_stats(self) -> dict:
        """Return stats dict compatible with MogliCPG interface."""
        return {
            'gait': self._gait_name or 'none',
            'phase': self._phase,
            'freq_scale': self._freq_scale,
            'amp_scale': self._amp_scale,
            'steering': self._steering,
            'maturation': self._maturation,
            'mode': self._mode.name,
            'drift_estimate': 0.0,
            'vestibular_cycles': (self._step_count * 0.005
                                  / max(self.cycle_time, 0.01)),
        }

    def reset_episode(self):
        """Reset for new episode (keeps gait selection)."""
        self._phase = 0.0
        self._step_count = 0
        self._maturation = 0.0

    # ---- Internal ----

    def _step_gait(self, dt: float) -> np.ndarray:
        """Advance gait phase and interpolate between frames."""
        self._maturation = min(1.0, self._maturation + self._maturation_rate)

        n_frames = len(self._gait_frames)
        cycle_time = n_frames * FRAME_DT

        # Advance phase
        phase_advance = dt / cycle_time * self._freq_scale
        self._phase = (self._phase + phase_advance) % 1.0

        # Frame index and interpolation
        frame_pos = self._phase * n_frames
        i0 = int(frame_pos) % n_frames
        i1 = (i0 + 1) % n_frames
        alpha = frame_pos - int(frame_pos)

        # Interpolate between frames (absolute ctrl values)
        ctrl = (self._gait_frames[i0] * (1.0 - alpha)
                + self._gait_frames[i1] * alpha)

        # --- Steering ----------------------------------------------------------
        # Two mechanisms, selected by _steering_mode.
        #
        # 'offset' (legacy, default): a static bias added to the shoulder joints.
        #   Measured (knowledge #271, isolated MuJoCo trial, no SNN/reflexes):
        #   full lock produces ~2 deg/s of yaw -- LESS than the 2.84 deg/s the trot
        #   drifts on its own with no command at all.  The reason is structural: a
        #   trot turns by taking LONGER STRIDES on the outside of the curve, and an
        #   offset to the rest pose leaves stride length symmetric.  The robot walks
        #   straight no matter how large the offset gets.  Kept as the default so
        #   every existing run stays bit-identical.
        #
        # 'gait_blend': blend the whole stride toward OpenCat's own turning table.
        #   OpenCat ships a turning variant of every gait (trF->trL, wkF->wkL, ...)
        #   in which the entire step pattern differs, not just a joint offset.
        #   Measured: trL turns at 7.41 deg/s -- 3-4x the offset's full lock, same
        #   body, same physics, same ground.  The body was never the problem; the
        #   controller was asking for the turn the wrong way.
        #
        #   Sign convention follows the legacy offset (verified in the same trial):
        #   positive steering = positive yaw = LEFT.  trL is the left turn, so it is
        #   used directly; a right turn mirrors it across the body axis.
        #
        #   HW note: these tables come from the OpenCat firmware and run on the real
        #   Bittle.  A hand-rolled amplitude modulation would not.
        if self._steering_mode == 'gait_blend':
            self._last_steer_delta = np.zeros_like(STAND_CTRL)
            mag = min(1.0, abs(self._steering))
            if mag > 0.01:
                turn_frames = self._get_turn_frames()
                if turn_frames is not None:
                    t_ctrl = self._sample(turn_frames, self._phase)
                    if self._steering < 0.0:                  # right turn: mirror the left one
                        t_ctrl = self._mirror(t_ctrl)
                    # Capture the pure turning component BEFORE the mix + before the
                    # amplitude scaling below, so a consumer (apply_motor_output) can
                    # re-apply it undamped.  This IS the delta the blend adds to the
                    # straight stride: (turn - straight) * mag.
                    self._last_steer_delta = (t_ctrl - ctrl) * mag
                    ctrl = ctrl * (1.0 - mag) + t_ctrl * mag

        # Apply amplitude scaling around STAND_CTRL
        if self._amp_scale != 1.0 or self._maturation < 1.0:
            delta = ctrl - STAND_CTRL
            scale = self._amp_scale * self._maturation
            ctrl = STAND_CTRL + delta * scale

        # Legacy steering: differential shoulder bias.  Inert (see above), but it is
        # what every recorded run was produced with, so it stays the default.
        if self._steering_mode == 'offset' and abs(self._steering) > 0.01:
            s = self._steering * 0.1
            ctrl[0] += s   # RF shoulder
            ctrl[2] -= s   # LF shoulder
            ctrl[4] += s   # RR shoulder
            ctrl[6] -= s   # LR shoulder

        return ctrl

    # ---- Steering via turning gaits (mode 'gait_blend') ----

    @staticmethod
    def _sample(frames: np.ndarray, phase: float) -> np.ndarray:
        """Interpolate a gait table at a normalised phase (0..1).

        Phase rather than frame index, because the straight and turning tables do
        not necessarily hold the same number of frames.  Sampling both at the same
        phase keeps them in step whatever their lengths.
        """
        n = len(frames)
        pos = (phase % 1.0) * n
        i0 = int(pos) % n
        i1 = (i0 + 1) % n
        a = pos - int(pos)
        return frames[i0] * (1.0 - a) + frames[i1] * a

    @staticmethod
    def _mirror(ctrl: np.ndarray) -> np.ndarray:
        """Mirror a control vector across the body's long axis.

        Layout is [RF_sh, RF_kn, LF_sh, LF_kn, RR_sh, RR_kn, LR_sh, LR_kn], so a
        left/right mirror swaps the front pair and the rear pair.  The shoulder
        angles of a left and right leg carry the same sign in STAND_CTRL, so the
        swap alone is the mirror -- no negation.

        OpenCat only ships the LEFT turn of each gait; the right turn is its mirror.
        """
        m = ctrl.copy()
        m[0:2], m[2:4] = ctrl[2:4].copy(), ctrl[0:2].copy()   # front: RF <-> LF
        m[4:6], m[6:8] = ctrl[6:8].copy(), ctrl[4:6].copy()   # rear:  RR <-> LR
        return m

    def _get_turn_frames(self) -> Optional[np.ndarray]:
        """The turning table matching the current gait ('trF' -> 'trL'), cached.

        Returns None when the current gait has no turning variant, in which case
        steering silently does nothing -- better than turning with the wrong gait.
        """
        if self._gait_name is None:
            return None
        turn_name = self._gait_name[:-1] + 'L' if self._gait_name.endswith('F') else None
        if turn_name is None or turn_name not in GAITS:
            return None
        if turn_name not in self._gait_cache:
            self._gait_cache[turn_name] = _precompute_gait(turn_name)
        return self._gait_cache[turn_name]

    def _step_pose(self, dt: float) -> np.ndarray:
        """Blend toward target pose."""
        self._pose_blend = min(1.0, self._pose_blend + self._pose_speed * dt)

        ctrl = (self._current_ctrl * (1.0 - self._pose_blend)
                + self._target_pose * self._pose_blend)

        if self._pose_blend >= 1.0:
            self._mode = Mode.IDLE  # transition complete

        return ctrl

    @staticmethod
    def _resolve_gait_name(name: str) -> str:
        """Map friendly names to OpenCat gait codes."""
        aliases = {
            'walk': 'wkF', 'walk_forward': 'wkF', 'walk_left': 'wkL',
            'trot': 'trF', 'trot_forward': 'trF', 'trot_left': 'trL',
            'back': 'bkF', 'backward': 'bkF', 'back_left': 'bkL',
            'crawl': 'crF', 'crawl_forward': 'crF', 'crawl_left': 'crL',
            'gallop': 'gpF', 'gallop_forward': 'gpF', 'gallop_left': 'gpL',
            'bound': 'bdF',
            'halloween': 'hlw',
            'jump': 'jpF',
            'step': 'vtF', 'step_forward': 'vtF', 'step_left': 'vtL',
        }
        return aliases.get(name, name)

    # ---- MogliCPG-compatible compute() interface ----

    def compute(self, dt: float, arousal: float = 0.5,
                freq_scale: float = 1.0, amp_scale: float = 1.0,
                steering: float = 0.0, yaw_rate: float = 0.0,
                **kwargs) -> np.ndarray:
        """MogliCPG-compatible interface: returns delta from STAND_CTRL.

        This allows drop-in replacement in train_baby.py's CPG dispatch.
        """
        self._freq_scale = freq_scale
        self._amp_scale = amp_scale
        self._steering = steering

        if self._mode != Mode.GAIT and self._gait_frames is not None:
            self._mode = Mode.GAIT

        return self.step_delta(dt)
