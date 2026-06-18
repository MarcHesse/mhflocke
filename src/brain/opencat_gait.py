"""
MH-FLOCKE — OpenCat Gait Generator for Bittle
===============================================
Replays the hand-optimized OpenCat walk/trot gait frames as a CPG replacement.
Same interface as MogliCPG: compute() returns ctrl deltas.

The OpenCat gaits are proven stable on real hardware and in MuJoCo simulation.
This class cycles through the pre-recorded frames at the correct rate,
producing the exact same motor commands as the real Bittle firmware.
"""

__version__ = "1.0"       # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 27          # mh-logbuch module entry
__status__  = "neu"       # active | veraltet | neu

import math
import numpy as np
from typing import Optional


# Rest angles from OpenCat currentAng[] boot position
_REST_SHOULDER = 75   # indices 8-11
_REST_KNEE = -55      # indices 12-15

# OpenCat frame column order:
# [LF_sh, RF_sh, RR_sh, LR_sh, LF_kn, RF_kn, RR_kn, LR_kn]

# Mapping: (column_index, actuator_index, axis_sign, rest_angle)
# Actuator order: RF_sh(0), RF_kn(1), LF_sh(2), LF_kn(3),
#                 RR_sh(4), RR_kn(5), LR_sh(6), LR_kn(7)
_FRAME_MAP = [
    (1, 0, +1, _REST_SHOULDER),  # col 1 RF_sh -> act 0
    (5, 1, +1, _REST_KNEE),      # col 5 RF_kn -> act 1
    (0, 2, +1, _REST_SHOULDER),  # col 0 LF_sh -> act 2
    (4, 3, +1, _REST_KNEE),      # col 4 LF_kn -> act 3
    (2, 4, -1, _REST_SHOULDER),  # col 2 RR_sh -> act 4 (inverted axis)
    (6, 5, +1, _REST_KNEE),      # col 6 RR_kn -> act 5
    (3, 6, -1, _REST_SHOULDER),  # col 3 LR_sh -> act 6 (inverted axis)
    (7, 7, +1, _REST_KNEE),      # col 7 LR_kn -> act 7
]

# Stand ctrl (same as bittle.py STAND_CTRL)
_STAND_CTRL = np.array([
    -0.7854, +1.4835, -0.7854, +1.4835,
    +0.7854, +1.4835, +0.7854, +1.4835,
])


def _convert_frame(frame: list) -> np.ndarray:
    """Convert one OpenCat frame to 8 MJCF ctrl values."""
    ctrl = np.zeros(8)
    for col_idx, act_idx, axis_sign, rest in _FRAME_MAP:
        ctrl[act_idx] = (frame[col_idx] - rest) * axis_sign * math.pi / 180.0
    return ctrl


def _precompute_gait(raw_frames: list) -> np.ndarray:
    """Convert all frames to MJCF ctrl array (N x 8)."""
    return np.array([_convert_frame(f) for f in raw_frames])


# ============================================================
# Walk gait (wkF) — 116 frames
# ============================================================
_WK_F_RAW = [
    [21,58,61,55,1,8,-7,5],[20,59,60,57,2,8,-7,3],[18,59,57,57,4,9,-8,4],
    [17,60,56,58,6,9,-8,5],[15,60,54,58,10,10,-9,5],[14,60,51,59,12,11,-7,6],
    [13,61,49,59,14,11,-7,6],[15,61,47,60,14,12,-7,7],[15,61,44,60,14,13,-6,7],
    [16,62,42,61,13,13,-6,8],[18,62,40,61,12,14,-5,8],[18,62,36,62,12,15,-4,9],
    [19,63,35,62,11,15,-4,10],[21,63,31,62,10,16,-2,10],[21,63,30,63,10,17,-1,11],
    [23,64,28,63,9,18,0,12],[24,64,26,64,8,19,3,12],[25,64,24,64,8,20,5,13],
    [26,64,24,64,7,21,7,14],[27,64,22,64,7,22,11,15],[28,64,22,65,6,23,12,15],
    [29,63,24,65,6,26,11,16],[30,63,24,65,5,27,11,17],[31,64,25,65,5,26,9,18],
    [32,66,26,65,4,24,8,19],[33,69,27,66,4,20,8,20],[34,70,28,65,3,18,8,21],
    [35,71,29,65,3,14,7,22],[36,71,30,65,3,13,6,24],[37,71,32,66,2,13,6,24],
    [38,71,32,65,2,10,6,26],[39,71,33,65,2,8,6,27],[40,70,35,65,2,6,5,28],
    [41,70,35,66,2,3,5,29],[42,69,36,65,1,2,5,31],[43,68,37,65,1,0,4,32],
    [44,67,38,65,1,-1,4,33],[44,65,39,66,1,-2,3,33],[45,64,40,67,1,-3,3,33],
    [46,64,41,69,1,-6,3,31],[47,62,42,70,1,-6,3,29],[48,60,42,73,2,-7,3,22],
    [49,58,43,73,1,-8,3,21],[49,56,44,75,1,-8,3,18],[49,54,45,75,3,-9,3,18],
    [50,52,46,75,4,-9,3,16],[51,49,47,75,3,-9,3,13],[51,48,48,75,4,-9,3,10],
    [52,45,48,74,4,-8,4,8],[52,43,49,74,5,-9,3,7],[54,41,50,73,5,-9,4,3],
    [54,39,51,73,5,-8,3,2],[55,36,51,72,5,-7,4,1],[55,34,52,71,6,-6,4,-2],
    [56,31,52,68,6,-5,5,-1],[56,28,54,67,6,-4,5,-2],[57,26,54,66,7,-2,5,-4],
    [58,22,55,65,8,0,5,-5],[58,21,57,63,8,1,3,-6],[59,20,57,61,8,2,4,-7],
    [59,18,58,60,9,4,5,-7],[60,17,58,57,9,6,5,-8],[60,15,59,56,10,10,6,-8],
    [60,14,59,54,11,12,6,-9],[61,13,60,51,11,14,7,-7],[61,15,60,49,12,14,7,-7],
    [61,15,61,47,13,14,8,-7],[62,16,61,44,13,13,8,-6],[62,18,62,42,14,12,9,-6],
    [62,18,62,40,15,12,10,-5],[63,19,62,36,15,11,10,-4],[63,21,63,35,16,10,11,-4],
    [63,21,63,31,17,10,12,-2],[64,23,64,30,18,9,12,-1],[64,24,64,28,19,8,13,0],
    [64,25,64,26,20,8,14,3],[64,26,64,24,21,7,15,5],[64,27,65,24,22,7,15,7],
    [64,28,65,22,23,6,16,11],[63,29,65,22,26,6,17,12],[63,30,65,24,27,5,18,11],
    [64,31,65,24,26,5,19,11],[66,32,66,25,24,4,20,9],[69,33,65,26,20,4,21,8],
    [70,34,65,27,18,3,22,8],[71,35,65,28,14,3,24,8],[71,36,66,29,13,3,24,7],
    [71,37,65,30,13,2,26,6],[71,38,65,32,10,2,27,6],[71,39,65,32,8,2,28,6],
    [70,40,66,33,6,2,29,6],[70,41,65,35,3,2,31,5],[69,42,65,35,2,1,32,5],
    [68,43,65,36,0,1,33,5],[67,44,66,37,-1,1,33,4],[65,44,67,38,-2,1,33,4],
    [64,45,69,39,-3,1,31,3],[64,46,70,40,-6,1,29,3],[62,47,73,41,-6,1,22,3],
    [60,48,73,42,-7,2,21,3],[58,49,75,42,-8,1,18,3],[56,49,75,43,-8,1,18,3],
    [54,49,75,44,-9,3,16,3],[52,50,75,45,-9,4,13,3],[49,51,75,46,-9,3,10,3],
    [48,51,74,47,-9,4,8,3],[45,52,74,48,-8,4,7,3],[43,52,73,48,-9,5,3,4],
    [41,54,73,49,-9,5,2,3],[39,54,72,50,-8,5,1,4],[36,55,71,51,-7,5,-2,3],
    [34,55,68,51,-6,6,-1,4],[31,56,67,52,-5,6,-2,4],[28,56,66,52,-4,6,-4,5],
    [26,57,65,54,-2,7,-5,5],[22,58,63,54,0,8,-6,5],
]

# ============================================================
# Trot gait (trF) — 48 frames
# ============================================================
_TR_F_RAW = [
    [31,35,55,61,9,0,11,2],[34,32,57,60,8,0,12,-1],[36,27,58,56,8,3,14,-3],
    [39,21,59,52,7,5,16,-3],[41,17,59,49,9,9,20,-4],[43,11,59,44,9,13,23,-3],
    [45,5,60,39,9,18,26,-3],[47,4,60,38,9,19,29,-2],[49,2,60,36,10,22,32,0],
    [51,1,60,36,11,23,37,0],[52,-1,57,35,14,26,44,2],[54,-2,59,34,12,29,43,3],
    [54,-2,59,33,13,30,43,6],[55,-1,60,32,11,31,41,7],[58,-1,63,34,8,31,36,7],
    [58,2,65,36,5,27,33,6],[58,6,65,39,5,24,32,6],[59,9,67,42,3,21,29,5],
    [58,13,67,44,2,18,28,5],[57,16,69,47,-1,16,20,5],[53,19,68,49,-2,13,16,6],
    [49,21,67,50,-2,14,11,8],[45,25,66,52,-2,12,8,9],[41,28,64,54,-1,10,5,10],
    [35,31,61,55,0,9,2,11],[32,34,60,57,0,8,-1,12],[27,36,56,58,3,8,-3,14],
    [21,39,52,59,5,7,-3,16],[17,41,49,59,9,9,-4,20],[11,43,44,59,13,9,-3,23],
    [5,45,39,60,18,9,-3,26],[4,47,38,60,19,9,-2,29],[2,49,36,60,22,10,0,32],
    [1,51,36,60,23,11,0,37],[-1,52,35,57,26,14,2,44],[-2,54,34,59,29,12,3,43],
    [-2,54,33,59,30,13,6,43],[-1,55,32,60,31,11,7,41],[-1,58,34,63,31,8,7,36],
    [2,58,36,65,27,5,6,33],[6,58,39,65,24,5,6,32],[9,59,42,67,21,3,5,29],
    [13,58,44,67,18,2,5,28],[16,57,47,69,16,-1,5,20],[19,53,49,68,13,-2,6,16],
    [21,49,50,67,14,-2,8,11],[25,45,52,66,12,-2,9,8],[28,41,54,64,10,-1,10,5],
]

# Precompute MJCF ctrl arrays
WALK_CTRL = _precompute_gait(_WK_F_RAW)  # (116, 8)
TROT_CTRL = _precompute_gait(_TR_F_RAW)  # (48, 8)

# Precompute deltas from stand
WALK_DELTA = WALK_CTRL - _STAND_CTRL     # (116, 8)
TROT_DELTA = TROT_CTRL - _STAND_CTRL     # (48, 8)


class OpenCatGait:
    """OpenCat gait frame player with MogliCPG-compatible interface.

    Cycles through precomputed gait frames at the correct rate.
    Output is ctrl delta (same as MogliCPG: add to STAND_CTRL to get ctrl).

    OpenCat gaits run at 50 Hz (20ms per frame). The compute() method
    advances the phase based on elapsed dt and interpolates between frames.
    """

    def __init__(self, gait: str = 'walk'):
        if gait == 'trot':
            self._deltas = TROT_DELTA
            self._ctrls = TROT_CTRL
        else:
            self._deltas = WALK_DELTA
            self._ctrls = WALK_CTRL

        self._n_frames = len(self._deltas)
        self._frame_dt = 0.020  # 50 Hz, OpenCat standard
        self._cycle_time = self._n_frames * self._frame_dt
        self._phase = 0.0  # 0..1 through cycle
        self._freq_scale = 1.0
        self._amp_scale = 1.0
        self._maturation = 0.0
        self._maturation_rate = 1.0 / 500  # full amplitude after 500 steps
        self._step_count = 0
        self._babbling_noise = 0.0  # accept but ignore (no per-leg noise)

        # Stats (MogliCPG compatibility)
        self._stats = {
            'coupling_contra': 0.0,
            'coupling_ipsi': 0.0,
            'coupling_diag': 0.0,
            'drift_estimate': 0.0,
            'vestibular_correction': 0.0,
            'vestibular_cycles': 0,
        }

    def compute(self, dt: float, arousal: float = 0.5,
                freq_scale: float = 1.0, amp_scale: float = 1.0,
                steering: float = 0.0, yaw_rate: float = 0.0,
                **kwargs) -> np.ndarray:
        """Advance gait phase and return 8-element ctrl delta.

        Returns delta from STAND_CTRL (add to STAND_CTRL for actual ctrl).
        Uses linear interpolation between frames for smooth output.
        """
        self._step_count += 1
        self._maturation = min(1.0, self._maturation + self._maturation_rate)
        self._freq_scale = freq_scale
        self._amp_scale = amp_scale

        # Advance phase
        phase_advance = dt / self._cycle_time * freq_scale
        self._phase = (self._phase + phase_advance) % 1.0

        # Frame index and interpolation alpha
        frame_pos = self._phase * self._n_frames
        i0 = int(frame_pos) % self._n_frames
        i1 = (i0 + 1) % self._n_frames
        alpha = frame_pos - int(frame_pos)

        # Interpolate between frames
        delta = self._deltas[i0] * (1.0 - alpha) + self._deltas[i1] * alpha

        # Apply amplitude scaling and maturation
        delta = delta * amp_scale * self._maturation

        # Simple steering: add yaw bias to shoulders
        if abs(steering) > 0.01:
            # Left/right shoulder differential
            delta[0] += steering * 0.1   # RF shoulder
            delta[2] -= steering * 0.1   # LF shoulder
            delta[4] += steering * 0.1   # RR shoulder
            delta[6] -= steering * 0.1   # LR shoulder

        # Track cycles for stats
        self._stats['vestibular_cycles'] = self._step_count * dt / self._cycle_time

        return delta

    def get_stats(self) -> dict:
        """MogliCPG-compatible stats interface."""
        return self._stats.copy()

    def reset_episode(self):
        """Reset phase to start of gait cycle."""
        self._phase = 0.0
        self._step_count = 0
        self._maturation = 0.0

    @property
    def gait_name(self) -> str:
        return 'trot' if len(self._deltas) == 48 else 'walk'

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def cycle_time(self) -> float:
        return self._cycle_time

    # -- Checkpoint compatibility with MogliCPG/SpinalCPG --

    @property
    def _step(self) -> int:
        """Alias for _step_count (checkpoint save/restore)."""
        return self._step_count

    @_step.setter
    def _step(self, value: int):
        self._step_count = value

    @property
    def _phases(self) -> np.ndarray:
        """Return current phase as array (checkpoint save)."""
        return np.array([self._phase * 2 * np.pi])

    @_phases.setter
    def _phases(self, value):
        """Restore phase from array (checkpoint restore)."""
        if hasattr(value, '__len__') and len(value) > 0:
            self._phase = (float(value[0]) / (2 * np.pi)) % 1.0
