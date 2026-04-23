"""
MH-FLOCKE — Body Awareness v0.1.0
====================================

The dog must know its own body. Not as a list of joint angles, but as
a structural understanding: "I have 4 legs. Each leg has 3 joints.
My FR leg doesn't respond to commands anymore."

This module extends the existing BodySchema with limb-level awareness:
  - Groups joints into limbs (legs)
  - Tracks per-limb responsiveness (command → feedback correlation)
  - Detects limb failure automatically (no external flag needed)
  - Reports which limbs are healthy, degraded, or dead

When a limb is detected as dead, the system can:
  1. Disconnect CPG coupling to that limb's oscillator
  2. Notify the metacognition ("I lost a leg")
  3. Trigger drive change ("survival" → "adaptation")

All based on proprioceptive feedback — the dog FEELS that the leg
doesn't respond, just like a real animal.

RPi cost: ~0.05ms per step (4 correlations + 4 comparisons).

Biology:
  - Proprioception via muscle spindles + Golgi tendon organs
  - Deafferentation detection via efference copy mismatch
  - Body schema updating in parietal cortex (Maravita & Iriki 2004)
  - Phantom limb = brain still has leg representation after loss

Ref: Wolpert & Ghahramani 2000 — computational motor control
Ref: Maravita & Iriki 2004 — body schema plasticity

Author: MH-FLOCKE Level 15 v0.7.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BODY_AWARENESS_VERSION = "v0.1.0"


@dataclass
class LimbConfig:
    """Configuration for a single limb."""
    name: str                      # e.g. 'FL', 'FR', 'RL', 'RR'
    joint_indices: List[int]       # indices into joint_positions array
    joint_names: List[str]         # e.g. ['fl_hip_yaw', 'fl_hip_pitch', 'fl_knee']
    side: str = 'left'             # 'left' or 'right'
    position: str = 'front'        # 'front' or 'rear'


@dataclass
class LimbState:
    """Current state of a single limb."""
    name: str
    responsiveness: float = 1.0    # 0.0 = dead, 1.0 = fully responsive
    status: str = 'healthy'        # 'healthy', 'degraded', 'dead'
    command_magnitude: float = 0.0 # How much we're commanding this leg
    feedback_magnitude: float = 0.0 # How much the leg actually moves
    correlation: float = 1.0       # Command-feedback correlation
    steps_dead: int = 0            # How many steps detected as dead


class BodyAwareness:
    """
    Limb-level body awareness through proprioceptive monitoring.

    Detects limb failure by comparing motor commands to sensory feedback.
    A healthy limb: when you send commands, the joints move.
    A dead limb: commands have no effect on joint positions.

    Usage:
        body = BodyAwareness()
        for step in loop:
            body.update(motor_commands, joint_positions)
            for limb in body.get_dead_limbs():
                cpg.disconnect_oscillator(limb)
    """

    VERSION = BODY_AWARENESS_VERSION

    def __init__(self, joints_per_leg: int = 3, n_legs: int = 4,
                 buffer_size: int = 200,
                 dead_threshold: float = 0.15,
                 degraded_threshold: float = 0.5,
                 detection_delay: int = 100):
        """
        Args:
            joints_per_leg: Number of joints per leg (3 for Freenove/Go2).
            n_legs: Number of legs (4 for quadruped).
            buffer_size: Rolling buffer for correlation computation.
            dead_threshold: Responsiveness below this = dead.
            degraded_threshold: Responsiveness below this = degraded.
            detection_delay: Minimum steps before declaring a limb dead.
                Prevents false positives during startup transients.
        """
        self.joints_per_leg = joints_per_leg
        self.n_legs = n_legs
        self.buffer_size = buffer_size
        self.dead_threshold = dead_threshold
        self.degraded_threshold = degraded_threshold
        self.detection_delay = detection_delay

        # Define limb layout
        leg_names = ['FL', 'FR', 'RL', 'RR']
        sides = ['left', 'right', 'left', 'right']
        positions = ['front', 'front', 'rear', 'rear']
        joint_labels = ['yaw', 'pitch', 'knee']

        self.limbs: List[LimbConfig] = []
        for i in range(n_legs):
            indices = list(range(i * joints_per_leg, (i + 1) * joints_per_leg))
            names = [f'{leg_names[i].lower()}_{j}' for j in joint_labels[:joints_per_leg]]
            self.limbs.append(LimbConfig(
                name=leg_names[i],
                joint_indices=indices,
                joint_names=names,
                side=sides[i],
                position=positions[i],
            ))

        # Per-limb state
        self.limb_states: List[LimbState] = [
            LimbState(name=l.name) for l in self.limbs
        ]

        # Rolling buffers: commands per limb + raw joint positions per joint
        self._cmd_buf = np.zeros((buffer_size, n_legs), dtype=np.float64)
        n_total_joints = n_legs * joints_per_leg
        self._joint_pos_buf = np.zeros((buffer_size, n_total_joints), dtype=np.float64)
        self._cmd_per_joint_buf = np.zeros((buffer_size, n_total_joints), dtype=np.float64)
        self._steps_since_reset = 0
        self._buf_idx = 0
        self._buf_filled = 0

        self._step_count = 0
        self._steps_since_reset = 0
        self._prev_joints: Optional[np.ndarray] = None

        # Event log: records when limb state changes
        self._events: List[Dict] = []

    def update(self, motor_commands: np.ndarray,
               joint_positions: np.ndarray) -> None:
        """Update body awareness with current motor commands and joint feedback.

        Call every simulation step.

        Args:
            motor_commands: Array of motor commands sent to actuators.
                Length = n_legs * joints_per_leg.
            joint_positions: Array of actual joint positions from sensors.
                Length = n_legs * joints_per_leg.
        """
        self._step_count += 1
        self._steps_since_reset += 1
        n_j = self.joints_per_leg

        # Ensure numpy arrays
        motor_commands = np.asarray(motor_commands, dtype=np.float64)
        joint_positions = np.asarray(joint_positions, dtype=np.float64)

        n_total = self.n_legs * n_j
        # Store raw joint positions
        jlen = min(len(joint_positions), n_total)
        self._joint_pos_buf[self._buf_idx, :jlen] = joint_positions[:jlen]

        # Store per-limb command magnitude AND per-joint positions
        for i, limb in enumerate(self.limbs):
            idx = limb.joint_indices
            cmd_vals = motor_commands[idx[0]:idx[-1]+1] if idx[-1] < len(motor_commands) else np.zeros(n_j)
            cmd_mag = float(np.sqrt(np.mean(np.asarray(cmd_vals) ** 2)))
            self._cmd_buf[self._buf_idx, i] = cmd_mag
            self.limb_states[i].command_magnitude = cmd_mag

            # Also store the actual command values for correlation
            # (not just magnitude — we need the signal shape)
            for k, j_idx in enumerate(idx):
                if j_idx < len(motor_commands):
                    self._cmd_per_joint_buf[self._buf_idx, j_idx] = motor_commands[j_idx]

        self._buf_idx = (self._buf_idx + 1) % self.buffer_size
        self._buf_filled = min(self._buf_filled + 1, self.buffer_size)

        # Periodic analysis (every buffer_size/2 steps)
        if self._buf_filled >= self.buffer_size // 2 and self._step_count % 100 == 0:
            self._analyze()

    def _analyze(self) -> None:
        """Compute per-limb responsiveness from buffered data.
        
        Responsiveness v4: CORRELATION between command changes and joint changes.
        A healthy leg: when commands change, joints follow proportionally.
        A dead leg: joints move independently of commands (passive swinging).
        
        This distinguishes active movement (servo-driven) from passive
        movement (gravity/inertia), which range-only metrics cannot.
        """
        n = self._buf_filled
        n_j = self.joints_per_leg

        # Get ordered data from ring buffer
        if n >= self.buffer_size:
            positions = np.roll(self._joint_pos_buf, -self._buf_idx, axis=0)
            cmd_joints = np.roll(self._cmd_per_joint_buf, -self._buf_idx, axis=0)
            cmds_limb = np.roll(self._cmd_buf, -self._buf_idx, axis=0)
        else:
            positions = self._joint_pos_buf[:n]
            cmd_joints = self._cmd_per_joint_buf[:n]
            cmds_limb = self._cmd_buf[:n]

        for i, limb in enumerate(self.limbs):
            idx = limb.joint_indices
            cmd_mean = float(np.mean(np.abs(cmds_limb[:, i])))

            if cmd_mean < 0.01:
                continue

            # Per-joint correlation: does the joint position track the command?
            # For position actuators: joint_pos should approach cmd value.
            # Correlation between cmd[t] and pos[t] should be high for healthy legs.
            correlations = []
            for j_idx in idx:
                if j_idx < positions.shape[1] and j_idx < cmd_joints.shape[1]:
                    cmd_signal = cmd_joints[:, j_idx]
                    pos_signal = positions[:, j_idx]
                    # Only correlate if both have variance
                    if np.std(cmd_signal) > 0.001 and np.std(pos_signal) > 0.001:
                        r = float(np.corrcoef(cmd_signal, pos_signal)[0, 1])
                        if not np.isnan(r):
                            correlations.append(abs(r))  # abs: anti-correlation also means responsive

            if not correlations:
                continue

            # Mean absolute correlation across joints
            mean_corr = float(np.mean(correlations))
            self.limb_states[i].feedback_magnitude = mean_corr

            # Responsiveness = correlation strength
            # Healthy: commands and positions strongly correlated (0.7-1.0)
            # Dead: commands and positions uncorrelated (0.0-0.2)
            responsiveness = mean_corr
            corr = mean_corr

            old_status = self.limb_states[i].status
            self.limb_states[i].responsiveness = responsiveness
            self.limb_states[i].correlation = corr

            if responsiveness < self.dead_threshold:
                if self._steps_since_reset > self.detection_delay:
                    self.limb_states[i].status = 'dead'
                    self.limb_states[i].steps_dead += 100
            elif responsiveness < self.degraded_threshold:
                # Only upgrade from dead to degraded if sustained recovery
                if self.limb_states[i].status != 'dead':
                    self.limb_states[i].status = 'degraded'
                self.limb_states[i].steps_dead = 0
            else:
                # Only upgrade from dead to healthy if STRONG sustained signal
                # A dead limb flickers — single high reading shouldn't revive it
                if self.limb_states[i].status == 'dead':
                    # Require 3 consecutive healthy readings to revive
                    self.limb_states[i]._healthy_streak = getattr(
                        self.limb_states[i], '_healthy_streak', 0) + 1
                    if self.limb_states[i]._healthy_streak >= 3:
                        self.limb_states[i].status = 'healthy'
                        self.limb_states[i].steps_dead = 0
                        self.limb_states[i]._healthy_streak = 0
                else:
                    self.limb_states[i].status = 'healthy'
                    self.limb_states[i].steps_dead = 0

            new_status = self.limb_states[i].status
            if new_status != old_status:
                self._events.append({
                    'step': self._step_count,
                    'limb': limb.name,
                    'old_status': old_status,
                    'new_status': new_status,
                    'responsiveness': responsiveness,
                    'correlation': corr,
                })

    def get_dead_limbs(self) -> List[str]:
        """Return names of dead limbs."""
        return [ls.name for ls in self.limb_states if ls.status == 'dead']

    def get_healthy_limbs(self) -> List[str]:
        """Return names of healthy limbs."""
        return [ls.name for ls in self.limb_states if ls.status == 'healthy']

    def get_degraded_limbs(self) -> List[str]:
        """Return names of degraded limbs."""
        return [ls.name for ls in self.limb_states if ls.status == 'degraded']

    def get_limb_state(self, name: str) -> Optional[LimbState]:
        """Get state of a specific limb by name."""
        for ls in self.limb_states:
            if ls.name == name:
                return ls
        return None

    def is_limb_dead(self, name: str) -> bool:
        """Quick check if a specific limb is dead."""
        ls = self.get_limb_state(name)
        return ls is not None and ls.status == 'dead'

    def get_compensation_hint(self) -> Dict[str, float]:
        """Suggest which side needs more amplitude to compensate.

        Returns a dict with 'left_boost' and 'right_boost' values.
        These are SUGGESTIONS based on which limbs are dead —
        the actual learning happens through R-STDP, not hardcoding.

        This is the proprioceptive equivalent of "I feel unbalanced
        to the right, I should lean left."
        """
        dead = self.get_dead_limbs()
        if not dead:
            return {'left_boost': 0.0, 'right_boost': 0.0}

        right_dead = sum(1 for d in dead if d in ('FR', 'RR'))
        left_dead = sum(1 for d in dead if d in ('FL', 'RL'))

        # If right side lost a leg, left side needs to compensate
        # Magnitude is a HINT, not a prescription — the SNN decides
        return {
            'left_boost': right_dead * 0.15,
            'right_boost': left_dead * 0.15,
        }

    def get_active_oscillator_mask(self) -> List[bool]:
        """Return mask of which leg oscillators should be active.

        Dead legs should have their CPG oscillator disconnected
        so it doesn't interfere with the remaining legs' rhythm.
        """
        return [ls.status != 'dead' for ls in self.limb_states]

    def stats(self) -> Dict:
        """Compact stats for FLOG logging."""
        result = {}
        for ls in self.limb_states:
            prefix = f'body_{ls.name}'
            result[f'{prefix}_resp'] = ls.responsiveness
            result[f'{prefix}_status'] = ls.status
            result[f'{prefix}_corr'] = ls.correlation
        result['body_dead_limbs'] = len(self.get_dead_limbs())
        result['body_healthy_limbs'] = len(self.get_healthy_limbs())
        return result

    def get_events(self) -> List[Dict]:
        """Return and clear the event log."""
        events = self._events.copy()
        self._events.clear()
        return events

    def reset_after_physics_reset(self) -> None:
        """Reset all limb states after a physics reset (keyframe/auto-reset).
        
        After a physics reset, all joint positions jump discontinuously.
        The correlation buffer becomes invalid. Without reset, healthy legs
        can appear 'dead' because their pre-reset correlation data doesn't
        match post-reset physics.
        """
        for ls in self.limb_states:
            ls.status = 'healthy'
            ls.responsiveness = 1.0
            ls.correlation = 1.0
            ls.steps_dead = 0
            if hasattr(ls, '_healthy_streak'):
                ls._healthy_streak = 0
        self._buf_idx = 0
        self._buf_filled = 0
        self._cmd_buf[:] = 0
        self._joint_pos_buf[:] = 0
        self._cmd_per_joint_buf[:] = 0
        self._events.clear()
        self._steps_since_reset = 0
        # Detection delay is relative to last reset, not global step count
        self._step_count = self._step_count  # keep global count unchanged

    def __repr__(self) -> str:
        parts = []
        for ls in self.limb_states:
            parts.append(f'{ls.name}:{ls.status}({ls.responsiveness:.2f})')
        return f'BodyAwareness({", ".join(parts)})'
