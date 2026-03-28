"""
MH-FLOCKE — Developmental Schedule v0.4.1
=========================================
Motor babbling and developmental stage progression.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DevelopmentalConfig:
    """Configuration for developmental motor learning schedule."""

    # ── Perturbation (neonatal instability) ──
    # Biology: neonatal muscles have lower force-to-noise ratio.
    # Motor units fire irregularly, producing tremor-like perturbations.
    # This provides rich vestibular training signal even on flat terrain.
    #
    # IMPORTANT: Perturbation scales with COMPETENCE, not time.
    # A puppy doesn't stop wobbling after 3 weeks — it stops wobbling
    # because its motor system matures. Maturation IS competence.
    # Ref: Hadders-Algra 2018 — motor development is competence-driven
    perturb_enabled: bool = True
    perturb_force_max: float = 0.3       # Newtons peak force on torso
    perturb_interval: int = 100          # Apply every N steps (not every step)
    perturb_duration: int = 5            # How many steps each perturbation lasts

    # ── Forward model warmup ──
    # Biology: the internal forward model (cerebellar) develops gradually.
    # Neonates rely on feedback, older infants use prediction.
    # Also competence-gated: FM gain rises with motor competence.
    forward_model_warmup_steps: int = 10000  # Fallback time-based ramp if no competence signal


class DevelopmentalSchedule:
    """
    Sensorimotor development schedule for training.

    Provides developmental perturbations and sensor augmentation
    that enable cerebellar calibration from the start of training,
    regardless of terrain type.
    """

    def __init__(self, total_steps: int = 50000,
                 config: DevelopmentalConfig = None):
        self.total_steps = total_steps
        self.config = config or DevelopmentalConfig()

        # Perturbation state
        self._perturb_force = np.zeros(3)    # Current force vector [x, y, z]
        self._perturb_torque = np.zeros(3)   # Current torque vector
        self._perturb_countdown = 0          # Steps remaining for current perturbation
        self._next_perturb_step = 0          # When to apply next perturbation
        self._rng = np.random.RandomState(42)

        # Competence tracking (updated externally via set_competence)
        self._competence = 0.0               # 0.0 = neonate, 1.0 = mature
        self._competence_ema = 0.0           # Smoothed competence

        # Stats
        self.stats = {
            'perturb_magnitude': 0.0,
            'perturb_active': False,
            'developmental_phase': 0.0,    # 0.0 = neonate, 1.0 = mature
            'forward_model_gain': 0.0,
        }

    def set_competence(self, competence: float):
        """
        Update motor competence from external signal.

        Biology: motor maturation is driven by achieved competence,
        not by time. A puppy stops wobbling when its cerebellum is
        calibrated, not when a calendar says so.

        Competence sources (combined in train_v032.py):
          - actor_competence from CompetenceGate (speed-based)
          - upright ratio (stability-based)
          - CPG weight reduction (handoff progress)

        Args:
            competence: 0.0 (neonate) to 1.0 (mature motor system)
        """
        self._competence = max(0.0, min(1.0, competence))
        # Smooth with EMA to prevent jitter
        self._competence_ema = 0.95 * self._competence_ema + 0.05 * self._competence

    def get_developmental_phase(self, step: int) -> float:
        """
        How far through development we are.

        Primarily competence-driven. Falls back to time-based ramp
        only as a minimum floor (ensures some progress even if
        competence signal is stuck at 0).

        0.0 = neonatal (max perturbation, no forward model)
        1.0 = mature (no perturbation, full forward model)
        """
        # Competence-driven (primary)
        competence_phase = self._competence_ema
        # Time-based floor (fallback: at least some progression)
        time_floor = min(1.0, step / max(1, self.total_steps * 0.8))
        # Use whichever is higher
        return max(competence_phase, time_floor * 0.3)

    def get_forward_model_gain(self, step: int) -> float:
        """
        Forward model contribution ramp.

        Biology: infants develop internal models gradually.
        Competence-gated with time-based fallback.
        """
        # Competence: FM develops as motor system matures
        comp_gain = self._competence_ema
        # Time fallback: at least some FM after warmup
        time_gain = min(1.0, step / max(1, self.config.forward_model_warmup_steps))
        return max(comp_gain, time_gain * 0.5)

    def step(self, step: int, world, creature) -> None:
        """
        Apply developmental perturbation forces.

        Called once per training step. Applies small random forces
        to the creature's torso to provide vestibular training signal.

        Biology: this simulates neonatal muscle noise and postural
        instability. The perturbations decay as the animal "matures",
        representing improving motor unit recruitment and muscle
        coordination.

        Args:
            step: current training step
            world: MuJoCoWorld instance
            creature: MuJoCoCreature instance
        """
        if not self.config.perturb_enabled:
            self.stats['perturb_active'] = False
            self.stats['perturb_magnitude'] = 0.0
            return

        phase = self.get_developmental_phase(step)
        self.stats['developmental_phase'] = phase
        self.stats['forward_model_gain'] = self.get_forward_model_gain(step)

        # Past developmental period — no more perturbations
        if phase >= 1.0:
            self.stats['perturb_active'] = False
            self.stats['perturb_magnitude'] = 0.0
            return

        # Decaying force envelope
        force_scale = self.config.perturb_force_max * (1.0 - phase)

        # New perturbation?
        if step >= self._next_perturb_step and self._perturb_countdown <= 0:
            # Random direction (mostly lateral — roll/pitch training)
            direction = self._rng.randn(3)
            direction[2] *= 0.3  # Less vertical, more lateral
            direction /= max(np.linalg.norm(direction), 1e-6)

            self._perturb_force = direction * force_scale
            # Small torque too (rotational perturbation)
            torque_dir = self._rng.randn(3)
            torque_dir /= max(np.linalg.norm(torque_dir), 1e-6)
            self._perturb_torque = torque_dir * force_scale * 0.02

            self._perturb_countdown = self.config.perturb_duration
            self._next_perturb_step = step + self.config.perturb_interval

        # Apply active perturbation
        if self._perturb_countdown > 0:
            try:
                # Apply to torso body (body index 1 = torso for most models)
                body_id = 1
                # xfrc_applied: [nbody, 6] — force(3) + torque(3)
                world._data.xfrc_applied[body_id, :3] = self._perturb_force
                world._data.xfrc_applied[body_id, 3:] = self._perturb_torque
            except (IndexError, AttributeError):
                pass

            self._perturb_countdown -= 1
            self.stats['perturb_active'] = True
            self.stats['perturb_magnitude'] = float(np.linalg.norm(self._perturb_force))
        else:
            # Clear forces when not perturbing
            try:
                world._data.xfrc_applied[1, :] = 0.0
            except (IndexError, AttributeError):
                pass
            self.stats['perturb_active'] = False
            self.stats['perturb_magnitude'] = 0.0

    def get_sensor_augmentation(self, world, creature) -> dict:
        """
        Augment sensor_data with joint positions and motor commands
        for the cerebellar forward model.

        This fixes the pred_error=0 bug: the forward model needs
        joint_positions and motor_commands in sensor_data, which
        were previously never provided.

        Returns:
            dict to merge into sensor_data
        """
        aug = {}
        try:
            n_act = world._model.nu
            # Joint positions (qpos[7:] = joint angles, skip freejoint 7-DoF)
            qpos = world._data.qpos
            if len(qpos) > 7 + n_act:
                aug['joint_positions'] = qpos[7:7 + n_act].copy()
            elif len(qpos) > 7:
                aug['joint_positions'] = qpos[7:].copy()

            # Motor commands (ctrl = what we sent to actuators)
            aug['motor_commands'] = world._data.ctrl[:n_act].copy()
        except (AttributeError, IndexError):
            pass

        # Forward model gain ramp
        aug['forward_model_gain'] = self.stats.get('forward_model_gain', 0.0)

        return aug

    def get_stats(self) -> dict:
        """For FLOG logging."""
        return dict(self.stats)
