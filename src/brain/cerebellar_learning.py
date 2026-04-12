"""
MH-FLOCKE — Cerebellar Learning v0.4.3
========================================
Marr-Albus-Ito cerebellar forward model for motor correction.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from src.brain.multi_compartment import PurkinjeCompartmentLayer, PurkinjeConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CerebellarConfig:
    """Configuration for cerebellar architecture and learning."""

    # --- Population sizes ---
    n_granule: int = 4000          # GrC: expansion layer (biology: 50B)
    n_golgi: int = 200             # GoC: inhibitory interneurons
    n_purkinje: int = 32           # PkC: 2 per actuator (push/pull), auto-adjusted
    n_dcn: int = 32                # DCN: output neurons, same as PkC, auto-adjusted

    # --- Connectivity ---
    mf_per_granule: int = 4        # Each GrC gets exactly 4 MF inputs (biology: 4 dendrites)
    grc_goc_prob: float = 0.05     # GrC→GoC excitatory feedback
    goc_grc_prob: float = 0.05     # GoC→GrC inhibitory (each GoC inhibits ~20 local GrC)
    pf_pkc_prob: float = 0.4       # Parallel fiber → Purkinje (main plastic site)

    # --- Golgi sparseness control ---
    target_grc_sparseness: float = 0.03   # Target: 3% of GrC active (biology: 1-5%)
    golgi_gain: float = 0.5               # How strongly GoC bias maps to threshold (was 1.0, too aggressive)
    golgi_adaptation_rate: float = 0.01   # How fast GoC adapts (was 0.05, caused oscillation with resets)

    # --- Learning ---
    ltd_rate: float = 0.001        # PF→PkC LTD rate (when CF active = error)
    ltp_rate: float = 0.001        # PF→PkC LTP rate (when CF silent = consolidate)
    # Note: LTP balances LTD. In biology, LTP is slower but runs
    # 10x more often (CF fires ~1Hz, silent ~99% of time).
    # Equal rates ensure weights don't collapse.
    cf_threshold: float = 0.05     # Min error magnitude to activate climbing fiber
    eligibility_decay: float = 0.95  # Parallel fiber eligibility trace decay

    # --- Motor mixing ---
    snn_ramp_steps: int = 3000     # Steps before cerebellar corrections reach full strength (was 5000)
    snn_mix_end: float = 0.35      # Max correction magnitude (35% of CPG, was 15%)

    # --- DCN baseline ---
    dcn_tonic: float = 0.5         # DCN baseline activity (PkC inhibits from this)

    # --- Neuromodulation interface ---
    ne_exploration_boost: float = 0.02   # NE increases GrC noise
    da_ltp_modulation: float = 1.0       # DA boosts LTP (reward reinforces good patterns)
    da_ltd_suppression: float = 0.5       # DA suppresses LTD (reward protects from error correction)


# ============================================================================
# INFERIOR OLIVE — Climbing Fiber Error Signal
# ============================================================================

class InferiorOlive:
    """
    Computes climbing fiber error signals from body state AND motor predictions.

    This is the "teaching signal" — equivalent to the inferior olive
    which computes sensory prediction errors and sends them via
    climbing fibers to Purkinje cells.

    v0.3.5: FORWARD MODEL (Efferenzkopie)
    ========================================
    The biological inferior olive doesn't just detect "am I tilting?"
    It compares PREDICTED sensory outcome (from the motor command)
    with ACTUAL sensory outcome (from proprioception).

    Motor command (efference copy) → Forward Model → predicted joint delta
    Proprioception                 → actual joint delta
    Difference = prediction error  → Climbing Fiber

    This is the Wolpert 1998 forward model. Without it, the cerebellum
    cannot distinguish:
    - "I moved correctly but terrain surprised me" (terrain error)
    - "My motor command produced wrong movement" (motor error)

    Both types generate CF signals, but motor prediction errors are
    MORE informative for learning because they directly tell the
    cerebellum HOW its corrections are wrong.

    Ref: Wolpert, Miall & Kawato 1998 — Internal models in the cerebellum
    Ref: Bastian 2006 — Learning to predict the future
    Ref: Ito 2008 — Control and learning by the cerebellum

    Each Purkinje cell gets a CF: positive = error present, 0 = no error.
    The magnitude encodes error severity.
    """

    def __init__(self, n_actuators: int, config: CerebellarConfig):
        self.n_actuators = n_actuators
        self.config = config
        self.n_pkc = config.n_purkinje

        # Forward model state
        self._prev_joint_pos = np.zeros(n_actuators)
        self._prev_motor_cmd = np.zeros(n_actuators)
        self._initialized = False

        # Learned forward model: simple linear predictor
        # predicted_delta[j] = gain[j] * motor_cmd[j]
        # gain adapts over time to match the actual motor-to-movement ratio.
        # Biology: this is what the cerebellum LEARNS — the mapping from
        # motor commands to expected sensory consequences.
        self._forward_gain = np.ones(n_actuators) * 0.02  # initial guess: small
        self._forward_gain_lr = 0.001  # how fast the forward model adapts
        self._accum_actual_delta = np.zeros(n_actuators)
        self._accum_predicted_delta = np.zeros(n_actuators)

        # ── Vestibular Sensor (Issue #68) ──
        # Biology: vestibular nuclei (VN) in the inner ear detect:
        #   - Angular acceleration (semicircular canals)
        #   - Linear acceleration / gravity (otolith organs)
        # VN projects directly to inferior olive and fastigial nucleus.
        # When the head tilts or rotates unexpectedly, vestibular error
        # drives cerebellar correction via climbing fibers.
        # Ref: Angelaki & Cullen 2008 — Vestibular System
        # Ref: Barmack 2003 — Central vestibular system
        self._prev_angular_vel = np.zeros(3)
        self._angular_accel = np.zeros(3)       # semicircular canals
        self._angular_accel_ema = np.zeros(3)   # smoothed (hair cell adaptation)
        self._vest_ema_decay = 0.85             # ~6-7 step integration

        # Stats
        self._prediction_error_mean = 0.0
        self._terrain_error_mean = 0.0
        self._vestibular_error = 0.0

        # --- Navigation CF (Issue #81: Cerebellar VOR Gain Adaptation) ---
        # Biology: The flocculus of the cerebellum calibrates VOR gain.
        # When the VOR produces a steering command, the cerebellum predicts
        # how much the heading should change. If the actual heading change
        # doesn't match → CF fires → PkC learns → gain adjusts.
        # Ref: Boyden et al. 2004, Raymond & Lisberger 1998
        self._prev_heading = 0.0
        self._prev_steering = 0.0
        self._heading_gain = 0.01  # learned: steering → expected heading change
        self._heading_gain_lr = 0.0005
        self._navigation_cf = 0.0  # exposed for logging
        self._steering_gain_correction = 0.0  # DCN output: VOR gain modifier
        self._obstacle_cf = 0.0  # obstacle proximity CF (Issue #103)

    def compute_cf_signal(self, sensor_data: dict) -> np.ndarray:
        """
        Compute climbing fiber activation for each Purkinje cell.

        Biology: Olivary neurons fire at ~1Hz, not every timestep.
        ALL CF signals are pulsed to allow LTP recovery between bursts.
        Without pulsing, continuous terrain errors keep calcium high
        in PkC dendrites → LTP window never opens → weights collapse.

        Returns:
            cf_active: float array [n_purkinje], 0.0 (silent) to 1.0 (max error)
        """
        cf = np.zeros(self.n_pkc)

        step = sensor_data.get('step', 0)

        # --- Pulsing schedule ---
        # Balance errors (roll/pitch/height/yaw/lateral): fire every 50 steps (~4Hz)
        # Velocity error: fire every 100 steps (~2Hz)
        # This gives ~90% CF-silent time for LTP consolidation
        balance_pulse = (step % 50 == 0)
        velocity_pulse = (step % 100 == 0)

        # Forward model accumulation runs EVERY step (Issue #71 fix).
        # Only the CF signal generation is pulsed.
        motor_pred_pulse = (step % 50 == 25)  # offset from balance pulse
        self._accumulate_forward_model(sensor_data, motor_pred_pulse, cf)

        if not balance_pulse and not velocity_pulse:
            return cf  # Silent — allows LTP

        euler = sensor_data.get('orientation_euler', np.zeros(3))
        roll = euler[0]
        pitch = euler[1]
        height = sensor_data.get('height', 0.3)
        lat_vel = sensor_data.get('velocity', np.zeros(3))[1] if 'velocity' in sensor_data else 0.0

        ang_vel = sensor_data.get('angular_velocity', np.zeros(3))
        yaw_rate = ang_vel[2] if len(ang_vel) > 2 else 0.0

        n_legs = 4
        jpleg = self.n_actuators // n_legs

        if balance_pulse:
            # 1. ROLL → left/right leg correction
            roll_err = np.clip(roll / 0.3, -1.0, 1.0)
            if abs(roll_err) > self.config.cf_threshold:
                for leg in range(n_legs):
                    base = leg * jpleg * 2
                    side = -1.0 if leg < 2 else 1.0
                    err = roll_err * side * 0.5
                    if base + 1 < self.n_pkc:
                        cf[base] = max(cf[base], max(0.0, err))
                        cf[base + 1] = max(cf[base + 1], max(0.0, -err))

            # 2. PITCH → front/rear correction
            pitch_err = np.clip(pitch / 0.3, -1.0, 1.0)
            if abs(pitch_err) > self.config.cf_threshold:
                for leg in range(n_legs):
                    front_back = -1.0 if leg % 2 == 0 else 1.0
                    err = pitch_err * front_back * 0.3
                    knee_idx = leg * jpleg * 2 + 2
                    if knee_idx + 1 < self.n_pkc:
                        cf[knee_idx] = max(cf[knee_idx], max(0.0, err))
                        cf[knee_idx + 1] = max(cf[knee_idx + 1], max(0.0, -err))

            # 3. HEIGHT → extend all when dropping
            standing_h = sensor_data.get('standing_height', 0.28)
            h_err = np.clip((standing_h * 0.6 - height) / 0.1, 0.0, 1.0)
            if h_err > self.config.cf_threshold:
                for leg in range(n_legs):
                    for j in range(jpleg):
                        idx = (leg * jpleg + j) * 2
                        if idx < self.n_pkc:
                            cf[idx] = max(cf[idx], h_err * 0.3)

            # 4. YAW → diagonal hip correction
            yaw_err = np.clip(yaw_rate / 1.0, -1.0, 1.0)
            if abs(yaw_err) > self.config.cf_threshold:
                for leg in range(n_legs):
                    base = leg * jpleg * 2
                    diag = 1.0 if leg in [0, 3] else -1.0
                    err = yaw_err * diag * 0.2
                    if base + 1 < self.n_pkc:
                        cf[base] = max(cf[base], max(0.0, err))
                        cf[base + 1] = max(cf[base + 1], max(0.0, -err))

            # 5. LATERAL DRIFT → push opposite
            lat_err = np.clip(lat_vel / 0.5, -1.0, 1.0)
            if abs(lat_err) > self.config.cf_threshold:
                for leg in range(n_legs):
                    base = leg * jpleg * 2
                    side = -1.0 if leg < 2 else 1.0
                    err = lat_err * side * 0.2
                    if base + 1 < self.n_pkc:
                        cf[base] = max(cf[base], max(0.0, err))
                        cf[base + 1] = max(cf[base + 1], max(0.0, -err))

        # 6. VELOCITY ERROR → "you should be moving"
        if velocity_pulse:
            desired_vel = sensor_data.get('desired_velocity', 0.0)
            actual_vel = sensor_data.get('forward_velocity', 0.0)
            if desired_vel > 0:
                vel_err = np.clip((desired_vel - actual_vel) / desired_vel, 0.0, 1.0)
                if vel_err > self.config.cf_threshold:
                    for leg in range(n_legs):
                        for j in range(jpleg):
                            push_idx = (leg * jpleg + j) * 2
                            pull_idx = push_idx + 1
                            if push_idx < self.n_pkc:
                                cf[push_idx] = max(cf[push_idx], vel_err * 0.20)
                            if pull_idx < self.n_pkc:
                                cf[pull_idx] = max(cf[pull_idx], vel_err * 0.05)

        self._terrain_error_mean = float(np.mean(cf))

        # 7. VESTIBULAR ERROR (angular acceleration from semicircular canals)
        # Biology: sudden rotational acceleration = unexpected perturbation.
        # The vestibular nuclei project to the inferior olive which generates
        # CF signals proportional to vestibular error. This is the PRIMARY
        # feedback loop that prevents cerebellar corrections from causing
        # further instability — if a correction makes the animal tilt,
        # the vestibular system immediately signals "wrong direction".
        # Ref: Barmack 2003, Simpson et al. 1996
        if balance_pulse:
            ang_vel_now = sensor_data.get('angular_velocity', np.zeros(3))
            ang_vel_now = np.array(ang_vel_now[:3])

            # Semicircular canals: detect angular ACCELERATION
            self._angular_accel = (ang_vel_now - self._prev_angular_vel)
            self._prev_angular_vel = ang_vel_now.copy()

            # Hair cell adaptation (EMA smoothing)
            self._angular_accel_ema = (self._vest_ema_decay * self._angular_accel_ema
                                       + (1 - self._vest_ema_decay) * self._angular_accel)

            # Roll acceleration → lateral leg correction (same logic as roll error
            # but PREDICTIVE: catches the tilt before it shows in euler angles)
            roll_accel = np.clip(self._angular_accel_ema[0] / 2.0, -1.0, 1.0)
            # Pitch acceleration → front/rear correction
            pitch_accel = np.clip(self._angular_accel_ema[1] / 2.0, -1.0, 1.0)

            vest_err = abs(roll_accel) + abs(pitch_accel)
            self._vestibular_error = float(vest_err)

            if abs(roll_accel) > self.config.cf_threshold:
                for leg in range(n_legs):
                    base = leg * jpleg * 2
                    side = -1.0 if leg < 2 else 1.0
                    err = roll_accel * side * 0.4  # stronger than static roll
                    if base + 1 < self.n_pkc:
                        cf[base] = max(cf[base], max(0.0, err))
                        cf[base + 1] = max(cf[base + 1], max(0.0, -err))

            if abs(pitch_accel) > self.config.cf_threshold:
                for leg in range(n_legs):
                    front_back = -1.0 if leg % 2 == 0 else 1.0
                    err = pitch_accel * front_back * 0.3
                    knee_idx = leg * jpleg * 2 + 2
                    if knee_idx + 1 < self.n_pkc:
                        cf[knee_idx] = max(cf[knee_idx], max(0.0, err))
                        cf[knee_idx + 1] = max(cf[knee_idx + 1], max(0.0, -err))

        # 8. FORWARD MODEL — CF signals injected by _accumulate_forward_model()
        # which runs BEFORE the early return gate (Issue #71 fix).

        # 9. NAVIGATION CF (Issue #81: Cerebellar VOR Gain Adaptation)
        # Biology: Flocculus of cerebellum calibrates VOR gain in real-time.
        # The inferior olive compares expected heading change (from steering
        # command) with actual heading change (from proprioception).
        # CF = predicted_heading_delta - actual_heading_delta
        #   Under-steering (CF > 0): PkC learns → increase gain
        #   Over-steering (CF < 0): PkC learns → decrease gain
        # This runs every step, not pulsed — navigation needs continuous
        # feedback, unlike balance which is sampled at ~4Hz.
        # Ref: Raymond & Lisberger 1998 (VOR gain adaptation in flocculus)
        current_heading = sensor_data.get('ball_heading', 0.0)
        current_steering = sensor_data.get('steering_offset', 0.0)
        if abs(self._prev_steering) > 0.01:  # Only when actively steering
            # Predicted heading change: steering * learned gain
            predicted_delta = self._prev_steering * self._heading_gain
            # Actual heading change
            actual_delta = current_heading - self._prev_heading
            # Navigation prediction error
            nav_pe = predicted_delta - actual_delta
            # Adapt heading gain (forward model for navigation)
            self._heading_gain += self._heading_gain_lr * nav_pe * np.sign(self._prev_steering)
            self._heading_gain = np.clip(self._heading_gain, 0.001, 0.1)
            # CF magnitude (for PkC learning and gain correction)
            self._navigation_cf = float(np.clip(nav_pe, -1.0, 1.0))
            # Steering gain correction: integrate navigation CF
            # Positive CF (under-steering) → increase gain
            # Negative CF (over-steering) → decrease gain
            # EMA to prevent jitter
            self._steering_gain_correction = (
                0.95 * self._steering_gain_correction +
                0.05 * self._navigation_cf * 0.3
            )
            self._steering_gain_correction = np.clip(
                self._steering_gain_correction, -0.5, 0.5
            )
        self._prev_heading = current_heading
        self._prev_steering = current_steering

        # 10. OBSTACLE CF (Issue #103: Ultrasonic obstacle avoidance)
        # Biology: Trigeminal reflex — whisker/face contact triggers
        # immediate motor response via brainstem. The cerebellum receives
        # climbing fiber input from the inferior olive when the animal
        # approaches or contacts an obstacle.
        #
        # CRITICAL: CF must be ASYMMETRIC between push and pull PkC.
        # If push and pull get identical CF → identical calcium →
        # identical DCN inhibition → push-pull difference = 0 → 
        # corrections = 0 ALWAYS.
        #
        # Biology: When an animal hits a wall, the correction is:
        #   - REDUCE forward drive (push PkC get strong CF → LTD)
        #   - INCREASE braking (pull PkC get weak/no CF → LTP)
        # Result: DCN push < DCN pull → negative correction → slow down
        #
        # Three zones:
        #   distance < 0.10m: COLLISION — strong CF on push PkC only
        #   distance < 0.30m: DANGER — graded CF, push > pull  
        #   distance < 0.80m: WARNING — hip yaw CF for turning
        obstacle_dist = sensor_data.get('obstacle_distance', -1.0)
        self._obstacle_cf = 0.0  # for logging

        if obstacle_dist >= 0 and obstacle_dist < 4.0:
            n_legs = 4
            jpleg = self.n_actuators // n_legs

            if obstacle_dist < 0.10:
                # COLLISION: Strong CF on PUSH PkC, weak on PULL
                # This creates asymmetry: push DCN drops, pull DCN stays high
                # → negative correction → all joints retract/brake
                for j in range(min(self.n_actuators, n_legs * jpleg)):
                    push_idx = j * 2
                    pull_idx = push_idx + 1
                    if push_idx < self.n_pkc:
                        cf[push_idx] = max(cf[push_idx], 0.9)  # Strong: suppress push
                    if pull_idx < self.n_pkc:
                        cf[pull_idx] = max(cf[pull_idx], 0.1)  # Weak: preserve pull
                self._obstacle_cf = 0.9
            elif obstacle_dist < 0.30:
                # DANGER: Graded asymmetric CF
                danger = np.clip((0.30 - obstacle_dist) / 0.20, 0.0, 1.0) * 0.7
                for j in range(min(self.n_actuators, n_legs * jpleg)):
                    push_idx = j * 2
                    pull_idx = push_idx + 1
                    if push_idx < self.n_pkc:
                        cf[push_idx] = max(cf[push_idx], danger)        # Push: full CF
                    if pull_idx < self.n_pkc:
                        cf[pull_idx] = max(cf[pull_idx], danger * 0.15)  # Pull: much weaker
                self._obstacle_cf = float(danger)
            elif obstacle_dist < 0.80:
                # WARNING: Hip yaw CF for turning — front legs only
                warn = np.clip((0.80 - obstacle_dist) / 0.50, 0.0, 1.0) * 0.4
                # Front legs (0, 1): hip yaw push CF to turn
                for leg in [0, 1]:  # FL, FR
                    base = leg * jpleg * 2
                    if base + 1 < self.n_pkc:
                        cf[base] = max(cf[base], warn)          # Push: turn
                        cf[base + 1] = max(cf[base + 1], warn * 0.1)  # Pull: preserve
                self._obstacle_cf = float(warn)

        return np.clip(cf, 0.0, 1.0)

    def _accumulate_forward_model(self, sensor_data: dict,
                                   is_pulse: bool, cf: np.ndarray):
        """
        Forward model: Efferenzkopie vs actual joint movement.

        Runs EVERY step to accumulate joint positions and motor commands.
        On pulse steps (step%50==25), computes prediction error and
        adapts forward gain. This fixes Issue #71 where the FM was
        unreachable behind the balance/velocity pulse gate.

        Biology: The inferior olive continuously compares efference copy
        (motor command) with reafference (sensory feedback). The error
        signal is the MOST informative input for cerebellar learning.
        Ref: Wolpert 1998, Miall & Wolpert 1996.
        """
        joint_pos = sensor_data.get('joint_positions', None)
        motor_cmd = sensor_data.get('motor_commands', None)

        if joint_pos is None or motor_cmd is None:
            return

        joint_pos = np.array(joint_pos[:self.n_actuators])
        motor_cmd = np.array(motor_cmd[:self.n_actuators])

        if not self._initialized:
            self._prev_joint_pos = joint_pos.copy()
            self._prev_motor_cmd = motor_cmd.copy()
            self._initialized = True
            return

        if is_pulse:
            # Actual joint movement since last pulse
            actual_delta = joint_pos - self._prev_joint_pos

            # Predicted delta from forward model
            predicted_delta = self._prev_motor_cmd * self._forward_gain

            # Prediction error per joint
            pred_error = actual_delta - predicted_delta

            # Adapt forward model (gradient descent on prediction error)
            gain_update = (self._forward_gain_lr * pred_error
                           * np.sign(self._prev_motor_cmd))
            self._forward_gain += gain_update
            self._forward_gain = np.clip(self._forward_gain, 0.001, 0.5)

            # Convert per-joint prediction error to per-PkC CF signal
            n_legs = 4
            jpleg = self.n_actuators // n_legs
            for j in range(min(self.n_actuators, n_legs * jpleg)):
                err_mag = abs(pred_error[j])
                cf_strength = np.clip(err_mag / 0.05, 0.0, 1.0) * 0.4

                if cf_strength > self.config.cf_threshold:
                    push_idx = j * 2
                    pull_idx = push_idx + 1
                    if pred_error[j] > 0:
                        if pull_idx < self.n_pkc:
                            cf[pull_idx] = max(cf[pull_idx], cf_strength)
                    else:
                        if push_idx < self.n_pkc:
                            cf[push_idx] = max(cf[push_idx], cf_strength)

            self._prediction_error_mean = float(np.mean(np.abs(pred_error)))

        # Always update state for next pulse
        self._prev_joint_pos = joint_pos.copy()
        self._prev_motor_cmd = motor_cmd.copy()

    def get_forward_model_stats(self) -> dict:
        """Stats for logging and debugging."""
        return {
            'prediction_error': self._prediction_error_mean,
            'terrain_error': self._terrain_error_mean,
            'vestibular_error': self._vestibular_error,
            'angular_accel': self._angular_accel_ema.tolist(),
            'forward_gain_mean': float(np.mean(self._forward_gain)),
            'forward_gain_std': float(np.std(self._forward_gain)),
            'navigation_cf': self._navigation_cf,
            'steering_gain_correction': self._steering_gain_correction,
            'heading_gain': self._heading_gain,
            'obstacle_cf': self._obstacle_cf,
        }

    def get_steering_gain_correction(self) -> float:
        """Get cerebellar VOR gain modifier.
        
        Positive = under-steering, increase VOR gain.
        Negative = over-steering, decrease VOR gain.
        Range: -0.5 to +0.5
        
        Usage in training loop:
            cb_mod = cerebellum.inferior_olive.get_steering_gain_correction()
            calibrated_steering = raw_vor_steering * (1.0 + cb_mod)
        """
        return self._steering_gain_correction


# ============================================================================
# CEREBELLAR LEARNING MODULE
# ============================================================================

class CerebellarLearning:
    """
    Cerebellar sensorimotor learning with biologically realistic architecture.

    Manages:
    1. Granule cell sparse coding (Golgi inhibition)
    2. Parallel fiber → Purkinje cell plasticity (LTD/LTP)
    3. Climbing fiber error signals (InferiorOlive)
    4. DCN output (tonic - PkC inhibition)
    5. Motor correction generation

    Population names used (defined by MuJoCoCreatureBuilder):
    - 'mossy_fibers' (= input neurons)
    - 'granule_cells'
    - 'golgi_cells'
    - 'purkinje_cells'
    - 'dcn'
    """

    def __init__(self, snn, n_actuators: int,
                 config: CerebellarConfig = None,
                 device: str = 'cpu'):
        self.snn = snn
        self.n_actuators = n_actuators
        self.config = config or CerebellarConfig()
        self.device = device

        # Auto-adjust PkC/DCN count to match actuators (2 per actuator: push/pull)
        needed = n_actuators * 2
        if self.config.n_purkinje != needed:
            self.config.n_purkinje = needed
            self.config.n_dcn = needed

        self.inferior_olive = InferiorOlive(n_actuators, self.config)

        # Population indices (set by builder)
        self._pop_grc: Optional[torch.Tensor] = None
        self._pop_goc: Optional[torch.Tensor] = None
        self._pop_pkc: Optional[torch.Tensor] = None
        self._pop_dcn: Optional[torch.Tensor] = None
        self._pop_mf: Optional[torch.Tensor] = None

        # PF→PkC weights (managed separately for biological accuracy)
        self._pf_pkc_weights: Optional[torch.Tensor] = None
        self._pf_pkc_mask: Optional[torch.Tensor] = None
        self._pf_eligibility: Optional[torch.Tensor] = None

        # Purkinje Multi-Compartment Layer (Level 14)
        self._pkc_layer: Optional[PurkinjeCompartmentLayer] = None

        # Golgi adaptation
        self._goc_threshold_bias: float = 0.0

        # GrC EMA spike rates
        self._grc_ema: Optional[torch.Tensor] = None
        self._ema_decay: float = 0.92

        # Last DCN output for compute_corrections
        self._last_dcn: Optional[np.ndarray] = None

        # Stats for dashboard
        self.stats = {
            'grc_sparseness': 0.0,
            'cf_magnitude': 0.0,
            'pf_pkc_mean_weight': 0.0,
            'ltd_applied': 0.0,
            'ltp_applied': 0.0,
            'correction_magnitude': 0.0,
            'dcn_activity': 0.0,
        }

        self._step = 0
        self.n_hidden = self.config.n_granule  # compat with llm_bridge

        # Accumulated LTD/LTP for logging (reset each read)
        self._ltd_acc = 0.0
        self._ltp_acc = 0.0

    def set_populations(self, mf_ids, grc_ids, goc_ids, pkc_ids, dcn_ids):
        """Called by builder after defining SNN populations."""
        self._pop_mf = mf_ids
        self._pop_grc = grc_ids
        self._pop_goc = goc_ids
        self._pop_pkc = pkc_ids
        self._pop_dcn = dcn_ids

        n_grc = len(grc_ids)
        n_pkc = len(pkc_ids)

        # PF→PkC: sparse random connectivity, Xavier-like init
        mask = torch.rand(n_grc, n_pkc, device=self.device) < self.config.pf_pkc_prob
        w_init = 0.5 / (self.config.pf_pkc_prob * n_grc) ** 0.5
        weights = torch.rand(n_grc, n_pkc, device=self.device) * w_init + w_init * 0.5
        self._pf_pkc_weights = weights * mask.float()
        self._pf_pkc_mask = mask
        self._pf_eligibility = torch.zeros(n_grc, n_pkc, device=self.device)

        self._grc_ema = torch.zeros(n_grc, device=self.device)

        # Initialize Purkinje multi-compartment layer
        pkc_cfg = PurkinjeConfig(
            n_neurons=n_pkc,
            tau_soma=15.0,
            tau_apical=80.0,   # Slow PF integration
            tau_basal=5.0,     # Fast CF response
            device=self.device,
        )
        self._pkc_layer = PurkinjeCompartmentLayer(pkc_cfg)

    def update(self, creature, sensor_data: dict):
        """
        One cerebellar learning step. Called after SNN.step().

        1. Read GrC spikes, update EMA
        2. Adapt GoC threshold for sparseness
        3. Compute PkC activity via PF→PkC
        4. Compute CF signal from sensor errors
        5. Apply LTD/LTP to PF→PkC weights
        6. Compute DCN output
        """
        if self._pop_grc is None:
            return

        self._step += 1

        # 1. GrC spike rates (accumulated across all substeps)
        # creature._accumulated_spikes sums spikes over 6 substeps
        if hasattr(creature, '_accumulated_spikes') and creature is not None:
            grc_spikes = creature._accumulated_spikes[self._pop_grc]
        else:
            grc_spikes = self.snn.spikes[self._pop_grc].float()
        # Normalize by substep count for rate
        grc_rates = grc_spikes / max(getattr(creature, 'SNN_SUBSTEPS', 6), 1)
        self._grc_ema = self._ema_decay * self._grc_ema + (1 - self._ema_decay) * grc_rates

        # 2. Golgi sparseness control
        # Threshold 0.01: counts GrC that fired recently (within ~10 steps)
        # With substep normalization, a GrC firing 1/6 substeps gives rate ~0.17,
        # EMA with decay 0.92 accumulates to ~0.013 after one step.
        active_frac = (self._grc_ema > 0.01).float().mean().item()
        self.stats['grc_sparseness'] = active_frac

        sp_err = active_frac - self.config.target_grc_sparseness
        # Proportional-Integral controller for sparseness
        # P-term: immediate correction
        # I-term: accumulated bias — BIDIRECTIONAL (was one-way, killed GrC)
        #   If too active (sp_err > 0): raise threshold to suppress
        #   If too quiet (sp_err < 0): lower threshold to reactivate
        self._goc_threshold_bias += sp_err * self.config.golgi_adaptation_rate
        self._goc_threshold_bias = max(-0.5, min(2.0, self._goc_threshold_bias))  # clamp range
        p_term = sp_err * 1.0  # moderate proportional response (was 2.0, too reactive)
        i_term = self._goc_threshold_bias * self.config.golgi_gain
        # Base threshold 0.5 + adaptive component (can go below 0.5 to reactivate GrC)
        new_thr = max(0.1, 0.5 + p_term + i_term)  # floor at 0.1 (never fully disable threshold)
        self.snn._thresholds[self._pop_grc] = new_thr
        self.stats['grc_threshold'] = float(new_thr)

        # 3. PkC via Multi-Compartment (Level 14)
        #    Apical = Parallel Fiber input (weighted GrC rates)
        #    Basal = Climbing Fiber error signal
        pf_rates = self._grc_ema
        pf_to_pkc = pf_rates @ self._pf_pkc_weights  # [n_pkc] — apical input

        # 4. Climbing fiber signal (basal input)
        cf = self.inferior_olive.compute_cf_signal(sensor_data)
        cf_t = torch.tensor(cf, device=self.device, dtype=torch.float32)
        self.stats['cf_magnitude'] = float(cf_t.abs().mean())

        # 5. Step Purkinje compartment layer
        #    apical integrates PF slowly, basal responds to CF fast
        #    calcium spike triggered by CF gates LTD
        self._pkc_layer.step(pf_input=pf_to_pkc, cf_input=cf_t)

        # 6. Eligibility traces (PF activity)
        self._pf_eligibility = (self._pf_eligibility * self.config.eligibility_decay +
                                pf_rates.unsqueeze(1) * self._pf_pkc_mask.float())

        # 7. LTD/LTP — gated by dendritic calcium (not raw CF)
        #    Calcium persists ~12 steps after CF → wider learning window
        #    This is biologically more accurate than binary CF gating
        ltd_gate = self._pkc_layer.get_ltd_gate()  # [n_pkc], 0..2
        cf_silent = (ltd_gate < 0.05).float()       # No recent CF

        # LTD: calcium x eligibility (dendritic computation)
        ltd = self._pf_eligibility * ltd_gate.unsqueeze(0)
        # LTP: no calcium x eligibility (consolidation)
        ltp = self._pf_eligibility * cf_silent.unsqueeze(0)

        # DA modulation (Level 15: reward → DA → LTP↑, LTD↓)
        # Biology: DA from VTA signals reward. High DA:
        #   - Boosts LTP: consolidate patterns that produced reward
        #   - Suppresses LTD: don't erase patterns that worked
        # Low DA (no reward): normal error correction dominates
        da_level = 0.0
        if hasattr(self.snn, 'neuromod_levels'):
            da_level = self.snn.neuromod_levels.get('da', 0.0)

        # LTP boost: da=0 → 1.0x, da=1.0 → 2.0x (reward doubles consolidation)
        ltp_da_mod = 1.0 + da_level * self.config.da_ltp_modulation
        # LTD suppression: da=0 → 1.0x, da=1.0 → 0.5x (reward halves error correction)
        ltd_da_mod = 1.0 - da_level * self.config.da_ltd_suppression
        ltd_da_mod = max(0.1, ltd_da_mod)  # never fully disable LTD

        dw = -self.config.ltd_rate * ltd_da_mod * ltd + self.config.ltp_rate * ltp_da_mod * ltp

        self.stats['da_level'] = da_level
        self.stats['ltp_da_mod'] = ltp_da_mod
        self.stats['ltd_da_mod'] = ltd_da_mod
        self._pf_pkc_weights = (self._pf_pkc_weights + dw).clamp(0.0, 2.0)
        self._pf_pkc_weights *= self._pf_pkc_mask.float()

        self._ltd_acc += float(ltd.sum())
        self._ltp_acc += float(ltp.sum())
        self.stats['ltd_applied'] = self._ltd_acc
        self.stats['ltp_applied'] = self._ltp_acc
        self.stats['pf_pkc_mean_weight'] = float(
            self._pf_pkc_weights.sum() / max(1, self._pf_pkc_mask.sum()))

        # 8. DCN output: tonic - PkC inhibition
        #
        # BUG FIX (Issue #103): The original used only PkC spike activity
        # (self._pkc_layer.activity) which is an EMA of spikes. Since PkC
        # rarely fire spikes (pf_to_pkc is too weak for V_threshold=1.0),
        # activity ≈ 0 for all PkC → DCN = tonic(0.5) for all → 
        # push - pull = 0 → corrections = 0. ALWAYS.
        #
        # The fix: Use the PkC COMPARTMENT activity (V_apical + V_basal)
        # as continuous inhibition, not just binary spikes. This is more
        # biologically accurate: PkC inhibit DCN via graded synaptic
        # release, not just spike-triggered release.
        #
        # Biology: PkC→DCN synapses are GABAergic (inhibitory) and show
        # graded release proportional to dendritic calcium + membrane
        # potential. The DCN fires at its tonic rate minus the PkC
        # inhibition. When CF activates specific PkC → those PkC have
        # high calcium + basal voltage → stronger inhibition → their
        # DCN partner goes below tonic → asymmetric correction.
        #
        # Ref: Gauck & Jaeger 2000 — DCN rebound from PkC inhibition
        pkc_state = self._pkc_layer.get_compartment_state()
        # Graded PkC output: combine apical (PF drive) + calcium (CF drive)
        # Calcium is the key differentiator — it's high only where CF fires
        pkc_graded = (pkc_state['apical'].abs() * 0.3 +
                      pkc_state['calcium'] * 0.5 +
                      self._pkc_layer.activity * 0.2)
        # Normalize to 0..1 range
        pkc_graded = pkc_graded.clamp(0.0, 1.5) / 1.5

        dcn_out = self.config.dcn_tonic - pkc_graded * 0.8
        dcn_out = dcn_out.clamp(-1.0, 1.0)

        self.stats['dcn_activity'] = float(dcn_out.abs().mean())
        self.stats['dcn_push_pull_diff'] = float(
            (dcn_out[0::2] - dcn_out[1::2]).abs().mean()) if len(dcn_out) > 1 else 0.0
        self._last_dcn = dcn_out.detach().cpu().numpy()

        # PkC compartment stats for dashboard (reuse pkc_state from above)
        self.stats['pkc_calcium'] = float(pkc_state['calcium'].mean())
        self.stats['pkc_apical'] = float(pkc_state['apical'].mean())
        self.stats['pkc_complex_spikes'] = int(pkc_state['complex_spike'].sum())

        # Return dict for llm_bridge compatibility
        return {
            'loss': self.stats['cf_magnitude'],
            'grc_sparseness': self.stats['grc_sparseness'],
            'dcn_activity': self.stats['dcn_activity'],
        }

    def compute_corrections(self, snn_controls: list,
                             upright: float = 1.0) -> np.ndarray:
        """
        DCN output → motor corrections for CPG blending,
        OR full motor commands in pure SNN mode (no CPG).

        Vestibular safety gate (Issue #68):
        Correction magnitude is scaled by upright value.
        If the creature is tilting (upright < 1.0), corrections
        are proportionally reduced. This prevents the cerebellum
        from amplifying instability — the biological equivalent
        of the vestibulo-cerebellar inhibition loop where
        vestibular nuclei suppress DCN output during falls.

        Ref: Pompeiano 1998 — Vestibulo-cerebellar interactions
        """
        if self._last_dcn is None:
            return np.zeros(self.n_actuators)

        ramp = min(1.0, self._step / max(1, self.config.snn_ramp_steps))
        mix = ramp * self.config.snn_mix_end

        # Vestibular gate: scale corrections by uprightness
        # Linear gate: upright=1.0 → 100%, upright=0.5 → 50%
        # (was squared: too aggressive on slopes where upright=0.95)
        # upright<0.3 → near-zero (creature is falling, don't interfere)
        vest_gate = max(0.0, min(1.0, upright))
        mix *= vest_gate

        corrections = np.zeros(self.n_actuators)
        dcn = self._last_dcn

        for j in range(self.n_actuators):
            push_idx = j * 2
            pull_idx = j * 2 + 1
            if push_idx < len(dcn) and pull_idx < len(dcn):
                corrections[j] = (dcn[push_idx] - dcn[pull_idx]) * mix
            elif push_idx < len(dcn):
                corrections[j] = dcn[push_idx] * mix

        # Pure SNN mode: full range [-1, 1] (no CPG base to correct)
        # CPG mode: corrections up to ±0.5 (was ±0.3, too restrictive for terrain)
        if self.config.snn_mix_end > 0.4:  # heuristic: >40% = pure mode
            corrections = np.clip(corrections, -1.0, 1.0)
        else:
            corrections = np.clip(corrections, -0.5, 0.5)
        self.stats['correction_magnitude'] = float(np.abs(corrections).mean())
        return corrections

    def get_stats(self) -> dict:
        """Return stats for dashboard/logging. Resets LTD/LTP accumulators."""
        s = dict(self.stats)
        self._ltd_acc = 0.0
        self._ltp_acc = 0.0
        return s

    def reset_episode(self):
        """Reset per-episode state (called after creature falls/resets).
        
        IMPORTANT: We do NOT reset Golgi threshold/bias or GrC EMA.
        These represent learned network state that must persist across episodes.
        Resetting them causes oscillation: EMA=0 → threshold drops → burst →
        threshold spikes → GrC die → repeat every episode.
        
        We only reset:
        - Eligibility traces (episode-specific credit assignment)
        - PkC layer (compartment voltages)
        - Last DCN output (stale motor command)
        """
        if self._pf_eligibility is not None:
            self._pf_eligibility.zero_()
        # Do NOT zero _grc_ema — it's learned state, not episode state.
        # Instead, gently decay toward target sparseness level.
        if self._grc_ema is not None:
            # Soft reset: blend toward a baseline that matches target sparseness
            # This prevents the post-reset oscillation
            target_val = 0.015  # just above EMA threshold of 0.01
            self._grc_ema = self._grc_ema * 0.5 + target_val * 0.5
        if self._pkc_layer is not None:
            self._pkc_layer.reset()
        self._last_dcn = None

    def get_snn_mix(self) -> float:
        """Current SNN correction mixing level (0..snn_mix_end)."""
        ramp = min(1.0, self._step / max(1, self.config.snn_ramp_steps))
        return ramp * self.config.snn_mix_end

    def get_metrics_summary(self, window: int = 500) -> dict:
        """Metrics summary compatible with llm_bridge console logging."""
        return {
            'avg_loss': self.stats.get('cf_magnitude', 0.0),
            'avg_correction': self.stats.get('correction_magnitude', 0.0),
            'snn_mix': self.get_snn_mix(),
            'grc_sparseness': self.stats.get('grc_sparseness', 0.0),
            'pf_pkc_weight': self.stats.get('pf_pkc_mean_weight', 0.0),
            'dcn_activity': self.stats.get('dcn_activity', 0.0),
        }

    def state_dict(self) -> dict:
        """Serialize for checkpoint."""
        sd = {
            'pf_pkc_weights': self._pf_pkc_weights.cpu() if self._pf_pkc_weights is not None else None,
            'pf_pkc_mask': self._pf_pkc_mask.cpu() if self._pf_pkc_mask is not None else None,
            'goc_threshold_bias': self._goc_threshold_bias,
            'step': self._step,
        }
        if self._pkc_layer is not None:
            sd['pkc_layer'] = self._pkc_layer.state_dict()
        return sd

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        if state.get('pf_pkc_weights') is not None:
            self._pf_pkc_weights = state['pf_pkc_weights'].to(self.device)
        if state.get('pf_pkc_mask') is not None:
            self._pf_pkc_mask = state['pf_pkc_mask'].to(self.device)
        self._goc_threshold_bias = state.get('goc_threshold_bias', 0.0)
        self._step = state.get('step', 0)
        if self._pkc_layer is not None and 'pkc_layer' in state:
            self._pkc_layer.load_state_dict(state['pkc_layer'])
