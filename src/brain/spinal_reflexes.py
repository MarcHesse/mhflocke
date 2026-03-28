"""
MH-FLOCKE — Spinal Reflexes v0.4.1
========================================
Postural reflexes, stretch reflexes, and crossed extension coordination.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════════════
# SPINAL SEGMENT — per-joint reflexes (always active)
# ═══════════════════════════════════════════════════════════

@dataclass
class SpinalSegmentConfig:
    """
    Configuration for spinal segment reflexes.
    These are the lowest-level motor control — monosynaptic arcs
    that operate per joint, independent of posture or brain state.

    Biology:
    - Muscle tone: gamma motor neuron loop maintains baseline tension.
      Without tone, a limb is flaccid (like under anaesthesia).
      Ref: Granit 1970 — The Basis of Motor Control

    - Stretch reflex (myotatic): muscle spindle detects lengthening
      → Ia afferent → alpha motor neuron → contraction.
      Latency ~30ms in humans, monosynaptic. This is what keeps
      you standing without thinking about it.
      Ref: Liddell & Sherrington 1924

    - Golgi tendon reflex (inverse myotatic): tendon organ detects
      excessive force → Ib afferent → inhibitory interneuron → relaxation.
      Protects muscles/tendons from tearing. Also regulates force output.
      Ref: Houk & Henneman 1967
    """

    # ── Muscle Tone ──
    # Baseline activation that keeps joints in a neutral, slightly flexed
    # resting position. A quadruped at rest has ~15-25% of max extensor
    # activation just from tone. This prevents collapse when the brain
    # is inactive or between motor commands.
    tone_enabled: bool = True
    tone_gain: float = 0.2          # baseline fraction of max activation

    # Per joint-type tone (extensors need more tone than flexors because gravity)
    # For a 4-joint leg: [abduction, hip, knee, ankle]
    # Hip extensors: hold leg under body
    # Knee extensors: prevent collapse
    # Ankle: mild plantarflexion
    tone_profile_4j: tuple = (0.0, -0.15, 0.25, -0.10)  # abd, hip, knee, ankle
    tone_profile_3j: tuple = (-0.15, 0.25, -0.10)        # hip, knee, ankle

    # ── Stretch Reflex ──
    # Resists perturbations proportional to velocity of joint displacement.
    # Fast stretch → strong contraction (phasic component)
    # Sustained stretch → mild contraction (tonic component)
    # This is the primary mechanism for postural stability.
    stretch_enabled: bool = True
    stretch_gain: float = 0.05      # proportional to joint velocity
    # NOTE: was 0.4 which caused stretch reflex to REVERSE CPG outputs.
    # At dt=0.005, joint_vel = delta/dt amplifies by 200x.
    # A 0.01rad movement → vel 2.0 → reflex 0.8 (was > CPG ~0.3).
    # Biological stretch reflex stabilizes but never overpowers
    # voluntary motor commands. 0.05 gives ~0.10 reflex for
    # typical movements, which damps without blocking.
    stretch_damping: float = 0.8    # velocity smoothing (EMA alpha)

    # ── Golgi Tendon Organ ──
    # Force limiter: when actuator command exceeds threshold,
    # inhibit proportionally. Prevents jerky over-corrections
    # and protects joints from excessive torque.
    golgi_enabled: bool = True
    golgi_threshold: float = 0.85   # command magnitude above which inhibition starts
    # NOTE: was 0.75 which clipped 10/12 joints constantly.
    # CPG(0.3) + tone(0.05) already near threshold.
    # 0.85 allows normal movement, only clips real spikes.
    golgi_gain: float = 0.5         # inhibition strength


class SpinalSegments:
    """
    Per-joint spinal reflexes — always active, no state machine needed.

    These run AFTER the reflex chains and motor commands are computed,
    and MODIFY the final motor output. They are the last layer before
    the actuators.

    Processing order:
      1. Muscle tone (baseline)
      2. Stretch reflex (resist perturbation)
      3. Golgi tendon (limit force)

    The output is ADDED to existing motor commands for tone,
    and the stretch reflex modifies commands in-place.
    Golgi acts as a multiplicative limiter.
    """

    def __init__(self, n_actuators: int, config: SpinalSegmentConfig = None):
        self.n_actuators = n_actuators
        self.config = config or SpinalSegmentConfig()
        self.n_legs = 4
        self.jpleg = n_actuators // self.n_legs

        # Joint velocity tracking for stretch reflex
        self._prev_joint_pos = np.zeros(n_actuators)
        self._joint_vel_ema = np.zeros(n_actuators)  # smoothed velocity
        self._initialized = False

        # Build tone profile for all actuators
        profile = (self.config.tone_profile_4j if self.jpleg == 4
                   else self.config.tone_profile_3j)
        self._tone_baseline = np.tile(profile, self.n_legs)

        # Stats
        self.stats = {
            'tone_magnitude': 0.0,
            'stretch_magnitude': 0.0,
            'golgi_clipped': 0,
        }

    def compute_tone(self) -> np.ndarray:
        """
        Muscle tone: constant baseline activation.

        Biology: gamma motor neurons maintain spindle tension,
        which through the stretch reflex loop maintains alpha
        motor neuron firing. The result is a steady low-level
        contraction that keeps joints in a neutral position.

        A sleeping dog doesn't collapse into a heap — tone holds
        its posture. An anaesthetized dog does collapse.
        """
        if not self.config.tone_enabled:
            return np.zeros(self.n_actuators)

        tone = self._tone_baseline * self.config.tone_gain
        self.stats['tone_magnitude'] = float(np.abs(tone).mean())
        return tone

    def compute_stretch_reflex(self, joint_positions: np.ndarray,
                                sim_dt: float = 0.005) -> np.ndarray:
        """
        Stretch reflex: resist rapid joint displacement.

        Biology: muscle spindle Ia afferents detect velocity of
        muscle stretch. Monosynaptic connection to alpha motor
        neurons causes immediate contraction opposing the stretch.

        This is why you don't fall when someone bumps you —
        before you're even aware of it, your stretch reflexes
        have already compensated.

        The reflex acts like a damper/spring on each joint:
        - Fast perturbation → strong resistance (phasic)
        - Slow movement → mild resistance (allows voluntary movement)

        Ref: Liddell & Sherrington 1924
        Ref: Matthews 1972 — Mammalian Muscle Receptors
        """
        if not self.config.stretch_enabled:
            return np.zeros(self.n_actuators)

        if not self._initialized:
            self._prev_joint_pos = joint_positions.copy()
            self._initialized = True
            return np.zeros(self.n_actuators)

        # Joint velocity (finite difference)
        joint_vel = (joint_positions - self._prev_joint_pos) / sim_dt
        self._prev_joint_pos = joint_positions.copy()

        # EMA smoothing (simulates spindle adaptation)
        alpha = 1.0 - self.config.stretch_damping
        self._joint_vel_ema = (self.config.stretch_damping * self._joint_vel_ema
                               + alpha * joint_vel)

        # Reflex: oppose the velocity
        # Negative sign: if joint moves positive, reflex pushes negative
        reflex = -self._joint_vel_ema * self.config.stretch_gain

        self.stats['stretch_magnitude'] = float(np.abs(reflex).mean())
        return reflex

    def apply_golgi(self, motor_commands: np.ndarray) -> np.ndarray:
        """
        Golgi tendon organ: limit excessive force.

        Biology: Golgi tendon organs (GTOs) sit at the muscle-tendon
        junction and measure FORCE (not length like spindles).
        When force exceeds a threshold, Ib afferents activate
        inhibitory interneurons that reduce alpha motor neuron firing.

        This prevents: tendon rupture, joint damage, jerky movements.
        It also smooths motor output — biological movement is smooth
        partly because GTOs prevent sudden force spikes.

        Ref: Houk & Henneman 1967
        Ref: Jami 1992 — Golgi tendon organs in mammalian muscle
        """
        if not self.config.golgi_enabled:
            return motor_commands

        result = motor_commands.copy()
        threshold = self.config.golgi_threshold
        clipped = 0

        for i in range(len(result)):
            magnitude = abs(result[i])
            if magnitude > threshold:
                # Proportional inhibition above threshold
                excess = magnitude - threshold
                inhibition = excess * self.config.golgi_gain
                # Reduce command toward threshold
                sign = np.sign(result[i])
                result[i] = sign * max(threshold, magnitude - inhibition)
                clipped += 1

        self.stats['golgi_clipped'] = clipped
        return result

    def process(self, motor_commands: np.ndarray,
                joint_positions: np.ndarray,
                sim_dt: float = 0.005) -> np.ndarray:
        """
        Full spinal segment processing pipeline.

        Called AFTER all higher-level motor commands (CPG, actor,
        reflex chains) have been combined. This is the final
        biological filter before signals reach the muscles.

        Order matters:
          1. Add muscle tone (ensures baseline activation)
          2. Add stretch reflex (resist perturbations)
          3. Apply Golgi (limit total force)
          4. Clip to actuator range
        """
        # 1. Muscle tone
        output = motor_commands + self.compute_tone()

        # 2. Stretch reflex
        output = output + self.compute_stretch_reflex(joint_positions, sim_dt)

        # 3. Golgi tendon organ
        output = self.apply_golgi(output)

        # 4. Final clip
        output = np.clip(output, -1.0, 1.0)

        return output

    def get_stats(self) -> dict:
        return dict(self.stats)


# ═══════════════════════════════════════════════════════════
# POSTURE STATE — where is the creature right now?
# ═══════════════════════════════════════════════════════════

class PostureState(Enum):
    """
    Developmental posture states. Each state has specific reflexes
    that are active, and specific transition conditions to the next state.

    Biology: Magnus 1924 described these as "Stellreflexe" (righting reflexes)
    that form a chain from supine → prone → sitting → standing.
    """
    SUPINE = 'supine'           # On back (upright < 0)
    ROLLING = 'rolling'         # Transitioning from back to belly
    PRONE = 'prone'             # On belly (0 < upright < 0.35)
    PUSHING_UP = 'pushing_up'   # Lifting torso off ground
    CROUCHING = 'crouching'     # Limbs under body, not yet standing (0.35 < upright < 0.7)
    RISING = 'rising'           # Extending legs to stand
    STANDING = 'standing'       # Upright and stable (upright > 0.85)
    WALKING = 'walking'         # Standing + forward velocity
    DESTABILIZED = 'destabilized'  # Standing but losing balance (0.7 < upright < 0.85)


@dataclass
class ReflexConfig:
    """Configuration for primitive reflexes."""

    # Reflex gains per posture phase
    # Each gain is strongest in the phase where it's needed
    tlr_gain: float = 1.0
    stnr_gain: float = 0.7
    righting_gain: float = 0.9
    extensor_gain: float = 0.8
    crossed_ext_gain: float = 0.4
    kick_freq: float = 2.5
    kick_amplitude: float = 0.7
    kick_onset_steps: int = 30      # faster onset than before (was 50)

    # Posture thresholds
    supine_threshold: float = 0.0       # upright < 0 = on back
    prone_threshold: float = 0.35       # upright 0..0.35 = on belly
    crouch_threshold: float = 0.7       # upright 0.35..0.7 = crouching
    standing_threshold: float = 0.85    # upright > 0.85 = standing
    walking_speed: float = 0.02         # m/s to count as walking

    # Transition timing
    min_phase_steps: int = 20           # minimum steps in a phase before transition
    rolling_duration: int = 80          # steps to complete a roll

    # Reflex fade (cerebellum takes over)
    fade_start: int = 50000
    fade_end: int = 500000
    min_gain: float = 0.1


class SpinalReflexes:
    """
    Reflex chain sequencer — spinal cord level motor control.

    Instead of firing all reflexes simultaneously, this module
    tracks the creature's posture state and activates only the
    reflexes appropriate for the CURRENT phase of recovery/locomotion.

    The chain:
      SUPINE → [TLR + kick] → ROLLING → [asymmetric push] → PRONE
      PRONE → [Landau + extensor] → PUSHING_UP → [push harder] → CROUCHING
      CROUCHING → [righting] → RISING → [extend] → STANDING
      STANDING → [crossed extension] → WALKING → [minimal reflexes]
      DESTABILIZED → [righting + extensor] → STANDING or → PRONE (if fail)
    """

    def __init__(self, n_actuators: int, config: ReflexConfig = None):
        self.n_actuators = n_actuators
        self.config = config or ReflexConfig()
        self.n_legs = 4
        self.jpleg = n_actuators // self.n_legs

        # State machine
        self.posture = PostureState.STANDING  # assume starting upright
        self._phase_steps = 0                 # steps in current phase
        self._total_steps = 0
        self._fallen_steps = 0

        # Per-leg phase offsets for diagonal alternation
        self._leg_phases = [0.0, np.pi, np.pi, 0.0]

        # Rolling state
        self._roll_direction = 1.0  # +1 or -1, chosen when entering ROLLING

        # Stats
        self.stats = {
            'reflex_magnitude': 0.0,
            'active_reflexes': '',
            'gain_multiplier': 1.0,
            'posture_state': 'standing',
        }

    def _fade_multiplier(self) -> float:
        """Primitive reflexes fade as cerebellum matures."""
        if self._total_steps < self.config.fade_start:
            return 1.0
        if self._total_steps > self.config.fade_end:
            return self.config.min_gain
        progress = (self._total_steps - self.config.fade_start) / (
            self.config.fade_end - self.config.fade_start)
        return 1.0 - progress * (1.0 - self.config.min_gain)

    # ═══════════════════════════════════════════════════════
    # POSTURE DETECTION — classify body state from sensors
    # ═══════════════════════════════════════════════════════

    def _classify_posture(self, sensor_data: dict, is_fallen: bool) -> PostureState:
        """
        Determine posture from sensor data. This is proprioception —
        the creature knows its own body orientation.

        Biology: vestibular system (inner ear) + proprioceptors (joints/muscles)
        combine to give continuous posture awareness. Even newborns have this.
        """
        upright = sensor_data.get('upright', 1.0)
        speed = abs(sensor_data.get('forward_velocity', 0.0))
        height = sensor_data.get('height', 0.35)

        # On back
        if upright < self.config.supine_threshold:
            if self.posture == PostureState.ROLLING and \
               self._phase_steps < self.config.rolling_duration:
                return PostureState.ROLLING  # still rolling, don't interrupt
            return PostureState.SUPINE

        # On belly
        if upright < self.config.prone_threshold:
            if self.posture == PostureState.ROLLING:
                # Roll succeeded! Now prone.
                return PostureState.PRONE
            if self.posture == PostureState.PUSHING_UP and \
               self._phase_steps < 200:
                return PostureState.PUSHING_UP  # still pushing, don't regress
            return PostureState.PRONE

        # Crouching / rising
        if upright < self.config.crouch_threshold:
            if self.posture == PostureState.RISING and \
               self._phase_steps < 150:
                return PostureState.RISING  # still rising
            return PostureState.CROUCHING

        # Destabilized (was standing, now wobbling)
        if upright < self.config.standing_threshold:
            if self.posture in (PostureState.STANDING, PostureState.WALKING,
                                PostureState.DESTABILIZED):
                return PostureState.DESTABILIZED
            return PostureState.RISING  # still getting up

        # Standing or walking
        if speed > self.config.walking_speed:
            return PostureState.WALKING
        return PostureState.STANDING

    # ═══════════════════════════════════════════════════════
    # POSTURE TRANSITIONS — chain logic
    # ═══════════════════════════════════════════════════════

    def _update_posture(self, sensor_data: dict, is_fallen: bool):
        """
        Update posture state machine. Transitions follow the
        developmental cascade — you can't skip states.
        """
        new_posture = self._classify_posture(sensor_data, is_fallen)

        # Enforce minimum phase duration (prevent jitter)
        if self._phase_steps < self.config.min_phase_steps:
            if new_posture != self.posture:
                # Exception: falling is always immediate
                upright = sensor_data.get('upright', 1.0)
                if upright < self.config.supine_threshold:
                    pass  # allow immediate transition to SUPINE
                else:
                    return  # stay in current phase

        # Initiate transitions
        if new_posture == PostureState.SUPINE and \
           self.posture != PostureState.SUPINE:
            # Just fell on back — prepare to roll
            self._fallen_steps = 0

        if self.posture == PostureState.SUPINE and \
           self._phase_steps > self.config.min_phase_steps:
            # Been on back long enough, start rolling
            euler = sensor_data.get('orientation_euler', np.zeros(3))
            roll = euler[0] if len(euler) > 0 else 0.0
            self._roll_direction = 1.0 if roll >= 0 else -1.0
            new_posture = PostureState.ROLLING

        if self.posture == PostureState.PRONE and \
           self._phase_steps > self.config.min_phase_steps:
            new_posture = PostureState.PUSHING_UP

        if self.posture == PostureState.CROUCHING and \
           self._phase_steps > self.config.min_phase_steps:
            new_posture = PostureState.RISING

        # Apply transition
        if new_posture != self.posture:
            self.posture = new_posture
            self._phase_steps = 0
        else:
            self._phase_steps += 1

    # ═══════════════════════════════════════════════════════
    # REFLEX GENERATORS — one per posture state
    # ═══════════════════════════════════════════════════════

    def _reflex_supine(self, sensor_data: dict, fade: float) -> tuple:
        """
        ON BACK: TLR + rhythmic kicking to generate momentum for roll.
        Biology: TLR (Capute 1978) — supine → extension pattern.
        Spinal oscillator (Grillner 1985) — rhythmic movement to
        generate enough momentum to roll over.
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.tlr_gain * fade

        # TLR: extend all legs (push against whatever surface)
        for leg in range(self.n_legs):
            base = leg * self.jpleg
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)
            commands[hip_idx] += -0.6 * gain
            commands[knee_idx] += 0.5 * gain

        # Spinal oscillator: kick to build momentum
        if self._fallen_steps > self.config.kick_onset_steps:
            t = self._fallen_steps * 0.005  # assume 5ms timestep
            omega = 2.0 * np.pi * self.config.kick_freq
            stress = min(1.5, 1.0 + self._fallen_steps * 0.002)
            kick_gain = self.config.kick_amplitude * fade

            for leg in range(self.n_legs):
                base = leg * self.jpleg
                phase = omega * t + self._leg_phases[leg]
                hip_idx = base + (1 if self.jpleg == 4 else 0)
                knee_idx = base + (2 if self.jpleg == 4 else 1)
                commands[hip_idx] += np.sin(phase) * 0.8 * kick_gain * stress
                commands[knee_idx] += np.cos(phase) * 0.6 * kick_gain * stress

        return commands, 'TLR+KICK'

    def _reflex_rolling(self, sensor_data: dict, fade: float) -> tuple:
        """
        ROLLING: asymmetric push to roll from back to belly.
        Biology: neck-righting reflex (Magnus 1924) — head turns,
        body follows segmentally. In quadrupeds: asymmetric limb
        extension rolls the body to prone position.
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.righting_gain * fade
        d = self._roll_direction

        # Asymmetric: extend legs on one side, flex on other
        for leg in range(self.n_legs):
            base = leg * self.jpleg
            is_left = leg in [0, 2]
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            if (is_left and d > 0) or (not is_left and d < 0):
                # Push side: extend hard
                commands[hip_idx] += -0.8 * gain
                commands[knee_idx] += 0.7 * gain
            else:
                # Pull side: flex to make room
                commands[hip_idx] += 0.4 * gain
                commands[knee_idx] += -0.3 * gain

            # Abduction helps roll
            if self.jpleg == 4:
                abd_force = 0.5 * gain * d * (1.0 if is_left else -1.0)
                commands[base] += abd_force

        return commands, 'ROLL'

    def _reflex_prone(self, sensor_data: dict, fade: float) -> tuple:
        """
        ON BELLY: Landau reflex — lift head and trunk.
        Biology: Landau reflex (3-12 months) — prone infant extends
        spine and lifts head. Foundation for pushing up.
        Ref: Palmar 1978, Fiorentino 1981
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.extensor_gain * fade
        upright = sensor_data.get('upright', 0.2)

        # Urgency: flatter = push harder
        urgency = np.clip((0.35 - upright) / 0.3, 0.3, 1.0)

        for leg in range(self.n_legs):
            base = leg * self.jpleg
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            # Front legs: push torso up (primary)
            # Rear legs: prepare but less force (secondary)
            is_front = leg in [0, 1]
            force = 1.0 if is_front else 0.6

            commands[hip_idx] += -0.7 * gain * urgency * force
            commands[knee_idx] += 0.8 * gain * urgency * force

            if self.jpleg == 4:
                commands[base + 3] += -0.4 * gain * urgency * force  # ankle

        return commands, 'LANDAU'

    def _reflex_pushing_up(self, sensor_data: dict, fade: float) -> tuple:
        """
        PUSHING UP: stronger extensor thrust, all four limbs engaged.
        Biology: extensor thrust reflex (Sherrington 1906) — ground
        contact triggers leg extension. Combined with Landau = push-up.
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.extensor_gain * fade

        for leg in range(self.n_legs):
            base = leg * self.jpleg
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            # Full extension — all legs push maximally
            commands[hip_idx] += -0.9 * gain
            commands[knee_idx] += 1.0 * gain

            if self.jpleg == 4:
                # Abduction: spread legs for wider base of support
                is_left = leg in [0, 2]
                commands[base] += (0.3 if is_left else -0.3) * gain
                commands[base + 3] += -0.5 * gain  # ankle push

        return commands, 'PUSH'

    def _reflex_crouching(self, sensor_data: dict, fade: float) -> tuple:
        """
        CROUCHING: limbs under body, preparing to extend.
        Biology: symmetrical tonic neck reflex (STNR) integration
        stage — the creature has weight on all fours but isn't
        fully extended yet.
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.righting_gain * fade

        euler = sensor_data.get('orientation_euler', np.zeros(3))
        roll = euler[0] if len(euler) > 0 else 0.0
        pitch = euler[1] if len(euler) > 1 else 0.0

        for leg in range(self.n_legs):
            base = leg * self.jpleg
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            # Gentle extension toward standing
            commands[hip_idx] += -0.4 * gain
            commands[knee_idx] += 0.5 * gain

            # Tilt correction: push harder on the low side
            is_left = leg in [0, 2]
            roll_correction = np.clip(roll / 0.3, -1.0, 1.0)
            side_bias = (-roll_correction if is_left else roll_correction) * 0.3 * gain
            commands[knee_idx] += side_bias

            # Pitch correction
            is_front = leg in [0, 1]
            pitch_correction = np.clip(pitch / 0.3, -1.0, 1.0)
            pitch_bias = (pitch_correction if is_front else -pitch_correction) * 0.2 * gain
            commands[hip_idx] += pitch_bias

        return commands, 'CROUCH'

    def _reflex_rising(self, sensor_data: dict, fade: float) -> tuple:
        """
        RISING: final extension to standing.
        Biology: labyrinthine righting reflex — vestibular-driven
        extension of all limbs to achieve upright posture.
        Ref: Magnus 1924
        """
        commands = np.zeros(self.n_actuators)
        gain = self.config.righting_gain * fade

        for leg in range(self.n_legs):
            base = leg * self.jpleg
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            # Strong extension
            commands[hip_idx] += -0.6 * gain
            commands[knee_idx] += 0.7 * gain

            if self.jpleg == 4:
                commands[base + 3] += -0.4 * gain

        return commands, 'RISE'

    def _reflex_standing(self, sensor_data: dict, fade: float) -> tuple:
        """
        STANDING: balance maintenance via righting + crossed extension.
        Biology: postural reflexes — continuous small corrections
        to maintain upright stance. These are lifelong, never fully fade.
        """
        commands = np.zeros(self.n_actuators)

        euler = sensor_data.get('orientation_euler', np.zeros(3))
        roll = euler[0] if len(euler) > 0 else 0.0
        pitch = euler[1] if len(euler) > 1 else 0.0
        joint_pos = sensor_data.get('joint_positions', np.zeros(self.n_actuators))
        if len(joint_pos) < self.n_actuators:
            joint_pos = np.zeros(self.n_actuators)

        # Righting: correct tilt (always active, even in adults)
        if abs(roll) > 0.05 or abs(pitch) > 0.05:
            gain = self.config.righting_gain * fade * 0.5  # gentler when standing
            roll_norm = np.clip(roll / 0.3, -1.0, 1.0)
            pitch_norm = np.clip(pitch / 0.3, -1.0, 1.0)

            for leg in range(self.n_legs):
                base = leg * self.jpleg
                is_left = leg in [0, 2]
                is_front = leg in [0, 1]
                hip_idx = base + (1 if self.jpleg == 4 else 0)
                knee_idx = base + (2 if self.jpleg == 4 else 1)

                # Roll correction
                side_sign = -roll_norm if is_left else roll_norm
                commands[knee_idx] += side_sign * 0.3 * gain
                if self.jpleg == 4:
                    commands[base] += side_sign * 0.2 * gain

                # Pitch correction
                pitch_sign = pitch_norm if is_front else -pitch_norm
                commands[hip_idx] += pitch_sign * 0.25 * gain

        # Crossed extension: diagonal coordination for gait preparation
        gain = self.config.crossed_ext_gain * fade
        diag_pairs = [(0, 3), (1, 2)]
        for leg_a, leg_b in diag_pairs:
            knee_a = leg_a * self.jpleg + (2 if self.jpleg == 4 else 1)
            knee_b = leg_b * self.jpleg + (2 if self.jpleg == 4 else 1)
            if knee_a < len(joint_pos) and knee_b < len(joint_pos):
                diff = joint_pos[knee_a] - joint_pos[knee_b]
                if abs(diff) > 0.08:
                    sign = np.sign(diff)
                    commands[knee_b] += sign * 0.2 * gain
                    commands[knee_a] -= sign * 0.2 * gain

        active = []
        if abs(roll) > 0.05 or abs(pitch) > 0.05:
            active.append('RIGHT')
        active.append('CROSS')
        return commands, '+'.join(active)

    def _reflex_destabilized(self, sensor_data: dict, fade: float) -> tuple:
        """
        DESTABILIZED: stronger corrections to prevent fall.
        Biology: stumble reflex / protective extension — when balance
        is threatened, reflexes intensify before voluntary correction.
        """
        commands = np.zeros(self.n_actuators)
        euler = sensor_data.get('orientation_euler', np.zeros(3))
        roll = euler[0] if len(euler) > 0 else 0.0
        pitch = euler[1] if len(euler) > 1 else 0.0

        # Stronger righting (1.5x normal)
        gain = self.config.righting_gain * fade * 1.5
        roll_norm = np.clip(roll / 0.25, -1.0, 1.0)
        pitch_norm = np.clip(pitch / 0.25, -1.0, 1.0)

        for leg in range(self.n_legs):
            base = leg * self.jpleg
            is_left = leg in [0, 2]
            is_front = leg in [0, 1]
            hip_idx = base + (1 if self.jpleg == 4 else 0)
            knee_idx = base + (2 if self.jpleg == 4 else 1)

            # Strong roll correction
            side_sign = -roll_norm if is_left else roll_norm
            commands[knee_idx] += side_sign * 0.5 * gain
            if self.jpleg == 4:
                commands[base] += side_sign * 0.4 * gain

            # Strong pitch correction
            pitch_sign = pitch_norm if is_front else -pitch_norm
            commands[hip_idx] += pitch_sign * 0.4 * gain

            # Widen stance (abduction) for stability
            if self.jpleg == 4:
                abd_sign = 1.0 if is_left else -1.0
                commands[base] += abd_sign * 0.3 * gain

        return commands, 'STABIL'

    def _reflex_walking(self, sensor_data: dict, fade: float) -> tuple:
        """
        WALKING: minimal reflexes — crossed extension only.
        CPG and actor handle locomotion, reflexes just maintain
        diagonal coordination.
        """
        commands = np.zeros(self.n_actuators)
        joint_pos = sensor_data.get('joint_positions', np.zeros(self.n_actuators))
        if len(joint_pos) < self.n_actuators:
            joint_pos = np.zeros(self.n_actuators)

        gain = self.config.crossed_ext_gain * fade * 0.5  # gentle during walk
        diag_pairs = [(0, 3), (1, 2)]
        for leg_a, leg_b in diag_pairs:
            knee_a = leg_a * self.jpleg + (2 if self.jpleg == 4 else 1)
            knee_b = leg_b * self.jpleg + (2 if self.jpleg == 4 else 1)
            if knee_a < len(joint_pos) and knee_b < len(joint_pos):
                diff = joint_pos[knee_a] - joint_pos[knee_b]
                if abs(diff) > 0.1:
                    sign = np.sign(diff)
                    commands[knee_b] += sign * 0.15 * gain
                    commands[knee_a] -= sign * 0.15 * gain

        return commands, 'CROSS'

    # ═══════════════════════════════════════════════════════
    # MAIN COMPUTE — dispatch to current posture's reflexes
    # ═══════════════════════════════════════════════════════

    def compute(self, sensor_data: dict, is_fallen: bool,
                sim_dt: float = 0.005) -> np.ndarray:
        """
        Compute reflex motor commands based on current posture state.

        The state machine ensures reflexes fire in the correct
        developmental sequence, not all at once.
        """
        self._total_steps += 1
        if is_fallen:
            self._fallen_steps += 1
        else:
            self._fallen_steps = 0

        # Update posture state machine
        self._update_posture(sensor_data, is_fallen)

        fade = self._fade_multiplier()
        self.stats['gain_multiplier'] = fade
        self.stats['posture_state'] = self.posture.value

        # Dispatch to posture-specific reflex generator
        dispatch = {
            PostureState.SUPINE: self._reflex_supine,
            PostureState.ROLLING: self._reflex_rolling,
            PostureState.PRONE: self._reflex_prone,
            PostureState.PUSHING_UP: self._reflex_pushing_up,
            PostureState.CROUCHING: self._reflex_crouching,
            PostureState.RISING: self._reflex_rising,
            PostureState.STANDING: self._reflex_standing,
            PostureState.WALKING: self._reflex_walking,
            PostureState.DESTABILIZED: self._reflex_destabilized,
        }

        handler = dispatch.get(self.posture, self._reflex_standing)
        commands, active_label = handler(sensor_data, fade)

        commands = np.clip(commands, -1.0, 1.0)
        self.stats['reflex_magnitude'] = float(np.abs(commands).mean())
        self.stats['active_reflexes'] = active_label

        return commands

    def get_stats(self) -> dict:
        return dict(self.stats)
