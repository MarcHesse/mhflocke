"""
MH-FLOCKE — Mogli Oscillator v0.1.0
========================================
SNN-based coupled half-center CPG for quadruped locomotion.

Replaces the mathematical SpinalCPG (sin/cos) with biologically
grounded Izhikevich half-center oscillators. Each leg has its own
oscillator that runs independently; inter-leg coupling via
interneurons produces gait patterns.

Named after the project's test dog Mogli.

Biology:
    Thomas Graham Brown (1911): Two neurons with mutual inhibition
    + intrinsic adaptation = alternating rhythm. Every vertebrate
    spinal CPG uses this principle.

    Grillner (2003): Coupled oscillators in the lamprey spinal cord
    produce swimming via phase-locked left-right alternation.

    Ijspeert (2008): CPG models for locomotion control in robotics.

Architecture:
    Per leg: 1 MogliHalfCenter per joint (abd, hip, knee)
             = 2 Izhikevich neurons per joint × 3 joints × 4 legs
             = 24 CPG neurons total (fits in Freenove's 232-neuron SNN)

    Inter-leg coupling:
             Left↔Right: commissural inhibition (alternation)
             Front↔Hind: propriospinal delay (walking sequence)

    The coupling produces gait patterns:
             Strong L-R coupling → walk (alternating)
             Weak L-R coupling → bound (synchronized)

    Tonic drive from brainstem (MLR) sets frequency.
    Steering = asymmetric drive (more drive → longer steps).
    Proprioceptive feedback entrains oscillators to actual movement.

Ref: Brown 1911, Grillner 2003, Ijspeert 2008, Danner et al. 2017
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MogliConfig:
    """Configuration for the Mogli Oscillator CPG."""

    # Izhikevich parameters for CPG interneurons
    # Intrinsically Bursting: produces rhythmic burst-pause patterns
    # that naturally generate flexion/extension alternation
    izh_a: float = 0.02    # Recovery time scale
    izh_b: float = 0.2     # Sensitivity of u to V
    izh_c: float = -55.0   # After-spike reset V (IB: -55)
    izh_d: float = 4.0     # After-spike reset u increment (IB: 4)

    # Half-center coupling
    w_mutual: float = -20.0     # Mutual inhibition (strong → clean alternation)
    w_self_excite: float = 1.0  # Self-excitation (sustains burst during active phase)

    # Tonic drive from brainstem (MLR)
    tonic_drive_base: float = 12.0  # Baseline current to each neuron
    tonic_drive_range: float = 5.0  # Modulation range (arousal scales this)

    # Inter-leg coupling weights
    # These produce a WALK gait pattern (diagonal pairs in phase):
    #   FL+RR move together, FR+RL move together
    #   Left↔Right alternate, Front↔Hind alternate
    # Contralateral (left↔right): inhibitory → alternation
    w_contralateral: float = -12.0
    # Ipsilateral (front↔hind same side): inhibitory → alternation
    w_ipsilateral: float = -5.0
    # Diagonal (FL↔RR, FR↔RL): excitatory → synchronization
    w_diagonal: float = 10.0

    # Output scaling
    # Biology: Motor neuron excitability is NOT fixed. It is modulated by:
    # 1. Serotonin (5-HT) from brainstem raphe nuclei → sets baseline gain
    # 2. Cerebellar DCN output → adjusts gain per-joint via reticulospinal tract
    # 3. Developmental maturation → gain increases as infant learns
    #
    # Implementation: output_gain starts at initial_gain (cautious infant)
    # and ramps toward max_gain over gain_ramp_steps. The cerebellum can
    # further modulate via set_gain_modulation().
    initial_gain: float = 3.0      # Starting gain (cautious but visible newborn)
    max_gain: float = 8.0          # Maximum gain after maturation
    gain_ramp_steps: int = 2000    # Steps to reach max_gain (developmental)
    output_smoothing: float = 0.9  # EMA smoothing on output (prevents jitter)

    # Per-joint amplitude scaling (relative)
    abd_scale: float = 0.3    # Abduction: small lateral sway
    hip_scale: float = 1.0    # Hip: main swing joint
    knee_scale: float = 0.8   # Knee: flex/extend for clearance

    # Proprioceptive feedback gain
    proprio_gain: float = 0.5  # How much joint angle feedback entrains oscillator

    # Phase offsets for walk gait (fraction of cycle)
    # FL=0.0, FR=0.5, RL=0.75, RR=0.25 → standard walk
    initial_phase_offsets: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 0.25])


class IzhikevichCPGNeuron:
    """
    Single Izhikevich neuron for CPG half-center oscillator.

    Uses Intrinsically Bursting (IB) parameters that produce
    rhythmic burst-pause patterns suitable for CPG oscillation.

    The key property: adaptation variable `u` accumulates during
    firing and eventually suppresses the neuron → partner takes over.
    """

    def __init__(self, a=0.02, b=0.2, c=-55.0, d=4.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # State
        self.V = c + np.random.uniform(-5, 5)  # Membrane potential (mV)
        self.u = b * self.V                     # Recovery variable
        self.fired = False                       # Spike flag
        self.firing_rate = 0.0                   # Smoothed firing rate

    def step(self, I_total: float, dt_steps: int = 1) -> bool:
        """
        Advance neuron by one timestep.

        Args:
            I_total: Total input current (tonic + coupling + feedback)
            dt_steps: Number of 0.5ms sub-steps for numerical stability

        Returns:
            True if neuron fired a spike
        """
        self.fired = False
        for _ in range(dt_steps):
            if self.V >= 30.0:
                self.V = self.c
                self.u += self.d
                self.fired = True

            # Izhikevich dynamics (two half-steps for stability)
            dV = 0.04 * self.V * self.V + 5.0 * self.V + 140.0 - self.u + I_total
            self.V += 0.5 * dV
            dV = 0.04 * self.V * self.V + 5.0 * self.V + 140.0 - self.u + I_total
            self.V += 0.5 * dV
            du = self.a * (self.b * self.V - self.u)
            self.u += du

        # Smoothed firing rate (for output)
        self.firing_rate = 0.9 * self.firing_rate + 0.1 * (1.0 if self.fired else 0.0)
        return self.fired


class MogliHalfCenter:
    """
    Half-center oscillator: two Izhikevich neurons with mutual inhibition.

    Flexor and extensor alternate through adaptation-driven switching:
    1. Flexor fires → inhibits extensor → u accumulates
    2. Flexor fatigues (u too high) → stops firing
    3. Extensor released from inhibition → fires → inhibits flexor
    4. Extensor fatigues → cycle repeats

    Output: flexor_rate - extensor_rate → push-pull motor command
    """

    def __init__(self, config: MogliConfig):
        self.config = config
        self.flexor = IzhikevichCPGNeuron(
            config.izh_a, config.izh_b, config.izh_c, config.izh_d)
        self.extensor = IzhikevichCPGNeuron(
            config.izh_a, config.izh_b, config.izh_c, config.izh_d)
        self._output_ema = 0.0  # Smoothed output

    def set_initial_phase(self, phase_fraction: float):
        """
        Set initial phase by biasing one neuron.
        phase_fraction 0.0 = flexor dominant, 0.5 = extensor dominant.
        """
        if phase_fraction < 0.5:
            # Flexor starts active
            self.flexor.V = -40.0 + phase_fraction * 20.0
            self.extensor.V = self.config.izh_c
        else:
            # Extensor starts active
            self.extensor.V = -40.0 + (phase_fraction - 0.5) * 20.0
            self.flexor.V = self.config.izh_c

    def step(self, tonic_drive: float, external_coupling: float = 0.0,
             proprio_feedback: float = 0.0) -> float:
        """
        One oscillator step.

        Args:
            tonic_drive: Brainstem tonic input (sets frequency)
            external_coupling: Input from other oscillators
            proprio_feedback: Joint angle feedback (entrainment)

        Returns:
            Output value: positive = flexion, negative = extension
        """
        cfg = self.config

        # Mutual inhibition: each neuron inhibits the other
        flex_inhibit = cfg.w_mutual * self.extensor.firing_rate
        ext_inhibit = cfg.w_mutual * self.flexor.firing_rate

        # Self-excitation (maintains firing during burst)
        flex_self = cfg.w_self_excite * self.flexor.firing_rate
        ext_self = cfg.w_self_excite * self.extensor.firing_rate

        # Proprioceptive feedback: positive angle → boost flexor
        proprio_flex = proprio_feedback * cfg.proprio_gain
        proprio_ext = -proprio_feedback * cfg.proprio_gain

        # Total current
        I_flex = tonic_drive + flex_inhibit + flex_self + external_coupling + proprio_flex
        I_ext = tonic_drive + ext_inhibit + ext_self - external_coupling + proprio_ext

        # Step both neurons
        self.flexor.step(I_flex)
        self.extensor.step(I_ext)

        # Push-pull output
        # Flexor-dominant: positive output = flexion
        raw_output = self.flexor.firing_rate - self.extensor.firing_rate

        # Smooth output (prevents motor jitter)
        self._output_ema = (cfg.output_smoothing * self._output_ema +
                           (1 - cfg.output_smoothing) * raw_output)

        return self._output_ema

    def get_state(self) -> dict:
        return {
            'flex_V': self.flexor.V,
            'ext_V': self.extensor.V,
            'flex_u': self.flexor.u,
            'ext_u': self.extensor.u,
            'flex_rate': self.flexor.firing_rate,
            'ext_rate': self.extensor.firing_rate,
            'output': self._output_ema,
        }


class MogliCPG:
    """
    Four-leg coupled oscillator network for quadruped locomotion.

    Each leg has 3 half-center oscillators (abd, hip, knee).
    Legs are coupled via commissural (L↔R) and propriospinal (F↔H)
    interneurons. The coupling topology determines the gait pattern.

    Tonic drive from brainstem sets overall frequency.
    Asymmetric drive produces steering (turning).

    API-compatible with SpinalCPG.compute() for drop-in replacement.
    """

    def __init__(self, n_actuators: int = 12, joints_per_leg: int = 3,
                 config: MogliConfig = None):
        self.config = config or MogliConfig()
        self.n_actuators = n_actuators
        self.n_legs = 4
        self.jpleg = joints_per_leg

        # Leg names for clarity
        self._leg_names = ['FL', 'FR', 'RL', 'RR']

        # Create oscillators: 1 half-center per joint per leg
        self.oscillators: Dict[str, MogliHalfCenter] = {}
        joint_names = ['abd', 'hip', 'knee'][:joints_per_leg]

        for leg_idx, leg_name in enumerate(self._leg_names):
            for joint_name in joint_names:
                key = f'{leg_name}_{joint_name}'
                osc = MogliHalfCenter(self.config)
                # Set initial phase based on gait pattern
                phase = self.config.initial_phase_offsets[leg_idx]
                # Offset within leg: hip-knee phase relationship determines
                # walking direction. Knee LAGGING hip = forward walking.
                # Knee LEADING hip = backward walking.
                # Biology: stance extension pushes body forward, then knee
                # flexes to lift and swing the leg for the next step.
                if joint_name == 'knee':
                    phase += -0.25  # Knee lags hip → forward
                elif joint_name == 'abd':
                    phase += 0.0  # In-phase with hip
                osc.set_initial_phase(phase % 1.0)
                self.oscillators[key] = osc

        # Inter-leg coupling state
        self._coupling_outputs: Dict[str, float] = {
            name: 0.0 for name in self._leg_names}

        # === LEARNABLE COUPLING WEIGHTS (R-STDP) ===
        # Instead of fixed w_contralateral/w_ipsilateral/w_diagonal,
        # each leg-to-leg connection has a learnable weight.
        # The weight matrix starts at the config values (innate bias)
        # and adapts through R-STDP: reward strengthens active couplings,
        # absent feedback weakens unused couplings.
        #
        # Biology: Neonatal CPG has innate coupling from genetics.
        # Postnatal experience refines it. Limb loss causes Hebbian
        # decay of connections to the missing limb's oscillator.
        #
        # Matrix layout: coupling_weights[source_leg][target_leg]
        # Positive = excitatory (sync), Negative = inhibitory (alternate)
        cfg = self.config
        self.coupling_weights = np.array([
            #  FL→      FR→      RL→      RR→     (source)
            [  0.0,  cfg.w_contralateral, cfg.w_ipsilateral, cfg.w_diagonal],  # →FL
            [cfg.w_contralateral,   0.0,  cfg.w_diagonal, cfg.w_ipsilateral],  # →FR
            [cfg.w_ipsilateral, cfg.w_diagonal,   0.0,  cfg.w_contralateral],  # →RL
            [cfg.w_diagonal, cfg.w_ipsilateral, cfg.w_contralateral,   0.0],   # →RR
        ], dtype=np.float64)

        # Eligibility traces for coupling weights (for R-STDP)
        self._coupling_eligibility = np.zeros((4, 4), dtype=np.float64)
        self._coupling_trace_decay = 0.95

        # R-STDP learning rate for coupling weights
        self._coupling_lr = 0.01
        # Weight bounds (prevent sign flips from destabilizing gait)
        self._coupling_w_min = -25.0
        self._coupling_w_max = 15.0

        # Step counter (for compatibility with SpinalCPG)
        self._step = 0

        # Adaptive output gain (developmental maturation)
        # Biology: Neonatal motor neurons have low excitability.
        # 5-HT from raphe nuclei gradually increases gain over days.
        # Here: linear ramp from initial_gain to max_gain over ramp_steps.
        self._current_gain = cfg.initial_gain
        self._gain_modulation = 1.0  # Cerebellar modulation (set externally)

        # Babbling noise (set by drive loop)
        self._babbling_noise = 0.0

        # Stats
        self.stats = {
            'freq_estimate': 0.0,
            'left_right_phase': 0.0,
        }

    def compute(self, dt: float = 0.005, arousal: float = 0.5,
                freq_scale: float = 1.0, amp_scale: float = 1.0,
                steering: float = 0.0) -> np.ndarray:
        """
        Generate one step of CPG motor commands.

        API-compatible with SpinalCPG.compute() for drop-in replacement.

        Args:
            dt: simulation timestep (affects sub-stepping)
            arousal: 0-1, modulates tonic drive (NE level)
            freq_scale: frequency multiplier (>1 = faster)
            amp_scale: amplitude multiplier (0 = stopped)
            steering: -1..+1, asymmetric drive for turning

        Returns:
            np.ndarray of shape (n_actuators,) with values in [-1, 1]
        """
        cfg = self.config
        self._step += 1

        # Developmental gain maturation
        # Biology: Motor neuron excitability ramps up over postnatal days.
        # Serotonergic innervation from raphe nuclei increases gradually.
        maturation = min(1.0, self._step / max(1, cfg.gain_ramp_steps))
        self._current_gain = (cfg.initial_gain +
                             (cfg.max_gain - cfg.initial_gain) * maturation)
        # Apply cerebellar modulation (set via set_gain_modulation())
        effective_gain = self._current_gain * self._gain_modulation

        # Tonic drive: brainstem MLR → sets oscillation frequency
        # Higher drive = faster oscillation (biology: MLR stimulation
        # increases locomotor speed monotonically)
        base_drive = cfg.tonic_drive_base + cfg.tonic_drive_range * arousal
        base_drive *= freq_scale

        # Amplitude scaling with CPG autonomy floor
        # Biology: The spinal CPG maintains autonomous rhythm independent
        # of cortical commands. A decerebrate cat still walks on a treadmill.
        # The behavior planner (cortex) can modulate amplitude but cannot
        # silence the CPG below a floor — that requires active inhibition
        # from the basal ganglia (not implemented yet).
        # Exception: amp_scale=0 (rest) explicitly stops locomotion.
        if amp_scale < 0.01:
            drive = base_drive * 0.01  # Rest: near-zero
        else:
            _min_amp = 0.7  # CPG autonomy floor: never below 70%
            effective_amp = max(_min_amp, amp_scale)
            drive = base_drive * effective_amp

        # Steering: TWO mechanisms (biology: reticulospinal + rubrospinal)
        #
        # 1. Tonic drive asymmetry → changes oscillation FREQUENCY per side
        #    Biology: MLR sends asymmetric tonic drive via reticulospinal tract
        #    Result: inside legs oscillate slower (shorter steps)
        #
        # 2. Output amplitude scaling → changes STEP LENGTH per side
        #    Biology: Rubrospinal tract modulates motor neuron gain
        #    Result: inside legs produce smaller movements
        #
        # Combined: inside legs are both slower AND shorter → natural curve
        # Ref: Grillner 2003 (lamprey turning), Rybak 2006 (L-R coupling)
        steering_clamped = np.clip(steering, -0.8, 0.8)
        # Mechanism 1: Tonic drive asymmetry (frequency)
        left_drive_mod = 1.0 + steering_clamped * 0.3
        right_drive_mod = 1.0 - steering_clamped * 0.3
        # Mechanism 2: Output amplitude scaling (step length)
        left_amp_mod = 1.0 + steering_clamped * 0.5
        right_amp_mod = 1.0 - steering_clamped * 0.5

        # Compute inter-leg coupling signals
        coupling = self._compute_coupling()

        # Step all oscillators
        commands = np.zeros(self.n_actuators)
        joint_names = ['abd', 'hip', 'knee'][:self.jpleg]
        joint_scales = [cfg.abd_scale, cfg.hip_scale, cfg.knee_scale][:self.jpleg]

        for leg_idx, leg_name in enumerate(self._leg_names):
            # Per-leg tonic drive (steering mechanism 1: frequency)
            is_left = (leg_idx % 2 == 0)  # FL=0, RL=2
            drive_mod = left_drive_mod if is_left else right_drive_mod
            amp_mod = left_amp_mod if is_left else right_amp_mod
            leg_drive = drive * drive_mod

            # Per-leg babbling noise
            if self._babbling_noise > 0:
                leg_drive *= (1.0 + np.random.uniform(
                    -self._babbling_noise, self._babbling_noise))

            for joint_idx, (joint_name, joint_scale) in enumerate(
                    zip(joint_names, joint_scales)):
                key = f'{leg_name}_{joint_name}'
                osc = self.oscillators[key]

                # Inter-leg coupling for this leg's hip oscillator
                ext_coupling = coupling[leg_name] if joint_name == 'hip' else 0.0

                # Step oscillator
                output = osc.step(
                    tonic_drive=leg_drive,
                    external_coupling=ext_coupling,
                    proprio_feedback=0.0,  # TODO: connect to actual joint angles
                )

                # Scale output per joint type + steering amplitude (mechanism 2)
                motor_idx = leg_idx * self.jpleg + joint_idx
                if motor_idx < self.n_actuators:
                    commands[motor_idx] = output * effective_gain * joint_scale * amp_mod

        return np.clip(commands, -1.0, 1.0)

    def _compute_coupling(self) -> Dict[str, float]:
        """
        Compute inter-leg coupling signals using LEARNABLE weight matrix.

        Each leg's hip oscillator sends its output to all other legs,
        weighted by coupling_weights[source][target]. The weights are
        initialized from config (innate bias) and adapted by apply_rstdp().

        Biology: Commissural and propriospinal interneurons in the spinal
        cord connect the four leg CPGs. Their synaptic strengths are
        modulated by activity-dependent plasticity. When a limb is lost,
        the connections to its oscillator decay through Hebbian weakening
        (no correlated activity → synapse weakens).

        Returns:
            Dict mapping leg_name → total coupling input for that leg's hip
        """
        # Get hip oscillator outputs for each leg
        hip_out = np.zeros(4)
        for i, leg_name in enumerate(self._leg_names):
            key = f'{leg_name}_hip'
            if key in self.oscillators:
                osc = self.oscillators[key]
                hip_out[i] = osc.flexor.firing_rate - osc.extensor.firing_rate

        # Matrix multiply: coupling = weights^T @ hip_out
        # Each target leg gets weighted sum of all source leg outputs
        coupling_input = self.coupling_weights.T @ hip_out

        # Update eligibility traces (for R-STDP learning)
        # Outer product of pre (source) × post (target) activity
        pre_post = np.outer(hip_out, hip_out)
        self._coupling_eligibility = (self._coupling_trace_decay *
                                      self._coupling_eligibility + pre_post)

        # Store for stats
        self._last_hip_out = hip_out.copy()

        return {name: float(coupling_input[i])
                for i, name in enumerate(self._leg_names)}

    def apply_coupling_rstdp(self, reward_signal: float):
        """
        Apply R-STDP to inter-leg coupling weights.

        Called from training loop with the locomotion reward signal.
        Reward > 0: strengthen couplings that were active together
        Reward < 0: weaken couplings that were active together

        Biology: Dopamine (reward) modulates synaptic plasticity in
        spinal interneurons. Successful gait patterns are reinforced.
        When a limb is absent, its oscillator produces no output →
        no correlated activity → eligibility = 0 → no reinforcement →
        weight decay slowly weakens the connection.

        Args:
            reward_signal: DA-like reward (positive = good gait)
        """
        # dw = lr × reward × eligibility
        dw = self._coupling_lr * reward_signal * self._coupling_eligibility

        # Clip update magnitude
        dw = np.clip(dw, -0.05, 0.05)

        # Don't learn self-connections (diagonal = 0)
        np.fill_diagonal(dw, 0.0)

        # Apply
        self.coupling_weights += dw

        # Weight decay: all weights slowly drift toward zero
        # Biology: synaptic homeostasis — connections that aren't
        # actively maintained by correlated activity slowly weaken.
        # This is what causes the dead-leg connections to fade.
        self.coupling_weights *= 0.9999  # very slow decay

        # Clamp to bounds (prevent sign flips)
        self.coupling_weights = np.clip(
            self.coupling_weights, self._coupling_w_min, self._coupling_w_max)

        # Keep diagonal zero (no self-coupling)
        np.fill_diagonal(self.coupling_weights, 0.0)

        # Decay eligibility after use
        self._coupling_eligibility *= 0.3

    def compute_tendon(self, dt: float = 0.005, arousal: float = 0.5,
                       freq_scale: float = 1.0, amp_scale: float = 1.0) -> np.ndarray:
        """
        Tendon-coupled output (Bommel/dm_quadruped compatibility).
        For now, same as compute() without steering.
        """
        return self.compute(dt=dt, arousal=arousal,
                           freq_scale=freq_scale, amp_scale=amp_scale)

    def get_phase_input(self) -> np.ndarray:
        """
        Get CPG phase signal for sensor encoding (Bridge compatibility).
        Uses FL hip oscillator output as phase reference.
        """
        fl_hip = self.oscillators.get('FL_hip')
        if fl_hip:
            # Approximate phase from flexor/extensor balance
            flex_rate = fl_hip.flexor.firing_rate
            ext_rate = fl_hip.extensor.firing_rate
            # Map to sin/cos for compatibility with hardware sensor encoding
            phase_proxy = np.arctan2(flex_rate - ext_rate,
                                     flex_rate + ext_rate + 1e-6)
            return np.array([np.sin(phase_proxy), np.cos(phase_proxy)],
                           dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    def set_gain_modulation(self, modulation: float):
        """
        Cerebellar gain modulation.

        Called by CerebellarLearning to adjust oscillator amplitude
        based on DCN rebound output. This is the biological pathway
        where the cerebellum calibrates motor gain via the
        reticulospinal tract.

        Args:
            modulation: 0.5-2.0, where 1.0 = no change,
                       <1.0 = reduce amplitude (too much error),
                       >1.0 = increase amplitude (movements too small)
        """
        self._gain_modulation = np.clip(modulation, 0.3, 2.0)

    def get_cpg_weight(self) -> float:
        """Compatibility with SpinalCPG API."""
        return 0.9  # Fixed for now

    def get_stats(self) -> dict:
        """Return oscillator statistics for logging."""
        stats = {}
        stats['mogli_gain'] = self._current_gain
        stats['mogli_gain_mod'] = self._gain_modulation
        stats['mogli_effective_gain'] = self._current_gain * self._gain_modulation
        for key, osc in self.oscillators.items():
            s = osc.get_state()
            stats[f'{key}_flex_rate'] = s['flex_rate']
            stats[f'{key}_ext_rate'] = s['ext_rate']
            stats[f'{key}_output'] = s['output']

        # Estimate frequency from FL hip zero-crossings
        fl_hip = self.oscillators.get('FL_hip')
        if fl_hip:
            stats['fl_hip_output'] = fl_hip._output_ema
            stats['fl_hip_flex_V'] = fl_hip.flexor.V
            stats['fl_hip_ext_V'] = fl_hip.extensor.V

        # Coupling weight statistics
        stats['coupling_mean'] = float(np.abs(self.coupling_weights).mean())
        stats['coupling_contra'] = float(self.coupling_weights[0, 1])  # FL→FR
        stats['coupling_ipsi'] = float(self.coupling_weights[0, 2])    # FL→RL
        stats['coupling_diag'] = float(self.coupling_weights[0, 3])    # FL→RR

        return stats

    def reset_episode(self):
        """Reset oscillator states for new episode (keeps coupling weights)."""
        for osc in self.oscillators.values():
            osc.flexor.V = self.config.izh_c + np.random.uniform(-5, 5)
            osc.flexor.u = self.config.izh_b * osc.flexor.V
            osc.extensor.V = self.config.izh_c + np.random.uniform(-5, 5)
            osc.extensor.u = self.config.izh_b * osc.extensor.V
            osc._output_ema = 0.0

        # Re-apply initial phases
        for leg_idx, leg_name in enumerate(self._leg_names):
            phase = self.config.initial_phase_offsets[leg_idx]
            joint_names = ['abd', 'hip', 'knee'][:self.jpleg]
            for joint_name in joint_names:
                key = f'{leg_name}_{joint_name}'
                if key in self.oscillators:
                    jp = phase
                    if joint_name == 'knee':
                        jp += 0.25
                    self.oscillators[key].set_initial_phase(jp % 1.0)
