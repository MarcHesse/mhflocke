"""
MH-FLOCKE — Mogli Oscillator v0.2.1
========================================
SNN-based coupled half-center CPG for quadruped locomotion.

Architecture (v0.2.1 — Firing-Rate Phase Accumulation):
    The Mogli Oscillator uses Izhikevich half-center neurons as RHYTHM
    GENERATORS, not as direct motor command sources. Each leg has one
    hip half-center oscillator (2 neurons). The oscillator output is
    converted to a PHASE signal, and joint commands are computed from
    that phase using sin/cos — exactly like the SpinalCPG.

    This mirrors the biological two-layer CPG architecture:
    Layer 1 (Rhythm Generator): Half-center interneurons produce
        alternating rhythm via mutual inhibition + adaptation.
    Layer 2 (Pattern Formation): Motoneurons translate rhythm into
        joint-specific activation patterns.

    Ref: Rybak et al. 2006 "Modelling spinal circuitry involved in
         locomotor pattern generation"

    4 Hip oscillators (8 neurons) + inter-leg coupling = gait
    Joint commands derived from phase: sin(phase) for hip,
    sin(phase + offset) for knee, matching SpinalCPG exactly.

    This means: Everything that works with SpinalCPG (forward walking,
    steering, reflexes, wall avoidance) works identically with Mogli.
    The ONLY difference is WHERE the phase comes from — biological
    neurons instead of a mathematical phase accumulator.

Named after the project's test dog Mogli.

Ref: Brown 1911, Grillner 2003, Ijspeert 2008, Rybak 2006
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MogliConfig:
    """Configuration for the Mogli Oscillator CPG."""

    # Izhikevich parameters for rhythm generator interneurons
    # Biology: Spinal CPG interneurons are FAST neurons with quick
    # recovery (high a). Regular Spiking (a=0.02) bursts and pauses;
    # Fast Spiking (a=0.1) fires rhythmically — which is what CPG
    # half-centers need for stable alternation.
    # c=-50, d=2: moderate reset, less adaptation = steadier rhythm.
    # Ref: Izhikevich 2003 Table 2 — Fast Spiking interneurons
    izh_a: float = 0.1
    izh_b: float = 0.2
    izh_c: float = -50.0
    izh_d: float = 2.0

    # Half-center coupling
    w_mutual: float = -20.0
    w_self_excite: float = 1.0

    # Tonic drive from brainstem (MLR)
    tonic_drive_base: float = 12.0
    tonic_drive_range: float = 5.0

    # Inter-leg coupling weights (walk gait)
    w_contralateral: float = -12.0
    w_ipsilateral: float = -5.0
    w_diagonal: float = 10.0

    # Joint amplitudes (matched to SpinalCPG for equivalent motor output)
    abd_amplitude: float = 0.05
    hip_amplitude: float = 0.60
    knee_amplitude: float = 0.50

    # Stance/swing asymmetry (same as SpinalCPG)
    stance_power: float = 1.3
    swing_power: float = 0.8

    # Phase relationships within leg (same as SpinalCPG)
    knee_phase_offset: float = 0.25  # Knee leads hip by 90°

    # Overall amplitude
    base_amplitude: float = 0.50
    max_amplitude: float = 0.80
    maturation_steps: int = 5000

    # Output smoothing
    output_smoothing: float = 0.9

    # Phase offsets for walk gait
    initial_phase_offsets: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 0.25])


class IzhikevichCPGNeuron:
    """Single Izhikevich neuron for CPG half-center oscillator."""

    def __init__(self, a=0.02, b=0.2, c=-55.0, d=4.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = c  # Deterministic init
        self.u = b * self.V
        self.fired = False
        self.firing_rate = 0.0

    def step(self, I_total: float, dt_steps: int = 1) -> bool:
        self.fired = False
        for _ in range(dt_steps):
            if self.V >= 30.0:
                self.V = self.c
                self.u += self.d
                self.fired = True
            dV = 0.04 * self.V * self.V + 5.0 * self.V + 140.0 - self.u + I_total
            self.V += 0.5 * dV
            dV = 0.04 * self.V * self.V + 5.0 * self.V + 140.0 - self.u + I_total
            self.V += 0.5 * dV
            du = self.a * (self.b * self.V - self.u)
            self.u += du
        self.firing_rate = 0.7 * self.firing_rate + 0.3 * (1.0 if self.fired else 0.0)
        return self.fired


class MogliHalfCenter:
    """Half-center oscillator: two Izhikevich neurons with mutual inhibition."""

    def __init__(self, config: MogliConfig):
        self.config = config
        self.flexor = IzhikevichCPGNeuron(
            config.izh_a, config.izh_b, config.izh_c, config.izh_d)
        self.extensor = IzhikevichCPGNeuron(
            config.izh_a, config.izh_b, config.izh_c, config.izh_d)
        self._output_ema = 0.0
        # Phase tracking
        self._phase = 0.0  # Extracted phase in radians
        self._prev_output = 0.0
        self._phase_velocity = 0.0  # Estimated angular velocity

    def set_initial_phase(self, phase_fraction: float):
        """Set initial phase by biasing one neuron."""
        self._phase = phase_fraction * 2.0 * np.pi
        if phase_fraction < 0.5:
            self.flexor.V = -40.0 + phase_fraction * 20.0
            self.extensor.V = self.config.izh_c
        else:
            self.extensor.V = -40.0 + (phase_fraction - 0.5) * 20.0
            self.flexor.V = self.config.izh_c

    def step(self, tonic_drive: float, external_coupling: float = 0.0) -> float:
        """Step the oscillator and extract phase.

        Phase extraction v0.2.1: Firing-rate-driven accumulation.

        Biology: Spinal motoneurons integrate interneuron activity into
        a firing frequency. The frequency directly determines step rate.
        Higher tonic drive from MLR -> faster firing -> faster stepping.
        This is the rate code that the Pattern Formation Layer reads.

        The combined firing rate of both half-center neurons determines
        how fast the phase advances. When the oscillator is active
        (neurons alternating), the total rate is ~0.5 (one fires while
        the other is silent). This maps to ~1Hz stepping via
        freq_gain=2*pi.

        When a leg is disabled (no tonic drive), firing rates drop to
        zero, phase stops advancing, and that leg freezes. The
        remaining legs adjust via R-STDP coupling weights.

        Ref: Grillner 2003 - rate coding in spinal locomotor circuits
        Ref: Ijspeert 2008 - frequency control via tonic drive
        """
        cfg = self.config

        flex_inhibit = cfg.w_mutual * self.extensor.firing_rate
        ext_inhibit = cfg.w_mutual * self.flexor.firing_rate
        flex_self = cfg.w_self_excite * self.flexor.firing_rate
        ext_self = cfg.w_self_excite * self.extensor.firing_rate

        I_flex = tonic_drive + flex_inhibit + flex_self + external_coupling
        I_ext = tonic_drive + ext_inhibit + ext_self - external_coupling

        self.flexor.step(I_flex)
        self.extensor.step(I_ext)

        # Raw output (for inter-leg coupling signals)
        raw_output = self.flexor.firing_rate - self.extensor.firing_rate
        self._output_ema = (cfg.output_smoothing * self._output_ema +
                           (1 - cfg.output_smoothing) * raw_output)

        # === PHASE EXTRACTION v0.2.1 ===
        # Firing-rate-driven phase accumulation.
        #
        # The TOTAL firing activity (flexor + extensor) determines
        # how fast this leg cycles. In a healthy half-center, exactly
        # one neuron fires at a time, so total_rate ~ 0.5.
        # freq_gain converts this to radians/step.
        #
        # freq_gain = 2*pi: total_rate=0.5 -> pi rad/step -> ~1Hz
        # This matches SpinalCPG's default 1Hz frequency.
        #
        # The phase is monotonically increasing (no zero-crossing
        # artifacts) and directly proportional to neural activity.
        total_rate = self.flexor.firing_rate + self.extensor.firing_rate
        freq_gain = 2.0 * np.pi  # radians per unit firing rate per step
        self._phase += total_rate * freq_gain * 0.030  # 0.054*(1.0/1.78) for ~1Hz with FS neurons

        return self._output_ema

    def get_phase(self) -> float:
        """Get the current extracted phase in radians."""
        return self._phase

    def get_state(self) -> dict:
        return {
            'flex_V': self.flexor.V,
            'ext_V': self.extensor.V,
            'flex_u': self.flexor.u,
            'ext_u': self.extensor.u,
            'flex_rate': self.flexor.firing_rate,
            'ext_rate': self.extensor.firing_rate,
            'output': self._output_ema,
            'phase': self._phase,
        }


class MogliCPG:
    """
    Four-leg coupled oscillator network with phase-based joint mapping.

    Layer 1 (Rhythm Generator): 4 Izhikevich half-center oscillators
        produce the basic locomotion rhythm via mutual inhibition.
    Layer 2 (Pattern Formation): Phase extracted from oscillators is
        mapped to joint angles using sin(phase) — identical to SpinalCPG.

    This means: Forward walking, steering, reflexes all work identically
    to SpinalCPG. The only difference is the rhythm source.

    API-compatible with SpinalCPG.compute() for drop-in replacement.
    """

    def __init__(self, n_actuators: int = 12, joints_per_leg: int = 3,
                 config: MogliConfig = None):
        self.config = config or MogliConfig()
        self.n_actuators = n_actuators
        self.n_legs = 4
        self.jpleg = joints_per_leg
        self._leg_names = ['FL', 'FR', 'RL', 'RR']

        # Create ONE oscillator per leg (rhythm generator)
        self.oscillators: Dict[str, MogliHalfCenter] = {}
        for leg_idx, leg_name in enumerate(self._leg_names):
            key = f'{leg_name}_hip'
            osc = MogliHalfCenter(self.config)
            phase = self.config.initial_phase_offsets[leg_idx]
            osc.set_initial_phase(phase)
            self.oscillators[key] = osc

        # Phase state per leg (for SpinalCPG-compatible output)
        self._phases = np.array([
            offset * 2.0 * np.pi
            for offset in self.config.initial_phase_offsets
        ], dtype=np.float64)

        # Inter-leg coupling
        self._coupling_outputs: Dict[str, float] = {
            name: 0.0 for name in self._leg_names}

        # Learnable coupling weights
        cfg = self.config
        self.coupling_weights = np.array([
            [0.0, cfg.w_contralateral, cfg.w_ipsilateral, cfg.w_diagonal],
            [cfg.w_contralateral, 0.0, cfg.w_diagonal, cfg.w_ipsilateral],
            [cfg.w_ipsilateral, cfg.w_diagonal, 0.0, cfg.w_contralateral],
            [cfg.w_diagonal, cfg.w_ipsilateral, cfg.w_contralateral, 0.0],
        ], dtype=np.float64)

        self._coupling_eligibility = np.zeros((4, 4), dtype=np.float64)
        self._coupling_trace_decay = 0.95
        self._coupling_lr = 0.01
        self._coupling_w_min = -25.0
        self._coupling_w_max = 15.0

        self._step = 0
        self._gain_modulation = 1.0
        self._babbling_noise = 0.0

        self.stats = {'freq_estimate': 0.0, 'left_right_phase': 0.0}

    def compute(self, dt: float = 0.005, arousal: float = 0.5,
                freq_scale: float = 1.0, amp_scale: float = 1.0,
                steering: float = 0.0) -> np.ndarray:
        """Generate motor commands. API-compatible with SpinalCPG."""
        cfg = self.config
        self._step += 1

        # Amplitude maturation (same as SpinalCPG)
        maturation = min(1.0, self._step / max(1, cfg.maturation_steps))
        amplitude = cfg.base_amplitude + (cfg.max_amplitude - cfg.base_amplitude) * maturation
        amplitude *= amp_scale * self._gain_modulation

        # CPG autonomy floor
        if amp_scale < 0.01:
            amplitude *= 0.01
        elif amp_scale < 0.7:
            amplitude = max(amplitude, cfg.base_amplitude * 0.7)

        # Tonic drive for rhythm generators
        # Biology: MLR tonic drive has a threshold component (keeps neurons
        # above firing threshold) and a modulatory component (controls
        # frequency). freq_scale only modulates the excess above threshold,
        # never silences the oscillator completely. A cat on a treadmill
        # walks slower with less MLR stimulation, but doesn't stop until
        # stimulation is removed entirely.
        # Ref: Shik & Orlovsky 1976 — MLR threshold for locomotion
        _threshold_drive = 8.0  # minimum to maintain rhythmic firing
        _modulatory_drive = (cfg.tonic_drive_base - _threshold_drive +
                            cfg.tonic_drive_range * arousal)
        base_drive = _threshold_drive + _modulatory_drive * freq_scale

        # Steering (same as SpinalCPG)
        steering_clamped = np.clip(steering, -0.6, 0.6)

        # Step rhythm generators and extract phases
        coupling = self._compute_coupling()

        for leg_idx, leg_name in enumerate(self._leg_names):
            is_left = (leg_idx % 2 == 0)
            drive_mod = 1.0 + steering_clamped * 0.3 * (1.0 if is_left else -1.0)
            leg_drive = base_drive * drive_mod

            if self._babbling_noise > 0:
                leg_drive *= (1.0 + np.random.uniform(
                    -self._babbling_noise, self._babbling_noise))

            # Step the rhythm generator
            hip_key = f'{leg_name}_hip'
            osc = self.oscillators[hip_key]
            osc.step(
                tonic_drive=leg_drive,
                external_coupling=coupling[leg_name],
            )

            # Use the oscillator's extracted phase
            self._phases[leg_idx] = osc.get_phase()

        # === PATTERN FORMATION LAYER ===
        # Generate joint commands from phases — IDENTICAL to SpinalCPG.
        # This is the part that actually produces forward walking.
        commands = np.zeros(self.n_actuators)

        for leg_idx in range(self.n_legs):
            phase = self._phases[leg_idx]
            base = leg_idx * self.jpleg
            leg_amplitude = amplitude

            # Raw sine for this leg
            raw_sin = np.sin(phase)

            # Stance/swing asymmetry (identical to SpinalCPG)
            if raw_sin > 0:
                power = cfg.stance_power
            else:
                power = cfg.swing_power

            # ABD
            if self.jpleg >= 1:
                abd_cmd = raw_sin * cfg.abd_amplitude * leg_amplitude * 0.5
                if abs(steering_clamped) > 0.02:
                    is_left = (leg_idx % 2 == 0)
                    steer_offset = steering_clamped * cfg.abd_amplitude * 5.0
                    if is_left:
                        abd_cmd += steer_offset
                    else:
                        abd_cmd -= steer_offset
                commands[base + 0] = abd_cmd

            # HIP (identical to SpinalCPG)
            if self.jpleg >= 2:
                hip_cmd = raw_sin * cfg.hip_amplitude * leg_amplitude * power
                commands[base + 1] = hip_cmd

            # KNEE (identical to SpinalCPG)
            if self.jpleg >= 3:
                knee_phase = phase + cfg.knee_phase_offset * 2 * np.pi
                knee_sin = np.sin(knee_phase)
                knee_cmd = knee_sin * cfg.knee_amplitude * leg_amplitude
                commands[base + 2] = knee_cmd

        return np.clip(commands, -1.0, 1.0)

    def _compute_coupling(self) -> Dict[str, float]:
        """Compute inter-leg coupling signals."""
        hip_out = np.zeros(4)
        for i, leg_name in enumerate(self._leg_names):
            key = f'{leg_name}_hip'
            if key in self.oscillators:
                osc = self.oscillators[key]
                hip_out[i] = osc.flexor.firing_rate - osc.extensor.firing_rate

        coupling_input = self.coupling_weights.T @ hip_out
        pre_post = np.outer(hip_out, hip_out)
        self._coupling_eligibility = (self._coupling_trace_decay *
                                      self._coupling_eligibility + pre_post)
        self._last_hip_out = hip_out.copy()

        return {name: float(coupling_input[i])
                for i, name in enumerate(self._leg_names)}

    def apply_coupling_rstdp(self, reward_signal: float):
        """Apply R-STDP to inter-leg coupling weights."""
        dw = self._coupling_lr * reward_signal * self._coupling_eligibility
        dw = np.clip(dw, -0.05, 0.05)
        np.fill_diagonal(dw, 0.0)
        self.coupling_weights += dw
        self.coupling_weights *= 0.9999
        self.coupling_weights = np.clip(
            self.coupling_weights, self._coupling_w_min, self._coupling_w_max)
        np.fill_diagonal(self.coupling_weights, 0.0)
        self._coupling_eligibility *= 0.3

    def compute_tendon(self, dt=0.005, arousal=0.5,
                       freq_scale=1.0, amp_scale=1.0):
        return self.compute(dt=dt, arousal=arousal,
                           freq_scale=freq_scale, amp_scale=amp_scale)

    def get_phase_input(self) -> np.ndarray:
        """Get CPG phase for sensor encoding (Bridge compatibility)."""
        import math
        phase_rad = float(self._phases[0])
        return np.array([math.sin(phase_rad), math.cos(phase_rad)],
                       dtype=np.float32)

    def set_gain_modulation(self, modulation: float):
        self._gain_modulation = np.clip(modulation, 0.3, 2.0)

    def get_cpg_weight(self) -> float:
        return 0.9

    def get_stats(self) -> dict:
        stats = {}
        stats['mogli_gain'] = self._gain_modulation
        for key, osc in self.oscillators.items():
            s = osc.get_state()
            stats[f'{key}_flex_rate'] = s['flex_rate']
            stats[f'{key}_ext_rate'] = s['ext_rate']
            stats[f'{key}_output'] = s['output']
            stats[f'{key}_phase'] = s['phase']

        stats['coupling_mean'] = float(np.abs(self.coupling_weights).mean())
        stats['coupling_contra'] = float(self.coupling_weights[0, 1])
        stats['coupling_ipsi'] = float(self.coupling_weights[0, 2])
        stats['coupling_diag'] = float(self.coupling_weights[0, 3])
        return stats

    def reset_episode(self):
        """Reset oscillator states for new episode."""
        for leg_idx, leg_name in enumerate(self._leg_names):
            key = f'{leg_name}_hip'
            osc = self.oscillators[key]
            osc.flexor.V = self.config.izh_c
            osc.flexor.u = self.config.izh_b * osc.flexor.V
            osc.extensor.V = self.config.izh_c
            osc.extensor.u = self.config.izh_b * osc.extensor.V
            osc._output_ema = 0.0
            osc._prev_output = 0.0
            phase = self.config.initial_phase_offsets[leg_idx]
            osc.set_initial_phase(phase)
            self._phases[leg_idx] = phase * 2.0 * np.pi
