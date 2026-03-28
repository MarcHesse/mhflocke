"""
MH-FLOCKE — Spinal CPG v0.4.1
========================================
Innate rhythmic locomotion patterns — genetically encoded, not learned.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpinalCPGConfig:
    """Configuration for the innate spinal CPG."""
    # Gait timing
    frequency: float = 1.0          # Hz — slow walk (newborn)
    phase_offsets: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 0.25])  # Walk gait

    # Per-joint amplitudes (fraction of ctrl range)
    # These are the innate "wiring strengths" — not learned
    abd_amplitude: float = 0.05     # Lateral sway for balance
    hip_amplitude: float = 0.35     # Main swing joint — needs to overcome inertia
    knee_amplitude: float = 0.25    # Flex/extend for ground clearance
    ankle_amplitude: float = 0.15   # Push-off

    # Asymmetry (stance vs swing phase)
    stance_power: float = 1.3       # Stronger during ground contact
    swing_power: float = 0.8        # Lighter during leg lift

    # Phase relationships within leg
    knee_phase_offset: float = 0.25   # Knee leads hip by ~90° (π/2)
    ankle_phase_offset: float = 0.4   # Ankle lags knee

    # Overall amplitude modulation
    base_amplitude: float = 0.30    # Start visible (newborn wobble)
    max_amplitude: float = 0.70     # Full walking amplitude

    # Maturation: CPG amplitude ramps up over training
    maturation_steps: int = 5000    # Full amplitude after N steps
    
    # CPG weight in motor blend (rest is cerebellum)
    cpg_weight_start: float = 0.8   # 80% CPG at birth
    cpg_weight_end: float = 0.2     # 20% CPG after maturation (cerebellum takes over)
    cpg_weight_fade_steps: int = 200000  # Slow fade — cerebellum needs time to learn


class SpinalCPG:
    """
    Innate spinal central pattern generator for quadruped locomotion.
    
    Produces rhythmic motor commands that serve as a baseline pattern
    for the cerebellum to learn corrections on top of.
    
    Can load evolved CPG parameters from cpg_config.json (produced by
    evolve_creature.py or evolve_cpg_params.py).
    """

    def __init__(self, n_actuators: int, joints_per_leg: int = 4,
                 config: SpinalCPGConfig = None):
        self.config = config or SpinalCPGConfig()
        self.n_actuators = n_actuators
        self.n_legs = 4
        self.jpleg = joints_per_leg  # abd, hip, knee, ankle

        # Phase state (continuous, wraps at 2π)
        self._phases = np.array(self.config.phase_offsets) * 2 * np.pi
        self._step = 0

    @classmethod
    def from_evolved(cls, cpg_config_path: str, n_actuators: int,
                     joints_per_leg: int = 4) -> 'SpinalCPG':
        """Load evolved CPG parameters from cpg_config.json.
        
        Maps evolved CPG genome fields to SpinalCPGConfig.
        The evolved params come from evolve_creature.py or evolve_cpg_params.py.
        """
        import json
        with open(cpg_config_path) as f:
            data = json.load(f)

        cfg = SpinalCPGConfig(
            frequency=data.get('base_frequency', 1.0),
            phase_offsets=data.get('phase_offsets', [0.0, 0.5, 0.75, 0.25]),
            # Map evolved amplitudes to per-joint
            hip_amplitude=data.get('shoulder_hip_amp', 0.35),
            knee_amplitude=data.get('elbow_stifle_amp', 0.25),
            ankle_amplitude=data.get('carpus_hock_amp', 0.15),
            abd_amplitude=data.get('abduction_amp', 0.05),
            # Asymmetry
            stance_power=data.get('stance_power', 1.3),
            swing_power=data.get('swing_power', 0.8),
            # Phase offsets within leg
            knee_phase_offset=data.get('knee_phase_shift', 0.25),
            ankle_phase_offset=data.get('hock_phase_shift', 0.4),
            # Overall amplitude = evolved base_amplitude
            base_amplitude=data.get('base_amplitude', 0.30),
            max_amplitude=data.get('base_amplitude', 0.30) * 2.0,
        )
        cpg = cls(n_actuators=n_actuators, joints_per_leg=joints_per_leg, config=cfg)
        cpg._evolved_from = cpg_config_path
        return cpg

    def compute(self, dt: float = 0.005, arousal: float = 0.5,
                freq_scale: float = 1.0, amp_scale: float = 1.0,
                steering: float = 0.0) -> np.ndarray:
        """
        Generate one step of CPG motor commands (per-joint, direct control).
        
        Use this for robots with direct joint actuators (Go2, etc.).
        Use compute_tendon() for robots with tendon-coupled actuators (Bommel).
        
        Args:
            dt: simulation timestep
            arousal: 0-1, modulates amplitude (NE level, excitement)
            freq_scale: BehaviorPlanner frequency multiplier (1.0=walk, 1.4=trot)
            amp_scale: BehaviorPlanner amplitude multiplier (1.0=normal, 0.0=rest)
            steering: -1..+1, asymmetric amplitude for turning. Positive = turn right.
                Biology: Reticulospinal neurons modulate left/right CPG amplitude
                asymmetrically. Longer stride on outside, shorter on inside.
                Ref: Grillner 2003 (lamprey turning), Rybak 2006 (left-right coupling)
            
        Returns:
            np.ndarray of shape (n_actuators,) with values in [-1, 1]
        """
        cfg = self.config
        self._step += 1

        # Phase advance — modulated by behavior frequency scale
        d_phase = 2 * np.pi * cfg.frequency * freq_scale * dt
        self._phases += d_phase

        # Maturation: amplitude ramps up
        maturation = min(1.0, self._step / max(cfg.maturation_steps, 1))
        amplitude = cfg.base_amplitude + (cfg.max_amplitude - cfg.base_amplitude) * maturation

        # Arousal modulation: higher arousal = slightly larger movements
        amplitude *= (0.7 + 0.6 * arousal)

        # Behavior amplitude modulation
        amplitude *= amp_scale

        # --- Asymmetric ABDUCTION for turning (Issue #76d) ---
        # Biology: Reticulospinal projection modulates left/right abduction
        # asymmetrically. This produces body ROTATION, not lateral tilt.
        # Key insight: scaling ALL joint amplitudes causes lateral tipping
        # (the creature leans sideways instead of turning). Only abduction
        # asymmetry produces actual yaw rotation on flat ground.
        # Ref: Grillner 2003 (lamprey turning), Ijspeert 2008 (salamander)
        #
        # Positive steering = turn right:
        #   Left legs get positive abduction offset (push body right)
        #   Right legs get negative abduction offset (pull body right)
        #   Hip/knee amplitudes stay SYMMETRIC (stable gait preserved)
        steering_clamped = np.clip(steering, -0.6, 0.6)

        # Generate per-joint commands
        commands = np.zeros(self.n_actuators)

        for leg_idx in range(self.n_legs):
            phase = self._phases[leg_idx]
            base = leg_idx * self.jpleg
            # Hip/knee amplitude is SYMMETRIC (no per-leg scaling)
            leg_amplitude = amplitude

            # Raw sine for this leg
            raw_sin = np.sin(phase)

            # Stance/swing asymmetry
            if raw_sin > 0:
                power = cfg.stance_power
            else:
                power = cfg.swing_power

            # === Abduction (index 0 in leg) ===
            if self.jpleg >= 1:
                # Base: slight lateral sway, in-phase with hip
                abd_cmd = raw_sin * cfg.abd_amplitude * leg_amplitude * 0.5
                # Steering: DC offset on abduction to produce yaw rotation.
                # Left legs (0,2) get +offset, right legs (1,3) get -offset.
                # This pushes the body into a turn without destabilizing the gait.
                if abs(steering_clamped) > 0.02:
                    is_left = (leg_idx % 2 == 0)  # FL=0, RL=2 are left
                    steer_offset = steering_clamped * cfg.abd_amplitude * 5.0
                    if is_left:
                        abd_cmd += steer_offset
                    else:
                        abd_cmd -= steer_offset
                commands[base + 0] = abd_cmd

            # === Hip (index 1 in leg) ===
            if self.jpleg >= 2:
                # Main swing: forward/backward
                # v0.3.4: removed negation — Go2 walks FORWARD now
                hip_cmd = raw_sin * cfg.hip_amplitude * leg_amplitude * power
                commands[base + 1] = hip_cmd

            # === Knee (index 2 in leg) ===
            if self.jpleg >= 3:
                # Knee has phase offset: flexes during swing for ground clearance
                knee_phase = phase + cfg.knee_phase_offset * 2 * np.pi
                knee_sin = np.sin(knee_phase)
                knee_cmd = knee_sin * cfg.knee_amplitude * leg_amplitude
                commands[base + 2] = knee_cmd

            # === Ankle (index 3 in leg) ===
            if self.jpleg >= 4:
                ankle_phase = phase + cfg.ankle_phase_offset * 2 * np.pi
                ankle_sin = np.sin(ankle_phase)
                ankle_cmd = ankle_sin * cfg.ankle_amplitude * leg_amplitude
                commands[base + 3] = ankle_cmd

        return np.clip(commands, -1.0, 1.0)

    def get_cpg_weight(self) -> float:
        """Current CPG weight in motor blend (decreases as cerebellum matures)."""
        cfg = self.config
        fade = min(1.0, self._step / max(cfg.cpg_weight_fade_steps, 1))
        return cfg.cpg_weight_start + (cfg.cpg_weight_end - cfg.cpg_weight_start) * fade

    def compute_tendon(self, dt: float = 0.005, arousal: float = 0.5,
                       freq_scale: float = 1.0,
                       amp_scale: float = 1.0) -> np.ndarray:
        """
        Generate CPG commands for tendon-actuated quadrupeds (dm_control style).
        
        Instead of per-joint commands [abd, hip, knee, ankle],
        produces per-leg commands [yaw, lift, extend]:
          - yaw: lateral rotation (~abd)
          - lift: raise/lower leg (pitch + ankle coupled)
          - extend: stretch/retract leg (pitch + knee + ankle coupled)

        Args:
            dt: simulation timestep
            arousal: 0-1, NE level modulates amplitude
            freq_scale: BehaviorPlanner frequency multiplier (1.0=walk, 1.4=trot, 0.0=rest)
            amp_scale: BehaviorPlanner amplitude multiplier (1.0=normal, 0.4=sniff, 0.0=rest)
        
        Returns: np.ndarray of shape (12,) for 4 legs × 3 actuators
        """
        cfg = self.config
        self._step += 1

        # Phase advance — modulated by behavior frequency scale
        # Biology: brainstem locomotor region (MLR) controls CPG frequency
        # via reticulospinal projections. Higher drive = faster gait.
        d_phase = 2 * np.pi * cfg.frequency * freq_scale * dt
        self._phases += d_phase

        # Maturation
        maturation = min(1.0, self._step / max(cfg.maturation_steps, 1))
        amplitude = cfg.base_amplitude + (cfg.max_amplitude - cfg.base_amplitude) * maturation
        amplitude *= (0.7 + 0.6 * arousal)

        # Behavior amplitude modulation
        # Biology: motor cortex gain control on spinal interneurons
        amplitude *= amp_scale

        commands = np.zeros(12)  # 4 legs × 3 actuators

        # Motor babbling: per-leg amplitude noise for neonatal exploration.
        # Biology: immature motor units fire irregularly, creating
        # asymmetric limb movements that shift the center of mass.
        # This provides vestibular training signal on flat terrain.
        # babbling_noise is set externally by DriveMotorBridge when
        # motor_babbling behavior is active.
        babble = getattr(self, '_babbling_noise', 0.0)

        for leg_idx in range(4):
            phase = self._phases[leg_idx]
            base = leg_idx * 3  # [yaw, lift, extend]

            # Per-leg babbling modulation (slow random walk)
            if babble > 0:
                # Use phase-offset noise so each leg gets different modulation
                leg_noise = np.sin(self._step * 0.003 + leg_idx * 1.7) * babble
                leg_noise += np.sin(self._step * 0.0071 + leg_idx * 2.3) * babble * 0.5
            else:
                leg_noise = 0.0

            raw_sin = np.sin(phase)
            raw_cos = np.cos(phase)

            # Stance/swing asymmetry
            power = cfg.stance_power if raw_sin > 0 else cfg.swing_power

            # Yaw: slight lateral sway for balance (+ babbling asymmetry)
            commands[base + 0] = raw_sin * cfg.abd_amplitude * amplitude * 0.3 + leg_noise * 0.3

            # Lift: raise leg during swing, plant during stance
            # Main locomotion driver — needs clear swing phase
            lift_cmd = raw_sin * cfg.hip_amplitude * amplitude * power * (1.0 + leg_noise)
            commands[base + 1] = lift_cmd

            # Extend: stretch forward during swing, push back during stance
            extend_phase = phase + cfg.knee_phase_offset * 2 * np.pi
            extend_cmd = np.sin(extend_phase) * cfg.knee_amplitude * amplitude * 0.8 * (1.0 + leg_noise * 0.5)
            commands[base + 2] = extend_cmd

        return np.clip(commands, -1.0, 1.0)

    def get_stats(self) -> dict:
        """Stats for logging."""
        return {
            'cpg_step': self._step,
            'cpg_weight': self.get_cpg_weight(),
            'frequency': self.config.frequency,
            'maturation': min(1.0, self._step / max(self.config.maturation_steps, 1)),
        }
