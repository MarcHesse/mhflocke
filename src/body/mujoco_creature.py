"""
MH-FLOCKE — MuJoCo Creature v0.5.2
========================================
SNN-MuJoCo bridge: population coding, sense-think-act cycle.

v0.5.2: Bilateral MH->Output symmetry fix (Issue #145).
  - Random MH->Output init creates L/R asymmetry that R-STDP amplifies
    into drift (positive feedback). Fix: average weights between bilateral
    leg pairs (FL<->FR, RL<->RR) at init time.
  - Drift analysis (2026-04-20): CPG-only walks straight (7m/50k),
    all SNN seeds drifted rightward. After fix: dr ≈ 0.
v0.5.1: Continuous topology scaling (Issue #142).
  - Removed n_hidden >= 500 cliff that jumped from ~560 to 4308 neurons.
  - Same scaling formula applies for all n_hidden values.
  - Motor Hidden (30% of budget) always present.
  v0.5.0: Per-population Izhikevich parameters (Issue #104).
  v0.4.3: Ultrasonic sensor, additive CPG blending.
  v0.4.2: Scalable cerebellar architecture for hardware transfer.
  - profile.json drives SNN topology (n_input, n_hidden, n_output)
  - Cerebellar populations scale proportionally
  - Hardware-matched sensor encoding (--hardware-sensors flag)
"""

__version__ = "0.5.3"     # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 53          # mh-logbuch module entry
__status__  = "active"     # active | veraltet | neu

import torch
import numpy as np
from typing import Dict, List, Optional
from src.body.genome import Genome
from src.brain.snn_controller import SNNController, SNNConfig
from src.brain.evolved_plasticity import (
    PlasticityGenome, EvolvedPlasticityRule
)
from src.brain.cognitive_brain import CognitiveBrain, CognitiveBrainConfig


# ================================================================
# ENCODING / DECODING
# ================================================================

def rate_encode(value: float, n_neurons: int,
                min_val: float = -1.0, max_val: float = 1.0,
                gain: float = 2.0) -> torch.Tensor:
    """
    Gaussian Tuning Curves: one value -> population of spike currents.
    
    Each neuron has a "preferred value" (uniformly distributed).
    Activation = Gauss(value - preferred) * gain.
    
    Biologisch: wie Place Cells, Head Direction Cells, etc.
    """
    if n_neurons == 0:
        return torch.tensor([])
    preferred = torch.linspace(min_val, max_val, n_neurons)
    sigma = (max_val - min_val) / max(n_neurons - 1, 1) * 1.5  # Overlap
    activations = torch.exp(-0.5 * ((value - preferred) / max(sigma, 0.01)) ** 2)
    return activations * gain


def decode_motor_spikes(spike_counts: torch.Tensor, n_per_joint: int,
                        substeps: int) -> List[float]:
    """
    Population Voting: Output-Spikes → Controls (-1..1).
    
    Pro Gelenk: n_per_joint/2 "push" Neuronen + n_per_joint/2 "pull" Neuronen.
    Control = (push_rate - pull_rate), normalisiert.
    """
    controls = []
    half = max(1, n_per_joint // 2)
    
    for j in range(0, len(spike_counts), n_per_joint):
        push = spike_counts[j:j + half].sum().item()
        pull = spike_counts[j + half:j + n_per_joint].sum().item()
        total = push + pull
        if total > 0:
            ctrl = (push - pull) / (substeps * half)  # Normalisiert auf ~-1..1
        else:
            ctrl = 0.0
        controls.append(np.clip(ctrl, -1.0, 1.0))
    
    return controls


# ================================================================
# HARDWARE-MATCHED SENSOR ENCODING (Freenove Bridge v2.5 compatible)
# ================================================================

# Reference angles from MJCF (degrees) — joint rest positions
# Order: FL(yaw,pitch,knee), FR(yaw,pitch,knee), RL(yaw,pitch,knee), RR(yaw,pitch,knee)
_MJCF_REF_DEG = [0, -38, 91, 0, -38, 91, 0, -38, 91, 0, -38, 91]

# Leg types: which legs are left-side, which are right-side
# Left legs (FL, RL): servo_pitch = 90 - abs_pitch, servo_knee = abs_knee
# Right legs (FR, RR): servo_pitch = 90 + abs_pitch, servo_knee = 180 - abs_knee
_LEG_IS_LEFT = [True, False, True, False]  # FL, FR, RL, RR


def qpos_to_servo_normalized(joint_angles_rad):
    """
    Convert MuJoCo joint angles (radians, relative to ref) to
    normalized servo values (0-1) matching Bridge encode_sensory().
    
    MuJoCo qpos = 0 means the joint is at its ref angle.
    Bridge normalizes absolute servo degrees / 180.
    """
    servo_norm = np.zeros(12, dtype=np.float32)
    
    for leg_idx in range(4):
        is_left = _LEG_IS_LEFT[leg_idx]
        base = leg_idx * 3
        
        # Absolute angles = ref + qpos (qpos in radians -> degrees)
        yaw_abs = _MJCF_REF_DEG[base + 0] + np.degrees(joint_angles_rad[base + 0])
        pitch_abs = _MJCF_REF_DEG[base + 1] + np.degrees(joint_angles_rad[base + 1])
        knee_abs = _MJCF_REF_DEG[base + 2] + np.degrees(joint_angles_rad[base + 2])
        
        if is_left:
            servo_yaw = yaw_abs + 90       # ~90 at standing
            servo_pitch = 90 - pitch_abs   # ~128 at standing
            servo_knee = knee_abs          # ~91 at standing
        else:
            servo_yaw = 90 - yaw_abs       # ~90 at standing
            servo_pitch = 90 + pitch_abs   # ~52 at standing
            servo_knee = 180 - knee_abs    # ~89 at standing
        
        servo_norm[base + 0] = np.clip(servo_yaw, 0, 180) / 180.0
        servo_norm[base + 1] = np.clip(servo_pitch, 0, 180) / 180.0
        servo_norm[base + 2] = np.clip(servo_knee, 0, 180) / 180.0
    
    return servo_norm


def encode_sensory_hardware_matched(joint_angles_rad, cpg_phase, imu_data,
                                    obstacle_distance: float = -1.0):
    """
    Encode sensor data in the EXACT same format as the Pi Bridge v2.5+.
    
    Layout (48 channels, matches freenove_bridge.py encode_sensory()):
      Channels  0-11: 12 joint angles (servo positions, normalized /180)
      Channels 12-13:  2 CPG phase (sin, cos)
      Channels 14-17:  4 IMU (pitch/90, roll/90, yaw/180, upright)
      Channel  18:      1 Ultrasonic proximity (0=far/none, 1=touching)
      Channels 19-47: 29 zeros (padding to 48)
    
    NO position, NO velocity, NO height, NO forward_velocity,
    NO joint_velocities, NO visual channels — the real robot
    doesn't have those sensors.
    
    Args:
        joint_angles_rad: 12 joint angles in radians from MuJoCo (relative to ref)
        cpg_phase: [sin(phase), cos(phase)] from CPG
        imu_data: dict with pitch, roll, yaw (degrees), upright (0-1)
        obstacle_distance: distance to obstacle in meters (-1 or max_range = none)
    
    Returns:
        np.array of 48 float32 values
    """
    s = np.zeros(48, dtype=np.float32)
    
    # Channels 0-11: Servo angles normalized to [0,1]
    s[:12] = qpos_to_servo_normalized(joint_angles_rad)
    
    # Channels 12-13: CPG phase
    s[12:14] = cpg_phase
    
    # Channels 14-17: IMU (matches Bridge v2.5 exactly)
    s[14] = imu_data['pitch'] / 90.0 * 0.5 + 0.5   # [-90,90] -> [0,1]
    s[15] = imu_data['roll'] / 90.0 * 0.5 + 0.5
    s[16] = imu_data['yaw'] / 180.0 * 0.5 + 0.5     # [-180,180] -> [0,1]
    s[17] = imu_data['upright']                       # [0,1]
    
    # Channel 18: Ultrasonic proximity (Issue #103)
    # Biology: trigeminal/whisker proximity sense — binary urgency signal.
    # 0.0 = no obstacle or far away (>= max_range)
    # 1.0 = obstacle touching (distance ~0)
    # Nonlinear mapping: strong signal only when close (< 0.5m)
    # HC-SR04 max range = 4.0m, effective for avoidance < 1.0m
    _US_MAX_RANGE = 4.0
    if obstacle_distance < 0 or obstacle_distance >= _US_MAX_RANGE:
        s[18] = 0.0
    else:
        # Inverse: closer = higher value. Nonlinear for urgency.
        # 4.0m → 0.0, 1.0m → 0.25, 0.5m → 0.5, 0.1m → 0.9, 0.0m → 1.0
        proximity = 1.0 - min(obstacle_distance / _US_MAX_RANGE, 1.0)
        # Square for urgency: gentle at distance, steep when close
        s[18] = float(np.clip(proximity ** 0.5, 0.0, 1.0))
    
    # Channels 19-47: zeros (padding, matches Bridge)
    return s


def encode_sensory_bittle(joint_angles_rad, cpg_phase, imu_data,
                         obstacle_distance: float = -1.0):
    """
    Encode Bittle sensor data for hardware-matched SNN input.

    Layout (19 channels, Bittle 8-DOF):
      Channels  0-7:   8 joint angles (rad/pi + 0.5 -> [0,1])
      Channels  8-9:   2 CPG phase (sin, cos)
      Channels 10-13:  4 IMU (pitch/90, roll/90, yaw/180, upright)
      Channel  14:     1 ultrasonic proximity
      Channels 15-18:  4 padding (zeros)
    = 19 channels total

    Same structure as Freenove encode_sensory_hardware_matched() but
    adapted for 8-DOF (2 joints per leg, no abduction).
    """
    s = np.zeros(19, dtype=np.float32)

    # Channels 0-7: Joint angles normalized to [0, 1]
    n_joints = min(8, len(joint_angles_rad))
    for i in range(n_joints):
        s[i] = np.clip(joint_angles_rad[i] / np.pi + 0.5, 0.0, 1.0)

    # Channels 8-9: CPG phase
    s[8:10] = cpg_phase

    # Channels 10-13: IMU (same as Freenove encoding)
    s[10] = imu_data['pitch'] / 90.0 * 0.5 + 0.5
    s[11] = imu_data['roll'] / 90.0 * 0.5 + 0.5
    s[12] = imu_data['yaw'] / 180.0 * 0.5 + 0.5
    s[13] = imu_data['upright']

    # Channel 14: Ultrasonic proximity
    _US_MAX_RANGE = 4.0
    if obstacle_distance < 0 or obstacle_distance >= _US_MAX_RANGE:
        s[14] = 0.0
    else:
        proximity = 1.0 - min(obstacle_distance / _US_MAX_RANGE, 1.0)
        s[14] = float(np.clip(proximity ** 0.5, 0.0, 1.0))

    # Channels 15-18: padding
    return s


def extract_imu_from_mujoco(qpos):
    """
    Extract IMU-equivalent data from MuJoCo qpos quaternion.
    
    Args:
        qpos: MuJoCo qpos array (at least 7 elements: 3 pos + 4 quat)
    
    Returns:
        dict with pitch, roll, yaw (degrees), upright (0-1)
    """
    import math
    w, x, y, z = qpos[3:7]
    
    # Quaternion to euler
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    tilt = math.sqrt(float(pitch)**2 + float(roll)**2)
    upright = max(0.0, 1.0 - tilt / 45.0)
    
    return {
        'pitch': float(pitch),
        'roll': float(roll),
        'yaw': float(yaw),
        'upright': float(upright),
    }


# ================================================================
# MUJOCO CREATURE
# ================================================================

class MuJoCoCreature:
    """Kreatur mit SNN-Gehirn in MuJoCo-Welt."""

    # Anzahl SNN-Schritte pro Creature-Step
    # 6 suffices for 2 propagation cycles (Relay->Hidden->Output = 3 steps)
    SNN_SUBSTEPS = 6
    
    # Neuronen pro Sensor-Wert (Population Coding)
    NEURONS_PER_SENSOR = 8
    
    # Neurons per motor joint (push/pull pairs)
    NEURONS_PER_MOTOR = 6  # 3 push + 3 pull

    # Actor-Critic integration (set by builder or externally)
    actor_critic: Optional['ActorCriticIntegration'] = None

    def __init__(self, genome: Genome, snn: SNNController,
                 world, body_name: str = "creature_s0",
                 creature_name: str = "creature"):
        """
        Args:
            genome: Kreatur-Morphologie
            snn: Fertig verdrahteter SNN-Controller
            world: MuJoCoWorld Instanz
            body_name: Name des Root-Body in MuJoCo
            creature_name: Prefix fuer MJCF-Namen
        """
        self.genome = genome
        self.snn = snn
        self.world = world
        self.body_name = body_name
        self.creature_name = creature_name

        # Cognitive Brain (wird vom Builder gesetzt)
        self.brain: Optional[CognitiveBrain] = None

        # Evolved plasticity rule (fallback when no CognitiveBrain)
        self.plasticity_rule: Optional[EvolvedPlasticityRule] = None
        if genome.plasticity_genome:
            pg = PlasticityGenome(**genome.plasticity_genome)
            self.plasticity_rule = EvolvedPlasticityRule(pg)

        # Hardware-matched sensor mode (set by builder)
        self._hardware_sensors = False
        self._hardware_n_input = 48  # default Freenove channel count

        # Sensor/Motor Dimensionen
        self._n_sensor_channels = self._count_sensor_channels()
        self._n_motors = world.n_actuators if hasattr(world, '_model') and world._model else genome.n_joints

        # qpos/qvel offset: actuated joints may not start at index 7
        # (e.g. Bittle has a non-actuated neck joint before leg joints).
        # Formula: offset = nq - n_actuators (actuated joints are last in qpos).
        if hasattr(world, '_model') and world._model:
            self._qpos_offset = world._model.nq - self._n_motors
            self._qvel_offset = world._model.nv - self._n_motors
        else:
            self._qpos_offset = 7   # default: free joint (3 pos + 4 quat)
            self._qvel_offset = 6   # default: free joint (3 vel + 3 angvel)
        self.n_input_neurons = self._n_sensor_channels * self.NEURONS_PER_SENSOR
        self.n_output_neurons = self._n_motors * self.NEURONS_PER_MOTOR

        # State
        self._step_count = 0
        self._energy_spent = 0.0
        self._start_position: Optional[np.ndarray] = None
        self._standing_steps = 0
        self._last_controls: Optional[List[float]] = None
        self._prediction_error = 0.0

        # Record Startposition nach erstem Forward
        self._init_start_pos()

    def _count_sensor_channels(self) -> int:
        """Zaehle Sensor-Kanaele aus MuJoCo Sensor-Data."""
        # In hardware-matched mode, channel count comes from profile
        if self._hardware_sensors:
            return self._hardware_n_input
        # Standard mode: position(3) + velocity(3) + orientation(3) + 
        #               height(1) + upright(1) + forward_vel(1) +
        #               joint_angles(n) + joint_velocities(n) +
        #               visual_target_heading(1) + visual_target_distance(1)
        if hasattr(self.world, '_model') and self.world._model:
            n_joints = self.world.n_actuators
        else:
            n_joints = len(self.genome.joints)
        return 3 + 3 + 3 + 1 + 1 + 1 + n_joints + n_joints + 2  # +2 = vision (heading + distance)

    def _init_start_pos(self):
        """Startposition initialisieren."""
        try:
            if self.world._built:
                data = self.world.get_sensor_data(self.body_name)
                self._start_position = data['position'].copy()
        except Exception:
            pass

    def _extract_raw_sensors(self, data: dict) -> list:
        """Extrahiert rohe Sensor-Werte fuer CognitiveBrain."""
        if self._hardware_sensors:
            return self._extract_raw_sensors_hardware(data)
        values = []
        pos = data.get('position', np.zeros(3))
        values.extend([pos[0] / 10, pos[1] / 10, pos[2] / 10])
        vel = data.get('velocity', np.zeros(3))
        values.extend([vel[0] / 5, vel[1] / 5, vel[2] / 5])
        euler = data.get('orientation_euler', np.zeros(3))
        values.extend([euler[0] / np.pi, euler[1] / np.pi, euler[2] / np.pi])
        values.append(data.get('height', 0) / 2.0)
        values.append(data.get('upright', 0))
        values.append(data.get('forward_velocity', 0) / 5.0)
        for a in data.get('joint_angles', []):
            values.append(np.clip(a / 2.0, -1, 1))
        for v in data.get('joint_velocities', []):
            values.append(np.clip(v / 10.0, -1, 1))
        # Vision channels: target heading (-1..+1) and distance (0..1)
        values.append(getattr(self, '_visual_target_heading', 0.0))
        values.append(getattr(self, '_visual_target_distance', 0.0))
        return values

    def _extract_raw_sensors_hardware(self, data: dict) -> list:
        """Extract raw sensor values in hardware-matched layout for CognitiveBrain."""
        # Use the same 48-channel layout as the Pi Bridge
        n_act = self.world.n_actuators if hasattr(self.world, '_model') else 12
        joint_angles_rad = self.world._data.qpos[self._qpos_offset:self._qpos_offset+n_act] if hasattr(self.world, '_data') else np.zeros(n_act)
        
        # CPG phase from creature state
        cpg_cmd = getattr(self, '_cpg_cmd', None)
        if cpg_cmd is not None:
            # Reconstruct phase from CPG command pattern (approximate)
            import math
            cpg_phase_val = getattr(self, '_cpg_phase_input', np.array([0.0, 1.0]))
        else:
            cpg_phase_val = np.array([0.0, 1.0])
        
        # IMU from MuJoCo quaternion
        imu_data = extract_imu_from_mujoco(self.world._data.qpos)
        
        sensory = encode_sensory_bittle(joint_angles_rad, cpg_phase_val, imu_data) \
            if n_act <= 8 else \
            encode_sensory_hardware_matched(joint_angles_rad, cpg_phase_val, imu_data)
        return sensory.tolist()

    # ================================================================
    # SENSE
    # ================================================================

    def get_sensor_input(self) -> torch.Tensor:
        """
        MuJoCo Sensoren -> SNN Input Tensor.
        
        In hardware-matched mode: uses encode_sensory_hardware_matched()
        to produce the exact same 48-channel layout as the Pi Bridge.
        
        In standard mode: uses Population Coding with Gaussian Tuning Curves.
        """
        if self._hardware_sensors:
            return self._get_sensor_input_hardware()
        
        data = self.world.get_sensor_data(self.body_name)
        
        # Build sensor vector (all normalized to ~-1..1)
        sensor_values = []
        
        # Position (normalisiert auf -10..10 Welt)
        pos = data['position']
        sensor_values.extend([pos[0] / 10, pos[1] / 10, pos[2] / 10])
        
        # Velocity (normalisiert auf -5..5 m/s)
        vel = data['velocity']
        sensor_values.extend([vel[0] / 5, vel[1] / 5, vel[2] / 5])
        
        # Orientation (Euler, already ~-pi..pi -> normalized)
        euler = data['orientation_euler']
        sensor_values.extend([euler[0] / np.pi, euler[1] / np.pi, euler[2] / np.pi])
        
        # Height (0..2m)
        sensor_values.append(data['height'] / 2.0)
        
        # Upright (-1..1, MuJoCo provides Z component directly)
        sensor_values.append(data['upright'])
        
        # Forward velocity (-5..5)
        sensor_values.append(data['forward_velocity'] / 5.0)
        
        # Joint angles (normalisiert auf Joint-Limits)
        angles = data['joint_angles']
        for a in angles:
            sensor_values.append(np.clip(a / 2.0, -1, 1))
        
        # Joint velocities
        vels = data['joint_velocities']
        for v in vels:
            sensor_values.append(np.clip(v / 10.0, -1, 1))

        # --- Vision channels (Issue #76d): Target object direction + distance ---
        _vis_heading = getattr(self, '_visual_target_heading', 0.0)
        _vis_distance = getattr(self, '_visual_target_distance', 0.0)
        sensor_values.append(np.clip(_vis_heading, -1.0, 1.0))
        sensor_values.append(np.clip(_vis_distance, -1.0, 1.0))

        # --- Obstacle proximity (Issue #103): Rangefinder / Ultrasonic ---
        obstacle_dist = data.get('obstacle_distance', -1.0)
        self._obstacle_distance = obstacle_dist  # Store for reward computation
        _us_max = 4.0
        if obstacle_dist < 0 or obstacle_dist >= _us_max:
            sensor_values.append(0.0)
        else:
            proximity = 1.0 - min(obstacle_dist / _us_max, 1.0)
            sensor_values.append(float(np.clip(proximity ** 0.5, 0.0, 1.0)))

        # Population Coding -> SNN Input
        n = self.snn.config.n_neurons
        snn_input = torch.zeros(n, device=self.snn.device, dtype=self.snn.dtype)
        
        threshold = self.snn.config.v_threshold
        idx = 0
        for val in sensor_values:
            encoded = rate_encode(float(val), self.NEURONS_PER_SENSOR,
                                  min_val=-1.0, max_val=1.0,
                                  gain=threshold * 2.0)
            end = min(idx + self.NEURONS_PER_SENSOR, n)
            snn_input[idx:end] = encoded[:end - idx]
            idx = end
            if idx >= n:
                break

        # Tonischer Hintergrundstrom auf Hidden
        tonic = getattr(self.snn, '_hidden_tonic_current', 0.02)
        if tonic > 0 and 'hidden' in self.snn.populations:
            hidden_ids = self.snn.populations['hidden']
            snn_input[hidden_ids] += tonic

        return snn_input

    def _get_sensor_input_hardware(self) -> torch.Tensor:
        """
        Hardware-matched sensor encoding for Freenove.
        
        Uses encode_sensory_hardware_matched() to produce the EXACT same
        48-channel layout as freenove_bridge.py. The 48 raw values are
        then population-coded into n_input SNN neurons.
        
        This ensures that a brain trained in simulation can be directly
        transferred to the Raspberry Pi without sensor mismatch.
        """
        n_act = self.world.n_actuators if hasattr(self.world, '_model') else 12
        joint_angles_rad = self.world._data.qpos[self._qpos_offset:self._qpos_offset+n_act] if hasattr(self.world, '_data') else np.zeros(n_act)
        
        # CPG phase input (stored by training loop)
        cpg_phase_val = getattr(self, '_cpg_phase_input', np.array([0.0, 1.0], dtype=np.float32))
        
        # IMU from MuJoCo quaternion
        imu_data = extract_imu_from_mujoco(self.world._data.qpos)
        
        # Ultrasonic rangefinder (Channel 18, Issue #103)
        obstacle_dist = self.world._read_rangefinder() if hasattr(self.world, '_read_rangefinder') else -1.0
        # Store for training loop access (reward computation)
        self._obstacle_distance = obstacle_dist
        
        # Encode in Bridge-identical format — dispatch by DOF
        if n_act <= 8:
            sensory = encode_sensory_bittle(
                joint_angles_rad, cpg_phase_val, imu_data,
                obstacle_distance=obstacle_dist)
        else:
            sensory = encode_sensory_hardware_matched(
                joint_angles_rad, cpg_phase_val, imu_data,
                obstacle_distance=obstacle_dist)
        
        # Population Coding: 48 values -> n_input SNN neurons
        n = self.snn.config.n_neurons
        snn_input = torch.zeros(n, device=self.snn.device, dtype=self.snn.dtype)
        
        threshold = self.snn.config.v_threshold
        n_input = len(self.snn.populations.get('input', []))
        neurons_per_channel = max(1, n_input // len(sensory)) if n_input > 0 else 1
        
        idx = 0
        for val in sensory:
            encoded = rate_encode(float(val), neurons_per_channel,
                                  min_val=0.0, max_val=1.0,  # Hardware values are 0-1
                                  gain=threshold * 2.0)
            end = min(idx + neurons_per_channel, n_input, n)
            actual_len = end - idx
            if actual_len > 0:
                snn_input[idx:end] = encoded[:actual_len]
            idx = end
            if idx >= n:
                break
        
        # Tonic current on hidden neurons
        tonic = getattr(self.snn, '_hidden_tonic_current', 0.02)
        if tonic > 0 and 'hidden' in self.snn.populations:
            hidden_ids = self.snn.populations['hidden']
            snn_input[hidden_ids] += tonic
        
        return snn_input

    # ================================================================
    # THINK
    # ================================================================

    def think(self, sensor_input: torch.Tensor) -> torch.Tensor:
        """
        SNN-Simulation: mehrere Substeps fuer Spike-Propagation.
        
        Returns:
            Akkumulierte Output-Spikes (fuer Motor-Decoding)
        """
        n_out_start = self.n_input_neurons
        n_out_end = n_out_start + self.n_output_neurons
        n_out_end = min(n_out_end, self.snn.config.n_neurons)
        actual_n_out = n_out_end - n_out_start

        output_spike_count = torch.zeros(actual_n_out,
                                          device=self.snn.device,
                                          dtype=self.snn.dtype)

        # Accumulate GrC spikes across substeps for cerebellar learning
        n = self.snn.config.n_neurons
        self._accumulated_spikes = torch.zeros(n, device=self.snn.device,
                                                dtype=self.snn.dtype)

        for t in range(self.SNN_SUBSTEPS):
            spikes = self.snn.step(sensor_input)
            output_spike_count += spikes[n_out_start:n_out_end].float()
            self._accumulated_spikes += spikes.float()

        return output_spike_count

    # ================================================================
    # ACT
    # ================================================================

    def apply_motor_output(self, output_spikes: torch.Tensor):
        """
        Output-Spikes -> MuJoCo Actuator Controls.
        
        Wenn CPG aktiv (_cpg_base gesetzt): CPG-Basis + SNN-Korrekturen
        Ohne CPG: Population Voting direkt.
        """
        snn_controls = decode_motor_spikes(
            output_spikes, self.NEURONS_PER_MOTOR, self.SNN_SUBSTEPS)

        # Pad/truncate auf Anzahl Actuatoren
        n_act = self.world.n_actuators
        while len(snn_controls) < n_act:
            snn_controls.append(0.0)
        snn_controls = snn_controls[:n_act]

        # Motor output routing: CPG+SNN fusion OR pure SNN
        if hasattr(self, '_cpg_base') and self._cpg_base is not None:
            # OLD MODE: CPG base + SNN corrections (v0.3.0)
            cpg_base = self._cpg_base

            if self.actor_critic is not None:
                _quat = self.world._data.qpos[3:7]
                _up = max(0, 1.0 - 2.0 * (_quat[1]**2 + _quat[2]**2))
                compute_fn = getattr(self.actor_critic, 'compute_corrections',
                                     getattr(self.actor_critic, 'compute_snn_corrections', None))
                if compute_fn:
                    try:
                        snn_corr = compute_fn(snn_controls, upright=_up)
                    except TypeError:
                        snn_corr = compute_fn(snn_controls)
                else:
                    snn_corr = np.zeros(len(cpg_base))
                controls = list(np.clip(cpg_base + snn_corr, -1.0, 1.0))
            else:
                controls = list(np.clip(cpg_base, -1.0, 1.0))
        elif self.actor_critic is not None:
            _quat2 = self.world._data.qpos[3:7]
            _up2 = max(0, 1.0 - 2.0 * (_quat2[1]**2 + _quat2[2]**2))
            try:
                corrections = self.actor_critic.compute_corrections(snn_controls, upright=_up2)
            except TypeError:
                corrections = self.actor_critic.compute_corrections(snn_controls)
            controls = list(np.clip(corrections, -1.0, 1.0))
        else:
            controls = snn_controls

        # Scale cerebellum/SNN output
        per_joint_scale = getattr(self, 'per_joint_scale', None)
        motor_scale = getattr(self, 'motor_scale', 1.0)
        if per_joint_scale is not None:
            controls = [c * s for c, s in zip(controls, per_joint_scale)]
        elif motor_scale != 1.0:
            controls = [c * motor_scale for c in controls]

        # Blend with spinal CPG (innate rhythmic pattern)
        # Biology: CPG provides the base rhythm. Cerebellar corrections
        # are ADDED to the CPG output, not weighted against it.
        # The CPG weight controls how much the CPG contributes,
        # but corrections are always fully applied on top.
        #
        # Old (broken): cpg * 0.9 + correction * 0.1 → corrections = 10%
        # New: cpg * 0.9 + correction * 1.0 → corrections at full strength
        #
        # This matches biology: the cerebellum modulates the CPG output
        # via the reticulospinal tract, it doesn't compete with it.
        cpg_cmd = getattr(self, '_cpg_cmd', None)
        cpg_weight = getattr(self, '_cpg_weight', 0.0)
        # Task #94 diagnosis: keep the SNN's own contribution before it is mixed with
        # the CPG.  The CPG is damped by cpg_weight; `controls` (the SNN/cerebellar
        # output) is added at FULL weight.  If the SNN has learned to trot straight --
        # and it sees the IMU, so it can -- it will cancel any turn the CPG commands
        # while leaving forward motion untouched.  That is exactly the signature we
        # measure: gait matches the damped chain, steering does not.  Read-only.
        self._dbg_snn_controls = list(controls)
        if cpg_cmd is not None and cpg_weight > 0:
            controls = [cpg * cpg_weight + cb
                       for cpg, cb in zip(cpg_cmd, controls)]
        self._dbg_mixed_controls = list(controls)

        # Add reflex commands (emergency overrides)
        reflex_cmd = getattr(self, '_reflex_cmd', None)
        if reflex_cmd is not None:
            reflex_scale = getattr(self, 'reflex_scale', 1.0)
            controls = [c + r * reflex_scale for c, r in zip(controls, reflex_cmd)]

        # Terrain reflex corrections (Phase B ATR)
        terrain_corr = getattr(self, '_terrain_corr', None)
        if terrain_corr is not None:
            controls = [c + t for c, t in zip(controls, terrain_corr)]

        # Spinal segments: muscle tone + stretch reflex + Golgi tendon organ
        spinal_seg = getattr(self, '_spinal_segments', None)
        if spinal_seg is not None:
            joint_pos = self.world._data.qpos[self._qpos_offset:self._qpos_offset+n_act] if hasattr(self.world, '_data') else np.zeros(n_act)
            controls = spinal_seg.process(
                np.array(controls), joint_pos,
                sim_dt=getattr(self, '_sim_dt', 0.005)
            ).tolist()
        else:
            # Clip to actuator range (not [-1,1] which truncates position actuators)
            pd = getattr(self, '_pd_controller', None)
            if pd is not None and pd.get('native_position', False):
                lo = pd['lo']
                hi = pd['hi']
                controls = [float(np.clip(c, lo[i], hi[i])) for i, c in enumerate(controls)]
            else:
                controls = [np.clip(c, -1.0, 1.0) for c in controls]

        self.world.set_controls(np.array(controls))
        self._last_controls = controls
        self._energy_spent += sum(abs(c) for c in controls)

        # PD Controller hook: convert position targets -> torques (or native position)
        pd = getattr(self, '_pd_controller', None)
        if pd is not None:
            n = self.world.n_actuators
            raw_ctrl = self.world._data.ctrl[:n].copy()
            standing = pd['standing']
            base_scale = pd.get('scale', 0.4)
            fallen_scale = pd.get('fallen_scale', 1.5)
            _quat = self.world._data.qpos[3:7]
            _upright = max(-1, min(1, 1.0 - 2.0 * (_quat[1]**2 + _quat[2]**2)))
            _urgency = max(0.0, min(1.0, (1.0 - _upright) / 2.0))
            scale = base_scale + (fallen_scale - base_scale) * _urgency
            target_q = standing + raw_ctrl * scale

            # Task #92/#94: re-apply the PURE turning component UNDAMPED.
            # raw_ctrl was damped by cpg_weight (mix) and by `scale` (amplitude), which
            # together shrink a 7.41 deg/s turning gait to ~1 (measured, scripts/
            # prove_motor_chain.py). The forward gait MUST stay damped for stability,
            # but the turn is a steering command, not part of the base rhythm. The
            # controller exposed the turn-only delta as _steer_delta (already a delta
            # around the standing pose, so it adds straight onto target_q). Weight is
            # tunable via _steer_undamped_weight; 0.0 => bit-identical (this whole block
            # is skipped). Joint deltas only -> works on the real Bittle.
            _steer_delta = getattr(self, '_steer_delta', None)
            _steer_w = getattr(self, '_steer_undamped_weight', 0.0)
            if _steer_delta is not None and _steer_w != 0.0:
                _sd = np.asarray(_steer_delta, dtype=float)
                if _sd.shape[0] >= n:
                    target_q = target_q + _sd[:n] * _steer_w
            if pd.get('native_position', False):
                # Position actuators: send target angles directly.
                # raw_ctrl contains DELTAS from standing.
                # Apply OpenCat balance corrections if available.
                balance = getattr(self, '_opencat_balance', None)
                if balance is not None:
                    quat = self.world._data.qpos[3:7]
                    balance_corr = balance.compute(quat)
                    target_q = target_q + balance_corr[:n]
                self.world._data.ctrl[:n] = np.clip(target_q, pd['lo'], pd['hi'])
            else:
                # Torque actuators: PD controller
                current_q = self.world._data.qpos[self._qpos_offset:self._qpos_offset+n]
                current_v = self.world._data.qvel[self._qvel_offset:self._qvel_offset+n]
                torques = pd['kp'] * (target_q - current_q) - pd['kd'] * current_v
                torques = np.clip(torques, pd['lo'], pd['hi'])
                self.world._data.ctrl[:n] = torques

    # ================================================================
    # STEP (Sense-Think-Act)
    # ================================================================

    def step(self, reward_signal: float = 0.0,
             extra_sensor_data: Optional[Dict] = None) -> dict:
        """
        Vollstaendiger Sense-Think-Act Zyklus.
        """
        if self._start_position is None:
            self._init_start_pos()

        import time as _t

        # 1. Sense
        _t0 = _t.perf_counter()
        sensor_data = {}
        try:
            sensor_data = self.world.get_sensor_data(self.body_name)
        except Exception:
            pass

        # 2. Sense -> SNN Encoding
        sensor_input = self.get_sensor_input()
        _t1 = _t.perf_counter()

        # 3. Think
        output_spikes = self.think(sensor_input)
        _t2 = _t.perf_counter()

        # 4. Act
        _t_act0 = _t.perf_counter()
        self.apply_motor_output(output_spikes)
        _t_act1 = _t.perf_counter()

        # 5. Physik-Step (MuJoCo)
        self.world.step()
        _t3 = _t.perf_counter()

        # 6. Cognitive cycle
        brain_result = {}
        if self.brain:
            if self.actor_critic is not None:
                self.brain.protect_snn_weights = True
            else:
                self.brain.protect_snn_weights = False
            raw_sensors = self._extract_raw_sensors(sensor_data)
            brain_result = self.brain.process(
                sensor_values=raw_sensors,
                snn_input=sensor_input,
                output_spikes=output_spikes,
                controls=self._last_controls or [],
                external_reward=reward_signal,
                is_fallen=self.is_fallen(),
                extra_sensor_data=extra_sensor_data,
            )
        _t4 = _t.perf_counter()

        # 7. Tracking
        self._step_count += 1
        if not self.is_fallen():
            self._standing_steps += 1

        # Profile logging (every 5000 steps)
        if self._step_count % 5000 == 0:
            print(f'  [CREATURE PROFILE step {self._step_count}] '
                  f'sense:{(_t1-_t0)*1000:.1f}ms  '
                  f'think:{(_t2-_t1)*1000:.1f}ms  '
                  f'act:{(_t_act1-_t_act0)*1000:.1f}ms  '
                  f'mujoco:{(_t3-_t_act1)*1000:.1f}ms  '
                  f'brain:{(_t4-_t3)*1000:.1f}ms  '
                  f'extract:{(_t3-_t2)*1000-((_t_act1-_t_act0)+(_t3-_t_act1))*1000:.1f}ms')

        return {
            'step': self._step_count,
            'total_spikes': int(self.snn.spikes.sum()),
            'output_spikes': int(output_spikes.sum()),
            'controls': self._last_controls,
            'reward': reward_signal,
            'brain': brain_result,
        }

    # ================================================================
    # EVALUATION
    # ================================================================

    def get_position(self) -> np.ndarray:
        try:
            data = self.world.get_sensor_data(self.body_name)
            return data['position'].copy()
        except Exception:
            return np.zeros(3)

    def get_distance_traveled(self) -> float:
        if self._start_position is None:
            return 0.0
        pos = self.get_position()
        diff = pos - self._start_position
        return float(np.sqrt(diff[0]**2 + diff[1]**2))

    def is_fallen(self) -> bool:
        try:
            height = float(self.world._data.qpos[2])
            qw, qx, qy, qz = self.world._data.qpos[3:7]
            upright = max(0, 1.0 - 2.0 * (qx*qx + qy*qy))
            h_thresh = getattr(self, '_fallen_height_threshold', 0.08)
            return height < h_thresh or upright < 0.3
        except Exception:
            return True

    def get_energy_spent(self) -> float:
        return self._energy_spent

    def get_state(self) -> dict:
        try:
            data = self.world.get_sensor_data(self.body_name)
        except Exception:
            data = {}
        
        return {
            'step': self._step_count,
            'position': data.get('position', np.zeros(3)).tolist(),
            'height': data.get('height', 0),
            'upright': data.get('upright', 0),
            'velocity': data.get('forward_velocity', 0),
            'distance': self.get_distance_traveled(),
            'energy': self._energy_spent,
            'fallen': self.is_fallen(),
            'standing_ratio': self._standing_steps / max(self._step_count, 1),
            'controls': self._last_controls,
            'total_spikes': int(self.snn.spikes.sum()) if self.snn.spikes is not None else 0,
        }

    def apply_reward(self, reward: float):
        self.snn.apply_rstdp(reward_signal=reward)
        if reward > 0:
            self.snn.set_neuromodulator('ne', min(0.8, 0.3 + reward * 0.3))
        else:
            self.snn.set_neuromodulator('5ht', min(0.8, 0.5 + abs(reward) * 0.2))

    def reset(self):
        self.world.reset()
        self._step_count = 0
        self._energy_spent = 0.0
        self._standing_steps = 0
        self._last_controls = None
        self._start_position = None


# ================================================================
# BUILDER — Scalable Cerebellar Architecture
# ================================================================

class MuJoCoCreatureBuilder:
    """Factory: Genome -> MuJoCoCreature (komplett verdrahtet)."""

    @staticmethod
    def _compute_cerebellar_populations(n_hidden: int, n_actuators: int,
                                        motor_hidden_ratio: float = 0.30):
        """
        Compute cerebellar population sizes scaled to available hidden neurons.
        
        v0.7.1: Continuous scaling for ALL n_hidden values. No more cliff
        at n_hidden=500 that jumped from ~560 to 4308 total neurons.
        The same formula applies whether n_hidden is 172 or 5000.
        
        Layout of hidden neurons:
          - PkC + DCN: fixed at 2*n_actuators each (output interface)
          - Remaining: 70% cerebellum (GrC + GoC), 30% motorcortex (MH)
          - GrC:GoC ratio within cerebellum: 85:15
        
        Returns:
            dict with population sizes: n_granule, n_golgi, n_purkinje, n_dcn
        """
        n_purkinje = n_actuators * 2  # 2 per actuator (push/pull)
        n_dcn = n_actuators * 2       # same as PkC
        
        # Reserve space for PkC + DCN first (fixed output interface)
        fixed_neurons = n_purkinje + n_dcn  # e.g. 24 + 24 = 48 for 12 actuators
        available = max(4, n_hidden - fixed_neurons)
        
        # Reserve motor_hidden_ratio as free hidden layer (motorcortex)
        free_hidden_ratio = motor_hidden_ratio
        cerebellum_budget = max(4, int(available * (1.0 - free_hidden_ratio)))
        
        # Split cerebellum budget: 85% GrC (expansion), 15% GoC (inhibition)
        n_golgi = max(4, int(cerebellum_budget * 0.15))
        n_granule = max(4, cerebellum_budget - n_golgi)
        
        return {
            'n_granule': n_granule,
            'n_golgi': n_golgi,
            'n_purkinje': n_purkinje,
            'n_dcn': n_dcn,
        }

    @staticmethod
    def build(genome: Genome, world=None,
              n_hidden_neurons: int = 1000,
              device: str = 'cpu',
              creature_name: str = 'creature',
              xml_path: str = None,
              xml_string: str = None,
              hardware_sensors: bool = False,
              no_vision: bool = False,
              profile: dict = None,
              izh_cerebellum: bool = True,
              protect_cerebellum: bool = True) -> 'MuJoCoCreature':
        """
        Baut eine MuJoCo-Kreatur aus Genome (oder fertigem XML).
        
        Args:
            genome: Kreatur-Morphologie
            world: MuJoCoWorld (wird erstellt wenn None)
            n_hidden_neurons: Anzahl Hidden-Neuronen im SNN
            device: 'cpu' oder 'cuda'
            creature_name: Name-Prefix
            xml_path: Pfad zu fertigem MJCF XML
            xml_string: XML as string
            hardware_sensors: Use hardware-matched sensor encoding (Bridge v2.5 layout)
            no_vision: Disable visual heading/distance channels
            profile: Creature profile dict (from profile.json). If provided,
                     overrides n_hidden_neurons with profile['snn']['n_hidden'].
            izh_cerebellum: Enable Izhikevich dynamics on cerebellar populations.
                     Default True (v0.5.0). Set False for v0.4.3 LIF-LTC behavior.
            protect_cerebellum: Protect cerebellar populations from R-STDP.
                     Default True (v0.5.0). Set False for v0.4.3 behavior where
                     R-STDP could modify cerebellar weights.
                     overrides n_hidden_neurons with profile['snn']['n_hidden'].
            
        Returns:
            MuJoCoCreature mit verdrahtetem SNN
        """
        from src.body.mujoco_world import MuJoCoWorld
        from src.body.mjcf_generator import MJCFGenerator

        # --- Profile-driven topology ---
        snn_profile = profile.get('snn', {}) if profile else {}
        if snn_profile.get('n_hidden') is not None:
            n_hidden_neurons = snn_profile['n_hidden']
        profile_n_input = snn_profile.get('n_input', None)
        profile_n_output = snn_profile.get('n_output', None)

        # 1. MuJoCo Welt
        if world is None:
            world = MuJoCoWorld(render=False)

        # 2. Load MJCF
        if xml_string:
            world.load_from_xml_string(xml_string)
        elif xml_path:
            import os
            world.load_from_xml_path(xml_path)
        else:
            world.add_ground()
            body_xml, act_xml, sens_xml = MJCFGenerator.generate(
                genome, name=creature_name)
            world.load_creature_xml(body_xml, act_xml, sens_xml)
            world._build()

        # 3. SNN topology — Scalable Cerebellar Architecture v0.4.3
        #
        # Profile-driven: if profile.json has snn.n_input, use that.
        # Hardware-matched: if --hardware-sensors, use 48 raw channels.
        # Standard: compute from MuJoCo sensor channels + population coding.
        
        if hardware_sensors and profile_n_input is not None:
            # Hardware mode with profile: n_input from profile (e.g. 48 for Freenove)
            n_input = profile_n_input
            n_sensor_channels = n_input  # 1:1, no population coding expansion
        elif hardware_sensors:
            # Hardware mode without profile: compute from creature
            n_sensor_channels = 48  # default Freenove layout
            n_input = n_sensor_channels
        else:
            # Standard mode: ALWAYS compute from MuJoCo, ignore profile n_input
            n_vision = 0 if no_vision else 2
            n_sensor_channels = 12 + 2 * world.n_actuators + n_vision
            n_input = n_sensor_channels * MuJoCoCreature.NEURONS_PER_SENSOR

        # Output neurons
        if profile_n_output is not None:
            n_output = profile_n_output
        else:
            n_output = world.n_actuators * MuJoCoCreature.NEURONS_PER_MOTOR

        _mh_ratio = snn_profile.get('motor_hidden_ratio', 0.30) if profile else 0.30

        # Cerebellar populations scaled to n_hidden
        cb_pops = MuJoCoCreatureBuilder._compute_cerebellar_populations(
            n_hidden_neurons, world.n_actuators, motor_hidden_ratio=_mh_ratio)
        n_granule = cb_pops['n_granule']
        n_golgi = cb_pops['n_golgi']
        n_purkinje = cb_pops['n_purkinje']
        n_dcn = cb_pops['n_dcn']

        # v0.7.0: Motor hidden neurons (motorcortex)
        # Ratio configurable via profile (default 30%).
        # Set to 0 for creatures using firmware-level gaits (OpenCat)
        # where SNN only needs cerebellar corrections.
        n_cerebellum = n_granule + n_golgi + n_purkinje + n_dcn
        n_motor_hidden = max(0, n_hidden_neurons - n_cerebellum) if _mh_ratio > 0 else 0

        # Total neuron count (now includes motor_hidden)
        total_neurons = n_input + n_output + n_granule + n_golgi + n_purkinje + n_dcn + n_motor_hidden

        # Neuron ID ranges
        grc_start = n_input + n_output
        goc_start = grc_start + n_granule
        pkc_start = goc_start + n_golgi
        dcn_start = pkc_start + n_purkinje
        motor_hidden_start = dcn_start + n_dcn

        print(f'  SNN Topology: {total_neurons} neurons '
              f'({n_input}i + {n_output}o + {n_granule}GrC + {n_golgi}GoC + '
              f'{n_purkinje}PkC + {n_dcn}DCN + {n_motor_hidden}MH)')
        if hardware_sensors:
            print(f'  Sensor mode: HARDWARE-MATCHED (Bridge v2.5 layout, {n_input} channels)')
        print(f'  Topology: {n_hidden_neurons} hidden '
              f'(GrC:{n_granule} GoC:{n_golgi} PkC:{n_purkinje} DCN:{n_dcn} MH:{n_motor_hidden})')

        # Update CerebellarConfig with scaled values
        from src.brain.cerebellar_learning import CerebellarConfig
        cb_cfg = CerebellarConfig()
        cb_cfg.n_granule = n_granule
        cb_cfg.n_golgi = n_golgi
        cb_cfg.n_purkinje = n_purkinje
        cb_cfg.n_dcn = n_dcn
        # Scale connectivity proportionally to GrC count.
        # For large GrC counts the formulas converge to standard values
        # (e.g. 4000 GrC → grc_goc_prob=0.05, same as old hardcoded).
        # For small GrC counts they increase connectivity to maintain
        # information flow through the cerebellar circuit.
        cb_cfg.grc_goc_prob = min(0.3, 0.05 * (4000 / max(n_granule, 1)))
        cb_cfg.pf_pkc_prob = min(0.8, 0.4 * (4000 / max(n_granule, 1)))
        cb_cfg.mf_per_granule = min(4, max(2, n_input // max(n_granule, 1)))

        # 4. SNN Controller
        snn = SNNController(SNNConfig(
            n_neurons=total_neurons,
            connectivity_prob=0.0,
            homeostatic_interval=200,
            device=device,
        ))

        # === Populations — Cerebellar Architecture ===
        mf_ids = torch.arange(0, n_input)
        snn.define_population('input', mf_ids)
        snn.define_population('mossy_fibers', mf_ids)

        out_ids = torch.arange(n_input, n_input + n_output)
        snn.define_population('output', out_ids)

        grc_ids = torch.arange(grc_start, grc_start + n_granule)
        snn.define_population('granule_cells', grc_ids)

        goc_ids = torch.arange(goc_start, goc_start + n_golgi)
        snn.define_population('golgi_cells', goc_ids)

        pkc_ids = torch.arange(pkc_start, pkc_start + n_purkinje)
        snn.define_population('purkinje_cells', pkc_ids)

        dcn_ids = torch.arange(dcn_start, dcn_start + n_dcn)
        snn.define_population('dcn', dcn_ids)

        # v0.7.0: Motor hidden population (motorcortex)
        # These neurons learn motor patterns via R-STDP.
        # Biology: primary motor cortex — pattern generation.
        # NOT protected from R-STDP (unlike cerebellum).
        if n_motor_hidden > 0:
            mh_ids = torch.arange(motor_hidden_start, motor_hidden_start + n_motor_hidden)
            snn.define_population('motor_hidden', mh_ids)
            # Include in 'hidden' for tonic current
            all_hidden_ids = torch.cat([grc_ids, mh_ids])
            snn.define_population('hidden', all_hidden_ids)
        else:
            mh_ids = torch.tensor([], dtype=torch.long)
            snn.define_population('hidden', grc_ids)

        # === Connectivity — Biologically structured ===
        # MF -> GrC: each GrC receives mf_per_granule MF inputs
        n_mf = len(mf_ids)
        mf_per_grc = cb_cfg.mf_per_granule
        src_list = []
        tgt_list = []
        for g in range(n_granule):
            mf_choices = torch.randint(0, n_mf, (mf_per_grc,))
            for m in mf_choices:
                src_list.append(mf_ids[m].item())
                tgt_list.append(grc_ids[g].item())
        if src_list:
            src_t = torch.tensor(src_list, device=device)
            tgt_t = torch.tensor(tgt_list, device=device)
            w = torch.rand(len(src_list), device=device) * 1.0 + 1.0
            new_idx = torch.stack([src_t, tgt_t])
            snn._weight_indices = new_idx
            snn._weight_values = w
            snn._eligibility = torch.zeros(len(src_list), device=device)

        # GrC -> GoC
        snn.connect_populations('granule_cells', 'golgi_cells',
                                prob=cb_cfg.grc_goc_prob, weight_range=(0.3, 0.8))

        # GoC -> GrC (inhibitory)
        snn.neuron_types[goc_ids] = -1.0
        snn.connect_populations('golgi_cells', 'granule_cells',
                                prob=0.02, weight_range=(0.2, 0.4))

        # MF -> GoC (scale for small input populations)
        _mf_goc_prob = max(0.1, 3.0 / max(1, len(mf_ids)))
        snn.connect_populations('mossy_fibers', 'golgi_cells',
                                prob=_mf_goc_prob, weight_range=(0.5, 1.0))

        # GrC -> legacy output (scale prob for small populations)
        _grc_out_prob = max(0.02, 3.0 / max(1, n_granule))
        snn.connect_populations('granule_cells', 'output',
                                prob=_grc_out_prob, weight_range=(0.3, 0.8))

        # Direct input -> output (corticospinal fast path)
        # Bypasses cerebellar expansion for immediate motor corrections.
        # Biology: corticospinal tract provides direct cortex->motoneuron
        # connections alongside the slower cerebellar pathway.
        # Critical for small SNNs where cerebellar expansion dilutes signal.
        _in_out_prob = max(0.2, 5.0 / max(1, len(mf_ids)))
        snn.connect_populations('input', 'output',
                                prob=_in_out_prob, weight_range=(0.8, 1.5))

        # v0.7.0: Motor hidden connectivity (motorcortex)
        # Input -> motor_hidden: sparse sensory feed
        # motor_hidden -> Output: motor pattern projection
        # These connections are the ones R-STDP can modify!
        if n_motor_hidden > 0:
            # MF -> motor_hidden (sensory input)
            # Scale prob so each MH neuron gets ~5 input connections minimum
            _mf_mh_prob = max(0.1, 5.0 / max(1, len(mf_ids)))
            snn.connect_populations('mossy_fibers', 'motor_hidden',
                                    prob=_mf_mh_prob, weight_range=(0.5, 1.5))
            # motor_hidden -> output (motor drive)
            _mh_out_prob = max(0.15, 5.0 / max(1, n_motor_hidden))
            snn.connect_populations('motor_hidden', 'output',
                                    prob=_mh_out_prob, weight_range=(0.8, 1.5))

            # v0.5.2: Bilateral symmetry for MH->Output weights (Issue #145)
            # Biology: bilateral animals have symmetric motor innervation at birth.
            # Without this, random MH->Output init creates L/R asymmetry that
            # R-STDP amplifies into drift (positive feedback loop).
            # Drift analysis (2026-04-20): CPG-only walks straight (7m/50k),
            # SNN runs drift consistently. All seeds show same rightward pattern.
            # Fix: average MH->Output weights between bilateral leg pairs.
            # Actuator layout depends on creature:
            #   12-DOF: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
            #   8-DOF:  RF(0-1), LF(2-3), RR(4-5), LR(6-7)
            _jpleg = world.n_actuators // 4
            if _jpleg == 3:  # 12-DOF: FL<->FR, RL<->RR
                _bilateral_pairs = [(0, 3), (1, 4), (2, 5),
                                    (6, 9), (7, 10), (8, 11)]
            elif _jpleg == 2:  # 8-DOF: RF<->LF, RR<->LR
                _bilateral_pairs = [(0, 2), (1, 3),
                                    (4, 6), (5, 7)]
            else:
                _bilateral_pairs = []
            _out_start = out_ids[0].item()
            _n_per_joint = max(1, n_output // world.n_actuators)
            _idx = snn._weight_indices
            _w = snn._weight_values
            # Only average MH->Output weights; GrC->Output and Input->Output
            # have different source populations and must not be mixed in.
            _mh_set = set(mh_ids.tolist())
            _is_mh_source = torch.tensor(
                [s.item() in _mh_set for s in _idx[0]], device=device)
            for _left_j, _right_j in _bilateral_pairs:
                _left_neurons = list(range(_out_start + _left_j * _n_per_joint,
                                           _out_start + (_left_j + 1) * _n_per_joint))
                _right_neurons = list(range(_out_start + _right_j * _n_per_joint,
                                            _out_start + (_right_j + 1) * _n_per_joint))
                for _ln, _rn in zip(_left_neurons, _right_neurons):
                    _left_mask = (_idx[1] == _ln) & _is_mh_source
                    _right_mask = (_idx[1] == _rn) & _is_mh_source
                    if _left_mask.any() and _right_mask.any():
                        _avg = (_w[_left_mask].mean() + _w[_right_mask].mean()) / 2.0
                        _w[_left_mask] = _avg
                        _w[_right_mask] = _avg
            print(f'  MH->Output bilateral symmetry enforced ({len(_bilateral_pairs)} joint pairs)')

            # motor_hidden recurrent (pattern memory)
            snn.connect_populations('motor_hidden', 'motor_hidden',
                                    prob=0.1, weight_range=(0.2, 0.6))

        # === Per-population membrane time constants ===
        snn._tau_base[grc_ids] = 5.0
        snn._tau_base[goc_ids] = 20.0
        snn._tau_base[pkc_ids] = 15.0
        snn._tau_base[dcn_ids] = 10.0
        if n_motor_hidden > 0:
            snn._tau_base[mh_ids] = 10.0  # Medium time constant (cortical)

        # === Thresholds ===
        snn._thresholds[grc_ids] = 0.5
        snn._thresholds[goc_ids] = 0.5
        snn._thresholds[pkc_ids] = 1.0
        snn._thresholds[dcn_ids] = 1.0
        snn._thresholds[out_ids] = 0.4
        if n_motor_hidden > 0:
            snn._thresholds[mh_ids] = 0.5  # Same as GrC
        snn._hidden_tonic_current = 0.015

        # === Per-population Izhikevich parameters (Issue #104) ===
        # Enable biologically accurate firing dynamics per cell type.
        # Can be disabled with izh_cerebellum=False for v0.4.3 compatibility.
        if izh_cerebellum:
            snn.set_izhikevich_params('granule_cells', a=0.02, b=0.2, c=-65, d=8)   # RS
            snn.set_izhikevich_params('golgi_cells',   a=0.02, b=0.2, c=-55, d=4)   # IB
            snn.set_izhikevich_params('purkinje_cells', a=0.02, b=0.2, c=-50, d=2)  # CH
            snn.set_izhikevich_params('dcn',           a=0.03, b=0.25, c=-52, d=0)  # Rebound
            # Motor hidden: Regular Spiking (like cortical pyramidal cells)
            if n_motor_hidden > 0:
                snn.set_izhikevich_params('motor_hidden', a=0.02, b=0.2, c=-65, d=8)  # RS
            # Output neurons: Regular Spiking (motoneurons are tonic/RS).
            # v0.5.1: Enabling Izhikevich on ALL populations allows the
            # unified fast path in step() (no boolean masking overhead).
            # Biology: alpha motoneurons fire tonically — RS is appropriate.
            snn.set_izhikevich_params('output', a=0.02, b=0.2, c=-65, d=8)  # RS
            # Input neurons: Fast Spiking (thalamic relay neurons).
            # FS fires 3x faster than RS → more input spikes per frame.
            # Biology: thalamic relay cells are fast-spiking, not regular.
            # Having them FS also enables the all_izh fast path in step().
            snn.set_izhikevich_params('input', a=0.1, b=0.2, c=-65, d=2)   # FS

        # Protect cerebellar populations from R-STDP
        # Can be disabled with protect_cerebellum=False for v0.4.3 compatibility
        # where R-STDP could modify cerebellar weights (biologically incorrect
        # but empirically helped Go2 stability).
        if protect_cerebellum:
            snn.protected_populations = {
                'mossy_fibers', 'granule_cells', 'golgi_cells',
                'purkinje_cells', 'dcn',
            }

        snn._rebuild_sparse_weights()

        # 5. Creature erstellen
        body_name = f"{creature_name}_s0"
        creature = MuJoCoCreature(genome, snn, world, body_name, creature_name)

        # Set hardware sensor mode
        creature._hardware_sensors = hardware_sensors
        if hardware_sensors and profile_n_input:
            creature._hardware_n_input = profile_n_input
        # Recalculate dimensions with correct sensor mode
        creature._n_sensor_channels = n_sensor_channels if hardware_sensors else creature._count_sensor_channels()
        creature.n_input_neurons = n_input
        creature.n_output_neurons = n_output
        # Adjust NEURONS_PER_MOTOR when profile overrides n_output
        if profile_n_output is not None and world.n_actuators > 0:
            creature.NEURONS_PER_MOTOR = max(1, n_output // world.n_actuators)

        # 6. Cognitive Brain
        plasticity_pg = None
        if genome.plasticity_genome:
            plasticity_pg = PlasticityGenome(**genome.plasticity_genome)

        brain_config = CognitiveBrainConfig(
            world_model_hidden=max(50, n_granule // 10),
            curiosity_weight=0.3,
            dream_interval=200,
            dream_steps=10,
            hebbian_rate=0.003,
            device=device,
        )
        creature.brain = CognitiveBrain(
            snn=snn,
            n_sensor_channels=n_sensor_channels,
            n_motors=world.n_actuators,
            config=brain_config,
            plasticity_genome=plasticity_pg,
        )

        return creature
