"""
MH-FLOCKE — MuJoCo Creature v0.4.1
========================================
SNN-MuJoCo bridge: population coding, sense-think-act cycle.
"""

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
    Gaussianische Tuning Curves: ein Wert → Population von Spike-Strömen.
    
    Jedes Neuron hat ein "preferred value" (gleichmäßig verteilt).
    Aktivierung = Gauss(value - preferred) × gain.
    
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

        # Sensor/Motor Dimensionen
        self._n_sensor_channels = self._count_sensor_channels()
        self._n_motors = world.n_actuators if hasattr(world, '_model') and world._model else genome.n_joints
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
        # Feste Kanaele: position(3) + velocity(3) + orientation(3) + 
        #               height(1) + upright(1) + forward_vel(1) +
        #               joint_angles(n) + joint_velocities(n) +
        #               visual_target_heading(1) + visual_target_distance(1)
        # n_joints from MuJoCo (actuators), not from genome
        # Vision channels (Issue #76d): The creature can SEE a target object.
        # These 2 channels encode direction and distance to the visual target,
        # like a dog's visual cortex projecting to motor areas.
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

    # ================================================================
    # SENSE
    # ================================================================

    def get_sensor_input(self) -> torch.Tensor:
        """
        MuJoCo Sensoren -> SNN Input Tensor.
        
        Verwendet Population Coding: jeder Sensor-Wert wird durch
        NEURONS_PER_SENSOR Neuronen mit Gaussianischen Tuning Curves encodiert.
        """
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
        # Biology: Visual cortex (V1->MT->FEF) projects target location to
        # premotor cortex. The dog SEES the ball and knows where it is.
        # 2 channels: heading (-1=far left, +1=far right), distance (0=far, 1=touching)
        # These are population-coded like all other sensors = 16 new input neurons.
        # The SNN learns via R-STDP: "heading>0 + approach_reward -> turn right"
        _vis_heading = getattr(self, '_visual_target_heading', 0.0)
        _vis_distance = getattr(self, '_visual_target_distance', 0.0)
        sensor_values.append(np.clip(_vis_heading, -1.0, 1.0))
        sensor_values.append(np.clip(_vis_distance, -1.0, 1.0))

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

        # Teil A (asymmetric input current) REMOVED — replaced by dedicated
        # vision sensor channels above (target_heading + target_distance).
        # The SNN now has proper "eyes" instead of a crude current bias.

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
        # (snn.spikes only holds the LAST substep)
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
                # Vestibular gate: pass upright to cerebellum (Issue #68)
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
            # NEW MODE: Pure SNN via cerebellum (v0.3.1)
            # No CPG base -- cerebellum DCN output IS the motor command.
            # compute_corrections() uses pure_mode internally for [-1,1] range.
            _quat2 = self.world._data.qpos[3:7]
            _up2 = max(0, 1.0 - 2.0 * (_quat2[1]**2 + _quat2[2]**2))
            try:
                corrections = self.actor_critic.compute_corrections(snn_controls, upright=_up2)
            except TypeError:
                corrections = self.actor_critic.compute_corrections(snn_controls)
            controls = list(np.clip(corrections, -1.0, 1.0))
        else:
            # Fallback: raw SNN spike decoding (no cerebellum)
            controls = snn_controls

        # Scale cerebellum/SNN output (morphology needs fine control)
        per_joint_scale = getattr(self, 'per_joint_scale', None)
        motor_scale = getattr(self, 'motor_scale', 1.0)
        if per_joint_scale is not None:
            controls = [c * s for c, s in zip(controls, per_joint_scale)]
        elif motor_scale != 1.0:
            controls = [c * motor_scale for c in controls]

        # Blend with spinal CPG (innate rhythmic pattern)
        # Motor = CPG * cpg_weight + Cerebellum * (1 - cpg_weight) + Reflexes
        cpg_cmd = getattr(self, '_cpg_cmd', None)
        cpg_weight = getattr(self, '_cpg_weight', 0.0)
        if cpg_cmd is not None and cpg_weight > 0:
            controls = [cpg * cpg_weight + cb * (1.0 - cpg_weight)
                       for cpg, cb in zip(cpg_cmd, controls)]

        # Add reflex commands (emergency overrides)
        reflex_cmd = getattr(self, '_reflex_cmd', None)
        if reflex_cmd is not None:
            reflex_scale = getattr(self, 'reflex_scale', 1.0)
            controls = [c + r * reflex_scale for c, r in zip(controls, reflex_cmd)]

        # Terrain reflex: proprioceptive slope/contact corrections (Phase B ATR)
        # Added AFTER CPG+cerebellum+reflexes, BEFORE spinal segments
        terrain_corr = getattr(self, '_terrain_corr', None)
        if terrain_corr is not None:
            controls = [c + t for c, t in zip(controls, terrain_corr)]

        # Spinal segments: muscle tone + stretch reflex + Golgi tendon organ
        # This is the LAST biological layer before the muscles.
        # Tone keeps joints from collapsing, stretch reflex resists
        # perturbations, Golgi limits excessive force.
        spinal_seg = getattr(self, '_spinal_segments', None)
        if spinal_seg is not None:
            joint_pos = self.world._data.qpos[7:7+n_act] if hasattr(self.world, '_data') else np.zeros(n_act)
            controls = spinal_seg.process(
                np.array(controls), joint_pos,
                sim_dt=getattr(self, '_sim_dt', 0.005)
            ).tolist()
        else:
            controls = [np.clip(c, -1.0, 1.0) for c in controls]

        # Teil B motor-hack steering REMOVED — steering now happens inside
        # the CPG itself via asymmetric per-leg amplitude (spinal_cpg.compute(
        # steering=...)). This is biologically correct: Reticulospinal neurons
        # modulate left/right CPG half-centers, not post-hoc motor offsets.
        # Ref: Grillner 2003 (lamprey turning via asymmetric CPG)

        self.world.set_controls(np.array(controls))
        self._last_controls = controls
        self._energy_spent += sum(abs(c) for c in controls)

        # PD Controller hook: convert position targets -> torques
        # For torque-actuated robots (Go2), the CPG/SNN output +/-1.0
        # must be interpreted as joint angle offsets, not raw torques.
        pd = getattr(self, '_pd_controller', None)
        if pd is not None:
            n = self.world.n_actuators
            raw_ctrl = self.world._data.ctrl[:n].copy()
            standing = pd['standing']
            # Dynamic scale: fine control when upright, full range when fallen
            # Biology: muscle co-activation increases with arousal/panic.
            # A fallen animal thrashes with maximum range to right itself.
            base_scale = pd.get('scale', 0.4)
            fallen_scale = pd.get('fallen_scale', 1.5)
            _quat = self.world._data.qpos[3:7]
            _upright = max(-1, min(1, 1.0 - 2.0 * (_quat[1]**2 + _quat[2]**2)))
            # Smooth transition: upright(1.0) -> base_scale, fallen(-1.0) -> fallen_scale
            _urgency = max(0.0, min(1.0, (1.0 - _upright) / 2.0))  # 0=upright, 1=inverted
            scale = base_scale + (fallen_scale - base_scale) * _urgency
            target_q = standing + raw_ctrl * scale
            current_q = self.world._data.qpos[7:7+n]
            current_v = self.world._data.qvel[6:6+n]
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
        
        Args:
            reward_signal: Reward fuer R-STDP Lernen
            extra_sensor_data: Optional dict with enriched sensor data from
                training loop (smell_strength, scent_reward, etc.) that is
                not available in raw MuJoCo sensors. Passed through to
                CognitiveBrain.process() for Drive computation.
            
        Returns:
            dict mit Step-Info (spikes, controls, etc.)
        """
        # Startposition bei erstem Step setzen
        if self._start_position is None:
            self._init_start_pos()

        # 1. Sense (raw sensor values for CognitiveBrain)
        sensor_data = {}
        try:
            sensor_data = self.world.get_sensor_data(self.body_name)
        except Exception:
            pass

        # 2. Sense -> SNN Encoding
        sensor_input = self.get_sensor_input()

        # 3. Think
        output_spikes = self.think(sensor_input)

        # 4. Act
        self.apply_motor_output(output_spikes)

        # 5. Physik-Step (MuJoCo)
        self.world.step()

        # 6. Cognitive cycle
        #    When cerebellar learning is active, we enable CognitiveBrain
        #    but protect the SNN from weight modifications that would
        #    interfere with cerebellar learning.
        #    Protected: R-STDP, Hebbian, GWT broadcast to hidden, 
        #               Synaptogenesis apical injection
        #    Active: Emotions, Drives, Memory, Body Schema, World Model,
        #            Metacognition, Consistency, Dream recording
        brain_result = {}
        if self.brain:
            # When cerebellar learning is active, disable SNN weight mods
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

        # 7. Tracking
        self._step_count += 1
        if not self.is_fallen():
            self._standing_steps += 1

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
        """Aktuelle Position des Root-Body."""
        try:
            data = self.world.get_sensor_data(self.body_name)
            return data['position'].copy()
        except Exception:
            return np.zeros(3)

    def get_distance_traveled(self) -> float:
        """Horizontale Distanz vom Startpunkt (XY-Ebene)."""
        if self._start_position is None:
            return 0.0
        pos = self.get_position()
        diff = pos - self._start_position
        return float(np.sqrt(diff[0]**2 + diff[1]**2))  # XY-Ebene (MuJoCo: Z=up)

    def is_fallen(self) -> bool:
        """Kreatur umgefallen? (Height unter Threshold oder nicht aufrecht)."""
        try:
            # Read directly from qpos (reliable, no sensor needed)
            height = float(self.world._data.qpos[2])
            qw, qx, qy, qz = self.world._data.qpos[3:7]
            upright = max(0, 1.0 - 2.0 * (qx*qx + qy*qy))
            # Height threshold accounts for motor_scale reducing standing height
            h_thresh = getattr(self, '_fallen_height_threshold', 0.08)
            return height < h_thresh or upright < 0.3
        except Exception:
            return True

    def get_energy_spent(self) -> float:
        return self._energy_spent

    def get_state(self) -> dict:
        """Kompletter Zustand fuer Logging."""
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
        """Reward -> R-STDP + Neuromodulator-Anpassung."""
        self.snn.apply_rstdp(reward_signal=reward)
        if reward > 0:
            self.snn.set_neuromodulator('ne', min(0.8, 0.3 + reward * 0.3))
        else:
            self.snn.set_neuromodulator('5ht', min(0.8, 0.5 + abs(reward) * 0.2))

    def reset(self):
        """State zuruecksetzen (SNN bleibt, Physik wird resettet)."""
        self.world.reset()
        self._step_count = 0
        self._energy_spent = 0.0
        self._standing_steps = 0
        self._last_controls = None
        self._start_position = None


# ================================================================
# BUILDER
# ================================================================

class MuJoCoCreatureBuilder:
    """Factory: Genome -> MuJoCoCreature (komplett verdrahtet)."""

    @staticmethod
    def build(genome: Genome, world=None,
              n_hidden_neurons: int = 1000,
              device: str = 'cpu',
              creature_name: str = 'creature',
              xml_path: str = None,
              xml_string: str = None) -> 'MuJoCoCreature':
        """
        Baut eine MuJoCo-Kreatur aus Genome (oder fertigem XML).
        
        Args:
            genome: Kreatur-Morphologie
            world: MuJoCoWorld (wird erstellt wenn None)
            n_hidden_neurons: Anzahl Hidden-Neuronen im SNN
            device: 'cpu' oder 'cuda'
            creature_name: Name-Prefix
            xml_path: Pfad zu fertigem MJCF XML (Mesh-Modell).
                      Wenn angegeben, wird das XML direkt geladen
                      statt aus dem Genome generiert.
            
        Returns:
            MuJoCoCreature mit verdrahtetem SNN
        """
        from src.body.mujoco_world import MuJoCoWorld
        from src.body.mjcf_generator import MJCFGenerator

        # 1. MuJoCo Welt
        if world is None:
            world = MuJoCoWorld(render=False)

        # 2. Load MJCF: either existing file or generated from genome
        if xml_string:
            # Szenen-XML direkt laden (SceneBuilder hat MJCF schon kombiniert)
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

        # 3. SNN dimensionieren -- Cerebellar Architecture v0.3.0
        # Based on Marr-Albus theory and Shinji et al. 2024
        #
        # Layout:  MF(input) -> GrC -> PkC -> DCN(output)
        #           \-> GoC <--/ (feedback inhibition for sparseness)
        #
        # Old layout was: input -> relay -> hidden -> output (unstructured)
        # New layout has functional populations with biological roles.
        #
        # Sensor-Channels aus MuJoCo (+2 vision channels: target heading + distance)
        n_sensor_channels = 12 + 2 * world.n_actuators + 2  # +2 = vision (Issue #76d)
        n_input = n_sensor_channels * MuJoCoCreature.NEURONS_PER_SENSOR

        # Cerebellar population sizes
        from src.brain.cerebellar_learning import CerebellarConfig
        cb_cfg = CerebellarConfig()
        n_granule = cb_cfg.n_granule    # 4000 GrC (expansion layer)
        n_golgi = cb_cfg.n_golgi        # 200 GoC (inhibitory)
        # PkC/DCN = 2 per actuator (push/pull), auto-sized to morphology
        n_purkinje = world.n_actuators * 2  # e.g. 16 actuators -> 32 PkC
        n_dcn = world.n_actuators * 2       # same as PkC
        cb_cfg.n_purkinje = n_purkinje
        cb_cfg.n_dcn = n_dcn

        # Legacy output neurons (kept for compatibility with think()/decode)
        n_output = world.n_actuators * MuJoCoCreature.NEURONS_PER_MOTOR

        # Total: MF + Output(legacy) + GrC + GoC + PkC + DCN
        total_neurons = n_input + n_output + n_granule + n_golgi + n_purkinje + n_dcn

        # Neuron ID ranges
        grc_start = n_input + n_output
        goc_start = grc_start + n_granule
        pkc_start = goc_start + n_golgi
        dcn_start = pkc_start + n_purkinje

        # 4. SNN Controller
        snn = SNNController(SNNConfig(
            n_neurons=total_neurons,
            connectivity_prob=0.0,
            homeostatic_interval=200,
            device=device,
        ))

        # === Populations -- Cerebellar Architecture ===
        # Mossy fibers = input population (sensor encoding)
        mf_ids = torch.arange(0, n_input)
        snn.define_population('input', mf_ids)           # compat name
        snn.define_population('mossy_fibers', mf_ids)     # cerebellar name

        # Legacy output (still used by think()/decode_motor_spikes)
        out_ids = torch.arange(n_input, n_input + n_output)
        snn.define_population('output', out_ids)

        # Granule cells -- expansion layer, sparse coding
        grc_ids = torch.arange(grc_start, grc_start + n_granule)
        snn.define_population('granule_cells', grc_ids)
        snn.define_population('hidden', grc_ids)  # compat: cognitive_brain uses 'hidden'

        # Golgi cells -- inhibitory feedback on GrC
        goc_ids = torch.arange(goc_start, goc_start + n_golgi)
        snn.define_population('golgi_cells', goc_ids)

        # Purkinje cells -- cerebellar output, learns via LTD
        pkc_ids = torch.arange(pkc_start, pkc_start + n_purkinje)
        snn.define_population('purkinje_cells', pkc_ids)

        # Deep Cerebellar Nuclei -- motor correction output
        dcn_ids = torch.arange(dcn_start, dcn_start + n_dcn)
        snn.define_population('dcn', dcn_ids)

        # === Connectivity -- Biologically structured ===

        # MF -> GrC: each GrC receives exactly 4 MF inputs (biology: 4 dendrites)
        # Using structured sparse connectivity instead of random probability
        n_mf = len(mf_ids)
        mf_per_grc = cb_cfg.mf_per_granule
        src_list = []
        tgt_list = []
        for g in range(n_granule):
            # Each GrC samples 4 random MF inputs
            mf_choices = torch.randint(0, n_mf, (mf_per_grc,))
            for m in mf_choices:
                src_list.append(mf_ids[m].item())
                tgt_list.append(grc_ids[g].item())
        if src_list:
            src_t = torch.tensor(src_list, device=device)
            tgt_t = torch.tensor(tgt_list, device=device)
            w = torch.rand(len(src_list), device=device) * 1.0 + 1.0  # [1.0, 2.0]
            new_idx = torch.stack([src_t, tgt_t])
            snn._weight_indices = new_idx
            snn._weight_values = w
            snn._eligibility = torch.zeros(len(src_list), device=device)

        # GrC -> GoC: excitatory feedback (GoC monitors GrC activity)
        snn.connect_populations('granule_cells', 'golgi_cells',
                                prob=cb_cfg.grc_goc_prob, weight_range=(0.3, 0.8))

        # GoC -> GrC: INHIBITORY feedback (enforces sparseness)
        # Mark GoC neurons as inhibitory
        snn.neuron_types[goc_ids] = -1.0  # inhibitory
        snn.connect_populations('golgi_cells', 'granule_cells',
                                prob=0.02, weight_range=(0.2, 0.4))

        # MF -> GoC: direct excitation (GoC also receives sensory input)
        snn.connect_populations('mossy_fibers', 'golgi_cells',
                                prob=0.1, weight_range=(0.5, 1.0))

        # GrC -> legacy output: keep some direct path for SNN motor output
        snn.connect_populations('granule_cells', 'output',
                                prob=0.02, weight_range=(0.3, 0.8))

        # === Per-population membrane time constants (tau_mem) ===
        snn._tau_base[grc_ids] = 5.0
        snn._tau_base[goc_ids] = 20.0
        snn._tau_base[pkc_ids] = 15.0
        snn._tau_base[dcn_ids] = 10.0

        # === Thresholds ===
        snn._thresholds[grc_ids] = 0.5
        snn._thresholds[goc_ids] = 0.5
        snn._thresholds[pkc_ids] = 1.0
        snn._thresholds[dcn_ids] = 1.0
        snn._thresholds[out_ids] = 0.4
        snn._hidden_tonic_current = 0.015

        snn._rebuild_sparse_weights()

        # 5. Creature erstellen
        body_name = f"{creature_name}_s0"
        creature = MuJoCoCreature(genome, snn, world, body_name, creature_name)

        # 6. Cognitive Brain aufsetzen
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
