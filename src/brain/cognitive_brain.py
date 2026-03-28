"""
MH-FLOCKE — Cognitive Brain v0.4.1
========================================
15-step closed-loop cognitive architecture orchestrating all neural modules.
"""

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from src.brain.snn_controller import SNNController
from src.brain.world_model import SpikingWorldModel, WorldModelConfig, DreamEngine
from src.brain.gwt_bridge import GlobalWorkspaceBridge, GWTBridgeConfig, GWTModule
from src.brain.evolved_plasticity import PlasticityGenome, EvolvedPlasticityRule
from src.brain.embodied_emotions import EmbodiedEmotions
from src.brain.body_schema import BodySchema, BodySchemaConfig
from src.brain.sensomotor_memory import SensomotorMemory
from src.brain.drives import MotivationalDrives
from src.brain.embodied_metacognition import EmbodiedMetacognition
from src.brain.consistency_checker import ConsistencyChecker
from src.brain.synaptogenesis import Synaptogenesis, SynaptogenesisConfig
from src.brain.astrocyte_gate import AstrocyteGate
from src.brain.theory_of_mind import TheoryOfMind, ToMConfig
from src.brain.modular_skills import SkillRegistry
from src.behavior.behavior_knowledge import BehaviorKnowledge
from src.behavior.behavior_planner import BehaviorPlanner, Situation
from src.behavior.behavior_executor import BehaviorExecutor
from src.brain.curiosity import CuriosityDrive, CuriosityConfig
from src.brain.empowerment import EmpowermentDrive, EmpowermentConfig


@dataclass
class CognitiveBrainConfig:
    """Konfiguration für das kognitive Gehirn."""
    # World Model
    world_model_hidden: int = 200
    curiosity_weight: float = 0.5
    curiosity_decay: float = 0.99

    # Intrinsische Motivation (Curiosity + Empowerment)
    curiosity_alpha: float = 0.3        # Balance intrinsisch/extrinsisch
    curiosity_boredom_steps: int = 200
    empowerment_weight: float = 0.3
    empowerment_history: int = 100

    # GWT
    gwt_broadcast_strength: float = 0.3
    gwt_competition: str = 'softmax'

    # Dream
    dream_interval: int = 100
    dream_steps: int = 20
    dream_replay_ratio: float = 0.7

    # Hebbian
    hebbian_rate: float = 0.005

    # Memory
    memory_max_episodes: int = 500
    memory_fragment_length: int = 20

    # Synaptogenesis
    synaptogenesis_interval: int = 10  # Spikes beobachten alle N steps
    synaptogenesis_consolidate_interval: int = 200  # Konsolidieren alle N steps

    # Astrocyte
    astrocyte_cluster_size: int = 100

    # PCI
    pci_interval: int = 500  # PCI messen alle N steps (teuer!)

    device: str = 'cpu'


class CognitiveBrain:
    """
    Kognitive Schicht über dem SNN — orchestriert ALLE Module.

    Integriert alle Level (1-9) als sensomotorische Äquivalente.
    """

    def __init__(self, snn: SNNController, n_sensor_channels: int,
                 n_motors: int, config: CognitiveBrainConfig = None,
                 plasticity_genome: Optional[PlasticityGenome] = None):
        self.snn = snn
        self.config = config or CognitiveBrainConfig()
        self.device = snn.device
        self.n_sensor_channels = n_sensor_channels
        self.n_motors = n_motors

        # --- World Model (Level 8C: Verstehen) ---
        wm_config = WorldModelConfig(
            n_input=n_sensor_channels + n_motors,
            n_hidden=self.config.world_model_hidden,
            n_output=n_sensor_channels,
            device=str(self.device),
        )
        self.world_model = SpikingWorldModel(wm_config)

        # --- Dream Engine (Level 8C: Schlafen) ---
        self.dream_engine = DreamEngine(self.world_model, snn)

        # --- GWT Bridge (Level 6: Bewusstsein) ---
        gwt_config = GWTBridgeConfig(
            n_neurons=min(snn.config.n_neurons, 1000),
            broadcast_strength=self.config.gwt_broadcast_strength,
            competition_method=self.config.gwt_competition,
            device=str(self.device),
        )
        self.gwt = GlobalWorkspaceBridge(gwt_config)
        n_gwt = gwt_config.n_neurons
        for name in ['sensory', 'motor', 'predictive', 'error', 'memory']:
            self.gwt.register_module(GWTModule(name, n_gwt, str(self.device)))

        # --- Evolved Plasticity (Level 7: Meta-Learning) ---
        self.plasticity_rule: Optional[EvolvedPlasticityRule] = None
        if plasticity_genome:
            self.plasticity_rule = EvolvedPlasticityRule(plasticity_genome)

        # --- Embodied Emotions (Level 6: Somatic Markers) ---
        self.emotions = EmbodiedEmotions()

        # --- Body Schema (Level 4: Self-Model) ---
        self.body_schema = BodySchema(
            n_joints=n_motors,
            n_sensors=n_sensor_channels,
            config=BodySchemaConfig(learning_rate=0.05),
        )

        # --- Sensomotor Memory (Level 3+: Episodic) ---
        self.memory = SensomotorMemory(
            max_episodes=self.config.memory_max_episodes,
            fragment_length=self.config.memory_fragment_length,
            n_sensors=n_sensor_channels,
            n_motors=n_motors,
        )

        # --- Motivational Drives (Level 5: Goals) ---
        self.drives = MotivationalDrives()

        # --- Metacognition (Level 5: self-monitoring) ---
        self.metacognition = EmbodiedMetacognition()

        # --- Consistency Checker (Level 8: Integrity) ---
        self.consistency = ConsistencyChecker()

        # --- Synaptogenesis (Level 9: SNN ↔ Konzept-Graph) ---
        synapto_config = SynaptogenesisConfig(
            pattern_dimensions=64,
            max_concepts=500,
            consolidation_threshold=3,
            similarity_threshold=0.6,
            retrieval_strength=0.3,
            device=str(self.device),
        )
        self.synaptogenesis = Synaptogenesis(synapto_config, snn)

        # --- Astrocyte Gating (Level 8D: biologisches Gating) ---
        self.astrocytes = AstrocyteGate(
            n_neurons=snn.config.n_neurons,
            cluster_size=self.config.astrocyte_cluster_size,
        )

        # --- Theory of Mind (Level 9+, activated for multi-agent) ---
        self.tom: Optional[TheoryOfMind] = None  # Lazy: call enable_tom()

        # --- Modular Skills (Selective Learning/Forgetting) ---
        self.skills = SkillRegistry(snn)

        # --- Behavior System (Phase 11: autonomous behavior) ---
        self.behavior_knowledge = BehaviorKnowledge(creature_type='dog')
        self.behavior_planner = BehaviorPlanner(self.behavior_knowledge)
        self.behavior_executor = BehaviorExecutor(n_actuators=n_motors)
        self._current_behavior = 'walk'
        self._motor_modulation = None  # (freq_scale, amp_scale, overrides)

        # --- Intrinsische Motivation (Phase 9W) ---
        self.curiosity = CuriosityDrive(CuriosityConfig(
            alpha=self.config.curiosity_alpha,
            boredom_steps=self.config.curiosity_boredom_steps,
        ))
        self.empowerment = EmpowermentDrive(EmpowermentConfig(
            weight=self.config.empowerment_weight,
            history_size=self.config.empowerment_history,
        ))

        # --- LLM-Bridge (Level 12: language-capable) ---
        self.language_bridge = None  # Optional: set via attach_language_bridge()

        # --- PCI (Level 6: Bewusstseinsmetrik) ---
        self._last_pci = 0.0
        self._pci_error_logged = False
        self._has_dreamed = False

        # --- Hebbian Counter ---
        self._hebbian_updates = 0
        self._hebbian_total_dw = 0.0
        self._hebbian_step_dw = 0.0  # per-step delta for dashboard

        # --- State ---
        self._step_count = 0
        self._last_sensor: Optional[torch.Tensor] = None
        self._last_motor: Optional[torch.Tensor] = None
        self._last_raw_sensors: Optional[list] = None
        self._prediction_error = 0.0
        self._curiosity_reward = 0.0
        self._cumulative_curiosity = 0.0
        self._empowerment_reward = 0.0
        self._gwt_winner = ''
        self._consciousness_level = 0

    def process(self, sensor_values, snn_input: torch.Tensor,
                output_spikes: torch.Tensor, controls: list,
                external_reward: float = 0.0,
                is_fallen: bool = False,
                extra_sensor_data: Optional[Dict] = None) -> Dict:
        """
        Vollständiger 15-Step kognitiver Zyklus.

        Aufgerufen von MuJoCoCreature.step() NACH dem SNN-Step.

        Args:
            sensor_values: Raw sensor values from MuJoCo
            snn_input: SNN input tensor (population coded)
            output_spikes: SNN output spikes
            controls: Motor control values
            external_reward: Reward from training loop
            is_fallen: Whether creature is fallen
            extra_sensor_data: Optional dict with enriched sensor data from
                training loop (smell_strength, scent_reward, etc.) that is
                not available in raw MuJoCo sensors. Passed to Drives.
        """
        sensor_t = torch.tensor(sensor_values, dtype=torch.float32,
                                device=self.device) if not isinstance(
                                    sensor_values, torch.Tensor) else sensor_values
        motor_t = torch.tensor(controls, dtype=torch.float32,
                               device=self.device)
        raw_sensors = list(sensor_values) if not isinstance(sensor_values, list) else sensor_values

        # ============================================================
        # 1. SENSE — Raw sensors already available
        # ============================================================
        sensor_data = {
            'height': raw_sensors[9] * 2.0 if len(raw_sensors) > 9 else 0.5,
            'upright': raw_sensors[10] if len(raw_sensors) > 10 else 0.5,
            'forward_velocity': raw_sensors[11] * 5.0 if len(raw_sensors) > 11 else 0.0,
            'joint_angles': raw_sensors[12:12 + self.n_motors] if len(raw_sensors) > 12 else [],
            'joint_velocities': raw_sensors[12 + self.n_motors:] if len(raw_sensors) > 12 + self.n_motors else [],
        }

        # ============================================================
        # 2. BODY SCHEMA — Check efference copy
        # ============================================================
        body_result = self.body_schema.update(
            motor_command=controls,
            current_sensors=raw_sensors,
            previous_sensors=self._last_raw_sensors,
        )
        body_anomaly = body_result.get('anomaly', 0.0)
        body_confidence = body_result.get('body_confidence', 0.0)

        # ============================================================
        # 3. WORLD MODEL — Vorhersage + Prediction Error
        # ============================================================
        prediction_error = 0.0
        if self._last_sensor is not None:
            prediction_error = self.world_model.train_step(
                self._last_sensor[:self.n_sensor_channels],
                self._last_motor[:self.n_motors],
                sensor_t[:self.n_sensor_channels]
            )
        self._prediction_error = prediction_error

        # Task-specific PE: "Is the ball getting closer?" (Issue #79 fix)
        # The global World Model PE is ~0.004 (noise). It tells the SNN
        # "you predicted the world correctly" — but correctly predicting
        # that you walk straight past the ball is NOT the goal.
        #
        # Cortical Labs DishBrain principle: unpredictable stimulation
        # when the task is NOT being accomplished. If ball distance
        # increases → high PE. If ball distance decreases → low PE.
        # This gives the SNN a clear signal EVERY step.
        #
        # Biology: the task PE is like frustration/satisfaction.
        # Ball getting closer = satisfying = low PE = consolidate.
        # Ball getting further = frustrating = high PE = change behavior.
        _extra = extra_sensor_data or {}
        _ball_dist = _extra.get('ball_distance', 0.0)
        _ball_heading = _extra.get('ball_heading', 0.0)
        if _ball_dist > 0.1:  # Ball scene active
            # Task PE based on ABSOLUTE STATE, not step-delta.
            #
            # Problem with delta-based PE: per-step distance change is ~0.001m.
            # After scaling: TPE = 0.003. That's noise, not a learning signal.
            #
            # Solution: State-based PE. "How good is my CURRENT situation?"
            #   Ball at 1m = very good (TPE = -1.0)
            #   Ball at 3m = neutral (TPE = 0.0)
            #   Ball at 8m = very bad (TPE = +1.0)
            #   Ball heading 0.0 = facing ball (bonus -0.3)
            #   Ball heading 0.9 = ball behind me (penalty +0.3)
            #
            # This gives STRONG contrast every single step.
            # The SNN feels: "I'm close to the ball = calm = consolidate"
            #            vs: "I'm far from ball = chaos = change behavior"
            #
            # Reference distance: 3.0m (starting distance from ball)
            _ref_dist = 3.0
            # Distance PE: linear, centered at reference distance
            _dist_pe = (_ball_dist - _ref_dist) / _ref_dist  # 0m→-1.0, 3m→0.0, 6m→+1.0
            # Issue #79b: Asymmetric PE — walking AWAY from ball hurts 2x more
            # than approaching helps. Biology: loss aversion (Kahneman 1979).
            # This creates a "ratchet" effect: gains are normal, but losses
            # are amplified → the SNN strongly avoids increasing ball distance.
            if hasattr(self, '_prev_ball_dist') and self._prev_ball_dist > 0.1:
                if _ball_dist > self._prev_ball_dist:  # Walking away
                    _dist_pe *= 2.0
            # Issue #79c: Proximity brake — when close to ball, PE becomes
            # purely about STAYING close, not about approaching further.
            # Any movement away from ball < 0.5m generates strong PE.
            if _ball_dist < 0.5:
                _dist_pe = max(_dist_pe, 0.0)  # Never reward further approach at < 0.5m
                if hasattr(self, '_prev_ball_dist') and self._prev_ball_dist > 0.1:
                    _departure = _ball_dist - self._prev_ball_dist
                    if _departure > 0:  # Moving away while close
                        _dist_pe = _departure * 10.0  # STRONG penalty
            # Heading PE: only when far from ball (not during close approach)
            _heading_pe = abs(_ball_heading) * 0.3 if _ball_dist > 1.5 else 0.0
            _task_pe = _dist_pe + _heading_pe
            self._task_prediction_error = max(-2.0, min(2.0, _task_pe))
            self._prev_ball_dist = _ball_dist
            # REPLACE the global PE with task PE for R-STDP learning
            prediction_error = self._task_prediction_error

            # DishBrain-style LOCAL stimulation of vision neurons (Issue #79)
            # Global TPE modulates ALL 93k synapses equally → no directional learning.
            # DishBrain: electrodes stimulate LOCALLY → neurons near the error get chaos.
            #
            # Our equivalent: when TPE is positive (failing), BOOST the vision input
            # neurons. This forces the SNN to "pay attention" to the ball signal.
            # The heading neurons get extra current proportional to TPE → the SNN
            # can't ignore the ball anymore. It MUST process the vision signal.
            #
            # When TPE is negative (succeeding), vision neurons get baseline → calm.
            # This is exactly DishBrain: chaos when wrong, calm when right.
            if 'input' in self.snn.populations and self._task_prediction_error > 0.05:
                input_ids = self.snn.populations['input']
                n_input = len(input_ids)
                # Last 16 input neurons = vision (8 heading + 8 distance)
                vision_start = max(0, n_input - 16)
                vision_ids = input_ids[vision_start:]
                # Boost proportional to TPE: TPE=1.0 → +0.5 extra current
                boost = self._task_prediction_error * 0.5
                self.snn.V[vision_ids] += boost
        else:
            self._task_prediction_error = 0.0

        # ============================================================
        # 4. EMOTIONS — Valence-arousal from body signals
        # ============================================================
        emotional_state = self.emotions.update(
            sensor_data=sensor_data,
            prediction_error=prediction_error,
            reward=external_reward,
            is_fallen=is_fallen,
        )
        somatic_markers = self.emotions.get_somatic_markers()

        # ============================================================
        # 5. MEMORY — Record episodes + retrieve similar ones
        # ============================================================
        self.memory.record_step(
            sensors=raw_sensors,
            motors=controls,
            reward=external_reward,
            valence=emotional_state.valence,
            arousal=emotional_state.arousal,
        )

        memory_recall = []
        memory_mismatch = 0.0
        if prediction_error > 0.2 and self.memory.episodes:
            memory_recall = self.memory.recall_similar(
                np.array(raw_sensors[:self.n_sensor_channels], dtype=np.float32),
                k=2
            )
            if memory_recall:
                # Mismatch: aktuelle Situation vs. erinnerte
                recalled_mean = memory_recall[0].sensor_seq.mean(axis=0)
                current = np.array(raw_sensors[:self.n_sensor_channels], dtype=np.float32)
                diff = current[:len(recalled_mean)] - recalled_mean[:len(current)]
                memory_mismatch = float(np.sqrt(np.mean(diff ** 2)))

                # Fix: Memory recall → motor influence (Module Audit 2026-03-18)
                # If a recalled episode had positive reward (e.g. ball approach),
                # replay its motor pattern as top-down bias into SNN output neurons.
                # Biology: hippocampal replay during waking → motor priming.
                if memory_recall[0].total_reward > 0.5 and 'output' in self.snn.populations:
                    out_ids = self.snn.populations['output']
                    recalled_motors = memory_recall[0].motor_seq.mean(axis=0)
                    m_len = min(len(recalled_motors), len(out_ids))
                    motor_bias = torch.tensor(recalled_motors[:m_len],
                                             dtype=torch.float32, device=self.device)
                    self.snn.V[out_ids[:m_len]] += motor_bias * 0.15  # gentle top-down bias

        # ============================================================
        # 6. DRIVES — Dominanten Antrieb bestimmen
        # ============================================================
        # Merge extra sensor data from training loop (smell, sound, etc.)
        # These enriched values (e.g. smell_strength from SensoryEnvironment)
        # are not available in raw MuJoCo sensors — they come from the
        # training loop via extra_sensor_data.
        _extra = extra_sensor_data or {}
        drive_state = self.drives.compute_drive_strengths({
            'upright': sensor_data.get('upright', 0.5),
            'height': sensor_data.get('height', 0.5),
            'prediction_error': prediction_error,
            'energy_spent': sum(abs(c) for c in controls),
            'is_fallen': is_fallen,
            'learning_progress': self.metacognition.learning_progress,
            'smell_strength': _extra.get('smell_strength', 0.0),
            'scent_reward': _extra.get('scent_reward', 0.0),
        })

        # ============================================================
        # 6b. BEHAVIOR — Drive + knowledge -> autonomous behavior
        # ============================================================
        situation = Situation(
            upright=sensor_data.get('upright', 0.5),
            speed=abs(sensor_data.get('forward_velocity', 0.0)),
            height=sensor_data.get('height', 0.35),
            is_fallen=is_fallen,
            steps_alive=self._step_count,
            prediction_error=prediction_error,
            energy_spent=sum(abs(c) for c in controls),
        )
        self._current_behavior = self.behavior_planner.update(
            self.drives.get_state(), situation)

        # Update executor with current behavior definition
        current_beh_def = self.behavior_knowledge.get_behavior(self._current_behavior)
        self.behavior_executor.set_behavior(
            current_beh_def,
            blend_factor=self.behavior_planner.get_blend_factor())

        # Get motor modulation (freq_scale, amp_scale, overrides)
        planner_state = self.behavior_planner.get_state()
        self._motor_modulation = self.behavior_executor.step(
            behavior_name=self._current_behavior,
            behavior_step=planner_state.get('behavior_step', 0),
        )

        # ============================================================
        # 7. GWT — Competition mit Emotion + Drive Modulation
        # ============================================================
        n_gwt = self.gwt.config.n_neurons
        signals = {}
        for name in ['sensory', 'motor', 'predictive', 'error', 'memory']:
            signals[name] = torch.zeros(n_gwt, device=self.device)

        # Sensory: strength of sensor inputs
        s_len = min(len(sensor_t), n_gwt)
        signals['sensory'][:s_len] = sensor_t[:s_len].abs()
        # Motor: strength of motor outputs
        m_len = min(len(motor_t), n_gwt)
        signals['motor'][:m_len] = motor_t[:m_len].abs()
        # Predictive: World Model Confidence
        signals['predictive'][0] = max(0, 1.0 - prediction_error * 10)
        # Error: Prediction Error Salienz
        signals['error'][0] = prediction_error * 5.0
        # Memory: Recall-Stärke
        if memory_recall:
            signals['memory'][0] = 1.0 - memory_mismatch

        # Emotion moduliert Salience
        emotion_mod = self.emotions.get_gwt_salience_modulation()
        drive_bias = self.drives.get_gwt_bias()
        for name in signals:
            mod = emotion_mod.get(name, 1.0) * drive_bias.get(name, 1.0)
            signals[name] *= mod

        gwt_result = self.gwt.step(signals)
        self._gwt_winner = gwt_result['winning_module']

        # Broadcast → SNN hidden neurons (cerebellar pops protected at SNN level)
        broadcast = gwt_result['broadcast_signal']
        if 'hidden' in self.snn.populations:
            hidden_ids = self.snn.populations['hidden']
            b_len = min(len(broadcast), len(hidden_ids))
            self.snn.V[hidden_ids[:b_len]] += broadcast[:b_len] * 0.1

        # ============================================================
        # 8. METACOGNITION — Confidence, Consciousness Level
        # ============================================================
        # Behavior-Repertoire zählen
        n_behaviors = len(self.behavior_knowledge.get_all_behaviors()) if hasattr(
            self.behavior_knowledge, 'get_all_behaviors') else 3
        n_skills = len(self.skills.get_frozen_skills())
        has_dreamed = hasattr(self, '_has_dreamed') and self._has_dreamed
        has_bridge = hasattr(self, 'language_bridge') and self.language_bridge is not None

        metacog_result = self.metacognition.assess_situation(
            prediction_error=prediction_error,
            body_anomaly=body_anomaly,
            body_confidence=body_confidence,
            fitness=external_reward,
            n_episodes=len(self.memory.episodes),
            behavior_count=n_behaviors,
            skill_count=n_skills,
            modules_active={
                'world_model': True,
                'body_schema': True,
                'emotions': True,
                'memory': len(self.memory.episodes) > 0,
                'drives': True,
                'consistency': True,
                'synaptogenesis': self.synaptogenesis.graph.size() > 0,
                'theory_of_mind': self.tom is not None,
                'behavior_planner': True,
                'dream_engine': has_dreamed,
                'language_bridge': has_bridge,
            },
        )
        self._consciousness_level = metacog_result['consciousness_level']

        # Fix: Metacognition → Exploration (Module Audit 2026-03-18)
        # If learning progress is stagnant, boost NE for exploration.
        # Biology: frustration / boredom → norepinephrine → try something new.
        if self.metacognition.learning_progress < 0.01 and self._step_count > 5000:
            current_ne = self.snn.neuromod_levels.get('ne', 0.2)
            if current_ne < 0.6:  # Don't overshoot
                self.snn.set_neuromodulator('ne', min(0.6, current_ne + 0.02))

        # ============================================================
        # 9. CONSISTENCY — Integrity Check
        # ============================================================
        consistency_result = self.consistency.check(
            prediction_error=prediction_error,
            body_anomaly=body_anomaly,
            memory_mismatch=memory_mismatch,
        )

        # ============================================================
        # 10. COMBINED REWARD (extern + curiosity + empowerment + drive + emotion)
        # ============================================================

        # 10a. Curiosity: World Model Prediction Error → intrinsischer Reward
        #      Running-Mean-normalisiert, mit Boredom-Impuls bei Stagnation
        self._curiosity_reward = self.curiosity.compute_intrinsic_reward(prediction_error)
        self._cumulative_curiosity = (
            self._cumulative_curiosity * self.config.curiosity_decay +
            self._curiosity_reward
        )

        # 10b. Empowerment: Action→State Mutual Information
        #      Kreatur will Kontrolle über ihre Umgebung behalten
        motor_np = np.array(controls, dtype=np.float32)
        sensor_np = np.array(raw_sensors[:self.n_sensor_channels], dtype=np.float32)
        self.empowerment.record(motor_np, sensor_np)
        self._empowerment_reward = self.empowerment.compute_reward()

        # 10c. Curiosity → Neuromodulator-Signale (novelty/boredom)
        # Neuromodulation from curiosity (protected pops keep base tau at SNN level)
        curiosity_neuromod = self.curiosity.get_neuromodulator_signals()
        if curiosity_neuromod['novelty'] > 0.3:
            self.snn.set_neuromodulator('da',
                min(1.0, self.snn.neuromod_levels.get('da', 0.5) + 0.05))
        if curiosity_neuromod['boredom'] > 0.5:
            self.snn.set_neuromodulator('ne',
                min(1.0, self.snn.neuromod_levels.get('ne', 0.5) + 0.1))

        # 10d. Combine: extrinsisch + intrinsisch
        intrinsic_total = self._curiosity_reward + self._empowerment_reward
        combined_extrinsic = external_reward

        # Curiosity alpha: Balance intrinsisch/extrinsisch
        alpha = self.curiosity.config.alpha
        base_reward = (1 - alpha) * combined_extrinsic + alpha * intrinsic_total

        # Drive moduliert Reward
        upright_bonus = 0.1 if sensor_data.get('upright', 0) > 0.7 else 0.0
        combined_reward = self.drives.modulate_reward(
            base_reward=base_reward,
            upright_bonus=upright_bonus,
            curiosity_bonus=self._curiosity_reward,
        )

        # Emotion moduliert: negative Valence dämpft Reward (Vorsicht!)
        emotion_factor = 0.7 + emotional_state.valence * 0.3
        combined_reward *= max(0.3, emotion_factor)

        # ============================================================
        # 11. LEARNING — Evolved Plasticity / R-STDP
        #     Cerebellar populations protected at SNN level (protected_populations)
        # ============================================================
        if abs(combined_reward) > 1e-6 or abs(prediction_error) > 0.01:
            if self.plasticity_rule:
                self.plasticity_rule.apply(self.snn, reward=combined_reward)
            else:
                # Issue #79: Prediction Error as learning signal (Free Energy Principle)
                # Ref: Friston 2010, Cortical Labs DishBrain 2022/2026
                # PE fires every step → 8x more learning events than reward alone
                self.snn.apply_rstdp(
                    reward_signal=combined_reward,
                    prediction_error=prediction_error,
                )

        # 11b. SKILL PROTECTION — EWC-Schutz eingefrorener Skills
        if self.skills._current_skill:
            self.skills.on_training_step()

        # ============================================================
        # 12. SYNAPTOGENESIS — SNN-Pattern → Konzept-Graph
        # ============================================================
        synapto_result = None
        if self._step_count % self.config.synaptogenesis_interval == 0:
            # Spikes beobachten
            self.synaptogenesis.observe_spikes(self.snn.spikes)
            # Erfahrung aufzeichnen mit Kontext
            # Behavior als semantisches Label fuer den Graph
            behavior_name = self.behavior_planner.get_state().get(
                'current_behavior', 'unknown')
            # Fix: Add navigation context (Module Audit 2026-03-18)
            # Without ball_heading/distance, concepts are generic ("walk", "chase").
            # With them, concepts become navigation-specific:
            #   "ball_right_approach" (heading>0, distance decreasing, positive valence)
            #   "ball_left_diverge" (heading<0, distance increasing, negative valence)
            # These concepts can then be retrieved to prime the SNN for navigation.
            _extra = extra_sensor_data or {}
            self.synaptogenesis.record_experience(
                context={
                    'event': behavior_name,
                    'object_type': behavior_name,
                    'upright': sensor_data.get('upright', 0),
                    'height': sensor_data.get('height', 0),
                    'speed': sensor_data.get('speed', 0),
                    'drive': drive_state.dominant,
                    'emotion': emotional_state.dominant_emotion,
                    'ball_heading': _extra.get('ball_heading', 0.0),
                    'ball_distance': _extra.get('ball_distance', 0.0),
                    'steering': _extra.get('steering_offset', 0.0),
                },
                valence=emotional_state.valence,
            )
            # Retrieval: Konzept → Apikaler Kontext ins SNN
            apical = self.synaptogenesis.retrieve()
            if (apical.abs().sum() > 0 and 'hidden' in self.snn.populations):
                hidden_ids = self.snn.populations['hidden']
                ap_len = min(len(apical), len(hidden_ids))
                self.snn.V[hidden_ids[:ap_len]] += apical[:ap_len] * 0.05

        # 12b. ASTROCYTE GATING — Calcium-Update
        if self.snn.spikes is not None:
            spikes_np = self.snn.spikes.cpu().numpy() if self.snn.spikes.is_cuda else self.snn.spikes.numpy()
            self.astrocytes.update(spikes_np, dt=1.0)

        # ============================================================
        # 13. HEBBIAN — Koinzidenz-Lernen
        #     Cerebellar populations protected at SNN level (protected_populations)
        # ============================================================
        if self.config.hebbian_rate > 0:
            self._hebbian_step()

        # ============================================================
        # 14. DREAM — Periodisches Offline-Replay + Memory Consolidation
        # ============================================================
        self.dream_engine.record_experience(sensor_t, motor_t, combined_reward)

        dream_result = None
        if (self.config.dream_interval > 0 and
                self._step_count > 0 and
                self._step_count % self.config.dream_interval == 0):
            # dream() calls snn.step() + apply_rstdp() — cerebellar pops
            # are protected at SNN level via protected_populations.
            dream_result = self.dream_engine.dream(
                n_steps=self.config.dream_steps,
                replay_ratio=self.config.dream_replay_ratio)
            self._has_dreamed = True
            # Memory Consolidation
            consolidation = self.memory.consolidate()
            # Synaptogenesis Konsolidierung: Erfahrungen → Konzepte
            synapto_result = self.synaptogenesis.consolidate()

        # ============================================================
        # 15. NEUROMOD — Anpassung aus Emotion + Consistency
        #     Protected populations keep base tau at SNN level (get_neuromodulator_tau)
        # ============================================================
        # Somatic Markers als primäre Quelle
        for nm, level in somatic_markers.items():
            if nm in ('da', '5ht', 'ne', 'ach'):
                self.snn.set_neuromodulator(nm, float(level))

        # Consistency Override bei schwerer Dissonanz
        neuromod_reset = consistency_result.get('neuromod_reset', {})
        for nm, level in neuromod_reset.items():
            self.snn.set_neuromodulator(nm, float(level))

        # ============================================================
        # 15b. PCI — Periodische Bewusstseinsmessung (teuer!)
        # ============================================================
        # Approximation: Statt echte Perturbation (braucht numpy-SNN API)
        # berechne Spike-Komplexitaet aus aktuellem Aktivitaetsmuster.
        # Methode: Lempel-Ziv Komplexitaet des binärisierten Spike-Vektors.
        if (self.config.pci_interval > 0 and
                self._step_count > 0 and
                self._step_count % self.config.pci_interval == 0):
            try:
                spikes = self.snn.spikes.cpu().numpy().astype(bool)
                if spikes.sum() > 0:
                    pci_val = self._lz_complexity(spikes)
                    if pci_val > 0:
                        self._last_pci = pci_val
            except Exception as e:
                if not self._pci_error_logged:
                    import sys
                    print(f"  [PCI] Error: {e}", file=sys.stderr)
                    self._pci_error_logged = True

        # State Update
        self._last_sensor = sensor_t.detach().clone()
        self._last_motor = motor_t.detach().clone()
        self._last_raw_sensors = raw_sensors
        self._step_count += 1

        return {
            'combined_reward': combined_reward,
            'external_reward': external_reward,
            'curiosity_reward': self._curiosity_reward,
            'empowerment_reward': self._empowerment_reward,
            'empowerment': self.empowerment.last_empowerment,
            'prediction_error': prediction_error,
            'gwt_winner': self._gwt_winner,
            'emotion': self.emotions.get_state(),
            'body_schema': self.body_schema.get_state(),
            'drives': self.drives.get_state(),
            'metacognition': metacog_result,
            'consistency': consistency_result,
            'memory_episodes': len(self.memory.episodes),
            'consciousness_level': self._consciousness_level,
            'concept_graph_size': self.synaptogenesis.graph.size(),
            'knowledge_graph': self.synaptogenesis.graph.to_dict(),
            'astrocyte_active': self.astrocytes.above_threshold_count,
            'pci': self._last_pci,
            'dreamed': dream_result is not None,
        }

    def _hebbian_step(self):
        """Reines Hebb-Learning: koinzidente Spikes verstärken Synapsen."""
        if self.snn._n_synapses == 0 or self.snn._weight_values is None:
            return

        spike_f = self.snn.spikes.float()
        pre_idx = self.snn._weight_indices[0]
        post_idx = self.snn._weight_indices[1]

        co_active = spike_f[pre_idx] * spike_f[post_idx]
        dw = self.config.hebbian_rate * co_active

        signs = self.snn.neuron_types[pre_idx]
        dw = dw * signs.abs()

        # Zero out updates for protected synapses (cerebellar populations)
        if self.snn.protected_populations:
            protected_mask = self.snn._get_protected_synapse_mask()
            dw[protected_mask] = 0.0

        # Count actual updates (non-zero weight changes)
        n_updates = int((dw.abs() > 1e-8).sum())
        self._hebbian_updates += n_updates
        step_dw = float(dw.abs().sum())
        self._hebbian_total_dw += step_dw
        self._hebbian_step_dw = step_dw

        self.snn._weight_values = self.snn._weight_values + dw
        exc_mask = self.snn.neuron_types[pre_idx] > 0
        self.snn._weight_values = torch.where(
            exc_mask,
            self.snn._weight_values.clamp(min=0.0, max=2.0),
            self.snn._weight_values.clamp(min=-2.0, max=0.0)
        )

    @staticmethod
    def _lz_complexity(binary_vector: 'np.ndarray') -> float:
        """Lempel-Ziv complexity of a binary vector, normalized to [0,1].
        Measures information content of spike pattern.
        Higher = more complex/conscious-like activity."""
        import numpy as np
        s = ''.join('1' if b else '0' for b in binary_vector.flatten())
        n = len(s)
        if n == 0:
            return 0.0
        # LZ76 algorithm
        complexity = 1
        i = 0
        k = 1
        kmax = 1
        while i + k <= n:
            substring = s[i + 1:i + k + 1]
            prefix = s[:i + kmax]
            if substring in prefix:
                k += 1
            else:
                complexity += 1
                i += kmax
                k = 1
                kmax = k
            if k > kmax:
                kmax = k
        # Normalize: random binary string has LZ ~ n/log2(n)
        import math
        max_lz = n / max(math.log2(n), 1.0)
        return min(1.0, complexity / max(max_lz, 1.0))

    # ================================================================
    # LANGUAGE BRIDGE (Level 12)
    # ================================================================

    def attach_language_bridge(self, bridge=None):
        """
        Aktiviert Level 12: Sprachfähigkeit.
        
        Args:
            bridge: ExperienceNarrator oder TrainingOrchestrator
                    Wenn None, wird ein Standard-Narrator erstellt.
        """
        if bridge is None:
            from src.bridge.llm_bridge import ExperienceNarrator
            bridge = ExperienceNarrator()
        self.language_bridge = bridge

    def narrate_experience(self) -> str:
        """Die Kreatur beschreibt ihren aktuellen Zustand in Sprache."""
        if self.language_bridge is None:
            return "(keine Sprachfähigkeit)"
        state = self.get_state()
        return self.language_bridge.narrate_state(state)

    def narrate_training(self, stats: Dict) -> str:
        """Die Kreatur beschreibt ihren Trainingsfortschritt."""
        if self.language_bridge is None:
            return "(keine Sprachfähigkeit)"
        return self.language_bridge.narrate_training(stats)

    def understand_task(self, text: str) -> Dict:
        """Die Kreatur versteht eine Aufgabe in natürlicher Sprache."""
        from src.bridge.llm_bridge import TaskParser
        parser = TaskParser()
        return parser.parse(text).__dict__

    # ================================================================
    # Level 13: Theory of Mind Activation
    # ================================================================

    def enable_tom(self, n_agents: int = 2, device: str = 'cpu'):
        """
        Activate Theory of Mind for multi-agent scenarios.

        Creates a ToM module that models other agents' behavior
        using mirror neurons and cooperation decisions.

        Args:
            n_agents: Number of other agents to model.
            device: Torch device.
        """
        if self.tom is not None:
            return  # Already active

        config = ToMConfig(
            n_mirror_neurons=200,
            n_decision_neurons=100,
            n_observed_features=10,
            n_action_features=self.n_motors,
            device=device,
        )
        self.tom = TheoryOfMind(config)
        self._tom_n_agents = n_agents

    def disable_tom(self):
        """Deactivate Theory of Mind."""
        self.tom = None
        self._tom_n_agents = 0

    def get_state(self) -> Dict:
        """Kognitiver Zustand für Dashboard/Logging."""
        return {
            'prediction_error': self._prediction_error,
            'curiosity_reward': self._curiosity_reward,
            'cumulative_curiosity': self._cumulative_curiosity,
            'empowerment_reward': self._empowerment_reward,
            'empowerment': self.empowerment.last_empowerment,
            'curiosity_boredom': self.curiosity.boredom_counter,
            'curiosity_novelty': self.curiosity.last_prediction_error,
            'gwt_winner': self._gwt_winner,
            'consciousness_level': self._consciousness_level,
            'emotion': self.emotions.get_state(),
            'body_schema': self.body_schema.get_state(),
            'drives': self.drives.get_state(),
            'metacognition': self.metacognition.get_state(),
            'consistency': self.consistency.get_state(),
            'memory': self.memory.get_state(),
            'world_model': self.world_model.get_state(),
            'synaptogenesis': self.synaptogenesis.get_stats(),
            'knowledge_graph': self.synaptogenesis.graph.to_dict(),
            'astrocytes': self.astrocytes.get_stats(),
            'pci': self._last_pci,
            'tom_active': self.tom is not None,
            'replay_buffer_size': self.dream_engine.get_replay_buffer_size(),
            'neuromod': dict(self.snn.neuromod_levels),
            'step_count': self._step_count,
            'hebbian_updates': self._hebbian_updates,
            'hebbian_total_dw': self._hebbian_total_dw,
            'hebbian_step_dw': self._hebbian_step_dw,
            'skills': {
                'active': self.skills.get_active_skills(),
                'frozen': self.skills.get_frozen_skills(),
                'current': self.skills._current_skill,
                'count': len(self.skills.skills),
            },
            'behavior': self.behavior_planner.get_state(),
            'behavior_motor': self.behavior_executor.get_state() if self._motor_modulation else {},
            'language_bridge_active': self.language_bridge is not None,
            'has_dreamed': self._has_dreamed,
        }
