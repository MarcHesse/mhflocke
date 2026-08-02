#!/usr/bin/env python3
"""
MH-FLOCKE — Baby-KI Training v0.8.0-alpha
=========================================
"A puppy learns to walk because falling feels bad and moving feels good."

Based on train_v032.py v0.4.3 with ONE surgical change:
  The reward signal sent to creature.step() is replaced by intrinsic reward
  from CognitiveBrain.get_intrinsic_reward(). External reward is still
  computed for FLOG logging and metrics but NOT used for learning.
  --reward-blend controls the mix (0.0 = pure intrinsic, 1.0 = pure external).

v0.4.3: Obstacle avoidance, graded DCN, additive CPG blending.
v0.4.8: Run-and-Tumble chemotaxis replaces continuous olfactory steering.
v0.5.x: Phototaxis navigation logging — ground truth pos/dist plus the
  creature's own SpatialMap snapshot (landmarks + visit grid) so a
  renderer can show both the world view and the dog's internal map.
#   v0.4.2: Scalable SNN for hardware brain transfer.
  - --n-hidden: SNN hidden neuron count (default from profile.json)
  - --hardware-sensors: Bridge v2.5 sensor layout (12 servo + 2 CPG + 4 IMU)
  - --no-vision: Disable visual channels for camera-less robots
  - Cerebellar populations scale proportionally for small neuron counts
  - Full 15-step cognitive loop runs with 232 neurons (Freenove)

The creature learns WHAT to do (knowledge), WHY (drives), and HOW (actor + cerebellum).

Full pipeline:
  User: "walk on hilly grassland"
    -> TaskParser -> UnderstandEngine (LLM or builtin)
    -> SceneInstruction (auto-generated from knowledge)
    -> Terrain (real MuJoCo heightfield)
    -> CPG baseline + SNN Actor (R-STDP) + Cerebellum (Marr-Albus-Ito)
    -> Reward -> DA -> reinforces good patterns
    -> Balance errors from terrain -> CF -> cerebellum corrects
    -> Competence gate: CPG fades only when actor proves speed

What's new vs v0.3.2:
  - Issue #57: Autonomous Drive Loop — BehaviorPlanner + MotorPattern → CPG modulation
    Drives decide behavior (walk/trot/sniff/rest/alert), MotorPattern scales CPG freq/amp.
    The creature has its own motivation instead of relying purely on external reward.
  - CPG loads evolved params from creatures/{name}/cpg_config.json (if present)
  - creatures/ directory is the central registry for all creature data

Usage:
  python scripts/train_v032.py --scene "walk on hilly grassland"
  python scripts/train_v032.py --scene "run on rocky terrain" --steps 200000
  python scripts/train_v032.py --scene "walk on flat meadow" --no-terrain
  python scripts/train_v032.py --steps 50000 --difficulty 0.4

  # Freenove brain transfer (232 neurons, hardware-matched sensors):
  python scripts/train_v032.py --creature-name freenove \\
    --scene "walk on flat meadow" --steps 50000 --no-terrain --no-sensory \\
    --no-vision --hardware-sensors --auto-reset 500

Author: MH-FLOCKE Level 15 v0.4.3
"""

__version__ = "0.8.3"     # module version (MAJOR.MINOR); Baby-KI trainer
__logbook__ = 63          # mh-logbuch module entry
__status__  = "active"     # active | veraltet | neu

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if sys.platform != 'win32':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import time
import json
import base64
import logging
import numpy as np
import torch
import mujoco
from collections import deque
import gc as _gc

from src.body.genome import Genome
from src.body.mujoco_creature import MuJoCoCreature, MuJoCoCreatureBuilder
from src.body.mujoco_world import MuJoCoWorld
from src.brain.cerebellar_learning import CerebellarLearning, CerebellarConfig
from src.brain.spinal_reflexes import SpinalReflexes, ReflexConfig, SpinalSegments, SpinalSegmentConfig
from src.brain.spinal_cpg import SpinalCPG, SpinalCPGConfig
from src.brain.mogli_oscillator import MogliCPG, MogliConfig
from src.brain.developmental_schedule import DevelopmentalSchedule, DevelopmentalConfig
from src.brain.gait_quality import GaitQualityAnalyzer, GaitQualityConfig, GAIT_QUALITY_VERSION
from src.brain.body_awareness import BodyAwareness, BODY_AWARENESS_VERSION
from src.brain.spatial_map import SpatialMap, SPATIAL_MAP_VERSION
from src.brain.directed_learning import DirectedLearning, DIRECTED_LEARNING_VERSION
from src.brain.episode_analyzer import EpisodeAnalyzer
from src.brain.strategy_adapter import StrategyAdapter
from src.brain.curiosity_hypothesis import CuriosityExplorer, HypothesisGenerator
from src.body.terrain import (
    TerrainConfig, generate_heightfield, inject_terrain, inject_terrain_geoms,
    terrain_type_from_scene, difficulty_from_scene, inject_ball, inject_wall,
    inject_light,
)

logger = logging.getLogger(__name__)


# === SNN DIAGNOSTICS (temporary, remove after debugging) ===
def diagnose_snn(creature, step):
    """Full SNN signal chain diagnosis."""
    snn = creature.snn
    n = snn.config.n_neurons
    pops = snn.populations

    print(f'\n{"="*70}')
    print(f'  SNN DIAGNOSIS — Step {step}')
    print(f'{"="*70}')

    # 1. Topology
    pop_sizes = {name: len(ids) for name, ids in pops.items()}
    print(f'  Topology: {n} total | {pop_sizes}')
    print(f'  hardware_sensors={creature._hardware_sensors} | '
          f'n_sensors={creature._n_sensor_channels} | '
          f'n_motors={creature._n_motors}')
    print(f'  qpos_offset={creature._qpos_offset} | '
          f'qvel_offset={creature._qvel_offset} | '
          f'motor_scale={getattr(creature, "motor_scale", "N/A")}')

    # 2. Sensor Input
    sensor_input = creature.get_sensor_input()
    si_nz = (sensor_input.abs() > 1e-6).sum().item()
    print(f'\n  SENSOR INPUT [{n}]:')
    print(f'    nonzero={si_nz}/{n} min={sensor_input.min().item():.4f} '
          f'max={sensor_input.max().item():.4f}')
    for pname in ['input', 'motor_hidden', 'output']:
        ids = pops.get(pname, [])
        if len(ids) > 0:
            vals = sensor_input[ids]
            nz = (vals.abs() > 1e-6).sum().item()
            print(f'    {pname:20s}: nonzero={nz}/{len(ids)} '
                  f'max={vals.max().item():.4f}')
    inp_ids = pops.get('input', [])
    if len(inp_ids) > 0:
        first20 = sensor_input[inp_ids[:20]].tolist()
        print(f'    First 20 input: {[f"{v:.3f}" for v in first20]}')

    # 3. Membrane Potentials
    V = snn.V
    u = snn._u
    print(f'\n  MEMBRANE V & RECOVERY u:')
    for pname, ids in pops.items():
        if len(ids) > 0 and pname != 'mossy_fibers':
            Vp = V[ids]; up = u[ids]
            print(f'    {pname:20s}: V [{Vp.min().item():7.1f}, {Vp.max().item():7.1f}] '
                  f'mean={Vp.mean().item():7.1f} | u mean={up.mean().item():7.1f}')

    # 4. Spike threshold proximity
    print(f'\n  THRESHOLD PROXIMITY (Izh fires at V>=30):')
    for pname in ['input', 'motor_hidden', 'output', 'granule_cells', 'dcn']:
        ids = pops.get(pname, [])
        if len(ids) > 0:
            Vp = V[ids]
            at_rest = ((Vp > -70) & (Vp < -60)).sum().item()
            rising = (Vp > -20).sum().item()
            spiking = (Vp >= 30).sum().item()
            print(f'    {pname:20s}: rest={at_rest} rising={rising} spiking={spiking}')

    # 5. Accumulated Spikes
    acc = getattr(creature, '_accumulated_spikes', None)
    if acc is not None:
        print(f'\n  SPIKES (last frame, {creature.SNN_SUBSTEPS} substeps):')
        for pname, ids in pops.items():
            if len(ids) > 0 and pname != 'mossy_fibers':
                ps = acc[ids].sum().item()
                rate = ps / (len(ids) * creature.SNN_SUBSTEPS) if ps > 0 else 0
                print(f'    {pname:20s}: {ps:6.0f} spikes '
                      f'(rate={rate:.4f}/neuron/substep)')

    # 6. Connectivity per pathway
    print(f'\n  CONNECTIVITY ({snn._n_synapses} synapses):')
    if snn._weight_values is not None:
        wv = snn._weight_values
        idx = snn._weight_indices
        for src_n, tgt_n in [('input','motor_hidden'), ('input','granule_cells'),
                             ('motor_hidden','output'), ('motor_hidden','motor_hidden'),
                             ('dcn','output'), ('granule_cells','output')]:
            src_ids = pops.get(src_n, [])
            tgt_ids = pops.get(tgt_n, [])
            if len(src_ids) > 0 and len(tgt_ids) > 0:
                src_s = set(src_ids.tolist()) if torch.is_tensor(src_ids) else set(src_ids)
                tgt_s = set(tgt_ids.tolist()) if torch.is_tensor(tgt_ids) else set(tgt_ids)
                mask = torch.zeros(idx.shape[1], dtype=torch.bool)
                for i in range(idx.shape[1]):
                    if idx[0,i].item() in src_s and idx[1,i].item() in tgt_s:
                        mask[i] = True
                nc = mask.sum().item()
                if nc > 0:
                    pw = wv[mask]
                    print(f'    {src_n}->{tgt_n}: {nc} syn '
                          f'w=[{pw.min().item():.3f},{pw.max().item():.3f}] '
                          f'abs_mean={pw.abs().mean().item():.3f}')
                else:
                    print(f'    {src_n}->{tgt_n}: NO CONNECTIONS!')

    # 7. Tonic current bug check
    tonic = getattr(snn, '_hidden_tonic_current', 0.0)
    has_hidden = 'hidden' in pops
    has_mh = 'motor_hidden' in pops
    print(f'\n  TONIC CURRENT: {tonic:.4f}')
    print(f'    pop "hidden" exists: {has_hidden}')
    print(f'    pop "motor_hidden" exists: {has_mh}')
    if not has_hidden and has_mh and tonic > 0:
        print(f'    >>> BUG: tonic={tonic} set but code checks for "hidden"!')
        print(f'    >>> motor_hidden gets ZERO tonic current!')

    # 8. Neuromodulation
    ne = snn.neuromod_levels.get('ne', 0.0)
    da = snn.neuromod_levels.get('da', 0.0)
    print(f'\n  NEUROMOD: DA={da:.3f} NE={ne:.3f}')

    # 9. Critical analysis
    si_max = sensor_input.max().item()
    print(f'\n  CRITICAL:')
    print(f'    max sensor_input={si_max:.4f} -> after *10 = {si_max*10:.1f} mV')
    print(f'    Izh RS at rest: dV = -3 + I   (need I > 3 mV for dV > 0)')
    print(f'    Input neurons: direct I={si_max*10:.1f} mV -> should fire')
    print(f'    Motor_hidden: synaptic only (w=0.5-1.5 * 10 = 5-15 mV/spike)')
    print(f'{"="*70}\n')
# === END SNN DIAGNOSTICS ===


def resolve_creature_paths(creature_name: str, xml_arg: str):
    """
    Resolve creature file paths from the creatures/ registry.

    Search order for XML:
      1. Explicit --xml argument (if not default)
      2. creatures/{name}/scene_mhflocke.xml (Go2 / Menagerie models)
      3. creatures/{name}/creature.xml
      4. editor/creatures/{name}_creature.xml (legacy)
      5. Fall back to --xml default

    Search order for CPG config:
      1. creatures/{name}/cpg_config.json
      2. checkpoints/{name}/cpg_config.json (legacy)
      3. None (use defaults)

    Returns:
        (xml_path, cpg_config_path_or_None, profile_or_None)
    """
    name_lower = creature_name.lower()
    default_xml = 'creatures/dm_quadruped/creature.xml'

    # --- XML ---
    if xml_arg != default_xml and os.path.exists(xml_arg):
        xml_path = xml_arg
    elif os.path.exists(f'creatures/{name_lower}/scene_mhflocke.xml'):
        # Menagerie / external MJCF with <include> — must use from_xml_path
        xml_path = f'creatures/{name_lower}/scene_mhflocke.xml'
    elif os.path.exists(f'creatures/{name_lower}/creature.xml'):
        xml_path = f'creatures/{name_lower}/creature.xml'
    elif os.path.exists(f'editor/creatures/{name_lower}_creature.xml'):
        xml_path = f'editor/creatures/{name_lower}_creature.xml'
    else:
        xml_path = xml_arg

    # --- CPG Config ---
    cpg_config_path = None
    candidates = [
        f'creatures/{name_lower}/cpg_config.json',
        f'checkpoints/{name_lower}/cpg_config.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            cpg_config_path = path
            break

    # --- Profile (Go2 etc.) ---
    profile = None
    profile_path = f'creatures/{name_lower}/profile.json'
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)

    return xml_path, cpg_config_path, profile


# Phase 0: Knowledge Engine
def acquire_knowledge(scene_text, creature_type='dog', use_llm=True):
    try:
        from src.bridge.llm_bridge import TaskParser
        from src.bridge.understand import UnderstandEngine
    except ImportError:
        TaskParser = None
        UnderstandEngine = None
    from src.behavior.scene_instruction import SceneInstruction, SCENE_INSTRUCTIONS

    result = {'scene_instruction': None, 'terrain_config': None,
              'behaviors': [], 'understand_result': None, 'source': 'builtin'}

    # Fallback if bridge modules not available (public release)
    if TaskParser is None or UnderstandEngine is None:
        print('  Bridge modules not available — using builtin defaults')
        terrain_type = terrain_type_from_scene(scene_text)
        difficulty = difficulty_from_scene(scene_text)
        terrain_cfg = TerrainConfig(terrain_type=terrain_type, difficulty=difficulty)
        result['terrain_config'] = terrain_cfg
        si = SceneInstruction(text=scene_text,
                              description=f'Default: {scene_text}',
                              drive_biases={'exploration': 0.7, 'play': 0.5},
                              behavior_weights={'walk': 1.0, 'trot': 0.6})
        result['scene_instruction'] = si
        print(f'  Terrain: {terrain_type} (difficulty={difficulty:.2f})')
        return result

    parser = TaskParser()
    task = parser.parse(scene_text)
    print(f'  TaskParser: type={task.task_type.value} '
          f'env={task.environment_hints} diff={task.difficulty:.2f} '
          f'conf={task.confidence:.2f}')

    llm_adapter = None
    if use_llm:
        try:
            from src.utils.config import LLM_API_KEYS
            from src.llm.llm_adapter import MultiLLMAdapter
            llm_adapter = MultiLLMAdapter(keys=LLM_API_KEYS)
            if llm_adapter.enabled:
                stats = llm_adapter.get_statistics()
                active = [p['name'] for p in stats['providers'] if p['enabled']]
                print(f'  LLM: {len(active)} providers active: {", ".join(active)}')
            else:
                print(f'  LLM: no API keys configured, using builtin knowledge')
                llm_adapter = None
        except Exception as e:
            print(f'  LLM: init failed ({e}), using builtin knowledge')
            llm_adapter = None

    # dm_quadruped IS a dog — all knowledge/behavior lookup uses 'dog'
    knowledge_type = 'dog'

    engine = UnderstandEngine(llm_adapter=llm_adapter)
    understand = engine.understand(task, creature_type=knowledge_type)
    result['understand_result'] = understand
    result['behaviors'] = understand.behaviors
    result['source'] = understand.source

    print(f'  Knowledge: {len(understand.behaviors)} behaviors (source: {understand.source})')
    for b in understand.behaviors[:5]:
        print(f'    - {b.name} (p={b.priority:.1f}, drive={b.drive}): {b.description}')

    env_name = task.environment_hints[0] if task.environment_hints else ''
    preset = SCENE_INSTRUCTIONS.get(env_name)
    if preset:
        result['scene_instruction'] = preset
        print(f'  Scene: preset "{env_name}"')
    else:
        drive_biases = {}
        behavior_weights = {}
        for b in understand.behaviors:
            drive_biases[b.drive] = max(drive_biases.get(b.drive, 0.0), b.priority)
            behavior_weights[b.name] = b.priority
        si = SceneInstruction(text=scene_text,
                              description=f'Auto-generated from: {scene_text}',
                              drive_biases=drive_biases,
                              behavior_weights=behavior_weights)
        result['scene_instruction'] = si
        print(f'  Scene: auto-generated (drives: {drive_biases})')

    terrain_type = terrain_type_from_scene(scene_text)
    difficulty = difficulty_from_scene(scene_text)
    if task.difficulty > 0.5 and task.confidence > 0.5:
        difficulty = task.difficulty * 0.8
    terrain_cfg = TerrainConfig(terrain_type=terrain_type, difficulty=difficulty)
    result['terrain_config'] = terrain_cfg
    print(f'  Terrain: {terrain_type} (difficulty={difficulty:.2f})')

    try:
        from src.behavior.behavior_knowledge import BehaviorKnowledge
        bk = BehaviorKnowledge(creature_type=knowledge_type)
        injected = engine.inject_into_behavior_knowledge(understand, bk)
        if injected > 0:
            print(f'  Injected {injected} new behaviors into BehaviorKnowledge')
    except Exception as e:
        logger.debug(f'BehaviorKnowledge injection skipped: {e}')

    return result


def validate_morphology(xml_path, timestep=0.005):
    import re
    result = {'passed': False, 'errors': [], 'warnings': []}
    if not os.path.exists(xml_path):
        result['errors'].append(f'XML not found: {xml_path}')
        return result
    try:
        # Try from_xml_path first (handles <include>, meshdir, etc.)
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
        except Exception:
            # Fallback: load as string (legacy creatures without <include>)
            with open(xml_path, encoding='utf-8') as f:
                xml = f.read()
            xml = re.sub(r'timestep="[0-9.]+"', f'timestep="{timestep}"', xml)
            model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception as e:
        result['errors'].append(f'XML parse error: {e}')
        return result
    result['n_actuators'] = model.nu
    mujoco.mj_forward(model, data)
    result['init_height'] = round(float(data.qpos[2]), 4)
    result['passed'] = len(result['errors']) == 0
    return result


def patch_xml_timestep(xml_path, new_timestep=0.005):
    import re
    with open(xml_path, encoding='utf-8') as f:
        xml = f.read()
    return re.sub(r'timestep="[0-9.]+"', f'timestep="{new_timestep}"', xml)


class CompetenceGate:
    """Competence-gated CPG->Actor handoff. Resolves Issue #45.
    
    v0.6.0: Pure IMU gate. Only uses signals available on real hardware:
    upright (from IMU accelerometer) and fall detection. No velocity,
    no world position — the real robot doesn't have those.
    
    Biology: Motor competence in neonates is assessed by postural
    stability. A foal that stands steadily is ready to walk.
    The vestibular system (= IMU) is the primary feedback.
    
    Logic:
    - Upright and not fallen → competence grows
    - Fallen → competence shrinks (fast)
    - Not upright but not fallen → competence shrinks (slow)
    - CPG weight decreases as competence increases
    """

    def __init__(self, grow_rate=0.0002, shrink_rate=0.0003,
                 cpg_min=0.40, cpg_max=0.9, upright_threshold=0.6,
                 stability_window=1000, **kwargs):
        self.grow_rate = grow_rate
        self.shrink_rate = shrink_rate
        self.cpg_min = cpg_min
        self.cpg_max = cpg_max
        self.upright_threshold = upright_threshold
        self.actor_competence = 0.0
        self.cpg_weight = cpg_max
        # Stability tracking
        self.stability_window = stability_window
        self._recent_falls = 0
        self._window_step = 0
        self._fall_rate = 0.0        # falls per 1000 steps
        self._upright_ema = 1.0      # smoothed upright value
        # Keep for backward compat but unused in gate logic
        self.vel_ema = 0.0
        self.speed_threshold = kwargs.get('speed_threshold', 0.0)

    def update(self, step, vel_mps, is_fallen, upright=1.0):
        # Track upright EMA (IMU-derived, available on real hardware)
        self._upright_ema = self._upright_ema * 0.995 + upright * 0.005
        
        # Track fall rate in sliding window
        self._window_step += 1
        if is_fallen:
            self._recent_falls += 1
        if self._window_step >= self.stability_window:
            self._fall_rate = self._recent_falls / (self.stability_window / 1000.0)
            self._recent_falls = 0
            self._window_step = 0

        # Keep vel_ema updated for logging only (not used in gate logic)
        self.vel_ema = self.vel_ema * 0.99 + vel_mps * 0.01

        if is_fallen:
            # Fallen: fast shrink
            self.actor_competence = max(0.0, self.actor_competence - self.shrink_rate * 5)
            self._recompute_cpg()
            return
        
        # Pure IMU gate: only upright stability matters
        is_upright = upright > self.upright_threshold
        is_stable = self._fall_rate < 5.0 and self._upright_ema > self.upright_threshold
        
        if is_upright and is_stable:
            # Standing stably: competence grows
            self.actor_competence = min(1.0, self.actor_competence + self.grow_rate)
        elif is_upright:
            # Momentarily upright but recent instability: slow growth
            self.actor_competence = min(1.0, self.actor_competence + self.grow_rate * 0.3)
        else:
            # Not upright, not fallen (tilted, sliding): mild shrink
            self.actor_competence = max(0.0, self.actor_competence - self.shrink_rate)
        
        self._recompute_cpg()
    
    def _recompute_cpg(self):
        """CPG weight from actor competence. Higher competence = less CPG."""
        self.cpg_weight = max(self.cpg_min,
                              self.cpg_max - self.actor_competence * (self.cpg_max - self.cpg_min))

    def get_cpg_weight(self):
        return self.cpg_weight

    def get_stats(self):
        is_stable = self._fall_rate < 5.0 and self._upright_ema > self.upright_threshold
        return {'actor_competence': self.actor_competence, 'cpg_weight': self.cpg_weight,
                'vel_ema': self.vel_ema, 'fall_rate': self._fall_rate,
                'upright_ema': self._upright_ema, 'is_stable': is_stable}


def main():
    main._ball_ep = 0  # Ball episode counter (persists across resets)
    main._ball_stage = 0  # Curriculum stage (Issue #86)
    main._ball_best_dist = 99.0  # Best ball distance in current stage
    # Ball curriculum: start close and centered, progressively harder
    main._ball_positions = [
        (1.5, 0.0, 0.12),   # Stage 0: straight ahead, easy
        (2.0, 0.5, 0.12),   # Stage 1: slight lateral (~14°)
        (2.5, 1.0, 0.12),   # Stage 2: moderate (~22°)
        (3.0, 1.5, 0.12),   # Stage 3: significant (~27°)
        (3.0, 2.0, 0.12),   # Stage 4: original position (~34°)
    ]
    parser = argparse.ArgumentParser(description='MH-FLOCKE Level 15 v0.4.3')
    parser.add_argument('--scene', type=str, default='walk on hilly grassland')
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--xml', type=str, default='creatures/dm_quadruped/creature.xml')
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--record-interval', type=int, default=10,
                        help='Cadence (steps) for logging the body POSE (qpos/qvel) used for MuJoCo '
                             'replay in render_bittle. Default 10 (~20 Hz, bit-identical). Set 1 for '
                             'smooth render-quality playback (denser FLOG; dynamics unchanged).')
    parser.add_argument('--coord-reward-weight', type=float, default=0.0,
                        help='Increment b (#208): weight of buffered IMU gait-band coordination '
                             'reward (acc_x+pitch concentration over ~2000-step rolling buffer). '
                             '0.0 = OFF (bit-identical baseline).')
    parser.add_argument('--block-aversion-weight', type=float, default=0.0,
                        help='#213/#206: weight of the INTRINSIC block-aversion term — a persistent '
                             'yaw-scrub (rolling yaw_rate std) added to vestibular_discomfort so that '
                             'being stuck against a wall becomes intrinsically aversive. NOT external '
                             'reward and NOT a hardcoded turn: it only makes the blocked state '
                             'unpleasant; the existing R-STDP/drive machinery must LEARN that turning '
                             'reduces it. 0.0 = OFF (bit-identical baseline).')
    parser.add_argument('--block-aversion-window', type=int, default=200,
                        help='Rolling window (steps) for the yaw-scrub block signal (default 200).')
    parser.add_argument('--imu-obstacle', action='store_true',
                        help='Blind-IMU obstacle avoidance via Run-and-Tumble (#108 RT): the IMU '
                             'block signal drives a discrete RUN (straight, full gait = kwkF) -> '
                             'SNIFF (evaluate block_aversion) -> TUMBLE (committed turn at full gait '
                             '= kwkL) cycle, mirroring the chemotaxis state machine. Replaces the '
                             'continuous throttle+bias reflex that traps the robot in slow-turn '
                             'limbo. No external reward (intrinsic). Default OFF (bit-identical).')
    parser.add_argument('--imu-ob-run', type=int, default=60,
                        help='Run-and-Tumble: straight RUN length (steps) between block evaluations (default 60).')
    parser.add_argument('--imu-ob-tumble', type=int, default=40,
                        help='Run-and-Tumble: committed TUMBLE (turn) length in steps (default 40).')
    parser.add_argument('--imu-ob-turn-gain', type=float, default=1.0,
                        help='Run-and-Tumble: committed turn steering magnitude during TUMBLE (kwkL strength, default 1.0).')
    parser.add_argument('--imu-ob-block-on', type=float, default=0.45,
                        help='Run-and-Tumble: block_aversion at a SNIFF above which the robot tumbles (default 0.45).')
    parser.add_argument('--wall-memory-weight', type=float, default=0.0,
                        help='Task #84 step 2: weight of the anticipatory boundary-aversion term '
                             'added to block_aversion when near a remembered DANGER landmark (the '
                             'wall, anchored once at first contact in step 1). Reuses the existing '
                             'aversion+Run-and-Tumble machinery: in the canonical run (--imu-obstacle) '
                             'it makes the existing tumble fire EARLIER (anticipatory turn-away); with '
                             '--block-aversion-weight>0 it ALSO feeds the intrinsic learn signal. '
                             '0.0 = OFF (bit-identical). SCAFFOLD: the landmark geometry is read from '
                             'the dead-reckoned map (privileged vel_mps/cur_x); HW-honest IMU odometry '
                             'is the named follow-up.')
    parser.add_argument('--wall-memory-radius', type=float, default=0.25,
                        help='Task #84 step 2: anticipation radius (m) for the remembered-danger term; '
                             'aversion ramps from 0 at this distance to max at the landmark.')
    parser.add_argument('--danger-steer-weight', type=float, default=0.0,
                        help='Task #84 step 4: the missing ACTOR. Steps 1-3 all produce a SIGNAL '
                             '(block_aversion, wall_mem, curiosity) but the only thing that ever turns '
                             'the body is the Run-and-Tumble SNIFF -- a discrete check every ~40 steps. '
                             'Between SNIFFs the remembered wall has no path to the motor at all. This '
                             'term adds a quiet, CONTINUOUS away-from-danger steering bias straight into '
                             '_cpg_steering whenever a remembered DANGER landmark sits ahead and close. '
                             'It is neither reward nor curiosity -- it is a drive, and it acts every step. '
                             'Magnitude is constant (this weight), shaped only by proximity x aheadness. '
                             'Applied BEFORE the efference-copy buffer, so the yaw it generates is '
                             'subtracted out of block_aversion and cannot be mistaken for wall scrub. '
                             '0.0 = OFF (bit-identical). Suggested first probe: 0.15-0.30.')
    parser.add_argument('--danger-steer-radius', type=float, default=0.35,
                        help='Task #84 step 4: radius (m) within which the away-from-danger drive acts. '
                             'Slightly wider than --wall-memory-radius by default, so the body starts '
                             'drifting away before the aversion signal itself peaks.')
    parser.add_argument('--steering-mode', choices=('offset', 'gait_blend'), default='offset',
                        help='Task #92: HOW a steering command turns the body (OpenCat creatures only). '
                             "'offset' (default) is the legacy path: a static bias on the shoulder joints. "
                             'Measured in isolation (knowledge #271): ~4 deg/s of steering span at full '
                             'lock, against a gait that drifts 2.84 deg/s on its own with no command. The '
                             'command is smaller than the drift it has to fight -- effectively inert. The '
                             'reason is structural: a trot turns by taking LONGER STRIDES on the outside '
                             'of the curve, and an offset to the rest pose leaves stride length symmetric. '
                             "'gait_blend' blends the whole stride toward OpenCat's own turning table "
                             '(trF -> trL, mirrored for right turns). Measured: 13.6 deg/s of span, 3.4x '
                             'the legacy path, and bit-identical to it at zero steering. Those tables come '
                             'from the OpenCat firmware, so this also works on the real Bittle. '
                             'Every recorded run predating task #92 used offset, which is why it stays '
                             'the default -- but no wall-avoidance mechanism can work with it.')
    parser.add_argument('--steering-constant', type=float, default=0.0,
                        help='Task #92/#94 DIAGNOSTIC: inject a constant steering command into '
                             '_cpg_steering every step (added AFTER all reflex/danger terms), so the '
                             'isolated turn test has a stimulus with no wall present. +left / -right, '
                             'magnitude ~0..1 (clamped by the controller). 0.0 = OFF (bit-identical). '
                             'This is a measurement scaffold, NOT a control mechanism -- it lets '
                             'analyze_turn_test.py measure the real deg/s that gait_blend produces.')
    parser.add_argument('--steer-undamped', type=float, default=0.0,
                        help='Task #92/#94 FIX: re-apply the pure turning component of gait_blend '
                             'UNDAMPED in apply_motor_output, so the steer survives the '
                             'cpg_weight*pd_scale damping that shrinks a 7.41 deg/s turn to ~1. '
                             'The forward gait stays damped (stability); only the turn-only delta '
                             'is added back at this weight. Needs --steering-mode gait_blend. '
                             '0.0 = OFF (bit-identical). First probe ~0.6 (fills the missing 60%%); '
                             'tune with analyze_turn_test.py. Joint deltas only -> HW-portable.')
    parser.add_argument('--curiosity-steer-weight', type=float, default=0.0,
                        help='Task #96 Weg 2: intrinsic curiosity steering. Every step, the SpatialMap '
                             'says which way the least-visited (unexplored) space lies '
                             '(direction_to_unexplored); this term steers the body that way, scaled by '
                             'the CuriosityExplorer drive (bored/high-PE = turn more) and by how '
                             'one-sided the novelty is. NOT external reward and NOT a hardcoded turn: '
                             'the robot turns toward where the world is still unknown. Wall avoidance '
                             'falls out for free -- behind a wall there is nothing new, so the vector '
                             'points away. Added into _cpg_steering; use with --steer-undamped so it '
                             'actually turns the body. 0.0 = OFF (bit-identical). First probe ~0.3-0.5.')
    parser.add_argument('--curiosity-steer-radius', type=float, default=2.0,
                        help='Task #96: radius (m) the unexplored-direction scan covers (default 2.0).')
    parser.add_argument('--wall-distance', type=float, default=0.0,
                        help='Override wall X distance in meters when scene contains a wall '
                             '(0.0 = use scene default 0.8/1.5/3.0). Used for the stuck/wall test.')
    parser.add_argument('--no-wall-reset', dest='no_wall_reset', action='store_true',
                        help='Do not teleport-reset on wall contact (Issue #103) — robot stays '
                             'blocked against the wall. For continuous learning / the wall test.')
    parser.add_argument('--timestep', type=float, default=0.005)
    parser.add_argument('--snn-substeps', type=int, default=3)
    parser.add_argument('--no-flog', action='store_true')
    parser.add_argument('--dashboard', action='store_true',
        help='Live dashboard: push the real flog_data dict to the ws://localhost:5001 '
             'broadcaster (dashboard_views #61) for src/viz/sim_live.html. Default OFF = '
             'bit-identical. Live cadence follows --log-every (use 1 for a smooth view); '
             'needs FLOG on and `pip install websockets`. Pure observability, no SNN change.')
    parser.add_argument('--no-cerebellum', action='store_true')
    parser.add_argument('--no-terrain', action='store_true')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--no-drives', action='store_true', help='Disable autonomous drive loop')
    parser.add_argument('--no-sensory', action='store_true', help='Disable sensory environment (no scent/sound)')
    parser.add_argument('--phototaxis', action='store_true',
                        help='Use light-based navigation instead of scent (camera bilateral brightness)')
    parser.add_argument('--drift-profile', type=str, default=None,
                        help='Path to hardware drift profile JSON (e.g. creatures/freenove/drift_profiles/measured_marc_01.json)')
    parser.add_argument('--difficulty', type=float, default=None)
    parser.add_argument('--pci-interval', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fresh', action='store_true',
                        help='Skip loading brain.pt — start with completely fresh SNN + cognitive state')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip-morph-check', action='store_true')
    parser.add_argument('--creature-name', type=str, default='Mogli')
    parser.add_argument('--auto-reset', type=int, default=0,
                        help='Auto-reset after N consecutive fallen steps (0=disabled, 500=recommended for Go2). '
                             'Biology: mother helps fallen pup. SNN/cerebellum weights preserved.')
    parser.add_argument('--n-hidden', type=int, default=None,
                        help='Number of hidden neurons in SNN. Default: from profile.json or 1000. '
                             'Freenove: 172, Go2: 1000+')
    parser.add_argument('--hardware-sensors', action='store_true',
                        help='Use hardware-matched sensor encoding (Bridge v2.5 layout). '
                             'Only channels available on real hardware: 12 servo + 2 CPG + 4 IMU. '
                             'Required for sim-to-real brain transfer.')
    parser.add_argument('--no-vision', action='store_true',
                        help='Disable visual heading/distance sensor channels. '
                             'Use for robots without camera/vision sensor.')
    parser.add_argument('--neural-cpg', action='store_true',
                        help='Use Mogli Oscillator (SNN half-center CPG) instead of '
                             'mathematical SpinalCPG.')
    parser.add_argument('--no-competence-gate', dest='no_competence_gate',
                        action='store_true',
                        help='Pin cpg_weight=1.0 (disable the CompetenceGate fade). '
                             'CPG/OpenCat gait then runs at FULL amplitude with '
                             'SNN/cerebellum corrections added on top -- identical in '
                             'form to the hardware bridge. A/B control for the '
                             'sim-cm-vs-hardware-m discrepancy (known_issue #60).')
    parser.add_argument('--no-balance', dest='no_balance',
                        action='store_true',
                        help='Disable the OpenCatBalance controller (IMU->servo '
                             'corrections) in simulation. Mirrors the hardware '
                             'bridge with balance OFF (gb). A/B control for the '
                             'sim ~38deg roll vs hardware ~10deg roll gap '
                             '(Task #51, known_issue #60).')
    parser.add_argument('--legacy-cerebellum', action='store_true',
                        help='Disable Izhikevich on cerebellum AND allow R-STDP on '
                             'cerebellar weights. Reproduces v0.4.3 Go2 paper results '
                             '(45.15m, 0 falls). Use with --creature-name go2.')
    parser.add_argument('--leg-damage', type=str, default='',
                        choices=['', 'FL', 'FR', 'RL', 'RR'],
                        help='Disable one leg (zero its actuators). Biology: leg injury.')
    parser.add_argument('--pin-base-at', type=int, default=-1,
                        help='DIAGNOSTIC: from this step, viscously lock the base x/y '
                             'translation (true "wedged" stuck; legs keep gaiting). '
                             '-1 = off. Observation-only test for accel stuck detection.')
    parser.add_argument('--leg-damage-at', type=int, default=0,
                        help='Step at which leg damage occurs (0=from start). '
                             'Biology: injury during locomotion. The creature must '
                             'detect the change and adapt autonomously.')
    parser.add_argument('--reward-blend', type=float, default=0.0,
                        help='Baby-KI reward blend: 0.0 = pure intrinsic (default), '
                             '1.0 = pure external (v0.4.3 behavior). '
                             '0.1 = 10%% external + 90%% intrinsic (Stufe 1).')
    parser.add_argument('--snn-motor-scale', type=float, default=None,
                        help='How strongly the SNN overlays the CPG pattern on the '
                             'motors (creature.motor_scale). Default None = 0.3 '
                             '(unchanged => bit-identical). This is the LAST untested '
                             'link in the learning chain (26.07.2026): substrate, '
                             'reward, modulator and eligibility were all measured and '
                             'repaired without any behavioural change, so the open '
                             'question is whether the SNN reaches the motors at all. '
                             'Raising this is a direct test; 0.0 disables the SNN '
                             'overlay entirely (control: does behaviour change AT ALL?).')
    parser.add_argument('--eligibility-decay', type=float, default=None,
                        help='Own decay for the ELIGIBILITY trace, per SNN step. '
                             'Default None = same as the STDP trace (0.95) => '
                             'bit-identical. Finding 26.07.2026: one constant served '
                             'two biologically distinct roles -- the coincidence '
                             'window (~20 ms) and the reward-waiting window (0.3-2 s, '
                             'Yagishita 2014). At 0.95 with --snn-substeps 10 the '
                             'half-life is ~1.35 control steps, so delayed reward is '
                             'structurally unlearnable. 0.999 gives ~693 SNN steps.')
    parser.add_argument('--eligibility-consume', type=float, default=None,
                        help='Factor the eligibility is multiplied by after each '
                             'apply_rstdp (default 0.3). 1.0 = reward does not consume '
                             'the trace. Default None = unchanged => bit-identical.')
    parser.add_argument('--pe-blend', type=float, default=None,
                        help='Weight of the prediction error in the R-STDP learning '
                             'signal: combined = (1-w)*R + w*(-PE). Default None = '
                             'config value 0.9 (unchanged => bit-identical). Measured '
                             '26.07.2026: the PE branch fires 98.7%% of calls and '
                             'contributes 70.8%% of the signal magnitude, so the '
                             'intrinsic drives (curiosity, empowerment, vestibular, '
                             'proprioception) barely shape learning. Lower w gives them '
                             'more weight; w=0 learns from R alone.')
    parser.add_argument('--reward-baseline', action='store_true',
                        help='Schultz 1997: modulate R-STDP with the DEVIATION from '
                             'expected reward instead of the raw value. Measured '
                             '25.07.2026: R has mean +0.68 (min -0.16), so the raw '
                             'modulator is positive >90%% of the time and dw is almost '
                             'always positive -- every active synapse is strengthened '
                             'regardless of its contribution, i.e. reward-modulated '
                             'learning degenerates into plain Hebbian potentiation. '
                             'Subtracting a running expectation makes the modulator '
                             'zero-mean: better than expected strengthens, worse '
                             'weakens. Default OFF => bit-identical.')
    parser.add_argument('--reward-baseline-alpha', type=float, default=0.01,
                        help='EMA rate for the reward expectation (default 0.01, i.e. '
                             'time constant ~100 R-STDP calls). Only used with '
                             '--reward-baseline.')
    parser.add_argument('--empowerment-weight', type=float, default=None,
                        help="Override CognitiveBrain empowerment_weight_intrinsic (the "
                             "weight of empowerment in the intrinsic-reward sum). Default "
                             "None = config value (unchanged => bit-identical). Set 0 to "
                             "drop empowerment from R for the A/B (#230). Changes r when "
                             "set => behaviour A/B, not a bit-identical regression.")
    parser.add_argument('--curiosity-learning-progress', action='store_true',
                        help='Lever C (Task #84): switch the CuriosityDrive from rewarding raw '
                             'prediction-error surprise to rewarding learning PROGRESS (error '
                             'decreasing). Unlearnable chaos (a wall the world-model cannot '
                             'predict) stops being rewarded -> stops being a magnet. Default off '
                             '=> bit-identical. Changes behaviour when set => A/B, not a regression.')
    parser.add_argument('--curiosity-lp-scale', type=float, default=1.0,
                        help='Lever C: scale of the learning-progress reward (default 1.0).')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_steps = args.steps
    log_every = args.log_every
    # Reproducibility: same seed = same result, even on GPU
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'\n{"="*65}')
    print(f'  MH-FLOCKE -- Baby-KI v0.8.0-alpha')
    print(f'  "A puppy learns to walk because falling feels bad."')
    _blend = args.reward_blend
    if _blend == 0.0:
        print(f'  Reward: PURE INTRINSIC (no external reward)')
    elif _blend == 1.0:
        print(f'  Reward: PURE EXTERNAL (v0.4.3 behavior)')
    else:
        print(f'  Reward: BLEND {_blend:.0%} external + {1.0-_blend:.0%} intrinsic')
    print(f'{"="*65}')
    print(f'  Scene: "{args.scene}"')
    print(f'  Steps: {total_steps:,}  Device: {device}')
    print(f'{"="*65}')

    # --- Resolve creature paths from creatures/ registry ---
    xml_path, cpg_config_path, profile = resolve_creature_paths(args.creature_name, args.xml)
    is_external_mjcf = xml_path.endswith('scene_mhflocke.xml')  # Go2 etc.
    print(f'\n  Creature: {args.creature_name}')
    print(f'  XML: {xml_path} ({"external MJCF" if is_external_mjcf else "inline"})')
    print(f'  CPG config: {cpg_config_path or "defaults (no evolved config found)"}')
    if profile:
        print(f'  Profile: {profile.get("n_joints", "?")} joints, '
              f'{profile.get("joints_per_leg", "?")} per leg, '
              f'standing_h={profile.get("standing_height", "?")}')
    # Go2 body name is 'base', dm_quadruped is '{name}_s0'
    root_body_name = profile.get('root_body', 'base') if profile else f'{args.creature_name.lower()}_s0'

    # --- Resolve n_hidden from CLI > profile > default ---
    n_hidden = args.n_hidden
    if n_hidden is None and profile and 'snn' in profile:
        n_hidden = profile['snn'].get('n_hidden', 1000)
    if n_hidden is None:
        n_hidden = 1000
    if args.hardware_sensors:
        print(f'  Hardware sensors: ON (Bridge v2.5 layout)')
    if args.no_vision:
        print(f'  Vision channels: DISABLED')
    print(f'  SNN hidden neurons: {n_hidden}')

    # Cerebellum mode flags
    _izh_cerebellum = not args.legacy_cerebellum
    _protect_cerebellum = not args.legacy_cerebellum
    if args.legacy_cerebellum:
        print(f'  Legacy cerebellum: ON (v0.4.3 compat — no Izhikevich, R-STDP on cerebellum)')

    print(f'\n  -- Phase 0: Knowledge Acquisition --')
    knowledge = acquire_knowledge(args.scene, creature_type='dog', use_llm=not args.no_llm)
    scene_inst = knowledge['scene_instruction']
    terrain_cfg = knowledge['terrain_config']
    if args.difficulty is not None:
        terrain_cfg.difficulty = args.difficulty
        print(f'  Difficulty override: {args.difficulty:.2f}')
    if args.no_terrain:
        terrain_cfg.terrain_type = 'flat'
        terrain_cfg.difficulty = 0.0
        print(f'  Terrain disabled (flat ground)')

    if not args.skip_morph_check:
        morph = validate_morphology(xml_path, args.timestep)
        if morph['passed']:
            print(f'\n  Morphology OK: {morph.get("n_actuators", "?")} actuators, h={morph.get("init_height", 0):.3f}m')
        else:
            print(f'  Morphology FAILED: {morph["errors"]}')
            sys.exit(1)

    print(f'\n  -- Phase 3: Building World --')

    genome = Genome()
    world = MuJoCoWorld(render=False)

    # Detect ball scene from scene text (inject ball as scene object, not model part)
    _scene_has_ball = any(w in args.scene.lower() for w in ['ball', 'toy', 'fetch', 'spielzeug'])

    # Detect wall/obstacle scene (Issue #103: ultrasonic obstacle avoidance)
    _scene_has_wall = any(w in args.scene.lower() for w in ['wall', 'wand', 'obstacle', 'hindernis', 'barrier'])
    _scene_has_light = getattr(args, 'phototaxis', False)  # Phototaxis: inject light source
    _wall_distance = 0.8  # meters from origin — close for fast episodic learning
    if 'far' in args.scene.lower() or 'weit' in args.scene.lower():
        _wall_distance = 1.5
    elif 'very far' in args.scene.lower():
        _wall_distance = 3.0
    if getattr(args, 'wall_distance', 0.0) and args.wall_distance > 0.0:
        _wall_distance = float(args.wall_distance)
        print(f'  Wall distance override: {_wall_distance:.2f}m (--wall-distance)')

    if is_external_mjcf:
        # Go2 / Menagerie: load via from_xml_path (handles <include>, meshdir)
        # Terrain/Ball injection for external MJCF: write patched XML as temp file
        # Use PID in filenames to support parallel runs
        _pid = os.getpid()
        hfield_path = os.path.join('output', f'mhflocke_terrain_{_pid}.png')
        os.makedirs('output', exist_ok=True)
        _needs_temp_xml = (terrain_cfg.terrain_type != 'flat') or _scene_has_ball or _scene_has_wall or _scene_has_light
        if _needs_temp_xml:
            with open(xml_path, encoding='utf-8') as f:
                xml_string = f.read()
            if terrain_cfg.terrain_type != 'flat':
                xml_string = inject_terrain_geoms(xml_string, terrain_cfg)
                print(f'  Terrain: 3D hill geoms (h_max={terrain_cfg.max_height * terrain_cfg.difficulty / 0.3:.3f}m)')
            else:
                print(f'  Terrain: flat (no heightfield)')
            if _scene_has_ball:
                _init_ball = main._ball_positions[0]  # Start at curriculum Stage 0
                xml_string = inject_ball(xml_string, pos=_init_ball)
                print(f'  Ball: injected at {_init_ball} -- curriculum Stage 0')
            if _scene_has_wall:
                xml_string = inject_wall(xml_string, distance=_wall_distance)
            if _scene_has_light:
                xml_string = inject_light(xml_string, pos=(2.0, 0.0, 0.02))
                print(f'  Light: injected at (2.0, 0.0) — 2m straight ahead')
            # Write temp file next to original (so <include> paths resolve)
            temp_xml = os.path.join(os.path.dirname(xml_path), f'_train_temp_{_pid}.xml')
            with open(temp_xml, 'w') as f:
                f.write(xml_string)
            creature = MuJoCoCreatureBuilder.build(
                genome, world=world, device=device,
                creature_name=args.creature_name.lower(),
                xml_path=temp_xml,
                n_hidden_neurons=n_hidden,
                hardware_sensors=args.hardware_sensors,
                no_vision=args.no_vision,
                profile=profile,
                izh_cerebellum=_izh_cerebellum,
                protect_cerebellum=_protect_cerebellum)
            os.remove(temp_xml)
        else:
            creature = MuJoCoCreatureBuilder.build(
                genome, world=world, device=device,
                creature_name=args.creature_name.lower(),
                xml_path=xml_path,
                n_hidden_neurons=n_hidden,
                hardware_sensors=args.hardware_sensors,
                no_vision=args.no_vision,
                profile=profile,
                izh_cerebellum=_izh_cerebellum,
                protect_cerebellum=_protect_cerebellum)
            print(f'  Terrain: flat (no heightfield)')
    else:
        # Legacy inline MJCF (dm_quadruped etc.)
        xml_string = patch_xml_timestep(xml_path, args.timestep)
        hfield_path = os.path.join('output', 'mhflocke_terrain.png')
        os.makedirs('output', exist_ok=True)
        if terrain_cfg.terrain_type != 'flat':
            xml_string = inject_terrain(xml_string, terrain_cfg, os.path.abspath(hfield_path))
            print(f'  Terrain injected: {terrain_cfg.terrain_type} (h_max={terrain_cfg.max_height * terrain_cfg.difficulty / 0.3:.3f}m)')
        else:
            print(f'  Terrain: flat (no heightfield)')
        if _scene_has_wall:
            xml_string = inject_wall(xml_string, distance=_wall_distance)
        if _scene_has_light:
            xml_string = inject_light(xml_string, pos=(2.0, 0.0, 0.02))
            print(f'  Light: injected at (2.0, 0.0) — 2m straight ahead')
        creature = MuJoCoCreatureBuilder.build(genome, world=world, device=device,
            creature_name=args.creature_name.lower(), xml_string=xml_string,
            n_hidden_neurons=n_hidden,
            hardware_sensors=args.hardware_sensors,
            no_vision=args.no_vision,
            profile=profile,
            izh_cerebellum=_izh_cerebellum,
            protect_cerebellum=_protect_cerebellum)
    creature.SNN_SUBSTEPS = args.snn_substeps
    # Protect cerebellar populations from CognitiveBrain learning
    # (GrC patterns must stay stable for Marr-Albus learning)
    # With --legacy-cerebellum: R-STDP can modify cerebellar weights (v0.4.3 behavior)
    if _protect_cerebellum:
        creature.snn.protected_populations = {
            'mossy_fibers', 'granule_cells', 'golgi_cells',
            'purkinje_cells', 'dcn',
        }
    creature.brain.config.pci_interval = args.pci_interval
    # Eligibility-Zeitkonstante (#215). Default None => Config unberuehrt.
    _el_args = (getattr(args, 'eligibility_decay', None),
                getattr(args, 'eligibility_consume', None))
    if any(a is not None for a in _el_args):
        _snn_el = getattr(creature, 'snn', None)
        if _snn_el is not None and hasattr(_snn_el, 'config'):
            if _el_args[0] is not None:
                # Lokaler Alias: main() enthaelt weiter unten ein 'import math',
                # wodurch 'math' fuer die gesamte Funktion lokal wird und hier noch
                # nicht gebunden ist (UnboundLocalError).
                import math as _math
                _snn_el.config.eligibility_decay = float(_el_args[0])
                _snn_el._elig_decay = float(_el_args[0])
                _d = float(_el_args[0])
                _hl = _math.log(0.5) / _math.log(_d) if 0 < _d < 1 else float('inf')
                print(f'  Eligibility decay: {_el_args[0]} per SNN step '
                      f'(half-life ~{_hl:.0f} SNN steps = ~{_hl/10:.0f} control steps '
                      f'at substeps 10; STDP coincidence trace unchanged at 0.95)')
            if _el_args[1] is not None:
                _snn_el.config.eligibility_consume = float(_el_args[1])
                print(f'  Eligibility consume: {_el_args[1]} after each reward')
        else:
            print('  WARNING: --eligibility-* requested but no SNN config found')
    # PE/R-Mischung im Lernsignal (#215). Default None => Config unberuehrt.
    if getattr(args, 'pe_blend', None) is not None:
        _snn_pe = getattr(creature, 'snn', None)
        if _snn_pe is not None and hasattr(_snn_pe, 'config'):
            _snn_pe.config.pe_blend = float(args.pe_blend)
            print(f'  PE blend: {args.pe_blend} '
                  f'(learning signal = {1.0-float(args.pe_blend):.2f}*R + '
                  f'{float(args.pe_blend):.2f}*(-PE))')
        else:
            print('  WARNING: --pe-blend requested but no SNN config found')
    # Reward-Baseline (Schultz 1997) -- siehe SNNConfig.reward_baseline.
    # Default off => Config unberuehrt => bit-identisch.
    if getattr(args, 'reward_baseline', False):
        _snn = getattr(creature, 'snn', None)
        if _snn is not None and hasattr(_snn, 'config'):
            _snn.config.reward_baseline = True
            _snn.config.reward_baseline_alpha = float(args.reward_baseline_alpha)
            print(f'  Reward baseline: ON (alpha={args.reward_baseline_alpha}) '
                  f'-- R-STDP modulates with R - E[R], not raw R')
        else:
            print('  WARNING: --reward-baseline requested but no SNN config found')
    # Empowerment A/B (#230): override the intrinsic-sum weight of empowerment.
    # Default None => leave the config value (1.0) => bit-identical. The empowerment
    # term currently saturates at a constant +0.30 offset (dead weight); set 0 to
    # remove it and A/B distance/behaviour against the default run (same seed).
    if args.empowerment_weight is not None:
        creature.brain.config.empowerment_weight_intrinsic = float(args.empowerment_weight)
        print(f'  Empowerment weight (intrinsic sum): {args.empowerment_weight} (override; A/B #230)')
    # Lever C (Task #84): learning-progress curiosity. Set live on the already-built
    # CuriosityDrive config (compute_intrinsic_reward reads self.config every step).
    # Default off => untouched => bit-identical.
    if args.curiosity_learning_progress and hasattr(creature.brain, 'curiosity'):
        creature.brain.curiosity.config.learning_progress_mode = True
        creature.brain.curiosity.config.lp_scale = float(args.curiosity_lp_scale)
        print(f'  Curiosity: learning-progress mode ON (lp_scale={args.curiosity_lp_scale}; Lever C, Task #84)')
    # Dream consolidation: enabled for wall training (Issue #103).
    # Periodic dreams at interval=500 (~every 16s at 30ms/step).
    # PLUS: explicit dream after each wall hit for obstacle pattern consolidation.
    # Previous: interval=0 (disabled) or 100 (too expensive at 400ms/cycle).
    # At 500: ~40 dream cycles per 20k run, manageable overhead.
    creature.brain.config.dream_interval = 500
    # Baby-KI: arousal drive disabled (v0.4.3-v0.4.5 experiments showed
    # it hurts distance: 0.30m/0.47m/1.57m vs 3.37m baseline without it).
    # RAS modulation doesn't help when the fundamental problem is that
    # intrinsic reward is direction-agnostic. Stufe 3 (Sensory Environment)
    # addresses this by giving the creature scent targets to walk toward.
    # if args.reward_blend < 1.0:
    #     creature.brain.config.intrinsic_arousal_drive = True
    #     print(f'  Arousal Drive: INTERNAL (brain controls NE + tonic from arousal)')
    print(f'  Arousal Drive: DISABLED (v0.4.5 conclusion: use Sensory Environment instead)')
    print(f'  Olfactory: Run-and-Tumble v0.4.8 (RUN={40} TUMBLE={12} dead_zone={0.15:.2f}rad min_sm={0.10})')
    creature.per_joint_scale = None
    creature.body_name = root_body_name
    standing_h = profile.get('standing_height', 0.48) if profile else 0.48
    creature._fallen_height_threshold = standing_h * 0.45  # ~45% of standing height (Go2: 0.12m)

    # --- PD Controller for torque-actuated robots (Go2) ---
    pd_controller = None
    if is_external_mjcf:
        n_act = world.n_actuators
        # PD gains from profile or defaults
        kp_vals = profile.get('pd_kp', [60, 60, 80] * (n_act // 3)) if profile else [60] * n_act
        kd_vals = profile.get('pd_kd', [2, 2, 3] * (n_act // 3)) if profile else [2] * n_act
        pd_kp = np.array(kp_vals[:n_act], dtype=np.float64)
        pd_kd = np.array(kd_vals[:n_act], dtype=np.float64)
        ctrl_lo = world._model.actuator_ctrlrange[:, 0].copy()
        ctrl_hi = world._model.actuator_ctrlrange[:, 1].copy()
        print(f'  PD Controller: Kp={pd_kp[0]:.0f}/{pd_kp[1]:.0f}/{pd_kp[2]:.0f}  Kd={pd_kd[0]:.1f}/{pd_kd[1]:.1f}/{pd_kd[2]:.1f}')
        # Standing pose: must be in ACTUATOR order (not qpos order!)
        # qpos order follows body-tree traversal, actuator order follows <actuator> section.
        # These can differ (e.g. Bittle: qpos has RF,RR,LF,LR but ctrl has RF,LF,RR,LR).
        import mujoco
        standing_qpos = np.zeros(n_act)
        for i in range(n_act):
            joint_id = world._model.actuator_trnid[i, 0]
            qpos_addr = world._model.jnt_qposadr[joint_id]
            standing_qpos[i] = world._data.qpos[qpos_addr]
        print(f'  Standing pose (actuator order): {np.degrees(standing_qpos).round(1)}')
        # Attach PD controller to creature (runs inside apply_motor_output, BEFORE world.step)
        pd_scale = profile.get('pd_scale', 0.5) if profile else 0.5
        pd_fallen_scale = profile.get('pd_fallen_scale', 1.5) if profile else 1.5
        pd_controller = {
            'kp': pd_kp, 'kd': pd_kd, 'lo': ctrl_lo, 'hi': ctrl_hi,
            'standing': standing_qpos, 'scale': pd_scale, 'fallen_scale': pd_fallen_scale,
        }
        print(f'  PD scale: {pd_scale} (standing) / {pd_fallen_scale} (fallen)')

    # train_baby.py also needs to check native_position_control
    if pd_controller and profile and profile.get('native_position_control', False):
        pd_controller['native_position'] = True
        print(f'  Position control: NATIVE (MuJoCo position actuators, no PD torque conversion)')
    elif pd_controller:
        print(f'  Position control: CUSTOM PD (torque actuators)')
    if pd_controller:
        creature._pd_controller = pd_controller

    # --- Dynamic reflex scaling based on creature mass ---
    total_mass = sum(world._model.body_mass)
    mass_factor = total_mass / 20.0
    creature._reflex_scale_standing = 0.15 * mass_factor
    creature._reflex_scale_fallen = 0.9 * mass_factor
    creature.reflex_scale = creature._reflex_scale_standing
    print(f'  Mass: {total_mass:.1f}kg  Reflex scale: standing={creature._reflex_scale_standing:.2f} fallen={creature._reflex_scale_fallen:.2f}')
    print(f'  {args.creature_name}: {creature.snn.config.n_neurons} neurons, {world.n_actuators} actuators')

    print(f'\n  -- Phase 1: Cerebellum + DA Reward --')
    cb = None
    if not args.no_cerebellum:
        cb_cfg = CerebellarConfig(snn_ramp_steps=2000, snn_mix_end=1.0, ltd_rate=0.001, ltp_rate=0.001)
        cb = CerebellarLearning(snn=creature.snn, n_actuators=world.n_actuators, config=cb_cfg, device=device)
        cb.set_populations(
            mf_ids=creature.snn.populations['mossy_fibers'],
            grc_ids=creature.snn.populations['granule_cells'],
            goc_ids=creature.snn.populations['golgi_cells'],
            pkc_ids=creature.snn.populations['purkinje_cells'],
            dcn_ids=creature.snn.populations['dcn'])
        creature.actor_critic = cb
        print(f'  Cerebellum ON: GrC={cb_cfg.n_granule} PkC={cb_cfg.n_purkinje} DCN={cb_cfg.n_dcn}')
        print(f'  DA modulation: reward -> LTP boost, LTD suppression')

    reflexes = SpinalReflexes(n_actuators=world.n_actuators)
    spinal_segments = SpinalSegments(n_actuators=world.n_actuators)

    # --- Terrain-Adaptive Locomotion (Phase A+B) ---
    from src.brain.terrain_reflex import FootContactSensor, TerrainReflex, TerrainReflexConfig
    foot_sensor = FootContactSensor()
    foot_sensor.initialize(world._model)
    terrain_reflex = TerrainReflex(config=TerrainReflexConfig(), n_actuators=world.n_actuators)
    print(f'  TerrainReflex: pitch_gain={terrain_reflex.config.pitch_gain} roll_gain={terrain_reflex.config.roll_gain}')

    # --- Issue #76d: Visual Orienting Response (VOR) ---
    # Biology: Superior Colliculus -> Tectospinal Tract -> asymmetric motor activation.
    # Hardwired reflex: creature turns toward visual target immediately.
    # The SNN vision channels (target_heading, target_distance) provide input for
    # LEARNING (what to do with the target), the VOR provides the immediate motor
    # response (turn toward it). Like vestibular reflexes: hardwired, not learned.
    from src.brain.visual_orienting import VisualOrientingResponse, VORConfig
    vor = VisualOrientingResponse(
        config=VORConfig(
            hip_gain=0.45,       # PD controller is more efficient, less raw gain needed
            abd_gain=0.25,
            smoothing=0.6,       # Slightly more responsive for PD damping to work
            max_output=0.55,     # Moderate: enough to turn, not enough to spin in place
            deadzone=0.05,       # ~5.4 degrees
        ),
        n_actuators=world.n_actuators,
    )
    print(f'  VOR: hip={vor.config.hip_gain} abd={vor.config.abd_gain} smooth={vor.config.smoothing} max={vor.config.max_output}')

    # --- Issue #78: Embodied Closed-Loop Adapter ---
    # THIS IS THE KEY: closes the loop between experience and adaptation.
    # The system evaluates its own progress and adjusts its own parameters.
    # No more manual gain tuning. The creature learns to navigate autonomously.
    from src.brain.embodied_closed_loop import EmbodiedClosedLoop, EmbodiedExperience
    closed_loop = EmbodiedClosedLoop(
        snn=creature.snn,
        vor=vor,
        eval_interval=2000,
    )
    print(f'  Closed-Loop: eval every {closed_loop.eval_interval} steps (autonomous adaptation)')
    creature._spinal_segments = spinal_segments
    creature._sim_dt = args.timestep
    # For external MJCF (Go2, Bittle): use the model's actual timestep,
    # not args.timestep which defaults to 0.005 and may differ.
    if is_external_mjcf and hasattr(world, '_model') and world._model:
        actual_dt = world._model.opt.timestep
        if abs(actual_dt - args.timestep) > 1e-6:
            print(f'  NOTE: Model timestep={actual_dt} differs from --timestep={args.timestep}, using model value')
            args.timestep = actual_dt
            creature._sim_dt = actual_dt
    print(f'  Spinal Segments: tone={spinal_segments.config.tone_gain:.2f}'
          f'  stretch={spinal_segments.config.stretch_gain:.2f}'
          f'  golgi@{spinal_segments.config.golgi_threshold:.2f}')

    # --- Load evolved CPG params or use defaults ---
    # CPG dimensions from profile (Bittle: 12/3 with abd=0; Freenove/Go2: 12/3)
    _cpg_n_act = 12
    _cpg_jpleg = 3
    if profile and 'cpg_config' in profile:
        _cpg_n_act = profile['cpg_config'].get('n_actuators', 12)
        _cpg_jpleg = profile['cpg_config'].get('joints_per_leg', 3)

    _use_opencat_gait = (getattr(args, 'neural_cpg', False)
                         and profile and profile.get('joints_per_leg', 3) == 2)

    if _use_opencat_gait:
        # Bittle: use OpenCat controller (firmware-level abstraction)
        from src.body.opencat_controller import OpenCatController
        spinal_cpg = OpenCatController(steering_mode=args.steering_mode)
        spinal_cpg.set_gait('trot')
        print(f'  CPG: OPENCAT CONTROLLER (walk={spinal_cpg.cycle_time:.3f}s, '
              f'{len(spinal_cpg._gait_frames)} frames, 50 Hz interpolated)')
        print(f'    gaits: {spinal_cpg.available_gaits}')
        print(f'    poses: {spinal_cpg.available_poses}')
    elif getattr(args, 'neural_cpg', False):
        # Mogli Oscillator: SNN-based half-center CPG (Issue #111)
        mogli_cfg = MogliConfig()
        # Apply creature-specific CPG amplitudes from profile
        if profile and 'cpg_config' in profile:
            pcpg = profile['cpg_config']
            if 'hip_amplitude' in pcpg:
                mogli_cfg.hip_amplitude = pcpg['hip_amplitude']
            if 'knee_amplitude' in pcpg:
                mogli_cfg.knee_amplitude = pcpg['knee_amplitude']
            if 'abd_amplitude' in pcpg:
                mogli_cfg.abd_amplitude = pcpg['abd_amplitude']
            if 'maturation_steps' in pcpg:
                mogli_cfg.maturation_steps = pcpg['maturation_steps']
        spinal_cpg = MogliCPG(n_actuators=_cpg_n_act, joints_per_leg=_cpg_jpleg, config=mogli_cfg)
        print(f'  CPG: MOGLI OSCILLATOR v0.3.3 (8 Izhikevich neurons + dead-band vestibular + step-length steering)')
        print(f'    n_actuators={_cpg_n_act}  joints_per_leg={_cpg_jpleg}')
        print(f'    w_mutual={mogli_cfg.w_mutual}  w_contra={mogli_cfg.w_contralateral}'
              f'  drive={mogli_cfg.tonic_drive_base}  amp={mogli_cfg.base_amplitude}→{mogli_cfg.max_amplitude}'
              f'  mat={mogli_cfg.maturation_steps}')
        print(f'    vestibular: gain={mogli_cfg.vestibular_gain}  dead_band={mogli_cfg.vestibular_dead_band}  output_smoothing={mogli_cfg.vestibular_output_smoothing}')
    elif cpg_config_path:
        spinal_cpg = SpinalCPG.from_evolved(cpg_config_path, n_actuators=12, joints_per_leg=3)
        print(f'  CPG: evolved params from {cpg_config_path}')
        print(f'    freq={spinal_cpg.config.frequency:.3f}Hz  hip={spinal_cpg.config.hip_amplitude:.3f}'
              f'  knee={spinal_cpg.config.knee_amplitude:.3f}  abd={spinal_cpg.config.abd_amplitude:.3f}')
    else:
        cpg_cfg = SpinalCPGConfig(frequency=1.0, hip_amplitude=0.60, knee_amplitude=0.50,
                                   abd_amplitude=0.05, base_amplitude=0.50, max_amplitude=0.80,
                                   cpg_weight_start=0.9, cpg_weight_end=0.2, cpg_weight_fade_steps=999999999)
        spinal_cpg = SpinalCPG(n_actuators=12, joints_per_leg=3, config=cpg_cfg)
        print(f'  CPG: default params (no evolved config), freq={cpg_cfg.frequency}Hz')

    # OpenCatGait is self-stable — spinal reflexes destabilize the gait.
    # SNN/cerebellum corrections at reduced scale so the network can learn
    # without overwhelming the stable CPG pattern.
    _disable_reflexes = _use_opencat_gait
    if _disable_reflexes:
        # SNN-Anteil am Motor. Fest verdrahtet 0.3; seit 26.07.2026 ueber
        # --snn-motor-scale einstellbar, Default unveraendert.
        _ms = getattr(args, 'snn_motor_scale', None)
        _ms = 0.3 if _ms is None else float(_ms)
        creature.motor_scale = _ms          # damped SNN — learn without destabilizing
        creature._spinal_segments = None    # disable: OpenCat position control, not torque
        _profile_balance = profile.get('opencat_balance', True) if profile else True
        if getattr(args, 'no_balance', False) or not _profile_balance:
            # A/B control (Task #51, #60) + profile default. Balance OFF mirrors the
            # hardware bridge (balance not run on HW). Measured (knowledge #88):
            # OpenCatBalance costs ~40% distance and is the only fall source in sim.
            # Bittle profile sets opencat_balance:false; default True keeps
            # Mogli/Freenove/Go2 behaviour unchanged.
            creature._opencat_balance = None
            _bal_why = '--no-balance' if getattr(args, 'no_balance', False) else 'profile'
            print(f'  Motor overlays: SNN at {_ms:.0%}, reflexes OFF, spinal segments OFF, '
                  f'OpenCat balance OFF ({_bal_why})')
        else:
            # OpenCat balance controller (IMU → servo corrections)
            from src.body.opencat_balance import OpenCatBalance
            creature._opencat_balance = OpenCatBalance(
                n_actuators=world.n_actuators,
                joint_mapping=[
                    4,   # servo 0: RF shoulder -> OC joint 4
                    8,   # servo 1: RF knee     -> OC joint 8
                    5,   # servo 2: LF shoulder -> OC joint 5
                    9,   # servo 3: LF knee     -> OC joint 9
                    6,   # servo 4: RB shoulder -> OC joint 6
                    10,  # servo 5: RB knee     -> OC joint 10
                    7,   # servo 6: LB shoulder -> OC joint 7
                    11,  # servo 7: LB knee     -> OC joint 11
                ],
                gain_scale=0.6,    # reduce gains for MuJoCo (real HW coefficients too aggressive)
            )
            print(f'  Motor overlays: SNN at {_ms:.0%}, reflexes OFF, spinal segments OFF, '
                  'OpenCat balance ON')

    print(f'\n  -- Phase 2: Competence-Gated Handoff --')
    gate = CompetenceGate(grow_rate=0.0005, shrink_rate=0.0002,
                          cpg_min=0.40, cpg_max=0.9, upright_threshold=0.6)
    print(f'  Gate: CPG {gate.cpg_max:.0%} -> {gate.cpg_min:.0%} | pure IMU'
          f'  upright>{gate.upright_threshold}  grow={gate.grow_rate}  shrink={gate.shrink_rate}')
    if getattr(args, 'no_competence_gate', False):
        # A/B control (#60): pin the gate so the CPG never fades -> the OpenCat
        # gait runs at full amplitude + additive SNN/cerebellum, byte-for-form
        # identical to bridge_bittle_wifi.py. Competence is still tracked (logged)
        # but no longer attenuates the gait.
        gate.cpg_min = 1.0
        gate.cpg_max = 1.0
        gate.cpg_weight = 1.0
        print('  Gate: DISABLED via --no-competence-gate '
              '(cpg_weight pinned 1.0 -> full CPG, bridge-equivalent gait application)')

    # --- Leg damage (Issue #133: leg-loss survival) ---
    _damaged_actuators = []
    _leg_damage_applied = False
    _leg_damage_at_step = args.leg_damage_at  # 0 = from start
    _pin_applied = False  # --pin-base-at diagnostic (true wedged stuck)
    if args.leg_damage and _leg_damage_at_step == 0:
        # Immediate damage from start
        n_act = world.n_actuators
        leg_map = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}
        leg_id = leg_map.get(args.leg_damage.upper())
        if leg_id is not None:
            jpleg = n_act // 4
            _damaged_actuators = list(range(leg_id * jpleg, (leg_id + 1) * jpleg))
            # Make leg physically limp: zero actuator gains + minimal joint damping
            # Biology: a paralyzed leg has no muscle tone — it hangs limp
            # and provides no support. MuJoCo position actuators with kp=0
            # exert zero torque; near-zero damping lets the joint swing freely.
            leg_prefix = args.leg_damage.lower()  # 'fl', 'fr', 'rl', 'rr'
            joint_names = [f'{leg_prefix}_hip_yaw', f'{leg_prefix}_hip_pitch', f'{leg_prefix}_knee']
            for jname in joint_names:
                jid = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    world._model.jnt_stiffness[jid] = 0.0
                    world._model.dof_damping[world._model.jnt_dofadr[jid]] = 0.01  # near-zero, not zero (stability)
            for aid in _damaged_actuators:
                if aid < world._model.nu:
                    world._model.actuator_gainprm[aid, 0] = 0.0  # kp = 0 → no servo torque
                    world._model.actuator_biasprm[aid, 1] = 0.0  # kp bias = 0
            print(f'\n  -- Leg Damage: {args.leg_damage.upper()} LIMP (actuator kp=0, joint damping=0.01) --')
            print(f'     Actuators {_damaged_actuators} zeroed, joints {joint_names} freed')
            _leg_damage_applied = True
    elif args.leg_damage and _leg_damage_at_step > 0:
        print(f'\n  -- Leg Damage: {args.leg_damage.upper()} scheduled at step {_leg_damage_at_step} --')

    # --- Issue #57: Autonomous Drive Loop ---
    drive_bridge = None
    if not args.no_drives:
        try:
            from src.behavior.drive_motor_bridge import DriveMotorBridge
            drive_bridge = DriveMotorBridge(
                creature_type='dog',
                scene_instruction=scene_inst,
                drive_limits=profile.get('drive_limits') if profile else None,
            )
            print(f'\n  -- Drive Loop: ACTIVE --')
            print(f'  Behaviors: {", ".join(drive_bridge.knowledge.get_all_names())}')
            print(f'  Drive → BehaviorPlanner → MotorPattern → CPG freq/amp modulation')
            if profile and 'drive_limits' in profile:
                dl = profile['drive_limits']
                print(f'  Drive limits: freq={dl.get("freq_min", "none")}-{dl.get("freq_max", "none")} amp={dl.get("amp_min", "none")}-{dl.get("amp_max", "none")}')
        except Exception as e:
            print(f'\n  Drive Loop: init failed ({e}), running without drives')
            drive_bridge = None
    else:
        print(f'\n  -- Drive Loop: DISABLED (--no-drives) --')

    # --- Issue #75: Sensory Environment ---
    sensory_env = None
    visual_env = None  # Phototaxis mode
    # Pre-init light body IDs so they always exist (used in FLOG block).
    # The 'if visual_env:' branch overwrites these when phototaxis is active.
    _lt_body_id = -1
    _lt_jnt_id = -1
    _lt_qposadr = -1
    _lt_dofadr = -1
    if drive_bridge and not args.no_sensory:
        try:
            if _scene_has_light:
                # Phototaxis mode: VisualEnvironment with camera
                from src.body.visual_environment import VisualEnvironment
                visual_env = VisualEnvironment(
                    world_size=10.0, seed=args.seed,
                    cam_width=64, cam_height=48,
                )
                # Skip camera renderer in sim — use geometric fallback
                # Camera is only needed on real hardware (cv2)
                # visual_env.init_renderer(world._model, world._data)
                visual_env.spawn_lights(count=1, min_dist=1.5, max_dist=3.0)
                # Set light body position via qpos
                light_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'light_target')
                if light_id >= 0:
                    light_jnt_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'light_joint')
                    light_qposadr = world._model.jnt_qposadr[light_jnt_id]
                    light_pos = visual_env.get_light_positions()[0]
                    world._data.qpos[light_qposadr:light_qposadr + 3] = light_pos
                    world._data.qpos[light_qposadr + 3:light_qposadr + 7] = [1, 0, 0, 0]
                    light_dofadr = world._model.jnt_dofadr[light_jnt_id]
                    world._data.qvel[light_dofadr:light_dofadr + 6] = 0.0
                    mujoco.mj_forward(world._model, world._data)
                print(f'  Sensory: PHOTOTAXIS MODE — 1 light source, camera {64}x{48}')
                # Cache MuJoCo IDs for light body (used every step)
                _lt_body_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'light_target')
                _lt_jnt_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'light_joint')
                if _lt_body_id >= 0 and _lt_jnt_id >= 0:
                    _lt_qposadr = world._model.jnt_qposadr[_lt_jnt_id]
                    _lt_dofadr = world._model.jnt_dofadr[_lt_jnt_id]
                else:
                    _lt_qposadr = -1
                    _lt_dofadr = -1
            else:
                from src.body.sensory_environment import SensoryEnvironment, ScentSource
                sensory_env = SensoryEnvironment(
                    world_size=10.0, seed=args.seed,
                    sound_interval=2000, sound_duration=500,
                )
                # Check if ball exists in scene
                ball_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
                if ball_id >= 0:
                    # Ball scene: set ball qpos explicitly (freejoint ignores XML body pos=)
                    ball_jnt_id = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
                    ball_qposadr = world._model.jnt_qposadr[ball_jnt_id]
                    ball_target = np.array(main._ball_positions[main._ball_stage])
                    world._data.qpos[ball_qposadr:ball_qposadr + 3] = ball_target
                    world._data.qpos[ball_qposadr + 3:ball_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
                    ball_dofadr = world._model.jnt_dofadr[ball_jnt_id]
                    world._data.qvel[ball_dofadr:ball_dofadr + 6] = 0.0
                    mujoco.mj_forward(world._model, world._data)
                    ball_xpos = world._data.xpos[ball_id].copy()
                    print(f'  Ball qpos set: target=({ball_target[0]:.1f}, {ball_target[1]:.1f}, {ball_target[2]:.2f})'
                          f'  actual xpos=({ball_xpos[0]:.1f}, {ball_xpos[1]:.1f}, {ball_xpos[2]:.2f})')
                    # Ball as single scent source (visual salience proxy)
                    sensory_env._scents = [ScentSource(
                        position=ball_xpos.copy(), strength=3.0, radius=1.5,
                        name='ball_scent', fixed=True
                    )]
                    creature._steer_gain = 0.15
                    print(f'  Sensory: BALL MODE -- target at ({ball_xpos[0]:.1f}, {ball_xpos[1]:.1f})')
                else:
                    sensory_env.spawn_scent(count=2, min_dist=1.0, max_dist=2.0)
                    print(f'  Sensory: 2 scent sources (1-2m), sounds every ~2k steps')
        except Exception as e:
            print(f'  Sensory env: init failed ({e})')
            sensory_env = None

    # === v4.2: HardwareDrift — inject measured mechanical drift ===
    from src.body.hardware_drift import HardwareDrift
    if args.drift_profile:
        hardware_drift = HardwareDrift.from_profile(args.drift_profile)
    else:
        hardware_drift = HardwareDrift()  # No-op

    # === v4.2: LightMemory — return to last known yaw when light lost ===
    light_memory = None
    if _scene_has_light:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from freenove_bridge import LightMemory
        light_memory = LightMemory(return_gain=0.4, timeout_seconds=10.0,
                                   z_sign=-1.0)  # Simulator: +Z → turn left
        print(f'  LightMemory: enabled (gain=0.4, timeout=10s)')

    start_step = 0
    _resume_spatial_map = None   # spatial_map is built later (~L1524); stash + apply after creation
    if args.resume and os.path.exists(args.resume):
        print(f'\n  Resuming from {args.resume}...')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # SNN file: try stored path first, then relative to checkpoint dir
        snn_file = ckpt.get('snn_file')
        if snn_file and not os.path.exists(snn_file):
            # Path was absolute on different machine — try sibling
            snn_file = os.path.join(os.path.dirname(args.resume), 'snn_state.pt')
        if snn_file and os.path.exists(snn_file):
            creature.snn.load(snn_file)
            print(f'  SNN loaded: {snn_file}')
        else:
            print(f'  ⚠ SNN state not found, starting fresh weights')
        if cb and 'cerebellum_state' in ckpt:
            cb.load_state_dict(ckpt['cerebellum_state'])
        # v4.2: Restore spatial map — DEFERRED: the spatial_map object is built
        # later (~L1524). Stash the state now, load it right after creation.
        if 'spatial_map' in ckpt:
            _resume_spatial_map = ckpt['spatial_map']
        start_step = ckpt.get('step', 0)
        gate.actor_competence = ckpt.get('actor_competence', 0.0)
        gate.cpg_weight = ckpt.get('cpg_weight', gate.cpg_max)
        gate.vel_ema = ckpt.get('vel_ema', 0.0)
        cpg_phases = ckpt.get('cpg_phases', None)
        cpg_step = ckpt.get('cpg_step', 0)
        if cpg_phases is not None and hasattr(spinal_cpg, '_phases'):
            spinal_cpg._phases = np.array(cpg_phases)
        spinal_cpg._step = cpg_step
        max_dist = ckpt.get('max_dist', 0.0)
        fall_count = ckpt.get('falls', 0)
        recovery_count = ckpt.get('recoveries', 0)
        best_upright_streak = ckpt.get('best_upright_streak', 0)
        last_pci = ckpt.get('pci', 0.0)
        total_steps = start_step + args.steps
        print(f'  Resumed at step {start_step:,} (cpg_step={cpg_step}, '
              f'competence={gate.actor_competence:.3f}, cpg={gate.cpg_weight:.0%})')

    # ================================================================
    # BRAIN PERSISTENCE: Auto-load brain if it exists (Issue #85)
    # ================================================================
    # This runs ALWAYS — not just on --resume. The brain is the creature's
    # long-term memory. A new run without --resume still loads existing
    # knowledge (episodic memory, concept graph, world model, skills).
    # Only SNN weights are reset on a fresh run (no --resume).
    # With --resume: SNN weights + brain both loaded.
    # Without --resume: fresh SNN weights + existing brain knowledge.
    # This is biologically correct: a puppy born with fresh synapses
    # but inheriting the body schema and instincts of its species.
    creature_base = os.path.join('creatures', args.creature_name.lower())
    brain_file_auto = os.path.join(creature_base, 'brain', 'brain.pt')
    if args.fresh:
        print(f'\n  --fresh: Skipping brain.pt — completely fresh start')
    elif os.path.exists(brain_file_auto) and hasattr(creature, 'brain') and creature.brain:
        from src.brain.brain_persistence import load_brain, brain_info
        bi = brain_info(brain_file_auto)
        print(f'\n  Brain found: {bi.get("n_episodes", 0)} episodes, '
              f'{bi.get("n_concepts", 0)} concepts, '
              f'{bi.get("snn_steps", 0)} SNN steps')
        # TOPOLOGY CHECK: if saved SNN size != current SNN size,
        # strip the SNN state from brain.pt before loading.
        # Cognitive components (memory, concepts, world model) still load.
        _brain_state = torch.load(brain_file_auto, map_location='cpu', weights_only=False)
        _snn_mismatch = False
        if 'snn' in _brain_state:
            _saved_v = _brain_state['snn'].get('V', None)
            if _saved_v is not None and _saved_v.shape[0] != creature.snn.config.n_neurons:
                _snn_mismatch = True
                print(f'  ⚠ Brain topology mismatch: saved SNN={_saved_v.shape[0]} neurons, '
                      f'current SNN={creature.snn.config.n_neurons}. Stripping SNN state.')
                del _brain_state['snn']
        # Save stripped state to temp file, load via standard load_brain()
        _tmp_brain = brain_file_auto + '.tmp'
        torch.save(_brain_state, _tmp_brain)
        del _brain_state
        load_brain(creature.brain, creature.snn, _tmp_brain)
        os.remove(_tmp_brain)
        if _snn_mismatch:
            print(f'  Brain loaded (cognitive only, SNN fresh): {brain_file_auto}')
        else:
            print(f'  Brain loaded: {brain_file_auto}')
        print(f'  → Episodic memory, concept graph, world model, skills restored')
    else:
        print(f'\n  No brain.pt found — starting with fresh cognitive state')

    recorder = None
    flog_path = None
    run_id = f'v043_{int(time.time())}'
    creature_dir = f'creatures/{args.creature_name.lower()}/{run_id}'
    if not args.no_flog:
        try:
            from src.brain.creature_store import TrainingRecorder
            os.makedirs(creature_dir, exist_ok=True)
            flog_path = os.path.join(creature_dir, 'training_log.bin')
            flog_meta = {
                'creature': args.creature_name.lower(),
                'task': args.scene,
                'scene': terrain_cfg.terrain_type,
                'difficulty': terrain_cfg.difficulty,
                'steps': total_steps,
                'record_interval': args.record_interval,   # pose-logging cadence; render_bittle reads this for sim_dt timing
                'device': device,
                'version': 'v0.8.1',
                'n_neurons': creature.snn.config.n_neurons,
                'population_sizes': {
                    'n_input': creature.n_input_neurons,
                    'n_output': creature.n_output_neurons,
                    'n_granule': len(creature.snn.populations.get('granule_cells', [])),
                    'n_golgi': len(creature.snn.populations.get('golgi_cells', [])),
                    'n_purkinje': len(creature.snn.populations.get('purkinje_cells', [])),
                    'n_dcn': len(creature.snn.populations.get('dcn', [])),
                    'n_motor_hidden': len(creature.snn.populations.get('motor_hidden', [])),
                    'n_total': creature.snn.config.n_neurons,
                },
                # Observation-only: makes every FLOG self-describing so a
                # variance study can pair runs by seed + the gated knobs.
                # Metadata only — does NOT touch physics/learning (bit-identical).
                #
                # The behaviour-defining flags below were MISSING until task #95, and it
                # cost a full session: run B (v043_1782653493) turned out to have used
                # seed 4 rather than the default 42, and a different wall_memory_weight
                # than the log entries claimed.  Both had to be back-computed from the
                # measurements, and every A/B comparison made against it in the meantime
                # was void.  A run that cannot say how it was produced is not evidence.
                # If a flag changes behaviour, it belongs here.
                'config': {
                    'seed': args.seed,
                    'wall_distance': args.wall_distance,
                    'no_wall_reset': bool(args.no_wall_reset),
                    'no_balance': bool(args.no_balance),
                    'no_drives': bool(args.no_drives),
                    'coord_reward_weight': args.coord_reward_weight,
                    'reward_blend': args.reward_blend,
                    'snn_substeps': args.snn_substeps,
                    # --- task #84 wall avoidance -------------------------------------
                    'imu_obstacle': bool(getattr(args, 'imu_obstacle', False)),
                    'block_aversion_weight': float(getattr(args, 'block_aversion_weight', 0.0)),
                    'wall_memory_weight': float(getattr(args, 'wall_memory_weight', 0.0)),
                    'wall_memory_radius': float(getattr(args, 'wall_memory_radius', 0.0)),
                    'danger_steer_weight': float(getattr(args, 'danger_steer_weight', 0.0)),
                    'danger_steer_radius': float(getattr(args, 'danger_steer_radius', 0.0)),
                    # --- task #92 steering -------------------------------------------
                    'steering_mode': str(getattr(args, 'steering_mode', 'offset')),
                    # --- learning / curiosity ----------------------------------------
                    'curiosity_learning_progress': bool(getattr(args, 'curiosity_learning_progress', False)),
                    'curiosity_lp_scale': float(getattr(args, 'curiosity_lp_scale', 1.0)),
                    'neural_cpg': bool(getattr(args, 'neural_cpg', False)),
                    'hardware_sensors': bool(getattr(args, 'hardware_sensors', False)),
                    'fresh': bool(getattr(args, 'fresh', False)),
                    'steps': int(getattr(args, 'steps', 0)),
                    'scene': str(getattr(args, 'scene', '')),
                    'creature_name': str(getattr(args, 'creature_name', '')),
                },
            }
            recorder = TrainingRecorder(flog_path, meta=flog_meta)
            print(f'  FLOG: {flog_path}')
        except Exception as e:
            print(f'  FLOG init failed: {e}')

    try:
        knowledge_log = {
            'scene': args.scene, 'source': knowledge['source'],
            'behaviors': [{'name': b.name, 'priority': b.priority, 'drive': b.drive, 'description': b.description} for b in knowledge['behaviors']],
            'terrain': {
                'type': terrain_cfg.terrain_type,
                'difficulty': terrain_cfg.difficulty,
                'max_height': terrain_cfg.max_height,
                'size_x': terrain_cfg.size_x,
                'size_y': terrain_cfg.size_y,
                'resolution': terrain_cfg.resolution,
                'seed': terrain_cfg.seed,
            },
            'drives': scene_inst.drive_biases if scene_inst else {},
        }
        # Save knowledge log into the run directory (alongside FLOG)
        knowledge_dir = creature_dir if creature_dir else f'creatures/{args.creature_name.lower()}'
        os.makedirs(knowledge_dir, exist_ok=True)
        with open(os.path.join(knowledge_dir, 'knowledge.json'), 'w') as f:
            json.dump(knowledge_log, f, indent=2)
    except Exception:
        pass

    print(f'\n{"="*65}')
    print(f'  {args.creature_name} begins learning: "{args.scene}"')
    print(f'{"="*65}\n')

    # Training Loop
    t_start = time.perf_counter()

    # Step-time profiling (Issue: step-time explosion)
    _profile = {'sensor': 0.0, 'sensory_env': 0.0, 'creature_step': 0.0,
                'brain': 0.0, 'flog': 0.0, 'mujoco': 0.0, 'other': 0.0}
    _profile_window = 1000
    max_dist = 0.0
    fall_count = 0
    recovery_count = 0
    best_upright_streak = 0
    current_upright_streak = 0
    step_times = deque(maxlen=2000)  # Bounded — prevents O(N) growth over long runs
    last_pci = 0.0
    brain_result = {}

    # --- Stuck detection (Increment A2, observation only — Decision #203) ---------
    # Sim ground-truth label vs. hardware-available accelerometer proxy. Updated
    # every step (below); only the current value is written to FLOG at log_every.
    # Constants are starting points, to be tuned against the proxy<->label
    # correlation study. NOT used for learning or control (runs stay bit-identical).
    STUCK_WINDOW = 30          # ~1.9 s at ~16 sps (hardware-sensors run)
    STUCK_DISP_EPS = 0.002     # m net displ. over window; ~1/2 the observed steady-walk floor (~0.004, Run-D #95)
    STUCK_SPEED_EPS = 0.02     # m/s instantaneous (logged for comparison only)
    _acc_grav_ema = None        # lazy-init from first valid reading (avoids 9.81-vs-0 startup spike)
    _acc_dyn_ema = 0.0          # high-pass dynamic-acceleration energy (hardware-able proxy)
    _horiz_speed = 0.0
    _progress = float('nan')
    _pos_window = deque(maxlen=STUCK_WINDOW)   # base (x,y) history for net-displacement label
    _stuck_truth = False

    # Drive loop state (for logging)
    current_behavior = 'walk'
    current_freq_scale = 1.0
    current_amp_scale = 1.0

    # Auto-reset state
    auto_reset_limit = args.auto_reset
    consecutive_fallen = 0
    reset_count = 0

    # Issue #110: Velocity-based stuck detection for Go2
    # The Go2 can lie on its side with upright ~0.35, which doesn't
    # trigger is_fallen() (threshold 0.3). It then lies motionless
    # for thousands of steps wasting training time.
    # Fix: if velocity ≈ 0 AND upright < 0.7 for 200+ steps → stuck.
    _stuck_counter = 0
    _STUCK_THRESHOLD_STEPS = 200  # ~6 seconds at 30ms/step
    _STUCK_VELOCITY_MAX = 0.005   # m/s — basically motionless
    _STUCK_UPRIGHT_MAX = 0.7      # below this = not standing properly

    # Issue #125: Progress-based stuck detection
    # The upright-based detector misses cases where the robot stands
    # upright (0.83) but walks in place or rolls without forward progress.
    # This detector resets if the robot makes zero forward progress
    # for an extended period, regardless of upright state.
    # Biology: a dog that isn't going anywhere needs help.
    _progress_stuck_counter = 0
    _PROGRESS_STUCK_STEPS = 500   # ~15 seconds — longer than upright detector
    _progress_last_max_dist = 0.0

    # --- Issue #103: Wall episode state ---
    wall_episode_count = 0
    _wall_last_obs_dist = 4.0  # hysteresis: remember last known distance
    _wall_obs_cooldown = 0     # steps since last wall detection
    _wall_pause_counter = 0    # steps to stand still after wall contact before reset
    _WALL_PAUSE_STEPS = 50     # ~1.5s at 33 sps — visible "perplex" moment

    # --- Run-and-Tumble chemotaxis state machine (v0.4.8) ---
    # Biology: chemotaxis is NOT continuous steering. It is discrete:
    #   SNIFF → TUMBLE (orient) → RUN (straight) → SNIFF again.
    # Ref: Berg & Brown 1972 (E. coli), Catania 2013 (star-nosed mole).
    # Continuous steering causes circling (v0.4.7 failure mode).
    _RT_STATE = 'RUN'          # Current state: 'RUN', 'SNIFF', 'TUMBLE'
    _RT_TIMER = 0              # Steps remaining in current state
    _RT_RUN_DURATION = 40      # Steps per RUN phase (~1.3s at 33sps, ~1.5 gait cycles)
    _RT_RUN_DURATION_BASE = 40 # Base value (adapts based on improvement)
    _RT_RUN_DURATION_MAX = 120 # Max extended RUN (was 200 — shorter to stay near scents)
    _RT_TUMBLE_DURATION = 12   # Steps per TUMBLE phase (~0.4s — quick head turn)
    _RT_TUMBLE_IMPULSE = 0.0   # Steering impulse computed during SNIFF
    _RT_SM_BEFORE = 0.0        # Smell strength at start of RUN (for improvement check)
    _RT_DEAD_ZONE = 0.15       # rad (~8.6°) — don't tumble if already aimed
    _RT_MIN_SM = 0.10          # Minimum smell strength to trigger steering

    # --- Issue #76d: Ball approach reward state ---
    # Biology: Dopamine burst on approach to salient stimulus (Schultz 1997).
    # The SNN learns to navigate toward ball via DA signal, not motor hacks.
    prev_ball_dist = None  # initialized on first step when ball exists
    ball_approach_reward = 0.0
    # Track best ball distance DURING episodes (not at reset time)
    # At reset, prev_ball_dist is >8m so we'd miss the actual minimum.
    main._ball_best_dist_running = 99.0  # Updated every step

    # --- Developmental Schedule (Issue #68b) ---
    # Biology: neonatal sensorimotor development.
    # Perturbation forces + forward model sensor augmentation.
    dev_schedule = DevelopmentalSchedule(
        total_steps=total_steps,
        config=DevelopmentalConfig(
            perturb_enabled=False,          # Disabled: perturbation hurts flat, not needed for hilly
            perturb_force_max=0.3,
            perturb_interval=100,
            perturb_duration=5,
            forward_model_warmup_steps=10000,
        )
    )
    print(f'  Developmental Schedule: perturbation 0.3N (competence-gated), FM warmup 10k steps')

    # --- Gait Quality Metrics (v0.7.0 Pillar 3) ---
    gait_cfg = GaitQualityConfig(standing_height=standing_h)
    gait_analyzer = GaitQualityAnalyzer(config=gait_cfg)
    print(f'  Gait Quality: {GAIT_QUALITY_VERSION} (analysis every {gait_cfg.analysis_interval} steps, buffer {gait_cfg.joint_buffer_size})')

    # --- Body Awareness (v0.7.0 Pillar 1) ---
    # Dead-leg threshold: tuned for 12-DOF quadrupeds (Freenove/Go2).
    # Smaller creatures (Bittle 8-DOF) have lower responsiveness due to
    # smaller joint amplitudes — use profile-aware threshold.
    _dead_thresh = 0.20  # default for 12-DOF
    _ba_enabled = True
    if profile and profile.get('joints_per_leg', 3) == 2:
        _ba_enabled = False  # Bittle: amplitudes too small for reliable limb detection
    body_awareness = BodyAwareness(
        joints_per_leg=world.n_actuators // 4,
        n_legs=4,
        detection_delay=500,
        dead_threshold=_dead_thresh,
        degraded_threshold=0.35,
    )
    body_awareness.enabled = _ba_enabled
    print(f'  Body Awareness: {BODY_AWARENESS_VERSION} (limb detection, auto-disconnect)'
          f'{" — DISABLED (small creature)" if not _ba_enabled else ""}')

    # --- Spatial Map (v0.7.0 Pillar 2) ---
    spatial_map = SpatialMap(world_size=10.0, grid_resolution=20)
    print(f'  Spatial Map: {SPATIAL_MAP_VERSION} (path integration, {spatial_map.grid_resolution}x{spatial_map.grid_resolution} grid)')
    if _resume_spatial_map is not None:
        spatial_map.load_state_dict(_resume_spatial_map)
        print(f'  Spatial map restored from checkpoint')

    # --- Directed Learning (v0.7.0 Pillar 5) ---
    directed_learning = DirectedLearning(
        eval_interval=2000,    # Evaluate every 2k steps (~60s at 33sps)
        test_duration=1000,    # Test each hypothesis for 1k steps (~30s)
    )
    print(f'  Directed Learning: {DIRECTED_LEARNING_VERSION} (eval every {directed_learning.eval_interval} steps, test {directed_learning.test_duration} steps)')

    # --- Episode Analyzer (Meta-Learning Loop Phase A) ---
    episode_analyzer = EpisodeAnalyzer(
        min_events_for_analysis=4,
        max_events=200,
        confidence_threshold=0.5,
    )
    print(f'  EpisodeAnalyzer: v0.1.0 (Phase A meta-learning loop)')

    # --- Strategy Adapter (Meta-Learning Loop Phase B) ---
    strategy_adapter = StrategyAdapter(
        confidence_threshold=0.5,
        max_adjustment_pct=0.3,
    )
    # Initialize with current RT parameters
    strategy_adapter.params['rt_run_duration'] = float(_RT_RUN_DURATION)
    strategy_adapter.params['rt_tumble_duration'] = float(_RT_TUMBLE_DURATION)
    print(f'  StrategyAdapter: v0.1.0 (Phase B meta-learning loop)')

    # --- Curiosity Explorer (Meta-Learning Loop Phase C) ---
    curiosity_explorer = CuriosityExplorer()
    print(f'  CuriosityExplorer: v0.1.0 (Phase C meta-learning loop)')

    # --- Hypothesis Generator (Meta-Learning Loop Phase D) ---
    hypothesis_generator = HypothesisGenerator(min_insight_confidence=0.5)
    print(f'  HypothesisGenerator: v0.1.0 (Phase D meta-learning loop)')

    # Increment b (#208): buffered IMU coordination reward (flag-gated, default OFF).
    # Rolling ~2000-step buffer of acc_x + pitch_rate; every _COORD_K steps the gait-band
    # spectral concentration (peak/sum, periods 8..120, detrended) is recomputed and held.
    # Validated walk-vs-babble separation 1.69x @ W=2000 (substrate #207/#208). When the
    # weight is 0.0 the whole computation is gated out -> bit-identical to baseline.
    _coord_w = float(args.coord_reward_weight)
    _COORD_W = 2000          # buffer length (validated sweet spot)
    _COORD_K = 250           # recompute interval (value held between)
    _coord_buf_ax = deque(maxlen=_COORD_W)
    _coord_buf_pr = deque(maxlen=_COORD_W)
    _coord_concentration = 0.0
    # Block aversion (#213/#206): rolling yaw-scrub buffer. Intrinsic, flag-gated.
    _block_w = float(args.block_aversion_weight)
    _block_buf = deque(maxlen=args.block_aversion_window)
    _steer_buf = deque(maxlen=args.block_aversion_window)   # commanded steering, fed back into the variance (efference copy)
    _block_aversion = 0.0
    _wall_mem_w = float(args.wall_memory_weight)   # Task #84 step 2: anticipatory boundary-aversion weight (0.0=off=bit-identical)
    _wall_mem_r = float(args.wall_memory_radius)   # Task #84 step 2: anticipation radius (m)
    # Task #84 step 4: the away-from-danger DRIVE (the actor steps 1-3 never had).
    _danger_w = float(args.danger_steer_weight)    # 0.0 = off = bit-identical
    _danger_r = float(args.danger_steer_radius)
    _danger_steer = 0.0                            # steering contribution applied this step
    _danger_side = 0                               # committed evasion side: +1 = turn right, -1 = turn left, 0 = not committed
    _DANGER_HEADON_EPS = 0.15                      # |sin(bearing)| below this counts as head-on
    _danger_steps = 0                              # diagnostic: steps the drive was active
    if _danger_w > 0.0:
        print(f'  Danger steer drive: ON  weight={_danger_w:.3f}  radius={_danger_r:.2f}m  '
              f'(continuous away-from-danger actor, pre-efference)')
    _wall_anchored = False   # Task #84 step 1: latch -- danger landmark stamped ONCE at first wall contact (no per-step re-averaging -> no drift)
    _WALL_ANCHOR_MIN_X = 0.15   # Task #84 step 1 fix: a progress-stall only counts as WALL contact once the robot has walked INTO the wall zone (true fwd x; wall face 0.25 / torso stop ~0.174). Without it the stall fires during the early maturation/arc phase and anchors the danger landmark near home (observed: conf 0.37 => written ~step 57). Sim-side scaffold; HW-honest IMU-only blocked-detector is the deferred pre-step-2 audit.
    # Obstacle Run-and-Tumble (#108 RT, Marc): discrete RUN/SNIFF/TUMBLE driven by
    # the IMU block signal, mirroring the chemotaxis state machine. Active only under
    # --imu-obstacle. RUN = straight, full gait (kwkF); TUMBLE = committed turn at
    # full gait (kwkL); SNIFF = evaluate block_aversion, tumble again if still blocked.
    # Replaces the continuous throttle+bias reflex that trapped the robot in slow-turn
    # limbo. The primitives (forward, turn) already exist — this just sequences them.
    _OB_STATE = 'RUN'
    _OB_TIMER = int(args.imu_ob_run)
    _OB_STEER = 0.0                                   # steering applied this step (0 in RUN, committed in TUMBLE)
    _OB_RUN = int(args.imu_ob_run)
    _OB_TUMBLE = int(args.imu_ob_tumble)
    _OB_TURN_GAIN = float(args.imu_ob_turn_gain)
    _OB_BLOCK_ON = float(args.imu_ob_block_on)
    _ob_tumbles = 0
    _yaw_scrub_val = 0.0
    if _block_w > 0.0:
        print(f'  Block aversion: ON  weight={_block_w:.3f}  window={args.block_aversion_window}  '
              f'(intrinsic; the drive machinery must LEARN to avoid)')
    if _coord_w > 0.0:
        print(f'  Coordination reward: ON  weight={_coord_w:.3f}  buffer W={_COORD_W} K={_COORD_K}')

    # ---- Live dashboard (Task #61/#70) ---------------------------------------
    # Push the REAL flog_data stats dict to the ws://localhost:5001 broadcaster
    # (dashboard_views #61); viewer = src/viz/sim_live.html. ECHTE WERTE ODER
    # NICHTS (#196): forward exactly the FLOG dict, never a derived value.
    # Default OFF => no call => bit-identical. ON => additive observability only
    # (no SNN/training touch, like the #152 logging fix). Live cadence follows
    # --log-every (the push sits in the flog gate); the wall test runs at 1.
    _dash_on = bool(getattr(args, 'dashboard', False))
    _dash_push = None
    if _dash_on:
        try:
            from src.viz.dashboard_views import start_websocket, update_training_state
            start_websocket()
            _dash_meta = {'source': 'sim', 'creature': args.creature_name,
                          'scene': args.scene, 'wall_distance': float(args.wall_distance),
                          'total_steps': total_steps, 'seed': args.seed}

            def _dash_push(_fd, _m=_dash_meta, _u=update_training_state):
                _u({**_fd, **_m})

            print('  \U0001f4e1 Live dashboard: ON  ws://localhost:5001  '
                  '(open src/viz/sim_live.html in a browser)')
            print(f'     live cadence = --log-every ({log_every}); use 1 for a smooth view')
            if recorder is None:
                print('     \u26a0 FLOG is OFF (--no-flog) -> no live data will be pushed')
        except Exception as _dash_e:
            _dash_on = False
            print(f'  \u26a0 Live dashboard disabled ({_dash_e}); run continues normally')

    for step in range(start_step, total_steps):
        t_step = time.perf_counter()
        _tp0 = t_step  # profile marker
        sensor_data = {}
        try:
            sensor_data = world.get_sensor_data(creature.body_name)
        except Exception:
            pass

        cur_x = float(world._data.qpos[0])
        prev_x = getattr(creature, '_prev_x', cur_x)
        forward_vel = cur_x - prev_x
        creature._prev_x = cur_x
        upright = sensor_data.get('upright', 1.0)
        height = sensor_data.get('height', 0.3)
        prev_upright = getattr(creature, '_prev_upright', upright)
        creature._prev_upright = upright
        is_fallen = creature.is_fallen()
        vel_mps = max(0.0, forward_vel / args.timestep)

        # OpenCat recovery: physical firmware-style recovery (fold to 'dropped',
        # then push 'up' to stand), mirroring the hardware bridge's kbalance
        # recovery (#29) instead of a teleport reset. If 'up' does not right the
        # body within a cycle, re-fold and retry (like the bridge's max_tries)
        # so the robot keeps trying instead of freezing. The teleport auto-reset
        # below is only a FAR fallback for this path (#25/#58).
        _OC_DROP_HOLD = 80    # steps holding 'dropped' before pushing 'up'
        _OC_CYCLE = 200       # full retry period: re-fold if still down after this
        if _use_opencat_gait and is_fallen:
            _oc = getattr(creature, '_oc_recovery_step', 0)
            if _oc == 0:
                # Just fell — fold legs into the 'dropped' pose
                spinal_cpg.set_pose('dropped', speed=3.0)
                creature._oc_recovery_step = 1
                if step < 5000 or fall_count < 5:
                    print(f'  [OPENCAT RECOVERY at step {step}]')
            else:
                _ph = _oc % _OC_CYCLE
                if _ph == _OC_DROP_HOLD:
                    spinal_cpg.set_pose('up', speed=2.0)        # push up to stand
                elif _ph == 0:
                    spinal_cpg.set_pose('dropped', speed=3.0)   # up failed: retry
                creature._oc_recovery_step = _oc + 1
        elif _use_opencat_gait and not is_fallen:
            if getattr(creature, '_oc_recovery_step', 0) != 0:
                # Was recovering, now upright — resume walking
                spinal_cpg.set_gait('trot')
                creature._oc_recovery_step = 0
                recovery_count += 1
                if step < 5000 or recovery_count < 5:
                    print(f'  [OPENCAT RECOVERED at step {step}]')

        # v0.7.0: Heading-aligned velocity (reward forward motion in ANY direction)
        # Biology: a dog wants to move in the direction it's facing.
        # Using X-axis velocity penalizes lateral or rotated movement
        # even when the CPG drives forward relative to the body.
        _cur_y = float(world._data.qpos[1])
        _prev_y = getattr(creature, '_prev_y', _cur_y)
        creature._prev_y = _cur_y
        _dx = cur_x - prev_x
        _dy = _cur_y - _prev_y
        _displacement = float(np.sqrt(_dx**2 + _dy**2))
        # Heading from quaternion
        _qw_h, _qx_h, _qy_h, _qz_h = world._data.qpos[3:7]
        _heading = float(np.arctan2(2.0 * (_qw_h * _qz_h + _qx_h * _qy_h),
                                    1.0 - 2.0 * (_qy_h**2 + _qz_h**2)))
        # Dot product: displacement direction vs heading
        if _displacement > 1e-5:
            _move_angle = float(np.arctan2(_dy, _dx))
            _heading_alignment = float(np.cos(_move_angle - _heading))
            # forward_vel_aligned: positive when moving in facing direction
            forward_vel_aligned = _displacement * _heading_alignment
        else:
            forward_vel_aligned = 0.0

        # Detect physics reset: large position jump means keyframe reset happened
        # Body Awareness buffers become invalid after reset
        if abs(cur_x - prev_x) > 0.2 and step > 100:
            body_awareness.reset_after_physics_reset()

        # --- Delayed leg damage: injury during locomotion ---
        if args.leg_damage and _leg_damage_at_step > 0 and not _leg_damage_applied and step >= _leg_damage_at_step:
            n_act = world.n_actuators
            leg_map = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}
            leg_id = leg_map.get(args.leg_damage.upper())
            if leg_id is not None:
                jpleg = n_act // 4
                _damaged_actuators = list(range(leg_id * jpleg, (leg_id + 1) * jpleg))
                leg_prefix = args.leg_damage.lower()
                joint_names = [f'{leg_prefix}_hip_yaw', f'{leg_prefix}_hip_pitch', f'{leg_prefix}_knee']
                for jname in joint_names:
                    jid = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        world._model.jnt_stiffness[jid] = 0.0
                        world._model.dof_damping[world._model.jnt_dofadr[jid]] = 0.01
                for aid in _damaged_actuators:
                    if aid < world._model.nu:
                        world._model.actuator_gainprm[aid, 0] = 0.0
                        world._model.actuator_biasprm[aid, 1] = 0.0
                _leg_damage_applied = True
                print(f'\n  *** INJURY at step {step}: {args.leg_damage.upper()} leg FAILED ***')
                print(f'      The creature must detect and adapt autonomously.\n')

        # --- Diagnostic: pin base horizontally (true "wedged" stuck, observation only) ---
        # Viscously locks base x/y translation IN THE PHYSICS, so the accelerometer
        # reflects a body that cannot translate while the gait keeps churning. Only
        # active with --pin-base-at; runs without the flag are unchanged (bit-identical).
        # Decisive test for the hardware-stuck question (Lesson #205): does the accel
        # separate a body that is genuinely held? Fallback record before the forward-reward pivot.
        if args.pin_base_at >= 0 and not _pin_applied and step >= args.pin_base_at:
            _base_dof = 0
            for _j in range(world._model.njnt):
                if world._model.jnt_type[_j] == mujoco.mjtJoint.mjJNT_FREE:
                    _base_dof = int(world._model.jnt_dofadr[_j])
                    break
            world._model.dof_damping[_base_dof:_base_dof + 2] = 1000.0  # lock horizontal transl.
            _pin_applied = True
            print(f'\n  *** BASE PINNED at step {step}: horizontal translation locked (wedged) ***\n')

        ne_lvl_drive = creature.snn.neuromod_levels.get('ne', 0.2)
        desired_speed = max(0.15, ne_lvl_drive * 0.5)
        sensor_data['desired_velocity'] = desired_speed
        sensor_data['forward_velocity'] = vel_mps
        sensor_data['step'] = step
        sensor_data['standing_height'] = standing_h

        # Efferenzkopie: joint positions + last motor commands for forward model
        n_act = world.n_actuators
        sensor_data['joint_positions'] = world._data.qpos[7:7+n_act].copy()
        sensor_data['motor_commands'] = getattr(creature, '_last_controls', np.zeros(n_act))

        # --- Stuck detection update (observation only; Decision #203) ------------
        # Label = gait commanded but NO NET PROGRESS over the window. Instantaneous
        # |xy-velocity| is fooled by in-place oscillation/babbling (#88), so the
        # label uses windowed net displacement of the base (framepos = sim-only
        # trainer ground truth, brain never sees it; #59). Proxy = dynamic
        # (gravity-removed) accelerometer energy, which DOES exist on hardware.
        _gait_active = _use_opencat_gait and not is_fallen
        _horiz_speed = float(np.linalg.norm(world._data.qvel[0:2]))   # instantaneous (oscillation-fooled)
        _xy = world._data.qpos[0:2].copy()                            # sim-only ground truth (#59)
        _pos_window.append(_xy)
        _progress = (float(np.linalg.norm(_xy - _pos_window[0]))
                     if len(_pos_window) == _pos_window.maxlen else float('nan'))
        _acc_vec = np.asarray(sensor_data.get('linear_acceleration', np.zeros(3)), dtype=np.float64)
        if _acc_grav_ema is None and float(np.linalg.norm(_acc_vec)) > 1.0:
            _acc_grav_ema = _acc_vec.copy()                           # lazy seed from first valid reading
        if _acc_grav_ema is not None:
            _acc_grav_ema = 0.98 * _acc_grav_ema + 0.02 * _acc_vec
            _acc_dyn = float(np.linalg.norm(_acc_vec - _acc_grav_ema))
            _acc_dyn_ema = 0.9 * _acc_dyn_ema + 0.1 * _acc_dyn
        # Increment b (#208): feed buffered IMU coordination signal (gated; OFF -> no-op).
        if _coord_w > 0.0:
            _coord_buf_ax.append(float(_acc_vec[0]))
            _coord_buf_pr.append(float(np.asarray(sensor_data.get('angular_velocity', (0.0, 0.0, 0.0)))[1]))
            if step % _COORD_K == 0 and len(_coord_buf_ax) == _COORD_W:
                _cc = []
                for _b in (_coord_buf_ax, _coord_buf_pr):
                    _x = np.asarray(_b, dtype=np.float64)
                    _x = _x - _x.mean()
                    _tt = np.arange(len(_x))
                    _x = _x - np.polyval(np.polyfit(_tt, _x, 1), _tt)
                    _sp = np.abs(np.fft.rfft(_x * np.hanning(len(_x)))) ** 2
                    _fr = np.fft.rfftfreq(len(_x))
                    _pe = np.where(_fr > 0, 1.0 / np.maximum(_fr, 1e-12), np.inf)
                    _bd = (_pe >= 8.0) & (_pe <= 120.0)
                    _s = _sp[_bd].sum()
                    _cc.append(float(_sp[_bd].max() / _s) if _s > 0 else 0.0)
                _coord_concentration = float(np.mean(_cc))
        _stuck_truth = bool(_gait_active
                            and len(_pos_window) == _pos_window.maxlen
                            and _progress < STUCK_DISP_EPS)

        # Foot contact sensing (Phase A of Terrain-Adaptive Locomotion)
        foot_sensor.update(world._model, world._data, step)
        sensor_data.update(foot_sensor.get_data())

        # Gait quality: record joint positions + height + foot contacts every step
        _gait_joints = world._data.qpos[7:7+n_act].copy() if len(world._data.qpos) > 7+n_act else np.zeros(n_act)
        _gait_feet = foot_sensor.contacts.copy() if hasattr(foot_sensor, 'contacts') else None
        gait_analyzer.update(_gait_joints, height, _gait_feet)
        # Periodic analysis (every analysis_interval steps)
        if step % gait_cfg.analysis_interval == 0 and step > 0:
            gait_analyzer.analyze()
            _gq = gait_analyzer.stats()
            gait_analyzer._cached_gq = _gq.get('gait_quality', 0.5)
            gait_analyzer._cached_per = _gq.get('gait_periodicity', 0.0)
            gait_analyzer._cached_jit = _gq.get('gait_jitter', 0.0)
            gait_analyzer._cached_hr = _gq.get('gait_height_ratio', 0.5)

        # Body Awareness: detect limb failure from proprioceptive feedback
        _ba_cmds = getattr(creature, '_last_controls', None)
        if _ba_cmds is None:
            _ba_cmds = np.zeros(n_act)
        _ba_joints = world._data.qpos[7:7+n_act].copy() if len(world._data.qpos) > 7+n_act else np.zeros(n_act)
        if _ba_enabled:
            body_awareness.update(_ba_cmds, _ba_joints)
        # Check for limb state changes and auto-disconnect dead oscillators
        for evt in (body_awareness.get_events() if _ba_enabled else []):
            if evt['new_status'] == 'dead' and _is_mogli:
                leg_map = {'FL': 0, 'FR': 1, 'RL': 2, 'RR': 3}
                dead_idx = leg_map.get(evt['limb'])
                if dead_idx is not None:
                    spinal_cpg.coupling_weights[dead_idx, :] = 0.0
                    spinal_cpg.coupling_weights[:, dead_idx] = 0.0
                    print(f'  [BODY] {evt["limb"]} detected DEAD at step {step} '
                          f'(resp={evt["responsiveness"]:.3f}) — CPG coupling disconnected')

        # Developmental schedule: competence-driven perturbation + forward model
        # Biology: motor maturation is driven by competence, not time.
        # Competence = blend of: stability (upright), handoff progress (CPG→actor),
        # and actor speed. A creature that walks well stops wobbling.
        _cpg_w = getattr(gate, 'cpg_weight', 0.9)
        _actor_c = getattr(gate, 'actor_competence', 0.0)
        _up_ratio = current_upright_streak / max(step + 1, 1)
        _handoff = 1.0 - (_cpg_w - gate.cpg_min) / max(gate.cpg_max - gate.cpg_min, 0.01)
        _motor_competence = 0.4 * _handoff + 0.3 * min(1.0, _actor_c * 10) + 0.3 * _up_ratio
        dev_schedule.set_competence(_motor_competence)
        dev_schedule.step(step, world, creature)
        sensor_data['forward_model_gain'] = dev_schedule.get_forward_model_gain(step)
        _tp1 = time.perf_counter(); _profile['sensor'] += _tp1 - _tp0

        # --- Spatial Map: update position + observe landmarks ---
        _qw_sp, _qx_sp, _qy_sp, _qz_sp = world._data.qpos[3:7]
        _yaw_sp = float(np.arctan2(2.0 * (_qw_sp * _qz_sp + _qx_sp * _qy_sp),
                                    1.0 - 2.0 * (_qy_sp * _qy_sp + _qz_sp * _qz_sp)))
        # Task #84 step 1: wall-zone flag, used ONLY by the danger-anchor below (a
        # write-once landmark stamp, which the brain does not read). IMPORTANT: do NOT
        # gate update_position on this. spatial_map.position and get_explored_ratio() ARE
        # read by the brain via extra_sensor_data (spatial_x/y/explored/dist_home) and by
        # curiosity grid_coverage -- so ANY change to the dead-reckoned advance is NOT
        # control-neutral (it shifted the headline 0.292->0.274). The write-once latch
        # already fixes landmark drift, so no freeze is needed -- update_position stays
        # exactly as baseline.
        _at_wall_zone = (cur_x >= _WALL_ANCHOR_MIN_X)
        spatial_map.update_position(vel_mps, _yaw_sp, dt=args.timestep)
        _ball_id_sp = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
        if _ball_id_sp >= 0:
            _bp_sp = world._data.xpos[_ball_id_sp][:2].copy()
            _rel_ball_sp = _bp_sp - spatial_map.position
            spatial_map.observe_landmark('ball', _rel_ball_sp, category='goal',
                                        valence=0.8, distance=float(np.linalg.norm(_rel_ball_sp)))
        # v4.2: Observe light source as landmark (same as ball)
        if _lt_body_id >= 0:
            _lp_sp = world._data.xpos[_lt_body_id][:2].copy()
            _rel_light_sp = _lp_sp - spatial_map.position
            _light_dist = float(np.linalg.norm(_rel_light_sp))
            if _light_dist < 5.0:  # Only observe within camera range
                spatial_map.observe_landmark('light', _rel_light_sp, category='goal',
                                            valence=1.0, distance=_light_dist)
        _obs_dist_sp = sensor_data.get('obstacle_distance', -1.0)
        if _obs_dist_sp >= 0 and _obs_dist_sp < 0.5:
            # Rangefinder path (creatures with an ultrasonic sensor; never fires for
            # the Bittle, which is IMU-only -> obstacle_distance stays at no-hit).
            _wall_rel_sp = np.array([_obs_dist_sp * np.cos(_yaw_sp), _obs_dist_sp * np.sin(_yaw_sp)])
            spatial_map.observe_landmark('wall', _wall_rel_sp, category='danger', valence=-1.0,
                                         distance=float(_obs_dist_sp))
        elif _stuck_truth and _at_wall_zone and not _wall_anchored:
            # Contact-anchored wall memory (#232 / Task #84 step 1): the blind dog stores
            # the "pain" at the FIRST moment forward translation collapses while the gait
            # is still driving AND the robot has reached the wall zone (cur_x >=
            # _WALL_ANCHOR_MIN_X -- the wall-zone guard rejects the early maturation/arc
            # stall that otherwise anchored near home at ~step 57). Progress-stall over
            # STUCK_WINDOW=30 steps ~1-2 s -- contact-near, vs the ~622-step / ~30 s lag of
            # block_aversion. Stamped ONCE and latched
            # (_wall_anchored): re-averaging it every stuck step let the 0.10-m-ahead point
            # sweep as yaw scrubbed -> the observed rightward drift. Written as a DANGER
            # landmark (negative valence = the stored pain; readable by get_danger_nearby/
            # direction_to) ~0.10 m ahead of the dead-reckoned position in the current heading.
            # WRITE-ONLY: nothing reads it back to steer yet -- that is step 2 (the gated A/B,
            # the biological turn-away). HONESTY NOTE: _stuck_truth uses sim framepos (trainer
            # bookkeeping, the brain never sees it, #59); before step 2 reads this into control,
            # trigger + map position MUST be audited for IMU-only honesty (no privileged qpos
            # in a brain-read path).
            _wall_rel_sp = np.array([0.10 * np.cos(_yaw_sp), 0.10 * np.sin(_yaw_sp)])
            spatial_map.observe_landmark('wall', _wall_rel_sp, category='danger', valence=-1.0,
                                         distance=0.10)
            _wall_anchored = True

        # --- Issue #75: Sensory environment ---
        if visual_env:
            # --- PHOTOTAXIS MODE ---
            creature_pos = np.array([float(world._data.qpos[0]),
                                     float(world._data.qpos[1]),
                                     float(world._data.qpos[2])])
            # Geometric gradient (no camera in sim — uses position math)
            light_str, light_dir = visual_env.get_light_gradient(creature_pos)
            sensor_data['smell_strength'] = light_str
            sensor_data['smell_direction'] = float(np.arctan2(light_dir[1], light_dir[0]))
            sensor_data['sound_intensity'] = 0.0
            sensor_data['sound_direction'] = 0.0

            # Check if creature reached a light source
            _light_reached = visual_env.check_light_reached(creature_pos)
            if _light_reached:
                sensor_data['scent_reward'] = 0.5
                # Meta-Learning Loop Phase A: record successful navigation
                episode_analyzer.record_event('found', {
                    'smell_strength': sensor_data.get('smell_strength', 0.0),
                    'gait_quality': getattr(gait_analyzer, "_cached_gq", 0.5) if gait_analyzer else 0.0,
                    'heading_error': abs(getattr(creature, '_ball_heading', 0.0)),
                    'steering_offset': getattr(creature, '_steering_offset', 0.0),
                    'upright': upright,
                    'velocity': vel_mps,
                    'cpg_weight': gate.cpg_weight if gate else 0.9,
                    'actor_competence': gate.actor_competence if gate else 0.0,
                    'steps_since_last': step - getattr(creature, '_last_found_step', 0),
                    'cumulative_turn': getattr(creature, '_cumulative_turn', 0.0),
                }, step=step)
                creature._last_found_step = step
                creature._cumulative_turn = 0.0

            # Meta-Learning: periodic "missed" event if no light found for too long
            # Biology: foraging timeout — if no reward in N steps, strategy was bad
            _MISSED_THRESHOLD = 15000  # ~5 minutes of walking
            _steps_since = step - getattr(creature, '_last_found_step', 0)
            if (_steps_since > 0 and _steps_since % _MISSED_THRESHOLD == 0
                    and step > 5000):
                episode_analyzer.record_event('missed', {
                    'smell_strength': sensor_data.get('smell_strength', 0.0),
                    'gait_quality': getattr(gait_analyzer, "_cached_gq", 0.5) if gait_analyzer else 0.0,
                    'heading_error': abs(getattr(creature, '_ball_heading', 0.0)),
                    'steering_offset': getattr(creature, '_steering_offset', 0.0),
                    'upright': upright,
                    'velocity': vel_mps,
                    'cpg_weight': gate.cpg_weight if gate else 0.9,
                    'actor_competence': gate.actor_competence if gate else 0.0,
                    'steps_since_last': _steps_since,
                    'cumulative_turn': getattr(creature, '_cumulative_turn', 0.0),
                }, step=step)
            # Sync light body position in MuJoCo every 10 steps (not every step)
            if _lt_qposadr >= 0 and step % 10 == 0 and len(visual_env.get_light_positions()) > 0:
                _lt_pos = visual_env.get_light_positions()[0]
                world._data.qpos[_lt_qposadr:_lt_qposadr + 3] = _lt_pos
                world._data.qpos[_lt_qposadr + 3:_lt_qposadr + 7] = [1, 0, 0, 0]
                world._data.qvel[_lt_dofadr:_lt_dofadr + 6] = 0.0

            # Phototactic steering — v0.5.0 IMU PD closed-loop
            # Same approach as Bridge v4.4: camera provides target_yaw,
            # PD controller on yaw error drives asymmetric stride via CPG.
            # Hardware-validated: Test C (2026-05-03) proved this 3x more
            # effective than VOR/abduction-offset steering.
            _qw_olf, _qx_olf, _qy_olf, _qz_olf = world._data.qpos[3:7]
            heading = float(np.arctan2(2.0 * (_qw_olf * _qz_olf + _qx_olf * _qy_olf),
                                       1.0 - 2.0 * (_qy_olf**2 + _qz_olf**2)))
            _yaw_deg = float(np.degrees(heading))

            # Track cumulative turning for EpisodeAnalyzer
            if hasattr(creature, '_prev_yaw_for_turn'):
                _yaw_change = abs(_yaw_deg - creature._prev_yaw_for_turn)
                if _yaw_change > 180: _yaw_change = 360 - _yaw_change
                creature._cumulative_turn = getattr(creature, '_cumulative_turn', 0.0) + _yaw_change
            creature._prev_yaw_for_turn = _yaw_deg

            _actor_comp = getattr(gate, 'actor_competence', 0.0)
            if _actor_comp > 0.1 or step > 5000:
                olf_steer = visual_env.get_phototactic_steering(
                    creature_pos, heading)
            else:
                olf_steer = 0.0
            sensor_data['olfactory_steering'] = olf_steer
            sensor_data['scents_found'] = visual_env.lights_found

            # Initialize PID controller state (once)
            if not hasattr(creature, '_pd_yaw_target'):
                creature._pd_yaw_target = _yaw_deg
                creature._pd_prev_error = 0.0
                creature._pd_integral = 0.0
                creature._pd_steering = 0.0
                # PID gains — sim needs higher Kp than hardware (0.03) because
                # HardwareDrift.apply() torque is stronger than real mechanical drift
                creature._pd_kp = 0.08
                creature._pd_ki = 0.005   # I-term: eliminates steady-state drift offset
                creature._pd_kd = 0.02
                creature._pd_max = 0.6
                creature._pd_integral_max = 30.0  # anti-windup clamp

            # Camera/phototaxis provides target heading:
            # olf_steer convention from get_phototactic_steering():
            #   positive = light is to the LEFT (angle_diff = light_angle - heading)
            #   negative = light is to the RIGHT
            # MuJoCo yaw convention: positive = counterclockwise (left)
            # So: olf_steer positive → need to turn left → increase yaw
            _HALF_FOV_DEG = 31.0
            _cam_salience = sensor_data.get('smell_strength', 0.0)
            if _cam_salience > 0.02:
                # Always update target when light is visible, even if heading is small
                # Old threshold (0.05) prevented target updates when dog was already
                # pointed roughly at the light, causing it to drift past
                creature._pd_yaw_target = _yaw_deg + olf_steer * _HALF_FOV_DEG
                creature._ball_heading = olf_steer
                creature._ball_salience = _cam_salience
            else:
                creature._ball_heading = 0.0
                creature._ball_salience = 0.0

            # v4.2: LightMemory — when light lost, steer to remembered yaw
            if light_memory:
                _mem_z = light_memory.update(
                    cam_salience=_cam_salience,
                    cam_heading=olf_steer,
                    current_yaw=_yaw_deg,
                    current_time=time.time(),
                )
                if _cam_salience <= 0.05 and light_memory.state == 'returning':
                    creature._pd_yaw_target = light_memory._target_yaw

            # PID controller: yaw error -> steering
            _yaw_error = creature._pd_yaw_target - _yaw_deg
            # Normalize to [-180, 180]
            while _yaw_error > 180: _yaw_error -= 360
            while _yaw_error < -180: _yaw_error += 360

            # I-term: accumulate error over time (eliminates steady-state offset)
            # Biology: cerebellar LTD accumulates over seconds/minutes
            creature._pd_integral += _yaw_error * args.timestep
            creature._pd_integral = max(-creature._pd_integral_max,
                                        min(creature._pd_integral_max, creature._pd_integral))

            _error_rate = (_yaw_error - creature._pd_prev_error) / args.timestep
            creature._pd_prev_error = _yaw_error

            _steering = (creature._pd_kp * _yaw_error +
                         creature._pd_ki * creature._pd_integral +
                         creature._pd_kd * _error_rate)
            # Negate: MuJoCo yaw positive = left, but CPG steering positive = right
            # Positive yaw_error means "need to turn left" → negative CPG steering
            _steering = -_steering
            _steering = max(-creature._pd_max, min(creature._pd_max, _steering))

            # Low-pass filter
            _alpha = 0.3
            creature._pd_steering = creature._pd_steering * (1.0 - _alpha) + _steering * _alpha

            # Apply to creature steering
            creature._steering_offset = creature._pd_steering

        elif sensory_env:
            creature_pos = np.array([float(world._data.qpos[0]),
                                     float(world._data.qpos[1]),
                                     float(world._data.qpos[2])])
            smell_str, smell_dir = sensory_env.get_smell_gradient(creature_pos)
            sensor_data['smell_strength'] = smell_str
            sensor_data['smell_direction'] = float(np.arctan2(smell_dir[1], smell_dir[0]))

            sound = sensory_env.update_sound(step)
            if sound:
                sensor_data['sound_intensity'] = sound.intensity
                sensor_data['sound_direction'] = float(
                    np.arctan2(sound.direction[1], sound.direction[0]))
            else:
                sensor_data['sound_intensity'] = 0.0
                sensor_data['sound_direction'] = 0.0

            # Check if creature reached a scent source
            if sensory_env.check_scent_reached(creature_pos):
                sensor_data['scent_reward'] = 0.5

            # Olfactory steering: modulate abduction to turn toward scent
            # Biology: bilateral olfactory comparison creates turning
            # tendency (chemotaxis). Ref: Catania 2006, Porter et al. 2007
            # GATE: only steer when actor has basic competence (prevents
            # destabilizing the CPG gait during early learning).
            # v0.4.8 fix: extract yaw from quaternion (qpos[3] is qw, NOT heading!)
            _qw_olf, _qx_olf, _qy_olf, _qz_olf = world._data.qpos[3:7]
            heading = float(np.arctan2(2.0 * (_qw_olf * _qz_olf + _qx_olf * _qy_olf),
                                       1.0 - 2.0 * (_qy_olf**2 + _qz_olf**2)))
            _actor_comp = getattr(gate, 'actor_competence', 0.0)
            if _actor_comp > 0.1 or step > 5000:  # Steer once minimally stable, or after 5k
                olf_steer = sensory_env.get_olfactory_steering(creature_pos, heading)
            else:
                olf_steer = 0.0  # Too early — let CPG establish gait first
            sensor_data['olfactory_steering'] = olf_steer
            # Pass navigation data to sensor_data for cerebellar CF (Issue #81)
            sensor_data['ball_heading'] = getattr(creature, '_ball_heading', 0.0)
            sensor_data['steering_offset'] = getattr(creature, '_steering_offset', 0.0)

            # Update ball scent position to follow the ball (if it moved)
            ball_id_rt = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
            if ball_id_rt >= 0 and len(sensory_env._scents) > 0 and sensory_env._scents[0].name == 'ball_scent':
                sensory_env._scents[0].position = world._data.xpos[ball_id_rt].copy()

            # --- Issue #76d: Ball heading + salience for SNN steering ---
            # Compute relative heading to ball and salience (inverse distance).
            # These are set as creature attributes and read by get_sensor_input()
            # (Teil A: SNN input bias) and apply_motor_output() (Teil B: steering offset).
            if ball_id_rt >= 0:
                _c_pos = np.array([float(world._data.qpos[0]), float(world._data.qpos[1])])
                _b_pos = world._data.xpos[ball_id_rt][:2].copy()
                _to_ball = _b_pos - _c_pos
                _ball_dist_2d = float(np.linalg.norm(_to_ball))
                # Salience: 1.0 when touching, fades over 15m (was 5m — too aggressive)
                # A dog can see/smell a ball from much further than 5m.
                _ball_salience = max(0.0, 1.0 - _ball_dist_2d / 15.0)
                # Heading: angle to ball relative to creature facing direction
                # qpos[3:7] is quaternion — extract yaw from quaternion
                _qw, _qx, _qy, _qz = world._data.qpos[3:7]
                _creature_yaw = float(np.arctan2(2.0 * (_qw * _qz + _qx * _qy),
                                                  1.0 - 2.0 * (_qy * _qy + _qz * _qz)))
                _ball_angle = float(np.arctan2(_to_ball[1], _to_ball[0]))
                _heading_error = _ball_angle - _creature_yaw
                # Normalize to [-pi, pi]
                while _heading_error > np.pi: _heading_error -= 2 * np.pi
                while _heading_error < -np.pi: _heading_error += 2 * np.pi
                # Normalize to [-1, 1] where -1=ball is 180deg left, +1=180deg right
                _ball_heading = np.clip(_heading_error / np.pi, -1.0, 1.0)
                creature._ball_heading = _ball_heading
                creature._ball_salience = _ball_salience
                # Vision system: set visual target for SNN input (Issue #76d)
                # _visual_target_heading: -1 (ball far left) to +1 (ball far right)
                # _visual_target_distance: 0 (far/no target) to 1 (touching)
                creature._visual_target_heading = _ball_heading
                creature._visual_target_distance = _ball_salience
                # VOR: Superior Colliculus reflex — turn body toward visual target
                # This is hardwired (not learned). The SNN learns WHAT to do
                # with the target via the vision channels + DA reward.
                _cur_cpg_w = getattr(gate, 'cpg_weight', 0.9)
                _vor_steer = vor.compute(_ball_heading, _ball_salience, upright, cpg_weight=_cur_cpg_w)
                # Issue #81: Cerebellum calibrates VOR gain in real-time
                # Like the flocculus adapting saccade gain after every eye movement
                if cb:
                    _cb_mod = cb.inferior_olive.get_steering_gain_correction()
                    _vor_steer = _vor_steer * (1.0 + _cb_mod)
                creature._steering_offset = _vor_steer
            else:
                creature._ball_heading = 0.0
                creature._ball_salience = 0.0
                creature._visual_target_heading = 0.0
                creature._visual_target_distance = 0.0
                _cur_cpg_w = getattr(gate, 'cpg_weight', 0.9)
                creature._steering_offset = vor.compute(0.0, 0.0, upright, cpg_weight=_cur_cpg_w)

        # Auto-reset: if creature has been fallen for too long, reset to standing
        # Biology: in real RL, episodes reset. In nature, a parent helps.
        # The SNN/cerebellum weights are preserved — only physics resets.
        if is_fallen:
            consecutive_fallen += 1
        else:
            consecutive_fallen = 0

        # ================================================================
        # BALL EPISODE RESET (Issue #76d: episodic learning)
        # ================================================================
        # Biology: A puppy that misses the ball doesn't walk 20m away.
        # Its owner picks it up and places it near the ball again.
        # Each attempt is short (seconds, not minutes). The puppy gets
        # hundreds of attempts per play session. Each attempt teaches
        # the SNN a little more. Weights accumulate across episodes.
        #
        # Without this, the Go2 gets ONE chance in 50k steps:
        #   - Approach ball (8k steps, reward signal)
        #   - Walk past (42k steps, ZERO reward signal)
        # That's 84% of training with no learning at all.
        #
        # With episodic reset at bd > 8m:
        #   - Attempt 1: approach, miss, reset at step ~10k
        #   - Attempt 2: approach, miss (less), reset at step ~18k
        #   - Attempt 3: approach, reach ball at step ~24k!
        # Each attempt has dense reward signal. SNN learns 5x faster.
        # ================================================================
        _ball_episode_reset = False
        if _scene_has_ball and prev_ball_dist is not None:
            if prev_ball_dist > 8.0 and step > 5000:
                # Go2 has walked too far from ball — reset episode
                _ball_episode_reset = True
                ball_episode_count = getattr(main, '_ball_ep', 0) + 1
                main._ball_ep = ball_episode_count
                # Reset physics to standing pose (keyframe 0)
                if world._model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(world._model, world._data, 0)
                else:
                    world._data.qpos[:] = 0
                    world._data.qvel[:] = 0
                    world._data.qpos[2] = standing_h + 0.02
                mujoco.mj_forward(world._model, world._data)
                consecutive_fallen = 0
                is_fallen = False
                creature._was_fallen = False
                creature._prev_x = float(world._data.qpos[0])
                # Curriculum: advance stage if Go2 got close to ball (Issue #86)
                # Use running minimum (tracked every step), not prev_ball_dist
                # (which is >8m at reset time — that's WHY we reset!)
                if main._ball_best_dist_running < 0.5:  # Ball contact!
                    old_stage = main._ball_stage
                    main._ball_stage = min(main._ball_stage + 1, len(main._ball_positions) - 1)
                    main._ball_best_dist = 99.0  # Reset for new stage
                    main._ball_best_dist_running = 99.0  # Reset running tracker
                    if main._ball_stage > old_stage:
                        _bp_new = main._ball_positions[main._ball_stage]
                        print(f'  [CURRICULUM ADVANCE → Stage {main._ball_stage}: ball at ({_bp_new[0]:.1f}, {_bp_new[1]:.1f})]')
                # Set ball position from curriculum stage
                _bp = main._ball_positions[min(main._ball_stage, len(main._ball_positions) - 1)]
                _ball_id_ep = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
                if _ball_id_ep >= 0:
                    _ball_jnt_ep = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
                    _bqa_ep = world._model.jnt_qposadr[_ball_jnt_ep]
                    world._data.qpos[_bqa_ep:_bqa_ep + 3] = list(_bp)
                    world._data.qpos[_bqa_ep + 3:_bqa_ep + 7] = [1.0, 0.0, 0.0, 0.0]
                    _bda_ep = world._model.jnt_dofadr[_ball_jnt_ep]
                    world._data.qvel[_bda_ep:_bda_ep + 6] = 0.0
                    mujoco.mj_forward(world._model, world._data)
                    # Update scent source to match new ball position
                    if sensory_env and len(sensory_env._scents) > 0:
                        sensory_env._scents[0].position = np.array([_bp[0], _bp[1], 0.0])
                # Reset cerebellum episode state (not weights!)
                if cb:
                    cb.reset_episode()
                # Reset Body Awareness after physics reset
                body_awareness.reset_after_physics_reset()
                # Negative DA burst: "you missed the ball" — weakens recent synapses
                if hasattr(creature, 'brain') and creature.brain:
                    creature.brain.snn.set_neuromodulator('da', 0.0)
                    # Dream consolidation between episodes (biology: sleep replay)
                    # The puppy rests, its hippocampus replays the approach,
                    # strengthens patterns that led to reward, prunes the rest.
                    if hasattr(creature.brain, 'dream_engine'):
                        try:
                            creature.brain.dream_engine.dream_step(
                                creature.brain.snn, n_steps=10)
                        except Exception:
                            pass  # Dream mode may not be fully connected yet
                    # Synaptogenesis: consolidate patterns from this episode
                    if hasattr(creature.brain, 'synaptogenesis'):
                        try:
                            creature.brain.synaptogenesis.consolidate()
                        except Exception:
                            pass  # May fail if buffer empty
                prev_ball_dist = None  # Reset tracking
                print(f'  [BALL EPISODE #{ball_episode_count} at step {step} — bd>{8.0:.0f}m, resetting]')

        # OpenCat path: the physical pose-recovery above is primary; the teleport
        # reset is only a far fallback, so give it 4x longer to right itself (#25).
        _eff_reset_limit = auto_reset_limit * (4 if _use_opencat_gait else 1)
        if auto_reset_limit > 0 and consecutive_fallen >= _eff_reset_limit:
            # Reset physics to standing pose (keyframe 0)
            if world._model.nkey > 0:
                mujoco.mj_resetDataKeyframe(world._model, world._data, 0)
            else:
                world._data.qpos[:] = 0
                world._data.qvel[:] = 0
                world._data.qpos[2] = standing_h + 0.02
            mujoco.mj_forward(world._model, world._data)
            consecutive_fallen = 0
            reset_count += 1
            is_fallen = False
            creature._was_fallen = False
            creature._prev_x = float(world._data.qpos[0])
            _stuck_counter = 0  # Issue #110: reset stuck counter
            _progress_stuck_counter = 0  # Issue #125: reset progress detector
            _progress_last_max_dist = 0.0
            body_awareness.reset_after_physics_reset()  # v0.7.0: prevent false positives
            # Restore ball position after reset (keyframe reset zeros all qpos)
            _ball_id_reset = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
            if _ball_id_reset >= 0:
                _ball_jnt = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
                _bqa = world._model.jnt_qposadr[_ball_jnt]
                world._data.qpos[_bqa:_bqa + 3] = [3.0, 2.0, 0.12]
                world._data.qpos[_bqa + 3:_bqa + 7] = [1.0, 0.0, 0.0, 0.0]
                _bda = world._model.jnt_dofadr[_ball_jnt]
                world._data.qvel[_bda:_bda + 6] = 0.0
                mujoco.mj_forward(world._model, world._data)
            if step < 100 or reset_count <= 3 or reset_count % 10 == 0:
                print(f'  [RESET #{reset_count} at step {step}]')

        # Issue #110: Velocity-based stuck detection
        # Catches Go2 lying on its side (upright ~0.35, not detected as fallen)
        # or any creature that is motionless but not technically "fallen".
        # Biology: A stuck animal thrashes and tries to right itself.
        # Our auto-reset is the "parent picking up the puppy".
        if auto_reset_limit > 0 and not is_fallen:
            if vel_mps < _STUCK_VELOCITY_MAX and upright < _STUCK_UPRIGHT_MAX:
                _stuck_counter += 1
            else:
                _stuck_counter = 0
            if _stuck_counter >= _STUCK_THRESHOLD_STEPS:
                # Stuck: motionless + not upright for too long → reset
                if world._model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(world._model, world._data, 0)
                else:
                    world._data.qpos[:] = 0
                    world._data.qvel[:] = 0
                    world._data.qpos[2] = standing_h + 0.02
                mujoco.mj_forward(world._model, world._data)
                _stuck_counter = 0
                consecutive_fallen = 0
                reset_count += 1
                is_fallen = False
                creature._was_fallen = False
                creature._prev_x = float(world._data.qpos[0])
                _progress_stuck_counter = 0  # Issue #125
                _progress_last_max_dist = 0.0
                body_awareness.reset_after_physics_reset()  # v0.7.0: prevent false positives
                # Restore ball position
                _ball_id_stuck = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
                if _ball_id_stuck >= 0:
                    _ball_jnt_s = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
                    _bqa_s = world._model.jnt_qposadr[_ball_jnt_s]
                    world._data.qpos[_bqa_s:_bqa_s + 3] = [3.0, 2.0, 0.12]
                    world._data.qpos[_bqa_s + 3:_bqa_s + 7] = [1.0, 0.0, 0.0, 0.0]
                    _bda_s = world._model.jnt_dofadr[_ball_jnt_s]
                    world._data.qvel[_bda_s:_bda_s + 6] = 0.0
                    mujoco.mj_forward(world._model, world._data)
                if reset_count <= 10 or reset_count % 10 == 0:
                    print(f'  [STUCK RESET #{reset_count} at step {step} — vel={vel_mps:.4f} up={upright:.2f}]')

        # Track fall *transitions*, not every step while fallen
        was_fallen = getattr(creature, '_was_fallen', False)
        if is_fallen:
            upright_delta = upright - prev_upright
            # Recovery reward: any improvement in upright is strongly rewarded
            # Even going from -1.0 to -0.8 should produce DA for learning
            recovery_reward = max(0, upright_delta) * 30.0
            # Baseline: small reward proportional to how close to upright
            # upright ranges from -1 (inverted) to 1 (standing)
            # Normalize to 0..1 range for reward
            upright_normalized = (upright + 1.0) / 2.0  # -1→0, 0→0.5, 1→1
            stability_reward = upright_normalized * 1.0
            reward = recovery_reward + stability_reward - 0.05
            if not was_fallen:
                fall_count += 1  # Count only the fall transition
            current_upright_streak = 0
        else:
            reward = forward_vel_aligned * 15.0 + max(0, upright) * 2.0
            # v0.7.0: Backward locomotion discomfort
            # Biology: an animal walking backward cannot see where it goes.
            if forward_vel_aligned < -0.001:
                reward += forward_vel_aligned * 5.0  # backward penalty
            # Upright bonus: strong nonlinear reward for being truly upright
            # Biology: proprioceptive reward — standing feels good, tilting hurts
            # This drives the SNN to find motor patterns that maximize upright
            # Critical for leg-loss recovery: the difference between 0.77 (dragging)
            # and 0.98 (upright) must produce a large reward difference
            if upright > 0.9:
                reward += (upright - 0.9) * 30.0  # up to +3.0 for perfect upright
            elif upright < 0.7:
                reward -= (0.7 - upright) * 5.0   # penalty for significant tilt
            # Gait quality reward (v0.7.0 Pillar 3)
            # Only after enough data collected (first ~400 steps)
            _gait_rewards = gait_analyzer.get_reward_components()
            reward += _gait_rewards.get('gait_reward', 0.0)
            current_upright_streak += 1
            if current_upright_streak > best_upright_streak:
                best_upright_streak = current_upright_streak
            if was_fallen and upright > 0.6:
                reward += 15.0  # Strong recovery bonus
                recovery_count += 1
        creature._was_fallen = is_fallen

        # --- Issue #76d: Ball approach reward (Schultz 1997) ---
        # DA signal proportional to distance decrease toward ball.
        # The SNN learns HOW to navigate — no motor hack needed.
        # Only active in ball scenes, only when not fallen.
        ball_approach_reward = 0.0
        _ball_id_reward = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
        if _ball_id_reward >= 0 and not is_fallen:
            creature_pos_2d = np.array([float(world._data.qpos[0]), float(world._data.qpos[1])])
            ball_pos_2d = world._data.xpos[_ball_id_reward][:2].copy()
            ball_dist = float(np.linalg.norm(creature_pos_2d - ball_pos_2d))
            if prev_ball_dist is not None:
                # Reward for getting closer (positive delta = approaching)
                approach_delta = prev_ball_dist - ball_dist
                ball_approach_reward = max(0.0, approach_delta) * 10.0
                # Heading reward: bonus when facing the ball
                creature_heading = float(world._data.qpos[3])  # quaternion w component
                to_ball = ball_pos_2d - creature_pos_2d
                to_ball_angle = float(np.arctan2(to_ball[1], to_ball[0]))
                heading_error = abs(to_ball_angle - creature_heading)
                if heading_error > np.pi:
                    heading_error = 2 * np.pi - heading_error
                # Max heading bonus when facing ball (error=0), zero at 90deg+
                heading_reward = max(0.0, 1.0 - heading_error / (np.pi * 0.5)) * 0.5
                ball_approach_reward += heading_reward
                # Contact bonus: big reward when very close to ball
                if ball_dist < 0.3:
                    ball_approach_reward += 5.0
            prev_ball_dist = ball_dist
            # Track running minimum for curriculum (Issue #86)
            if ball_dist < main._ball_best_dist_running:
                main._ball_best_dist_running = ball_dist
            reward += ball_approach_reward

        # --- Issue #103: Obstacle avoidance reward ---
        # Strong negative DA when hitting wall, positive when avoiding.
        # This gives the SNN a clear, binary learning signal.
        #
        # Hysteresis: After the rangefinder detects a wall and then loses it
        # (e.g. robot deflects sideways), keep the negative reward active
        # for a cooldown period. This prevents the reward from flickering
        # between "wall" and "safe" on every step.
        obstacle_reward = 0.0
        _obs_dist = sensor_data.get('obstacle_distance', -1.0)
        
        # Hysteresis: if rangefinder recently saw wall, use last known distance
        if _obs_dist >= 0 and _obs_dist < 2.0:
            _wall_last_obs_dist = _obs_dist
            _wall_obs_cooldown = 50  # keep active for 50 steps after last detection
        elif _wall_obs_cooldown > 0:
            _wall_obs_cooldown -= 1
            _obs_dist = _wall_last_obs_dist  # use last known distance
        
        if _scene_has_wall and _obs_dist >= 0 and not is_fallen:
            if _obs_dist < 0.15:
                # COLLISION: strong punishment — "you hit the wall"
                obstacle_reward = -15.0
                sensor_data['obstacle_collision'] = True
            elif _obs_dist < 0.30:
                # DANGER ZONE: moderate punishment — "too close, brake!"
                obstacle_reward = -3.0 * (0.30 - _obs_dist) / 0.15
            elif _obs_dist < 0.80:
                # WARNING: small punishment — "pay attention"
                obstacle_reward = -0.5 * (0.80 - _obs_dist) / 0.50
            else:
                # SAFE: small reward for maintaining distance
                obstacle_reward = 0.2
            reward += obstacle_reward
        # IMU-as-rangefinder (#108 reuse, #213 variance): when there is no real
        # rangefinder (blind IMU Bittle -> _obs_dist < 0), synthesize the distance
        # from OUR block variance. Maps block_aversion 0..1 -> 0.30..0.0 m so the
        # EXISTING brake/turn/reverse branch (above + #108) consumes it unchanged:
        # free walk (aversion 0) -> 0.30 m = no trigger; more blocked = "closer" =
        # slow -> turn -> stop -> reverse. Injected AFTER the external obstacle_reward,
        # so that reward stays -1 (OFF) and only the REFLEX sees it -> intrinsic, no
        # external reward. Uses previous step's _block_aversion (200-step rolling;
        # 1-step lag negligible). Flag-gated, default off -> bit-identical.
        if args.imu_obstacle:
            # Run-and-Tumble obstacle avoidance (#108 RT, Marc). The blind IMU has no
            # distance sense (the Bittle rangefinder reads a constant ~4.0), so instead
            # of the continuous throttle+bias reflex (which trapped the robot in slow-turn
            # limbo), the IMU block signal drives a discrete cycle that SEQUENCES the
            # primitives that already exist:
            #   RUN    — straight, full gait (kwkF). Going straight, block_aversion is a
            #            clean wall reading (residual == raw scrub).
            #   SNIFF  — at the end of a RUN: still blocked? -> commit a turn.
            #   TUMBLE — a committed turn at full gait (kwkL) for a fixed span. The
            #            variance feedback subtracts this commanded yaw, so the next
            #            SNIFF still reads only wall-scrub, not our own turning.
            # Over several cycles the heading accumulates until a RUN stays clear =
            # escape. obstacle_distance is left at its real reading, so the continuous
            # #108 reflex stays dormant and this is the sole obstacle response. Uses the
            # PREVIOUS step's _block_aversion (1-step lag, negligible). _OB_STEER is
            # added to _cpg_steering below. No external reward -> intrinsic.
            _OB_STEER = 0.0
            if _OB_TIMER > 0:
                _OB_TIMER -= 1
            if _OB_STATE == 'RUN':
                if _OB_TIMER <= 0:
                    # SNIFF (instantaneous): evaluate the clean straight-run block reading.
                    if _block_aversion >= _OB_BLOCK_ON:
                        _OB_STATE = 'TUMBLE'
                        _OB_TIMER = _OB_TUMBLE
                        _ob_tumbles += 1
                    else:
                        _OB_TIMER = _OB_RUN          # clear -> keep running straight
            if _OB_STATE == 'TUMBLE':
                _OB_STEER = _OB_TURN_GAIN           # committed turn (kwkL), full gait
                if _OB_TIMER <= 0:
                    _OB_STATE = 'RUN'
                    _OB_TIMER = _OB_RUN
        # Pass obstacle distance to sensor_data for cerebellar CF
        sensor_data['obstacle_distance'] = _obs_dist

        # --- Issue #103: Episodic wall reset ---
        # Biology: A puppy that bumps into a wall doesn't teleport away.
        # It stands there perplexed for a moment, then backs up.
        # The pause is visible in video and gives the cerebellum time
        # to process the collision CF signal before the episode resets.
        #
        # Two-phase reset:
        #   Phase 1: Wall contact detected → CPG killed, robot stands still
        #            for _WALL_PAUSE_STEPS (~50 steps / ~1.5s)
        #   Phase 2: After pause → physics reset to start position
        _wall_reset = False

        # Phase 1: Detect wall contact, start pause
        if _scene_has_wall and not getattr(args, 'no_wall_reset', False) and _obs_dist >= 0 and _obs_dist < 0.10 and step > 500 and _wall_pause_counter == 0:
            _wall_pause_counter = _WALL_PAUSE_STEPS
            wall_episode_count += 1
            # Negative DA burst: "you hit the wall"
            creature.snn.set_neuromodulator('da', 0.0)
            if wall_episode_count <= 20 or wall_episode_count % 5 == 0:
                print(f'  [WALL HIT #{wall_episode_count} at step {step}, x={cur_x:.2f}]')

        # During pause: count down, CPG override happens below (before CPG compute)
        if _wall_pause_counter > 0:
            _wall_pause_counter -= 1
            if _wall_pause_counter == 0:
                # Phase 2: Pause over → reset physics
                _wall_reset = True
                if world._model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(world._model, world._data, 0)
                else:
                    world._data.qpos[:] = 0
                    world._data.qvel[:] = 0
                    world._data.qpos[2] = standing_h + 0.02
                mujoco.mj_forward(world._model, world._data)
                consecutive_fallen = 0
                is_fallen = False
                creature._was_fallen = False
                creature._prev_x = float(world._data.qpos[0])
                _wall_last_obs_dist = 4.0
                _wall_obs_cooldown = 0
                # Reset cerebellum episode state (not weights!)
                if cb:
                    cb.reset_episode()
                body_awareness.reset_after_physics_reset()  # v0.7.0: prevent false positives
                # Dream consolidation: replay obstacle approach patterns
                if hasattr(creature, 'brain') and creature.brain:
                    if hasattr(creature.brain, 'dream_engine'):
                        try:
                            creature.brain.dream_engine.dream_step(
                                creature.brain.snn, n_steps=10)
                        except Exception:
                            pass
                    if hasattr(creature.brain, 'synaptogenesis'):
                        try:
                            creature.brain.synaptogenesis.consolidate()
                        except Exception:
                            pass

        # Debug: show obstacle distance for first few steps
        # (uncomment for debugging: print(f'  [DBG step {step}] obs_dist=...'))

        # ================================================================
        # BABY-KI: Compute learning signal from intrinsic reward blend
        # ================================================================
        # External reward (`reward`) is still computed above for FLOG and
        # metrics. The LEARNING signal sent to the SNN is the blend:
        #   blend=0.0 → pure intrinsic (Baby-KI default)
        #   blend=1.0 → pure external (v0.4.3 RL behavior)
        # get_intrinsic_reward() reads signals populated by creature.step()
        # → process(), so we call it AFTER creature.step() below and
        # apply R-STDP retroactively. For the DA neuromodulator and the
        # reward_signal passed to process(), we use the blend of the
        # PREVIOUS step's intrinsic reward (which is 0.0 on step 0 — safe).
        _prev_intrinsic = creature.brain._intrinsic_reward if creature.brain else 0.0
        _blend = args.reward_blend
        learning_signal = _blend * reward + (1.0 - _blend) * _prev_intrinsic

        # DA signal: derived from learning_signal, not raw external reward
        da_signal = np.clip(learning_signal / 10.0, 0.05, 1.0)
        creature.snn.neuromod_levels['da'] = float(da_signal)

        reflex_cmd = reflexes.compute(sensor_data, is_fallen, sim_dt=args.timestep)
        if _disable_reflexes:
            reflex_cmd = np.zeros_like(reflex_cmd)

        # Dynamic reflex scale — proportional to instability
        instability = max(0.0, 1.0 - upright)
        urgency = min(1.0, instability * instability / (0.7 * 0.7))
        creature.reflex_scale = (creature._reflex_scale_standing +
            (creature._reflex_scale_fallen - creature._reflex_scale_standing) * urgency)

        if is_fallen:
            stress = np.clip(1.0 - upright, 0.0, 1.0)
            ne_level = min(0.9, 0.3 + stress * 0.5)
            creature.snn.set_neuromodulator('ne', ne_level)
            creature.snn._hidden_tonic_current = 0.02 + stress * 0.08
        elif not is_fallen:
            # v0.4.5: Trainer ALWAYS computes NE/tonic (no longer skipped
            # when intrinsic_arousal_drive is True). The brain's arousal
            # oscillator is read and ADDED on top. RAS modulates cortex.
            prev_brain = brain_result
            cur_reward_b = prev_brain.get('curiosity_reward', 0.0) if prev_brain else 0.0
            drv = prev_brain.get('drives', {}) if prev_brain else {}
            expl_drive = drv.get('exploration', 0.3)
            boredom = prev_brain.get('boredom', 0.0) if prev_brain else 0.0
            ne_base = 0.25
            ne_curiosity = cur_reward_b * 0.3
            ne_boredom = min(boredom, 1.0) * 0.2
            ne_level = ne_base + ne_curiosity + ne_boredom
            tonic_base = 0.02
            tonic_explore = expl_drive * 0.04
            tonic_curiosity = cur_reward_b * 0.03
            tonic = tonic_base + tonic_explore + tonic_curiosity

            # v0.4.5: Additive arousal oscillator from brain
            arousal_osc = 0.0
            if (hasattr(creature, 'brain') and creature.brain
                    and creature.brain.config.intrinsic_arousal_drive):
                arousal_osc = getattr(creature.brain, '_arousal_oscillator', 0.0)
                ne_level += arousal_osc * 0.15   # gentle NE boost from oscillator
                tonic += arousal_osc * 0.03      # gentle tonic boost from oscillator

            creature.snn.set_neuromodulator('ne', min(0.8, ne_level))
            creature.snn._hidden_tonic_current = tonic

        if cb:
            cb.update(creature, sensor_data)

        ne_lvl = creature.snn.neuromod_levels.get('ne', 0.2)

        # --- Issue #57: Drive Loop → CPG modulation ---
        if drive_bridge and step > 500:
            # Wait 500 steps for brain to warm up before drive loop kicks in
            current_freq_scale, current_amp_scale, current_behavior = \
                drive_bridge.update(brain_result, sensor_data, is_fallen)
        else:
            current_freq_scale = 1.0
            current_amp_scale = 1.0
            current_behavior = 'walk'

        # Motor babbling: set per-leg noise when neonatal behavior active
        # Biology: fidgety movements (Prechtl 1997) create asymmetric
        # limb activity that shifts CoM → vestibular calibration signal.
        if current_behavior == 'motor_babbling' and hasattr(spinal_cpg, '_babbling_noise'):
            spinal_cpg._babbling_noise = 0.25  # 25% per-leg variation — gentle weight shifts
        elif hasattr(spinal_cpg, '_babbling_noise'):
            spinal_cpg._babbling_noise = 0.0

        # Terrain reflex: compute corrections + CPG modulation (Phase B)
        terrain_corr = terrain_reflex.compute(sensor_data)
        tr_freq = terrain_reflex.freq_scale
        tr_amp = terrain_reflex.amp_scale
        if _disable_reflexes:
            terrain_corr = np.zeros_like(terrain_corr)
            tr_freq = 1.0
            tr_amp = 1.0

        # VOR steering signal for CPG asymmetric amplitude (Issue #76d)
        # Biology: Reticulospinal projection modulates left/right CPG amplitude.
        # This is the CORRECT way to steer — inside the CPG rhythm, not as
        # an external offset that fights the CPG pattern.
        _cpg_steering = getattr(creature, '_steering_offset', 0.0)

        # Issue #79c: CPG amplitude reduction near ball ("proximity brake")
        # Biology: animals decelerate when approaching a target (optic flow).
        # When ball_dist < 1.0m, scale CPG amplitude down linearly.
        # At 0.3m, amplitude is 30% of normal → dog slows to a stop near ball.
        _proximity_amp_scale = 1.0
        if _scene_has_ball and not is_fallen:
            _bid_prox = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
            if _bid_prox >= 0:
                _cpos_prox = np.array([float(world._data.qpos[0]), float(world._data.qpos[1])])
                _bpos_prox = world._data.xpos[_bid_prox][:2].copy()
                _bdist_prox = float(np.linalg.norm(_cpos_prox - _bpos_prox))
                if _bdist_prox < 1.0:
                    # Linear: 1.0m → 1.0, 0.3m → 0.3, 0.0m → 0.1
                    _proximity_amp_scale = max(0.1, 0.3 + 0.7 * (_bdist_prox / 1.0))

        # Issue #107: Reticulospinal CPG Inhibition (DCN → CPG pathway)
        # =============================================================
        # Biology: The reticulospinal tract (RST) carries DCN output from
        # the cerebellum to the spinal CPG. When the cerebellum detects
        # an obstacle (via climbing fiber from inferior olive), DCN
        # rebound bursts produce STRONG output that INHIBITS the CPG.
        #
        # This replaces the old trainer-side _proximity_amp_scale hack.
        # The old hack externally reduced CPG amplitude based on distance
        # — the brain never learned to brake. Now the DCN output directly
        # modulates CPG amplitude through a biologically real pathway.
        #
        # The DCN correction magnitude naturally increases near obstacles:
        #   - obstacle_cf fires → PkC calcium → PkC inhibits DCN
        #   - calcium decays → PkC releases → DCN REBOUND BURST
        #   - burst magnitude scales with obstacle proximity
        #   - burst → CPG inhibition (this code)
        #
        # Two components:
        #   1. LEARNED: DCN rebound strength → CPG inhibition (trains over time)
        #   2. REFLEX: Brainstem proximity brake (hardwired, immediate safety)
        #
        # The reflex provides a safety floor; the cerebellum learns to
        # anticipate and brake earlier/smoother than the reflex alone.
        #
        # Ref: Drew et al. 2004 — Reticulospinal control of locomotion
        # Ref: Takakusaki 2013 — Brainstem-spinal cord locomotor circuits
        _dcn_cpg_inhibition = 1.0  # 1.0 = no inhibition
        if _scene_has_wall and not is_fallen and hasattr(creature, 'actor_critic') and creature.actor_critic is not None:
            cb = creature.actor_critic
            # Learned pathway: DCN rebound → CPG inhibition
            # DCN rebound_strength is high when CF fired recently and
            # calcium is decaying → the cerebellum "knows" it needs to brake
            _dcn_reb = cb.stats.get('dcn_rebound_strength', 0.0)
            # obstacle_cf tells us IF the obstacle triggered the cerebellum
            _obs_cf = cb.inferior_olive._obstacle_cf
            if _obs_cf > 0.05 and _dcn_reb > 0.01:
                # CPG inhibition proportional to rebound × obstacle_cf
                # Max inhibition: 70% (always keep some movement for recovery)
                _dcn_inhibit = min(0.70, _dcn_reb * _obs_cf * 3.0)
                _dcn_cpg_inhibition = 1.0 - _dcn_inhibit

        # Brainstem obstacle reflexes: HARDWARE-MATCHED
        # ================================================
        # These must be IDENTICAL to freenove_bridge.py so that the brain
        # trains in the same environment it will run on hardware.
        # A dog has the same reflexes in training as in the real world.
        #
        # Three zones (same distances as freenove_bridge.py):
        #   <0.05m (5cm):  REVERSE — CPG runs backward (back up)
        #   <0.10m (10cm): STOP    — CPG killed (full stop)
        #   <0.30m (30cm): SLOW    — graded CPG inhibition
        #   >=0.30m:       CLEAR   — full CPG output
        #
        # v0.5.0: Tightened from 8/15/50cm to 5/10/30cm.
        # The old distances were too conservative — the robot barely
        # reached the wall (1 hit per 10k steps). A puppy needs to
        # BUMP its nose to learn. The reflex prevents DAMAGE, not CONTACT.
        # HC-SR04 is only reliable below ~30cm anyway.
        _REFLEX_REVERSE_M = 0.05
        _REFLEX_STOP_M = 0.10
        _REFLEX_SLOW_M = 0.30
        _reflex_cpg_inhibition = 1.0
        _reflex_reverse = False
        if _scene_has_wall and not is_fallen:
            _obs_dist_brake = sensor_data.get('obstacle_distance', -1.0)
            if _obs_dist_brake >= 0:
                if _obs_dist_brake < _REFLEX_REVERSE_M:
                    # CONTACT: Reverse CPG — back up
                    _reflex_cpg_inhibition = 0.6  # reduced amplitude
                    _reflex_reverse = True
                elif _obs_dist_brake < _REFLEX_STOP_M:
                    # COLLISION IMMINENT: CPG kill — full stop
                    _reflex_cpg_inhibition = 0.0
                elif _obs_dist_brake < _REFLEX_SLOW_M:
                    # DANGER: Graded inhibition — slow proportionally
                    _slow = (_obs_dist_brake - _REFLEX_STOP_M) / (_REFLEX_SLOW_M - _REFLEX_STOP_M)
                    _reflex_cpg_inhibition = max(0.2, _slow)

        # Combined: use whichever is MORE inhibitory (min = more brake)
        # Early training: reflex dominates (DCN hasn't learned yet)
        # Late training: DCN anticipates and brakes before reflex fires
        _combined_inhibition = min(_dcn_cpg_inhibition, _reflex_cpg_inhibition)
        _proximity_amp_scale *= _combined_inhibition

        # Reverse: negate CPG frequency to walk backward
        # (same as freenove_bridge.py set_reverse())
        _cpg_freq_direction = -1.0 if _reflex_reverse else 1.0

        # Issue #108: Obstacle Avoidance Turn Reflex (hardwired)
        # ======================================================
        # Biology: Trigeminal avoidance — when whiskers/nose detect
        # an obstacle, the brainstem produces ASYMMETRIC motor commands:
        # ipsilateral retraction + contralateral extension → animal
        # TURNS AWAY. This is a reflex, not learned behavior.
        # The cerebellum calibrates timing and gain over experience.
        #
        # Implementation: Add steering signal to CPG proportional to
        # obstacle proximity. Direction is fixed (always turn left)
        # because HC-SR04 is a single forward sensor — can't tell
        # left from right. Many insects have fixed turning chirality.
        #
        # Ref: Nguyen & Bhatt 2018 — Trigeminal avoidance circuitry
        # Ref: Dean et al. 1986 — Brainstem avoidance reflexes in rats
        _REFLEX_TURN_GAIN = 0.4  # max steering magnitude
        _reflex_turn_steering = 0.0
        if _scene_has_wall and not is_fallen and not _reflex_reverse:
            _obs_dist_turn = sensor_data.get('obstacle_distance', -1.0)
            if _obs_dist_turn >= 0 and _obs_dist_turn < _REFLEX_SLOW_M:
                # Strength: 0 at 30cm, full at 5cm
                _turn_strength = 1.0 - (_obs_dist_turn / _REFLEX_SLOW_M)
                _reflex_turn_steering = _turn_strength * _REFLEX_TURN_GAIN
        # Add to existing steering (VOR etc.)
        _cpg_steering += _reflex_turn_steering

        # Run-and-Tumble committed turn (#108 RT): during a TUMBLE phase, BACK AWAY
        # WHILE TURNING (kbk + veer). A robot rammed nose-first into the wall cannot
        # pivot in place — the feet scrub but the body does not rotate (66 tumbles,
        # 27% turning, heading still 14deg, pos_x pinned). Reversing first unjams it,
        # and the steering veer rotates the heading as it backs off. This is the dog's
        # actual escape: back up, turn away, then run. RUN stays forward (kwkF), steer 0.
        # The continuous #108 reflex above is dormant under --imu-obstacle (obs ~4.0).
        # The variance feedback removes this commanded yaw from the block read.
        if args.imu_obstacle:
            _cpg_steering += _OB_STEER
            if _OB_STATE == 'TUMBLE':
                _cpg_freq_direction = -1.0          # reverse to unjam from the wall (kbk)

        # v0.4.8: Run-and-Tumble chemotaxis (replaces v0.4.7 continuous steering)
        # Biology: chemotaxis in animals is NOT continuous proportional steering.
        # It is a discrete cycle: SNIFF → TUMBLE → RUN → SNIFF again.
        # Continuous steering caused circling (v0.4.7 failure: dr consistently
        # negative, CPG at 88-90%, actor never got control, sf:0).
        #
        # Run-and-Tumble (Berg & Brown 1972 for bacteria, analogous pattern
        # in C. elegans, star-nosed moles — Catania 2013):
        #   SNIFF: measure smell gradient, compute angle to source
        #   TUMBLE: brief steering impulse to orient toward source (~12 steps)
        #   RUN: straight-line locomotion, no steering (~80 steps)
        #   Then repeat. If smell improved during RUN → extend next RUN.
        #
        # This prevents circling because steering is IMPULSE not CONTINUOUS.
        # The dog sniffs, turns, runs straight, sniffs again.
        _olf_steer_raw = sensor_data.get('olfactory_steering', 0.0)
        _sm_now = sensor_data.get('smell_strength', 0.0)

        # State machine tick
        if _RT_TIMER > 0:
            _RT_TIMER -= 1

        if _RT_STATE == 'RUN':
            # Running straight — no olfactory steering on CPG
            if _RT_TIMER <= 0:
                # RUN phase over → transition to SNIFF
                _RT_STATE = 'SNIFF'
                # (SNIFF is instantaneous — processed in same step)

        if _RT_STATE == 'SNIFF':
            # Measure gradient and decide whether to tumble
            _sm_after_run = _sm_now
            _angle_diff = abs(_olf_steer_raw)  # Already normalized [-1,1] by SensoryEnv

            # Improvement check: did smell get stronger during last RUN?
            if _sm_after_run > _RT_SM_BEFORE + 0.02:
                # Getting closer — extend next RUN (reward straight walking)
                _RT_RUN_DURATION = min(_RT_RUN_DURATION_MAX,
                                       int(_RT_RUN_DURATION * 1.5))
            else:
                # Not improving — reset to base duration
                _RT_RUN_DURATION = _RT_RUN_DURATION_BASE

            # Decision: tumble or skip?
            if _sm_now >= _RT_MIN_SM and _angle_diff > _RT_DEAD_ZONE:
                # Need to turn: compute impulse and enter TUMBLE
                # _olf_steer_raw is in [-1, 1], maps angle_diff/pi * sm
                # Impulse gain 0.5 = half a radian for full error at full sm
                _RT_TUMBLE_IMPULSE = _olf_steer_raw * 0.5
                _RT_STATE = 'TUMBLE'
                _RT_TIMER = _RT_TUMBLE_DURATION
            else:
                # Already aimed correctly or too weak signal → skip tumble, go to RUN
                _RT_TUMBLE_IMPULSE = 0.0
                _RT_STATE = 'RUN'
                _RT_TIMER = _RT_RUN_DURATION
                _RT_SM_BEFORE = _sm_now  # Record for next improvement check

        if _RT_STATE == 'TUMBLE':
            # Apply steering impulse to CPG (constant during tumble phase)
            _cpg_steering += _RT_TUMBLE_IMPULSE
            if _RT_TIMER <= 0:
                # Tumble done → start RUN
                _RT_STATE = 'RUN'
                _RT_TIMER = _RT_RUN_DURATION
                _RT_SM_BEFORE = _sm_now  # Record for improvement check

        # Wall pause: kill CPG completely during "perplex" phase
        # This must be AFTER all other _proximity_amp_scale calculations
        # so it can't be overridden by ball proximity or reflex logic.
        if _wall_pause_counter > 0:
            _proximity_amp_scale = 0.0

        # --- Task #84 step 4: away-from-danger DRIVE (the missing actor) ---------
        # Steps 1-3 established that the robot can FEEL the wall (block_aversion),
        # REMEMBER it (danger landmark + wall_mem) and stop being fascinated by it
        # (learning-progress curiosity) -- and still it stands there. Log #184: the
        # three are not additive. The reason is structural, not a tuning issue:
        # every one of them produces a SIGNAL, and the only thing in the loop that
        # ever turns the body is the Run-and-Tumble SNIFF -- a discrete decision taken
        # once per RUN (~40 steps). Between two SNIFFs the remembered wall has no path
        # to the motor at all. Feeling without an actor is paralysis; that is exactly
        # what the runs show.
        #
        # So: a drive, not a reward. A quiet constant push away from a remembered
        # danger, applied EVERY step, straight into the steering the CPG already takes.
        # Biology: this is not deliberation, it is the same reflexive negative taxis a
        # woodlouse has -- gradient in, turn out, no representation in between.
        #
        # Geometry (spatial_map.direction_to): bearing 0 = straight ahead, +pi/2 = LEFT,
        # -pi/2 = RIGHT. CPG steering convention is the opposite sign (train_baby negates
        # the yaw PID at #2089: positive steering = turn RIGHT). Danger on the LEFT
        # (sin > 0) must therefore produce POSITIVE steering, and vice versa.
        #
        # Head-on (bearing ~ 0, sin -> 0) is an unstable equilibrium: the pure gradient
        # gives no turn precisely when the robot is nose-to-the-wall -- the situation this
        # whole task is about. Break the symmetry the way an animal does: COMMIT to a side
        # and keep it while the danger stays ahead (no per-step re-decision, no dithering).
        # The commitment is released as soon as the danger is behind or out of range.
        #
        # Placement is load-bearing: this must land BEFORE _steer_buf.append() below, so
        # the efference copy subtracts the yaw WE commanded. Applied after that point, the
        # robot's own escape turn would read as unexplained yaw = wall scrub, block_aversion
        # would rise from its own evasion, and it would trap itself in the slow-turn limbo
        # the efference-copy fix removed. Signal stays clean, actor stays free.
        #
        # Flag-gated: weight 0.0 => whole block skipped => bit-identical.
        # SCAFFOLD (same caveat as step 2): the landmark geometry is read from the
        # dead-reckoned map (privileged vel_mps/cur_x). HW-honest IMU odometry is the
        # named follow-up -- the drive itself is HW-portable, only its input is not yet.
        _danger_steer = 0.0
        if _danger_w > 0.0:
            _dgr_s = spatial_map.get_danger_nearby(radius=_danger_r)
            _dir_s = spatial_map.direction_to(_dgr_s.name) if _dgr_s is not None else None
            if _dir_s is not None:
                _rel_s, _dist_s = _dir_s
                _prox_s = max(0.0, 1.0 - _dist_s / max(_danger_r, 1e-6))   # 1 at the landmark -> 0 at radius edge
                _ahead_s = max(0.0, float(np.cos(_rel_s)))                 # 1 straight ahead -> 0 abeam/behind
                if _prox_s > 0.0 and _ahead_s > 0.0:
                    _lat_s = float(np.sin(_rel_s))                          # >0 danger left, <0 danger right
                    if abs(_lat_s) >= _DANGER_HEADON_EPS:
                        # Clear bearing: turn away from the side the danger is on.
                        _danger_side = 1 if _lat_s > 0.0 else -1
                    elif _danger_side == 0:
                        # Head-on and not yet committed: pick a side and stay with it.
                        _danger_side = 1
                    _danger_steer = _danger_w * float(_danger_side) * _prox_s * _ahead_s
                    _danger_steps += 1
                else:
                    _danger_side = 0        # danger abeam/behind/out of range -> release the commitment
            else:
                _danger_side = 0            # nothing remembered nearby
            _cpg_steering += _danger_steer

        # Task #96 Weg 2: intrinsic curiosity steering. Turn toward the least-visited
        # nearby space the SpatialMap knows about, scaled by the CuriosityExplorer
        # drive (bored/high-PE -> turn more) and by how one-sided the novelty is.
        # The robot heads where the world is still unknown; a wall is self-avoiding
        # because nothing behind it ever gets visited. NOT external reward, NOT a
        # hardcoded turn. Uses the previous step's curiosity drive (1-step lag,
        # negligible, same as the other terms). Added before the efference-copy buffer
        # so it reads as commanded steering, not wall scrub. 0.0 -> bit-identical.
        _curiosity_steer = 0.0
        if getattr(args, 'curiosity_steer_weight', 0.0) != 0.0:
            _unexp = spatial_map.direction_to_unexplored(
                radius=float(getattr(args, 'curiosity_steer_radius', 2.0)))
            if _unexp is not None:
                _cur_rel, _cur_nov = _unexp
                _cur_drive = curiosity_explorer.get_exploration_drive()
                # rel_angle -> steering sign: +left/+, -right/-. sin() saturates the
                # command for large angles (no runaway) and gives 0 straight ahead.
                _curiosity_steer = (float(args.curiosity_steer_weight)
                                    * _cur_drive * _cur_nov * float(np.sin(_cur_rel)))
                _curiosity_steer = float(np.clip(_curiosity_steer, -1.0, 1.0))
                _cpg_steering += _curiosity_steer

        # Task #92/#94 DIAGNOSTIC: constant steering injection (measurement scaffold).
        # Added AFTER every reflex/danger term and BEFORE the efference-copy buffer, so
        # the yaw it generates is correctly treated as commanded steering (not wall
        # scrub). Default 0.0 -> bit-identical. Lets analyze_turn_test.py see the real
        # deg/s that gait_blend produces with a known, steady command and no wall.
        if getattr(args, 'steering_constant', 0.0) != 0.0:
            _cpg_steering += float(args.steering_constant)

        # Vestibulospinal reflex (Issue #122): extract yaw rate from IMU
        # sensor_data['angular_velocity'] = [wx, wy, wz] in rad/s
        # wz > 0 = turning left (counterclockwise from above)
        # On hardware: MPU6050 gyro_z provides the same signal.
        _yaw_rate = float(sensor_data.get('angular_velocity', [0, 0, 0])[2])

        # Block aversion (intrinsic, #213/#206): the body's own sense of being
        # stuck, with the robot's OWN behaviour fed back into the variance (Marc).
        # Raw yaw-scrub is the wall signature, BUT commanded turning ALSO makes yaw,
        # so a closed-loop escape (turning away) keeps raw scrub high and traps the
        # robot in slow-turn limbo. Fix (efference copy): over the window, remove the
        # yaw EXPLAINED by the commanded steering; only the UNEXPLAINED yaw (the wall
        # scrubbing the body) counts as block. Self-generated turning becomes
        # invisible to the signal, so the escape can resolve it. Going straight
        # (steering ~const) -> residual == raw scrub, so the block is still detected.
        # Computed every step (cheap, logged); used only via _block_w / --imu-obstacle.
        _block_buf.append(float(_yaw_rate))
        _steer_buf.append(float(_cpg_steering))
        if len(_block_buf) >= 20:
            _y_arr = np.asarray(_block_buf, dtype=float)
            _s_arr = np.asarray(_steer_buf, dtype=float)
            _s_var = float(np.var(_s_arr))
            if _s_var > 1e-6:
                _a_sy = float(np.mean((_s_arr - _s_arr.mean()) * (_y_arr - _y_arr.mean())) / _s_var)
                _resid = _y_arr - _a_sy * _s_arr      # yaw minus the commanded (expected) yaw
            else:
                _resid = _y_arr                       # going straight: nothing commanded to remove
            _yaw_scrub_val = float(np.std(_resid))
            _block_aversion = min(1.0, max(0.0, (_yaw_scrub_val - 0.12) / 0.23))
        else:
            _yaw_scrub_val = 0.0
            _block_aversion = 0.0

        # Task #84 step 2: anticipatory boundary aversion (the dog's boundary-cell memory).
        # When a remembered DANGER landmark (the wall, written ONCE at first contact in step 1)
        # is nearby AND roughly ahead, raise block_aversion BEFORE contact -> (1) the existing
        # Run-and-Tumble SNIFF (--imu-obstacle) crosses _OB_BLOCK_ON earlier and turns away,
        # and (2) vestibular_discomfort rises IF --block-aversion-weight>0 (the intrinsic learn
        # signal). One insertion, both flavours, through the existing aversion+turn machinery --
        # no new hardwired steer. Flag-gated: weight 0.0 => whole block skipped => bit-identical.
        # SCAFFOLD: direction_to uses the dead-reckoned map (privileged vel_mps/cur_x); HW-honest
        # IMU odometry is the named follow-up. The memory only exists AFTER first contact, so the
        # first approach still hits and teaches; later approaches are anticipated.
        _wall_mem_av = 0.0
        if _wall_mem_w > 0.0:
            _dgr = spatial_map.get_danger_nearby(radius=_wall_mem_r)
            if _dgr is not None:
                _dir_dg = spatial_map.direction_to(_dgr.name)
                if _dir_dg is not None:
                    _rel_dg, _dist_dg = _dir_dg
                    _prox = max(0.0, 1.0 - _dist_dg / max(_wall_mem_r, 1e-6))   # 0 at radius edge -> 1 at the landmark
                    _ahead = max(0.0, float(np.cos(_rel_dg)))                   # 1 straight ahead -> 0 to the side/behind
                    _wall_mem_av = _wall_mem_w * _prox * _ahead
                    _block_aversion = min(1.0, _block_aversion + _wall_mem_av)

        # Build CPG kwargs — yaw_rate only for Mogli (SpinalCPG doesn't use it)
        _cpg_kwargs = dict(
            dt=args.timestep, arousal=ne_lvl,
            freq_scale=current_freq_scale * tr_freq * _cpg_freq_direction * directed_learning.freq_scale_modifier,
            amp_scale=current_amp_scale * tr_amp * _proximity_amp_scale * directed_learning.amp_scale_modifier,
        )
        _is_mogli = hasattr(spinal_cpg, '_drift_estimate')
        if _is_mogli:
            _cpg_kwargs['steering'] = _cpg_steering
            _cpg_kwargs['yaw_rate'] = _yaw_rate

        if _is_mogli:
            # Mogli: always use compute() (handles both Freenove and Go2)
            cpg_cmd = spinal_cpg.compute(**_cpg_kwargs)
        elif is_external_mjcf:
            # Go2 with SpinalCPG: direct joint control
            _cpg_kwargs['steering'] = _cpg_steering
            cpg_cmd = spinal_cpg.compute(**_cpg_kwargs)
        else:
            # Freenove/Bommel with SpinalCPG: use compute() with steering
            # v0.5.0: steering via asymmetric stride must work for ALL creatures
            _cpg_kwargs['steering'] = _cpg_steering
            cpg_cmd = spinal_cpg.compute(**_cpg_kwargs)

        # Apply leg damage: zero actuators for disabled leg
        for _dj in _damaged_actuators:
            if _dj < len(cpg_cmd):
                cpg_cmd[_dj] = 0.0

        # Creature-specific CPG output remapping (e.g. Bittle 12->8 with axis inversion)
        # Skip if cpg_cmd is already 8-element (OpenCatGait outputs 8-element deltas directly)
        if (profile and 'cpg_config' in profile and profile.get('n_joints', 12) < 12
                and len(cpg_cmd) > 8):
            from src.body.bittle import cpg_output_to_ctrl, STAND_CTRL
            # cpg_output_to_ctrl extracts HIP+KNEE from 12-element output,
            # remaps leg order, inverts rear shoulders, adds STAND_CTRL.
            # But creature.apply_motor_output adds stand_angles itself,
            # so we return the delta only (without STAND_CTRL).
            _ctrl_8 = cpg_output_to_ctrl(cpg_cmd)  # STAND + delta
            cpg_cmd = _ctrl_8 - STAND_CTRL          # delta only

        gate.update(step, vel_mps, is_fallen, upright=upright)
        cpg_weight = gate.get_cpg_weight()
        creature._cpg_cmd = cpg_cmd
        creature._cpg_weight = cpg_weight
        # Task #92/#94: hand the controller's pure turn-only delta + its weight to
        # apply_motor_output, so the steer can be re-applied undamped. Default weight
        # 0.0 => bit-identical. Only OpenCatController exposes _last_steer_delta.
        creature._steer_delta = getattr(spinal_cpg, '_last_steer_delta', None)
        creature._steer_undamped_weight = float(getattr(args, 'steer_undamped', 0.0))
        creature._reflex_cmd = reflex_cmd
        creature._terrain_corr = terrain_corr  # Phase B terrain reflex corrections
        creature._olfactory_steering = sensor_data.get('olfactory_steering', 0.0)

        # Store CPG phase for hardware sensor encoding (Bridge v2.5 compatibility)
        if args.hardware_sensors:
            if hasattr(spinal_cpg, '_phases'):
                import math
                _phase_rad = float(spinal_cpg._phases[0])
                creature._cpg_phase_input = np.array(
                    [math.sin(_phase_rad), math.cos(_phase_rad)], dtype=np.float32)
            elif hasattr(spinal_cpg, 'get_phase_input'):
                creature._cpg_phase_input = spinal_cpg.get_phase_input()

        _tp2 = time.perf_counter(); _profile['sensory_env'] += _tp2 - _tp1

        # v4.2: Apply hardware drift before physics step
        hardware_drift.apply(creature)

        step_result = creature.step(
            reward_signal=learning_signal,
            extra_sensor_data={
                'smell_strength': sensor_data.get('smell_strength', 0.0),
                'scent_reward': sensor_data.get('scent_reward', 0.0),
                # Navigation context for Synaptogenesis (Module Audit Fix)
                'ball_heading': getattr(creature, '_ball_heading', 0.0),
                'ball_distance': prev_ball_dist if prev_ball_dist is not None else 0.0,
                'steering_offset': getattr(creature, '_steering_offset', 0.0),
                # v0.7.0 Pillar 1-3: Body Awareness + Gait Quality + Spatial Map
                # Cache stats — only recompute every 100 steps
                'gait_quality': getattr(gait_analyzer, '_cached_gq', 0.5),
                # Increment b (#209 fix): buffered IMU coordination -> dampens the exploration
                # drive (coordinated gait = not bored = no need to babble). 0 when weight=0.
                'coordination': (_coord_w * min(1.0, _coord_concentration / 0.1)) if _coord_w > 0.0 else 0.0,
                'gait_periodicity': getattr(gait_analyzer, '_cached_per', 0.0),
                'gait_jitter': getattr(gait_analyzer, '_cached_jit', 0.0),
                'gait_height_ratio': getattr(gait_analyzer, '_cached_hr', 0.5),
                'limb_dead': body_awareness.get_dead_limbs(),
                'limb_degraded': body_awareness.get_degraded_limbs(),
                'ball_salience': getattr(creature, '_ball_salience', 0.0),
                'spatial_x': spatial_map.position[0],
                'spatial_y': spatial_map.position[1],
                'spatial_explored': spatial_map.get_explored_ratio(),
                'spatial_dist_home': spatial_map.direction_to_home()[1],
                # v0.7.0: Vestibular discomfort from sustained rotation
                # Biology: semicircular canals signal sustained rotation.
                # A puppy spinning in circles gets dizzy. This feeds into
                # the emotion system as negative valence, not as a motor
                # correction (that's the vestibulospinal reflex).
                'vestibular_discomfort': min(1.0, (min(1.0, max(0.0, (abs(_yaw_rate) - 0.3) * 2.0)) if abs(_yaw_rate) > 0.3 else 0.0) + _block_w * _block_aversion),
            },
        )

        # === SNN DIAGNOSTICS (temporary) ===
        if step % 500 == 0 and step <= 5000:
            diagnose_snn(creature, step)

        # ================================================================
        # BABY-KI: Compute intrinsic reward for NEXT step's learning signal.
        _tp3 = time.perf_counter(); _profile['creature_step'] += _tp3 - _tp2
        # process() has just populated vestibular_discomfort, curiosity,
        # empowerment, and proprioceptive_delta. Now compute the composite.
        # ================================================================
        if creature.brain:
            creature.brain.get_intrinsic_reward()

        # --- Issue #78: Closed-Loop — record experience + adapt ---
        if _scene_has_ball and sensory_env:
            closed_loop.record(EmbodiedExperience(
                step=step,
                ball_dist=prev_ball_dist if prev_ball_dist is not None else -1.0,
                ball_heading=getattr(creature, '_ball_heading', 0.0),
                upright=upright,
                velocity=vel_mps,
                da_reward=ball_approach_reward,
                steering_offset=getattr(creature, '_steering_offset', 0.0),
                cpg_weight=getattr(gate, 'cpg_weight', 0.9),
                behavior=current_behavior,
                is_fallen=is_fallen,
                prediction_error=getattr(creature.brain, '_prediction_error', 0.0),
            ))
            if step % closed_loop.eval_interval == 0 and step > 0:
                closed_loop.adapt()

        # --- Directed Learning: evaluate hypotheses + generate new ones ---
        if directed_learning.should_evaluate(step):
            _dl_emo = brain_result.get('emotion', {}).get('dominant_emotion', '')
            _dl_gq = gait_analyzer.stats().get('gait_quality', 0.5)
            _dl_result = directed_learning.evaluate_and_adapt(
                step=step,
                gait_quality=_dl_gq,
                upright=upright,
                dead_limbs=body_awareness.get_dead_limbs(),
                degraded_limbs=body_awareness.get_degraded_limbs(),
                emotion=_dl_emo,
                obstacle_hits=wall_episode_count,
                obstacle_distance=sensor_data.get('obstacle_distance', -1.0),
                cpg=spinal_cpg,
            )

        # Memory management: periodic GC to clean up Python object cycles.
        if step % 5000 == 0 and step > 0:
            _gc.collect()
        brain_result = step_result.get('brain', {})
        pci_val = brain_result.get('pci', None)
        if pci_val is not None and pci_val > 0:
            last_pci = pci_val

        # Phase C: CuriosityExplorer — update exploration drive from PE
        _pe_val = getattr(creature.brain, '_prediction_error', 0.0) if hasattr(creature, 'brain') else 0.0
        _grid_cov = 0.0
        if spatial_map:
            _sp_stats = spatial_map.stats()
            _grid_cov = _sp_stats.get('cells_visited', 0) / max(1, _sp_stats.get('grid_size', 400))
        curiosity_explorer.update(_pe_val, grid_coverage=_grid_cov)
        # R-STDP is now handled by CognitiveBrain.process() (step 11)
        # with protected_populations filtering cerebellar synapses.
        # No duplicate apply_rstdp needed here.

        d = creature.get_distance_traveled()
        if d > max_dist:
            max_dist = d

        # Issue #125: Progress-based stuck detection
        # Catches robots that are technically upright but making no forward
        # progress (rolling in place, oscillating, stuck against nothing).
        # The upright-based detector (#110) misses cases where upright > 0.7
        # but the robot walks in place. This detector only checks whether
        # max_dist is growing.
        if auto_reset_limit > 0 and not is_fallen and step > 1000:
            if d <= _progress_last_max_dist:
                _progress_stuck_counter += 1
            else:
                _progress_stuck_counter = 0
                _progress_last_max_dist = d
            if _progress_stuck_counter >= _PROGRESS_STUCK_STEPS:
                if world._model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(world._model, world._data, 0)
                else:
                    world._data.qpos[:] = 0
                    world._data.qvel[:] = 0
                    world._data.qpos[2] = standing_h + 0.02
                mujoco.mj_forward(world._model, world._data)
                _progress_stuck_counter = 0
                _progress_last_max_dist = 0.0  # Reset distance threshold
                consecutive_fallen = 0
                _stuck_counter = 0
                reset_count += 1
                is_fallen = False
                creature._was_fallen = False
                creature._prev_x = float(world._data.qpos[0])
                if reset_count <= 10 or reset_count % 10 == 0:
                    print(f'  [PROGRESS RESET #{reset_count} at step {step} — no dist gain for {_PROGRESS_STUCK_STEPS} steps, up={upright:.2f}]')
                body_awareness.reset_after_physics_reset()  # v0.7.0: prevent false positives
                # Meta-Learning Loop Phase A: record timeout
                episode_analyzer.record_event('timeout', {
                    'smell_strength': sensor_data.get('smell_strength', 0.0),
                    'gait_quality': getattr(gait_analyzer, "_cached_gq", 0.5) if gait_analyzer else 0.0,
                    'heading_error': abs(getattr(creature, '_ball_heading', 0.0)),
                    'steering_offset': getattr(creature, '_steering_offset', 0.0),
                    'upright': upright,
                    'velocity': vel_mps,
                    'cpg_weight': gate.cpg_weight if gate else 0.9,
                    'actor_competence': gate.actor_competence if gate else 0.0,
                    'steps_since_last': step - getattr(creature, '_last_found_step', 0),
                    'cumulative_turn': getattr(creature, '_cumulative_turn', 0.0),
                }, step=step)
        _tp4 = time.perf_counter(); _profile['brain'] += _tp4 - _tp3
        step_dt = time.perf_counter() - t_step
        step_times.append(step_dt)

        # Profiling report every 5000 steps
        if step > 0 and step % 5000 == 0:
            total_prof = sum(_profile.values())
            if total_prof > 0:
                print(f'  [PROFILE step {step}] sensor:{_profile["sensor"]*1000/5000:.1f}ms  '
                      f'sensory:{_profile["sensory_env"]*1000/5000:.1f}ms  '
                      f'creature:{_profile["creature_step"]*1000/5000:.1f}ms  '
                      f'brain:{_profile["brain"]*1000/5000:.1f}ms  '
                      f'total:{total_prof*1000/5000:.1f}ms/step')
            _profile = {k: 0.0 for k in _profile}

        # EpisodeAnalyzer + StrategyAdapter + CuriosityExplorer + HypothesisGenerator
        if step > 0 and step % 10000 == 0:
            # Phase A → Phase B: insights → parameter adjustments
            _new_insights = episode_analyzer.get_new_insights()
            if _new_insights:
                strategy_adapter.process_insights(_new_insights, step=step)
                # Phase D: insights → hypotheses
                _new_hyps = hypothesis_generator.generate_from_insights(_new_insights, step=step)
                for _ins in _new_insights:
                    print(f'  [INSIGHT] {_ins.description} (conf={_ins.confidence:.2f})')
                for _adj in strategy_adapter.get_pending_adjustments():
                    print(f'  [STRATEGY] {_adj.parameter}: {_adj.old_value:.1f} → {_adj.new_value:.1f} ({_adj.reason})')
                for _hyp in hypothesis_generator.get_pending():
                    print(f'  [HYPOTHESIS] {_hyp.description} (param={_hyp.parameter} val={_hyp.value:.2f})')

            # Apply adapted parameters back to RT state machine
            _RT_RUN_DURATION_BASE = strategy_adapter.get_rt_run_duration()
            _RT_TUMBLE_DURATION = strategy_adapter.get_rt_tumble_duration()

            # Phase C: curiosity modulates RT on top of strategy
            _curiosity_mods = curiosity_explorer.get_rt_modulation()
            _RT_RUN_DURATION_BASE = int(_RT_RUN_DURATION_BASE * _curiosity_mods['rt_run_scale'])
            _RT_TUMBLE_DURATION = max(6, int(_RT_TUMBLE_DURATION * _curiosity_mods['rt_tumble_scale']))

            # PID Kp scaling
            if hasattr(creature, '_pd_kp'):
                creature._pd_kp = 0.08 * strategy_adapter.get_pid_kp_scale()

            _ea_sr = episode_analyzer.get_success_rate()
            _ea_n = len(episode_analyzer.events)
            _hg_s = hypothesis_generator.stats()
            _ce_d = curiosity_explorer.get_exploration_drive()
            if _ea_n > 0:
                print(f'  [META-LOOP] events={_ea_n} sr={_ea_sr:.0%} insights={len(episode_analyzer.insights)} adj={len(strategy_adapter.adjustments)} hyp={_hg_s["hypothesis_total"]} confirmed={_hg_s["hypothesis_confirmed"]} explore={_ce_d:.2f}')

        # GC every 2000 steps to prevent PyTorch memory fragmentation
        if step > 0 and step % 2000 == 0:
            _gc.collect()

        if recorder and step % args.record_interval == 0:
            try:
                qpos = world._data.qpos
                extra_creature = dict(step=step, x=float(qpos[0]), y=float(qpos[1]))
                # Store ball position in FLOG for video rendering
                _ball_id_flog = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
                if _ball_id_flog >= 0:
                    bp = world._data.xpos[_ball_id_flog]
                    extra_creature['ball_pos'] = [float(bp[0]), float(bp[1]), float(bp[2])]
                # Per-frame phototaxis fields for renderer minimap.
                # dist_to_light = ground-truth distance from creature to light body
                # intent_yaw_rate = current motor steering command (what the dog *wants*)
                if _lt_body_id >= 0:
                    _lp_flog = world._data.xpos[_lt_body_id][:2]
                    _to_light_flog = _lp_flog - np.array([float(qpos[0]), float(qpos[1])])
                    extra_creature['dist_to_light'] = float(np.linalg.norm(_to_light_flog))
                else:
                    extra_creature['dist_to_light'] = -1.0
                extra_creature['intent_yaw_rate'] = float(getattr(creature, '_steering_offset', 0.0))
                recorder.record_creature(joint_positions=qpos.copy(), joint_velocities=world._data.qvel.copy(),
                    center_of_mass=qpos[:3].copy(), heading=float(qpos[3]), speed=vel_mps, **extra_creature)
            except Exception as e:
                if step < 100:  # Only warn on first few failures
                    print(f'  ⚠ FLOG creature write failed at step {step}: {e}')

        if recorder and step % log_every == 0 and step > 0:
            try:
                # Helper: safe attribute read with default, so missing attrs
                # don't crash the whole stats write. Without this, one stale
                # attribute reference tanks the entire FLOG output.
                def _safe(obj, attr, default=0.0):
                    try:
                        return getattr(obj, attr, default)
                    except Exception:
                        return default
                flog_data = {'phase': 'level15', 'step': step, 'distance': d, 'max_distance': max_dist,
                    'falls': fall_count, 'reward': reward, 'upright': upright,
                    'is_fallen': 1 if is_fallen else 0, 'recoveries': recovery_count,
                    'da_reward': da_signal, 'vel_mps': vel_mps,
                    'actor_competence': gate.actor_competence, 'cpg_weight': cpg_weight,
                    'terrain_type': terrain_cfg.terrain_type, 'terrain_difficulty': terrain_cfg.difficulty,
                    'pci': last_pci, 'consciousness_level': brain_result.get('consciousness_level', 0),
                    # Issue #57: drive loop stats
                    'behavior': current_behavior,
                    'freq_scale': current_freq_scale,
                    'amp_scale': current_amp_scale,
                    'posture_state': reflexes.get_stats().get('posture_state', ''),
                    'reflex_active': reflexes.get_stats().get('active_reflexes', ''),
                    'reflex_magnitude': reflexes.get_stats().get('reflex_magnitude', 0.0),
                    'tone_magnitude': spinal_segments.get_stats().get('tone_magnitude', 0.0),
                    'stretch_magnitude': spinal_segments.get_stats().get('stretch_magnitude', 0.0),
                    'golgi_clipped': spinal_segments.get_stats().get('golgi_clipped', 0),
                    # Terrain reflex (Phase B ATR)
                    'terrain_reflex_mag': terrain_reflex.stats.get('terrain_reflex_mag', 0.0),
                    'terrain_pitch_ema': terrain_reflex.stats.get('terrain_pitch_ema', 0.0),
                    'terrain_roll_ema': terrain_reflex.stats.get('terrain_roll_ema', 0.0),
                    'terrain_freq_scale': terrain_reflex.stats.get('terrain_freq_scale', 1.0),
                    'terrain_amp_scale': terrain_reflex.stats.get('terrain_amp_scale', 1.0),
                    # Foot contacts (Phase A ATR)
                    'foot_contact_count': int(foot_sensor.contacts.sum()),
                    'foot_FL': bool(foot_sensor.contacts[0]),
                    'foot_FR': bool(foot_sensor.contacts[1]),
                    'foot_RL': bool(foot_sensor.contacts[2]),
                    'foot_RR': bool(foot_sensor.contacts[3]),
                }
                if cb:
                    s = cb.get_stats()
                    flog_data.update({'grc_sparseness': s.get('grc_sparseness', 0.0),
                        'cf_magnitude': s.get('cf_magnitude', 0.0),
                        'pf_pkc_weight': s.get('pf_pkc_mean_weight', 0.0),
                        'correction_mag': s.get('correction_magnitude', 0.0),
                        'dcn_activity': s.get('dcn_activity', 0.0),
                        'snn_mix': cb.get_snn_mix(), 'pkc_calcium': s.get('pkc_calcium', 0.0)})
                    # Forward model stats
                    fm = cb.inferior_olive.get_forward_model_stats()
                    flog_data.update({
                        'pred_error': fm['prediction_error'],
                        'terrain_error': fm['terrain_error'],
                        'vestibular_error': fm.get('vestibular_error', 0.0),
                        'forward_gain_mean': fm['forward_gain_mean'],
                    })
                # Developmental schedule stats
                dev_stats = dev_schedule.get_stats()
                flog_data['dev_phase'] = dev_stats['developmental_phase']
                flog_data['dev_perturb'] = dev_stats['perturb_magnitude']
                flog_data['dev_fm_gain'] = dev_stats['forward_model_gain']
                flog_data['dev_competence'] = dev_schedule._competence_ema
                # Issue #122: Vestibulospinal reflex stats
                # v0.3.1 cycle-integrator writes drift_estimate + vestibular_cycles.
                # Any missing attribute simply defaults to 0.0 (no crash).
                flog_data['yaw_rate'] = _yaw_rate
                flog_data['drift_estimate'] = _safe(spinal_cpg, '_drift_estimate', 0.0)
                flog_data['vestibular_correction'] = _safe(spinal_cpg, '_vestibular_correction', 0.0)
                flog_data['vestibular_cycles'] = _safe(spinal_cpg, '_cycles_completed', 0)
                # Stuck detection (Increment A2): real sim label + hardware-able proxy.
                flog_data['stuck'] = 1 if _stuck_truth else 0      # gait commanded but no NET progress over window
                flog_data['progress'] = (0.0 if _progress != _progress else _progress)  # net displacement over window (NaN→0 until full)
                flog_data['stuck_speed'] = _horiz_speed            # instantaneous |xy-vel| (logged for comparison)
                flog_data['accel_dyn'] = _acc_dyn_ema              # accel-only proxy (also on hardware)
                _al = sensor_data.get('linear_acceleration', np.zeros(3))
                flog_data['acc_x'] = float(_al[0]); flog_data['acc_y'] = float(_al[1]); flog_data['acc_z'] = float(_al[2])
                # Efference copy (motor command) + CPG phase driver. The forward
                # model predicts the IMU consequence of the command; without the
                # command logged it cannot be built. Observation-only, bit-identical.
                _mc = sensor_data.get('motor_commands', getattr(creature, '_last_controls', None))
                if _mc is not None:
                    flog_data['motor_cmd'] = [float(v) for v in np.asarray(_mc).ravel()]
                _cp = getattr(creature, '_cpg_phase_input', None)
                if _cp is not None:
                    flog_data['cpg_phase'] = [float(v) for v in np.asarray(_cp).ravel()]
                emo = brain_result.get('emotion', {})
                drv_r = brain_result.get('drives', {})
                flog_data['emotion_dominant'] = emo.get('dominant_emotion', '')
                flog_data['valence'] = emo.get('valence', 0.0)
                flog_data['arousal'] = emo.get('arousal', 0.0)
                # Real neuromodulator levels — no derived/placeholder values.
                # cognitive_brain sets da/5ht/ne/ach in snn.neuromod_levels from
                # somatic markers; process() exposes them as brain_result['neuromod'].
                # DA is already logged as 'da_reward'. Write the real 5HT/NE/ACh when
                # present; if the cognitive path is inactive the key is omitted and the
                # overlay shows its fallback rather than a placeholder.
                _neuromod = brain_result.get('neuromod', {})
                if '5ht' in _neuromod:
                    flog_data['serotonin'] = float(_neuromod['5ht'])
                if 'ne' in _neuromod:
                    flog_data['noradrenaline'] = float(_neuromod['ne'])
                if 'ach' in _neuromod:
                    flog_data['acetylcholine'] = float(_neuromod['ach'])
                flog_data['drive_dominant'] = drv_r.get('dominant', '')
                flog_data['curiosity_reward'] = brain_result.get('curiosity_reward', 0.0)
                # Baby-KI intrinsic reward components
                # intrinsic_reward must come from the SAME compute as ir_* below:
                # get_intrinsic_reward() (line ~2961) sets creature.brain._intrinsic_reward
                # AND _intrinsic_components in one pass. brain_result['intrinsic_reward']
                # is process()'s return = the PREVIOUS step's value (verified: R[t]==Σir[t-1],
                # 100% of frames), so logging it here lagged R by one step vs its own
                # decomposition. Log the post-get_intrinsic_reward value so R == Σ ir_* (#227).
                flog_data['intrinsic_reward'] = float(getattr(creature.brain, '_intrinsic_reward', 0.0)) \
                    if creature.brain else brain_result.get('intrinsic_reward', 0.0)
                # Real intrinsic-reward decomposition: signed, WEIGHTED contributions
                # from cognitive_brain.get_intrinsic_reward() that SUM to intrinsic_reward
                # (#227). The raw cur/vest/prop signals above never summed to R; these do.
                _irc = getattr(creature.brain, '_intrinsic_components', None) if creature.brain else None
                if _irc:
                    flog_data['ir_vestibular']  = float(_irc.get('vestibular', 0.0))
                    flog_data['ir_curiosity']   = float(_irc.get('curiosity', 0.0))
                    flog_data['ir_empowerment'] = float(_irc.get('empowerment', 0.0))
                    flog_data['ir_proprio']     = float(_irc.get('proprio', 0.0))
                    flog_data['ir_scent']       = float(_irc.get('scent', 0.0))
                flog_data['vestibular_discomfort'] = brain_result.get('vestibular_discomfort', 0.0)
                flog_data['block_aversion'] = _block_aversion      # intrinsic stuck-aversion (#213/#206)
                flog_data['yaw_scrub'] = _yaw_scrub_val            # rolling yaw std (the wall signature)
                flog_data['wall_mem_aversion'] = _wall_mem_av      # Task #84 step 2: anticipatory boundary-aversion contribution (0.0 when flag off)
                flog_data['ob_steer'] = float(_OB_STEER)           # committed RT turn applied (0 in RUN, gain in TUMBLE)
                flog_data['danger_steer'] = float(_danger_steer)   # Task #84 step 4: continuous away-from-danger drive applied to steering (0.0 when flag off)
                flog_data['danger_side'] = int(_danger_side)       # Task #84 step 4: committed evasion side (+1 right, -1 left, 0 not committed)
                # Task #84 step 4 diagnosis: steering in a trot works ONLY through asymmetric
                # stride length -- no step, no turn. If the brake reflex has zeroed the gait
                # amplitude, every steering command (drive AND tumble) is applied to a gait that
                # is not running. These two fields make that visible instead of inferred.
                flog_data['amp_scale'] = float(current_amp_scale * tr_amp * _proximity_amp_scale * directed_learning.amp_scale_modifier)
                flog_data['prox_amp_scale'] = float(_proximity_amp_scale)   # 0.0 = brake reflex killed the CPG
                # Task #92: the FINAL steering value handed to the controller, after every
                # contribution has been summed.  Without this the individual terms
                # (danger_steer, ob_steer, reflex_turn, the VOR's steering_offset) can each
                # look correct while cancelling each other out -- which is exactly what a
                # 15x gap between the isolated bench (6.65 deg/s) and the training loop
                # (0.45 deg/s) looks like.  Log the sum, not just the parts.
                flog_data['cpg_steering'] = float(_cpg_steering)
                flog_data['reflex_turn_steering'] = float(_reflex_turn_steering)
                flog_data['curiosity_steer'] = float(_curiosity_steer)   # Task #96 Weg 2 intrinsic curiosity turn
                # Task #94: a turn IS a left/right asymmetry in the joint commands.
                # Layout is [RF_sh, RF_kn, LF_sh, LF_kn, RR_sh, RR_kn, LR_sh, LR_kn],
                # so right = indices 0,1,4,5 and left = 2,3,6,7.  Log the asymmetry at
                # three points in the chain: what the CPG asks for, what the SNN adds,
                # and what survives the mix.  If the SNN's asymmetry is the NEGATIVE of
                # the CPG's, the brain is steering back against its own body -- which is
                # the only remaining explanation for a gait that matches the damped chain
                # while the turn does not.
                def _asym(v):
                    if v is None or len(v) < 8:
                        return 0.0
                    right = (float(v[0]) + float(v[1]) + float(v[4]) + float(v[5])) / 4.0
                    left = (float(v[2]) + float(v[3]) + float(v[6]) + float(v[7])) / 4.0
                    return right - left
                flog_data['asym_cpg'] = _asym(getattr(creature, '_cpg_cmd', None))
                flog_data['asym_snn'] = _asym(getattr(creature, '_dbg_snn_controls', None))
                flog_data['asym_mixed'] = _asym(getattr(creature, '_dbg_mixed_controls', None))
                flog_data['ob_tumbles'] = int(_ob_tumbles)         # cumulative tumble count (#108 RT)
                flog_data['proprio_delta'] = brain_result.get('proprioceptive_delta', 0.0)
                flog_data['body_anomaly_ema'] = brain_result.get('body_anomaly_ema', 0.0)
                flog_data['learning_signal'] = learning_signal
                # Substrat-Gesundheit (Logbuch #214, 25.07.2026). Feuerraten wurden
                # bisher NIRGENDS geloggt -- ein Netz mit 30-74 % stummen Neuronen
                # blieb ueber saemtliche Laeufe unsichtbar, waehrend am Lernsignal
                # gearbeitet wurde. R-STDP braucht Koinzidenz, Koinzidenz braucht
                # Spikes. snn_rate_window mitloggen: das Zaehlfenster wird nach jedem
                # Homoeostase-Intervall genullt, bei kleinem Fenster sind die Raten
                # Rauschen und duerfen nicht ausgewertet werden.
                try:
                    flog_data.update(creature.snn.get_health())
                except AttributeError:
                    pass   # aeltere SNN-Version ohne get_health()
                flog_data['reward_blend'] = _blend
                if _coord_w > 0.0:
                    flog_data['coord_conc'] = _coord_concentration
                # Issue #75: Sensory environment
                if sensory_env or visual_env:
                    flog_data['smell_strength'] = sensor_data.get('smell_strength', 0.0)
                    flog_data['smell_direction'] = sensor_data.get('smell_direction', 0.0)
                    flog_data['sound_intensity'] = sensor_data.get('sound_intensity', 0.0)
                    flog_data['sound_direction'] = sensor_data.get('sound_direction', 0.0)
                    flog_data['scents_found'] = visual_env.lights_found if visual_env else sensory_env.scents_found
                    flog_data['olfactory_steering'] = sensor_data.get('olfactory_steering', 0.0)
                    # Episode Analyzer stats (Meta-Learning Loop Phase A)
                    flog_data.update(episode_analyzer.stats())
                    # Strategy Adapter stats (Meta-Learning Loop Phase B)
                    flog_data.update(strategy_adapter.stats())
                    # Curiosity Explorer stats (Meta-Learning Loop Phase C)
                    flog_data.update(curiosity_explorer.stats())
                    # Hypothesis Generator stats (Meta-Learning Loop Phase D)
                    flog_data.update(hypothesis_generator.stats())
                    # Run-and-Tumble state machine (v0.4.8)
                    flog_data['rt_state'] = _RT_STATE
                    flog_data['rt_timer'] = _RT_TIMER
                    flog_data['rt_run_duration'] = _RT_RUN_DURATION
                    flog_data['rt_tumble_impulse'] = _RT_TUMBLE_IMPULSE
                    flog_data['rt_sm_before'] = _RT_SM_BEFORE
                    flog_data['ball_approach_reward'] = ball_approach_reward
                    flog_data['ball_dist'] = prev_ball_dist if prev_ball_dist is not None else -1.0
                    flog_data['ball_heading'] = getattr(creature, '_ball_heading', 0.0)
                    flog_data['ball_salience'] = getattr(creature, '_ball_salience', 0.0)
                    flog_data['steering_offset'] = getattr(creature, '_steering_offset', 0.0)
                    vor_stats = vor.get_stats()
                    flog_data['vor_raw'] = vor_stats.get('vor_raw', 0.0)
                    flog_data['vor_smoothed'] = vor_stats.get('vor_smoothed', 0.0)
                    # Closed-Loop stats (Issue #78)
                    cl_stats = closed_loop.get_stats()
                    flog_data['ball_episode'] = getattr(main, '_ball_ep', 0)
                    flog_data['task_pe'] = getattr(creature.brain, '_task_prediction_error', 0.0)
                    flog_data['cl_adaptations'] = cl_stats.get('cl_adaptations', 0)
                    flog_data['cl_best_ball_dist'] = cl_stats.get('cl_best_ball_dist', -1.0)
                    flog_data['cl_consec_improve'] = cl_stats.get('cl_consec_improve', 0)
                    flog_data['cl_consec_fail'] = cl_stats.get('cl_consec_fail', 0)
                    flog_data['cl_vor_hip_gain'] = cl_stats.get('cl_vor_hip_gain', 0.0)
                    # Cerebellar navigation stats (Issue #81)
                    if cb:
                        fm_stats = cb.inferior_olive.get_forward_model_stats()
                        flog_data['nav_cf'] = fm_stats.get('navigation_cf', 0.0)
                        flog_data['cb_steer_correction'] = fm_stats.get('steering_gain_correction', 0.0)
                        flog_data['cb_heading_gain'] = fm_stats.get('heading_gain', 0.0)
                    # Scent/light source positions for video rendering
                    if sensory_env:
                        for si, sc in enumerate(sensory_env._scents):
                            flog_data[f'scent_{si}_x'] = float(sc.position[0])
                            flog_data[f'scent_{si}_y'] = float(sc.position[1])
                    elif visual_env:
                        for si, lp in enumerate(visual_env.get_light_positions()):
                            flog_data[f'scent_{si}_x'] = float(lp[0])
                            flog_data[f'scent_{si}_y'] = float(lp[1])
                # Issue #121: Extended sensor data for analysis
                # Obstacle distance (HC-SR04 / MuJoCo rangefinder)
                flog_data['obstacle_distance'] = float(sensor_data.get('obstacle_distance', -1.0))
                # IMU: yaw as separate float (angular_velocity Z-axis)
                _ang_vel = sensor_data.get('angular_velocity', np.zeros(3))
                flog_data['pitch_rate'] = float(_ang_vel[1])
                flog_data['roll_rate'] = float(_ang_vel[0])
                # Orientation (for heading analysis)
                _orient = sensor_data.get('orientation_euler', np.zeros(3))
                flog_data['yaw'] = float(_orient[2]) if len(_orient) > 2 else 0.0
                flog_data['pitch'] = float(_orient[1]) if len(_orient) > 1 else 0.0
                flog_data['roll'] = float(_orient[0]) if len(_orient) > 0 else 0.0
                # Y position (for drift/circle detection)
                flog_data['y'] = float(world._data.qpos[1])
                # ============================================================
                # Phototaxis navigation logging
                # ============================================================
                # Ground truth (from MuJoCo physics — what really happened):
                #   pos_x, pos_y         — creature world position (canonical)
                #   dist_to_light        — true distance to light body
                #   heading_to_light     — angle from creature to light (world rad)
                #   intent_yaw_rate      — current motor steering command
                #
                # Brain map (from SpatialMap — what the dog *believes*):
                #   brain_pos_x/y        — dead-reckoned position
                #   brain_pos_error      — drift between belief and truth
                #   brain_landmarks_json — known landmarks with confidence/valence
                #   brain_visit_grid_b64 — uint8-quantized 20x20 visit heatmap
                #
                # On hardware no ground truth exists — only the brain map.
                # Renderer uses ground truth for the outer minimap, brain
                # map for the inner "what the dog knows" view.
                flog_data['pos_x'] = float(world._data.qpos[0])
                flog_data['pos_y'] = float(world._data.qpos[1])
                if _lt_body_id >= 0:
                    _lp_train = world._data.xpos[_lt_body_id][:2]
                    _to_light_train = _lp_train - np.array([flog_data['pos_x'], flog_data['pos_y']])
                    flog_data['dist_to_light'] = float(np.linalg.norm(_to_light_train))
                    flog_data['heading_to_light'] = float(np.arctan2(_to_light_train[1], _to_light_train[0]))
                else:
                    flog_data['dist_to_light'] = -1.0
                    flog_data['heading_to_light'] = -999.0
                flog_data['intent_yaw_rate'] = float(getattr(creature, '_steering_offset', 0.0))
                # Brain-map snapshot — what the dog believes about the world.
                # Cheap to write at log_every interval; landmarks ~1KB,
                # visit grid ~540 bytes (uint8 + base64). Total ~1.5KB/snapshot.
                try:
                    _bx, _by = float(spatial_map.position[0]), float(spatial_map.position[1])
                    flog_data['brain_pos_x'] = _bx
                    flog_data['brain_pos_y'] = _by
                    flog_data['brain_pos_error'] = float(np.sqrt(
                        (_bx - flog_data['pos_x'])**2 + (_by - flog_data['pos_y'])**2))
                    _lm_payload = []
                    for _lm_name, _lm in spatial_map.landmarks.items():
                        if _lm.confidence < 0.05:
                            continue
                        _lm_payload.append({
                            'name': _lm_name,
                            'x': float(_lm.position[0]),
                            'y': float(_lm.position[1]),
                            'cat': _lm.category,
                            'conf': round(float(_lm.confidence), 3),
                            'val': round(float(_lm.valence), 3),
                            'visits': int(_lm.visit_count),
                            'last_seen': int(_lm.last_seen_step),
                        })
                    flog_data['brain_landmarks_json'] = json.dumps(_lm_payload, separators=(',', ':'))
                    # Quantize visit grid to uint8 (clamp >=255 visits to 255).
                    # Most cells visited <50 times so quantization loss is minor.
                    _vg = spatial_map.visit_grid
                    _vg_u8 = np.clip(_vg, 0, 255).astype(np.uint8)
                    flog_data['brain_visit_grid_b64'] = base64.b64encode(_vg_u8.tobytes()).decode('ascii')
                    flog_data['brain_grid_shape'] = list(_vg_u8.shape)
                except Exception as _bm_e:
                    # Never crash the FLOG write because of brain-map serialization.
                    if not hasattr(main, '_brain_map_warn_shown'):
                        print(f'  \u26a0 Brain-map serialize failed at step {step}: {_bm_e}')
                        main._brain_map_warn_shown = True
                # Gait quality metrics (v0.7.0 Pillar 3)
                gq = gait_analyzer.stats()
                for gk, gv in gq.items():
                    flog_data[gk] = gv
                # Body awareness (v0.7.0 Pillar 1)
                ba = body_awareness.stats()
                for bk, bv in ba.items():
                    flog_data[bk] = bv
                # Spatial map (v0.7.0 Pillar 2)
                sp = spatial_map.stats()
                for sk, sv in sp.items():
                    flog_data[sk] = sv
                # Directed Learning (v0.7.0 Pillar 5)
                dl = directed_learning.stats()
                for dk, dv in dl.items():
                    flog_data[dk] = dv
                # Motor Hidden spikes (v0.7.0 motorcortex activity)
                _mh_pop = creature.snn.populations.get('motor_hidden', [])
                if len(_mh_pop) > 0 and hasattr(creature, '_accumulated_spikes'):
                    _mh_spikes = creature._accumulated_spikes[_mh_pop].sum().item()
                    flog_data['mh_spike_count'] = _mh_spikes
                    flog_data['mh_spike_rate'] = _mh_spikes / max(len(_mh_pop), 1)
                    flog_data['mh_n_neurons'] = len(_mh_pop)
                else:
                    flog_data['mh_spike_count'] = 0
                    flog_data['mh_spike_rate'] = 0.0
                    flog_data['mh_n_neurons'] = 0
                # Log the real per-neuron spike raster so the Brain3D shows genuine
                # activity, never a fabricated one. Ordered by population to match the
                # brain_3d layout (input → granule → golgi → purkinje → dcn →
                # motor_hidden → output). If unavailable the key is omitted → Brain3D
                # shows no activity (empty), never random.
                try:
                    if hasattr(creature, '_accumulated_spikes'):
                        _acc = creature._accumulated_spikes
                        if hasattr(_acc, 'detach'):
                            _acc = _acc.detach().cpu().numpy()
                        _acc = np.asarray(_acc).reshape(-1)
                        _pops = creature.snn.populations
                        _order = ['input', 'granule_cells', 'golgi_cells',
                                  'purkinje_cells', 'dcn', 'motor_hidden', 'output']
                        _parts = []
                        for _pn in _order:
                            _pop = _pops.get(_pn, [])
                            _idx = np.asarray(list(_pop), dtype=int) if len(_pop) else None
                            if _idx is not None and _idx.size:
                                _parts.append((_acc[_idx] > 0).astype(int))
                        if _parts:
                            # Key 'spike_raster' (NOT 'spikes') on purpose: 'spikes'
                            # collides with record_training's spikes= param, which
                            # down-samples to 200 and breaks the population layout.
                            flog_data['spike_raster'] = np.concatenate(_parts).tolist()
                except Exception:
                    pass
                # Mogli Oscillator stats (per-leg phase, firing rates, coupling)
                if hasattr(spinal_cpg, 'oscillators'):
                    mogli_stats = spinal_cpg.get_stats()
                    for mk, mv in mogli_stats.items():
                        flog_data[f'mogli_{mk}'] = float(mv)
                recorder.record_training_stats(flog_data)
                if _dash_push is not None:
                    try:
                        _dash_push(flog_data)
                    except Exception:
                        pass  # never let the live push affect the run
            except Exception as e:
                # Show traceback ONCE so we can diagnose persistent failures;
                # subsequent failures show only the message to avoid log spam.
                if not hasattr(main, '_flog_stats_tb_shown'):
                    import traceback
                    print(f'  ⚠ FLOG stats write failed at step {step}: {e}')
                    print('  Full traceback (shown once):')
                    traceback.print_exc()
                    main._flog_stats_tb_shown = True
                else:
                    print(f'  ⚠ FLOG stats write failed at step {step}: {e}')

        if step > 0 and step % log_every == 0:
            avg_ms = np.mean(list(step_times)[-log_every:]) * 1000
            eta_min = (total_steps - step) * (avg_ms / 1000) / 60
            line1 = (f'  {step:>7,}/{total_steps:,}  dist:{max_dist:>5.2f}m  x:{cur_x:.2f}'
                     f'  vel:{vel_mps:.3f}m/s'
                     f'  up:{upright:.2f}  F:{"Y" if is_fallen else "N"}  falls:{fall_count}  rec:{recovery_count}')
            if cb:
                s = cb.get_stats()
                line1 += f'  w:{s["pf_pkc_mean_weight"]:.4f}  mix:{cb.get_snn_mix():.0%}'
            pci_marker = 'Y' if last_pci > 0.31 else '.'
            line1 += f'  PCI:{last_pci:.3f}{pci_marker}  {avg_ms:.1f}ms  ETA:{eta_min:.1f}m'
            print(line1)

            emo = brain_result.get('emotion', {})
            drv_r = brain_result.get('drives', {})
            line2 = (f'           emo:{emo.get("dominant_emotion", "?")[:4]}  V:{emo.get("valence", 0):.2f}'
                     f'  drv:{drv_r.get("dominant", "?")[:4]}  cur:{brain_result.get("curiosity_reward", 0):.3f}'
                     f'  emp:{brain_result.get("empowerment", 0):.3f}  CL:{brain_result.get("consciousness_level", 0)}')
            print(line2)

            cb_stats = cb.stats if cb else {}
            gs = gate.get_stats()
            rs = reflexes.get_stats()
            # Issue #57: show behavior + freq/amp scale in log
            beh_tag = current_behavior[:6] if current_behavior else '?'
            posture = rs.get('posture_state', '?')[:5]
            ss = spinal_segments.get_stats()
            line3 = (f'           beh:{beh_tag:<6s}  fq:{current_freq_scale:.2f}  am:{current_amp_scale:.2f}'
                     f'  rfx:{rs["active_reflexes"][:10]:<10s}  pos:{posture:<5s}  cpg:{cpg_weight:.0%}'
                     f'  act:{gs["actor_competence"]:.3f}  DA:{da_signal:.2f}'
                     f'  uEMA:{gs.get("upright_ema", 0):.2f}  fR:{gs.get("fall_rate", 0):.1f}  vE:{gs.get("vel_ema", 0):.4f}  stb:{"Y" if gs.get("is_stable", False) else "N"}'
                     f'  CF:{cb_stats.get("cf_magnitude", 0.0):.3f}'
                     f'  corr:{cb_stats.get("correction_magnitude", 0.0):.4f}'
                     f'  reb:{cb_stats.get("dcn_rebound_strength", 0.0):.3f}'
                     f'  terr:{terrain_cfg.difficulty:.2f}'
                     f'  TR:{terrain_reflex.stats["terrain_reflex_mag"]:.3f}'
                     f'  ft:{int(foot_sensor.contacts.sum())}')
            if _scene_has_wall:
                line3 += f'  od:{_obs_dist:.2f}'
                _rfx_state = ('REV' if _reflex_reverse else
                              'STOP' if _reflex_cpg_inhibition < 0.01 else
                              f'SLO{_reflex_cpg_inhibition:.0%}' if _reflex_cpg_inhibition < 1.0 else '')
                line3 += f'  inh:{_combined_inhibition:.2f} {_rfx_state}'
                if _reflex_turn_steering > 0.01:
                    line3 += f' T:{_reflex_turn_steering:.2f}'
            # Issue #122: Vestibulospinal reflex
            if hasattr(spinal_cpg, '_drift_estimate'):
                line3 += f'  yr:{_yaw_rate:+.3f}  dr:{spinal_cpg._drift_estimate:+.3f}  vc:{spinal_cpg._vestibular_correction:+.2f}'
            elif hasattr(spinal_cpg, '_yaw_rate_ema'):
                line3 += f'  yr:{spinal_cpg._yaw_rate_ema:+.3f}  vc:{spinal_cpg._vestibular_correction:+.2f}'
            # Gait quality (v0.7.0)
            gq = gait_analyzer.stats()
            line3 += f'  GQ:{gq.get("gait_quality", 0):.2f}  per:{gq.get("gait_periodicity", 0):.2f}  jit:{gq.get("gait_jitter", 0):.3f}  hR:{gq.get("gait_height_ratio", 0):.2f}'
            # Body awareness
            _dead = body_awareness.get_dead_limbs()
            if _dead:
                line3 += f'  DEAD:{",".join(_dead)}'
            # Issue #75: sensory info
            if sensory_env or visual_env:
                sm = sensor_data.get('smell_strength', 0.0)
                sf = visual_env.lights_found if visual_env else sensory_env.scents_found
                line3 += f'  sm:{sm:.2f}  sf:{sf}  RT:{_RT_STATE[0]}({_RT_TIMER})'
                bh = getattr(creature, '_ball_heading', 0.0)
                bs = getattr(creature, '_ball_salience', 0.0)
                so = getattr(creature, '_steering_offset', 0.0)
                bd = prev_ball_dist if prev_ball_dist is not None else -1.0
                vr = vor.stats.get('vor_raw', 0.0) if 'vor' in dir() else 0.0
                vs = so  # v0.5.0: show actual steering offset (PD or VOR)
                _cb_sc = cb.inferior_olive._steering_gain_correction if cb else 0.0
                _tpe = getattr(creature.brain, '_task_prediction_error', 0.0) if hasattr(creature, 'brain') else 0.0
                line3 += f'  bh:{bh:+.2f}  bd:{bd:.1f}  VOR:{vs:+.2f}  CB:{_cb_sc:+.2f}  TPE:{_tpe:+.2f}'
            print(line3)

    total_time = time.perf_counter() - t_start
    avg_ms = total_time / max(1, total_steps - start_step) * 1000
    print(f'\n{"="*65}')
    print(f'  Level 15 Training Complete')
    print(f'{"="*65}')
    print(f'  Scene: "{args.scene}"')
    print(f'  Knowledge: {knowledge["source"]} ({len(knowledge["behaviors"])} behaviors)')
    print(f'  Terrain: {terrain_cfg.terrain_type} (diff={terrain_cfg.difficulty:.2f})')
    print(f'  Steps: {total_steps:,}  Time: {total_time/60:.1f}m')
    print(f'  Speed: {avg_ms:.2f}ms/step ({1000/avg_ms:.0f} sps)')
    print(f'  Max distance: {max_dist:.3f}m')
    print(f'  Falls: {fall_count:,}  Recoveries: {recovery_count}  Resets: {reset_count}')
    if _scene_has_wall:
        print(f'  Wall hits: {wall_episode_count}  (episodic resets)')
    print(f'  Best upright streak: {best_upright_streak}')
    gs = gate.get_stats()
    print(f'  Actor competence: {gs["actor_competence"]:.3f}')
    print(f'  Final CPG weight: {gs["cpg_weight"]:.0%}')
    pci_status = 'ABOVE' if last_pci > 0.31 else 'BELOW'
    print(f'  Final PCI: {last_pci:.4f} ({pci_status})')
    if cb:
        s = cb.get_stats()
        print(f'  Final PF->PkC weight: {s["pf_pkc_mean_weight"]:.4f}')
        print(f'  Final correction: {s["correction_magnitude"]:.4f}')
    if drive_bridge:
        ds = drive_bridge.get_state()
        print(f'  Final behavior: {ds["behavior"]} (freq={ds["freq_scale"]:.2f} amp={ds["amp_scale"]:.2f})')
        print(f'  Behavior history: {" → ".join(ds["behavior_history"][-8:])}')
    print(f'{"="*65}')

    # Close FLOG recorder (flush all buffered data to disk)
    if recorder:
        recorder.close()
        flog_size = os.path.getsize(flog_path) / 1024 if os.path.exists(flog_path) else 0
        print(f'  FLOG closed: {flog_path} ({recorder.frame_count} frames, {flog_size:.0f} KB)')

    # Save checkpoints into the run directory (creatures/{name}/{run_id}/)
    ckpt_dir = creature_dir if creature_dir else f'creatures/{args.creature_name.lower()}'
    os.makedirs(ckpt_dir, exist_ok=True)
    snn_file = os.path.join(ckpt_dir, 'snn_state.pt')
    creature.snn.save(snn_file)
    print(f'  SNN saved: {snn_file}')

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pt')
    ckpt_data = {
        'snn_file': os.path.basename(snn_file), 'step': total_steps, 'max_dist': max_dist,
        'falls': fall_count, 'recoveries': recovery_count,
        'best_upright_streak': best_upright_streak, 'pci': last_pci,
        'actor_competence': gate.actor_competence, 'cpg_weight': gate.cpg_weight,
        'vel_ema': gate.vel_ema,
        'cpg_phases': spinal_cpg._phases.tolist() if hasattr(spinal_cpg, '_phases') else [],
        'cpg_step': spinal_cpg._step,
        'cpg_type': 'opencat' if _use_opencat_gait else ('mogli' if getattr(args, 'neural_cpg', False) else 'spinal'),
        'version': 'v0.8.1', 'scene': args.scene, 'seed': args.seed,
        'terrain_type': terrain_cfg.terrain_type, 'terrain_difficulty': terrain_cfg.difficulty,
        'flog_path': flog_path,
        'flog_frames': recorder.frame_count if recorder else 0,
    }
    if cb:
        ckpt_data['cerebellum_state'] = cb.state_dict()
    # v4.2: Spatial Map persistence
    ckpt_data['spatial_map'] = spatial_map.state_dict()
    torch.save(ckpt_data, ckpt_path)
    print(f'  Checkpoint: {ckpt_path}')
    print(f'  Resume: python scripts/train_v032.py --resume {ckpt_path} --steps 100000')

    # === BRAIN PERSISTENCE: Save COMPLETE cognitive state ===
    # This includes everything the creature has learned:
    #   SNN weights, World Model, Episodic Memory, Concept Graph,
    #   Emotional Markers, Body Schema, Skills, Dream Replay Buffer.
    # Without this, the creature starts fresh every run — no transfer
    # between tasks, no long-term memory, no accumulated knowledge.
    # With this, a dog that learned "ball right → steer right" keeps
    # that knowledge when it later learns to climb stairs.
    if hasattr(creature, 'brain') and creature.brain:
        from src.brain.brain_persistence import save_brain
        import shutil
        # Brain lives in a persistent directory per creature (git-tracked)
        # NOT in the run directory (ephemeral, not in git)
        creature_base = os.path.join('creatures', args.creature_name.lower())
        brain_dir = os.path.join(creature_base, 'brain')
        os.makedirs(brain_dir, exist_ok=True)
        brain_file = os.path.join(brain_dir, 'brain.pt')
        brain_meta = {
            'creature': args.creature_name,
            'scene': args.scene,
            'steps': total_steps,
            'falls': fall_count,
            'max_distance': max_dist,
            'seed': args.seed,
            'ball_episodes': getattr(main, '_ball_ep', 0),
            'run_id': os.path.basename(ckpt_dir),
        }
        # Issue #159: persist the cerebellum INTO brain.pt (the transfer
        # artifact), not only into checkpoint.pt (which is --resume-only).
        # cb is None if --no-cerebellum; save_brain handles None safely.
        save_brain(creature.brain, creature.snn, brain_file, metadata=brain_meta,
                   cerebellum=cb)
        print(f'  Brain saved: {brain_file}'
              f'{" (incl. cerebellum)" if cb else " (no cerebellum)"}')
        # History snapshot (timestamped copy for comparison)
        history_dir = os.path.join(brain_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        scene_tag = args.scene.replace(' ', '_')[:20] if args.scene else 'unknown'
        history_file = os.path.join(history_dir, f'brain_{ts}_{total_steps//1000}k_{scene_tag}.pt')
        shutil.copy2(brain_file, history_file)
        print(f'  Brain history: {history_file}')
        # Append to brain_log.jsonl (provenance tracking for Brain Editor)
        brain_log_path = os.path.join(brain_dir, 'brain_log.jsonl')
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'run_id': os.path.basename(ckpt_dir),
            'scene': args.scene,
            'steps': total_steps,
            'falls': fall_count,
            'max_distance': max_dist,
            'seed': args.seed,
            'ball_episodes': getattr(main, '_ball_ep', 0),
            'n_episodes': len(creature.brain.memory.episodes) if hasattr(creature.brain, 'memory') else 0,
            'n_concepts': creature.brain.synaptogenesis.graph.size() if hasattr(creature.brain, 'synaptogenesis') else 0,
            'snapshot': os.path.basename(history_file),
        }
        with open(brain_log_path, 'a') as blf:
            blf.write(json.dumps(log_entry) + '\n')


if __name__ == '__main__':
    main()
