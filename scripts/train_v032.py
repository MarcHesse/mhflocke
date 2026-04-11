#!/usr/bin/env python3
"""
MH-FLOCKE — Level 15 Training v0.4.2
=======================================
"A dog doesn't stand still on a meadow."

v0.4.2: Scalable SNN for hardware brain transfer.
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

Author: MH-FLOCKE Level 15 v0.4.2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if sys.platform != 'win32':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import time
import json
import logging
import numpy as np
import torch
import mujoco

from src.body.genome import Genome
from src.body.mujoco_creature import MuJoCoCreature, MuJoCoCreatureBuilder
from src.body.mujoco_world import MuJoCoWorld
from src.brain.cerebellar_learning import CerebellarLearning, CerebellarConfig
from src.brain.spinal_reflexes import SpinalReflexes, ReflexConfig, SpinalSegments, SpinalSegmentConfig
from src.brain.spinal_cpg import SpinalCPG, SpinalCPGConfig
from src.brain.developmental_schedule import DevelopmentalSchedule, DevelopmentalConfig
from src.body.terrain import (
    TerrainConfig, generate_heightfield, inject_terrain, inject_terrain_geoms,
    terrain_type_from_scene, difficulty_from_scene, inject_ball,
)

logger = logging.getLogger(__name__)


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
            with open(xml_path) as f:
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
    with open(xml_path) as f:
        xml = f.read()
    return re.sub(r'timestep="[0-9.]+"', f'timestep="{new_timestep}"', xml)


class CompetenceGate:
    """Competence-gated CPG->Actor handoff. Resolves Issue #45.
    
    v0.3.5: Stability-aware gate. Actor competence now requires BOTH
    speed AND stability. The actor must prove it can maintain upright
    posture while moving, not just push the creature forward into falls.
    
    Ablation finding: Speed-only gate let actor reach cpg_min (40%) by
    step 4k, then falls escalated because actor disrupted CPG rhythm
    without maintaining balance. Now: fall_rate resets competence hard.
    """

    def __init__(self, speed_threshold=0.03, grow_rate=0.0002,
                 shrink_rate=0.0003, cpg_min=0.40, cpg_max=0.9,
                 vel_ema_decay=0.99, stability_window=1000):
        self.speed_threshold = speed_threshold
        self.grow_rate = grow_rate
        self.shrink_rate = shrink_rate
        self.cpg_min = cpg_min
        self.cpg_max = cpg_max
        self.actor_competence = 0.0
        self.cpg_weight = cpg_max
        self.vel_ema_decay = vel_ema_decay
        self.vel_ema = 0.0
        # Stability tracking
        self.stability_window = stability_window
        self._recent_falls = 0       # falls in current window
        self._window_step = 0        # steps since window reset
        self._fall_rate = 0.0        # falls per 1000 steps (smoothed)
        self._upright_ema = 1.0      # smoothed upright value

    def update(self, step, vel_mps, is_fallen, upright=1.0):
        # Track upright EMA
        self._upright_ema = self._upright_ema * 0.995 + upright * 0.005
        
        # Track fall rate in sliding window
        self._window_step += 1
        if is_fallen:
            self._recent_falls += 1
        if self._window_step >= self.stability_window:
            self._fall_rate = self._recent_falls / (self.stability_window / 1000.0)
            self._recent_falls = 0
            self._window_step = 0

        if is_fallen:
            # On fall: actor clearly not competent, fast shrink
            self.actor_competence = max(0.0, self.actor_competence - self.shrink_rate * 10)
            self._recompute_cpg()
            return
        
        # Smooth velocity
        self.vel_ema = self.vel_ema * self.vel_ema_decay + vel_mps * (1.0 - self.vel_ema_decay)
        
        # Competence grows only if BOTH speed and stability are good
        is_moving = self.vel_ema > self.speed_threshold
        is_stable = self._fall_rate < 5.0 and self._upright_ema > 0.85
        
        if is_moving and is_stable:
            # Actor is both moving AND stable: grow competence
            self.actor_competence = min(1.0, self.actor_competence + self.grow_rate)
        elif is_moving and not is_stable:
            # Moving but unstable: actor is causing problems, shrink faster
            self.actor_competence = max(0.0, self.actor_competence - self.shrink_rate * 3)
        else:
            # Stopped: shrink normally
            self.actor_competence = max(0.0, self.actor_competence - self.shrink_rate)
        
        self._recompute_cpg()
    
    def _recompute_cpg(self):
        """CPG weight from actor competence. Higher competence = less CPG."""
        self.cpg_weight = max(self.cpg_min,
                              self.cpg_max - self.actor_competence * (self.cpg_max - self.cpg_min))

    def get_cpg_weight(self):
        return self.cpg_weight

    def get_stats(self):
        return {'actor_competence': self.actor_competence, 'cpg_weight': self.cpg_weight,
                'vel_ema': self.vel_ema, 'fall_rate': self._fall_rate,
                'upright_ema': self._upright_ema}


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
    parser = argparse.ArgumentParser(description='MH-FLOCKE Level 15 v0.4.2')
    parser.add_argument('--scene', type=str, default='walk on hilly grassland')
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--xml', type=str, default='creatures/dm_quadruped/creature.xml')
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--timestep', type=float, default=0.005)
    parser.add_argument('--snn-substeps', type=int, default=3)
    parser.add_argument('--no-flog', action='store_true')
    parser.add_argument('--no-cerebellum', action='store_true')
    parser.add_argument('--no-terrain', action='store_true')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--no-drives', action='store_true', help='Disable autonomous drive loop')
    parser.add_argument('--no-sensory', action='store_true', help='Disable sensory environment (no scent/sound)')
    parser.add_argument('--difficulty', type=float, default=None)
    parser.add_argument('--pci-interval', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
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
    print(f'  MH-FLOCKE -- Level 15 v0.4.2')
    print(f'  "A dog doesn\'t stand still on a meadow."')
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

    if is_external_mjcf:
        # Go2 / Menagerie: load via from_xml_path (handles <include>, meshdir)
        # Terrain/Ball injection for external MJCF: write patched XML as temp file
        # Use PID in filenames to support parallel runs
        _pid = os.getpid()
        hfield_path = os.path.join('output', f'mhflocke_terrain_{_pid}.png')
        os.makedirs('output', exist_ok=True)
        _needs_temp_xml = (terrain_cfg.terrain_type != 'flat') or _scene_has_ball
        if _needs_temp_xml:
            with open(xml_path) as f:
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
                profile=profile)
            os.remove(temp_xml)
        else:
            creature = MuJoCoCreatureBuilder.build(
                genome, world=world, device=device,
                creature_name=args.creature_name.lower(),
                xml_path=xml_path,
                n_hidden_neurons=n_hidden,
                hardware_sensors=args.hardware_sensors,
                no_vision=args.no_vision,
                profile=profile)
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
        creature = MuJoCoCreatureBuilder.build(genome, world=world, device=device,
            creature_name=args.creature_name.lower(), xml_string=xml_string,
            n_hidden_neurons=n_hidden,
            hardware_sensors=args.hardware_sensors,
            no_vision=args.no_vision,
            profile=profile)
    creature.SNN_SUBSTEPS = args.snn_substeps
    # Protect cerebellar populations from CognitiveBrain learning
    # (GrC patterns must stay stable for Marr-Albus learning)
    # Hidden/Input/Output neurons learn freely via R-STDP, Hebbian, Dreams, Neuromod
    creature.snn.protected_populations = {
        'mossy_fibers', 'granule_cells', 'golgi_cells',
        'purkinje_cells', 'dcn',
    }
    creature.brain.config.pci_interval = args.pci_interval
    # Disable periodic dreaming during embodied training (too expensive).
    # Dreams happen between ball episodes instead (Issue #76d).
    # Default dream_interval=100 caused 400+ dream cycles per 50k run
    # with 3900ms/step spikes from synaptogenesis consolidation.
    creature.brain.config.dream_interval = 0
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
        # Standing pose from keyframe
        standing_qpos = world._data.qpos[7:7+n_act].copy()  # from keyframe reset
        print(f'  Standing pose: {standing_qpos[:6]}...')
        # Attach PD controller to creature (runs inside apply_motor_output, BEFORE world.step)
        pd_scale = profile.get('pd_scale', 0.5) if profile else 0.5
        pd_fallen_scale = profile.get('pd_fallen_scale', 1.5) if profile else 1.5
        pd_controller = {
            'kp': pd_kp, 'kd': pd_kd, 'lo': ctrl_lo, 'hi': ctrl_hi,
            'standing': standing_qpos, 'scale': pd_scale, 'fallen_scale': pd_fallen_scale,
        }
        print(f'  PD scale: {pd_scale} (standing) / {pd_fallen_scale} (fallen)')

    # Attach PD controller to creature object
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
    print(f'  Spinal Segments: tone={spinal_segments.config.tone_gain:.2f}'
          f'  stretch={spinal_segments.config.stretch_gain:.2f}'
          f'  golgi@{spinal_segments.config.golgi_threshold:.2f}')

    # --- Load evolved CPG params or use defaults ---
    if cpg_config_path:
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

    print(f'\n  -- Phase 2: Competence-Gated Handoff --')
    gate = CompetenceGate(speed_threshold=0.03, cpg_min=0.40, cpg_max=0.9)
    print(f'  Gate: CPG {gate.cpg_max:.0%} -> {gate.cpg_min:.0%} when actor speed > {gate.speed_threshold} m/s')

    # --- Issue #57: Autonomous Drive Loop ---
    drive_bridge = None
    if not args.no_drives:
        try:
            from src.behavior.drive_motor_bridge import DriveMotorBridge
            drive_bridge = DriveMotorBridge(
                creature_type='dog',
                scene_instruction=scene_inst,
            )
            print(f'\n  -- Drive Loop: ACTIVE --')
            print(f'  Behaviors: {", ".join(drive_bridge.knowledge.get_all_names())}')
            print(f'  Drive → BehaviorPlanner → MotorPattern → CPG freq/amp modulation')
        except Exception as e:
            print(f'\n  Drive Loop: init failed ({e}), running without drives')
            drive_bridge = None
    else:
        print(f'\n  -- Drive Loop: DISABLED (--no-drives) --')

    # --- Issue #75: Sensory Environment ---
    sensory_env = None
    if drive_bridge and not args.no_sensory:
        try:
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
                    position=ball_xpos.copy(), strength=3.0, radius=1.5,  # 1.5m find radius (0.5 too tight)
                    name='ball_scent', fixed=True
                )]
                creature._steer_gain = 0.15  # VOR uses this as base gain for motor corrections
                print(f'  Sensory: BALL MODE -- target at ({ball_xpos[0]:.1f}, {ball_xpos[1]:.1f})')
            else:
                # Normal scene: random scent sources for behavior motivation
                sensory_env.spawn_scent(count=2, min_dist=2.0, max_dist=4.0)
                print(f'  Sensory: 2 scent sources, sounds every ~2k steps')
        except Exception as e:
            print(f'  Sensory env: init failed ({e})')
            sensory_env = None

    start_step = 0
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
        start_step = ckpt.get('step', 0)
        gate.actor_competence = ckpt.get('actor_competence', 0.0)
        gate.cpg_weight = ckpt.get('cpg_weight', gate.cpg_max)
        gate.vel_ema = ckpt.get('vel_ema', 0.0)
        cpg_phases = ckpt.get('cpg_phases', None)
        cpg_step = ckpt.get('cpg_step', 0)
        if cpg_phases is not None:
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
    if os.path.exists(brain_file_auto) and hasattr(creature, 'brain') and creature.brain:
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
    run_id = f'v034_{int(time.time())}'
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
                'device': device,
                'version': 'v0.4.2',
                'n_neurons': creature.snn.config.n_neurons,
                'population_sizes': {
                    'n_input': creature.n_input_neurons,
                    'n_output': creature.n_output_neurons,
                    'n_granule': len(creature.snn.populations.get('granule_cells', [])),
                    'n_golgi': len(creature.snn.populations.get('golgi_cells', [])),
                    'n_purkinje': len(creature.snn.populations.get('purkinje_cells', [])),
                    'n_dcn': len(creature.snn.populations.get('dcn', [])),
                    'n_total': creature.snn.config.n_neurons,
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
    max_dist = 0.0
    fall_count = 0
    recovery_count = 0
    best_upright_streak = 0
    current_upright_streak = 0
    step_times = []
    last_pci = 0.0
    brain_result = {}

    # Drive loop state (for logging)
    current_behavior = 'walk'
    current_freq_scale = 1.0
    current_amp_scale = 1.0

    # Auto-reset state
    auto_reset_limit = args.auto_reset
    consecutive_fallen = 0
    reset_count = 0

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

    for step in range(start_step, total_steps):
        t_step = time.perf_counter()
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

        # Foot contact sensing (Phase A of Terrain-Adaptive Locomotion)
        foot_sensor.update(world._model, world._data, step)
        sensor_data.update(foot_sensor.get_data())

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

        # --- Issue #75: Sensory environment ---
        if sensory_env:
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
            heading = float(world._data.qpos[3]) if len(world._data.qpos) > 3 else 0.0
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

        if auto_reset_limit > 0 and consecutive_fallen >= auto_reset_limit:
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
            reward = forward_vel * 5.0 + max(0, upright) * 2.0
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

        # DA signal: ensure minimum DA even when fallen so learning continues
        da_signal = np.clip(reward / 10.0, 0.05, 1.0)
        creature.snn.neuromod_levels['da'] = float(da_signal)

        reflex_cmd = reflexes.compute(sensor_data, is_fallen, sim_dt=args.timestep)

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
        else:
            prev_brain = brain_result
            cur_reward_b = prev_brain.get('curiosity_reward', 0.0) if prev_brain else 0.0
            drv = prev_brain.get('drives', {}) if prev_brain else {}
            expl_drive = drv.get('exploration', 0.3)
            boredom = prev_brain.get('boredom', 0.0) if prev_brain else 0.0
            ne_base = 0.25
            ne_curiosity = cur_reward_b * 0.3
            ne_boredom = min(boredom, 1.0) * 0.2
            ne_level = min(0.7, ne_base + ne_curiosity + ne_boredom)
            creature.snn.set_neuromodulator('ne', ne_level)
            tonic_base = 0.02
            tonic_explore = expl_drive * 0.04
            tonic_curiosity = cur_reward_b * 0.03
            creature.snn._hidden_tonic_current = tonic_base + tonic_explore + tonic_curiosity

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
        if current_behavior == 'motor_babbling':
            spinal_cpg._babbling_noise = 0.25  # 25% per-leg variation — gentle weight shifts
        else:
            spinal_cpg._babbling_noise = 0.0

        # Terrain reflex: compute corrections + CPG modulation (Phase B)
        terrain_corr = terrain_reflex.compute(sensor_data)
        tr_freq = terrain_reflex.freq_scale
        tr_amp = terrain_reflex.amp_scale

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

        if is_external_mjcf:
            # Go2: direct joint control (per-joint amplitudes)
            cpg_cmd = spinal_cpg.compute(
                dt=args.timestep, arousal=ne_lvl,
                freq_scale=current_freq_scale * tr_freq,
                amp_scale=current_amp_scale * tr_amp * _proximity_amp_scale,
                steering=_cpg_steering,
            )
        else:
            # Bommel: tendon-coupled actuators
            cpg_cmd = spinal_cpg.compute_tendon(
                dt=args.timestep, arousal=ne_lvl,
                freq_scale=current_freq_scale * tr_freq,
                amp_scale=current_amp_scale * tr_amp * _proximity_amp_scale,
            )

        gate.update(step, vel_mps, is_fallen, upright=upright)
        cpg_weight = gate.get_cpg_weight()
        creature._cpg_cmd = cpg_cmd
        creature._cpg_weight = cpg_weight
        creature._reflex_cmd = reflex_cmd
        creature._terrain_corr = terrain_corr  # Phase B terrain reflex corrections
        creature._olfactory_steering = sensor_data.get('olfactory_steering', 0.0)

        # Store CPG phase for hardware sensor encoding (Bridge v2.5 compatibility)
        if args.hardware_sensors and hasattr(spinal_cpg, '_phases'):
            import math
            _phase_rad = float(spinal_cpg._phases[0])
            creature._cpg_phase_input = np.array(
                [math.sin(_phase_rad), math.cos(_phase_rad)], dtype=np.float32)

        step_result = creature.step(
            reward_signal=reward,
            extra_sensor_data={
                'smell_strength': sensor_data.get('smell_strength', 0.0),
                'scent_reward': sensor_data.get('scent_reward', 0.0),
                # Navigation context for Synaptogenesis (Module Audit Fix)
                'ball_heading': getattr(creature, '_ball_heading', 0.0),
                'ball_distance': prev_ball_dist if prev_ball_dist is not None else 0.0,
                'steering_offset': getattr(creature, '_steering_offset', 0.0),
            },
        )

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

        # Memory management: periodic GC to clean up Python object cycles.
        if step % 5000 == 0 and step > 0:
            import gc
            gc.collect()
        brain_result = step_result.get('brain', {})
        pci_val = brain_result.get('pci', None)
        if pci_val is not None and pci_val > 0:
            last_pci = pci_val
        # R-STDP is now handled by CognitiveBrain.process() (step 11)
        # with protected_populations filtering cerebellar synapses.
        # No duplicate apply_rstdp needed here.

        d = creature.get_distance_traveled()
        if d > max_dist:
            max_dist = d
        step_dt = time.perf_counter() - t_step
        step_times.append(step_dt)

        if recorder and step % 10 == 0:
            try:
                qpos = world._data.qpos
                extra_creature = dict(step=step, x=float(qpos[0]), y=float(qpos[1]))
                # Store ball position in FLOG for video rendering
                _ball_id_flog = mujoco.mj_name2id(world._model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
                if _ball_id_flog >= 0:
                    bp = world._data.xpos[_ball_id_flog]
                    extra_creature['ball_pos'] = [float(bp[0]), float(bp[1]), float(bp[2])]
                recorder.record_creature(joint_positions=qpos.copy(), joint_velocities=world._data.qvel.copy(),
                    center_of_mass=qpos[:3].copy(), heading=float(qpos[3]), speed=vel_mps, **extra_creature)
            except Exception as e:
                if step < 100:  # Only warn on first few failures
                    print(f'  ⚠ FLOG creature write failed at step {step}: {e}')

        if recorder and step % log_every == 0 and step > 0:
            try:
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
                emo = brain_result.get('emotion', {})
                drv_r = brain_result.get('drives', {})
                flog_data['emotion_dominant'] = emo.get('dominant_emotion', '')
                flog_data['valence'] = emo.get('valence', 0.0)
                flog_data['arousal'] = emo.get('arousal', 0.0)
                flog_data['drive_dominant'] = drv_r.get('dominant', '')
                flog_data['curiosity_reward'] = brain_result.get('curiosity_reward', 0.0)
                # Issue #75: Sensory environment
                if sensory_env:
                    flog_data['smell_strength'] = sensor_data.get('smell_strength', 0.0)
                    flog_data['smell_direction'] = sensor_data.get('smell_direction', 0.0)
                    flog_data['sound_intensity'] = sensor_data.get('sound_intensity', 0.0)
                    flog_data['sound_direction'] = sensor_data.get('sound_direction', 0.0)
                    flog_data['scents_found'] = sensory_env.scents_found
                    flog_data['olfactory_steering'] = sensor_data.get('olfactory_steering', 0.0)
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
                    # Scent source positions for video rendering
                    for si, sc in enumerate(sensory_env._scents):
                        flog_data[f'scent_{si}_x'] = float(sc.position[0])
                        flog_data[f'scent_{si}_y'] = float(sc.position[1])
                recorder.record_training_stats(flog_data)
            except Exception as e:
                print(f'  ⚠ FLOG stats write failed at step {step}: {e}')

        if step > 0 and step % log_every == 0:
            avg_ms = np.mean(step_times[-log_every:]) * 1000
            eta_min = (total_steps - step) * (avg_ms / 1000) / 60
            line1 = (f'  {step:>7,}/{total_steps:,}  dist:{max_dist:>5.2f}m  vel:{vel_mps:.3f}m/s'
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
                     f'  CF:{cb_stats.get("cf_magnitude", 0.0):.3f}'
                     f'  corr:{cb_stats.get("correction_magnitude", 0.0):.4f}'
                     f'  terr:{terrain_cfg.difficulty:.2f}'
                     f'  TR:{terrain_reflex.stats["terrain_reflex_mag"]:.3f}'
                     f'  ft:{int(foot_sensor.contacts.sum())}')
            # Issue #75: sensory info
            if sensory_env:
                sm = sensor_data.get('smell_strength', 0.0)
                sf = sensory_env.scents_found
                line3 += f'  sm:{sm:.2f}  sf:{sf}'
                bh = getattr(creature, '_ball_heading', 0.0)
                bs = getattr(creature, '_ball_salience', 0.0)
                so = getattr(creature, '_steering_offset', 0.0)
                bd = prev_ball_dist if prev_ball_dist is not None else -1.0
                vr = vor.stats.get('vor_raw', 0.0) if 'vor' in dir() else 0.0
                vs = vor.stats.get('vor_smoothed', 0.0) if 'vor' in dir() else 0.0
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
        'cpg_phases': spinal_cpg._phases.tolist(),
        'cpg_step': spinal_cpg._step,
        'version': 'v0.4.2', 'scene': args.scene, 'seed': args.seed,
        'terrain_type': terrain_cfg.terrain_type, 'terrain_difficulty': terrain_cfg.difficulty,
        'flog_path': flog_path,
        'flog_frames': recorder.frame_count if recorder else 0,
    }
    if cb:
        ckpt_data['cerebellum_state'] = cb.state_dict()
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
        save_brain(creature.brain, creature.snn, brain_file, metadata=brain_meta)
        print(f'  Brain saved: {brain_file}')
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
