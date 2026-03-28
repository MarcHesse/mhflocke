# MH-FLOCKE — System Architecture (Level 15 v0.3.2)
# Updated: 25 February 2026

## Biological AI — Born, not programmed.
## Author: Marc Hesse, Potsdam, Germany
## Project Start: 31 January 2026

---

## What is MH-FLOCKE?

A scientific experiment: Can a system develop genuine understanding — not through
programming, but through embodied experience?

The system gets a body, a world, and neurons.
No calibration. No motor mapping. No hardcoded strategies.
It discovers what it is, what it can do, and what the world is.

Every component exists as proven science. The integration is new.
Nobody has built the complete system.

**Named after Flocke (†) and Mogli — Marc's dogs.**

---

## The Complete Module Inventory

### Package: src/brain/ (40+ modules)

| Module | File | Phase | Status in Training | Function |
|--------|------|-------|-------------------|----------|
| SNN Controller | `snn_controller.py` | 3 | ✅ Core | LIF-LTC neurons, Small-World topology, 5k neurons, populations (MF/GrC/GoC/PkC/DCN/hidden) |
| Cognitive Brain | `cognitive_brain.py` | 10-I | ✅ Core | 15-step cognitive cycle orchestrator — calls ALL modules below |
| Cerebellar Learning | `cerebellar_learning.py` | 15 | ✅ Core | Marr-Albus-Ito model: Mossy Fibers → Granule Cells → Purkinje → DCN, LTD/LTP, CF error signal |
| Spinal CPG | `spinal_cpg.py` | 15 | ✅ Core | Central Pattern Generator for tendon-mode quadruped locomotion |
| Spinal Reflexes | `spinal_reflexes.py` | 15 | ✅ Core | Righting reflex, cross-extension, vestibulospinal gain |
| Embodied Emotions | `embodied_emotions.py` | 6 | ✅ Active | Valence-Arousal model from body signals, somatic markers → neuromodulators |
| Emotions (legacy) | `emotions.py` | 6 | ✅ Active | Emotion state management |
| Drives | `drives.py` | 5 | ✅ Active | Motivational drives: survival, exploration, comfort, social, curiosity |
| Body Schema | `body_schema.py` | 4 | ✅ Active | Efference copy checking, body model learning, anomaly detection |
| World Model | `world_model.py` | 8C | ✅ Active | Spiking world model: prediction + prediction error + dream engine |
| GWT Bridge | `gwt_bridge.py` | 6 | ✅ Active | Global Workspace Theory: competition between sensory/motor/predictive/error/memory modules |
| GWT | `gwt.py` | 6 | ✅ Active | GWT core implementation |
| Sensomotor Memory | `sensomotor_memory.py` | 3+ | ✅ Active | Episodic recording + similarity recall |
| Metacognition | `embodied_metacognition.py` | 5 | ✅ Active | Confidence assessment, consciousness level (0-5), learning progress |
| Metacognition (base) | `metacognition.py` | 5 | ✅ Active | Base metacognition module |
| Consistency Checker | `consistency_checker.py` | 8 | ✅ Active | Integrity check: prediction error + body anomaly + memory mismatch |
| Synaptogenesis | `synaptogenesis.py` | 9 | ✅ Active | SNN spike patterns → concept graph, observe + consolidate + retrieve |
| Astrocyte Gate | `astrocyte_gate.py` | 8D | ✅ Active | Glial calcium dynamics, metabolic coupling, cluster-based gating |
| Curiosity | `curiosity.py` | 9W | ✅ Active | Intrinsic motivation: prediction error → curiosity reward, boredom detection |
| Empowerment | `empowerment.py` | 9W | ✅ Active | Action→State mutual information, agent wants control over environment |
| Dream Mode | `dream_mode.py` | 8 | ✅ Active* | Offline replay + memory consolidation (*SNN replay protected by protect_snn_weights) |
| Neuromodulators | `neuromodulators.py` | 5 | ✅ Active | DA, 5-HT, NE, ACh — dynamic modulation per situation |
| Hebbian Learning | `hebbian_learning.py` | 5 | ❌ Protected | Co-activation learning (blocked by protect_snn_weights to preserve GrC patterns) |
| Evolved Plasticity | `evolved_plasticity.py` | 7 | ❌ Protected | Meta-learned plasticity rules (blocked by protect_snn_weights) |
| Modular Skills | `modular_skills.py` | 11 | ✅ Active | Skill registry, EWC protection for frozen skills |
| Theory of Mind | `theory_of_mind.py` | 9+ | ⏸ Lazy | Mirror neurons, cooperation decisions — activated via enable_tom() for multi-agent |
| Active Inference | `active_inference.py` | 9W | 📦 Available | Free Energy minimization framework |
| Predictive Coding | `predictive_coding.py` | 8C | 📦 Available | Hierarchical prediction error minimization |
| Self Model | `self_model.py` | 6 | 📦 Available | Self-representation |
| Multi-Compartment | `multi_compartment.py` | 8D | 📦 Available | Dendritic computation (BrainCog-inspired) |
| Homeostatic Plasticity | `homeostatic_plasticity.py` | 5 | 📦 Available | Activity stabilization |
| Knowledge Synthesis | `knowledge_synthesis.py` | 2 | 📦 Available | Knowledge graph operations |
| Hybrid Datastore | `hybrid_datastore.py` | 2 | 📦 Available | Long-term knowledge graph storage |
| Consciousness Eval | `consciousness_eval.py` | 6 | 📦 Available | Extended PCI measurement |
| Consciousness Metrics | `consciousness_metrics.py` | 6 | 📦 Available | Consciousness metric suite |
| Goal System | `goal_system.py` | 5 | 📦 Available | Autonomous goal generation |
| Goal Generator | `goal_generator.py` | 5 | 📦 Available | Goal proposals from drives |
| Working Memory | `working_memory.py` | 4 | 📦 Available | Short-term buffer |
| Episodic Memory | `episodic_memory.py` | 3+ | 📦 Available | Longer-term episodic storage |
| Inner Life | `inner_life.py` | 8 | 📦 Available | Internal narrative |
| Circadian | `circadian.py` | 8 | 📦 Available | Sleep/wake cycle |
| GPU SNN | `gpu_snn.py` | 9 | 📦 Available | GPU-accelerated SNN for scaling |
| Brain Persistence | `brain_persistence.py` | 10 | 📦 Available | Save/load brain state |
| Creature Store | `creature_store.py` | 14 | ✅ Active | "Brain Git": versioned snapshots, FLOG binary recording |
| Actor Critic | `actor_critic.py` | 3 | ✅ Active | Base class for cerebellar learning |
| CPG (legacy) | `cpg.py` | 9 | Replaced | Old CPG, replaced by spinal_cpg.py |
| Neural Config | `neural_config.py` | 3 | ✅ Active | SNN configuration parameters |
| Topology Monitor | `topology_monitor.py` | 5 | 📦 Available | Network topology analysis |

Legend:
- ✅ Core = Essential for training, called directly by train_v032.py
- ✅ Active = Called by CognitiveBrain.process() every step
- ❌ Protected = Module runs but SNN modifications blocked by protect_snn_weights
- ⏸ Lazy = Available but not activated by default
- 📦 Available = Module exists, not currently wired into CognitiveBrain

### Package: src/body/ (8 modules)

| Module | File | Function |
|--------|------|----------|
| Genome | `genome.py` | Creature genome (morphology parameters) |
| MuJoCo Creature | `mujoco_creature.py` | Creature physics body, step() calls CognitiveBrain.process() |
| MuJoCo World | `mujoco_world.py` | Physics simulation, sensor data extraction |
| MuJoCo Creature Builder | `mujoco_creature.py` | Builds creature from genome + XML |
| Scene Builder | `scene_builder.py` | Visual scenes (neon_grassland, flat_grass, ice, etc.) |
| Terrain | `terrain.py` | MuJoCo heightfield generation + injection |
| Body Types | `body_types.py` | Body type definitions |
| Morphology | `morphology.py` | Morphology utilities |

### Package: src/behavior/ (4 modules)

| Module | File | Function |
|--------|------|----------|
| Behavior Knowledge | `behavior_knowledge.py` | Dog behavior database (walk, trot, sniff, rest, alert, etc.) |
| Behavior Planner | `behavior_planner.py` | Drive → Situation → Behavior selection |
| Behavior Executor | `behavior_executor.py` | Behavior → Motor modulation (freq_scale, amp_scale) |
| Scene Instruction | `scene_instruction.py` | Scene → Drive biases, preset environments |

### Package: src/bridge/ (9 modules)

| Module | File | Function |
|--------|------|----------|
| LLM Bridge | `llm_bridge.py` | TaskParser, ExperienceNarrator, TrainingOrchestrator |
| Understand | `understand.py` | UnderstandEngine: LLM/builtin → behaviors + scene |
| Grow | `grow.py` | Post-training suggestions for autonomous learning |
| Other modules | Various | Multi-LLM routing, task specs, verification |

### Package: src/integrity/ (5 modules)

| Module | File | Function |
|--------|------|----------|
| Dissonance Detector | `dissonance_detector.py` | Graph-based contradiction detection |
| Inhibition Controller | `inhibition_controller.py` | Stops implausible outputs |
| Integrity Validator | `integrity_validator.py` | Multi-source verification |
| Claim Extractor | `claim_extractor.py` | Extracts claims for verification |
| GPT2 Generator | `gpt2_generator.py` | Local generation for testing |

### Package: src/viz/ (14+ modules)

| Module | File | Function |
|--------|------|----------|
| Video Assembler | (scripts/) `assemble_video.py` | YAML-driven video editor: segments, overlays, cameras |
| Cerebellar Panel | `cerebellar_panel.py` | Cerebellar dashboard overlay for videos |
| Brain Overlay | `brain_overlay.py` | SNN/emotion/GWT overlay for videos |
| Evo Overlay | `evo_overlay.py` | Evolution progress overlay |
| Camera System | `camera_system.py` | CinematicCamera: follow, orbit, frontal, dramatic_low, etc. |
| Post Processing | `post_processing.py` | Cinematic color grading, vignette |
| Overlay Base | `overlay_base.py` | Glass compositing (blur-behind transparency) |
| FLOG Replay | `flog_replay.py` | Binary log → NPZ physics + CSV stats for video |
| Brain 3D | `brain_3d_network.py` | 3D neural network visualization |
| Flocke Editor | `flocke_editor.py` | Timeline, events, subtitles, effects |
| Recording | `recording.py` | Re-export shim for backwards compat |

### Package: src/self_improvement/ (7 modules)

| Module | File | Function |
|--------|------|----------|
| Self-Improve Pipeline | `self_improve_pipeline.py` | Recursive self-improvement loop |
| Self Analyzer | `self_analyzer.py` | Performance analysis |
| Algorithm Improver | `algorithm_improver.py` | Algorithm optimization suggestions |
| Parameter Tuner | `parameter_tuner.py` | Hyperparameter optimization |
| Neuroevolution | `neuroevolution.py` | Evolutionary architecture search |
| Code Reader | `code_reader.py` | Self-inspection |
| Safety Gate | `safety_gate.py` | Safety constraints for self-modification |

---

## The 15-Step Cognitive Cycle (CognitiveBrain.process)

Called every simulation step via `creature.step()`:

```
Step  1: SENSE           → Raw sensor data from MuJoCo
Step  2: BODY SCHEMA     → Efference copy check, anomaly detection
Step  3: WORLD MODEL     → Prediction + prediction error (SpikingWorldModel)
Step  4: EMOTIONS        → Valence-Arousal from body signals (EmbodiedEmotions)
Step  5: MEMORY          → Record step + recall similar episodes (SensomotorMemory)
Step  6: DRIVES          → Dominant drive from situation (MotivationalDrives)
Step 6b: BEHAVIOR        → Drive + Knowledge → behavior selection (BehaviorPlanner)
Step  7: GWT             → Competition: sensory/motor/predictive/error/memory (GlobalWorkspaceBridge)
Step  8: METACOGNITION   → Confidence, consciousness level 0-5 (EmbodiedMetacognition)
Step  9: CONSISTENCY     → Integrity check (ConsistencyChecker)
Step 10: REWARD          → Combined: external + curiosity + empowerment + drive + emotion
Step 11: LEARNING        → R-STDP or Evolved Plasticity [PROTECTED]
Step 12: SYNAPTOGENESIS  → SNN patterns → concept graph + retrieval [retrieval PROTECTED]
Step 12b: ASTROCYTE      → Calcium dynamics, cluster gating
Step 13: HEBBIAN         → Co-activation learning [PROTECTED]
Step 14: DREAM           → Offline replay + memory consolidation [SNN replay PROTECTED]
Step 15: NEUROMOD        → Somatic markers → SNN parameters [PROTECTED]
Step 15b: PCI            → Lempel-Ziv complexity of spike patterns (every 500 steps)
```

### protect_snn_weights Flag

When `protect_snn_weights = True` (default in train_v032.py):
- CognitiveBrain modules still RUN and collect data
- But direct SNN weight/parameter modifications are BLOCKED
- This protects cerebellar Granule Cell patterns from interference

Protected operations:
- GWT broadcast → SNN hidden neurons (Step 7)
- R-STDP / Evolved Plasticity (Step 11)
- Synaptogenesis retrieval → SNN apical context (Step 12)
- Hebbian co-activation learning (Step 13)
- Dream replay SNN steps (Step 14, memory consolidation still runs)
- Neuromodulator somatic markers → SNN tau/threshold (Step 15)

Unprotected (always active):
- All observation/recording (memory, synaptogenesis observe, astrocyte calcium)
- All internal state updates (emotions, drives, world model training, metacognition)
- PCI measurement
- Behavior planning
- Curiosity/empowerment reward computation

---

## Training Architecture (train_v032.py)

```
User: "walk on hilly grassland"
  │
  ▼
Phase 0: Knowledge Engine
  TaskParser → UnderstandEngine (LLM or builtin)
  → 5 behaviors + SceneInstruction + TerrainConfig
  │
  ▼
Phase 3: World Building
  XML + Terrain heightfield → MuJoCo model
  Creature: SNN (5000 neurons) + CognitiveBrain (all modules)
  │
  ▼
Phase 1: Cerebellum Setup
  CerebellarLearning (Marr-Albus-Ito)
  Populations: MF → GrC → GoC → PkC → DCN
  DA modulation: reward → LTP boost (2×), LTD suppression (0.5×)
  │
  ▼
Phase 2: Competence Gate
  CPG weight: 90% → 40% as actor proves speed > 0.03 m/s
  │
  ▼
Training Loop (continuous, NO resets):
  ┌─────────────────────────────────────────────┐
  │ 1. Sensors from MuJoCo                      │
  │ 2. Reward computation (forward vel + upright)│
  │ 3. DA signal from reward                     │
  │ 4. Reflex computation (vestibulospinal)      │
  │ 5. Dynamic reflex scale (mass × instability) │
  │ 6. Cerebellum update (CF error → LTD/LTP)    │
  │ 7. CPG tendon command                        │
  │ 8. CompetenceGate update                     │
  │ 9. creature.step(reward_signal=reward)        │
  │    └→ SNN forward pass (3 substeps)          │
  │    └→ CognitiveBrain.process() ← ALL 15 STEPS│
  │    └→ Motor command: CPG×w + Actor×(1-w) + Reflex│
  │ 10. FLOG recording (physics every 10, stats every 1000)│
  └─────────────────────────────────────────────┘
```

### What train_v032.py handles directly:
- SpinalCPG (tendon mode)
- CerebellarLearning (external to CognitiveBrain)
- SpinalReflexes (external to CognitiveBrain)
- CompetenceGate (CPG→Actor blend)
- FLOG recording
- Reward computation
- Dynamic reflex scaling

### What CognitiveBrain handles (inside creature.step):
- Everything else: World Model, Emotions, Drives, Memory, GWT, Metacognition,
  Consistency, Synaptogenesis, Astrocytes, Curiosity, Empowerment, Dreams, PCI

---

## Creature Pipeline

```
1. DESIGN    → Flocke Creature Editor v3 (HTML)
                Sliders: body, legs, head, eyes, stripes, tail, beacon
                9 presets (Mogli, FLOCKE, Scout, Tank, Spider, Baby, Racer, Ghost, Lava, Neon)
                Export: MJCF XML + Preset JSON

2. EVOLVE    → evolve_cpg_params.py
                bootstrap_genome_from_morphology(model)
                → measures leg length, mass, joint ranges
                → derives frequency, amplitudes, stance/swing
                → detects tendon actuators
                Population: 40% bootstrap + 30% gait variants + 30% random
                Output: checkpoints/{name}/cpg_config.json

3. TRAIN     → train_v032.py
                Full pipeline: Knowledge → World → Cerebellum → CPG → SNN → CognitiveBrain
                Output: creatures/{name}/v032_xxx/training_log.bin (FLOG)
                        checkpoints/{name}/snn_v032_state.pt
                        checkpoints/{name}/v032_checkpoint.pt

4. VIDEO     → assemble_video.py
                YAML cut template → FLOG export → MuJoCo replay + overlay
                Scene: SceneBuilder (visual) + Terrain (physics)
                Camera: CinematicCamera with YAML cuts
                Overlay: Cerebellar or Brain dashboard + glass compositing
                Output: .mp4 + YouTube chapters/description/thumbnail
```

---

## Current Creatures

### Mogli (dm_quadruped reference)
- Standard dm_quadruped with tendon system
- 12 actuators (yaw/lift/extend × 4 legs)
- Reference mass: ~20kg
- Established CPG parameters

### Bommel (custom from Editor v3)
- Red body (0.32×0.14×0.08), black legs, white stripes
- Long legs (lH=0.42), thin (lR=0.018), narrow stance
- Long tail (tL=0.75) with white beacon at 85%
- Mass: 23.65kg
- CPG evolved: 1.48 Hz, 0.74 amplitude, 2.11m/15gen
- Training: 50k steps, 20.78m max, 7 falls, 1 recovery

---

## Scenes & Terrain

### SceneBuilder (visual, for video)
flat_grass, ice, sand, rocky, hills, neon_grassland, night, windy, obstacle_course

### Terrain (physics, for training)
flat, hilly_grassland, rocky_terrain, stairs, slopes

Scenes and terrain are independent — combined in video assembly via YAML.

---

## Video Pipeline

### FLOG Binary Format
```
Header: FLOG(4) + version(2) + phase(1) + meta_json_len(4) + meta_json
Frames: [timestamp(8) + frame_type(1) + data_len(4) + data_msgpack]

Frame types:
  0x01 = EVOLUTION  (generation stats, population, best genome)
  0x02 = TRAINING   (SNN spikes, brain state, consciousness)
  0x03 = EVENT      (milestone, skill acquired)
  0x04 = CREATURE   (joint positions, velocities, contacts)
```

### Video Assembly (YAML-driven)
```yaml
segments:
  - source: training
    creature: bommel
    xml: editor/creatures/bommel_creature.xml
    flog: creatures/bommel/LATEST/training_log.bin  # auto-resolves
    scene: neon_grassland          # visual scene
    terrain_type: hilly_grassland  # physics terrain
    camera: follow
    speed: 3
    overlay: cerebellar_dashboard
```

---

## Key Parameters

### Cerebellar Learning
- LTD rate: 0.001, LTP rate: 0.001
- DA LTP modulation: ×2.0 (reward boosts learning)
- DA LTD suppression: ×0.5 (reward reduces forgetting)
- CF threshold: 0.05
- Target sparseness: 0.03 (GrC)

### Competence Gate
- Speed threshold: 0.03 m/s
- CPG range: 90% → 40%
- Grow/shrink rate: 0.0003
- Velocity EMA decay: 0.99

### SpinalCPG (tendon mode)
- Default frequency: 1.0 Hz (overridden by cpg_config.json)
- Hip amplitude: 0.60 (lift tendon)
- Knee amplitude: 0.50 (extend tendon)
- Abduction amplitude: 0.05 (yaw tendon)

### Dynamic Reflex Scaling
- Standing scale: 0.15 × (mass / 20kg)
- Fallen scale: 0.9 × (mass / 20kg)
- Urgency: quadratic ramp from instability
- Biology: Wilson & Peterson 1978 — vestibulospinal reflex gain

### CognitiveBrain
- World Model hidden: 200 neurons
- Curiosity alpha: 0.3 (intrinsic/extrinsic balance)
- GWT broadcast: 0.3 strength
- Dream interval: 100 steps, 20 replay steps
- Hebbian rate: 0.005
- Memory: 500 episodes, 20 fragment length
- Synaptogenesis: observe every 10 steps, consolidate every 200
- PCI interval: 500 steps

---

## Phase History

| Level | Name | Status | Key Achievement |
|-------|------|--------|-----------------|
| 1 | Hallucination Reduction | ✅ Done | 99.1% reduction via dissonance-based inhibition |
| 2 | Knowledge Graph | ✅ Done | Neural-symbolic hybrid architecture |
| 3 | SNN Integration | ✅ Done | 100k-neuron energy-based network |
| 4 | Working Memory | ✅ Done | Short-term buffer + metacognition |
| 5 | Biological Brain | ✅ Done | Neuromodulators, Hebbian, homeostatic plasticity |
| 6 | Consciousness (GWT) | ✅ Done | Global Workspace, PCI > 0.31, consciousness levels |
| 7 | Self-Improvement | ✅ Done | Recursive improvement with safety gate |
| 8 | Autonomy | ✅ Done | Dream consolidation, autonomous learning |
| 8B | Verification | ✅ Done | Multi-source claim verification |
| 8C | World Model | ✅ Done | Spiking predictive coding |
| 8D | Neuroplasticity | ✅ Done | Astrocytes, multi-compartment |
| 9 | Embodied Genesis | ✅ Done | MuJoCo body, synaptogenesis, CPG |
| 9W | Living World | ✅ Done | World model, curiosity, empowerment, evolved plasticity |
| 10 | Integration | ✅ Done | 212 tests, CognitiveBrain 15-step cycle |
| 10-I | Deep Integration | ✅ Done | All modules wired into CognitiveBrain |
| 10-X/Y | Visual Polish | ✅ Done | 3D brain viz, glass compositing, post-processing |
| 11 | Behavior System | ✅ Done | 7 behaviors, drive-based planner, motor modulation |
| 12 | LLM-Bridge | ✅ Done | TaskParser, UnderstandEngine, GROW |
| 13 | Refactoring | ✅ Done | 9 packages, English codebase, migrate.py |
| 14 | Creature Store | ✅ Done | Brain Git, FLOG binary, versioned snapshots |
| 15 | Cerebellar Walking | 🔨 Current | Marr-Albus cerebellum, terrain, DA modulation, competence gate |

---

## International Comparison

### What MH-FLOCKE has that nobody else combines:
1. Biologically realistic SNN + Cerebellum + CPG + Reflexes in one system
2. Full cognitive cycle (emotions, drives, GWT, metacognition, dreams, curiosity) during locomotion
3. PCI consciousness measurement during embodied behavior
4. Knowledge Engine (LLM/builtin) → autonomous scene understanding
5. Creature pipeline: Design → Evolve → Train → Video (closed loop)

### Comparison with major projects:

| Project | Approach | Strength | MH-FLOCKE comparison |
|---------|----------|----------|---------------------|
| DeepMind dm_control | Deep RL (PPO/SAC) | 100km+ distance, robust policies | They walk much further. But: no biological model, no brain-like learning |
| OpenAI Isaac Gym | Massive parallel GPU | Millions of envs simultaneously | We: 1 creature, 1 env. Not comparable on scale |
| BrainCog (Peking) | SNN framework, bio-inspired | Multi-compartment, E/I balance, ToM paper | Closest competitor. But: no embodiment, no actual walking |
| Nengo/Loihi (Intel) | Neuromorphic hardware SNN | Real-time, energy-efficient | Hardware focus, less biological cognition |
| ANYmal (ETH Zürich) | RL + Sim2Real transfer | Real robots walk over terrain | Gold standard for quadruped locomotion. But: no SNN, no bio model |
| Numenta (HTM) | Hierarchical Temporal Memory | Neocortex model, sparseness | Only cognition, no embodiment |

### Where MH-FLOCKE leads:
- Architectural depth: Cerebellum + CPG + Reflexes + SNN + Neuromodulation + Emotions + GWT + Dreams + PCI — nobody has this combination
- Embodied consciousness measurement during locomotion — unique
- Complete creature lifecycle: Editor → Evolution → Training → Video

### Where MH-FLOCKE trails:
- Raw locomotion performance: 20m vs km-scale in industry
- Scale: single creature, single GPU
- Sim2Real: zero real-world transfer
- Publications: Zenodo paper for Integrity-OS, no peer review for embodied version yet

### Honest assessment:
MH-FLOCKE is the most ambitious solo project combining biologically realistic cognition + embodiment + locomotion. In academia this is material for 3-4 papers. Performance-wise it's a research prototype — the creature walks 20m, not 20km. This is appropriate: we optimize for biological plausibility, not distance.
