# Changelog

## v0.7.0 (2026-04-23)

### Baby-KI: Intrinsic Reward Learning
- New `train_baby.py` training script with `--reward-blend` flag.
  0.0 = pure intrinsic reward (no external signal), 1.0 = v0.4.3 behavior.
- Intrinsic reward from body signals: vestibular discomfort, proprioceptive
  delta, curiosity, empowerment. The dog learns because falling feels bad
  and moving feels good — no reward shaping.
- Arousal Drive (RAS) tested and disabled: constant, oscillator, and additive
  variants all hurt locomotion (0.30-1.57m vs 3.37m baseline).

### Run-and-Tumble Chemotaxis (v0.4.8) — NEW
- Biologically correct navigation replaces continuous olfactory steering.
  Three-state machine: SNIFF (1 step), TUMBLE (12 steps), RUN (40 steps).
- Adaptive RUN duration: extends 1.5x when scent improves, max 120 steps.
- Result: **5.43m, 0 falls, 4 scent targets found** (vs 3.37m baseline,
  vs 0.51m with continuous steering which caused circling).
- Biology: Berg & Brown 1972, Catania 2013.
- Heading bug fixed: qpos[3] is quaternion W (~1.0), not yaw angle.
  All olfactory navigation failures since v0.4.6 traced to this.

### Sensory Environment v0.4.8
- Heading-aware scent respawn: sources spawn in ±30° cone ahead of creature.
- Scent radius 0.5→0.8m (sized for Freenove's short stride).
- Fade distance 5.0→3.0m (faster recycling of passed sources).

### v0.7.0 Pillars — Autonomous Self-Learning
- **Body Awareness** (`body_awareness.py`): Proprioceptive limb failure
  detection via command-position correlation. Auto-disconnects dead CPG
  oscillators. Thresholds: dead < 0.20, degraded < 0.35.
- **Spatial Map** (`spatial_map.py`): 2D cognitive map from IMU path
  integration. 20×20 grid, landmark memory, direction-to-home.
- **Gait Quality** (`gait_quality.py`): Periodicity (autocorrelation),
  jitter, height ratio, step amplitude. Feeds into training reward.
- **Directed Learning** (`directed_learning.py`): Systematic self-
  experimentation. Generates hypotheses, tests them, evaluates results.
  Autonomously confirmed frequency hypothesis (GQ 0.39→0.57).
- **Emotion Loop Connected**: Emotions receive gait quality, body awareness,
  obstacle distance, ball salience. Dead limb → fear, bad gait → negative
  valence, ball → joy.

### SNN Optimization for Pi4
- n_hidden=500 (was 172): continuous topology scaling, no cliff.
- Pi4 benchmark: 77 Hz in Bridge mode (12.9ms/step). 55 Hz full cognitive.
- Motor Hidden population (30% of hidden budget) with R-STDP.
- Izhikevich RS on all populations including input + output.

### Bilateral Symmetry Fix (v0.5.2)
- Random MH→Output init creates L/R asymmetry that R-STDP amplifies
  into systematic drift. Fix: average weights between bilateral leg pairs.
- DreamEngine buffer list→deque (O(N)→O(1) eviction). Fixed step time
  explosion from 20ms to 449ms over 33k steps.

### Dashboard Improvements
- Header spacing fix: creature name + scene text no longer overlap.
- Behavior widget: adaptive row layout scales to panel height.
- Scent source markers visible in rendered videos (cyan glow circles).
- "Targets found: N" counter in video overlay.
- Version override for FLOG metadata.

### XML Encoding Fix
- All `open(xml_path)` calls now use `encoding='utf-8'`.
  Emoji in creature.xml (🐾) crashed Python's charmap codec on Windows.

## v0.6.0 (2026-04-17)

### Vestibulospinal Reflex Fix (Issue #130) — CRITICAL
- **Dead-band on vestibular correction.** The v0.3.1 cycle-integrator reflex
  caused circling (gain=0.3: 36% uptime, 2 falls) or spinning (gain=0.6: 45
  revolutions) because gait-synchronous yaw oscillation was mistaken for drift.
  Fix: if |drift| < 0.3 rad/s, no correction is applied. The cycle-integrator
  still runs for logging and future learning — only the motor output is gated.
- Result: all gain values now produce identical straight walking (4.24m, 100%
  uptime, 0 falls). The reflex is silent during normal gait but activates on
  real disturbances (shove, slope, leg loss).
- Biology: vestibular afferent thresholds (Goldberg & Fernandez 1971)

### Step-Length Steering (Issue #131) — NEW
- **Replaced drive-mod + ABD-offset steering** with hip amplitude asymmetry
  in the Pattern Formation Layer. Inner legs get shorter steps, outer legs get
  longer steps. This is how real quadrupeds turn.
- Old steering: chaotic resonance at random values (R²=0.03), ABD had zero
  effect. New steering: monotonic from -3.7 to +3.7 revolutions, **R²=0.969**.
- Biology: Maes & Abourachid 2013, quadruped turning kinematics

### Ball Tracking — FIRST SUCCESSFUL GOAL-DIRECTED NAVIGATION
- Dog follows a moving figure-8 target for 10 full loops (120k steps).
  Mean distance 1.64m, 61% of time within 2m, 0 falls.
- With FR leg disabled: mean 1.76m, 63% within 2m, 0 falls.
  Dog tracks ball with missing leg — nearly identical to 4-leg performance.

### TectospinalBias v0.7.0 — NEW
- Mid-brain steering-bias adapter learns constant offset for body asymmetry.
  Converges in ~50 gait cycles. With FR leg disabled: improves tracking from
  63% to 72% within 2m, reduces max distance from 3.67m to 2.43m.
- Simple integrator with empirically confirmed sign convention.
  Replaces failed v0.6.0-v0.6.2 (sign-probing) versions.

### Full-Stack CognitiveBrain Integration
- First successful training run with all 15 CognitiveBrain steps active on
  a stable quadruped: Drives, Emotions, Memory, DreamEngine, Synaptogenesis,
  Metacognition, Astrocyte gating, PCI monitoring.
- 50k steps: 2.41m, 1 fall, 40098-step upright streak, actor competence 0.97.
- SNN takes over 60% of motor control from CPG (CPG weight 42%).
- Autonomous behavior selection: motor_babbling -> alert -> chase -> walk.

### Drive-Limits (Issue #124)
- Creature-specific freq_min/max and amp_min/max in profile.json.
  Prevents destabilizing frequency jumps between behaviors.

### Progress-Based Stuck Detection (Issue #125)
- Resets episode when max distance hasn't grown for 500 steps.
  Catches creatures that are upright but making no forward progress.

### Mogli Oscillator Version History
- v0.3.0: EMA-smoothed vestibulospinal reflex (superseded)
- v0.3.1: Phase-locked cycle-integrator (Ito 1984) (superseded)
- v0.3.2: Dead-band on correction output (fixes circling)
- v0.3.3: Step-length steering replaces drive-mod + ABD-offset

### Freenove Capabilities at v0.6.0
- Straight walking: 12.4m in 250s, 0 falls
- Controlled turning: R²=0.969, ±3.7 revolutions
- Ball tracking: 1.64m mean distance to moving target
- Leg-loss survival: walks and tracks with 3 legs
- Drift compensation: TectospinalBias learns in ~50 cycles
- Full CognitiveBrain loop: 15 steps, all systems active

## v0.5.0 (2026-04-14)

### Mogli Oscillator v0.1.0 (Issue #111) — NEW
- SNN-based CPG replacing the mathematical SpinalCPG (sin/cos)
- 24 Izhikevich neurons: 2 per joint (flexor/extensor) × 3 joints × 4 legs
- Half-center oscillation via mutual inhibition (Brown 1911)
- Inter-leg coupling produces walk gait: FL↔FR=-0.78, FL↔RR=+0.73
- Adaptive gain: developmental maturation from 3.0→8.0 over 2000 steps
  (biology: 5-HT from raphe nuclei increases motor neuron excitability)
- Dual steering: tonic drive asymmetry + output amplitude scaling
- CPG autonomy floor: spinal CPG maintains minimum rhythm independent
  of cortical commands (biology: decerebrate cats still walk)
- Learnable coupling weights prepared for R-STDP (not yet active)
- 50k result: 1.21m, 0 falls, actor competence 0.649, CPG→58%
- Enable with `--neural-cpg` flag

### Izhikevich Neuron Dynamics (Issue #104)
- Per-population Izhikevich (a,b,c,d) parameters replace uniform LIF-LTC
- 4 biologically accurate cell types: Regular Spiking (GrC), Intrinsically
  Bursting (GoC), Chattering (PkC), Rebound Burst (DCN)
- Output neurons remain LIF-LTC (motoneurons are RS/Tonic, not FS)
- Recovery variable `u` enables rebound bursting, pacemaker oscillation,
  and chattering dynamics that LIF cannot produce
- Backward-compatible: v0.4.x brains load correctly (defaults to LIF mode)

### DCN Rebound Burst Detection
- Post-inhibitory rebound bursting in Deep Cerebellar Nuclei
- When PkC inhibition releases after climbing fiber calcium decay,
  DCN fires a burst (REBOUND_BURST_GAIN=5.0) producing 5-10x stronger
  motor corrections than the v0.4.x tonic-minus-inhibition model
- Corrections improved from 0.01 (v0.4.3) to 0.06 (v0.5.0) in 50k training

### SNN Performance Optimizations (Issue #106)
- `torch.no_grad()` around step() core: 30-50% CPU speedup
- Dense matmul for n<500 neurons (L1 cache-friendly): 15-25% speedup
- Tau caching with dirty flag: recompute only on neuromodulator change
- Pre-allocated spike buffer: avoids .float() copy every step
- Combined: ~60% faster (benchmark: 1.2ms/step for 232 neurons)

### Hardware-Matched Obstacle Reflexes
- Brainstem-level reflexes identical in simulator and on hardware:
  REVERSE (<5cm), STOP (<10cm), SLOW (<30cm), TURN (proportional steering)
- Trigeminal avoidance turn reflex: robot turns away from obstacles
- Reticulospinal DCN→CPG inhibition pathway replaces trainer-side hack
- Wall pause: 50 steps visible standing before episodic reset

### Sim-to-Real Servo Compensation (Issue #105)
- Hardware correction gain ×5 to overcome SG90 servo dead band
- CPG frequency scale ×0.7 for servo tracking capability
- Cerebellar populations protected from R-STDP interference

### Go2 Stability Fix (Issue #110)
- Velocity-based stuck detection: resets robot when motionless (vel < 0.005)
  with upright < 0.7 for 200+ consecutive steps
- Go2 compatibility: `--legacy-cerebellum` flag for v0.4.x behavior

### Roadmap
- Mogli Oscillator: R-STDP coupling learning for automatic gait
  reorganization after limb loss
- Spatial planning: hippocampal place cells + prefrontal working memory
- Biological hierarchy: spinal reflexes → brainstem avoidance →
  cerebellar calibration → cortical planning

## v0.4.3 (2026-04-12)

### Obstacle Avoidance
- Ultrasonic sensor (HC-SR04 / MuJoCo rangefinder) on Channel 18
- Graded DCN output with asymmetric climbing fiber signals
- Additive CPG blending for cerebellar motor corrections
- Wall injection in MuJoCo scenes with episodic reset

### Sim-to-Real Transfer
- Unified Bridge v4.0: Pi and simulator share same src/brain/ codebase
- Brain3D visualization using real SNN topology and spike data
- Freenove demo: brain trained in MuJoCo, transferred to Raspberry Pi

## v0.4.2 (2026-04-06)

### Scalable Architecture
- Profile-driven SNN topology (profile.json per creature)
- Cerebellar populations scale proportionally for small neuron counts
- Hardware-matched sensor encoding (Bridge v2.5 layout)
- Freenove 232-neuron full cognitive loop
