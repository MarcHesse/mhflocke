# CHANGELOG

All notable changes to MH-FLOCKE. Dates are YYYY-MM-DD.

## v0.5.1 — PID Steering + Meta-Learning Loop (2026-05-05)

### Asymmetric Stride Steering (replaces Z-offset)
- **Z-offset steering proven useless** — Hardware isolation Test B: ±5mm Z-offset produces <5° effect against 70° mechanical drift. One measurement killed weeks of assumptions.
- **Asymmetric stride steering** — differential hip amplitude (left/right). Left legs longer stride → dog curves right. Biology: reticulospinal tract modulates stride asymmetry (Grillner 2003).
- **Hardware Test C validated**: Kp=0.03, Kd=0.015 reduces drift from 70° to 8.5°.
- **SpinalCPG v0.5.0**: `stride_scale = 1.0 ± steering_clamped` per side. Abduction stays symmetric.

### PID Closed-Loop Steering
- **IMU PID controller** replaces VOR-based steering in both sim and bridge. Camera provides target heading, PID on yaw error drives stride asymmetry. Closed loop — automatically compensates any mechanical drift, any surface, any battery level.
- **I-term added** — eliminates steady-state drift offset. Like cerebellar LTD accumulating corrections over time. Anti-windup ±30°.
- **Sign convention**: Sim negates PID output (`_steering = -_steering`) because MuJoCo yaw positive=left, CPG steering positive=right. Bridge does NOT negate (MPU6050 matches CPG, validated by Test C).
- **Sim gains**: Kp=0.08, Ki=0.005, Kd=0.02. **Bridge gains**: Kp=0.05, Ki=0.01, Kd=0.015.
- Sim result: dog navigates to 3 lights with measured drift profile, 0 falls, 0 resets.
- Hardware result: dog approaches light source from 0.52m to 0.17m in 60s with active drift compensation.

### CompetenceGate v0.5.0 — Stability-Primary
- **Gate no longer requires speed** — stability alone (upright + no falls) grows actor competence at 0.5× rate. Speed adds 1.5× bonus. Old speed-only gate blocked handoff when drift consumed locomotion energy.
- Actor reaches competence 1.0, CPG drops to 40% by step ~9k even with drift.

### Meta-Learning Loop (Phase A–D)
- **Phase A: EpisodeAnalyzer** (`src/brain/episode_analyzer.py`) — Compares successful vs unsuccessful navigation events. Identifies correlations between context variables (GQ, heading error, velocity, steering offset) and navigation success. Generates insights with confidence scores.
- **Phase B: StrategyAdapter** (`src/brain/strategy_adapter.py`) — Converts insights into parameter adjustments. Modulates RT run/tumble duration, PID Kp scaling, and exploration bias. Conservative: bounded changes, confidence-gated.
- **Phase C: CuriosityExplorer** (`src/brain/curiosity_hypothesis.py`) — World Model prediction error drives exploration. High PE → shorter runs, more tumbles (explore). Low PE → longer runs (exploit). Also uses SpatialMap grid coverage.
- **Phase D: HypothesisGenerator** (`src/brain/curiosity_hypothesis.py`) — Generates testable hypotheses from insights (e.g. "increase CPG frequency to 112%"). Designed to feed into DirectedLearning for autonomous testing.
- All four phases integrated in `train_baby.py`, logged via FLOG, with save/load persistence.
- Design reference: `docs/DESIGN_AUTONOMOUS_LOOP.md`.

### Renderer Fixes
- **World-centered minimaps** — trail shows actual path in world coordinates. Old robot-centered map was confusing.
- **FLOG format fix** — brain map reads `brain_visit_grid_b64` (base64) and `brain_landmarks_json` (JSON).
- **Reach radius circle** — shows 2m light detection radius on WORLD minimap.

### Bridge v4.4
- PID closed-loop phototaxis: Z-offset removed, asymmetric stride via `set_steering()`.
- `_pd_yaw_target` initialized to current IMU yaw (was 0.0 — caused saturation on startup).
- Salience threshold lowered to 0.02 (was 0.05).
- Stride and lift scale with `--speed` parameter (was hardcoded 12mm).
- PID gains tuned on hardware: Kp=0.05, Ki=0.01 (increased from 0.03/0.005 — hardware drift stronger than measured).

### Drift Profile Update
- `creatures/freenove/drift_profiles/measured_marc_01.json` updated to v3.
- Yaw drift rate corrected from -0.4 to -1.5 deg/s based on walking-load measurements.
- Under walking load, servo asymmetry creates stronger drift than at rest.
- Added `pid_gains_hardware` section documenting tuned PID parameters.

### Bug Fixes (9 found this session)
1. Z-offset too weak (hardware Test B)
2. Bridge comment wrong ("Z+ turns left" — actually turns right)
3. `compute_tendon()` routing — steering never reached CPG
4. VOR display showed proxy, not actual `_steering_offset`
5. CompetenceGate speed-only blocked actor handoff with drift
6. Bridge PD init `_pd_yaw_target = 0.0` instead of current yaw
7. MuJoCo yaw sign inverted vs CPG — needed `_steering = -_steering` in sim only
8. `abs(olf_steer) > 0.05` threshold prevented target updates near light
9. Renderer FLOG format mismatch (grid_visited_X_Y vs brain_visit_grid_b64)

### New Files
- `src/brain/episode_analyzer.py` — Meta-Learning Loop Phase A
- `src/brain/strategy_adapter.py` — Meta-Learning Loop Phase B
- `src/brain/curiosity_hypothesis.py` — Meta-Learning Loop Phase C+D

### Changed Files
- `src/brain/spinal_cpg.py` v0.4.1 → v0.5.0
- `scripts/freenove_bridge.py` v4.3 → v4.4
- `scripts/train_baby.py` — PID, stability gate, Meta-Learning Loop A-D
- `scripts/render_freenove.py` — world-centered maps, FLOG fix, reach radius
- `creatures/freenove/drift_profiles/measured_marc_01.json` — v3

---

## v0.5.0 — Sim-to-Real + LightMemory + Hardware Drift (2026-05-02)

### LightMemory — Spatial Yaw Recall
- **LightMemory class** in `freenove_bridge.py` — when light disappears, dog remembers last known yaw angle and steers back. Three states: TRACKING → RETURNING → LOST.
- **`z_sign` parameter** — hardware Z-convention is inverted vs simulator. `z_sign=+1.0` for Freenove hardware, `-1.0` for MuJoCo. Measured empirically from hardware CSV data.
- **Target yaw includes heading offset** — stores direction TO the light, not just body orientation. `target_yaw = body_yaw + heading * HALF_FOV`.
- Biology: Head Direction cells in postsubiculum maintain heading representation without visual landmarks.

### Hardware Drift Simulation
- **`src/body/hardware_drift.py`** — injects measured mechanical drift into MuJoCo via `xfrc_applied`. No-op without profile (zero cost). Public feature: any user can create drift profiles for their robot.
- **Drift profiles** in `creatures/freenove/drift_profiles/`: measured (Marc's unit), synthetic (left drift), control (no drift).
- **Calibrated**: `_TORQUE_PER_DEG_S = 0.05` (empirically measured via `calibrate_drift.py`). Produces -2.25 deg/s in simulator.
- **Hardware measurement**: actual drift is -0.4 deg/s (previous -2.0 estimate was inflated by accumulated drift_bias). Steering effectiveness: -0.22 deg/s per mm Z-offset.

### Neuron Alignment (232 → 560)
- **`topology.py` v0.7.1** — continuous scaling, 70% cerebellum / 30% motorcortex split. No more n_hidden>=500 cliff. Same formula for simulator and hardware.
- **Bridge v4.2**: `build_freenove_snn()` now uses n_hidden=500 = 560 total neurons. Motor Hidden (136 neurons) for R-STDP learning. Bilateral MH→Output symmetry enforced.
- **Per-population Izhikevich** on all populations including Output (RS, not FS).

### Spatial Map Persistence
- **`spatial_map.py`** — `state_dict()` / `load_state_dict()` methods. Grid, landmarks, trail, position, heading all persisted.
- Saved in checkpoint.pt, restored on `--resume`.
- Light source observed as landmark with `valence=1.0`.
- **Bridge v4.3**: hardware also runs SpatialMap. Path integration from CPG-derived velocity proxy (~0.04 m/s at full output, scaled by inhibition) and IMU yaw. Map state persisted to `~/spatial_map.pt`, restored on next run unless `--fresh`.

### FLOG Format v1.2 — Phototaxis Navigation Fields
- **FRAME_CREATURE** (every 10 steps) gains `dist_to_light` (m, sentinel `-1.0` if no light) and `intent_yaw_rate` (current motor steering command). Lets a renderer draw a per-frame heading-to-target arrow on the world minimap.
- **FRAME_TRAINING** (every `log_every` steps) gains:
  - Ground truth from physics: `pos_x`, `pos_y`, `dist_to_light`, `heading_to_light` (sentinel `-999.0` if no light), `intent_yaw_rate`.
  - Brain map snapshot from `SpatialMap`: `brain_pos_x`, `brain_pos_y`, `brain_pos_error` (drift between belief and ground truth), `brain_landmarks_json` (list of known landmarks with `confidence`, `valence`, `visit_count`, `last_seen_step`), `brain_visit_grid_b64` (uint8-quantized 20×20 visit heatmap, base64-encoded), `brain_grid_shape`.
- On hardware no ground truth exists; the ground-truth fields use sentinel values and only the brain-map fields contain meaningful data.
- Approx. 1.5 KB added per training-stats snapshot. ~150 KB per 100k-step run — negligible.
- **`docs/FLOG_FORMAT.md`** bumped to v1.2 with new sections "Phototaxis Navigation" and "Brain Map" (landmark JSON schema + Python decode snippet for the visit grid).
- **Bridge v4.3 CSV**: hardware CSV gets the same fields (`pos_x/y` and `dist_to_light` use sentinel values; `brain_pos_x/y` from SpatialMap). Brain-map snapshots written every 1000 steps to a JSONL sidecar (`<csv_stem>.brain.jsonl`), one snapshot per line, identical schema to the FLOG brain-map fields.

### Bug Fixes
- **Duplicate `step+=1`** in Bridge v4.1 — steps were counted double. Fixed.
- **Drift-bias accumulation** — old bias (+5.7) was itself a drift source. Must be reset between experiments.
- **VOR drift-learning loop**: `_YAW_PER_MM` was `+0.15` (guessed); hardware measurement gives `-0.22 deg/s/mm`. Sign was inverted, magnitude was off. Loop now also subtracts the expected steering rotation from the measured yaw delta before averaging — it learns only the *unmodelled* drift, not its own commanded rotation.

### New Tools
- `scripts/calibrate_drift.py` — measures actual yaw drift rate in simulator
- `scripts/smoke_test_phototaxis.py` — 18-point integration test for new components

### Files Changed
- `scripts/freenove_bridge.py` v4.1 → v4.3 (LightMemory, SpatialMap on hardware, CSV phototaxis fields, JSONL brain-map sidecar)
- `scripts/train_baby.py` — --drift-profile, LightMemory, spatial map checkpoint, light landmark, FLOG phototaxis/brain-map fields
- `src/brain/topology.py` v0.7.0 → v0.7.1
- `src/brain/spatial_map.py` — persistence
- `src/body/hardware_drift.py` — NEW
- `creatures/freenove/drift_profiles/` — NEW (3 profiles)
- `docs/FLOG_FORMAT.md` v1.1 → v1.2 — phototaxis navigation + brain map sections
- `.gitignore` — cleaned

---

## v0.4.8 — Phototaxis Navigation + 6× Performance Fix (2026-04-25)

### Phototaxis Navigation
- **VOR (Vestibulo-Ocular Response) steering** — hardwired reflex turns dog toward light source
- **Waypoint system** — fixed positions with relative spawning, respawn on miss (4.5m)
- **Run-and-Tumble integration** — RT state machine triggers Tumbles during navigation
- **Geometric light gradient** — `1/(0.5+dist)²` bilateral brightness computation
- **First successful navigation**: sf:2 (two waypoints reached), VOR up to +0.54
- **MuJoCo light body** — emissive sphere with spotlight injected into scene

### Performance (6× Speedup)
- **Root cause found**: Synaptogenesis `ExperienceBuffer` O(N²) clustering over 5000 entries
- **Fix**: `buffer.clear()` after consolidation, max_size 5000→500
- **Dense SNN threshold**: 500→600 (Freenove 560 neurons now uses fast dense path)
- **R-STDP lazy dirty flag**: dense matrix rebuilt only in next forward(), not after every update
- **Memory fixes**: deque replacements for list.pop(0) in world_model, spatial_map, directed_learning, embodied_emotions
- **Result**: 7 sps → 54 sps stable over 100k steps. 100k run in 30 min instead of 5+ hours.

### Video Rendering
- **Mini-map overlay** in render_freenove.py — bottom-left, shows trail + light waypoints
- **Instagram Reel renderer** (render_insta_reel.py) — 3:4 format
- **Thumbnail generator** (render_phototaxis_thumb.py)

### Documentation
- **HONEST_CLAIMS.md** — complete documentation of hardwired vs. learned components

---

## v0.4.5 — Baby-KI Autonomous Learning (2026-04-21)

- **`train_baby.py` v0.8.0-alpha** — autonomous learning without external reward
- **Arousal Drive (RAS)** in CognitiveBrain — `get_intrinsic_reward()` with `--reward-blend`
- **Drift root cause identified**: bilateral MH→Output weight asymmetry amplified by R-STDP
- **Fix**: bilateral symmetry enforcement at initialization (v0.5.2)
- **Cognitive brain v0.4.3**: intrinsic reward, Arousal Drive, deque fixes

---

## v0.4.2 — Freenove Sim-to-Real Unified Codebase (2026-04-11)

- **Bridge v4.0**: `freenove_bridge.py` imports `src/brain/` directly. Same PyTorch SNN on Pi and simulator.
- **`topology.py`**: Shared cerebellar population computation without MuJoCo dependency.
- **Brain3D visualization**: Population-aware layout from actual SNN topology.
- **Live dashboard**: Web-based real-time display of cerebellar populations on Pi.
- **Pi deployment guide**: `docs/FREENOVE_PI_DEPLOY.md`, `requirements-pi.txt`

---

## v0.4.0 — Freenove Hardware Integration (2026-04-06)

- Initial Freenove integration: Bridge v2.5, IMU support
- First real-world run: 8.2m, 0 falls
- Brain persistence across sessions (18,746 steps over 3 sessions)
- A/B test: fresh vs loaded brain — key paper finding
- Demo video: [youtube.com/watch?v=7iN8tB2xLHI](https://youtube.com/watch?v=7iN8tB2xLHI)

---

## v0.3.4 — Go2 Ablation Study (2026-03-28)

- **10-seed ablation**: B1 (SNN+Cerebellum) 45.15±0.67m vs PPO 12.83±7.78m (3.5×, low variance)
- B=C identity confirmed as honest architectural result
- Recovery Learning (4-phase RightingReflex, always-on)
- Ball interaction: asymmetric prediction error (loss aversion) + CPG proximity brake
- arXiv submission prepared (cs.NE + cs.RO + cs.AI)
- aiXiv preprint: [aixiv.science/abs/aixiv.260301.000002](https://aixiv.science/abs/aixiv.260301.000002)

---

## v0.3.0 — Go2 Integration (2026-02)

- Go2 integrated from MuJoCo Menagerie
- PD controller bridge (CPG outputs as torques)
- SNN spectral entropy analysis (3.9→6.9 bits)
- Hesse Neuron prototype (phase-coupled oscillator + Phase-STDP)

---

## v0.2.0 — Mogli Quadruped (2026-01)

- Custom quadruped model (Mogli) in MuJoCo
- 15-step cognitive cycle fully operational
- Global Workspace, episodic memory, drives, metacognition
- Video pipeline (Playwright renderer)

---

## v0.1.0 — Initial Release (2025-12)

- 100k-neuron SNN with STDP, homeostatic plasticity, neuromodulators
- Astrocyte gating, Phase 10 CognitiveBrain
- Integrity-OS hallucination prevention (Zenodo DOI 10.5281/zenodo.18450340)
