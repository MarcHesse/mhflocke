# CHANGELOG

All notable changes to MH-FLOCKE. Dates are YYYY-MM-DD.

## v0.8.2 — Substrate telemetry and learning-signal controls (2026-08-02)

The R-STDP learning signal was a single opaque number: when a run learned badly there was no
way to tell a bad gait from a substrate that had gone silent, saturated or drifted. This
release makes the substrate observable and the signal itself configurable. Every new flag
defaults to what the previous version did, and a regression run is bit-identical.

What the telemetry showed on the default settings, measured on a 20,000-step run: the
prediction-error branch of `apply_rstdp` is taken in 98.7 % of steps, so the reward term
contributes only `1 - pe_blend` = 10 % of the learning signal, and the threshold that is
supposed to select between the two branches separates nothing (mean |PE| is 0.204 against a
threshold of 0.05). Without the baseline the modulator is positive in 6.9 % of steps — a signal
with the same sign almost everywhere cannot distinguish a synapse that contributed from one that
did not. The new flags exist to change that; the defaults do not.

### Learning-signal controls (`scripts/train_baby.py` → 0.8.3)
- `--reward-baseline` / `--reward-baseline-alpha` — modulate with `R - E[R]` instead of the raw
  reward (Schultz 1997).
- `--pe-blend` — weight of the prediction error in the learning signal,
  `combined = (1-w)·R + w·(-PE)`. Default 0.9, which is the previous behaviour.
- `--eligibility-decay` — separate time constant for the eligibility trace. Default `None`
  keeps it tied to the STDP trace (0.95).
- `--eligibility-consume` — consumption factor after each `apply_rstdp`. Default 0.3.

### Substrate telemetry in the training log
Firing rates, threshold quantiles, silent and saturated fractions, homeostasis counts, error
and adaptation magnitudes, the reward baseline, and the split of the learning signal into its
reward and prediction-error parts are now written per logged step. The `snn_*` group in
`docs/FLOG_FORMAT.md` (doc version 1.4) lists them. No format change — these are optional keys,
older readers ignore them and older logs show a missing field as `—`.

### SNN controller (`src/brain/snn_controller.py` → 0.5.3)
- `get_health()` — controller self-report.
- Reward baseline and configurable PE/R mixing.
- Eligibility trace separated from the STDP trace, with its own decay and consumption.

### New analysis entry points
- `scripts/diagnose_homeostasis.py` — threshold distribution and weight growth from a saved
  `snn_state.pt`, without running training. Reports how much of the threshold range sits at the
  upper clamp, i.e. whether the homeostatic controller has run out of room.
- `scripts/analyze_substrate_health.py` — firing rates and threshold quantiles over time.
- `scripts/analyze_reward_gradient.py` — spread and structure of the learning signal.
- `scripts/analyze_locomotion.py` — path length vs. straight-line distance, speed over time.
- `scripts/smoke_test_release.py` — runs every entry point in the release once, in order: two
  short training runs (one with the new flags), the log check, the four analyses, the renderers
  and sonification, the dashboard server, and the hardware bridge's imports. It verifies that
  the new flags demonstrably change the modulator rather than just being accepted, and scans
  every output for leftover German text and private paths. A PASS means the release runs, not
  that a run learned anything.

### Fixed
- `--resume` crashed on a forward reference while restoring the spatial map (#231).

### What the new instrumentation shows

These are open findings, not fixes. They are listed because the tools exist to make them
visible, and because a run that looks healthy from the outside can be none of these things:

- **The threshold homeostasis is asymmetric.** `rate_error = actual - target` with a target of
  0.05 reaches +0.95 upward but only −0.05 downward, since a firing rate cannot go negative — a
  19:1 ratio. The controller corrects overactivity quickly and underactivity barely, so its
  fixed point sits below the target.
- **On the Bittle it does not correct at all.** `_homeostatic_update` zeroes its adjustment for
  Izhikevich neurons, and every Bittle population is Izhikevich — so the rate homeostasis is
  inert by construction. Measured over a 20,000-step run: the controller ran 1001 update cycles,
  reported a rate error of −0.039 throughout, and applied exactly 0.000000 every time. The
  firing rate sat at 0.005 against a target of 0.05, with 28–35 % of neurons silent. The
  thresholds still rose from 0.50 to 1.58 over the run, but that is `cerebellar_learning.py`
  writing granule-cell thresholds directly, not the homeostasis. `analyze_substrate_health.py`
  now reports this case explicitly instead of comparing the movement against a rule that was
  never applied.
- **The learning signal has a gradient; the modulator does not.** A first reading was that the
  intrinsic reward is close to constant on flat ground and therefore shapes nothing. Measured on
  a 20,000-step run, that is wrong: R has a coefficient of variation of 0.64 and a lag-one
  autocorrelation of 0.9965 — plenty of spread, and structure rather than noise. The problem is
  one stage later. What actually multiplies into the weight update is positive in only 6.9 % of
  steps, so in 93 % of steps it carries the same sign and cannot distinguish a synapse that
  contributed from one that did not. `--reward-baseline` addresses exactly this and is off by
  default.
- **Speed is not rewarded.** `corr(v, R) = +0.065` over the same run — walking faster does not
  make the reward better, which is consistent with a speed that stays flat.
- **The prediction-error branch is not a branch.** Independently reproduced on that run: the PE
  path is taken in 98.7 % of `apply_rstdp` calls, mean |PE| is 0.204 against a threshold of
  0.05, and the contributions split 29.6 % reward to 70.4 % prediction error. The threshold
  separates nothing and the reward-only path is effectively dead code.
- **Consistent with that, speed does not improve over a long run.** Measured on a 20,000-step
  flat run: 0.068 m/s in the first tenth, then 0.050–0.054 m/s for the remaining nine, with no
  upward trend. Nothing in the reward rewards being faster, and the robot does not get faster.
- **Two suspicions did not survive the measurement.** The concern that unchecked weight growth
  drives the thresholds into their upper clamp, leaving the short eligibility trace as the only
  brake, is not supported: on a 20,000-step network no threshold sits at either clamp and 90.9 %
  of weights lie between the bounds rather than pinned at them. Nor is the network saturated —
  the opposite: 0 % of neurons fire above 0.9, 28.6 % do not fire at all, the median rate is
  0.005 against a target of 0.05, and only 15 of 535 neurons sit in the target band.
- **Purkinje cells do not fire.** Over a 20,000-step run not one of the 16 Purkinje neurons
  spiked; over a 500-step run neither Purkinje nor DCN did. Every other population fires
  (input 19/19, Golgi 49/49, motor hidden 141/141, output 16/16, granule 172/278). Purkinje
  cells are the sole output of the cerebellar cortex, so this concerns the stage where
  Marr-Albus-Ito learning is supposed to take effect — while in the same run the parallel-fibre
  to Purkinje weight grew from 0.10 to 0.44. Whether the cerebellar correction is therefore
  inert depends on whether `cerebellar_learning` works on spikes or on membrane potential, which
  is not yet established. `check_flog.py` now reports this; the previous version passed it
  silently.
- **Distance figures need reading carefully.** "Max distance" does not distinguish walking in a
  circle from walking in a straight line. `analyze_locomotion.py` reports path length against
  straight-line displacement so the two cases can be told apart. On the same 20,000-step run:
  2.161 m of path for 1.039 m of displacement, a straightness of 0.48 — about half the distance
  walked goes sideways, and the lateral offset drifts steadily in one direction rather than
  oscillating.

### Also in this release

Modules that grew alongside the work since v0.8.1 and are shipped here without being broken out
individually: `mujoco_creature.py`, `mujoco_world.py`, `opencat_controller.py`, `terrain.py`,
`curiosity.py`, `spatial_map.py`, `drives.py`, `cognitive_brain.py`, `config.py`. The bulk of
`train_baby.py`'s diff is likewise wider than the flags described above. These were not tracked
change by change, so this entry names them rather than claiming a completeness it does not have.

- `scripts/render_bittle.py` → 0.1.1: the wall is drawn at the distance the run actually used,
  read from the log's meta, instead of a hardcoded 0.8 m — otherwise the robot appears to avoid
  empty space. The temporary scene XML is now written next to the original so the relative
  `<include>` and `meshdir` still resolve.

### Build
- `scripts/check_flog.py` → 0.2.0: per-population spike coverage is now counted over the whole
  run instead of the first frame. A first-frame count says nothing at a target rate of 0.05,
  where most neurons are silent in any given step — it read as if a population never fired. A
  population that stays silent for an entire run is now a WARN.
- The manifest generator now carries the version as a constant, lists `check_flog.py` as an
  entry point, and pins the two static HTML assets (`flog_dashboard.html`, `bridge_live.html`).
  All three had to be patched in by hand before, and the HTML files dropped out on every
  rebuild — the reason for the two hotfixes after v0.8.0.

---

## v0.8.1 — Real data in the video dashboard overlay (2026-06-20)

A few readouts in the rendered dashboard overlay were still placeholders, shown for
demonstration. As of v0.8.1 they are all wired through to the real values from the training
log — no placeholder values remain in the overlay. A reading that genuinely isn't available in a
run is shown as `—` rather than a stand-in.

- **New:** `scripts/check_flog.py` — a post-run check that a training log is well-formed and
  carries real values.
- Training-log format bumped, backward compatible (older logs still read fine).
- `src/viz/go2_dashboard.py` → `src/viz/bittle_dashboard.py`; package version → `0.8.1`.

---

## v0.8.0 — Bittle-only Public Release (2026-06-18)

The public `main` branch is now scoped to the Petoi Bittle X. The earlier Go2 and Freenove
platforms are preserved as tags, not deleted, and dropped from `main`.

### Platform freeze (preserve ≠ delete)
- **Go2 + Freenove kept as paper checkpoints**: `v0.4.1-paper1` (Go2 ablation), `v0.4.3-paper2`
  (Freenove sim-to-real). Reproduce either with `git checkout <tag>`.
- **Snapshot tag `v0.7-go2-freenove-final`** captures the last `main` that still carried both
  platforms (creatures, bridges, benchmarks) before the Bittle-only trim.
- 39 Go2/Freenove sources removed from `main`: `creatures/go2/`, `creatures/freenove/`, the Go2/
  Freenove renderers and bridges, the PPO baseline, the PCI benchmark, and `FREENOVE_PI_DEPLOY.md`.

### Manifest-driven public sync
- The public file set is now resolved from an explicit, generated manifest
  (`scripts/build_v080_manifest.py`): the transitive `import src.*` closure of the Bittle entry
  points plus the static Bittle assets — no wildcard copy, no Go2/Freenove code dragged along.
- Sync runs from that manifest (`scripts/sync_from_manifest.py`) with a private-pattern leak
  scan and an orphan report. Run outputs, checkpoints (`*.pt`/`*.bin`), the raw STL set, bridge
  telemetry, `*.bat`, and internal docs stay private.
- `.gitignore` extended for the Bittle-only split (raw `creatures/*/meshes/`, `creatures/*/bridge_*/`,
  `*.stl`, `*.bin`, `*.bat`).

### README rewritten Bittle-first + honest-claims sweep
- README is now Bittle-first; Go2/Freenove appear only as referenced paper tags.
- Overclaim language removed. The headline no longer says "no end-to-end RL required". The
  published B benchmark (Go2, 45.15 m) is now framed honestly: it trains on an external shaped
  reward `R_ext = 0.8·v_forward + 0.2·upright` via R-STDP on top of an innate CPG gait; the SNN +
  cerebellum's own marginal contribution over CPG-only is ≈+11% distance plus a variance collapse
  and zero falls. "From scratch / no reward shaping / no external reward" describe only the
  separate intrinsic-reward (`train_baby --reward-blend 0`) line, never the benchmark numbers.

### Package version
- `src/__init__.py` → `0.8.0` (closes the prior drift where `__init__` and tags disagreed with
  the CHANGELOG prose).

---

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
