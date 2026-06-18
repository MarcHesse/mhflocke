# MH-FLOCKE

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19336894-blue.svg)](https://doi.org/10.5281/zenodo.19336894)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Biologically Grounded Embodied Cognition for Quadruped Locomotion Learning**

A simulated quadruped learns to refine its locomotion through a 15-step closed-loop cognitive
architecture that integrates spiking neural networks, a cerebellar forward model, a central
pattern generator, embodied emotions, and reward-modulated spike-timing-dependent plasticity
(R-STDP). The system is a **hybrid by design**: an innate CPG provides the base gait, and the
SNN + cerebellum learn to refine it on top — closer to how a young animal's brainstem and
spinal cord come pre-wired while the cerebellum and cortex calibrate them with experience.

> **This `main` branch is the Bittle-only public release (v0.8.0).** The active hardware
> platform is the [Petoi Bittle X](https://www.petoi.com/products/petoi-bittle-x). Earlier
> platforms (Unitree Go2 ablation, Freenove sim-to-real) are preserved as paper checkpoints —
> see the tags below.

> **📄 Paper Checkpoints**
>
> | Paper | Focus | Tag | Preprint |
> |-------|-------|-----|----------|
> | **Paper 1** — Ablation study | Go2 10-seed validation, B vs PPO | `v0.4.1-paper1` | [aiXiv 260301.000002](https://aixiv.science/abs/aixiv.260301.000002) |
> | **Paper 2** — Sim-to-Real | Freenove hardware transfer, Bridge v4.4, phototaxis | `v0.4.3-paper2` | [aiXiv 260409.000002](https://aixiv.science/abs/aixiv.260409.000002) |
>
> Use `git checkout v0.4.1-paper1` or `git checkout v0.4.3-paper2` to reproduce the paper
> results with their original Go2 / Freenove assets. A further snapshot of the last `main`
> that still carried both platforms is tagged `v0.7-go2-freenove-final`.

## What is hardwired vs. learned

MH-FLOCKE is explicitly a hybrid. Being clear about which parts are programmed and which parts
adapt is central to the project:

**Hardwired (programmed, present "from birth"):** the CPG gait oscillator, spinal reflexes
(righting, cross-extension, terrain), vestibular/light reflexes, the run-and-tumble navigation
state machine, the drive/behaviour planner, and the competence gate. The SNN does **not**
generate the gait — the CPG does.

**Learned (adapts through experience):** the SNN weights via R-STDP, the cerebellar correction
(Marr-Albus-Ito, climbing-fibre error → Purkinje learning), the CPG→actor handoff as the actor
proves competence, Hebbian co-activation, world-model prediction, emotional state, and
activity-dependent synaptogenesis.

The interesting behaviour lives in the interaction: reflexes provide the scaffold, learning
refines it. This is a design principle, not a limitation.

## Architecture

A 15-step closed-loop processing cycle runs at every simulation timestep:

```
SENSE → BODY SCHEMA → WORLD MODEL → EMOTIONS → MEMORY →
DRIVES → GLOBAL WORKSPACE → METACOGNITION → CONSISTENCY →
COMBINED REWARD → R-STDP LEARNING → SYNAPTOGENESIS →
HEBBIAN → DREAM MODE → NEUROMODULATION
```

Operating across nested timescales:

- **Spinal reflexes** (every step) — posture maintenance, stretch reflexes
- **Central Pattern Generator** — innate gait, competence-gated blending with the learned actor
- **Cerebellar forward model** — Marr-Albus-Ito framework, prediction-error-driven corrections
- **SNN with R-STDP** — Izhikevich neurons (≈535–756 depending on sensor configuration), reward-modulated STDP
- **Cognitive layers** — Global Workspace Theory, embodied emotions, episodic memory, drives
- **Meta-learning loop** — EpisodeAnalyzer, StrategyAdapter, CuriosityExplorer, HypothesisGenerator

The CPG provides a locomotion prior from step 1. As the SNN actor learns, a competence gate
transitions from ≈90% CPG toward ≈40% CPG / 60% actor — the CPG floor stays at 40% by design,
so the SNN refines the gait rather than replacing it. The creature walks from the start and
improves through learning.

## Quick Start

```bash
# Clone
git clone https://github.com/MarcHesse/mhflocke.git
cd mhflocke

# Install dependencies
pip install -r requirements.txt

# Train the Bittle on flat ground (OpenCat Trot gait + SNN refinement)
# --neural-cpg is REQUIRED for the Bittle: it loads the OpenCat Trot controller.
# Without it the SpinalCPG path is used and the robot falls immediately.
python scripts/train_baby.py --creature-name bittle --neural-cpg \
    --scene flat --steps 25000 --hardware-sensors --fresh --snn-substeps 10

# Analyze training data
python flog_server.py
# Open http://localhost:5050 for the dashboard
```

### Requirements

- Python 3.11+
- MuJoCo (via the `mujoco` pip package)
- PyTorch
- NumPy, msgpack

## Hardware — Petoi Bittle X

MH-FLOCKE targets the [Petoi Bittle X](https://www.petoi.com/products/petoi-bittle-x)
(BiBoard V0, ESP32, MPU6050 IMU, 8 leg servos). The brain runs on a host PC and talks to the
robot over a WiFi WebSocket, using the **same `src/brain/` code** as the simulator — one
codebase, two platforms. OpenCat's onboard balance is disabled so the motor commands pass
through unmodified.

The bridge needs `websocket-client`; the live `--dashboard` additionally needs `websockets`
(both are in `requirements.txt`). Each run writes per-step telemetry and the learned weights to
`creatures/bittle/bridge_<timestamp>/`.

```bash
# 1) Verify the WiFi / IMU channel — no motion, just reads the IMU.
#    Replace <robot-ip> with the Bittle's address on your network.
python scripts/bridge_bittle_wifi.py --ip <robot-ip>

# 2) Live gait: the innate OpenCat Trot with a fresh SNN learning on top via R-STDP.
#    No pre-trained brain needed — intrinsic drives (vestibular, curiosity) supply the reward.
python scripts/bridge_bittle_wifi.py --ip <robot-ip> --gait-loop --snn --duration 30

# 3) Same run, plus the live telemetry dashboard. Open the local file
#    src/viz/bridge_live.html directly in a browser (double-click or a file:// URL);
#    it connects to the bridge's WebSocket. Do NOT browse to localhost:5001 — that
#    port is a raw WebSocket, not a web page.
python scripts/bridge_bittle_wifi.py --ip <robot-ip> --gait-loop --snn --dashboard --duration 30
```

To load a simulation-trained brain instead of learning fresh, pass `--snn-brain <path/to/brain.pt>`
(sim-to-real transfer is active research — see the note below); `--cerebellum` adds the cerebellar
drift correction, and `--yaw-pid` closes an IMU yaw loop so the robot compensates mechanical
drift, surface, and battery level. The Bittle does **not** self-right from a supine fall — set it
upright by hand, or pass `--recover` to drive the OpenCat stand posture on side/forward falls
(learned weights are kept).

> **Sim-to-real note.** Closing the simulation-to-hardware gap on the Bittle is an active
> research arc, not a solved benchmark. Distances and roll amplitudes differ between sim and
> hardware, and the public code reflects work in progress rather than a finished result.

## Published validation (Paper 1, Unitree Go2)

The 10-seed ablation in Paper 1 was run on the Unitree Go2 (reproducible at tag
`v0.4.1-paper1`):

| Config | Distance (m) | Falls | Variance |
|--------|-------------|-------|----------|
| **B — SNN + Cerebellum** | **45.15 ± 0.67** | **0** | σ = 0.67 |
| A — CPG only | 40.73 ± 6.14 | 0.2 | σ = 6.14 |
| PPO baseline | 12.83 ± 7.78 | 0 | σ = 7.78 |

**How to read this honestly.** The B configurations train on an external, shaped reward —
`R_ext(t) = 0.8·v_forward + 0.2·upright` — applied via R-STDP on top of the innate CPG gait.
The large gap over PPO is **mostly the CPG locomotion prior**, not the SNN learning to walk by
itself: the SNN + cerebellum's own marginal contribution over CPG-only (A → B) is about **+11%
distance**, alongside a collapse in seed-to-seed variance (σ 6.14 → 0.67) and zero falls. The
phrases "from scratch", "no reward shaping", and "no end-to-end RL" do **not** apply to these
numbers.

A separate **intrinsic-reward line** (`train_baby.py --reward-blend 0`) learns from body signals
alone — vestibular comfort, curiosity, proprioceptive prediction error — with no external
reward. That configuration trades distance for autonomy and is **not** the source of the
benchmark numbers above.

## Ablation Design

Three configurations isolate component contributions:

- **A (CPG only)** — spinal reflexes + vestibular. The minimal baseline.
- **B (SNN + Cerebellum)** — adds R-STDP learning, cerebellar forward model, drives, behaviour planner.
- **C (Full system)** — all 15 cognitive steps including GWT, metacognition, dream mode, synaptogenesis.

## FLOG Dashboard

The training logger writes binary FLOG files (msgpack-encoded frames at 10-step intervals). The
standalone dashboard provides real-time analysis:

```bash
python flog_server.py
```

Features: distance/velocity charts, fall detection, CPG/actor weight tracking, cerebellar
prediction error, behavioural state timeline.

## Video Rendering

Render training runs with the full dashboard overlay and data-driven sonification:

```bash
# Render a Bittle training video
python scripts/render_bittle.py creatures/bittle/<run>/training_log.bin

# Instagram-format reel
python scripts/render_insta_reel_bittle.py creatures/bittle/<run>/training_log.bin

# Add data-driven audio (SNN crackle, CPG heartbeat, cerebellum tones, DA melody)
python scripts/sonify_flog.py --flog creatures/bittle/<run>/training_log.bin --speed 2 --mux output.mp4
```

> **Requires `ffmpeg`** on your PATH (external tool, not a pip package) for video
> encoding and audio muxing — install via `apt install ffmpeg`, `brew install ffmpeg`,
> or the Windows build from [ffmpeg.org](https://ffmpeg.org/download.html).

The Brain3D visualization in rendered videos shows actual SNN topology and spike activity from
the training data.

## Project Structure

```
mhflocke/
├── scripts/
│   ├── train_baby.py               # Baby-KI training loop (intrinsic + shaped reward)
│   ├── bridge_bittle_wifi.py       # Bittle hardware bridge (WiFi/WebSocket, same src/brain/)
│   ├── render_bittle.py            # Bittle training-video renderer (dashboard overlay)
│   ├── render_insta_reel_bittle.py # Instagram-format renderer
│   └── sonify_flog.py              # Data-driven audio from FLOG
├── src/
│   ├── body/                       # MuJoCo creature, terrain, OpenCat balance/controller
│   │   ├── bittle.py               # Bittle body model
│   │   ├── hardware_drift.py       # Mechanical-drift simulation (robot-agnostic, no-op without profile)
│   │   └── ...
│   ├── brain/                      # SNN, cerebellum, CPG, cognitive brain
│   │   ├── snn_controller.py       # Izhikevich SNN with R-STDP
│   │   ├── cerebellar_learning.py  # Marr-Albus-Ito cerebellum
│   │   ├── spinal_cpg.py           # Central pattern generator
│   │   ├── topology.py             # Shared population sizing (no MuJoCo dep)
│   │   ├── spatial_map.py          # Path integration + landmarks
│   │   ├── episode_analyzer.py     # Meta-learning: episode comparison
│   │   ├── strategy_adapter.py     # Meta-learning: parameter adaptation
│   │   ├── curiosity_hypothesis.py # Meta-learning: exploration + hypothesis generation
│   │   └── ...
│   ├── bridge/                     # Task parsing, scene generation, curriculum
│   ├── viz/                        # Brain3D, dashboard overlay
│   └── behavior/                   # Drive-based behaviour planner
├── creatures/
│   └── bittle/                     # Petoi Bittle X configuration
│       ├── bittle.xml              # MJCF (measured inertia)
│       ├── scene_mhflocke.xml      # Training scene
│       ├── profile.json            # Robot profile + SNN topology
│       ├── cpg_config.json         # Evolved CPG parameters
│       └── meshes_obj/             # Collision/visual meshes
├── docs/
│   └── FLOG_FORMAT.md
├── flog_server.py                  # FLOG analysis + dashboard
├── requirements.txt                # Simulator dependencies
└── requirements-pi.txt             # On-device dependencies (CPU-only)
```

## Documentation

Full documentation with architecture details, API references, mathematical formulations, and
biological background:

**[mhflocke.com/docs](https://mhflocke.com/docs/)**

## Papers

> **Paper 1 — Ablation Study:**
> MH-FLOCKE: Biologically Grounded Embodied Cognition Through a 15-Step Closed-Loop Architecture
> for Quadruped Locomotion Learning. Marc Hesse (2026).
> Preprint: [aiXiv 260301.000002](https://aixiv.science/abs/aixiv.260301.000002)

> **Paper 2 — Sim-to-Real:**
> MH-FLOCKE: Sim-to-Real Transfer of Biologically Grounded Spiking Neural Networks for Quadruped
> Locomotion. Marc Hesse (2026).
> Preprint: [aiXiv 260409.000002](https://aixiv.science/abs/aixiv.260409.000002)

## Videos

- [YouTube Channel: @mhflocke](https://www.youtube.com/@mhflocke)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The Unitree Go2 MJCF model used in the Paper 1 ablation (available at tag `v0.4.1-paper1`) is
from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) project (Google
DeepMind), derived from [Unitree Robotics](https://www.unitree.com/) URDF descriptions and
licensed under BSD-3-Clause.

## Named After

MH-FLOCKE is named after the author's late dog Flocke. The current test pilot is Mogli.

## Citation

```bibtex
@article{hesse2026mhflocke,
  title={MH-FLOCKE: Biologically Grounded Embodied Cognition Through a 15-Step Closed-Loop Architecture for Quadruped Locomotion Learning},
  author={Hesse, Marc},
  year={2026},
  note={Independent Researcher, Potsdam, Germany}
}
```

## Contact

- Website: [mhflocke.com](https://mhflocke.com)
- Email: info@mhflocke.com
- Reddit: [u/mhflocke](https://reddit.com/u/mhflocke)
