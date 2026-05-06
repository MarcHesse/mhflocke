# MH-FLOCKE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19336894.svg)](https://doi.org/10.5281/zenodo.19336894)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Biologically Grounded Embodied Cognition for Quadruped Locomotion Learning**

A simulated quadruped learns to walk through a 15-step closed-loop cognitive architecture integrating spiking neural networks, cerebellar forward models, central pattern generators, embodied emotions, and reward-modulated spike-timing-dependent plasticity — no end-to-end RL required.

> **📄 Paper Checkpoints**
>
> This repository accompanies two preprints:
>
> | Paper | Focus | Preprint | Data |
> |-------|-------|----------|------|
> | **Paper 1** — Ablation study | Go2 10-seed validation, B1 vs PPO 3.5× | [aiXiv 260301.000002](https://aixiv.science/abs/aixiv.260301.000002) | [Zenodo 10.5281/zenodo.19336894](https://doi.org/10.5281/zenodo.19336894) |
> | **Paper 2** — Sim-to-Real | Freenove hardware transfer, Bridge v4.x | *In preparation* | This repo (`creatures/freenove/`) |
>
> The code at tag `v0.3.4` corresponds to the ablation results in Paper 1.
> The code at `main` includes sim-to-real extensions (Bridge, phototaxis, LightMemory).

## Key Results (10-Seed Validation, Unitree Go2)

| Config | Distance (m) | Falls | Variance |
|--------|-------------|-------|----------|
| **B1 SNN+Cerebellum** | **45.15 ± 0.67** | **0** | **σ = 0.67** |
| A1 CPG only | 40.73 ± 6.14 | 0.2 | σ = 6.14 |
| PPO Baseline | 12.83 ± 7.78 | 0 | σ = 7.78 |

**3.5x faster learning than PPO with 11.6x lower variance** at identical sample budgets (50k steps). Zero falls across all 10 seeds.

## Quick Start

```bash
# Clone
git clone https://github.com/MarcHesse/mhflocke.git
cd mhflocke

# Install dependencies
pip install -r requirements.txt

# Train Go2 on flat terrain (10k steps, ~20 min)
python scripts/train_v032.py \
    --creature-name go2 \
    --scene "walk on flat meadow" \
    --steps 10000 \
    --skip-morph-check \
    --no-terrain \
    --auto-reset 500 \
    --seed 42

# Analyze training data
python flog_server.py
# Open http://localhost:5050 for the dashboard
```

### Requirements

- Python 3.11+
- MuJoCo (included via `mujoco` pip package)
- PyTorch
- NumPy, msgpack

## Freenove Robot Dog — Sim-to-Real

MH-FLOCKE runs on real hardware using the Freenove Robot Dog Kit (FNK0050, ~100€).
The Raspberry Pi 4 runs the **same SNN and cerebellum code** as the MuJoCo simulator — one codebase, two platforms. A brain trained in simulation transfers directly to the real robot.

### Hardware Setup

- Kit: [Freenove FNK0050](https://www.freenove.com/fnk0050) (~100€)
- Compute: Raspberry Pi 4 (2GB+ RAM)
- 12 SG90 servos, PCA9685 driver, MPU6050 IMU, Pi Camera v2
- SNN: 560 neurons (48 MF + 269 GrC + 47 GoC + 24 PkC + 24 DCN + 136 MH + 12 OUT)
- Control loop: ~15 Hz with PyTorch CPU-only

### Running on Pi

```bash
# Install PyTorch CPU-only
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Deploy brain code (same src/brain/ as simulator)
scp -r src/brain/ admin@<pi-hostname>:~/mhflocke/src/brain/
scp scripts/freenove_bridge.py admin@<pi-hostname>:~/mhflocke/scripts/

# Walk with SNN + Cerebellum
python3 scripts/freenove_bridge.py --gait walk --snn --fresh --verbose --duration 120

# Phototaxis: navigate toward a flashlight
python3 scripts/freenove_bridge.py --gait walk --snn --fresh --phototaxis --verbose --duration 60

# Transfer sim-trained brain
scp creatures/freenove/brain/brain.pt admin@<pi-hostname>:~/brain.pt
python3 scripts/freenove_bridge.py --gait walk --snn --verbose --duration 120
```

### Hardware Drift Simulation

Real robots have mechanical asymmetries that cause drift. MH-FLOCKE includes a drift simulation module for testing robustness in the simulator:

```bash
# Train with measured hardware drift profile
python scripts/train_baby.py --creature-name freenove --phototaxis \
    --drift-profile creatures/freenove/drift_profiles/measured_marc_01.json \
    --hardware-sensors --no-terrain --steps 50000
```

Drift profiles in `creatures/freenove/drift_profiles/` describe per-robot mechanical characteristics. Create your own profile from hardware measurements.

### Live Dashboard

The Bridge includes a web dashboard showing real-time SNN activity:

```bash
python3 scripts/freenove_bridge.py --gait walk --snn --dashboard --verbose --duration 300
# Open http://<pi-hostname>:8080
```

Displays all 6 cerebellar populations with live spike data, servo angles, competence gate, and neuromodulation levels.

Full deployment guide: [docs/FREENOVE_PI_DEPLOY.md](docs/FREENOVE_PI_DEPLOY.md)

## Architecture

MH-FLOCKE implements a 15-step closed-loop processing cycle that runs at every simulation timestep (200 Hz):

```
SENSE → BODY SCHEMA → WORLD MODEL → EMOTIONS → MEMORY →
DRIVES → GLOBAL WORKSPACE → METACOGNITION → CONSISTENCY →
COMBINED REWARD → R-STDP LEARNING → SYNAPTOGENESIS →
HEBBIAN → DREAM MODE → NEUROMODULATION
```

The architecture operates across nested timescales:

- **Spinal reflexes** (every step) — posture maintenance, stretch reflexes
- **Central Pattern Generator** — innate gait patterns, competence-gated blending with learned actor
- **Cerebellar forward model** — Marr-Albus-Ito framework, prediction error-driven motor corrections
- **SNN with R-STDP** — 560+ Izhikevich neurons, reward-modulated spike-timing-dependent plasticity
- **Cognitive layers** — Global Workspace Theory, embodied emotions, episodic memory, motivational drives

The CPG provides a locomotion prior from step 1. As the SNN actor learns, a competence gate smoothly transitions from 90% CPG to 40% CPG / 60% actor. The creature walks immediately and improves through learning — no random exploration phase required.

## Ablation Design

Three configurations isolate component contributions:

- **A (CPG only)** — Spinal reflexes + vestibular. The anencephalic baseline.
- **B (SNN + Cerebellum)** — Adds R-STDP learning, cerebellar forward model, drives, behavior planner.
- **C (Full system)** — All 15 cognitive steps including GWT, metacognition, dream mode, synaptogenesis.

Each tested on flat and hilly terrain, 10 random seeds, yielding 80 total runs.

## FLOG Dashboard

The training logger writes binary FLOG files (msgpack-encoded frames at 10-step intervals). The standalone dashboard provides real-time analysis:

```bash
python flog_server.py
```

Features: distance/velocity charts, fall detection, CPG/actor weight tracking, cerebellar prediction error, behavioral state timeline.

## Video Rendering

Render training runs with the full dashboard overlay and data-driven sonification:

```bash
# Render Go2 training video
python scripts/render_go2_mujoco.py creatures/go2/v034_.../training_log.bin

# Render Freenove training video
python scripts/render_freenove.py creatures/freenove/v043_.../training_log.bin

# Add data-driven audio (SNN crackle, CPG heartbeat, cerebellum tones, DA melody)
python scripts/sonify_flog.py --flog creatures/.../training_log.bin --speed 2 --mux output.mp4
```

The Brain3D visualization in rendered videos shows actual SNN topology and spike activity from the training data, with correct population sizes for each creature.

## Project Structure

```
mhflocke/
├── scripts/
│   ├── train_v032.py           # Go2 training loop
│   ├── train_baby.py           # Baby-KI autonomous learning (Freenove)
│   ├── freenove_bridge.py      # Pi hardware bridge v4.2 (unified codebase)
│   ├── calibrate_drift.py      # Hardware drift calibration
│   ├── smoke_test_phototaxis.py # Component integration test
│   ├── render_freenove.py      # Freenove video renderer
│   ├── render_go2_mujoco.py    # Go2 video renderer
│   └── sonify_flog.py          # Data-driven audio from FLOG
├── src/
│   ├── body/                   # MuJoCo creature, terrain, genome
│   │   ├── hardware_drift.py   # Mechanical drift simulation
│   │   └── ...
│   ├── brain/                  # SNN, cerebellum, CPG, cognitive brain
│   │   ├── snn_controller.py   # Izhikevich SNN with R-STDP
│   │   ├── cerebellar_learning.py  # Marr-Albus-Ito cerebellum
│   │   ├── topology.py         # Shared population sizing (no MuJoCo dep)
│   │   ├── spatial_map.py      # Path integration + landmarks
│   │   └── ...
│   ├── viz/                    # Brain3D, dashboard overlay
│   └── behavior/               # Drive-based behavior planner
├── creatures/
│   ├── go2/                    # Unitree Go2 configuration
│   └── freenove/               # Freenove Robot Dog configuration
│       ├── drift_profiles/     # Hardware drift characterization
│       ├── dashboard/          # Live web dashboard (real SNN data)
│       ├── profile.json        # Robot profile + SNN topology
│       └── servo_config.json   # Channel mapping
├── docs/
│   ├── FREENOVE_PI_DEPLOY.md   # Complete Pi deployment guide
│   ├── ARCHITECTURE.md
│   └── FLOG_FORMAT.md
├── flog_server.py              # FLOG analysis + dashboard
├── requirements.txt            # Simulator dependencies
└── requirements-pi.txt         # Raspberry Pi dependencies (CPU-only)
```

## Documentation

Full documentation with architecture details, API references, mathematical formulations, and biological background:

**[mhflocke.com/docs](https://mhflocke.com/docs/)**

25 pages covering: Architecture, SNN Controller, R-STDP, Cerebellum, CPG, Task Prediction Error, Reflexes, Emotions & Drives, Training Pipeline, FLOG Format, World Model, Global Workspace, Body Schema, Memory, Metacognition, and more.

## Papers

> **Paper 1 — Ablation Study:**
> MH-FLOCKE: Biologically Grounded Embodied Cognition Through a 15-Step Closed-Loop Architecture for Quadruped Locomotion Learning.
> Marc Hesse (2026). Preprint: [aiXiv 260301.000002](https://aixiv.science/abs/aixiv.260301.000002)

> **Paper 2 — Sim-to-Real:** *In preparation.*
> Freenove Robot Dog hardware transfer, phototaxis navigation, LightMemory spatial recall.

## Videos

- [Freenove Robot Dog — SNN on Real Hardware](https://www.youtube.com/watch?v=7iN8tB2xLHI)
- [Video #3: Go2 Ball Interaction](https://www.youtube.com/watch?v=Jo7UM6pEFMg)
- [YouTube Channel: @mhflocke](https://www.youtube.com/@mhflocke)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## Acknowledgments

The Unitree Go2 MJCF model is from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) project (Google DeepMind), derived from [Unitree Robotics](https://www.unitree.com/) URDF descriptions. Licensed under BSD-3-Clause — see `creatures/go2/LICENSE_unitree_go2`.

## Named After

MH-FLOCKE is named after the author's late dog Flocke. The current test pilot is Mogli.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The Unitree Go2 model files in `creatures/go2/` are licensed under BSD-3-Clause — see `creatures/go2/LICENSE_unitree_go2`.

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
