# MH-FLOCKE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19336894.svg)](https://doi.org/10.5281/zenodo.19336894)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Biologically Grounded Embodied Cognition for Quadruped Locomotion Learning**

A simulated quadruped learns to walk through a 15-step closed-loop cognitive architecture integrating Izhikevich spiking neural networks, cerebellar forward models with DCN rebound bursting, central pattern generators, embodied emotions, and reward-modulated spike-timing-dependent plasticity — no end-to-end RL required.

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
- 12 SG90 servos, PCA9685 driver, MPU6050 IMU, HC-SR04 ultrasonic
- SNN: 232 Izhikevich neurons (48 MF + 106 GrC + 18 GoC + 24 PkC + 24 DCN + 12 OUT)
- Per-population neuron dynamics: Regular Spiking, Intrinsically Bursting, Chattering, Rebound Burst, Fast Spiking
- Obstacle reflexes: hardware-matched brainstem-level STOP/SLOW/REVERSE/TURN
- Control loop: 29Hz with PyTorch CPU-only

### Running on Pi

```bash
# Install PyTorch CPU-only
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Deploy brain code (same src/brain/ as simulator)
scp -r src/brain/ admin@<pi-hostname>:~/src/brain/
scp scripts/freenove_bridge.py admin@<pi-hostname>:~/

# Walk with SNN + Cerebellum
python3 freenove_bridge.py --gait walk --snn --fresh --verbose --duration 120

# Transfer sim-trained brain
scp creatures/freenove/brain/brain.pt admin@<pi-hostname>:~/brain.pt
python3 freenove_bridge.py --gait walk --snn --verbose --duration 120
```

### Live Dashboard

The Bridge includes a web dashboard showing real-time SNN activity:

```bash
python3 freenove_bridge.py --gait walk --snn --dashboard --verbose --duration 300
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

The architecture operates across nested timescales, following the biological hierarchy of motor control:

- **Spinal reflexes** (every step) — posture maintenance, stretch reflexes, muscle tone
- **Brainstem reflexes** — obstacle avoidance (stop, slow, reverse, turn), righting reflexes
- **Central Pattern Generator** — innate gait patterns, competence-gated blending with learned actor
- **Cerebellar forward model** — Marr-Albus-Ito framework with DCN rebound bursting, prediction error-driven motor corrections via reticulospinal pathway
- **SNN with R-STDP** — 5000+ Izhikevich neurons with per-population dynamics, reward-modulated spike-timing-dependent plasticity
- **Cognitive layers** — Global Workspace Theory, embodied emotions, episodic memory, motivational drives

The CPG provides a locomotion prior from step 1. As the SNN actor learns, a competence gate smoothly transitions from 90% CPG to 40% CPG / 60% actor. The creature walks immediately and improves through learning — no random exploration phase required.

### Roadmap: Spatial Planning

The current architecture implements biological layers 1-3 (spinal, brainstem, cerebellum). We are working toward a spatial planning layer inspired by hippocampal place cells and prefrontal cortex working memory. This will enable the robot to build a cognitive map of its environment and plan paths around obstacles — moving from reactive avoidance to goal-directed navigation.

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
python scripts/render_freenove.py creatures/freenove/v034_.../training_log.bin

# Add data-driven audio (SNN crackle, CPG heartbeat, cerebellum tones, DA melody)
python scripts/sonify_flog.py --flog creatures/.../training_log.bin --speed 2 --mux output.mp4
```

The Brain3D visualization in rendered videos shows actual SNN topology and spike activity from the training data, with correct population sizes for each creature.

## Project Structure

```
mhflocke/
├── scripts/
│   ├── train_v032.py           # Main training loop
│   ├── freenove_bridge.py      # Pi hardware bridge (unified codebase)
│   ├── freenove_calibrate.py   # Servo calibration tool
│   ├── render_go2_mujoco.py    # Go2 video renderer
│   ├── render_freenove.py      # Freenove video renderer
│   └── sonify_flog.py          # Data-driven audio from FLOG
├── src/
│   ├── body/                   # MuJoCo creature, terrain, genome
│   ├── brain/                  # SNN, cerebellum, CPG, cognitive brain
│   │   ├── snn_controller.py   # Izhikevich SNN with R-STDP
│   │   ├── cerebellar_learning.py  # Marr-Albus-Ito cerebellum
│   │   ├── topology.py         # Shared population sizing (no MuJoCo dep)
│   │   └── ...
│   ├── viz/                    # Brain3D, dashboard overlay
│   └── behavior/               # Drive-based behavior planner
├── creatures/
│   ├── go2/                    # Unitree Go2 configuration
│   └── freenove/               # Freenove Robot Dog configuration
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

## Paper

> **MH-FLOCKE: Biologically Grounded Embodied Cognition Through a 15-Step Closed-Loop Architecture for Quadruped Locomotion Learning**
>
> Marc Hesse (2026). Independent Researcher, Potsdam, Germany.
>
> Preprint: [aixiv.science](https://aixiv.science)

## Videos

- [Freenove Robot Dog — SNN on Real Hardware](https://www.youtube.com/watch?v=7iN8tB2xLHI)
- [Video #3: Go2 Ball Interaction](https://www.youtube.com/watch?v=Jo7UM6pEFMg)
- [YouTube Channel: @mhflocke](https://www.youtube.com/@mhflocke)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

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
