# MH-FLOCKE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19336894.svg)](https://doi.org/10.5281/zenodo.19336894)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Biologically Grounded Embodied Cognition for Quadruped Locomotion Learning**

A simulated quadruped learns to walk through a 15-step closed-loop cognitive architecture integrating spiking neural networks, cerebellar forward models, central pattern generators, embodied emotions, and reward-modulated spike-timing-dependent plasticity — no end-to-end RL required.

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
- **SNN with R-STDP** — 5000+ Izhikevich neurons, reward-modulated spike-timing-dependent plasticity
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

## Project Structure

```
mhflocke/
├── scripts/train_v032.py       # Main training loop
├── src/
│   ├── body/                   # MuJoCo creature, terrain, genome
│   ├── brain/                  # SNN, cerebellum, CPG, cognitive brain
│   └── behavior/               # Drive-based behavior planner
├── creatures/go2/              # Unitree Go2 configuration
├── flog_server.py              # FLOG analysis + dashboard
└── docs/                       # Format specs
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

- [Video #3: Go2 Ball Interaction](https://www.youtube.com/watch?v=Jo7UM6pEFMg)
- [YouTube Channel: @mhflocke](https://www.youtube.com/@mhflocke)

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
