# Changelog

## v0.5.0 (2026-04-13)

### Izhikevich Neuron Dynamics (Issue #104)
- Per-population Izhikevich (a,b,c,d) parameters replace uniform LIF-LTC
- 5 biologically accurate cell types: Regular Spiking (GrC), Intrinsically
  Bursting (GoC), Chattering (PkC), Rebound Burst (DCN), Fast Spiking (Output)
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

### Roadmap
We're working toward a spatial planning layer inspired by hippocampal
place cells and prefrontal cortex working memory. This will enable
path planning around obstacles — not just reactive avoidance. The
biological hierarchy: spinal reflexes → brainstem avoidance →
cerebellar calibration → cortical planning.

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
