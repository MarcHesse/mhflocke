# Changelog

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
