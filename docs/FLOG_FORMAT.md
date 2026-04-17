# FLOG Binary Format — Structure Documentation

**MH-FLOCKE Level 15 v0.5.0**
**Document Version:** 1.1 — April 15, 2026

---

## 1. File Structure

```
┌─────────────────────────────────────────────────┐
│  HEADER                                          │
│  ├── Magic:     4 bytes  "FLOG" (0x464C4F47)    │
│  ├── Version:   uint16 LE (currently 1)          │
│  ├── Phase:     uint8 (0 = training)             │
│  ├── Meta Len:  uint32 LE (JSON metadata size)   │
│  └── Meta JSON: variable (UTF-8 JSON string)     │
├─────────────────────────────────────────────────┤
│  FRAMES (repeated until EOF)                     │
│  ├── Timestamp: float64 LE (seconds since start) │
│  ├── Type:      uint8 (frame type)               │
│  ├── Data Len:  uint32 LE (payload size)         │
│  └── Payload:   variable (MessagePack-encoded)   │
└─────────────────────────────────────────────────┘
```

### Header: 11 + meta_len bytes

| Field     | Type      | Bytes | Description                           |
|-----------|-----------|-------|---------------------------------------|
| magic     | char[4]   | 4     | Always `FLOG` (0x464C4F47)           |
| version   | uint16 LE | 2     | Format version (currently 1)          |
| phase     | uint8     | 1     | Training phase (0 = standard)         |
| meta_len  | uint32 LE | 4     | Length of JSON metadata in bytes      |
| meta_json | UTF-8     | var   | JSON metadata (see §2)                |

### Frame: 13 + data_len bytes

| Field     | Type      | Bytes | Description                           |
|-----------|-----------|-------|---------------------------------------|
| timestamp | float64 LE| 8     | Seconds since training start          |
| type      | uint8     | 1     | Frame type constant (see §3)          |
| data_len  | uint32 LE | 4     | Length of MessagePack payload         |
| payload   | msgpack   | var   | MessagePack-encoded dict              |

---

## 2. Header Metadata (JSON)

```json
{
  "creature": "go2",
  "task": "dog plays with ball on grass",
  "scene": "flat",
  "difficulty": 0.0,
  "steps": 50000,
  "device": "cpu",
  "version": "v0.3.4"
}
```

---

## 3. Frame Types

| Constant         | Value | Description                              | Frequency        |
|------------------|-------|------------------------------------------|------------------|
| FRAME_EVOLUTION  | 1     | CPG evolution data (gen, fitness)         | Per generation    |
| FRAME_TRAINING   | 2     | Training stats (reward, distance, etc.)   | Every 1000 steps  |
| FRAME_EVENT      | 3     | Milestone events (skill_start, etc.)      | Sparse            |
| FRAME_CREATURE   | 4     | Physics snapshot (qpos, qvel, com)        | Every 10 steps    |

---

## 4. Frame Payloads (MessagePack Dict)

### 4.1 FRAME_CREATURE (type=4) — Physics Snapshot

Recorded every 10 simulation steps. Contains full MuJoCo state.

| Key       | Type         | Description                                  |
|-----------|--------------|----------------------------------------------|
| step      | int          | Training step number                         |
| heading   | float        | Forward heading (0-1, dot product)           |
| speed     | float        | Forward velocity (m/s)                       |
| x         | float        | Creature X position (world coords)           |
| y         | float        | Creature Y position (world coords)           |
| ball_pos  | [f, f, f]    | Ball position [x, y, z] (world coords)       |
| pos       | [f × 26]     | Full qpos array (Go2: 7 root + 12 joints + 7 ball) |
| vel       | [f × 24]     | Full qvel array (Go2: 6 root + 12 joints + 6 ball) |
| com       | [f, f, f]    | Center of mass [x, y, z]                     |

**qpos layout (26 values):**
- `[0:3]` — root position (x, y, z)
- `[3:7]` — root quaternion (w, x, y, z)
- `[7:19]` — 12 joint angles (FL_hip, FL_thigh, FL_calf, FR_*, RL_*, RR_*)
- `[19:26]` — ball free joint (x, y, z, qw, qx, qy, qz)

### 4.2 FRAME_TRAINING (type=2) — Training Stats

Recorded every 1000 steps. Contains all brain/body metrics.

**Core Metrics:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| step               | int    | Training step                              |
| distance           | float  | Distance traveled since last reset         |
| max_distance       | float  | Max distance ever traveled                 |
| falls              | int    | Cumulative fall count                      |
| reward             | float  | Current reward signal                      |
| upright            | float  | Uprightness (0-1)                          |
| is_fallen          | int    | Currently fallen? (0/1)                    |
| recoveries         | int    | Recovery count                             |
| vel_mps            | float  | Velocity in m/s                            |
| behavior           | string | Current behavior (see §5)                  |

**Brain / SNN:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| gwt                | string | Global Workspace broadcast winner          |
| c_level            | int    | Consciousness level (0-15)                 |
| pci                | float  | Perturbational Complexity Index            |
| spike_count        | int    | SNN spike count this interval              |
| phase              | string | Dev phase (e.g. "level15")                 |
| actor_competence   | float  | Actor competence (0-1, drives CPG blend)   |
| cpg_weight         | float  | CPG weight (1.0 = pure CPG, 0.4 = SNN mix)|
| snn_mix            | float  | SNN contribution ratio                     |

**Neuromodulation:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| da_reward          | float  | Dopamine reward signal                     |
| emotion_dominant   | string | Dominant emotion (content, neutral, etc.)  |
| valence            | float  | Emotional valence (-1 to 1)                |
| arousal            | float  | Arousal level (0-1)                        |
| drive_dominant     | string | Dominant drive (exploration, play, etc.)   |
| curiosity_reward   | float  | Curiosity-driven reward                    |

**Cerebellum:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| grc_sparseness     | float  | Granule cell sparseness                    |
| cf_magnitude       | float  | Climbing fiber error signal magnitude      |
| pf_pkc_weight      | float  | Parallel fiber → Purkinje cell weight      |
| correction_mag     | float  | Cerebellar correction magnitude            |
| dcn_activity       | float  | Deep cerebellar nuclei activity            |
| pred_error         | float  | Forward model prediction error             |
| cb_steer_correction| float  | Cerebellar steering correction             |
| cb_heading_gain    | float  | Cerebellar heading gain                    |

**Ball Interaction:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| ball_dist          | float  | Current ball distance (m)                  |
| ball_heading       | float  | Ball heading relative to creature          |
| ball_salience      | float  | Ball salience (attention weight)           |
| ball_approach_reward| float | Ball approach reward component             |
| ball_episode       | int    | Current ball episode number                |
| task_pe            | float  | Task-specific prediction error             |
| steering_offset    | float  | Steering offset toward ball                |

**Curriculum Learning:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| cl_adaptations     | int    | Number of curriculum adaptations           |
| cl_best_ball_dist  | float  | Best ball distance achieved (m)            |
| cl_consec_improve  | int    | Consecutive improvement count              |
| cl_consec_fail     | int    | Consecutive failure count                  |
| cl_vor_hip_gain    | float  | VOR hip gain                               |

**Locomotion / CPG:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| freq_scale         | float  | CPG frequency scale                        |
| amp_scale          | float  | CPG amplitude scale                        |
| posture_state      | string | Posture state                              |
| reflex_active      | bool   | Righting reflex active?                    |
| reflex_magnitude   | float  | Reflex magnitude                           |
| foot_contact_count | int    | Number of feet on ground                   |
| foot_FL/FR/RL/RR   | float  | Individual foot contact forces             |

**Terrain:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| terrain_type       | string | Terrain type (flat, hilly, etc.)           |
| terrain_difficulty | float  | Terrain difficulty (0-1)                   |
| terrain_reflex_mag | float  | Terrain reflex magnitude                   |
| terrain_pitch_ema  | float  | Terrain pitch EMA                          |
| terrain_roll_ema   | float  | Terrain roll EMA                           |

**IMU / Vestibular (v0.5.0, Issues #121, #122):**

| Key                   | Type   | Description                                |
|-----------------------|--------|--------------------------------------------|
| yaw_rate              | float  | Gyroscope Z raw (rad/s, >0 = turning left) |
| yaw_rate_ema          | float  | Smoothed yaw rate (Mogli only, EMA α=0.98) |
| vestibular_correction | float  | L/R tonic drive correction (Mogli only)    |
| pitch_rate            | float  | Gyroscope Y (rad/s)                        |
| roll_rate             | float  | Gyroscope X (rad/s)                        |
| yaw                   | float  | Orientation yaw (Euler, rad)               |
| pitch                 | float  | Orientation pitch (Euler, rad)             |
| roll                  | float  | Orientation roll (Euler, rad)              |
| y                     | float  | Y position (for drift/circle detection)    |

**Obstacle Avoidance (v0.5.0, Issue #121):**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| obstacle_distance  | float  | HC-SR04 / rangefinder distance (m, -1=none)|

**Mogli Oscillator (v0.5.0, --neural-cpg only):**

Per-leg keys prefixed `mogli_` from `MogliCPG.get_stats()`:

| Key                       | Type   | Description                          |
|---------------------------|--------|--------------------------------------|
| mogli_gain                | float  | CPG gain modulation                  |
| mogli_{leg}_hip_flex_rate | float  | Flexor firing rate (per leg)         |
| mogli_{leg}_hip_ext_rate  | float  | Extensor firing rate (per leg)       |
| mogli_{leg}_hip_output    | float  | Half-center output EMA (per leg)     |
| mogli_{leg}_hip_phase     | float  | Extracted phase in radians (per leg) |
| mogli_coupling_mean       | float  | Mean abs coupling weight             |
| mogli_coupling_contra     | float  | Contralateral coupling weight        |
| mogli_coupling_ipsi       | float  | Ipsilateral coupling weight          |
| mogli_coupling_diag       | float  | Diagonal coupling weight             |
| mogli_yaw_rate_ema        | float  | Vestibular smoothed yaw rate         |
| mogli_vestibular_correction| float | Vestibular L/R drive correction      |

Where `{leg}` is one of: `FL`, `FR`, `RL`, `RR`.

**Olfactory / Sensory:**

| Key                | Type   | Description                                |
|--------------------|--------|--------------------------------------------|
| smell_strength     | float  | Olfactory signal strength                  |
| smell_direction    | float  | Olfactory signal direction                 |
| sound_intensity    | float  | Auditory signal intensity                  |
| sound_direction    | float  | Auditory signal direction                  |
| scents_found       | int    | Number of scent sources found              |
| olfactory_steering | float  | Olfactory steering contribution            |
| scent_0_x/y        | float  | First scent source position                |

### 4.3 FRAME_EVENT (type=3) — Milestone Events

| Key  | Type   | Description                     |
|------|--------|---------------------------------|
| type | string | Event type (skill_start, etc.)  |
| msg  | string | Human-readable message          |

### 4.4 FRAME_EVOLUTION (type=1) — CPG Evolution

| Key          | Type  | Description                 |
|--------------|-------|-----------------------------|
| gen          | int   | Generation number           |
| best_fitness | float | Best fitness this gen       |
| avg_fitness  | float | Average fitness             |
| best_distance| float | Best distance traveled      |
| upright      | float | Uprightness score           |
| stood        | bool  | Successfully stood up?      |

---

## 5. Behavior States

Emergent behaviors observed during training (not programmed):

| Behavior         | Description                                |
|------------------|--------------------------------------------|
| motor_babbling   | Random motor exploration (early training)   |
| sniff            | Low-speed orientation / sensory sampling    |
| walk             | Relaxed forward locomotion                  |
| trot             | Faster rhythmic gait                        |
| chase            | High-speed pursuit toward target            |
| alert            | Stationary, heightened sensory attention     |
| play             | Exploratory interaction with objects         |

---

## 6. Reading a FLOG file (Python)

```python
import struct, json, msgpack

with open('training_log.bin', 'rb') as f:
    magic = f.read(4)  # b'FLOG'
    version = struct.unpack('<H', f.read(2))[0]
    phase = struct.unpack('<B', f.read(1))[0]
    meta_len = struct.unpack('<I', f.read(4))[0]
    meta = json.loads(f.read(meta_len))

    frames = []
    while True:
        header = f.read(13)
        if len(header) < 13:
            break
        ts, frame_type, data_len = struct.unpack('<dBI', header)
        payload = f.read(data_len)
        data = msgpack.unpackb(payload, raw=False)
        frames.append({'timestamp': ts, 'type': frame_type, 'data': data})
```

See also: `src/viz/flog_replay.py` (FlogReplay class)

---

## 7. Dependencies

- **msgpack** — Frame payload encoding (pip install msgpack)
- Fallback: JSON encoding (if msgpack not available)
- Written by: `src/body/creature_store.py`
- Read by: `src/viz/flog_replay.py`, `flog_server.py`
