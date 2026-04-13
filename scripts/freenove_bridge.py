#!/usr/bin/env python3
"""
MH-FLOCKE — Freenove Robot Dog Bridge v4.1
=============================================
v4.1: Hardware servo compensation (Issue #105):
      - HARDWARE_CORRECTION_GAIN ×5 amplification for servo dead band
      - HARDWARE_CPG_FREQ_SCALE ×0.7 for servo tracking
      - Obstacle reflexes: CPG-kill <15cm, inhibition <50cm
      - Per-population Izhikevich parameters (Issue #104)
v4.0: UNIFIED CODEBASE — uses src/brain/ directly (same as simulator).
      No more separate NumPy SNN. Same SNNController, same CerebellarLearning,
      same brain_persistence on Pi AND in MuJoCo. PyTorch on Pi.

Usage:
    python3 freenove_bridge.py --gait walk --snn --verbose --duration 300

Requires on Pi:
    pip3 install torch --break-system-packages
    Copy src/brain/ (incl. topology.py) to Pi

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

import time, math, argparse, os, signal, sys

HOME_DIR = os.path.expanduser('~')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to path (works both from repo and from Pi home)
for _root in [os.path.join(SCRIPT_DIR, '..'), HOME_DIR, os.path.join(HOME_DIR, 'mhflocke')]:
    if os.path.exists(os.path.join(_root, 'src', 'brain', 'snn_controller.py')):
        sys.path.insert(0, _root)
        break

FREENOVE_SEARCH_PATHS = [
    os.path.join(HOME_DIR, 'freenove_server'),
    os.path.join(HOME_DIR, 'Freenove_Robot_Dog_Kit_for_Raspberry_Pi', 'Code', 'Server'),
    '/home/pi/freenove_server',
]
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ

# === Hardware Servo Compensation (Issue #105) ===
# Strategy 1: Amplify cerebellar corrections to overcome servo dead band (~3°).
# Corrections of 0.01 (3°) are invisible to SG90; ×5 = 0.05 (15°) = visible.
# Biology: Reticulospinal tract amplifies cerebellar DCN output for
# spinal motor neurons — the gain is set by brainstem circuits.
HARDWARE_CORRECTION_GAIN = 5.0

# Strategy 3: Slow CPG for servo tracking capability.
# SG90: ~60°/100ms at no load. At CPG 0.8Hz, servos can barely track.
# ×0.7 → 0.56Hz gives servos 40% more time per cycle.
# Biology: Animals naturally slow down on difficult terrain.
HARDWARE_CPG_FREQ_SCALE = 0.7

# === Obstacle Reflexes (brainstem, not learned) ===
# These are HARDWIRED reflexes that the cerebellum calibrates but
# cannot override. Like a dog flinching when its nose touches a wall.
# Biology: Trigeminal reflex arc (face→brainstem→motor, ~10ms latency)
# v0.5.0: Tightened from 15/50/8cm to 10/30/5cm.
# The reflex prevents DAMAGE, not CONTACT. The robot needs to bump
# its nose to learn — the old distances were too conservative.
REFLEX_STOP_DISTANCE_CM = 10.0      # CPG kill — full stop
REFLEX_SLOW_DISTANCE_CM = 30.0      # Graded CPG inhibition
REFLEX_REVERSE_DISTANCE_CM = 5.0    # Reverse CPG — back up
DASHBOARD_PATHS = [
    os.path.join(SCRIPT_DIR, '..', 'creatures', 'freenove', 'dashboard'),
    os.path.join(HOME_DIR, 'dashboard'),
]

# ================================================================
# HARDWARE HELPERS (same as v2.5)
# ================================================================

def ik(x, y, z, l1=23, l2=55, l3=55):
    a = math.pi/2 - math.atan2(z, y)
    x4 = l1*math.sin(a); x5 = l1*math.cos(a)
    l23 = math.sqrt((z-x5)**2+(y-x4)**2+x**2)
    b = math.asin(round(x/l23,2)) - math.acos(round((l2*l2+l23*l23-l3*l3)/(2*l2*l23),2))
    c = math.pi - math.acos(round((l2**2+l3**2-l23**2)/(2*l3*l2),2))
    return round(math.degrees(a)), round(math.degrees(b)), round(math.degrees(c))

def set_legs(servo, points):
    angles = [list(ik(p[0],p[1],p[2])) for p in points]
    for i in range(2):
        servo.setServoAngle(4+i*3, max(0,min(180,angles[i][0])))
        servo.setServoAngle(3+i*3, max(0,min(180,90-angles[i][1])))
        servo.setServoAngle(2+i*3, max(0,min(180,angles[i][2])))
        servo.setServoAngle(8+i*3, max(0,min(180,angles[i+2][0])))
        servo.setServoAngle(9+i*3, max(0,min(180,90+angles[i+2][1])))
        servo.setServoAngle(10+i*3, max(0,min(180,180-angles[i+2][2])))

def get_servo_angles(points):
    angles = [list(ik(p[0],p[1],p[2])) for p in points]
    s = {}
    for i in range(2):
        s[4+i*3]=max(0,min(180,angles[i][0])); s[3+i*3]=max(0,min(180,90-angles[i][1]))
        s[2+i*3]=max(0,min(180,angles[i][2])); s[8+i*3]=max(0,min(180,angles[i+2][0]))
        s[9+i*3]=max(0,min(180,90+angles[i+2][1])); s[10+i*3]=max(0,min(180,180-angles[i+2][2]))
    return s

class FreenoveCPG:
    def __init__(self, speed=1.0, height=99):
        self.height=height; self.stride=12; self.lift=6
        self.frequency=0.8*speed*HARDWARE_CPG_FREQ_SCALE; self._phase=90.0
        self._inhibition = 1.0  # 1.0=full, 0.0=stopped
        self._reverse = False   # True = walk backward
    def set_inhibition(self, level):
        """Set CPG output inhibition. 0.0=stopped, 1.0=full stride."""
        self._inhibition = max(0.0, min(1.0, level))
    def set_reverse(self, reverse):
        """Set reverse walking mode."""
        self._reverse = reverse
    def compute(self, dt=None):
        if dt is None: dt=CONTROL_DT
        direction = -1.0 if self._reverse else 1.0
        self._phase += 360.0*self.frequency*dt*direction; h=self.height
        pr=self._phase*math.pi/180.0; po=(self._phase+180.0)*math.pi/180.0
        stride = self.stride * self._inhibition
        lift = self.lift * self._inhibition
        X1=stride*math.cos(pr); Y1=lift*math.sin(pr)+h
        if Y1>h: Y1=h
        X2=stride*math.cos(po); Y2=lift*math.sin(po)+h
        if Y2>h: Y2=h
        return [[X1+10,Y1,10],[X2+10,Y2,10],[X1+10,Y1,-10],[X2+10,Y2,-10]]
    def get_phase_input(self):
        import numpy as np
        rad=(self._phase%360.0)/360.0*2*math.pi
        return np.array([math.sin(rad),math.cos(rad)],dtype=np.float32)

class FreenoveServoDriver:
    def __init__(self, dry_run=False):
        self.dry_run=dry_run; self._servo=None
        if not dry_run:
            for path in FREENOVE_SEARCH_PATHS:
                if os.path.exists(os.path.join(path,'Servo.py')):
                    sys.path.insert(0,path)
                    try:
                        from Servo import Servo
                        self._servo=Servo(); print('  PCA9685 initialized'); break
                    except Exception as e: print(f'  WARNING: {e}')
            if not self._servo: self.dry_run=True
    def get_servo(self): return self._servo
    def emergency_stop(self):
        print('\n  EMERGENCY STOP')
        if self._servo:
            for ch in range(2,14):
                try: self._servo.setServoAngle(ch,90)
                except: pass

class IMUReader:
    def __init__(self):
        self.imu = None
        self.pitch = 0.0; self.roll = 0.0; self.yaw = 0.0
        try:
            for p in FREENOVE_SEARCH_PATHS:
                if os.path.exists(os.path.join(p, 'IMU.py')) and p not in sys.path:
                    sys.path.insert(0, p)
            from IMU import IMU
            self.imu = IMU()
            for _ in range(50): self.imu.imuUpdate()
            print('  IMU: MPU6050 initialized (pitch/roll/yaw)')
        except Exception as e:
            print(f'  IMU: not available ({e}) — using defaults')
    def update(self):
        if self.imu is None:
            return {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0, 'upright': 1.0}
        try: self.pitch, self.roll, self.yaw = self.imu.imuUpdate()
        except: pass
        tilt = math.sqrt(self.pitch**2 + self.roll**2)
        upright = max(0.0, 1.0 - tilt / 45.0)
        return {'pitch': self.pitch, 'roll': self.roll, 'yaw': self.yaw, 'upright': upright}


class UltrasonicReader:
    """
    HC-SR04 ultrasonic distance sensor reader.

    Mounted on Freenove head bracket, facing forward.
    GPIO pins: TRIG=GPIO 27, ECHO=GPIO 22 (adjustable).
    Range: 2cm – 400cm, ~15° beam width.

    Biology: Echolocation (bats) / whisker-based proximity sensing.
    The ultrasonic sensor gives the robot a simple "obstacle ahead"
    signal — the digital equivalent of a dog's nose bumping a wall.

    Issue #103: This sensor provides the clear binary signal needed
    to test whether the SNN can learn active avoidance behavior.
    """
    TRIG_PIN = 27
    ECHO_PIN = 22
    MAX_RANGE_CM = 400.0
    TIMEOUT_S = 0.03  # 30ms timeout (~5m max at speed of sound)

    def __init__(self):
        self.gpio = None
        self.distance_cm = self.MAX_RANGE_CM
        self._available = False
        self._history = []  # Moving median filter (5 samples)
        self._filter_size = 5
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.TRIG_PIN, GPIO.OUT)
            GPIO.setup(self.ECHO_PIN, GPIO.IN)
            GPIO.output(self.TRIG_PIN, False)
            time.sleep(0.1)  # Settle
            self._available = True
            print(f'  Ultrasonic: HC-SR04 initialized (TRIG={self.TRIG_PIN}, ECHO={self.ECHO_PIN}, median={self._filter_size})')
        except Exception as e:
            print(f'  Ultrasonic: not available ({e}) — using max range')

    def _read_raw(self) -> float:
        """Single raw measurement in meters."""
        if not self._available or self.gpio is None:
            return self.MAX_RANGE_CM / 100.0

        try:
            GPIO = self.gpio
            # Send 10µs trigger pulse
            GPIO.output(self.TRIG_PIN, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG_PIN, False)

            # Wait for echo start (rising edge)
            t_timeout = time.time() + self.TIMEOUT_S
            while GPIO.input(self.ECHO_PIN) == 0:
                pulse_start = time.time()
                if pulse_start > t_timeout:
                    return self.MAX_RANGE_CM / 100.0

            # Wait for echo end (falling edge)
            t_timeout = time.time() + self.TIMEOUT_S
            while GPIO.input(self.ECHO_PIN) == 1:
                pulse_end = time.time()
                if pulse_end > t_timeout:
                    return self.MAX_RANGE_CM / 100.0

            # Distance = time * speed_of_sound / 2
            pulse_duration = pulse_end - pulse_start
            distance_cm = pulse_duration * 34300 / 2.0

            # Clamp to valid range
            if distance_cm < 2.0 or distance_cm > self.MAX_RANGE_CM:
                distance_cm = self.MAX_RANGE_CM

            self.distance_cm = distance_cm
            return distance_cm / 100.0  # Convert to meters

        except Exception:
            return self.MAX_RANGE_CM / 100.0

    def read(self) -> float:
        """
        Filtered measurement in meters (moving median, 5 samples).

        Biology: Sensory neurons integrate over time — a single spike
        doesn't trigger a response, but a sustained signal does.
        The median filter removes outliers from beam reflections
        off the floor or nearby objects outside the target.
        """
        raw = self._read_raw()
        self._history.append(raw)
        if len(self._history) > self._filter_size:
            self._history.pop(0)
        # Median: sort and take middle value (robust against outliers)
        return sorted(self._history)[len(self._history) // 2]

    def cleanup(self):
        """Release GPIO pins."""
        if self._available and self.gpio:
            try:
                self.gpio.cleanup([self.TRIG_PIN, self.ECHO_PIN])
            except Exception:
                pass

# ================================================================
# SENSOR ENCODING (identical to mujoco_creature.py hardware mode)
# ================================================================

def encode_sensory(servo_angles_prev, cpg_phase, step, imu_data=None,
                   obstacle_distance=-1.0):
    import numpy as np
    s = np.zeros(48, dtype=np.float32)
    if servo_angles_prev:
        for i, a in enumerate(servo_angles_prev[:12]): s[i] = a / 180.0
    s[12:14] = cpg_phase
    if imu_data:
        s[14] = imu_data['pitch'] / 90.0 * 0.5 + 0.5
        s[15] = imu_data['roll'] / 90.0 * 0.5 + 0.5
        s[16] = imu_data['yaw'] / 180.0 * 0.5 + 0.5
        s[17] = imu_data['upright']
    else:
        s[14:18] = 0.5
    # Channel 18: Ultrasonic proximity (Issue #103)
    # HC-SR04: 2cm-400cm range. Same encoding as mujoco_creature.py.
    _US_MAX_RANGE = 4.0
    if obstacle_distance >= 0 and obstacle_distance < _US_MAX_RANGE:
        proximity = 1.0 - min(obstacle_distance / _US_MAX_RANGE, 1.0)
        s[18] = float(np.clip(proximity ** 0.5, 0.0, 1.0))
    return s

# ================================================================
# REWARD (same as v2.5)
# ================================================================

class RewardComputer:
    def __init__(self):
        self.prev_angles = None; self.prev_delta = None
        self.smoothness_ema = 0.5; self.symmetry_ema = 0.5
    def compute(self, current_angles, imu_data=None):
        import numpy as np
        if current_angles is None or len(current_angles) < 12: return 0.5
        angles = np.array(current_angles[:12], dtype=np.float32)
        if self.prev_angles is None:
            self.prev_angles = angles.copy(); return 0.5
        delta = angles - self.prev_angles
        jerk_mag = np.mean(np.abs(delta))
        smoothness = max(0, 1.0 - jerk_mag / 5.0)
        self.smoothness_ema = 0.95 * self.smoothness_ema + 0.05 * smoothness
        jerk_reward = 0.5
        if self.prev_delta is not None:
            jerk_reward = max(0, 1.0 - np.mean(np.abs(delta - self.prev_delta)) / 3.0)
        fl=np.abs(delta[0:3]); fr=np.abs(delta[3:6]); rl=np.abs(delta[6:9]); rr=np.abs(delta[9:12])
        sym = max(0, 1.0 - (np.mean(np.abs(fl-rr)) + np.mean(np.abs(fr-rl))) / 4.0)
        self.symmetry_ema = 0.95 * self.symmetry_ema + 0.05 * sym
        activity = min(1.0, jerk_mag / 1.0)
        stability = imu_data['upright'] if imu_data else 0.5
        reward = (self.smoothness_ema*0.25 + jerk_reward*0.15 +
                  self.symmetry_ema*0.25 + activity*0.15 + stability*0.20)
        self.prev_delta = delta.copy(); self.prev_angles = angles.copy()
        return float(np.clip(0.2 + reward * 0.6, 0.1, 0.9))

# ================================================================
# SNN BUILDER — creates the same topology as MuJoCoCreatureBuilder
# ================================================================

def build_freenove_snn(device='cpu'):
    """
    Build a 232-neuron SNN with cerebellar architecture.
    IDENTICAL topology to MuJoCoCreatureBuilder.build() with profile.json.
    
    Returns: snn, cerebellar_learning, population_info
    """
    import torch
    from src.brain.snn_controller import SNNController, SNNConfig
    from src.brain.cerebellar_learning import CerebellarLearning, CerebellarConfig
    from src.brain.topology import compute_cerebellar_populations

    n_input = 48
    n_output = 12
    n_hidden = 172
    n_actuators = 12

    # Compute cerebellar populations (same function as simulator)
    cb_pops = compute_cerebellar_populations(n_hidden, n_actuators)
    n_granule = cb_pops['n_granule']   # 106
    n_golgi = cb_pops['n_golgi']       # 18
    n_purkinje = cb_pops['n_purkinje'] # 24
    n_dcn = cb_pops['n_dcn']           # 24

    total_neurons = n_input + n_output + n_granule + n_golgi + n_purkinje + n_dcn  # 232

    # Neuron ID ranges (must match simulator build() order)
    grc_start = n_input + n_output  # 60
    goc_start = grc_start + n_granule  # 166
    pkc_start = goc_start + n_golgi  # 184
    dcn_start = pkc_start + n_purkinje  # 208

    # Create SNN
    snn = SNNController(SNNConfig(
        n_neurons=total_neurons,
        connectivity_prob=0.0,  # We build connectivity manually
        homeostatic_interval=200,
        device=device,
    ))

    # Define populations (identical to simulator)
    mf_ids = torch.arange(0, n_input)
    snn.define_population('input', mf_ids)
    snn.define_population('mossy_fibers', mf_ids)

    out_ids = torch.arange(n_input, n_input + n_output)
    snn.define_population('output', out_ids)

    grc_ids = torch.arange(grc_start, grc_start + n_granule)
    snn.define_population('granule_cells', grc_ids)
    snn.define_population('hidden', grc_ids)

    goc_ids = torch.arange(goc_start, goc_start + n_golgi)
    snn.define_population('golgi_cells', goc_ids)

    pkc_ids = torch.arange(pkc_start, pkc_start + n_purkinje)
    snn.define_population('purkinje_cells', pkc_ids)

    dcn_ids = torch.arange(dcn_start, dcn_start + n_dcn)
    snn.define_population('dcn', dcn_ids)

    # Build connectivity (identical to simulator)
    mf_per_grc = min(4, max(2, n_input // max(n_granule, 1)))
    src_list, tgt_list = [], []
    for g in range(n_granule):
        mf_choices = torch.randint(0, n_input, (mf_per_grc,))
        for m in mf_choices:
            src_list.append(mf_ids[m].item())
            tgt_list.append(grc_ids[g].item())
    if src_list:
        src_t = torch.tensor(src_list, device=device)
        tgt_t = torch.tensor(tgt_list, device=device)
        w = torch.rand(len(src_list), device=device) * 1.0 + 1.0
        snn._weight_indices = torch.stack([src_t, tgt_t])
        snn._weight_values = w
        snn._eligibility = torch.zeros(len(src_list), device=device)

    grc_goc_prob = min(0.3, 0.05 * (4000 / max(n_granule, 1)))
    snn.connect_populations('granule_cells', 'golgi_cells',
                            prob=grc_goc_prob, weight_range=(0.3, 0.8))
    snn.neuron_types[goc_ids] = -1.0
    snn.connect_populations('golgi_cells', 'granule_cells',
                            prob=0.02, weight_range=(0.2, 0.4))
    snn.connect_populations('mossy_fibers', 'golgi_cells',
                            prob=0.1, weight_range=(0.5, 1.0))
    snn.connect_populations('granule_cells', 'output',
                            prob=0.02, weight_range=(0.3, 0.8))

    # Per-population time constants
    snn._tau_base[grc_ids] = 5.0
    snn._tau_base[goc_ids] = 20.0
    snn._tau_base[pkc_ids] = 15.0
    snn._tau_base[dcn_ids] = 10.0

    # Thresholds
    snn._thresholds[grc_ids] = 0.5
    snn._thresholds[goc_ids] = 0.5
    snn._thresholds[pkc_ids] = 1.0
    snn._thresholds[dcn_ids] = 1.0
    snn._thresholds[out_ids] = 0.4
    snn._hidden_tonic_current = 0.015

    snn._rebuild_sparse_weights()

    # Cerebellar Learning (same class as simulator)
    cb_cfg = CerebellarConfig(
        n_granule=n_granule, n_golgi=n_golgi,
        n_purkinje=n_purkinje, n_dcn=n_dcn,
        snn_ramp_steps=2000, snn_mix_end=1.0,
        ltd_rate=0.001, ltp_rate=0.001,
    )
    cb = CerebellarLearning(snn=snn, n_actuators=n_actuators, config=cb_cfg, device=device)
    cb.set_populations(
        mf_ids=mf_ids, grc_ids=grc_ids, goc_ids=goc_ids,
        pkc_ids=pkc_ids, dcn_ids=dcn_ids)

    # Protect cerebellar populations from R-STDP
    snn.protected_populations = {
        'mossy_fibers', 'granule_cells', 'golgi_cells',
        'purkinje_cells', 'dcn',
    }

    # Per-population Izhikevich parameters (Issue #104)
    # These enable biologically accurate firing dynamics:
    # - DCN: Rebound Burst → 5-10x stronger corrections
    # - GoC: Intrinsically Bursting / Pacemaker → stable sparseness
    # - PkC: Chattering → baseline Simple Spike rate
    # - Output: Fast Spiking → quick motor response
    # Ref: Izhikevich 2003, Table 2
    snn.set_izhikevich_params('granule_cells', a=0.02, b=0.2, c=-65, d=8)   # RS
    snn.set_izhikevich_params('golgi_cells',   a=0.02, b=0.2, c=-55, d=4)   # IB
    snn.set_izhikevich_params('purkinje_cells', a=0.02, b=0.2, c=-50, d=2)  # CH
    snn.set_izhikevich_params('dcn',           a=0.03, b=0.25, c=-52, d=0)  # Rebound
    snn.set_izhikevich_params('output',        a=0.1,  b=0.2, c=-65, d=2)   # FS

    print(f'  SNN: {total_neurons} neurons, {snn._n_synapses} synapses')
    print(f'  Cerebellum: GrC={n_granule} GoC={n_golgi} PkC={n_purkinje} DCN={n_dcn}')
    print(f'  Izhikevich: DCN=Rebound, GoC=IB, PkC=CH, Output=FS')

    return snn, cb

# ================================================================
# COMPETENCE GATE (same logic as simulator CompetenceGate)
# ================================================================

class CompetenceGate:
    def __init__(self, cpg_max=0.9, cpg_min=0.4):
        self.cpg_max = cpg_max; self.cpg_min = cpg_min
        self.cpg_weight = cpg_max; self.actor_competence = 0.0
        self._upright_streak = 0
    def update(self, is_upright, velocity=0.0):
        if is_upright: self._upright_streak += 1
        else: self._upright_streak = max(0, self._upright_streak - 5)
        target = min(1.0, self._upright_streak / 2000.0)
        self.actor_competence += 0.001 * (target - self.actor_competence)
        self.cpg_weight = self.cpg_max - (self.cpg_max - self.cpg_min) * self.actor_competence
    def blend(self, cpg_points, actor_output):
        blended = []
        for i, point in enumerate(cpg_points):
            p = list(point)
            if i < len(actor_output) // 3:
                off = i * 3
                for j in range(3):
                    if off + j < len(actor_output):
                        mod = (actor_output[off + j] - 0.5) * 10.0
                        p[j] = self.cpg_weight * p[j] + (1 - self.cpg_weight) * (p[j] + mod)
            blended.append(p)
        return blended
    def get_stats(self):
        return {'actor_competence': self.actor_competence, 'cpg_weight': self.cpg_weight}

# ================================================================
# MAIN
# ================================================================

def main():
    import torch
    import numpy as np

    parser = argparse.ArgumentParser(description='MH-FLOCKE Freenove Bridge v4.1')
    parser.add_argument('--gait', default='stand', choices=['stand','stop','walk'])
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=0,
                        help='Fixed number of steps (overrides --duration). '
                             'Use for fair A/B comparison: both runs do exactly the same number of servo updates.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--snn', action='store_true', help='Enable SNN + Cerebellum')
    parser.add_argument('--best', action='store_true', help='Load best brain')
    parser.add_argument('--fresh', action='store_true', help='Start fresh')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log-csv', type=str, default=None,
                        help='Log sensor data to CSV file for A/B comparison')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'  MH-FLOCKE — Freenove Bridge v4.1 (unified + servo comp)')
    print(f'  Same SNN + Cerebellum as simulator (PyTorch)')
    print(f'  Gait: {args.gait}  Speed: {args.speed}  Duration: {args.duration}s')
    print(f'  CPG freq scale: {HARDWARE_CPG_FREQ_SCALE}  Correction gain: {HARDWARE_CORRECTION_GAIN}x')
    print(f'  SNN: {"ON" if args.snn else "OFF"}  Dashboard: {"ON" if args.dashboard else "OFF"}')
    print(f'{"="*60}\n')

    # CSV logger for A/B comparison experiments
    csv_file = None
    if args.log_csv:
        csv_file = open(args.log_csv, 'w')
        csv_file.write('step,time,obstacle_dist,cpg_weight,competence,correction,da,pitch,roll,upright\n')
        print(f'  CSV log: {args.log_csv}')

    imu = IMUReader()
    ultrasonic = UltrasonicReader()

    dashboard = None
    if args.dashboard:
        for dp in DASHBOARD_PATHS:
            if os.path.exists(os.path.join(dp,'freenove_dashboard.py')):
                sys.path.insert(0,dp); break
        try:
            from freenove_dashboard import DashboardServer
            dashboard=DashboardServer(port=8080); dashboard.start()
        except ImportError: dashboard=None

    # === SNN + Cerebellum (same code as simulator) ===
    snn = None; cb = None; gate = None; reward_computer = None
    brain_path = os.path.join(HOME_DIR, 'brain.pt')

    if args.snn:
        try:
            snn, cb = build_freenove_snn(device='cpu')
            gate = CompetenceGate(cpg_max=0.9, cpg_min=0.4)
            reward_computer = RewardComputer()

            # Load brain (same format as simulator)
            if os.path.exists(brain_path) and not args.fresh:
                try:
                    state = torch.load(brain_path, map_location='cpu', weights_only=False)
                    # Check topology
                    if 'snn' in state:
                        saved_n = state['snn'].get('V', torch.zeros(0)).shape[0]
                        if saved_n == snn.config.n_neurons:
                            from src.brain.brain_persistence import _load_snn
                            _load_snn(snn, state['snn'])
                            if 'cerebellum_state' in state and cb:
                                cb.load_state_dict(state['cerebellum_state'])
                            gate.actor_competence = state.get('actor_competence', 0.0)
                            gate.cpg_weight = state.get('cpg_weight', 0.9)
                            print(f'  Brain loaded: {brain_path}')
                            print(f'  Competence: {gate.actor_competence:.3f}  CPG: {gate.cpg_weight:.0%}')
                        else:
                            print(f'  Brain topology mismatch ({saved_n} vs {snn.config.n_neurons}) — fresh start')
                    else:
                        print(f'  Brain format not recognized — fresh start')
                except Exception as e:
                    print(f'  Brain load failed ({e}) — fresh start')
            elif args.fresh:
                print(f'  Fresh start (ignoring saved brain)')
            else:
                print(f'  No brain found — starting fresh')

        except ImportError as e:
            print(f'  ERROR: Cannot import src.brain modules: {e}')
            print(f'  Make sure src/brain/ is accessible and PyTorch is installed')
            args.snn = False

    driver = FreenoveServoDriver(dry_run=args.dry_run)
    servo = driver.get_servo()
    step = 0

    def sig_handler(sig, frame):
        _save_brain()
        driver.emergency_stop(); sys.exit(0)

    def _save_brain():
        if snn and cb and gate:
            print(f'\n  Saving brain...')
            state = {
                'snn': {
                    'V': snn.V.cpu(),
                    'spikes': snn.spikes.cpu(),
                    'refractory_counter': snn.refractory_counter.cpu(),
                    'neuron_types': snn.neuron_types.cpu(),
                    'neuromod_levels': dict(snn.neuromod_levels),
                    'neuromod_sensitivity': snn.neuromod_sensitivity.cpu(),
                    'populations': {k: v.cpu() for k, v in snn.populations.items()},
                    'weight_indices': snn._weight_indices.cpu() if snn._weight_indices is not None else None,
                    'weight_values': snn._weight_values.cpu() if snn._weight_values is not None else None,
                    'eligibility': snn._eligibility.cpu(),
                    'pre_trace': snn._pre_trace.cpu(),
                    'post_trace': snn._post_trace.cpu(),
                    'thresholds': snn._thresholds.cpu(),
                    'astro_calcium': snn._astro_calcium.cpu(),
                    'spike_count_window': snn._spike_count_window.cpu(),
                    'homeostatic_step_count': snn._homeostatic_step_count,
                    'step_count': snn.step_count,
                },
                'cerebellum_state': cb.state_dict() if cb else None,
                'actor_competence': gate.actor_competence,
                'cpg_weight': gate.cpg_weight,
                'step': step,
                'version': 'v4.1',
            }
            torch.save(state, brain_path)
            print(f'  Brain saved: {brain_path} (step {step}, '
                  f'competence {gate.actor_competence:.3f})')

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    standing = [[10,99,10],[10,99,10],[10,99,-10],[10,99,-10]]
    if args.gait in ('stand','stop'):
        if servo: set_legs(servo, standing)
        print(f'  Standing. Press Ctrl+C to stop.')
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: driver.emergency_stop()
        return

    cpg = FreenoveCPG(speed=args.speed)
    print(f'  CPG: freq={cpg.frequency:.2f}Hz  stride={cpg.stride}mm  lift={cpg.lift}mm')
    if servo: set_legs(servo, standing)
    time.sleep(1.0)
    print(f'  Walking! Press Ctrl+C to stop.\n')

    t_start=time.time(); t_loop=time.time()
    prev_servo=None; da_signal=0.5
    max_steps = args.steps if args.steps > 0 else 999999999
    max_duration = 9999 if args.steps > 0 else args.duration

    try:
        while (time.time()-t_start) < max_duration and step < max_steps:
            t_step=time.time()
            imu_data = imu.update()
            obstacle_dist = ultrasonic.read()
            obstacle_dist_cm = obstacle_dist * 100.0

            # === OBSTACLE REFLEXES (brainstem, hardwired) ===
            # Priority: reverse > stop > slow > normal
            # The cerebellum calibrates WHEN and HOW MUCH, but these
            # reflexes fire regardless of SNN state.
            # Biology: Trigeminal reflex arc (face contact → brainstem
            # → immediate motor response, ~10ms latency in mammals)
            reflex_active = False
            if obstacle_dist_cm < REFLEX_REVERSE_DISTANCE_CM and obstacle_dist_cm > 0:
                # CONTACT: Reverse CPG — back up
                cpg.set_inhibition(0.6)
                cpg.set_reverse(True)
                reflex_active = True
            elif obstacle_dist_cm < REFLEX_STOP_DISTANCE_CM:
                # COLLISION IMMINENT: CPG kill — full stop
                cpg.set_inhibition(0.0)
                cpg.set_reverse(False)
                reflex_active = True
            elif obstacle_dist_cm < REFLEX_SLOW_DISTANCE_CM:
                # DANGER: Graded CPG inhibition — slow down proportionally
                slow_factor = (obstacle_dist_cm - REFLEX_STOP_DISTANCE_CM) / (REFLEX_SLOW_DISTANCE_CM - REFLEX_STOP_DISTANCE_CM)
                cpg.set_inhibition(max(0.2, slow_factor))
                cpg.set_reverse(False)
                reflex_active = True
            else:
                # CLEAR: Full CPG output
                cpg.set_inhibition(1.0)
                cpg.set_reverse(False)

            cpg_points = cpg.compute(CONTROL_DT)

            # Issue #108: Obstacle Avoidance Turn Reflex
            # Asymmetric Z-offset on CPG points → robot turns away.
            # Same logic as train_v032.py _reflex_turn_steering.
            # Z > 0 = left side, Z < 0 = right side.
            # Positive Z-offset on all legs → robot turns left.
            _REFLEX_TURN_GAIN_HW = 5.0  # mm of Z-offset at max proximity
            if reflex_active and not cpg._reverse:
                _turn_prox = max(0.0, 1.0 - obstacle_dist_cm / REFLEX_SLOW_DISTANCE_CM)
                _turn_z = _turn_prox * _REFLEX_TURN_GAIN_HW
                for i in range(4):
                    cpg_points[i][2] += _turn_z  # shift all legs laterally

            if args.snn and snn and gate:
                sensory = encode_sensory(prev_servo, cpg.get_phase_input(), step, imu_data,
                                         obstacle_distance=obstacle_dist)

                # SNN step — same as simulator creature.think()
                sensor_tensor = torch.zeros(snn.config.n_neurons, dtype=torch.float32)
                # Population coding: raw values → input current
                threshold = snn.config.v_threshold
                for i, val in enumerate(sensory):
                    if i < len(snn.populations.get('input', [])):
                        sensor_tensor[i] = float(val) * threshold * 2.0
                # Tonic on hidden
                hidden_ids = snn.populations.get('hidden', torch.tensor([]))
                if len(hidden_ids) > 0:
                    sensor_tensor[hidden_ids] += 0.015

                # 3 substeps (same as bridge v2.5)
                output_ids = snn.populations.get('output', torch.tensor([]))
                output_spikes = torch.zeros(len(output_ids))
                for _ in range(3):
                    spikes = snn.step(sensor_tensor)
                    output_spikes += spikes[output_ids].float()

                # Decode output
                rates = (output_spikes / 3.0).numpy()
                actor_output = np.clip((rates - 0.5) * 2.0, -1.0, 1.0)

                # Reward
                da_signal = reward_computer.compute(prev_servo, imu_data)
                snn.neuromod_levels['da'] = da_signal

                # R-STDP (same as simulator)
                snn.apply_rstdp(reward_signal=da_signal)

                # Cerebellum update — SAME code as simulator
                if cb:
                    # Build sensor_data dict matching what InferiorOlive expects
                    sensor_data = {
                        'orientation_euler': np.array([
                            np.radians(imu_data['roll']),
                            np.radians(imu_data['pitch']),
                            0.0
                        ]),
                        'height': 0.12,  # standing height
                        'standing_height': 0.12,
                        'upright': imu_data['upright'],
                        'forward_velocity': 0.05,  # estimated
                        'desired_velocity': 0.15,
                        'step': step,
                        'velocity': np.array([0.05, 0.0, 0.0]),
                        'angular_velocity': np.array([0.0, 0.0, 0.0]),
                        'obstacle_distance': obstacle_dist,
                    }
                    if prev_servo:
                        n_act = min(12, len(prev_servo))
                        sensor_data['joint_positions'] = np.array(prev_servo[:n_act]) / 180.0
                        sensor_data['motor_commands'] = actor_output[:n_act]
                    cb.update(None, sensor_data)  # creature=None, uses sensor_data directly
                    cb_corr = cb.compute_corrections(actor_output.tolist(), upright=imu_data['upright'])
                else:
                    cb_corr = None

                is_upright = imu_data['upright'] > 0.5
                gate.update(is_upright=is_upright, velocity=0.1)
                final_points = gate.blend(cpg_points, actor_output)

                # Apply cerebellar corrections WITH hardware amplification
                # Strategy 1 (Issue #105): ×5 gain to overcome servo dead band
                if cb_corr is not None:
                    for leg_idx in range(4):
                        base = leg_idx * 3
                        if base + 2 < len(cb_corr):
                            final_points[leg_idx][0] += cb_corr[base] * 3.0 * HARDWARE_CORRECTION_GAIN
                            final_points[leg_idx][1] += cb_corr[base+1] * 2.0 * HARDWARE_CORRECTION_GAIN
                            final_points[leg_idx][2] += cb_corr[base+2] * 1.0 * HARDWARE_CORRECTION_GAIN
            else:
                final_points = cpg_points

            if servo: set_legs(servo, final_points)

            if args.snn:
                try:
                    prev_servo=[]
                    for p in final_points:
                        a,b,c=ik(p[0],p[1],p[2]); prev_servo.extend([a,b,c])
                except: prev_servo=None

            if args.verbose and step%50==0 and step>0:
                ms=(time.time()-t_step)*1000
                if args.snn and gate and snn:
                    gs=gate.get_stats()
                    fr=float(snn.spikes.float().mean())
                    cf = cb.stats.get('cf_magnitude', 0.0) if cb else 0.0
                    corr = cb.stats.get('correction_magnitude', 0.0) if cb else 0.0
                    pw = cb.stats.get('pf_pkc_mean_weight', 0.0) if cb else 0.0
                    reb = cb.stats.get('dcn_rebound_strength', 0.0) if cb else 0.0
                    rfx = 'REV' if cpg._reverse else ('STOP' if cpg._inhibition < 0.01 else
                          (f'SLOW{cpg._inhibition:.0%}' if cpg._inhibition < 1.0 else ''))
                    print(f'  [{step:5d}] {ms:.1f}ms  CPG:{gs["cpg_weight"]:.0%}  '
                          f'Act:{gs["actor_competence"]:.3f}  DA:{da_signal:.2f}  '
                          f'FR:{fr:.3f}  CF:{cf:.3f}  corr:{corr:.4f}  reb:{reb:.3f}  '
                          f'P:{imu_data["pitch"]:+.1f}  '
                          f'R:{imu_data["roll"]:+.1f}  Up:{imu_data["upright"]:.2f}  '
                          f'OD:{obstacle_dist:.2f}m {rfx}')
                else:
                    print(f'  [{step:5d}] {ms:.1f}ms  CPG only  '
                          f'P:{imu_data["pitch"]:+.1f}  R:{imu_data["roll"]:+.1f}  '
                          f'OD:{obstacle_dist:.2f}m')

            # CSV logging — every 10 steps for smooth curves
            if csv_file and step % 10 == 0:
                t_elapsed = time.time() - t_start
                _csv_cpg = gate.get_stats()['cpg_weight'] if gate else 1.0
                _csv_comp = gate.get_stats()['actor_competence'] if gate else 0.0
                _csv_corr = cb.stats.get('correction_magnitude', 0.0) if cb else 0.0
                _csv_da = da_signal if args.snn else 0.0
                csv_file.write(f'{step},{t_elapsed:.3f},{obstacle_dist:.4f},'
                               f'{_csv_cpg:.4f},{_csv_comp:.4f},{_csv_corr:.6f},'
                               f'{_csv_da:.4f},{imu_data["pitch"]:.2f},'
                               f'{imu_data["roll"]:.2f},{imu_data["upright"]:.4f}\n')


            # Dashboard update — real SNN data
            if dashboard:
                ms_step = (time.time()-t_step)*1000
                dash_state = {
                    "step": step, "gait": args.gait, "ms_per_step": ms_step,
                    "snn_enabled": args.snn, "cpg_phase": cpg._phase,
                    "uptime": time.time()-t_start,
                }
                if prev_servo:
                    servo_dict = {}
                    ch_map = [4,3,2, 7,6,5, 8,9,10, 11,12,13]
                    for ci, ch in enumerate(ch_map):
                        if ci < len(prev_servo):
                            servo_dict[ch] = prev_servo[ci]
                    dash_state["servos"] = servo_dict
                if args.snn and snn and gate:
                    gs = gate.get_stats()
                    dash_state["snn"] = {
                        "firing_rate": float(snn.spikes.float().mean()),
                        "mean_weight": float(snn._weight_values.abs().mean()) if snn._weight_values is not None else 0.0,
                        "da_level": da_signal,
                    }
                    dash_state["gate"] = gs
                    # Real neuron spikes (232 bools)
                    dash_state["neurons"] = snn.spikes.bool().tolist()
                dashboard.update(dash_state)
            step+=1; t_loop+=CONTROL_DT
            s=t_loop-time.time()
            if s>0: time.sleep(s)
            else: t_loop=time.time()

    except KeyboardInterrupt: pass

    _save_brain()

    print(f'\n  Stopping... ({step} steps in {time.time()-t_start:.1f}s)')
    if servo: set_legs(servo, standing)
    ultrasonic.cleanup()
    if csv_file:
        csv_file.close()
        print(f'  CSV log saved: {args.log_csv} ({step // 10} data points)')
    if args.snn and gate:
        gs=gate.get_stats()
        print(f'  Final: competence={gs["actor_competence"]:.3f}  CPG={gs["cpg_weight"]:.0%}')
    print(f'  Done.')

if __name__=='__main__':
    main()
