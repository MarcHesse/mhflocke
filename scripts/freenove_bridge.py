#!/usr/bin/env python3
"""
MH-FLOCKE -- Freenove Robot Dog Bridge v4.4
=============================================
v4.4: Asymmetric stride steering with IMU closed-loop (Issue #76d v2).
      Hardware tests (2026-05-03) proved Z-offset steering too weak
      to overcome mechanical drift. New approach: differential hip stride
      (left vs right) with PD controller on IMU yaw feedback.
      Replaces Z-offset phototaxis with IMU-closed-loop target-yaw.
v4.3: SpatialMap on hardware -- dead-reckoned position + landmark memory
      from path integration (no GPS, no SLAM). CSV gains phototaxis
      navigation columns, brain-map snapshots written to JSONL sidecar.
      Same FLOG schema as the simulator (FLOG_FORMAT.md v1.2).
v4.2: LightMemory — spatial yaw memory for phototaxis recovery.
      When the light disappears from the camera, the dog remembers
      the last known yaw angle and steers back toward it.
      Biology: Hippocampal place cell → head direction cell circuit.
      Also fixes: duplicate step+=1 bug from v4.1.
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

import time, math, argparse, os, signal, sys, json

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
        self.height=height; self.stride=int(12 * speed); self.lift=int(6 * speed)
        self.frequency=0.8*speed*HARDWARE_CPG_FREQ_SCALE; self._phase=90.0
        self._inhibition = 1.0  # 1.0=full, 0.0=stopped
        self._reverse = False   # True = walk backward
        self._steering = 0.0    # v4.4: asymmetric stride, -1..+1
    def set_inhibition(self, level):
        """Set CPG output inhibition. 0.0=stopped, 1.0=full stride."""
        self._inhibition = max(0.0, min(1.0, level))
    def set_reverse(self, reverse):
        """Set reverse walking mode."""
        self._reverse = reverse
    def set_steering(self, steering):
        """Set asymmetric stride for turning. Positive = turn right.
        Hardware-validated (Test C, 2026-05-03): asymmetric stride is
        3x more effective than Z-offset for overcoming mechanical drift.
        Convention: steering > 0 -> left legs longer stride, right shorter -> turns right.
        """
        self._steering = max(-0.6, min(0.6, steering))
    def compute(self, dt=None):
        if dt is None: dt=CONTROL_DT
        direction = -1.0 if self._reverse else 1.0
        self._phase += 360.0*self.frequency*dt*direction; h=self.height
        pr=self._phase*math.pi/180.0; po=(self._phase+180.0)*math.pi/180.0
        stride = self.stride * self._inhibition
        lift = self.lift * self._inhibition
        # v4.4: Asymmetric stride for steering (differential/tank steering)
        stride_left  = stride * (1.0 + self._steering)
        stride_right = stride * (1.0 - self._steering)
        # Left legs (indices 0,1): Front-Left, Rear-Left
        X1_L=stride_left*math.cos(pr); Y1=lift*math.sin(pr)+h
        if Y1>h: Y1=h
        X2_L=stride_left*math.cos(po); Y2=lift*math.sin(po)+h
        if Y2>h: Y2=h
        # Right legs (indices 2,3): Front-Right, Rear-Right
        X1_R=stride_right*math.cos(pr); Y1_R=lift*math.sin(pr)+h
        if Y1_R>h: Y1_R=h
        X2_R=stride_right*math.cos(po); Y2_R=lift*math.sin(po)+h
        if Y2_R>h: Y2_R=h
        return [[X1_L+10,Y1,10],[X2_L+10,Y2,10],[X1_R+10,Y1_R,-10],[X2_R+10,Y2_R,-10]]
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


class CameraReader:
    """
    Pi Camera light-source detector for phototaxis.

    Uses libcamera (rpicam-vid) on Pi, falls back to cv2.VideoCapture on other platforms.
    Detects the brightest region in the camera frame and returns
    heading + salience for VOR steering.

    Biology: Superior Colliculus — detects bright stimulus location
    and drives saccade/orienting reflex toward it.
    """

    def __init__(self, width=320, height=240, brightness_threshold=200):
        self.cap = None
        self._pipe = None
        self._available = False
        self._mode = None  # 'libcamera' or 'cv2'
        self.width = width
        self.height = height
        self.threshold = brightness_threshold
        self.heading = 0.0
        self.salience = 0.0
        self.cv2 = None
        self._frame_size = width * height * 3  # RGB bytes per frame

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            print('  Camera: cv2 not installed (pip3 install opencv-python-headless)')
            return

        # Try libcamera first (Pi Camera)
        if self._try_libcamera():
            return

        # Fallback: cv2 VideoCapture (USB cameras, desktop)
        if self._try_cv2():
            return

        print('  Camera: no working camera found')

    def _try_libcamera(self):
        """Start rpicam-vid as a pipe, streaming raw RGB frames."""
        try:
            import subprocess, shutil
            rpicam = shutil.which('rpicam-vid')
            if not rpicam:
                return False
            cmd = [
                rpicam, '-t', '0',  # run indefinitely
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', '15',
                '--codec', 'yuv420',
                '--nopreview',
                '-o', '-'  # output to stdout
            ]
            self._pipe = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=self._frame_size * 2)
            # Read and discard first few frames (warmup)
            yuv_size = self.width * self.height * 3 // 2  # YUV420
            self._yuv_size = yuv_size
            for _ in range(5):
                self._pipe.stdout.read(yuv_size)
            self._mode = 'libcamera'
            self._available = True
            print(f'  Camera: libcamera {self.width}x{self.height} @ 15fps (phototaxis)')
            return True
        except Exception as e:
            print(f'  Camera: libcamera failed ({e})')
            if self._pipe:
                self._pipe.kill()
                self._pipe = None
            return False

    def _try_cv2(self):
        """Try cv2 VideoCapture (USB cameras, desktop)."""
        try:
            self.cap = self.cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(self.cv2.CAP_PROP_FPS, 15)
                # Test if we actually get frames
                import time
                time.sleep(0.5)
                ret, _ = self.cap.read()
                if ret:
                    self._mode = 'cv2'
                    self._available = True
                    print(f'  Camera: cv2 {self.width}x{self.height} @ 15fps (phototaxis)')
                    return True
                else:
                    self.cap.release()
                    self.cap = None
            return False
        except Exception as e:
            print(f'  Camera: cv2 failed ({e})')
            return False

    def update(self):
        """Capture frame, find brightest region, return heading + salience."""
        if not self._available:
            return {'heading': 0.0, 'salience': 0.0}

        cv2 = self.cv2
        gray = None

        if self._mode == 'libcamera':
            # Read YUV420 frame from pipe
            raw = self._pipe.stdout.read(self._yuv_size)
            if len(raw) < self._yuv_size:
                return {'heading': 0.0, 'salience': 0.0}
            # Y channel is the first W*H bytes (grayscale)
            import numpy as np
            gray = np.frombuffer(raw[:self.width * self.height],
                                dtype=np.uint8).reshape(self.height, self.width)
        elif self._mode == 'cv2':
            ret, frame = self.cap.read()
            if not ret:
                return {'heading': 0.0, 'salience': 0.0}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray is None:
            return {'heading': 0.0, 'salience': 0.0}

        # Gaussian blur to smooth noise
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # Find brightest pixel
        _, max_val, _, max_loc = cv2.minMaxLoc(gray)

        if max_val < self.threshold:
            self.heading = 0.0
            self.salience = 0.0
            return {'heading': 0.0, 'salience': 0.0}

        # Heading: normalize x position to [-1, 1]
        center_x = max_loc[0]
        self.heading = (center_x / self.width - 0.5) * 2.0

        # Salience: brightness + area of bright region
        _, bright_mask = cv2.threshold(gray, int(self.threshold * 0.8), 255, cv2.THRESH_BINARY)
        bright_area = cv2.countNonZero(bright_mask)
        area_ratio = bright_area / (self.width * self.height)
        brightness_norm = (max_val - self.threshold) / (255 - self.threshold)
        self.salience = min(1.0, brightness_norm * 0.6 + area_ratio * 10.0)

        return {'heading': self.heading, 'salience': self.salience}

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self._pipe:
            self._pipe.kill()
            self._pipe.wait()
            self._pipe = None


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
    Build a 560-neuron SNN with cerebellar + motor hidden architecture.
    IDENTICAL topology to MuJoCoCreatureBuilder.build() with profile.json.
    
    v4.2: Aligned with simulator — 500 hidden neurons = 560 total.
          Includes Motor Hidden (motorcortex) for R-STDP learning.
          Bilateral MH→Output symmetry (drift fix v0.5.2).
    
    Returns: snn, cerebellar_learning
    """
    import torch
    from src.brain.snn_controller import SNNController, SNNConfig
    from src.brain.cerebellar_learning import CerebellarLearning, CerebellarConfig
    from src.brain.topology import compute_cerebellar_populations

    n_input = 48
    n_output = 12
    n_hidden = 500   # Same as profile.json snn.n_hidden
    n_actuators = 12

    # Compute cerebellar populations (same function as simulator, v0.7.1)
    cb_pops = compute_cerebellar_populations(n_hidden, n_actuators)
    n_granule = cb_pops['n_granule']   # 269
    n_golgi = cb_pops['n_golgi']       # 47
    n_purkinje = cb_pops['n_purkinje'] # 24
    n_dcn = cb_pops['n_dcn']           # 24

    # Motor hidden neurons (motorcortex) — same as simulator v0.7.0
    # Free hidden neurons for R-STDP motor pattern learning.
    # NOT part of the cerebellum and NOT protected from R-STDP.
    n_cerebellum = n_granule + n_golgi + n_purkinje + n_dcn
    n_motor_hidden = max(0, n_hidden - n_cerebellum)  # 136

    total_neurons = n_input + n_output + n_cerebellum + n_motor_hidden  # 560

    # Neuron ID ranges (must match simulator build() order)
    grc_start = n_input + n_output                    # 60
    goc_start = grc_start + n_granule                 # 329
    pkc_start = goc_start + n_golgi                   # 376
    dcn_start = pkc_start + n_purkinje                # 400
    motor_hidden_start = dcn_start + n_dcn            # 424

    # Create SNN
    snn = SNNController(SNNConfig(
        n_neurons=total_neurons,
        connectivity_prob=0.0,  # We build connectivity manually
        homeostatic_interval=200,
        device=device,
    ))

    # === Populations — Cerebellar Architecture (identical to simulator) ===
    mf_ids = torch.arange(0, n_input)
    snn.define_population('input', mf_ids)
    snn.define_population('mossy_fibers', mf_ids)

    out_ids = torch.arange(n_input, n_input + n_output)
    snn.define_population('output', out_ids)

    grc_ids = torch.arange(grc_start, grc_start + n_granule)
    snn.define_population('granule_cells', grc_ids)

    goc_ids = torch.arange(goc_start, goc_start + n_golgi)
    snn.define_population('golgi_cells', goc_ids)

    pkc_ids = torch.arange(pkc_start, pkc_start + n_purkinje)
    snn.define_population('purkinje_cells', pkc_ids)

    dcn_ids = torch.arange(dcn_start, dcn_start + n_dcn)
    snn.define_population('dcn', dcn_ids)

    # Motor hidden population (motorcortex) — same as simulator v0.7.0
    if n_motor_hidden > 0:
        mh_ids = torch.arange(motor_hidden_start, motor_hidden_start + n_motor_hidden)
        snn.define_population('motor_hidden', mh_ids)
        # Include in 'hidden' for tonic current
        all_hidden_ids = torch.cat([grc_ids, mh_ids])
        snn.define_population('hidden', all_hidden_ids)
    else:
        mh_ids = torch.tensor([], dtype=torch.long)
        snn.define_population('hidden', grc_ids)

    # === Connectivity — Biologically structured (identical to simulator) ===

    # Scale connectivity proportionally to GrC count
    grc_goc_prob = min(0.3, 0.05 * (4000 / max(n_granule, 1)))
    mf_per_grc = min(4, max(2, n_input // max(n_granule, 1)))

    # MF -> GrC: each GrC receives mf_per_grc MF inputs
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

    # GrC -> GoC
    snn.connect_populations('granule_cells', 'golgi_cells',
                            prob=grc_goc_prob, weight_range=(0.3, 0.8))
    # GoC -> GrC (inhibitory)
    snn.neuron_types[goc_ids] = -1.0
    snn.connect_populations('golgi_cells', 'granule_cells',
                            prob=0.02, weight_range=(0.2, 0.4))
    # MF -> GoC
    snn.connect_populations('mossy_fibers', 'golgi_cells',
                            prob=0.1, weight_range=(0.5, 1.0))
    # GrC -> Output (legacy direct path)
    snn.connect_populations('granule_cells', 'output',
                            prob=0.02, weight_range=(0.3, 0.8))

    # Motor hidden connectivity (motorcortex) — same as simulator v0.7.0
    if n_motor_hidden > 0:
        # MF -> motor_hidden (sensory input)
        snn.connect_populations('mossy_fibers', 'motor_hidden',
                                prob=0.1, weight_range=(0.5, 1.5))
        # motor_hidden -> output (motor drive)
        snn.connect_populations('motor_hidden', 'output',
                                prob=0.15, weight_range=(0.3, 1.0))

        # v0.5.2: Bilateral symmetry for MH->Output weights (Issue #145)
        # Without this, random init creates L/R asymmetry → drift.
        # Actuator layout: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
        _bilateral_pairs = [(0, 3), (1, 4), (2, 5),   # FL <-> FR
                            (6, 9), (7, 10), (8, 11)]  # RL <-> RR
        _out_start = out_ids[0].item()
        _n_per_joint = max(1, n_output // n_actuators)
        _idx = snn._weight_indices
        _w = snn._weight_values
        for _left_j, _right_j in _bilateral_pairs:
            _left_neurons = list(range(_out_start + _left_j * _n_per_joint,
                                       _out_start + (_left_j + 1) * _n_per_joint))
            _right_neurons = list(range(_out_start + _right_j * _n_per_joint,
                                        _out_start + (_right_j + 1) * _n_per_joint))
            for _ln, _rn in zip(_left_neurons, _right_neurons):
                _left_mask = (_idx[1] == _ln)
                _right_mask = (_idx[1] == _rn)
                if _left_mask.any() and _right_mask.any():
                    _avg = (_w[_left_mask].mean() + _w[_right_mask].mean()) / 2.0
                    _w[_left_mask] = _avg
                    _w[_right_mask] = _avg
        print(f'  MH->Output bilateral symmetry enforced ({len(_bilateral_pairs)} joint pairs)')

        # motor_hidden recurrent (pattern memory)
        snn.connect_populations('motor_hidden', 'motor_hidden',
                                prob=0.1, weight_range=(0.2, 0.6))

    # Per-population time constants
    snn._tau_base[grc_ids] = 5.0
    snn._tau_base[goc_ids] = 20.0
    snn._tau_base[pkc_ids] = 15.0
    snn._tau_base[dcn_ids] = 10.0
    if n_motor_hidden > 0:
        snn._tau_base[mh_ids] = 10.0  # Medium (cortical)

    # Thresholds
    snn._thresholds[grc_ids] = 0.5
    snn._thresholds[goc_ids] = 0.5
    snn._thresholds[pkc_ids] = 1.0
    snn._thresholds[dcn_ids] = 1.0
    snn._thresholds[out_ids] = 0.4
    if n_motor_hidden > 0:
        snn._thresholds[mh_ids] = 0.5
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

    # Protect cerebellar populations from R-STDP (motor_hidden is NOT protected)
    snn.protected_populations = {
        'mossy_fibers', 'granule_cells', 'golgi_cells',
        'purkinje_cells', 'dcn',
    }

    # Per-population Izhikevich parameters (Issue #104)
    snn.set_izhikevich_params('granule_cells', a=0.02, b=0.2, c=-65, d=8)   # RS
    snn.set_izhikevich_params('golgi_cells',   a=0.02, b=0.2, c=-55, d=4)   # IB
    snn.set_izhikevich_params('purkinje_cells', a=0.02, b=0.2, c=-50, d=2)  # CH
    snn.set_izhikevich_params('dcn',           a=0.03, b=0.25, c=-52, d=0)  # Rebound
    if n_motor_hidden > 0:
        snn.set_izhikevich_params('motor_hidden', a=0.02, b=0.2, c=-65, d=8)  # RS
    # Output neurons: RS (motoneurons are tonic/RS, not FS)
    snn.set_izhikevich_params('output', a=0.02, b=0.2, c=-65, d=8)  # RS
    snn.set_izhikevich_params('input', a=0.02, b=0.2, c=-65, d=8)   # RS

    print(f'  SNN: {total_neurons} neurons ({n_input}i + {n_output}o + '
          f'{n_granule}GrC + {n_golgi}GoC + {n_purkinje}PkC + {n_dcn}DCN + {n_motor_hidden}MH)')
    print(f'  Izhikevich: DCN=Rebound, GoC=IB, PkC=CH, MH=RS, Output=RS')

    return snn, cb

# ================================================================
# LIGHT MEMORY — spatial yaw recall for phototaxis recovery
# ================================================================

class LightMemory:
    """
    Remembers the last known yaw angle when the light was visible.
    When the light disappears, steers back toward the remembered yaw.

    This is the minimal spatial memory needed to close the phototaxis
    loop: instead of walking blindly when the light is lost, the dog
    turns back toward where it last saw the light.

    Biology: Head direction cells in the postsubiculum maintain a
    persistent representation of the animal's heading even without
    visual landmarks. When a landmark reappears, the HD cell ring
    recalibrates. This class models that — a persistent yaw target
    that the dog steers toward when the visual cue is lost.

    States:
      TRACKING  — light visible, continuously updating memory
      RETURNING — light lost, steering back to remembered yaw
      LOST      — timeout expired, no memory-based steering
                  (future: trigger Run-and-Tumble search)

    RPi cost: ~0.01ms per step (2 float comparisons + 1 subtraction).
    """

    # State constants
    TRACKING = 'tracking'
    RETURNING = 'returning'
    LOST = 'lost'

    def __init__(self, return_gain: float = 0.4, timeout_seconds: float = 10.0,
                 arrival_threshold_deg: float = 15.0, z_sign: float = -1.0):
        """
        Args:
            return_gain: Proportional gain for yaw return steering (deg → mm Z-offset).
                         0.4 = moderate. Lower than PHOTOTAXIS_GAIN to avoid overshoot.
            timeout_seconds: How long to try returning before giving up.
            arrival_threshold_deg: Within this many degrees of target = "arrived".
            z_sign: Sign convention for Z-offset to yaw mapping.
                    -1.0 = simulator (positive Z → turn left → positive yaw)
                    +1.0 = Freenove hardware (positive Z → turn right → negative yaw)
                    Memory steering needs the OPPOSITE sign of the live
                    phototaxis steering: we are pushing yaw_error toward
                    zero, not driving the body toward a heading offset.
                    Hardware-measured 2026-05-02 from CSV step 600–750.
        """
        self.return_gain = return_gain
        self.timeout_seconds = timeout_seconds
        self.arrival_threshold_deg = arrival_threshold_deg
        self.z_sign = z_sign

        # State
        self.state = self.LOST  # Start as LOST (no memory yet)
        self._target_yaw = 0.0           # Remembered yaw angle (degrees)
        self._last_heading = 0.0         # Camera heading when last seen
        self._lost_time = 0.0            # When light was lost (time.time())
        self._tracking_steps = 0         # How long we tracked before losing

    def update(self, cam_salience: float, cam_heading: float,
               current_yaw: float, current_time: float) -> float:
        """
        Update memory state and compute steering correction.

        Args:
            cam_salience: Camera brightness salience (0 = no light, 1 = bright).
            cam_heading: Camera heading (-1 left, +1 right).
            current_yaw: IMU yaw in degrees.
            current_time: time.time() for timeout tracking.

        Returns:
            Z-offset in mm for CPG leg points. 0.0 if no memory steering.
            This is ADDED to the phototaxis Z-offset (which is 0 when no light).
        """
        if cam_salience > 0.05:
            # === LIGHT VISIBLE → TRACKING ===
            # Save the yaw TOWARD the light, not just body orientation.
            # cam_heading [-1,+1] maps to ~[-31°,+31°] (half Pi Camera FOV).
            # target_yaw = body_yaw + heading_offset = direction to light.
            _HALF_FOV_DEG = 31.0  # Pi Camera v2 FOV ~62°
            self.state = self.TRACKING
            self._target_yaw = current_yaw + cam_heading * _HALF_FOV_DEG
            self._last_heading = cam_heading
            self._tracking_steps += 1
            return 0.0  # No memory steering needed — camera handles it

        # === LIGHT NOT VISIBLE ===
        if self.state == self.TRACKING:
            # Just lost the light — transition to RETURNING
            self.state = self.RETURNING
            self._lost_time = current_time
            # target_yaw already holds the last known position

        if self.state == self.RETURNING:
            # Check timeout
            elapsed = current_time - self._lost_time
            if elapsed > self.timeout_seconds:
                self.state = self.LOST
                self._tracking_steps = 0
                return 0.0  # Give up

            # Compute yaw error: how far are we from the remembered yaw?
            yaw_error = current_yaw - self._target_yaw
            # Normalize to [-180, 180]
            while yaw_error > 180.0: yaw_error -= 360.0
            while yaw_error < -180.0: yaw_error += 360.0

            # If close enough, hold position (don't oscillate)
            if abs(yaw_error) < self.arrival_threshold_deg:
                return 0.0

            # Proportional steering back toward target
            # z_sign controls the mapping from yaw_error to Z-offset:
            #   Simulator (z_sign=-1): +error → -Z → turn left toward target
            #   Hardware  (z_sign=+1): +error → +Z → turn right toward target (inverted coupling)
            memory_z = self.z_sign * yaw_error * self.return_gain

            # Clamp: hardware measurement shows -12mm produces only 0.21 deg/s.
            # To turn 21° in 10s timeout → need ~2.1 deg/s → ~100mm.
            # Set clamp at ±50mm — strong but not destabilizing.
            # (Camera steering uses ±12mm but is closed-loop;
            #  Memory steering is open-loop and needs more authority.)
            memory_z = max(-50.0, min(50.0, memory_z))

            return memory_z

        # state == LOST: no memory, no steering
        return 0.0

    def get_state_label(self) -> str:
        """Short label for CSV/dashboard."""
        return self.state[0].upper()  # 'T', 'R', 'L'

    def get_stats(self) -> dict:
        """Stats for dashboard/logging."""
        return {
            'memory_state': self.state,
            'target_yaw': self._target_yaw,
            'tracking_steps': self._tracking_steps,
        }


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

    parser = argparse.ArgumentParser(description='MH-FLOCKE Freenove Bridge v4.3')
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
    parser.add_argument('--phototaxis', action='store_true',
                        help='Enable camera-based phototaxis (follow flashlight)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log-csv', type=str, default=None,
                        help='Log sensor data to CSV file for A/B comparison')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'  MH-FLOCKE — Freenove Bridge v4.3 (unified + servo comp + light memory + spatial map)')
    print(f'  Same SNN + Cerebellum as simulator (PyTorch)')
    print(f'  Gait: {args.gait}  Speed: {args.speed}  Duration: {args.duration}s')
    print(f'  CPG freq scale: {HARDWARE_CPG_FREQ_SCALE}  Correction gain: {HARDWARE_CORRECTION_GAIN}x')
    print(f'  SNN: {"ON" if args.snn else "OFF"}  Dashboard: {"ON" if args.dashboard else "OFF"}')
    print(f'{"="*60}\n')

    # CSV logger for A/B comparison experiments
    csv_file = None
    brain_jsonl_path = None
    brain_jsonl = None
    if args.log_csv:
        csv_file = open(args.log_csv, 'w')
        # Same field semantics as simulator FLOG (FLOG_FORMAT.md v1.2):
        #   pos_x/pos_y           — ground truth (sentinel -999 on hardware, no GT)
        #   dist_to_light         — ground truth distance (sentinel -1 on hardware)
        #   heading_to_light      — ground truth heading (sentinel -999 on hardware)
        #   intent_yaw_rate       — current motor steering Z-offset (mm)
        #   brain_pos_x/y         — dead-reckoned belief from SpatialMap
        #   brain_pos_error       — sentinel -1 on hardware (no GT to compare with)
        csv_file.write('step,time,obstacle_dist,cpg_weight,competence,correction,da,'
                       'pitch,roll,upright,yaw,cam_heading,cam_salience,steer_z,drift_bias,'
                       'memory_state,memory_z,'
                       'pos_x,pos_y,dist_to_light,heading_to_light,intent_yaw_rate,'
                       'brain_pos_x,brain_pos_y,brain_pos_error\n')
        csv_file.flush()
        print(f'  CSV log: {args.log_csv}')
        # JSONL sidecar for brain-map snapshots (landmarks + visit grid)
        # One snapshot per call, same schema as FLOG brain_landmarks_json /
        # brain_visit_grid_b64 fields. Read by renderer to draw the
        # "what the dog believes" minimap.
        _csv_stem, _ = os.path.splitext(args.log_csv)
        brain_jsonl_path = _csv_stem + '.brain.jsonl'
        brain_jsonl = open(brain_jsonl_path, 'w')
        print(f'  Brain map sidecar: {brain_jsonl_path}')

    imu = IMUReader()
    ultrasonic = UltrasonicReader()
    camera = CameraReader() if args.phototaxis else None
    light_memory = LightMemory(return_gain=0.8, timeout_seconds=10.0,
                               arrival_threshold_deg=0.0,
                               z_sign=+1.0) if args.phototaxis else None

    # SpatialMap — dead-reckoned position + landmark memory.
    # Path integration from velocity proxy (CPG inhibition) + IMU yaw.
    # No GPS, no external odometry. Drift is real and visible — that is
    # the point. The renderer compares brain belief vs. measured ground
    # truth (in sim) or shows belief alone (on hardware).
    spatial_map = None
    try:
        from src.brain.spatial_map import SpatialMap
        spatial_map = SpatialMap(world_size=10.0, grid_resolution=20)
        # Try to restore previous map state for continuity across runs.
        _spatial_state_path = os.path.join(HOME_DIR, 'spatial_map.pt')
        if os.path.exists(_spatial_state_path) and not args.fresh:
            try:
                import torch as _torch
                _sm_state = _torch.load(_spatial_state_path, map_location='cpu', weights_only=False)
                spatial_map.load_state_dict(_sm_state)
                print(f'  SpatialMap loaded: {_spatial_state_path} '
                      f'({len(spatial_map.landmarks)} landmarks, '
                      f'pos=({spatial_map.position[0]:.2f}, {spatial_map.position[1]:.2f}))')
            except Exception as _sm_e:
                print(f'  SpatialMap restore failed ({_sm_e}) — fresh map')
        else:
            print(f'  SpatialMap: fresh (no prior state)')
    except Exception as _sm_e:
        print(f'  SpatialMap unavailable ({_sm_e}) — brain map will not be logged')

    # Velocity proxy for path integration.
    # The robot has no wheel encoder, no foot-fall sensor, no optical flow.
    # We estimate forward speed from CPG state: when the gait is running
    # at full output the legs cycle at ~one full step per ~1.25s with a
    # ~5 cm stride — roughly 0.04 m/s linear. CPG inhibition scales this
    # linearly. This is a coarse estimate; SpatialMap drift is expected.
    _VELOCITY_AT_FULL_CPG = 0.04  # m/s, empirical Freenove walk gait

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
                'version': 'v4.3',
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
            _memory_z = 0.0  # v4.2: reset each step (set by LightMemory if active)
            imu_data = imu.update()
            obstacle_dist = ultrasonic.read()
            obstacle_dist_cm = obstacle_dist * 100.0

            # SpatialMap update — dead-reckoned position from velocity proxy + IMU yaw.
            # Velocity proxy: CPG inhibition * empirical full-walk speed.
            # When the CPG is killed (obstacle reflex) the dog isn't moving forward.
            if spatial_map is not None:
                _vel_est = _VELOCITY_AT_FULL_CPG * cpg._inhibition
                if cpg._reverse:
                    _vel_est = -_vel_est
                _yaw_rad = math.radians(imu_data.get('yaw', 0.0))
                spatial_map.update_position(_vel_est, _yaw_rad, dt=CONTROL_DT)

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
            # v4.4: Obstacle avoidance now uses asymmetric stride instead of Z-offset.
            # Positive steering = turn right (away from obstacle assumed on left).
            _REFLEX_TURN_GAIN_HW = 0.3  # steering units at max proximity
            if reflex_active and not cpg._reverse:
                _turn_prox = max(0.0, 1.0 - obstacle_dist_cm / REFLEX_SLOW_DISTANCE_CM)
                cpg.set_steering(_turn_prox * _REFLEX_TURN_GAIN_HW)

            # --- v4.4: Phototaxis with IMU closed-loop steering ---
            # Replaces Z-offset steering (v4.2-v4.3) which was too weak
            # to overcome mechanical drift (measured: <5 deg effect vs 70 deg drift).
            #
            # New approach: Camera provides target_yaw (relative to current heading).
            # PD controller on IMU yaw error drives asymmetric stride via CPG.
            # This automatically compensates ANY mechanical drift.
            #
            # Biology: Vestibulospinal tract — IMU (vestibular organ) feedback
            # directly modulates left/right stride amplitude. The cerebellum
            # does NOT need to learn drift compensation — the closed loop handles it.
            #
            # Hardware-validated: Test C (2026-05-03) reduced drift from -70 deg
            # to +/-8.5 deg over 45 seconds with Kp=0.03, Kd=0.015.
            if camera and args.phototaxis:
                cam_data = camera.update()
                _cam_heading = cam_data['heading']    # -1 left, +1 right
                _cam_salience = cam_data['salience']  # 0 none, 1 bright+close

                # Initialize PD controller state (once)
                if not hasattr(camera, '_pd_yaw_target'):
                    camera._pd_yaw_target = imu_data.get('yaw', 0.0)  # Start at current heading
                    camera._pd_yaw_start = imu_data.get('yaw', 0.0)
                    camera._pd_prev_error = 0.0
                    camera._pd_integral = 0.0
                    camera._pd_steering = 0.0          # smoothed output
                    camera._last_z_offset = 0.0        # compat for CSV logging
                    # PID gains (tuned on hardware 2026-05-05, increased for stronger drift)
                    camera._pd_kp = 0.05
                    camera._pd_ki = 0.01               # I-term: doubled for faster drift elimination
                    camera._pd_kd = 0.015
                    camera._pd_max = 0.6               # max steering asymmetry
                    camera._pd_integral_max = 30.0     # anti-windup clamp

                # Camera provides target yaw:
                # heading > 0 means light is to the RIGHT
                # -> we want to increase yaw_target (turn right)
                # Scale: heading [-1,+1] maps to ~[-30,+30] deg FOV
                _HALF_FOV_DEG = 31.0  # Pi Camera v2 half-FOV
                if _cam_salience > 0.02:
                    # Update target: current IMU yaw + camera heading in degrees
                    _current_yaw = imu_data.get('yaw', 0.0)
                    camera._pd_yaw_target = _current_yaw + _cam_heading * _HALF_FOV_DEG

                # v4.2 compat: LightMemory for when light disappears
                if light_memory:
                    _memory_z = light_memory.update(
                        cam_salience=_cam_salience,
                        cam_heading=_cam_heading,
                        current_yaw=imu_data.get('yaw', 0.0),
                        current_time=time.time(),
                    )
                    # When light lost, LightMemory provides target yaw directly
                    if _cam_salience <= 0.05 and light_memory.state == 'returning':
                        camera._pd_yaw_target = light_memory._target_yaw

                # PID controller: IMU yaw error -> steering
                _current_yaw = imu_data.get('yaw', 0.0)
                _yaw_error = camera._pd_yaw_target - _current_yaw
                # Normalize to [-180, 180]
                while _yaw_error > 180: _yaw_error -= 360
                while _yaw_error < -180: _yaw_error += 360

                # I-term: accumulate error (eliminates steady-state drift offset)
                camera._pd_integral += _yaw_error * CONTROL_DT
                camera._pd_integral = max(-camera._pd_integral_max,
                                          min(camera._pd_integral_max, camera._pd_integral))

                _error_rate = (_yaw_error - camera._pd_prev_error) / CONTROL_DT
                camera._pd_prev_error = _yaw_error

                _steering = (camera._pd_kp * _yaw_error +
                             camera._pd_ki * camera._pd_integral +
                             camera._pd_kd * _error_rate)
                # NO negation on hardware — MPU6050 yaw and CPG match (validated Test C)
                _steering = max(-camera._pd_max, min(camera._pd_max, _steering))

                # Low-pass filter to smooth output (reduce IMU noise amplification)
                _alpha = 0.3  # smoothing factor (0=no change, 1=instant)
                camera._pd_steering = camera._pd_steering * (1.0 - _alpha) + _steering * _alpha

                # Apply to CPG (overrides obstacle reflex steering if phototaxis active)
                if not reflex_active:
                    cpg.set_steering(camera._pd_steering)

                # Compat: store for CSV logging
                camera._last_z_offset = camera._pd_steering

                if step % 50 == 0 and args.verbose:
                    _mem_label = light_memory.get_state_label() if light_memory else '-'
                    print('  CAM h:%+.2f s:%.2f  tgt:%+.1f  yaw:%+.1f  err:%+.1f  steer:%+.3f  mem:%s' % (
                        _cam_heading, _cam_salience, camera._pd_yaw_target,
                        _current_yaw, _yaw_error, camera._pd_steering, _mem_label))

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

            # Cerebellar VOR calibration update — runs EVERY step.
            # v4.4: Drift learning loop REMOVED — the IMU PD controller
            # handles drift compensation automatically. No need to learn
            # a separate drift bias when the closed loop already corrects it.

            # Non-SNN verbose output
            if args.verbose and step%50==0 and step>0 and not (args.snn and gate):
                ms=(time.time()-t_step)*1000
                print(f'  [{step:5d}] {ms:.1f}ms  CPG only  '
                      f'P:{imu_data["pitch"]:+.1f}  R:{imu_data["roll"]:+.1f}  '
                      f'OD:{obstacle_dist:.2f}m')

            # CSV logging — every 50 steps, flush immediately
            if csv_file and step % 50 == 0:
                t_elapsed = time.time() - t_start
                _csv_cpg = gate.get_stats()['cpg_weight'] if gate else 1.0
                _csv_comp = gate.get_stats()['actor_competence'] if gate else 0.0
                _csv_corr = cb.stats.get('correction_magnitude', 0.0) if cb else 0.0
                _csv_da = da_signal if args.snn else 0.0
                _csv_cam_h = camera.heading if camera else 0.0
                _csv_cam_s = camera.salience if camera else 0.0
                _csv_steer_z = getattr(camera, '_pd_steering', 0.0) if camera else 0.0
                _csv_drift = 0.0  # v4.4: drift_bias removed, PD handles it
                _csv_mem_state = light_memory.get_state_label() if light_memory else '-'
                _csv_mem_z = _memory_z if (camera and light_memory) else 0.0
                # Phototaxis navigation fields (FLOG_FORMAT.md v1.2 schema):
                # Hardware has no ground truth — sentinels for pos_x/y, dist_to_light,
                # heading_to_light, brain_pos_error. brain_pos_x/y comes from SpatialMap.
                _csv_pos_x = -999.0  # GT not available on hardware
                _csv_pos_y = -999.0
                _csv_dist_light = -1.0
                _csv_heading_light = -999.0
                _csv_intent_yaw = _csv_steer_z  # motor steering Z is the intent signal
                if spatial_map is not None:
                    _csv_brain_x = float(spatial_map.position[0])
                    _csv_brain_y = float(spatial_map.position[1])
                else:
                    _csv_brain_x = -999.0
                    _csv_brain_y = -999.0
                _csv_brain_err = -1.0  # No GT → cannot compute error
                csv_file.write(f'{step},{t_elapsed:.3f},{obstacle_dist:.4f},'
                               f'{_csv_cpg:.4f},{_csv_comp:.4f},{_csv_corr:.6f},'
                               f'{_csv_da:.4f},{imu_data["pitch"]:.2f},'
                               f'{imu_data["roll"]:.2f},{imu_data["upright"]:.4f},'
                               f'{imu_data["yaw"]:.2f},'
                               f'{_csv_cam_h:.4f},{_csv_cam_s:.4f},{_csv_steer_z:.4f},{_csv_drift:.4f},'
                               f'{_csv_mem_state},{_csv_mem_z:.4f},'
                               f'{_csv_pos_x:.4f},{_csv_pos_y:.4f},'
                               f'{_csv_dist_light:.4f},{_csv_heading_light:.4f},'
                               f'{_csv_intent_yaw:.4f},'
                               f'{_csv_brain_x:.4f},{_csv_brain_y:.4f},{_csv_brain_err:.4f}\n')
                csv_file.flush()

            # Brain-map JSONL sidecar — same schema as FLOG brain_landmarks_json /
            # brain_visit_grid_b64. Written every 1000 steps (~20s of walk).
            # Each line is one snapshot, parseable independently.
            if brain_jsonl is not None and spatial_map is not None and step % 1000 == 0 and step > 0:
                try:
                    import base64
                    import numpy as _np
                    _lm_payload = []
                    for _lm_name, _lm in spatial_map.landmarks.items():
                        if _lm.confidence < 0.05:
                            continue
                        _lm_payload.append({
                            'name': _lm_name,
                            'x': float(_lm.position[0]),
                            'y': float(_lm.position[1]),
                            'cat': _lm.category,
                            'conf': round(float(_lm.confidence), 3),
                            'val': round(float(_lm.valence), 3),
                            'visits': int(_lm.visit_count),
                            'last_seen': int(_lm.last_seen_step),
                        })
                    _vg_u8 = _np.clip(spatial_map.visit_grid, 0, 255).astype(_np.uint8)
                    _snapshot = {
                        'step': step,
                        'time': round(time.time() - t_start, 3),
                        'brain_pos_x': float(spatial_map.position[0]),
                        'brain_pos_y': float(spatial_map.position[1]),
                        'brain_landmarks': _lm_payload,
                        'brain_visit_grid_b64': base64.b64encode(_vg_u8.tobytes()).decode('ascii'),
                        'brain_grid_shape': list(_vg_u8.shape),
                    }
                    brain_jsonl.write(json.dumps(_snapshot, separators=(',', ':')) + '\n')
                    brain_jsonl.flush()
                except Exception as _bj_e:
                    if not hasattr(main, '_brain_jsonl_warn'):
                        print(f'  Brain map sidecar write failed at step {step}: {_bj_e}')
                        main._brain_jsonl_warn = True


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
                    # Real neuron spikes (560 bools — aligned with simulator)
                    dash_state["neurons"] = snn.spikes.bool().tolist()
                # Camera data for dashboard
                if camera:
                    dash_state["camera"] = {
                        "heading": camera.heading,
                        "salience": camera.salience,
                        "steering_z": getattr(camera, '_last_z_offset', 0.0),
                    }
                # Light memory data for dashboard
                if light_memory:
                    dash_state["light_memory"] = light_memory.get_stats()
                # Brain map for dashboard (current SpatialMap belief)
                if spatial_map is not None:
                    dash_state["brain_map"] = {
                        "pos_x": float(spatial_map.position[0]),
                        "pos_y": float(spatial_map.position[1]),
                        "heading": float(spatial_map.heading),
                        "n_landmarks": len(spatial_map.landmarks),
                        "explored_ratio": float(spatial_map.get_explored_ratio()),
                    }
                # IMU data for dashboard
                dash_state["imu"] = {
                    "pitch": imu_data.get('pitch', 0.0),
                    "roll": imu_data.get('roll', 0.0),
                    "yaw": imu_data.get('yaw', 0.0),
                    "upright": imu_data.get('upright', 1.0),
                }
                # Cerebellum data
                if cb:
                    dash_state["cerebellum"] = {
                        "correction": cb.stats.get('correction_magnitude', 0.0),
                        "cf_error": cb.stats.get('climbing_fiber', 0.0),
                        "pf_weight": cb.stats.get('pf_pkc_weight', 0.0),
                        "rebound": cb.stats.get('rebound', 0.0),
                    }
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
    if camera:
        camera.close()
        # v4.4: Drift calibration file no longer used — PD controller handles drift.
        pass
    if csv_file:
        csv_file.close()
        print(f'  CSV log saved: {args.log_csv} ({step // 10} data points)')
    if brain_jsonl is not None:
        brain_jsonl.close()
        print(f'  Brain map sidecar saved: {brain_jsonl_path}')
    # Persist SpatialMap state for next run — the dog remembers where it has been.
    if spatial_map is not None:
        try:
            import torch as _torch
            _spatial_state_path = os.path.join(HOME_DIR, 'spatial_map.pt')
            _torch.save(spatial_map.state_dict(), _spatial_state_path)
            print(f'  SpatialMap saved: {_spatial_state_path} '
                  f'({len(spatial_map.landmarks)} landmarks, '
                  f'pos=({spatial_map.position[0]:.2f}, {spatial_map.position[1]:.2f}))')
        except Exception as _sm_e:
            print(f'  SpatialMap save failed: {_sm_e}')
    if args.snn and gate:
        gs=gate.get_stats()
        print(f'  Final: competence={gs["actor_competence"]:.3f}  CPG={gs["cpg_weight"]:.0%}')
    print(f'  Done.')

if __name__=='__main__':
    main()
