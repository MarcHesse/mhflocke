# Freenove Robot Dog — Complete Setup & Deployment Guide
# MH-FLOCKE Bridge v4.0 — Unified Codebase
# =========================================
#
# This guide covers everything from unboxing the Freenove FNK0050 kit
# to running MH-FLOCKE's spiking neural network on the robot.
#
# Kit: Freenove Robot Dog Kit for Raspberry Pi (FNK0050), ~100€
# URL: https://www.freenove.com/fnk0050
# GitHub: https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi

# ============================================================
# 1. HARDWARE ASSEMBLY
# ============================================================
#
# Follow the Freenove tutorial PDF (included in the kit).
# Key points:
#   - 12 SG90 servos (3 per leg: hip_yaw, hip_pitch, knee)
#   - PCA9685 servo driver board (I2C address 0x40)
#   - MPU6050 IMU (I2C address 0x68, on the main board)
#   - Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
#   - 18650 batteries (2x, not included) or USB-C power
#   - Ultrasonic sensor HC-SR04 (head, optional for MH-FLOCKE)
#
# Servo Channel Mapping (PCA9685):
#   FL (Front Left):  hip_yaw=ch4,  hip_pitch=ch3,  knee=ch2
#   FR (Front Right): hip_yaw=ch11, hip_pitch=ch12, knee=ch13
#   RL (Rear Left):   hip_yaw=ch7,  hip_pitch=ch6,  knee=ch5
#   RR (Rear Right):  hip_yaw=ch8,  hip_pitch=ch9,  knee=ch10
#
# IMPORTANT: Servo calibration is critical. After assembly, each servo
# should be at 90° when the leg is in its neutral position. Use the
# Freenove calibration tool first, then fine-tune with freenove_calibrate.py.

# ============================================================
# 2. RASPBERRY PI OS SETUP
# ============================================================
#
# Recommended: Raspberry Pi OS Lite (64-bit, Bookworm)
# Flash with Raspberry Pi Imager (https://www.raspberrypi.com/software/)
#
# During imaging, configure:
#   - Hostname: robot
#   - Username: admin
#   - Password: (your choice)
#   - WLAN: your network SSID + password
#   - Locale: Europe/Berlin (or your timezone)
#   - SSH: Enable
#
# After first boot:
ssh admin@robot
sudo raspi-config
#   → Interface Options → I2C → Enable
#   → Interface Options → SSH → Enable (should already be)
#   → Finish → Reboot

# Verify I2C devices:
sudo i2cdetect -y 1
# Expected output:
#   0x40 = PCA9685 (servo driver)
#   0x68 = MPU6050 (IMU)
# If 0x40 missing: check servo board power and I2C cable
# If 0x68 missing: IMU is optional, Bridge works without it

# ============================================================
# 3. FREENOVE SOFTWARE
# ============================================================
#
# Install the Freenove server code (required for Servo.py, IMU.py):
cd ~
git clone https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi.git
ln -s ~/Freenove_Robot_Dog_Kit_for_Raspberry_Pi/Code/Server ~/freenove_server
#
# Test basic servo control:
cd ~/freenove_server
python3 Servo.py
# (servos should twitch briefly)
#
# Test IMU:
python3 -c "from IMU import IMU; i = IMU(); print(i.imuUpdate())"
# Should print (pitch, roll, yaw) values

# ============================================================
# 4. SERVO CALIBRATION (Freenove tool)
# ============================================================
#
# The Freenove kit includes a calibration procedure.
# Run their calibration first:
cd ~/freenove_server
python3 Calibration.py
#
# This creates calibration offsets stored in the Freenove config.
# The MH-FLOCKE bridge reads these automatically through the
# Freenove Servo class.
#
# After Freenove calibration, verify with MH-FLOCKE:
python3 ~/freenove_calibrate.py          # Standing position (IK x=10, y=99, z=10)
python3 ~/freenove_calibrate.py --zero   # All servos to 90° (neutral)
python3 ~/freenove_calibrate.py --sweep  # Sweep each joint ±15° (check range)

# ============================================================
# 5. PYTORCH INSTALLATION (one-time, ~5 min)
# ============================================================
#
# IMPORTANT: Use --index-url for CPU-only version (~150MB).
# Without it, pip downloads CUDA version (~2GB) → fails on Pi.
#
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip3 install numpy --break-system-packages
#
# Verify:
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
#
# If "No space left on device":
pip3 cache purge
sudo apt autoremove -y && sudo apt clean
# Then retry

# ============================================================
# 6. MH-FLOCKE DEPLOYMENT
# ============================================================
#
# From your Windows/Mac development machine (in mhflocke-work directory):
#
# Deploy src/brain/ (6 files — the unified codebase):
ssh admin@robot "mkdir -p ~/src/brain"
scp src/brain/__init__.py admin@robot:~/src/brain/
scp src/brain/snn_controller.py admin@robot:~/src/brain/
scp src/brain/cerebellar_learning.py admin@robot:~/src/brain/
scp src/brain/multi_compartment.py admin@robot:~/src/brain/
scp src/brain/brain_persistence.py admin@robot:~/src/brain/
scp src/brain/topology.py admin@robot:~/src/brain/
#
# Deploy Bridge + Calibration:
scp scripts/freenove_bridge.py admin@robot:~/freenove_bridge.py
scp scripts/freenove_calibrate.py admin@robot:~/freenove_calibrate.py
#
# Deploy creature config (optional, for reference):
ssh admin@robot "mkdir -p ~/creatures/freenove"
scp creatures/freenove/profile.json admin@robot:~/creatures/freenove/
scp creatures/freenove/servo_config.json admin@robot:~/creatures/freenove/
#
# Verify imports:
ssh admin@robot
python3 -c "
from src.brain.snn_controller import SNNController, SNNConfig
from src.brain.cerebellar_learning import CerebellarLearning, CerebellarConfig
from src.brain.topology import compute_cerebellar_populations
pops = compute_cerebellar_populations(172, 12)
print(f'All imports OK — topology: {pops}')
"

# ============================================================
# 7. RUNNING THE BRIDGE
# ============================================================
#
# Standing test (no walking, no SNN):
python3 freenove_bridge.py --gait stand
#
# CPG-only walk (no SNN, just the Central Pattern Generator):
python3 freenove_bridge.py --gait walk --duration 30
#
# Full SNN + Cerebellum, fresh brain:
python3 freenove_bridge.py --gait walk --snn --fresh --verbose --duration 120
#
# Load existing brain (resumes learning):
python3 freenove_bridge.py --gait walk --snn --verbose --duration 120
#
# With web dashboard (http://robot:8080):
# First kill any previous dashboard process on that port:
sudo fuser -k 8080/tcp
python3 freenove_bridge.py --gait walk --snn --dashboard --verbose --duration 300
#
# Then open in browser: http://robot:8080
#
# The dashboard shows REAL data from the running SNN:
#   - 232 neurons in 6 cerebellar populations (MF, GrC, GoC, PkC, DCN, OUT)
#   - Live spike activity (neurons light up when they fire)
#   - Servo angles for all 12 joints
#   - SNN firing rate, DA level, mean synaptic weight
#   - Competence Gate (CPG vs Actor balance)
#   - Performance metrics (step count, ms/step, uptime)
#
# NOTE: The dashboard polls at 200ms intervals. At 34ms/step the Pi
# serves data between SNN steps. If the dashboard shows "Disconnected",
# it usually reconnects within a few seconds. The robot continues
# walking regardless of dashboard connection status.
#
# Key parameters:
#   --gait stand|walk     Walking mode
#   --snn                 Enable SNN + Cerebellum (232 neurons)
#   --fresh               Start with new brain (ignore saved brain.pt)
#   --verbose             Print stats every 50 steps
#   --duration N          Run for N seconds
#   --speed F             CPG speed multiplier (default 1.0)
#   --dry-run             No servo output (for testing without hardware)
#   --dashboard           Start web dashboard on port 8080

# ============================================================
# 8. SIM-TO-REAL BRAIN TRANSFER
# ============================================================
#
# Train in simulator (on Windows/Mac with MuJoCo):
# cd D:\claude\mhflocke-work
# python scripts/train_v032.py --creature-name freenove ^
#   --scene "walk on flat meadow" --steps 50000 ^
#   --no-terrain --no-sensory --no-vision --hardware-sensors ^
#   --auto-reset 500 --no-llm --n-hidden 172
#
# Copy trained brain to Pi:
scp creatures/freenove/brain/brain.pt admin@robot:~/brain.pt
#
# Run with transferred brain:
ssh admin@robot
python3 freenove_bridge.py --gait walk --snn --verbose --duration 120
#
# The brain is in PyTorch format, identical between simulator and Pi.
# The SNN continues learning on the Pi (R-STDP + Cerebellar adaptation).

# ============================================================
# 9. FILE STRUCTURE ON PI
# ============================================================
#
# ~/
# ├── freenove_bridge.py          # Bridge v4.0 (main control script)
# ├── freenove_calibrate.py       # Servo calibration tool
# ├── brain.pt                    # Saved brain (auto-created on first run)
# ├── src/
# │   └── brain/
# │       ├── __init__.py
# │       ├── snn_controller.py   # SNNController (LIF-LTC, R-STDP)
# │       ├── cerebellar_learning.py  # CerebellarLearning (Marr-Albus-Ito)
# │       ├── multi_compartment.py    # PurkinjeCompartmentLayer
# │       ├── brain_persistence.py    # Save/Load brain state
# │       └── topology.py            # compute_cerebellar_populations()
# ├── freenove_server/            # Freenove kit code (symlink)
# │   ├── Servo.py                # PCA9685 servo driver
# │   ├── IMU.py                  # MPU6050 IMU reader
# │   ├── Calibration.py          # Servo calibration
# │   └── ...
# ├── creatures/freenove/         # Config files (optional)
# │   ├── profile.json            # Robot profile (dimensions, SNN topology)
# │   └── servo_config.json       # Servo channel mapping
# └── Freenove_Robot_Dog_Kit_.../  # Full Freenove kit repo
#     └── Code/Server/            # ← freenove_server symlink target

# ============================================================
# 10. WLAN & REMOTE ACCESS
# ============================================================
#
# The Pi connects to your WLAN (configured during imaging).
# Find the Pi's IP address:
#   - From your router's admin page, or:
#   - ping robot.local (macOS/Linux), or:
#   - Use Fing app (mobile) to scan the network
#
# SSH access:
ssh admin@robot
#
# If hostname 'robot' doesn't resolve:
ssh admin@<IP-address>
#
# To change WLAN after setup:
sudo nmtui                   # Network Manager TUI (Bookworm)
# or edit:
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf   # (older OS)
#
# For headless WLAN config (before first boot):
# Create wpa_supplicant.conf on the boot partition of the SD card.
#
# Static IP (optional, recommended for reliable SSH):
sudo nmtui → Edit connection → IPv4 → Manual
# Set: Address 192.168.x.y/24, Gateway 192.168.x.1, DNS 192.168.x.1

# ============================================================
# 11. BATTERY & POWER
# ============================================================
#
# The Freenove kit uses 2x 18650 Li-ion batteries (NOT included).
# Recommended: Samsung 25R or Sony VTC6 (high drain, 2500mAh+)
#
# IMPORTANT: The Pi 4 draws 1-3A under load. With 12 servos active,
# total current can reach 4-5A. Use batteries rated for >5A discharge.
#
# Battery life: ~30-60 minutes of walking (depends on servos + SNN load)
#
# Alternative: USB-C power supply (5V/3A) for bench testing.
# Connect USB-C to Pi AND battery board for reliable servo power.
#
# Low battery warning: servos will jitter or fail to hold position.
# The Bridge will continue running but the robot won't walk properly.

# ============================================================
# 12. TROUBLESHOOTING
# ============================================================
#
# "No module named 'torch'":
#   pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
#
# "No space left on device" during pip install:
#   pip3 cache purge && sudo apt clean && sudo apt autoremove -y
#   Then retry with --index-url for CPU-only PyTorch
#
# "No module named 'src.brain'":
#   Bridge looks for src/ in: script parent dir, $HOME, $HOME/mhflocke
#   Verify: ls ~/src/brain/ (should show 6 .py files)
#
# "PCA9685 not found" / servos don't move:
#   sudo i2cdetect -y 1    # Should show 0x40
#   Check: I2C enabled? Servo board powered? Cable connected?
#
# "IMU not available":
#   sudo i2cdetect -y 1    # Should show 0x68
#   IMU is optional — Bridge uses defaults (upright=1.0) without it
#
# "Brain topology mismatch":
#   Saved brain.pt was trained with different neuron count.
#   Fix: python3 freenove_bridge.py --gait walk --snn --fresh --verbose
#
# Robot tips over immediately:
#   1. python3 freenove_calibrate.py    # Check servo alignment
#   2. python3 freenove_bridge.py --gait stand   # Test standing
#   3. Check battery level (low = servos can't hold position)
#
# Slow step time (>30ms):
#   Normal for PyTorch on Pi 4 with 232 neurons (~34ms/step = 29Hz).
#   The robot walks fine at 29Hz. If >50ms, check CPU temperature:
#   vcgencmd measure_temp    # Should be <70°C
#
# Servos jitter at startup:
#   Normal — the PCA9685 initializes channels sequentially.
#   Wait 1 second after startup before walking.
#
# SSH connection drops during walk:
#   WLAN interference from servo PWM. Use 5GHz band if available.
#   Or run in tmux/screen: tmux new -s walk
#   Then: python3 freenove_bridge.py --gait walk --snn --verbose --duration 600
#   Detach: Ctrl+B, D    Reattach: tmux attach -t walk
#
# "Address already in use" (port 8080):
#   A previous dashboard process is still running.
#   Fix: sudo fuser -k 8080/tcp
#   Then restart the Bridge with --dashboard.
#
# Dashboard shows "Disconnected":
#   Normal — the Pi serves HTTP between SNN steps.
#   The dashboard reconnects automatically. If it stays
#   disconnected, check that the Bridge is still running
#   in the terminal (should show step counter advancing).
#
# Brain not saving on Ctrl+C:
#   The Bridge catches SIGINT and saves brain.pt before exit.
#   If it doesn't work, check disk space: df -h /

# ============================================================
# 13. SPECS & REFERENCE
# ============================================================
#
# Robot Dimensions:
#   Body: 136mm × 76mm × 30mm (L×W×H without legs)
#   Standing height: ~99mm (from foot to hip joint)
#   Total mass: ~620g (with Pi + batteries)
#
# Servo Kinematics:
#   Hip offset (l1): 23mm
#   Upper leg (l2): 55mm
#   Lower leg (l3): 55mm
#   Servo range: 18°–162° (SG90, limited by mechanical stops)
#
# SNN Topology (Freenove profile):
#   Input: 48 neurons (12 servo + 2 CPG + 4 IMU + 30 padding)
#   Output: 12 neurons (1 per actuator)
#   Granule Cells: 106
#   Golgi Cells: 18
#   Purkinje Cells: 24
#   DCN: 24
#   Total: 232 neurons, ~933 synapses
#
# Control Loop:
#   Target: 50Hz (20ms/step)
#   Actual: ~29Hz (34ms/step) with PyTorch on Pi 4
#   CPG frequency: 0.8Hz (adjustable via --speed)
#
# I2C Addresses:
#   0x40 = PCA9685 servo driver (12-bit PWM, 50Hz)
#   0x68 = MPU6050 IMU (gyro + accelerometer, 6-axis)

# ============================================================
# 14. COMPLETE DEPENDENCY LIST
# ============================================================
#
# --- On Raspberry Pi (runtime) ---
#
# System packages (pre-installed on Raspberry Pi OS):
#   python3 (3.11+)
#   python3-pip
#   python3-smbus / python3-smbus2
#   i2c-tools
#   git
#
# Python packages (install with pip3):
#   torch>=2.0.0          # PyTorch CPU-only (~150MB)
#   numpy>=1.24.0         # NumPy (usually pre-installed)
#
# Install command:
#   pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
#   pip3 install numpy --break-system-packages
#
# Freenove dependencies (from their repo, auto-installed):
#   RPi.GPIO              # GPIO access (pre-installed on Pi OS)
#   smbus2                # I2C communication (PCA9685, MPU6050)
#   rpi_ws281x            # LED strip (optional, not used by MH-FLOCKE)
#
# --- On development machine (training + rendering) ---
#
# Python packages:
#   torch>=2.0.0          # PyTorch (with CUDA for GPU training)
#   numpy>=1.24.0
#   mujoco>=3.0.0         # Physics simulator
#   Pillow>=10.0          # Image processing (dashboard overlay)
#   msgpack               # FLOG binary format
#   soundfile             # Audio sonification (optional)
#   pedalboard            # Audio mastering (optional)
#
# Install command:
#   pip install torch numpy mujoco Pillow msgpack
#   pip install soundfile pedalboard   # optional, for video sound
#
# Freenove source code (required on Pi):
#   git clone https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi.git
#   The Bridge imports Servo.py and IMU.py from this repo.
#
# MH-FLOCKE source code:
#   git clone https://github.com/MarcHesse/mhflocke.git
#   Only src/brain/ (6 files) is needed on the Pi.

# ============================================================
# 15. QUICK REFERENCE CARD
# ============================================================
#
# --- Pi commands ---
# python3 freenove_bridge.py --gait walk --snn --fresh --verbose --duration 120
# python3 freenove_bridge.py --gait walk --snn --dashboard --verbose --duration 300
# python3 freenove_calibrate.py
# python3 freenove_calibrate.py --zero
# python3 freenove_calibrate.py --sweep
#
# --- Deploy from dev machine ---
# scp src/brain/{__init__,snn_controller,cerebellar_learning,multi_compartment,brain_persistence,topology}.py admin@robot:~/src/brain/
# scp scripts/freenove_bridge.py admin@robot:~/freenove_bridge.py
# scp scripts/freenove_calibrate.py admin@robot:~/freenove_calibrate.py
# scp creatures/freenove/brain/brain.pt admin@robot:~/brain.pt
#
# --- Training (dev machine) ---
# python scripts/train_v032.py --creature-name freenove --scene "walk on flat meadow" --steps 50000 --no-terrain --no-sensory --no-vision --hardware-sensors --auto-reset 500 --no-llm --n-hidden 172
#
# --- Rendering (dev machine) ---
# python scripts/render_freenove.py creatures/freenove/v034_XXXXX/training_log.bin
# python scripts/sonify_flog.py --flog creatures/freenove/v034_XXXXX/training_log.bin --speed 2 --mux creatures/freenove/v034_XXXXX/freenove_render.mp4

# ============================================================
# 16. SUPPORT & CONTACT
# ============================================================
#
# For questions about installation, hardware setup, or MH-FLOCKE:
#
#   Email: marc@mhflocke.com
#
# Resources:
#   GitHub:  https://github.com/MarcHesse/mhflocke
#   Website: https://mhflocke.com
#   Paper:   https://doi.org/10.5281/zenodo.19336894
#   YouTube: https://www.youtube.com/@mhflocke
#
# Freenove Kit Support:
#   Freenove provides their own support for hardware assembly
#   and their base software (Servo.py, IMU.py, Calibration.py).
#   URL: https://www.freenove.com/fnk0050
#   GitHub: https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi
