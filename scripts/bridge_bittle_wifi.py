"""Bittle Bridge - WiFi / WebSocket Test Client
==============================================
Connects to the Bittle X over WiFi (WebSocket on port 81), disables OpenCat's
onboard balance so it does not modify our motor commands, enables continuous
IMU print, triggers a small motion, and reads the IMU lines back.

This verifies the real control channel (WiFi), not USB. USB was only used for
first-contact / WiFi setup.

Usage:
    py -3.11 scripts/bridge_bittle_wifi.py --ip <robot-ip>
    py -3.11 scripts/bridge_bittle_wifi.py --ip <robot-ip> --keep-balance

Protocol (verified against ref/PetoiWebCoding/js/petoi_async_client.js and
ref/OpenCatEsp32/src/webServer.h):
    Send:    {"type":"command","taskId":"<id>","commands":["<cmd>",...],
              "timestamp":<ms>}
    Receive: {"type":"connected", ...}                       (on connect)
             {"type":"response","taskId":...,"status":"running"}
             {"type":"response","taskId":...,"status":"completed",
              "results":["<webResponse per command>"]}
    The output of a command (e.g. the 'MCU:' IMU line from print6Axis) is
    collected into webResponse and returned in 'results'.

OpenCat command tokens used here:
    gb  -> gyro balance OFF (motor commands pass through untouched)
    gP  -> print IMU continuously (streams inside transform() during motion)
    gp  -> print IMU once / stop continuous
    kbalance / kup -> small posture move so transform() runs and IMU streams
    d   -> rest (servos relax)

The IMU line is prefixed with "MCU:" (MPU6050) or "ICM:" (ICM42670) and holds
accel x/y/z then yaw/pitch/roll.
"""

__version__ = "3.3"       # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 29          # mh-logbuch module entry
__status__  = "active"     # active | veraltet | neu

import argparse
import json
import math
import os
import re
import sys
import time

# Ensure repo root is on sys.path so 'src' is importable when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websocket  # websocket-client (sync)


def _now_ms() -> int:
    return int(time.time() * 1000)


def parse_imu(text: str):
    """Extract (label, [floats]) from a print6Axis line, ignoring MCU:/ICM:."""
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        label = ""
        low = ln.lower()
        if "mcu" in low:
            label = "MPU6050"
        elif "icm" in low:
            label = "ICM42670"
        toks = ln.replace(",", " ").replace(":", " ").split()
        vals = []
        for t in toks:
            # The firmware glues the command token echo onto the last number,
            # e.g. "0.2g". Strip a trailing non-numeric suffix before parsing.
            m = re.match(r"[-+]?\d*\.?\d+", t)
            if m:
                try:
                    vals.append(float(m.group(0)))
                except ValueError:
                    pass
        if len(vals) >= 3:
            return label, vals
    return None


class BittleWS:
    def __init__(self, ip, port=81, timeout=6.0):
        self.url = f"ws://{ip}:{port}"
        self.timeout = timeout
        self.ws = None

    def connect(self):
        print(f"Connecting to {self.url} ...")
        self.ws = websocket.create_connection(self.url, timeout=self.timeout)
        # Server sends a {"type":"connected"} greeting; read it.
        try:
            greeting = self.ws.recv()
            print(f"  <- {greeting}")
        except Exception:
            pass
        print("  Connected.")

    def send_commands(self, commands, settle=0.0, timeout=None):
        """Send a command group, collect results until 'completed'/'error'.

        Args:
            timeout: override the instance timeout for this call (seconds).
                     Useful for short IMU reads in tight loops.
        """
        task_id = str(_now_ms())
        msg = {
            "type": "command",
            "taskId": task_id,
            "commands": commands,
            "timestamp": _now_ms(),
        }
        self.ws.send(json.dumps(msg))

        results = []
        t = timeout if timeout is not None else self.timeout
        deadline = time.time() + t
        while time.time() < deadline:
            try:
                self.ws.settimeout(max(0.1, deadline - time.time()))
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            except Exception as e:
                print(f"  recv error: {e}")
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            status = data.get("status")
            if status == "completed":
                results = data.get("results", [])
                break
            if status == "error":
                print(f"  task error: {data.get('error')}")
                break
            # status == 'running' or heartbeat -> keep waiting
        if settle:
            time.sleep(settle)
        return results

    def send_joints(self, ctrl, settle: float = 0.2) -> list:
        """Send 8 MJCF joint targets (radians) as an OpenCat 'i' command.

        Converts MJCF actuator radians → OpenCat integer degrees and sends
        them as a single indexed simultaneous move command:
            i <oc_idx> <deg> <oc_idx> <deg> ...

        Args:
            ctrl: array-like of 8 floats, MJCF actuator order
                  (RF_sh, RF_kn, LF_sh, LF_kn, RR_sh, RR_kn, LR_sh, LR_kn).
            settle: seconds to wait after command completes.

        Returns:
            results list from firmware (usually empty for motion commands).
        """
        from src.body.opencat_controller import mjcf_ctrl_to_oc_i_cmd
        cmd = mjcf_ctrl_to_oc_i_cmd(ctrl)
        return self.send_commands([cmd], settle=settle)

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass


_CB_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CB_BRAIN_DEFAULT = os.path.join(_CB_REPO_ROOT, "creatures", "bittle", "brain", "brain.pt")
_CB_SNN_SUBSTEPS = 6    # substeps per cerebellar step
_CB_MF_CURRENT = 1.0   # input current injected into mossy fibers
_CB_SENSOR_HEIGHT = 0.11  # Bittle approximate standing height in meters
_CB_N_GRANULE = 398    # match existing brain.pt for future transfer compatibility
_CB_N_GOLGI = 70

_SNN_SUBSTEPS = 6       # substeps per SNN step
_SNN_N_HIDDEN = 500     # hidden neurons (creatures/bittle/profile.json)
_SNN_MAX_CORR = 0.3     # max correction magnitude in radians
_SNN_RAMP_STEPS = 200   # steps to reach full correction (longer than Cerebellum)

# --- Fall detection + OpenCat recovery (v3.1) ---
# Mechanical self-reset for unattended fresh learning on hardware: if the IMU
# reports the body down for a few reads, drive a built-in OpenCat stand posture
# until upright. Weights are kept (only the body resets) so learning continues.
_FALL_CONSEC_TRIGGER = 3        # consecutive fallen IMU reads before recovery fires
_RECOVER_DEFAULT_SKILL = "kbalance"  # OpenCat posture skill -> legs to standing pose


def _read_imu_once(bot, timeout: float = 0.3):
    """Send one 'gp' and return (yaw, pitch, roll) in degrees, or None."""
    for chunk in bot.send_commands(["gp"], settle=0.0, timeout=timeout):
        parsed = parse_imu(chunk)
        if parsed and len(parsed[1]) >= 6:
            v = parsed[1]
            return (v[3], v[4], v[5])
    return None


def _recover_bittle(bot, skill: str, fall_threshold: float,
                    settle: float = 1.8, max_tries: int = 3) -> bool:
    """Mechanical self-reset: drive the Bittle to a standing posture via an
    OpenCat firmware skill (default 'kbalance') until the IMU reports upright.

    Hardware analogue of the simulator's auto-reset ("mother helps the fallen
    pup"): on a fall, command the built-in OpenCat stand/balance posture, wait
    for it to settle, verify via IMU. SNN/cerebellum weights are NOT touched —
    only the body is reset — so learning continues across falls. Returns True if
    upright after recovery, False if still down.

    NOTE: a static posture skill rights typical side/forward falls; a full
    upside-down flip may need a dynamic self-right skill (future work).
    """
    for _ in range(max_tries):
        bot.send_commands([skill], settle=settle)
        ypr = _read_imu_once(bot)
        if ypr is not None and abs(ypr[1]) < fall_threshold and abs(ypr[2]) < fall_threshold:
            return True
    return False


def _load_cb_fresh(n_actuators: int = 8, cb_state_path: str = None):
    """Create a CerebellarLearning instance on the SHARED SNN topology (#159).

    The SNN is built by build_snn_from_profile() — identical cerebellar
    populations (GrC/GoC/PkC/DCN) to the simulator — and CerebellarLearning is
    configured from the builder's OWN sizes/connectivity (built.meta). This
    replaces the old hand-sized net (398 GrC / 70 GoC) that could never match a
    simulator-trained cerebellum.

    cb_state_path: optional CerebellarLearning.state_dict() checkpoint. The
    learned PF->PkC weights live HERE, not in brain.pt (see logbook #23) —
    brain.pt is the CognitiveBrain bundle and contains no cerebellum. Loaded
    best-effort (try/except). NOTE: as of 2026-06-07 no current Bittle
    cerebellum checkpoint is known to exist; the sim must save cb.state_dict()
    to a sibling file before transfer is possible (follow-up).

    Returns (cb, snn) or (None, None) on error.
    """
    import json
    import torch
    from src.brain.snn_builder import build_snn_from_profile
    from src.brain.cerebellar_learning import CerebellarConfig, CerebellarLearning

    profile_path = os.path.join(_CB_REPO_ROOT, "creatures", "bittle", "profile.json")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Shared builder = identical cerebellar topology to the simulator.
    built = build_snn_from_profile(profile, n_actuators=n_actuators, device="cpu")
    snn = built.snn
    m = built.meta
    pops = built.pops

    # Cerebellar config mirrors the builder's sizes + connectivity exactly, so a
    # simulator cerebellum state_dict (same profile) loads without shape clash.
    cb_config = CerebellarConfig(
        n_granule=m["n_granule"],
        n_golgi=m["n_golgi"],
        n_purkinje=m["n_purkinje"],
        n_dcn=m["n_dcn"],
        mf_per_granule=m["mf_per_granule"],
        grc_goc_prob=m["grc_goc_prob"],
        pf_pkc_prob=m["pf_pkc_prob"],
        snn_mix_end=0.35,
        snn_ramp_steps=100,  # 100 steps × 5 Hz = 20s ramp (sim default 3000 × 50Hz)
    )
    cb = CerebellarLearning(snn, n_actuators=n_actuators, config=cb_config, device="cpu")
    cb.set_populations(pops["mossy_fibers"], pops["granule_cells"],
                       pops["golgi_cells"], pops["purkinje_cells"], pops["dcn"])

    if cb_state_path and os.path.exists(cb_state_path):
        try:
            cb.load_state_dict(torch.load(cb_state_path, map_location="cpu"))
            print(f"  Cerebellum: loaded state from {cb_state_path}")
        except Exception as e:
            print(f"  Cerebellum: state load failed ({e}); fresh weights")
    elif cb_state_path:
        print(f"  Cerebellum: state path not found ({cb_state_path}); fresh weights")

    print(f"  Cerebellum: shared builder topology "
          f"(GrC={m['n_granule']} GoC={m['n_golgi']} PkC={m['n_purkinje']} "
          f"DCN={m['n_dcn']}, matches simulator), ramp={cb_config.snn_ramp_steps} steps")
    return cb, snn


def _cb_step(cb, snn, cb_state: dict, joints_rad,
             cpg_phase: float, imu_ypr: tuple):
    """One cerebellar step: run SNN substeps, update weights, return corrections[8].

    cf signal comes from vestibular error (yaw_rate, pitch, roll) derived from WiFi IMU.
    Returns np.ndarray[8] in MJCF radian space.
    """
    import numpy as np
    import torch

    yaw_deg, pitch_deg, roll_deg = imu_ypr
    step = cb_state.get("step", 0)
    dt = cb_state.get("dt", 0.2)

    prev_yaw = cb_state.get("prev_yaw", yaw_deg)
    yaw_rate_deg = (yaw_deg - prev_yaw) / max(dt, 0.01)
    # Wrap yaw_rate to [-180, 180] (handles the ±180° boundary)
    if yaw_rate_deg > 180:
        yaw_rate_deg -= 360
    elif yaw_rate_deg < -180:
        yaw_rate_deg += 360
    cb_state["prev_yaw"] = yaw_deg

    sensor_data = {
        "step": step,
        "orientation_euler": np.array([math.radians(roll_deg),
                                        math.radians(pitch_deg),
                                        math.radians(yaw_deg)]),
        "angular_velocity": np.array([0.0, 0.0, math.radians(yaw_rate_deg)]),
        "height": _CB_SENSOR_HEIGHT,
        "standing_height": _CB_SENSOR_HEIGHT,
        "motor_commands": np.array(joints_rad[:8], dtype=np.float32),
        "joint_positions": np.array(joints_rad[:8], dtype=np.float32),
        "velocity": np.zeros(3),
        "forward_velocity": 0.0,
        "desired_velocity": 0.2,
    }

    n = snn.config.n_neurons
    snn_input = torch.zeros(n)
    mf = [0.0] * 19
    for i, j in enumerate(joints_rad[:8]):
        mf[i] = max(0.0, min(1.0, float(j) / math.pi + 0.5))
    mf[8]  = math.sin(2 * math.pi * cpg_phase)
    mf[9]  = math.cos(2 * math.pi * cpg_phase)
    mf[10] = max(-1.0, min(1.0, pitch_deg / 90.0))
    mf[11] = max(-1.0, min(1.0, roll_deg / 90.0))
    mf[12] = max(-1.0, min(1.0, yaw_deg / 180.0))
    mf[13] = 1.0 - min(1.0, (abs(pitch_deg) + abs(roll_deg)) / 90.0)
    snn_input[:19] = torch.tensor(mf, dtype=torch.float32) * _CB_MF_CURRENT

    for _ in range(_CB_SNN_SUBSTEPS):
        snn.step(snn_input)

    # Update PF->PkC weights via LTD/LTP; compute DCN output.
    cb.update(None, sensor_data)

    upright = mf[13]
    corrections = cb.compute_corrections([], upright=upright)

    cb_state["step"] = step + 1
    return corrections


def _load_snn_fresh(n_actuators: int = 8, brain_path: str = None):
    """Create an SNNController + CognitiveBrain for Bittle (Issue #159).

    Topology is now built by the SHARED builder build_snn_from_profile(), the
    same code the simulator uses, from creatures/bittle/profile.json. This is
    the fix for #159: previously this function hand-built a FLAT input/hidden/
    output net (535 neurons but no cerebellar GrC/GoC/PkC/DCN/MH structure), so
    a brain.pt trained in simulation could never be loaded onto hardware —
    population ID ranges and connectivity did not match. Now sim and hardware
    produce byte-identical topology for a given seed, so transfer is possible.

    Args:
        n_actuators: actuated joints (Bittle: 8).
        brain_path:  optional path to a simulation-trained brain.pt. If given
                     and present, its weights are loaded into the (now matching)
                     topology. NOTE: this restores the SNNController weights
                     only; cerebellar PF->PkC weights live in
                     CerebellarLearning.state_dict() (see logbook #23) and are
                     NOT in brain.pt — full transfer needs both.

    Starts from random weights (or trained brain.pt) and learns locomotion via
    R-STDP with intrinsic rewards (vestibular discomfort, curiosity,
    empowerment) from WiFi IMU.
    Returns (brain, snn, out_ids) or (None, None, None) on error.
    """
    import json
    from src.brain.snn_builder import build_snn_from_profile
    from src.brain.cognitive_brain import CognitiveBrainConfig, CognitiveBrain

    profile_path = os.path.join(_CB_REPO_ROOT, "creatures", "bittle", "profile.json")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    n_input = profile["snn"]["n_input"]  # 19

    # Shared builder = identical topology to the simulator (cerebellar + MH).
    built = build_snn_from_profile(profile, n_actuators=n_actuators, device="cpu")
    snn = built.snn
    out_ids = built.pops["output"]

    brain = CognitiveBrain(snn, n_sensor_channels=n_input,
                           n_motors=n_actuators, config=CognitiveBrainConfig())

    # Optional: load a simulation-trained brain.pt. brain.pt is the
    # CognitiveBrain bundle (SNN + cognitive state) written by
    # brain_persistence.save_brain() — NOT the SNNController's own .save()
    # format — so it must be loaded via load_brain(). The cerebellum is NOT in
    # brain.pt (see logbook #23); its checkpoint is loaded separately in
    # _load_cb_fresh(). A synapse-count precheck avoids the #150 topology crash
    # on a stale checkpoint.
    if brain_path and os.path.exists(brain_path):
        from src.brain.brain_persistence import load_brain, brain_info
        try:
            info = brain_info(brain_path)
            saved_syn = info.get("n_synapses")
            if saved_syn is not None and saved_syn != snn._n_synapses:
                print(f"  SNN: brain.pt synapses {saved_syn} != built {snn._n_synapses} "
                      f"(topology mismatch, see #150) — skipping load, fresh weights")
            else:
                load_brain(brain, snn, brain_path)
                print(f"  SNN: loaded trained brain from {brain_path}")
        except Exception as e:
            print(f"  SNN: brain.pt load failed ({e}); continuing with fresh weights")
    elif brain_path:
        print(f"  SNN: brain_path not found ({brain_path}); fresh weights")

    print(f"  SNN: shared builder topology, {snn.config.n_neurons} neurons "
          f"(matches simulator; input={n_input} output={len(out_ids)}), "
          f"ramp={_SNN_RAMP_STEPS} steps")
    return brain, snn, out_ids


def _snn_step(brain, snn, out_ids, snn_state: dict,
              joints_rad, cpg_phase: float, imu_ypr: tuple):
    """One SNN step: encode sensors, run substeps, decode corrections, update brain.

    R-STDP driven by intrinsic reward: vestibular_discomfort, curiosity,
    empowerment, proprioceptive_delta. No external reward.
    Returns (corrections[8], intrinsic_reward).
    """
    import numpy as np
    import torch
    from src.body.mujoco_creature import decode_motor_spikes

    yaw_deg, pitch_deg, roll_deg = imu_ypr
    step = snn_state.get("step", 0)
    dt   = snn_state.get("dt", 0.2)

    # 1. Encode 19 sensor channels (same layout as Cerebellum mossy fibers)
    n = snn.config.n_neurons
    snn_input = torch.zeros(n)
    for i, j in enumerate(joints_rad[:8]):
        snn_input[i] = max(0.0, min(1.0, float(j) / math.pi + 0.5))
    snn_input[8]  = math.sin(2 * math.pi * cpg_phase)
    snn_input[9]  = math.cos(2 * math.pi * cpg_phase)
    snn_input[10] = max(-1.0, min(1.0, pitch_deg / 90.0))
    snn_input[11] = max(-1.0, min(1.0, roll_deg  / 90.0))
    snn_input[12] = max(-1.0, min(1.0, yaw_deg   / 180.0))
    upright = 1.0 - min(1.0, (abs(pitch_deg) + abs(roll_deg)) / 90.0)
    snn_input[13] = upright

    # Tonic background current on hidden neurons (matches mujoco_creature.py)
    tonic = getattr(snn, '_hidden_tonic_current', 0.0)
    if tonic > 0:
        hid_pop = snn.populations.get('hidden')
        if hid_pop is not None:
            snn_input[hid_pop] += tonic

    # 2. Run substeps, accumulate output spikes
    out_spikes = torch.zeros(len(out_ids))
    last_spikes = None
    for _ in range(_SNN_SUBSTEPS):
        last_spikes = snn.step(snn_input)
        out_spikes += last_spikes[out_ids].float()

    # 3. Decode output spikes → motor corrections with ramp blend
    raw = decode_motor_spikes(out_spikes, n_per_joint=2, substeps=_SNN_SUBSTEPS)
    ramp = min(1.0, step / max(1, _SNN_RAMP_STEPS))
    corrections = np.array(raw, dtype=np.float32) * ramp * _SNN_MAX_CORR

    # 4. Vestibular discomfort: penalty for yaw_rate + body tilt
    prev_yaw = snn_state.get("prev_yaw", yaw_deg)
    yaw_rate = (yaw_deg - prev_yaw) / max(dt, 0.01)
    if yaw_rate > 180:   yaw_rate -= 360
    elif yaw_rate < -180: yaw_rate += 360
    snn_state["prev_yaw"] = yaw_deg
    vest = min(1.0, max(0.0, abs(yaw_rate) * 0.02 + (1.0 - upright)))

    controls_list = [float(j) for j in joints_rad[:8]]
    is_fallen = abs(pitch_deg) > 60 or abs(roll_deg) > 60

    # 5. CognitiveBrain 15-step cognitive cycle
    brain.process(
        sensor_values=snn_input[:19].numpy().tolist(),
        snn_input=snn_input[:19],
        output_spikes=(last_spikes[out_ids]
                       if last_spikes is not None
                       else torch.zeros(len(out_ids))),
        controls=controls_list,
        external_reward=0.0,
        is_fallen=is_fallen,
        extra_sensor_data={
            "vestibular_discomfort": vest,
            "smell_strength": 0.0,
            "scent_reward": 0.0,
        },
    )

    # 6. Intrinsic reward → R-STDP weight update
    reward = brain.get_intrinsic_reward()
    snn.apply_rstdp(reward_signal=reward)

    snn_state["step"] = step + 1
    return corrections, float(reward)


def run_gait_loop(bot, gait_name: str, duration: float, keep_balance: bool,
                  batch: int = 1, yaw_pid: bool = False,
                  yaw_invert: bool = False, brain_mode: bool = False,
                  cerebellum: bool = False, cb_state_path: str = None,
                  use_snn: bool = False, snn_brain_path: str = None,
                  recover: bool = False, fall_threshold: float = 50.0,
                  recover_skill: str = _RECOVER_DEFAULT_SKILL,
                  recover_settle: float = 1.8, dashboard: bool = False,
                  telemetry: bool = True):
    """Stream a gait via 'i' tokens, log IMU, optionally close yaw PID loop.

    brain_mode=False (default): OpenCatController.step(dt) — absolute ctrl.
    brain_mode=True: OpenCatGait.compute(dt, steering=s) — delta + STAND_CTRL.
    use_snn=True: fresh SNNController + CognitiveBrain, learns via R-STDP with
      intrinsic rewards (vestibular, curiosity, empowerment). Applied before
      Cerebellum corrections.
    cerebellum=True: fresh CerebellarLearning, applies drift corrections on top.
    """
    import numpy as np
    from src.body.opencat_controller import mjcf_ctrl_to_oc_i_cmd

    if not keep_balance:
        print("  Disabling balance (gb) ...")
        bot.send_commands(["gb"])

    # --- SNN setup (locomotion learner + drives) ---
    snn_brain, snn_ctrl, snn_out_ids = None, None, None
    snn_state = {}
    if use_snn:
        try:
            snn_brain, snn_ctrl, snn_out_ids = _load_snn_fresh(brain_path=snn_brain_path)
            snn_state = {'step': 0, 'prev_yaw': 0.0, 'dt': 0.2}
        except Exception as e:
            print(f"  SNN disabled (load failed): {e}")
            use_snn = False

    # --- Cerebellum setup ---
    cb, cb_snn = None, None
    cb_state = {}
    if cerebellum:
        try:
            cb, cb_snn = _load_cb_fresh(cb_state_path=cb_state_path)
            cb_state = {'step': 0, 'prev_yaw': 0.0, 'dt': 0.2}
        except Exception as e:
            print(f"  Cerebellum disabled (load failed): {e}")
            cerebellum = False

    # --- Live dashboard (reuses dashboard_views WS broadcaster on :5001) ---
    _dash_push = None
    if dashboard:
        try:
            from src.viz.dashboard_views import (start_websocket,
                                                 update_training_state as _dp)
            start_websocket()
            _dash_push = _dp
            print("  Dashboard: live on ws://localhost:5001 "
                  "(open src/viz/bridge_live.html in a browser)")
        except Exception as e:
            print(f"  Dashboard disabled (init failed): {e}")

    if brain_mode:
        from src.brain.opencat_gait import OpenCatGait
        from src.body.opencat_controller import STAND_CTRL
        cpg = OpenCatGait(gait_name)
        cpg._maturation = 1.0
        print(f"  CPG: OpenCatGait (brain interface, delta mode)")
    else:
        from src.body.opencat_controller import OpenCatController
        cpg = OpenCatController()
        cpg.set_gait(gait_name)
        cpg._maturation = 1.0
        print(f"  CPG: OpenCatController (direct mode)")

    # Yaw PID state  (Kp/Ki from CLAUDE.md Bridge values)
    KP = 0.05
    KI = 0.005
    yaw_ref = None       # set on first IMU sample
    yaw_integral = 0.0
    yaw_error = 0.0
    t_last_imu = None
    last_imu_ypr = (0.0, 0.0, 0.0)  # (yaw, pitch, roll) — updated on every IMU read

    step = 0
    imu_count = 0
    fall_consec = 0
    recovery_count = 0
    t_start = time.time()
    t_deadline = t_start + duration
    t_prev = t_start

    mode_tags = []
    if yaw_pid:
        mode_tags.append("yaw-PID")
    if use_snn:
        mode_tags.append("snn")
    if cerebellum:
        mode_tags.append("cerebellum")
    if recover:
        mode_tags.append("recover")
    mode_str = " " + "+".join(mode_tags) if mode_tags else ""
    print(f"\n--- Gait loop: '{gait_name}', {duration}s, batch={batch}{mode_str}"
          f" (Ctrl+C to stop) ---")

    # --- Telemetry log + per-run output dir (scientific record of the run) ---
    # Each hardware run gets its own dir: creatures/bittle/bridge_<ts>/ with
    #   telemetry.jsonl  (per-step IMU/cmd/correction/reward stream)
    #   brain.pt         (learned SNN + cerebellum, saved on exit)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_CB_REPO_ROOT, "creatures", "bittle", f"bridge_{run_id}")
    _telem = None
    if telemetry:
        try:
            os.makedirs(run_dir, exist_ok=True)
            _telem = open(os.path.join(run_dir, "telemetry.jsonl"), "w",
                          encoding="utf-8")
            _telem.write(json.dumps({
                "type": "header", "run_id": run_id, "t_start": t_start,
                "gait": gait_name, "duration": duration, "batch": batch,
                "modes": mode_tags, "fall_threshold": fall_threshold,
                "yaw_pid": bool(yaw_pid), "snn": bool(use_snn),
                "cerebellum": bool(cerebellum), "recover": bool(recover),
                "ip": getattr(bot, "ip", None),
            }) + "\n")
            print(f"  Telemetry: {os.path.join(run_dir, 'telemetry.jsonl')}")
        except Exception as e:
            print(f"  Telemetry disabled (open failed): {e}")
            _telem = None

    try:
        while time.time() < t_deadline:
            t0 = time.time()
            actual_dt = t0 - t_prev
            t_prev = t0

            # Compute steering from yaw PID.
            # Negate: positive yaw error (left drift) → positive steering
            # (right correction). Verified empirically on Bittle 2026-05-31.
            steering = 0.0
            if yaw_pid and yaw_ref is not None:
                raw = KP * yaw_error + KI * yaw_integral
                steering = max(-1.0, min(1.0, raw if yaw_invert else -raw))

            # Build batch of N gait frames.
            cmds = []
            cb_max_corr = 0.0
            snn_max_corr = 0.0
            snn_last_reward = 0.0
            for b in range(batch):
                dt = max(actual_dt, 0.001) if b == 0 else 0.020
                if brain_mode:
                    delta = cpg.compute(dt=dt, steering=steering)
                    joints = STAND_CTRL + delta
                else:
                    cpg.set_steering(steering)
                    joints = cpg.step(dt=dt)

                cpg_ph = getattr(cpg, '_phase', (step / 48.0) % 1.0)

                # SNN corrections first (locomotion learner, drives)
                if use_snn and snn_brain is not None:
                    snn_state['dt'] = max(actual_dt, 0.01)
                    snn_corr, snn_reward = _snn_step(
                        snn_brain, snn_ctrl, snn_out_ids, snn_state,
                        joints, cpg_ph, last_imu_ypr)
                    joints = joints + snn_corr
                    snn_max_corr = max(snn_max_corr, float(np.abs(snn_corr).max()))
                    snn_last_reward = snn_reward

                # Cerebellum corrections on top (drift fine-tuning)
                if cerebellum and cb is not None:
                    cb_state['dt'] = max(actual_dt, 0.01)
                    corrections = _cb_step(cb, cb_snn, cb_state, joints,
                                           cpg_ph, last_imu_ypr)
                    joints = joints + corrections
                    cb_max_corr = max(cb_max_corr, float(np.abs(corrections).max()))

                cmds.append(mjcf_ctrl_to_oc_i_cmd(joints))

            bot.send_commands(cmds, settle=0.0)

            # IMU every 5th step; update PID state when pid active.
            imu_str = "---"
            if step % 5 == 0:
                imu_results = bot.send_commands(["gp"], settle=0.0, timeout=0.3)
                for chunk in imu_results:
                    parsed = parse_imu(chunk)
                    if parsed:
                        _, vals = parsed
                        if len(vals) >= 6:
                            imu_str = (f"ypr {vals[3]:>6.1f} {vals[4]:>6.1f}"
                                       f" {vals[5]:>6.1f}")
                            last_imu_ypr = (vals[3], vals[4], vals[5])
                            if yaw_pid:
                                yaw_now = vals[3]
                                t_now = time.time()
                                if yaw_ref is None:
                                    yaw_ref = yaw_now
                                    t_last_imu = t_now
                                else:
                                    err = yaw_now - yaw_ref
                                    # Wrap to [-180, 180]
                                    if err > 180: err -= 360
                                    elif err < -180: err += 360
                                    dt_imu = t_now - t_last_imu
                                    yaw_error = err
                                    yaw_integral += err * dt_imu
                                    yaw_integral = max(-20.0, min(20.0,
                                                       yaw_integral))
                                    t_last_imu = t_now
                        imu_count += 1
                        break

            # --- Fall detection + OpenCat recovery (mechanical self-reset) ---
            # If the IMU says we're down for a few reads, drive the built-in
            # OpenCat stand posture and verify upright. Weights kept; only the
            # body resets, so fresh learning runs unattended across falls.
            if recover:
                _pitch, _roll = abs(last_imu_ypr[1]), abs(last_imu_ypr[2])
                if _pitch > fall_threshold or _roll > fall_threshold:
                    fall_consec += 1
                else:
                    fall_consec = 0
                if fall_consec >= _FALL_CONSEC_TRIGGER:
                    print(f"  [FALL] pitch={last_imu_ypr[1]:.0f} "
                          f"roll={last_imu_ypr[2]:.0f} -> recovery ({recover_skill})")
                    ok = _recover_bittle(bot, recover_skill, fall_threshold,
                                         settle=recover_settle)
                    fall_consec = 0
                    recovery_count += 1
                    # Reset episode state (NOT weights) so learning continues.
                    if cerebellum and cb is not None and hasattr(cb, "reset_episode"):
                        try:
                            cb.reset_episode()
                        except Exception:
                            pass
                    # Restart gait phase + re-baseline yaw PID after the reset.
                    if hasattr(cpg, "_phase"):
                        cpg._phase = 0.0
                    yaw_ref = None
                    yaw_integral = 0.0
                    print("  [RECOVERED]" if ok
                          else "  [RECOVERY INCOMPLETE -- still down]")

            step += 1
            elapsed_ms = int((time.time() - t0) * 1000)
            extra = ""
            if yaw_pid and yaw_ref is not None:
                extra += f"  steer: {steering:+.3f}"
            if use_snn:
                extra += f"  snn: {snn_max_corr:.3f} r:{snn_last_reward:+.3f}"
            if cerebellum:
                extra += f"  cb: {cb_max_corr:.3f}"
            print(f"[{step:>4}] {elapsed_ms:>3}ms  joints: {cmds[-1]:<50}"
                  f"  imu: {imu_str}{extra}")

            if _dash_push is not None:
                _dpitch, _droll, _dyaw = last_imu_ypr[1], last_imu_ypr[2], last_imu_ypr[0]
                _dash_push({
                    'source': 'bittle-bridge', 'step': step,
                    'hz': (1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0,
                    'pitch': _dpitch, 'roll': _droll, 'yaw': _dyaw,
                    'reward': snn_last_reward if use_snn else 0.0,
                    'snn_corr': snn_max_corr if use_snn else 0.0,
                    'cb_corr': cb_max_corr if cerebellum else 0.0,
                    'steering': steering,
                    'fallen': bool(abs(_dpitch) > fall_threshold or abs(_droll) > fall_threshold),
                    'recoveries': recovery_count,
                })

            if _telem is not None:
                try:
                    _telem.write(json.dumps({
                        "step": step, "t": round(time.time() - t_start, 3),
                        "ms": elapsed_ms,
                        "yaw": last_imu_ypr[0], "pitch": last_imu_ypr[1],
                        "roll": last_imu_ypr[2],
                        "cmd": cmds[-1] if cmds else "",
                        "steering": round(steering, 4),
                        "snn_corr": round(snn_max_corr, 4) if use_snn else 0.0,
                        "reward": round(snn_last_reward, 4) if use_snn else 0.0,
                        "cb_corr": round(cb_max_corr, 4) if cerebellum else 0.0,
                        "fall_consec": fall_consec,
                        "recoveries": recovery_count,
                    }) + "\n")
                except Exception:
                    pass

            remaining = 0.020 - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        bot.send_commands(["d"])
        t_total = time.time() - t_start
        hz = step / t_total if t_total > 0 else 0.0
        print(f"\nLoop done: {step} steps in {t_total:.1f}s"
              f" = {hz:.1f} Hz, {imu_count} IMU samples received"
              f", {recovery_count} recoveries")

        # --- Persist what was learned (SNN + cerebellum) so the run is not lost.
        # brain.pt here is byte-compatible with the sim (shared topology #159):
        # it can be reloaded into the bridge (--snn-brain) or inspected offline.
        if use_snn and snn_brain is not None and snn_ctrl is not None:
            try:
                from src.brain.brain_persistence import save_brain
                os.makedirs(run_dir, exist_ok=True)
                brain_out = os.path.join(run_dir, "brain.pt")
                save_brain(snn_brain, snn_ctrl, brain_out,
                           metadata={"source": "bittle-bridge", "run_id": run_id,
                                     "steps": step, "hz": round(hz, 2),
                                     "recoveries": recovery_count,
                                     "gait": gait_name},
                           cerebellum=(cb if cerebellum else None))
                print(f"  Brain saved: {brain_out}"
                      f"{' (incl. cerebellum)' if cerebellum else ''}")
            except Exception as e:
                print(f"  Brain save FAILED: {e}")

        if _telem is not None:
            try:
                _telem.write(json.dumps({
                    "type": "footer", "steps": step,
                    "t_total": round(t_total, 2), "hz": round(hz, 2),
                    "imu_samples": imu_count, "recoveries": recovery_count,
                }) + "\n")
                _telem.close()
                print(f"  Telemetry closed: "
                      f"{os.path.join(run_dir, 'telemetry.jsonl')}")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Bittle WiFi/WebSocket test client")
    parser.add_argument("--ip", required=True, help="Bittle IP, e.g. 192.168.1.100")
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--keep-balance", action="store_true",
                        help="Do NOT disable OpenCat balance (default: disable via 'gb').")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of IMU samples to read during motion.")
    parser.add_argument("--test-joints", action="store_true",
                        help="Send STAND_CTRL via 'i' token and print the command (no IMU).")
    parser.add_argument("--gait-loop", action="store_true",
                        help="Stream a gait at ~50 Hz and log IMU per step.")
    parser.add_argument("--gait", default="trot",
                        help="Gait name for --gait-loop (default: trot).")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration in seconds for --gait-loop (default: 5).")
    parser.add_argument("--batch", type=int, default=1,
                        help="Commands per round-trip for --gait-loop (default: 1).")
    parser.add_argument("--yaw-pid", action="store_true",
                        help="Enable IMU yaw feedback PID for straight-line correction.")
    parser.add_argument("--yaw-invert", action="store_true",
                        help="Invert yaw PID sign if robot turns the wrong way.")
    parser.add_argument("--brain", action="store_true",
                        help="Use OpenCatGait.compute() brain interface (delta mode)."
                             " Same gait, drop-in for future SNN integration.")
    parser.add_argument("--snn", action="store_true",
                        help="Fresh SNNController + CognitiveBrain: learns locomotion via"
                             " R-STDP with intrinsic rewards (vestibular, curiosity,"
                             " empowerment). No pre-training needed.")
    parser.add_argument("--cerebellum", action="store_true",
                        help="Fresh CerebellarLearning: applies drift corrections on top"
                             " of CPG (and SNN if --snn). Learns from IMU vestibular error.")
    parser.add_argument("--cerebellum-state", dest="cerebellum_state", default=None,
                        help="Path to a CerebellarLearning state_dict() checkpoint for"
                             " --cerebellum. This is the cerebellum's OWN checkpoint, NOT"
                             " brain.pt (brain.pt is the CognitiveBrain/SNN bundle and"
                             " contains no cerebellum; see logbook #23). Default: fresh.")
    parser.add_argument("--snn-brain", default=None,
                        help="Path to a simulation-trained brain.pt to load into the"
                             " shared-builder SNN topology (#159 sim->hardware transfer)."
                             " Default: fresh random weights.")
    parser.add_argument("--recover", action="store_true",
                        help="Mechanical self-reset: on a detected fall, drive the"
                             " built-in OpenCat stand posture (kbalance) until upright,"
                             " then resume. Keeps SNN/cerebellum weights -> enables"
                             " unattended fresh learning on hardware.")
    parser.add_argument("--fall-threshold", dest="fall_threshold", type=float, default=50.0,
                        help="Pitch/roll (deg) beyond which the Bittle counts as fallen"
                             " (default 50).")
    parser.add_argument("--recover-skill", dest="recover_skill", default=_RECOVER_DEFAULT_SKILL,
                        help="OpenCat posture skill token used to stand up (default kbalance).")
    parser.add_argument("--recover-settle", dest="recover_settle", type=float, default=1.8,
                        help="Seconds to wait for the recovery posture to settle (default 1.8).")
    parser.add_argument("--dashboard", action="store_true",
                        help="Stream live telemetry to a browser dashboard via the"
                             " dashboard_views WebSocket (ws://localhost:5001). Open"
                             " src/viz/bridge_live.html. Needs: pip install websockets.")
    parser.add_argument("--no-telemetry", dest="no_telemetry", action="store_true",
                        help="Disable the per-run telemetry JSONL + brain.pt save"
                             " (both ON by default; written to"
                             " creatures/bittle/bridge_<timestamp>/).")
    args = parser.parse_args()

    bot = BittleWS(args.ip, args.port)
    bot.connect()

    try:
        # --gait-loop: stream gait + log IMU.
        if args.gait_loop:
            run_gait_loop(bot, args.gait, args.duration, args.keep_balance,
                          batch=args.batch, yaw_pid=args.yaw_pid,
                          yaw_invert=args.yaw_invert, brain_mode=args.brain,
                          cerebellum=args.cerebellum, cb_state_path=args.cerebellum_state,
                          use_snn=args.snn, snn_brain_path=args.snn_brain,
                          recover=args.recover, fall_threshold=args.fall_threshold,
                          recover_skill=args.recover_skill, recover_settle=args.recover_settle,
                          dashboard=args.dashboard, telemetry=not args.no_telemetry)
            return

        # --test-joints: verify the send path with a single safe pose.
        if args.test_joints:
            from src.body.opencat_controller import STAND_CTRL, mjcf_ctrl_to_oc_i_cmd
            cmd = mjcf_ctrl_to_oc_i_cmd(STAND_CTRL)
            print(f"\n--- Joint send test ---")
            print(f"  STAND_CTRL → '{cmd}'")
            if not args.keep_balance:
                print("  Disabling balance (gb) ...")
                bot.send_commands(["gb"])
            print(f"  Sending stand pose ...")
            r = bot.send_commands([cmd], settle=0.5)
            print(f"  results: {r}")
            print("  Done.")
            return

        # 1) Disable OpenCat onboard balance so it doesn't modify motor commands.
        if not args.keep_balance:
            print("\n--- Disabling OpenCat balance (gb) ---")
            r = bot.send_commands(["gb"])
            print(f"  results: {r}")
        else:
            print("\n--- Keeping OpenCat balance ON (--keep-balance) ---")

        # 2) Enable continuous IMU print.
        print("\n--- Enabling continuous IMU print (gP) ---")
        r = bot.send_commands(["gP"])
        print(f"  results: {r}")

        # 3) Read IMU samples. Each 'gp' triggers one print6Axis directly.
        print(f"\n--- Reading {args.samples} IMU samples ---")
        got = 0
        for i in range(args.samples):
            results = bot.send_commands(["gp"])
            imu = None
            for chunk in results:
                imu = parse_imu(chunk)
                if imu:
                    break
            if imu:
                got += 1
                label, vals = imu
                if len(vals) >= 6:
                    ypr = " ".join(f"{v:>7.1f}" for v in vals[3:6])
                    acc = " ".join(f"{v:>6.2f}" for v in vals[:3])
                    print(f"  [{got:>2}] {label or '?':<8} ypr: {ypr}   acc: {acc}")
                else:
                    print(f"  [{got:>2}] {label or '?'}: {vals}")
            time.sleep(0.25)  # respect print6Axis 200 ms rate limit
        print(f"\n  Got {got}/{args.samples} IMU samples.")

        # 4) Stop continuous print (gp single shot already set printGyroQ off).
        print("\n--- Done. Leaving balance as set; IMU print returned to once. ---")

    finally:
        bot.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
