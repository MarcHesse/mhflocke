"""OpenCat Balance Controller — ported from motion.h adjust().

Reads roll/pitch from MuJoCo IMU (quaternion) and computes per-joint
servo angle corrections to keep the Bittle upright.

Reference: PetoiCamp/OpenCatEsp32-Quadruped-Robot src/motion.h lines 260-332
"""
import numpy as np
import math


class OpenCatBalance:
    """IMU-based balance corrections for OpenCat gaits in MuJoCo."""

    # Dead zones (degrees) — body still considered level
    ROLL_TOLERANCE = 5.0
    PITCH_TOLERANCE = 3.0

    # Damping — max correction change per step (degrees)
    ADJUSTMENT_DAMPER = 15.0  # increased from 5.0 for MuJoCo (no servo inertia)

    # Factors from OpenCat source
    LEFT_RIGHT_FACTOR = 2.0
    POSTURE_WALKING_FACTOR = 0.5

    # Per-joint balance coefficients [roll_factor, pitch_factor]
    # From motion.h adaptiveParameterArray (X_LEG config)
    # OpenCat 16-joint layout:
    #   0-3: head group (neck pan, neck tilt, spare, spare)
    #   4-7: shoulders (RF, LF, RB, LB)
    #   8-11: upper legs (RF, LF, RB, LB)
    #   12-15: lower legs/knees (RF, LF, RB, LB)
    #
    # Bittle 8-servo mapping (our MJCF order):
    #   servo 0-3 = shoulders = OpenCat joints 4-7
    #   servo 4-7 = knees = OpenCat joints 8-11
    #
    # Coefficients (will be divided by radPerDeg in adjust):
    _sRF = 50    # shoulder roll factor
    _sPF = 12    # shoulder pitch factor
    _uRF = 50    # upper leg roll factor
    _uPF = 50    # upper leg pitch factor
    _lRF = -75   # lower leg roll factor (-1.5 * uRF)
    _lPF = -75   # lower leg pitch factor (-1.5 * uPF)

    # Bittle X_LEG coefficients for joints 4-11 (our 8 servos)
    # [roll_coeff, pitch_coeff] per OpenCat joint
    ADAPT_XLEG = {
        # Shoulders (OpenCat joints 4-7)
        4: [_sRF, -_sPF],     # RF shoulder
        5: [-_sRF, -_sPF],    # LF shoulder
        6: [-_sRF, _sPF],     # RB shoulder
        7: [_sRF, _sPF],      # LB shoulder
        # Upper legs / knees (OpenCat joints 8-11)
        8: [_uRF, _uPF],      # RF upper
        9: [_uRF, _uPF],      # LF upper
        10: [-_uRF, _uPF],    # RB upper
        11: [-_uRF, _uPF],    # LB upper
    }

    def __init__(self, n_actuators=8, joint_mapping=None, gain_scale=1.0):
        """
        Args:
            n_actuators: Number of actuated joints (8 for Bittle).
            joint_mapping: Map from our servo index to OpenCat joint index.
                          Default: [4,5,6,7,8,9,10,11] for Bittle.
            gain_scale: Scale factor for all balance gains (1.0 = original OpenCat).
        """
        self.n_act = n_actuators
        self.deg_per_rad = 180.0 / math.pi
        self.rad_per_deg = math.pi / 180.0

        # Map our servo indices to OpenCat joint indices
        if joint_mapping is None:
            # Default Bittle: servos 0-3 = shoulders (OC 4-7),
            #                 servos 4-7 = knees (OC 8-11)
            self.joint_map = [4, 5, 6, 7, 8, 9, 10, 11]
        else:
            self.joint_map = joint_mapping

        # Build coefficient arrays aligned to our servo order
        self.roll_coeff = np.zeros(n_actuators)
        self.pitch_coeff = np.zeros(n_actuators)
        for servo_idx, oc_joint in enumerate(self.joint_map):
            if oc_joint in self.ADAPT_XLEG:
                self.roll_coeff[servo_idx] = self.ADAPT_XLEG[oc_joint][0] * gain_scale
                self.pitch_coeff[servo_idx] = self.ADAPT_XLEG[oc_joint][1] * gain_scale

        # State
        self.current_adjust = np.zeros(n_actuators)  # damped corrections (degrees)
        self.balance_slope = np.array([1.0, 1.0])     # roll, pitch direction

    def quat_to_roll_pitch(self, quat):
        """Convert MuJoCo quaternion [w,x,y,z] to roll, pitch in degrees."""
        w, x, y, z = quat
        # Roll (rotation around X axis)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp) * self.deg_per_rad
        # Pitch (rotation around Y axis)
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp) * self.deg_per_rad
        return roll, pitch

    def compute(self, quat, expected_roll=0.0, expected_pitch=0.0):
        """Compute balance corrections for all joints.

        Args:
            quat: MuJoCo body quaternion [w, x, y, z].
            expected_roll: Expected roll angle in degrees (0 for flat walking).
            expected_pitch: Expected pitch angle in degrees.

        Returns:
            corrections: Array of angle corrections in RADIANS for each servo.
        """
        roll_deg, pitch_deg = self.quat_to_roll_pitch(quat)

        # Deviation from expected orientation
        roll_dev = roll_deg - expected_roll
        pitch_dev = pitch_deg - expected_pitch

        # Apply dead zone (levelTolerance)
        if abs(roll_dev) < self.ROLL_TOLERANCE:
            roll_dev = 0.0
        else:
            roll_dev = math.copysign(abs(roll_dev) - self.ROLL_TOLERANCE, roll_deg)

        if abs(pitch_dev) < self.PITCH_TOLERANCE:
            pitch_dev = 0.0
        else:
            pitch_dev = math.copysign(abs(pitch_dev) - self.PITCH_TOLERANCE, pitch_deg)

        # Cutoff for walking (reduce noise)
        cutoff = 15.0
        roll_dev = max(-cutoff, min(cutoff, roll_dev))
        pitch_dev = max(-cutoff, min(cutoff, pitch_dev))

        # Compute per-joint corrections
        for i in range(self.n_act):
            oc_joint = self.joint_map[i]

            # Pitch adjustment
            pitch_adj = self.pitch_coeff[i] * pitch_dev

            # Roll adjustment with left/right asymmetry
            left_q = (oc_joint - 1) % 4 > 1
            left_right_factor = 1.0
            if ((left_q and self.balance_slope[0] * roll_dev > 0) or
                    (not left_q and self.balance_slope[0] * roll_dev < 0)):
                left_right_factor = self.LEFT_RIGHT_FACTOR * abs(self.balance_slope[0])

            # Upper legs (joints 8+) use abs(roll), shoulders use signed roll
            if oc_joint > 7:
                roll_adj = abs(roll_dev) * self.roll_coeff[i] * left_right_factor
            else:
                roll_adj = roll_dev * self.roll_coeff[i] * left_right_factor

            # Walking factor for leg joints (not head)
            walking_factor = self.POSTURE_WALKING_FACTOR if oc_joint > 3 else 1.0

            # Ideal adjustment (in degrees, then convert to radians internally)
            ideal = self.rad_per_deg * walking_factor * (
                self.balance_slope[0] * roll_adj -
                self.balance_slope[1] * pitch_adj
            )

            # Damping: limit change rate
            delta = ideal - self.current_adjust[i]
            delta = max(-self.ADJUSTMENT_DAMPER, min(self.ADJUSTMENT_DAMPER, delta))
            self.current_adjust[i] += delta

            # Clamp (asymmetric for front legs)
            upper_limit = 15.0 if (oc_joint > 3 and oc_joint % 4 < 2) else 45.0
            self.current_adjust[i] = max(-45.0, min(upper_limit, self.current_adjust[i]))

        # Return corrections in radians (current_adjust is in degrees)
        return self.current_adjust * self.rad_per_deg

    def reset(self):
        """Reset all corrections to zero."""
        self.current_adjust[:] = 0.0
