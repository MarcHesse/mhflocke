"""
MH-FLOCKE — Bittle Body Profile
================================
8 DOF quadruped body profile for the Petoi Bittle X.
Maps between MH-FLOCKE internal leg order and OpenCat/MJCF conventions.

Joint order (MH-FLOCKE actuator index):
  0: RF shoulder   1: RF knee
  2: LF shoulder   3: LF knee
  4: RR shoulder   5: RR knee
  6: LR shoulder   7: LR knee
"""

__version__ = "1.0"
__logbook__ = 123

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BittleConfig:
    """Static configuration for the Petoi Bittle X."""

    # Model paths
    mjcf_path: str = "creatures/bittle/scene.xml"
    creature_xml: str = "creatures/bittle/bittle.xml"

    # Degrees of freedom
    n_dof: int = 8
    n_legs: int = 4
    dof_per_leg: int = 2  # shoulder + knee

    # Physical dimensions (meters)
    body_length: float = 0.120   # front-to-rear servo distance
    body_width: float = 0.072   # left-to-right servo distance
    thigh_length: float = 0.050
    shank_length: float = 0.060
    body_mass: float = 0.165    # kg (torso only)

    # Simulation
    timestep: float = 0.002
    actuator_kp: float = 20.0

    # OpenCat rest angles (boot position from currentAng[])
    rest_shoulder_deg: float = 75.0
    rest_knee_deg: float = -55.0


# Joint definitions
JOINT_NAMES = [
    "shrfs_joint",  # 0: RF shoulder
    "shrft_joint",  # 1: RF knee
    "shlfs_joint",  # 2: LF shoulder
    "shlft_joint",  # 3: LF knee
    "shrrs_joint",  # 4: RR shoulder
    "shrrt_joint",  # 5: RR knee
    "shlrs_joint",  # 6: LR shoulder
    "shlrt_joint",  # 7: LR knee
]

ACTUATOR_NAMES = [
    "rf_shoulder_motor",
    "rf_knee_motor",
    "lf_shoulder_motor",
    "lf_knee_motor",
    "rr_shoulder_motor",
    "rr_knee_motor",
    "lr_shoulder_motor",
    "lr_knee_motor",
]

# Leg grouping (actuator indices)
LEG_RF = (0, 1)
LEG_LF = (2, 3)
LEG_RR = (4, 5)
LEG_LR = (6, 7)

LEGS = {
    "RF": LEG_RF,
    "LF": LEG_LF,
    "RR": LEG_RR,
    "LR": LEG_LR,
}

# Bilateral symmetry pairs (for weight averaging)
BILATERAL_PAIRS = [
    (LEG_RF, LEG_LF),  # front pair
    (LEG_RR, LEG_LR),  # rear pair
]

# Foot site names (for contact detection)
FOOT_SITES = ["rf_foot_site", "lf_foot_site", "rr_foot_site", "lr_foot_site"]

# Contact sensor names
CONTACT_SENSORS = ["rf_contact", "lf_contact", "rr_contact", "lr_contact"]

# IMU sensor names
IMU_ACCEL = "imu_accel"
IMU_GYRO = "imu_gyro"
TORSO_QUAT = "torso_quat"
TORSO_POS = "torso_pos"
TORSO_VEL = "torso_vel"


# ============================================================
# OpenCat <-> MJCF conversion
# ============================================================

# Axis signs per joint (from bittle.xml axis attributes)
# +1 for axis="0 1 0", -1 for axis="0 -1 0"
AXIS_SIGNS = {
    "shrfs_joint": +1,  # RF shoulder
    "shrft_joint": +1,  # RF knee
    "shlfs_joint": +1,  # LF shoulder
    "shlft_joint": +1,  # LF knee
    "shrrs_joint": -1,  # RR shoulder (axis="0 -1 0")
    "shrrt_joint": +1,  # RR knee
    "shlrs_joint": -1,  # LR shoulder (axis="0 -1 0")
    "shlrt_joint": +1,  # LR knee
}

# OpenCat servo index to MH-FLOCKE actuator index mapping
# OpenCat indices: 8=LF_sh, 9=RF_sh, 10=RR_sh, 11=LR_sh,
#                  12=LF_kn, 13=RF_kn, 14=RR_kn, 15=LR_kn
OPENCAT_TO_MHFLOCKE = {
    8: 2,   # LF_sh -> act 2
    9: 0,   # RF_sh -> act 0
    10: 4,  # RR_sh -> act 4
    11: 6,  # LR_sh -> act 6
    12: 3,  # LF_kn -> act 3
    13: 1,  # RF_kn -> act 1
    14: 5,  # RR_kn -> act 5
    15: 7,  # LR_kn -> act 7
}

# OpenCat frame column order: [LF_sh, RF_sh, RR_sh, LR_sh, LF_kn, RF_kn, RR_kn, LR_kn]
# Maps column index -> (MH-FLOCKE actuator index, is_shoulder)
FRAME_COL_MAP = [
    (2, True),   # col 0: LF_sh
    (0, True),   # col 1: RF_sh
    (4, True),   # col 2: RR_sh
    (6, True),   # col 3: LR_sh
    (3, False),  # col 4: LF_kn
    (1, False),  # col 5: RF_kn
    (5, False),  # col 6: RR_kn
    (7, False),  # col 7: LR_kn
]


def opencat_to_mjcf_angle(opencat_deg: float, joint_name: str) -> float:
    """Convert a single OpenCat angle (degrees) to MJCF radians.

    Formula: mjcf_angle = (opencat_angle - rest_angle) * axis_sign * pi/180
    """
    is_shoulder = "s_joint" in joint_name  # shrfs, shlfs, shrrs, shlrs
    rest = BittleConfig.rest_shoulder_deg if is_shoulder else BittleConfig.rest_knee_deg
    axis_sign = AXIS_SIGNS[joint_name]
    return (opencat_deg - rest) * axis_sign * math.pi / 180.0


def mjcf_to_opencat_angle(mjcf_rad: float, joint_name: str) -> float:
    """Convert MJCF radians back to OpenCat degrees.

    Inverse: opencat_angle = (mjcf_angle / axis_sign) * 180/pi + rest_angle
    """
    is_shoulder = "s_joint" in joint_name
    rest = BittleConfig.rest_shoulder_deg if is_shoulder else BittleConfig.rest_knee_deg
    axis_sign = AXIS_SIGNS[joint_name]
    return (mjcf_rad / axis_sign) * 180.0 / math.pi + rest


def convert_opencat_frame(frame: list) -> np.ndarray:
    """Convert an 8-value OpenCat frame to 8 MJCF ctrl values.

    Input:  [LF_sh, RF_sh, RR_sh, LR_sh, LF_kn, RF_kn, RR_kn, LR_kn]
    Output: [RF_sh, RF_kn, LF_sh, LF_kn, RR_sh, RR_kn, LR_sh, LR_kn]
    """
    ctrl = np.zeros(8)
    for col_idx, (act_idx, is_shoulder) in enumerate(FRAME_COL_MAP):
        joint_name = JOINT_NAMES[act_idx]
        ctrl[act_idx] = opencat_to_mjcf_angle(frame[col_idx], joint_name)
    return ctrl


# ============================================================
# Stand pose (from bittle.xml keyframe)
# ============================================================

STAND_CTRL = np.array([
    -0.7854,  # RF shoulder
    +1.4835,  # RF knee
    -0.7854,  # LF shoulder
    +1.4835,  # LF knee
    +0.7854,  # RR shoulder
    +1.4835,  # RR knee
    +0.7854,  # LR shoulder
    +1.4835,  # LR knee
])

STAND_QPOS_BODY = np.array([
    0, 0, 0.06,   # position
    1, 0, 0, 0,   # quaternion
])

NECK_STAND_ANGLE = -0.48  # radians, straightens head


# ============================================================
# CPG interface (for MogliCPG integration)
# ============================================================

# MogliCPG with joints_per_leg=2 produces:
#   [FL_abd(=0), FL_hip, FR_abd(=0), FR_hip, RL_abd(=0), RL_hip, RR_abd(=0), RR_hip]
# ABD is always 0 because abd_amplitude=0. The HIP command is on odd indices.
# But Bittle needs HIP on shoulder and KNEE (which MogliCPG doesn't produce
# with jpleg=2).
#
# Solution: Run MogliCPG with joints_per_leg=3 and abd_amplitude=0,
# then pick HIP and KNEE from the 12-element output.
# This way the oscillator generates proper hip+knee commands.
#
# CPG output (12 elements, jpleg=3):
#   [FL_abd, FL_hip, FL_knee, FR_abd, FR_hip, FR_knee,
#    RL_abd, RL_hip, RL_knee, RR_abd, RR_hip, RR_knee]
#
# We extract HIP→shoulder, KNEE→knee for each leg.
# CRITICAL: RR and LR shoulders have inverted axis (axis="0 -1 0"),
# so their HIP commands must be negated relative to FL/FR.
CPG12_TO_ACTUATOR = {
    # (cpg_index, actuator_index, sign)
    1:  (2, +1),   # FL_hip  -> act 2 (LF_sh), normal axis
    2:  (3, +1),   # FL_knee -> act 3 (LF_kn)
    4:  (0, +1),   # FR_hip  -> act 0 (RF_sh), normal axis
    5:  (1, +1),   # FR_knee -> act 1 (RF_kn)
    7:  (6, -1),   # RL_hip  -> act 6 (LR_sh), INVERTED axis
    8:  (7, +1),   # RL_knee -> act 7 (LR_kn)
    10: (4, -1),   # RR_hip  -> act 4 (RR_sh), INVERTED axis
    11: (5, +1),   # RR_knee -> act 5 (RR_kn)
}


# Knee bias: REMOVED. The MogliCPG swings symmetrically around 0,
# and STAND_CTRL already positions the knees correctly.
# Previous attempts with negative bias lifted feet off ground.
KNEE_BIAS = 0.0  # no additional bias


def cpg_output_to_ctrl(cpg_output: np.ndarray) -> np.ndarray:
    """Map CPG output to 8 Bittle actuator ctrl values.

    Supports two modes:
    - 12-element MogliCPG output: extracts HIP+KNEE, remaps legs, inverts rear shoulders
    - 8-element OpenCatGait output: direct delta, no remapping needed
    """
    if len(cpg_output) == 8:
        # OpenCatGait: already in correct actuator order
        return STAND_CTRL + cpg_output

    # MogliCPG 12-element: extract and remap
    ctrl = np.zeros(8)
    for cpg_idx, (act_idx, sign) in CPG12_TO_ACTUATOR.items():
        val = cpg_output[cpg_idx] * sign
        if act_idx % 2 == 1:  # knee
            val += KNEE_BIAS
        ctrl[act_idx] = val
    return STAND_CTRL + ctrl


def apply_bilateral_symmetry(weights: np.ndarray) -> np.ndarray:
    """Average weight pairs between left and right legs.

    Prevents the known drift issue from asymmetric weight initialization.
    """
    result = weights.copy()
    for (leg_a, leg_b) in BILATERAL_PAIRS:
        for i in range(len(leg_a)):
            avg = (result[leg_a[i]] + result[leg_b[i]]) / 2.0
            result[leg_a[i]] = avg
            result[leg_b[i]] = avg
    return result
