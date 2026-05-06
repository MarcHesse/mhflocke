"""
MH-FLOCKE — Hardware Drift Simulation v0.1.3
================================================
Injects realistic mechanical drift into the MuJoCo simulator,
based on measured data from a physical robot or synthetic profiles.

v0.1.3: Calibrated _TORQUE_PER_DEG_S from 0.0005 to 0.05.
        0.0005 was 100x too weak because foot contact forces
        dominate the base body dynamics. Empirically calibrated
        via calibrate_drift.py to produce ~2 deg/s yaw drift.
v0.1.2: Fix NaN crash — SET not accumulate xfrc_applied.
v0.1.1: Body name fallback.

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

import json
import numpy as np
from typing import Optional, Dict


class HardwareDrift:
    """
    Simulates mechanical drift from hardware asymmetry.

    Applies external torque and force perturbations to the robot's
    base body in MuJoCo, mimicking measured or synthetic drift
    characteristics.

    When no profile is loaded, all methods are no-ops (zero cost).
    """

    VERSION = "v0.1.3"
    _ROOT_BODY_NAMES = ['base', 'torso', 'body', 'trunk', 'chassis']

    def __init__(self):
        """Create an inactive (no-op) drift instance."""
        self.active = False
        self.profile = None
        self._rng = np.random.default_rng(seed=None)
        self._yaw_rate = 0.0
        self._yaw_noise_std = 0.0
        self._yaw_max_rate = 0.0
        self._roll_bias = 0.0
        self._roll_noise_std = 0.0
        self._servo_scale = {}
        self._body_id = None
        self._resolved = False

    @classmethod
    def from_profile(cls, profile_path: str) -> 'HardwareDrift':
        """Load drift characteristics from a JSON profile."""
        instance = cls()
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)
        except FileNotFoundError:
            print(f'  HardwareDrift: profile not found ({profile_path}) — disabled')
            return instance
        except json.JSONDecodeError as e:
            print(f'  HardwareDrift: invalid JSON ({e}) — disabled')
            return instance

        instance.profile = profile
        instance.active = True

        yaw = profile.get('yaw_drift', {})
        instance._yaw_rate = yaw.get('rate_deg_per_s', 0.0)
        instance._yaw_noise_std = yaw.get('noise_std_deg_per_s', 0.0)
        instance._yaw_max_rate = yaw.get('max_rate_deg_per_s', instance._yaw_rate * 2)

        roll = profile.get('roll_bias', {})
        instance._roll_bias = roll.get('mean_deg', 0.0)
        instance._roll_noise_std = roll.get('noise_std_deg', 0.0)

        servo = profile.get('servo_asymmetry', {})
        for leg in ['FL', 'FR', 'RL', 'RR']:
            instance._servo_scale[leg] = servo.get(leg, 1.0)

        name = profile.get('name', profile.get('robot_id', 'unknown'))
        print(f'  HardwareDrift: loaded "{name}"')
        print(f'    yaw_rate={instance._yaw_rate:.1f} deg/s, '
              f'roll_bias={instance._roll_bias:.1f} deg, '
              f'servo_asym={instance._servo_scale}')

        return instance

    def _resolve_body_id(self, creature) -> bool:
        """Find the MuJoCo body ID for the robot's base body."""
        if self._resolved:
            return self._body_id is not None
        self._resolved = True
        try:
            import mujoco as mj
            model = creature.world._model
            for name in [creature.body_name] + self._ROOT_BODY_NAMES:
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    self._body_id = bid
                    if name != creature.body_name:
                        print(f'  HardwareDrift: body "{creature.body_name}" not found, '
                              f'using "{name}" (id={bid})')
                    else:
                        print(f'  HardwareDrift: body "{name}" (id={bid})')
                    return True
            print(f'  HardwareDrift: no root body found — disabled')
            self.active = False
            return False
        except Exception as e:
            print(f'  HardwareDrift: body resolution failed ({e}) — disabled')
            self.active = False
            return False

    def apply(self, creature, dt: float = 0.002) -> None:
        """
        Apply drift forces to the robot in MuJoCo.
        Call BEFORE world.step(). Uses SET (=) not accumulate (+=).
        """
        if not self.active:
            return
        if not self._resolved:
            if not self._resolve_body_id(creature):
                return

        data = creature.world._data
        data.xfrc_applied[self._body_id] = 0.0

        # === Yaw drift torque ===
        # Calibrated: 0.05 Nm per deg/s for Freenove (0.5kg, 4 feet on ground).
        # Contact friction dominates — the feet hold the base, so torque must
        # overcome static friction × lever arm to produce yaw rotation.
        # Empirically verified via calibrate_drift.py.
        _TORQUE_PER_DEG_S = 0.05

        yaw_rate = self._yaw_rate + self._rng.normal(0, self._yaw_noise_std)
        yaw_rate = np.clip(yaw_rate, min(self._yaw_max_rate, self._yaw_rate),
                           max(-self._yaw_max_rate, -self._yaw_rate))
        data.xfrc_applied[self._body_id, 5] = yaw_rate * _TORQUE_PER_DEG_S

        # === Roll bias force ===
        _MASS = 0.50
        _G = 9.81
        roll_deg = self._roll_bias + self._rng.normal(0, self._roll_noise_std)
        data.xfrc_applied[self._body_id, 1] = _MASS * _G * np.sin(np.radians(roll_deg))

    def apply_servo_asymmetry(self, leg_name: str, torque: float) -> float:
        """Scale servo torque for a specific leg to simulate asymmetry."""
        if not self.active:
            return torque
        return torque * self._servo_scale.get(leg_name, 1.0)

    def get_stats(self) -> Dict:
        """Stats for FLOG logging / dashboard."""
        if not self.active:
            return {'drift_active': False}
        return {
            'drift_active': True,
            'yaw_rate': self._yaw_rate,
            'roll_bias': self._roll_bias,
            'name': self.profile.get('name', 'unknown') if self.profile else 'unknown',
        }
