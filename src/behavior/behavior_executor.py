"""
MH-FLOCKE — Behavior Executor v0.4.1
========================================
CPG modulation from behavior state (frequency, amplitude).
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.behavior.behavior_knowledge import BehaviorDef, MotorPattern
from src.behavior.behavior_planner import BehaviorPlanner, PlannerState


# Actuator index map for Mogli (21 actuators)
# Muss zum MJCF passen (mogli_mesh.xml <actuator> Reihenfolge)
ACTUATOR_MAP = {
    'neck': 0,
    'fl_shoulder_ab': 1, 'fl_shoulder': 2, 'fl_elbow': 3, 'fl_carpus': 4,
    'fr_shoulder_ab': 5, 'fr_shoulder': 6, 'fr_elbow': 7, 'fr_carpus': 8,
    'rl_hip_ab': 9, 'rl_hip': 10, 'rl_stifle': 11, 'rl_hock': 12,
    'rr_hip_ab': 13, 'rr_hip': 14, 'rr_stifle': 15, 'rr_hock': 16,
    'tail': 17,
    'ear_l': 18,
    'ear_r': 19,
    'jaw': 20,
}

# Cosmetic actuators (not controlled by CPG)
COSMETIC_ACTUATORS = {'neck', 'tail', 'ear_l', 'ear_r', 'jaw'}

# Locomotion Actuatoren (von CPG gesteuert)
LOCOMOTION_ACTUATORS = {
    'fl_shoulder_ab', 'fl_shoulder', 'fl_elbow', 'fl_carpus',
    'fr_shoulder_ab', 'fr_shoulder', 'fr_elbow', 'fr_carpus',
    'rl_hip_ab', 'rl_hip', 'rl_stifle', 'rl_hock',
    'rr_hip_ab', 'rr_hip', 'rr_stifle', 'rr_hock',
}

# Joint ranges (from MJCF, for normalization angle -> ctrl[-1..1])
JOINT_RANGES = {
    'neck': (-30.0, 45.0),
    'fl_shoulder_ab': (-15.0, 25.0), 'fl_shoulder': (-45.0, 60.0),
    'fl_elbow': (-90.0, 10.0), 'fl_carpus': (-30.0, 45.0),
    'fr_shoulder_ab': (-25.0, 15.0), 'fr_shoulder': (-45.0, 60.0),
    'fr_elbow': (-90.0, 10.0), 'fr_carpus': (-30.0, 45.0),
    'rl_hip_ab': (-15.0, 30.0), 'rl_hip': (-60.0, 45.0),
    'rl_stifle': (-100.0, 10.0), 'rl_hock': (-20.0, 50.0),
    'rr_hip_ab': (-30.0, 15.0), 'rr_hip': (-60.0, 45.0),
    'rr_stifle': (-100.0, 10.0), 'rr_hock': (-20.0, 50.0),
    'tail': (-30.0, 60.0),
    'ear_l': (-20.0, 20.0),
    'ear_r': (-20.0, 20.0),
    'jaw': (-15.0, 0.0),
}


def angle_to_ctrl(angle_deg: float, joint_name: str) -> float:
    """Konvertiert Winkel in Grad zu normalisiertem Control-Signal [-1..1]."""
    lo, hi = JOINT_RANGES.get(joint_name, (-45, 45))
    if hi == lo:
        return 0.0
    return np.clip(2.0 * (angle_deg - lo) / (hi - lo) - 1.0, -1.0, 1.0)


class BehaviorExecutor:
    """
    Uebersetzt aktives Behavior in Motor-Modulation.

    Zwei Ausgaben pro Step:
    1. CPG-Modulation: frequency_scale, amplitude_scale
    2. Cosmetic-Controls: Ziel-Werte fuer neck, tail, ears, jaw

    Die kosmetischen Controls werden smooth interpoliert.
    Locomotion-Joints bleiben unter CPG+SNN-Kontrolle, werden
    nur skaliert (amplitude/frequency).
    """

    def __init__(self, n_actuators: int = 21):
        self.n_actuators = n_actuators

        # Current target values for cosmetic joints (normalized -1..1)
        self._cosmetic_targets: Dict[str, float] = {
            'neck': 0.0,
            'tail': 0.0,
            'ear_l': 0.0,
            'ear_r': 0.0,
            'jaw': 0.0,
        }

        # Aktuelle interpolierte Werte
        self._cosmetic_current: Dict[str, float] = dict(self._cosmetic_targets)

        # CPG-Modulation
        self._cpg_freq_scale: float = 1.0
        self._cpg_amp_scale: float = 1.0

        # Blend targets (for transitions)
        self._target_freq_scale: float = 1.0
        self._target_amp_scale: float = 1.0

        # Leg overrides (for mark behavior etc.)
        self._leg_override_targets: Dict[str, float] = {}
        self._leg_override_current: Dict[str, float] = {}

        # Look-around sweep state
        self._sweep_phase: float = 0.0

        self._blend_speed: float = 0.05

    def set_behavior(self, behavior: Optional[BehaviorDef],
                     blend_factor: float = 1.0):
        """
        Setzt neues Verhalten. Wird vom Planner aufgerufen.

        Args:
            behavior: BehaviorDef oder None (idle/walk)
            blend_factor: 0..1 wie weit der Uebergang ist
        """
        if behavior is None:
            self._target_freq_scale = 1.0
            self._target_amp_scale = 1.0
            self._leg_override_targets = {}
            self._blend_speed = 0.05
            return

        motor = behavior.motor
        self._target_freq_scale = motor.cpg_frequency_scale
        self._target_amp_scale = motor.cpg_amplitude_scale
        self._blend_speed = motor.blend_speed

        # Kosmetische Ziele setzen
        if motor.neck_angle is not None:
            self._cosmetic_targets['neck'] = angle_to_ctrl(
                motor.neck_angle, 'neck')
        if motor.jaw_angle is not None:
            self._cosmetic_targets['jaw'] = angle_to_ctrl(
                motor.jaw_angle, 'jaw')
        if motor.ear_angle is not None:
            self._cosmetic_targets['ear_l'] = angle_to_ctrl(
                motor.ear_angle, 'ear_l')
            self._cosmetic_targets['ear_r'] = angle_to_ctrl(
                motor.ear_angle, 'ear_r')
        if motor.tail_angle is not None:
            self._cosmetic_targets['tail'] = angle_to_ctrl(
                motor.tail_angle, 'tail')

        # Leg overrides
        self._leg_override_targets = dict(motor.leg_overrides)

    def step(self, behavior_name: Optional[str] = None,
             behavior_step: int = 0) -> Tuple[float, float, np.ndarray]:
        """
        Einen Step ausfuehren — interpoliert alle Werte.

        Args:
            behavior_name: Fuer Spezial-Logik (look_around sweep)
            behavior_step: Fuer zeitabhaengige Patterns

        Returns:
            (cpg_freq_scale, cpg_amp_scale, cosmetic_overrides)
            cosmetic_overrides: Array[n_actuators] mit NaN wo CPG bestimmt
        """
        # ── Smooth CPG-Modulation ──
        self._cpg_freq_scale += (
            self._target_freq_scale - self._cpg_freq_scale) * self._blend_speed
        self._cpg_amp_scale += (
            self._target_amp_scale - self._cpg_amp_scale) * self._blend_speed

        # ── Smooth kosmetische Joints ──
        for name in self._cosmetic_current:
            target = self._cosmetic_targets[name]

            # Spezial: look_around Kopf-Sweep
            if behavior_name == 'look_around' and name == 'neck':
                self._sweep_phase += 0.005
                sweep = np.sin(self._sweep_phase * 2 * np.pi) * 0.6
                target = sweep  # Hin-und-her

            self._cosmetic_current[name] += (
                target - self._cosmetic_current[name]) * self._blend_speed

        # ── Smooth Leg Overrides ──
        for name in self._leg_override_targets:
            current = self._leg_override_current.get(name, 0.0)
            target = self._leg_override_targets[name]
            self._leg_override_current[name] = current + (
                target - current) * self._blend_speed

        # Reset inactive overrides back to 0
        expired = [n for n in self._leg_override_current
                   if n not in self._leg_override_targets]
        for name in expired:
            self._leg_override_current[name] *= (1.0 - self._blend_speed)
            if abs(self._leg_override_current[name]) < 0.01:
                del self._leg_override_current[name]

        # ── Output Array bauen ──
        overrides = np.full(self.n_actuators, np.nan)

        # Kosmetische Joints setzen
        for name, val in self._cosmetic_current.items():
            idx = ACTUATOR_MAP.get(name)
            if idx is not None and idx < self.n_actuators:
                overrides[idx] = val

        # Leg Overrides setzen
        for name, val in self._leg_override_current.items():
            idx = ACTUATOR_MAP.get(name)
            if idx is not None and idx < self.n_actuators:
                overrides[idx] = val

        return self._cpg_freq_scale, self._cpg_amp_scale, overrides

    def apply_to_controls(self, cpg_controls: np.ndarray,
                          freq_scale: float, amp_scale: float,
                          overrides: np.ndarray) -> np.ndarray:
        """
        Wendet Behavior-Modulation auf CPG-Controls an.

        Args:
            cpg_controls: Rohe CPG-Ausgabe (n_actuators,)
            freq_scale: CPG-Frequency Skalierung (bereits in CPG angewendet)
            amp_scale: CPG-Amplitude Skalierung
            overrides: Array mit NaN (CPG bestimmt) oder Override-Wert

        Returns:
            Modifizierte Controls
        """
        result = cpg_controls.copy()

        # Amplitude skalieren (nur Locomotion-Actuatoren)
        for name in LOCOMOTION_ACTUATORS:
            idx = ACTUATOR_MAP.get(name)
            if idx is not None and idx < len(result):
                if np.isnan(overrides[idx]):
                    result[idx] *= amp_scale
                else:
                    # Override: Behavior bestimmt direkt
                    result[idx] = overrides[idx]

        # Kosmetische Overrides direkt setzen
        for name in COSMETIC_ACTUATORS:
            idx = ACTUATOR_MAP.get(name)
            if idx is not None and idx < len(result):
                if not np.isnan(overrides[idx]):
                    result[idx] = overrides[idx]

        return result

    def get_cpg_frequency_scale(self) -> float:
        """Fuer CPG-Modulation: Frequenz-Skalierung."""
        return self._cpg_freq_scale

    def get_state(self) -> Dict:
        """Fuer Dashboard/Logging."""
        return {
            'cpg_freq_scale': round(self._cpg_freq_scale, 3),
            'cpg_amp_scale': round(self._cpg_amp_scale, 3),
            'cosmetic': {k: round(v, 3) for k, v in self._cosmetic_current.items()},
            'leg_overrides': {k: round(v, 3) for k, v in self._leg_override_current.items()},
        }
