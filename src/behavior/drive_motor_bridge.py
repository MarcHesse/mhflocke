"""
MH-FLOCKE — Drive-Motor Bridge v0.4.1
========================================
Maps motivational drives to motor pattern parameters.
"""

import logging
from typing import Dict, Optional, Tuple

from src.behavior.behavior_knowledge import BehaviorKnowledge
from src.behavior.behavior_planner import BehaviorPlanner, Situation
from src.behavior.scene_instruction import SceneInstruction

logger = logging.getLogger(__name__)


class DriveMotorBridge:
    """
    Bridges motivational drives to CPG motor modulation.

    Owns: BehaviorKnowledge, BehaviorPlanner
    Input: drive_state from CognitiveBrain, sensor situation
    Output: (freq_scale, amp_scale, behavior_name) for CPG modulation
    """

    def __init__(self, creature_type: str = 'dog',
                 scene_instruction: Optional[SceneInstruction] = None):
        """
        Args:
            creature_type: 'dog' loads full dog behavior repertoire
            scene_instruction: optional scene context for behavior weighting
        """
        self.knowledge = BehaviorKnowledge(creature_type=creature_type)
        self.planner = BehaviorPlanner(self.knowledge)

        if scene_instruction:
            self.planner.set_scene(scene_instruction)

        # Smoothed output (EMA to avoid jitter between behaviors)
        self._freq_scale = 1.0
        self._amp_scale = 1.0
        self._smooth_rate = 0.02  # slow blend, ~50 steps to transition

        self._step_count = 0
        self._behavior_history = []  # last N behavior names for logging

    def update(self, brain_result: Dict, sensor_data: Dict,
               is_fallen: bool = False) -> Tuple[float, float, str]:
        """
        One step of the drive-motor loop.

        Args:
            brain_result: dict from creature.step()['brain'], contains:
                - drives: {survival, exploration, comfort, social, dominant}
                - emotion: {valence, arousal, dominant_emotion}
                - curiosity_reward, empowerment, etc.
            sensor_data: dict with upright, height, speed, etc.
            is_fallen: whether creature is currently fallen

        Returns:
            (freq_scale, amp_scale, behavior_name)
            freq_scale: 0.0 (stop) to 1.5 (fast trot), modulates CPG frequency
            amp_scale: 0.0 (no movement) to 1.2 (large strides), modulates CPG amplitude
        """
        self._step_count += 1

        # Extract drive state from brain result
        drive_state = brain_result.get('drives', {})
        if not drive_state:
            # Fallback: default drives
            drive_state = {
                'survival': 0.4,
                'exploration': 0.3,
                'comfort': 0.2,
                'social': 0.0,
                'dominant': 'survival' if is_fallen else 'exploration',
            }

        # Build situation from sensor data (including sensory environment)
        situation = Situation(
            upright=sensor_data.get('upright', 1.0),
            speed=sensor_data.get('forward_velocity', 0.0),
            height=sensor_data.get('height', 0.35),
            is_fallen=is_fallen,
            steps_alive=self._step_count,
            prediction_error=brain_result.get('prediction_error', 0.0),
            energy_spent=brain_result.get('energy_spent', 0.0),
            # Sensory environment (Issue #75)
            smell_strength=sensor_data.get('smell_strength', 0.0),
            smell_direction=sensor_data.get('smell_direction', 0.0),
            sound_intensity=sensor_data.get('sound_intensity', 0.0),
            sound_direction=sensor_data.get('sound_direction', 0.0),
        )

        # BehaviorPlanner decides next behavior
        behavior_name = self.planner.update(drive_state, situation)

        # Get the MotorPattern for this behavior
        behavior_def = self.knowledge.get_behavior(behavior_name)
        if behavior_def:
            target_freq = behavior_def.motor.cpg_frequency_scale
            target_amp = behavior_def.motor.cpg_amplitude_scale
        else:
            target_freq = 1.0
            target_amp = 1.0

        # Smooth transition (EMA)
        self._freq_scale += (target_freq - self._freq_scale) * self._smooth_rate
        self._amp_scale += (target_amp - self._amp_scale) * self._smooth_rate

        # Track behavior history (keep last 10 for logging)
        if (not self._behavior_history or
                self._behavior_history[-1] != behavior_name):
            self._behavior_history.append(behavior_name)
            if len(self._behavior_history) > 10:
                self._behavior_history.pop(0)

        return self._freq_scale, self._amp_scale, behavior_name

    def get_state(self) -> Dict:
        """For dashboard/logging."""
        planner_state = self.planner.get_state()
        return {
            'freq_scale': round(self._freq_scale, 3),
            'amp_scale': round(self._amp_scale, 3),
            'behavior': planner_state.get('current_behavior', 'idle'),
            'planner_state': planner_state.get('planner_state', 'idle'),
            'behavior_progress': planner_state.get('behavior_progress', 0.0),
            'blend_progress': planner_state.get('blend_progress', 0.0),
            'behavior_history': list(self._behavior_history),
        }
