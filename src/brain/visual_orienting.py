"""
MH-FLOCKE — Visual Orienting v0.4.1
========================================
Vestibulo-ocular reflex and visual attention.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class VORConfig:
    """Configuration for Visual Orienting Response."""
    # Gain: how strongly the creature turns toward the target.
    # 0.15 = moderate turn, enough to curve toward a ball at 3m offset
    # over ~5000 steps. Too high (>0.3) destabilizes the CPG gait.
    hip_gain: float = 0.15       # Hip flexion asymmetry gain
    abd_gain: float = 0.10       # Hip abduction asymmetry gain
    # Deadzone: heading errors below this are ignored (prevents wobble
    # when nearly facing the target). In radians equivalent: ~6 degrees.
    deadzone: float = 0.03       # Normalized heading, ~0.03 * pi = 5.4 deg
    # Upright gate: VOR only active when creature is sufficiently upright.
    # A fallen creature should not try to orient — it needs to right itself first.
    upright_threshold: float = 0.5
    # Smoothing: EMA on the steering signal to prevent jerk.
    # Biology: Tectospinal pathway has inherent temporal smoothing.
    smoothing: float = 0.9       # EMA decay (0.9 = smooth, 0.5 = responsive)
    # Max output: clip to prevent destabilizing the gait.
    max_output: float = 0.25


class VisualOrientingResponse:
    """
    Brainstem reflex: turn body toward visual target.
    
    This is NOT a learned behavior. It's a hardwired orienting reflex
    comparable to the vestibulo-ocular reflex or startle response.
    The SNN vision channels provide input for LEARNING (what to do
    with the target), while the VOR provides the immediate motor
    response (turn toward it).
    
    Usage in training loop:
        vor = VisualOrientingResponse()
        ...
        steering = vor.compute(heading, distance, upright)
        creature._steering_offset = steering
    """
    
    def __init__(self, config: VORConfig = None, n_actuators: int = 12):
        self.config = config or VORConfig()
        self.n_actuators = n_actuators
        self._steering_ema = 0.0
        self._prev_heading = 0.0  # For D-term (derivative damping)
        self._step = 0
        # Stats for logging
        self.stats = {
            'vor_raw': 0.0,
            'vor_smoothed': 0.0,
            'vor_gated': True,
        }
    
    def compute(self, target_heading: float, target_distance: float,
                upright: float = 1.0, cpg_weight: float = 0.9) -> float:
        """
        Compute steering signal from visual target heading.
        
        Args:
            target_heading: -1 (target far left) to +1 (target far right)
                           Normalized: -1 = -180deg, 0 = straight ahead, +1 = +180deg
            target_distance: 0 (far/no target) to 1 (touching).
                           Currently unused for steering magnitude (dog turns
                           toward ball regardless of distance), but available
                           for future use (e.g. slow down when close).
            upright: 0..1, creature uprightness. VOR gated off when fallen.
            cpg_weight: 0..1, current CPG weight in motor blend. When CPG is
                dominant (0.9), less steering is needed. When SNN takes over
                (0.4), the CPG asymmetry gets diluted and we need more.
                Biology: Reticulospinal gain is modulated by brainstem arousal.
            
        Returns:
            Steering signal: -max..+max. Positive = turn right.
            Set this as creature._steering_offset in the training loop.
        """
        self._step += 1
        
        # Gate: no orienting when fallen or nearly fallen
        if upright < self.config.upright_threshold:
            self.stats['vor_gated'] = False
            self.stats['vor_raw'] = 0.0
            self.stats['vor_smoothed'] = self._steering_ema
            return self._steering_ema  # Let EMA decay naturally
        
        self.stats['vor_gated'] = True
        
        # Deadzone: ignore tiny heading errors (prevents wobble at target)
        if abs(target_heading) < self.config.deadzone:
            raw_steer = 0.0
        else:
            # PD controller: Proportional + Derivative damping
            # P-term: proportional to heading error (turn toward ball)
            # D-term: proportional to heading CHANGE RATE (brake if overshooting)
            # Biology: Superior Colliculus has both position and velocity
            # sensitive neurons. Velocity damping prevents saccade overshoot.
            # Ref: Sparks 1986 (SC burst neurons encode displacement + velocity)
            sign = 1.0 if target_heading > 0 else -1.0
            magnitude = abs(target_heading) - self.config.deadzone
            
            # P-term: sqrt response (gentle small corrections, assertive large)
            p_term = sign * np.sqrt(magnitude) * self.config.hip_gain
            
            # D-term: damping based on heading change rate
            # If heading is decreasing (we're turning the right way), brake
            # If heading is increasing (we're turning the wrong way), boost
            heading_delta = target_heading - self._prev_heading
            d_gain = 0.5  # Damping coefficient
            d_term = heading_delta * d_gain
            
            # CPG-weight compensation: dilution by SNN output
            cpg_boost = 1.0 / max(0.3, cpg_weight)
            
            raw_steer = (p_term - d_term) * cpg_boost
        
        self._prev_heading = target_heading
        
        self.stats['vor_raw'] = raw_steer
        
        # EMA smoothing: prevents jerk from frame-to-frame heading noise
        alpha = 1.0 - self.config.smoothing
        self._steering_ema = self._steering_ema * self.config.smoothing + raw_steer * alpha
        
        # Clip to max
        self._steering_ema = np.clip(self._steering_ema, -self.config.max_output, self.config.max_output)
        
        self.stats['vor_smoothed'] = self._steering_ema
        
        return self._steering_ema
    
    def get_motor_corrections(self, steering: float) -> np.ndarray:
        """
        Convert steering signal to per-joint corrections.
        
        For quadrupeds with 12 actuators (hip, knee, abd per leg):
          FL=0,1,2  FR=3,4,5  RL=6,7,8  RR=9,10,11
          
        Positive steering (turn right):
          Left legs: +hip flexion, +abduction (longer outward stride)
          Right legs: -hip flexion, -abduction (shorter inward stride)
        
        Returns:
            np.ndarray of shape (n_actuators,) with corrections to add to controls.
        """
        corr = np.zeros(self.n_actuators)
        if self.n_actuators < 12 or abs(steering) < 0.001:
            return corr
        
        hip_c = steering * self.config.hip_gain / self.config.max_output  # Normalize
        abd_c = steering * self.config.abd_gain / self.config.max_output
        
        # Left legs: FL(0), RL(6) hip; FL(2), RL(8) abduction
        corr[0] += hip_c     # FL hip
        corr[2] += abd_c     # FL abduction
        corr[6] += hip_c     # RL hip
        corr[8] += abd_c     # RL abduction
        # Right legs: FR(3), RR(9) hip; FR(5), RR(11) abduction
        corr[3] -= hip_c     # FR hip
        corr[5] -= abd_c     # FR abduction
        corr[9] -= hip_c     # RR hip
        corr[11] -= abd_c    # RR abduction
        
        return corr
    
    def get_stats(self) -> dict:
        """Stats for FLOG logging."""
        return dict(self.stats)
