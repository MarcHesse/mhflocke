"""
MH-FLOCKE — Gait Quality Metrics v0.1.0
==========================================

Measures what the reward function cannot: is the dog actually WALKING
or just twitching its way forward?

A good quadruped gait has:
  - Rhythmic periodicity (autocorrelation peak in hip joints)
  - Adequate step height (knees lift clear of ground)
  - Body height near standing height (not crouching)
  - Smooth joint velocities (low jitter)
  - Proper foot contact alternation (stance/swing pattern)

These metrics serve two purposes:
  1. REWARD SHAPING: feed into training reward to prevent local optima
     like "twitch knees while crouching" which gets forward_vel reward
     but is not walking.
  2. METACOGNITION: the system can evaluate its own gait quality and
     trigger directed learning when quality is poor.

All computations use rolling buffers — no growing memory.
RPi cost: ~0.1ms per step (buffer updates + periodic analysis).

Ref: Hildebrand 1965 — symmetrical gaits of horses
Ref: Griffin et al. 2004 — biomechanics of quadruped locomotion
Ref: Fuchs & Goldner 1986 — gait quality assessment

Author: MH-FLOCKE Level 15 v0.7.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GAIT_QUALITY_VERSION = "v0.1.0"


@dataclass
class GaitQualityConfig:
    """Configuration for gait quality assessment."""

    # Buffer sizes (in simulation steps, not frames)
    joint_buffer_size: int = 400       # ~2 seconds at 200Hz — enough for 2 gait cycles
    analysis_interval: int = 200       # Analyze every N steps (~1 second)

    # Height
    standing_height: float = 0.12      # Freenove standing CoM height (meters)
    height_target_ratio: float = 0.90  # Target: 90% of standing height

    # Periodicity detection
    autocorr_min_lag: int = 20         # Minimum period: 20 steps = 0.1s = 10Hz (too fast)
    autocorr_max_lag: int = 200        # Maximum period: 200 steps = 1.0s = 1Hz (slow walk)
    periodicity_threshold: float = 0.3 # Minimum autocorrelation peak to count as periodic

    # Jitter
    jitter_good_threshold: float = 0.01   # Below this = smooth movement
    jitter_bad_threshold: float = 0.04    # Above this = twitching

    # Step height (knee joint amplitude)
    min_step_amplitude: float = 0.15   # Minimum knee swing (radians) for a real step

    # Foot contacts
    min_contact_ratio: float = 0.30    # Each active leg should touch ground >30% of time
    max_contact_ratio: float = 0.80    # Each active leg should lift >20% of time (swing phase)

    # Number of joints per leg
    joints_per_leg: int = 3

    # Leg layout: [FL, FR, RL, RR], each with [yaw, pitch, knee]
    n_legs: int = 4


class GaitQualityAnalyzer:
    """
    Rolling gait quality assessment from joint positions and body state.

    Usage:
        analyzer = GaitQualityAnalyzer(config)
        for step in training_loop:
            analyzer.update(joint_positions, height, foot_contacts)
            if step % config.analysis_interval == 0:
                metrics = analyzer.analyze()
                reward_bonus = analyzer.get_reward_components()
    """

    VERSION = GAIT_QUALITY_VERSION

    def __init__(self, config: GaitQualityConfig = None):
        self.config = config or GaitQualityConfig()
        c = self.config
        n_joints = c.n_legs * c.joints_per_leg

        # Rolling buffers (fixed size, overwrite oldest)
        self._joint_buf = np.zeros((c.joint_buffer_size, n_joints), dtype=np.float64)
        self._height_buf = np.zeros(c.joint_buffer_size, dtype=np.float64)
        self._foot_buf = np.zeros((c.joint_buffer_size, c.n_legs), dtype=np.float64)
        self._buf_idx = 0
        self._buf_filled = 0  # How many entries are valid

        self._step_count = 0

        # Latest analysis results
        self._metrics: Dict[str, float] = {}
        self._leg_names = ['FL', 'FR', 'RL', 'RR']
        self._joint_names = []
        for leg in self._leg_names:
            for joint in ['yaw', 'pitch', 'knee']:
                self._joint_names.append(f'{leg}_{joint}')

    def update(self, joint_positions: np.ndarray, height: float,
               foot_contacts: Optional[np.ndarray] = None) -> None:
        """Record one step of data. Call every simulation step.

        Args:
            joint_positions: Array of joint angles [n_legs * joints_per_leg].
                Order: FL_yaw, FL_pitch, FL_knee, FR_yaw, ...
            height: Center of mass height (meters).
            foot_contacts: Boolean array [n_legs] — True if foot touching ground.
                None if not available.
        """
        c = self.config
        idx = self._buf_idx

        n_j = min(len(joint_positions), c.n_legs * c.joints_per_leg)
        self._joint_buf[idx, :n_j] = joint_positions[:n_j]
        self._height_buf[idx] = height

        if foot_contacts is not None:
            n_f = min(len(foot_contacts), c.n_legs)
            self._foot_buf[idx, :n_f] = foot_contacts[:n_f].astype(np.float64)

        self._buf_idx = (idx + 1) % c.joint_buffer_size
        self._buf_filled = min(self._buf_filled + 1, c.joint_buffer_size)
        self._step_count += 1

    def analyze(self) -> Dict[str, float]:
        """Run full gait analysis on current buffer.

        Call periodically (every analysis_interval steps).
        Returns dict of metric name → value.
        """
        if self._buf_filled < 100:
            # Not enough data yet
            self._metrics = {'quality_score': 0.0, 'enough_data': 0}
            return self._metrics

        c = self.config
        m = {}

        # Get ordered data from ring buffer
        if self._buf_filled >= c.joint_buffer_size:
            # Buffer is full — reorder so oldest is first
            joints = np.roll(self._joint_buf, -self._buf_idx, axis=0)
            heights = np.roll(self._height_buf, -self._buf_idx)
            feet = np.roll(self._foot_buf, -self._buf_idx, axis=0)
        else:
            joints = self._joint_buf[:self._buf_filled]
            heights = self._height_buf[:self._buf_filled]
            feet = self._foot_buf[:self._buf_filled]

        n = len(joints)

        # ================================================================
        # 1. HEIGHT RATIO — how close to standing height
        # ================================================================
        mean_height = float(np.mean(heights))
        height_ratio = mean_height / max(c.standing_height, 0.01)
        m['height_ratio'] = height_ratio
        m['height_mean'] = mean_height

        # ================================================================
        # 2. PERIODICITY — autocorrelation of hip pitch joints
        # ================================================================
        # Hip pitch (index 1 per leg) is the main walking joint
        periodicities = []
        periods = []
        for leg_idx in range(c.n_legs):
            pitch_idx = leg_idx * c.joints_per_leg + 1  # pitch is joint[1]
            signal = joints[:, pitch_idx]
            period, strength = self._find_periodicity(signal)
            periodicities.append(strength)
            periods.append(period)
            m[f'{self._leg_names[leg_idx]}_periodicity'] = strength
            m[f'{self._leg_names[leg_idx]}_period'] = period

        m['periodicity_mean'] = float(np.mean(periodicities))
        m['period_mean'] = float(np.mean([p for p in periods if p > 0]) if any(p > 0 for p in periods) else 0)

        # ================================================================
        # 3. JITTER — smoothness of joint movement
        # ================================================================
        jitters = []
        for j in range(joints.shape[1]):
            vel = np.diff(joints[:, j])
            jitter = float(np.std(vel))
            jitters.append(jitter)
            m[f'{self._joint_names[j]}_jitter'] = jitter

        m['jitter_mean'] = float(np.mean(jitters))

        # Per-leg jitter (average of 3 joints per leg)
        for leg_idx in range(c.n_legs):
            base = leg_idx * c.joints_per_leg
            leg_jitter = float(np.mean(jitters[base:base + c.joints_per_leg]))
            m[f'{self._leg_names[leg_idx]}_jitter'] = leg_jitter

        # ================================================================
        # 4. STEP AMPLITUDE — are the legs actually stepping?
        # ================================================================
        for leg_idx in range(c.n_legs):
            knee_idx = leg_idx * c.joints_per_leg + 2  # knee is joint[2]
            pitch_idx = leg_idx * c.joints_per_leg + 1
            knee_amp = float(joints[:, knee_idx].max() - joints[:, knee_idx].min())
            pitch_amp = float(joints[:, pitch_idx].max() - joints[:, pitch_idx].min())
            m[f'{self._leg_names[leg_idx]}_knee_amp'] = knee_amp
            m[f'{self._leg_names[leg_idx]}_pitch_amp'] = pitch_amp

        # ================================================================
        # 5. FOOT CONTACT PATTERN — stance/swing alternation
        # ================================================================
        for leg_idx in range(c.n_legs):
            contact_ratio = float(np.mean(feet[:, leg_idx]))
            m[f'{self._leg_names[leg_idx]}_contact'] = contact_ratio

        # Support pattern: how many legs touch ground simultaneously
        support_counts = feet.sum(axis=1)
        m['support_mean'] = float(np.mean(support_counts))
        m['support_std'] = float(np.std(support_counts))

        # ================================================================
        # 6. COMPOSITE QUALITY SCORE — 0 (terrible) to 1 (perfect gait)
        # ================================================================
        score = self._compute_quality_score(m)
        m['quality_score'] = score
        m['enough_data'] = 1

        self._metrics = m
        return m

    def get_reward_components(self) -> Dict[str, float]:
        """Convert gait metrics into reward bonus/penalty components.

        Returns dict with named reward components. Sum these and add
        to the training reward.

        RPi-friendly: just reads cached metrics, no computation.
        """
        m = self._metrics
        if not m or m.get('enough_data', 0) == 0:
            return {'gait_reward': 0.0}

        c = self.config
        rewards = {}

        # HEIGHT BONUS: being near standing height
        hr = m.get('height_ratio', 0.0)
        if hr > c.height_target_ratio:
            rewards['height_bonus'] = (hr - c.height_target_ratio) * 10.0  # up to +1.0
        elif hr < 0.7:
            rewards['height_penalty'] = -(0.7 - hr) * 5.0  # penalty for crouching
        else:
            rewards['height_bonus'] = 0.0

        # PERIODICITY BONUS: rhythmic movement
        per = m.get('periodicity_mean', 0.0)
        if per > c.periodicity_threshold:
            rewards['periodicity_bonus'] = per * 2.0  # up to +2.0
        else:
            rewards['periodicity_penalty'] = -0.5  # no rhythm = penalty

        # JITTER PENALTY: twitching
        jit = m.get('jitter_mean', 0.0)
        if jit > c.jitter_bad_threshold:
            rewards['jitter_penalty'] = -(jit - c.jitter_bad_threshold) * 10.0
        elif jit < c.jitter_good_threshold:
            rewards['smoothness_bonus'] = 0.5  # smooth movement bonus

        # STEP AMPLITUDE: knees must actually lift
        # Only count active legs (amplitude > 0.01 in any joint)
        step_bonus = 0.0
        for leg_idx in range(c.n_legs):
            knee_amp = m.get(f'{self._leg_names[leg_idx]}_knee_amp', 0.0)
            pitch_amp = m.get(f'{self._leg_names[leg_idx]}_pitch_amp', 0.0)
            if pitch_amp > c.min_step_amplitude:
                step_bonus += 0.25  # each stepping leg gets bonus
        rewards['step_bonus'] = step_bonus

        # Total
        rewards['gait_reward'] = sum(rewards.values())
        return rewards

    def get_metrics(self) -> Dict[str, float]:
        """Return latest cached metrics."""
        return self._metrics.copy()

    def get_quality_score(self) -> float:
        """Return composite quality score (0-1)."""
        return self._metrics.get('quality_score', 0.0)

    def _find_periodicity(self, signal: np.ndarray) -> tuple:
        """Find periodicity in a 1D signal using autocorrelation.

        Returns (period_in_steps, strength).
        period=0 and strength=0 if no periodic signal found.
        """
        c = self.config
        x = signal - np.mean(signal)
        if np.std(x) < 1e-6:
            return (0, 0.0)

        # Autocorrelation via numpy correlate (faster than FFT for small N)
        max_lag = min(c.autocorr_max_lag, len(x) // 2)
        ac = np.correlate(x, x, 'full')
        ac = ac[len(x) - 1:]  # Take positive lags only
        ac = ac / (ac[0] + 1e-10)  # Normalize

        # Find first peak after min_lag
        best_lag = 0
        best_val = 0.0
        for i in range(c.autocorr_min_lag, min(max_lag, len(ac) - 1)):
            if ac[i - 1] < ac[i] > ac[i + 1] and ac[i] > best_val:
                best_lag = i
                best_val = float(ac[i])
                break  # First peak is the fundamental period

        if best_val < c.periodicity_threshold:
            return (0, 0.0)

        return (best_lag, best_val)

    def _compute_quality_score(self, m: Dict[str, float]) -> float:
        """Compute composite gait quality score from individual metrics.

        Score components (all normalized to 0-1):
          - Height: 0 at 50%, 1 at 95%+ of standing
          - Periodicity: 0 at 0, 1 at strong rhythm
          - Smoothness: 0 at high jitter, 1 at smooth
          - Stepping: 0 if no step amplitude, 1 if all legs step

        Equal weights for now — can be tuned later.
        """
        c = self.config

        # Height score: linear 0.5→0, 0.95→1
        hr = m.get('height_ratio', 0.0)
        height_score = np.clip((hr - 0.5) / 0.45, 0.0, 1.0)

        # Periodicity score: 0→0, 0.7→1
        per = m.get('periodicity_mean', 0.0)
        period_score = np.clip(per / 0.7, 0.0, 1.0)

        # Smoothness score: inverse jitter, 0.05→0, 0.005→1
        jit = m.get('jitter_mean', 0.05)
        smooth_score = np.clip(1.0 - (jit - 0.005) / 0.045, 0.0, 1.0)

        # Stepping score: how many legs have adequate pitch amplitude
        stepping = 0.0
        active_legs = 0
        for leg_idx in range(c.n_legs):
            pitch_amp = m.get(f'{self._leg_names[leg_idx]}_pitch_amp', 0.0)
            knee_amp = m.get(f'{self._leg_names[leg_idx]}_knee_amp', 0.0)
            total_amp = pitch_amp + knee_amp
            if total_amp > 0.05:  # Leg is trying to move
                active_legs += 1
                if pitch_amp > c.min_step_amplitude:
                    stepping += 1.0
        step_score = stepping / max(active_legs, 1)

        # Weighted composite
        score = (0.25 * height_score +
                 0.30 * period_score +
                 0.25 * smooth_score +
                 0.20 * step_score)

        return float(np.clip(score, 0.0, 1.0))

    def stats(self) -> Dict:
        """Compact stats for logging/FLOG."""
        m = self._metrics
        return {
            'gait_quality': m.get('quality_score', 0.0),
            'gait_height_ratio': m.get('height_ratio', 0.0),
            'gait_periodicity': m.get('periodicity_mean', 0.0),
            'gait_jitter': m.get('jitter_mean', 0.0),
            'gait_support': m.get('support_mean', 0.0),
            'gait_period': m.get('period_mean', 0.0),
        }
