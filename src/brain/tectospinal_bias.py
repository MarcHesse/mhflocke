"""
MH-FLOCKE — Tectospinal Bias Adapter (v0.7.0)
==============================================

Mid-brain steering-bias calibration: learns a constant steering offset
that compensates for body asymmetry (e.g. leg loss, servo wear).

History
-------
v0.6.0-v0.6.2: Failed because steering had no authority (R²=0.03).
    Sign-probing produced 196 flips (noise-chasing).
v0.7.0: Rebuilt after steering fix (v0.3.3 step-length steering,
    R²=0.969, slope=+0.296). Simple integrator with empirically
    confirmed sign convention: bias -= lr * drift.

Biology
-------
Superior colliculus / reticulospinal tracts carry voluntary steering.
When the body drifts due to asymmetry (missing leg, worn servo),
this pathway learns a constant offset. Fast (seconds), strong
(uses the full steering channel), and specific (only compensates
the DC component, not gait oscillation).

Ref: Shadmehr & Mussa-Ivaldi 1994 — motor program recalibration
"""

import numpy as np

TECTOSPINAL_BIAS_VERSION = "v0.7.0-simple-integrator"


class TectospinalBias:
    """Learns a steering-bias offset from drift measurements.

    Sign convention (empirically confirmed on Freenove MJCF v0.3.3):
        drift > 0  (turning left)  → bias decreases → steers right
        drift < 0  (turning right) → bias increases → steers left
        Rule: bias -= lr * drift

    Usage:
        tb = TectospinalBias()
        for step in loop:
            if cpg_cycle_just_completed:
                tb.update(cpg._drift_estimate, cpg._cycles_completed)
            steering_out = user_steering + tb.get_bias()
            cpg.compute(steering=steering_out, ...)
    """

    VERSION = TECTOSPINAL_BIAS_VERSION

    def __init__(
        self,
        learning_rate: float = 0.05,
        decay_rate: float = 0.001,
        bias_max: float = 0.4,
        warmup_cycles: int = 3,
        drift_dead_band: float = 0.03,
    ):
        """
        Args:
            learning_rate: Fraction of drift applied per cycle.
                At lr=0.05 and drift=0.5 rad/s, bias changes by 0.025/cycle.
                Convergence in ~20 cycles for typical asymmetries.
            decay_rate: Gentle pull toward zero when drift is in dead band.
                Prevents stale bias persisting after recovery from injury.
            bias_max: Hard clamp. Leaves room for user steering on top
                (Mogli clamps total steering to ±0.6).
            warmup_cycles: Skip first N cycles (gait still settling).
            drift_dead_band: Ignore drifts below this (gait noise floor).
        """
        self.lr = learning_rate
        self.decay = decay_rate
        self.bias_max = bias_max
        self.warmup = warmup_cycles
        self.dead_band = drift_dead_band

        self._bias = 0.0
        self._prev_cycle = 0
        self._updates_applied = 0
        self._last_drift_seen = 0.0

    def update(self, drift_estimate: float, cycles_completed: int) -> None:
        """Consume one drift measurement per gait cycle.

        Args:
            drift_estimate: Mean yaw_rate over last cycle (rad/s).
            cycles_completed: CPG cycle counter (for dedup + warmup).
        """
        if cycles_completed == self._prev_cycle:
            return
        self._prev_cycle = cycles_completed
        self._last_drift_seen = drift_estimate

        if cycles_completed <= self.warmup:
            return

        if abs(drift_estimate) > self.dead_band:
            self._bias -= self.lr * drift_estimate
            self._updates_applied += 1
        else:
            self._bias *= (1.0 - self.decay)

        self._bias = float(np.clip(self._bias, -self.bias_max, self.bias_max))

    def get_bias(self) -> float:
        """Current learned offset. Add to user steering."""
        return self._bias

    def reset(self) -> None:
        self._bias = 0.0
        self._prev_cycle = 0
        self._updates_applied = 0
        self._last_drift_seen = 0.0

    def stats(self) -> dict:
        return {
            'tectospinal_bias': self._bias,
            'tectospinal_updates': self._updates_applied,
            'tectospinal_last_drift': self._last_drift_seen,
        }
