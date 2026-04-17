"""
MH-FLOCKE — Coupling Reward Signal for Inter-Leg R-STDP (v0.6.0)
================================================================

Generates the reward signal that Mogli's `apply_coupling_rstdp()` consumes
to adapt inter-leg coupling weights. The goal is that over many gait
cycles, the spinal circuits learn the body's own symmetry — so that even
without any other correction mechanism, the CPG produces straight walking.

Biology
-------
Spinal commissural interneurons connect the four limbs (Kiehn 2006).
These connections are plastic and shaped during development: pups that
grow up with asymmetric bodies develop asymmetric coupling weights that
compensate. The teaching signal is proprioceptive and vestibular — the
spinal cord notices when walking is "good" (stable, straight, efficient)
and reinforces the coupling patterns that produced that state.

This is slower than cortical/colliculus steering learning but more
fundamental: it's the BODY LEARNING ITS OWN STRUCTURE.

Reward rule
-----------
After each gait cycle we have:
    drift       = mean yaw_rate over the cycle
    bias_norm   = abs(tectospinal_bias), how much external help we need
    forward_vel = how fast we moved

A stable, self-symmetric gait requires LESS tectospinal bias to walk
straight AND has LESS residual drift. So the reward signal is:

    reward = tanh(straightness_gain) - bias_norm * bias_penalty

Where straightness_gain rises when |drift| drops from the recent baseline,
and bias_norm penalizes reliance on the tectospinal crutch. The coupling
weights that were active during high-reward cycles get strengthened via
R-STDP; weights active during low-reward cycles get weakened.

This closes a beautiful loop:
    1. Tectospinal bias compensates body asymmetry QUICKLY (seconds)
    2. Straight walking with bias = reward → R-STDP strengthens useful
       couplings (minutes)
    3. As couplings internalize the compensation, the bias decays toward
       zero (because drift disappears without it)
    4. At convergence: the body walks straight with minimal bias AND
       minimal drift. The animal has "learned its body."

References
----------
- Kiehn 2006 — Locomotor circuits in the mammalian spinal cord
- Legrain et al. 2011 — Developmental plasticity of spinal networks
- Ijspeert 2008 — Central pattern generators as adaptive systems
"""

import numpy as np
from typing import Optional

COUPLING_REWARD_VERSION = "v0.6.2-waits-for-bias-convergence"


class CouplingRewardSignal:
    """Compute a reward signal for Mogli's inter-leg R-STDP.

    Usage:
        reward = CouplingRewardSignal()
        for step in loop:
            if cpg.new_cycle_completed():
                r = reward.compute(
                    drift=cpg._drift_estimate,
                    bias=tectospinal.get_bias(),
                    forward_vel=forward_velocity_estimate,
                )
                cpg.apply_coupling_rstdp(r)
    """

    VERSION = COUPLING_REWARD_VERSION

    def __init__(
        self,
        drift_tolerance: float = 0.15,
        bias_penalty: float = 1.5,
        min_forward_vel: float = 0.02,
        reward_clip: float = 0.5,
        warmup_cycles: int = 10,
        drift_gate: float = 0.35,
    ):
        """
        Args:
            drift_tolerance: Drifts below this magnitude are considered
                "fine" — produce positive reward. Above this, negative.
            bias_penalty: How strongly |tectospinal_bias| reduces reward.
                Default 1.5 means a bias of 0.3 roughly cancels a small
                positive drift reward.
            min_forward_vel: Below this, punish regardless of drift (a
                stationary animal isn't walking straight, it's just not
                walking).
            reward_clip: Hard limit on reward magnitude. Prevents one
                great or terrible cycle from dominating the learning.
            warmup_cycles: Don't emit rewards during warmup. Early cycles
                are noisy and would corrupt the learned couplings.
        """
        self.drift_tol = drift_tolerance
        self.bias_penalty = bias_penalty
        self.min_vel = min_forward_vel
        self.clip = reward_clip
        self.warmup = warmup_cycles
        # If the tectospinal bias adapter still sees |drift| above this
        # threshold, we freeze the R-STDP learner. It means the fast
        # adapter has not yet converged, and any reward we'd emit now
        # would reflect that transient rather than a genuine gait
        # signature.
        self.drift_gate = drift_gate

        self._prev_cycle = 0
        self._last_reward = 0.0
        self._gated_cycles = 0

    def compute(
        self,
        drift: float,
        bias: float,
        forward_vel: float,
        cycles_completed: int,
    ) -> Optional[float]:
        """Compute reward for the just-completed cycle. Returns None during
        warmup or on duplicate calls.
        """
        if cycles_completed == self._prev_cycle:
            return None
        self._prev_cycle = cycles_completed

        if cycles_completed <= self.warmup:
            return None

        # Gate: while drift is still large, the tectospinal layer is
        # likely still converging. R-STDP would pick up the transient
        # gait instability caused by the bias ramping up and learn
        # bad couplings. Return None until drift has settled.
        if abs(drift) > self.drift_gate:
            self._gated_cycles += 1
            return None

        # Term 1: straightness. Positive when drift is small, negative
        # when large. tanh gives smooth saturation at the extremes.
        drift_term = 1.0 - abs(drift) / self.drift_tol
        drift_term = np.tanh(drift_term)  # [-1, +1]

        # Term 2: bias penalty. Bias consumes energy and signals that
        # the compensation hasn't been internalized yet.
        bias_term = abs(bias) * self.bias_penalty

        # Term 3: forward-motion gate. If the animal isn't moving,
        # straightness is meaningless — punish regardless.
        if forward_vel < self.min_vel:
            # Negative even if drift happens to be zero — we don't want
            # to reward "stand still."
            reward = -0.3
        else:
            reward = drift_term - bias_term

        reward = float(np.clip(reward, -self.clip, self.clip))
        self._last_reward = reward
        return reward

    def stats(self) -> dict:
        return {
            'coupling_reward': self._last_reward,
            'coupling_reward_gated_cycles': self._gated_cycles,
        }

    def reset(self) -> None:
        self._prev_cycle = 0
        self._last_reward = 0.0
        self._gated_cycles = 0
