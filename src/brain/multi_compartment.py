"""
MH-FLOCKE — Multi-Compartment Neurons v0.4.1
============================================
Soma-basal-apical compartment model with burst mode.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MultiCompartmentConfig:
    """Configuration for Multi-Compartment Neurons."""
    n_neurons: int = 500_000

    # Compartment Time Constants
    tau_soma: float = 20.0
    tau_basal: float = 10.0
    tau_apical: float = 50.0

    # Threshold
    v_threshold: float = 1.0
    v_rest: float = 0.0
    v_reset: float = 0.0

    # Coupling
    basal_to_soma_weight: float = 1.0
    apical_gain_factor: float = 0.5

    # Burst
    burst_threshold: float = 0.3
    burst_multiplier: float = 2.0

    # Adaptive E/I
    ei_adaptation_rate: float = 0.001
    target_firing_rate: float = 0.05

    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


class MultiCompartmentLayer:
    """
    Multi-Compartment Neuron Layer.

    Basal (Bottom-Up) → additively drives Soma
    Apical (Top-Down) → multiplicatively modulates Gain
    Soma → Spike generation
    """

    def __init__(self, config: MultiCompartmentConfig):
        self.config = config
        n = config.n_neurons
        self.device = config.device
        self.dtype = config.dtype

        # Compartment potentials
        self.V_soma = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.V_basal = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.V_apical = torch.zeros(n, device=self.device, dtype=self.dtype)

        # Persistent apical context (Global Workspace Broadcast)
        self._apical_context = torch.zeros(n, device=self.device, dtype=self.dtype)

        # Burst tracking
        self._burst_mask = torch.zeros(n, device=self.device, dtype=torch.bool)

        # Refractory
        self._refractory = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._refractory_period = 2.0

        # E/I Balance Tracking
        self._firing_rate_ema = torch.ones(n, device=self.device, dtype=self.dtype) * config.target_firing_rate
        self._ei_gain = torch.ones(n, device=self.device, dtype=self.dtype)

    def step(self, basal_input: torch.Tensor,
             apical_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One simulation step.

        Returns:
            spikes: Bool tensor [n_neurons]
        """
        cfg = self.config

        # 1. Basal dendrite: leaky integration
        self.V_basal += (basal_input - self.V_basal) / cfg.tau_basal

        # 2. Apical dendrite: leaky integration (slower)
        apical_total = self._apical_context.clone()
        if apical_input is not None:
            apical_total = apical_total + apical_input
        self.V_apical += (apical_total - self.V_apical) / cfg.tau_apical

        # 3. Apical gain: multiplicative modulation
        gain = 1.0 + cfg.apical_gain_factor * torch.sigmoid(self.V_apical)

        # 4. Soma: leaky integration with gain-modulated basal input
        basal_drive = cfg.basal_to_soma_weight * self.V_basal * gain * self._ei_gain
        self.V_soma += (basal_drive - (self.V_soma - cfg.v_rest)) / cfg.tau_soma

        # Refractory: neurons in refractory period cannot fire
        refractory_mask = self._refractory > 0
        self.V_soma[refractory_mask] = cfg.v_reset
        self._refractory = torch.clamp(self._refractory - 1.0, min=0.0)

        # 5. Spike generation
        spikes = self.V_soma >= cfg.v_threshold

        # 6. Burst mode: strong apical input amplifies spikes
        self._burst_mask = self.V_apical > cfg.burst_threshold

        # Reset after spike
        self.V_soma[spikes] = cfg.v_reset
        self._refractory[spikes] = self._refractory_period

        # Update firing rate EMA
        self._firing_rate_ema = 0.99 * self._firing_rate_ema + 0.01 * spikes.float()

        return spikes

    def get_compartment_state(self) -> Dict[str, torch.Tensor]:
        """Return all compartment potentials."""
        return {
            'soma': self.V_soma.clone(),
            'basal': self.V_basal.clone(),
            'apical': self.V_apical.clone(),
        }

    def get_burst_mask(self) -> torch.Tensor:
        """Which neurons are in burst mode?"""
        return self._burst_mask.clone()

    def set_apical_context(self, context: torch.Tensor):
        """Set persistent apical context (Global Workspace Broadcast)."""
        self._apical_context = context.to(device=self.device, dtype=self.dtype)

    def adapt_ei_balance(self):
        """
        Adaptive E/I balance per neuron.
        Neurons firing too much → gain down.
        Neurons firing too little → gain up.
        """
        rate = self.config.ei_adaptation_rate
        target = self.config.target_firing_rate

        error = self._firing_rate_ema - target
        self._ei_gain -= rate * error
        self._ei_gain = torch.clamp(self._ei_gain, 0.1, 5.0)

    def reset(self):
        """Reset all state (episode boundary)."""
        n = self.config.n_neurons
        self.V_soma.zero_()
        self.V_basal.zero_()
        self.V_apical.zero_()
        self._burst_mask.zero_()
        self._refractory.zero_()


# ============================================================================
# PURKINJE CELL COMPARTMENT LAYER
# ============================================================================

@dataclass
class PurkinjeConfig:
    """Configuration for Purkinje cell multi-compartment layer.

    Biologically: PkC have the most complex dendritic tree of any neuron.
    - Apical dendrite: receives ~200,000 parallel fiber (PF) inputs from GrC
    - Basal dendrite: receives 1 climbing fiber (CF) from Inferior Olive
    - Soma: generates output spikes → inhibits DCN

    The apical dendrite is the main learning site:
    - PF→PkC synapses undergo LTD when CF is active (error)
    - PF→PkC synapses undergo LTP when CF is silent (consolidate)
    - CF causes dendritic calcium spike → gates LTD
    """
    n_neurons: int = 42          # 2 per joint (push/pull), matches CerebellarConfig

    # Compartment time constants (PkC biology)
    tau_soma: float = 15.0       # PkC soma: medium speed
    tau_apical: float = 80.0     # Apical dendrite: slow integration of many PF inputs
    tau_basal: float = 5.0       # Basal: fast CF response (single fiber, all-or-nothing)

    # Thresholds
    v_threshold: float = 1.0
    v_rest: float = 0.0
    v_reset: float = 0.0

    # Dendritic calcium spike (CF-triggered)
    calcium_decay: float = 0.92      # Slow decay — calcium persists ~12 steps
    calcium_spike_mag: float = 1.0   # How strongly CF activates calcium

    # Apical → Soma coupling
    apical_to_soma_weight: float = 0.8   # PF input drives PkC output
    basal_modulation: float = 0.3        # CF modulates (not drives) soma

    # Burst: strong CF + PF coincidence → complex spike
    complex_spike_threshold: float = 0.5

    device: str = 'cpu'
    dtype: torch.dtype = torch.float32


class PurkinjeCompartmentLayer:
    """
    Multi-compartment Purkinje cell layer for cerebellar learning.

    Architecture per neuron:
        Parallel Fibers (GrC) ──→ [Apical Dendrite] ──┐
                                                       ├──→ [Soma] ──→ DCN (inhibitory)
        Climbing Fiber (IO)  ──→ [Basal Dendrite]  ──┘
                                       │
                                 Calcium Spike → gates LTD in apical

    Key biological properties:
    - Apical dendrite integrates thousands of weak PF inputs (slow tau)
    - Basal dendrite receives single strong CF input (fast tau)
    - CF triggers dendritic calcium spike → opens LTD window
    - Simple spikes: regular output driven by PF (50-100 Hz)
    - Complex spikes: CF-triggered bursts (1-2 Hz), signal error
    """

    def __init__(self, config: PurkinjeConfig = None):
        self.config = config or PurkinjeConfig()
        cfg = self.config
        n = cfg.n_neurons
        self.device = cfg.device
        self.dtype = cfg.dtype

        # Compartment potentials
        self.V_soma = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.V_apical = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.V_basal = torch.zeros(n, device=self.device, dtype=self.dtype)

        # Dendritic calcium (CF-triggered, gates LTD)
        self.calcium = torch.zeros(n, device=self.device, dtype=self.dtype)

        # Complex spike flag (CF + PF coincidence)
        self.complex_spike = torch.zeros(n, device=self.device, dtype=torch.bool)

        # Simple spike output
        self.spikes = torch.zeros(n, device=self.device, dtype=torch.bool)

        # Refractory
        self._refractory = torch.zeros(n, device=self.device, dtype=self.dtype)

        # Output activity (continuous, for DCN input)
        self.activity = torch.zeros(n, device=self.device, dtype=self.dtype)

    def step(self,
             pf_input: torch.Tensor,
             cf_input: torch.Tensor) -> torch.Tensor:
        """
        One simulation step.

        Args:
            pf_input: [n_neurons] — parallel fiber input (from PF→PkC weights × GrC rates)
            cf_input: [n_neurons] — climbing fiber signal (0.0 = silent, >0 = error)

        Returns:
            spikes: Bool tensor [n_neurons] — simple spikes (regular output)
        """
        cfg = self.config

        # 1. Apical dendrite: slow integration of parallel fiber input
        #    This is where learning happens — PF→PkC weights determine this signal
        self.V_apical += (pf_input - self.V_apical) / cfg.tau_apical

        # 2. Basal dendrite: fast CF response
        self.V_basal += (cf_input - self.V_basal) / cfg.tau_basal

        # 3. CF triggers dendritic calcium spike
        cf_active = cf_input > 0.01
        self.calcium = (self.calcium * cfg.calcium_decay +
                        cf_active.float() * cfg.calcium_spike_mag)
        self.calcium = self.calcium.clamp(0.0, 2.0)

        # 4. Soma: apical drives output, basal modulates
        #    PkC fires simple spikes driven by PF input
        #    CF doesn't directly drive soma — it modulates via calcium
        soma_drive = (cfg.apical_to_soma_weight * self.V_apical +
                      cfg.basal_modulation * self.V_basal)
        self.V_soma += (soma_drive - (self.V_soma - cfg.v_rest)) / cfg.tau_soma

        # Refractory
        refr_mask = self._refractory > 0
        self.V_soma[refr_mask] = cfg.v_reset
        self._refractory = (self._refractory - 1.0).clamp(min=0.0)

        # 5. Simple spikes
        self.spikes = self.V_soma >= cfg.v_threshold
        self.V_soma[self.spikes] = cfg.v_reset
        self._refractory[self.spikes] = 2.0

        # 6. Complex spikes: CF + strong PF coincidence (rare, ~1-2 Hz)
        self.complex_spike = (cf_active &
                              (self.V_apical > cfg.complex_spike_threshold))

        # 7. Continuous activity for DCN (smoothed spike rate)
        self.activity = 0.9 * self.activity + 0.1 * self.spikes.float()

        return self.spikes

    def get_ltd_gate(self) -> torch.Tensor:
        """
        Returns calcium level — gates LTD in PF→PkC synapses.
        High calcium = CF was recently active = error present = LTD window open.

        Used by CerebellarLearning to modulate weight updates:
            dw = -ltd_rate * ltd_gate * pf_eligibility  (LTD)
        """
        return self.calcium.clone()

    def get_compartment_state(self) -> Dict[str, torch.Tensor]:
        """Return all compartment potentials for visualization."""
        return {
            'soma': self.V_soma.clone(),
            'apical': self.V_apical.clone(),
            'basal': self.V_basal.clone(),
            'calcium': self.calcium.clone(),
            'activity': self.activity.clone(),
            'complex_spike': self.complex_spike.clone(),
        }

    def reset(self):
        """Reset all state (episode boundary)."""
        self.V_soma.zero_()
        self.V_apical.zero_()
        self.V_basal.zero_()
        self.calcium.zero_()
        self.complex_spike.zero_()
        self.spikes.zero_()
        self.activity.zero_()
        self._refractory.zero_()

    def state_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            'V_soma': self.V_soma.cpu(),
            'V_apical': self.V_apical.cpu(),
            'V_basal': self.V_basal.cpu(),
            'calcium': self.calcium.cpu(),
            'activity': self.activity.cpu(),
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        for key in ['V_soma', 'V_apical', 'V_basal', 'calcium', 'activity']:
            if key in state:
                getattr(self, key).copy_(state[key].to(self.device))
