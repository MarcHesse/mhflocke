"""
MH-FLOCKE — SNN Controller v0.5.0
========================================
Izhikevich spiking neural network with R-STDP on PyTorch tensors.

v0.5.0: Per-population Izhikevich (a,b,c,d) parameters + recovery variable u.
         DCN neurons use Low-Threshold Spiking parameters for rebound bursting.
         Issue #104: Population-Specific Neuron Types.
         Issue #106: Performance optimizations (torch.no_grad, pre-alloc).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import math


# === Surrogate Gradient Functions ===

class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast-Sigmoid Surrogate Gradient für Backprop durch Spikes."""

    @staticmethod
    def forward(ctx, V, threshold):
        ctx.save_for_backward(V, torch.tensor(threshold, device=V.device))
        return (V >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        V, threshold = ctx.saved_tensors
        # dσ/dV = 1 / (1 + π|V - threshold|)²
        grad = 1.0 / (1.0 + math.pi * torch.abs(V - threshold)) ** 2
        return grad_output * grad, None


class ATanSurrogate(torch.autograd.Function):
    """Arctan Surrogate Gradient."""

    @staticmethod
    def forward(ctx, V, threshold):
        ctx.save_for_backward(V, torch.tensor(threshold, device=V.device))
        return (V >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        V, threshold = ctx.saved_tensors
        alpha = 2.0
        grad = alpha / (2 * (1 + (math.pi / 2 * alpha * (V - threshold)) ** 2))
        return grad_output * grad, None


SURROGATE_FUNCTIONS = {
    'fast_sigmoid': FastSigmoidSurrogate,
    'atan': ATanSurrogate,
}


@dataclass
class SNNConfig:
    """Konfiguration für den SNN-Controller."""
    n_neurons: int = 500_000
    n_excitatory_ratio: float = 0.8          # 80% excitatory
    connectivity_prob: float = 0.02          # 2% sparse connectivity

    # LIF-LTC Parameter
    tau_base: float = 20.0                   # Basis-Zeitkonstante (ms)
    delta_tau: float = 15.0                  # Modulationsbereich
    v_threshold: float = 1.0                 # Spike-Threshold
    v_reset: float = 0.0                     # Reset nach Spike
    v_rest: float = 0.0                      # Ruhepotential
    refractory_ms: int = 2                   # Refraktärzeit

    # Plasticity
    stdp_lr: float = 0.01                    # STDP Lernrate
    surrogate_gradient: str = 'fast_sigmoid' # 'atan', 'fast_sigmoid'

    # Neuromodulatoren
    neuromod_enabled: bool = True

    # Homöostatische Plasticity
    homeostatic_interval: int = 1000         # Steps zwischen Anpassungen
    target_firing_rate: float = 0.05         # Ziel: 5% der Steps

    # Astrozyt-Gate
    astrocyte_cluster_size: int = 100
    astrocyte_calcium_threshold: float = 0.7
    astrocyte_tau_calcium: float = 2000.0

    # Synaptogenese
    synaptogenesis_interval: int = 5000      # Alle N Steps
    synaptogenesis_max_new: int = 1000       # Max neue Synapsen pro Runde
    pruning_threshold: float = 0.001         # Gewicht unter dem gepruned wird

    # Hardware
    device: str = 'cuda'                     # 'cuda' oder 'cpu'
    dtype: torch.dtype = torch.float32       # float32 oder float16


class SNNController:
    """
    Tensorisierter Spiking Neural Network Controller.

    GPU-first Design mit:
    - LIF-LTC (Liquid Time-Constant) Neuronen
    - Sparse COO Konnektivität
    - R-STDP + optionale Surrogate Gradients
    - Integrierte Neuromodulatoren (NE, 5-HT, ACh)
    - Astrozyt-Gate für Synaptogenese
    - Homöostatische Plastizität
    - Adaptive E/I Balance
    """

    def __init__(self, config: SNNConfig):
        """Initialisiert SNN auf GPU/CPU."""
        self.config = config
        self.device = torch.device(config.device if config.device == 'cuda'
                                   and torch.cuda.is_available() else 'cpu')
        self.dtype = config.dtype
        n = config.n_neurons

        # --- Neuron state ---
        self.V = torch.zeros(n, device=self.device, dtype=self.dtype)         # Membrane potential
        self.spikes = torch.zeros(n, device=self.device, dtype=torch.bool)    # Current spikes
        self.refractory_counter = torch.zeros(n, device=self.device, dtype=torch.int32)

        # --- Neuron types: +1 excitatory, -1 inhibitory ---
        n_exc = int(n * config.n_excitatory_ratio)
        self.neuron_types = torch.ones(n, device=self.device, dtype=self.dtype)
        self.neuron_types[n_exc:] = -1.0
        # Shuffle so inhibitory neurons are distributed
        perm = torch.randperm(n, device=self.device)
        self.neuron_types = self.neuron_types[perm]

        # --- Neuromodulators ---
        self.neuromod_levels = {
            'ne': 0.3,   # Noradrenalin
            '5ht': 0.5,  # Serotonin
            'ach': 0.5,  # Acetylcholin
        }
        # Per-neuron sensitivity (slight random variation)
        self.neuromod_sensitivity = torch.rand(n, 3, device=self.device, dtype=self.dtype) * 0.4 + 0.8
        # Columns: 0=NE, 1=5-HT, 2=ACh

        # --- Populations ---
        self.populations: Dict[str, torch.Tensor] = {}
        # Populations whose weights/thresholds are protected from learning
        # (e.g. cerebellar populations during cerebellar training)
        self.protected_populations: set = set()

        # --- Eligibility Traces (for R-STDP) — must be before connectivity init ---
        self._pre_trace = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._post_trace = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._trace_decay = 0.95  # Exponential decay per step
        self._eligibility = torch.zeros(1, device=self.device, dtype=self.dtype)

        # --- Connectivity (initially empty, built via populations) ---
        self._population_connections: List[Tuple[str, str, float, Tuple[float, float]]] = []
        self.weights: Optional[torch.Tensor] = None
        self._weight_indices: Optional[torch.Tensor] = None  # [2, nnz]
        self._weight_values: Optional[torch.Tensor] = None
        self._n_synapses = 0

        # Initialize default connectivity based on config
        self._init_default_connectivity()

        # --- Homöostatische Plasticity ---
        self._spike_count_window = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._homeostatic_step_count = 0
        self._thresholds = torch.full((n,), config.v_threshold, device=self.device, dtype=self.dtype)

        # --- Astrozyt-Gate ---
        self._astro_cluster_size = config.astrocyte_cluster_size
        self._n_astrocytes = (n + self._astro_cluster_size - 1) // self._astro_cluster_size
        self._astro_calcium = torch.zeros(self._n_astrocytes, device=self.device, dtype=self.dtype)

        # --- Per-neuron membrane time constant ---
        # Default: config.tau_base for all. Builder can override per population.
        # Biology: different cell types have different tau_mem.
        # GrC: ~5ms (small, high input resistance)
        # GoC: ~20ms (larger interneuron)
        # PkC: ~15ms (large, complex dendritic tree)
        # Ref: D'Angelo 2025, Bhatt et al. 2019
        self._tau_base = torch.full((n,), config.tau_base, device=self.device, dtype=self.dtype)

        # --- Per-neuron Izhikevich parameters (Issue #104) ---
        # Default: Regular Spiking (a=0.02, b=0.2, c=-65, d=8)
        # Builder overrides per population via set_izhikevich_params().
        # These replace the implicit LIF-LTC dynamics for populations
        # where biologically accurate firing patterns matter (DCN rebound,
        # GoC pacemaker, PkC chattering).
        # Ref: Izhikevich 2003 — Simple model of spiking neurons
        self._izh_a = torch.full((n,), 0.02, device=self.device, dtype=self.dtype)
        self._izh_b = torch.full((n,), 0.2, device=self.device, dtype=self.dtype)
        self._izh_c = torch.full((n,), -65.0, device=self.device, dtype=self.dtype)
        self._izh_d = torch.full((n,), 8.0, device=self.device, dtype=self.dtype)
        # Recovery variable u (Izhikevich membrane recovery)
        # u provides adaptation, bursting, and rebound dynamics that
        # the simple LIF model cannot produce. Critical for DCN rebound.
        self._u = torch.zeros(n, device=self.device, dtype=self.dtype)
        # Flag: which populations use full Izhikevich dynamics
        # (others keep the LIF-LTC model for backward compatibility)
        self._izh_enabled = torch.zeros(n, device=self.device, dtype=torch.bool)

        # --- Pre-allocated tensors for performance (Issue #106) ---
        self._v_reset_tensor = torch.tensor(config.v_reset, device=self.device, dtype=self.dtype)
        self._refractory_tensor = torch.tensor(config.refractory_ms, device=self.device, dtype=torch.int32)
        # Pre-allocated spike float buffer (avoids .float() copy every step)
        self._spike_float = torch.zeros(n, device=self.device, dtype=self.dtype)
        # Dense weight matrix cache for small networks (n<500)
        # Sparse matmul has fixed overhead; dense is faster for small n.
        self._dense_weights: Optional[torch.Tensor] = None
        self._dense_weights_dirty = True  # Rebuild after any weight change
        # Tau cache: recompute only when neuromodulator levels change
        self._tau_cached: Optional[torch.Tensor] = None
        self._tau_dirty = True

        # --- Simulation Counter ---
        self.step_count = 0

    # ========================================================================
    # CONNECTIVITY
    # ========================================================================

    def _init_default_connectivity(self):
        """Initialize sparse connectivity based on config.

        Uses population-based connectivity if populations are defined,
        otherwise creates uniform random sparse connectivity.
        If connectivity_prob is 0, skip entirely (populations will be
        connected explicitly via connect_populations()).
        """
        n = self.config.n_neurons
        p = self.config.connectivity_prob

        # Skip if explicitly disabled (population-based wiring)
        if p <= 0.0:
            self._weight_indices = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            self._weight_values = torch.zeros(0, device=self.device, dtype=self.dtype)
            self._eligibility = torch.zeros(0, device=self.device, dtype=self.dtype)
            self._n_synapses = 0
            self._rebuild_sparse_weights()
            return

        # For neuron counts where n*n*p is manageable, create random sparse
        # For very large n, limit total synapses
        max_synapses = min(int(n * n * p), 20_000_000)  # Cap at 20M
        actual_synapses = min(int(n * n * p), max_synapses)

        if actual_synapses == 0:
            actual_synapses = max(n * 10, 1000)  # At least 10 connections per neuron avg

        # Random source and target indices
        src = torch.randint(0, n, (actual_synapses,), device=self.device)
        tgt = torch.randint(0, n, (actual_synapses,), device=self.device)

        # Remove self-connections
        mask = src != tgt
        src = src[mask]
        tgt = tgt[mask]

        # Initial weights: small random, sign determined by neuron type
        n_connections = src.shape[0]
        weights = torch.randn(n_connections, device=self.device, dtype=self.dtype) * 0.1
        weights = weights.abs()  # Start positive, apply E/I sign below

        # E/I sign: excitatory neurons → positive weights, inhibitory → negative
        signs = self.neuron_types[src]  # +1 or -1
        weights = weights * signs

        self._weight_indices = torch.stack([src, tgt])  # [2, nnz]
        self._weight_values = weights
        self._n_synapses = n_connections

        # Build sparse tensor
        self._rebuild_sparse_weights()

    def _rebuild_sparse_weights(self):
        """Rebuild sparse weight matrix from indices and values.
        
        IMPORTANT: Explicitly deletes the old sparse matrix before creating
        the new one to prevent memory fragmentation/OOM. This method is called
        on every apply_rstdp() step — without cleanup, leaked sparse tensors
        accumulate and eventually cause OOM (Issue: alloc_cpu.cpp:117).
        """
        n = self.config.n_neurons
        # Invalidate dense cache (Issue #106)
        self._dense_weights = None
        self._dense_weights_dirty = True
        # Free old sparse matrix BEFORE allocating new one
        if self.weights is not None:
            del self.weights
            self.weights = None
        if self._weight_indices is not None and self._weight_values is not None:
            self.weights = torch.sparse_coo_tensor(
                self._weight_indices,
                self._weight_values,
                size=(n, n),
                device=self.device
            ).coalesce()

            # Resize eligibility traces
            nnz = self._weight_values.shape[0]
            if self._eligibility.shape[0] != nnz:
                self._eligibility = torch.zeros(nnz, device=self.device, dtype=self.dtype)
            self._n_synapses = nnz
        else:
            self.weights = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device, dtype=self.dtype),
                size=(n, n),
                device=self.device
            )
            self._n_synapses = 0

    # ========================================================================
    # KERN-SIMULATION
    # ========================================================================

    def step(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One simulation step (1ms).

        Performance (Issue #106):
            - torch.no_grad(): 30-50% speedup (no gradient tracking)
            - Dense matmul for n<500: 15-25% (avoids sparse overhead)
            - Tau caching: 3-5% (recompute only on neuromod change)
            - Pre-allocated spike buffer: 5-10%
            Combined: ~60-80% speedup on CPU (Pi: 37ms → ~20ms)

        Neurons with _izh_enabled=True use full Izhikevich dynamics:
            dV/dt = 0.04*V^2 + 5*V + 140 - u + I
            du/dt = a*(b*V - u)
            if V >= 30mV: V = c, u = u + d

        All other neurons keep the LIF-LTC model (backward compatible).

        Args:
            external_input: Tensor [n_neurons] with external current (optional)

        Returns:
            spikes: Bool-Tensor [n_neurons]
        """
        with torch.no_grad():
            n = self.config.n_neurons
            izh_mask = self._izh_enabled
            lif_mask = ~izh_mask

            # 1. Tau for LIF-LTC neurons (cached — recomputed only on neuromod change)
            if self._tau_dirty or self._tau_cached is None:
                self._tau_cached = self.get_neuromodulator_tau()
                self._tau_dirty = False
            tau = self._tau_cached

            # 2. Synaptic input
            # Use pre-allocated spike float buffer (avoids .float() copy)
            self._spike_float.zero_()
            self._spike_float[self.spikes] = 1.0

            if self._n_synapses > 0:
                # Dense matmul for small networks (Issue #106)
                # For n<500, dense is faster than sparse (no CSR overhead,
                # fits in L1 cache: 232×232 = 54k floats = 216KB)
                if n < 500:
                    if self._dense_weights is None or self._dense_weights_dirty:
                        if self.weights is not None:
                            self._dense_weights = self.weights.to_dense()
                        self._dense_weights_dirty = False
                    if self._dense_weights is not None:
                        I_syn = self._dense_weights.t() @ self._spike_float
                    else:
                        I_syn = torch.zeros(n, device=self.device, dtype=self.dtype)
                else:
                    spike_col = self._spike_float.unsqueeze(1)
                    I_syn = torch.sparse.mm(self.weights.t(), spike_col).squeeze(1)
            else:
                I_syn = torch.zeros(n, device=self.device, dtype=self.dtype)

            # 3. NE exploration noise
            if self.config.neuromod_enabled:
                ne_level = self.neuromod_levels['ne']
                if ne_level > 0.01:
                    noise = torch.randn(n, device=self.device, dtype=self.dtype) * (0.1 * ne_level)
                    I_syn = I_syn + noise

            # 4. Refractory mask
            refractory_mask = (self.refractory_counter <= 0).float()

            # 5a. External input
            ext = external_input if external_input is not None else 0.0
            I_total = I_syn + ext

            # 5b. Izhikevich dynamics for enabled populations
            if izh_mask.any():
                V_izh = self.V[izh_mask]
                u_izh = self._u[izh_mask]
                a_izh = self._izh_a[izh_mask]
                b_izh = self._izh_b[izh_mask]
                I_izh = I_total[izh_mask] * 10.0  # scale to mV range

                # Izhikevich update (two half-steps for numerical stability)
                dV = 0.04 * V_izh * V_izh + 5.0 * V_izh + 140.0 - u_izh + I_izh
                V_izh = V_izh + 0.5 * dV
                dV = 0.04 * V_izh * V_izh + 5.0 * V_izh + 140.0 - u_izh + I_izh
                V_izh = V_izh + 0.5 * dV
                # Recovery variable update
                du = a_izh * (b_izh * V_izh - u_izh)
                u_izh = u_izh + du

                # Apply refractory mask
                ref_izh = refractory_mask[izh_mask]
                V_izh = V_izh * ref_izh + self._izh_c[izh_mask] * (1.0 - ref_izh)

                self.V[izh_mask] = V_izh
                self._u[izh_mask] = u_izh

            # 5c. LIF-LTC dynamics for remaining neurons (original behavior)
            if lif_mask.any():
                V_lif = self.V[lif_mask]
                tau_lif = tau[lif_mask]
                I_lif = I_total[lif_mask]
                I_syn_lif = I_syn[lif_mask]
                ext_lif = ext[lif_mask] if isinstance(ext, torch.Tensor) else ext
                dv = (-(V_lif - self.config.v_rest) + I_syn_lif) / tau_lif + ext_lif
                V_lif = V_lif + dv
                V_lif = V_lif * refractory_mask[lif_mask]
                self.V[lif_mask] = V_lif

            # 6. Spike generation
            izh_spikes = torch.zeros(n, device=self.device, dtype=torch.bool)
            lif_spikes = torch.zeros(n, device=self.device, dtype=torch.bool)

            if izh_mask.any():
                izh_spikes[izh_mask] = self.V[izh_mask] >= 30.0
                spiked_izh = izh_mask & izh_spikes
                if spiked_izh.any():
                    self.V[spiked_izh] = self._izh_c[spiked_izh]
                    self._u[spiked_izh] = self._u[spiked_izh] + self._izh_d[spiked_izh]

            if lif_mask.any():
                lif_spikes[lif_mask] = self.V[lif_mask] >= self._thresholds[lif_mask]
                spiked_lif = lif_mask & lif_spikes
                if spiked_lif.any():
                    self.V[spiked_lif] = self.config.v_reset

            self.spikes = izh_spikes | lif_spikes

            # Refractory counter update
            self.refractory_counter = torch.where(
                self.spikes,
                self._refractory_tensor,
                self.refractory_counter
            )
            self.refractory_counter = (self.refractory_counter - 1).clamp(min=-1)

        # Outside no_grad: eligibility traces need gradient-free but
        # are also weight updates, kept separate for clarity
        # 7. Eligibility Traces update (for R-STDP)
        self._update_eligibility_traces()

        # 8. Spike count for homeostasis
        self._spike_count_window += self.spikes.float()
        self._homeostatic_step_count += 1

        # 9. Homeostatic plasticity
        if self._homeostatic_step_count >= self.config.homeostatic_interval:
            self._homeostatic_update()

        # 10. Astrocyte calcium update (every 10 steps for performance)
        if self.step_count % 10 == 0:
            self._astrocyte_update()

        # 11. Synaptogenesis (infrequent)
        if self.config.synaptogenesis_interval > 0 and \
           self.step_count > 0 and \
           self.step_count % self.config.synaptogenesis_interval == 0:
            self.synaptogenesis_step()

        self.step_count += 1
        return self.spikes

    def simulate(self, external_input: Optional[torch.Tensor] = None,
                 duration_ms: int = 1) -> torch.Tensor:
        """
        Mehrere Simulationsschritte.

        Args:
            external_input: Tensor [n_neurons] (wird jeden Step angelegt)
            duration_ms: Anzahl Millisekunden

        Returns:
            spike_history: Bool-Tensor [duration_ms, n_neurons]
        """
        n = self.config.n_neurons
        history = torch.zeros(duration_ms, n, device=self.device, dtype=torch.bool)

        for t in range(duration_ms):
            spikes = self.step(external_input)
            history[t] = spikes

        return history

    # ========================================================================
    # ELIGIBILITY TRACES
    # ========================================================================

    def _update_eligibility_traces(self):
        """Update pre/post spike traces and per-synapse eligibility."""
        spike_f = self.spikes.float()

        # Decay traces
        self._pre_trace = self._pre_trace * self._trace_decay + spike_f
        self._post_trace = self._post_trace * self._trace_decay + spike_f

        # Per-synapse eligibility: STDP-like coincidence detection
        if self._weight_indices is not None and self._n_synapses > 0:
            pre_idx = self._weight_indices[0]   # source neurons
            post_idx = self._weight_indices[1]  # target neurons

            # pre fires, post has recent trace → LTP-like
            # post fires, pre has recent trace → LTD-like
            pre_trace_at_syn = self._pre_trace[pre_idx]
            post_trace_at_syn = self._post_trace[post_idx]
            pre_spike_at_syn = spike_f[pre_idx]
            post_spike_at_syn = spike_f[post_idx]

            # Eligibility accumulates: pre→post coincidence minus post→pre
            d_elig = (pre_spike_at_syn * post_trace_at_syn -
                      post_spike_at_syn * pre_trace_at_syn)

            self._eligibility = self._eligibility * self._trace_decay + d_elig

    # ========================================================================
    # PLASTIZITÄT
    # ========================================================================

    def apply_rstdp(self, reward_signal: float = 0.0, prediction_error: float = 0.0):
        """
        Reward + Prediction-Error moduliertes STDP.

        Eligibility Traces akkumulieren Koinzidenzen.
        Learning signal = blend of DA reward + prediction error.
        
        Biology: DA (reward) encodes "better than expected" (Schultz 1997).
        Prediction error encodes "my model was wrong" (Friston 2010).
        Combined: the synapse strengthens when the action was rewarding
        AND/OR when the world behaved differently than predicted.
        
        Ref: Cortical Labs DishBrain — neurons learn via Free Energy
        (prediction error minimization), not reward. 18x faster than RL.
        We blend both: 30% reward + 70% prediction error.
        """
        if self._n_synapses == 0 or self._weight_values is None:
            return

        # ACh modulation of learning rate
        lr = self.config.stdp_lr
        if self.config.neuromod_enabled:
            ach = self.neuromod_levels['ach']
            lr = lr * (1.0 + ach)  # ACh boosts learning

        # Weight update: Δw = lr * combined_signal * eligibility
        # combined = 30% reward (DA) + 70% task prediction error
        #
        # Task PE sign convention (DishBrain-inspired):
        #   Negative PE = task succeeding (ball getting closer) → REINFORCE
        #   Positive PE = task failing (ball getting further) → WEAKEN
        # We need to INVERT the PE for the weight update:
        #   -PE → positive signal → reinforce active synapses
        #   +PE → negative signal → weaken active synapses
        elig_clipped = self._eligibility.clamp(-1.0, 1.0)
        # When task PE is active (ball scene), it dominates.
        # Locomotion reward must NOT mask the navigation signal.
        if abs(prediction_error) > 0.05:  # Task PE active
            combined_signal = 0.1 * reward_signal + 0.9 * (-prediction_error)
        else:  # No task PE (flat meadow etc.) — use reward as before
            combined_signal = reward_signal
        dw = lr * combined_signal * elig_clipped

        # Max update per step (balanciert: schnell genug zum Lernen)
        dw = dw.clamp(-0.05, 0.05)

        # Zero out updates for protected synapses (cerebellar populations)
        if self.protected_populations:
            protected_mask = self._get_protected_synapse_mask()
            dw[protected_mask] = 0.0

        self._weight_values = self._weight_values + dw

        # Ensure E/I sign is preserved (tighter clamp for stability)
        exc_mask = self.neuron_types[self._weight_indices[0]] > 0
        inh_mask = ~exc_mask
        self._weight_values = torch.where(
            exc_mask,
            self._weight_values.clamp(min=0.0, max=1.0),
            self._weight_values.clamp(min=-1.0, max=0.0)
        )

        # Decay eligibility after use (prevents stale traces)
        self._eligibility *= 0.3

        # Rebuild sparse matrix
        self._rebuild_sparse_weights()

    def apply_surrogate_gradient_step(self, loss: torch.Tensor):
        """
        Ein Backprop-Through-Time Schritt mit Surrogate Gradients.

        Nur für hybrides Lernen — R-STDP bleibt primär.
        Requires that V has grad tracking enabled.
        """
        if loss.requires_grad:
            loss.backward()

    # ========================================================================
    # NEUROMODULATION
    # ========================================================================

    def set_neuromodulator(self, modulator: str, level: float):
        """
        Setze Neuromodulator-Level.

        Args:
            modulator: 'ne', '5ht', 'ach'
            level: 0.0–1.0

        Effekte:
            NE:  Senkt τ → schnellere Neuronen, erhöht Gain → Exploration
            5-HT: Erhöht τ → langsamere Neuronen → Geduld/Ruhe
            ACh: Erhöht Plastizität (STDP-LR) → Lernen
        """
        level = max(0.0, min(1.0, level))
        if modulator in self.neuromod_levels:
            self.neuromod_levels[modulator] = level
            self._tau_dirty = True  # Issue #106: recompute tau on next step

    def get_neuromodulator_tau(self) -> torch.Tensor:
        """
        Berechnet aktuelle τ pro Neuron basierend auf Neuromodulator-Levels.

        NE senkt tau (schnellere Response), 5-HT erhöht tau (langsamer).
        Formel: τ = τ_base + δτ * (5-HT_sensitivity * 5-HT - NE_sensitivity * NE)

        Returns:
            tau: Tensor [n_neurons]
        """
        if not self.config.neuromod_enabled:
            return self._tau_base.clone()

        ne = self.neuromod_levels['ne']
        sht = self.neuromod_levels['5ht']

        # Per-neuron modulation
        ne_effect = self.neuromod_sensitivity[:, 0] * ne    # Speeds up (lowers tau)
        sht_effect = self.neuromod_sensitivity[:, 1] * sht  # Slows down (raises tau)

        tau = self._tau_base + self.config.delta_tau * (sht_effect - ne_effect)

        # Protected populations keep base tau (cerebellar neurons need stable dynamics)
        if self.protected_populations:
            protected_mask = self._get_protected_neuron_mask()
            tau[protected_mask] = self._tau_base[protected_mask]

        # Clamp to reasonable range
        tau = tau.clamp(min=2.0, max=100.0)
        return tau

    # ========================================================================
    # HOMÖOSTATISCHE PLASTIZITÄT
    # ========================================================================

    def _homeostatic_update(self):
        """Adapt per-neuron thresholds toward target firing rate."""
        if self._homeostatic_step_count == 0:
            return

        actual_rates = self._spike_count_window / self._homeostatic_step_count
        target = self.config.target_firing_rate

        # Neurons firing too much → raise threshold
        # Neurons firing too little → lower threshold
        rate_error = actual_rates - target
        adaptation = 0.01 * rate_error  # Small adaptation steps

        # Skip adaptation for protected populations (cerebellar neurons)
        if self.protected_populations:
            protected_mask = self._get_protected_neuron_mask()
            adaptation[protected_mask] = 0.0

        # Skip adaptation for Izhikevich neurons (they use V>=30 threshold,
        # not _thresholds — adapting would have no effect and waste cycles)
        if self._izh_enabled.any():
            adaptation[self._izh_enabled] = 0.0

        self._thresholds = self._thresholds + adaptation
        self._thresholds = self._thresholds.clamp(min=0.3, max=3.0)

        # Reset window
        self._spike_count_window.zero_()
        self._homeostatic_step_count = 0

    # ========================================================================
    # ASTROZYT-GATE
    # ========================================================================

    def _astrocyte_update(self):
        """Update astrocyte calcium based on spike activity."""
        n = self.config.n_neurons
        cs = self._astro_cluster_size

        # Sum spikes per cluster
        spike_f = self.spikes.float()
        # Pad to multiple of cluster_size
        padded_len = self._n_astrocytes * cs
        if padded_len > n:
            padded = torch.zeros(padded_len, device=self.device, dtype=self.dtype)
            padded[:n] = spike_f
        else:
            padded = spike_f[:padded_len]

        cluster_spikes = padded.view(self._n_astrocytes, cs).sum(dim=1) / cs

        # Calcium influx
        self._astro_calcium = self._astro_calcium + cluster_spikes * 0.1

        # Decay
        decay = 1.0 / self.config.astrocyte_tau_calcium
        self._astro_calcium = self._astro_calcium * (1.0 - decay)

        # Clamp
        self._astro_calcium = self._astro_calcium.clamp(min=0.0, max=2.0)

    def _can_form_synapse(self, pre_neuron: int, post_neuron: int) -> bool:
        """Check if astrocyte gate allows synapse formation."""
        pre_cluster = pre_neuron // self._astro_cluster_size
        post_cluster = post_neuron // self._astro_cluster_size

        threshold = self.config.astrocyte_calcium_threshold
        return (self._astro_calcium[pre_cluster].item() > threshold and
                self._astro_calcium[post_cluster].item() > threshold)

    # ========================================================================
    # SYNAPTOGENESE
    # ========================================================================

    def synaptogenesis_step(self):
        """
        Synaptogenese: Neue Synapsen wachsen, unbenutzte sterben.

        - Ko-aktive Cluster mit Astrozyt-Gate → neue Synapse
        - Synapse mit Gewicht ≈ 0 → Pruning
        - Sparse-Matrix wird rebuilt
        """
        if self._weight_values is None:
            return

        changed = False

        # --- Pruning: remove near-zero synapses ---
        mask = self._weight_values.abs() > self.config.pruning_threshold
        if mask.sum() < self._n_synapses:
            self._weight_indices = self._weight_indices[:, mask]
            self._weight_values = self._weight_values[mask]
            self._eligibility = self._eligibility[mask]
            changed = True

        # --- Growth: add synapses in co-active astrocyte clusters ---
        active_clusters = torch.where(
            self._astro_calcium > self.config.astrocyte_calcium_threshold
        )[0]

        if len(active_clusters) >= 2:
            n_new = min(self.config.synaptogenesis_max_new, len(active_clusters) * 10)
            cs = self._astro_cluster_size
            new_src = []
            new_tgt = []

            for _ in range(n_new):
                # Pick two random active clusters
                idx = torch.randint(0, len(active_clusters), (2,), device=self.device)
                c1 = active_clusters[idx[0]].item()
                c2 = active_clusters[idx[1]].item()
                if c1 == c2:
                    continue

                # Random neuron from each cluster
                src = c1 * cs + torch.randint(0, min(cs, self.config.n_neurons - c1 * cs), (1,)).item()
                tgt = c2 * cs + torch.randint(0, min(cs, self.config.n_neurons - c2 * cs), (1,)).item()

                if src < self.config.n_neurons and tgt < self.config.n_neurons and src != tgt:
                    new_src.append(src)
                    new_tgt.append(tgt)

            if new_src:
                new_src_t = torch.tensor(new_src, device=self.device, dtype=torch.long)
                new_tgt_t = torch.tensor(new_tgt, device=self.device, dtype=torch.long)
                new_indices = torch.stack([new_src_t, new_tgt_t])

                # Initial weight: small, sign based on neuron type
                signs = self.neuron_types[new_src_t]
                new_weights = torch.randn(len(new_src), device=self.device, dtype=self.dtype).abs() * 0.05 * signs

                self._weight_indices = torch.cat([self._weight_indices, new_indices], dim=1)
                self._weight_values = torch.cat([self._weight_values, new_weights])
                new_elig = torch.zeros(len(new_src), device=self.device, dtype=self.dtype)
                self._eligibility = torch.cat([self._eligibility, new_elig])
                changed = True

        if changed:
            self._rebuild_sparse_weights()

    # ========================================================================
    # MONITORING
    # ========================================================================

    def get_state(self) -> Dict:
        """
        Gibt aktuellen Netzwerk-Zustand zurück.

        Returns:
            Dict mit: firing_rates, mean_potential, e_i_ratio,
                      n_synapses, neuromod_levels, tau_distribution
        """
        n = self.config.n_neurons
        exc_mask = self.neuron_types > 0
        inh_mask = ~exc_mask

        # Firing rates from homeostatic window
        if self._homeostatic_step_count > 0:
            firing_rates = self._spike_count_window / self._homeostatic_step_count
        else:
            firing_rates = torch.zeros(n, device=self.device, dtype=self.dtype)

        # E/I ratio: average excitatory rate / average inhibitory rate
        exc_rate = firing_rates[exc_mask].mean().item() if exc_mask.any() else 0.0
        inh_rate = firing_rates[inh_mask].mean().item() if inh_mask.any() else 0.001
        e_i_ratio = exc_rate / max(inh_rate, 1e-6)

        tau = self.get_neuromodulator_tau()

        return {
            'firing_rates': firing_rates,
            'mean_potential': self.V.mean().item(),
            'e_i_ratio': e_i_ratio,
            'n_synapses': self._n_synapses,
            'neuromod_levels': dict(self.neuromod_levels),
            'tau_distribution': {
                'mean': tau.mean().item(),
                'std': tau.std().item(),
                'min': tau.min().item(),
                'max': tau.max().item(),
            },
            'step_count': self.step_count,
            'n_neurons': n,
            'thresholds_mean': self._thresholds.mean().item(),
            'astro_active_clusters': int((self._astro_calcium > self.config.astrocyte_calcium_threshold).sum().item()),
        }

    def get_population_activity(self, population: str) -> torch.Tensor:
        """
        Aktivität einer benannten Population.

        Args:
            population: Name der Population

        Returns:
            Firing rates der Population
        """
        if population not in self.populations:
            raise KeyError(f"Population '{population}' not defined. "
                           f"Available: {list(self.populations.keys())}")

        ids = self.populations[population]

        if self._homeostatic_step_count > 0:
            rates = self._spike_count_window[ids] / self._homeostatic_step_count
        else:
            rates = torch.zeros(len(ids), device=self.device, dtype=self.dtype)

        return rates

    # ========================================================================
    # PERSISTENZ
    # ========================================================================

    def save(self, path: str):
        """Speichert SNN-Zustand."""
        state = {
            'config': self.config,
            'V': self.V.cpu(),
            'spikes': self.spikes.cpu(),
            'refractory_counter': self.refractory_counter.cpu(),
            'neuron_types': self.neuron_types.cpu(),
            'neuromod_levels': self.neuromod_levels,
            'neuromod_sensitivity': self.neuromod_sensitivity.cpu(),
            'populations': {k: v.cpu() for k, v in self.populations.items()},
            'weight_indices': self._weight_indices.cpu() if self._weight_indices is not None else None,
            'weight_values': self._weight_values.cpu() if self._weight_values is not None else None,
            'eligibility': self._eligibility.cpu(),
            'pre_trace': self._pre_trace.cpu(),
            'post_trace': self._post_trace.cpu(),
            'thresholds': self._thresholds.cpu(),
            'astro_calcium': self._astro_calcium.cpu(),
            'spike_count_window': self._spike_count_window.cpu(),
            'homeostatic_step_count': self._homeostatic_step_count,
            'step_count': self.step_count,
            # Izhikevich state (v0.5.0)
            'izh_a': self._izh_a.cpu(),
            'izh_b': self._izh_b.cpu(),
            'izh_c': self._izh_c.cpu(),
            'izh_d': self._izh_d.cpu(),
            'izh_u': self._u.cpu(),
            'izh_enabled': self._izh_enabled.cpu(),
        }
        torch.save(state, path)

    def load(self, path: str):
        """Lädt SNN-Zustand (backward-compatible with v0.4.x)."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.V = state['V'].to(self.device)
        self.spikes = state['spikes'].to(self.device)
        self.refractory_counter = state['refractory_counter'].to(self.device)
        self.neuron_types = state['neuron_types'].to(self.device)
        self.neuromod_levels = state['neuromod_levels']
        self.neuromod_sensitivity = state['neuromod_sensitivity'].to(self.device)
        self.populations = {k: v.to(self.device) for k, v in state['populations'].items()}

        self._weight_indices = state['weight_indices'].to(self.device) if state['weight_indices'] is not None else None
        self._weight_values = state['weight_values'].to(self.device) if state['weight_values'] is not None else None
        self._eligibility = state['eligibility'].to(self.device)
        self._pre_trace = state['pre_trace'].to(self.device)
        self._post_trace = state['post_trace'].to(self.device)
        self._thresholds = state['thresholds'].to(self.device)
        self._astro_calcium = state['astro_calcium'].to(self.device)
        self._spike_count_window = state['spike_count_window'].to(self.device)
        self._homeostatic_step_count = state['homeostatic_step_count']
        self.step_count = state['step_count']

        # Izhikevich state (v0.5.0) — backward-compatible with v0.4.x brains
        if 'izh_a' in state:
            self._izh_a = state['izh_a'].to(self.device)
            self._izh_b = state['izh_b'].to(self.device)
            self._izh_c = state['izh_c'].to(self.device)
            self._izh_d = state['izh_d'].to(self.device)
            self._u = state['izh_u'].to(self.device)
            self._izh_enabled = state['izh_enabled'].to(self.device)
        # else: keep defaults from __init__ (RS params, u=0, izh_enabled=False)

        self._n_synapses = self._weight_values.shape[0] if self._weight_values is not None else 0
        self._rebuild_sparse_weights()

    # ========================================================================
    # POPULATIONS-MANAGEMENT
    # ========================================================================

    def define_population(self, name: str, neuron_ids: torch.Tensor):
        """Definiert eine benannte Neuron-Population."""
        self.populations[name] = neuron_ids.to(self.device)

    def set_izhikevich_params(self, population: str,
                               a: float, b: float, c: float, d: float):
        """
        Set per-population Izhikevich (a,b,c,d) parameters and enable
        full Izhikevich dynamics for this population.

        Issue #104: Population-Specific Neuron Types.

        Standard parameter sets (Izhikevich 2003, Table 2):
          Regular Spiking (RS):      a=0.02, b=0.2,  c=-65, d=8
          Intrinsically Bursting:    a=0.02, b=0.2,  c=-55, d=4
          Chattering (CH):           a=0.02, b=0.2,  c=-50, d=2
          Low-Threshold Spiking:     a=0.02, b=0.25, c=-65, d=2
          Rebound Burst (DCN):       a=0.03, b=0.25, c=-52, d=0
          Fast Spiking (FS):         a=0.1,  b=0.2,  c=-65, d=2

        Args:
            population: Name of the population
            a: Recovery time scale (smaller = slower recovery)
            b: Sensitivity of u to V (larger = stronger coupling)
            c: After-spike reset value of V (mV)
            d: After-spike reset increment of u
        """
        if population not in self.populations:
            raise KeyError(f"Population '{population}' not defined. "
                           f"Available: {list(self.populations.keys())}")
        ids = self.populations[population]
        self._izh_a[ids] = a
        self._izh_b[ids] = b
        self._izh_c[ids] = c
        self._izh_d[ids] = d
        self._izh_enabled[ids] = True
        # Initialize u = b * V for this population
        self._u[ids] = b * self.V[ids]

    def _get_protected_neuron_mask(self) -> torch.Tensor:
        """Returns a boolean mask [n_neurons] where True = protected (no learning)."""
        mask = torch.zeros(self.config.n_neurons, dtype=torch.bool, device=self.device)
        for pop_name in self.protected_populations:
            if pop_name in self.populations:
                mask[self.populations[pop_name]] = True
        return mask

    def _get_protected_synapse_mask(self) -> torch.Tensor:
        """Returns a boolean mask [n_synapses] where True = protected.
        A synapse is protected if EITHER pre or post neuron is in a protected population."""
        if self._weight_indices is None or self._n_synapses == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        neuron_mask = self._get_protected_neuron_mask()
        pre_protected = neuron_mask[self._weight_indices[0]]
        post_protected = neuron_mask[self._weight_indices[1]]
        return pre_protected | post_protected

    def connect_populations(self, source: str, target: str,
                            prob: float = 0.05,
                            weight_range: Tuple[float, float] = (0.1, 0.5)):
        """Verbindet zwei Populationen mit gegebener Wahrscheinlichkeit."""
        if source not in self.populations or target not in self.populations:
            raise KeyError(f"Population not found. Available: {list(self.populations.keys())}")

        src_ids = self.populations[source]
        tgt_ids = self.populations[target]

        n_connections = int(len(src_ids) * len(tgt_ids) * prob)
        if n_connections == 0:
            return

        # Random connections
        src_pick = src_ids[torch.randint(0, len(src_ids), (n_connections,), device=self.device)]
        tgt_pick = tgt_ids[torch.randint(0, len(tgt_ids), (n_connections,), device=self.device)]

        # Remove self-connections
        mask = src_pick != tgt_pick
        src_pick = src_pick[mask]
        tgt_pick = tgt_pick[mask]

        if len(src_pick) == 0:
            return

        # Weights in range, with E/I sign
        n_actual = src_pick.shape[0]
        w_min, w_max = weight_range
        raw_weights = torch.rand(n_actual, device=self.device, dtype=self.dtype) * (w_max - w_min) + w_min
        signs = self.neuron_types[src_pick]
        new_weights = raw_weights * signs

        new_indices = torch.stack([src_pick, tgt_pick])

        # Append to existing
        if self._weight_indices is not None:
            self._weight_indices = torch.cat([self._weight_indices, new_indices], dim=1)
            self._weight_values = torch.cat([self._weight_values, new_weights])
            new_elig = torch.zeros(n_actual, device=self.device, dtype=self.dtype)
            self._eligibility = torch.cat([self._eligibility, new_elig])
        else:
            self._weight_indices = new_indices
            self._weight_values = new_weights
            self._eligibility = torch.zeros(n_actual, device=self.device, dtype=self.dtype)

        self._rebuild_sparse_weights()
        self._population_connections.append((source, target, prob, weight_range))
