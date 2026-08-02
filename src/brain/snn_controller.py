"""
MH-FLOCKE — SNN Controller v0.5.2
========================================
Izhikevich spiking neural network with R-STDP on PyTorch tensors.

v0.5.2: Overhead reduction for Pi4 n_hidden=500 target (Issue #144).
         Pre-allocated buffers for I_syn/refractory/noise. In-place Izh ops.
         Single trace vector (pre==post). Reuse _spike_float everywhere.
         Cached _all_izh flag. torch.set_num_threads(1) for small networks.
v0.5.1: Performance: unified Izhikevich fast path (no masking when all
         neurons are izh-enabled), lazy dense weight update in apply_rstdp.
         Issue #144: Pi4 real-time budget.
v0.5.0: Per-population Izhikevich (a,b,c,d) parameters + recovery variable u.
         DCN neurons use Low-Threshold Spiking parameters for rebound bursting.
         Issue #104: Population-Specific Neuron Types.
         Issue #106: Performance optimizations (torch.no_grad, pre-alloc).
"""

__version__ = "0.5.3"     # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 22          # mh-logbuch module entry
__status__  = "active"    # active | veraltet | neu

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import math


# === Surrogate Gradient Functions ===

class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast-Sigmoid Surrogate Gradient for backprop through spikes."""

    @staticmethod
    def forward(ctx, V, threshold):
        ctx.save_for_backward(V, torch.tensor(threshold, device=V.device))
        return (V >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        V, threshold = ctx.saved_tensors
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
    """Configuration for the SNN Controller."""
    n_neurons: int = 500_000
    n_excitatory_ratio: float = 0.8
    connectivity_prob: float = 0.02

    # LIF-LTC parameters
    tau_base: float = 20.0
    delta_tau: float = 15.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    refractory_ms: int = 2

    # Plasticity
    stdp_lr: float = 0.01
    surrogate_gradient: str = 'fast_sigmoid'

    # Reward baseline (25./26.07.2026, Logbuch #215).
    # apply_rstdp moduliert per Default mit dem ROHWERT des Lernsignals.
    # GEMESSEN (Lauf v043_1785080567, Seed 42): der Modulator hat Mittel -0.10
    # und ist nur zu 8,2 % positiv -- also naeherungsweise gleichfoermige
    # ABSCHWAECHUNG. Grund: bei |PE| > 0.05 rechnet apply_rstdp
    # 0.1 * R + 0.9 * (-PE); dieser Zweig dominiert, PE ist positiv.
    # (Die urspruengliche Vermutung lautete umgekehrt ">90 % positiv, degeneriert
    # zu Hebb'schem Potenzieren" -- das war im Vorzeichen falsch.)
    # So oder so: ein Modulator mit zu 92 % gleichem Vorzeichen unterscheidet
    # nicht zwischen Beitrag und Nicht-Beitrag. Der Docstring von apply_rstdp
    # nennt die richtige Definition -- Dopamin kodiert "better than expected"
    # (Schultz 1997) -- sie war nur nicht implementiert.
    # reward_baseline=True zieht einen gleitenden Erwartungswert ab und macht den
    # Modulator mittelwertfrei (gemessen: 47,6 % positiv, Streuung bleibt).
    # Default False -> bit-identisch zum bisherigen Verhalten.
    reward_baseline: bool = False
    reward_baseline_alpha: float = 0.01   # EMA-Rate; 0.01 ~ Zeitkonstante 100 Updates

    # Mischung Belohnung / Vorhersagefehler im Lernsignal (26.07.2026, #215).
    # apply_rstdp rechnet bei |PE| > pe_threshold:
    #     combined = (1 - pe_blend) * R + pe_blend * (-PE)
    # GEMESSEN (v043_1785083780, Seed 42, 10k): Der PE-Zweig greift zu 98,7 % --
    # der andere Zweig ist praktisch toter Code. |PE| liegt im Mittel bei 0.2004
    # gegen eine Schwelle von 0.05, also viermal darueber; die Schwelle trennt
    # nichts. Beitragsanteile am Betrag: R 29,2 %, PE 70,8 %.
    # Folge: 0.9 * 0.2004 = 0.18 ist ein nahezu KONSTANTER negativer Sockel; das
    # Lernsignal besteht aus diesem Sockel plus einer kleinen Schwankung, die
    # ueberwiegend aus dem PE stammt und nicht aus den intrinsischen Antrieben.
    # pe_blend senken => Neugier, Empowerment, Vestibulaer und Propriozeption
    # bekommen mehr Gewicht im Lernen. Default 0.9 -> bit-identisch.
    pe_blend: float = 0.9
    pe_threshold: float = 0.05

    # Eligibility-Trace (26.07.2026, #215).
    # BEFUND: _trace_decay = 0.95 bediente ZWEI biologisch verschiedene Dinge --
    #   (a) den STDP-Spurspeicher, also das KOINZIDENZFENSTER zwischen prae- und
    #       postsynaptischem Spike (biologisch ~20 ms), und
    #   (b) die ELIGIBILITY, also wie lange eine Koinzidenz auf Belohnung wartet
    #       (biologisch 0,3-2 s, Dopamin-Fenster, Yagishita et al. 2014).
    # Das sind zwei Groessenordnungen Unterschied an einer einzigen Konstante.
    # Wer den Trace verlaengert, um Credit Assignment zu ermoeglichen, verbreitert
    # zwangsläufig auch das Koinzidenzfenster und macht STDP unspezifisch.
    # Bei 0.95 pro SNN-Schritt und --snn-substeps 10 ergibt das 0.60 pro
    # Regelschritt, Halbwertszeit ~1,35 Regelschritte -- verzoegerte Belohnung ist
    # damit strukturell nicht lernbar.
    # eligibility_decay=None -> nimmt _trace_decay => bit-identisch.
    # eligibility_consume: Faktor, mit dem die Eligibility nach jedem apply_rstdp
    # verbraucht wird (bisher fest 0.3).
    eligibility_decay: Optional[float] = None
    eligibility_consume: float = 0.3

    # Neuromodulators
    neuromod_enabled: bool = True

    # Homeostatic plasticity
    homeostatic_interval: int = 1000
    target_firing_rate: float = 0.05

    # Astrocyte gate
    astrocyte_cluster_size: int = 100
    astrocyte_calcium_threshold: float = 0.7
    astrocyte_tau_calcium: float = 2000.0

    # Synaptogenesis
    synaptogenesis_interval: int = 5000
    synaptogenesis_max_new: int = 1000
    pruning_threshold: float = 0.001

    # Hardware
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


class SNNController:
    """
    Tensorized Spiking Neural Network Controller.

    GPU-first design with:
    - LIF-LTC (Liquid Time-Constant) neurons
    - Sparse COO connectivity
    - R-STDP + optional Surrogate Gradients
    - Integrated neuromodulators (NE, 5-HT, ACh)
    - Astrocyte gate for synaptogenesis
    - Homeostatic plasticity
    - Adaptive E/I balance
    """

    def __init__(self, config: SNNConfig):
        """Initialize SNN on GPU/CPU."""
        self.config = config
        self.device = torch.device(config.device if config.device == 'cuda'
                                   and torch.cuda.is_available() else 'cpu')
        self.dtype = config.dtype
        n = config.n_neurons

        # v0.5.2: Single thread is faster for n<1000 on ARM (Pi4).
        # Threading overhead > parallelism gain for small tensors.
        if n < 1000:
            torch.set_num_threads(1)

        # --- Neuron state ---
        self.V = torch.zeros(n, device=self.device, dtype=self.dtype)
        self.spikes = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.refractory_counter = torch.zeros(n, device=self.device, dtype=torch.int32)

        # --- Neuron types: +1 excitatory, -1 inhibitory ---
        n_exc = int(n * config.n_excitatory_ratio)
        self.neuron_types = torch.ones(n, device=self.device, dtype=self.dtype)
        self.neuron_types[n_exc:] = -1.0
        perm = torch.randperm(n, device=self.device)
        self.neuron_types = self.neuron_types[perm]

        # --- Neuromodulators ---
        self.neuromod_levels = {
            'ne': 0.3,
            '5ht': 0.5,
            'ach': 0.5,
        }
        self.neuromod_sensitivity = torch.rand(n, 3, device=self.device, dtype=self.dtype) * 0.4 + 0.8

        # --- Populations ---
        self.populations: Dict[str, torch.Tensor] = {}
        self.protected_populations: set = set()

        # --- Eligibility Traces (for R-STDP) ---
        # v0.5.2: Single trace vector. Pre and post traces were identical
        # (same decay, same spike input). Merged to halve trace ops.
        self._trace = torch.zeros(n, device=self.device, dtype=self.dtype)
        # Keep _pre_trace/_post_trace as aliases for backward compat with save/load
        self._pre_trace = self._trace
        self._post_trace = self._trace
        self._trace_decay = 0.95
        # Eigene Zeitkonstante fuer die Eligibility (siehe SNNConfig).
        # None => identisch zum Spurspeicher => bit-identisch zum alten Verhalten.
        _ed = getattr(config, 'eligibility_decay', None)
        self._elig_decay = self._trace_decay if _ed is None else float(_ed)
        self._eligibility = torch.zeros(1, device=self.device, dtype=self.dtype)

        # --- Connectivity ---
        self._population_connections: List[Tuple[str, str, float, Tuple[float, float]]] = []
        self.weights: Optional[torch.Tensor] = None
        self._weight_indices: Optional[torch.Tensor] = None
        self._weight_values: Optional[torch.Tensor] = None
        self._n_synapses = 0

        self._init_default_connectivity()

        # --- Homeostatic Plasticity ---
        self._spike_count_window = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._homeostatic_step_count = 0
        self._last_rate_err_mean = 0.0
        self._last_rate_err_max = 0.0
        self._last_adapt_mean = 0.0
        self._last_adapt_max = 0.0
        self._homeo_updates = 0
        # Erwartungswert der Belohnung (siehe SNNConfig.reward_baseline).
        self._reward_ema = 0.0
        self._last_modulator = 0.0
        self._rstdp_calls = 0
        self._pe_branch_hits = 0
        self._last_pe = 0.0
        self._last_reward_in = 0.0
        self._last_combined = 0.0
        self._thresholds = torch.full((n,), config.v_threshold, device=self.device, dtype=self.dtype)

        # --- Astrocyte Gate ---
        self._astro_cluster_size = config.astrocyte_cluster_size
        self._n_astrocytes = (n + self._astro_cluster_size - 1) // self._astro_cluster_size
        self._astro_calcium = torch.zeros(self._n_astrocytes, device=self.device, dtype=self.dtype)

        # --- Per-neuron membrane time constant ---
        self._tau_base = torch.full((n,), config.tau_base, device=self.device, dtype=self.dtype)

        # --- Per-neuron Izhikevich parameters (Issue #104) ---
        self._izh_a = torch.full((n,), 0.02, device=self.device, dtype=self.dtype)
        self._izh_b = torch.full((n,), 0.2, device=self.device, dtype=self.dtype)
        self._izh_c = torch.full((n,), -65.0, device=self.device, dtype=self.dtype)
        self._izh_d = torch.full((n,), 8.0, device=self.device, dtype=self.dtype)
        self._u = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._izh_enabled = torch.zeros(n, device=self.device, dtype=torch.bool)

        # --- Pre-allocated tensors for performance (Issue #106 + #144) ---
        self._v_reset_tensor = torch.tensor(config.v_reset, device=self.device, dtype=self.dtype)
        self._refractory_tensor = torch.tensor(config.refractory_ms, device=self.device, dtype=torch.int32)
        self._spike_float = torch.zeros(n, device=self.device, dtype=self.dtype)
        # v0.5.2: Pre-allocated buffers to eliminate per-step allocations
        self._I_syn_buf = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._refractory_mask_buf = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._dV_buf = torch.zeros(n, device=self.device, dtype=self.dtype)
        self._noise_buf = torch.zeros(n, device=self.device, dtype=self.dtype)
        # Dense weight matrix cache
        self._dense_weights: Optional[torch.Tensor] = None
        self._dense_weights_dirty = True
        # Tau cache
        self._tau_cached: Optional[torch.Tensor] = None
        self._tau_dirty = True
        # v0.5.2: Cached all_izh flag — set once by builder, never changes
        self._all_izh: bool = False

        # --- Simulation Counter ---
        self.step_count = 0

    # ========================================================================
    # CONNECTIVITY
    # ========================================================================

    def _init_default_connectivity(self):
        """Initialize sparse connectivity based on config."""
        n = self.config.n_neurons
        p = self.config.connectivity_prob

        if p <= 0.0:
            self._weight_indices = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            self._weight_values = torch.zeros(0, device=self.device, dtype=self.dtype)
            self._eligibility = torch.zeros(0, device=self.device, dtype=self.dtype)
            self._n_synapses = 0
            self._rebuild_sparse_weights()
            return

        max_synapses = min(int(n * n * p), 20_000_000)
        actual_synapses = min(int(n * n * p), max_synapses)
        if actual_synapses == 0:
            actual_synapses = max(n * 10, 1000)

        src = torch.randint(0, n, (actual_synapses,), device=self.device)
        tgt = torch.randint(0, n, (actual_synapses,), device=self.device)
        mask = src != tgt
        src = src[mask]
        tgt = tgt[mask]

        n_connections = src.shape[0]
        weights = torch.randn(n_connections, device=self.device, dtype=self.dtype) * 0.1
        weights = weights.abs()
        signs = self.neuron_types[src]
        weights = weights * signs

        self._weight_indices = torch.stack([src, tgt])
        self._weight_values = weights
        self._n_synapses = n_connections
        self._rebuild_sparse_weights()

    def _rebuild_sparse_weights(self):
        """Rebuild sparse weight matrix from indices and values."""
        n = self.config.n_neurons
        self._dense_weights = None
        self._dense_weights_dirty = True
        if self.weights is not None:
            del self.weights
            self.weights = None
        if self._weight_indices is not None and self._weight_values is not None:
            self.weights = torch.sparse_coo_tensor(
                self._weight_indices, self._weight_values,
                size=(n, n), device=self.device
            ).coalesce()
            nnz = self._weight_values.shape[0]
            if self._eligibility.shape[0] != nnz:
                self._eligibility = torch.zeros(nnz, device=self.device, dtype=self.dtype)
            self._n_synapses = nnz
        else:
            self.weights = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device, dtype=self.dtype),
                size=(n, n), device=self.device
            )
            self._n_synapses = 0

    # ========================================================================
    # CORE SIMULATION
    # ========================================================================

    def step(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One simulation step (1ms).

        v0.5.2 performance optimizations:
        - Cached _all_izh flag (no izh_mask.all() per step)
        - Pre-allocated I_syn, refractory_mask, dV, noise buffers
        - In-place Izhikevich ops (add_, mul_) to reduce allocations
        - Reuse _spike_float for eligibility + homeostasis (no redundant .float())
        - Single trace vector instead of separate pre/post traces
        - torch.set_num_threads(1) for n<1000

        Args:
            external_input: Tensor [n_neurons] with external current (optional)

        Returns:
            spikes: Bool-Tensor [n_neurons]
        """
        with torch.no_grad():
            n = self.config.n_neurons

            # 2. Synaptic input — reuse pre-allocated buffer
            sf = self._spike_float
            sf.zero_()
            sf[self.spikes] = 1.0

            if self._n_synapses > 0:
                if n < 600:  # Dense path for small SNNs (Freenove: 560)
                    if self._dense_weights is None or self._dense_weights_dirty:
                        if self._weight_indices is not None and self._weight_values is not None:
                            if self._dense_weights is None:
                                self._dense_weights = torch.zeros(n, n, device=self.device, dtype=self.dtype)
                            else:
                                self._dense_weights.zero_()
                            self._dense_weights[self._weight_indices[0], self._weight_indices[1]] = self._weight_values
                        self._dense_weights_dirty = False
                    if self._dense_weights is not None:
                        torch.mv(self._dense_weights.t(), sf, out=self._I_syn_buf)
                    else:
                        self._I_syn_buf.zero_()
                else:
                    spike_col = sf.unsqueeze(1)
                    I_syn_tmp = torch.sparse.mm(self.weights.t(), spike_col).squeeze(1)
                    self._I_syn_buf.copy_(I_syn_tmp)
            else:
                self._I_syn_buf.zero_()

            # 3. NE exploration noise — in-place
            if self.config.neuromod_enabled:
                ne_level = self.neuromod_levels['ne']
                if ne_level > 0.01:
                    self._noise_buf.normal_()
                    self._noise_buf.mul_(0.1 * ne_level)
                    self._I_syn_buf.add_(self._noise_buf)

            # 4. Refractory mask — in-place
            torch.le(self.refractory_counter, 0, out=self.spikes)  # temp bool reuse
            self._refractory_mask_buf.copy_(self.spikes.float())
            ref = self._refractory_mask_buf

            # 5a. External input
            if external_input is not None:
                self._I_syn_buf.add_(external_input)

            # 5b. Izhikevich dynamics — all-izh fast path with in-place ops
            if self._all_izh:
                # I_total is now in _I_syn_buf, scale to mV range
                I = self._I_syn_buf
                I.mul_(10.0)

                # Two half-steps for numerical stability
                # dV = 0.04*V*V + 5*V + 140 - u + I
                dV = self._dV_buf
                torch.mul(self.V, self.V, out=dV)
                dV.mul_(0.04)
                dV.add_(self.V, alpha=5.0)
                dV.add_(140.0)
                dV.sub_(self._u)
                dV.add_(I)
                self.V.add_(dV, alpha=0.5)

                # Second half-step (recompute dV with updated V)
                torch.mul(self.V, self.V, out=dV)
                dV.mul_(0.04)
                dV.add_(self.V, alpha=5.0)
                dV.add_(140.0)
                dV.sub_(self._u)
                dV.add_(I)
                self.V.add_(dV, alpha=0.5)

                # Recovery variable: du = a*(b*V - u)
                # u += du  →  u += a*(b*V - u)
                # In-place: _dV_buf = b*V; _dV_buf -= u; _dV_buf *= a; u += _dV_buf
                torch.mul(self._izh_b, self.V, out=dV)
                dV.sub_(self._u)
                dV.mul_(self._izh_a)
                self._u.add_(dV)

                # Refractory: V = V * ref + c * (1 - ref)
                self.V.mul_(ref)
                # c * (1 - ref): reuse dV
                torch.sub(torch.ones(1, device=self.device), ref, out=dV)
                dV.mul_(self._izh_c)
                self.V.add_(dV)

                # Spike generation
                torch.ge(self.V, 30.0, out=self.spikes)
                if self.spikes.any():
                    self.V[self.spikes] = self._izh_c[self.spikes]
                    self._u[self.spikes] = self._u[self.spikes] + self._izh_d[self.spikes]
            else:
                # Mixed path (large networks or legacy)
                izh_mask = self._izh_enabled
                any_izh = izh_mask.any()
                lif_mask = ~izh_mask
                any_lif = lif_mask.any()

                I_total = self._I_syn_buf

                if any_izh:
                    V_izh = self.V[izh_mask]
                    u_izh = self._u[izh_mask]
                    a_izh = self._izh_a[izh_mask]
                    b_izh = self._izh_b[izh_mask]
                    I_izh = I_total[izh_mask] * 10.0

                    dV = 0.04 * V_izh * V_izh + 5.0 * V_izh + 140.0 - u_izh + I_izh
                    V_izh = V_izh + 0.5 * dV
                    dV = 0.04 * V_izh * V_izh + 5.0 * V_izh + 140.0 - u_izh + I_izh
                    V_izh = V_izh + 0.5 * dV
                    du = a_izh * (b_izh * V_izh - u_izh)
                    u_izh = u_izh + du

                    ref_izh = ref[izh_mask]
                    V_izh = V_izh * ref_izh + self._izh_c[izh_mask] * (1.0 - ref_izh)
                    self.V[izh_mask] = V_izh
                    self._u[izh_mask] = u_izh

                if any_lif:
                    if self._tau_dirty or self._tau_cached is None:
                        self._tau_cached = self.get_neuromodulator_tau()
                        self._tau_dirty = False
                    tau = self._tau_cached

                    V_lif = self.V[lif_mask]
                    tau_lif = tau[lif_mask]
                    I_syn_lif = I_total[lif_mask]
                    ext_lif = external_input[lif_mask] if external_input is not None else 0.0
                    dv = (-(V_lif - self.config.v_rest) + I_syn_lif) / tau_lif + ext_lif
                    V_lif = V_lif + dv
                    V_lif = V_lif * ref[lif_mask]
                    self.V[lif_mask] = V_lif

                izh_spikes = torch.zeros(n, device=self.device, dtype=torch.bool)
                lif_spikes = torch.zeros(n, device=self.device, dtype=torch.bool)

                if any_izh:
                    izh_spikes[izh_mask] = self.V[izh_mask] >= 30.0
                    spiked_izh = izh_mask & izh_spikes
                    if spiked_izh.any():
                        self.V[spiked_izh] = self._izh_c[spiked_izh]
                        self._u[spiked_izh] = self._u[spiked_izh] + self._izh_d[spiked_izh]

                if any_lif:
                    lif_spikes[lif_mask] = self.V[lif_mask] >= self._thresholds[lif_mask]
                    spiked_lif = lif_mask & lif_spikes
                    if spiked_lif.any():
                        self.V[spiked_lif] = self.config.v_reset

                self.spikes = izh_spikes | lif_spikes

            # Refractory counter update — in-place
            self.refractory_counter[self.spikes] = self._refractory_tensor
            self.refractory_counter.sub_(1).clamp_(min=-1)

        # Update _spike_float to match new spikes (for eligibility + homeostasis)
        sf.zero_()
        sf[self.spikes] = 1.0

        # 7. Eligibility Traces — reuse sf, single trace
        self._trace.mul_(self._trace_decay).add_(sf)
        if self._weight_indices is not None and self._n_synapses > 0:
            pre_idx = self._weight_indices[0]
            post_idx = self._weight_indices[1]
            # d_elig = spike[pre] * trace[post] - spike[post] * trace[pre]
            d_elig = (sf[pre_idx] * self._trace[post_idx] -
                      sf[post_idx] * self._trace[pre_idx])
            # Eigene Zeitkonstante: die Eligibility wartet auf Belohnung, der
            # Spurspeicher oben misst Koinzidenz. Zwei Rollen, zwei Konstanten.
            self._eligibility.mul_(self._elig_decay).add_(d_elig)

        # 8. Spike count for homeostasis — reuse sf
        self._spike_count_window.add_(sf)
        self._homeostatic_step_count += 1

        # 9. Homeostatic plasticity
        if self._homeostatic_step_count >= self.config.homeostatic_interval:
            self._homeostatic_update()

        # 10. Astrocyte calcium update (every 10 steps)
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
        """Run multiple simulation steps."""
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
        """Update spike traces and per-synapse eligibility.
        
        NOTE: In v0.5.2 this is inlined into step() for performance.
        This method exists for backward compatibility / external callers.
        """
        sf = self._spike_float
        self._trace.mul_(self._trace_decay).add_(sf)

        if self._weight_indices is not None and self._n_synapses > 0:
            pre_idx = self._weight_indices[0]
            post_idx = self._weight_indices[1]
            d_elig = (sf[pre_idx] * self._trace[post_idx] -
                      sf[post_idx] * self._trace[pre_idx])
            self._eligibility.mul_(self._elig_decay).add_(d_elig)

    # ========================================================================
    # PLASTICITY
    # ========================================================================

    def apply_rstdp(self, reward_signal: float = 0.0, prediction_error: float = 0.0):
        """
        Reward + Prediction-Error modulated STDP.

        Eligibility Traces accumulate coincidences.
        Learning signal = blend of DA reward + prediction error.
        
        Biology: DA (reward) encodes "better than expected" (Schultz 1997).
        Prediction error encodes "my model was wrong" (Friston 2010).
        Combined: the synapse strengthens when the action was rewarding
        AND/OR when the world behaved differently than predicted.
        """
        if self._n_synapses == 0 or self._weight_values is None:
            return

        lr = self.config.stdp_lr
        if self.config.neuromod_enabled:
            ach = self.neuromod_levels['ach']
            lr = lr * (1.0 + ach)

        elig_clipped = self._eligibility.clamp(-1.0, 1.0)
        _thr = getattr(self.config, 'pe_threshold', 0.05)
        _w = getattr(self.config, 'pe_blend', 0.9)
        if abs(prediction_error) > _thr:
            combined_signal = (1.0 - _w) * reward_signal + _w * (-prediction_error)
            self._pe_branch_hits += 1
        else:
            combined_signal = reward_signal
        # Anteile getrennt festhalten (26.07.2026): Wenn der PE-Zweig fast immer
        # greift, bestimmt die intrinsische Belohnung das Lernen nur zu 10 %.
        # Der PE hier ist der ARGUMENTWERT von apply_rstdp -- nicht zwingend
        # dasselbe wie das FLOG-Feld 'pred_error' aus dem Forward-Modell.
        self._last_pe = float(prediction_error)
        self._last_reward_in = float(reward_signal)
        self._last_combined = float(combined_signal)

        # Schultz 1997: Dopamin kodiert die ABWEICHUNG von der Erwartung, nicht
        # den Rohwert. Ohne diesen Abzug ist der Modulator >90 % der Zeit positiv
        # und dw damit fast immer positiv -- verstaerkt wird alles, was aktiv war.
        # Mit Abzug wird der Modulator mittelwertfrei: besser als erwartet
        # verstaerkt, schlechter als erwartet schwaecht ab.
        if self.config.reward_baseline:
            modulator = combined_signal - self._reward_ema
            a = self.config.reward_baseline_alpha
            self._reward_ema += a * (combined_signal - self._reward_ema)
        else:
            modulator = combined_signal
        self._last_modulator = float(modulator)
        self._rstdp_calls += 1

        dw = lr * modulator * elig_clipped
        dw = dw.clamp(-0.05, 0.05)

        if self.protected_populations:
            protected_mask = self._get_protected_synapse_mask()
            dw[protected_mask] = 0.0

        self._weight_values = self._weight_values + dw

        exc_mask = self.neuron_types[self._weight_indices[0]] > 0
        self._weight_values = torch.where(
            exc_mask,
            self._weight_values.clamp(min=0.0, max=1.0),
            self._weight_values.clamp(min=-1.0, max=0.0)
        )

        self._eligibility *= getattr(self.config, 'eligibility_consume', 0.3)

        # Update weight representation — mark dirty, rebuild lazily in next forward()
        self._dense_weights_dirty = True

    def apply_surrogate_gradient_step(self, loss: torch.Tensor):
        """One BPTT step with Surrogate Gradients. R-STDP remains primary."""
        if loss.requires_grad:
            loss.backward()

    # ========================================================================
    # NEUROMODULATION
    # ========================================================================

    def set_neuromodulator(self, modulator: str, level: float):
        """Set neuromodulator level (0.0-1.0)."""
        level = max(0.0, min(1.0, level))
        if modulator in self.neuromod_levels:
            self.neuromod_levels[modulator] = level
            self._tau_dirty = True

    def get_neuromodulator_tau(self) -> torch.Tensor:
        """Compute current tau per neuron based on neuromodulator levels."""
        if not self.config.neuromod_enabled:
            return self._tau_base.clone()

        ne = self.neuromod_levels['ne']
        sht = self.neuromod_levels['5ht']
        ne_effect = self.neuromod_sensitivity[:, 0] * ne
        sht_effect = self.neuromod_sensitivity[:, 1] * sht
        tau = self._tau_base + self.config.delta_tau * (sht_effect - ne_effect)

        if self.protected_populations:
            protected_mask = self._get_protected_neuron_mask()
            tau[protected_mask] = self._tau_base[protected_mask]

        tau = tau.clamp(min=2.0, max=100.0)
        return tau

    # ========================================================================
    # HOMEOSTATIC PLASTICITY
    # ========================================================================

    def _homeostatic_update(self):
        """Adapt per-neuron thresholds toward target firing rate."""
        if self._homeostatic_step_count == 0:
            return

        actual_rates = self._spike_count_window / self._homeostatic_step_count
        target = self.config.target_firing_rate
        rate_error = actual_rates - target
        adaptation = 0.01 * rate_error

        if self.protected_populations:
            protected_mask = self._get_protected_neuron_mask()
            adaptation[protected_mask] = 0.0

        if self._izh_enabled.any():
            adaptation[self._izh_enabled] = 0.0

        self._thresholds = self._thresholds + adaptation
        self._thresholds = self._thresholds.clamp(min=0.3, max=3.0)

        # Selbstauskunft des Reglers (25.07.2026). Der gemessene Schwellenanstieg
        # in der Anfangsphase eines Laufes ist rund 6x schneller, als adaptation =
        # 0.01 * rate_error zulaesst (max 0.0095 pro Intervall, und das nur bei
        # Feuerrate 1.0). Die Plateauphase passt dagegen exakt zur Regel. Statt das
        # weiter zu erschliessen: aufzeichnen, was hier tatsaechlich angewandt wird.
        self._last_rate_err_mean = float(rate_error.mean())
        self._last_rate_err_max = float(rate_error.max())
        self._last_adapt_mean = float(adaptation.mean())
        self._last_adapt_max = float(adaptation.abs().max())
        self._homeo_updates += 1

        self._spike_count_window.zero_()
        self._homeostatic_step_count = 0

    # ========================================================================
    # ASTROCYTE GATE
    # ========================================================================

    def _astrocyte_update(self):
        """Update astrocyte calcium based on spike activity."""
        n = self.config.n_neurons
        cs = self._astro_cluster_size

        # v0.5.2: reuse _spike_float instead of self.spikes.float()
        spike_f = self._spike_float
        padded_len = self._n_astrocytes * cs
        if padded_len > n:
            padded = torch.zeros(padded_len, device=self.device, dtype=self.dtype)
            padded[:n] = spike_f
        else:
            padded = spike_f[:padded_len]

        cluster_spikes = padded.view(self._n_astrocytes, cs).sum(dim=1) / cs
        self._astro_calcium = self._astro_calcium + cluster_spikes * 0.1

        decay = 1.0 / self.config.astrocyte_tau_calcium
        self._astro_calcium = self._astro_calcium * (1.0 - decay)
        self._astro_calcium = self._astro_calcium.clamp(min=0.0, max=2.0)

    def _can_form_synapse(self, pre_neuron: int, post_neuron: int) -> bool:
        """Check if astrocyte gate allows synapse formation."""
        pre_cluster = pre_neuron // self._astro_cluster_size
        post_cluster = post_neuron // self._astro_cluster_size
        threshold = self.config.astrocyte_calcium_threshold
        return (self._astro_calcium[pre_cluster].item() > threshold and
                self._astro_calcium[post_cluster].item() > threshold)

    # ========================================================================
    # SYNAPTOGENESIS
    # ========================================================================

    def synaptogenesis_step(self):
        """Synaptogenesis: grow new synapses, prune unused ones."""
        if self._weight_values is None:
            return

        changed = False

        mask = self._weight_values.abs() > self.config.pruning_threshold
        if mask.sum() < self._n_synapses:
            self._weight_indices = self._weight_indices[:, mask]
            self._weight_values = self._weight_values[mask]
            self._eligibility = self._eligibility[mask]
            changed = True

        active_clusters = torch.where(
            self._astro_calcium > self.config.astrocyte_calcium_threshold
        )[0]

        if len(active_clusters) >= 2:
            n_new = min(self.config.synaptogenesis_max_new, len(active_clusters) * 10)
            cs = self._astro_cluster_size
            new_src = []
            new_tgt = []

            for _ in range(n_new):
                idx = torch.randint(0, len(active_clusters), (2,), device=self.device)
                c1 = active_clusters[idx[0]].item()
                c2 = active_clusters[idx[1]].item()
                if c1 == c2:
                    continue
                src = c1 * cs + torch.randint(0, min(cs, self.config.n_neurons - c1 * cs), (1,)).item()
                tgt = c2 * cs + torch.randint(0, min(cs, self.config.n_neurons - c2 * cs), (1,)).item()
                if src < self.config.n_neurons and tgt < self.config.n_neurons and src != tgt:
                    new_src.append(src)
                    new_tgt.append(tgt)

            if new_src:
                new_src_t = torch.tensor(new_src, device=self.device, dtype=torch.long)
                new_tgt_t = torch.tensor(new_tgt, device=self.device, dtype=torch.long)
                new_indices = torch.stack([new_src_t, new_tgt_t])
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

    def get_health(self) -> Dict:
        """Cheap scalar health snapshot of the substrate. Read-only, no side effects.

        Added 2026-07-25 (Logbuch #214). Firing rates were computed nowhere in the
        training loop and logged nowhere at all -- so a network in which 30-74 %% of
        the neurons were silent stayed invisible across every run. R-STDP needs
        coincidence and coincidence needs spikes; without this the substrate's
        state cannot be seen while working on the reward.

        Unlike get_state() this returns SCALARS only and skips the neuromodulator
        and tau bookkeeping, so it is cheap enough to call at FLOG cadence.

        Caveat: _spike_count_window is zeroed after every homeostatic interval, so
        the window is short right after a reset. 'window' is returned for exactly
        that reason -- at a small window the rates are noise.
        """
        w = self._homeostatic_step_count
        if w <= 0:
            return {'snn_rate_mean': 0.0, 'snn_rate_med': 0.0, 'snn_silent_frac': 0.0,
                    'snn_sat_frac': 0.0, 'snn_thr_med': float(self._thresholds.median()),
                    'snn_rate_window': 0}
        rates = self._spike_count_window / float(w)
        n = rates.numel()
        # Quantile statt nur Median: die Schwellen bilden wenige diskrete Cluster
        # (gemessen 25.07.: 4 Werte auf 535 Neuronen). Ein Median ueber eine solche
        # Verteilung SPRINGT, sobald ein Cluster die 50-%-Marke passiert -- er
        # verfolgt keine Population. Wer nur den Median loggt, liest Cluster-
        # Wechsel als Schwellenanstieg. Die Spannweite macht das unterscheidbar.
        thr = self._thresholds
        q = torch.quantile(thr.float(),
                           torch.tensor([0.1, 0.5, 0.9], device=thr.device))
        return {
            'snn_rate_mean': float(rates.mean()),
            'snn_rate_med': float(rates.median()),
            'snn_silent_frac': float((rates <= 0).sum()) / n,
            'snn_sat_frac': float((rates > 0.9).sum()) / n,
            'snn_thr_med': float(q[1]),
            'snn_thr_p10': float(q[0]),
            'snn_thr_p90': float(q[2]),
            'snn_thr_uniq': int(torch.unique(thr).numel()),
            'snn_rate_window': int(w),
            # Selbstauskunft des Reglers -- siehe _homeostatic_update
            'snn_homeo_n': int(getattr(self, '_homeo_updates', 0)),
            'snn_err_mean': float(getattr(self, '_last_rate_err_mean', 0.0)),
            'snn_err_max': float(getattr(self, '_last_rate_err_max', 0.0)),
            'snn_adapt_mean': float(getattr(self, '_last_adapt_mean', 0.0)),
            'snn_adapt_max': float(getattr(self, '_last_adapt_max', 0.0)),
            # Belohnungs-Baseline (Schultz): was WIRKLICH ins dw geht
            'snn_reward_ema': float(getattr(self, '_reward_ema', 0.0)),
            'snn_modulator': float(getattr(self, '_last_modulator', 0.0)),
            'snn_baseline_on': int(bool(getattr(self.config, 'reward_baseline', False))),
            # Welcher Zweig formt das Lernen? (siehe apply_rstdp)
            'snn_rstdp_calls': int(getattr(self, '_rstdp_calls', 0)),
            'snn_pe_branch_hits': int(getattr(self, '_pe_branch_hits', 0)),
            'snn_pe_in': float(getattr(self, '_last_pe', 0.0)),
            'snn_reward_in': float(getattr(self, '_last_reward_in', 0.0)),
            'snn_combined': float(getattr(self, '_last_combined', 0.0)),
            'snn_pe_blend': float(getattr(self.config, 'pe_blend', 0.9)),
        }

    def get_state(self) -> Dict:
        """Return current network state."""
        n = self.config.n_neurons
        exc_mask = self.neuron_types > 0
        inh_mask = ~exc_mask

        if self._homeostatic_step_count > 0:
            firing_rates = self._spike_count_window / self._homeostatic_step_count
        else:
            firing_rates = torch.zeros(n, device=self.device, dtype=self.dtype)

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
        """Activity of a named population."""
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
    # PERSISTENCE
    # ========================================================================

    def save(self, path: str):
        """Save SNN state."""
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
            'pre_trace': self._trace.cpu(),
            'post_trace': self._trace.cpu(),
            'thresholds': self._thresholds.cpu(),
            'astro_calcium': self._astro_calcium.cpu(),
            'spike_count_window': self._spike_count_window.cpu(),
            'homeostatic_step_count': self._homeostatic_step_count,
            'step_count': self.step_count,
            'izh_a': self._izh_a.cpu(),
            'izh_b': self._izh_b.cpu(),
            'izh_c': self._izh_c.cpu(),
            'izh_d': self._izh_d.cpu(),
            'izh_u': self._u.cpu(),
            'izh_enabled': self._izh_enabled.cpu(),
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load SNN state (backward-compatible with v0.4.x)."""
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
        # v0.5.2: load into single _trace (pre==post were always identical)
        self._trace = state['pre_trace'].to(self.device)
        self._pre_trace = self._trace
        self._post_trace = self._trace
        self._thresholds = state['thresholds'].to(self.device)
        self._astro_calcium = state['astro_calcium'].to(self.device)
        self._spike_count_window = state['spike_count_window'].to(self.device)
        self._homeostatic_step_count = state['homeostatic_step_count']
        self.step_count = state['step_count']

        if 'izh_a' in state:
            self._izh_a = state['izh_a'].to(self.device)
            self._izh_b = state['izh_b'].to(self.device)
            self._izh_c = state['izh_c'].to(self.device)
            self._izh_d = state['izh_d'].to(self.device)
            self._u = state['izh_u'].to(self.device)
            self._izh_enabled = state['izh_enabled'].to(self.device)

        self._n_synapses = self._weight_values.shape[0] if self._weight_values is not None else 0
        self._all_izh = bool(self._izh_enabled.all())
        self._rebuild_sparse_weights()

    # ========================================================================
    # POPULATION MANAGEMENT
    # ========================================================================

    def define_population(self, name: str, neuron_ids: torch.Tensor):
        """Define a named neuron population."""
        self.populations[name] = neuron_ids.to(self.device)

    def set_izhikevich_params(self, population: str,
                               a: float, b: float, c: float, d: float):
        """
        Set per-population Izhikevich (a,b,c,d) parameters.

        Standard parameter sets (Izhikevich 2003, Table 2):
          Regular Spiking (RS):      a=0.02, b=0.2,  c=-65, d=8
          Intrinsically Bursting:    a=0.02, b=0.2,  c=-55, d=4
          Chattering (CH):           a=0.02, b=0.2,  c=-50, d=2
          Low-Threshold Spiking:     a=0.02, b=0.25, c=-65, d=2
          Rebound Burst (DCN):       a=0.03, b=0.25, c=-52, d=0
          Fast Spiking (FS):         a=0.1,  b=0.2,  c=-65, d=2
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
        self._u[ids] = b * self.V[ids]
        # Update cached flag
        self._all_izh = bool(self._izh_enabled.all())

    def _get_protected_neuron_mask(self) -> torch.Tensor:
        """Boolean mask [n_neurons] where True = protected."""
        mask = torch.zeros(self.config.n_neurons, dtype=torch.bool, device=self.device)
        for pop_name in self.protected_populations:
            if pop_name in self.populations:
                mask[self.populations[pop_name]] = True
        return mask

    def _get_protected_synapse_mask(self) -> torch.Tensor:
        """Boolean mask [n_synapses] where True = protected."""
        if self._weight_indices is None or self._n_synapses == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        neuron_mask = self._get_protected_neuron_mask()
        pre_protected = neuron_mask[self._weight_indices[0]]
        post_protected = neuron_mask[self._weight_indices[1]]
        return pre_protected | post_protected

    def connect_populations(self, source: str, target: str,
                            prob: float = 0.05,
                            weight_range: Tuple[float, float] = (0.1, 0.5)):
        """Connect two populations with given probability."""
        if source not in self.populations or target not in self.populations:
            raise KeyError(f"Population not found. Available: {list(self.populations.keys())}")

        src_ids = self.populations[source]
        tgt_ids = self.populations[target]

        n_connections = int(len(src_ids) * len(tgt_ids) * prob)
        if n_connections == 0:
            return

        src_pick = src_ids[torch.randint(0, len(src_ids), (n_connections,), device=self.device)]
        tgt_pick = tgt_ids[torch.randint(0, len(tgt_ids), (n_connections,), device=self.device)]

        mask = src_pick != tgt_pick
        src_pick = src_pick[mask]
        tgt_pick = tgt_pick[mask]

        if len(src_pick) == 0:
            return

        n_actual = src_pick.shape[0]
        w_min, w_max = weight_range
        raw_weights = torch.rand(n_actual, device=self.device, dtype=self.dtype) * (w_max - w_min) + w_min
        signs = self.neuron_types[src_pick]
        new_weights = raw_weights * signs

        new_indices = torch.stack([src_pick, tgt_pick])

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
