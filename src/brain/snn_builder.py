"""
MH-FLOCKE — Shared SNN Builder (Issue #159)
============================================
Single source of truth for SNN topology construction.

Problem (#159): the simulator (src/body/mujoco_creature.py) and the WiFi
bridge (scripts/bridge_bittle_wifi.py) built *different* SNN topologies from
the same profile. A brain.pt trained in simulation could therefore never be
loaded onto hardware — population ID ranges and connectivity did not match.

Fix: both sides call build_snn_from_profile(). The wiring below is an EXACT
port of MuJoCoCreatureBuilder.build()'s SNN-construction block (v0.5.2),
kept in the same order so the RNG stream — and thus the produced weights for a
given seed — are identical to the simulator.

This module is intentionally free of MuJoCo / hardware dependencies so the
bridge can import it. n_actuators is passed in instead of reading world.n_actuators.

Used by:
  - src/body/mujoco_creature.py (simulator)            [after refactor]
  - scripts/bridge_bittle_wifi.py (Bittle WiFi bridge) [after refactor]

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

__version__ = "1.0"       # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 18          # mh-logbuch module entry
__status__  = "neu"       # active | veraltet | neu

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from src.brain.snn_controller import SNNController, SNNConfig
from src.brain.topology import compute_cerebellar_populations


@dataclass
class BuiltSNN:
    """Result of build_snn_from_profile()."""
    snn: SNNController
    pops: Dict[str, torch.Tensor]   # population name -> neuron id tensor
    meta: Dict = field(default_factory=dict)  # sizes + cerebellar connectivity params


def build_snn_from_profile(profile: dict,
                           n_actuators: int,
                           device: str = "cpu",
                           izh: bool = True,
                           protect_cerebellum: bool = True,
                           verbose: bool = True) -> BuiltSNN:
    """
    Build the full cerebellar + motor-cortex SNN from a creature profile.

    This reproduces the simulator topology exactly: same population sizes,
    same neuron ID ordering (input, output, GrC, GoC, PkC, DCN, MH), same
    connectivity sequence, same per-population Izhikevich/threshold/tau values.

    Args:
        profile:    creature profile dict; uses profile['snn'] with keys
                    n_input, n_hidden, n_output, motor_hidden_ratio.
        n_actuators: number of actuated joints (Bittle: 8). Drives PkC/DCN
                    sizing and the bilateral symmetry leg pairing.
        device:     'cpu' or 'cuda'.
        izh:        enable per-population Izhikevich dynamics (sim default True).
        protect_cerebellum: protect cerebellar populations from R-STDP
                    (sim default True).
        verbose:    print the topology summary (matches the simulator's log line).

    Returns:
        BuiltSNN(snn, pops, meta)
    """
    snn_profile = profile.get("snn", {}) if profile else {}
    n_input = snn_profile.get("n_input")
    n_hidden = snn_profile.get("n_hidden")
    n_output = snn_profile.get("n_output")
    mh_ratio = snn_profile.get("motor_hidden_ratio", 0.30)

    if n_input is None or n_hidden is None or n_output is None:
        raise ValueError(
            "profile['snn'] must define n_input, n_hidden and n_output "
            f"(got n_input={n_input}, n_hidden={n_hidden}, n_output={n_output})"
        )

    # --- Population sizes (shared formula in topology.py) ---
    cb_pops = compute_cerebellar_populations(n_hidden, n_actuators)
    n_granule = cb_pops["n_granule"]
    n_golgi = cb_pops["n_golgi"]
    n_purkinje = cb_pops["n_purkinje"]
    n_dcn = cb_pops["n_dcn"]

    n_cerebellum = n_granule + n_golgi + n_purkinje + n_dcn
    n_motor_hidden = max(0, n_hidden - n_cerebellum) if mh_ratio > 0 else 0

    total_neurons = (n_input + n_output + n_granule + n_golgi
                     + n_purkinje + n_dcn + n_motor_hidden)

    # --- Neuron ID ranges (ORDER IS CONTRACTUAL — do not reorder) ---
    grc_start = n_input + n_output
    goc_start = grc_start + n_granule
    pkc_start = goc_start + n_golgi
    dcn_start = pkc_start + n_purkinje
    motor_hidden_start = dcn_start + n_dcn

    # --- Cerebellar connectivity params (mirrors CerebellarConfig scaling) ---
    grc_goc_prob = min(0.3, 0.05 * (4000 / max(n_granule, 1)))
    pf_pkc_prob = min(0.8, 0.4 * (4000 / max(n_granule, 1)))   # used by CerebellarLearning, not here
    mf_per_granule = min(4, max(2, n_input // max(n_granule, 1)))

    if verbose:
        print(f'  SNN Topology: {total_neurons} neurons '
              f'({n_input}i + {n_output}o + {n_granule}GrC + {n_golgi}GoC + '
              f'{n_purkinje}PkC + {n_dcn}DCN + {n_motor_hidden}MH)')
        print(f'  Topology: {n_hidden} hidden '
              f'(GrC:{n_granule} GoC:{n_golgi} PkC:{n_purkinje} '
              f'DCN:{n_dcn} MH:{n_motor_hidden})')

    # --- SNN Controller (first RNG consumer: neuromod_sensitivity + randperm) ---
    snn = SNNController(SNNConfig(
        n_neurons=total_neurons,
        connectivity_prob=0.0,
        homeostatic_interval=200,
        device=device,
    ))

    # === Populations — Cerebellar Architecture ===
    mf_ids = torch.arange(0, n_input)
    snn.define_population('input', mf_ids)
    snn.define_population('mossy_fibers', mf_ids)

    out_ids = torch.arange(n_input, n_input + n_output)
    snn.define_population('output', out_ids)

    grc_ids = torch.arange(grc_start, grc_start + n_granule)
    snn.define_population('granule_cells', grc_ids)

    goc_ids = torch.arange(goc_start, goc_start + n_golgi)
    snn.define_population('golgi_cells', goc_ids)

    pkc_ids = torch.arange(pkc_start, pkc_start + n_purkinje)
    snn.define_population('purkinje_cells', pkc_ids)

    dcn_ids = torch.arange(dcn_start, dcn_start + n_dcn)
    snn.define_population('dcn', dcn_ids)

    if n_motor_hidden > 0:
        mh_ids = torch.arange(motor_hidden_start, motor_hidden_start + n_motor_hidden)
        snn.define_population('motor_hidden', mh_ids)
        all_hidden_ids = torch.cat([grc_ids, mh_ids])
        snn.define_population('hidden', all_hidden_ids)
    else:
        mh_ids = torch.tensor([], dtype=torch.long)
        snn.define_population('hidden', grc_ids)

    # === Connectivity — biologically structured (exact sim order) ===
    # MF -> GrC: each GrC receives mf_per_granule random MF inputs.
    # This directly initialises the weight arrays (connectivity_prob was 0).
    n_mf = len(mf_ids)
    mf_per_grc = mf_per_granule
    src_list = []
    tgt_list = []
    for g in range(n_granule):
        mf_choices = torch.randint(0, n_mf, (mf_per_grc,))
        for m in mf_choices:
            src_list.append(mf_ids[m].item())
            tgt_list.append(grc_ids[g].item())
    if src_list:
        src_t = torch.tensor(src_list, device=device)
        tgt_t = torch.tensor(tgt_list, device=device)
        w = torch.rand(len(src_list), device=device) * 1.0 + 1.0
        new_idx = torch.stack([src_t, tgt_t])
        snn._weight_indices = new_idx
        snn._weight_values = w
        snn._eligibility = torch.zeros(len(src_list), device=device)

    # GrC -> GoC (excitatory)
    snn.connect_populations('granule_cells', 'golgi_cells',
                            prob=grc_goc_prob, weight_range=(0.3, 0.8))

    # GoC made inhibitory, then GoC -> GrC
    snn.neuron_types[goc_ids] = -1.0
    snn.connect_populations('golgi_cells', 'granule_cells',
                            prob=0.02, weight_range=(0.2, 0.4))

    # MF -> GoC (scaled for small input populations)
    _mf_goc_prob = max(0.1, 3.0 / max(1, len(mf_ids)))
    snn.connect_populations('mossy_fibers', 'golgi_cells',
                            prob=_mf_goc_prob, weight_range=(0.5, 1.0))

    # GrC -> output (scaled prob for small populations)
    _grc_out_prob = max(0.02, 3.0 / max(1, n_granule))
    snn.connect_populations('granule_cells', 'output',
                            prob=_grc_out_prob, weight_range=(0.3, 0.8))

    # Direct input -> output (corticospinal fast path)
    _in_out_prob = max(0.2, 5.0 / max(1, len(mf_ids)))
    snn.connect_populations('input', 'output',
                            prob=_in_out_prob, weight_range=(0.8, 1.5))

    # Motor-hidden connectivity (the R-STDP-modifiable pathway)
    if n_motor_hidden > 0:
        _mf_mh_prob = max(0.1, 5.0 / max(1, len(mf_ids)))
        snn.connect_populations('mossy_fibers', 'motor_hidden',
                                prob=_mf_mh_prob, weight_range=(0.5, 1.5))
        _mh_out_prob = max(0.15, 5.0 / max(1, n_motor_hidden))
        snn.connect_populations('motor_hidden', 'output',
                                prob=_mh_out_prob, weight_range=(0.8, 1.5))

        # Bilateral symmetry for MH->Output weights (Issue #145).
        # Average MH->Output weights between bilateral leg pairs to avoid the
        # L/R asymmetry that R-STDP amplifies into drift.
        _jpleg = n_actuators // 4
        if _jpleg == 3:    # 12-DOF: FL<->FR, RL<->RR
            _bilateral_pairs = [(0, 3), (1, 4), (2, 5),
                                (6, 9), (7, 10), (8, 11)]
        elif _jpleg == 2:  # 8-DOF: RF<->LF, RR<->LR
            _bilateral_pairs = [(0, 2), (1, 3),
                                (4, 6), (5, 7)]
        else:
            _bilateral_pairs = []
        _out_start = out_ids[0].item()
        _n_per_joint = max(1, n_output // n_actuators)
        _idx = snn._weight_indices
        _w = snn._weight_values
        _mh_set = set(mh_ids.tolist())
        _is_mh_source = torch.tensor(
            [s.item() in _mh_set for s in _idx[0]], device=device)
        for _left_j, _right_j in _bilateral_pairs:
            _left_neurons = list(range(_out_start + _left_j * _n_per_joint,
                                       _out_start + (_left_j + 1) * _n_per_joint))
            _right_neurons = list(range(_out_start + _right_j * _n_per_joint,
                                        _out_start + (_right_j + 1) * _n_per_joint))
            for _ln, _rn in zip(_left_neurons, _right_neurons):
                _left_mask = (_idx[1] == _ln) & _is_mh_source
                _right_mask = (_idx[1] == _rn) & _is_mh_source
                if _left_mask.any() and _right_mask.any():
                    _avg = (_w[_left_mask].mean() + _w[_right_mask].mean()) / 2.0
                    _w[_left_mask] = _avg
                    _w[_right_mask] = _avg
        if verbose:
            print(f'  MH->Output bilateral symmetry enforced '
                  f'({len(_bilateral_pairs)} joint pairs)')

        # motor_hidden recurrent (pattern memory)
        snn.connect_populations('motor_hidden', 'motor_hidden',
                                prob=0.1, weight_range=(0.2, 0.6))

    # === Per-population membrane time constants ===
    snn._tau_base[grc_ids] = 5.0
    snn._tau_base[goc_ids] = 20.0
    snn._tau_base[pkc_ids] = 15.0
    snn._tau_base[dcn_ids] = 10.0
    if n_motor_hidden > 0:
        snn._tau_base[mh_ids] = 10.0

    # === Thresholds ===
    snn._thresholds[grc_ids] = 0.5
    snn._thresholds[goc_ids] = 0.5
    snn._thresholds[pkc_ids] = 1.0
    snn._thresholds[dcn_ids] = 1.0
    snn._thresholds[out_ids] = 0.4
    if n_motor_hidden > 0:
        snn._thresholds[mh_ids] = 0.5
    snn._hidden_tonic_current = 0.015

    # === Per-population Izhikevich parameters (Issue #104) ===
    if izh:
        snn.set_izhikevich_params('granule_cells', a=0.02, b=0.2, c=-65, d=8)   # RS
        snn.set_izhikevich_params('golgi_cells',   a=0.02, b=0.2, c=-55, d=4)   # IB
        snn.set_izhikevich_params('purkinje_cells', a=0.02, b=0.2, c=-50, d=2)  # CH
        snn.set_izhikevich_params('dcn',           a=0.03, b=0.25, c=-52, d=0)  # Rebound
        if n_motor_hidden > 0:
            snn.set_izhikevich_params('motor_hidden', a=0.02, b=0.2, c=-65, d=8)  # RS
        snn.set_izhikevich_params('output', a=0.02, b=0.2, c=-65, d=8)  # RS
        snn.set_izhikevich_params('input',  a=0.1,  b=0.2, c=-65, d=2)  # FS

    # Protect cerebellar populations from R-STDP
    if protect_cerebellum:
        snn.protected_populations = {
            'mossy_fibers', 'granule_cells', 'golgi_cells',
            'purkinje_cells', 'dcn',
        }

    snn._rebuild_sparse_weights()

    pops = {
        'input': mf_ids, 'mossy_fibers': mf_ids, 'output': out_ids,
        'granule_cells': grc_ids, 'golgi_cells': goc_ids,
        'purkinje_cells': pkc_ids, 'dcn': dcn_ids, 'motor_hidden': mh_ids,
    }
    meta = {
        'n_input': n_input, 'n_output': n_output, 'n_hidden': n_hidden,
        'n_granule': n_granule, 'n_golgi': n_golgi, 'n_purkinje': n_purkinje,
        'n_dcn': n_dcn, 'n_motor_hidden': n_motor_hidden,
        'total_neurons': total_neurons, 'n_actuators': n_actuators,
        'mh_ratio': mh_ratio,
        'grc_start': grc_start, 'goc_start': goc_start, 'pkc_start': pkc_start,
        'dcn_start': dcn_start, 'motor_hidden_start': motor_hidden_start,
        'grc_goc_prob': grc_goc_prob, 'pf_pkc_prob': pf_pkc_prob,
        'mf_per_granule': mf_per_granule,
    }
    return BuiltSNN(snn=snn, pops=pops, meta=meta)
