"""
MH-FLOCKE — Neural Topology Helpers
=====================================
Shared functions for computing SNN population sizes.
No MuJoCo, no hardware dependencies — pure math.

Used by:
  - src/body/mujoco_creature.py (simulator)
  - scripts/freenove_bridge.py (Pi hardware)

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""


def compute_cerebellar_populations(n_hidden: int, n_actuators: int) -> dict:
    """
    Compute cerebellar population sizes scaled to available hidden neurons.
    
    For large networks (n_hidden >= 500): use standard sizes (GrC=4000, GoC=200).
    For small networks (n_hidden < 500): scale proportionally.
    
    The architecture is preserved — just smaller. Like a mouse cerebellum
    vs an elephant cerebellum: same cell types, same connectivity, different scale.
    
    Returns:
        dict with population sizes: n_granule, n_golgi, n_purkinje, n_dcn
    """
    n_purkinje = n_actuators * 2  # 2 per actuator (push/pull)
    n_dcn = n_actuators * 2       # same as PkC
    
    if n_hidden >= 500:
        # Standard: large cerebellum (Go2, Bommel)
        return {
            'n_granule': 4000,
            'n_golgi': 200,
            'n_purkinje': n_purkinje,
            'n_dcn': n_dcn,
        }
    
    # Scaled: small cerebellum (Freenove, micro robots)
    # Reserve space for PkC + DCN first, rest goes to GrC + GoC
    fixed_neurons = n_purkinje + n_dcn  # e.g. 24 + 24 = 48 for 12 actuators
    available = max(4, n_hidden - fixed_neurons)
    
    # Split available: 85% GrC (expansion), 15% GoC (inhibition)
    n_golgi = max(4, int(available * 0.15))
    n_granule = max(4, available - n_golgi)
    
    return {
        'n_granule': n_granule,
        'n_golgi': n_golgi,
        'n_purkinje': n_purkinje,
        'n_dcn': n_dcn,
    }
