"""
MH-FLOCKE — Neural Topology Helpers
=====================================
Shared functions for computing SNN population sizes.
No MuJoCo, no hardware dependencies — pure math.

v0.7.1: Continuous scaling (no n_hidden>=500 cliff).
        30% motor hidden budget, 70% cerebellum.
        Same formula for all n_hidden values.

Used by:
  - src/body/mujoco_creature.py (simulator)
  - scripts/freenove_bridge.py (Pi hardware)

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

__version__ = "0.7.1"     # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 19          # mh-logbuch module entry
__status__  = "active"    # active | veraltet | neu


def compute_cerebellar_populations(n_hidden: int, n_actuators: int) -> dict:
    """
    Compute cerebellar population sizes scaled to available hidden neurons.

    v0.7.1: Continuous scaling for ALL n_hidden values. No more cliff
    at n_hidden=500 that jumped from ~560 to 4308 total neurons.
    The same formula applies whether n_hidden is 172 or 5000.

    Layout of hidden neurons:
      - PkC + DCN: fixed at 2*n_actuators each (output interface)
      - Remaining: 70% cerebellum (GrC + GoC), 30% motorcortex (MH)
      - GrC:GoC ratio within cerebellum: 85:15

    Returns:
        dict with population sizes: n_granule, n_golgi, n_purkinje, n_dcn
        (Motor hidden = n_hidden - sum of these four)
    """
    n_purkinje = n_actuators * 2  # 2 per actuator (push/pull)
    n_dcn = n_actuators * 2       # same as PkC

    # Reserve space for PkC + DCN first (fixed output interface)
    fixed_neurons = n_purkinje + n_dcn  # e.g. 24 + 24 = 48 for 12 actuators
    available = max(4, n_hidden - fixed_neurons)

    # Reserve 30% as free hidden layer (motorcortex / motor_hidden)
    free_hidden_ratio = 0.30
    cerebellum_budget = max(4, int(available * (1.0 - free_hidden_ratio)))

    # Split cerebellum budget: 85% GrC (expansion), 15% GoC (inhibition)
    n_golgi = max(4, int(cerebellum_budget * 0.15))
    n_granule = max(4, cerebellum_budget - n_golgi)

    return {
        'n_granule': n_granule,
        'n_golgi': n_golgi,
        'n_purkinje': n_purkinje,
        'n_dcn': n_dcn,
    }
