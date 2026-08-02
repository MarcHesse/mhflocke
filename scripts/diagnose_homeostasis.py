"""
Is the threshold homeostasis already exhausted?

Background: snn_controller applies firing-rate homeostasis to the THRESHOLDS
only (`_homeostatic_update`, clamped to 0.3-3.0). There is no counterforce on
the WEIGHTS -- no synaptic scaling, no normalisation, and weight bounds only at
initialisation.

Hypothesis: the only effective protection against runaway potentiation is the
very short eligibility trace (`_trace_decay` 0.95, times 0.3 after every
`apply_rstdp`). If that holds, the brake on runaway activity is a side effect
rather than a design, and the trace must NOT be lengthened before synaptic
scaling exists.

A testable prediction that needs no training run:
  if the weights grow unchecked, the threshold homeostasis has to push back
  until it runs into the upper clamp at 3.0. If a sizeable share of the
  thresholds sits there, the homeostasis is exhausted.

Usage:
    py -3.11 scripts/diagnose_homeostasis.py <snn_state.pt> [<another.pt> ...]

Worth running across several runs of different length -- then it also shows
whether that share GROWS with runtime, which would be the actual evidence.
"""
import sys
import os
import glob

# The pickle contains objects from src.* -- when started from scripts/ only
# scripts/ is on the path, not the repo root. Without this: "No module named 'src'".
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import torch
except ImportError:
    sys.exit("torch not found -- run this in the training environment")

CLAMP_LO, CLAMP_HI = 0.3, 3.0
EPS = 1e-4


def find(state, *names):
    """Find a tensor in the state dict, whether flat or nested."""
    if not isinstance(state, dict):
        return None
    for n in names:
        if n in state:
            return state[n]
    for v in state.values():
        if isinstance(v, dict):
            hit = find(v, *names)
            if hit is not None:
                return hit
    return None


def report(path):
    print(f"\n=== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} ===")
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  not loadable: {e}")
        return

    th = find(state, 'thresholds', '_thresholds')
    if th is None:
        print("  no thresholds found in the state -- check the key names:")
        if isinstance(state, dict):
            print("  ", list(state.keys())[:20])
        return

    th = th.detach().float().flatten()
    n = th.numel()
    at_hi = int((th >= CLAMP_HI - EPS).sum())
    at_lo = int((th <= CLAMP_LO + EPS).sum())

    print(f"  neurons               : {n}")
    print(f"  threshold min/med/max : {th.min():.3f} / {th.median():.3f} / {th.max():.3f}")
    print(f"  at the UPPER clamp    : {at_hi} ({100.0*at_hi/n:.1f} %)   <- homeostasis exhausted")
    print(f"  at the LOWER clamp    : {at_lo} ({100.0*at_lo/n:.1f} %)")

    w = find(state, 'weight_values', '_weight_values', '_weights', 'weights', 'w')
    if w is not None and hasattr(w, 'detach'):
        w = w.detach().float().flatten()
        aw = w.abs()
        nw = w.numel()
        at_zero = int((aw <= 1e-6).sum())
        at_bound = int((aw >= 1.0 - 1e-4).sum())
        middle = nw - at_zero - at_bound
        print(f"  weights n/mean|w|/max : {nw} / {aw.mean():.4f} / {aw.max():.4f}")
        print(f"  weight norm (L2)      : {w.norm():.2f}")
        print(f"  at 0                  : {at_zero} ({100.0*at_zero/nw:.1f} %)")
        print(f"  at the bound |w|=1    : {at_bound} ({100.0*at_bound/nw:.1f} %)")
        print(f"  IN BETWEEN            : {middle} ({100.0*middle/nw:.1f} %)  <- usable range")
        # coarse distribution over |w|
        edges = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.001]
        hist = []
        for lo, hi in zip(edges, edges[1:]):
            hist.append(int(((aw >= lo) & (aw < hi)).sum()))
        print("  |w| histogram         : " +
              " ".join(f"{lo:.2f}-{hi:.2f}:{c}" for lo, hi, c in zip(edges, edges[1:], hist)))
        if middle / nw < 0.20:
            print("  >> BIMODAL: almost no weights between the bounds. Hard clamps without")
            print("     multiplicative scaling drive STDP weights to the edges. The network")
            print("     then has no range left to learn in -- independently of trace length")
            print("     and reward.")
    else:
        print("  weights not found in the state (check the keys)")

    # Threshold distribution: median == max in every run so far suggests a
    # degenerate distribution (most neurons sitting on ONE value).
    uniq, counts = torch.unique(th, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top = [(float(uniq[i]), int(counts[i])) for i in order[:5]]
    print("  thresholds: most common: " +
          " ".join(f"{v:.3f}x{c}" for v, c in top) + f"  ({uniq.numel()} distinct values)")

    # DECISIVE: the actual firing rates. Four distinct thresholds across 535
    # neurons, with 278 of them locked on ONE value, suggests those 278 share a
    # rate -- and the direction (1.000 -> 1.5+) says: above target. Suspicion:
    # permanently saturated, rate ~1.0.
    sc = find(state, 'spike_count_window')
    hsc = find(state, 'homeostatic_step_count')
    if sc is not None and hsc:
        try:
            hsc = int(hsc)
        except Exception:
            hsc = 0
    if sc is not None and hsc:
        rates = sc.detach().float().flatten() / float(hsc)
        tgt = 0.05
        print(f"\n  FIRING RATES (window {hsc} steps, target {tgt}):")
        print(f"    min/med/max         : {rates.min():.3f} / {rates.median():.3f} / {rates.max():.3f}")
        print(f"    silent (rate 0)     : {int((rates <= 0).sum())} ({100.0*int((rates<=0).sum())/rates.numel():.1f} %)")
        print(f"    saturated (>0.9)    : {int((rates > 0.9).sum())} ({100.0*int((rates>0.9).sum())/rates.numel():.1f} %)")
        print(f"    near target (0.02-0.1): {int(((rates>=0.02)&(rates<=0.10)).sum())}")
        if int((rates > 0.9).sum()) > 0.2 * rates.numel():
            print("    >> A large part of the network fires on practically every step.")
            print("       A permanently saturated population carries no information --")
            print("       the quiet variant of an epileptic state. The threshold")
            print("       homeostasis pushes back, but far too slowly.")
    else:
        print("\n  Firing rates cannot be reconstructed (spike_count_window/step_count")
        print("  missing, or the window had just been reset when this was saved).")

    print("\n  Reading:")
    if at_hi / n > 0.10:
        print("  >10 % at the upper clamp -- the threshold homeostasis is maxed out.")
        print("  It can no longer compensate further weight growth. The short eligibility")
        print("  trace is then the only remaining protection, and lengthening it without")
        print("  synaptic scaling would be reckless.")
    elif at_hi / n > 0.01:
        print("  First neurons sit at the clamp -- exhaustion beginning. Compare across")
        print("  several run lengths: does the share grow?")
    else:
        print("  The clamp is practically unused -- the hypothesis is NOT confirmed in")
        print("  this form. Then the network stays stable by some other means, and what")
        print("  that is remains to be established.")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, '**', 'snn_state.pt'),
                                          recursive=True)))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"skipped (does not exist): {arg}")
    if not paths:
        sys.exit("no snn_state.pt found")
    for p in paths:
        report(p)
    if len(paths) > 1:
        print("\nWhat matters is the TREND across run lengths, not a single value.")


if __name__ == '__main__':
    main()
