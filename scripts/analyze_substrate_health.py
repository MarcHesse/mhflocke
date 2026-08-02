"""
Substrate health over the course of a run.

Shows what no run made visible before: how many neurons fire at all, how the
rate compares to the homeostatic target, and where the thresholds move.

The finding that made this necessary: the threshold homeostasis is asymmetric.
With `rate_error = actual - target` and a target of 0.05, the error reaches
+0.95 upward but only -0.05 downward, because a firing rate cannot go negative.
A ratio of 19:1. The controller corrects overactivity quickly and underactivity
barely, so its fixed point sits BELOW the target. Measured on two saved
networks: thresholds at 1.52 while rates were ten times under target, with
30-74 % of neurons silent in the measurement window.

What to expect from a fresh run if that reading is right:
  the rate starts ABOVE the target, the threshold rises, the rate drops below
  the target and stays there while the threshold stays up.
Seeing that sequence would show the ratchet in the run itself, rather than
inferring it from an end state.

Usage:
    py -3.11 scripts/analyze_substrate_health.py <flog> [<flog2> ...]
"""
import sys
import struct

try:
    import msgpack
except ImportError:
    sys.exit("msgpack not installed")

FRAME_TRAINING = 0x02
TARGET = 0.05          # snn_controller SNNConfig.target_firing_rate
MIN_WINDOW = 100       # below this the counting window is too short -> noise.
                       # NOTE: measured on a --snn-substeps 10 run the window
                       # never exceeds 190 and resets there, so the homeostatic
                       # interval in practice is ~200 SNN steps, not the 1000 in
                       # SNNConfig. A default of 200 filtered out every frame.
BUCKETS = 10


def parse(path):
    rows = []
    with open(path, 'rb') as f:
        if f.read(4) != b'FLOG':
            sys.exit(f"{path}: not a FLOG")
        f.read(3)
        mlen = struct.unpack('<I', f.read(4))[0]
        f.read(mlen)
        while True:
            h = f.read(13)
            if len(h) < 13:
                break
            _ts, ftype, plen = struct.unpack('<dBI', h)
            payload = f.read(plen)
            if len(payload) < plen or ftype != FRAME_TRAINING:
                continue
            try:
                d = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            except Exception:
                continue
            if isinstance(d, dict):
                rows.append(d)
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def report(path):
    print(f"\n=== {path} ===")
    rows = parse(path)
    if not rows:
        print("  no training frames")
        return

    have = sum(1 for r in rows if 'snn_rate_med' in r)
    if have == 0:
        print(f"  {len(rows)} frames, but NO substrate telemetry.")
        print("  Run predates train_baby 0.8.3 / snn_controller 0.5.3.")
        return

    # Look at the window sizes BEFORE filtering on them -- otherwise the script
    # throws the data away and you cannot see why.
    wins = sorted(int(r.get('snn_rate_window', -1)) for r in rows if 'snn_rate_med' in r)
    uniq = sorted(set(wins))
    print(f"  counting window (snn_rate_window): min {wins[0]} / max {wins[-1]} "
          f"/ {len(uniq)} distinct values")
    print(f"    most common: " + " ".join(
        f"{v}x{wins.count(v)}" for v in sorted(uniq, key=lambda v: -wins.count(v))[:6]))
    sample = next((r for r in rows if 'snn_rate_med' in r), None)
    if sample:
        print("    sample frame: " + " ".join(
            f"{k}={sample[k]}" for k in sorted(sample) if k.startswith('snn_')))

    min_window = MIN_WINDOW
    if '--min-window' in sys.argv:
        min_window = int(sys.argv[sys.argv.index('--min-window') + 1])

    # Drop frames whose counting window is too short -- their rates are noise
    usable = [r for r in rows
              if r.get('snn_rate_window', 0) >= min_window and 'snn_rate_med' in r]
    print(f"  frames total          : {len(rows)}")
    print(f"  with telemetry        : {have}")
    print(f"  of those window >= {min_window}: {len(usable)}")
    if not usable:
        print(f"  No frame reaches a window of {min_window}. Largest seen: {wins[-1]}.")
        print("  Lower it with --min-window <n>. If the largest window is well below")
        print("  SNNConfig.homeostatic_interval, the reset happens more often than the")
        print("  configured interval -- worth checking before reading the rates.")
        return

    steps = [int(r.get('step', i)) for i, r in enumerate(usable)]
    print(f"  steps                 : {steps[0]} .. {steps[-1]}")

    print(f"\n  Over time ({BUCKETS} segments, target rate {TARGET}):")
    print(f"  {'step':>10} {'rate avg':>9} {'silent %':>8} {'thr p10':>8} "
          f"{'thr p90':>8} {'#upd':>6} {'err avg':>9} {'adapt avg':>10} {'adapt max':>10}")
    size = max(1, len(usable) // BUCKETS)
    for b in range(0, len(usable), size):
        chunk = usable[b:b + size]
        if not chunk:
            continue
        print(f"  {int(chunk[-1].get('step', 0)):>10}"
              f" {mean([float(r.get('snn_rate_mean', 0)) for r in chunk]):>9.4f}"
              f" {100*mean([float(r.get('snn_silent_frac', 0)) for r in chunk]):>8.1f}"
              f" {mean([float(r.get('snn_thr_p10', 0)) for r in chunk]):>8.3f}"
              f" {mean([float(r.get('snn_thr_p90', 0)) for r in chunk]):>8.3f}"
              f" {mean([float(r.get('snn_homeo_n', 0)) for r in chunk]):>6.0f}"
              f" {mean([float(r.get('snn_err_mean', 0)) for r in chunk]):>9.4f}"
              f" {mean([float(r.get('snn_adapt_mean', 0)) for r in chunk]):>10.6f}"
              f" {mean([float(r.get('snn_adapt_max', 0)) for r in chunk]):>10.6f}")

    # The decisive check: does the observed threshold change match what the
    # homeostatic controller reports it applied? If the controller applied
    # nothing and the thresholds moved anyway, something else writes to them.
    n0 = float(usable[0].get('snn_homeo_n', 0))
    n1 = float(usable[-1].get('snn_homeo_n', 0))
    p90_0 = float(usable[0].get('snn_thr_p90', 0))
    p90_1 = float(usable[-1].get('snn_thr_p90', 0))
    adapt_max = max(abs(float(r.get('snn_adapt_max', 0))) for r in usable)
    dn = n1 - n0
    if dn > 0:
        print(f"\n  CROSS-CHECK:")
        print(f"    homeostatic updates in the run: {dn:.0f}")
        print(f"    largest adjustment it applied : {adapt_max:.6f}")
        print(f"    thr p90 {p90_0:.3f} -> {p90_1:.3f}  = {p90_1-p90_0:+.3f}")
        print(f"    per update                    : {(p90_1-p90_0)/dn:+.6f}")
        print(f"    maximum per the rule (0.01*0.95): 0.009500")
        if adapt_max < 1e-9 and abs(p90_1 - p90_0) > 0.01:
            print("    >> The homeostasis applied NOTHING (adapt = 0 throughout) and the")
            print("       thresholds moved anyway. Something else writes to them.")
            print("       Known writers besides _homeostatic_update:")
            print("         src/brain/snn_builder.py       - per-population init values")
            print("         src/brain/cerebellar_learning.py - granule cell thresholds")
            print("       Note that _homeostatic_update zeroes the adjustment for")
            print("       Izhikevich neurons, and on the Bittle every population is")
            print("       Izhikevich -- so the rate homeostasis is inert by construction.")
        elif abs((p90_1 - p90_0) / dn) > 0.0095:
            print("    >> The threshold moves FASTER than the rule allows. Either")
            print("       something else writes to it, or the interval is shorter than")
            print("       assumed. Compare snn_homeo_n against the step count to get")
            print("       the actual interval.")
        else:
            print("    >> Within the rule, and the controller did apply adjustments.")

    first, last = usable[0], usable[-1]
    r0, r1 = float(first['snn_rate_med']), float(last['snn_rate_med'])
    t0, t1 = float(first.get('snn_thr_med', 0)), float(last.get('snn_thr_med', 0))
    s1 = 100 * float(last.get('snn_silent_frac', 0))

    print("\n  Reading:")
    print(f"    rate  {r0:.4f} -> {r1:.4f}   (target {TARGET})")
    print(f"    threshold median {t0:.3f} -> {t1:.3f}")
    print(f"    silent at the end: {s1:.1f} %")
    uq = float(last.get('snn_thr_uniq', 0))
    if uq and uq < 20:
        print(f"    NOTE: only {uq:.0f} distinct threshold values. With so few clusters")
        print("    the median jumps between them and is NOT a usable trend measure --")
        print("    read p10/p90, not p50.")
    if r1 < TARGET * 0.5:
        print("    >> The network runs persistently below its set point. R-STDP needs")
        print("       coincidence and coincidence needs spikes. Whether homeostasis is")
        print("       the cause is decided by the p10/p90 trend, not by p50.")
    elif abs(r1 - TARGET) < TARGET * 0.5:
        print("    >> Rate close to target. Homeostasis regulates cleanly in this run.")
    else:
        print("    >> No clear picture.")


def main():
    # Keep flags out of the file list -- otherwise '--min-window' ends up as a
    # path in parse() and the script dies after the last report.
    args = sys.argv[1:]
    paths = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == '--min-window':
            skip = True
            continue
        if a.startswith('--'):
            continue
        paths.append(a)
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        report(p)


if __name__ == '__main__':
    main()
