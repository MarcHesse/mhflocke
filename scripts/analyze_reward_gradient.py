"""
Is there a gradient at all?

Hypothesis: the SNN does not learn because the reward signal is close to
constant across the whole run. R-STDP shapes nothing when R does not change --
not because the mechanism is broken, but because a constant signal carries no
direction. Competence 1.000, zero falls, an even gait straight ahead on flat
ground: vestibular satisfied, proprioceptive satisfied, prediction error
minimal (a straight line on flat terrain is maximally predictable). None of
that rewards speed.

This analysis tests the hypothesis on three questions:

  1. SPREAD    -- how much does the learning signal vary at all?
                  No spread, no gradient. This is the knockout criterion.
  2. STRUCTURE -- is the variation signal or noise? Lag-one autocorrelation:
                  near zero means white noise, which carries nothing learnable.
  3. COUPLING  -- does walking faster pay off? Correlation between instantaneous
                  speed and reward. If it is ~0 there is no reason to get
                  faster, and the robot stays at 2.7 cm/s however long it trains.

Usage:
    py -3.11 scripts/analyze_reward_gradient.py <flog> [<flog2> ...]
"""
import sys
import math
import struct

try:
    import msgpack
except ImportError:
    sys.exit("msgpack not installed")

FRAME_TRAINING = 0x02
SIM_DT = 0.002
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


def stats(v):
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / n
    return m, math.sqrt(var), min(v), max(v)


def corr(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 1e-12 and db > 1e-12 else 0.0


def report(path):
    print(f"\n=== {path} ===")
    rows = parse(path)
    key = 'learning_signal'
    have = [r for r in rows if key in r]
    if not have:
        print(f"  '{key}' missing from the FLOG -- check the field name.")
        avail = sorted({k for r in rows[:50] for k in r})
        print("  fields present:", ", ".join(avail[:40]))
        return

    R = [float(r[key]) for r in have]
    steps = [int(r.get('step', i)) for i, r in enumerate(have)]
    m, sd, lo, hi = stats(R)

    print(f"  frames with reward    : {len(R)}  (steps {steps[0]}..{steps[-1]})")
    print(f"  R  mean / sd          : {m:.6f} / {sd:.6f}")
    print(f"  R  min / max / range  : {lo:.6f} / {hi:.6f} / {hi-lo:.6f}")
    rel = sd / abs(m) if abs(m) > 1e-12 else float('inf')
    print(f"  coefficient of variation: {rel:.4f}   (sd / |mean|)")

    # 2. Structure: lag-one autocorrelation
    ac = corr(R[:-1], R[1:])
    print(f"  autocorrelation lag-1 : {ac:+.4f}")

    # 3. Coupling to speed
    v = []
    Rv = []
    if all('pos_x' in r and 'pos_y' in r for r in have[:5]):
        for i in range(1, len(have)):
            if 'pos_x' not in have[i] or 'pos_x' not in have[i-1]:
                continue
            d = math.hypot(float(have[i]['pos_x']) - float(have[i-1]['pos_x']),
                           float(have[i]['pos_y']) - float(have[i-1]['pos_y']))
            dt = (steps[i] - steps[i-1]) * SIM_DT
            if dt > 0:
                v.append(d / dt)
                Rv.append(R[i])
    if v:
        cv = corr(v, Rv)
        mv, sdv, _, _ = stats(v)
        print(f"  v mean / sd           : {mv:.4f} / {sdv:.4f} m/s")
        print(f"  corr(v, R)            : {cv:+.4f}   <- does going faster pay?")
    else:
        cv = None

    # Over time: does the reward improve?
    print(f"\n  Over time ({BUCKETS} segments):")
    print(f"  {'step':>10} {'R mean':>10} {'R sd':>10}")
    size = max(1, len(R) // BUCKETS)
    for b in range(0, len(R), size):
        chunk = R[b:b + size]
        if not chunk:
            continue
        cm, csd, _, _ = stats(chunk)
        print(f"  {steps[min(b+size, len(R))-1]:>10} {cm:>10.6f} {csd:>10.6f}")

    # CONTROL: learning_signal is the RAW R going into apply_rstdp. The baseline
    # is subtracted inside it. What actually reaches dw is snn_modulator. Without
    # this comparison an A/B on the baseline says nothing about its effect.
    mod = [float(r['snn_modulator']) for r in rows if 'snn_modulator' in r]
    ema = [float(r['snn_reward_ema']) for r in rows if 'snn_reward_ema' in r]
    on = next((int(r['snn_baseline_on']) for r in rows if 'snn_baseline_on' in r), None)
    if mod:
        mm, msd, mlo, mhi = stats(mod)
        pos = 100.0 * sum(1 for x in mod if x > 0) / len(mod)
        print(f"\n  MODULATOR (what actually reaches dw):")
        print(f"    baseline active     : {'YES' if on else 'no'}")
        print(f"    mean / sd           : {mm:+.6f} / {msd:.6f}")
        print(f"    min / max           : {mlo:+.6f} / {mhi:+.6f}")
        print(f"    share positive      : {pos:.1f} %   <- 50 % = zero-mean")
        if ema:
            em, _, elo, ehi = stats(ema)
            print(f"    E[R] mean/min/max   : {em:.4f} / {elo:.4f} / {ehi:.4f}")
        if on and abs(mm) > 0.05:
            print("    >> Baseline is on, but the modulator is NOT zero-mean. The EMA is")
            print("       then trailing the signal instead of centring it -- alpha too")
            print("       small, or apply_rstdp called less often than assumed.")
        elif on and pos > 60:
            print("    >> Modulator centred but still mostly positive: skewed")
            print("       distribution. Consider a median baseline instead of a mean.")
    else:
        print("\n  snn_modulator missing from the FLOG -- run predates snn_controller 0.5.3.")

    # WHICH BRANCH shapes the learning? apply_rstdp uses (1-w)*R + w*(-PE) above
    # the PE threshold and R alone below it. If the PE branch fires almost always,
    # the intrinsic reward determines only (1-w) of the learning -- and then no
    # curiosity drive can shape behaviour, however good it is.
    calls = [int(r['snn_rstdp_calls']) for r in rows if 'snn_rstdp_calls' in r]
    hits = [int(r['snn_pe_branch_hits']) for r in rows if 'snn_pe_branch_hits' in r]
    if calls and hits:
        dc = calls[-1] - calls[0]
        dh = hits[-1] - hits[0]
        print(f"\n  LEARNING PATH BRANCH:")
        print(f"    apply_rstdp calls   : {dc}")
        print(f"    of those PE branch  : {dh}  ({100.0*dh/dc if dc else 0:.1f} %)")
        pe = [abs(float(r['snn_pe_in'])) for r in rows if 'snn_pe_in' in r]
        if pe:
            pm, psd, plo, phi = stats(pe)
            print(f"    |PE| mean/min/max   : {pm:.4f} / {plo:.4f} / {phi:.4f}  (threshold 0.05)")
        rin = [float(r['snn_reward_in']) for r in rows if 'snn_reward_in' in r]
        # Do NOT hardcode the weighting -- it is settable via --pe-blend. A fixed
        # 0.1/0.9 reports the exact opposite of the truth at w=0.3.
        w = next((float(r['snn_pe_blend']) for r in rows if 'snn_pe_blend' in r), None)
        if w is None:
            w = 0.9
            print(f"    pe_blend            : not logged, assuming default {w}")
        else:
            print(f"    pe_blend            : {w:.2f}")
        if pe and rin:
            n = min(len(pe), len(rin))
            share_r = sum(abs((1.0 - w) * rin[i]) for i in range(n)) / n
            share_pe = sum(abs(w * pe[i]) for i in range(n)) / n
            total = share_r + share_pe
            if total > 1e-12:
                print(f"    share R  ({1-w:.2f}*R)   : {100*share_r/total:.1f} %")
                print(f"    share PE ({w:.2f}*PE)  : {100*share_pe/total:.1f} %")
                if share_pe > share_r:
                    print("    >> The prediction error dominates. Curiosity, empowerment,")
                    print("       vestibular and proprioceptive drives barely shape learning.")
                else:
                    print("    >> The intrinsic drives dominate the learning signal.")
        if dc and dh / dc > 0.9:
            print(f"    >> The PE branch fires {100.0*dh/dc:.1f} % of the time -- the other")
            print("       branch is effectively dead code. The threshold separates nothing.")
        elif dc:
            print("    >> Both branches occur.")

    print("\n  Reading:")
    if rel < 0.01:
        print("    >> NO GRADIENT. The reward is practically constant. R-STDP cannot")
        print("       shape anything from it -- a constant signal has no direction.")
        print("       Then the learning rule is not the problem; the system having no")
        print("       unmet drive is.")
    elif abs(ac) < 0.05:
        print("    >> The reward varies, but without structure (autocorrelation ~0).")
        print("       White noise carries nothing learnable; over many steps it")
        print("       averages out.")
    else:
        print("    >> The reward has both spread AND structure. Then the problem is not")
        print("       the signal but something further down the chain.")
    if cv is not None and abs(cv) < 0.05:
        print("    >> corr(v, R) ~ 0: going faster does not make the reward better, so")
        print("       there is no reason to get faster. That fully explains a speed that")
        print("       stays flat over tens of thousands of steps.")


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        report(p)


if __name__ == '__main__':
    main()
