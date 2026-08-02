"""
Does the Bittle move at all -- and if so, where to?

Starting point: 0.547 m "max distance" over 10,000 steps with zero falls and an
actor competence of 1.000. What that number measures is not obvious. This
analysis computes both path length and straight-line displacement from the
logged positions, which separates four cases:

  short path  + short displacement  -> standing (or stepping in place)
  long path   + short displacement  -> walking in circles / pendling
  long path   + long displacement   -> walking straight, just slowly
  path ~ displacement               -> directed movement

Also reported: speed over time (is it getting faster?), heading consistency,
and whether the movement comes in bursts or is even.

Usage:
    py -3.11 scripts/analyze_locomotion.py <flog> [<flog2> ...]
"""
import sys
import math
import struct

try:
    import msgpack
except ImportError:
    sys.exit("msgpack not installed")

FRAME_TRAINING = 0x02
BUCKETS = 10
SIM_DT = 0.002          # bittle.xml timestep; only used for the time figure


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


def report(path):
    print(f"\n=== {path} ===")
    rows = [r for r in parse(path) if 'pos_x' in r and 'pos_y' in r]
    if len(rows) < 2:
        print("  no position data in the FLOG")
        return

    xs = [float(r['pos_x']) for r in rows]
    ys = [float(r['pos_y']) for r in rows]
    steps = [int(r.get('step', i)) for i, r in enumerate(rows)]

    seg = [math.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) for i in range(len(xs)-1)]
    path_len = sum(seg)
    disp = math.hypot(xs[-1]-xs[0], ys[-1]-ys[0])
    max_disp = max(math.hypot(x-xs[0], y-ys[0]) for x, y in zip(xs, ys))
    span = steps[-1] - steps[0]
    sim_s = span * SIM_DT

    print(f"  frames / steps        : {len(rows)} / {steps[0]}..{steps[-1]}")
    print(f"  path length           : {path_len:.3f} m")
    print(f"  displacement start->end: {disp:.3f} m")
    print(f"  max distance from start: {max_disp:.3f} m")
    if path_len > 1e-9:
        print(f"  straightness          : {disp/path_len:.3f}  (1.0 = dead straight)")
    print(f"  mean speed            : {path_len/sim_s:.4f} m/s  ({sim_s:.1f} s sim time)")

    print(f"\n  Over time ({BUCKETS} segments):")
    print(f"  {'step':>10} {'path m':>8} {'v m/s':>8} {'x':>8} {'y':>8} {'from start':>11}")
    size = max(1, (len(rows)-1) // BUCKETS)
    for b in range(0, len(rows)-1, size):
        lo, hi = b, min(b + size, len(rows)-1)
        w = sum(seg[lo:hi])
        dstep = steps[hi] - steps[lo]
        v = w / (dstep * SIM_DT) if dstep else 0.0
        print(f"  {steps[hi]:>10} {w:>8.3f} {v:>8.4f} {xs[hi]:>8.3f} {ys[hi]:>8.3f}"
              f" {math.hypot(xs[hi]-xs[0], ys[hi]-ys[0]):>11.3f}")

    print("\n  Reading:")
    if path_len < 0.05:
        print("    >> STANDING. Practically no change of place. That is not a")
        print("       learning problem but a locomotion problem, and it makes any")
        print("       question about curves or exploration moot.")
    elif disp / path_len < 0.3:
        print("    >> WALKING, BUT NOT AWAY. Long path, little displacement: circles")
        print("       or pendling. For exploration that means it sees almost nothing")
        print("       new, however good the curiosity drive is.")
    else:
        print("    >> DIRECTED MOVEMENT. Distance is then a question of speed, not")
        print("       of heading.")


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        report(p)


if __name__ == '__main__':
    main()
