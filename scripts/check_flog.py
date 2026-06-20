#!/usr/bin/env python3
"""
MH-FLOCKE — check_flog.py
=========================
Post-run FLOG integrity & honesty checker.

Runs after a training run to verify that the FLOG (training_log.bin) is
well-formed AND that the values in it are real, not placeholder/derived:

  * structure parses (magic, header, every frame, no truncation)
  * schema version is current (LOG_VERSION >= 2 = real neuromodulators)
  * the real spike raster is present and FULL length (not the 200-sampled
    legacy 'spikes' key that breaks the Brain3D population layout)
  * real neuromodulators (serotonin/noradrenaline/acetylcholine) are present
    and in range — not derived/placeholder
  * no NaN in any numeric stats field (Inf is treated as a sentinel -> WARN)
  * per-population spike coverage is sane (when population_sizes is in meta)

Plus an informational run summary (steps / falls / distance / competence).

Exit code: 0 if no FAIL, 1 if any FAIL (so it can gate a pipeline / post-run
hook), 2 if no FLOG was found / msgpack missing. WARN never fails the check.

Usage:
    py -3.11 scripts/check_flog.py [FLOG_PATH]
    py -3.11 scripts/check_flog.py                 # newest FLOG under creatures/
    py -3.11 scripts/check_flog.py --creature bittle
    py -3.11 scripts/check_flog.py --strict        # also FAIL on falls > 0
    py -3.11 scripts/check_flog.py --json
"""

__version__ = "0.1.0"
__logbook__ = 201

import argparse
import glob
import json
import math
import os
import struct
import sys

try:
    import msgpack
except ImportError:
    print("pip install msgpack"); sys.exit(2)

FLOG_MAGIC = b"FLOG"
FRAME_EVOLUTION, FRAME_TRAINING, FRAME_EVENT, FRAME_CREATURE = 1, 2, 3, 4
# Population order used by train_baby's spike_raster writer / brain_3d layout.
POP_ORDER = ["input", "granule", "golgi", "purkinje", "dcn", "motor_hidden", "output"]
POP_META_KEYS = {  # population name -> meta key in population_sizes
    "input": "n_input", "granule": "n_granule", "golgi": "n_golgi",
    "purkinje": "n_purkinje", "dcn": "n_dcn", "motor_hidden": "n_motor_hidden",
    "output": "n_output",
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.items = []  # (level, name, detail)

    def add(self, level, name, detail=""):
        self.items.append((level, name, detail))

    @property
    def failed(self):
        return any(lv == FAIL for lv, _, _ in self.items)

    @property
    def warned(self):
        return any(lv == WARN for lv, _, _ in self.items)


# --------------------------------------------------------------------------- #
# FLOG location + parsing
# --------------------------------------------------------------------------- #

def find_flog(path, creature):
    """Resolve a FLOG path, a creature run dir, or the newest FLOG on disk."""
    if path and os.path.isfile(path):
        return path
    if path and os.path.isdir(path):
        cand = os.path.join(path, "training_log.bin")
        if os.path.isfile(cand):
            return cand
        path = None  # fall through to glob inside this dir
    base = path or "creatures"
    pat = os.path.join(base, "**", "training_log.bin")
    files = glob.glob(pat, recursive=True)
    if creature:
        files = [f for f in files if creature.lower() in f.replace("\\", "/").lower()]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_flog(path):
    """Parse a FLOG into (meta, version, phase, frames). Raises on structural error.

    frames is a list of (frame_type, payload_dict). Truncation is reported by
    raising ValueError so the caller turns it into a FAIL.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != FLOG_MAGIC:
            raise ValueError(f"bad magic {magic!r} (not a FLOG file)")
        hdr = f.read(7)
        if len(hdr) < 7:
            raise ValueError("truncated header")
        version = struct.unpack("<H", hdr[0:2])[0]
        phase = struct.unpack("<B", hdr[2:3])[0]
        meta_len = struct.unpack("<I", hdr[3:7])[0]
        raw_meta = f.read(meta_len)
        if len(raw_meta) < meta_len:
            raise ValueError("truncated meta block")
        try:
            meta = json.loads(raw_meta) if meta_len else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"meta is not valid JSON: {e}")

        frames = []
        idx = 0
        while True:
            h = f.read(13)
            if len(h) == 0:
                break
            if len(h) < 13:
                raise ValueError(f"truncated frame header at frame {idx}")
            _ts, ftype, dlen = struct.unpack("<dBI", h)
            payload = f.read(dlen)
            if len(payload) < dlen:
                raise ValueError(f"truncated frame payload at frame {idx}")
            try:
                data = msgpack.unpackb(payload, raw=False)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"frame {idx} payload not valid msgpack: {e}")
            frames.append((ftype, data))
            idx += 1
    return meta, version, phase, frames


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _bad_floats(d):
    """Return (nan_keys, inf_keys) (recurses one level into lists).

    NaN is treated as a real numeric bug (FAIL); Inf is usually a legitimate
    sentinel (e.g. best-distance initialised to +inf = "none yet") and only WARNs.
    """
    nans, infs = [], []
    for k, v in d.items():
        if _is_num(v):
            if math.isnan(v):
                nans.append(k)
            elif math.isinf(v):
                infs.append(k)
        elif isinstance(v, (list, tuple)):
            for i, e in enumerate(v):
                if _is_num(e) and math.isnan(e):
                    nans.append(f"{k}[{i}]"); break
                if _is_num(e) and math.isinf(e):
                    infs.append(f"{k}[{i}]"); break
    return nans, infs


def check_structure(rep, meta, version, frames):
    type_counts = {}
    for ft, _ in frames:
        type_counts[ft] = type_counts.get(ft, 0) + 1
    n_train = type_counts.get(FRAME_TRAINING, 0)
    n_creat = type_counts.get(FRAME_CREATURE, 0)
    rep.add(PASS, "structure",
            f"{len(frames)} frames (training={n_train}, creature={n_creat}, "
            f"event={type_counts.get(FRAME_EVENT, 0)}, evo={type_counts.get(FRAME_EVOLUTION, 0)})")

    if version >= 2:
        rep.add(PASS, "schema", f"LOG_VERSION={version} (real neuromodulators)")
    else:
        rep.add(WARN, "schema",
                f"LOG_VERSION={version} < 2 — predates real 5HT/NE/ACh fields (legacy run)")

    if n_train == 0:
        rep.add(FAIL, "training_frames", "no training frames — nothing to display/verify")
    return n_train


def check_neuromod(rep, train):
    """Real neuromodulators present and in range across training frames."""
    keys = ["da_reward", "serotonin", "noradrenaline", "acetylcholine"]
    present = {k: 0 for k in keys}
    out_of_range = {}
    for d in train:
        for k in keys:
            if k in d and _is_num(d[k]):
                present[k] += 1
                lo, hi = (-1.0, 1.0) if k == "da_reward" else (0.0, 1.0)
                if not (lo - 1e-6 <= d[k] <= hi + 1e-6):
                    out_of_range.setdefault(k, d[k])
    n = len(train)
    real = [k for k in ("serotonin", "noradrenaline") if present[k] > 0]
    if len(real) == 2:
        rep.add(PASS, "neuromod_real",
                f"serotonin+noradrenaline present in {present['serotonin']}/{n} frames "
                f"(ACh in {present['acetylcholine']}/{n})")
    elif real:
        rep.add(WARN, "neuromod_real", f"only {', '.join(real)} present — partial")
    else:
        rep.add(WARN, "neuromod_real",
                "no real serotonin/noradrenaline — legacy FLOG (overlay shows '—')")
    if out_of_range:
        rep.add(FAIL, "neuromod_range",
                "out of range: " + ", ".join(f"{k}={v:.3f}" for k, v in out_of_range.items()))


def _pop_total_from_meta(meta):
    ps = meta.get("population_sizes") or {}
    if not ps:
        return None, None
    total = 0
    seg = []
    for name in POP_ORDER:
        n = int(ps.get(POP_META_KEYS[name], 0) or 0)
        seg.append((name, n))
        total += n
    return (total if total > 0 else None), seg


def check_spike_raster(rep, meta, train):
    """The real, FULL-length spike raster must be present (not the 200-sampled
    legacy 'spikes' key, which breaks the Brain3D population layout)."""
    has_new = any("spike_raster" in d for d in train)
    sample = None
    key = None
    for d in train:
        if "spike_raster" in d:
            sample, key = d["spike_raster"], "spike_raster"
            break
    if sample is None:
        for d in train:
            if "spikes" in d:
                sample, key = d["spikes"], "spikes"
                break

    if sample is None:
        rep.add(WARN, "spike_raster",
                "no spike raster in FLOG — Brain3D shows no activity (empty)")
        return

    length = len(sample)
    vals = set(int(x) for x in sample[:2000])
    binary = vals.issubset({0, 1})
    total, seg = _pop_total_from_meta(meta)

    if key == "spikes" and not has_new:
        # legacy key — gets sampled to 200 by record_training, breaks layout
        if length == 200 and (total is None or total != 200):
            rep.add(FAIL, "spike_raster",
                    "only legacy 'spikes' key, length 200 (down-sampled) — Brain3D "
                    "will pad and only the first ~200 display-neurons light. Write "
                    "'spike_raster' (full) from train_baby instead.")
        else:
            rep.add(WARN, "spike_raster", f"legacy 'spikes' key, length {length}")
        return

    if not binary:
        rep.add(FAIL, "spike_raster", f"values are not 0/1 (saw {sorted(vals)[:6]})")
        return

    if total is not None and length != total:
        rep.add(WARN, "spike_raster",
                f"length {length} != population total {total} from meta — "
                f"Brain3D will resample (check writer ordering)")
    else:
        rep.add(PASS, "spike_raster",
                f"real raster '{key}', length {length}, binary"
                + (f", matches population total {total}" if total else ""))

    # per-population coverage (sanity: not impossible, raster not all-zero)
    if seg and length == total:
        off = 0
        cov = []
        nonzero_pops = 0
        for name, n in seg:
            if n <= 0:
                continue
            s = sum(int(x) > 0 for x in sample[off:off + n])
            off += n
            cov.append(f"{name}:{s}/{n}")
            if s > 0:
                nonzero_pops += 1
        if nonzero_pops == 0:
            rep.add(WARN, "spike_coverage", "raster is all-zero in this frame (no neuron fired)")
        else:
            rep.add(PASS, "spike_coverage", "  ".join(cov))


def check_nan(rep, train):
    nan_fields, inf_fields = {}, {}
    for i, d in enumerate(train):
        nans, infs = _bad_floats(d)
        for k in nans:
            nan_fields.setdefault(k, i)
        for k in infs:
            inf_fields.setdefault(k, i)
    if nan_fields:
        rep.add(FAIL, "nan",
                "NaN in: " + ", ".join(f"{k}@frame{ix}" for k, ix in list(nan_fields.items())[:8]))
    else:
        rep.add(PASS, "nan", "no NaN in numeric stats fields")
    if inf_fields:
        rep.add(WARN, "inf",
                "Inf (usually a sentinel, e.g. best-distance not yet set): "
                + ", ".join(list(inf_fields.keys())[:8]))


def run_summary(rep, meta, train, creature_frames, strict):
    last = train[-1] if train else {}

    def g(*keys, default=None):
        for src in (last, meta):
            for k in keys:
                if k in src:
                    return src[k]
        return default

    falls = g("falls", default=None)
    dist = g("max_distance", "current_distance", default=None)
    comp = g("actor_competence", default=None)
    cpg = g("cpg_weight", default=None)
    beh = g("behavior", default=None)
    steps = g("step", "steps", default=None)

    parts = []
    if steps is not None:
        parts.append(f"steps={steps}")
    if falls is not None:
        parts.append(f"falls={falls}")
    if dist is not None:
        parts.append(f"dist={float(dist):.3f}m")
    if comp is not None:
        parts.append(f"competence={float(comp):.3f}")
    if cpg is not None:
        parts.append(f"cpg={float(cpg):.0%}")
    if beh:
        parts.append(f"behavior={beh}")
    rep.add(PASS, "run_summary", "  ".join(parts) if parts else "(no summary fields)")

    if strict and falls is not None and falls > 0:
        rep.add(FAIL, "run_quality", f"falls={falls} > 0 (--strict; Bittle canonical expects 0)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Post-run FLOG integrity & honesty checker")
    ap.add_argument("flog", nargs="?", default=None,
                    help="FLOG path or run dir (default: newest training_log.bin under creatures/)")
    ap.add_argument("--creature", default=None, help="filter newest-FLOG search by creature name")
    ap.add_argument("--strict", action="store_true", help="also FAIL on falls > 0")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON report")
    args = ap.parse_args()

    path = find_flog(args.flog, args.creature)
    if not path:
        print("No FLOG found.", file=sys.stderr)
        return 2

    rep = Report()
    try:
        meta, version, phase, frames = parse_flog(path)
    except ValueError as e:
        rep.add(FAIL, "parse", str(e))
        _emit(rep, path, args.json)
        return 1

    train = [d for ft, d in frames if ft == FRAME_TRAINING]
    creat = [d for ft, d in frames if ft == FRAME_CREATURE]

    n_train = check_structure(rep, meta, version, frames)
    if n_train:
        check_neuromod(rep, train)
        check_spike_raster(rep, meta, train)
        check_nan(rep, train)
    run_summary(rep, meta, train, creat, args.strict)

    _emit(rep, path, args.json)
    return 1 if rep.failed else 0


def _emit(rep, path, as_json):
    if as_json:
        print(json.dumps({
            "flog": path,
            "result": "FAIL" if rep.failed else ("WARN" if rep.warned else "PASS"),
            "checks": [{"level": lv, "name": n, "detail": d} for lv, n, d in rep.items],
        }, indent=2))
        return
    print(f"\nFLOG check: {path}")
    print("=" * 70)
    for lv, name, detail in rep.items:
        print(f"  [{lv}] {name:16s} {detail}")
    print("=" * 70)
    verdict = "FAIL" if rep.failed else ("PASS (with warnings)" if rep.warned else "PASS")
    print(f"  => {verdict}\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
