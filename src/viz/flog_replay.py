"""
MH-FLOCKE — FLOG Replay v0.4.1
========================================
Binary FLOG reader for training data analysis and visualization.
"""

import os
import csv
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass, field

# Frame type constants (must match creature_store.py)
FRAME_EVOLUTION = 1
FRAME_TRAINING = 2
FRAME_EVENT = 3       # 0x03 in creature_store.py
FRAME_CREATURE = 4    # 0x04 in creature_store.py
LOG_MAGIC = b'FLOG'

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


# ═══════════════════════════════════════════════════════════
# FLOG REPLAY
# ═══════════════════════════════════════════════════════════

class FlogReplay:
    """
    High-level FLOG reader for video pipeline and dashboard.

    Separates raw frame reading (from TrainingLogReader) from
    domain-specific extraction (physics, stats, events).
    """

    def __init__(self, flog_path: str):
        self.path = Path(flog_path)
        if not self.path.exists():
            raise FileNotFoundError(f"FLOG not found: {flog_path}")

        self._meta: Dict = {}
        self._frames: List[Dict] = []
        self._read()

        # Cached extractions
        self._physics_cache: Optional[List[Dict]] = None
        self._stats_cache: Optional[List[Dict]] = None

    def _read(self):
        """Read binary FLOG file."""
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != LOG_MAGIC:
                raise ValueError(f"Not a FLOG file: {self.path}")

            version = struct.unpack('<H', f.read(2))[0]
            _phase = struct.unpack('<B', f.read(1))[0]
            meta_len = struct.unpack('<I', f.read(4))[0]
            meta_bytes = f.read(meta_len)
            self._meta = json.loads(meta_bytes)

            while True:
                header = f.read(13)
                if len(header) < 13:
                    break
                ts, frame_type, data_len = struct.unpack('<dBI', header)
                payload = f.read(data_len)
                if len(payload) < data_len:
                    break

                if HAS_MSGPACK:
                    data = msgpack.unpackb(payload, raw=False)
                else:
                    data = json.loads(payload)

                self._frames.append({
                    'timestamp': ts,
                    'type': frame_type,
                    'data': data,
                })

    # ── Properties ──────────────────────────────────────────

    @property
    def meta(self) -> Dict:
        """FLOG header metadata (creature, task, scene, etc.)."""
        return self._meta

    @property
    def creature(self) -> str:
        return self._meta.get('creature', 'unknown')

    @property
    def task(self) -> str:
        return self._meta.get('task', '')

    @property
    def skill(self) -> str:
        return self._meta.get('skill', '')

    @property
    def duration_s(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1]['timestamp']

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def n_physics(self) -> int:
        return len([f for f in self._frames if f['type'] == FRAME_CREATURE])

    @property
    def n_stats(self) -> int:
        return len([f for f in self._frames if f['type'] == FRAME_TRAINING])

    @property
    def n_events(self) -> int:
        return len([f for f in self._frames if f['type'] == FRAME_EVENT])

    @property
    def n_evolution(self) -> int:
        return len([f for f in self._frames if f['type'] == FRAME_EVOLUTION])

    # ── Physics Extraction ─────────────────────────────────

    def physics_frames(self) -> List[Dict]:
        """All creature physics frames (qpos, qvel, com, speed)."""
        if self._physics_cache is None:
            self._physics_cache = [
                {'timestamp': f['timestamp'], **f['data']}
                for f in self._frames if f['type'] == FRAME_CREATURE
            ]
        return self._physics_cache

    def export_physics_npz(self, output_path: str):
        """
        Export physics to npz for MuJoCo replay.

        Creates arrays:
          qpos: (n_frames, n_joints) — joint positions
          qvel: (n_frames, n_joints) — joint velocities
          timestamps: (n_frames,) — seconds since start
          steps: (n_frames,) — training step numbers
        """
        frames = self.physics_frames()
        if not frames:
            print(f"  ⚠ No physics frames in {self.path}")
            return

        # Extract arrays
        timestamps = np.array([f['timestamp'] for f in frames], dtype=np.float64)
        steps = np.array([f.get('step', i * 10) for i, f in enumerate(frames)], dtype=np.int32)

        # qpos/qvel may vary in length — use first frame as reference
        qpos_list = [np.array(f['pos']) for f in frames if 'pos' in f]
        qvel_list = [np.array(f['vel']) for f in frames if 'vel' in f]

        if qpos_list:
            qpos = np.stack(qpos_list)
        else:
            qpos = np.zeros((len(frames), 1))

        if qvel_list:
            qvel = np.stack(qvel_list)
        else:
            qvel = np.zeros((len(frames), 1))

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez_compressed(output_path,
                            qpos=qpos, qvel=qvel,
                            timestamps=timestamps, steps=steps)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  📦 Physics exported: {output_path} "
              f"({len(frames)} frames, {size_kb:.0f} KB)")

    def iter_physics(self, fps: float = 30.0) -> Iterator[Dict]:
        """
        Iterate physics frames at target FPS for video rendering.

        Interpolates between logged frames (every 10 steps) to produce
        smooth playback at target framerate.
        """
        frames = self.physics_frames()
        if not frames:
            return

        duration = frames[-1]['timestamp']
        dt = 1.0 / fps
        t = 0.0
        idx = 0

        while t <= duration:
            # Find surrounding frames
            while idx < len(frames) - 1 and frames[idx + 1]['timestamp'] < t:
                idx += 1

            if idx >= len(frames) - 1:
                yield frames[-1]
                break

            f0 = frames[idx]
            f1 = frames[idx + 1]

            # Interpolation factor
            dt_frames = f1['timestamp'] - f0['timestamp']
            if dt_frames > 0:
                alpha = (t - f0['timestamp']) / dt_frames
            else:
                alpha = 0.0
            alpha = max(0.0, min(1.0, alpha))

            # Interpolate qpos/qvel
            interp = {'timestamp': t}
            if 'pos' in f0 and 'pos' in f1:
                p0 = np.array(f0['pos'])
                p1 = np.array(f1['pos'])
                interp['qpos'] = (1 - alpha) * p0 + alpha * p1
            elif 'pos' in f0:
                interp['qpos'] = np.array(f0['pos'])

            if 'vel' in f0 and 'vel' in f1:
                v0 = np.array(f0['vel'])
                v1 = np.array(f1['vel'])
                interp['qvel'] = (1 - alpha) * v0 + alpha * v1
            elif 'vel' in f0:
                interp['qvel'] = np.array(f0['vel'])

            if 'com' in f0:
                interp['com'] = np.array(f0['com'])
            interp['speed'] = f0.get('speed', 0.0)
            interp['step'] = f0.get('step', 0)
            interp['skill'] = f0.get('skill', '')

            yield interp
            t += dt

    # ── Stats Extraction ────────────────────────────────────

    def stats_frames(self) -> List[Dict]:
        """All training stats frames (distance, reward, emotions, etc.)."""
        if self._stats_cache is None:
            self._stats_cache = [
                {'timestamp': f['timestamp'], **f['data']}
                for f in self._frames if f['type'] == FRAME_TRAINING
            ]
        return self._stats_cache

    def evolution_frames(self) -> List[Dict]:
        """All CPG evolution frames."""
        return [
            {'timestamp': f['timestamp'], **f['data']}
            for f in self._frames if f['type'] == FRAME_EVOLUTION
        ]

    def export_stats_csv(self, output_path: str):
        """
        Export stats to CSV compatible with flocke_editor TimelineEngine.

        Columns match TimelineRow fields used by FlockeEditor:
          step, timestamp, distance, max_distance, best_episode, avg_episode,
          falls, resets, reward, consciousness_level, emotion, valence, skill
        """
        frames = self.stats_frames()
        if not frames:
            print(f"  ⚠ No stats frames in {self.path}")
            return

        fieldnames = [
            'step', 'timestamp', 'distance', 'max_distance',
            'best_episode', 'avg_episode', 'falls', 'resets',
            'reward', 'consciousness_level', 'emotion', 'valence', 'skill',
            # Cerebellar architecture data (v0.3.0)
            'grc_sparseness', 'cf_magnitude', 'pf_pkc_weight',
            'pf_pkc_weight_std', 'ltd_applied', 'ltp_applied',
            'dcn_activity', 'correction_mag',
            'snn_mix', 'golgi_rate', 'upright', 'episode',
        ]

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for frame in frames:
                row = {k: frame.get(k, '') for k in fieldnames}
                row['timestamp'] = f"{frame.get('timestamp', 0):.3f}"
                writer.writerow(row)

        print(f"  📊 Stats exported: {output_path} ({len(frames)} rows)")

    # ── Events ──────────────────────────────────────────────

    def events(self) -> List[Dict]:
        """All milestone events (skill_start, skill_freeze, etc.)."""
        return [
            {'timestamp': f['timestamp'], **f['data']}
            for f in self._frames if f['type'] == FRAME_EVENT
        ]

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of FLOG contents."""
        lines = [
            f"FLOG: {self.path.name}",
            f"  Creature:  {self.creature}",
            f"  Task:      {self.task}",
            f"  Duration:  {self.duration_s:.1f}s",
            f"  Physics:   {self.n_physics} frames",
            f"  Stats:     {self.n_stats} frames",
            f"  Evolution: {self.n_evolution} frames",
            f"  Events:    {self.n_events}",
            f"  Total:     {self.total_frames} frames",
        ]
        for ev in self.events():
            lines.append(f"    {ev['timestamp']:>6.1f}s  [{ev.get('type', '?')}] {ev.get('msg', '')}")
        return "\n".join(lines)

    def __repr__(self):
        return f"FlogReplay({self.path.name}: {self.total_frames} frames, {self.duration_s:.1f}s)"


# ═══════════════════════════════════════════════════════════
# FLOG → FLOCKE EDITOR BRIDGE
# ═══════════════════════════════════════════════════════════

def flog_to_editor(flog_path: str) -> 'FlockeEditor':
    """
    Create a FlockeEditor from a FLOG file.

    Exports stats to a temp CSV, then initializes the editor
    with that CSV. This bridges FLOG binary → existing CSV-based
    editor without rewriting the entire editor.
    """
    import tempfile
    from src.viz.flocke_editor import FlockeEditor

    replay = FlogReplay(flog_path)
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    replay.export_stats_csv(tmp.name)
    editor = FlockeEditor(tmp.name)
    editor._flog_replay = replay  # Keep reference for physics
    editor._flog_path = flog_path
    return editor


def flog_to_assemble(flog_path: str, output_dir: str) -> Dict[str, str]:
    """
    Export FLOG to files needed by assemble_video.py.

    Returns dict with paths:
      {'physics_npz': ..., 'stats_csv': ..., 'events_json': ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    replay = FlogReplay(flog_path)

    physics_path = os.path.join(output_dir, 'physics.npz')
    stats_path = os.path.join(output_dir, 'stats.csv')
    events_path = os.path.join(output_dir, 'events.json')

    replay.export_physics_npz(physics_path)
    replay.export_stats_csv(stats_path)

    # Export events
    with open(events_path, 'w') as f:
        json.dump(replay.events(), f, indent=2, default=str)
    print(f"  📋 Events exported: {events_path} ({len(replay.events())} events)")

    print(f"\n  Summary:\n{replay.summary()}")

    return {
        'physics_npz': physics_path,
        'stats_csv': stats_path,
        'events_json': events_path,
    }


# ═══════════════════════════════════════════════════════════
# NPZ PHYSICS REPLAY — drop-in replacement for deprecated PhysicsReplay
# ═══════════════════════════════════════════════════════════

class NpzPhysicsReplay:
    """
    Replays physics from NPZ (exported by flog_to_assemble).
    Same API as deprecated src.viz.recording.PhysicsReplay.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.qpos = data['qpos']
        self.qvel = data['qvel']
        self.steps = data['steps'] if 'steps' in data else np.arange(len(self.qpos)) * 10
        self.timestamps = data['timestamps'] if 'timestamps' in data else np.arange(len(self.qpos)) * 0.01
        self.n_frames = len(self.qpos)

    def apply_frame(self, fi: int, model, data):
        """Apply frame fi to MuJoCo model/data."""
        fi = min(fi, self.n_frames - 1)
        nq = min(len(self.qpos[fi]), model.nq)
        nv = min(len(self.qvel[fi]), model.nv)
        # Warn once on mismatch
        if fi == 0 and len(self.qpos[0]) != model.nq:
            print(f"    \u26a0 qpos mismatch: recorded nq={len(self.qpos[0])}, "
                  f"model nq={model.nq} \u2014 limbs may appear detached!")
        data.qpos[:nq] = self.qpos[fi][:nq]
        data.qvel[:nv] = self.qvel[fi][:nv]

    def get_step(self, fi: int) -> int:
        fi = min(fi, self.n_frames - 1)
        return int(self.steps[fi])

    def get_creature_pos(self, fi: int) -> Tuple[float, float, float]:
        fi = min(fi, self.n_frames - 1)
        q = self.qpos[fi]
        return (float(q[0]), float(q[1]), float(q[2]))


def flog_evo_gen_frames(flog_path: str) -> List[Dict]:
    """
    Extract per-generation creature frames from an evolution FLOG.

    Returns list of dicts per generation:
      [{'gen': 0, 'qpos': array, 'qvel': array, 'steps': array, ...}, ...]
    """
    replay = FlogReplay(flog_path)
    physics = replay.physics_frames()
    evo_data = replay.evo_frames() if hasattr(replay, 'evo_frames') else []

    # Group creature frames by 'gen' field
    gen_groups = {}
    for f in physics:
        g = f.get('gen', 0)
        if g not in gen_groups:
            gen_groups[g] = []
        gen_groups[g].append(f)

    result = []
    for gen_idx in sorted(gen_groups.keys()):
        frames = gen_groups[gen_idx]
        qpos = np.array([f['pos'] for f in frames if 'pos' in f])
        qvel = np.array([f['vel'] for f in frames if 'vel' in f])
        steps = np.array([f.get('step', i * 10) for i, f in enumerate(frames)])
        result.append({
            'gen': gen_idx,
            'qpos': qpos,
            'qvel': qvel,
            'steps': steps,
            'n_frames': len(qpos),
        })

    return result


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='FLOG Replay — extract training data')
    parser.add_argument('flog', help='Path to .bin FLOG file')
    parser.add_argument('--export', '-e', default=None,
                        help='Export directory (creates physics.npz + stats.csv + events.json)')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Print summary only')
    parser.add_argument('--physics-npz', default=None,
                        help='Export physics to specific npz path')
    parser.add_argument('--stats-csv', default=None,
                        help='Export stats to specific CSV path')
    args = parser.parse_args()

    replay = FlogReplay(args.flog)

    if args.summary:
        print(replay.summary())
        return

    if args.export:
        flog_to_assemble(args.flog, args.export)
        return

    if args.physics_npz:
        replay.export_physics_npz(args.physics_npz)

    if args.stats_csv:
        replay.export_stats_csv(args.stats_csv)

    if not args.physics_npz and not args.stats_csv:
        print(replay.summary())


if __name__ == '__main__':
    main()
