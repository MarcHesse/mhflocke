#!/usr/bin/env python3
"""
MH-FLOCKE — FLOG Analysis Server
====================================
Lightweight Flask server for the FLOG Analysis Dashboard.
Runs locally without MuJoCo — only reads FLOG binary files.

Usage:
    python flog_server.py
    → Open http://localhost:5050

    python flog_server.py --port 8080
    python flog_server.py --flog creatures/mogli/v001_.../training_log.bin
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import struct
import threading

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════
# LIVE WATCH — incremental FLOG reading for live dashboards
# ═══════════════════════════════════════════════════════════

# Per-path state: tracks file offset so we only read NEW frames
_watch_state = {}  # path -> { 'offset': int, 'stats': [], 'creature': [], 'meta': {} }
_watch_lock = threading.Lock()

LOG_MAGIC = b'FLOG'
FRAME_TRAINING = 2
FRAME_CREATURE = 4


def _read_flog_incremental(flog_path: str):
    """Read new frames from a FLOG file since last call.
    
    Uses file offset tracking to avoid re-reading the entire file.
    Safe to call while training is writing to the same file.
    """
    flog_path = os.path.normpath(flog_path)
    with _watch_lock:
        if flog_path not in _watch_state:
            _watch_state[flog_path] = {
                'offset': 0, 'stats': [], 'creature': [],
                'meta': {}, 'header_read': False
            }
        state = _watch_state[flog_path]

    if not os.path.exists(flog_path):
        return state

    file_size = os.path.getsize(flog_path)
    if file_size <= state['offset']:
        return state  # No new data

    try:
        with open(flog_path, 'rb') as f:
            # First read: parse header
            if not state['header_read']:
                magic = f.read(4)
                if magic != LOG_MAGIC:
                    return state
                _version = struct.unpack('<H', f.read(2))[0]
                _phase = struct.unpack('<B', f.read(1))[0]
                meta_len = struct.unpack('<I', f.read(4))[0]
                meta_bytes = f.read(meta_len)
                state['meta'] = json.loads(meta_bytes)
                state['offset'] = f.tell()
                state['header_read'] = True
            else:
                f.seek(state['offset'])

            # Read frames from current offset
            new_stats = []
            new_creature = []
            while True:
                pos_before = f.tell()
                header = f.read(13)
                if len(header) < 13:
                    f.seek(pos_before)  # Incomplete header — writer mid-write
                    break
                ts, frame_type, data_len = struct.unpack('<dBI', header)
                payload = f.read(data_len)
                if len(payload) < data_len:
                    f.seek(pos_before)  # Incomplete frame
                    break

                try:
                    if HAS_MSGPACK:
                        data = msgpack.unpackb(payload, raw=False)
                    else:
                        data = json.loads(payload)
                except Exception:
                    continue

                if frame_type == FRAME_TRAINING:
                    row = {'timestamp': ts}
                    for k, v in data.items():
                        if isinstance(v, float):
                            row[k] = round(v, 4)
                        elif isinstance(v, (int, bool, str)):
                            row[k] = v
                    new_stats.append(row)
                elif frame_type == FRAME_CREATURE:
                    new_creature.append({'timestamp': ts, **data})

            state['offset'] = f.tell()

            with _watch_lock:
                state['stats'].extend(new_stats)
                state['creature'].extend(new_creature)

    except (IOError, OSError):
        pass  # File locked by writer — retry next poll

    return state

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

CREATURES_DIR = PROJECT_ROOT / 'creatures'
DASHBOARD_HTML = PROJECT_ROOT / 'src' / 'viz' / 'flog_dashboard.html'


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════


@app.route('/api/flog/watch')
def api_flog_watch():
    """Live watch endpoint — returns new stats since last poll.
    
    Usage:
        GET /api/flog/watch?path=creatures/go2/v034_.../training_log.bin&since=0
    
    Parameters:
        path:  FLOG file path
        since: index of last received stats frame (0 = get all)
    
    Returns:
        { stats: [...new frames...], total: N, creature: '...', task: '...' }
    """
    flog_path = request.args.get('path', '')
    since = int(request.args.get('since', 0))

    if not flog_path:
        return jsonify({'error': 'No path specified'}), 400

    # Normalize path separators (Windows backslash vs URL forward slash)
    flog_path = os.path.normpath(flog_path)

    state = _read_flog_incremental(flog_path)
    all_stats = state.get('stats', [])
    meta = state.get('meta', {})

    return jsonify({
        'stats': all_stats[since:],
        'total': len(all_stats),
        'creature': meta.get('creature', ''),
        'task': meta.get('task', ''),
    })


@app.route('/api/flog/watch/reset')
def api_flog_watch_reset():
    """Reset watch state for a FLOG file (force re-read from start)."""
    flog_path = request.args.get('path', '')
    with _watch_lock:
        if flog_path in _watch_state:
            del _watch_state[flog_path]
    return jsonify({'ok': True})

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return send_file(DASHBOARD_HTML)


@app.route('/api/flog/list')
def api_flog_list():
    """List all available FLOG files."""
    flogs = []
    for flog_path in sorted(glob.glob(str(CREATURES_DIR / '*/v*/training_log.bin'))):
        p = Path(flog_path)
        creature = p.parent.parent.name
        version = p.parent.name

        # Try to read meta.json
        meta_path = p.parent / 'meta.json'
        task = ''
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                task = meta.get('task', meta.get('description', ''))
            except Exception:
                pass

        flogs.append({
            'path': str(flog_path),
            'creature': creature,
            'version': version,
            'task': task,
            'size_kb': round(p.stat().st_size / 1024),
        })
    return jsonify(flogs)


@app.route('/api/flog/analyze')
def api_flog_analyze():
    """Load and analyze a FLOG file, return all data as JSON."""
    flog_path = request.args.get('path', '')

    if not flog_path or not os.path.exists(flog_path):
        return jsonify({'error': f'FLOG not found: {flog_path}'}), 404

    try:
        from src.viz.flog_replay import FlogReplay
        replay = FlogReplay(flog_path)

        # Build response with all frame data
        stats = replay.stats_frames()
        evo = replay.evolution_frames()
        events = replay.events()
        physics_summary = _physics_summary(replay)

        return jsonify({
            'creature': replay.creature,
            'task': replay.task,
            'skill': replay.skill,
            'duration_s': round(replay.duration_s, 1),
            'total_frames': replay.total_frames,
            'n_physics': replay.n_physics,
            'n_stats': replay.n_stats,
            'n_evolution': replay.n_evolution,
            'n_events': replay.n_events,
            'meta': replay.meta,
            'stats': _clean_stats(stats),
            'evolution': _clean_evo(evo),
            'events': _clean_events(events),
            'physics_summary': physics_summary,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# DATA CLEANING (ensure JSON-serializable)
# ═══════════════════════════════════════════════════════════

def _clean_stats(stats):
    """Clean stats frames for JSON.
    
    Passes through ALL numeric/string keys from FLOG stats frames.
    This makes the dashboard forward-compatible with new metrics
    without needing server-side changes.
    """
    out = []
    for s in stats:
        row = {}
        for k, v in s.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
            elif isinstance(v, (int, bool, str)):
                row[k] = v
            # skip complex types (lists, dicts, bytes)
        out.append(row)
    return out


def _clean_evo(evo):
    """Clean evolution frames for JSON."""
    out = []
    for e in evo:
        out.append({
            'timestamp': round(e.get('timestamp', 0), 2),
            'gen': e.get('gen', e.get('generation', 0)),
            'best_fitness': round(e.get('best_fitness', e.get('best', 0)) or 0, 4),
            'avg_fitness': round(e.get('avg_fitness', e.get('avg', 0)) or 0, 4),
            'best_distance': round(e.get('best_distance', e.get('dist', 0)) or 0, 3),
            'upright': round(e.get('upright', e.get('up', 0)) or 0, 3),
            'stood': e.get('stood', False),
        })
    return out


def _clean_events(events):
    """Clean event frames for JSON."""
    out = []
    for ev in events:
        out.append({
            'timestamp': round(ev.get('timestamp', 0), 2),
            'type': ev.get('type', 'unknown'),
            'msg': ev.get('msg', ''),
        })
    return out


def _physics_summary(replay):
    """Summarize physics data (don't send all 5000 frames)."""
    frames = replay.physics_frames()
    if not frames:
        return {'count': 0}

    speeds = [f.get('speed', 0) or 0 for f in frames]
    headings = [f.get('heading', 0) or 0 for f in frames]

    # Sample every 50th frame for position trajectory
    trajectory = []
    for i in range(0, len(frames), 50):
        f = frames[i]
        com = f.get('com', f.get('pos', [0, 0, 0]))
        if com and len(com) >= 3:
            trajectory.append({
                't': round(f.get('timestamp', 0), 1),
                'x': round(com[0], 3),
                'y': round(com[1], 3),
                'z': round(com[2], 3),
            })

    return {
        'count': len(frames),
        'avg_speed': round(sum(speeds) / max(len(speeds), 1), 4),
        'max_speed': round(max(speeds) if speeds else 0, 4),
        'trajectory': trajectory,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MH-FLOCKE FLOG Analysis Server')
    parser.add_argument('--port', type=int, default=5050, help='Port (default: 5050)')
    parser.add_argument('--host', default='127.0.0.1', help='Host (default: 127.0.0.1)')
    parser.add_argument('--flog', help='Auto-load specific FLOG file')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════╗
║  MH-FLOCKE — FLOG Analysis Dashboard            ║
║                                                  ║
║  http://{args.host}:{args.port}                        ║
║                                                  ║
║  Creatures: {CREATURES_DIR}
║  FLOGs found: {len(glob.glob(str(CREATURES_DIR / '*/v*/training_log.bin')))}
╚══════════════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
