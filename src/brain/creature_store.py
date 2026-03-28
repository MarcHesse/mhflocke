"""
MH-FLOCKE — Creature Store v0.4.1
========================================
FLOG binary logger for training data.
"""

import json
import struct
import time
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═══════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════

@dataclass
class VersionMeta:
    """Metadata for one creature version."""
    version_id: str          # e.g. "v001"
    created: str             # ISO datetime
    tag: str = ""            # user label: "learned grass walking"
    branch: str = "main"
    task: str = ""           # "dog walk on grass"
    scene: str = ""          # "flat_grass"
    
    # Training results
    best_fitness: float = 0.0
    total_generations: int = 0
    training_time_s: float = 0.0
    device: str = "cpu"
    
    # Brain state
    n_neurons: int = 0
    consciousness_level: int = 0
    n_concepts: int = 0
    n_skills: int = 0
    skills: List[str] = field(default_factory=list)
    
    # Data sizes
    brain_size_kb: float = 0.0
    log_frames: int = 0
    log_size_kb: float = 0.0


@dataclass
class CreatureManifest:
    """Manifest for one creature — all versions + branches."""
    name: str
    created: str
    body_type: str = "synpaw"
    versions: List[VersionMeta] = field(default_factory=list)
    active_version: str = ""  # currently loaded version
    branches: List[str] = field(default_factory=lambda: ["main"])


# ═══════════════════════════════════════════════
# TRAINING RECORDER
# ═══════════════════════════════════════════════

# Binary format constants
LOG_MAGIC = b'FLOG'
LOG_VERSION = 1
FRAME_EVOLUTION = 0x01
FRAME_TRAINING = 0x02
FRAME_EVENT = 0x03
FRAME_CREATURE = 0x04


class TrainingRecorder:
    """
    Records training data to binary log file.
    
    Two phases recorded:
      EVOLUTION: generation stats, population fitness, best genome params
      TRAINING:  SNN spikes (sampled), neuromod levels, GWT, emotions, consciousness
    
    Plus: creature physics (joints, contacts) and events (milestones).
    
    All data is compact binary (msgpack). No video frames stored.
    Video is reconstructed later from this data + MuJoCo replay.
    """
    
    def __init__(self, path: str, meta: Optional[Dict] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._frame_count = 0
        self._t_start = time.time()
        self._open(meta or {})
    
    def _open(self, meta: Dict):
        """Write header and open for appending."""
        self._file = open(self.path, 'wb')
        
        # Header
        meta_bytes = json.dumps(meta, default=str).encode('utf-8')
        self._file.write(LOG_MAGIC)                           # 4 bytes
        self._file.write(struct.pack('<H', LOG_VERSION))      # 2 bytes
        self._file.write(struct.pack('<B', 0))                # 1 byte phase tag (0=mixed)
        self._file.write(struct.pack('<I', len(meta_bytes)))  # 4 bytes
        self._file.write(meta_bytes)
        self._file.flush()
    
    def _serialize(self, data: Dict) -> bytes:
        """Serialize frame data to bytes."""
        # Convert numpy arrays to lists for serialization
        clean = {}
        for k, v in data.items():
            if HAS_NUMPY and isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif isinstance(v, (list, tuple, int, float, str, bool, type(None))):
                clean[k] = v
            elif isinstance(v, dict):
                clean[k] = {str(kk): vv.tolist() if HAS_NUMPY and isinstance(vv, np.ndarray) else vv 
                           for kk, vv in v.items()}
            else:
                clean[k] = str(v)
        
        if HAS_MSGPACK:
            return msgpack.packb(clean, use_bin_type=True)
        else:
            return json.dumps(clean, default=str).encode('utf-8')
    
    def _write_frame(self, frame_type: int, data: Dict):
        """Write one frame to log."""
        if not self._file:
            return
        payload = self._serialize(data)
        ts = time.time() - self._t_start
        # Frame: timestamp(8) + type(1) + length(4) + payload
        self._file.write(struct.pack('<dBI', ts, frame_type, len(payload)))
        self._file.write(payload)
        self._frame_count += 1
        
        # Flush every 100 frames
        if self._frame_count % 100 == 0:
            self._file.flush()
    
    def record_evolution(self, generation: int, best_fitness: float,
                         avg_fitness: float, population_size: int,
                         best_distance: float = 0.0,
                         curriculum_stage: str = "",
                         best_genome: Optional[Dict] = None,
                         population_diversity: float = 0.0,
                         **extra):
        """Record one evolution generation."""
        data = {
            'gen': generation,
            'best': best_fitness,
            'avg': avg_fitness,
            'pop': population_size,
            'dist': best_distance,
            'stage': curriculum_stage,
            'diversity': population_diversity,
            **extra,
        }
        if best_genome:
            data['genome'] = best_genome
        self._write_frame(FRAME_EVOLUTION, data)
    
    def record_training(self, step: int,
                        spikes: Optional[Any] = None,
                        neuromod: Optional[Dict] = None,
                        gwt_winner: str = "",
                        emotions: Optional[Dict] = None,
                        consciousness_level: int = 0,
                        pci: float = 0.0,
                        spike_count: int = 0,
                        **extra):
        """Record one SNN training step."""
        data = {
            'step': step,
            'gwt': gwt_winner,
            'c_level': consciousness_level,
            'pci': pci,
            'spike_count': spike_count,
            **extra,
        }
        if spikes is not None:
            # Sample spikes to keep size small (max 200 neurons)
            if HAS_NUMPY and isinstance(spikes, np.ndarray):
                n = len(spikes)
                if n > 200:
                    idx = np.linspace(0, n-1, 200, dtype=int)
                    data['spikes'] = spikes[idx].astype(int).tolist()
                else:
                    data['spikes'] = spikes.astype(int).tolist()
            else:
                data['spikes'] = list(spikes)[:200]
        if neuromod:
            data['neuromod'] = neuromod
        if emotions:
            data['emotions'] = emotions
        self._write_frame(FRAME_TRAINING, data)
    
    def record_creature(self, joint_positions: Any = None,
                        joint_velocities: Any = None,
                        contacts: Optional[List] = None,
                        center_of_mass: Optional[Any] = None,
                        heading: float = 0.0,
                        speed: float = 0.0,
                        **extra):
        """Record creature physics state."""
        data = {'heading': heading, 'speed': speed, **extra}
        if joint_positions is not None:
            data['pos'] = joint_positions.tolist() if HAS_NUMPY and isinstance(joint_positions, np.ndarray) else list(joint_positions)
        if joint_velocities is not None:
            data['vel'] = joint_velocities.tolist() if HAS_NUMPY and isinstance(joint_velocities, np.ndarray) else list(joint_velocities)
        if contacts:
            data['contacts'] = contacts
        if center_of_mass is not None:
            data['com'] = center_of_mass.tolist() if HAS_NUMPY and isinstance(center_of_mass, np.ndarray) else list(center_of_mass)
        self._write_frame(FRAME_CREATURE, data)
    
    def record_event(self, event_type: str, message: str, **extra):
        """Record milestone/event."""
        data = {'type': event_type, 'msg': message, **extra}
        self._write_frame(FRAME_EVENT, data)
    
    def record_training_stats(self, data: Dict):
        """Record training stats (convenience wrapper for log_frame)."""
        self.log_frame(data)

    def log_frame(self, data: Dict):
        """Generic frame logger — auto-detects frame type from data keys."""
        if 'gen' in data or 'generation' in data or data.get('phase') == 'cpg_evolution':
            self.record_evolution(
                generation=data.get('generation', data.get('gen', 0)),
                best_fitness=data.get('best_fitness', 0),
                avg_fitness=data.get('avg_fitness', 0),
                population_size=data.get('pop', 0),
                best_distance=data.get('best_distance', data.get('dist', 0)),
                curriculum_stage=data.get('phase', ''),
            )
        elif 'step' in data or data.get('phase') == 'snn_training':
            self.record_training(
                step=data.get('step', 0),
                consciousness_level=data.get('consciousness_level', 0),
                spike_count=data.get('spike_count', 0),
                **{k: v for k, v in data.items()
                   if k not in ('step', 'consciousness_level', 'spike_count')},
            )
        else:
            self._write_frame(FRAME_EVENT, data)

    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def close(self):
        """Close log file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
    
    def __del__(self):
        self.close()


class TrainingLogReader:
    """Read back a training log for video rendering / analysis."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._frames = []
        self._meta = {}
        self._read()
    
    def _read(self):
        with open(self.path, 'rb') as f:
            # Header
            magic = f.read(4)
            if magic != LOG_MAGIC:
                raise ValueError(f"Not a FLOG file: {self.path}")
            version = struct.unpack('<H', f.read(2))[0]
            _phase = struct.unpack('<B', f.read(1))[0]
            meta_len = struct.unpack('<I', f.read(4))[0]
            meta_bytes = f.read(meta_len)
            self._meta = json.loads(meta_bytes)
            
            # Frames
            while True:
                header = f.read(13)  # 8 + 1 + 4
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
    
    @property
    def meta(self) -> Dict:
        return self._meta
    
    @property
    def frames(self) -> List[Dict]:
        return self._frames
    
    def evolution_frames(self) -> List[Dict]:
        return [f for f in self._frames if f['type'] == FRAME_EVOLUTION]
    
    def training_frames(self) -> List[Dict]:
        return [f for f in self._frames if f['type'] == FRAME_TRAINING]
    
    def creature_frames(self) -> List[Dict]:
        return [f for f in self._frames if f['type'] == FRAME_CREATURE]
    
    def events(self) -> List[Dict]:
        return [f for f in self._frames if f['type'] == FRAME_EVENT]
    
    @property
    def duration_s(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1]['timestamp']
    
    def __len__(self):
        return len(self._frames)


# ═══════════════════════════════════════════════
# CREATURE STORE
# ═══════════════════════════════════════════════

class CreatureStore:
    """
    Versioned creature lifecycle management.
    
    Usage:
        store = CreatureStore('creatures')
        
        # Start recording during training
        recorder = store.start_recording('mogli', task='dog walk on grass')
        recorder.record_evolution(gen=1, best=0.12, avg=0.05, pop=50)
        ...
        
        # Save snapshot after training
        store.snapshot('mogli', tag='first grass walk',
                      brain_state=snn.state_dict(),
                      knowledge=graph.export(),
                      training_result={...})
        
        # Later: restore
        store.restore('mogli', 'v001')
        
        # List versions
        store.list_versions('mogli')
        
        # Branch for experiment
        store.branch('mogli', 'v003', 'forest_experiment')
    """
    
    def __init__(self, base_dir: str = 'creatures'):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._active_recorders: Dict[str, TrainingRecorder] = {}
    
    def _manifest_path(self, name: str) -> Path:
        return self.base / name / 'manifest.json'
    
    def _load_manifest(self, name: str) -> CreatureManifest:
        p = self._manifest_path(name)
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            versions = [VersionMeta(**v) for v in data.get('versions', [])]
            return CreatureManifest(
                name=data['name'],
                created=data['created'],
                body_type=data.get('body_type', 'synpaw'),
                versions=versions,
                active_version=data.get('active_version', ''),
                branches=data.get('branches', ['main']),
            )
        return CreatureManifest(
            name=name,
            created=datetime.now().isoformat(),
        )
    
    def _save_manifest(self, manifest: CreatureManifest):
        p = self._manifest_path(manifest.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'name': manifest.name,
            'created': manifest.created,
            'body_type': manifest.body_type,
            'versions': [asdict(v) for v in manifest.versions],
            'active_version': manifest.active_version,
            'branches': manifest.branches,
        }
        p.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    
    def _next_version_id(self, manifest: CreatureManifest) -> str:
        """Generate next version ID: v001, v002, ..."""
        existing = [v.version_id for v in manifest.versions]
        n = len(existing) + 1
        return f"v{n:03d}"
    
    def _version_dir(self, name: str, version_id: str, branch: str = "main") -> Path:
        if branch == "main":
            return self.base / name / f"{version_id}_{datetime.now().strftime('%Y-%m-%dT%H-%M')}"
        return self.base / name / "branches" / branch / version_id
    
    def list_creatures(self) -> List[str]:
        """List all creatures."""
        return [d.name for d in self.base.iterdir() if d.is_dir() and (d / 'manifest.json').exists()]
    
    def list_versions(self, name: str) -> List[VersionMeta]:
        """List all versions for a creature."""
        manifest = self._load_manifest(name)
        return manifest.versions
    
    # ── Recording ──
    
    def start_recording(self, name: str, task: str = "",
                        scene: str = "", device: str = "cpu",
                        **extra_meta) -> TrainingRecorder:
        """Start recording training data for a creature."""
        manifest = self._load_manifest(name)
        vid = self._next_version_id(manifest)
        vdir = self._version_dir(name, vid)
        vdir.mkdir(parents=True, exist_ok=True)
        
        log_path = vdir / 'training_log.bin'
        meta = {
            'creature': name,
            'version': vid,
            'task': task,
            'scene': scene,
            'device': device,
            'started': datetime.now().isoformat(),
            **extra_meta,
        }
        recorder = TrainingRecorder(str(log_path), meta)
        self._active_recorders[name] = recorder
        
        # Store version dir for later snapshot
        recorder._version_dir = vdir
        recorder._version_id = vid
        recorder._task = task
        recorder._scene = scene
        recorder._device = device
        
        print(f"  📼 Recording started: {name}/{vid} → {log_path}")
        return recorder
    
    def stop_recording(self, name: str) -> Optional[TrainingRecorder]:
        """Stop recording but keep in store for snapshot to read frame_count."""
        recorder = self._active_recorders.get(name)
        if recorder:
            recorder.close()
            print(f"  📼 Recording stopped: {name} ({recorder.frame_count} frames)")
        return recorder
    
    # ── Snapshots ──
    
    def snapshot(self, name: str, tag: str = "",
                 brain_state: Optional[Dict] = None,
                 knowledge: Optional[Dict] = None,
                 skills: Optional[List[str]] = None,
                 training_result: Optional[Dict] = None,
                 consciousness_level: int = 0,
                 n_neurons: int = 0,
                 n_concepts: int = 0) -> str:
        """
        Save a versioned snapshot of the creature's current state.
        Returns version ID.
        """
        manifest = self._load_manifest(name)
        
        # If there's an active recorder, use its version dir
        recorder = self._active_recorders.get(name)
        if recorder and hasattr(recorder, '_version_dir'):
            vdir = recorder._version_dir
            vid = recorder._version_id
            task = recorder._task
            scene = recorder._scene
            device = recorder._device
            log_frames = recorder.frame_count
        else:
            vid = self._next_version_id(manifest)
            vdir = self._version_dir(name, vid)
            vdir.mkdir(parents=True, exist_ok=True)
            task = ""
            scene = ""
            device = "cpu"
            log_frames = 0
        
        # Save brain state
        brain_size_kb = 0.0
        if brain_state:
            brain_path = vdir / 'brain.pt'
            try:
                import torch
                torch.save(brain_state, brain_path)
                brain_size_kb = brain_path.stat().st_size / 1024
            except ImportError:
                # Fallback: save as JSON
                brain_path = vdir / 'brain.json'
                brain_path.write_text(json.dumps(brain_state, default=str), encoding='utf-8')
                brain_size_kb = brain_path.stat().st_size / 1024
        
        # Save knowledge graph
        if knowledge:
            kg_path = vdir / 'knowledge.json'
            kg_path.write_text(json.dumps(knowledge, default=str, indent=1), encoding='utf-8')
        
        # Save skills
        if skills:
            skills_dir = vdir / 'skills'
            skills_dir.mkdir(exist_ok=True)
            for skill_name in skills:
                # Copy skill checkpoint if it exists
                src = Path(f'checkpoints/{name}/skill_{skill_name}.json')
                if src.exists():
                    shutil.copy2(src, skills_dir / f'{skill_name}.json')
        
        # Save metadata
        tr = training_result or {}
        log_size_kb = 0.0
        log_path = vdir / 'training_log.bin'
        if log_path.exists():
            log_size_kb = log_path.stat().st_size / 1024
        
        version_meta = VersionMeta(
            version_id=vid,
            created=datetime.now().isoformat(),
            tag=tag,
            task=task or tr.get('task', ''),
            scene=scene or tr.get('scene', ''),
            best_fitness=tr.get('best_fitness', 0.0),
            total_generations=tr.get('total_generations', 0),
            training_time_s=tr.get('training_time_s', 0.0),
            device=device or tr.get('device', 'cpu'),
            n_neurons=n_neurons,
            consciousness_level=consciousness_level,
            n_concepts=n_concepts,
            n_skills=len(skills) if skills else 0,
            skills=skills or [],
            brain_size_kb=brain_size_kb,
            log_frames=log_frames,
            log_size_kb=log_size_kb,
        )
        
        meta_path = vdir / 'meta.json'
        meta_path.write_text(json.dumps(asdict(version_meta), indent=2, default=str), encoding='utf-8')
        
        # Update manifest
        manifest.versions.append(version_meta)
        manifest.active_version = vid
        self._save_manifest(manifest)
        
        # Clean up recorder reference
        self._active_recorders.pop(name, None)
        
        print(f"  💾 Snapshot saved: {name}/{vid}"
              f" (brain:{brain_size_kb:.0f}KB, log:{log_size_kb:.0f}KB, {log_frames} frames)")
        if tag:
            print(f"     Tag: \"{tag}\"")
        
        return vid
    
    # ── Restore ──
    
    def restore(self, name: str, version_id: str) -> Optional[Dict]:
        """
        Load a specific version. Returns dict with paths to saved files.
        """
        manifest = self._load_manifest(name)
        
        # Find version
        vmeta = None
        for v in manifest.versions:
            if v.version_id == version_id:
                vmeta = v
                break
        
        if not vmeta:
            print(f"  ❌ Version {version_id} not found for {name}")
            return None
        
        # Find version directory
        vdir = None
        for d in (self.base / name).iterdir():
            if d.is_dir() and d.name.startswith(version_id):
                vdir = d
                break
        
        if not vdir:
            # Check branches
            branches_dir = self.base / name / 'branches'
            if branches_dir.exists():
                for branch_dir in branches_dir.iterdir():
                    for d in branch_dir.iterdir():
                        if d.is_dir() and d.name.startswith(version_id):
                            vdir = d
                            break
        
        if not vdir:
            print(f"  ❌ Directory for {version_id} not found")
            return None
        
        manifest.active_version = version_id
        self._save_manifest(manifest)
        
        result = {
            'version': vmeta,
            'dir': str(vdir),
            'brain_path': None,
            'knowledge_path': None,
            'log_path': None,
            'skills_dir': None,
        }
        
        brain_pt = vdir / 'brain.pt'
        brain_json = vdir / 'brain.json'
        if brain_pt.exists():
            result['brain_path'] = str(brain_pt)
        elif brain_json.exists():
            result['brain_path'] = str(brain_json)
        
        kg = vdir / 'knowledge.json'
        if kg.exists():
            result['knowledge_path'] = str(kg)
        
        log = vdir / 'training_log.bin'
        if log.exists():
            result['log_path'] = str(log)
        
        skills = vdir / 'skills'
        if skills.exists():
            result['skills_dir'] = str(skills)
        
        print(f"  🔄 Restored: {name}/{version_id} — \"{vmeta.tag}\"")
        return result
    
    # ── Branch ──
    
    def branch(self, name: str, from_version: str, branch_name: str) -> Optional[str]:
        """Create a branch from a version."""
        manifest = self._load_manifest(name)
        
        # Find source version directory
        src_dir = None
        for d in (self.base / name).iterdir():
            if d.is_dir() and d.name.startswith(from_version):
                src_dir = d
                break
        
        if not src_dir:
            print(f"  ❌ Source version {from_version} not found")
            return None
        
        # Create branch
        dst_dir = self.base / name / 'branches' / branch_name / f"{from_version}_base"
        if dst_dir.exists():
            print(f"  ❌ Branch {branch_name} already exists")
            return None
        
        shutil.copytree(src_dir, dst_dir)
        
        if branch_name not in manifest.branches:
            manifest.branches.append(branch_name)
        self._save_manifest(manifest)
        
        print(f"  🌿 Branch created: {name}/{branch_name} (from {from_version})")
        return branch_name
    
    # ── Prune ──
    
    def prune(self, name: str, keep_last: int = 5, keep_tagged: bool = True) -> int:
        """Delete old versions, keep last N + tagged ones."""
        manifest = self._load_manifest(name)
        
        to_keep = set()
        # Keep last N
        for v in manifest.versions[-keep_last:]:
            to_keep.add(v.version_id)
        # Keep tagged
        if keep_tagged:
            for v in manifest.versions:
                if v.tag:
                    to_keep.add(v.version_id)
        
        removed = 0
        new_versions = []
        for v in manifest.versions:
            if v.version_id in to_keep:
                new_versions.append(v)
            else:
                # Find and delete directory
                for d in (self.base / name).iterdir():
                    if d.is_dir() and d.name.startswith(v.version_id):
                        shutil.rmtree(d)
                        removed += 1
                        break
        
        manifest.versions = new_versions
        self._save_manifest(manifest)
        
        if removed:
            print(f"  🗑️  Pruned {removed} versions of {name}, kept {len(new_versions)}")
        return removed
    
    # ── Summary ──
    
    def summary(self, name: str) -> str:
        """Human-readable summary of a creature."""
        manifest = self._load_manifest(name)
        lines = [f"🐾 {name} — {len(manifest.versions)} versions"]
        for v in manifest.versions:
            tag_str = f' "{v.tag}"' if v.tag else ''
            active = " ◄ ACTIVE" if v.version_id == manifest.active_version else ""
            lines.append(
                f"  {v.version_id} [{v.created[:16]}]{tag_str}"
                f"  fitness={v.best_fitness:.2f}  skills={v.n_skills}"
                f"  C{v.consciousness_level}{active}"
            )
        if manifest.branches and len(manifest.branches) > 1:
            lines.append(f"  Branches: {', '.join(manifest.branches)}")
        return '\n'.join(lines)
    
    def compare(self, name: str, vid_a: str, vid_b: str) -> str:
        """Compare two versions of a creature. Returns human-readable diff."""
        manifest = self._load_manifest(name)
        version_map = {v.version_id: v for v in manifest.versions}
        
        a = version_map.get(vid_a)
        b = version_map.get(vid_b)
        if not a:
            return f"❌ Version {vid_a} not found for {name}"
        if not b:
            return f"❌ Version {vid_b} not found for {name}"
        
        lines = [
            f"\n🔍 COMPARE {name}: {vid_a} vs {vid_b}",
            f"{'='*50}",
            f"  {'':20s} {'   ' + vid_a:>12s}  {'   ' + vid_b:>12s}  {'delta':>10s}",
            f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}",
        ]
        
        # Fitness
        df = b.best_fitness - a.best_fitness
        arrow = '↑' if df > 0 else '↓' if df < 0 else '='
        lines.append(f"  {'Fitness':20s} {a.best_fitness:12.3f} {b.best_fitness:12.3f} {arrow} {df:+.3f}")
        
        # Generations
        dg = b.total_generations - a.total_generations
        lines.append(f"  {'Generations':20s} {a.total_generations:12d} {b.total_generations:12d} {dg:+10d}")
        
        # Training time
        dt = b.training_time_s - a.training_time_s
        lines.append(f"  {'Training (s)':20s} {a.training_time_s:12.1f} {b.training_time_s:12.1f} {dt:+10.1f}")
        
        # Consciousness
        dc = b.consciousness_level - a.consciousness_level
        arrow = '↑' if dc > 0 else '↓' if dc < 0 else '='
        lines.append(f"  {'Consciousness':20s} {'C'+str(a.consciousness_level):>12s} {'C'+str(b.consciousness_level):>12s} {arrow} {dc:+d}")
        
        # Neurons
        lines.append(f"  {'Neurons':20s} {a.n_neurons:12,d} {b.n_neurons:12,d}")
        
        # Skills
        skills_a = set(a.skills)
        skills_b = set(b.skills)
        gained = skills_b - skills_a
        lost = skills_a - skills_b
        kept = skills_a & skills_b
        lines.append(f"  {'Skills':20s} {a.n_skills:12d} {b.n_skills:12d}")
        if gained:
            lines.append(f"    🌟 Gained: {', '.join(gained)}")
        if lost:
            lines.append(f"    ❌ Lost: {', '.join(lost)}")
        if kept:
            lines.append(f"    ✅ Kept: {', '.join(kept)}")
        
        # Data sizes
        lines.append(f"  {'Brain (KB)':20s} {a.brain_size_kb:12.1f} {b.brain_size_kb:12.1f}")
        lines.append(f"  {'Log frames':20s} {a.log_frames:12d} {b.log_frames:12d}")
        
        # Tags
        lines.append(f"\n  {vid_a}: {a.tag or '(no tag)'}  [{a.created[:16]}]")
        lines.append(f"  {vid_b}: {b.tag or '(no tag)'}  [{b.created[:16]}]")
        
        return '\n'.join(lines)
