"""
MH-FLOCKE — Brain Persistence v0.4.1
========================================
Save and load brain state across training sessions.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, Optional, Any
from pathlib import Path


def save_brain(brain, snn, path: str, metadata: Optional[Dict] = None):
    """
    Speichert kompletten CognitiveBrain-State.
    
    Args:
        brain: CognitiveBrain Instanz
        snn: SNNController Instanz
        path: Zielpfad (.pt Datei)
        metadata: Optional dict mit Name, Generation, etc.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    state = {
        'version': 2,
        'metadata': metadata or {},
        'step_count': brain._step_count,
        'consciousness_level': brain._consciousness_level,
        'prediction_error': brain._prediction_error,
        'curiosity_reward': brain._curiosity_reward,
        'cumulative_curiosity': brain._cumulative_curiosity,
        'gwt_winner': brain._gwt_winner,
        'last_pci': brain._last_pci,
    }
    
    # === 1. SNN (the core) ===
    state['snn'] = _save_snn(snn)
    
    # === 2. World Model ===
    state['world_model'] = _save_world_model(brain.world_model)
    
    # === 3. GWT ===
    state['gwt'] = _save_gwt(brain.gwt)
    
    # === 4. Emotions ===
    state['emotions'] = _save_emotions(brain.emotions)
    
    # === 5. Body Schema ===
    state['body_schema'] = _save_body_schema(brain.body_schema)
    
    # === 6. Sensomotor Memory ===
    state['memory'] = _save_memory(brain.memory)
    
    # === 7. Drives ===
    state['drives'] = _save_drives(brain.drives)
    
    # === 8. Metacognition ===
    state['metacognition'] = _save_metacognition(brain.metacognition)
    
    # === 9. Consistency ===
    state['consistency'] = _save_consistency(brain.consistency)
    
    # === 10. Synaptogenesis (Graph + Buffer) ===
    state['synaptogenesis'] = _save_synaptogenesis(brain.synaptogenesis)
    
    # === 11. Astrocyte ===
    state['astrocytes'] = _save_astrocytes(brain.astrocytes)
    
    # === 12. Dream Engine ===
    state['dream'] = _save_dream_engine(brain.dream_engine)
    
    # === 13. Modular Skills ===
    state['skills'] = brain.skills.save_state()
    
    torch.save(state, path)
    return path


def load_brain(brain, snn, path: str) -> Dict:
    """
    Lädt kompletten CognitiveBrain-State.
    
    Args:
        brain: CognitiveBrain Instanz (muss bereits initialisiert sein)
        snn: SNNController Instanz
        path: Quellpfad (.pt Datei)
        
    Returns:
        Metadata dict aus dem gespeicherten State
    """
    state = torch.load(path, map_location='cpu', weights_only=False)
    
    # Basis-State
    brain._step_count = state.get('step_count', 0)
    brain._consciousness_level = state.get('consciousness_level', 0)
    brain._prediction_error = state.get('prediction_error', 0.0)
    brain._curiosity_reward = state.get('curiosity_reward', 0.0)
    brain._cumulative_curiosity = state.get('cumulative_curiosity', 0.0)
    brain._gwt_winner = state.get('gwt_winner', '')
    brain._last_pci = state.get('last_pci', 0.0)
    
    # Module laden
    if 'snn' in state:
        _load_snn(snn, state['snn'])
    if 'world_model' in state:
        _load_world_model(brain.world_model, state['world_model'])
    if 'gwt' in state:
        _load_gwt(brain.gwt, state['gwt'])
    if 'emotions' in state:
        _load_emotions(brain.emotions, state['emotions'])
    if 'body_schema' in state:
        _load_body_schema(brain.body_schema, state['body_schema'])
    if 'memory' in state:
        _load_memory(brain.memory, state['memory'])
    if 'drives' in state:
        _load_drives(brain.drives, state['drives'])
    if 'metacognition' in state:
        _load_metacognition(brain.metacognition, state['metacognition'])
    if 'consistency' in state:
        _load_consistency(brain.consistency, state['consistency'])
    if 'synaptogenesis' in state:
        _load_synaptogenesis(brain.synaptogenesis, state['synaptogenesis'])
    if 'astrocytes' in state:
        _load_astrocytes(brain.astrocytes, state['astrocytes'])
    if 'dream' in state:
        _load_dream_engine(brain.dream_engine, state['dream'])
    if 'skills' in state:
        brain.skills.load_state(state['skills'])
    
    return state.get('metadata', {})


# ============================================================
# SNN Serialization
# ============================================================

def _save_snn(snn) -> dict:
    """SNN-Gewichte, Traces, Neuromodulatoren."""
    return {
        'V': snn.V.cpu(),
        'spikes': snn.spikes.cpu(),
        'refractory_counter': snn.refractory_counter.cpu(),
        'neuron_types': snn.neuron_types.cpu(),
        'neuromod_levels': dict(snn.neuromod_levels),
        'neuromod_sensitivity': snn.neuromod_sensitivity.cpu(),
        'populations': {k: v.cpu() for k, v in snn.populations.items()},
        'weight_indices': snn._weight_indices.cpu() if snn._weight_indices is not None else None,
        'weight_values': snn._weight_values.cpu() if snn._weight_values is not None else None,
        'eligibility': snn._eligibility.cpu(),
        'pre_trace': snn._pre_trace.cpu(),
        'post_trace': snn._post_trace.cpu(),
        'thresholds': snn._thresholds.cpu(),
        'astro_calcium': snn._astro_calcium.cpu(),
        'spike_count_window': snn._spike_count_window.cpu(),
        'homeostatic_step_count': snn._homeostatic_step_count,
        'step_count': snn.step_count,
    }


def _load_snn(snn, data: dict):
    """SNN-State wiederherstellen."""
    device = snn.device
    snn.V = data['V'].to(device)
    snn.spikes = data['spikes'].to(device)
    snn.refractory_counter = data['refractory_counter'].to(device)
    snn.neuron_types = data['neuron_types'].to(device)
    snn.neuromod_levels = data['neuromod_levels']
    snn.neuromod_sensitivity = data['neuromod_sensitivity'].to(device)
    snn.populations = {k: v.to(device) for k, v in data['populations'].items()}
    
    if data['weight_indices'] is not None:
        snn._weight_indices = data['weight_indices'].to(device)
    if data['weight_values'] is not None:
        snn._weight_values = data['weight_values'].to(device)
        snn._rebuild_sparse_weights()
    
    snn._eligibility = data['eligibility'].to(device)
    snn._pre_trace = data['pre_trace'].to(device)
    snn._post_trace = data['post_trace'].to(device)
    snn._thresholds = data['thresholds'].to(device)
    snn._astro_calcium = data['astro_calcium'].to(device)
    snn._spike_count_window = data['spike_count_window'].to(device)
    snn._homeostatic_step_count = data['homeostatic_step_count']
    snn.step_count = data['step_count']


# ============================================================
# World Model
# ============================================================

def _save_world_model(wm) -> dict:
    """World Model SNN + Prediction History."""
    return {
        'snn': _save_snn(wm.snn),
        'prediction_history': list(wm._prediction_history)[-200:] if hasattr(wm, '_prediction_history') else [],
        'n_sensors': wm.n_sensors,
        'n_motors': wm.n_motors,
    }


def _load_world_model(wm, data: dict):
    """World Model wiederherstellen."""
    if 'snn' in data:
        _load_snn(wm.snn, data['snn'])
    if hasattr(wm, '_prediction_history'):
        wm._prediction_history = data.get('prediction_history', [])


# ============================================================
# GWT (Global Workspace Theory)
# ============================================================

def _save_gwt(gwt) -> dict:
    """GWT Module-States + Broadcast."""
    modules = {}
    for name, mod in gwt._modules.items():
        modules[name] = {
            'activation': mod.activation.cpu() if hasattr(mod, 'activation') else None,
            'salience_history': list(getattr(mod, '_salience_history', []))[-50:],
        }
    return {
        'modules': modules,
        'winning_module': gwt._winning_module,
        'broadcast_signal': gwt._broadcast_signal.cpu(),
        'broadcast_age': gwt._broadcast_age,
        'history': list(gwt._history)[-100:] if gwt._history else [],
    }


def _load_gwt(gwt, data: dict):
    """GWT wiederherstellen."""
    device = gwt.device
    gwt._winning_module = data.get('winning_module', '')
    gwt._broadcast_signal = data['broadcast_signal'].to(device) if 'broadcast_signal' in data else gwt._broadcast_signal
    gwt._broadcast_age = data.get('broadcast_age', 0)
    gwt._history = data.get('history', [])
    
    for name, mod_data in data.get('modules', {}).items():
        if name in gwt._modules:
            mod = gwt._modules[name]
            if mod_data.get('activation') is not None:
                mod.activation = mod_data['activation'].to(device)


# ============================================================
# Emotions
# ============================================================

def _save_emotions(emo) -> dict:
    """Emotionaler Zustand + History."""
    return {
        'valence': emo.state.valence,
        'arousal': emo.state.arousal,
        'dominant_emotion': emo.state.dominant_emotion,
        'history': [
            {'valence': h.valence, 'arousal': h.arousal, 'emotion': h.dominant_emotion}
            for h in list(emo.history)[-200:]
        ],
        'step_count': emo._step_count,
    }


def _load_emotions(emo, data: dict):
    """Emotionen wiederherstellen."""
    emo.state.valence = data.get('valence', 0.0)
    emo.state.arousal = data.get('arousal', 0.5)
    emo.state.dominant_emotion = data.get('dominant_emotion', 'neutral')
    emo._step_count = data.get('step_count', 0)


# ============================================================
# Body Schema
# ============================================================

def _save_body_schema(bs) -> dict:
    """Forward Model + Confidences."""
    return {
        'forward_weights': bs.forward_weights.tolist(),
        'forward_bias': bs.forward_bias.tolist(),
        'joint_confidence': bs.joint_confidence.tolist(),
        'body_confidence': bs._body_confidence,
        'update_count': bs._update_count,
        'total_error': bs._total_error,
    }


def _load_body_schema(bs, data: dict):
    """Body Schema wiederherstellen."""
    if 'forward_weights' in data:
        w = np.array(data['forward_weights'], dtype=np.float32)
        if w.shape == bs.forward_weights.shape:
            bs.forward_weights = w
    if 'forward_bias' in data:
        b = np.array(data['forward_bias'], dtype=np.float32)
        if b.shape == bs.forward_bias.shape:
            bs.forward_bias = b
    if 'joint_confidence' in data:
        jc = np.array(data['joint_confidence'], dtype=np.float32)
        if jc.shape == bs.joint_confidence.shape:
            bs.joint_confidence = jc
    bs._body_confidence = data.get('body_confidence', 0.0)
    bs._update_count = data.get('update_count', 0)
    bs._total_error = data.get('total_error', 0.0)


# ============================================================
# Sensomotor Memory
# ============================================================

def _save_memory(mem) -> dict:
    """Episoden serialisieren."""
    episodes = []
    for ep in mem.episodes:
        episodes.append({
            'sensor_seq': ep.sensor_seq.tolist(),
            'motor_seq': ep.motor_seq.tolist(),
            'reward_seq': ep.reward_seq.tolist(),
            'valence': ep.valence,
            'arousal': ep.arousal,
            'total_reward': ep.total_reward,
            'start_step': ep.start_step,
            'pattern_hash': ep.pattern_hash,
        })
    return {
        'episodes': episodes,
        'total_recorded': mem._total_recorded,
        'total_recalled': mem._total_recalled,
    }


def _load_memory(mem, data: dict):
    """Episoden laden."""
    from src.brain.sensomotor_memory import Episode
    
    mem.episodes = []
    for ep_data in data.get('episodes', []):
        ep = Episode(
            sensor_seq=np.array(ep_data['sensor_seq'], dtype=np.float32),
            motor_seq=np.array(ep_data['motor_seq'], dtype=np.float32),
            reward_seq=np.array(ep_data['reward_seq'], dtype=np.float32),
            valence=ep_data.get('valence', 0.0),
            arousal=ep_data.get('arousal', 0.5),
            total_reward=ep_data.get('total_reward', 0.0),
            start_step=ep_data.get('start_step', 0),
            pattern_hash=ep_data.get('pattern_hash', 0),
        )
        mem.episodes.append(ep)
    mem._total_recorded = data.get('total_recorded', 0)
    mem._total_recalled = data.get('total_recalled', 0)


# ============================================================
# Drives
# ============================================================

def _save_drives(drives) -> dict:
    """Drive-Levels."""
    return {
        'survival': drives.state.survival,
        'exploration': drives.state.exploration,
        'comfort': drives.state.comfort,
        'social': drives.state.social,
        'dominant': drives.state.dominant,
        'step_count': drives._step_count,
    }


def _load_drives(drives, data: dict):
    """Drives wiederherstellen."""
    drives.state.survival = data.get('survival', 0.5)
    drives.state.exploration = data.get('exploration', 0.3)
    drives.state.comfort = data.get('comfort', 0.2)
    drives.state.social = data.get('social', 0.0)
    drives.state.dominant = data.get('dominant', 'survival')
    drives._step_count = data.get('step_count', 0)


# ============================================================
# Metacognition
# ============================================================

def _save_metacognition(meta) -> dict:
    """Metacognition-State."""
    return {
        'world_model_accuracy': meta.world_model_accuracy,
        'body_schema_confidence': meta.body_schema_confidence,
        'learning_progress': meta.learning_progress,
        'consciousness_level': meta.consciousness_level,
        'pe_history': list(meta._pe_history),
        'fitness_history': list(meta._fitness_history),
        'modules_active': dict(meta._modules_active),
        'step_count': meta._step_count,
    }


def _load_metacognition(meta, data: dict):
    """Metacognition wiederherstellen."""
    meta.world_model_accuracy = data.get('world_model_accuracy', 0.5)
    meta.body_schema_confidence = data.get('body_schema_confidence', 0.0)
    meta.learning_progress = data.get('learning_progress', 0.0)
    meta.consciousness_level = data.get('consciousness_level', 0)
    from collections import deque
    meta._pe_history = deque(data.get('pe_history', []), maxlen=meta.ACCURACY_WINDOW)
    meta._fitness_history = deque(data.get('fitness_history', []), maxlen=meta.PROGRESS_WINDOW)
    meta._modules_active = data.get('modules_active', meta._modules_active)
    meta._step_count = data.get('step_count', 0)


# ============================================================
# Consistency Checker
# ============================================================

def _save_consistency(cc) -> dict:
    """Consistency State."""
    return {
        'last_dissonance': cc._last_dissonance,
        'dissonance_history': list(cc._dissonance_history)[-200:],
        'step_count': cc._step_count,
        'alert_count': cc._alert_count,
    }


def _load_consistency(cc, data: dict):
    """Consistency wiederherstellen."""
    cc._last_dissonance = data.get('last_dissonance', 0.0)
    cc._dissonance_history = data.get('dissonance_history', [])
    cc._step_count = data.get('step_count', 0)
    cc._alert_count = data.get('alert_count', 0)


# ============================================================
# Synaptogenesis (Graph + Buffer)
# ============================================================

def _save_synaptogenesis(synapto) -> dict:
    """Concept Graph + Experience Buffer."""
    # Graph
    concepts = {}
    for cid, c in synapto.graph._concepts.items():
        concepts[str(cid)] = {
            'id': c['id'],
            'label': c['label'],
            'pattern': c['pattern'].cpu().tolist(),
            'valence': c['valence'],
            'properties': c['properties'],
            'activation_count': c['activation_count'],
        }
    
    relations = [
        {'src': r[0], 'tgt': r[1], 'type': r[2], 'weight': r[3]}
        for r in synapto.graph._relations
    ]
    
    # Experience Buffer (deque doesn't support slicing — convert to list first)
    buffer_data = []
    if hasattr(synapto, 'buffer') and hasattr(synapto.buffer, '_buffer'):
        for entry in list(synapto.buffer._buffer)[-500:]:
            if hasattr(entry, 'pattern'):
                buffer_data.append({
                    'pattern': entry.pattern.cpu().tolist() if torch.is_tensor(entry.pattern) else entry.pattern,
                    'valence': getattr(entry, 'valence', 0.0),
                    'step': getattr(entry, 'step', 0),
                })
    
    return {
        'concepts': concepts,
        'relations': relations,
        'next_id': synapto.graph._next_id,
        'buffer': buffer_data,
        'spike_window': [s.cpu().tolist() if torch.is_tensor(s) else s 
                         for s in list(synapto._spike_window)[-50:]],
    }


def _load_synaptogenesis(synapto, data: dict):
    """Synaptogenesis wiederherstellen."""
    # Graph
    synapto.graph._concepts = {}
    for cid_str, c in data.get('concepts', {}).items():
        cid = int(cid_str)
        synapto.graph._concepts[cid] = {
            'id': c['id'],
            'label': c['label'],
            'pattern': torch.tensor(c['pattern'], dtype=torch.float32),
            'valence': c['valence'],
            'properties': c.get('properties', {}),
            'activation_count': c.get('activation_count', 1),
        }
    
    synapto.graph._relations = [
        (r['src'], r['tgt'], r['type'], r['weight'])
        for r in data.get('relations', [])
    ]
    synapto.graph._next_id = data.get('next_id', 0)


# ============================================================
# Astrocyte
# ============================================================

def _save_astrocytes(astro) -> dict:
    """Calcium-Levels."""
    return {
        'calcium': astro.calcium.tolist(),
        'update_count': astro.update_count,
        'above_threshold_count': astro.above_threshold_count,
    }


def _load_astrocytes(astro, data: dict):
    """Astrocyte wiederherstellen."""
    if 'calcium' in data:
        ca = np.array(data['calcium'], dtype=np.float32)
        if ca.shape == astro.calcium.shape:
            astro.calcium = ca
    astro.update_count = data.get('update_count', 0)
    astro.above_threshold_count = data.get('above_threshold_count', 0)


# ============================================================
# Dream Engine
# ============================================================

def _save_dream_engine(dream) -> dict:
    """Replay Buffer."""
    buffer = []
    if hasattr(dream, '_replay_buffer'):
        for entry in list(dream._replay_buffer)[-200:]:
            if isinstance(entry, dict):
                serializable = {}
                for k, v in entry.items():
                    if torch.is_tensor(v):
                        serializable[k] = v.cpu().tolist()
                    elif isinstance(v, np.ndarray):
                        serializable[k] = v.tolist()
                    else:
                        serializable[k] = v
                buffer.append(serializable)
    return {
        'replay_buffer': buffer,
        'total_dreams': getattr(dream, '_total_dreams', 0),
    }


def _load_dream_engine(dream, data: dict):
    """Dream Engine wiederherstellen."""
    if hasattr(dream, '_replay_buffer'):
        dream._replay_buffer = []
        for entry in data.get('replay_buffer', []):
            restored = {}
            for k, v in entry.items():
                if isinstance(v, list) and len(v) > 0:
                    restored[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    restored[k] = v
            dream._replay_buffer.append(restored)
    if hasattr(dream, '_total_dreams'):
        dream._total_dreams = data.get('total_dreams', 0)


# ============================================================
# Convenience: Brain-Info ohne Laden
# ============================================================

def brain_info(path: str) -> dict:
    """
    Liest Metadaten eines gespeicherten Brain-States ohne alles zu laden.
    
    Returns:
        Dict mit version, step_count, consciousness_level, etc.
    """
    state = torch.load(path, map_location='cpu', weights_only=False)
    info = {
        'version': state.get('version', 1),
        'metadata': state.get('metadata', {}),
        'step_count': state.get('step_count', 0),
        'consciousness_level': state.get('consciousness_level', 0),
        'last_pci': state.get('last_pci', 0.0),
    }
    
    # Module-Presence
    info['modules'] = {
        key: key in state
        for key in ['snn', 'world_model', 'gwt', 'emotions', 'body_schema',
                     'memory', 'drives', 'metacognition', 'consistency',
                     'synaptogenesis', 'astrocytes', 'dream']
    }
    
    # SNN stats
    if 'snn' in state:
        snn_data = state['snn']
        if 'weight_values' in snn_data and snn_data['weight_values'] is not None:
            info['n_synapses'] = len(snn_data['weight_values'])
        info['snn_steps'] = snn_data.get('step_count', 0)
    
    # Memory stats
    if 'memory' in state:
        info['n_episodes'] = len(state['memory'].get('episodes', []))
    
    # Graph stats
    if 'synaptogenesis' in state:
        info['n_concepts'] = len(state['synaptogenesis'].get('concepts', {}))
        info['n_relations'] = len(state['synaptogenesis'].get('relations', []))
    
    return info
