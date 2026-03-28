#!/usr/bin/env python3
"""
MH-FLOCKE — Neural Configuration v0.4.1
========================================
Adaptive SNN size profiles based on hardware detection.
"""

import time
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuralProfile:
    """Configuration profile for neural network."""
    name: str
    n_neurons: int
    simulation_steps: int
    workspace_query_steps: int
    workspace_max_concepts: int
    stdp_frequency: int
    stdp_max_updates: int
    energy_calc_frequency: int
    neural_query_cache_ttl: int
    estimated_step_ms: float
    description: str


# PROFILE DEFINITIONS
PROFILES = {
    'minimal': NeuralProfile(
        name='minimal',
        n_neurons=10_000,
        simulation_steps=10,
        workspace_query_steps=5,
        workspace_max_concepts=1,
        stdp_frequency=50,
        stdp_max_updates=100,
        energy_calc_frequency=5,
        neural_query_cache_ttl=120,
        estimated_step_ms=2.0,
        description='Lightweight (RAM < 4GB, old CPUs)'
    ),
    
    'standard': NeuralProfile(
        name='standard',
        n_neurons=50_000,
        simulation_steps=25,
        workspace_query_steps=10,
        workspace_max_concepts=2,
        stdp_frequency=10,
        stdp_max_updates=1000,
        energy_calc_frequency=3,
        neural_query_cache_ttl=60,
        estimated_step_ms=2.0,
        description='Balanced (RAM 4-16GB, modern CPUs)'
    ),
    
    'performance': NeuralProfile(
        name='performance',
        n_neurons=150_000,  # ← NEW! 3x more neurons
        simulation_steps=50,
        workspace_query_steps=20,
        workspace_max_concepts=3,
        stdp_frequency=5,
        stdp_max_updates=2000,
        energy_calc_frequency=2,
        neural_query_cache_ttl=30,
        estimated_step_ms=2.0,  # Still fast with Numba!
        description='High-performance (RAM 16GB+, fast CPUs, Numba-optimized)'
    ),
    
    'extreme': NeuralProfile(
        name='extreme',
        n_neurons=300_000,  # ← NEW! Maximum capacity
        simulation_steps=100,
        workspace_query_steps=50,
        workspace_max_concepts=5,
        stdp_frequency=3,
        stdp_max_updates=5000,
        energy_calc_frequency=1,
        neural_query_cache_ttl=15,
        estimated_step_ms=1.0,  # GPU or very fast CPU
        description='Maximum (RAM 32GB+, CUDA available, research-grade)'
    ),
}


def benchmark_neural_performance(n_test: int = 10_000) -> float:
    """
    Benchmark: Measure neural simulation speed on this hardware.
    
    Returns:
        Milliseconds per simulation step (10k neurons)
    """
    try:
        from scipy.sparse import random as sparse_random
    except ImportError:
        return 5.0  # Conservative fallback
    
    # Create mini network for benchmark
    n = n_test
    V = np.zeros(n, dtype=np.float32)
    thresholds = np.random.normal(0.5, 0.05, n).astype(np.float32)
    spikes = np.zeros(n, dtype=bool)
    
    # Sparse synapses (realistic density)
    density = min(50.0 / n, 0.01)
    W = sparse_random(n, n, density=density, format='csr', dtype=np.float32)
    
    # Run 10 simulation steps
    start = time.perf_counter()
    for _ in range(10):
        spike_vec = spikes.astype(np.float32)
        syn_current = W.T.dot(spike_vec)
        leak = -0.1 * (V - (-0.065))
        V += (leak + syn_current) * 0.001
        spikes = V > thresholds
        V[spikes] = -0.065
    elapsed = time.perf_counter() - start
    
    ms_per_step = (elapsed / 10) * 1000
    return ms_per_step


def detect_profile(force: Optional[str] = None) -> NeuralProfile:
    """
    Auto-detect optimal neural profile for current hardware.
    
    Detection strategy (Phase 8):
    1. Check for CUDA → 'extreme' if 32GB+ RAM
    2. Check RAM:
       - 32GB+ → 'extreme' (if CUDA) or 'performance' (if no CUDA)
       - 16GB+ → 'performance'
       - 8GB+  → 'standard'
       - <8GB  → 'minimal'
    3. Verify with benchmark cache if available
    
    Args:
        force: Override with specific profile name
    
    Returns:
        NeuralProfile
    """
    # Force override
    if force and force in PROFILES:
        profile = PROFILES[force]
        print(f"   ⚙️  Neural Profile: {profile.name} (forced)")
        print(f"      → {profile.n_neurons:,} neurons, {profile.description}")
        return profile
    
    # Detect hardware
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count() or 4
    
    # Check for CUDA
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            cuda_available = True
        except:
            pass
    
    # Profile selection logic
    if cuda_available and ram_gb >= 32:
        profile = PROFILES['extreme']
        print(f"   ⚙️  Neural Profile: {profile.name} (CUDA detected, {ram_gb:.0f}GB RAM)")
    elif ram_gb >= 32:
        profile = PROFILES['performance']
        print(f"   ⚙️  Neural Profile: {profile.name} ({ram_gb:.0f}GB RAM, {cpu_count} cores)")
    elif ram_gb >= 16:
        profile = PROFILES['performance']
        print(f"   ⚙️  Neural Profile: {profile.name} ({ram_gb:.0f}GB RAM)")
    elif ram_gb >= 8:
        profile = PROFILES['standard']
        print(f"   ⚙️  Neural Profile: {profile.name} ({ram_gb:.0f}GB RAM)")
    else:
        profile = PROFILES['minimal']
        print(f"   ⚙️  Neural Profile: {profile.name} ({ram_gb:.0f}GB RAM)")
    
    print(f"      → {profile.n_neurons:,} neurons, {profile.description}")
    
    # Check cached benchmark
    cache_file = os.path.join('data', 'neural_benchmark.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            ms = cached.get('ms_per_step_10k', 0)
            if ms > 0:
                print(f"      → Cached benchmark: {ms:.1f}ms/step (10k neurons)")
        except:
            pass
    
    return profile


def _run_and_cache_benchmark(cache_file: str) -> float:
    """Run benchmark and cache results."""
    print("   ⏱️  Running neural benchmark...")
    ms = benchmark_neural_performance()
    print(f"      → Result: {ms:.1f}ms per step (10k neurons)")
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'ms_per_step_10k': round(ms, 2),
                'timestamp': time.time(),
            }, f)
    except:
        pass
    
    return ms


if __name__ == '__main__':
    """Test profile detection"""
    print("="*70)
    print("NEURAL CONFIGURATION - PHASE 8 PROFILES")
    print("="*70)
    
    print("\n📊 Available Profiles:")
    for name, profile in PROFILES.items():
        print(f"\n  {name.upper()}:")
        print(f"    Neurons: {profile.n_neurons:,}")
        print(f"    Description: {profile.description}")
        print(f"    STDP Frequency: every {profile.stdp_frequency} steps")
        print(f"    Max STDP Updates: {profile.stdp_max_updates:,}")
    
    print("\n" + "="*70)
    print("🔍 Auto-Detection:")
    print("="*70)
    
    profile = detect_profile()
    
    print(f"\n✅ Selected: {profile.name.upper()}")
    print(f"   Neurons: {profile.n_neurons:,}")
    print(f"   Simulation Steps: {profile.simulation_steps}")
    print(f"   Description: {profile.description}")
    
    print("\n" + "="*70)
