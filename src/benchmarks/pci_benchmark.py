"""
MH-FLOCKE — PCI Benchmark v0.4.1
========================================
Perturbational Complexity Index measurement for SNN evaluation.
"""

import numpy as np
from typing import Dict, Any, Optional
import time


class PCIBenchmark:
    """Berechnet den Perturbational Complexity Index für SNNs."""

    def __init__(self, nn):
        self.nn = nn

    def compute_pci(self, stimulus_size: int = 100,
                     baseline_ms: int = 200,
                     response_ms: int = 300,
                     sample_neurons: int = 1000,
                     n_bins_per_ms: float = 0.1) -> Dict[str, Any]:
        """
        Berechnet PCI.
        
        Args:
            stimulus_size: Anzahl stimulierter Neuronen
            baseline_ms: Baseline-Aufzeichnung (keine Stimulation)
            response_ms: Antwort-Aufzeichnung (nach Stimulus)
            sample_neurons: Anzahl aufgezeichneter Neuronen
            n_bins_per_ms: Zeitauflösung (0.1 = 10ms Bins)
            
        Returns:
            Dict mit pci, heatmap, threshold_info
        """
        nn = self.nn
        nn._ensure_csr()
        n = nn.n_neurons

        # Select stimulus and recording neurons
        stimulus_neurons = np.random.choice(n, min(stimulus_size, n), replace=False)
        record_neurons = np.random.choice(n, min(sample_neurons, n), replace=False)
        record_set = {int(nid): i for i, nid in enumerate(record_neurons)}

        bin_size_ms = int(1.0 / n_bins_per_ms)  # 10ms Bins
        n_baseline_bins = baseline_ms // bin_size_ms
        n_response_bins = response_ms // bin_size_ms

        # Save state
        saved_state = nn.neuron_state.copy()
        saved_energy = nn.neuron_energy.copy()

        # === Phase 1: Baseline (tonische Hintergrund-Stimulation) ===
        baseline_activity = np.zeros((len(record_neurons), n_baseline_bins), dtype=np.float32)
        for t in range(baseline_ms):
            # Tonische Stimulation (wie sensorischer Input)
            n_stim = max(1, n // 50)
            stim_ids = np.random.choice(n, n_stim, replace=False)
            tonic = np.zeros(n, dtype=np.float32)
            tonic[stim_ids] = np.random.uniform(0.5, 1.2, n_stim).astype(np.float32)
            nn.simulate(tonic, duration_ms=1)
            bin_idx = t // bin_size_ms
            if bin_idx >= n_baseline_bins:
                break
            spiked = np.where(nn.current_spikes)[0]
            for nid in spiked:
                if int(nid) in record_set:
                    baseline_activity[record_set[int(nid)], bin_idx] += 1

        # Baseline statistics (mean + std per neuron)
        baseline_mean = baseline_activity.mean(axis=1, keepdims=True)
        baseline_std = baseline_activity.std(axis=1, keepdims=True)
        baseline_std[baseline_std < 0.1] = 0.1  # Minimum Std

        # === Phase 2: Stimulus ===
        # Injiziere starke Aktivierung in Stimulus-Neuronen
        nn.neuron_state[stimulus_neurons] = nn.spike_threshold * 2.0

        # === Phase 3: Response-Aufzeichnung (mit tonischer Stimulation) ===
        response_activity = np.zeros((len(record_neurons), n_response_bins), dtype=np.float32)
        for t in range(response_ms):
            # Gleiche tonische Stimulation wie Baseline
            n_stim = max(1, n // 50)
            stim_ids = np.random.choice(n, n_stim, replace=False)
            tonic = np.zeros(n, dtype=np.float32)
            tonic[stim_ids] = np.random.uniform(0.5, 1.2, n_stim).astype(np.float32)
            nn.simulate(tonic, duration_ms=1)
            bin_idx = t // bin_size_ms
            if bin_idx >= n_response_bins:
                break
            spiked = np.where(nn.current_spikes)[0]
            for nid in spiked:
                if int(nid) in record_set:
                    response_activity[record_set[int(nid)], bin_idx] += 1

        # Reset state
        nn.neuron_state[:] = saved_state
        nn.neuron_energy[:] = saved_energy

        # === Phase 4: Signifikante Antworten binarisieren ===
        # z-Score: (response - baseline_mean) / baseline_std
        z_scores = (response_activity - baseline_mean) / baseline_std
        
        # Significance threshold: z > 3.0
        z_threshold = 3.0
        binary_matrix = (z_scores > z_threshold).astype(np.uint8)

        # Remove empty rows and columns
        row_sums = binary_matrix.sum(axis=1)
        col_sums = binary_matrix.sum(axis=0)
        active_rows = row_sums > 0
        active_cols = col_sums > 0
        trimmed = binary_matrix[active_rows][:, active_cols]

        if trimmed.size == 0:
            return {
                'pci': 0.0,
                'reason': 'no_significant_response',
                'threshold': 0.31,
                'above_threshold': False
            }

        # === Phase 5: Lempel-Ziv Complexity ===
        # Flatten Matrix zeilenweise zu Bitstring
        bitstring = trimmed.flatten()
        lz_complexity = self._lempel_ziv_complexity(bitstring)

        # Normalisierung: LZ / (n_bits / log2(n_bits))
        n_bits = len(bitstring)
        if n_bits > 1:
            normalization = n_bits / np.log2(n_bits)
            pci = lz_complexity / normalization
        else:
            pci = 0.0

        pci = min(1.0, max(0.0, pci))  # Clamp [0, 1]

        # Heatmap data for dashboard (subsampled)
        max_display_neurons = 100
        max_display_bins = 50
        heatmap_step_n = max(1, len(record_neurons) // max_display_neurons)
        heatmap_step_t = max(1, n_response_bins // max_display_bins)

        return {
            'pci': round(pci, 4),
            'threshold': 0.31,
            'above_threshold': pci > 0.31,
            'lz_complexity': round(lz_complexity, 2),
            'binary_matrix_size': list(trimmed.shape),
            'n_significant_elements': int(trimmed.sum()),
            'total_elements': int(trimmed.size),
            'significant_fraction': round(float(trimmed.sum()) / max(trimmed.size, 1), 4),
            'heatmap': {
                'z_scores': z_scores[::heatmap_step_n, ::heatmap_step_t].tolist(),
                'binary': binary_matrix[::heatmap_step_n, ::heatmap_step_t].tolist(),
                'n_neurons': min(max_display_neurons, len(record_neurons)),
                'n_bins': min(max_display_bins, n_response_bins),
                'bin_size_ms': bin_size_ms
            },
            'stimulus': {
                'n_neurons': len(stimulus_neurons),
                'strength': float(nn.spike_threshold * 2.0)
            },
            'timing': {
                'baseline_ms': baseline_ms,
                'response_ms': response_ms
            }
        }

    def _lempel_ziv_complexity(self, sequence: np.ndarray) -> float:
        """
        Berechnet Lempel-Ziv-Komplexität einer binären Sequenz.
        
        Implementierung nach Lempel & Ziv (1976).
        Zählt die Anzahl distinkter Subsequenzen.
        """
        n = len(sequence)
        if n == 0:
            return 0.0

        s = ''.join(str(int(b)) for b in sequence)
        complexity = 1
        i = 0
        k = 1
        kmax = 1
        length = len(s)

        while i + k <= length:
            # Check if s[i:i+k] appears as substring in s[0:i+k-1]
            substring = s[i:i + k]
            search_space = s[0:i + k - 1] if i + k - 1 > 0 else ''
            
            if substring in search_space:
                k += 1
                if k > kmax:
                    kmax = k
            else:
                complexity += 1
                i += k if k > 1 else 1
                k = 1
                kmax = 1

        return float(complexity)

    # ===== FULL BENCHMARK =====

    def run_all(self, n_repeats: int = 5) -> Dict[str, Any]:
        """
        Vollständiger PCI-Benchmark mit Wiederholungen.
        
        Args:
            n_repeats: Anzahl Wiederholungen (verschiedene Stimulus-Positionen)
            
        Returns:
            Vollständiges Ergebnis-Dict
        """
        t0 = time.time()
        print("  📊 B2: PCI (Perturbational Complexity Index) Benchmark")
        print(f"     Network: {self.nn.n_neurons:,} neurons")
        print(f"     {n_repeats} Stimulus-Positionen\n")

        results = []
        for i in range(n_repeats):
            r = self.compute_pci()
            results.append(r)
            status = '✅' if r['above_threshold'] else '❌'
            print(f"     Run {i+1}: PCI = {r['pci']:.4f} {status} (>0.31)")

        pci_values = [r['pci'] for r in results]
        mean_pci = float(np.mean(pci_values))
        std_pci = float(np.std(pci_values))
        
        elapsed = time.time() - t0

        result = {
            'benchmark': 'B2_pci',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mean_pci': round(mean_pci, 4),
            'std_pci': round(std_pci, 4),
            'threshold': 0.31,
            'above_threshold': mean_pci > 0.31,
            'n_above': sum(1 for v in pci_values if v > 0.31),
            'n_total': len(pci_values),
            'values': [round(v, 4) for v in pci_values],
            'best_heatmap': results[int(np.argmax(pci_values))].get('heatmap'),
            'total_time_seconds': round(elapsed, 2)
        }

        status = '✅ ABOVE' if mean_pci > 0.31 else '❌ BELOW'
        print(f"\n     📊 Mean PCI: {mean_pci:.4f} ± {std_pci:.4f}")
        print(f"        Threshold: 0.31 → {status}")
        print(f"        {result['n_above']}/{result['n_total']} runs above threshold")
        print(f"     ⏱️  Total: {elapsed:.1f}s")

        return result
