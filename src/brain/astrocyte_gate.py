"""
MH-FLOCKE — Astrocyte Gate v0.4.1
========================================
Calcium-based gating for synaptogenesis.
"""

import numpy as np


class AstrocyteGate:
    """Astrocyte network for gating synaptogenesis."""
    
    def __init__(self, n_neurons: int, cluster_size: int = 100,
                 calcium_threshold: float = 0.7, tau_calcium: float = 2000.0):
        """
        Args:
            n_neurons: Total neurons in the network
            cluster_size: Neurons per astrocyte domain
            calcium_threshold: Calcium level needed for synapse formation
            tau_calcium: Calcium decay time constant (ms)
        """
        self.n_neurons = n_neurons
        self.cluster_size = cluster_size
        self.n_astrocytes = (n_neurons + cluster_size - 1) // cluster_size
        self.calcium_threshold = calcium_threshold
        self.tau_calcium = tau_calcium
        
        # Calcium levels per astrocyte (0.0 - 1.0+)
        self.calcium = np.zeros(self.n_astrocytes, dtype=np.float32)
        
        # Statistics
        self.update_count = 0
        self.above_threshold_count = 0
    
    def neuron_to_cluster(self, neuron_idx: int) -> int:
        """Map neuron index to its astrocyte cluster."""
        return neuron_idx // self.cluster_size
    
    def update(self, spike_mask: np.ndarray, dt: float = 1.0):
        """Update calcium levels based on spike activity.
        
        Args:
            spike_mask: Boolean array [n_neurons] — which neurons spiked
            dt: Timestep in ms
        """
        # Accumulate spikes per cluster
        for c in range(self.n_astrocytes):
            start = c * self.cluster_size
            end = min(start + self.cluster_size, self.n_neurons)
            cluster_spikes = spike_mask[start:end]
            spike_fraction = np.sum(cluster_spikes) / (end - start)
            
            # Calcium influx from local activity
            self.calcium[c] += spike_fraction * 0.1
        
        # Calcium decay
        decay = dt / self.tau_calcium
        self.calcium *= (1.0 - decay)
        
        # Clamp
        np.clip(self.calcium, 0.0, 2.0, out=self.calcium)
        
        self.update_count += 1
        self.above_threshold_count = int(np.sum(self.calcium > self.calcium_threshold))
    
    def can_form_synapse(self, pre_neuron: int, post_neuron: int) -> bool:
        """Check if synaptogenesis is allowed between two neurons.
        
        Both the pre and post astrocyte clusters must have calcium
        above threshold for a new synapse to form.
        
        Args:
            pre_neuron: Presynaptic neuron index
            post_neuron: Postsynaptic neuron index
            
        Returns:
            True if new synapse formation is permitted
        """
        pre_cluster = self.neuron_to_cluster(pre_neuron)
        post_cluster = self.neuron_to_cluster(post_neuron)
        
        return (self.calcium[pre_cluster] > self.calcium_threshold and
                self.calcium[post_cluster] > self.calcium_threshold)
    
    def get_active_clusters(self) -> np.ndarray:
        """Return indices of clusters above calcium threshold."""
        return np.where(self.calcium > self.calcium_threshold)[0]
    
    def get_stats(self) -> dict:
        """Return statistics for dashboard/monitoring."""
        return {
            'n_astrocytes': self.n_astrocytes,
            'cluster_size': self.cluster_size,
            'calcium_threshold': self.calcium_threshold,
            'above_threshold': self.above_threshold_count,
            'mean_calcium': float(np.mean(self.calcium)),
            'max_calcium': float(np.max(self.calcium)),
            'update_count': self.update_count
        }
