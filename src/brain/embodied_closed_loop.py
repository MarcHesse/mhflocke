"""
MH-FLOCKE — Embodied Closed Loop v0.4.1
========================================
Targeted synaptogenesis from embodied experience.
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Deque
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EmbodiedExperience:
    """A single experience snapshot from the training loop."""
    step: int
    ball_dist: float        # Distance to target
    ball_heading: float     # Heading error to target (-1..+1)
    upright: float          # Uprightness (0..1)
    velocity: float         # Forward velocity (m/s)
    da_reward: float        # Dopamine reward received
    steering_offset: float  # VOR output applied
    cpg_weight: float       # Current CPG/SNN mix
    behavior: str           # Current behavior label
    is_fallen: bool
    prediction_error: float = 0.0  # World model PE (Issue #79)


@dataclass
class AdaptationReport:
    """Report from one adaptation cycle."""
    step: int
    ball_dist_trend: float    # Negative = approaching (good), positive = diverging
    actions_taken: List[str]
    vor_gain_before: float
    vor_gain_after: float
    abd_gain_before: float
    abd_gain_after: float
    dream_triggered: bool
    synaptogenesis_biased: bool
    neuromod_changes: Dict[str, float]


class EmbodiedClosedLoop:
    """
    Closes the loop between embodied experience and self-modification.
    
    Called every eval_interval steps during training. Evaluates whether
    the creature is making progress toward the target and adapts
    parameters autonomously.
    
    This replaces manual gain tuning. The system finds its own parameters.
    
    Usage in training loop:
        closed_loop = EmbodiedClosedLoop(snn=creature.snn, vor=vor)
        ...
        # Every step: record experience
        closed_loop.record(EmbodiedExperience(...))
        # Every eval_interval steps: adapt
        if step % closed_loop.eval_interval == 0:
            report = closed_loop.adapt()
    """
    
    def __init__(self, snn=None, vor=None, eval_interval: int = 2000,
                 history_size: int = 500):
        """
        Args:
            snn: SNNController instance (for synaptogenesis + neuromod)
            vor: VisualOrientingResponse instance (for gain adaptation)
            eval_interval: Steps between adaptation cycles
            history_size: Number of experiences to keep in memory
        """
        self.snn = snn
        self.vor = vor
        self.eval_interval = eval_interval
        
        # Experience buffer (ring buffer)
        self._history: Deque[EmbodiedExperience] = deque(maxlen=history_size)
        
        # Adaptation state
        self._adaptation_count = 0
        self._last_avg_ball_dist = None
        self._consecutive_improvements = 0
        self._consecutive_failures = 0
        
        # Parameter bounds (the system explores within these)
        self._vor_hip_gain_range = (0.10, 0.80)
        self._vor_abd_gain_range = (0.05, 0.50)
        self._vor_smoothing_range = (0.4, 0.9)
        self._abd_steering_range = (2.0, 10.0)  # CPG abd_amplitude multiplier
        
        # Learning rates for parameter adaptation
        self._param_lr = 0.1  # How much to adjust per cycle
        
        # Track what works
        self._best_ball_dist = float('inf')
        self._best_params = {}
        
        # Reports
        self.reports: List[AdaptationReport] = []
    
    def record(self, exp: EmbodiedExperience):
        """Record a single experience from the training loop."""
        self._history.append(exp)
        
        # Track best ever
        if exp.ball_dist >= 0 and exp.ball_dist < self._best_ball_dist:
            self._best_ball_dist = exp.ball_dist
    
    def adapt(self) -> Optional[AdaptationReport]:
        """
        Evaluate recent experiences and adapt parameters.
        
        This is the CLOSED LOOP. It:
        1. Measures ball_dist trend (are we getting closer?)
        2. If YES: lock parameters, consolidate (dream)
        3. If NO: adjust VOR gains, boost exploration (NE), grow synapses
        4. If STUCK: try bigger parameter changes
        
        Returns AdaptationReport or None if insufficient data.
        """
        if len(self._history) < 20:
            return None
        
        self._adaptation_count += 1
        actions = []
        
        # --- 1. Evaluate: is ball_dist decreasing? ---
        recent = list(self._history)
        
        # Split into first half and second half
        mid = len(recent) // 2
        first_half = [e for e in recent[:mid] if e.ball_dist >= 0]
        second_half = [e for e in recent[mid:] if e.ball_dist >= 0]
        
        if not first_half or not second_half:
            return None
        
        avg_first = np.mean([e.ball_dist for e in first_half])
        avg_second = np.mean([e.ball_dist for e in second_half])
        trend = avg_second - avg_first  # Negative = approaching (good)
        
        # Check stability
        n_fallen = sum(1 for e in recent if e.is_fallen)
        fall_rate = n_fallen / len(recent)
        avg_upright = np.mean([e.upright for e in recent])
        
        # Save current params
        vor_hip_before = self.vor.config.hip_gain if self.vor else 0
        abd_before = getattr(self, '_current_abd_mult', 5.0)
        
        # --- 2. Decide: adapt or consolidate? ---
        
        if trend < -0.5 and fall_rate < 0.05:
            # APPROACHING and STABLE — consolidate what works!
            self._consecutive_improvements += 1
            self._consecutive_failures = 0
            actions.append(f"APPROACHING (trend={trend:.2f}m, Δ{self._consecutive_improvements})")
            
            # Save best parameters
            if self.vor:
                self._best_params = {
                    'hip_gain': self.vor.config.hip_gain,
                    'abd_gain': self.vor.config.abd_gain,
                    'smoothing': self.vor.config.smoothing,
                    'max_output': self.vor.config.max_output,
                }
            
            # Boost ACh (consolidation mode)
            if self.snn:
                self.snn.set_neuromodulator('ach', min(0.9, self.snn.neuromod_levels['ach'] + 0.1))
                self.snn.set_neuromodulator('ne', max(0.1, self.snn.neuromod_levels['ne'] - 0.05))
                actions.append("ACh↑ NE↓ (consolidate)")
            
            # Dream: strengthen successful pathways
            if self._consecutive_improvements >= 3 and self.snn:
                self._dream_consolidate()
                actions.append("DREAM: consolidate approach patterns")
                self._consecutive_improvements = 0  # Reset counter
        
        elif trend > 0.5 or (trend > 0 and self._consecutive_failures > 2):
            # DIVERGING — adapt parameters
            self._consecutive_failures += 1
            self._consecutive_improvements = 0
            actions.append(f"DIVERGING (trend=+{trend:.2f}m, fails={self._consecutive_failures})")
            
            if fall_rate > 0.1:
                # Falling too much — reduce steering strength
                actions.append("FALLING: reduce steering")
                self._adjust_vor_gain(-self._param_lr * 1.5)
            elif avg_upright < 0.7:
                # Unstable — reduce steering slightly
                actions.append("UNSTABLE: reduce steering slightly")
                self._adjust_vor_gain(-self._param_lr * 0.5)
            else:
                # Stable but not reaching ball — adjust steering
                # Check if heading is changing (are we turning at all?)
                headings = [e.ball_heading for e in second_half if abs(e.ball_heading) > 0.01]
                if headings:
                    heading_var = np.std(headings)
                    avg_heading = np.mean(np.abs(headings))
                    
                    if heading_var < 0.05 and avg_heading > 0.3:
                        # Heading is stable but pointing away — increase steering
                        actions.append(f"NOT TURNING (h={avg_heading:.2f}): increase steering")
                        self._adjust_vor_gain(self._param_lr)
                    elif heading_var > 0.5:
                        # Oscillating — increase damping
                        actions.append(f"OSCILLATING (var={heading_var:.2f}): increase smoothing")
                        if self.vor:
                            self.vor.config.smoothing = min(0.9, self.vor.config.smoothing + 0.05)
                    else:
                        # Turning but not enough — slight increase
                        actions.append(f"TURNING TOO SLOW: slight increase")
                        self._adjust_vor_gain(self._param_lr * 0.5)
            
            # Boost NE (exploration mode)
            if self.snn:
                self.snn.set_neuromodulator('ne', min(0.8, self.snn.neuromod_levels['ne'] + 0.1))
                self.snn.set_neuromodulator('ach', max(0.3, self.snn.neuromod_levels['ach'] - 0.05))
                actions.append("NE↑ ACh↓ (explore)")
        
        else:
            # NEUTRAL — small oscillation, wait
            actions.append(f"NEUTRAL (trend={trend:+.2f}m)")
            self._consecutive_failures = max(0, self._consecutive_failures - 1)
        
        # --- 3. Targeted synaptogenesis bias ---
        # Tell the SNN which neuron pairs had high DA correlation
        synaptogenesis_biased = False
        if self.snn and self._consecutive_improvements >= 2:
            # During good approach episodes: bias astrocyte calcium
            # toward vision→output clusters
            self._bias_synaptogenesis_toward_vision()
            synaptogenesis_biased = True
            actions.append("SYNAPTOGENESIS: biased toward vision→motor")
        
        # --- 4. Recovery: if stuck for too long, try bigger changes ---
        if self._consecutive_failures >= 5:
            actions.append("STUCK: trying larger parameter change")
            # Big change: flip exploration direction
            if self.vor:
                current = self.vor.config.hip_gain
                # If we've been increasing, try decreasing (and vice versa)
                if current > np.mean(self._vor_hip_gain_range):
                    self._adjust_vor_gain(-self._param_lr * 3)
                else:
                    self._adjust_vor_gain(self._param_lr * 3)
            self._consecutive_failures = 0  # Reset
        
        # Build report
        report = AdaptationReport(
            step=self._history[-1].step if self._history else 0,
            ball_dist_trend=trend,
            actions_taken=actions,
            vor_gain_before=vor_hip_before,
            vor_gain_after=self.vor.config.hip_gain if self.vor else 0,
            abd_gain_before=abd_before,
            abd_gain_after=getattr(self, '_current_abd_mult', 5.0),
            dream_triggered='DREAM' in ' '.join(actions),
            synaptogenesis_biased=synaptogenesis_biased,
            neuromod_changes={
                'ne': self.snn.neuromod_levels['ne'] if self.snn else 0,
                'ach': self.snn.neuromod_levels['ach'] if self.snn else 0,
                '5ht': self.snn.neuromod_levels['5ht'] if self.snn else 0,
            },
        )
        self.reports.append(report)
        
        # Print summary
        print(f"\n  [CLOSED-LOOP #{self._adaptation_count}]")
        for a in actions:
            print(f"    → {a}")
        if self.vor:
            print(f"    VOR: hip={self.vor.config.hip_gain:.3f} smooth={self.vor.config.smoothing:.2f} max={self.vor.config.max_output:.2f}")
        
        self._last_avg_ball_dist = avg_second
        return report
    
    def _adjust_vor_gain(self, delta: float):
        """Adjust VOR hip and abd gains within bounds."""
        if not self.vor:
            return
        
        lo, hi = self._vor_hip_gain_range
        new_hip = np.clip(self.vor.config.hip_gain + delta, lo, hi)
        self.vor.config.hip_gain = float(new_hip)
        
        # Abd gain tracks hip gain proportionally
        abd_lo, abd_hi = self._vor_abd_gain_range
        new_abd = np.clip(self.vor.config.abd_gain + delta * 0.5, abd_lo, abd_hi)
        self.vor.config.abd_gain = float(new_abd)
        
        # Max output tracks hip gain
        self.vor.config.max_output = float(np.clip(new_hip * 1.3, 0.2, 0.8))
    
    def _dream_consolidate(self):
        """
        Consolidation during training (mini dream mode).
        
        Biology: Sharp-wave ripples in hippocampus during brief rest periods
        replay recent experiences and strengthen active synapses.
        
        We boost weights of synapses that had high eligibility during
        high-DA (approach reward) moments. This is DA-gated Hebbian:
        connections that fired together during success get stronger.
        """
        if not self.snn or self.snn._weight_values is None:
            return
        
        # Identify high-DA experiences
        high_da = [e for e in self._history if e.da_reward > 0.3]
        if not high_da:
            return
        
        # During high-DA moments, eligibility traces captured which
        # synapses were co-active. Boost those weights slightly.
        # This is equivalent to a "replay" — we don't re-run the SNN,
        # we just strengthen what was already active.
        elig = self.snn._eligibility
        if elig is not None and len(elig) > 0:
            # Only boost positive eligibility (LTP-like)
            boost = elig.clamp(min=0.0) * 0.02  # Small consolidation boost
            # Don't touch protected populations
            if self.snn.protected_populations:
                mask = self.snn._get_protected_synapse_mask()
                boost[mask] = 0.0
            self.snn._weight_values = self.snn._weight_values + boost
            # Clamp
            exc_mask = self.snn.neuron_types[self.snn._weight_indices[0]] > 0
            self.snn._weight_values = torch.where(
                exc_mask,
                self.snn._weight_values.clamp(min=0.0, max=1.0),
                self.snn._weight_values.clamp(min=-1.0, max=0.0)
            )
            self.snn._rebuild_sparse_weights()
    
    def _bias_synaptogenesis_toward_vision(self):
        """
        Bias astrocyte calcium toward vision input clusters.
        
        Biology: Activity-dependent neurotrophic factors (BDNF) are
        released by active neurons and promote synapse formation in
        their vicinity. We simulate this by boosting calcium in
        clusters that contain vision input neurons.
        
        This makes synaptogenesis_step() more likely to grow
        connections FROM vision neurons TO motor output neurons.
        """
        if not self.snn:
            return
        
        # Find vision input neuron clusters
        if 'input' in self.snn.populations:
            input_ids = self.snn.populations['input']
            # Vision neurons are the LAST 16 input neurons (2 channels × 8 population)
            n_input = len(input_ids)
            if n_input >= 16:
                vision_ids = input_ids[-16:]  # Last 16 = vision channels
                cs = self.snn._astro_cluster_size
                vision_clusters = set()
                for vid in vision_ids:
                    cluster = vid.item() // cs
                    if cluster < len(self.snn._astro_calcium):
                        vision_clusters.add(cluster)
                
                # Boost calcium in vision clusters
                for c in vision_clusters:
                    self.snn._astro_calcium[c] = min(2.0, self.snn._astro_calcium[c] + 0.3)
        
        # Also boost output neuron clusters
        if 'output' in self.snn.populations:
            output_ids = self.snn.populations['output']
            cs = self.snn._astro_cluster_size
            output_clusters = set()
            for oid in output_ids:
                cluster = oid.item() // cs
                if cluster < len(self.snn._astro_calcium):
                    output_clusters.add(cluster)
            
            for c in output_clusters:
                self.snn._astro_calcium[c] = min(2.0, self.snn._astro_calcium[c] + 0.2)
    
    def get_stats(self) -> Dict:
        """Stats for FLOG logging."""
        return {
            'cl_adaptations': self._adaptation_count,
            'cl_best_ball_dist': self._best_ball_dist,
            'cl_consec_improve': self._consecutive_improvements,
            'cl_consec_fail': self._consecutive_failures,
            'cl_vor_hip_gain': self.vor.config.hip_gain if self.vor else 0,
        }
