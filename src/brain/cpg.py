"""
MH-FLOCKE — CPG Core v0.4.1
========================================
Phase-coupled oscillator network for rhythmic gait generation.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CPGConfig:
    """Konfiguration fuer den Central Pattern Generator."""
    base_frequency: float = 1.2      # Hz
    base_amplitude: float = 0.5      # Position-Control braucht groessere Werte
    
    # Amplituden pro Gelenktyp
    # Amplituden als Zielwinkel-Auslenkung (Radians vom Referenz-Winkel)
    shoulder_hip_amp: float = 0.4    # Schulter/Huefte Flex/Ext (~23 deg)
    elbow_stifle_amp: float = 0.25   # Ellbogen/Knie (~14 deg)
    carpus_hock_amp: float = 0.15    # Handgelenk/Sprunggelenk (~9 deg)
    abduction_amp: float = 0.08      # Seitliche Stabilisierung (~5 deg)
    
    # Offsets = 0, da Position-Actuatoren bei ctrl=0 die ref-Stellung halten
    elbow_offset: float = 0.0
    stifle_offset: float = 0.0
    carpus_offset: float = 0.0
    hock_offset: float = 0.0
    
    # Phasen-Offsets [FL, FR, RL, RR]
    phase_offsets: List[float] = field(default_factory=lambda: [0.0, 0.5, 0.75, 0.25])
    
    # Asymmetrie (evolvierbar)
    stance_power: float = 1.4       # Kraft beim Abstossen
    swing_power: float = 0.7        # Kraft beim Vorholen
    knee_phase_shift: float = 0.3   # pi-Multiplikator
    hock_phase_shift: float = 0.5   # pi-Multiplikator
    knee_swing_mult: float = 1.2
    knee_stance_mult: float = 0.4
    
    # SNN-Modulation
    freq_mod_range: float = 1.0
    amp_mod_range: float = 0.3
    
    dt: float = 0.002
    
    @staticmethod
    def load(path: str) -> 'CPGConfig':
        """Laedt evolved CPG-Parameter aus JSON."""
        with open(path) as f:
            data = json.load(f)
        cfg = CPGConfig()
        for key in ['base_frequency', 'base_amplitude', 'shoulder_hip_amp',
                     'elbow_stifle_amp', 'carpus_hock_amp', 'abduction_amp',
                     'stance_power', 'swing_power', 'knee_phase_shift',
                     'hock_phase_shift', 'knee_swing_mult', 'knee_stance_mult']:
            if key in data:
                setattr(cfg, key, data[key])
        # Aliases: CPGGenome uses 'frequency'/'amplitude', CPGConfig uses 'base_frequency'/'base_amplitude'
        if 'frequency' in data and 'base_frequency' not in data:
            cfg.base_frequency = data['frequency']
        if 'amplitude' in data and 'base_amplitude' not in data:
            cfg.base_amplitude = data['amplitude']
        if 'phase_offsets' in data:
            cfg.phase_offsets = data['phase_offsets']
        return cfg
    
    @staticmethod
    def auto_load(creature: str = 'mogli', skill: str = None,
                  search_dirs: List[str] = None) -> 'CPGConfig':
        """Sucht CPG-Config fuer eine Kreatur + Skill.
        
        Search order:
          1. checkpoints/{creature}/cpg_{skill}.json  (skill-specific)
          2. checkpoints/{creature}/cpg_config.json   (generic fallback)
          3. checkpoints/cpg_config.json
          4. CPGConfig() defaults
        """
        if search_dirs is None:
            search_dirs = [
                f'checkpoints/{creature}',
                'checkpoints',
                f'assets/meshes/{creature}',
                '.',
            ]
        
        # First: try skill-specific config
        if skill:
            for d in search_dirs:
                path = os.path.join(d, f'cpg_{skill}.json')
                if os.path.exists(path):
                    cfg = CPGConfig.load(path)
                    print(f"  CPG Config geladen: {path} (skill: {skill})")
                    return cfg
        
        # Fallback: generic cpg_config.json
        for d in search_dirs:
            path = os.path.join(d, 'cpg_config.json')
            if os.path.exists(path):
                cfg = CPGConfig.load(path)
                print(f"  CPG Config geladen: {path}")
                return cfg
        
        print(f"  CPG Config: defaults (no evolved config found)")
        return CPGConfig()

class CentralPatternGenerator:
    """
    CPG v2 fuer anatomisch korrekte Vierbein-Lokomotion.
    
    Pro Bein 4 Actuatoren:
      - Abduktion (seitlich, X-Achse): Gewichtsverlagerung
      - Flex/Ext (vor/zurueck, Y-Achse): Hauptbewegung
      - Ellbogen/Knie: Bein beugen/strecken
      - Carpus/Hock: Abstoss + Federung
    """
    
    def __init__(self, config: Optional[CPGConfig] = None, n_actuators: int = 21,
                 leg_map: dict = None):
        self.config = config or CPGConfig()
        self.n_actuators = n_actuators
        self.phases = np.array(self.config.phase_offsets, dtype=np.float64) * 2 * np.pi
        
        self.freq_mod = 0.0
        self.amp_mod = 0.0
        self.balance_corr = np.zeros(self.n_actuators)
        self.step_count = 0
        
        # Build leg mapping
        if leg_map is not None:
            self._legs = leg_map['legs']
            self._cosmetic = leg_map.get('cosmetic', [])
        elif n_actuators >= 17:
            self._legs = [
                {'abduction': 1,  'hip': 2,  'knee': 3,  'ankle': 4,  'is_rear': False},
                {'abduction': 5,  'hip': 6,  'knee': 7,  'ankle': 8,  'is_rear': False},
                {'abduction': 9,  'hip': 10, 'knee': 11, 'ankle': 12, 'is_rear': True},
                {'abduction': 13, 'hip': 14, 'knee': 15, 'ankle': 16, 'is_rear': True},
            ]
            self._cosmetic = [('neck', 0), ('tail', 17), ('ear_l', 18), ('ear_r', 19), ('jaw', 20)]
        else:
            self._legs = [
                {'abduction': None, 'hip': 0, 'knee': 1,  'ankle': 2,  'is_rear': False},
                {'abduction': None, 'hip': 3, 'knee': 4,  'ankle': 5,  'is_rear': False},
                {'abduction': None, 'hip': 6, 'knee': 7,  'ankle': 8,  'is_rear': True},
                {'abduction': None, 'hip': 9, 'knee': 10, 'ankle': 11, 'is_rear': True},
            ]
            self._cosmetic = []
    
    def step(self) -> np.ndarray:
        cfg = self.config
        
        freq = cfg.base_frequency + np.clip(self.freq_mod, -cfg.freq_mod_range, cfg.freq_mod_range)
        freq = max(0.3, freq)
        
        amp = cfg.base_amplitude + np.clip(self.amp_mod, -cfg.amp_mod_range, cfg.amp_mod_range)
        amp = np.clip(amp, 0.1, 1.0)
        
        d_phase = 2 * np.pi * freq * cfg.dt
        self.phases += d_phase
        
        controls = np.zeros(self.n_actuators)
        
        for leg_idx, leg in enumerate(self._legs):
            phase = self.phases[leg_idx] % (2 * np.pi)
            raw_sin = np.sin(phase)
            is_rear = leg.get('is_rear', False)
            
            # ── Abduction (if present) ──
            ab_i = leg.get('abduction')
            if ab_i is not None:
                if raw_sin > 0:
                    controls[ab_i] = cfg.abduction_amp * amp * 0.5
                else:
                    controls[ab_i] = -cfg.abduction_amp * amp * 0.3
            
            # ── Hip flex/ext (asymmetric stance/swing) ──
            hip_i = leg.get('hip')
            if hip_i is not None:
                if raw_sin > 0:
                    controls[hip_i] = -raw_sin * cfg.shoulder_hip_amp * amp * cfg.stance_power
                else:
                    controls[hip_i] = -raw_sin * cfg.shoulder_hip_amp * amp * cfg.swing_power
            
            # ── Knee ──
            knee_i = leg.get('knee')
            if knee_i is not None:
                knee_phase = phase + np.pi * cfg.knee_phase_shift
                knee_sin = np.sin(knee_phase)
                mid_amp = cfg.elbow_stifle_amp
                mid_offset = cfg.stifle_offset if is_rear else cfg.elbow_offset
                if knee_sin > 0:
                    controls[knee_i] = knee_sin * mid_amp * amp * cfg.knee_swing_mult + mid_offset
                else:
                    controls[knee_i] = knee_sin * mid_amp * amp * cfg.knee_stance_mult + mid_offset * 0.5
            
            # ── Ankle ──
            ankle_i = leg.get('ankle')
            if ankle_i is not None:
                low_phase = phase + np.pi * cfg.hock_phase_shift
                low_sin = np.sin(low_phase)
                low_amp = cfg.carpus_hock_amp
                low_offset = cfg.hock_offset if is_rear else cfg.carpus_offset
                if low_sin > 0:
                    controls[ankle_i] = low_sin * low_amp * amp + low_offset
                else:
                    controls[ankle_i] = low_sin * low_amp * amp * 0.6 + low_offset * 0.5
        
        # ── Cosmetic joints (neck, tail, etc.) ──
        for name, idx in self._cosmetic:
            if idx >= self.n_actuators:
                continue
            if 'neck' in name:
                fl_hip = self._legs[0].get('hip')
                fr_hip = self._legs[1].get('hip')
                if fl_hip is not None and fr_hip is not None:
                    controls[idx] = -(controls[fl_hip] + controls[fr_hip]) / 2 * 0.15
            elif 'tail' in name:
                controls[idx] = np.sin(self.phases[0] * 2) * 0.15
            # ears, jaw etc. stay at 0
        
        # SNN Balance-Korrekturen
        controls += self.balance_corr
        controls = np.clip(controls, -1.0, 1.0)
        
        self.step_count += 1
        return controls
    
    def set_modulation(self, freq_mod: float = 0.0, amp_mod: float = 0.0,
                       balance: Optional[np.ndarray] = None):
        self.freq_mod = freq_mod
        self.amp_mod = amp_mod
        if balance is not None:
            self.balance_corr = np.clip(balance, -0.3, 0.3)
    
    def set_gait(self, gait: str = 'walk'):
        gaits = {
            'walk':  [0.0, 0.5, 0.75, 0.25],
            'trot':  [0.0, 0.5, 0.5, 0.0],
            'bound': [0.0, 0.0, 0.5, 0.5],
            'stand': [0.0, 0.0, 0.0, 0.0],
        }
        if gait in gaits:
            current_base = self.phases[0]
            self.phases = np.array(gaits[gait]) * 2 * np.pi + current_base
    
    def reset(self):
        self.phases = np.array(self.config.phase_offsets) * 2 * np.pi
        self.step_count = 0
        self.freq_mod = 0.0
        self.amp_mod = 0.0
        self.balance_corr = np.zeros(self.n_actuators)
