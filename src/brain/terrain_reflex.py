"""
MH-FLOCKE — Terrain Reflex v0.4.1
========================================
Adaptive reflexes for terrain variation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class TerrainReflexConfig:
    """Configuration for terrain-adaptive reflexes."""
    # Slope compensation gains
    pitch_gain: float = 0.6         # how strongly pitch affects leg adjustment
    roll_gain: float = 0.4          # how strongly roll affects lateral adjustment
    
    # Contact timing
    expected_stance_phase: float = 0.5  # fraction of gait cycle in stance (50%)
    contact_timing_gain: float = 0.3    # correction for early/late contact
    
    # Force distribution
    force_balance_gain: float = 0.2     # correction for force asymmetry
    
    # Smoothing
    ema_alpha: float = 0.1              # exponential moving average for pitch/roll
    
    # Limits
    max_correction: float = 0.5         # max reflex correction per joint
    
    # Ramp-in (don't interfere with initial motor babbling)
    warmup_steps: int = 2000            # reflexes reach full strength after N steps
    
    # Enable/disable
    enabled: bool = True


class FootContactSensor:
    """
    Reads per-foot ground contact from MuJoCo data.
    
    Biology: Cutaneous mechanoreceptors in the foot pad detect:
    - Contact onset (Meissner corpuscles — fast adapting)
    - Contact force (Merkel cells — slow adapting)
    - Contact timing relative to gait phase
    
    The Go2 has 4 foot geoms: FL, FR, RL, RR.
    We detect contact by checking MuJoCo's contact array for
    collisions between foot geoms and the ground.
    """
    
    # Go2 foot geom names
    FOOT_NAMES = ['FL', 'FR', 'RL', 'RR']
    
    def __init__(self):
        self._foot_geom_ids = {}   # name → geom_id
        self._initialized = False
        
        # Per-foot state
        self.contacts = np.zeros(4, dtype=bool)       # is foot touching ground?
        self.forces = np.zeros(4, dtype=np.float64)    # normal contact force per foot
        self.prev_contacts = np.zeros(4, dtype=bool)   # previous step contacts
        self.contact_onset = np.full(4, -1, dtype=int) # step when contact started
        self.contact_offset = np.full(4, -1, dtype=int) # step when contact ended
        
    def initialize(self, model):
        """Find foot geom IDs in the MuJoCo model."""
        import mujoco
        for i, name in enumerate(self.FOOT_NAMES):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._foot_geom_ids[name] = gid
            else:
                print(f'  WARNING: Foot geom "{name}" not found in model')
        self._initialized = len(self._foot_geom_ids) == 4
        if self._initialized:
            print(f'  FootContactSensor: 4 feet initialized ({list(self._foot_geom_ids.values())})')
        else:
            print(f'  FootContactSensor: INCOMPLETE — found {len(self._foot_geom_ids)}/4 feet')
    
    def update(self, model, data, step: int):
        """Read contact data from MuJoCo. Call every simulation step."""
        import mujoco
        
        if not self._initialized:
            return
            
        self.prev_contacts[:] = self.contacts
        self.contacts[:] = False
        self.forces[:] = 0.0
        
        # Check all active contacts
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            
            # Check if either geom is a foot
            for fi, name in enumerate(self.FOOT_NAMES):
                foot_gid = self._foot_geom_ids.get(name, -1)
                if foot_gid < 0:
                    continue
                    
                if g1 == foot_gid or g2 == foot_gid:
                    self.contacts[fi] = True
                    # Get actual contact force
                    c_force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, c_force)
                    self.forces[fi] = max(self.forces[fi], abs(c_force[0]))
        
        # Track contact onset/offset timing
        for fi in range(4):
            if self.contacts[fi] and not self.prev_contacts[fi]:
                self.contact_onset[fi] = step  # foot just touched down
            elif not self.contacts[fi] and self.prev_contacts[fi]:
                self.contact_offset[fi] = step  # foot just lifted off
    
    def get_data(self) -> Dict:
        """Return foot contact data for sensor_data dict."""
        return {
            'foot_contacts': self.contacts.copy(),
            'foot_forces': self.forces.copy(),
            'foot_any_contact': bool(self.contacts.any()),
            'foot_contact_count': int(self.contacts.sum()),
            # Per-foot for FLOG logging
            'foot_FL': bool(self.contacts[0]),
            'foot_FR': bool(self.contacts[1]),
            'foot_RL': bool(self.contacts[2]),
            'foot_RR': bool(self.contacts[3]),
            'foot_force_FL': float(self.forces[0]),
            'foot_force_FR': float(self.forces[1]),
            'foot_force_RL': float(self.forces[2]),
            'foot_force_RR': float(self.forces[3]),
        }


class TerrainReflex:
    """
    Terrain-adaptive motor reflexes.
    
    Three reflex mechanisms, all biologically grounded:
    
    1. SLOPE COMPENSATION (vestibulospinal reflex):
       IMU detects pitch/roll → immediate leg length adjustment.
       Uphill: front legs shorter, rear legs push harder.
       Downhill: front legs brake, rear legs shorter.
       Biology: Vestibulospinal tract adjusts extensor tone based on head tilt.
    
    2. CONTACT FORCE BALANCING (load reflex):
       Asymmetric foot forces → weight shifting correction.
       If left feet bear more load, shift right.
       Biology: Golgi tendon organs detect force → autogenic inhibition.
    
    3. TERRAIN PITCH → CPG MODULATION:
       Steep uphill → higher amplitude (more force), lower frequency (slower steps).
       Steep downhill → lower amplitude (careful), lower frequency (braking).
       Returns freq_scale and amp_scale for CPG modulation.
    
    The corrections are ADDED to motor commands after CPG + cerebellum.
    """
    
    def __init__(self, config: TerrainReflexConfig = None, n_actuators: int = 12):
        self.config = config or TerrainReflexConfig()
        self.n_act = n_actuators
        self._step = 0
        
        # Smoothed sensor values
        self._pitch_ema = 0.0
        self._roll_ema = 0.0
        self._force_ema = np.zeros(4)
        
        # CPG modulation output
        self.freq_scale = 1.0
        self.amp_scale = 1.0
        
        # Stats for logging
        self.stats = {
            'terrain_reflex_mag': 0.0,
            'terrain_pitch_ema': 0.0,
            'terrain_roll_ema': 0.0,
            'terrain_freq_scale': 1.0,
            'terrain_amp_scale': 1.0,
        }
    
    def compute(self, sensor_data: Dict) -> np.ndarray:
        """
        Compute terrain reflex corrections.
        
        Args:
            sensor_data: dict with orientation_euler, foot_contacts, foot_forces
            
        Returns:
            corrections: np.ndarray[n_actuators] to ADD to motor commands
        """
        if not self.config.enabled:
            return np.zeros(self.n_act)
        
        self._step += 1
        
        # Ramp in
        ramp = min(1.0, self._step / max(1, self.config.warmup_steps))
        
        corrections = np.zeros(self.n_act)
        
        # Get sensor values
        euler = sensor_data.get('orientation_euler', np.zeros(3))
        pitch = float(euler[1])  # positive = nose up = uphill
        roll = float(euler[0])   # positive = right side down
        
        foot_forces = sensor_data.get('foot_forces', np.zeros(4))
        foot_contacts = sensor_data.get('foot_contacts', np.zeros(4, dtype=bool))
        
        # Smooth with EMA
        a = self.config.ema_alpha
        self._pitch_ema = self._pitch_ema * (1 - a) + pitch * a
        self._roll_ema = self._roll_ema * (1 - a) + roll * a
        self._force_ema = self._force_ema * (1 - a) + foot_forces * a
        
        p = self._pitch_ema
        r = self._roll_ema
        pg = self.config.pitch_gain
        rg = self.config.roll_gain
        
        # ── 1. SLOPE COMPENSATION ──
        # Go2 leg order: FL=0, FR=1, RL=2, RR=3
        # Each leg has 3 joints: [hip_yaw, hip_lift, knee_extend]
        # Indices: FL=[0,1,2], FR=[3,4,5], RL=[6,7,8], RR=[9,10,11]
        
        if abs(p) > 0.03:  # >1.7 degrees
            # PITCH compensation
            # Uphill (p > 0): front legs lift more, rear legs push harder
            # Downhill (p < 0): front legs brake (less extension), rear shorter
            
            # Front legs: hip lift
            corrections[1] += p * pg * 0.5   # FL hip: more lift uphill
            corrections[4] += p * pg * 0.5   # FR hip: more lift uphill
            
            # Front legs: knee extension
            corrections[2] -= p * pg * 0.3   # FL knee: less extend uphill (shorter step)
            corrections[5] -= p * pg * 0.3   # FR knee: less extend uphill
            
            # Rear legs: more push uphill
            corrections[7] -= p * pg * 0.3   # RL hip: push back harder uphill
            corrections[10] -= p * pg * 0.3  # RR hip: push back harder uphill
            
            # Rear legs: more extension uphill (longer stance)
            corrections[8] += p * pg * 0.4   # RL knee: more extend uphill
            corrections[11] += p * pg * 0.4  # RR knee: more extend uphill
        
        if abs(r) > 0.03:  # >1.7 degrees
            # ROLL compensation
            # Right tilt (r > 0): extend right legs, shorten left
            
            # Hip yaw for lateral balance
            corrections[0] += r * rg * 0.3   # FL yaw
            corrections[3] -= r * rg * 0.3   # FR yaw
            corrections[6] += r * rg * 0.3   # RL yaw
            corrections[9] -= r * rg * 0.3   # RR yaw
            
            # Hip lift: extend downhill side
            corrections[1] -= r * rg * 0.2   # FL hip
            corrections[4] += r * rg * 0.2   # FR hip
            corrections[7] -= r * rg * 0.2   # RL hip
            corrections[10] += r * rg * 0.2  # RR hip
        
        # ── 2. CONTACT FORCE BALANCING ──
        total_force = self._force_ema.sum()
        if total_force > 0.1 and foot_contacts.sum() >= 2:
            fg = self.config.force_balance_gain
            # Left/right asymmetry
            left_force = self._force_ema[0] + self._force_ema[2]   # FL + RL
            right_force = self._force_ema[1] + self._force_ema[3]  # FR + RR
            lr_imbalance = (right_force - left_force) / (total_force + 1e-6)
            
            # Front/rear asymmetry
            front_force = self._force_ema[0] + self._force_ema[1]  # FL + FR
            rear_force = self._force_ema[2] + self._force_ema[3]   # RL + RR
            fr_imbalance = (rear_force - front_force) / (total_force + 1e-6)
            
            # Correct: shift weight toward lighter side
            for leg in range(4):
                base = leg * 3
                is_left = leg in (0, 2)
                is_front = leg in (0, 1)
                
                # Lateral: extend lighter side legs
                side_corr = lr_imbalance * fg * (1.0 if is_left else -1.0)
                corrections[base + 1] += side_corr * 0.2  # hip lift
                
                # Fore-aft: extend lighter end legs
                fa_corr = fr_imbalance * fg * (1.0 if is_front else -1.0)
                corrections[base + 2] += fa_corr * 0.2  # knee extend
        
        # ── 3. CPG MODULATION ──
        terrain_slope = abs(self._pitch_ema)
        if terrain_slope > 0.05:
            # Steeper → more force, slower steps
            self.amp_scale = 1.0 + terrain_slope * 1.5   # up to 1.75x at 30°
            self.freq_scale = 1.0 - terrain_slope * 0.5   # down to 0.85x at 30°
            self.freq_scale = max(0.6, self.freq_scale)
            self.amp_scale = min(2.0, self.amp_scale)
        else:
            self.amp_scale = 1.0
            self.freq_scale = 1.0
        
        # Apply ramp and clip
        corrections *= ramp
        corrections = np.clip(corrections, -self.config.max_correction, self.config.max_correction)
        
        # Stats
        self.stats['terrain_reflex_mag'] = float(np.abs(corrections).mean())
        self.stats['terrain_pitch_ema'] = float(self._pitch_ema)
        self.stats['terrain_roll_ema'] = float(self._roll_ema)
        self.stats['terrain_freq_scale'] = float(self.freq_scale)
        self.stats['terrain_amp_scale'] = float(self.amp_scale)
        
        return corrections
    
    def get_stats(self) -> Dict:
        """Return stats for logging."""
        return dict(self.stats)


class RightingReflex:
    """
    Vestibular righting reflex — the oldest reflex in mammals.
    
    Biology:
      Newborn kittens can right themselves within hours. This is a
      vestibular reflex mediated by the brainstem, not a learned behavior.
      The vestibular system detects head orientation relative to gravity,
      and the righting reflex produces asymmetric limb extension to
      rotate the body back to upright.
      
      Ref: Magnus (1924) — Body posture (Koerperstellung)
      Ref: Pellis et al. (1991) — Ontogeny of the righting reflex
    
    When the Go2 is on its side (|roll| > 30 degrees):
    - Push with the legs on the ground side to flip back
    - Tuck the legs on the air side to reduce resistance
    - Use foot contact to determine which legs have ground contact
    
    Also handles prone/supine (|pitch| > 45 degrees):
    - Face down: push with front legs, tuck rear
    - Face up: more complex, needs belly-down first
    
    This reflex is ADDED to motor commands only when is_fallen=True.
    It provides a starting point for the SNN to refine.
    """
    
    def __init__(self, n_actuators: int = 12):
        self.n_act = n_actuators
        self._active = False
        self._magnitude = 0.0
    
    def compute(self, sensor_data: Dict, is_fallen: bool) -> np.ndarray:
        """
        Compute righting reflex corrections.
        Only active when is_fallen=True.
        
        Returns:
            corrections: np.ndarray[12] to ADD to motor commands
        """
        if not is_fallen:
            self._active = False
            self._magnitude = 0.0
            self._fallen_step_count = 0  # Reset sequence counter
            return np.zeros(self.n_act)
        
        euler = sensor_data.get('orientation_euler', np.zeros(3))
        roll = float(euler[0])    # positive = right side down
        pitch = float(euler[1])   # positive = nose up
        foot_contacts = sensor_data.get('foot_contacts', np.zeros(4, dtype=bool))
        foot_forces = sensor_data.get('foot_forces', np.zeros(4))
        
        corrections = np.zeros(self.n_act)
        
        # Go2 leg layout: FL=0, FR=1, RL=2, RR=3
        # Each leg: [hip_yaw, hip_lift, knee_extend] = indices [0,1,2], [3,4,5], [6,7,8], [9,10,11]
        
        # Multi-phase righting sequence:
        # Phase 1 (steps 0-80): TUCK — all legs pull in tight
        # Phase 2 (steps 80-160): ROLL — push with ground-side legs to get belly-down
        # Phase 3 (steps 160-250): CROUCH — all legs under body
        # Phase 4 (steps 250+): PUSH UP — extend all legs to stand
        #
        # Biology: This mirrors the mammalian righting sequence observed in
        # cats and dogs. The vestibular system triggers a stereotyped
        # motor program that unfolds in phases.
        
        if not hasattr(self, '_fallen_step_count'):
            self._fallen_step_count = 0
        self._fallen_step_count += 1
        phase_step = self._fallen_step_count
        
        self._active = True
        
        if phase_step < 80:
            # Phase 1: TUCK — pull all legs in, reduce moment of inertia
            for leg in range(4):
                corrections[leg * 3 + 0] = 0.0       # yaw neutral
                corrections[leg * 3 + 1] = 0.9       # hip: pull up tight
                corrections[leg * 3 + 2] = -0.9      # knee: flex tight
        
        elif phase_step < 160:
            # Phase 2: ROLL TO BELLY — push with ground-side legs
            if abs(roll) > 0.3:
                strength = min(1.0, abs(roll) / 1.0)
                if roll > 0:  # right side down
                    corrections[4] = -1.0 * strength    # FR hip: push hard
                    corrections[5] = 0.8 * strength     # FR knee: extend
                    corrections[10] = -1.0 * strength   # RR hip: push hard
                    corrections[11] = 0.8 * strength    # RR knee: extend
                    corrections[1] = 0.8                # FL hip: tuck
                    corrections[7] = 0.8                # RL hip: tuck
                else:  # left side down
                    corrections[1] = -1.0 * strength    # FL hip: push hard
                    corrections[2] = 0.8 * strength     # FL knee: extend
                    corrections[7] = -1.0 * strength    # RL hip: push hard
                    corrections[8] = 0.8 * strength     # RL knee: extend
                    corrections[4] = 0.8                # FR hip: tuck
                    corrections[10] = 0.8               # RR hip: tuck
            elif abs(pitch) > 0.5:
                # On back or face — roll to side first
                for leg in [0, 2]:  # left legs push
                    corrections[leg * 3 + 1] = -0.8
                    corrections[leg * 3 + 2] = 0.6
            else:
                # Already belly-down — skip to phase 3
                self._fallen_step_count = 160
        
        elif phase_step < 250:
            # Phase 3: CROUCH — get legs under body
            for leg in range(4):
                corrections[leg * 3 + 0] = 0.0       # yaw neutral
                corrections[leg * 3 + 1] = 0.3       # hip: partially down
                corrections[leg * 3 + 2] = -0.5      # knee: partially flexed
        
        else:
            # Phase 4: PUSH UP — extend all legs
            for leg in range(4):
                corrections[leg * 3 + 1] = -0.8      # hip: push down hard
                corrections[leg * 3 + 2] = 0.7       # knee: extend
            # If we've been pushing for too long (>100 steps), restart sequence
            if phase_step > 350:
                self._fallen_step_count = 0
        
        self._magnitude = float(np.abs(corrections).mean())
        return np.clip(corrections, -1.0, 1.0)
    
    def get_stats(self) -> Dict:
        return {
            'righting_active': self._active,
            'righting_magnitude': self._magnitude,
        }
