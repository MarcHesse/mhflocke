"""
MH-FLOCKE — Behavior Planner v0.4.1
========================================
Drive-based behavior selection from situation assessment.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.behavior.behavior_knowledge import BehaviorKnowledge, BehaviorDef
from src.behavior.scene_instruction import SceneInstruction


class PlannerState(Enum):
    IDLE = 'idle'
    TRANSITIONING = 'transitioning'
    EXECUTING = 'executing'
    COMPLETING = 'completing'


@dataclass
class Situation:
    """Aktuelle Situation der Kreatur."""
    upright: float = 1.0          # 0..1
    speed: float = 0.0            # m/s
    height: float = 0.35          # Koerperhoehe
    is_fallen: bool = False
    steps_alive: int = 0
    prediction_error: float = 0.0
    energy_spent: float = 0.0
    # Sensory environment (Issue #75)
    smell_strength: float = 0.0   # 0..1, olfactory gradient intensity
    smell_direction: float = 0.0  # angle to strongest scent (-pi..pi)
    sound_intensity: float = 0.0  # 0..1, current sound event strength
    sound_direction: float = 0.0  # angle to sound source (-pi..pi)


class BehaviorPlanner:
    """
    Waehlt autonomes Verhalten basierend auf Drive-State und Situation.
    Smooth-Uebergaenge zwischen Behaviors, keine abrupten Wechsel.
    
    v0.3.5: Boredom + Novelty + Drive-Konkurrenz
    Ablation finding: Without boredom, the planner locks into
    walk<->alert on flat terrain and the creature stagnates.
    A real dog gets restless when nothing happens — it sniffs,
    marks, explores. Now: stagnation drives exploration up,
    boredom triggers behavior variety, drives compete.
    """

    def __init__(self, knowledge: BehaviorKnowledge):
        self.knowledge = knowledge
        self.state = PlannerState.IDLE
        self.current_behavior: Optional[str] = None
        self.next_behavior: Optional[str] = None
        self.behavior_step: int = 0          # Steps im aktuellen Behavior
        self.behavior_duration: int = 1000   # Geplante Dauer
        self.blend_progress: float = 0.0     # 0..1 Uebergang
        self._cooldowns: Dict[str, int] = {} # behavior_name -> remaining_cooldown
        self._step_count: int = 0
        self._last_behavior_end: int = 0
        self._scene_instruction: Optional[SceneInstruction] = None
        # Boredom + novelty tracking
        self._boredom: float = 0.0           # 0..1, rises with stagnation
        self._last_distance: float = 0.0     # for stagnation detection
        self._stagnation_steps: int = 0      # consecutive steps with no progress
        self._behavior_counts: Dict[str, int] = {}  # how often each was chosen
        self._novelty_bonus: Dict[str, float] = {}  # bonus for rarely-chosen behaviors

    def set_scene(self, instruction: SceneInstruction):
        """Setzt Szenen-Kontext fuer Behavior-Auswahl."""
        self._scene_instruction = instruction

    def update(self, drive_state: Dict, situation: Situation) -> str:
        """
        Hauptlogik — aufrufen pro Step.

        Args:
            drive_state: Dict mit survival, exploration, comfort, social, dominant
            situation: Aktuelle Situation

        Returns:
            Name des aktiven Behaviors
        """
        self._step_count += 1
        self._decay_cooldowns()
        self._update_boredom(situation)

        # Override drives with boredom: a bored dog explores regardless of
        # what the dominant drive says. This is biologically correct —
        # boredom is a meta-drive that overrides the survival default.
        drive_state = self._apply_boredom_to_drives(drive_state)

        # Gefallen -> sofort survival/recovery
        if situation.is_fallen:
            self._force_behavior('walk', duration=500)
            self._boredom = 0.0  # reset: falling is exciting enough
            return self.current_behavior or 'walk'

        if self.state == PlannerState.IDLE:
            self._choose_next(drive_state, situation)
            if self.next_behavior:
                self.state = PlannerState.TRANSITIONING
                self.blend_progress = 0.0

        elif self.state == PlannerState.TRANSITIONING:
            beh = self.knowledge.get_behavior(self.next_behavior)
            speed = beh.motor.blend_speed if beh else 0.05
            self.blend_progress += speed
            if self.blend_progress >= 1.0:
                self.blend_progress = 1.0
                self.current_behavior = self.next_behavior
                self.next_behavior = None
                self.behavior_step = 0
                self.state = PlannerState.EXECUTING
                # Track behavior diversity
                self._behavior_counts[self.current_behavior] = \
                    self._behavior_counts.get(self.current_behavior, 0) + 1

        elif self.state == PlannerState.EXECUTING:
            self.behavior_step += 1
            if self.behavior_step >= self.behavior_duration:
                self.state = PlannerState.COMPLETING

        elif self.state == PlannerState.COMPLETING:
            # Cooldown setzen
            if self.current_behavior:
                beh = self.knowledge.get_behavior(self.current_behavior)
                if beh:
                    self._cooldowns[self.current_behavior] = beh.cooldown
            self._last_behavior_end = self._step_count
            self.state = PlannerState.IDLE

        return self.current_behavior or 'walk'

    def _update_boredom(self, situation: Situation):
        """
        Boredom rises when nothing happens. A dog that stands still on
        a flat meadow gets restless — it wants to sniff, explore, move.
        
        Boredom triggers:
          - No forward progress (stagnation)
          - Same behavior repeated (monotony)
          - Low prediction error (no surprises)
        
        Boredom decays when:
          - Moving forward (new ground)
          - Behavior changes (variety)
          - High prediction error (something unexpected)
        """
        # Stagnation: distance not increasing
        current_dist = getattr(situation, '_distance', 0.0)
        if hasattr(situation, 'speed'):
            # Approximate: if speed near zero, we're stagnating
            if situation.speed < 0.01:
                self._stagnation_steps += 1
            else:
                self._stagnation_steps = max(0, self._stagnation_steps - 5)
        
        # Boredom grows with stagnation
        stagnation_factor = min(1.0, self._stagnation_steps / 3000.0)
        
        # Boredom grows with prediction error absence (nothing surprising)
        novelty_factor = max(0.0, 1.0 - situation.prediction_error * 5.0)
        
        # Boredom grows
        boredom_growth = 0.0002 * (0.5 + stagnation_factor + novelty_factor * 0.5)
        self._boredom = min(1.0, self._boredom + boredom_growth)
        
        # Boredom decays with movement and behavior variety
        if situation.speed > 0.05:
            self._boredom = max(0.0, self._boredom - 0.001)
        if situation.prediction_error > 0.3:
            self._boredom = max(0.0, self._boredom - 0.005)

    def _apply_boredom_to_drives(self, drive_state: Dict) -> Dict:
        """
        Boredom overrides drive balance. High boredom shifts the
        dominant drive from survival toward exploration/comfort.
        
        Biology: Hypothalamic boredom circuits compete with survival.
        A safe dog doesn't stay in permanent alert — it explores.
        """
        if self._boredom < 0.2:
            return drive_state  # not bored enough to matter
        
        modified = dict(drive_state)
        
        # Boost exploration proportional to boredom
        base_expl = modified.get('exploration', 0.3)
        modified['exploration'] = min(1.0, base_expl + self._boredom * 0.5)
        
        # Boost comfort (sniff, rest, mark are comfort/exploration)
        base_comfort = modified.get('comfort', 0.2)
        modified['comfort'] = min(0.8, base_comfort + self._boredom * 0.3)
        
        # Suppress survival dominance when bored and safe
        if modified.get('dominant') == 'survival' and self._boredom > 0.35:
            # Switch dominant to exploration if bored enough
            if modified['exploration'] > modified.get('survival', 0.4) * 0.8:
                modified['dominant'] = 'exploration'
        
        return modified

    def _choose_next(self, drive_state: Dict, situation: Situation):
        """Waehlt naechstes Behavior basierend auf Drive-Konkurrenz.
        
        v0.3.5: Instead of only checking the dominant drive, we now
        gather candidates from ALL drives above a threshold, weighted
        by drive strength. This means exploration behaviors compete
        with survival behaviors based on actual drive levels.
        """
        # Drive-Konkurrenz: collect candidates from all active drives,
        # not just the dominant one. Each drive contributes candidates
        # weighted by its strength.
        all_candidates = []
        for drive_name in ['survival', 'exploration', 'comfort', 'social', 'play']:
            drive_strength = drive_state.get(drive_name, 0.0)
            if drive_strength < 0.15:
                continue
            drive_behs = self.knowledge.get_behaviors_for_drive(
                drive_name, min_affinity=0.2)
            for beh in drive_behs:
                all_candidates.append((beh, drive_name, drive_strength))
        
        # Deduplicate by behavior name (keep highest drive strength)
        seen = {}
        for beh, drive_name, strength in all_candidates:
            if beh.name not in seen or strength > seen[beh.name][2]:
                seen[beh.name] = (beh, drive_name, strength)
        candidates = list(seen.values())

        if not candidates:
            self.next_behavior = 'walk'
            self.behavior_duration = 2000
            return

        best_score = -1.0
        best_beh = None

        # Compute novelty bonus: rarely-chosen behaviors get a boost
        total_choices = max(1, sum(self._behavior_counts.values()))

        for beh, drive_name, drive_strength in candidates:
            # Cooldown-Check
            if self._cooldowns.get(beh.name, 0) > 0:
                continue

            # Precondition-Check
            if beh.requires_upright and situation.upright < 0.5:
                continue
            if beh.requires_still and situation.speed > 0.05:
                continue
            if beh.min_steps_alive > situation.steps_alive:
                continue

            # Base score: drive_strength * affinity * priority
            affinity = beh.drive_affinity.get(drive_name, 0)
            score = drive_strength * affinity * beh.priority

            # Scene-Instruction Gewichtung
            if self._scene_instruction and self._scene_instruction.behavior_weights:
                scene_w = self._scene_instruction.behavior_weights.get(beh.name, 0.5)
                score *= scene_w

            # Situation-Modifiers
            if beh.name == 'rest' and situation.energy_spent > 200:
                score *= 1.5
            if beh.name == 'alert' and situation.prediction_error > 0.5:
                score *= 2.0
            # ── Sensory stimulus modifiers (Issue #75) ──
            # Biology: behavior switching is stimulus-driven, not random.
            # Tinbergen 1951: external stimuli release fixed action patterns.

            # Olfactory: smell triggers sniff (nearby) or walk/trot (distant)
            if situation.smell_strength > 0.05:
                if beh.name == 'sniff' and situation.smell_strength > 0.3:
                    score *= (1.0 + situation.smell_strength * 3.0)  # Strong sniff drive near scent
                elif beh.name in ('walk', 'trot') and situation.smell_strength > 0.05:
                    score *= (1.0 + situation.smell_strength * 2.0)  # Walk toward scent
                elif beh.name in ('rest', 'alert', 'look_around'):
                    score *= max(0.3, 1.0 - situation.smell_strength)  # Don't stop when scent is strong

            # Acoustic: sound triggers alert (brief) then look_around
            if situation.sound_intensity > 0.1:
                if beh.name == 'alert':
                    score *= (1.0 + situation.sound_intensity * 4.0)  # Strong alert on sound
                elif beh.name == 'look_around':
                    score *= (1.0 + situation.sound_intensity * 2.0)  # Orient toward sound
                elif beh.name in ('walk', 'trot'):
                    score *= max(0.5, 1.0 - situation.sound_intensity)  # Slow down briefly

            # Alert suppression during stable locomotion WITHOUT sound stimulus.
            # Biology: a walking animal doesn't freeze-alert without cause.
            if beh.name == 'alert' and situation.speed > 0.05 and situation.sound_intensity < 0.1:
                score *= 0.3  # Hard to trigger alert while moving without cause
            if beh.name == 'sniff' and situation.speed < 0.1 and situation.smell_strength < 0.1:
                score *= 1.3

            # Motor babbling: neonatal phase only.
            # Biology: fidgety movements peak at 2-4 months, then fade
            # as goal-directed movement emerges (Prechtl 1997).
            # For us: high score in first 5k steps, fades to 0 by 10k.
            if beh.name == 'motor_babbling':
                maturity = min(1.0, situation.steps_alive / 10000.0)
                if maturity < 0.5:
                    score *= 3.0   # Strong neonatal drive — overrides alert
                elif maturity < 0.7:
                    score *= max(0.1, 2.0 - maturity * 3.0)  # Fade
                else:
                    score = 0.0   # Completely gone after 7k steps

            # Novelty bonus: behaviors chosen less often get a boost
            # A dog that always walks will eventually try something else
            choice_count = self._behavior_counts.get(beh.name, 0)
            novelty = 1.0 - (choice_count / total_choices) if total_choices > 3 else 0.5
            score *= (0.7 + novelty * 0.6)  # range: 0.7..1.3

            # Boredom boost for non-locomotion behaviors
            # When bored, sniffing and marking become more attractive
            if self._boredom > 0.3 and beh.name in ('sniff', 'mark', 'look_around', 'trot'):
                score *= (1.0 + self._boredom)

            # Anti-alert when bored: alert is the opposite of what a bored dog does
            if self._boredom > 0.4 and beh.name == 'alert':
                score *= 0.2

            # Repetition penalty — weaker for locomotion during learning.
            # Biology: a puppy learning to walk doesn't stop every 30s
            # to sniff or alert. Locomotion has intrinsic momentum
            # (reticulospinal drive maintains gait once initiated).
            # The full novelty/variety behavior emerges AFTER motor maturity.
            if beh.name == self.current_behavior:
                if beh.name in ('walk', 'trot', 'motor_babbling'):
                    score *= 0.7  # Mild penalty — locomotion has momentum
                else:
                    score *= 0.2  # Strong penalty — non-locomotion should vary

            # Reduced randomness — stimuli provide natural variety now.
            # Biology: motor noise is in execution, not decision.
            score *= (0.95 + 0.1 * np.random.random())

            if score > best_score:
                best_score = score
                best_beh = beh

        if best_beh:
            self.next_behavior = best_beh.name
            self.behavior_duration = np.random.randint(
                best_beh.min_duration, best_beh.max_duration + 1)
            # Choosing a new behavior reduces boredom slightly
            if best_beh.name != self.current_behavior:
                self._boredom = max(0.0, self._boredom - 0.05)
        else:
            self.next_behavior = 'walk'
            self.behavior_duration = 2000

    def _force_behavior(self, name: str, duration: int = 500):
        """Erzwingt Behavior-Wechsel (z.B. bei Fall)."""
        self.current_behavior = name
        self.next_behavior = None
        self.behavior_step = 0
        self.behavior_duration = duration
        self.blend_progress = 1.0
        self.state = PlannerState.EXECUTING

    def _decay_cooldowns(self):
        """Reduziert alle Cooldowns um 1."""
        expired = []
        for name in self._cooldowns:
            self._cooldowns[name] -= 1
            if self._cooldowns[name] <= 0:
                expired.append(name)
        for name in expired:
            del self._cooldowns[name]

    def get_blend_factor(self) -> float:
        """Wie weit der Uebergang zum naechsten Behavior ist (0..1)."""
        if self.state == PlannerState.TRANSITIONING:
            return self.blend_progress
        return 1.0

    def get_state(self) -> Dict:
        """Fuer Dashboard/Logging."""
        return {
            'current_behavior': self.current_behavior or 'idle',
            'next_behavior': self.next_behavior or '',
            'planner_state': self.state.value,
            'behavior_step': self.behavior_step,
            'behavior_duration': self.behavior_duration,
            'blend_progress': round(self.blend_progress, 3),
            'behavior_progress': round(
                self.behavior_step / max(self.behavior_duration, 1), 3),
            'boredom': round(self._boredom, 3),
            'stagnation_steps': self._stagnation_steps,
            'behavior_diversity': len(self._behavior_counts),
        }
