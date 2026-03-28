"""
MH-FLOCKE — Modular Skills v0.4.1
========================================
Skill registry with EWC protection for learned behaviors.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import copy


@dataclass
class Skill:
    """Ein gelerntes Fähigkeits-Modul."""
    
    name: str
    description: str = ''
    
    # Weight changes relative to base state
    weight_deltas: Optional[torch.Tensor] = None  # Sparse: nur geänderte Gewichte
    threshold_deltas: Optional[torch.Tensor] = None
    
    # Fisher Information Matrix (for EWC protection)
    # Higher values = more important weights = stronger protection
    importance: Optional[torch.Tensor] = None
    
    # Memory episodes belonging to this skill
    episode_indices: List[int] = field(default_factory=list)
    
    # Concept graph nodes created by this skill
    concept_ids: List[int] = field(default_factory=list)
    
    # Metadaten
    base_skills: List[str] = field(default_factory=list)  # Aufgebaut auf welchen Skills
    frozen: bool = False
    active: bool = True
    created_at: str = ''
    frozen_at: str = ''
    training_steps: int = 0
    best_metric: float = 0.0
    metric_name: str = 'distance'
    
    # Neuromodulator-Snapshot beim Freeze
    neuromod_snapshot: Optional[Dict[str, float]] = None


class SkillRegistry:
    """
    Verwaltet modulare Skills für ein CognitiveBrain.
    
    Kernfunktionen:
    - begin_skill(): Startet Aufzeichnung eines neuen Skills
    - freeze_skill(): Friert Skill ein, berechnet Importance
    - activate/deactivate(): Skills an/ausschalten
    - compose(): Kombiniert aktive Skills zu effektiven Gewichten
    - protect(): EWC-basierter Schutz eingefrorener Skills beim Training
    """
    
    def __init__(self, snn):
        self.snn = snn
        self.skills: Dict[str, Skill] = {}
        self._current_skill: Optional[str] = None
        self._baseline_weights: Optional[torch.Tensor] = None
        self._baseline_thresholds: Optional[torch.Tensor] = None
        self._training_step_count: int = 0
        
        # EWC lambda — protection strength of frozen skills
        self.ewc_lambda: float = 1000.0
        
    def begin_skill(self, name: str, description: str = '',
                    base_skills: Optional[List[str]] = None):
        """
        Startet Aufzeichnung eines neuen Skills.
        
        Speichert aktuellen Gewichtszustand als Baseline.
        Alle Gewichtsänderungen ab jetzt gehören zu diesem Skill.
        
        Args:
            name: Eindeutiger Skill-Name
            description: Was dieser Skill kann
            base_skills: Auf welchen eingefrorenen Skills aufgebaut wird
        """
        if name in self.skills and self.skills[name].frozen:
            raise ValueError(f"Skill '{name}' ist bereits eingefroren. "
                           f"Nutze einen anderen Namen.")
        
        # Baseline speichern
        self._baseline_weights = self.snn._weight_values.clone()
        self._baseline_thresholds = self.snn._thresholds.clone()
        self._training_step_count = 0
        
        # Create or update skill
        base = base_skills or []
        # Validate that base_skills exist and are frozen
        for bs in base:
            if bs not in self.skills:
                print(f"  ⚠ Base-Skill '{bs}' nicht gefunden, ignoriert")
            elif not self.skills[bs].frozen:
                print(f"  ⚠ Base-Skill '{bs}' nicht eingefroren, ignoriert")
        
        self.skills[name] = Skill(
            name=name,
            description=description,
            base_skills=[bs for bs in base if bs in self.skills],
            created_at=datetime.now().isoformat(),
            frozen=False,
            active=True,
        )
        
        self._current_skill = name
        print(f"  🎯 Skill '{name}' gestartet — Training beginnt")
        if base:
            print(f"     Basis: {', '.join(base)}")
    
    def freeze_skill(self, name: Optional[str] = None,
                     metric_value: float = 0.0,
                     metric_name: str = 'distance'):
        """
        Friert einen Skill ein.
        
        1. Berechnet Weight-Deltas (aktuelle Gewichte - Baseline)
        2. Berechnet Fisher Information (Importance) via Eligibility Traces
        3. Markiert Skill als frozen
        
        Args:
            name: Skill-Name (default: aktueller Skill)
            metric_value: Erreichte Leistung
            metric_name: Name der Metrik
        """
        name = name or self._current_skill
        if name is None:
            raise ValueError("Kein aktiver Skill zum Einfrieren")
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' nicht gefunden")
        
        skill = self.skills[name]
        if skill.frozen:
            print(f"  ⚠ Skill '{name}' ist bereits eingefroren")
            return
        
        # 1. Weight-Deltas berechnen
        if self._baseline_weights is not None:
            current_w = self.snn._weight_values
            baseline_w = self._baseline_weights
            # Synaptogenesis kann Synapsen hinzufuegen -> Groesse anpassen
            if current_w.shape[0] != baseline_w.shape[0]:
                new_size = current_w.shape[0]
                old_size = baseline_w.shape[0]
                if new_size > old_size:
                    # Neue Synapsen: Baseline mit 0 padden (Delta = aktueller Wert)
                    padded = torch.zeros(new_size, device=baseline_w.device)
                    padded[:old_size] = baseline_w
                    baseline_w = padded
                else:
                    # Synapsen entfernt (pruning): Baseline kuerzen
                    baseline_w = baseline_w[:new_size]
            skill.weight_deltas = (current_w - baseline_w).clone()

            current_t = self.snn._thresholds
            baseline_t = self._baseline_thresholds
            if current_t.shape[0] != baseline_t.shape[0]:
                new_size = current_t.shape[0]
                old_size = baseline_t.shape[0]
                if new_size > old_size:
                    padded = torch.zeros(new_size, device=baseline_t.device)
                    padded[:old_size] = baseline_t
                    baseline_t = padded
                else:
                    baseline_t = baseline_t[:new_size]
            skill.threshold_deltas = (current_t - baseline_t).clone()
        else:
            # No baseline -> all weights are the skill
            skill.weight_deltas = self.snn._weight_values.clone()
            skill.threshold_deltas = self.snn._thresholds.clone()
        
        # 2. Importance via Eligibility Traces (EWC-Approximation)
        # Gewichte die viel gelernt haben (hohe Eligibility) sind wichtiger
        elig = self.snn._eligibility
        # Eligibility muss gleiche Groesse wie weight_deltas haben
        if elig.shape[0] != skill.weight_deltas.shape[0]:
            target_size = skill.weight_deltas.shape[0]
            if elig.shape[0] < target_size:
                padded = torch.zeros(target_size, device=elig.device)
                padded[:elig.shape[0]] = elig.abs()
                skill.importance = padded
            else:
                skill.importance = elig[:target_size].abs().clone()
        else:
            skill.importance = elig.abs().clone()
        # Normalisieren
        imp_max = skill.importance.max()
        if imp_max > 0:
            skill.importance = skill.importance / imp_max
        
        # 3. Neuromodulator-Snapshot
        skill.neuromod_snapshot = dict(self.snn.neuromod_levels)
        
        # 4. Metadaten
        skill.frozen = True
        skill.frozen_at = datetime.now().isoformat()
        skill.training_steps = self._training_step_count
        skill.best_metric = metric_value
        skill.metric_name = metric_name
        
        self._current_skill = None
        
        n_changed = (skill.weight_deltas.abs() > 1e-6).sum().item()
        total = len(skill.weight_deltas)
        print(f"  ❄️  Skill '{name}' eingefroren")
        print(f"     {n_changed}/{total} Gewichte verändert "
              f"({100*n_changed/max(total,1):.1f}%)")
        print(f"     {metric_name}: {metric_value:.3f}")
        print(f"     Training: {skill.training_steps} Steps")
    
    def activate(self, names: List[str]):
        """Aktiviert Skills."""
        for name in names:
            if name in self.skills:
                self.skills[name].active = True
    
    def deactivate(self, names: List[str]):
        """Deaktiviert Skills (Gewichte werden nicht angewendet)."""
        for name in names:
            if name in self.skills:
                self.skills[name].active = False
    
    def delete_skill(self, name: str):
        """Löscht einen Skill komplett."""
        if name in self.skills:
            was_active = self.skills[name].active
            del self.skills[name]
            print(f"  🗑️  Skill '{name}' gelöscht")
            if was_active:
                print(f"     ⚠ War aktiv — compose() neu aufrufen!")
    
    def compose(self) -> torch.Tensor:
        """
        Kombiniert alle aktiven, eingefrorenen Skills zu effektiven Gewichten.
        
        Reihenfolge: Baseline + Skill1_deltas + Skill2_deltas + ...
        (additiv, mit Clipping)
        
        Returns:
            Effektive Gewichte als Tensor
        """
        if self._baseline_weights is None:
            return self.snn._weight_values
        
        # Start from baseline
        effective = self._baseline_weights.clone()
        
        # Aktive, eingefrorene Skills aufsummieren
        applied = []
        for name, skill in self.skills.items():
            if skill.active and skill.frozen and skill.weight_deltas is not None:
                effective += skill.weight_deltas
                applied.append(name)
        
        # Aktueller (nicht-eingefrorener) Skill
        if self._current_skill and self._current_skill in self.skills:
            current = self.skills[self._current_skill]
            if not current.frozen:
                # Aktuelle Deltas = live Gewichte - Baseline
                current_delta = self.snn._weight_values - self._baseline_weights
                effective += current_delta
                applied.append(f"{self._current_skill}(training)")
        
        # Clipping
        effective = effective.clamp(-3.0, 3.0)
        
        return effective
    
    def apply_composed(self):
        """Wendet komponierte Gewichte auf SNN an."""
        composed = self.compose()
        self.snn._weight_values = composed
        self.snn._rebuild_sparse_weights()
    
    def protect_frozen_skills(self):
        """
        EWC-basierter Schutz: Zieht Gewichte zurück zu eingefrorenen Skills.
        
        Sollte NACH jedem R-STDP/Hebbian Lernschritt aufgerufen werden.
        Stärke des Schutzes skaliert mit ewc_lambda und skill.importance.
        
        Mathematik:
          penalty = lambda * importance * (current_weight - frozen_weight)²
          gradient = 2 * lambda * importance * (current_weight - frozen_weight)
          
        In der Praxis: Gewichte die für eingefrorene Skills wichtig waren
        werden sanft zurückgezogen wenn Training sie zu weit verschiebt.
        """
        if not any(s.frozen and s.active for s in self.skills.values()):
            return  # Nichts zu schützen
        
        total_pull = torch.zeros_like(self.snn._weight_values)
        
        for skill in self.skills.values():
            if not (skill.frozen and skill.active and 
                    skill.weight_deltas is not None and
                    skill.importance is not None):
                continue
            
            # Ziel-Gewichte für diesen Skill
            target = self._baseline_weights + skill.weight_deltas
            
            # Abweichung vom Ziel
            deviation = self.snn._weight_values - target
            
            # EWC Penalty: wichtige Gewichte stärker zurückziehen
            pull = self.ewc_lambda * skill.importance * deviation
            total_pull += pull
        
        # Anwenden (sanft — nur 1% des Pulls pro Step)
        self.snn._weight_values -= 0.01 * total_pull
    
    def on_training_step(self):
        """
        Wird nach jedem Lernschritt aufgerufen.
        Zählt Steps und schützt eingefrorene Skills.
        """
        self._training_step_count += 1
        
        # EWC Protection
        if self._current_skill:
            self.protect_frozen_skills()
    
    def get_active_skills(self) -> List[str]:
        """Liste aktiver Skill-Namen."""
        return [n for n, s in self.skills.items() if s.active]
    
    def get_frozen_skills(self) -> List[str]:
        """Liste eingefrorener Skill-Namen."""
        return [n for n, s in self.skills.items() if s.frozen]
    
    def get_skill_info(self, name: str) -> Optional[Dict]:
        """Info über einen Skill."""
        if name not in self.skills:
            return None
        s = self.skills[name]
        return {
            'name': s.name,
            'description': s.description,
            'frozen': s.frozen,
            'active': s.active,
            'base_skills': s.base_skills,
            'training_steps': s.training_steps,
            'best_metric': s.best_metric,
            'metric_name': s.metric_name,
            'n_weight_changes': int((s.weight_deltas.abs() > 1e-6).sum()) 
                               if s.weight_deltas is not None else 0,
            'created_at': s.created_at,
            'frozen_at': s.frozen_at,
        }
    
    def summary(self) -> str:
        """Übersicht aller Skills."""
        lines = [f"  Skills ({len(self.skills)}):"]
        for name, s in self.skills.items():
            status = '❄️' if s.frozen else '🔥'
            active = '✅' if s.active else '⬜'
            metric = f"{s.best_metric:.3f}" if s.best_metric > 0 else '-'
            lines.append(f"    {active} {status} {name}: "
                        f"{s.training_steps} steps, "
                        f"{s.metric_name}={metric}")
            if s.base_skills:
                lines.append(f"       basis: {', '.join(s.base_skills)}")
        if self._current_skill:
            lines.append(f"  🎯 Training: {self._current_skill} "
                        f"(step {self._training_step_count})")
        return '\n'.join(lines)
    
    # ================================================================
    # PERSISTENCE
    # ================================================================
    
    def save_state(self) -> Dict:
        """Serialisiert alle Skills für brain_persistence."""
        skills_data = {}
        for name, s in self.skills.items():
            skills_data[name] = {
                'name': s.name,
                'description': s.description,
                'weight_deltas': s.weight_deltas.cpu() if s.weight_deltas is not None else None,
                'threshold_deltas': s.threshold_deltas.cpu() if s.threshold_deltas is not None else None,
                'importance': s.importance.cpu() if s.importance is not None else None,
                'episode_indices': s.episode_indices,
                'concept_ids': s.concept_ids,
                'base_skills': s.base_skills,
                'frozen': s.frozen,
                'active': s.active,
                'created_at': s.created_at,
                'frozen_at': s.frozen_at,
                'training_steps': s.training_steps,
                'best_metric': s.best_metric,
                'metric_name': s.metric_name,
                'neuromod_snapshot': s.neuromod_snapshot,
            }
        return {
            'skills': skills_data,
            'current_skill': self._current_skill,
            'baseline_weights': self._baseline_weights.cpu() if self._baseline_weights is not None else None,
            'baseline_thresholds': self._baseline_thresholds.cpu() if self._baseline_thresholds is not None else None,
            'training_step_count': self._training_step_count,
            'ewc_lambda': self.ewc_lambda,
        }
    
    def load_state(self, data: Dict):
        """Stellt alle Skills wieder her."""
        device = self.snn.device
        
        self._current_skill = data.get('current_skill')
        self._training_step_count = data.get('training_step_count', 0)
        self.ewc_lambda = data.get('ewc_lambda', 1000.0)
        
        if data.get('baseline_weights') is not None:
            self._baseline_weights = data['baseline_weights'].to(device)
        if data.get('baseline_thresholds') is not None:
            self._baseline_thresholds = data['baseline_thresholds'].to(device)
        
        self.skills = {}
        for name, sd in data.get('skills', {}).items():
            skill = Skill(
                name=sd['name'],
                description=sd.get('description', ''),
                base_skills=sd.get('base_skills', []),
                frozen=sd.get('frozen', False),
                active=sd.get('active', True),
                created_at=sd.get('created_at', ''),
                frozen_at=sd.get('frozen_at', ''),
                training_steps=sd.get('training_steps', 0),
                best_metric=sd.get('best_metric', 0.0),
                metric_name=sd.get('metric_name', 'distance'),
                episode_indices=sd.get('episode_indices', []),
                concept_ids=sd.get('concept_ids', []),
                neuromod_snapshot=sd.get('neuromod_snapshot'),
            )
            if sd.get('weight_deltas') is not None:
                skill.weight_deltas = sd['weight_deltas'].to(device)
            if sd.get('threshold_deltas') is not None:
                skill.threshold_deltas = sd['threshold_deltas'].to(device)
            if sd.get('importance') is not None:
                skill.importance = sd['importance'].to(device)
            
            self.skills[name] = skill
