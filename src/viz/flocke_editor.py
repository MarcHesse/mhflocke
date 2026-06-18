"""
MH-FLOCKE Video Editor -- Datengetriebener Post-Production Editor
====================================================================
Liest CSV Training-Log + Brain-State, erkennt Schluesselereignisse,
und rendert ein fertig editiertes Video mit:
  - Automatische Untertitel bei Events
  - Timeline-Graphs (Fitness, Emotions, Spikes, Neuromod)
  - Kamera-Follow + Orbit bei Milestones
  - Visuelle Effekte (Vignette, Glow, Dream-Blur, Color-Shift)
  - Audio-Generierung (Ambient, Spike-Sounds, Heartbeat)
  - YouTube Chapter Export

Architektur:
  CSV (38 Felder) --+
                     +--> Timeline Engine --> Frame Compositor --> ffmpeg --> MP4
  Brain.pt --------+
  Config (dict) ---+

Usage:
    from src.viz.flocke_editor import FlockeEditor
    editor = FlockeEditor('checkpoints/mogli/training_log.csv')
    events = editor.detect_events()
    editor.print_chapters()
"""

__version__ = "0.1.0"
__logbook__ = 154

import csv
import os
import time
import math
import struct
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFilter


# =====================================================================
# COLOR PALETTE (consistent with brain_overlay.py)
# =====================================================================
DEEP_SPACE = (13, 17, 23)
CYAN_ACCENT = (34, 211, 238)
FLOCKE_GOLD = (251, 191, 36)
TEXT_WHITE = (241, 245, 249)
TEXT_DIM = (100, 116, 139)
VALENCE_POS = (34, 197, 94)
VALENCE_NEG = (239, 68, 68)
DA_ORANGE = (255, 107, 53)
SEROTONIN_CYAN = (6, 182, 212)
NE_GOLD = (245, 158, 11)
ACH_GREEN = (16, 185, 129)
DREAM_VIOLET = (139, 92, 246)
FEAR_DARK = (30, 20, 20)


def _get_font(size=14, bold=False):
    """Safe font loading."""
    from PIL import ImageFont
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(name, size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


# =====================================================================
# DATA TYPES
# =====================================================================

@dataclass
class Event:
    """Ein erkanntes Ereignis in der Training-Timeline."""
    step: int
    time_sec: float
    event_type: str          # 'first_step', 'emotion_change', 'consciousness_up', etc.
    title: str               # Untertitel-Text
    subtitle: str = ''       # Zweite Zeile (optional)
    emoji: str = ''          # Prefix-Emoji
    effect: str = ''         # 'slowmo', 'glow', 'vignette', 'blur', 'flash', 'color_shift'
    color: Tuple = TEXT_WHITE
    duration_steps: int = 500  # Wie lange der Effekt sichtbar ist
    priority: int = 1        # Hoehere Prio = wichtiger (fuer Ueberlappungen)

    @property
    def chapter_time(self) -> str:
        """YouTube Chapter Timestamp (MM:SS)."""
        m, s = divmod(int(self.time_sec), 60)
        return f"{m:02d}:{s:02d}"


@dataclass
class TimelineRow:
    """Eine Zeile aus dem CSV Training-Log."""
    step: int = 0
    timestamp: str = ''
    time_sec: float = 0.0
    distance: float = 0.0
    reward: float = 0.0
    falls: int = 0
    valence: float = 0.0
    arousal: float = 0.0
    dominant_emotion: str = ''
    consciousness_level: int = 0
    pci: float = 0.0
    dominant_drive: str = ''
    curiosity: float = 0.0
    survival: float = 0.0
    competence: float = 0.0
    social: float = 0.0
    gwt_winner: str = ''
    n_episodes: int = 0
    memory_recall_count: int = 0
    prediction_error: float = 0.0
    concept_count: int = 0
    new_connections: int = 0
    dopamine: float = 0.0
    serotonin: float = 0.0
    norepinephrine: float = 0.0
    acetylcholine: float = 0.0
    total_spikes: int = 0
    spike_rate: float = 0.0
    n_synapses: int = 0
    astrocyte_active: int = 0
    body_error: float = 0.0
    proprioception_quality: float = 0.0
    consistency_score: float = 0.0
    hebbian_updates: int = 0
    hebbian_delta: float = 0.0
    dreamed: int = 0
    replay_buffer_size: int = 0
    avg_step_ms: float = 0.0
    skill: str = ''
    scene: str = ''       # MJCF-Dateiname (optional, fuer Szenen-Wechsel)
    # Cerebellar architecture data (v0.3.0)
    grc_sparseness: float = 0.0
    cf_magnitude: float = 0.0
    pf_pkc_weight: float = 0.0
    pf_pkc_weight_std: float = 0.0
    ltd_applied: float = 0.0
    ltp_applied: float = 0.0
    dcn_activity: float = 0.0
    correction_mag: float = 0.0
    snn_mix: float = 0.0
    golgi_rate: float = 0.0
    max_distance: float = 0.0
    best_episode: float = 0.0
    avg_episode: float = 0.0
    resets: int = 0
    upright: float = 0.0
    episode: int = 0


# =====================================================================
# 1. TIMELINE ENGINE -- CSV laden und parsen
# =====================================================================

class TimelineEngine:
    """Laedt CSV Training-Log und bietet Zugriff auf Zeitreihen."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.rows: List[TimelineRow] = []
        self._load()

    @staticmethod
    def _safe_int(val, default=0):
        try:
            return int(val) if val != '' and val is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(val, default=0.0):
        try:
            return float(val) if val != '' and val is not None else default
        except (ValueError, TypeError):
            return default

    def _load(self):
        """CSV parsen in TimelineRow-Objekte."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV nicht gefunden: {self.csv_path}")

        _si = self._safe_int
        _sf = self._safe_float

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = TimelineRow()
                row.step = _si(raw.get('step', 0))
                row.timestamp = raw.get('timestamp', '')
                row.time_sec = _sf(raw.get('time_sec', 0))
                row.distance = _sf(raw.get('distance', 0))
                row.reward = _sf(raw.get('reward', 0))
                row.falls = _si(raw.get('falls', 0))
                row.valence = _sf(raw.get('valence', 0))
                row.arousal = _sf(raw.get('arousal', 0))
                row.dominant_emotion = raw.get('dominant_emotion', '')
                row.consciousness_level = _si(raw.get('consciousness_level', 0))
                row.pci = _sf(raw.get('pci', 0))
                row.dominant_drive = raw.get('dominant_drive', '')
                row.curiosity = _sf(raw.get('curiosity', 0))
                row.survival = _sf(raw.get('survival', 0))
                row.competence = _sf(raw.get('competence', 0))
                row.social = _sf(raw.get('social', 0))
                row.gwt_winner = raw.get('gwt_winner', '')
                row.n_episodes = _si(raw.get('n_episodes', 0))
                row.memory_recall_count = _si(raw.get('memory_recall_count', 0))
                row.prediction_error = _sf(raw.get('prediction_error', 0))
                row.concept_count = _si(raw.get('concept_count', 0))
                row.new_connections = _si(raw.get('new_connections', 0))
                row.dopamine = _sf(raw.get('dopamine', 0))
                row.serotonin = _sf(raw.get('serotonin', 0))
                row.norepinephrine = _sf(raw.get('norepinephrine', 0))
                row.acetylcholine = _sf(raw.get('acetylcholine', 0))
                row.total_spikes = _si(raw.get('total_spikes', 0))
                row.spike_rate = _sf(raw.get('spike_rate', 0))
                row.n_synapses = _si(raw.get('n_synapses', 0))
                row.astrocyte_active = _si(raw.get('astrocyte_active', 0))
                row.body_error = _sf(raw.get('body_error', 0))
                row.proprioception_quality = _sf(raw.get('proprioception_quality', 0))
                row.consistency_score = _sf(raw.get('consistency_score', 0))
                row.hebbian_updates = _si(raw.get('hebbian_updates', 0))
                row.hebbian_delta = _sf(raw.get('hebbian_delta', 0.0))
                row.dreamed = _si(raw.get('dreamed', 0))
                row.replay_buffer_size = _si(raw.get('replay_buffer_size', 0))
                row.avg_step_ms = _sf(raw.get('avg_step_ms', 0))
                row.skill = raw.get('skill', '')
                row.scene = raw.get('scene', '')
                # Cerebellar architecture data (v0.3.0)
                row.grc_sparseness = _sf(raw.get('grc_sparseness', 0))
                row.cf_magnitude = _sf(raw.get('cf_magnitude', 0))
                row.pf_pkc_weight = _sf(raw.get('pf_pkc_weight', 0))
                row.pf_pkc_weight_std = _sf(raw.get('pf_pkc_weight_std', 0))
                row.ltd_applied = _sf(raw.get('ltd_applied', 0))
                row.ltp_applied = _sf(raw.get('ltp_applied', 0))
                row.dcn_activity = _sf(raw.get('dcn_activity', 0))
                row.correction_mag = _sf(raw.get('correction_mag', 0))
                row.snn_mix = _sf(raw.get('snn_mix', 0))
                row.golgi_rate = _sf(raw.get('golgi_rate', 0))
                row.max_distance = _sf(raw.get('max_distance', 0))
                row.best_episode = _sf(raw.get('best_episode', 0))
                row.avg_episode = _sf(raw.get('avg_episode', 0))
                row.resets = _si(raw.get('resets', 0))
                row.upright = _sf(raw.get('upright', 0))
                row.episode = _si(raw.get('episode', 0))
                self.rows.append(row)

        print(f"  Timeline: {len(self.rows)} Datenpunkte geladen aus {self.csv_path}")

    def get_series(self, field: str) -> List[float]:
        """Extrahiert eine Zeitreihe (z.B. 'distance', 'valence')."""
        return [getattr(r, field, 0) for r in self.rows]

    def get_steps(self) -> List[int]:
        return [r.step for r in self.rows]

    @property
    def total_steps(self) -> int:
        return self.rows[-1].step if self.rows else 0

    @property
    def total_time(self) -> float:
        return self.rows[-1].time_sec if self.rows else 0

    @property
    def max_distance(self) -> float:
        return max(r.distance for r in self.rows) if self.rows else 0

    def row_at_step(self, step: int) -> Optional[TimelineRow]:
        """Findet die naechste Zeile fuer einen gegebenen Step."""
        for r in self.rows:
            if r.step >= step:
                return r
        return self.rows[-1] if self.rows else None

    def interpolate_at(self, step: int, field: str) -> float:
        """Lineare Interpolation eines Feldes an beliebigem Step."""
        if not self.rows:
            return 0.0
        if step <= self.rows[0].step:
            return getattr(self.rows[0], field, 0)
        if step >= self.rows[-1].step:
            return getattr(self.rows[-1], field, 0)
        for i in range(len(self.rows) - 1):
            if self.rows[i].step <= step <= self.rows[i+1].step:
                t = (step - self.rows[i].step) / max(self.rows[i+1].step - self.rows[i].step, 1)
                v0 = getattr(self.rows[i], field, 0)
                v1 = getattr(self.rows[i+1], field, 0)
                return v0 + t * (v1 - v0)
        return 0.0


# =====================================================================
# 2. EVENT DETECTOR -- Schluesselereignisse finden
# =====================================================================

class EventDetector:
    """Erkennt Schluesselereignisse in der Training-Timeline."""

    def __init__(self, timeline: TimelineEngine):
        self.tl = timeline

    def detect_all(self) -> List[Event]:
        """Erkennt alle Events und sortiert nach Step."""
        events = []
        events.extend(self._detect_scene_changes())
        events.extend(self._detect_first_movement())
        events.extend(self._detect_distance_milestones())
        events.extend(self._detect_emotion_changes())
        events.extend(self._detect_consciousness_changes())
        events.extend(self._detect_first_concept())
        events.extend(self._detect_dream_phases())
        events.extend(self._detect_fall_crises())
        events.extend(self._detect_dopamine_spikes())
        events.extend(self._detect_drive_changes())
        events.extend(self._detect_spike_anomalies())
        events.sort(key=lambda e: e.step)
        # Deduplizieren: min 2000 Steps Abstand zwischen gleichen Event-Typen
        filtered = []
        for e in events:
            if not filtered or e.step - filtered[-1].step >= 2000:
                filtered.append(e)
            elif e.priority > filtered[-1].priority:
                filtered[-1] = e
        return filtered

    def _detect_scene_changes(self) -> List[Event]:
        """Erkennt Skill- und Szenen-Wechsel als Kapitelmarken.
        Zwei Quellen:
          - skill Feld: 'walk_grass' -> 'walk_ice' = neuer Trainingsabschnitt
          - scene Feld: 'flat.xml' -> 'hills.xml' = neue MuJoCo-Welt
        Jeder Wechsel = prominentes Kapitel (Priority 5)."""
        events = []
        last_skill = ''
        last_scene = ''
        for r in self.tl.rows:
            skill = r.skill.strip() if r.skill else ''
            scene = r.scene.strip() if r.scene else ''
            # Skill-Wechsel
            if skill and skill != last_skill:
                display = skill.replace('_', ' ').title()
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='scene_change',
                    title=f'Skill: {display}',
                    subtitle='Training gestartet',
                    emoji='\U0001F3AC', color=FLOCKE_GOLD, priority=5,
                    effect='glow', duration_steps=1500))
            # Szenen-Wechsel (MJCF)
            if scene and scene != last_scene and last_scene:
                scene_name = os.path.splitext(os.path.basename(scene))[0]
                display = scene_name.replace('_', ' ').title()
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='world_change',
                    title=f'Neue Welt: {display}',
                    subtitle=scene,
                    emoji='\U0001F30D', color=CYAN_ACCENT, priority=5,
                    effect='flash', duration_steps=1500))
            last_skill = skill
            last_scene = scene
        return events

    def _detect_first_movement(self) -> List[Event]:
        for r in self.tl.rows:
            if r.distance > 0.05:
                return [Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='first_movement',
                    title='First Movement',
                    emoji='\U0001F43E', color=FLOCKE_GOLD, priority=3)]
        return []

    def _detect_distance_milestones(self) -> List[Event]:
        events = []
        thresholds = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        reached = set()
        for r in self.tl.rows:
            for t in thresholds:
                if r.distance >= t and t not in reached:
                    reached.add(t)
                    events.append(Event(
                        step=r.step, time_sec=r.time_sec,
                        event_type='distance_milestone',
                        title=f'{t:.1f}m erreicht' if t < 1 else f'{t:.0f}m erreicht',
                        emoji='\U0001F3C3', color=VALENCE_POS, priority=2,
                        effect='glow'))
        return events

    def _detect_emotion_changes(self) -> List[Event]:
        events = []
        last_emotion = ''
        emotion_labels = {
            'fearful': ('Fear', '\U0001F628', VALENCE_NEG, 'vignette'),
            'content': ('Content', '\U0001F60C', VALENCE_POS, ''),
            'curious': ('Curiosity', '\U0001F914', CYAN_ACCENT, ''),
            'excited': ('Excitement', '\U0001F525', DA_ORANGE, 'flash'),
            'calm': ('Calm', '\U0001F54A', SEROTONIN_CYAN, ''),
            'frustrated': ('Frustration', '\U0001F620', VALENCE_NEG, 'vignette'),
            'neutral': ('', '', TEXT_DIM, ''),  # Skip neutral transitions
        }
        for r in self.tl.rows:
            emo = r.dominant_emotion
            if emo and emo != last_emotion and emo != 'neutral' and last_emotion:
                info = emotion_labels.get(emo, (emo.capitalize(), '\U0001F9E0', TEXT_WHITE, ''))
                if info[0]:
                    events.append(Event(
                        step=r.step, time_sec=r.time_sec,
                        event_type='emotion_change',
                        title=info[0],
                        emoji=info[1], color=info[2], priority=1,
                        effect=info[3]))
            last_emotion = emo
        return events

    def _detect_consciousness_changes(self) -> List[Event]:
        events = []
        last_level = 0
        for r in self.tl.rows:
            if r.consciousness_level > last_level:
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='consciousness_up',
                    title=f'Consciousness Level {r.consciousness_level}',
                    emoji='\U0001F9E0', color=CYAN_ACCENT, priority=3,
                    effect='glow', duration_steps=1000))
                last_level = r.consciousness_level
        return events

    def _detect_first_concept(self) -> List[Event]:
        for r in self.tl.rows:
            if r.concept_count > 0:
                return [Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='first_concept',
                    title='First Concept Formed',
                    emoji='\U0001F4A1', color=FLOCKE_GOLD, priority=3,
                    effect='flash')]
        return []

    def _detect_dream_phases(self) -> List[Event]:
        events = []
        for r in self.tl.rows:
            if r.dreamed:
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='dream',
                    title='Dream Phase',
                    subtitle='Consolidation',
                    emoji='\U0001F4A4', color=DREAM_VIOLET, priority=2,
                    effect='blur', duration_steps=500))
        return events

    def _detect_fall_crises(self) -> List[Event]:
        """Detects phases with above-average fall rates.
        Falls ist kumulativ -> Delta zwischen Rows berechnen.
        Nur einmal pro Krise triggern, nicht bei jedem Row."""
        events = []
        if len(self.tl.rows) < 5:
            return events
        # Sturz-Rate pro Intervall berechnen
        fall_rates = []
        for i in range(1, len(self.tl.rows)):
            delta = self.tl.rows[i].falls - self.tl.rows[i-1].falls
            step_delta = max(self.tl.rows[i].step - self.tl.rows[i-1].step, 1)
            fall_rates.append(delta / step_delta)  # Falls per step
        if not fall_rates:
            return events
        avg_rate = np.mean(fall_rates)
        std_rate = max(np.std(fall_rates), 0.001)
        # Nur bei deutlich erhoehter Rate triggern, max 1x pro 5000 Steps
        last_crisis_step = -10000
        for i, rate in enumerate(fall_rates):
            row = self.tl.rows[i + 1]
            if rate > avg_rate + 2 * std_rate and row.step - last_crisis_step > 5000:
                delta = self.tl.rows[i+1].falls - self.tl.rows[i].falls
                events.append(Event(
                    step=row.step, time_sec=row.time_sec,
                    event_type='fall_crisis',
                    title='Struggling',
                    subtitle=f'+{delta} Falls',
                    emoji='\U0001F4A5', color=VALENCE_NEG, priority=1,
                    effect='vignette'))
                last_crisis_step = row.step
        return events

    def _detect_dopamine_spikes(self) -> List[Event]:
        events = []
        for r in self.tl.rows:
            if r.dopamine > 0.7:
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='dopamine_spike',
                    title='Reward Spike!',
                    emoji='\u2B50', color=DA_ORANGE, priority=2,
                    effect='flash', duration_steps=300))
        return events

    def _detect_drive_changes(self) -> List[Event]:
        events = []
        last_drive = ''
        drive_labels = {
            'exploration': ('Exploration', '\U0001F50E'),
            'survival': ('Survival Mode', '\U0001F6E1'),
            'comfort': ('Seeking Comfort', '\U0001F3E0'),
            'social': ('Social', '\U0001F465'),
        }
        for r in self.tl.rows:
            d = r.dominant_drive
            if d and d != last_drive and last_drive:
                info = drive_labels.get(d, (d, '\U0001F3AF'))
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='drive_change',
                    title=info[0],
                    emoji=info[1], color=NE_GOLD, priority=1))
            last_drive = d
        return events

    def _detect_spike_anomalies(self) -> List[Event]:
        events = []
        spikes = [r.total_spikes for r in self.tl.rows]
        if len(spikes) < 5:
            return events
        mean_spikes = np.mean(spikes)
        std_spikes = max(np.std(spikes), 1)
        for i, r in enumerate(self.tl.rows):
            if r.total_spikes > mean_spikes + 2.5 * std_spikes:
                events.append(Event(
                    step=r.step, time_sec=r.time_sec,
                    event_type='spike_burst',
                    title='Neural Burst',
                    subtitle=f'{r.total_spikes} Spikes',
                    emoji='\u26A1', color=CYAN_ACCENT, priority=1))
        return events


# =====================================================================
# 3. SUBTITLE RENDERER
# =====================================================================

class SubtitleRenderer:
    """Rendert Untertitel mit Fade-In/Out auf einem Frame."""

    def __init__(self, width: int, height: int, font_size: int = 24):
        self.width = width
        self.height = height
        self.font = _get_font(font_size, bold=True)
        self.font_small = _get_font(font_size - 6)
        self.fade_steps = 15  # Frames fuer Fade

    def render(self, frame: Image.Image, event: Event,
               progress: float) -> Image.Image:
        """
        Rendert Untertitel auf Frame.
        progress: 0.0 (Anfang) bis 1.0 (Ende der Duration)
        """
        # Fade alpha
        if progress < 0.15:
            alpha = progress / 0.15
        elif progress > 0.85:
            alpha = (1.0 - progress) / 0.15
        else:
            alpha = 1.0
        alpha = max(0, min(1, alpha))
        if alpha < 0.01:
            return frame

        # Subtitle-Box
        text = f"{event.emoji} {event.title}" if event.emoji else event.title
        draw = ImageDraw.Draw(frame)
        tw = draw.textlength(text, font=self.font)

        # Position: unteres Drittel
        cx = self.width // 2
        cy = int(self.height * 0.82)

        # Semi-transparenter Background
        overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        pad = 16
        box = (cx - tw//2 - pad, cy - 18, cx + tw//2 + pad, cy + 22)
        if event.subtitle:
            box = (box[0], box[1], box[2], box[3] + 20)
        bg_alpha = int(160 * alpha)
        od.rounded_rectangle(box, radius=8, fill=(0, 0, 0, bg_alpha))

        # Text
        txt_alpha = int(255 * alpha)
        color = (*event.color[:3], txt_alpha)
        od.text((cx - tw//2, cy - 14), text, fill=color, font=self.font)
        if event.subtitle:
            sw = od.textlength(event.subtitle, font=self.font_small)
            od.text((cx - sw//2, cy + 10), event.subtitle,
                    fill=(200, 200, 200, txt_alpha), font=self.font_small)

        # Composite
        frame = frame.convert('RGBA')
        frame = Image.alpha_composite(frame, overlay)
        return frame.convert('RGB')


# =====================================================================
# 4. TIMELINE GRAPH RENDERER
# =====================================================================

class TimelineGraphRenderer:
    """Rendert Mini-Graphs die ueber das Video wachsen."""

    def __init__(self, width: int = 300, height: int = 80):
        self.width = width
        self.height = height
        self.font = _get_font(9)

    def render_fitness(self, data: List[float], current_idx: int,
                       label: str = "Distance") -> Image.Image:
        """Fitness-Kurve bis zum aktuellen Punkt."""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 120))
        draw = ImageDraw.Draw(img)

        # Label
        draw.text((4, 2), label, fill=(*TEXT_DIM, 200), font=self.font)

        if not data or current_idx < 1:
            return img

        visible = data[:current_idx + 1]
        max_val = max(max(visible), 0.01)

        # Wert anzeigen
        val_text = f"{visible[-1]:.2f}m"
        draw.text((self.width - 60, 2), val_text, fill=(*FLOCKE_GOLD, 220), font=self.font)

        # Kurve zeichnen
        margin_top = 16
        margin_bottom = 6
        plot_h = self.height - margin_top - margin_bottom
        n = len(visible)
        step_x = (self.width - 8) / max(len(data) - 1, 1)

        points = []
        for i, v in enumerate(visible):
            x = 4 + i * step_x
            y = margin_top + plot_h * (1.0 - v / max_val)
            points.append((x, y))

        if len(points) > 1:
            draw.line(points, fill=(*VALENCE_POS, 200), width=2)
            # Aktueller Punkt
            px, py = points[-1]
            draw.ellipse((px-3, py-3, px+3, py+3), fill=(*FLOCKE_GOLD, 255))

        return img

    def render_sparkline(self, data: List[float], current_idx: int,
                         color: Tuple, label: str = "") -> Image.Image:
        """Generische Sparkline."""
        h = 30
        img = Image.new('RGBA', (self.width, h), (0, 0, 0, 80))
        draw = ImageDraw.Draw(img)

        if label:
            draw.text((4, 1), label, fill=(*TEXT_DIM, 180), font=self.font)

        if not data or current_idx < 1:
            return img

        visible = data[:current_idx + 1]
        max_val = max(max(abs(v) for v in visible), 0.01)
        n = len(visible)
        step_x = (self.width - 8) / max(len(data) - 1, 1)

        points = []
        for i, v in enumerate(visible):
            x = 4 + i * step_x
            y = h//2 + (h//2 - 4) * (-v / max_val)
            points.append((x, y))

        if len(points) > 1:
            draw.line(points, fill=(*color, 180), width=1)

        return img

    def render_neuromod_bands(self, da: List[float], sht: List[float],
                               ne: List[float], ach: List[float],
                               current_idx: int) -> Image.Image:
        """4 Neuromod-Sparklines uebereinander."""
        band_h = 20
        total_h = band_h * 4 + 16
        img = Image.new('RGBA', (self.width, total_h), (0, 0, 0, 100))
        draw = ImageDraw.Draw(img)
        draw.text((4, 1), "Neuromodulators", fill=(*TEXT_DIM, 180), font=self.font)

        bands = [
            (da, DA_ORANGE, 'DA'),
            (sht, SEROTONIN_CYAN, '5HT'),
            (ne, NE_GOLD, 'NE'),
            (ach, ACH_GREEN, 'ACh'),
        ]
        y_off = 14
        for data, color, label in bands:
            spark = self.render_sparkline(data, current_idx, color, label)
            img.paste(spark, (0, y_off), spark)
            y_off += band_h

        return img


# =====================================================================
# 5. EFFECT RENDERER
# =====================================================================

class EffectRenderer:
    """Visuelle Effekte auf Frames anwenden."""

    @staticmethod
    def vignette(frame: Image.Image, strength: float = 0.5) -> Image.Image:
        """Dark corners (fear/stress)."""
        w, h = frame.size
        vig = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(vig)
        cx, cy = w // 2, h // 2
        max_r = math.sqrt(cx*cx + cy*cy)
        steps = 20
        for i in range(steps):
            r = max_r * (1.0 - i / steps)
            alpha = int(strength * 180 * (1.0 - i / steps))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=None, outline=(0, 0, 0, alpha), width=int(max_r/steps)+2)
        frame = frame.convert('RGBA')
        return Image.alpha_composite(frame, vig).convert('RGB')

    @staticmethod
    def glow(frame: Image.Image, color: Tuple = CYAN_ACCENT,
             strength: float = 0.3) -> Image.Image:
        """Heller Rand-Glow."""
        w, h = frame.size
        glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)
        alpha = int(strength * 100)
        for i in range(8):
            pad = i * 3
            draw.rectangle((pad, pad, w-pad-1, h-pad-1),
                           outline=(*color, max(0, alpha - i*12)), width=2)
        frame = frame.convert('RGBA')
        return Image.alpha_composite(frame, glow).convert('RGB')

    @staticmethod
    def dream_blur(frame: Image.Image, strength: float = 0.5) -> Image.Image:
        """Gaussischer Blur + violetter Tint fuer Dream-Phasen."""
        blurred = frame.filter(ImageFilter.GaussianBlur(radius=3 * strength))
        # Violetter Tint
        tint = Image.new('RGB', frame.size, DREAM_VIOLET)
        return Image.blend(blurred, tint, 0.1 * strength)

    @staticmethod
    def flash(frame: Image.Image, color: Tuple = FLOCKE_GOLD,
              strength: float = 0.3) -> Image.Image:
        """Kurzer Helligkeits-Flash."""
        bright = Image.new('RGB', frame.size, color)
        return Image.blend(frame, bright, strength)

    @staticmethod
    def color_shift(frame: Image.Image, valence: float) -> Image.Image:
        """Warm (positiv) / Kalt (negativ) basierend auf Valence."""
        if abs(valence) < 0.1:
            return frame
        if valence > 0:
            tint = Image.new('RGB', frame.size, (255, 200, 100))
        else:
            tint = Image.new('RGB', frame.size, (100, 150, 255))
        return Image.blend(frame, tint, min(abs(valence) * 0.08, 0.12))


# =====================================================================
# 6. AUDIO ENGINE
# =====================================================================

class AudioEngine:
    """Generiert Audio-Track aus Training-Daten."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def generate(self, timeline: TimelineEngine, events: List[Event],
                 duration_sec: float) -> np.ndarray:
        """Generiert kompletten Audio-Track als float32 Array."""
        n_samples = int(duration_sec * self.sr)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Layer 1: Ambient Pad (Consciousness-Level bestimmt Tonhoehe)
        audio += self._ambient_pad(timeline, n_samples) * 0.15

        # Layer 2: Heartbeat (Arousal bestimmt Tempo)
        audio += self._heartbeat(timeline, n_samples) * 0.1

        # Layer 3: Spike-Clicks
        audio += self._spike_texture(timeline, n_samples) * 0.08

        # Layer 4: Event-Sounds
        audio += self._event_sounds(events, duration_sec, n_samples) * 0.2

        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.85

        return audio

    def _ambient_pad(self, tl: TimelineEngine, n: int) -> np.ndarray:
        """Synth pad — pitch follows consciousness level. Fully vectorized."""
        audio = np.zeros(n, dtype=np.float32)
        if not tl.rows:
            return audio
        t = np.arange(n, dtype=np.float32) / self.sr
        total_time = tl.total_time or 1.0
        total_steps = tl.total_steps or 1
        # Sample consciousness level at ~100 Hz (not per-sample)
        check_interval = max(1, self.sr // 100)
        n_checks = (n + check_interval - 1) // check_interval
        cl_values = np.zeros(n_checks, dtype=np.float32)
        for ci in range(n_checks):
            sec = ci * check_interval / self.sr
            step = int(sec / total_time * total_steps)
            cl_values[ci] = tl.interpolate_at(step, 'consciousness_level')
        # Upsample to full resolution via repeat
        cl_full = np.repeat(cl_values, check_interval)[:n]
        freq = 80.0 + cl_full * 40.0  # L0=80Hz, L5=280Hz
        # Phase accumulation for smooth frequency changes
        phase = np.cumsum(2 * np.pi * freq / self.sr)
        audio = 0.5 * np.sin(phase) + 0.3 * np.sin(phase * 1.5)
        # Smooth with moving average
        kernel_size = min(1000, n // 4)
        if kernel_size > 1:
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            audio = np.convolve(audio, kernel, mode='same')
        return audio.astype(np.float32)

    def _heartbeat(self, tl: TimelineEngine, n: int) -> np.ndarray:
        """Heartbeat synced to arousal. Pre-compute beat positions, vectorized thumps."""
        audio = np.zeros(n, dtype=np.float32)
        if not tl.rows:
            return audio
        total_time = tl.total_time or 1.0
        total_steps = tl.total_steps or 1
        # Pre-compute beat positions
        beat_positions = []
        sec = 0.0
        while sec < total_time:
            step = int(sec / total_time * total_steps)
            arousal = tl.interpolate_at(step, 'arousal')
            bpm = 40 + arousal * 80
            beat_positions.append(sec)
            sec += 60.0 / max(bpm, 30)
        # Vectorized thump waveform
        dur = int(0.08 * self.sr)
        t_local = np.arange(dur, dtype=np.float32) / self.sr
        thump = np.sin(2 * np.pi * 50 * t_local) * np.exp(-t_local * 30)
        # Place thumps
        for bp in beat_positions:
            idx = int(bp * self.sr)
            end = min(idx + dur, n)
            if idx >= 0 and idx < n:
                audio[idx:end] += thump[:end - idx]
        return audio.astype(np.float32)

    def _spike_texture(self, tl: TimelineEngine, n: int) -> np.ndarray:
        """Crackling noise proportional to spike rate. Vectorized."""
        audio = np.zeros(n, dtype=np.float32)
        if not tl.rows:
            return audio
        total_time = tl.total_time or 1.0
        total_steps = tl.total_steps or 1
        rng = np.random.RandomState(42)
        check_interval = self.sr // 100  # 100 Hz
        n_checks = n // check_interval
        # Pre-compute click waveform
        click_dur = int(0.002 * self.sr)
        t_click = np.arange(click_dur, dtype=np.float32) / self.sr
        decay = np.exp(-t_click * 2000)
        # Batch: sample spike rates and random thresholds
        check_secs = np.arange(n_checks, dtype=np.float32) * check_interval / self.sr
        check_steps = (check_secs / total_time * total_steps).astype(int)
        thresholds = rng.random(n_checks).astype(np.float32)
        # Place clicks where threshold met
        for ci in range(n_checks):
            sr_val = tl.interpolate_at(int(check_steps[ci]), 'spike_rate')
            if thresholds[ci] < sr_val * 0.5:
                idx = ci * check_interval
                end = min(idx + click_dur, n)
                noise = rng.uniform(-1, 1, end - idx).astype(np.float32)
                audio[idx:end] += noise * decay[:end - idx]
        return audio.astype(np.float32)

    def _event_sounds(self, events: List[Event], total_sec: float,
                      n: int) -> np.ndarray:
        """Sound effects at events. Vectorized waveform generation."""
        audio = np.zeros(n, dtype=np.float32)
        for ev in events:
            sample_pos = int(ev.time_sec / max(total_sec, 1) * n)
            sample_pos = min(sample_pos, n - self.sr)
            if sample_pos < 0:
                continue
            if ev.event_type in ('consciousness_up', 'distance_milestone', 'first_concept'):
                # Rising chime
                dur = int(0.5 * self.sr)
                length = min(dur, n - sample_pos)
                t = np.arange(length, dtype=np.float32) / self.sr
                freq = 400 + 200 * t
                phase = np.cumsum(2 * np.pi * freq / self.sr)
                audio[sample_pos:sample_pos + length] += np.sin(phase) * np.exp(-t * 3)
            elif ev.event_type == 'dream':
                # Soft wash
                dur = int(1.0 * self.sr)
                length = min(dur, n - sample_pos)
                t = np.arange(length, dtype=np.float32) / self.sr
                audio[sample_pos:sample_pos + length] += (
                    np.sin(2 * np.pi * 200 * t) * np.exp(-t * 1.5) * 0.5)
            elif ev.event_type == 'fall_crisis':
                # Low rumble
                dur = int(0.3 * self.sr)
                length = min(dur, n - sample_pos)
                t = np.arange(length, dtype=np.float32) / self.sr
                audio[sample_pos:sample_pos + length] += (
                    np.sin(2 * np.pi * 40 * t) * np.exp(-t * 5))
        return audio.astype(np.float32)

    def save_wav(self, audio: np.ndarray, path: str):
        """Speichert als 16-bit WAV."""
        n = len(audio)
        audio_16 = (audio * 32767).astype(np.int16)
        with open(path, 'wb') as f:
            # WAV header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + n * 2))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<IHHIIHH', 16, 1, 1, self.sr, self.sr * 2, 2, 16))
            f.write(b'data')
            f.write(struct.pack('<I', n * 2))
            f.write(audio_16.tobytes())


# =====================================================================
# 7. CAMERA CONTROLLER
# =====================================================================

# ── Kamera-Shot Definitionen ─────────────────────────────────────

@dataclass
class CameraShot:
    """Ein Kamera-Shot mit Start/End-Parametern fuer Interpolation."""
    name: str
    start_step: int
    end_step: int
    # Start-Parameter
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -20.0
    # End-Parameter (fuer Interpolation, None = gleich wie Start)
    end_distance: float = None
    end_azimuth: float = None
    end_elevation: float = None
    # Optionen
    track_creature: bool = True    # Lookat folgt Kreatur
    lookat_offset: Tuple = (0, 0, 0.2)  # Offset ueber Kreatur-Mitte
    slowmo_factor: float = 1.0     # >1.0 = Zeitlupe
    easing: str = 'smooth'         # 'linear', 'smooth' (ease-in-out), 'ease_in'


class CameraController:
    """Cinematischer Kamera-Controller mit Shot-System.

    Shot-Typen:
      - WIDE:     Totale, weit weg, sieht alles
      - FOLLOW:   Standard-Verfolgung von hinten/seitlich
      - CLOSE_UP: Nah an Kreatur, Details sichtbar (Pfoten, Kopf)
      - ORBIT:    Langsame Rotation um Kreatur
      - TOP_DOWN: Draufsicht (zeigt Fortschritt/Pfad)
      - SIDE:     Seitenansicht (Ganganalyse)
      - DRAMATIC: Niedriger Winkel, leicht aufwarts (macht Kreatur 'gross')
      - ZOOM_IN:  Smooth Zoom von weit zu nah (fuer Milestones)
      - ZOOM_OUT: Smooth Zoom von nah zu weit (fuer Uebersicht)
      - DOLLY:    Kamera bewegt sich parallel zur Kreatur
    """

    # Vordefinierte Shot-Templates
    SHOTS = {
        'wide':     dict(distance=6.0, azimuth=135, elevation=-15),
        'follow':   dict(distance=3.0, azimuth=135, elevation=-20),
        'close_up': dict(distance=1.2, azimuth=150, elevation=-10,
                         lookat_offset=(0, 0, 0.15)),
        'orbit':    dict(distance=2.5, azimuth=0,   elevation=-15),
        'top_down': dict(distance=5.0, azimuth=90,  elevation=-85),
        'side':     dict(distance=3.5, azimuth=90,  elevation=-5),
        'dramatic': dict(distance=2.0, azimuth=160, elevation=-5,
                         lookat_offset=(0, 0, 0.1)),
        'dolly':    dict(distance=3.0, azimuth=90,  elevation=-15),
        'paw_cam':  dict(distance=0.8, azimuth=170, elevation=-3,
                         lookat_offset=(0.1, 0, -0.1)),
    }

    def __init__(self, default_distance: float = 3.0,
                 default_azimuth: float = 135.0,
                 default_elevation: float = -20.0):
        self.default_dist = default_distance
        self.default_az = default_azimuth
        self.default_el = default_elevation
        self._shots: List[CameraShot] = []
        self._auto_cut_interval = 3000  # Automatischer Schnitt alle N Steps

    def setup_from_events(self, events: List[Event], total_steps: int = 50000):
        """Generiert cinematische Shot-Liste basierend auf Events."""
        self._shots = []
        rng = np.random.RandomState(42)

        # Default-Rotation: Follow mit langsam wechselndem Azimuth
        # Dazwischen Event-getriebene Shots
        event_steps = {ev.step: ev for ev in events}
        step = 0

        while step < total_steps:
            # Check ob Event bei diesem Step
            ev = None
            for ev_step, ev_obj in event_steps.items():
                if ev_step >= step and ev_step < step + self._auto_cut_interval:
                    ev = ev_obj
                    break

            if ev and ev.priority >= 3:
                # ── Grosses Event: Cinematischer Shot ──
                if ev.event_type == 'first_movement':
                    # Zoom In auf ersten Schritt
                    self._shots.append(CameraShot(
                        name='zoom_in_first_step', start_step=ev.step - 200,
                        end_step=ev.step + 500,
                        distance=5.0, azimuth=135, elevation=-15,
                        end_distance=1.5, end_azimuth=155, end_elevation=-8,
                        slowmo_factor=2.0, easing='smooth'))
                    step = ev.step + 500
                elif ev.event_type == 'consciousness_up':
                    # Orbit
                    self._shots.append(CameraShot(
                        name='orbit_consciousness', start_step=ev.step,
                        end_step=ev.step + 800,
                        distance=2.5, azimuth=0, elevation=-15,
                        end_azimuth=180, easing='smooth'))
                    step = ev.step + 800
                elif ev.event_type in ('scene_change', 'world_change'):
                    # Wide shot fuer neue Szene
                    self._shots.append(CameraShot(
                        name='wide_new_scene', start_step=ev.step,
                        end_step=ev.step + 1000,
                        distance=7.0, azimuth=120, elevation=-25,
                        end_distance=3.0, end_azimuth=140, end_elevation=-20,
                        easing='smooth'))
                    step = ev.step + 1000
                elif ev.event_type == 'first_concept':
                    # Dramatic low angle
                    self._shots.append(CameraShot(
                        name='dramatic_concept', start_step=ev.step,
                        end_step=ev.step + 600,
                        distance=2.0, azimuth=160, elevation=-5,
                        easing='smooth'))
                    step = ev.step + 600
                else:
                    step = ev.step + 500

            elif ev and ev.priority == 2:
                # ── Mittleres Event: Close-Up oder Side ──
                shot_type = rng.choice(['close_up', 'side', 'dramatic'])
                tmpl = self.SHOTS[shot_type]
                self._shots.append(CameraShot(
                    name=f'{shot_type}_{ev.event_type}',
                    start_step=ev.step - 100, end_step=ev.step + 500,
                    **tmpl, easing='smooth'))
                step = ev.step + 500

            else:
                # ── Kein Event: Abwechselnde Standard-Shots ──
                dur = self._auto_cut_interval + rng.randint(-500, 500)
                shot_type = rng.choice(['follow', 'wide', 'side',
                                        'follow', 'follow', 'dolly'])
                tmpl = dict(self.SHOTS[shot_type])
                offset = tmpl.pop('lookat_offset', (0, 0, 0.2))
                # Leichte Azimuth-Variation
                tmpl['azimuth'] += rng.uniform(-20, 20)
                self._shots.append(CameraShot(
                    name=shot_type, start_step=step,
                    end_step=step + dur,
                    lookat_offset=offset, **tmpl))
                step += dur

    def get_camera_params(self, step: int, creature_pos: np.ndarray = None
                          ) -> Dict:
        """Gibt interpolierte Kamera-Parameter fuer einen Step zurueck."""
        # Finde aktiven Shot
        active_shot = None
        for shot in self._shots:
            if shot.start_step <= step <= shot.end_step:
                active_shot = shot
                break

        if active_shot is None:
            # Fallback: Default Follow
            return {
                'distance': self.default_dist,
                'azimuth': self.default_az,
                'elevation': self.default_el,
                'lookat': creature_pos,
                'lookat_offset': (0, 0, 0.2),
                'mode': 'follow',
                'slowmo': 1.0,
            }

        # Interpolation
        dur = max(active_shot.end_step - active_shot.start_step, 1)
        t = (step - active_shot.start_step) / dur  # 0..1
        t = max(0, min(1, t))

        # Easing
        if active_shot.easing == 'smooth':
            t = t * t * (3 - 2 * t)  # Smoothstep
        elif active_shot.easing == 'ease_in':
            t = t * t

        # Interpoliere Parameter
        end_dist = active_shot.end_distance or active_shot.distance
        end_az = active_shot.end_azimuth or active_shot.azimuth
        end_el = active_shot.end_elevation or active_shot.elevation

        dist = active_shot.distance + (end_dist - active_shot.distance) * t
        az = active_shot.azimuth + (end_az - active_shot.azimuth) * t
        el = active_shot.elevation + (end_el - active_shot.elevation) * t

        # Lookat
        lookat = creature_pos
        if active_shot.track_creature and creature_pos is not None:
            offset = active_shot.lookat_offset or (0, 0, 0.2)
            lookat = list(creature_pos)
            lookat[0] += offset[0]
            lookat[1] += offset[1]
            lookat[2] += offset[2]

        return {
            'distance': dist,
            'azimuth': az,
            'elevation': el,
            'lookat': lookat,
            'lookat_offset': active_shot.lookat_offset,
            'mode': active_shot.name,
            'slowmo': active_shot.slowmo_factor,
        }

    def apply_to_renderer(self, renderer, params: Dict, model, data):
        """Wendet Kamera-Parameter auf MuJoCo-Renderer an."""
        import mujoco
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = params['distance']
        camera.azimuth = params['azimuth']
        camera.elevation = params['elevation']
        if params.get('lookat') is not None:
            camera.lookat[:] = params['lookat'][:3]
        renderer.update_scene(data, camera)

    def get_shot_list(self) -> List[CameraShot]:
        """Gibt aktuelle Shot-Liste zurueck (fuer Debug/Preview)."""
        return self._shots


# =====================================================================
# 8. FLOCKE EDITOR -- Hauptklasse
# =====================================================================

class FlockeEditor:
    """
    Datengetriebener Video-Editor.

    Usage:
        editor = FlockeEditor('checkpoints/mogli/training_log.csv')
        events = editor.detect_events()
        editor.print_chapters()
        # Spaeter: editor.render('output/edited.mp4', brain_path=..., mesh=True)
    """

    def __init__(self, csv_path: str):
        self.timeline = TimelineEngine(csv_path)
        self.detector = EventDetector(self.timeline)
        self.subtitle_renderer = None  # Init bei render()
        self.graph_renderer = None
        self.effects = EffectRenderer()
        self.camera = CameraController()
        self.audio = AudioEngine()
        self.events: List[Event] = []

    def detect_events(self) -> List[Event]:
        """Erkennt alle Events in der Timeline."""
        self.events = self.detector.detect_all()
        print(f"\n  {len(self.events)} Events erkannt:")
        for ev in self.events:
            print(f"    Step {ev.step:>6d} [{ev.chapter_time}] "
                  f"{ev.emoji} {ev.title}"
                  f"{' -- ' + ev.subtitle if ev.subtitle else ''}"
                  f"{' [' + ev.effect + ']' if ev.effect else ''}")
        return self.events

    def print_chapters(self):
        """YouTube-Chapter-Format ausgeben."""
        print(f"\n  YouTube Chapters:")
        print(f"  00:00 Intro")
        for ev in self.events:
            if ev.priority >= 2:
                print(f"  {ev.chapter_time} {ev.emoji} {ev.title}")
        total_time = self.timeline.total_time
        m, s = divmod(int(total_time), 60)
        print(f"  {m:02d}:{s:02d} Ende")

    def export_chapters(self, path: str):
        """YouTube Chapters in Datei exportieren."""
        lines = ["00:00 Intro"]
        for ev in self.events:
            if ev.priority >= 2:
                lines.append(f"{ev.chapter_time} {ev.emoji} {ev.title}")
        total_time = self.timeline.total_time
        m, s = divmod(int(total_time), 60)
        lines.append(f"{m:02d}:{s:02d} Ende")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Chapters exportiert: {path}")

    def generate_audio(self, output_path: str = None) -> str:
        """Generiert Audio-Track als WAV."""
        if not self.events:
            self.detect_events()
        duration = self.timeline.total_time
        if duration < 1:
            print("  Audio: Timeline zu kurz")
            return ''
        print(f"\n  Audio generieren ({duration:.0f}s)...")
        audio = self.audio.generate(self.timeline, self.events, duration)
        path = output_path or 'output/training_audio.wav'
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.audio.save_wav(audio, path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  Audio: {path} ({size_kb:.0f} KB)")
        return path

    def summary(self):
        """Zusammenfassung der Timeline."""
        tl = self.timeline
        print(f"\n  Timeline Summary:")
        print(f"    Steps: {tl.total_steps}")
        print(f"    Zeit: {tl.total_time:.0f}s ({tl.total_time/60:.1f}min)")
        print(f"    Max Distance: {tl.max_distance:.3f}m")
        print(f"    Datenpunkte: {len(tl.rows)}")
        if tl.rows:
            last = tl.rows[-1]
            print(f"    Letzte Emotion: {last.dominant_emotion}")
            print(f"    Consciousness: Level {last.consciousness_level}")
            print(f"    Episoden: {last.n_episodes}")
            print(f"    Konzepte: {last.concept_count}")
            print(f"    Skill: {last.skill}")


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MH-Flocke Video Editor')
    parser.add_argument('--csv', type=str, default='checkpoints/mogli/training_log.csv')
    parser.add_argument('--detect', action='store_true', help='Events erkennen')
    parser.add_argument('--chapters', action='store_true', help='YouTube Chapters')
    parser.add_argument('--export-chapters', type=str, default=None)
    parser.add_argument('--audio', action='store_true', help='Audio generieren')
    parser.add_argument('--audio-output', type=str, default='output/training_audio.wav')
    parser.add_argument('--summary', action='store_true', help='Timeline Summary')
    args = parser.parse_args()

    editor = FlockeEditor(args.csv)

    if args.summary:
        editor.summary()

    if args.detect or args.chapters or args.export_chapters or args.audio:
        editor.detect_events()

    if args.chapters:
        editor.print_chapters()

    if args.export_chapters:
        editor.export_chapters(args.export_chapters)

    if args.audio:
        editor.generate_audio(args.audio_output)

    if not any([args.detect, args.chapters, args.export_chapters, args.audio, args.summary]):
        editor.summary()
        editor.detect_events()
        editor.print_chapters()


if __name__ == '__main__':
    main()
