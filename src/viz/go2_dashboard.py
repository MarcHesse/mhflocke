"""
MH-FLOCKE — Go2 Dashboard v0.4.3
========================================
PIL-based dashboard overlay for video rendering.
"""

import gc
import math
import numpy as np
from typing import Dict, Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════

CYAN = (0, 200, 220)
ICE = (160, 195, 235)
BLUE = (60, 130, 200)
GOLD = (255, 200, 60)
RED = (255, 60, 40)
VIOLET = (185, 110, 240)
GREEN = (50, 200, 100)
ORANGE = (255, 160, 40)
GREY = (150, 160, 180)
WHITE = (220, 225, 235)
PINK = (255, 100, 160)
LIME = (160, 255, 80)
BAR_BG = (25, 30, 42)
GLASS_BG = (6, 10, 18, 230)        # Darker + more opaque for YouTube compression survival
GLASS_BORDER = (0, 200, 220, 120)  # Brighter border for edge definition after recompression

# Behavior colors
BEH_COLORS = {
    'walk': GREEN, 'trot': CYAN, 'alert': GOLD, 'look_around': ORANGE,
    'rest': GREY, 'sniff': VIOLET, 'motor_babbling': RED,
}

# Emotion colors
EMO_COLORS = {
    # From embodied_emotions.py EMOTION_MAP
    'excited': GOLD, 'content': GREEN, 'fearful': RED, 'sad': BLUE, 'neutral': GREY,
    # Derived emotions (dashboard fallback when label is neutral)
    'calm': ICE, 'tense': ORANGE, 'focused': CYAN,
    # Legacy labels
    'curiosity': CYAN, 'satisfaction': GREEN, 'frustration': RED,
    'fear': VIOLET, 'surprise': GOLD, 'boredom': GREY,
}


# ═══════════════════════════════════════════════════════════
# FONT CACHE
# ═══════════════════════════════════════════════════════════

_FONTS = {}
_FONT_SCALE = 1.0  # Set by Go2DashboardOverlay.__init__ based on resolution
_MIN_FONT_PX = 14  # Minimum font size in pixels (survives YouTube compression)

def _f(size, bold=False):
    """Get cached font, scaled by global _FONT_SCALE.
    
    YouTube compression destroys thin fonts below ~14px. We enforce a
    minimum and prefer DejaVu Sans Mono (thicker strokes) over Consolas.
    """
    scaled = max(_MIN_FONT_PX, int(size * _FONT_SCALE))
    key = (scaled, bold)
    if key not in _FONTS:
        suffix = '-Bold' if bold else ''
        # DejaVu first — thicker strokes survive YouTube recompression better
        for path in [
            f'/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
            f'C:/Windows/Fonts/consola{"b" if bold else ""}.ttf',
        ]:
            try:
                _FONTS[key] = ImageFont.truetype(path, scaled)
                break
            except (OSError, IOError):
                continue
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


# ═══════════════════════════════════════════════════════════
# GLASS PANEL HELPER
# ═══════════════════════════════════════════════════════════

def _glass(w, h, radius=6):
    """Create transparent glass panel with border."""
    panel = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=radius,
                           fill=GLASS_BG, outline=GLASS_BORDER)
    return panel, draw


# ═══════════════════════════════════════════════════════════
# RIGHT COLUMN WIDGETS
# ═══════════════════════════════════════════════════════════

def _widget_cpg_mix(w, h, s):
    """CPG/SNN mix gauge with competence bar."""
    panel, d = _glass(w, h)
    cpg = s.get('cpg_weight', 0.9)
    snn = 1.0 - cpg
    comp = s.get('actor_competence', 0.0)
    snn_mix = s.get('snn_mix', 0.0)

    d.text((12, 6), 'CPG / SNN MIX', fill=(*CYAN, 240), font=_f(15, True))

    # Big numbers
    d.text((12, 26), f'{cpg:.0%}', fill=(*ORANGE, 255), font=_f(32, True))
    d.text((100, 34), 'CPG', fill=(*GREY, 200), font=_f(12))
    d.text((w // 2 + 10, 26), f'{snn:.0%}', fill=(*CYAN, 255), font=_f(32, True))
    d.text((w // 2 + 98, 34), 'SNN', fill=(*GREY, 200), font=_f(12))

    # Mix bar
    bar_y, bar_h, bx, bw = 66, 12, 12, w - 24
    d.rounded_rectangle([(bx, bar_y), (bx + bw, bar_y + bar_h)],
                        radius=4, fill=(*BAR_BG, 200))
    cpg_px = max(2, int(bw * cpg))
    d.rounded_rectangle([(bx, bar_y), (bx + cpg_px, bar_y + bar_h)],
                        radius=4, fill=(*ORANGE, 220))
    if bw - cpg_px > 2:
        d.rounded_rectangle([(bx + cpg_px, bar_y), (bx + bw, bar_y + bar_h)],
                            radius=4, fill=(*CYAN, 220))

    # Competence + SNN mix on same line
    d.text((12, 84), f'Competence: {comp:.3f}', fill=(*ICE, 200), font=_f(12))
    d.text((w // 2, 84), f'SNN mix: {snn_mix:.3f}', fill=(*ICE, 200), font=_f(12))
    return panel


def _widget_cerebellum(w, h, s):
    """Cerebellum circuit: CF→GrC→PkC→DCN with live values."""
    panel, d = _glass(w, h)
    cf = s.get('cf_magnitude', 0.0)
    corr = s.get('correction_mag', 0.0)
    grc = s.get('grc_sparseness', 0.0)
    pkc = s.get('pf_pkc_weight', 0.0)
    dcn = s.get('dcn_activity', 0.0)

    d.text((12, 5), 'CEREBELLUM', fill=(*CYAN, 240), font=_f(13, True))

    # Flow diagram: MF → GrC → PkC → DCN (scaled positions)
    ns = _FONT_SCALE
    nodes = [('MF', int(50*ns), 28, ICE), ('GrC', int(140*ns), 28, BLUE),
             ('PkC', int(230*ns), 28, VIOLET), ('DCN', int(320*ns), 28, GOLD)]
    for label, nx, ny, col in nodes:
        d.rounded_rectangle([(nx-20, ny), (nx+20, ny+16)], radius=3,
                            fill=(*col, 60), outline=(*col, 160))
        d.text((nx-12, ny+2), label, fill=(*col, 240), font=_f(10, True))
    # Arrows
    for i in range(3):
        x1 = nodes[i][1] + 22
        x2 = nodes[i+1][1] - 22
        ny = nodes[i][2] + 8
        d.line([(x1, ny), (x2, ny)], fill=(*CYAN, 100), width=1)
        d.polygon([(x2, ny-3), (x2, ny+3), (x2+4, ny)], fill=(*CYAN, 140))

    # CF arrow (top, error signal)
    d.line([(w//2, 22), (w//2, 28)], fill=(*ORANGE, 150), width=1)
    d.text((w//2-8, 14), 'CF', fill=(*ORANGE, 180), font=_f(8))

    # Metric bars — all bars start at x=100 for alignment
    metrics = [
        ('CF Error', cf, ORANGE),
        ('Correction', corr, GREEN),
        ('GrC Sparse', grc, ICE),
        ('PF-PkC Wt', pkc, VIOLET),
        ('DCN Act', dcn, GOLD),
    ]
    y = 50
    bar_x = int(110 * _FONT_SCALE)  # scales with font size to prevent overlap
    bw = w - bar_x - int(65 * _FONT_SCALE)  # leave room for value text
    for label, val, color in metrics:
        d.text((12, y), label, fill=(*GREY, 210), font=_f(11))
        by = y + 2
        d.rounded_rectangle([(bar_x, by), (bar_x + bw, by + 10)], radius=3, fill=(*BAR_BG, 180))
        fw = max(0, min(bw, int(bw * min(1.0, float(val)))))
        if fw > 2:
            d.rounded_rectangle([(bar_x, by), (bar_x + fw, by + 10)], radius=3, fill=(*color, 200))
        d.text((bar_x + bw + 4, y), f'{float(val):.4f}', fill=(*color, 200), font=_f(10))
        y += 18
    return panel


def _widget_neuromod(w, h, s):
    """Neuromodulator bars: DA, Serotonin (via emotion), Noradrenaline (via arousal)."""
    panel, d = _glass(w, h)
    da = s.get('da_reward', 0.0)
    # Derive neuromodulator levels from actual training signals
    # Serotonin ~ stability/upright (calm = high 5-HT)
    upright = s.get('upright', 0.5)
    vel = s.get('vel_mps', 0.0)
    serotonin = np.clip(upright * 0.6 + (1.0 - min(1, abs(da))) * 0.3 + 0.1, 0, 1)
    # Noradrenaline ~ arousal/alertness (falls, reflex, error)
    cf = s.get('cf_magnitude', 0.0)
    reflex = s.get('reflex_magnitude', 0.0)
    fallen = s.get('is_fallen', 0)
    noradrenaline = np.clip(cf * 2.0 + reflex * 1.5 + fallen * 0.5 + vel * 0.2, 0, 1)

    d.text((12, 5), 'NEUROMODULATORS', fill=(*PINK, 240), font=_f(13, True))

    mods = [
        ('Dopamine', (da + 1.0) / 2.0, PINK),   # DA is [-1,1] → [0,1]
        ('Serotonin', serotonin, GREEN),
        ('Noradren.', noradrenaline, ORANGE),
    ]
    bar_x = int(110 * _FONT_SCALE)  # aligned with cerebellum bars
    bw = w - bar_x - int(55 * _FONT_SCALE)
    y = 26
    for label, val, color in mods:
        d.text((12, y), label, fill=(*GREY, 210), font=_f(12))
        by = y + 2
        d.rounded_rectangle([(bar_x, by), (bar_x + bw, by + 12)], radius=3, fill=(*BAR_BG, 180))
        fw = max(0, min(bw, int(bw * max(0, min(1, val)))))
        if fw > 2:
            d.rounded_rectangle([(bar_x, by), (bar_x + fw, by + 12)], radius=3, fill=(*color, 220))
        d.text((bar_x + bw + 4, y), f'{val:.2f}', fill=(*color, 210), font=_f(12))
        y += 26
    return panel


def _widget_forward_model(w, h, s):
    """Forward model prediction errors + Closed-Loop adaptation state."""
    panel, d = _glass(w, h)
    pred = s.get('pred_error', 0.0)
    terrain = s.get('terrain_error', 0.0)
    vestib = s.get('vestibular_error', 0.0)
    gain = s.get('forward_gain_mean', 0.0)

    d.text((12, 5), 'FORWARD MODEL', fill=(*BLUE, 240), font=_f(13, True))

    errors = [
        ('Prediction', pred, ORANGE),
        ('Terrain', terrain, RED),
        ('Vestibular', vestib, VIOLET),
    ]
    bar_x = int(110 * _FONT_SCALE)  # aligned with other widgets
    bw = w - bar_x - int(65 * _FONT_SCALE)
    y = 26
    for label, val, color in errors:
        d.text((12, y), label, fill=(*GREY, 210), font=_f(12))
        by = y + 2
        d.rounded_rectangle([(bar_x, by), (bar_x + bw, by + 12)], radius=3, fill=(*BAR_BG, 180))
        fw = max(0, min(bw, int(bw * min(1.0, float(val)))))
        if fw > 2:
            d.rounded_rectangle([(bar_x, by), (bar_x + fw, by + 12)], radius=3, fill=(*color, 220))
        d.text((bar_x + bw + 4, y), f'{float(val):.4f}', fill=(*color, 210), font=_f(11))
        y += 22

    # Closed-Loop adaptation state (Issue #78)
    cl_adapt = s.get('cl_adaptations', 0)
    cl_vor = s.get('cl_vor_hip_gain', 0.0)
    cl_improve = s.get('cl_consec_improve', 0)
    cl_fail = s.get('cl_consec_fail', 0)
    if cl_adapt > 0:
        # Show CL status: green=improving, red=failing, grey=neutral
        if cl_improve > 0:
            cl_col = GREEN
            cl_label = f'CL: \u2191{cl_improve}'
        elif cl_fail > 0:
            cl_col = RED
            cl_label = f'CL: \u2193{cl_fail}'
        else:
            cl_col = GREY
            cl_label = f'CL: \u2014'
        d.text((12, y+2), cl_label, fill=(*cl_col, 220), font=_f(11, True))
        cb_sc = s.get('cb_steer_correction', 0.0)
        cb_label = f'VOR:{cl_vor:.2f} CB:{cb_sc:+.2f} #{cl_adapt}'
        d.text((bar_x, y+2), cb_label, fill=(*ICE, 200), font=_f(10))
    else:
        d.text((12, y+2), f'FM Gain: {gain:.4f}', fill=(*BLUE, 210), font=_f(12))
    return panel


def _widget_reflex(w, h, s):
    """Reflex status badges."""
    panel, d = _glass(w, h)
    posture = s.get('posture_state', '')
    active = s.get('reflex_active', '')
    mag = s.get('reflex_magnitude', 0.0)
    tone = s.get('tone_magnitude', 0.0)
    stretch = s.get('stretch_magnitude', 0.0)

    d.text((12, 5), 'REFLEXES', fill=(*ORANGE, 240), font=_f(13, True))

    # Posture badge
    pc = GREEN if posture == 'stable' else (ORANGE if posture else GREY)
    d.rounded_rectangle([(12, 24), (w//2-4, 42)], radius=4,
                        fill=(*pc, 60), outline=(*pc, 160))
    d.text((18, 26), posture.upper() if posture else 'UNKNOWN',
           fill=(*pc, 240), font=_f(11, True))

    # Active reflexes
    if active:
        reflexes = [r.strip() for r in str(active).split(',') if r.strip()]
        x = w // 2 + 4
        for r in reflexes[:3]:
            tw = min(len(r) * 7 + 12, w - x - 4)
            d.rounded_rectangle([(x, 24), (x + tw, 42)], radius=4,
                                fill=(*RED, 60), outline=(*RED, 160))
            d.text((x + 4, 26), r[:8], fill=(*RED, 240), font=_f(10, True))
            x += tw + 4

    # Magnitude bars — aligned at x=100 like all other widgets
    bars = [('Reflex', mag, RED), ('Tone', tone, ICE), ('Stretch', stretch, BLUE)]
    bar_x = int(110 * _FONT_SCALE)
    bw = w - bar_x - int(65 * _FONT_SCALE)
    y = 50
    for label, val, color in bars:
        d.text((12, y), label, fill=(*GREY, 210), font=_f(12))
        by = y + 2
        d.rounded_rectangle([(bar_x, by), (bar_x + bw, by + 12)], radius=3, fill=(*BAR_BG, 180))
        fw = max(0, min(bw, int(bw * min(1.0, float(val)))))
        if fw > 2:
            d.rounded_rectangle([(bar_x, by), (bar_x + fw, by + 12)], radius=3, fill=(*color, 200))
        d.text((bar_x + bw + 4, y), f'{float(val):.4f}', fill=(*color, 200), font=_f(12))
        y += 26
    return panel


def _widget_behavior_emotion(w, h, s):
    """Combined behavior state + dominant emotion + drive."""
    panel, d = _glass(w, h)
    beh = s.get('behavior', 'walk')
    freq = s.get('freq_scale', 1.0)
    amp = s.get('amp_scale', 1.0)
    emo = s.get('emotion_dominant', '')
    drv = s.get('drive_dominant', '')
    curiosity = s.get('curiosity_reward', 0.0)
    da = s.get('da_reward', 0.0)

    bc = BEH_COLORS.get(beh, ICE)
    ec = EMO_COLORS.get(emo, GREY)

    # Adaptive layout: evenly spaced rows from top to bottom
    spacing = max(14, (h - 8) // 6)   # 6 rows, evenly distributed
    row0 = 4                          # BEHAVIOR / EMOTION
    row1 = row0 + spacing             # WALK / EXCITED
    row2 = row1 + spacing             # Freq:Amp / Drive
    row3 = row2 + spacing             # Cur:
    row4 = row3 + spacing             # DA bar

    # Behavior (left half)
    d.text((12, row0), 'BEHAVIOR', fill=(*CYAN, 240), font=_f(14, True))
    d.text((12, row1), beh.upper(), fill=(*bc, 255), font=_f(18, True))
    d.text((12, row2), f'Freq:{freq:.2f}  Amp:{amp:.2f}', fill=(*ICE, 200), font=_f(11))

    # Emotion + Drive (right half)
    mid = w // 2 + 20
    d.text((mid, row0), 'EMOTION', fill=(*VIOLET, 240), font=_f(14, True))
    # Derive emotion from signals when label is neutral/empty
    # (embodied_emotions.py thresholds are too narrow, always returns neutral)
    if not emo or emo == 'neutral':
        da_val = s.get('da_reward', 0.0)
        vel_val = s.get('vel_mps', 0.0)
        fallen_val = s.get('is_fallen', 0)
        cf_val = s.get('cf_magnitude', 0.0)
        if fallen_val:
            emo = 'fearful'
        elif da_val > 0.1 and vel_val > 0.05:
            emo = 'excited'
        elif da_val > 0.05 and vel_val < 0.02:
            emo = 'content'
        elif cf_val > 0.1:
            emo = 'tense'
        elif vel_val > 0.1:
            emo = 'focused'
        else:
            emo = 'calm'
    ec = EMO_COLORS.get(emo, GREY)
    emo_nice = {'excited': 'EXCITED', 'content': 'CONTENT', 'fearful': 'FEARFUL',
                'sad': 'SAD', 'calm': 'CALM', 'tense': 'TENSE',
                'focused': 'FOCUSED', 'neutral': 'NEUTRAL'}
    label = emo_nice.get(emo, emo.upper())
    d.text((mid, row1), label, fill=(*ec, 255), font=_f(18, True))

    # Drive (right half)
    if drv:
        d.text((mid, row2), f'Drive: {drv}', fill=(*GOLD, 200), font=_f(11))
    # Curiosity (own line)
    d.text((12, row3), f'Cur:{curiosity:.3f}', fill=(*CYAN, 200), font=_f(9))
    # DA reward as compact bar (bottom line)
    d.text((12, row4), 'DA', fill=(*PINK, 200), font=_f(9))
    da_bw = w - 100
    d.rounded_rectangle([(32, row4 + 2), (32 + da_bw, row4 + 8)], radius=2, fill=(*BAR_BG, 180))
    da_px = max(0, min(da_bw, int(da_bw * (da + 1.0) / 2.0)))
    if da_px > 2:
        dc = GREEN if da > 0 else RED
        d.rounded_rectangle([(32, row4 + 2), (32 + da_px, row4 + 8)], radius=2, fill=(*dc, 200))
    d.text((32 + da_bw + 4, row4), f'{da:+.2f}', fill=(*ICE, 200), font=_f(9))
    return panel


# ═══════════════════════════════════════════════════════════
def _widget_brain_status(w, h, s):
    """Brain status: Dream, Memory, Synaptogenesis, TPE, Episode.
    Combined widget for ball-scene context showing cognitive state."""
    panel, d = _glass(w, h)
    d.text((10, 4), 'BRAIN', fill=(*CYAN, 240), font=_f(13, True))

    # Adaptive line spacing: fit all rows into available height
    n_rows = 6
    line = max(11, (h - 20) // n_rows)
    y = 18
    _fs = 10 if h < 100 else 11  # smaller font when panel is tight

    # Task PE (DishBrain signal)
    tpe = s.get('task_pe', 0.0)
    tpe_c = GREEN if tpe < -0.2 else (GOLD if tpe < 0.3 else RED)
    d.text((10, y), 'Task PE', fill=(*GREY, 200), font=_f(_fs))
    d.text((w - 60, y), f'{tpe:+.2f}', fill=(*tpe_c, 240), font=_f(_fs, True))
    y += line

    # Ball episode
    ball_ep = s.get('ball_episode', 0)
    d.text((10, y), 'Episode', fill=(*GREY, 200), font=_f(_fs))
    d.text((w - 60, y), f'#{ball_ep}', fill=(*PINK, 220), font=_f(_fs, True))
    y += line

    # Consciousness Level
    cl = s.get('c_level', 0)
    d.text((10, y), 'Conscious', fill=(*GREY, 200), font=_f(_fs))
    d.text((w - 60, y), f'L{cl}', fill=(*ICE, 220), font=_f(_fs, True))
    y += line

    # PCI
    pci = s.get('pci', 0.0)
    pci_c = GREEN if pci > 0.3 else (GOLD if pci > 0.2 else RED)
    d.text((10, y), 'PCI', fill=(*GREY, 200), font=_f(_fs))
    d.text((w - 60, y), f'{pci:.3f}', fill=(*pci_c, 220), font=_f(_fs))
    y += line

    # Learning progress
    lp = s.get('learning_progress', s.get('pred_error', 0.0))
    d.text((10, y), 'Learning', fill=(*GREY, 200), font=_f(_fs))
    d.text((w - 60, y), f'{lp:.4f}', fill=(*ICE, 200), font=_f(_fs))
    y += line

    if y + 14 < h:  # only draw if it fits
        corr = s.get('correction_mag', 0.0)
        d.text((10, y), 'CB Corr', fill=(*GREY, 200), font=_f(_fs))
        d.text((w - 60, y), f'{corr:.4f}', fill=(*ORANGE, 200), font=_f(_fs))

    return panel


# BOTTOM BAR WIDGETS
# ═══════════════════════════════════════════════════════════

def _widget_distance(w, h, s):
    """Distance + speed."""
    panel, d = _glass(w, h)
    best = s.get('max_distance', 0.0)
    now = s.get('current_distance', best)
    speed = s.get('vel_mps', 0.0)
    d.text((10, 6), 'FROM START', fill=(*CYAN, 240), font=_f(13, True))
    d.text((10, 24), f'{now:.2f}m', fill=(*GREEN, 255), font=_f(28, True))
    d.text((10, 56), f'Speed: {speed:.3f} m/s', fill=(*ICE, 200), font=_f(12))
    d.text((10, 74), f'Best: {best:.2f}m', fill=(*GOLD, 200), font=_f(12))
    return panel


def _widget_falls(w, h, s):
    """Falls + upright + recoveries."""
    panel, d = _glass(w, h)
    falls = s.get('falls', 0)
    upright = s.get('upright', 1.0)
    fallen = s.get('is_fallen', 0)
    recoveries = s.get('recoveries', 0)
    fc = RED if falls > 10 else (ORANGE if falls > 3 else GREEN)
    status = 'FALLEN' if fallen else 'UPRIGHT'
    sc = RED if fallen else GREEN
    d.text((10, 6), 'STABILITY', fill=(*CYAN, 240), font=_f(13, True))
    d.text((10, 24), f'{falls}', fill=(*fc, 255), font=_f(30, True))
    d.text((55, 32), 'falls', fill=(*GREY, 200), font=_f(12))
    d.text((10, 56), status, fill=(*sc, 220), font=_f(14, True))
    d.text((10, 76), f'Rec:{recoveries}  Up:{upright:.2f}', fill=(*ICE, 200), font=_f(11))
    return panel


def _widget_vision(w, h, s):
    """Vision system: ball heading compass + distance + steering signal.
    Shows the creature's 'eyes' — where it sees the target object.
    """
    panel, d = _glass(w, h)
    heading = s.get('ball_heading', 0.0)    # -1 (left) to +1 (right)
    salience = s.get('ball_salience', 0.0)  # 0 (far) to 1 (touching)
    ball_dist = s.get('ball_dist', -1.0)    # meters, -1 = no ball
    steering = s.get('steering_offset', 0.0)
    approach_rw = s.get('ball_approach_reward', 0.0)

    d.text((10, 6), 'VISION', fill=(*PINK, 240), font=_f(13, True))

    # No target indicator
    if ball_dist < 0:
        d.text((10, 30), 'No target', fill=(*GREY, 150), font=_f(14))
        return panel

    # --- Compass: heading to ball ---
    cx, cy, r = w // 2, 55, 24
    # Outer ring — color indicates salience (bright=close, dim=far)
    ring_alpha = int(80 + salience * 175)
    d.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
             outline=(*PINK, ring_alpha), width=2)
    # Heading arrow (heading: -1=pi left, +1=pi right)
    arrow_angle = heading * math.pi  # convert [-1,+1] to [-pi,+pi]
    ax = cx + int(r * 0.85 * math.sin(arrow_angle))
    ay = cy - int(r * 0.85 * math.cos(arrow_angle))
    arrow_col = LIME if abs(heading) < 0.3 else (GOLD if abs(heading) < 0.6 else RED)
    d.line([(cx, cy), (ax, ay)], fill=(*arrow_col, 240), width=3)
    # Center dot
    d.ellipse([(cx - 2, cy - 2), (cx + 2, cy + 2)], fill=(*WHITE, 200))
    # N/S/E/W labels
    d.text((cx - 3, cy - r - 12), 'F', fill=(*GREY, 150), font=_f(9))  # Forward
    d.text((cx + r + 4, cy - 5), 'R', fill=(*GREY, 150), font=_f(9))   # Right
    d.text((cx - r - 10, cy - 5), 'L', fill=(*GREY, 150), font=_f(9))  # Left

    # --- Distance + Salience ---
    d.text((10, 88), f'Dist: {ball_dist:.1f}m', fill=(*ICE, 220), font=_f(12))
    # Salience bar
    bar_x, bar_w = 10, w - 20
    d.rounded_rectangle([(bar_x, 106), (bar_x + bar_w, 114)], radius=3, fill=(*BAR_BG, 180))
    sp = max(0, min(bar_w, int(bar_w * salience)))
    if sp > 2:
        d.rounded_rectangle([(bar_x, 106), (bar_x + sp, 114)], radius=3, fill=(*PINK, 220))
    d.text((bar_x + bar_w - 40, 88), f'sal:{salience:.2f}', fill=(*PINK, 200), font=_f(11))

    # --- Steering + TPE ---
    steer_c = GREEN if abs(steering) < 0.2 else (GOLD if abs(steering) < 0.5 else RED)
    d.text((10, 120), f'Steer:{steering:+.2f}', fill=(*steer_c, 220), font=_f(11))
    # Task Prediction Error (DishBrain signal)
    tpe = s.get('task_pe', 0.0)
    tpe_c = GREEN if tpe < -0.2 else (GOLD if tpe < 0.3 else RED)
    d.text((w // 2, 120), f'TPE:{tpe:+.2f}', fill=(*tpe_c, 240), font=_f(11, True))
    # Contact indicator
    if ball_dist >= 0 and ball_dist < 0.15:
        d.text((w // 2 - 30, 6), 'CONTACT!', fill=(*LIME, 255), font=_f(13, True))
    elif approach_rw > 0.01:
        d.text((w // 2 + 10, 88), f'DA+{approach_rw:.1f}', fill=(*GOLD, 240), font=_f(11, True))

    return panel


def _widget_sensory(w, h, s):
    """Sensory status: scent + sound + olfactory steering."""
    panel, d = _glass(w, h)
    smell = s.get('smell_strength', 0.0)
    smell_dir = s.get('smell_direction', 0.0)
    sound = s.get('sound_intensity', 0.0)
    scents_found = s.get('scents_found', 0)
    steering = s.get('olfactory_steering', 0.0)

    d.text((10, 6), 'SENSORY', fill=(*LIME, 240), font=_f(13, True))

    # Scent bar
    d.text((10, 26), 'Smell', fill=(*GREEN, 200), font=_f(11))
    smell_bw = w - 80
    d.rounded_rectangle([(50, 28), (50 + smell_bw, 38)], radius=3, fill=(*BAR_BG, 180))
    sp = max(0, min(smell_bw, int(smell_bw * smell)))
    if sp > 2:
        d.rounded_rectangle([(50, 28), (50 + sp, 38)], radius=3, fill=(*GREEN, 220))
    d.text((50 + smell_bw + 4, 26), f'{smell:.2f}', fill=(*GREEN, 200), font=_f(11))

    # Sound bar
    d.text((10, 46), 'Sound', fill=(*ORANGE, 200), font=_f(11))
    d.rounded_rectangle([(50, 48), (50 + smell_bw, 58)], radius=3, fill=(*BAR_BG, 180))
    sndp = max(0, min(smell_bw, int(smell_bw * sound)))
    if sndp > 2:
        d.rounded_rectangle([(50, 48), (50 + sndp, 58)], radius=3, fill=(*ORANGE, 220))
    d.text((50 + smell_bw + 4, 46), f'{sound:.2f}', fill=(*ORANGE, 200), font=_f(11))

    # Scents found + steering
    d.text((10, 68), f'Found: {scents_found}', fill=(*GOLD, 220), font=_f(13, True))
    steer_c = GREEN if abs(steering) < 0.3 else (ORANGE if abs(steering) < 0.7 else RED)
    d.text((10, 86), f'Steer: {steering:+.3f}', fill=(*steer_c, 220), font=_f(12))

    # Direction indicator (small compass arrow)
    if abs(smell_dir) > 0.01:
        cx, cy, r = w - 30, 80, 16
        d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(*GREEN, 120), width=1)
        ax = cx + int(r * 0.8 * math.cos(smell_dir))
        ay = cy - int(r * 0.8 * math.sin(smell_dir))
        d.line([(cx, cy), (ax, ay)], fill=(*GREEN, 220), width=2)

    return panel


def _widget_valence_arousal(w, h, s):
    """Valence/Arousal 2D compass — derived from multiple signals."""
    panel, d = _glass(w, h)
    # Try direct values first (if logged), otherwise derive from signals
    valence = s.get('valence', None)
    arousal = s.get('arousal', None)

    if valence is None:
        # Derive valence from: DA reward (+), falls (-), distance (+)
        da = s.get('da_reward', 0.0)
        fallen = s.get('is_fallen', 0)
        vel = s.get('vel_mps', 0.0)
        valence = np.clip(da * 0.5 + vel * 0.3 - fallen * 0.8, -1, 1)

    if arousal is None:
        # Derive arousal from: velocity, reflex activity, CF error
        vel = s.get('vel_mps', 0.0)
        cf = s.get('cf_magnitude', 0.0)
        reflex = s.get('reflex_magnitude', 0.0)
        arousal = np.clip(vel * 0.4 + cf * 2.0 + reflex * 1.5 + 0.1, 0, 1)

    d.text((10, 6), 'VALENCE / AROUSAL', fill=(*VIOLET, 240), font=_f(12, True))

    # 2D compass
    cx, cy, r = w // 2, 62, 30
    d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(*GREY, 100), width=1)
    d.line([(cx - r, cy), (cx + r, cy)], fill=(*GREY, 60), width=1)
    d.line([(cx, cy - r), (cx, cy + r)], fill=(*GREY, 60), width=1)

    # Dot position
    dx = cx + int(r * 0.85 * valence)
    dy = cy - int(r * 0.85 * arousal)
    if valence > 0 and arousal > 0.3:
        dot_c = GREEN
    elif valence > 0:
        dot_c = CYAN
    elif arousal > 0.5:
        dot_c = RED
    else:
        dot_c = GREY
    d.ellipse([(dx - 4, dy - 4), (dx + 4, dy + 4)], fill=(*dot_c, 240))

    # Labels
    d.text((cx + r + 3, cy - 6), '+V', fill=(*GREEN, 150), font=_f(9))
    d.text((cx - r - 14, cy - 6), '-V', fill=(*RED, 150), font=_f(9))
    d.text((cx - 4, cy - r - 12), '+A', fill=(*ORANGE, 150), font=_f(9))

    # Values
    d.text((10, h - 22), f'V:{valence:+.2f}', fill=(*ICE, 200), font=_f(11))
    d.text((w // 2, h - 22), f'A:{arousal:.2f}', fill=(*ICE, 200), font=_f(11))
    return panel


def _widget_pci(w, h, s):
    """PCI consciousness index."""
    panel, d = _glass(w, h)
    pci = s.get('pci', 0.0)
    cl = s.get('consciousness_level', s.get('c_level', 0))
    dev = s.get('dev_phase', '')
    pc = VIOLET if pci > 0.31 else ICE
    d.text((10, 6), 'PCI', fill=(*VIOLET, 240), font=_f(13, True))
    d.text((10, 24), f'{pci:.4f}', fill=(*pc, 255), font=_f(28, True))
    th = 'ABOVE' if pci > 0.31 else 'below'
    d.text((10, 56), f'Threshold: {th}', fill=(*GREY, 200), font=_f(12))
    d.text((10, 74), f'Level: {cl}', fill=(*ICE, 200), font=_f(12))
    if dev:
        d.text((10, 90), f'Dev: {dev}', fill=(*CYAN, 180), font=_f(11))
    return panel


def _widget_step(w, h, s):
    """Training step counter + dev schedule info."""
    panel, d = _glass(w, h)
    step = s.get('step', 0)
    dev_comp = s.get('dev_competence', 0.0)
    dev_perturb = s.get('dev_perturb', 0.0)

    d.text((10, 6), 'TRAINING', fill=(*CYAN, 240), font=_f(13, True))
    d.text((10, 24), f'{step:,}', fill=(*WHITE, 255), font=_f(24, True))
    d.text((10, 52), 'steps', fill=(*GREY, 200), font=_f(12))

    # Ball episode + curriculum stage
    ball_ep = s.get('ball_episode', 0)
    if ball_ep > 0:
        d.text((10, 70), f'Episode #{ball_ep}', fill=(*PINK, 220), font=_f(11))
    elif dev_comp > 0:
        d.text((10, 70), f'Dev comp: {dev_comp:.3f}', fill=(*ICE, 200), font=_f(11))
    if dev_perturb > 0:
        d.text((10, 86), f'Perturb: {dev_perturb:.3f}', fill=(*ORANGE, 180), font=_f(11))
    return panel


# ═══════════════════════════════════════════════════════════
# BRAND BAR
# ═══════════════════════════════════════════════════════════

def _brand_bar(w, h, stats):
    """MH-FLOCKE branding + status bar — top left."""
    panel = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=4,
                           fill=(8, 10, 16, 160), outline=(0, 200, 220, 50))

    step = stats.get('step', 0)
    total = stats.get('total_steps', 50000)

    draw.text((10, 4), 'MH-FLOCKE', fill=(*CYAN, 240), font=_f(17, True))
    version = stats.get('version', 'v0.4.3')
    draw.text((130, 6), f'Level 15 {version}', fill=(*GREY, 180), font=_f(13))

    creature = stats.get('creature', 'Go2')
    task = stats.get('task', '')
    _cf = _f(17, True)
    draw.text((280, 6), creature, fill=(*WHITE, 200), font=_cf)
    _cw = int(draw.textlength(creature, font=_cf))
    if task:
        draw.text((280 + _cw + 10, 6), task, fill=(*GREY, 140), font=_f(13))

    if total > 0:
        prog = min(1.0, step / total)
        bar_y = h - 4
        bar_w = w - 20
        draw.line([(10, bar_y), (10 + bar_w, bar_y)],
                  fill=(30, 35, 50, 120), width=2)
        if prog > 0.001:
            draw.line([(10, bar_y), (10 + int(bar_w * prog), bar_y)],
                      fill=(*CYAN, 160), width=2)

    return panel


# ═══════════════════════════════════════════════════════════
# SCENE OVERLAYS (scent markers, steering arrow)
# ═══════════════════════════════════════════════════════════

def _project_world_to_screen(wx, wy, wz, cam_params, screen_w, screen_h):
    """
    Project MuJoCo world coordinates to screen pixel using camera params.
    MuJoCo camera: azimuth=angle around Z, elevation=angle from horizon,
    distance=from lookat, lookat=target point.
    Returns (px, py) or None if behind camera.
    """
    az = cam_params.get('azimuth', 225) * math.pi / 180
    el = cam_params.get('elevation', -15) * math.pi / 180
    dist = cam_params.get('distance', 3.5)
    la = cam_params.get('lookat', [0, 0, 0.3])

    # Camera position in world
    cam_x = la[0] + dist * math.cos(el) * math.cos(az)
    cam_y = la[1] + dist * math.cos(el) * math.sin(az)
    cam_z = la[2] + dist * math.sin(-el)

    # Vector from camera to point
    dx = wx - cam_x
    dy = wy - cam_y
    dz = wz - cam_z

    # Camera frame: forward = toward lookat, up = Z
    fx = la[0] - cam_x
    fy = la[1] - cam_y
    fz = la[2] - cam_z
    fd = math.sqrt(fx*fx + fy*fy + fz*fz)
    if fd < 1e-6:
        return None
    fx /= fd; fy /= fd; fz /= fd

    # Right vector (cross forward x world-up)
    rx = fy * 1.0 - fz * 0.0  # forward x (0,0,1)
    ry = fz * 0.0 - fx * 1.0
    rz = fx * 0.0 - fy * 0.0
    rd = math.sqrt(rx*rx + ry*ry + rz*rz)
    if rd < 1e-6:
        return None
    rx /= rd; ry /= rd; rz /= rd

    # Up vector (cross right x forward)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx

    # Project onto camera axes
    depth = dx*fx + dy*fy + dz*fz
    if depth < 0.1:
        return None  # behind camera
    screen_r = dx*rx + dy*ry + dz*rz
    screen_u = dx*ux + dy*uy + dz*uz

    # Perspective projection (MuJoCo default fov ~45 deg)
    fov = 34 * math.pi / 180  # match MuJoCo renderer
    scale = screen_w / (2 * math.tan(fov / 2) * depth)

    px = int(screen_w / 2 + screen_r * scale)
    py = int(screen_h / 2 - screen_u * scale)
    return (px, py)


def _draw_scent_markers(canvas, stats, cam_params):
    """
    Draw scent source positions as labeled markers on the scene.
    Scent = green diamond with 'S' label, Sound = blue ring.
    Uses proper perspective projection matching MuJoCo camera.
    """
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    smell_str = stats.get('smell_strength', 0.0)
    sound_int = stats.get('sound_intensity', 0.0)

    for si in range(5):
        sx = stats.get(f'scent_{si}_x', None)
        sy = stats.get(f'scent_{si}_y', None)
        if sx is None or sy is None:
            continue

        pt = _project_world_to_screen(sx, sy, 0.02, cam_params, w, h)
        if pt is None:
            continue
        px, py = pt

        if 20 < px < w - 500 and 40 < py < h - 160:
            # Scent marker: green diamond shape
            r = 6
            diamond = [(px, py - r), (px + r, py), (px, py + r), (px - r, py)]
            # Glow intensity based on smell strength
            glow_a = min(80, max(20, int(smell_str * 120)))
            draw.polygon(diamond, fill=(50, 220, 100, glow_a),
                         outline=(80, 255, 120, min(160, glow_a * 2)))

    # Sound event: blue pulse ring at creature position (center of scene)
    if sound_int > 0.05:
        cx, cy = w // 3, int(h * 0.5)  # approximate creature screen pos
        ring_r = int(20 + sound_int * 30)
        ring_a = min(100, int(sound_int * 150))
        draw.ellipse([(cx - ring_r, cy - ring_r), (cx + ring_r, cy + ring_r)],
                     outline=(60, 140, 255, ring_a), width=2)


def _draw_steering_arrow(canvas, stats, cam_params):
    """Draw VOR steering direction as an arrow near the creature.
    
    Updated from olfactory-only to VOR (Visual Orienting Response).
    Shows the direction the creature is trying to turn, with color
    coding: green=on target, gold=correcting, red=far off.
    Also draws ball marker and distance line when ball is in scene.
    """
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    # --- Ball marker: red star at ball position ---
    ball_dist = stats.get('ball_dist', -1.0)
    if ball_dist >= 0:
        # Try to project ball position to screen
        for si in range(5):
            sx = stats.get(f'scent_{si}_x', None)
            sy = stats.get(f'scent_{si}_y', None)
            if sx is None or sy is None:
                continue
            pt = _project_world_to_screen(sx, sy, 0.15, cam_params, w, h)
            if pt is None:
                continue
            bx, by = pt
            if 20 < bx < w - 500 and 40 < by < h - 160:
                # Ball marker: red/pink circle with glow
                salience = stats.get('ball_salience', 0.0)
                ball_r = max(4, min(12, int(8 + salience * 8)))
                ball_a = max(60, min(255, int(100 + salience * 155)))
                draw.ellipse([(bx - ball_r, by - ball_r), (bx + ball_r, by + ball_r)],
                             fill=(255, 80, 120, ball_a),
                             outline=(255, 180, 200, min(255, ball_a + 40)))
                # Distance label
                d_label = f'{ball_dist:.1f}m'
                draw.text((bx - 12, by - ball_r - 14), d_label,
                          fill=(255, 180, 200, ball_a), font=_f(10))
            break  # Only first scent = ball

    # --- VOR steering arrow (replaces old olfactory arrow) ---
    steering = stats.get('steering_offset', 0.0)
    vor_smoothed = stats.get('vor_smoothed', steering)
    ball_heading = stats.get('ball_heading', 0.0)
    if abs(vor_smoothed) < 0.005 and abs(ball_heading) < 0.02:
        return  # No visual target, no arrow

    # Arrow origin: creature screen position (approximate center)
    ox, oy = w // 2, int(h * 0.55)

    # Arrow length proportional to VOR signal strength
    arrow_len = min(70, max(15, int(abs(vor_smoothed) * 200 + 15)))
    # Direction: heading to ball, not just steering offset
    # ball_heading: -1..+1 mapped to angle relative to forward
    angle = math.pi / 2 + ball_heading * math.pi * 0.8
    ax = ox + int(arrow_len * math.cos(angle))
    ay = oy - int(arrow_len * math.sin(angle))

    # Color: green=on target, gold=correcting, red=far off
    abs_h = abs(ball_heading)
    if abs_h < 0.15:
        arrow_c = (100, 255, 150, 200)  # Green: nearly facing ball
    elif abs_h < 0.4:
        arrow_c = (255, 200, 60, 200)   # Gold: moderate correction
    else:
        arrow_c = (255, 80, 80, 180)    # Red: ball far to the side/behind

    # Draw arrow shaft
    draw.line([(ox, oy), (ax, ay)], fill=arrow_c, width=3)
    # Arrowhead
    head_len = 10
    for side in [-0.4, 0.4]:
        ha = angle + math.pi + side
        hx = ax + int(head_len * math.cos(ha))
        hy = ay - int(head_len * math.sin(ha))
        draw.line([(ax, ay), (hx, hy)], fill=arrow_c, width=2)

    # VOR label
    if abs(vor_smoothed) > 0.01:
        label = f'VOR:{vor_smoothed:+.2f}'
        draw.text((ox + 8, oy + 8), label, fill=arrow_c, font=_f(10))


# ═══════════════════════════════════════════════════════════
# MAIN OVERLAY CLASS
# ═══════════════════════════════════════════════════════════

class Go2DashboardOverlay:
    """
    L-shaped glass overlay with Brain 3D flagship.
    Memory-safe: all PIL images explicitly closed after use.

    Layout:
      Right column: Brain 3D (large!) + CPG/SNN + Behavior/Emotion
      Bottom bar: Distance, Falls, Sensory, Valence/Arousal, PCI, Step
      Scene: Scent markers + steering arrow
    """

    def __init__(self, width=1920, height=1080, population_sizes=None):
        self.w = width
        self.h = height

        # Scale factor: all widget dimensions scale proportionally
        # Reference: 1920x1080. At 2560x1440, scale = 1.333
        self._scale = width / 1920.0
        s = self._scale

        # Scale all fonts globally — fixes tiny text at 1440p+
        global _FONT_SCALE, _FONTS
        _FONT_SCALE = s
        _FONTS = {}  # Clear cache when scale changes

        # Right column (25% of width, proportional)
        self.right_w = int(480 * s)
        self.right_x = width - self.right_w - int(10 * s)

        # Bottom bar
        self.bottom_h = int(110 * s)  # Was 140, reduced to prevent overlap at 1440p
        self.bottom_y = height - self.bottom_h - int(10 * s)

        # Right column: compute available height for widgets
        # Total height minus top padding, bottom bar, and bottom padding
        self._right_avail = self.bottom_y - 8  # available px for right column

        self._n = 0
        self._population_sizes = population_sizes

        # Brain 3D Network — flagship visualization (LARGE)
        self._brain3d_fn = None
        self._brain3d_state = None
        self._brain3d_h = int(320 * s)  # v3.1: Brain3D is the masterpiece, needs room
        try:
            from src.viz.brain_3d import render_brain_network, BrainNetworkState
            self._brain3d_fn = render_brain_network
            self._brain3d_state = BrainNetworkState(n_neurons=population_sizes.get("n_total", 350) if population_sizes else 350, n_display=min(500, population_sizes.get("n_total", 350) if population_sizes else 350), population_sizes=population_sizes)
            print(f'  Dashboard v2.0: L-overlay {width}x{height}, '
                  f'Brain 3D ACTIVE ({self.right_w}x{self._brain3d_h})')
        except ImportError as e:
            print(f'  Dashboard v2.0: L-overlay {width}x{height}, '
                  f'Brain 3D unavailable ({e})')

        # Camera params for scene overlays (updated per frame)
        self._cam = {'azimuth': 225, 'elevation': -15, 'distance': 3.5,
                     'lookat': [0, 0, 0.3]}

    def set_camera(self, cam_params: Dict):
        """Update camera params for scent marker projection."""
        self._cam.update(cam_params)

    def composite(self, scene_rgb, stats, cam_params=None) -> np.ndarray:
        """
        Overlay all widgets + scene markers onto 3D frame.

        Args:
            scene_rgb: numpy [H,W,3] uint8 from MuJoCo
            stats: dict from FLOG stats frame (45+ metrics)
            cam_params: optional camera dict for scent projection

        Returns:
            numpy [H,W,3] uint8 with overlay
        """
        if cam_params:
            self._cam.update(cam_params)

        scene = Image.fromarray(scene_rgb)
        if scene.size != (self.w, self.h):
            scene = scene.resize((self.w, self.h), Image.LANCZOS)

        canvas = scene.convert('RGBA')
        scene.close()

        rw, rx, pad = self.right_w, self.right_x, int(6 * self._scale)

        # ── SCENE OVERLAYS (below widgets) ──
        _draw_scent_markers(canvas, stats, self._cam)
        _draw_steering_arrow(canvas, stats, self._cam)

        # ── BRAND BAR (top left) ──
        brand_w = min(int(550 * self._scale), self.w - rw - 30)
        brand_stats = dict(stats)
        brand_stats['creature'] = stats.get('creature', 'Freenove')
        brand_stats['task'] = stats.get('task', stats.get('terrain_type', 'flat'))
        brand_stats['total_steps'] = stats.get('total_steps', 50000)
        brand_stats['version'] = stats.get('version', 'v0.4.3')
        w_brand = _brand_bar(brand_w, 28, brand_stats)
        canvas.paste(w_brand, (10, 10), w_brand)
        w_brand.close()

        # ── RIGHT COLUMN ──
        y = 8

        # Brain 3D Network (FLAGSHIP — large!)
        if self._brain3d_fn:
            try:
                snn_mix = stats.get('snn_mix', 0.1)
                activity = max(0.03, min(0.15, snn_mix * 0.2))
                # v0.4.2: Use real spike data from FLOG if available
                spikes_data = stats.get('spikes', None)
                if spikes_data is not None:
                    spike_raster = np.array(spikes_data, dtype=float)
                else:
                    # Fallback: generate from firing rate (legacy FLOGs without spike data)
                    n_viz = self._brain3d_state.n_display if self._brain3d_state else 350
                    spike_raster = np.random.binomial(1, activity, n_viz).astype(float)

                brain_img, self._brain3d_state = self._brain3d_fn(
                    spike_raster,
                    width=rw,
                    height=self._brain3d_h,
                    n_display=self._brain3d_state.n_display if self._brain3d_state else 350,
                    state=self._brain3d_state,
                    brain_state=stats,
                )
                brain_rgba = brain_img.convert('RGBA')
                brain_img.close()
                canvas.paste(brain_rgba, (rx, y), brain_rgba)
                brain_rgba.close()
                y += self._brain3d_h + pad
            except Exception as e:
                if self._n < 3:
                    print(f'  Brain3D render error: {e}')
                y += pad

        # Right column widgets — v3.2: fill to bottom edge of bottom bar widgets
        # The right column extends to the same y as the bottom bar's lower edge.
        # This gives ~130px extra vs stopping at bottom bar TOP.
        _right_bottom = self.bottom_y + self.bottom_h  # Bottom edge of bottom bar
        avail = _right_bottom - y  # No extra pad subtracted — last widget fills flush

        # Widget proportions (relative weights — bigger = more space)
        widgets = [
            ('cpg_mix',    3, _widget_cpg_mix),
            ('behavior',   4, _widget_behavior_emotion),
            ('cerebellum', 4, _widget_cerebellum),       # Needs more: has diagram
            ('neuromod',   3, _widget_neuromod),
            ('fwd_model',  3, _widget_forward_model),
            ('brain',      2, _widget_brain_status),
        ]
        total_weight = sum(w[1] for w in widgets)
        total_pads = (len(widgets) - 1) * pad
        usable = avail - total_pads

        for i, (name, weight, fn) in enumerate(widgets):
            wh = max(55, int(usable * weight / total_weight))
            # Last widget: fill exactly to bottom bar lower edge (flush)
            if i == len(widgets) - 1:
                wh = max(55, _right_bottom - y)
            w_img = fn(rw, wh, stats)
            canvas.paste(w_img, (rx, y), w_img)
            w_img.close()
            y += wh + pad

        # ── BOTTOM BAR (7 widgets) ──
        by = self.bottom_y
        bh = self.bottom_h
        bottom_w = self.w - rw - int(30 * self._scale)
        n_bot = 7
        ww = (bottom_w - (n_bot - 1) * pad) // n_bot

        bot_fns = [
            _widget_distance,
            _widget_falls,
            _widget_vision,
            _widget_sensory,
            _widget_valence_arousal,
            _widget_pci,
            _widget_step,
        ]
        for i, fn in enumerate(bot_fns):
            w_img = fn(ww, bh, stats)
            canvas.paste(w_img, (10 + i * (ww + pad), by), w_img)
            w_img.close()

        # Convert to RGB numpy (memory-safe: close intermediate)
        rgb = canvas.convert('RGB')
        canvas.close()
        result = np.asarray(rgb).copy()  # copy so we can close the PIL image
        rgb.close()

        self._n += 1
        if self._n % 30 == 0:
            gc.collect()

        return result
