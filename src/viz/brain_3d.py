"""
MH-FLOCKE — Brain 3D v0.4.2
========================================
3D network visualization of SNN activity patterns.

v0.4.2: Uses REAL population sizes from SNN topology.
  Old: hardcoded 350 neurons with fake proportions (10%/45%/10%/5%/5%/25%)
  New: actual population sizes from compute_cerebellar_populations()
       e.g. Freenove: 48 MF + 106 GrC + 18 GoC + 24 PkC + 24 DCN + 12 OUT = 232
       e.g. Go2: 304 MF + 4000 GrC + 200 GoC + 24 PkC + 24 DCN + 72 OUT = 4624

  Spike data: from FLOG (real spikes, max 200 sampled) — NOT random noise.
  Synapses: schematic (same connectivity pattern, correct proportions,
            but not the exact synapse indices from the SNN — too many to store).
  Positions: deterministic layout based on population sizes (golden ratio spacing).
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image, ImageDraw, ImageFilter

DEEP_SPACE = (10, 12, 18)
COOL_BLUE = (60, 130, 200)
DEEP_BLUE = (25, 40, 80)
SPIKE_RED = (255, 60, 40)
FIRE_ORANGE = (255, 140, 40)
VIOLET_PULSE = (160, 80, 220)
CYAN_ACCENT = (0, 200, 220)
ICE_BLUE = (140, 180, 220)
SYNAPSE_BASE = (30, 60, 120)
SYNAPSE_ACTIVE = (0, 200, 220)
OUTPUT_GOLD = (255, 200, 60)

# Cerebellar architecture — 6 populations
LAYER_COLORS = {
    0: CYAN_ACCENT,     # Mossy Fibers (input)
    1: COOL_BLUE,       # Granule Cells (expansion, sparse)
    2: (200, 80, 80),   # Golgi Cells (inhibitory, red-ish)
    3: VIOLET_PULSE,    # Purkinje Cells (learning output)
    4: FIRE_ORANGE,     # DCN (motor correction)
    5: OUTPUT_GOLD,     # Output neurons
}
LAYER_LABELS = ['MF', 'GrC', 'GoC', 'PkC', 'DCN', 'OUT']
DENDRITE_COLOR = (120, 60, 180)


def _lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(len(c1)))


def _rgba(color, alpha):
    return color[:3] + (max(0, min(255, int(alpha))),)


class BrainNetworkState:
    """Persistent state across frames.
    
    v0.4.2: Accepts real population sizes instead of hardcoded percentages.
    
    Args:
        n_neurons: total neuron count from SNN (e.g. 232 for Freenove, 4624 for Go2)
        n_display: max neurons to render (caps at n_neurons)
        population_sizes: dict with keys n_input, n_granule, n_golgi, n_purkinje, n_dcn, n_output
                         If None, falls back to proportional estimates (legacy behavior).
    """

    def __init__(self, n_neurons: int, n_display: int = 500,
                 population_sizes: Optional[Dict[str, int]] = None):
        self.n_neurons = n_neurons
        self.n_display = min(n_display, n_neurons) if n_neurons > 0 else n_display
        self.population_sizes = population_sizes
        self.angle = 0.0
        self.pulse_phase = 0.0
        self.fade = np.full(self.n_display, 10, dtype=np.int32)
        self.activity_history = []
        self.positions_3d = None
        self.layer_idx = None
        self.synapse_pairs = None
        self._ready = False

    def init_layout(self):
        n = self.n_display

        if self.population_sizes:
            # REAL population sizes from SNN topology
            ps = self.population_sizes
            n_mf = ps.get('n_input', 48)
            n_grc = ps.get('n_granule', 106)
            n_goc = ps.get('n_golgi', 18)
            n_pkc = ps.get('n_purkinje', 24)
            n_dcn = ps.get('n_dcn', 24)
            n_out = ps.get('n_output', 12)
            total_real = n_mf + n_grc + n_goc + n_pkc + n_dcn + n_out

            # Scale to n_display if needed (for large networks like Go2 4624 neurons)
            if total_real > n:
                scale = n / total_real
                n_mf = max(4, int(n_mf * scale))
                n_grc = max(4, int(n_grc * scale))
                n_goc = max(2, int(n_goc * scale))
                n_pkc = max(2, int(n_pkc * scale))
                n_dcn = max(2, int(n_dcn * scale))
                n_out = max(2, n - n_mf - n_grc - n_goc - n_pkc - n_dcn)
            elif total_real < n:
                # Pad output if n_display > total_real
                n_out = n_out + (n - total_real)
        else:
            # Legacy fallback: estimate from n_display
            n_mf = max(8, int(n * 0.10))
            n_grc = max(20, int(n * 0.45))
            n_goc = max(6, int(n * 0.10))
            n_pkc = max(4, int(n * 0.05))
            n_dcn = max(4, int(n * 0.05))
            n_out = max(4, n - n_mf - n_grc - n_goc - n_pkc - n_dcn)

        self.layer_sizes = [n_mf, n_grc, n_goc, n_pkc, n_dcn, n_out]

        # Layer index per neuron
        self.layer_idx = np.zeros(n, dtype=np.int32)
        off = 0
        for li, sz in enumerate(self.layer_sizes):
            self.layer_idx[off:off + sz] = li
            off += sz

        # 3D positions: cerebellar layout
        rng = np.random.RandomState(42)
        pts = np.zeros((n, 3), dtype=np.float32)

        layer_x = [-2.0, -0.7, -0.7, 0.7, 1.4, 2.0]
        layer_y_offset = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
        layer_radius = [0.5, 1.2, 0.4, 0.3, 0.3, 0.5]

        off = 0
        for li, sz in enumerate(self.layer_sizes):
            golden = math.pi * (3.0 - math.sqrt(5.0))
            for i in range(sz):
                r = layer_radius[li] * math.sqrt((i + 0.5) / max(sz, 1))
                theta = golden * i
                y = r * math.cos(theta) + layer_y_offset[li] + rng.uniform(-0.04, 0.04)
                z = r * math.sin(theta) + rng.uniform(-0.04, 0.04)
                x = layer_x[li] + rng.uniform(-0.12, 0.12)
                pts[off + i] = [x, y, z]
            off += sz

        self.positions_3d = pts

        # Synapses: schematic cerebellar connectivity
        # Uses CORRECT proportions but not exact synapse indices from SNN
        pairs = []
        off_mf = 0
        off_grc = n_mf
        off_goc = off_grc + n_grc
        off_pkc = off_goc + n_goc
        off_dcn = off_pkc + n_pkc
        off_out = off_dcn + n_dcn

        # MF → GrC (4 inputs per GrC, matching biology)
        for gi in range(off_grc, off_grc + n_grc):
            nc = min(4, n_mf)
            targets = rng.choice(range(off_mf, off_mf + n_mf), size=nc, replace=False)
            for ti in targets:
                pairs.append((ti, gi))

        # GrC → GoC (excitatory feedback)
        for gi in range(off_grc, off_grc + n_grc):
            if rng.rand() < 0.15:
                ti = rng.randint(off_goc, off_goc + n_goc)
                pairs.append((gi, ti))

        # GoC → GrC (inhibitory feedback)
        for gi in range(off_goc, off_goc + n_goc):
            nc = max(3, n_grc // 10)
            targets = rng.choice(range(off_grc, off_grc + n_grc), size=min(nc, n_grc), replace=False)
            for ti in targets:
                pairs.append((gi, ti))

        # GrC → PkC (parallel fibers)
        for gi in range(off_grc, off_grc + n_grc):
            if rng.rand() < 0.3:
                ti = rng.randint(off_pkc, off_pkc + n_pkc)
                pairs.append((gi, ti))

        # PkC → DCN (inhibitory)
        for pi in range(off_pkc, off_pkc + n_pkc):
            di = off_dcn + (pi - off_pkc) % n_dcn
            pairs.append((pi, di))

        # DCN → OUT (motor output)
        for di in range(off_dcn, off_dcn + n_dcn):
            nc = max(2, n_out // max(n_dcn, 1))
            targets = rng.choice(range(off_out, off_out + n_out), size=min(nc, n_out), replace=False)
            for ti in targets:
                pairs.append((di, ti))

        self.synapse_pairs = pairs
        self._ready = True


def render_brain_network(
    spike_raster: np.ndarray,
    width: int = 480,
    height: int = 400,
    n_display: int = 350,
    inhibitory_mask: Optional[np.ndarray] = None,
    state: Optional[BrainNetworkState] = None,
    rotation_speed: float = 0.4,
    brain_state: Optional[Dict] = None,
) -> Tuple[Image.Image, BrainNetworkState]:
    n_total = len(spike_raster) if isinstance(spike_raster, np.ndarray) and spike_raster.size > 0 else 0

    if state is None:
        state = BrainNetworkState(n_total, n_display)
    if not state._ready:
        state.init_layout()

    nd = state.n_display

    # Subsample spikes to n_display
    if n_total > nd:
        idx = np.linspace(0, n_total - 1, nd, dtype=int)
        spikes = spike_raster[idx].astype(bool)
    elif n_total > 0:
        spikes = np.zeros(nd, dtype=bool)
        spikes[:n_total] = spike_raster[:n_total].astype(bool)
    else:
        spikes = np.zeros(nd, dtype=bool)

    # Update fade (0=just fired, higher=resting)
    FADE_MAX = 8
    state.fade[spikes] = 0
    state.fade[~spikes] = np.minimum(state.fade[~spikes] + 1, FADE_MAX + 2)

    spike_rate = float(spikes.sum()) / max(nd, 1)
    state.activity_history.append(spike_rate)
    if len(state.activity_history) > 60:
        state.activity_history = state.activity_history[-60:]

    state.angle += rotation_speed
    state.pulse_phase += 0.07

    # ── 3D Transform ──
    a_y = math.radians(state.angle)
    cy, sy = math.cos(a_y), math.sin(a_y)
    a_x = math.radians(15.0)
    cx_r, sx_r = math.cos(a_x), math.sin(a_x)

    pts = state.positions_3d.copy()
    x2 = pts[:, 0] * cy + pts[:, 2] * sy
    z2 = -pts[:, 0] * sy + pts[:, 2] * cy
    pts[:, 0] = x2
    pts[:, 2] = z2
    y2 = pts[:, 1] * cx_r - pts[:, 2] * sx_r
    z3 = pts[:, 1] * sx_r + pts[:, 2] * cx_r
    pts[:, 1] = y2
    pts[:, 2] = z3

    mid_x, mid_y = width * 0.5, height * 0.48
    base_scale = min(width, height) * 0.38
    cam_dist = 3.5

    screen_x = np.zeros(nd, dtype=np.float32)
    screen_y = np.zeros(nd, dtype=np.float32)
    depth = pts[:, 2].copy()

    for i in range(nd):
        perspective = base_scale * cam_dist / (cam_dist + pts[i, 2])
        screen_x[i] = mid_x + pts[i, 0] * perspective
        screen_y[i] = mid_y - pts[i, 1] * perspective

    pulse = 0.5 + 0.5 * math.sin(state.pulse_phase)

    # ── Render ──
    synapse_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    neuron_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    glow_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    syn_d = ImageDraw.Draw(synapse_img)
    neu_d = ImageDraw.Draw(neuron_img)
    glow_d = ImageDraw.Draw(glow_img)

    # ── SYNAPSES ──
    for (src, dst) in state.synapse_pairs:
        x1, y1 = int(screen_x[src]), int(screen_y[src])
        x2, y2 = int(screen_x[dst]), int(screen_y[dst])

        avg_depth = (depth[src] + depth[dst]) * 0.5
        z_a = max(0.15, min(1.0, (avg_depth + 2.0) / 4.0))

        s_active = state.fade[src] <= 2
        d_active = state.fade[dst] <= 2

        if s_active and d_active:
            col = SYNAPSE_ACTIVE
            alpha = int(200 * z_a * (0.7 + 0.3 * pulse))
            w = 2
        elif s_active or d_active:
            col = COOL_BLUE
            alpha = int(100 * z_a)
            w = 1
        else:
            col = SYNAPSE_BASE
            alpha = int((35 + 10 * pulse) * z_a)
            w = 1

        if alpha > 3:
            syn_d.line([(x1, y1), (x2, y2)], fill=_rgba(col, alpha), width=w)

    # ── PkC DENDRITIC TREES ──
    pkc_calcium = None
    if brain_state and 'pkc_calcium' in brain_state:
        pkc_calcium = brain_state.get('pkc_calcium', 0.0)

    off_pkc = sum(state.layer_sizes[:3])
    n_pkc = state.layer_sizes[3]
    for pi in range(n_pkc):
        idx = off_pkc + pi
        sx, sy_pos = int(screen_x[idx]), int(screen_y[idx])
        z = depth[idx]
        z_fac = max(0.2, min(1.0, (z + 2.0) / 4.0))

        branch_h = int(28 * z_fac)
        spread = int(16 * z_fac)
        apical_alpha = int(60 * z_fac)
        if pkc_calcium and pkc_calcium > 0.05:
            ca_boost = min(1.0, pkc_calcium * 2)
            den_col = _lerp(DENDRITE_COLOR, (255, 80, 80), ca_boost)
            apical_alpha = int(apical_alpha + 80 * ca_boost)
        else:
            den_col = DENDRITE_COLOR

        syn_d.line([(sx, sy_pos), (sx, sy_pos - branch_h)],
                   fill=_rgba(den_col, apical_alpha), width=1)
        syn_d.line([(sx, sy_pos - branch_h), (sx - spread, sy_pos - branch_h - int(12 * z_fac))],
                   fill=_rgba(den_col, apical_alpha - 15), width=1)
        syn_d.line([(sx, sy_pos - branch_h), (sx + spread, sy_pos - branch_h - int(12 * z_fac))],
                   fill=_rgba(den_col, apical_alpha - 15), width=1)
        basal_h = int(10 * z_fac)
        syn_d.line([(sx, sy_pos), (sx, sy_pos + basal_h)],
                   fill=_rgba((200, 80, 80), int(50 * z_fac)), width=1)

    # ── NEURONS ──
    order = np.argsort(-depth)
    for idx in order:
        sx, sy_pos = int(screen_x[idx]), int(screen_y[idx])
        z = depth[idx]
        f = state.fade[idx]
        li = state.layer_idx[idx]
        z_fac = max(0.2, min(1.0, (z + 2.0) / 4.0))

        fire_col = LAYER_COLORS.get(li, OUTPUT_GOLD)
        if f == 0:
            col = fire_col
            r = int(7 * z_fac)
            gr = int(22 * z_fac)
            a = 255
        elif f <= 2:
            t = f / 2.0
            col = _lerp(fire_col, LAYER_COLORS.get(li, DEEP_BLUE), t)
            r = int(5 * z_fac)
            gr = int(12 * z_fac * (1 - t * 0.5))
            a = int(230 - 50 * t)
        elif f <= FADE_MAX:
            t = (f - 2) / max(FADE_MAX - 2, 1)
            col = _lerp(LAYER_COLORS[li], DEEP_BLUE, t * 0.5)
            r = int(4 * z_fac)
            gr = 0
            a = int(160 - 60 * t)
        else:
            col = _lerp(LAYER_COLORS[li], DEEP_SPACE, 0.4)
            r = max(2, int(3 * z_fac))
            gr = 0
            a = int(100 * z_fac)

        if gr > 2:
            glow_d.ellipse([sx - gr, sy_pos - gr, sx + gr, sy_pos + gr],
                           fill=_rgba(col, int(60 * z_fac)))

        if r >= 1:
            neu_d.ellipse([sx - r, sy_pos - r, sx + r, sy_pos + r],
                          fill=_rgba(col, a))

        if f <= 1 and r >= 3:
            cr = max(1, r - 2)
            neu_d.ellipse([sx - cr, sy_pos - cr, sx + cr, sy_pos + cr],
                          fill=_rgba((255, 255, 255), int(200 * z_fac)))

    # ── Composite ──
    syn_blur = synapse_img.filter(ImageFilter.GaussianBlur(radius=1))
    glow_blur = glow_img.filter(ImageFilter.GaussianBlur(radius=5))

    final = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    final = Image.alpha_composite(final, syn_blur)
    final = Image.alpha_composite(final, glow_blur)
    final = Image.alpha_composite(final, neuron_img)

    draw = ImageDraw.Draw(final)

    # ── Spike Rate sparkline ──
    if len(state.activity_history) > 2:
        sh = 18
        s_y = height - sh - 3
        s_x = 8
        s_w = width - 16
        mx = max(max(state.activity_history), 0.01)
        draw.rectangle([s_x - 1, s_y - 1, s_x + s_w + 1, s_y + sh + 1],
                       fill=_rgba(DEEP_SPACE, 140))
        points = []
        for i, v in enumerate(state.activity_history):
            px = s_x + int(i * s_w / max(len(state.activity_history) - 1, 1))
            py = s_y + sh - int((v / mx) * sh)
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=_rgba(CYAN_ACCENT, 180), width=1)
        try:
            from src.viz.overlay_base import get_font
            draw.text((s_x, s_y - 10), "SPIKE RATE",
                      fill=_rgba(ICE_BLUE, 120), font=get_font(7))
        except Exception:
            pass

    # ── Layer labels with neuron counts ──
    try:
        from src.viz.overlay_base import get_font
        lf = get_font(9, bold=True)
        sf = get_font(7)
        for li, lbl in enumerate(LAYER_LABELS):
            col = LAYER_COLORS[li]
            ly = int(height * 0.12 + li * 24)
            count = state.layer_sizes[li] if li < len(state.layer_sizes) else 0
            draw.text((5, ly), f"{lbl} {count}", fill=_rgba(col, 160), font=lf)
    except Exception:
        pass

    # To RGB
    bg = Image.new('RGB', (width, height), DEEP_SPACE)
    bg.paste(final, mask=final.split()[3])
    return bg, state


# ── Backward-compatible wrapper ──

def render_brain_sphere(
    spike_raster: np.ndarray,
    angle: float = 0.0,
    width: int = 350,
    height: int = 280,
    n_display: int = 200,
    inhibitory_mask: Optional[np.ndarray] = None,
    fade_state: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    if not hasattr(render_brain_sphere, '_state'):
        render_brain_sphere._state = None
    st = render_brain_sphere._state
    img, st = render_brain_network(
        spike_raster, width, height, n_display,
        inhibitory_mask, st, rotation_speed=0.0)
    if st:
        st.angle = angle
    render_brain_sphere._state = st
    return img, st.fade if st else fade_state
