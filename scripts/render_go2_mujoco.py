#!/usr/bin/env python3
"""
MH-FLOCKE — Go2 MuJoCo Video Renderer (YouTube Quality)
==========================================================
MuJoCo native renderer at 2560x1440 with full L-shaped dashboard overlay.

YouTube Quality Strategy (Issue #77):
  - Render at 1440p (YouTube gives VP9 codec at 1440p+, much better than AVC)
  - Pre-sharpening via FFmpeg unsharp filter (survives re-encoding)
  - Min font size 16px, DejaVu Sans Mono Bold (thick strokes survive compression)
  - High bitrate CRF 12 + maxrate 30M (YouTube re-encodes, so overshoot is fine)
  - Architecture info read from FLOG meta (universell, not hardcoded)

Usage:
    cd D:\\claude\\mhflocke
    py -3.11 scripts/render_go2_mujoco.py --flog creatures/go2/v034_.../training_log.bin
    py -3.11 scripts/render_go2_mujoco.py --flog ... --overlay --speed 2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if sys.platform != 'win32':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse, json, struct, subprocess, time, gc
import numpy as np
import mujoco
from PIL import Image

try:
    import msgpack
except ImportError:
    print("pip install msgpack"); sys.exit(1)

GO2_XML = os.path.join(os.path.dirname(__file__), '..', 'creatures', 'go2', 'scene_mhflocke.xml')
FLOG_MAGIC = b'FLOG'


# ═══════════════════════════════════════════════════════════
# WRITE FRAME — single function, proper cleanup
# ═══════════════════════════════════════════════════════════

def write_frame(pipe, img):
    """Write a PIL RGB image to ffmpeg pipe. Closes image after."""
    raw = img.tobytes()
    pipe.write(raw)
    img.close()
    del raw


# ═══════════════════════════════════════════════════════════
# EVENT OVERLAY
# ═══════════════════════════════════════════════════════════

class EventOverlay:
    def __init__(self, flog, speed, width=1920, height=1080):
        self.w = width
        self.h = height
        self.speed = speed
        self.events = []
        self._detect_events(flog)

    def _detect_events(self, flog):
        prev = {}
        dist_milestones = set()
        for sf in flog.stats_frames:
            step = sf.get('step', 0)
            dist = sf.get('max_distance', 0)
            beh = sf.get('behavior', '')
            falls = sf.get('falls', 0)
            cpg = sf.get('cpg_weight', 1.0)
            comp = sf.get('actor_competence', 0.0)
            scents = sf.get('scents_found', 0)
            pci = sf.get('pci', 0.0)
            for m in [1, 5, 10, 15, 20, 25, 30]:
                if dist >= m and m not in dist_milestones:
                    dist_milestones.add(m)
                    self.events.append((step, f'{m}m reached', 'distance'))
            prev_beh = prev.get('behavior', '')
            if beh and beh != prev_beh and prev_beh:
                nice = {'walk': 'Walking', 'trot': 'Trotting', 'alert': 'Alert',
                        'look_around': 'Looking around', 'sniff': 'Sniffing',
                        'rest': 'Resting', 'motor_babbling': 'Motor babbling'}
                self.events.append((step, nice.get(beh, beh.title()), 'behavior'))
            if falls > 0 and prev.get('falls', 0) == 0:
                self.events.append((step, 'First fall!', 'fall'))
            if cpg <= 0.5 and prev.get('cpg_weight', 1.0) > 0.5:
                self.events.append((step, 'SNN takes over', 'cpg'))
            if comp >= 0.5 and prev.get('actor_competence', 0.0) < 0.5:
                self.events.append((step, 'Competence > 50%', 'competence'))
            for m in [5, 10, 20]:
                if scents >= m and prev.get('scents_found', 0) < m:
                    self.events.append((step, f'{m} scents found', 'scent'))
            # Closed-Loop adaptation events (Issue #78)
            cl_adapt = sf.get('cl_adaptations', 0)
            prev_cl = prev.get('cl_adaptations', 0)
            if cl_adapt > prev_cl and cl_adapt > 0:
                cl_improve = sf.get('cl_consec_improve', 0)
                cl_fail = sf.get('cl_consec_fail', 0)
                vor_gain = sf.get('cl_vor_hip_gain', 0.0)
                if cl_improve > 0:
                    self.events.append((step, f'Brain adapts: improving (VOR={vor_gain:.2f})', 'brain'))
                elif cl_fail >= 3:
                    self.events.append((step, f'Brain adapts: exploring (VOR={vor_gain:.2f})', 'brain'))
            # Ball proximity events
            ball_dist = sf.get('ball_dist', -1)
            prev_bd = prev.get('ball_dist', -1)
            if ball_dist >= 0:
                if ball_dist < 1.0 and prev_bd >= 1.0:
                    self.events.append((step, f'Approaching! ({ball_dist:.1f}m)', 'ball'))
                if ball_dist < 0.15 and prev_bd >= 0.15:
                    self.events.append((step, '\u2b50 BALL CONTACT!', 'contact'))
                elif ball_dist < 0.5 and prev_bd >= 0.5:
                    self.events.append((step, f'Almost touching! ({ball_dist:.2f}m)', 'ball'))
                # TPE transition: negative = approaching
                tpe = sf.get('task_pe', 0)
                prev_tpe = prev.get('task_pe', 0)
                if tpe < -0.5 and prev_tpe >= -0.5:
                    self.events.append((step, 'Brain: ball getting closer', 'brain'))
            # Ball episode resets
            ep = sf.get('ball_episode', 0)
            prev_ep = prev.get('ball_episode', 0)
            if ep > prev_ep:
                self.events.append((step, f'Attempt #{ep} — try again!', 'episode'))
            prev = sf
        self.events.sort(key=lambda e: e[0])
        # Filter: min 500 steps apart, BUT contact events always pass
        filtered = []
        for ev in self.events:
            if ev[2] == 'contact':
                # Contact events always shown — remove any conflicting event
                if filtered and ev[0] - filtered[-1][0] < 500:
                    filtered[-1] = ev  # Replace nearby event with contact
                else:
                    filtered.append(ev)
            elif not filtered or ev[0] - filtered[-1][0] >= 500:
                filtered.append(ev)
        self.events = filtered
        print(f'  Events detected: {len(self.events)}')
        for s, t, c in self.events:
            print(f'    step {s:>6,}: {t} [{c}]')

    def render(self, pixels, step, video_time):
        """Render event overlay onto numpy frame. Returns numpy."""
        active = None
        for ev_step, ev_text, ev_cat in self.events:
            if ev_step <= step <= ev_step + 2000:
                steps_in = step - ev_step
                if steps_in < 200: alpha = steps_in / 200.0
                elif steps_in > 1600: alpha = (2000 - steps_in) / 400.0
                else: alpha = 1.0
                if alpha > 0.03:
                    active = (ev_text, ev_cat, min(1.0, alpha))
                    break
        if active is None:
            return pixels

        text, category, alpha = active
        from PIL import ImageDraw, ImageFont

        img = Image.fromarray(pixels)
        d = ImageDraw.Draw(img, 'RGBA')  # Draw RGBA on RGB — allows alpha shapes

        f_scale = max(1.0, self.w / 1920.0)
        # DejaVu Sans Mono Bold preferred — survives YouTube recompression
        f_event = ImageFont.load_default()
        for fp in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
                   'C:/Windows/Fonts/consolab.ttf']:
            try:
                f_event = ImageFont.truetype(fp, max(18, int(24 * f_scale)))
                break
            except (OSError, IOError):
                continue

        x, y = 20, self.h - 280
        cat_colors = {
            'distance': (50, 200, 100), 'behavior': (0, 200, 220),
            'fall': (255, 60, 40), 'cpg': (255, 200, 60),
            'competence': (185, 110, 240), 'scent': (160, 255, 80),
            'ball': (255, 120, 200), 'brain': (0, 220, 180),
            'episode': (255, 220, 60), 'contact': (80, 255, 80),
        }
        col = cat_colors.get(category, (200, 210, 220))
        a = int(255 * alpha)
        tw = int(len(text) * 13 * f_scale + 40 * f_scale)
        box_h = int(36 * f_scale)

        d.rounded_rectangle([(x, y), (x + tw, y + box_h)], radius=6,
                            fill=(10, 14, 24, int(220 * alpha)),
                            outline=(*col, int(120 * alpha)))
        d.rounded_rectangle([(x, y), (x + int(4 * f_scale), y + box_h)], radius=2,
                            fill=(*col, a))
        d.text((x + int(14 * f_scale), y + int(8 * f_scale)), text,
               fill=(*col, a), font=f_event)

        result = np.array(img)
        img.close()
        return result


def _load_font(size, bold=False):
    """Load font with DejaVu preference, Consolas fallback."""
    from PIL import ImageFont
    suffix = '-Bold' if bold else ''
    for fp in [f'/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
               f'C:/Windows/Fonts/consola{"b" if bold else ""}.ttf']:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_title_card(w, h, fps, dur, pipe, arch_info=None):
    """Render title card with architecture info from FLOG (universell)."""
    from PIL import ImageDraw, ImageFont
    n = int(fps * dur)
    sc = max(1.0, w / 1920.0)
    fb = _load_font(max(48, int(64 * sc)), bold=True)
    fm = _load_font(max(22, int(28 * sc)))
    fs = _load_font(max(16, int(20 * sc)))  # Slightly larger for YouTube survival
    fa = _load_font(max(14, int(16 * sc)))  # Architecture detail font
    ox = int(220 * sc)
    line_h = int(28 * sc)  # Slightly more spacing for readability
    ai = arch_info or {}
    task = ai.get('task', '')
    arch_desc = ai.get('description', 'SNN + CPG + Cerebellum')
    for i in range(n):
        t = i / n
        img = Image.new('RGB', (w, h), (6, 8, 14))
        d = ImageDraw.Draw(img)
        grid = int(40 * sc)
        for gx in range(0, w, grid):
            d.line([(gx, 0), (gx, h)], fill=(15, 20, 30), width=1)
        for gy in range(0, h, grid):
            d.line([(0, gy), (w, gy)], fill=(15, 20, 30), width=1)
        alpha = min(1.0, t * 3)
        a = lambda v: int(v * alpha)
        cy = h // 2
        d.text((w//2-ox, cy-int(90*sc)), 'MH-FLOCKE',
               fill=(0, a(200), a(220)), font=fb)
        d.text((w//2-ox, cy-int(15*sc)), 'Embodied AI \u00b7 Level 15 v0.4.2',
               fill=(a(160), a(195), a(235)), font=fm)
        # Task name (from FLOG)
        if task:
            d.text((w//2-ox, cy+int(20*sc)), f'\u201c{task}\u201d',
                   fill=(a(200), a(180), a(100)), font=fm)
        # Architecture (from FLOG, universell)
        lines = [
            f'Architecture: {arch_desc}',
            'Robot: Unitree Go2 \u00b7 MuJoCo Menagerie (BSD-3-Clause)',
            '',
            '\u00a9 2026 Marc Hesse \u00b7 mhflocke.com',
        ]
        for j, line in enumerate(lines):
            if line:
                col = (0, a(180), a(200)) if j == 0 else (a(120), a(130), a(150))
                font = fa if j > 0 else fs
                d.text((w//2-ox, cy+int(55*sc)+j*line_h), line, fill=col, font=font)
        lw = int((w - ox*2) * min(1.0, t * 2))
        if lw > 0:
            d.line([(w//2-ox, cy+int(44*sc)), (w//2-ox+lw, cy+int(44*sc))],
                   fill=(0, a(200), a(220)), width=max(2, int(2*sc)))
        write_frame(pipe, img)
    gc.collect()
    print(f'  Title card: {n} frames ({dur}s)')


def render_end_card(w, h, fps, dur, pipe, stats, arch_info=None):
    """Render end card with results + architecture (universell from FLOG)."""
    from PIL import ImageDraw, ImageFont
    n = int(fps * dur)
    sc = max(1.0, w / 1920.0)
    fb = _load_font(max(36, int(48 * sc)), bold=True)
    fm = _load_font(max(18, int(22 * sc)))
    fs = _load_font(max(16, int(18 * sc)))  # Larger for YouTube
    ox = int(200 * sc)
    line_h = int(28 * sc)
    ai = arch_info or {}
    for i in range(n):
        t = i / n
        img = Image.new('RGB', (w, h), (6, 8, 14))
        d = ImageDraw.Draw(img)
        grid = int(40 * sc)
        for gx in range(0, w, grid):
            d.line([(gx, 0), (gx, h)], fill=(15, 20, 30), width=1)
        for gy in range(0, h, grid):
            d.line([(0, gy), (w, gy)], fill=(15, 20, 30), width=1)
        alpha = min(1.0, t * 3)
        a = lambda v: int(v * alpha)
        cy = h // 2
        d.text((w//2-ox, cy-int(110*sc)), 'MH-FLOCKE \u2014 Go2',
               fill=(0, a(200), a(220)), font=fb)
        task = ai.get('task', '')
        if task:
            d.text((w//2-ox, cy-int(55*sc)), f'\u201c{task}\u201d',
                   fill=(a(200), a(180), a(100)), font=fm)
        d.line([(w//2-ox, cy-int(30*sc)), (w//2+ox, cy-int(30*sc))],
               fill=(0, a(200), a(220)), width=max(2, int(2*sc)))
        # Results
        lines = [
            f'Distance: {stats.get("max_distance", 0):.2f}m  \u00b7  Falls: {stats.get("falls", 0)}',
            f'CPG: {stats.get("cpg_weight", 0):.0%}  \u00b7  PCI: {stats.get("pci", 0):.4f}',
        ]
        # Ball stats (if ball scene)
        if ai.get('has_ball'):
            ball_min = stats.get('ball_dist_min', -1)
            ball_end = stats.get('ball_dist', -1)
            ball_contacts = stats.get('ball_contacts', 0)
            if ball_min >= 0:
                ball_line = f'Ball: min {ball_min:.3f}m'
                if ball_contacts > 0:
                    ball_line += f'  \u00b7  {ball_contacts} contact frames'
                lines.append(ball_line)
        # Architecture
        arch_desc = ai.get('description', '')
        if arch_desc:
            lines.append(f'Architecture: {arch_desc}')
        lines.extend(['', 'Paper: SSRN / aiXiv (MH-FLOCKE)',
                      'Web: mhflocke.com  \u00b7  \u00a9 2026 Marc Hesse'])
        y = cy - int(10 * sc)
        for line in lines:
            if line:
                d.text((w//2-ox, y), line,
                       fill=(a(140), a(155), a(175)), font=fs)
            y += line_h
        write_frame(pipe, img)
    gc.collect()
    print(f'  End card: {n} frames ({dur}s)')


class FLOGReader:
    def __init__(self, path):
        self.path = path
        self.meta = {}
        self.creature_frames = []
        self.stats_frames = []
        with open(path, 'rb') as f:
            assert f.read(4) == FLOG_MAGIC
            struct.unpack('<H', f.read(2))
            struct.unpack('<B', f.read(1))
            ml = struct.unpack('<I', f.read(4))[0]
            self.meta = json.loads(f.read(ml))
            while True:
                h = f.read(13)
                if len(h) < 13: break
                ts, ft, dl = struct.unpack('<dBI', h)
                p = f.read(dl)
                if len(p) < dl: break
                d = msgpack.unpackb(p, raw=False)
                if ft == 4: self.creature_frames.append(d)
                elif ft == 2: self.stats_frames.append(d)
    def get_qpos(self, idx): return np.array(self.creature_frames[idx]['pos'])
    def get_ball_pos(self, idx):
        bp = self.creature_frames[idx].get('ball_pos')
        return np.array(bp) if bp else None
    def get_qvel(self, idx): return np.array(self.creature_frames[idx]['vel'])
    def get_step(self, idx): return self.creature_frames[idx].get('step', idx * 10)
    def get_stats_at_step(self, step):
        if not self.stats_frames: return {}
        best = self.stats_frames[0]
        for sf in self.stats_frames:
            if sf.get('step', 0) <= step: best = sf
            else: break
        return best
    def __len__(self): return len(self.creature_frames)

    def get_architecture_info(self) -> dict:
        """Extract architecture info from FLOG meta + stats for renderer.
        Universell: reads what's available, shows only what was active."""
        arch = self.meta.get('architecture', {})
        info = {
            'task': self.meta.get('task', 'unknown'),
            'creature': self.meta.get('creature', 'unknown'),
            'learning': arch.get('learning', 'unknown'),
            'steering': arch.get('steering', 'unknown'),
            'adaptation': arch.get('adaptation', 'unknown'),
            'has_ball': False,
            'has_closed_loop': False,
            'has_cerebellum': False,
            'has_pe': False,
            'has_vor': False,
        }
        # Auto-detect from stats (for runs before CLI architecture profiles)
        for sf in self.stats_frames[:5]:
            if sf.get('ball_dist', -1) >= 0:
                info['has_ball'] = True
            if sf.get('cl_adaptations', 0) > 0:
                info['has_closed_loop'] = True
            if sf.get('cb_steer_correction', None) is not None:
                info['has_cerebellum'] = True
            if sf.get('vor_smoothed', None) is not None:
                info['has_vor'] = True
        # Detect PE from learning signal (if PE was active, correction > 0.0 even without reward)
        if info['learning'] == 'unknown':
            # Heuristic: if early stats have non-zero pf_pkc but no ball_approach_reward
            # PE was probably active
            for sf in self.stats_frames[:20]:
                if sf.get('ball_approach_reward', 0) == 0 and sf.get('pred_error', 0) > 0.01:
                    info['has_pe'] = True
                    break
        else:
            info['has_pe'] = info['learning'] in ('pe', 'hybrid')
        # Build description string
        parts = []
        if info['has_pe'] and info['learning'] == 'hybrid':
            ratio = arch.get('pe_ratio', 0.7)
            parts.append(f'Hybrid PE:{ratio:.0%}')
        elif info['has_pe']:
            parts.append('Free Energy (PE)')
        else:
            parts.append('R-STDP')
        if info['has_vor']:
            parts.append('VOR')
        if info['has_cerebellum']:
            parts.append('CB-Gain')
        if info['has_closed_loop']:
            parts.append('CL')
        if info['has_ball']:
            parts.append('Ball')
        info['description'] = ' + '.join(parts)
        return info


def main():
    p = argparse.ArgumentParser(description='MH-FLOCKE Go2 MuJoCo Renderer')
    p.add_argument('flog', nargs='?', default=None, help='FLOG path (positional or --flog)')
    p.add_argument('--flog', dest='flog_flag', default=None, help='FLOG path (flag form)')
    p.add_argument('--output', default=None)
    p.add_argument('--overlay', action='store_true', default=True)  # Default ON for ball videos
    p.add_argument('--no-overlay', dest='overlay', action='store_false')
    p.add_argument('--res', type=int, default=1440, choices=[720, 1080, 1440, 2160],
                   help='Resolution shortcut: 720/1080/1440/2160')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--short', action='store_true', default=False,
                   help='YouTube Short: 1080x1920 vertical, 60s max')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--speed', type=float, default=2.0)
    p.add_argument('--distance', type=float, default=3.5)
    p.add_argument('--azimuth', type=float, default=150)
    p.add_argument('--elevation', type=float, default=-20)
    args = p.parse_args()

    # Resolve FLOG path (positional or --flog flag)
    if args.flog is None and args.flog_flag:
        args.flog = args.flog_flag
    if args.flog is None:
        p.error('FLOG path required (positional or --flog)')

    # Resolution shortcuts
    RES_MAP = {720: (1280, 720), 1080: (1920, 1080), 1440: (2560, 1440), 2160: (3840, 2160)}
    if args.short:
        args.width = 1080
        args.height = 1920
    elif args.width is None or args.height is None:
        args.width, args.height = RES_MAP.get(args.res, (2560, 1440))

    if args.output is None:
        d = os.path.dirname(args.flog)
        tag = '_dash' if args.overlay else ''
        short_tag = '_short' if args.short else ''
        args.output = os.path.join(d, f'go2_mujoco{tag}{short_tag}.mp4')

    print(f'\n{"="*65}')
    print(f'  MH-FLOCKE \u2014 Go2 MuJoCo Video Renderer')
    print(f'{"="*65}')
    print(f'  FLOG: {args.flog}')
    print(f'  Output: {args.output}')
    mode = 'YouTube Short (vertical)' if args.short else f'YouTube {args.res}p'
    print(f'  {args.width}x{args.height} @ {args.fps}fps, {args.speed}x speed  [{mode}]')

    # Load FLOG
    flog = FLOGReader(args.flog)
    print(f'  {len(flog)} creature + {len(flog.stats_frames)} stats frames')

    # Timing
    ri = flog.meta.get('record_interval', 10)
    dt_val = flog.meta.get('dt', 0.005)
    if ri == 0:
        ri = (flog.get_step(1) - flog.get_step(0)) if len(flog) > 1 else 10
    sim_dt = ri * dt_val
    total_sim = len(flog) * sim_dt
    video_dur = total_sim / args.speed
    n_frames = int(video_dur * args.fps)
    print(f'  Sim: {total_sim:.1f}s -> Video: {video_dur:.1f}s ({n_frames} frames)')

    # Terrain
    terrain_type = None
    terrain_diff = 0.3
    knowledge_path = os.path.join(os.path.dirname(args.flog), 'knowledge.json')
    if os.path.exists(knowledge_path):
        with open(knowledge_path) as kf:
            kdata = json.load(kf)
            t_info = kdata.get('terrain', {})
            terrain_type = t_info.get('type', None)
            terrain_diff = t_info.get('difficulty', 0.3)
            print(f'  Terrain from knowledge.json: {terrain_type} (diff={terrain_diff})')
    if not terrain_type and flog.meta.get('terrain_type'):
        terrain_type = flog.meta['terrain_type']
        terrain_diff = flog.meta.get('terrain_difficulty', 0.3)

    # Load model (with terrain if applicable)
    if terrain_type and terrain_type not in ('flat', 'none', ''):
        print(f'  Injecting terrain: {terrain_type} (difficulty={terrain_diff})')
        with open(GO2_XML, 'r') as xf:
            xml_str = xf.read()
        try:
            from src.body.terrain import TerrainConfig, inject_terrain, inject_terrain_geoms
            # Read full terrain config from knowledge.json if available
            t_kwargs = dict(terrain_type=terrain_type, difficulty=terrain_diff, seed=42)
            if os.path.exists(knowledge_path):
                with open(knowledge_path) as kf2:
                    kd2 = json.load(kf2).get('terrain', {})
                    if 'max_height' in kd2: t_kwargs['max_height'] = kd2['max_height']
                    if 'size_x' in kd2: t_kwargs['size_x'] = kd2['size_x']
                    if 'size_y' in kd2: t_kwargs['size_y'] = kd2['size_y']
                    if 'resolution' in kd2: t_kwargs['resolution'] = kd2['resolution']
                    if 'seed' in kd2: t_kwargs['seed'] = kd2['seed']
            t_cfg = TerrainConfig(**t_kwargs)
            print(f'  TerrainConfig: {t_cfg.size_x}x{t_cfg.size_y}m, h={t_cfg.max_height}m')
            # Use heightfield or geoms based on knowledge.json
            terrain_mode = 'heightfield'  # default for existing runs
            if os.path.exists(knowledge_path):
                with open(knowledge_path) as kf3:
                    terrain_mode = json.load(kf3).get('terrain', {}).get('mode', 'heightfield')
            if terrain_mode == 'geoms':
                xml_str = inject_terrain_geoms(xml_str, t_cfg)
            else:
                hfield_path = os.path.abspath(
                    os.path.join(os.path.dirname(args.flog), '_render_terrain.png'))
                xml_str = inject_terrain(xml_str, t_cfg, hfield_path)
            go2_dir = os.path.dirname(os.path.abspath(GO2_XML))
            temp_xml = os.path.join(go2_dir, '_render_temp.xml')
            with open(temp_xml, 'w') as tf:
                tf.write(xml_str)
            model = mujoco.MjModel.from_xml_path(temp_xml)
            os.remove(temp_xml)
            print(f'  Terrain injected OK')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  Terrain failed: {e}, using flat')
            model = mujoco.MjModel.from_xml_path(GO2_XML)
    else:
        # Check if FLOG was a ball scene (ball_pos in creature frames)
        _has_ball = any(cf.get('ball_pos') for cf in flog.creature_frames[:10])
        if _has_ball:
            with open(GO2_XML, 'r') as xf:
                xml_str = xf.read()
            from src.body.terrain import inject_ball
            xml_str = inject_ball(xml_str, pos=(8.0, 1.0, 0.12))
            go2_dir = os.path.dirname(os.path.abspath(GO2_XML))
            temp_xml = os.path.join(go2_dir, '_render_ball_temp.xml')
            with open(temp_xml, 'w') as tf:
                tf.write(xml_str)
            model = mujoco.MjModel.from_xml_path(temp_xml)
            os.remove(temp_xml)
            print(f'  Ball: injected for rendering')
        else:
            model = mujoco.MjModel.from_xml_path(GO2_XML)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.lookat[:] = [0, 0, 0.28]

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    print(f'  MuJoCo: {model.nu} actuators, cam dist={args.distance} az={args.azimuth} el={args.elevation}')

    # Dashboard + Events
    dash = None
    if args.overlay:
        from src.viz.go2_dashboard import Go2DashboardOverlay
        dash = Go2DashboardOverlay(args.width, args.height)
    events = EventOverlay(flog, args.speed, args.width, args.height) if args.overlay else None

    # Single ffmpeg pipe — title + main + end all in one stream
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    # FFmpeg with pre-sharpening (Issue #77: YouTube re-encoding destroys fine detail)
    # unsharp=5:5:0.8 sharpens luma, 5:5:0.3 sharpens chroma (subtle)
    # This compensates for YouTube's re-encoding blur at 1440p.
    # CRF 12 is overkill but YouTube re-encodes anyway — source quality matters.
    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{args.width}x{args.height}', '-r', str(args.fps), '-i', '-',
         '-vf', 'unsharp=5:5:0.8:5:5:0.3',  # Pre-sharpen for YouTube survival
         '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
         '-profile:v', 'high', '-level', '4.2',  # 4.2 sufficient for 1440p
         '-b:v', '30M', '-maxrate', '40M', '-bufsize', '60M',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
         args.output],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract architecture info from FLOG (universell)
    arch_info = flog.get_architecture_info()
    print(f'  Architecture: {arch_info["description"]}')
    if arch_info['has_ball']:
        print(f'  Ball scene detected')

    # ── TITLE CARD ──
    render_title_card(args.width, args.height, args.fps, 3.0, ffmpeg.stdin, arch_info)

    # ── MAIN RENDER ──
    t0 = time.time()
    for vf in range(n_frames):
        t_sim = (vf / args.fps) * args.speed
        flog_f = t_sim / sim_dt
        i0 = max(0, min(int(flog_f), len(flog) - 1))
        i1 = min(i0 + 1, len(flog) - 1)
        alpha = flog_f - int(flog_f)

        qpos = flog.get_qpos(i0) * (1.0 - alpha) + flog.get_qpos(i1) * alpha
        qvel = flog.get_qvel(i0) * (1.0 - alpha) + flog.get_qvel(i1) * alpha

        nq = min(len(qpos), model.nq)
        nv = min(len(qvel), model.nv)
        data.qpos[:nq] = qpos[:nq]
        data.qvel[:nv] = qvel[:nv]
        # Set ball position from FLOG (if recorded)
        ball_pos = flog.get_ball_pos(i0)
        if ball_pos is not None:
            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
            ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_joint')
            if ball_jnt_id >= 0:
                qa = model.jnt_qposadr[ball_jnt_id]
                data.qpos[qa:qa+3] = ball_pos[:3]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam, scene_option=opt)
        pixels = renderer.render()

        if dash:
            step = flog.get_step(i0)
            stats = dict(flog.get_stats_at_step(step))
            # Live distance from qpos
            live_dist = float(np.sqrt(qpos[0]**2 + qpos[1]**2)) if len(qpos) > 1 else 0.0
            stats['current_distance'] = live_dist
            if live_dist > stats.get('max_distance', 0.0):
                stats['max_distance'] = live_dist
            creature_z = float(qpos[2]) if len(qpos) > 2 else 0.3
            if creature_z < 0.18:
                stats['is_fallen'] = 1
                if stats.get('falls', 0) == 0: stats['falls'] = 1
            cam_p = {'azimuth': args.azimuth, 'elevation': args.elevation,
                     'distance': args.distance,
                     'lookat': [float(qpos[0]), float(qpos[1]), float(qpos[2])]}
            pixels = dash.composite(pixels, stats, cam_p)

        if events:
            step = flog.get_step(i0)
            pixels = events.render(pixels, step, t_sim / args.speed)

        ffmpeg.stdin.write(pixels.tobytes())

        if vf % 100 == 0 and vf > 0:
            elapsed = time.time() - t0
            rate = vf / elapsed
            eta = (n_frames - vf) / rate if rate > 0 else 0
            step = flog.get_step(i0)
            s = flog.get_stats_at_step(step)
            print(f'  {vf}/{n_frames}  step={step}  dist={s.get("max_distance",0):.1f}m  '
                  f'cpg={s.get("cpg_weight",0):.0%}  beh={s.get("behavior","?")}  '
                  f'{rate:.1f}fps  ETA {eta:.0f}s')

        # Force cleanup every 100 frames
        if vf % 100 == 0:
            gc.collect()

    # ── END CARD ──
    final_stats = dict(flog.get_stats_at_step(flog.get_step(len(flog) - 1)))
    max_falls = max((sf.get('falls', 0) for sf in flog.stats_frames), default=0)
    max_dist = max((sf.get('max_distance', 0) for sf in flog.stats_frames), default=0)
    final_stats['max_distance'] = max(max_dist, final_stats.get('max_distance', 0))
    final_stats['falls'] = max_falls
    # Ball stats for end card — compute from creature frames (10-step resolution)
    if arch_info.get('has_ball'):
        # Stats-based (1000-step resolution)
        ball_dists = [sf.get('ball_dist', 99) for sf in flog.stats_frames if sf.get('ball_dist', -1) >= 0]
        if ball_dists:
            final_stats['ball_dist_min'] = min(ball_dists)
        # Creature-frame-based (10-step resolution) — more accurate
        _ball_init = np.array([1.5, 0.0])  # Stage 0 default
        # Try to detect ball position from scent data
        for sf in flog.stats_frames[:5]:
            bx = sf.get('scent_0_x', None)
            by = sf.get('scent_0_y', None)
            if bx is not None:
                _ball_init = np.array([bx, by])
                break
        _contacts = 0
        _cf_min = 99.0
        for cf in flog.creature_frames:
            _q = np.array(cf['pos'])
            _bd = float(np.linalg.norm(np.array([float(_q[0]), float(_q[1])]) - _ball_init))
            _cf_min = min(_cf_min, _bd)
            if _bd < 0.15:
                _contacts += 1
        if _cf_min < final_stats.get('ball_dist_min', 99):
            final_stats['ball_dist_min'] = _cf_min
        final_stats['ball_contacts'] = _contacts
    print(f'  End card: dist={final_stats["max_distance"]:.2f}m, falls={final_stats["falls"]}'
          + (f', ball_min={final_stats.get("ball_dist_min", -1):.3f}m, contacts={final_stats.get("ball_contacts", 0)}' if arch_info.get('has_ball') else ''))
    render_end_card(args.width, args.height, args.fps, 4.0, ffmpeg.stdin, final_stats, arch_info)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()

    elapsed = time.time() - t0
    size = os.path.getsize(args.output) / (1024 * 1024)
    print(f'\n{"="*65}')
    print(f'  Done! {args.output} ({size:.1f} MB)')
    print(f'  {video_dur + 7:.1f}s total, {n_frames} main frames')
    print(f'  Render: {elapsed:.0f}s ({n_frames/elapsed:.1f} fps)')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
