#!/usr/bin/env python3
"""
MH-FLOCKE — Freenove MuJoCo Video Renderer (with Dashboard)
==============================================================
Renders Freenove training FLOG with full dashboard overlay.
Based on render_go2_mujoco.py pipeline.

Usage:
    python scripts/render_freenove.py creatures/freenove/v034_.../training_log.bin
    python scripts/render_freenove.py creatures/freenove/v034_.../training_log.bin --speed 3
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if sys.platform != 'win32':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse, json, struct, subprocess, time, gc
import numpy as np
import mujoco
from PIL import Image, ImageDraw

try:
    import msgpack
except ImportError:
    print("pip install msgpack"); sys.exit(1)

FREENOVE_XML = os.path.join(os.path.dirname(__file__), '..', 'creatures', 'freenove', 'freenove.xml')
FLOG_MAGIC = b'FLOG'


def write_frame(pipe, img):
    raw = img.tobytes()
    pipe.write(raw)
    img.close()
    del raw


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


def _load_font(size, bold=False):
    from PIL import ImageFont
    suffix = '-Bold' if bold else ''
    for fp in [f'/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf',
               f'C:/Windows/Fonts/consola{"b" if bold else ""}.ttf']:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_title_card(w, h, fps, dur, pipe, meta=None):
    from PIL import ImageDraw
    meta = meta or {}
    version = meta.get('version', 'v0.4.3')
    scene = meta.get('task', meta.get('scene', 'walk on flat meadow'))
    n_neurons = meta.get('n_neurons', 232)
    total_steps = meta.get('steps', 20000)
    n = int(fps * dur)
    sc = max(1.0, w / 1920.0)
    fb = _load_font(max(48, int(64 * sc)), bold=True)
    fm = _load_font(max(22, int(28 * sc)))
    fs = _load_font(max(16, int(20 * sc)))
    fa = _load_font(max(14, int(16 * sc)))
    ox = int(220 * sc)
    line_h = int(28 * sc)
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
        d.text((w//2-ox, cy-int(15*sc)), f'Freenove Robot Dog \u00b7 Level 15 {version}',
               fill=(a(160), a(195), a(235)), font=fm)
        d.text((w//2-ox, cy+int(20*sc)), f'\u201c{scene}\u201d',
               fill=(a(200), a(180), a(100)), font=fm)
        lines = [
            'Architecture: Izhikevich SNN + CPG + R-STDP + Cerebellum',
            'Robot: Freenove FNK0050 \u00b7 100\u20ac Kit \u00b7 Raspberry Pi 4',
            f'{n_neurons} neurons \u00b7 12 actuators \u00b7 {total_steps:,} steps',
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


def render_end_card(w, h, fps, dur, pipe, stats):
    from PIL import ImageDraw
    n = int(fps * dur)
    sc = max(1.0, w / 1920.0)
    fb = _load_font(max(36, int(48 * sc)), bold=True)
    fm = _load_font(max(18, int(22 * sc)))
    fs = _load_font(max(16, int(18 * sc)))
    ox = int(200 * sc)
    line_h = int(28 * sc)
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
        d.text((w//2-ox, cy-int(110*sc)), 'MH-FLOCKE \u2014 Freenove',
               fill=(0, a(200), a(220)), font=fb)
        d.line([(w//2-ox, cy-int(55*sc)), (w//2+ox, cy-int(55*sc))],
               fill=(0, a(200), a(220)), width=max(2, int(2*sc)))
        lines = [
            f'Distance: {stats.get("max_distance", 0):.2f}m  \u00b7  Falls: {stats.get("falls", 0)}',
            f'Best upright streak: {stats.get("best_upright_streak", 0):,}',
            f'Actor competence: {stats.get("actor_competence", 0):.3f}  \u00b7  CPG: {stats.get("cpg_weight", 0):.0%}',
            f'Architecture: Izhikevich SNN + CPG + R-STDP + Cerebellum',
            '',
            'Paper: DOI 10.5281/zenodo.19336894',
            'GitHub: github.com/MarcHesse/mhflocke',
            '\u00a9 2026 Marc Hesse \u00b7 mhflocke.com',
        ]
        y = cy - int(30 * sc)
        for line in lines:
            if line:
                d.text((w//2-ox, y), line, fill=(a(140), a(155), a(175)), font=fs)
            y += line_h
        write_frame(pipe, img)
    gc.collect()
    print(f'  End card: {n} frames ({dur}s)')


def main():
    p = argparse.ArgumentParser(description='MH-FLOCKE Freenove Renderer')
    p.add_argument('flog', help='FLOG path')
    p.add_argument('--output', default=None)
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--speed', type=float, default=1.5)
    p.add_argument('--distance', type=float, default=0.6)
    p.add_argument('--azimuth', type=float, default=None,
                   help='Camera azimuth (default: auto, 30 for wall scenes, 150 for normal)')
    p.add_argument('--elevation', type=float, default=-15)
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.flog), 'freenove_render.mp4')

    print(f'\n{"="*60}')
    print(f'  MH-FLOCKE \u2014 Freenove Renderer (with Dashboard)')
    print(f'{"="*60}')
    print(f'  FLOG: {args.flog}')
    print(f'  Output: {args.output}')
    print(f'  {args.width}x{args.height} @ {args.fps}fps, {args.speed}x speed')

    flog = FLOGReader(args.flog)
    print(f'  {len(flog)} creature + {len(flog.stats_frames)} stats frames')

    ri = flog.meta.get('record_interval', 10)
    dt_val = flog.meta.get('dt', 0.005)
    if ri == 0:
        ri = (flog.get_step(1) - flog.get_step(0)) if len(flog) > 1 else 10
    sim_dt = ri * dt_val
    total_sim = len(flog) * sim_dt
    video_dur = total_sim / args.speed
    n_frames = int(video_dur * args.fps)
    print(f'  Sim: {total_sim:.1f}s -> Video: {video_dur:.1f}s ({n_frames} frames)')

    # --- Read knowledge.json from FLOG directory (scene context) ---
    knowledge = {}
    knowledge_path = os.path.join(os.path.dirname(args.flog), 'knowledge.json')
    if os.path.exists(knowledge_path):
        with open(knowledge_path) as kf:
            knowledge = json.load(kf)
        print(f'  Scene: "{knowledge.get("scene", "?")}"')

    # --- Detect wall/obstacle scene and inject into XML ---
    scene_text = knowledge.get('scene', flog.meta.get('task', ''))
    _scene_has_wall = any(w in scene_text.lower() for w in ['wall', 'wand', 'obstacle', 'hindernis', 'barrier'])

    xml_path = os.path.abspath(FREENOVE_XML)
    if _scene_has_wall:
        from src.body.terrain import inject_wall
        with open(xml_path) as xf:
            xml_string = xf.read()
        xml_string = inject_wall(xml_string, distance=0.8)
        model = mujoco.MjModel.from_xml_string(xml_string)
        print(f'  Wall: injected at x=0.8m (visible in render)')
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)

    # Auto-detect camera azimuth for wall scenes
    # Wall scenes: camera at 30° (front-right) so wall is visible ahead
    # Normal scenes: camera at 150° (rear-left, classic follow cam)
    if args.azimuth is None:
        if _scene_has_wall:
            args.azimuth = 30.0   # Front-right: see robot AND wall
            args.elevation = -25  # Slightly higher for overview
            args.distance = 0.8   # Wider to show wall context
        else:
            args.azimuth = 150.0  # Classic rear follow cam

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.lookat[:] = [0, 0, 0.08]
    print(f'  Camera: az={args.azimuth:.0f} el={args.elevation:.0f} dist={args.distance:.1f}'
          f'{" (wall-aware)" if _scene_has_wall else ""}')

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    print(f'  MuJoCo: {model.nq} qpos, {model.nu} actuators')
    print(f'  Rangefinder ray: HIDDEN')

    # Load dashboard overlay
    dash = None
    try:
        from src.viz.go2_dashboard import Go2DashboardOverlay
        # v0.4.3: Real population sizes from FLOG meta or defaults
        flog_pops = flog.meta.get('population_sizes', None)
        freenove_pops = flog_pops if flog_pops else {'n_input': 48, 'n_granule': 106, 'n_golgi': 18,
                         'n_purkinje': 24, 'n_dcn': 24, 'n_output': 12, 'n_total': 232}
        dash = Go2DashboardOverlay(args.width, args.height, population_sizes=freenove_pops)
        print(f'  Dashboard overlay: ON')
    except Exception as e:
        print(f'  Dashboard overlay: FAILED ({e})')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{args.width}x{args.height}', '-r', str(args.fps), '-i', '-',
         '-vf', 'unsharp=5:5:0.8:5:5:0.3',
         '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
         '-profile:v', 'high', '-level', '4.2',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
         args.output],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Title card
    render_title_card(args.width, args.height, args.fps, 3.0, ffmpeg.stdin, meta=flog.meta)

    # Main render
    t0 = time.time()
    stats = {}  # Initialize for minimap before first dashboard frame
    # Trail buffer for mini-map
    _trail = []
    _minimap_size = min(args.width // 5, args.height // 4)  # Smaller: 1/5 width
    _minimap_margin = 15
    _minimap_scale = 12.0  # pixels per meter
    _lights_found_history = []  # Track where lights were found
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
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam, scene_option=opt)
        pixels = renderer.render()

        # --- Stats + Dashboard (scent circles DISABLED, minimap shows them) ---
        if dash:
            step = flog.get_step(i0)
            stats = dict(flog.get_stats_at_step(step))
            creature_x, creature_y = float(qpos[0]), float(qpos[1])

            # Targets found text (drawn directly on pixels)
            _scent_img = Image.fromarray(pixels)
            _scent_draw = ImageDraw.Draw(_scent_img)
            sf_count = int(stats.get('scents_found', 0))
            _sf_font = _load_font(max(28, int(36 * max(1.0, args.width / 1920.0))), bold=True)
            _scent_draw.text((20, 90), f'Targets found: {sf_count}',
                             fill=(0, 255, 220), font=_sf_font)
            pixels = np.array(_scent_img)

            live_dist = float(np.sqrt(qpos[0]**2 + qpos[1]**2)) if len(qpos) > 1 else 0.0
            stats['current_distance'] = live_dist
            if live_dist > stats.get('max_distance', 0.0):
                stats['max_distance'] = live_dist
            stats['creature'] = flog.meta.get('creature', 'freenove').capitalize()
            stats['version'] = 'v0.4.8'
            stats['task'] = flog.meta.get('task', scene_text or 'flat')
            stats['total_steps'] = flog.meta.get('steps', 20000)
            cam_p = {'azimuth': args.azimuth, 'elevation': args.elevation,
                     'distance': args.distance,
                     'lookat': [float(qpos[0]), float(qpos[1]), float(qpos[2])]}
            pixels = dash.composite(pixels, stats, cam_p)

        # --- Mini-Map Overlay (top-right corner) ---
        creature_x = float(qpos[0])
        creature_y = float(qpos[1])
        _trail.append((creature_x, creature_y))
        step = flog.get_step(i0)
        stats_now = flog.get_stats_at_step(step) if not dash else stats

        mm = _minimap_size
        mm_img = Image.new('RGBA', (mm, mm), (10, 15, 10, 200))  # Semi-transparent dark
        mm_draw = ImageDraw.Draw(mm_img)

        # Draw border
        mm_draw.rectangle([0, 0, mm-1, mm-1], outline=(0, 180, 220, 255), width=2)

        # Center of minimap = robot position
        cx, cy_map = creature_x, creature_y
        s = _minimap_scale

        def w2p(wx, wy):
            """World coords to minimap pixel coords."""
            px = int(mm / 2 + (wx - cx) * s)
            py = int(mm / 2 - (wy - cy_map) * s)
            return px, py

        # Grid lines (1m spacing)
        grid_col = (30, 45, 30, 100)
        for gx in range(int(cx - mm / (2*s)) - 1, int(cx + mm / (2*s)) + 2):
            px, _ = w2p(gx, cy_map)
            if 0 < px < mm:
                mm_draw.line([(px, 0), (px, mm)], fill=grid_col, width=1)
        for gy in range(int(cy_map - mm / (2*s)) - 1, int(cy_map + mm / (2*s)) + 2):
            _, py = w2p(cx, gy)
            if 0 < py < mm:
                mm_draw.line([(0, py), (mm, py)], fill=grid_col, width=1)

        # Origin marker
        ox, oy = w2p(0, 0)
        if 5 < ox < mm-5 and 5 < oy < mm-5:
            mm_draw.ellipse([ox-3, oy-3, ox+3, oy+3], fill=(100, 100, 100, 200))

        # Trail
        trail_step = max(1, len(_trail) // 500)  # Limit points
        for ti in range(0, len(_trail) - 1, trail_step):
            p1 = w2p(_trail[ti][0], _trail[ti][1])
            p2 = w2p(_trail[min(ti + trail_step, len(_trail)-1)][0],
                     _trail[min(ti + trail_step, len(_trail)-1)][1])
            if (0 <= p1[0] < mm and 0 <= p1[1] < mm and
                0 <= p2[0] < mm and 0 <= p2[1] < mm):
                age = ti / max(1, len(_trail))
                c = int(80 + 120 * age)
                mm_draw.line([p1, p2], fill=(0, c, int(c*0.8), 200), width=2)

        # Light/waypoint position
        for si in range(4):
            lx = stats_now.get(f'scent_{si}_x', None)
            ly = stats_now.get(f'scent_{si}_y', None)
            if lx is not None and ly is not None:
                lpx, lpy = w2p(lx, ly)
                if 5 < lpx < mm-5 and 5 < lpy < mm-5:
                    # Glowing light circle — large and bright
                    for r in range(20, 6, -2):
                        alpha = int(40 + 30 * (20 - r) / 14)
                        mm_draw.ellipse([lpx-r, lpy-r, lpx+r, lpy+r],
                                        outline=(255, 240, 50, alpha), width=2)
                    mm_draw.ellipse([lpx-6, lpy-6, lpx+6, lpy+6],
                                    fill=(255, 255, 100, 255))
                    mm_draw.ellipse([lpx-3, lpy-3, lpx+3, lpy+3],
                                    fill=(255, 255, 255, 255))

        # Robot position (arrow showing heading)
        rpx, rpy = mm // 2, mm // 2  # Always centered
        heading_val = float(qpos[3:7].tolist()[0]) if len(qpos) > 6 else 0
        # Quaternion to yaw
        qw, qx_h, qy_h, qz_h = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        yaw = np.arctan2(2.0 * (qw * qz_h + qx_h * qy_h), 1.0 - 2.0 * (qy_h**2 + qz_h**2))
        arrow_len = 8
        ax = rpx + int(arrow_len * np.cos(yaw))
        ay = rpy - int(arrow_len * np.sin(yaw))
        mm_draw.ellipse([rpx-4, rpy-4, rpx+4, rpy+4], fill=(0, 255, 100, 255))
        mm_draw.line([(rpx, rpy), (ax, ay)], fill=(0, 255, 100, 255), width=2)

        # Label
        try:
            _mm_font = _load_font(11)
            sf_val = int(stats_now.get('scents_found', 0))
            mm_draw.text((5, mm - 16), f'sf:{sf_val}  sm:{stats_now.get("smell_strength", 0):.2f}',
                         fill=(0, 220, 200, 255), font=_mm_font)
        except:
            pass

        # Composite minimap onto main frame (bottom-left, above status bar)
        main_img = Image.fromarray(pixels)
        mm_x = 12  # Aligned with status bar edge
        mm_y = args.height - mm - 130  # Higher above the status bar
        main_img.paste(mm_img, (mm_x, mm_y), mm_img)  # Use alpha
        pixels = np.array(main_img)
        main_img.close()
        mm_img.close()

        ffmpeg.stdin.write(pixels.tobytes())

        if vf % 100 == 0 and vf > 0:
            elapsed = time.time() - t0
            rate = vf / elapsed
            eta = (n_frames - vf) / rate if rate > 0 else 0
            step = flog.get_step(i0)
            s = flog.get_stats_at_step(step)
            print(f'  {vf}/{n_frames}  step={step}  dist={s.get("max_distance",0):.1f}m  '
                  f'cpg={s.get("cpg_weight",0):.0%}  act={s.get("actor_competence",0):.3f}  '
                  f'{rate:.1f}fps  ETA {eta:.0f}s')

        if vf % 100 == 0:
            gc.collect()

    # End card
    final_stats = dict(flog.get_stats_at_step(flog.get_step(len(flog) - 1)))
    max_dist = max((sf.get('max_distance', 0) for sf in flog.stats_frames), default=0)
    final_stats['max_distance'] = max(max_dist, final_stats.get('max_distance', 0))
    final_stats['falls'] = max((sf.get('falls', 0) for sf in flog.stats_frames), default=0)
    render_end_card(args.width, args.height, args.fps, 4.0, ffmpeg.stdin, final_stats)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()

    elapsed = time.time() - t0
    size = os.path.getsize(args.output) / (1024 * 1024)
    print(f'\n{"="*60}')
    print(f'  Done! {args.output} ({size:.1f} MB)')
    print(f'  {n_frames} frames in {elapsed:.0f}s ({n_frames/elapsed:.1f} fps)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
