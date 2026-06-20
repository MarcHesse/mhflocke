#!/usr/bin/env python3
"""
MH-FLOCKE — Instagram Reel Renderer for BITTLE (9:16, 1080x1920)
================================================================
LINEAR single-pass playback of the whole FLOG (like render_bittle.py), so the
data sonification (sonify_flog.py) syncs perfectly — same convention as the main
video. Robot hero + hook text + behaviour caption + live neuron band + CPG->SNN bar,
then a CTA end card.

Clean look (collision hidden, headlight brightened). No title card (hook is on the
robot from frame 0); 4s CTA end card -> sonify with --title-dur 0 --end-dur 4.

Render:
    py -3.11 scripts/render_insta_reel_bittle.py creatures/bittle/v043_1781448987/training_log.bin --speed 5
Sound (SAME speed!):
    py -3.11 scripts/sonify_flog.py --flog creatures/bittle/v043_1781448987/training_log.bin \
        --speed 5 --title-dur 0 --end-dur 4 --mux creatures/bittle/v043_1781448987/training_log_insta_bittle.mp4
"""

__version__ = "0.1.0"
__logbook__ = 158

import sys, os, time, struct, json, subprocess, argparse, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

try:
    import msgpack
except ImportError:
    print("pip install msgpack"); sys.exit(1)

FLOG_MAGIC = b'FLOG'
OUT_W, OUT_H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM = 285, 1635
BITTLE_XML = os.path.join(os.path.dirname(__file__), '..', 'creatures', 'bittle', 'scene_mhflocke.xml')

CYAN = (0, 212, 255); GOLD = (255, 220, 60); WHITE = (235, 240, 245)
ICE = (160, 195, 235); GREEN = (60, 220, 110); BG = (8, 11, 16)
GREY = (95, 105, 115); ORANGE = (255, 165, 50); RED = (255, 80, 70); VIOLET = (185, 125, 255)


def _f(size, bold=False):
    suffix = 'b' if bold else ''
    for path in ['C:/Windows/Fonts/consola' + suffix + '.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSansMono' + ('-Bold' if bold else '') + '.ttf']:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _ctext(draw, cx, y, text, font, fill):
    w = draw.textlength(text, font=font)
    draw.text((cx - w / 2, y), text, fill=fill, font=font)


class FLOGReader:
    def __init__(self, path):
        self.meta = {}; self.creature_frames = []; self.stats_frames = []
        with open(path, 'rb') as f:
            assert f.read(4) == FLOG_MAGIC
            f.read(3); ml = struct.unpack('<I', f.read(4))[0]
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

    def get_qpos(self, idx): return np.array(self.creature_frames[min(idx, len(self)-1)]['pos'])
    def get_qvel(self, idx): return np.array(self.creature_frames[min(idx, len(self)-1)]['vel'])
    def get_step(self, idx): return self.creature_frames[min(idx, len(self)-1)].get('step', idx*10)
    def get_stats_at_step(self, step):
        best = {}
        for sf in self.stats_frames:
            if sf.get('step', 0) <= step: best = sf
            else: break
        return best
    def __len__(self): return len(self.creature_frames)


def setup_mujoco(render_w, render_h):
    model = mujoco.MjModel.from_xml_path(os.path.abspath(BITTLE_XML))
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, render_h, render_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if cam.trackbodyid < 0: cam.trackbodyid = 1
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    opt.geomgroup[1] = False
    return model, data, renderer, cam, opt


def render_3d(flog, flog_idx, model, data, renderer, cam, opt):
    i0 = max(0, min(int(flog_idx), len(flog)-1)); i1 = min(i0+1, len(flog)-1)
    a = max(0.0, min(1.0, flog_idx - int(flog_idx)))
    qpos = flog.get_qpos(i0)*(1-a) + flog.get_qpos(i1)*a
    qvel = flog.get_qvel(i0)*(1-a) + flog.get_qvel(i1)*a
    data.qpos[:min(len(qpos), model.nq)] = qpos[:min(len(qpos), model.nq)]
    data.qvel[:min(len(qvel), model.nv)] = qvel[:min(len(qvel), model.nv)]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render(), qpos


CAPTION = {'motor_babbling': 'learning to move', 'walk': 'walking',
           'sniff': 'curious — sniffing around', 'look_around': 'looking around',
           'investigate': 'investigating', 'alert': 'alert', 'chase': 'chasing'}


def main():
    p = argparse.ArgumentParser(description='MH-FLOCKE Instagram Reel (Bittle, linear, sonify-compatible)')
    p.add_argument('flog')
    p.add_argument('--output', default=None)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--speed', type=float, default=5.0, help='sim seconds per video second (MATCH in sonify)')
    p.add_argument('--elevation', type=float, default=-18)
    p.add_argument('--distance', type=float, default=0.55)
    p.add_argument('--azimuth', type=float, default=150)
    p.add_argument('--cta-dur', type=float, default=4.0, help='end card seconds (MATCH sonify --end-dur)')
    args = p.parse_args()

    print(f'Instagram Reel — Bittle ({OUT_W}x{OUT_H}), linear')
    flog = FLOGReader(args.flog)
    n_neurons = flog.meta.get('n_neurons', 535)
    sim_dt = flog.meta.get('sim_dt', 0.005) * flog.meta.get('log_interval', 10)
    total_sim = len(flog) * sim_dt
    body_dur = total_sim / args.speed
    print(f'  {len(flog)} frames, {n_neurons} neurons, body {body_dur:.1f}s + cta {args.cta_dur}s')
    if args.output is None:
        args.output = os.path.splitext(args.flog)[0] + '_insta_bittle.mp4'
    print(f'  Output: {args.output}')

    SCENE_W, SCENE_H = 1080, 810
    model, data, renderer, cam, opt = setup_mujoco(SCENE_W, SCENE_H)
    cam.distance = args.distance; cam.elevation = args.elevation; cam.azimuth = args.azimuth

    # optional neuron band
    brain_fn = None; brain_state = None
    try:
        from src.viz.brain_3d import render_brain_network, BrainNetworkState
        brain_fn = render_brain_network
        brain_state = BrainNetworkState(n_neurons=350, n_display=350)
        print('  Brain3D: loaded')
    except Exception as e:
        print(f'  Brain3D: unavailable ({e}) — band will be skipped')

    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{OUT_W}x{OUT_H}', '-r', str(args.fps), '-i', '-',
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.output],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    f_hook = _f(46, True); f_cap = _f(34, True); f_small = _f(20)
    f_big = _f(38, True); f_lbl = _f(20)
    ROBOT_Y = 150
    BRAIN_Y = 1040; BRAIN_W, BRAIN_H = 1000, 470
    BAR_Y = BRAIN_Y + BRAIN_H + 24

    t0 = time.time()
    n_body = int(args.fps * body_dur)
    for vf in range(n_body):
        t = vf / args.fps
        flog_idx = (t * args.speed) / sim_dt
        pixels, qpos = render_3d(flog, flog_idx, model, data, renderer, cam, opt)
        step = flog.get_step(max(0, min(int(flog_idx), len(flog)-1)))
        stats = flog.get_stats_at_step(step)

        scene = Image.fromarray(pixels).resize((OUT_W, int(SCENE_H * OUT_W / SCENE_W)), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (OUT_W, OUT_H), BG)
        canvas.paste(scene, (0, ROBOT_Y))
        d = ImageDraw.Draw(canvas); cx = OUT_W // 2

        # Hook (first 5s) then behaviour caption
        if t < 5.0:
            _ctext(d, cx, SAFE_TOP + 20, 'THIS ROBOT TAUGHT', f_hook, WHITE)
            _ctext(d, cx, SAFE_TOP + 72, 'ITSELF TO WALK', f_hook, WHITE)
            _ctext(d, cx, SAFE_TOP + 140, 'no external reward', f_cap, GOLD)
        else:
            beh = str(stats.get('behavior', ''))
            cap = CAPTION.get(beh, beh.replace('_', ' '))
            if cap:
                _ctext(d, cx, SAFE_TOP + 20, cap, f_cap, CYAN)

        # Neuron band — real spike raster from the FLOG only.
        n_spiking = None
        if brain_fn:
            try:
                raster_data = stats.get('spike_raster', stats.get('spikes', None))
                if raster_data is not None:
                    raster = np.asarray(raster_data, dtype=float)
                    n_spiking = int((raster > 0).sum())
                else:
                    # No real spikes in this frame -> show none, never fabricate.
                    raster = np.zeros(350, dtype=float)
                bimg, brain_state = brain_fn(raster, width=BRAIN_W, height=BRAIN_H,
                                             n_display=350, state=brain_state, brain_state=stats)
                brgba = bimg.convert('RGBA')
                canvas.paste(brgba, ((OUT_W - BRAIN_W)//2, BRAIN_Y), brgba)
                brgba.close(); bimg.close()
            except Exception as e:
                if vf == 0: print(f'  Brain3D error: {e}')
        _band_lbl = f'{n_spiking} neurons firing' if n_spiking is not None else f'{n_neurons} neurons'
        _ctext(d, cx, BRAIN_Y - 30, _band_lbl, f_lbl, ICE)

        # CPG -> SNN bar
        cpg = stats.get('cpg_weight', 0.9); snn = 1 - cpg
        d.text((90, BAR_Y), f'{int(cpg*100)}%', fill=ORANGE, font=f_big)
        d.text((205, BAR_Y + 6), 'CPG', fill=ICE, font=f_lbl)
        d.text((OUT_W - 235, BAR_Y), f'{int(snn*100)}%', fill=CYAN, font=f_big)
        d.text((OUT_W - 120, BAR_Y + 6), 'SNN', fill=ICE, font=f_lbl)
        bar_w = OUT_W - 180; cbar = int(bar_w * cpg); by = BAR_Y + 50
        d.rectangle([90, by, 90 + cbar, by + 14], fill=ORANGE)
        d.rectangle([90 + cbar, by, 90 + bar_w, by + 14], fill=CYAN)
        _ctext(d, cx, by + 24, 'the brain takes over', f_lbl, GREY)

        _ctext(d, cx, 18, 'MH-FLOCKE', f_small, GREY)
        _ctext(d, cx, OUT_H - 36, 'no external reward · open source · mhflocke.com', f_small, GREY)
        ffmpeg.stdin.write(np.array(canvas).tobytes())
        if vf % 60 == 0:
            print(f'  body {vf}/{n_body}  step:{step}  beh:{stats.get("behavior","")}  cpg:{cpg:.0%}')

    renderer.close(); gc.collect()

    # CTA end card
    n_cta = int(args.fps * args.cta_dur)
    f_b = _f(56, True); f_m = _f(34, True); f_s = _f(26)
    for vf in range(n_cta):
        canvas = Image.new('RGB', (OUT_W, OUT_H), BG); d = ImageDraw.Draw(canvas); cx = OUT_W // 2; cy = OUT_H // 2
        _ctext(d, cx, cy - 150, 'A robot dog that', f_m, ICE)
        _ctext(d, cx, cy - 104, 'taught itself to walk', f_m, ICE)
        _ctext(d, cx, cy - 14, 'NO EXTERNAL REWARD', f_b, GOLD)
        _ctext(d, cx, cy + 80, 'Full video on YouTube', f_s, CYAN)
        _ctext(d, cx, cy + 128, 'MH-FLOCKE · mhflocke.com', f_s, WHITE)
        ffmpeg.stdin.write(np.array(canvas).tobytes())

    ffmpeg.stdin.close(); ffmpeg.wait()
    print(f'\nDone: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB, {time.time()-t0:.0f}s)')
    print(f'Now add OUR sound (same speed!):')
    print(f'  py -3.11 scripts/sonify_flog.py --flog {args.flog} --speed {args.speed} '
          f'--title-dur 0 --end-dur {args.cta_dur} --mux {args.output}')


if __name__ == '__main__':
    main()
