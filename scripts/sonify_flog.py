#!/usr/bin/env python3
"""
MH-FLOCKE — Data Sonification Engine v1.0
=============================================
Generates audio from FLOG training data. Every sound is driven by real metrics.

Audio layers (mixed together):
  1. SNN Crackle     — spike activity as electrical crackling noise
  2. Motor Hum       — joint velocities as low-frequency drone
  3. Heartbeat Pulse — CPG rhythm as deep bass pulse
  4. Cerebellum Tone — CF error as mid-frequency sine wave
  5. Scent Chime     — scents_found triggers bell sounds
  6. Fall Impact     — is_fallen triggers impact boom
  7. DA Melody       — dopamine reward as rising/falling tone
  8. Ambient Pad     — valence/arousal control warm pad texture
  9. Ball Proximity   — rising ping when approaching ball, pan follows direction
 10. Wall Proximity   — warning hum near wall, deep impact thud on collision

Output: WAV file matching video duration, then mux with FFmpeg.

Usage:
    py -3.11 scripts/sonify_flog.py --flog creatures/go2/v034_.../training_log.bin --speed 2
    py -3.11 scripts/sonify_flog.py --flog ... --speed 2 --mux go2_mujoco_dash.mp4

Dependencies:
    pip install numpy soundfile
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, json, struct, subprocess, time
import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

try:
    import msgpack
except ImportError:
    print("pip install msgpack"); sys.exit(1)

FLOG_MAGIC = b'FLOG'
SAMPLE_RATE = 44100


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

    def get_step(self, idx):
        return self.creature_frames[idx].get('step', idx * 10)

    def get_stats_at_step(self, step):
        if not self.stats_frames: return {}
        best = self.stats_frames[0]
        for sf_item in self.stats_frames:
            if sf_item.get('step', 0) <= step: best = sf_item
            else: break
        return best

    def __len__(self):
        return len(self.creature_frames)


# ═══════════════════════════════════════════════════════════
# SYNTH PRIMITIVES v2 — organic, warm, textured
# ═══════════════════════════════════════════════════════════

def _lp_filter(sig, cutoff_ratio=0.1):
    """Simple one-pole lowpass to smooth harsh digital edges."""
    a = cutoff_ratio
    out = np.zeros_like(sig)
    out[0] = sig[0]
    for i in range(1, len(sig)):
        out[i] = out[i-1] + a * (sig[i] - out[i-1])
    return out


def _saturate(sig, drive=1.5):
    """Soft tube-style saturation — adds warmth and harmonics."""
    return np.tanh(sig * drive) / drive


def sine(freq, duration, sr=SAMPLE_RATE, amp=0.3):
    """Warm sine with subtle harmonics and drift."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Slight frequency wobble (organic feel)
    wobble = 1.0 + 0.003 * np.sin(2 * np.pi * 0.7 * t)
    sig = np.sin(2 * np.pi * freq * wobble * t)
    # Add 2nd and 3rd harmonic for warmth
    sig += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
    sig += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
    return _saturate(amp * sig, 1.3)


def click(sr=SAMPLE_RATE, amp=0.15):
    """Neural spike — short filtered noise burst, not a pure sine."""
    n = int(sr * 0.004)  # 4ms, slightly longer
    noise = np.random.randn(n)
    # Bandpass feel: multiply by carrier then envelope
    t = np.linspace(0, 1, n)
    carrier = np.sin(2 * np.pi * (1200 + np.random.randint(-400, 400)) * t)
    env = np.exp(-t * 6) * (1 - np.exp(-t * 80))  # sharp attack, smooth decay
    return amp * noise * 0.4 * env + amp * carrier * 0.6 * env


def bass_pulse(freq, duration, sr=SAMPLE_RATE, amp=0.25):
    """Deep organic bass — sub-bass + chest thump."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    env = np.exp(-t * 2.5)  # slower decay, more body
    # Sub-bass fundamental
    sig = np.sin(2 * np.pi * freq * t) * 0.6
    # Click transient at attack
    click_env = np.exp(-t * 30)
    sig += np.sin(2 * np.pi * freq * 3 * t) * 0.3 * click_env
    # Subtle noise thump
    sig += np.random.randn(len(t)) * 0.1 * np.exp(-t * 15)
    return _saturate(amp * sig * env, 1.4)


def bell_chime(freq, duration=0.6, sr=SAMPLE_RATE, amp=0.2):
    """Warm bell — inharmonic partials + reverb tail."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    env = np.exp(-t * 3)  # slower decay, more resonance
    # Inharmonic partials (bell character)
    sig = np.sin(2 * np.pi * freq * t) * 0.4
    sig += np.sin(2 * np.pi * freq * 2.76 * t) * 0.25
    sig += np.sin(2 * np.pi * freq * 5.4 * t) * 0.1
    sig += np.sin(2 * np.pi * freq * 8.93 * t) * 0.05
    # Reverb simulation: delayed quiet copy
    reverb_n = int(sr * 0.08)  # 80ms delay
    if len(sig) > reverb_n:
        sig[reverb_n:] += sig[:-reverb_n] * 0.2
    return amp * sig * env


def impact_boom(duration=0.5, sr=SAMPLE_RATE, amp=0.4):
    """Realistic impact — layered sub-bass + crunch + rattle."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    env = np.exp(-t * 4)  # slower, more dramatic
    # Pitch-dropping sub
    freq = 100 * np.exp(-t * 2.5) + 30
    phase = np.cumsum(2 * np.pi * freq / sr)
    sub = np.sin(phase) * 0.5
    # Crunch (filtered noise burst)
    crunch = np.random.randn(len(t)) * np.exp(-t * 8) * 0.4
    crunch = _lp_filter(crunch, 0.15)  # remove harshness
    # Metal rattle (high partials, fast decay)
    rattle = np.sin(2 * np.pi * 340 * t) * np.exp(-t * 12) * 0.15
    rattle += np.sin(2 * np.pi * 570 * t) * np.exp(-t * 15) * 0.08
    sig = sub + crunch + rattle
    return _saturate(amp * sig * env, 1.5)


def pad_tone(freq, duration, sr=SAMPLE_RATE, amp=0.08):
    """Lush ambient pad — 7 detuned voices + slow filter sweep."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    # 7 voices with wider detuning for richness
    for detune in [-5, -3, -1, 0, 1, 3, 5]:
        f = freq * (1 + detune * 0.003)
        # Each voice has slightly different phase
        phase_offset = detune * 0.5
        sig += np.sin(2 * np.pi * f * t + phase_offset)
    sig /= 7
    # Slow amplitude modulation (breathing)
    sig *= (1 + 0.25 * np.sin(2 * np.pi * 0.15 * t))
    # Slow filter sweep (brighter in middle)
    sweep = 0.5 + 0.3 * np.sin(2 * np.pi * 0.08 * t)
    # Add subtle 2nd harmonic that follows sweep
    sig += 0.1 * np.sin(2 * np.pi * freq * 2 * t) * sweep
    return _saturate(amp * sig, 1.2)


def servo_hum(freq, duration, sr=SAMPLE_RATE, amp=0.05):
    """Electric servo/motor sound — buzzy, mechanical."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sawtooth-ish wave (harmonically rich)
    sig = np.zeros_like(t)
    for h in range(1, 8):
        sig += np.sin(2 * np.pi * freq * h * t) / h
    # Add mechanical resonance
    sig += 0.2 * np.sin(2 * np.pi * freq * 12.5 * t) * np.exp(-t * 2)
    # Filter to remove harshness
    sig = _lp_filter(sig, 0.2)
    return amp * sig


# ═══════════════════════════════════════════════════════════
# MUSICAL SCALE + CHORD SYSTEM
# ═══════════════════════════════════════════════════════════

# C minor pentatonic — always sounds good, cinematic, moody
# Notes: C, Eb, F, G, Bb across multiple octaves
SCALE_FREQS = [
    # Octave 2 (bass)
    65.41, 77.78, 87.31, 98.00, 116.54,
    # Octave 3 (low)
    130.81, 155.56, 174.61, 196.00, 233.08,
    # Octave 4 (mid)
    261.63, 311.13, 349.23, 392.00, 466.16,
    # Octave 5 (high)
    523.25, 622.25, 698.46, 783.99, 932.33,
]

# Chord progressions for different phases
# Each chord = indices into SCALE_FREQS
CHORD_PROG = {
    'calm':     [[5, 7, 9],    [6, 8, 10]],      # Cm, Fm (stable)
    'tension':  [[6, 8, 11],   [7, 9, 12]],       # rising tension
    'resolve':  [[5, 7, 10],   [5, 8, 10]],       # resolution
    'triumph':  [[5, 9, 12],   [7, 10, 14]],      # wide, open
}

def _pick_note(metric_val, octave_range=(5, 14)):
    """Map a 0-1 metric to a note from the scale."""
    lo, hi = octave_range
    idx = lo + int(metric_val * (hi - lo))
    return SCALE_FREQS[min(idx, len(SCALE_FREQS) - 1)]

def _pick_chord(phase):
    """Get chord frequencies for a training phase."""
    prog = CHORD_PROG.get(phase, CHORD_PROG['calm'])
    chord_idx = prog[0]  # could alternate with time
    return [SCALE_FREQS[min(i, len(SCALE_FREQS) - 1)] for i in chord_idx]

def _get_phase(step, total_steps=50000):
    """Determine training phase from step."""
    pct = step / total_steps
    if pct < 0.1: return 'calm'       # orientation
    if pct < 0.4: return 'tension'    # learning, falls
    if pct < 0.7: return 'resolve'    # recovery
    return 'triumph'                   # stable locomotion


# ═══════════════════════════════════════════════════════════
# SONIFICATION ENGINE v2 — musical, harmonic, data-driven
# ═══════════════════════════════════════════════════════════

class FLOGSonifier:
    def __init__(self, flog, video_duration, title_duration=3.0,
                 end_duration=4.0, sr=SAMPLE_RATE):
        self.flog = flog
        self.sr = sr
        self.video_dur = video_duration
        self.title_dur = title_duration
        self.end_dur = end_duration
        self.total_dur = title_duration + video_duration + end_duration
        self.n_samples = int(self.total_dur * sr)
        self.mix = np.zeros((self.n_samples, 2), dtype=np.float64)
        self._prev_scents = 0
        self._prev_fallen = 0

    def _write(self, buf_mono, start_sample, pan=0.5):
        n = len(buf_mono)
        end = min(start_sample + n, self.n_samples)
        actual = end - start_sample
        if actual <= 0: return
        self.mix[start_sample:end, 0] += buf_mono[:actual] * (1.0 - pan * 0.5)
        self.mix[start_sample:end, 1] += buf_mono[:actual] * (0.5 + pan * 0.5)

    def generate(self, speed):
        print(f'  Sonifying {self.total_dur:.1f}s audio...')
        ri = self.flog.meta.get('record_interval', 10)
        dt = self.flog.meta.get('dt', 0.005)
        if ri == 0 and len(self.flog) > 1:
            ri = self.flog.get_step(1) - self.flog.get_step(0)
        if ri == 0: ri = 10
        sim_dt = ri * dt
        title_samples = int(self.title_dur * self.sr)

        # Title pad — Cm chord, mysterious
        for f in _pick_chord('calm'):
            self._write(pad_tone(f, self.title_dur, self.sr, amp=0.03), 0, 0.5)

        chunk_dur = 1.0 / 30
        chunk_samples = int(chunk_dur * self.sr)
        n_chunks = int(self.video_dur / chunk_dur)

        for ci in range(n_chunks):
            t_video = ci * chunk_dur
            t_sim = t_video * speed
            flog_idx = max(0, min(int(t_sim / sim_dt), len(self.flog) - 1))
            step = self.flog.get_step(flog_idx)
            s = self.flog.get_stats_at_step(step)
            sample_pos = title_samples + int(t_video * self.sr)

            # 1. SNN Crackle — very subtle, background texture only
            snn_mix = s.get('snn_mix', 0.1)
            activity = max(0.01, snn_mix * 0.15)  # halved density
            for _ in range(np.random.poisson(activity * 6)):  # fewer clicks
                offset = np.random.randint(0, chunk_samples)
                self._write(click(self.sr, amp=0.015 * activity),  # much quieter
                            sample_pos + offset, np.random.uniform(0.3, 0.7))

            # 2. Servo Motor Hum — gentle background
            vel = s.get('vel_mps', 0.0)
            if vel > 0.05:  # higher threshold
                self._write(servo_hum(35 + vel * 15, chunk_dur, self.sr,
                                      amp=0.02 * min(1, vel)), sample_pos, 0.5)

            # 3. CPG Heartbeat — bass note from scale, rhythmic
            cpg = s.get('cpg_weight', 0.9)
            freq_scale = s.get('freq_scale', 1.0)
            if cpg > 0.1:
                pulse_phase = (t_video * freq_scale * 1.5) % 1.0
                if pulse_phase < 0.05:
                    # Bass note follows training phase
                    phase = _get_phase(step)
                    bass_note = SCALE_FREQS[0] if phase == 'calm' else (
                        SCALE_FREQS[1] if phase == 'tension' else (
                        SCALE_FREQS[2] if phase == 'resolve' else SCALE_FREQS[3]))
                    self._write(bass_pulse(bass_note, 0.2, self.sr, amp=0.10 * cpg),
                                sample_pos, 0.5)

            # 4. Cerebellum Tone — scale note based on error magnitude
            cf = s.get('cf_magnitude', 0.0)
            if cf > 0.05:
                cb_note = _pick_note(min(1, cf * 3), octave_range=(8, 14))
                self._write(sine(cb_note, chunk_dur, self.sr,
                                 amp=0.012 * min(1, cf * 3)), sample_pos, 0.6)

            # 5. Scent Chime — harmonious notes from scale
            scents = s.get('scents_found', 0)
            if scents > self._prev_scents:
                # Pick a high note from the scale, different each time
                chime_idx = (scents * 3) % 5 + 15  # high octave notes
                chime_freq = SCALE_FREQS[min(chime_idx, len(SCALE_FREQS) - 1)]
                self._write(bell_chime(chime_freq, 0.5, self.sr, amp=0.12),
                            sample_pos, 0.3 + (scents % 3) * 0.2)  # pan moves
                self._prev_scents = scents

            # 6. Fall Impact
            fallen = s.get('is_fallen', 0)
            if fallen and not self._prev_fallen:
                self._write(impact_boom(0.4, self.sr, amp=0.3), sample_pos, 0.5)
            self._prev_fallen = fallen

            # 7. DA Melody — pentatonic note, pitch follows reward
            da = s.get('da_reward', 0.0)
            if abs(da) > 0.12:
                # Positive DA = higher note, negative = lower
                da_mapped = (da + 1) / 2  # -1..1 -> 0..1
                da_note = _pick_note(da_mapped, octave_range=(10, 17))
                self._write(sine(da_note, chunk_dur, self.sr,
                                 amp=0.010 * min(1, abs(da))), sample_pos, 0.4)

            # 8. Ambient Pad — chord progression follows training phase
            if ci % 15 == 0:  # update every ~0.5s for smoother transitions
                phase = _get_phase(step)
                chord_freqs = _pick_chord(phase)
                # Slowly alternate between chord voicings
                chord_alt = int(t_video / 4.0) % 2
                prog = CHORD_PROG.get(phase, CHORD_PROG['calm'])
                chord_idx = prog[chord_alt % len(prog)]
                chord_freqs = [SCALE_FREQS[min(i, len(SCALE_FREQS)-1)] for i in chord_idx]
                pad_dur = chunk_dur * 15
                for fi, freq in enumerate(chord_freqs):
                    pan = 0.3 + fi * 0.2  # spread voices across stereo
                    self._write(pad_tone(freq, pad_dur, self.sr, amp=0.035),
                                sample_pos, pan)

            # 9. Ball Proximity Ping — rising tone when approaching ball (Issue #76d)
            # Biology: Auditory representation of visual salience.
            # Pitch rises as Go2 gets closer, panning follows ball direction.
            ball_dist = s.get('ball_dist', -1.0)
            ball_heading = s.get('ball_heading', 0.0)
            ball_approach = s.get('ball_approach_reward', 0.0)
            if ball_dist >= 0 and ball_dist < 15.0:
                # Proximity tone: closer = higher pitch, further = lower
                # Map distance 0-15m to note index 18 (high) to 5 (low)
                prox = max(0, 1.0 - ball_dist / 15.0)  # 1=touching, 0=15m away
                note_idx = int(5 + prox * 13)
                prox_freq = SCALE_FREQS[min(note_idx, len(SCALE_FREQS) - 1)]
                # Pan follows ball direction: -1(left) to +1(right) -> 0.0 to 1.0
                prox_pan = 0.5 + ball_heading * 0.4
                # Amplitude: louder when close
                prox_amp = 0.04 * prox + 0.005
                # Ping every ~1 second, faster when close
                ping_interval = max(3, int(15 - prox * 12))  # 15 chunks (far) to 3 chunks (close)
                if ci % ping_interval == 0:
                    self._write(bell_chime(prox_freq, 0.3, self.sr, amp=prox_amp),
                                sample_pos, prox_pan)
                # Ball contact: satisfying impact + bright chime
                if ball_dist < 0.15:
                    # Deep impact thud (ball hit)
                    impact = sine(80, 0.15, self.sr, amp=0.15)
                    impact *= np.exp(-np.linspace(0, 8, len(impact)))  # Fast decay
                    self._write(impact, sample_pos, 0.5)  # Center
                    # Bright achievement chime (two notes, octave apart)
                    chime1 = bell_chime(SCALE_FREQS[-1], 0.5, self.sr, amp=0.08)
                    chime2 = bell_chime(SCALE_FREQS[-1] * 2, 0.5, self.sr, amp=0.05)
                    self._write(chime1, sample_pos + int(0.05 * self.sr), 0.4)
                    self._write(chime2, sample_pos + int(0.10 * self.sr), 0.6)
                # Approach reward burst: bright high note on DA+
                elif ball_approach > 0.1:
                    burst_freq = SCALE_FREQS[min(17, len(SCALE_FREQS) - 1)]
                    self._write(sine(burst_freq, 0.1, self.sr, amp=0.06),
                                sample_pos, prox_pan)

            # 10. Wall/Obstacle Proximity — warning hum + collision impact (Issue #103)
            # Biology: Auditory representation of trigeminal/whisker danger signal.
            # Low rumble grows louder as robot approaches wall. Sharp impact on collision.
            obs_dist = s.get('obstacle_distance', -1.0)
            if obs_dist < 0:
                obs_dist = s.get('od', -1.0)
            if obs_dist >= 0 and obs_dist < 1.0:
                # Warning hum: lower pitch + louder when closer
                wall_prox = max(0, 1.0 - obs_dist)
                warn_freq = 60 + wall_prox * 40
                warn_amp = 0.02 + wall_prox * 0.08
                if ci % 3 == 0:
                    warn_dur = min(chunk_dur * 3, 0.15)
                    warn = sine(warn_freq, warn_dur, self.sr, amp=warn_amp)
                    warn += np.random.randn(len(warn)) * warn_amp * 0.3
                    warn *= np.exp(-np.linspace(0, 3, len(warn)))
                    self._write(warn, sample_pos, 0.5)
            # Wall collision: deep thud + metallic scrape
            if obs_dist >= 0 and obs_dist < 0.15:
                if not getattr(self, '_prev_wall_hit', False):
                    wall_thud = sine(45, 0.3, self.sr, amp=0.35)
                    wall_thud *= np.exp(-np.linspace(0, 6, len(wall_thud)))
                    scrape_n = int(0.2 * self.sr)
                    scrape = np.random.randn(scrape_n) * 0.08
                    scrape = _lp_filter(scrape, cutoff_ratio=0.15)
                    scrape *= np.exp(-np.linspace(0, 5, scrape_n))
                    self._write(wall_thud, sample_pos, 0.5)
                    self._write(scrape, sample_pos + int(0.05 * self.sr), 0.5)
                self._prev_wall_hit = True
            else:
                self._prev_wall_hit = False

        # End card -- resolving chord, fade to silence
        end_start = title_samples + int(self.video_dur * self.sr)
        for fi, freq in enumerate(_pick_chord('resolve')):
            ep = pad_tone(freq, self.end_dur, self.sr, amp=0.04)
            ep *= np.linspace(1, 0, len(ep))
            self._write(ep, end_start, 0.3 + fi * 0.2)

        # Normalize
        peak = np.max(np.abs(self.mix))
        if peak > 0.01:
            self.mix /= peak
            self.mix *= 0.85
        self.mix = np.tanh(self.mix * 1.2) / 1.2
        print(f'  Audio: {self.total_dur:.1f}s, peak={peak:.3f}')
        return self.mix

    def save(self, path):
        if not HAS_SF:
            print("  ERROR: pip install soundfile"); return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        sf.write(path, self.mix.astype(np.float32), self.sr)
        print(f'  Saved: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)')


def mux_audio_video(video_path, audio_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f'{base}_sound{ext}'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path,
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
           '-shortest', output_path]
    print(f'  Muxing -> {output_path}')
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(output_path):
        print(f'  Done: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)')
    return output_path


def main():
    p = argparse.ArgumentParser(description='MH-FLOCKE Data Sonification v1.0')
    p.add_argument('--flog', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--speed', type=float, default=2.0)
    p.add_argument('--title-dur', type=float, default=3.0)
    p.add_argument('--end-dur', type=float, default=4.0)
    p.add_argument('--mux', default=None, help='Mux with this video file')
    args = p.parse_args()

    flog = FLOGReader(args.flog)
    print(f'\n{"="*60}')
    print(f'  MH-FLOCKE — Data Sonification v1.0')
    print(f'{"="*60}')
    print(f'  {len(flog)} creature + {len(flog.stats_frames)} stats frames')

    ri = flog.meta.get('record_interval', 10)
    dt_val = flog.meta.get('dt', 0.005)
    if ri == 0 and len(flog) > 1:
        ri = flog.get_step(1) - flog.get_step(0)
    if ri == 0: ri = 10
    video_dur = len(flog) * ri * dt_val / args.speed

    sonifier = FLOGSonifier(flog, video_dur,
                             title_duration=args.title_dur,
                             end_duration=args.end_dur)
    sonifier.generate(args.speed)

    # ── PEDALBOARD POST-PROCESSING ──
    # Adds studio-quality reverb, warmth, compression to remove
    # the harsh digital/blechern quality from raw synthesis.
    try:
        from pedalboard import (Pedalboard, Reverb, Chorus, Compressor,
                                LadderFilter, Gain, Limiter)
        print('  Applying pedalboard mastering chain...')
        board = Pedalboard([
            # Warm lowpass — cut harsh highs
            LadderFilter(mode=LadderFilter.Mode.LPF24,
                         cutoff_hz=6000, resonance=0.1),
            # Subtle chorus for stereo width and organic feel
            Chorus(rate_hz=0.3, depth=0.2, mix=0.2,
                   centre_delay_ms=8.0),
            # Room reverb — gives space and depth
            Reverb(room_size=0.45, damping=0.7, wet_level=0.28,
                   dry_level=0.72, width=1.0),
            # Gentle compression — glues layers together
            Compressor(threshold_db=-18, ratio=3.0,
                       attack_ms=15, release_ms=200),
            # Final gain + limiter for consistent volume
            Gain(gain_db=3),
            Limiter(threshold_db=-1.0, release_ms=100),
        ])
        # pedalboard expects (channels, samples) float32
        audio = sonifier.mix.astype(np.float32).T  # -> (2, N)
        processed = board(audio, SAMPLE_RATE)
        sonifier.mix = processed.T  # -> (N, 2)
        print(f'  Pedalboard: LPF->Chorus->Reverb->Compressor->Limiter applied')
    except ImportError:
        print('  pedalboard not installed — skipping mastering.')
        print('  Install with: pip install pedalboard')
    except Exception as e:
        print(f'  pedalboard error: {e} — using raw mix')

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.flog), 'go2_sonification.wav')
    sonifier.save(args.output)

    if args.mux:
        mux_audio_video(args.mux, args.output)


if __name__ == '__main__':
    main()
