"""
MH-FLOCKE — Unified Dashboard Views
======================================
Flask Blueprint that adds 4 views + WebSocket to the dashboard:

  /              → Brain (existing dashboard_v2.html, Knowledge Graph, 3D Neural)
  /training      → Training Live (distance, skill progress, narrator)
  /evolution     → CPG Evolution Live (fitness curves, parameters)
  /replay        → FLOG Replay (scrubber, timeline, events)

All views share a common tab bar and connect to the same WebSocket
for real-time updates.

WebSocket: ws://host:5001
  - Pushes training/evolution state every 500ms
  - Accepts commands (play/pause/seek for replay)

Usage:
    from src.viz.dashboard_views import views_bp, start_websocket
    app.register_blueprint(views_bp)
    start_websocket(app)
"""

__version__ = "1.0"      # module version (MAJOR.MINOR; MAJOR = contract change)
__logbook__ = 61         # mh-logbuch module entry
__status__  = "active"    # active | veraltet | neu

import os
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional

from flask import Blueprint, render_template_string, jsonify, request

# ═══════════════════════════════════════════════════════════
# BLUEPRINT
# ═══════════════════════════════════════════════════════════

views_bp = Blueprint('views', __name__)

# Shared state containers (set by mhflocke.py or dashboard_api.py)
_training_state: Dict = {}
_evolution_state: Dict = {}
_replay_engine = None  # ReplayEngine instance (set when /replay is used)

# WebSocket client set — use list wrapper to avoid UnboundLocalError in async
_ws_state = {'clients': set()}

def update_training_state(data: Dict):
    """Called by DashboardBridge in mhflocke.py."""
    global _training_state
    _training_state.update(data)

def update_evolution_state(data: Dict):
    """Called during CPG evolution."""
    global _evolution_state
    _evolution_state.update(data)

def set_replay_engine(engine):
    """Set replay engine for /replay view."""
    global _replay_engine
    _replay_engine = engine


# ═══════════════════════════════════════════════════════════
# TAB BAR (shared HTML)
# ═══════════════════════════════════════════════════════════

TAB_BAR = """
<div id="tab-bar">
  <a href="/" class="tab {brain_active}">🧠 Brain</a>
  <a href="/training" class="tab {training_active}">🏋 Training</a>
  <a href="/evolution" class="tab {evolution_active}">🧬 Evolution</a>
  <a href="/replay" class="tab {replay_active}">⏮ Replay</a>
  <span class="status" id="ws-status">⚫ disconnected</span>
</div>
"""

# ═══════════════════════════════════════════════════════════
# SHARED CSS + JS
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# NAV WRAPPER (tab bar + iframe)
# ═══════════════════════════════════════════════════════════

NAV_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>MH-FLOCKE Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }
  #tabs {
    background: #161b22; border-bottom: 1px solid #30363d;
    display: flex; align-items: center; height: 40px; padding: 0 12px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
  }
  .tab {
    color: #8b949e; text-decoration: none; padding: 8px 16px;
    border-bottom: 2px solid transparent; font-size: 12px;
    cursor: pointer; transition: all 0.15s;
  }
  .tab:hover { color: #e6edf3; background: #21262d; }
  .tab.active { color: #22d3ee; border-bottom-color: #22d3ee; }
  .brand { color: #22d3ee; font-size: 13px; font-weight: bold; margin-right: 16px; }
  #view { flex: 1; border: none; width: 100%; }
</style>
</head>
<body>
<div id="tabs">
  <span class="brand">MH-FLOCKE</span>
  <a class="tab" data-view="/">&#x1f9e0; Brain</a>
  <a class="tab" data-view="/training">&#x1f3cb; Training</a>
  <a class="tab" data-view="/evolution">&#x1f9ec; Evolution</a>
  <a class="tab" data-view="/replay">&#x23ee; Replay</a>
</div>
<iframe id="view" src="/"></iframe>
<script>
  const tabs = document.querySelectorAll('.tab');
  const iframe = document.getElementById('view');

  // Set active tab from URL hash
  function setView(path) {
    iframe.src = path;
    tabs.forEach(t => t.classList.toggle('active', t.dataset.view === path));
    location.hash = path;
  }

  tabs.forEach(t => t.addEventListener('click', (e) => {
    e.preventDefault();
    setView(t.dataset.view);
  }));

  // Restore from hash
  const hash = location.hash.replace('#', '') || '/';
  setView(hash);
</script>
</body></html>"""  # noqa: E501

SHARED_HEAD = r"""
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117; color: #e6edf3;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }
  #tab-bar {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 0 16px; display: flex; align-items: center;
    height: 40px; gap: 0;
  }
  .tab {
    color: #8b949e; text-decoration: none; padding: 8px 16px;
    border-bottom: 2px solid transparent; font-size: 12px;
    transition: all 0.2s;
  }
  .tab:hover { color: #e6edf3; background: #21262d; }
  .tab.active { color: #22d3ee; border-bottom-color: #22d3ee; }
  .status { margin-left: auto; font-size: 10px; color: #484f58; }
  .status.connected { color: #3fb950; }

  /* Cards */
  .card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px; margin: 8px;
  }
  .card h3 {
    color: #8b949e; font-size: 10px; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px;
  }
  .big-value { font-size: 28px; font-weight: bold; color: #22d3ee; }
  .sub { color: #484f58; font-size: 11px; margin-top: 2px; }
  .label { color: #8b949e; font-size: 11px; }
  .accent { color: #22d3ee; }
  .gold { color: #fbb436; }
  .green { color: #3fb950; }
  .red { color: #f85149; }

  /* Grid */
  .grid { display: grid; gap: 8px; padding: 8px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr 1fr; }
  .grid-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }

  /* Chart */
  .chart-container { position: relative; height: 200px; }
  .chart-container canvas { width: 100%; height: 100%; }

  /* Timeline / Scrubber */
  .scrubber {
    height: 48px; background: #0d1117; border: 1px solid #21262d;
    border-radius: 4px; position: relative; cursor: pointer; margin: 8px;
  }
  .scrubber canvas { width: 100%; height: 100%; }
  .playhead {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: #22d3ee; pointer-events: none;
  }

  /* Transport */
  .transport { display: flex; align-items: center; gap: 8px; padding: 4px 12px; }
  .transport button {
    background: #21262d; border: 1px solid #30363d; color: #e6edf3;
    border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 12px;
  }
  .transport button:hover { background: #30363d; }
  .transport button.active { background: #22d3ee; color: #0d1117; }

  /* Event list */
  .event { padding: 4px 8px; border-left: 3px solid #30363d; margin: 4px 0;
           font-size: 11px; cursor: pointer; border-radius: 0 4px 4px 0; }
  .event:hover { background: #21262d; }
  .event .etime { color: #22d3ee; font-size: 10px; }
  .event-skill_start { border-color: #3fb950; }
  .event-skill_freeze { border-color: #fbb436; }

  /* Narrator */
  .narrator {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 12px; margin: 8px; font-style: italic; color: #8b949e;
    min-height: 40px;
  }
</style>
<script>
// ── WebSocket (shared across all views) ────────────────
let ws = null;
let wsState = {};

function wsConnect() {
  const wsPort = parseInt(location.port) + 1;
  ws = new WebSocket('ws://' + location.hostname + ':' + wsPort);
  ws.onopen = () => {
    document.getElementById('ws-status').textContent = '🟢 connected';
    document.getElementById('ws-status').className = 'status connected';
    // Tell server which view we're on
    wsSend({action: 'subscribe', view: document.body.dataset.view || 'brain'});
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    wsState = msg;
    if (typeof onWsMessage === 'function') onWsMessage(msg);
  };
  ws.onclose = () => {
    document.getElementById('ws-status').textContent = '⚫ disconnected';
    document.getElementById('ws-status').className = 'status';
    setTimeout(wsConnect, 2000);
  };
}

function wsSend(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + sec.toString().padStart(2, '0');
}

document.addEventListener('DOMContentLoaded', wsConnect);
</script>
"""


# ═══════════════════════════════════════════════════════════
# VIEW: TRAINING
# ═══════════════════════════════════════════════════════════

TRAINING_HTML = r"""<!DOCTYPE html>
<html><head><title>MH-FLOCKE Training</title>
""" + SHARED_HEAD + r"""
</head>
<body data-view="training">
""" + TAB_BAR.format(brain_active='', training_active='active', evolution_active='', replay_active='') + r"""

<div class="grid grid-4">
  <div class="card">
    <h3>Distance</h3>
    <div class="big-value" id="v-distance">—</div>
    <div class="sub">Best: <span id="v-best">—</span></div>
  </div>
  <div class="card">
    <h3>Step</h3>
    <div class="big-value" id="v-step">—</div>
    <div class="sub">Phase: <span id="v-phase">—</span></div>
  </div>
  <div class="card">
    <h3>Skill</h3>
    <div class="big-value" id="v-skill">—</div>
    <div class="sub">Falls: <span id="v-falls" class="red">0</span></div>
  </div>
  <div class="card">
    <h3>Consciousness</h3>
    <div class="big-value" id="v-clevel">L0</div>
    <div class="sub">Emotion: <span id="v-emotion">—</span></div>
  </div>
</div>

<div class="card" style="margin:8px;">
  <h3>Distance Over Time</h3>
  <div class="chart-container"><canvas id="dist-chart"></canvas></div>
</div>

<div class="narrator" id="narrator">Waiting for training data...</div>

<script>
const distHistory = [];
const MAX_POINTS = 200;

function onWsMessage(msg) {
  if (msg.type !== 'training_update') return;
  const s = msg.stats || msg;

  document.getElementById('v-distance').textContent = (s.distance || s.best_distance || 0).toFixed(2) + 'm';
  document.getElementById('v-best').textContent = (s.best_episode || s.best_distance || 0).toFixed(2) + 'm';
  document.getElementById('v-step').textContent = msg.generation || msg.step || 0;
  document.getElementById('v-phase').textContent = s.curriculum_stage || s.phase || '—';
  document.getElementById('v-skill').textContent = s.skill || '—';
  document.getElementById('v-falls').textContent = s.falls || 0;
  document.getElementById('v-clevel').textContent = 'L' + (s.consciousness_level || 0);
  document.getElementById('v-emotion').textContent = s.emotion || 'neutral';
  if (msg.narrator) document.getElementById('narrator').textContent = msg.narrator;

  distHistory.push(parseFloat(s.best_episode || s.best_distance || 0));
  if (distHistory.length > MAX_POINTS) distHistory.shift();
  drawDistChart();
}

function drawDistChart() {
  const canvas = document.getElementById('dist-chart');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width; canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (distHistory.length < 2) return;

  const maxD = Math.max(...distHistory, 0.1);
  ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < distHistory.length; i++) {
    const x = (i / (distHistory.length - 1)) * canvas.width;
    const y = canvas.height - (distHistory[i] / maxD) * (canvas.height - 8) - 4;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════
# VIEW: EVOLUTION
# ═══════════════════════════════════════════════════════════

EVOLUTION_HTML = r"""<!DOCTYPE html>
<html><head><title>MH-FLOCKE Evolution</title>
""" + SHARED_HEAD + r"""
</head>
<body data-view="evolution">
""" + TAB_BAR.format(brain_active='', training_active='', evolution_active='active', replay_active='') + r"""

<div class="grid grid-3">
  <div class="card">
    <h3>Generation</h3>
    <div class="big-value" id="v-gen">—</div>
    <div class="sub">of <span id="v-total-gen">—</span></div>
  </div>
  <div class="card">
    <h3>Best Fitness</h3>
    <div class="big-value gold" id="v-fitness">—</div>
    <div class="sub">Avg: <span id="v-avg-fit">—</span></div>
  </div>
  <div class="card">
    <h3>Best Distance</h3>
    <div class="big-value green" id="v-dist">—</div>
    <div class="sub">Skill: <span id="v-skill">—</span></div>
  </div>
</div>

<div class="card" style="margin:8px;">
  <h3>Fitness Curve</h3>
  <div class="chart-container"><canvas id="fit-chart"></canvas></div>
</div>

<div class="card" style="margin:8px;">
  <h3>CPG Parameters (Best Genome)</h3>
  <div id="cpg-params" class="sub">Waiting for evolution data...</div>
</div>

<script>
const fitHistory = [], avgHistory = [];
const MAX_PTS = 100;

function onWsMessage(msg) {
  if (msg.type !== 'training_update') return;
  const s = msg.stats || msg;
  if ((s.curriculum_stage || s.phase || '') !== 'CPG EVOLUTION' && !msg.best_fitness) return;

  document.getElementById('v-gen').textContent = msg.generation || 0;
  document.getElementById('v-fitness').textContent = (msg.best_fitness || s.best_fitness || 0).toFixed(3);
  document.getElementById('v-avg-fit').textContent = (msg.mean_fitness || s.avg_fitness || 0).toFixed(3);
  document.getElementById('v-dist').textContent = (s.best_distance || 0).toFixed(2) + 'm';
  document.getElementById('v-skill').textContent = s.skill || '—';

  fitHistory.push(parseFloat(msg.best_fitness || s.best_fitness || 0));
  avgHistory.push(parseFloat(msg.mean_fitness || s.avg_fitness || 0));
  if (fitHistory.length > MAX_PTS) { fitHistory.shift(); avgHistory.shift(); }
  drawFitChart();
}

function drawFitChart() {
  const canvas = document.getElementById('fit-chart');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width; canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (fitHistory.length < 2) return;

  const maxF = Math.max(...fitHistory, 0.1);

  // Best fitness (gold)
  ctx.strokeStyle = '#fbb436'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < fitHistory.length; i++) {
    const x = (i / (fitHistory.length - 1)) * canvas.width;
    const y = canvas.height - (fitHistory[i] / maxF) * (canvas.height - 8) - 4;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Avg fitness (dim)
  ctx.strokeStyle = 'rgba(139,148,158,0.4)'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < avgHistory.length; i++) {
    const x = (i / (avgHistory.length - 1)) * canvas.width;
    const y = canvas.height - (avgHistory[i] / maxF) * (canvas.height - 8) - 4;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════
# VIEW: REPLAY
# ═══════════════════════════════════════════════════════════

REPLAY_HTML = r"""<!DOCTYPE html>
<html><head><title>MH-FLOCKE Replay</title>
""" + SHARED_HEAD + r"""
</head>
<body data-view="replay">
""" + TAB_BAR.format(brain_active='', training_active='', evolution_active='', replay_active='active') + r"""

<div class="grid grid-4">
  <div class="card">
    <h3>Distance</h3>
    <div class="big-value" id="v-distance">0.00m</div>
    <div class="sub">Best: <span id="v-best">0.00m</span></div>
  </div>
  <div class="card">
    <h3>Consciousness</h3>
    <div class="big-value" id="v-clevel">L0</div>
    <div class="sub" id="v-emotion">neutral</div>
  </div>
  <div class="card">
    <h3>Skill</h3>
    <div class="big-value" id="v-skill">—</div>
    <div class="sub" id="v-phase">—</div>
  </div>
  <div class="card">
    <h3>Training</h3>
    <div class="sub">Step: <span id="v-step">0</span></div>
    <div class="sub">Falls: <span id="v-falls">0</span></div>
    <div class="sub">Reward: <span id="v-reward">0</span></div>
  </div>
</div>

<div class="card" style="margin:8px;">
  <h3>Timeline</h3>
  <div class="scrubber" id="scrubber" onclick="scrubberClick(event)">
    <canvas id="scrub-canvas"></canvas>
    <div class="playhead" id="playhead" style="left:0"></div>
  </div>
  <div class="transport">
    <button onclick="wsSend({action:'restart'})">⏮</button>
    <button id="btn-play" onclick="wsSend({action:'toggle'})">▶</button>
    <button onclick="wsSend({action:'speed',value:0.5})">½×</button>
    <button onclick="wsSend({action:'speed',value:1})">1×</button>
    <button onclick="wsSend({action:'speed',value:5})">5×</button>
    <button onclick="wsSend({action:'speed',value:20})">20×</button>
    <span class="accent" id="time-display">0:00 / 0:00</span>
    <span class="label" id="speed-display">1.0×</span>
  </div>
</div>

<div style="display:flex; gap:0;">
  <div style="flex:1;">
    <div class="card">
      <h3>Events</h3>
      <div id="event-list" style="max-height:200px; overflow-y:auto;"></div>
    </div>
  </div>
  <div style="flex:1;">
    <div class="card">
      <h3>FLOG Info</h3>
      <div id="flog-info" class="sub">Load a FLOG file to start replay.</div>
    </div>
    <div class="card">
      <h3>Load FLOG</h3>
      <div class="sub">
        <input type="text" id="flog-path" placeholder="creatures/mogli/v003_.../training_log.bin"
               style="width:100%; background:#0d1117; border:1px solid #21262d; color:#e6edf3;
                      padding:6px; border-radius:4px; font-family:inherit; margin-bottom:6px;">
        <button onclick="loadFlog()" style="width:100%;">Load</button>
      </div>
    </div>
  </div>
</div>

<script>
let replayTimeline = [];
let replayEvents = [];
let replayDuration = 0;

function onWsMessage(msg) {
  if (msg.type === 'replay_init') {
    replayTimeline = msg.timeline || [];
    replayEvents = msg.events || [];
    replayDuration = msg.summary ? msg.summary.duration : 0;
    document.getElementById('flog-info').innerHTML =
      'Creature: ' + (msg.summary.creature || '?') + '<br>' +
      'Task: ' + (msg.summary.task || '?') + '<br>' +
      'Duration: ' + replayDuration.toFixed(0) + 's<br>' +
      'Physics: ' + (msg.summary.n_physics || 0) + ' frames<br>' +
      'Stats: ' + (msg.summary.n_stats || 0) + ' frames';
    renderEvents();
    drawScrubber();
    return;
  }

  if (msg.type === 'replay_state' || msg.type === 'state') {
    document.getElementById('v-distance').textContent = (msg.distance || 0).toFixed(2) + 'm';
    document.getElementById('v-best').textContent = (msg.best_episode || 0).toFixed(2) + 'm';
    document.getElementById('v-step').textContent = msg.step || 0;
    document.getElementById('v-clevel').textContent = 'L' + (msg.consciousness_level || 0);
    document.getElementById('v-emotion').textContent = msg.emotion || 'neutral';
    document.getElementById('v-skill').textContent = msg.skill || '—';
    document.getElementById('v-phase').textContent = msg.phase || '—';
    document.getElementById('v-falls').textContent = msg.falls || 0;
    document.getElementById('v-reward').textContent = (msg.reward || 0).toFixed(3);
    document.getElementById('time-display').textContent =
      fmtTime(msg.time || 0) + ' / ' + fmtTime(msg.duration || replayDuration);
    document.getElementById('speed-display').textContent = (msg.speed || 1).toFixed(1) + '×';
    document.getElementById('btn-play').textContent = msg.playing ? '⏸' : '▶';

    // Playhead
    const pct = (msg.duration || replayDuration) > 0
      ? ((msg.time || 0) / (msg.duration || replayDuration)) * 100 : 0;
    document.getElementById('playhead').style.left = pct + '%';
  }
}

function renderEvents() {
  const el = document.getElementById('event-list');
  el.innerHTML = '';
  for (const ev of replayEvents) {
    const div = document.createElement('div');
    div.className = 'event event-' + (ev.type || 'unknown');
    div.innerHTML = '<span class="etime">' + (ev.timestamp || 0).toFixed(1) + 's</span> ' +
                    (ev.type || '?') + ': ' + (ev.msg || '');
    div.onclick = () => wsSend({action:'seek', time: ev.timestamp});
    el.appendChild(div);
  }
}

function drawScrubber() {
  const canvas = document.getElementById('scrub-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width; canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  if (!replayTimeline.length) return;

  const maxD = Math.max(...replayTimeline.map(t => t.b || 0), 0.1);
  ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < replayTimeline.length; i++) {
    const x = (i / (replayTimeline.length - 1)) * canvas.width;
    const y = canvas.height - ((replayTimeline[i].b || 0) / maxD) * (canvas.height - 4);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Event markers
  for (const ev of replayEvents) {
    const x = (ev.timestamp / replayDuration) * canvas.width;
    ctx.strokeStyle = ev.type === 'skill_start' ? '#3fb950' :
                      ev.type === 'skill_freeze' ? '#fbb436' : '#484f58';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
}

function scrubberClick(e) {
  if (!replayDuration) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  wsSend({action: 'seek', time: pct * replayDuration});
}

function loadFlog() {
  const path = document.getElementById('flog-path').value.trim();
  if (path) wsSend({action: 'load_flog', path: path});
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); wsSend({action: 'toggle'}); }
  if (e.key === '1') wsSend({action: 'speed', value: 1});
  if (e.key === '2') wsSend({action: 'speed', value: 2});
  if (e.key === '5') wsSend({action: 'speed', value: 5});
});
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@views_bp.route('/nav')
def view_nav():
    """Navigation wrapper — tab bar + iframe for any view."""
    return NAV_HTML

@views_bp.route('/training')
def view_training():
    return TRAINING_HTML

@views_bp.route('/evolution')
def view_evolution():
    return EVOLUTION_HTML

@views_bp.route('/replay')
def view_replay():
    return REPLAY_HTML

# API endpoints for views
@views_bp.route('/api/views/training/state')
def api_training_state():
    return jsonify(_training_state)

@views_bp.route('/api/views/evolution/state')
def api_evolution_state():
    return jsonify(_evolution_state)

@views_bp.route('/api/views/replay/load', methods=['POST'])
def api_replay_load():
    """Load a FLOG file for replay."""
    data = request.get_json() or {}
    flog_path = data.get('path', '')
    if not flog_path or not os.path.exists(flog_path):
        return jsonify({'error': f'FLOG not found: {flog_path}'}), 404

    try:
        from src.viz.flog_replay import FlogReplay
        replay = FlogReplay(flog_path)
        # Store for WebSocket replay
        from src.viz.dashboard_views import _replay_data
        _replay_data['replay'] = replay
        return jsonify({'ok': True, 'summary': {
            'creature': replay.creature,
            'task': replay.task,
            'duration': replay.duration_s,
            'n_physics': replay.n_physics,
            'n_stats': replay.n_stats,
            'n_events': replay.n_events,
        }})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# WEBSOCKET SERVER
# ═══════════════════════════════════════════════════════════

_replay_data = {'replay': None, 'playing': False, 'time': 0.0, 'speed': 1.0}

def start_websocket(host='0.0.0.0', ws_port=5001):
    """Start WebSocket server in background thread."""
    try:
        import websockets
    except ImportError:
        print("  ⚠ websockets not installed. pip install websockets")
        return

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def handler(websocket, path=None):  # path param for older websockets
            _ws_state['clients'].add(websocket)
            print(f"  \U0001f310 WS client connected ({len(_ws_state['clients'])} total)")
            try:
                async for message in websocket:
                    try:
                        print(f"  \U0001f4e8 WS recv: {message[:200]}")
                        _handle_ws_command(message, websocket)
                    except Exception as e:
                        print(f"  \u26a0 WS command error: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"  \u26a0 WS handler error: {e}")
            finally:
                _ws_state['clients'].discard(websocket)
                print(f"  \U0001f310 WS client disconnected ({len(_ws_state['clients'])} total)")

        async def broadcaster():
            t_last = time.time()
            while True:
                if _ws_state['clients']:
                    now = time.time()
                    dt = now - t_last
                    t_last = now

                    # Replay tick
                    if _replay_data.get('playing') and _replay_data.get('replay'):
                        _replay_data['time'] += dt * _replay_data.get('speed', 1.0)
                        dur = _replay_data['replay'].duration_s
                        if _replay_data['time'] >= dur:
                            _replay_data['time'] = dur
                            _replay_data['playing'] = False

                    # Send pending init (after FLOG load)
                    pending = _replay_data.pop('pending_init', None)

                    # Build state message
                    msg = _build_ws_state()
                    payload = json.dumps(msg, default=str)
                    dead = set()
                    for ws in list(_ws_state['clients']):
                        try:
                            if pending:
                                await ws.send(json.dumps(pending, default=str))
                            await ws.send(payload)
                        except Exception:
                            dead.add(ws)
                    _ws_state['clients'] -= dead

                await asyncio.sleep(0.1)  # 10 Hz push rate

        async def main():
            server = await websockets.serve(handler, host, ws_port)
            print(f"  📡 WebSocket: ws://{host}:{ws_port}")
            await asyncio.gather(server.wait_closed(), broadcaster())

        try:
            loop.run_until_complete(main())
        except OSError as e:
            if 'address already in use' in str(e).lower() or e.errno == 10048:
                print(f"  \u26a0 WebSocket port {ws_port} already in use (another instance running?)")
            else:
                print(f"  \u26a0 WebSocket error: {e}")
        except Exception as e:
            print(f"  \u26a0 WebSocket error: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _build_ws_state() -> Dict:
    """Build unified state message for all views."""
    msg = {'type': 'state'}

    # Training state (live)
    if _training_state:
        msg.update({'type': 'training_update', **_training_state})

    # Replay state
    replay = _replay_data.get('replay')
    if replay:
        t = _replay_data.get('time', 0)
        # Use stats frames, fall back to evolution frames
        stats = replay.stats_frames() or replay.evolution_frames()
        # Find nearest frame
        frame = {}
        for f in stats:
            if f.get('timestamp', 0) <= t:
                frame = f
            else:
                break
        msg.update({
            'type': 'replay_state',
            'playing': _replay_data.get('playing', False),
            'speed': _replay_data.get('speed', 1.0),
            'time': round(t, 2),
            'duration': round(replay.duration_s, 1),
            'step': frame.get('step', frame.get('gen', 0)),
            'distance': round(frame.get('distance', frame.get('dist', 0)) or 0, 3),
            'best_episode': round(frame.get('best_episode', frame.get('best', 0)) or 0, 3),
            'falls': frame.get('falls', 0),
            'resets': frame.get('resets', 0),
            'reward': round(frame.get('reward', frame.get('avg', 0)) or 0, 4),
            'consciousness_level': frame.get('consciousness_level', frame.get('c_level', 0)),
            'emotion': frame.get('emotion', ''),
            'valence': round(frame.get('valence', 0) or 0, 2),
            'skill': frame.get('skill', frame.get('stage', '')),
            'phase': frame.get('phase', frame.get('stage', '')),
        })

    return msg


def _handle_ws_command(message: str, websocket=None):
    """Process commands from browser."""
    try:
        cmd = json.loads(message)
    except json.JSONDecodeError:
        return

    action = cmd.get('action', '')
    if action != 'subscribe':  # don't spam for subscribe
        print(f"  \u2b50 WS command: {action} | {cmd}")

    # Replay controls
    if action == 'toggle':
        _replay_data['playing'] = not _replay_data.get('playing', False)
    elif action == 'play':
        _replay_data['playing'] = True
    elif action == 'pause':
        _replay_data['playing'] = False
    elif action == 'restart':
        _replay_data['time'] = 0
        _replay_data['playing'] = True
    elif action == 'seek':
        t = cmd.get('time', 0)
        replay = _replay_data.get('replay')
        if replay:
            _replay_data['time'] = max(0, min(t, replay.duration_s))
    elif action == 'speed':
        _replay_data['speed'] = max(0.1, min(100, cmd.get('value', 1.0)))
    elif action == 'load_flog':
        path = cmd.get('path', '')
        # Resolve relative to project root
        if path and not os.path.isabs(path):
            for base in [os.getcwd(), 'D:/claude/mhflocke', 'D:\\claude\\mhflocke']:
                candidate = os.path.join(base, path)
                if os.path.exists(candidate):
                    path = candidate
                    break
        print(f"  \U0001f50d FLOG load request: {path} (exists={os.path.exists(path)})")
        if path and os.path.exists(path):
            try:
                from src.viz.flog_replay import FlogReplay
                replay = FlogReplay(path)
                _replay_data['replay'] = replay
                _replay_data['time'] = 0
                _replay_data['playing'] = False
                # Queue init message for next broadcaster tick
                _replay_data['pending_init'] = {
                    'type': 'replay_init',
                    'summary': {
                        'creature': replay.creature,
                        'task': replay.task,
                        'duration': replay.duration_s,
                        'n_physics': replay.n_physics,
                        'n_stats': replay.n_stats,
                        'n_events': replay.n_events,
                    },
                    'events': replay.events(),
                    'timeline': [
                        {'s': f.get('step', f.get('gen', 0)),
                         't': round(f.get('timestamp', 0), 2),
                         'd': round(f.get('distance', f.get('dist', 0)) or 0, 3),
                         'b': round(f.get('best_episode', f.get('best', 0)) or 0, 3),
                         'r': round(f.get('reward', f.get('avg', 0)) or 0, 4),
                         'c': f.get('consciousness_level', f.get('c_level', 0)),
                         'e': f.get('emotion', ''),
                         'v': round(f.get('valence', 0) or 0, 2)}
                        for f in (replay.stats_frames() or replay.evolution_frames())
                    ],
                }
                print(f"  ✅ FLOG loaded: {replay.creature} | {replay.total_frames} frames | {replay.duration_s:.1f}s")
            except Exception as e:
                print(f"  ⚠ FLOG load failed: {e}")
