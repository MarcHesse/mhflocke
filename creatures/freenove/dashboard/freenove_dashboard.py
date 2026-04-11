#!/usr/bin/env python3
"""
MH-FLOCKE — Freenove Live Dashboard v2
========================================
Now with live neuron visualization!

Displays:
  - 232 Izhikevich neurons (color-coded, firing animation)
  - Servo angles (12 channels)
  - SNN stats + Competence Gate
  - Performance metrics

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

import time, threading, json, os, sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

PORT = 8080

dashboard_state = {
    'step': 0, 'gait': 'stand', 'speed': 0.0, 'ms_per_step': 0.0,
    'snn_enabled': False, 'servos': {}, 'cpg_phase': 0.0,
    'snn': {'firing_rate': 0.0, 'mean_weight': 0.0, 'da_level': 0.5},
    'gate': {'cpg_weight': 0.9, 'actor_competence': 0.0},
    'neurons': [],  # List of 232 bools (fired this step)
    'uptime': 0.0,
}

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MH-FLOCKE — Freenove Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: #0a0e17;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
}
.header {
    background: #111827;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #00d4ff33;
}
.header h1 { font-size: 18px; font-weight: 500; color: #00d4ff; }
.header .status { font-size: 12px; color: #888; }
.header .status .dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #22c55e; margin-right: 6px; vertical-align: middle;
}
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px; padding: 12px;
    max-width: 1100px; margin: 0 auto;
}
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px;
}
.card h2 {
    font-size: 13px; font-weight: 500; color: #00d4ff;
    margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;
}
.card.full { grid-column: 1 / -1; }
.servo-row { display: flex; align-items: center; margin: 3px 0; font-size: 12px; }
.servo-label { width: 70px; color: #888; font-family: monospace; }
.servo-bar-bg { flex:1; height:14px; background:#1e293b; border-radius:3px; overflow:hidden; }
.servo-bar { height:100%; border-radius:3px; transition: width 0.1s; }
.servo-val { width:40px; text-align:right; font-family:monospace; color:#ccc; font-size:11px; margin-left:6px; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat { text-align:center; padding:8px; background:#0a0e17; border-radius:6px; }
.stat .value { font-size:22px; font-weight:500; color:#00d4ff; font-family:monospace; }
.stat .label { font-size:10px; color:#666; margin-top:2px; text-transform:uppercase; }
.gate-bar { height:24px; background:#1e293b; border-radius:4px; display:flex; overflow:hidden; margin:8px 0; }
.gate-cpg { background:#0066cc; transition:width 0.3s; display:flex; align-items:center; justify-content:center; font-size:11px; color:white; font-weight:500; }
.gate-actor { background:#00d4ff; transition:width 0.3s; display:flex; align-items:center; justify-content:center; font-size:11px; color:#0a0e17; font-weight:500; }

/* Neuron visualization */
.neuron-container {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.neuron-section {
    display: flex;
    align-items: center;
    gap: 8px;
}
.neuron-section-label {
    font-size: 10px;
    color: #666;
    width: 50px;
    text-align: right;
    text-transform: uppercase;
}
.neuron-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
}
.neuron {
    width: 6px;
    height: 6px;
    border-radius: 1px;
    transition: background 0.15s;
}
/* Input neurons: blue family */
.neuron.input { background: #0a2540; }
.neuron.input.fired { background: #00d4ff; box-shadow: 0 0 4px #00d4ff88; }
/* Hidden excitatory: gray/purple */
.neuron.hidden-exc { background: #1a1a2e; }
.neuron.hidden-exc.fired { background: #7f77dd; box-shadow: 0 0 4px #7f77dd88; }
/* Hidden inhibitory: gray/red */
.neuron.hidden-inh { background: #2a1a1a; }
.neuron.hidden-inh.fired { background: #e24b4a; box-shadow: 0 0 3px #e24b4a88; }
/* PkC neurons: purple */
.neuron.pkc { background: #1a0a2e; }
.neuron.pkc.fired { background: #a050dc; box-shadow: 0 0 4px #a050dc88; }
/* DCN neurons: orange */
.neuron.dcn { background: #2a1a0a; }
.neuron.dcn.fired { background: #ff8c28; box-shadow: 0 0 4px #ff8c2888; }
/* Output neurons: green family */
.neuron.output { background: #0a2a15; }
.neuron.output.fired { background: #22c55e; box-shadow: 0 0 5px #22c55e88; }

.neuron-legend {
    display: flex;
    gap: 14px;
    margin-top: 6px;
    font-size: 10px;
    color: #666;
}
.neuron-legend span {
    display: flex;
    align-items: center;
    gap: 4px;
}
.legend-dot {
    width: 8px; height: 8px;
    border-radius: 1px;
    display: inline-block;
}
</style>
</head>
<body>
<div class="header">
    <h1>MH-FLOCKE — Freenove</h1>
    <div class="status"><span class="dot" id="conn-dot"></span><span id="conn-text">Connecting...</span></div>
</div>

<div class="grid">
    <!-- Performance -->
    <div class="card">
        <h2>Performance</h2>
        <div class="stat-grid">
            <div class="stat"><div class="value" id="s-step">0</div><div class="label">Step</div></div>
            <div class="stat"><div class="value" id="s-ms">0.0</div><div class="label">ms/step</div></div>
            <div class="stat"><div class="value" id="s-gait">stand</div><div class="label">Gait</div></div>
            <div class="stat"><div class="value" id="s-uptime">0s</div><div class="label">Uptime</div></div>
        </div>
    </div>

    <!-- SNN / Gate -->
    <div class="card">
        <h2>SNN + Competence Gate</h2>
        <div class="gate-bar">
            <div class="gate-cpg" id="gate-cpg" style="width:90%">CPG 90%</div>
            <div class="gate-actor" id="gate-actor" style="width:10%">10%</div>
        </div>
        <div class="stat-grid">
            <div class="stat"><div class="value" id="s-fr">0.00</div><div class="label">Firing rate</div></div>
            <div class="stat"><div class="value" id="s-da">0.50</div><div class="label">DA level</div></div>
            <div class="stat"><div class="value" id="s-comp">0.00</div><div class="label">Competence</div></div>
            <div class="stat"><div class="value" id="s-weight">0.00</div><div class="label">Mean |W|</div></div>
        </div>
    </div>

    <!-- Neuron visualization -->
    <div class="card full">
        <h2>SNN — 232 Neurons (Cerebellar Architecture)</h2>
        <div class="neuron-container">
            <div class="neuron-section">
                <span class="neuron-section-label">MF<br>48</span>
                <div class="neuron-grid" id="neurons-mf"></div>
            </div>
            <div class="neuron-section">
                <span class="neuron-section-label">GrC<br>106</span>
                <div class="neuron-grid" id="neurons-grc"></div>
            </div>
            <div class="neuron-section">
                <span class="neuron-section-label">GoC<br>18</span>
                <div class="neuron-grid" id="neurons-goc"></div>
            </div>
            <div class="neuron-section">
                <span class="neuron-section-label">PkC<br>24</span>
                <div class="neuron-grid" id="neurons-pkc"></div>
            </div>
            <div class="neuron-section">
                <span class="neuron-section-label">DCN<br>24</span>
                <div class="neuron-grid" id="neurons-dcn"></div>
            </div>
            <div class="neuron-section">
                <span class="neuron-section-label">OUT<br>12</span>
                <div class="neuron-grid" id="neurons-out"></div>
            </div>
        </div>
        <div class="neuron-legend">
            <span><span class="legend-dot" style="background:#00d4ff"></span> MF (input)</span>
            <span><span class="legend-dot" style="background:#3c82c8"></span> GrC (granule)</span>
            <span><span class="legend-dot" style="background:#e24b4a"></span> GoC (inhibitory)</span>
            <span><span class="legend-dot" style="background:#a050dc"></span> PkC (purkinje)</span>
            <span><span class="legend-dot" style="background:#ff8c28"></span> DCN (motor corr.)</span>
            <span><span class="legend-dot" style="background:#22c55e"></span> OUT (motor)</span>
        </div>
    </div>

    <!-- Servo angles -->
    <div class="card full">
        <h2>Servo Angles</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px 16px;">
            <div id="servo-left"></div>
            <div id="servo-right"></div>
        </div>
    </div>
</div>

<script>
// Build neuron dots
// Real cerebellar populations (Freenove 232 neurons)
const N_MF = 48, N_GRC = 106, N_GOC = 18, N_PKC = 24, N_DCN = 24, N_OUT = 12;
const N_TOTAL = N_MF + N_GRC + N_GOC + N_PKC + N_DCN + N_OUT;
// Hidden neuron types (20% inhibitory) — indices relative to hidden start

function buildNeurons() {
    // Real cerebellar populations — matches SNN topology exactly
    const pops = [
        {el: 'neurons-mf',  n: N_MF,  cls: 'neuron input',      label: 'MF'},
        {el: 'neurons-grc', n: N_GRC, cls: 'neuron hidden-exc',  label: 'GrC'},
        {el: 'neurons-goc', n: N_GOC, cls: 'neuron hidden-inh',  label: 'GoC'},
        {el: 'neurons-pkc', n: N_PKC, cls: 'neuron pkc',         label: 'PkC'},
        {el: 'neurons-dcn', n: N_DCN, cls: 'neuron dcn',         label: 'DCN'},
        {el: 'neurons-out', n: N_OUT, cls: 'neuron output',      label: 'OUT'},
    ];
    let idx = 0;
    pops.forEach(function(pop) {
        const container = document.getElementById(pop.el);
        for (let i = 0; i < pop.n; i++) {
            const d = document.createElement('div');
            d.className = pop.cls;
            d.id = 'n-' + idx;
            d.title = pop.label + ' ' + i;
            if (pop.label === 'OUT') { d.style.width = '10px'; d.style.height = '10px'; }
            container.appendChild(d);
            idx++;
        }
    });
}
buildNeurons();

// Servo bars
const CHANNELS_LEFT = [
    {ch:4, name:'FL yaw'}, {ch:3, name:'FL pitch'}, {ch:2, name:'FL knee'},
    {ch:7, name:'RL yaw'}, {ch:6, name:'RL pitch'}, {ch:5, name:'RL knee'},
];
const CHANNELS_RIGHT = [
    {ch:11, name:'FR yaw'}, {ch:12, name:'FR pitch'}, {ch:13, name:'FR knee'},
    {ch:8, name:'RR yaw'}, {ch:9, name:'RR pitch'}, {ch:10, name:'RR knee'},
];
function buildServoRows(container, channels) {
    channels.forEach(c => {
        container.innerHTML += '<div class="servo-row">' +
            '<span class="servo-label">' + c.name + '</span>' +
            '<div class="servo-bar-bg"><div class="servo-bar" id="bar-' + c.ch + '" style="width:50%;background:#0066cc"></div></div>' +
            '<span class="servo-val" id="val-' + c.ch + '">90</span></div>';
    });
}
buildServoRows(document.getElementById('servo-left'), CHANNELS_LEFT);
buildServoRows(document.getElementById('servo-right'), CHANNELS_RIGHT);

function updateServo(ch, angle) {
    const pct = ((angle - 18) / (162 - 18)) * 100;
    const bar = document.getElementById('bar-' + ch);
    const val = document.getElementById('val-' + ch);
    if (bar) {
        bar.style.width = pct + '%';
        const hue = Math.max(0, 200 - Math.abs(angle - 90) * 3);
        bar.style.background = 'hsl(' + hue + ', 80%, 50%)';
    }
    if (val) val.textContent = Math.round(angle);
}

function updateNeurons(fired) {
    if (!fired || fired.length === 0) return;
    const total = N_TOTAL;
    for (let i = 0; i < Math.min(fired.length, total); i++) {
        const el = document.getElementById('n-' + i);
        if (el) {
            if (fired[i]) {
                el.classList.add('fired');
            } else {
                el.classList.remove('fired');
            }
        }
    }
}

function poll() {
    fetch('/api/state')
    .then(r => r.json())
    .then(d => {
        document.getElementById('s-step').textContent = d.step;
        document.getElementById('s-ms').textContent = d.ms_per_step.toFixed(1);
        document.getElementById('s-gait').textContent = d.gait;
        document.getElementById('s-uptime').textContent = Math.round(d.uptime) + 's';

        if (d.servos) {
            Object.entries(d.servos).forEach(function(e) {
                updateServo(parseInt(e[0]), e[1]);
            });
        }

        const snn = d.snn || {};
        const gate = d.gate || {};
        document.getElementById('s-fr').textContent = (snn.firing_rate || 0).toFixed(3);
        document.getElementById('s-da').textContent = (snn.da_level || 0.5).toFixed(2);
        document.getElementById('s-comp').textContent = (gate.actor_competence || 0).toFixed(3);
        document.getElementById('s-weight').textContent = (snn.mean_weight || 0).toFixed(2);

        const cpgW = Math.round((gate.cpg_weight || 0.9) * 100);
        const actW = 100 - cpgW;
        document.getElementById('gate-cpg').style.width = cpgW + '%';
        document.getElementById('gate-cpg').textContent = 'CPG ' + cpgW + '%';
        document.getElementById('gate-actor').style.width = actW + '%';
        document.getElementById('gate-actor').textContent = actW + '%';

        // Neurons
        updateNeurons(d.neurons || []);

        document.getElementById('conn-dot').style.background = '#22c55e';
        document.getElementById('conn-text').textContent = d.snn_enabled ? 'SNN active' : 'CPG only';
    })
    .catch(function() {
        document.getElementById('conn-dot').style.background = '#ef4444';
        document.getElementById('conn-text').textContent = 'Disconnected';
    });
}
setInterval(poll, 200);
poll();
</script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(dashboard_state).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


class DashboardServer:
    def __init__(self, port=PORT):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.server = socketserver.ThreadingTCPServer(('0.0.0.0', self.port), DashboardHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f'  Dashboard: http://robot:{self.port}')

    def update(self, state_dict):
        dashboard_state.update(state_dict)

    def stop(self):
        if self.server:
            self.server.shutdown()


if __name__ == '__main__':
    print('Starting dashboard server...')
    srv = DashboardServer()
    srv.start()
    print(f'Open http://localhost:{PORT}')

    import math, random
    step = 0
    try:
        while True:
            step += 1
            phase = step * 0.05
            test_servos = {}
            for ch in [2,3,4,5,6,7,8,9,10,11,12,13]:
                test_servos[ch] = 90 + 30 * math.sin(phase + ch * 0.5)

            # Simulated neuron firing (random spikes for testing)
            neurons = [random.random() < 0.05 for _ in range(232)]  # test mode only

            srv.update({
                'step': step, 'gait': 'walk', 'ms_per_step': 1.2,
                'snn_enabled': True, 'servos': test_servos,
                'cpg_phase': phase % (2 * math.pi),
                'neurons': neurons,
                'snn': {
                    'firing_rate': 0.05 + 0.02 * math.sin(phase * 0.1),
                    'mean_weight': 1.5 + 0.3 * math.sin(phase * 0.05),
                    'da_level': 0.5 + 0.2 * math.sin(phase * 0.03),
                },
                'gate': {
                    'cpg_weight': max(0.4, 0.9 - step * 0.0001),
                    'actor_competence': min(1.0, step * 0.0002),
                },
                'uptime': step * 0.02,
            })
            time.sleep(0.02)
    except KeyboardInterrupt:
        srv.stop()
        print('\nDone.')
