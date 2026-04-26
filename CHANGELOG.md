# CHANGELOG

## v0.4.8 — Phototaxis Navigation + 6× Performance Fix (2026-04-25)

### Phototaxis Navigation
- **VOR (Vestibulo-Ocular Response) steering** — hardwired reflex turns dog toward light source
- **Waypoint system** — fixed positions with relative spawning, respawn on miss (4.5m)
- **Run-and-Tumble integration** — RT state machine triggers Tumbles during navigation
- **Geometric light gradient** — `1/(0.5+dist)²` bilateral brightness computation
- **First successful navigation**: sf:2 (two waypoints reached), VOR up to +0.54
- **MuJoCo light body** — emissive sphere with spotlight injected into scene

### Performance (6× Speedup)
- **Root cause found**: Synaptogenesis `ExperienceBuffer` O(N²) clustering over 5000 entries
- **Fix**: `buffer.clear()` after consolidation, max_size 5000→500
- **Dense SNN threshold**: 500→600 (Freenove 560 neurons now uses fast dense path)
- **R-STDP lazy dirty flag**: dense matrix rebuilt only in next forward(), not after every update
- **Memory fixes**: deque replacements for list.pop(0) in world_model, spatial_map, directed_learning, embodied_emotions
- **Result**: 7 sps → 54 sps stable over 100k steps. 100k run in 30 min instead of 5+ hours.

### Video Rendering
- **Mini-map overlay** in render_freenove.py — bottom-left, shows trail + light waypoints
- **Scent circles in 3D view disabled** — replaced by cleaner mini-map visualization
- **Instagram Reel renderer** (render_insta_reel.py) — 3:4 format with RUN/MAP/BRAIN sections
- **Thumbnail generator** (render_phototaxis_thumb.py)

### Documentation
- **HONEST_CLAIMS.md** — complete documentation of hardwired vs. learned components
- **Review checklist** for public posts (Reddit, HN, papers)
- **SESSION_SUMMARY_20260424.md**

### Key Metrics
| Run | Steps | Time | SPS | Distance | Falls | sf | Correction |
|-----|-------|------|-----|----------|-------|----|-----------|
| Phototaxis 33k (sf:2) | 33k | 9.7m | 57 | 5.97m | 0 | 2 | 0.008 |
| Phototaxis 100k | 100k | 30.6m | 54 | 24.8m | 0 | 2 | 0.034 |
| Pre-fix baseline | 100k | 5h+ | 7 | 18.2m | 0 | - | 0.021 |

### Files Changed
- `src/brain/synaptogenesis.py` — buffer.clear(), max_size 500
- `src/brain/snn_controller.py` — dense threshold 600, lazy dirty flag
- `src/brain/world_model.py` — deque fix
- `src/brain/spatial_map.py` — deque fix
- `src/brain/directed_learning.py` — deque fix
- `src/brain/embodied_emotions.py` — deque fix
- `src/brain/brain_persistence.py` — deque slice fixes
- `src/brain/cognitive_brain.py` — pre-allocated tensor buffers
- `src/body/visual_environment.py` — waypoint system, phototaxis
- `scripts/train_baby.py` — VOR steering, profiling, gc
- `scripts/render_freenove.py` — mini-map overlay, scent circles disabled
- `scripts/render_insta_reel.py` — NEW: Instagram 3:4 renderer
- `scripts/render_phototaxis_thumb.py` — NEW: thumbnail generator
- `scripts/test_cpg_steering.py` — NEW: CPG steering verification
- `scripts/debug_flog_keys.py` — NEW: FLOG stats inspector
- `creatures/freenove/creature.xml` — offscreen buffer 1920x1920
- `docs/HONEST_CLAIMS.md` — NEW
- `docs/SESSION_SUMMARY_20260424.md` — NEW
