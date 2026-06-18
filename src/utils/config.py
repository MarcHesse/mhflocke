# config.py
"""
MH-FLOCKE - CENTRAL CONFIGURATION
Edit this file to configure all settings in one place.

IMPORTANT: Secrets (API keys, FTP passwords) are loaded from
environment variables or .env file — NEVER hardcode them here.
"""

__version__ = "0.1.0"
__logbook__ = 149

import os
from pathlib import Path

# Load .env file if present (secrets, API keys)
_env_file = Path(__file__).parent.parent.parent / '.env'
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, val = line.partition('=')
            os.environ.setdefault(key.strip(), val.strip())

# ============================================================
# DATA STORAGE PATHS
# ============================================================

DATA_DIR = "data"
SQLITE_DB = "data/integrity.db"
VECTORS_DIR = "data/vectors"

# ============================================================
# FTP SYNCHRONIZATION
# ============================================================

# All FTP settings come from the environment / .env (no host or account path
# hardcoded). Leave FTP_HOST empty to keep sync disabled.
FTP_HOST = os.environ.get('FTP_HOST', '')
FTP_PORT = int(os.environ.get('FTP_PORT', '21'))
FTP_USER = os.environ.get('FTP_USER', '')
FTP_PASS = os.environ.get('FTP_PASS', '')
FTP_BASE = os.environ.get('FTP_BASE', '/mhflocke')

# DISABLED by default — enable with --sync flag or env var
FTP_SYNC_ENABLED = os.environ.get('FTP_SYNC_ENABLED', 'false').lower() == 'true'

SYNC_CHECKPOINTS = True
SYNC_DATA = True
SYNC_PROFILES = True
SYNC_LOGS = True
AUTO_SYNC_INTERVAL = 300

# ============================================================
# LEARNING SETTINGS
# ============================================================

LEARNING_DEPTH = 3
CRAWLER_DELAY = 1.0
MAX_SEARCH_RESULTS = 5

# ============================================================
# CODE EXECUTION
# ============================================================

CODE_TIMEOUT = 10
CODE_REQUIRE_APPROVAL = True

# ============================================================
# GRAPH SETTINGS
# ============================================================

ENERGY_DECAY_FACTOR = 0.95
PRUNING_THRESHOLD = 0.1

# ============================================================
# BIOLOGICAL PARAMETERS
# ============================================================

HEBBIAN_WINDOW = 5.0
HEBBIAN_STRENGTHEN_FACTOR = 1.1
WORKING_MEMORY_CAPACITY = 7
WORKING_MEMORY_TTL = 60
METACOG_CONFIDENCE_THRESHOLD = 0.75
METACOG_WEAK_THRESHOLD = 0.4
CONSOLIDATION_INTERVAL = 100
CONSOLIDATION_IMPORTANCE_HIGH = 0.7
CONSOLIDATION_IMPORTANCE_LOW = 0.3

# ============================================================
# DASHBOARD SETTINGS
# ============================================================

DASHBOARD_PORT = 5000
DASHBOARD_UPDATE_INTERVAL = 2000
GRAPH_MAX_NODES = 500
GRAPH_SHOW_LABELS = True

# ============================================================
# ADVANCED
# ============================================================

PHASE1_ENABLED = True
DEBUG_MODE = False
FAST_MODE = os.environ.get('FAST_MODE', 'false').lower() == 'true'

# ============================================================
# NEURAL NETWORK SETTINGS
# ============================================================

NEURAL_N_NEURONS = 100_000
NEURAL_TOPOLOGY = 'small_world'
NEURAL_K_NEIGHBORS = 20
NEURAL_P_REWIRE = 0.05

# ============================================================
# LLM API Keys — loaded from environment variables
# ============================================================
# Set in .env file or export before running:
#   export GEMINI_API_KEY=AIza...
#   export GROQ_API_KEY=gsk_...
#   export MISTRAL_API_KEY=RXVe...
#   export OPENROUTER_API_KEY=sk-or-...
#
# Get free keys:
#   Gemini:      https://aistudio.google.com
#   Groq:        https://console.groq.com
#   Mistral:     https://console.mistral.ai
#   OpenRouter:  https://openrouter.ai

LLM_API_KEYS = {
    'gemini':      os.environ.get('GEMINI_API_KEY', ''),
    'groq':        os.environ.get('GROQ_API_KEY', ''),
    'mistral':     os.environ.get('MISTRAL_API_KEY', ''),
    'openrouter':  os.environ.get('OPENROUTER_API_KEY', ''),
    'huggingface': os.environ.get('HF_API_KEY', ''),
}
