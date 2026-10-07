"""Central configuration: paths, environment, and tunable constants.

Everything that more than one module needs lives here so the pipeline has
exactly one place to change a threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("GRAVITY_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"                 # private, gitignored working data
CACHE = DATA / "cache"               # per-source disk caches
MODELS = DATA / "models"             # trained model artifacts
LOGS = DATA / "logs"
SITE = ROOT / "docs"                 # GitHub Pages root (served as-is)
SITE_DATA = SITE / "data"            # JSON the site reads
PICK_LOG = SITE_DATA / "history"     # public, append-only daily pick log

for _p in (DATA, CACHE, MODELS, LOGS, SITE_DATA, PICK_LOG):
    _p.mkdir(parents=True, exist_ok=True)

# ── Environment (never commit these values) ──────────────────────────────
# SEC EDGAR rejects anonymous clients; it wants "Name contact@email".
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
# Optional: Danelfin's API (free tier = 500 calls/month). Off when unset.
DANELFIN_API_KEY = os.environ.get("DANELFIN_API_KEY", "").strip()

# ── Universe ─────────────────────────────────────────────────────────────
MIN_PRICE = 0.10                     # below this, quotes are noise
MAX_MARKET_CAP = 2_000_000_000       # small + micro + nano caps (point-in-time for training)
TRAIN_MAX_CAP = 20_000_000_000       # training pulls names up to $20B today, then keeps a row
                                     # only while that name was ≤ MAX_MARKET_CAP at the time
MIN_DOLLAR_VOLUME_20D = 50_000       # median daily $ volume to be tradeable at all
HISTORY_PERIOD = "3y"                # daily bars kept per symbol

# ── Outcome definitions (used by labels, evidence, scorecard) ────────────
# "Dump"    = open→close return ≤ DUMP_THRESHOLD on the session you'd short.
# "Squeeze" = open→high ≥ SQUEEZE_THRESHOLD (the move that stops shorts out).
DUMP_THRESHOLD = -0.05
BIG_DUMP_THRESHOLD = -0.15
SQUEEZE_THRESHOLD = 0.20

# ── Reference position (the user's INHD short) ───────────────────────────
REFERENCE_SYMBOL = "INHD"
REFERENCE_SHORT_DATE = "2026-08-28"   # "about a month ago" — approximate, user-editable in the UI

# ── Geography heuristics (issuer-level structural risk, not a judgement) ─
ASIA_COUNTRIES = {
    "China", "Hong Kong", "Singapore", "Malaysia", "Taiwan", "Japan",
    "Cayman Islands", "British Virgin Islands", "Macau", "Thailand",
    "Vietnam", "Indonesia", "Philippines",
}

# ── Output sizes ─────────────────────────────────────────────────────────
BOARD_SIZE = 25          # ranked picks shown on the board
DOSSIER_SIZE = 60        # names that get the full enrichment pass (news, borrow, SI)
TWINS_SIZE = 12          # INHD lookalikes
CHART_SESSIONS = 120     # daily bars embedded per pick for the chart

# ── Politeness ───────────────────────────────────────────────────────────
SEC_RPS = 6.0            # SEC's hard cap is 10/s
NASDAQ_RPS = 3.0
DEFAULT_RPS = 4.0
