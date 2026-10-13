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
TRAIN_MAX_CAP = float("inf")         # training loads every listed name, then keeps a row only
                                     # while that name was ≤ MAX_MARKET_CAP at the time
MIN_DOLLAR_VOLUME_20D = 50_000       # median daily $ volume to be tradeable at all
# Morning-model inputs (see features.M1_EXTRA): "official" | "honest" | "ext" | "ext_counts" | "ext_on"
# "ext" won the leak-free pre-registered comparison (data/research/r9_selection_rule.txt, r9c.json)
MORNING_SET = "pm"
# Which shortlist names the morning model scores: "shortlist" (all — it was trained
# with "no extended-hours trade" rows) or "traded" (only names with an extended-hours price)
MORNING_SCOPE = "shortlist"
# Pre-market window the morning model uses: trades before 9:00 ET (needs a run after 9:00 ET) or
# before 8:00 ET (complete by the 07:20 CT run, which finishes before the Mac goes to sleep).
MORNING_CUTOFF_ET = 8
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
BOARD_SIZE = 100         # ranked picks shown on the board, each with the full dossier
# Every name on the board (and the #1) must take a short of at least BOARD_MIN_USD without moving
# the price: <= 5% of its conservative expected session $ volume, <= 0.5% estimated one-way impact
# (capacity.market_capacity), IBKR lendable >= BOARD_MIN_USD, and a price >= BOARD_MIN_PRICE.
# Ranking inside that universe: R11 (data/research/r9_selection_rule.txt, r11.json).
BOARD_MIN_USD = 30_000
BOARD_MIN_PRICE = 1.0
# The ONE (round 13): the single main-page pick — the highest-accuracy dump call among names that take a short of at
# least ONE_MIN_USD without moving the price (same volume / impact / price rules, IBKR lendable >= ONE_MIN_USD).
# Score = rank-average of three dump models within the $30K-ready universe (R13, r9_selection_rule.txt).
ONE_MIN_USD = 100_000
# conviction tiers: the three models' summed rank shortfall from #1 among $30K-ready names (R13b)
ONE_TIER_HIGH = 2
ONE_TIER_MEDIUM = 7
DOSSIER_SIZE = 60        # legacy pool size (board selection now walks the ranking — cli.BOARD_POOL_MAX)
TWINS_SIZE = 12          # INHD lookalikes
CHART_SESSIONS = 120     # daily bars embedded per pick for the chart

# ── Politeness ───────────────────────────────────────────────────────────
SEC_RPS = 6.0            # SEC's hard cap is 10/s
NASDAQ_RPS = 3.0
DEFAULT_RPS = 4.0
