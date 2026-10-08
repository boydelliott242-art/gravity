"""Seed the production extended-hours store (data/exthours/ext.pkl) from the
research fetch (data/research/ext_hours.pkl) so the first production retrain
doesn't refetch two years of hourly bars. Research rows carry each session's
regular close (close_h); production rows carry the PREVIOUS session's close
(close_h_prev) next to that morning's extended-hours aggregates."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gravity.sources import exthours as X  # noqa: E402

r = pd.read_pickle(ROOT / "data" / "research" / "ext_hours.pkl")
r["date"] = pd.to_datetime(r["date"])
r = r.sort_values(["symbol", "date"]).reset_index(drop=True)
r["close_h_prev"] = r.groupby("symbol")["close_h"].shift(1)
cols = ["symbol", "date", "ah_last", "pm_last", "ext_last", "ext_high", "ext_low", "n_ext", "n_pm", "close_h_prev"]
store = r[cols]
X.STORE.parent.mkdir(parents=True, exist_ok=True)
if X.STORE.exists() and "--force" not in sys.argv:
    sys.exit(f"{X.STORE} exists; pass --force to overwrite")
store.to_pickle(X.STORE)
print(f"seeded {len(store):,} sessions for {store.symbol.nunique():,} symbols → {X.STORE}")
