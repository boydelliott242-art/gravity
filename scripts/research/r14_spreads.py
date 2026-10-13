"""R14 — calibrate the cost model's spread caps for liquid names from REAL
quotes (scripts/research/quote_sampler.py → data/spreads_sample/*.jsonl, every
$30K-ready name several times in a session).

The old caps (2% for $1–5M/day, 1% above) came from the Abdi–Ranaldo estimator's
behaviour and from quotes of thin board names. Proposed cap per liquidity tier =
max(90th percentile of the names' median quoted spread, 2 × the tier median),
rounded up to 0.05% — conservative because the strategy trades at the open and
the close, when spreads are wider than mid-session. Tiers under $1M/day keep
their caps (the live tape's quotes of thin names showed 4–9%).
Output: data/research/r14_spreads.json"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gravity import capacity as CAP  # noqa: E402

files = sorted((ROOT / "data" / "spreads_sample").glob("*.jsonl"))
rows = [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]
q = pd.DataFrame(rows)
q = q[(q["spread"] > 0) & (q["spread"] < 0.5)]
g = q.groupby("symbol").agg(spread=("spread", "median"), n=("spread", "size"), dvol20=("dvol20", "first"))
TIERS = [("1M-5M", 1e6, 5e6), ("5M-20M", 5e6, 2e7), ("20M+", 2e7, 1e15)]
OLD_CAPS = {"1M-5M": 0.02, "5M-20M": 0.01, "20M+": 0.01}          # the pre-R14 caps (capacity.py before 2026-10-13)
out = {"sessions": [f.stem for f in files], "snapshots": int(q["ts"].nunique()), "snapshot_stamps": sorted(q["ts"].unique().tolist()),
       "names": int(len(g)), "names_ready": int(q.groupby("symbol")["ready"].first().sum()) if "ready" in q else None,
       "note": "mid-session quotes only; each stamp is when that round of ~1,700 quotes finished", "tiers": {}}
for name, lo, hi in TIERS:
    x = g[(g.dvol20 >= lo) & (g.dvol20 < hi)]["spread"]
    p90, med = float(x.quantile(0.9)), float(x.median())
    cap = math.ceil(max(p90, 2 * med) / 0.0005) * 0.0005
    out["tiers"][name] = {"names": int(len(x)), "median": med, "p75": float(x.quantile(0.75)), "p90": p90,
                          "old_cap": OLD_CAPS[name], "proposed_cap": cap,
                          "applied_cap": float(CAP.spread_cap((lo + min(hi, 1e9)) / 2))}
out["applied_note"] = ("capacity.py applies the caps computed from the first three rounds (10:04–14:52 ET): 1.45% / 0.9% / "
                       "0.55%. Adding the 15:20 round (finished near the close, when spreads are tighter) lowers the proposal; "
                       "the higher caps are kept because spreads at the open, when these shorts are entered, were not sampled.")
(ROOT / "data" / "research" / "r14_spreads.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
