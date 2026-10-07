"""Reproduce gravity/capacity.py NEXT_DV_Q25: the 25th percentile of
next-session dollar volume ÷ 20-day MEDIAN dollar volume, by today's
relative volume (today's $ volume ÷ the same 20-day median) bucket."""
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gravity import capacity as C, cli  # noqa: E402
from gravity.sources import prices  # noqa: E402

panel = pickle.loads(cli.PANEL_PATH.read_bytes())
hist = prices.load_history(sorted(panel["symbol"].unique()), refresh=False)
rows = []
for s, df in hist.items():
    if df is None or len(df) < 30:
        continue
    d = df[["close", "volume"]].copy()
    d["dv"] = d.close * d.volume
    d["dv_next"] = d.dv.shift(-1)
    d["dv20"] = d.dv.rolling(20, min_periods=10).median()
    rows.append(d.dropna())
a = pd.concat(rows)
a = a[a.dv20 > 0]
a["rvol"] = a.dv / a.dv20
edges = list(C.RVOL_EDGES) + [1e12]
for lo, hi, coded in zip(edges[:-1], edges[1:], C.NEXT_DV_Q25):
    m = a[(a.rvol >= lo) & (a.rvol < hi)]
    print(f"[{lo:>4}, {hi:<6}) n={len(m):>8}  q25 = {(m.dv_next / m.dv20).quantile(0.25):.2f}   coded {coded:.2f}")
