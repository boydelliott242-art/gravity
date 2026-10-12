"""Baskets (top-K, each capped at capacity up to $10/25/50K) and rank-band precision for the adopted
pre-market-only morning model (D3_8) and the close-only model, for the Size Lab planner. Output: r10c.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402

sub = R.build_sub()
sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
for c in ("h_open", "c1030", "c1230", "close_h", "hi1030", "hi_all", "path_ok", "gap_ext"):
    if c not in sub.columns:
        sub[c] = np.nan
res = {}
for name, f in (("D3_8", "oos_r10_D3_8.npy"), ("A_m0", "oos_r9_A_m0.npy")):
    p = np.load(R.OUT / f)
    res[name] = {"baskets": R.baskets(sub, p)}
    print(name, {k: round(v["net_per_dollar"] * 100, 2) for k, v in res[name]["baskets"].items() if k.endswith("50k")}, flush=True)
(R.OUT / "r10c.json").write_text(json.dumps(res, indent=1, default=float))
print("DONE", flush=True)
