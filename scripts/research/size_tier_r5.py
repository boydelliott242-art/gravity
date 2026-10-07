"""Round 5: survivorship / selection check. Rebuild the panel on a wider
universe (today's names up to $20B) and apply the $2B cap POINT-IN-TIME
(close_t × today's share count), so names that grew out of small-cap range
are included and names that shrank into it are only counted while small.
Then rerun the dev-selected winner (H=20 expected return, liquid-trained)."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
import size_tier_r3 as R3  # noqa: E402
from gravity import config, features, model as M, net  # noqa: E402
from gravity.sources import prices, sec, universe  # noqa: E402

t0 = time.time()
config.MAX_MARKET_CAP = 20e9
uni = universe.load_universe(max_age_s=1)
uni = uni[uni["market_cap"] > 0]
uni["shares_now"] = uni["market_cap"] / uni["price"]
syms = sorted(uni["symbol"].unique())
print(f"wide universe: {len(syms)} names (≤ $20B today)", flush=True)
hist = prices.load_history(syms, refresh=True)
hist = {s: d for s, d in hist.items() if d is not None and len(d) >= 30}
print(f"history for {len(hist)} names ({time.time()-t0:.0f}s)", flush=True)
splits = prices.load_splits(list(hist))
bench = prices.benchmark_history()
events = sec.events_for_universe(list(hist)) if net.sec_enabled() else {}
static = uni.set_index("symbol")[["asia", "ipo_year"]]
static = static[~static.index.duplicated()]
panel = features.build_panel(hist, events, splits, static, bench)
lab = panel[panel["y_dump"].notna()].copy()
lab["date"] = pd.to_datetime(lab["date"])
sh = uni.drop_duplicates("symbol").set_index("symbol")["shares_now"]
lab["pit_mcap"] = lab["close"] * lab["symbol"].map(sh)
lab = lab.merge(R1.extra_features(hist), on=["date", "symbol"], how="left")
lab = lab.merge(R3.forward_labels(hist), on=["date", "symbol"], how="left")
lab["dvx"] = R1.exp_dvol(lab["dvol20"].to_numpy(float), lab["dv1"].to_numpy(float))
lab["cap"] = R1.capacity(lab["dvx"].to_numpy(float), lab["vol20"].to_numpy(float))
lab["log_dvx"] = np.log10(lab["dvx"].clip(lower=1)); lab["log_cap"] = np.log10(lab["cap"].clip(lower=1))
lab = lab.sort_values(["date", "symbol"]).reset_index(drop=True)
lab.to_pickle(R1.OUT / "wide_lab.pkl")
pit_small = (lab["pit_mcap"] <= 2e9).to_numpy()
print(f"wide panel {len(lab):,} rows; point-in-time ≤$2B rows {pit_small.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
mask_dev = (lab["date"] < pd.Timestamp("2025-11-01")).to_numpy()
F = features.FEATURES + R1.EXTRA
liquid = lab["dvx"].to_numpy() >= 2e6
for name, train_mask, eval_mask in (
        ("PIT small caps only (train + eval on point-in-time ≤ $2B)", liquid & pit_small, pit_small),
        ("all caps ≤ $20B (train + eval on everything)", liquid, np.ones(len(lab), bool)),
):
    s = R3.wf(lab, F, "fwd20", 20, regress=True, train_mask=train_mask)
    s = np.where(eval_mask, s, np.nan)
    R3.evaluate(lab, s, 20, mask_dev, f"H=20 expected return — {name}", ascending=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
