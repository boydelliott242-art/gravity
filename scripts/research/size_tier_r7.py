"""Round 7: the Size Lab's same-day tier table, recomputed on the CORRECTED
point-in-time universe with the production cost model (gravity/capacity.py,
tick-aware spread band). Saves data/research/size_tier_r7.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
from gravity import capacity as CAP, cli, model as M  # noqa: E402
from gravity import features as F  # noqa: E402
from gravity.sources import prices, sec, universe  # noqa: E402

lab = pd.read_pickle(R1.OUT / "wide_lab.pkl")
uni = universe.load_universe(max_cap=float("inf")).drop_duplicates("symbol")
shares_today = (uni.set_index("symbol")["market_cap"] / uni.set_index("symbol")["price"]).replace([np.inf, -np.inf], np.nan)
small_today = set(uni.loc[uni["market_cap"] <= 2e9, "symbol"])
splits = prices.load_splits(sorted(lab["symbol"].unique()))
pit = cli._pit_market_cap(lab, sec.shares_frames(), sec.cik_map(), splits, shares_today)
keep = ((pit <= 2e9) | (pit.isna() & lab["symbol"].isin(small_today))).to_numpy()
sub = lab[keep & lab["y_oc"].notna().to_numpy()].reset_index(drop=True)
F._cross_sectional(sub)
p = R1.walk_forward(sub, F.FEATURES)
spr = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
base_ok = (~M.ssr_next(sub)) & np.isfinite(p)
tiers = {
    "micro": (base_ok & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR), (1e4, 5e4)),
    "$500K": (base_ok & (sub["close"].to_numpy() >= 1) & (sub["cap"].to_numpy() >= 5e5), (5e5,)),
    "$1M": (base_ok & (sub["close"].to_numpy() >= 1) & (sub["cap"].to_numpy() >= 1e6), (1e6,)),
}
out = {}
for period, pm in (("dev", (sub["date"] < pd.Timestamp("2025-11-01")).to_numpy()), ("confirm", (sub["date"] >= pd.Timestamp("2025-11-01")).to_numpy())):
    for t, (mask, sizes) in tiers.items():
        m = mask & pm
        d = sub[m].assign(p=p[m], spr=spr[m]).sort_values(["date", "p"], ascending=[True, False])
        top = d.groupby("date").head(1)
        row = {"names_per_day": float(d.groupby("date").size().median()), "base_dump": float(d.y_dump.mean()),
               "top1_hit": float(top.y_dump.mean()), "top1_mean_oc": float(top.y_oc.mean())}
        for q in sizes:
            c = np.asarray(CAP.round_trip_cost(q, top["dvx"].to_numpy(float), top["vol20"].to_numpy(float), top["spr"].to_numpy(float)), float)
            row[f"cost_{int(q)}"] = float(c.mean())
            row[f"net_{int(q)}"] = float((-top.y_oc.to_numpy() - c).mean())
            row[f"net_tick_{int(q)}"] = float((-top.y_oc.to_numpy() - np.asarray(CAP.round_trip_cost(
                q, top["dvx"].to_numpy(float), top["vol20"].to_numpy(float),
                CAP.spread_used_vec(np.full(len(top), 0.0011), top["close"].to_numpy(float), np.full(len(top), 1e12))), float)).mean())
        out[f"{period}|{t}"] = row
        print(period, t, {k: round(v, 4) for k, v in row.items()}, flush=True)
(R1.OUT / "size_tier_r7.json").write_text(json.dumps(out, indent=1))
print("DONE")
