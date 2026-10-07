"""Round 6: redo the point-in-time checks with the CORRECTED filter (SEC
share counts as reported at the time, split-adjusted to the trade date)
and the production cost model (gravity/capacity.py, conservative spread
fallback). Re-evaluates: the main #1 (intraday) and the 20-session
liquid-name short."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
import size_tier_r3 as R3  # noqa: E402
from gravity import capacity as CAP, cli, model as M, net  # noqa: E402
from gravity.features import FEATURES  # noqa: E402
from gravity.sources import prices, sec, universe  # noqa: E402

t0 = time.time()
lab = pd.read_pickle(R1.OUT / "wide_lab.pkl")
uni = universe.load_universe(max_cap=float("inf")).drop_duplicates("symbol")
shares_today = (uni.set_index("symbol")["market_cap"] / uni.set_index("symbol")["price"]).replace([np.inf, -np.inf], np.nan)
small_today = set(uni.loc[uni["market_cap"] <= 2e9, "symbol"])
splits = prices.load_splits(sorted(lab["symbol"].unique()))
pit = cli._pit_market_cap(lab, sec.shares_frames(), sec.cik_map(), splits, shares_today)
keep = ((pit <= 2e9) | (pit.isna() & lab["symbol"].isin(small_today))).to_numpy()
print(f"corrected PIT keeps {keep.mean():.3f} of wide rows ({time.time()-t0:.0f}s)", flush=True)
old_keep = (lab["pit_mcap"] <= 2e9).to_numpy()
diluters_back = keep & ~old_keep
print(f"rows restored vs the flawed filter: {int(diluters_back.sum()):,} (their dump rate {lab.loc[diluters_back,'y_dump'].mean():.3f} vs kept {lab.loc[keep,'y_dump'].mean():.3f})", flush=True)

def spr_of(df):
    return CAP.spread_used_vec(df["spread_ar"].to_numpy(float), df["close"].to_numpy(float), df["dvol20"].to_numpy(float))


def costs(df, Q):
    return np.asarray(CAP.round_trip_cost(Q, df["dvx"].to_numpy(float), df["vol20"].to_numpy(float), spr_of(df)), float)

res = {}
sub = lab[keep & lab["y_oc"].notna().to_numpy()].reset_index(drop=True)
from gravity import features as F  # recompute cross-sectional features on the kept universe
F._cross_sectional(sub)
p = R1.walk_forward(sub, FEATURES)
ok = (~M.ssr_next(sub)) & np.isfinite(p) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
for period, pm in (("dev", (sub["date"] < pd.Timestamp("2025-11-01")).to_numpy()), ("confirm", (sub["date"] >= pd.Timestamp("2025-11-01")).to_numpy())):
    d = sub[ok & pm].assign(p=p[ok & pm]).sort_values(["date", "p"], ascending=[True, False])
    top = d.groupby("date").head(1)
    c10 = costs(top, 1e4)
    res[f"main|{period}"] = {"base": float(d.y_dump.mean()), "hit": float(top.y_dump.mean()), "mean_oc": float(top.y_oc.mean()),
                             "median_oc": float(top.y_oc.median()), "auc": M._auc(d.y_dump.to_numpy(), d.p.to_numpy()),
                             "cost10k": float(c10.mean()), "net10k": float((-top.y_oc.to_numpy() - c10).mean())}
    print("main", period, {k: round(v, 4) for k, v in res[f"main|{period}"].items()}, flush=True)

lab2 = lab[keep].reset_index(drop=True)
F._cross_sectional(lab2)
liquid = lab2["dvx"].to_numpy() >= 2e6
s20 = R3.wf(lab2, FEATURES + R1.EXTRA, "fwd20", 20, regress=True, train_mask=liquid)
okk = (~M.ssr_next(lab2)) & np.isfinite(s20) & np.isfinite(lab2["fwd20"].to_numpy()) & (lab2["close"].to_numpy() >= 1)
for period, pm in (("dev", (lab2["date"] < pd.Timestamp("2025-11-01")).to_numpy()), ("confirm", (lab2["date"] >= pd.Timestamp("2025-11-01")).to_numpy())):
    for tname, Q in (("$500K", 5e5), ("$1M", 1e6)):
        m = okk & pm & (lab2["cap"].to_numpy() >= Q)
        d = lab2[m].assign(s=s20[m]).sort_values(["date", "s"])
        top = d.groupby("date").head(1)
        # exit 20 sessions later priced on normal (not spike) volume
        exit_dvx = top["dvol20"].to_numpy(float) * CAP.NEXT_DV_Q25[0]
        c = np.asarray(CAP.impact(Q, top["dvx"].to_numpy(float), top["vol20"].to_numpy(float))
                       + CAP.impact(Q, exit_dvx, top["vol20"].to_numpy(float)) + spr_of(top) + 0.05 * 20 / 252, float)
        net_ = -top["fwd20"].to_numpy(float) - c
        no = net_[::20]
        res[f"h20|{period}|{tname}"] = {"trades": int(len(top)), "net": float(net_.mean()), "tier_mean": float(d["fwd20"].mean()),
                                        "win": float((net_ > 0).mean()), "indep_trades": int(len(no)),
                                        "t": float(no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))) if len(no) > 2 else None}
        print("h20", period, tname, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[f"h20|{period}|{tname}"].items()}, flush=True)
(R1.OUT / "size_tier_r6.json").write_text(json.dumps(res, indent=1, default=float))
print(f"DONE {time.time()-t0:.0f}s", flush=True)
