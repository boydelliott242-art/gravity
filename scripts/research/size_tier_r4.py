"""Round 4: robustness of the dev-selected winner (H=20 expected-return model
trained on liquid names) and the runner-up (H=10). Uses the saved OOS
predictions from round 3."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
import size_tier_r3 as R3  # noqa: E402
from gravity import cli, model as M  # noqa: E402
from gravity.sources import prices  # noqa: E402

panel = pickle.loads(cli.PANEL_PATH.read_bytes())
lab = panel[panel["y_dump"].notna()].copy()
hist = prices.load_history(sorted(lab["symbol"].unique()), refresh=False)
lab["date"] = pd.to_datetime(lab["date"])
lab = lab.merge(R1.extra_features(hist), on=["date", "symbol"], how="left")
lab = lab.merge(R3.forward_labels(hist), on=["date", "symbol"], how="left")
lab["dvx"] = R1.exp_dvol(lab["dvol20"].to_numpy(float), lab["dv1"].to_numpy(float))
lab["cap"] = R1.capacity(lab["dvx"].to_numpy(float), lab["vol20"].to_numpy(float))
lab = lab.sort_values(["date", "symbol"]).reset_index(drop=True)
iwm = prices.benchmark_history()
io, ic = iwm["open"], iwm["close"]
iwm_fwd = {H: (ic.shift(-H) / io.shift(-1) - 1) for H in (10, 20)}

for H, fn in ((20, "oos_h20_reg_liq.npy"), (10, "oos_h10_reg_liq.npy")):
    s = np.load(R1.OUT / fn)
    ok = (~M.ssr_next(lab)) & np.isfinite(s) & np.isfinite(lab[f"fwd{H}"].to_numpy()) & (lab["close"].to_numpy() >= 1)
    print(f"\n################ H={H} expected-return (liquid-trained)")
    for tname, Q in (("$500K", 5e5), ("$1M", 1e6)):
        m = ok & (lab["cap"].to_numpy() >= Q)
        df = pd.DataFrame({"date": lab["date"][m].to_numpy(), "sym": lab["symbol"][m].to_numpy(), "s": s[m],
                           "r": lab[f"fwd{H}"][m].to_numpy(), "mae": lab[f"mae{H}"][m].to_numpy(),
                           "dvx": lab["dvx"][m].to_numpy(), "sig": lab["vol20"][m].to_numpy(),
                           "spr": np.clip(np.nan_to_num(lab["spread_ar"][m].to_numpy(), nan=0.005), 0.001, 0.02)})
        df = df.sort_values(["date", "s"])
        df["iwm"] = df["date"].map(iwm_fwd[H])
        tier_avg = df.groupby("date")["r"].mean()
        for k in (1, 3, 5):
            top = df.groupby("date").head(k).copy()
            q = Q / k
            for mult_y, mult_s, tag in ((1, 1, "base cost"), (1.5, 2, "harsh cost")):
                cost = 2 * mult_y * R1.impact(q, top.dvx.to_numpy(), top.sig.to_numpy()) + mult_s * top.spr.to_numpy() + R3.BORROW_YR * H / 252
                top[f"net_{tag}"] = -top.r.to_numpy() - cost
            day = top.groupby("date").agg(r=("r", "mean"), net=("net_base cost", "mean"), harsh=("net_harsh cost", "mean"),
                                          mae=("mae", "max"), iwm=("iwm", "first"))
            day["excess_tier"] = -(day.r - tier_avg.reindex(day.index))
            day["hedged"] = -(day.r - day.iwm)          # short the pick, long IWM
            no = day.iloc[::H]                          # non-overlapping entries
            mon = day.groupby(day.index.to_period("M"))["net"].mean()
            print(f"{tname} top{k}: distinct names {top.sym.nunique():>3} | net/trade {day.net.mean():+.4f} (harsh {day.harsh.mean():+.4f}) "
                  f"win {np.mean(day.net>0):.2f} | excess vs tier {day.excess_tier.mean():+.4f} | IWM-hedged {day.hedged.mean():+.4f} | "
                  f"worst {day.net.min():+.3f} | squeeze p90 {day.mae.quantile(.9):+.2f} max {day.mae.max():+.2f} | "
                  f"months + {int((mon>0).sum())}/{len(mon)} | non-overlap n={len(no)} mean {no.net.mean():+.4f} "
                  f"t≈{no.net.mean()/ (no.net.std(ddof=1)/np.sqrt(len(no))):+.2f}", flush=True)
        if Q == 5e5:
            top1 = df.groupby("date").head(1)
            print("   most frequent #1 names:", top1.sym.value_counts().head(8).to_dict())
