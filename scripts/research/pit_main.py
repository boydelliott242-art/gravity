"""Does the MAIN daily #1 (intraday dump model) survive a point-in-time
universe? Same production recipe; compare today's-universe rows vs
point-in-time ≤ $2B rows from the wide panel."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
from gravity import model as M  # noqa: E402
from gravity.features import FEATURES  # noqa: E402

lab = pd.read_pickle(R1.OUT / "wide_lab.pkl")
lab = lab[lab["y_oc"].notna()].reset_index(drop=True)
today_small = set(pd.read_pickle(R1.OUT / "lab_meta.pkl")["symbol"].unique())
mask_dev = (lab["date"] < pd.Timestamp("2025-11-01")).to_numpy()
pit = (lab["pit_mcap"] <= 2e9).to_numpy()
todayu = lab["symbol"].isin(today_small).to_numpy()
for name, m in (("today's ≤$2B list (as published)", todayu), ("point-in-time ≤$2B", pit)):
    sub = lab[m].reset_index(drop=True)
    p = R1.walk_forward(sub, FEATURES)
    ok = (~M.ssr_next(sub)) & np.isfinite(p) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    for period, pm in (("dev", (sub["date"] < pd.Timestamp("2025-11-01")).to_numpy()), ("confirm", (sub["date"] >= pd.Timestamp("2025-11-01")).to_numpy())):
        df = pd.DataFrame({"date": sub["date"][ok & pm], "p": p[ok & pm], "y": sub["y_dump"][ok & pm], "oc": sub["y_oc"][ok & pm], "oh": sub["y_oh"][ok & pm]})
        top = df.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(1)
        print(f"{name:34s} {period:7s} base {df.y.mean():.3f} | #1 hit {top.y.mean():.3f} mean oc {top.oc.mean():+.4f} median {top.oc.median():+.4f} squeeze {np.mean(top.oh>=0.2):.3f} | AUC {M._auc(df.y.to_numpy(), df.p.to_numpy()):.3f}", flush=True)
print("DONE")
