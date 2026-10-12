"""R10d — is the pre-market morning model still better than close-only when a
quarter of its pre-market prices are missing (Yahoo's thinner feed since 10-07)?"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402
from gravity import features as F  # noqa: E402

OUT = R.OUT
sub = R.build_sub()
sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
for c in ("h_open", "c1030", "c1230", "close_h", "hi1030", "hi_all", "path_ok", "gap_ext"):
    if c not in sub.columns:
        sub[c] = np.nan
ext = pd.read_pickle(OUT / "ext_hours2.pkl"); ext["date"] = pd.to_datetime(ext["date"])
ext = ext.sort_values(["symbol", "date"]).reset_index(drop=True)
g = ext.groupby("symbol"); ext["prev_date"] = g["date"].shift(1); ext["close_h_prev"] = g["close_h"].shift(1)
cal = np.sort(sub["date"].unique()); pos = np.searchsorted(cal, sub["date"].to_numpy("datetime64[ns]"), side="right")
D = np.full(len(sub), np.datetime64("NaT"), dtype="datetime64[ns]"); ok = pos < len(cal); D[ok] = cal[pos[ok]]
m = pd.DataFrame({"symbol": sub["symbol"].to_numpy(), "D": D}).merge(
    ext.rename(columns={"date": "D"})[["symbol", "D", "prev_date", "close_h_prev", "pm_last8"]], on=["symbol", "D"], how="left")
cp = m["close_h_prev"].to_numpy(float)
aligned = (pd.to_datetime(m["prev_date"]).to_numpy("datetime64[ns]") == sub["date"].to_numpy("datetime64[ns]")) & np.isfinite(cp) & (cp > 0)
with np.errstate(invalid="ignore", divide="ignore"):
    gap = np.where(aligned, m["pm_last8"].to_numpy(float) / cp - 1, np.nan)
sub["gap_open"] = gap
rng = np.random.default_rng(11)
deg = gap.copy(); fin = np.isfinite(deg); deg[fin & (rng.random(len(deg)) < 0.25)] = np.nan
p = R.walk_forward(sub, list(F.FEATURES) + ["gap_open"], test_over={"gap_open": deg})
pa = np.load(OUT / "oos_r9_A_m0.npy"); p3 = np.load(OUT / "oos_r10_D3_8.npy")
res = {k: R.evaluate(sub, v) for k, v in (("A_m0", pa), ("D3_8", p3), ("D3_8_degraded25", p))}
for k, v in res.items():
    R.log(k, " | ".join(f"{per} hit {v[per]['top1']['hit']:.3f} mean {v[per]['top1']['mean_oc']:+.4f}" for per in ("dev", "confirm")))
ok_ = all(res["D3_8_degraded25"][per]["top1"]["mean_oc"] <= res["A_m0"][per]["top1"]["mean_oc"] for per in ("dev", "confirm"))
res["decision"] = {"guard_pm_ratio": 0.70 if ok_ else 0.85}
R.log("DECISION guard_pm_ratio:", res["decision"]["guard_pm_ratio"])
(OUT / "r10d.json").write_text(json.dumps(res, indent=1, default=float))
R.log("DONE")
