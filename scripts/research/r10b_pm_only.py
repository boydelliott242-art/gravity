"""R10b — morning model from PRE-MARKET trades only (8:00 ET cutoff), robust to
Yahoo's after-hours outage. Rule: scripts/research/r9_selection_rule.txt (R10b)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
import r10_cutoff as R10  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402
from gravity import features as F  # noqa: E402

OUT = R.OUT


def main():
    t0 = time.time()
    sub = R.build_sub()
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    for c in ("h_open", "c1030", "c1230", "close_h", "hi1030", "hi_all", "path_ok", "gap_ext"):
        if c not in sub.columns:
            sub[c] = np.nan
    ext = pd.read_pickle(OUT / "ext_hours2.pkl")
    ext["date"] = pd.to_datetime(ext["date"])
    ext = ext.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = ext.groupby("symbol")
    ext["prev_date"] = g["date"].shift(1)
    ext["close_h_prev"] = g["close_h"].shift(1)
    cal = np.sort(sub["date"].unique())
    pos = np.searchsorted(cal, sub["date"].to_numpy("datetime64[ns]"), side="right")
    D = np.full(len(sub), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(cal); D[ok] = cal[pos[ok]]
    key = pd.DataFrame({"symbol": sub["symbol"].to_numpy(), "D": D})
    m = key.merge(ext.rename(columns={"date": "D"})[["symbol", "D", "prev_date", "close_h_prev", "pm_last8"]], on=["symbol", "D"], how="left")
    cp = m["close_h_prev"].to_numpy(float)
    aligned = (pd.to_datetime(m["prev_date"]).to_numpy("datetime64[ns]") == sub["date"].to_numpy("datetime64[ns]")) & np.isfinite(cp) & (cp > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sub["gap_open"] = np.where(aligned, m["pm_last8"].to_numpy(float) / cp - 1, np.nan)
    R.log("pre-market-only gap coverage", float(np.isfinite(sub["gap_open"]).mean()))
    f = OUT / "oos_r10_D3_8.npy"
    if f.exists():
        p3 = np.load(f)
    else:
        p3 = R.walk_forward(sub, list(F.FEATURES) + ["gap_open"])
        np.save(f, p3)
    preds = {"A_m0": np.load(OUT / "oos_r9_A_m0.npy"), "D2_8": np.load(OUT / "oos_r10_D2_8.npy"), "D3_8": p3}
    res = {"variants": {k: R.evaluate(sub, p) for k, p in preds.items()},
           "paired_vs_A": {k: R.paired(sub, p, preds["A_m0"]) for k, p in preds.items() if k != "A_m0"}}
    for k, v in res["variants"].items():
        R.log(k, " | ".join(f"{per} hit {v[per]['top1']['hit']:.3f} mean {v[per]['top1']['mean_oc']:+.4f} sqz {v[per]['top1']['squeeze']:.3f} net10k {v[per]['top1']['net_10k']:+.4f}" for per in ("dev", "confirm")))
    a, d2, d3 = (res["variants"][k] for k in ("A_m0", "D2_8", "D3_8"))
    adopt = all(d3[per]["top1"]["mean_oc"] <= a[per]["top1"]["mean_oc"] and d3[per]["top1"]["mean_oc"] - d2[per]["top1"]["mean_oc"] <= 0.005
                for per in ("dev", "confirm"))
    res["decision"] = {"adopt_pm_only": bool(adopt)}
    R.log("DECISION adopt pre-market-only (D3_8):", adopt)
    (OUT / "r10b.json").write_text(json.dumps(res, indent=1, default=float))
    R.log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
