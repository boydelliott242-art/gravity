"""Round 10 — can the morning model run at 07:20 CT? Compare the leak-free
morning model (D2) with the pre-market window ending 8:00 ET (D2_8) vs 9:00 ET
(D2_9), and the close-only model. Rule: scripts/research/r9_selection_rule.txt (R10).
Input: data/research/ext_hours2.pkl (fetch_ext_hours2.py). Output: r10.json"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402
from gravity import features as F  # noqa: E402

OUT = R.OUT
SHAPE = ["ext_hi", "ext_lo", "ext_fade", "ah_ret"]


def add_ext2(sub, suffix):
    """Leak-free features from ext_hours2 for the given cutoff ('' = 9:00 ET, '8' = 8:00 ET)."""
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
    cols = ["prev_date", "close_h_prev", "ah_last", f"ext_last{suffix}", f"ext_high{suffix}", f"ext_low{suffix}"]
    m = key.merge(ext.rename(columns={"date": "D"})[["symbol", "D"] + cols], on=["symbol", "D"], how="left")
    cp = m["close_h_prev"].to_numpy(float)
    aligned = (pd.to_datetime(m["prev_date"]).to_numpy("datetime64[ns]") == sub["date"].to_numpy("datetime64[ns]")) & np.isfinite(cp) & (cp > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        el, eh, elo, ah = (m[c].to_numpy(float) for c in (f"ext_last{suffix}", f"ext_high{suffix}", f"ext_low{suffix}", "ah_last"))
        feats = {"gap_open": el / cp - 1, "ext_hi": eh / cp - 1, "ext_lo": elo / cp - 1, "ext_fade": el / eh - 1, "ah_ret": ah / cp - 1}
    return {k: np.where(aligned, v, np.nan) for k, v in feats.items()}


def main():
    t0 = time.time()
    sub = R.build_sub()
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    # hourly path columns for timing are not needed here; daily_top needs them present
    for c in ("h_open", "c1030", "c1230", "close_h", "hi1030", "hi_all", "path_ok", "gap_ext"):
        if c not in sub.columns:
            sub[c] = np.nan
    base = list(F.FEATURES)
    preds = {"A_m0": np.load(OUT / "oos_r9_A_m0.npy")}
    for name, suffix in (("D2_9", ""), ("D2_8", "8")):
        f = OUT / f"oos_r10_{name}.npy"
        fe = add_ext2(sub, suffix)
        for k, v in fe.items():
            sub[k] = v
        cov = float(np.isfinite(sub["gap_open"]).mean())
        if f.exists():
            preds[name] = np.load(f)
        else:
            t1 = time.time()
            preds[name] = R.walk_forward(sub, base + ["gap_open"] + SHAPE)
            np.save(f, preds[name])
            R.log(f"{name}: {time.time() - t1:.0f}s (ext coverage {cov:.3f})")
    res = {"variants": {}, "paired_vs_D2_9": {}, "paired_vs_A": {}}
    for name, p in preds.items():
        res["variants"][name] = R.evaluate(sub, p)
        if name != "D2_9":
            res["paired_vs_D2_9"][name] = R.paired(sub, p, preds["D2_9"])
        if name != "A_m0":
            res["paired_vs_A"][name] = R.paired(sub, p, preds["A_m0"])
        v = res["variants"][name]
        R.log(name, " | ".join(f"{per} hit {v[per]['top1']['hit']:.3f} mean {v[per]['top1']['mean_oc']:+.4f} sqz {v[per]['top1']['squeeze']:.3f} net10k {v[per]['top1']['net_10k']:+.4f}" for per in ("dev", "confirm")))
    a, d9, d8 = (res["variants"][k] for k in ("A_m0", "D2_9", "D2_8"))
    adopt = all(d8[per]["top1"]["mean_oc"] <= a[per]["top1"]["mean_oc"] and d8[per]["top1"]["mean_oc"] - d9[per]["top1"]["mean_oc"] <= 0.01
                for per in ("dev", "confirm"))
    res["decision"] = {"adopt_8am_cutoff": bool(adopt)}
    R.log("DECISION adopt 8:00 ET cutoff:", adopt)
    (OUT / "r10.json").write_text(json.dumps(res, indent=1, default=float))
    R.log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
