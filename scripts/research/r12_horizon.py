"""R12 — does holding longer pay for a $30K board? (rule: r9_selection_rule.txt, R12)
Same universe L and L3 recipe as R11, target = next open → close H sessions later (H = 5, 10). Output: r12.json"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r11_liquid as X  # noqa: E402
import r9_premarket as R  # noqa: E402
from gravity import model as M  # noqa: E402
from gravity import features as F  # noqa: E402

OUT = R.OUT
BORROW_YR = 0.05


def walk_h(sub, feats, target, H, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    s = M._settings(1, {"embargo": max(5, H + 1)})   # +1: a halted symbol's label can run past H sessions
    dates = np.unique(sub["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, sub["date"].to_numpy("datetime64[ns]"))
    Xm = sub[feats].to_numpy(np.float32)
    y = sub[target].to_numpy(float)
    L = sub["L"].to_numpy()
    tr = L & np.isfinite(y)
    oos = np.full(len(sub), np.nan)
    rng = np.random.default_rng(seed * 100)
    for fw in M._fold_windows(dates, s):
        idx = np.flatnonzero(tr & (di <= fw["train_end"]))
        te = np.flatnonzero(L & (di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te) or len(idx) < 1000:
            continue
        if len(idx) > s["max_fit_rows"]:
            idx = np.sort(rng.choice(idx, int(s["max_fit_rows"]), replace=False))
        sb = json.loads(json.dumps(s)); sb["hgb"]["random_state"] = seed
        reg = HistGradientBoostingRegressor(**sb["hgb"]).fit(Xm[idx], y[idx])
        oos[te] = -reg.predict(Xm[te])
    return oos


def main():
    t0 = time.time()
    sub = X.prep()
    L = sub["L"].to_numpy()
    feats = list(F.FEATURES) + ["gap_open"]
    res = {"same_day_L3": X.evaluate(sub, np.load(OUT / "oos_r11_L3.npy"))}
    for H, col in ((5, "fwd5"), (10, "fwd10")):
        y = sub[col].to_numpy(float)
        t = pd.DataFrame({"date": sub["date"].to_numpy()[L], "y": y[L]})
        med = t.groupby("date")["y"].transform("median").to_numpy()
        sub[f"y_dm{H}"] = np.nan
        sub.loc[L, f"y_dm{H}"] = np.clip(y[L] - med, -0.5, 0.5)
        f = OUT / f"oos_r12_H{H}.npy"
        if f.exists():
            p = np.load(f)
        else:
            p = walk_h(sub, feats, f"y_dm{H}", H); np.save(f, p)
        b = X.board(sub, p, L).join(sub[[col]], how="left")
        cost = b["c30"].to_numpy(float) + BORROW_YR * H / 252
        b["net"] = -b[col].to_numpy(float) - cost
        out = {}
        for per, pm in (("dev", b.date < X.SPLIT), ("confirm", b.date >= X.SPLIT)):
            x = b[pm.to_numpy()]
            r = {}
            for lo, hi in X.BANDS:
                z = x[(x.rk >= lo) & (x.rk <= hi)]
                r[f"{lo}-{hi}"] = {"n": int(len(z)), "gross": float(-z[col].mean()), "median_gross": float(-z[col].median()),
                                   "net": float(z.net.mean()), "net_per_session": float(z.net.mean() / H),
                                   "win": float((z.net > 0).mean()), "p95_adverse": float(z[col].quantile(0.95))}
            out[per] = r
        res[f"H{H}"] = out
        R.log(f"H{H}: " + " | ".join(f"{per} ALL100 net {out[per]['1-100']['net'] * 100:+.2f} (/session {out[per]['1-100']['net_per_session'] * 100:+.3f}) "
                                      f"top10 {out[per]['1-10']['net'] * 100:+.2f} #1 {out[per]['1-1']['net'] * 100:+.2f}" for per in ("dev", "confirm")))
    sd = res["same_day_L3"]
    offer = None
    for H in (5, 10):
        o = res[f"H{H}"]
        if all(o[per]["1-100"]["net"] > 0 and o[per]["1-100"]["net_per_session"] > sd[per]["1-100"]["net30"] for per in ("dev", "confirm")):
            offer = H
            break
    res["decision"] = {"offer_swing_H": offer}
    R.log("RULE: offer $30K swing board with H =", offer)
    (OUT / "r12.json").write_text(json.dumps(res, indent=1, default=float))
    R.log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
