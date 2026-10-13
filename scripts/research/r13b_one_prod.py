"""R13b — the adopted ONE recipe (O4: rank-average of three dump models within
the $30K-ready universe, #1 among $100K-ready names) measured the way production
runs it, with the current (R14-calibrated) cost model. Reporting only.

- evening / feed-fallback: every component close-only (A_m0, L1_m0, L2_m0)
- morning hybrid: components use the morning (pre-market gap) model for the
  shortlist — the top 450 $30K-ready names by the close-only $30K ranker plus
  the top 150 $100K-ready names by the close-only ONE score — close-only for the rest
- conviction tiers by RANK agreement (robust to the universe size): sum over the
  three models of (rank among $30K-ready names − 1); high ≤ 2, medium ≤ 7 (the R13
  DEV 80th / 50th percentile thresholds expressed in ranks)
Output: data/research/r13b.json"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r11_liquid as X  # noqa: E402
import r13_one as R13  # noqa: E402
from gravity import capacity as CAP, features as F  # noqa: E402

OUT = X.OUT
log = X.log
HIGH, MEDIUM = 2, 7


def ranks_in_L(sub, p):
    """Position from the top (1 = highest) of each L row among that session's L rows."""
    L = sub["L"].to_numpy()
    out = np.full(len(sub), np.nan)
    s = pd.Series(np.asarray(p, float)[L])
    out[np.flatnonzero(L)] = s.groupby(sub["date"].to_numpy()[L]).rank(ascending=False, method="min").to_numpy()
    return out


def one_score(sub, comps):
    """Rank-average (percentiles within the session's L rows) and the rank deficit."""
    sc = X.rank_avg(sub, comps)
    deficit = np.sum([ranks_in_L(sub, c) - 1 for c in comps], axis=0)
    return sc, deficit


def tiers(sub, score, deficit, U):
    m = U & np.isfinite(score)
    d = pd.DataFrame({"date": sub["date"].to_numpy()[m], "s": score[m], "dfc": deficit[m],
                      "y": sub["y_dump"].to_numpy()[m], "oc": sub["y_oc"].to_numpy()[m],
                      "oh": sub["y_oh"].to_numpy()[m]}).sort_values(["date", "s"], ascending=[True, False])
    top = d.groupby("date").head(1)
    out = {}
    for per, tm in (("dev", top.date < X.SPLIT), ("confirm", top.date >= X.SPLIT), ("all", top.date.notna())):
        t = top[tm]
        for name, sel in (("high", t.dfc <= HIGH), ("medium", (t.dfc > HIGH) & (t.dfc <= MEDIUM)), ("low", t.dfc > MEDIUM)):
            x = t[sel]
            out.setdefault(name, {})[per] = {"days": int(len(x)), "share": float(len(x) / max(len(t), 1)),
                                             "hit": float(x.y.mean()) if len(x) else None,
                                             "mean_oc": float(x.oc.mean()) if len(x) else None,
                                             "big10": float((x.oc <= -0.10).mean()) if len(x) else None,
                                             "squeeze": float((x.oh >= 0.20).mean()) if len(x) else None}
    return out


def main():
    t0 = time.time()
    sub = X.prep()
    px = np.where(np.isfinite(sub["price"].to_numpy(float)), sub["price"].to_numpy(float), sub["close"].to_numpy(float))
    U = sub["pub"].to_numpy() & (px >= 1.0) & (sub["cap"].to_numpy(float) >= R13.FLOOR)
    for q, c in ((100_000, "c100"), (500_000, "c500")):
        sub[c] = np.asarray(CAP.round_trip_cost(q, sub["dvx"].to_numpy(float), sub["vol20"].to_numpy(float), sub["spr"].to_numpy(float)), float)
    sub["y_d3"] = np.where(np.isfinite(sub["y_oc"]), (sub["y_oc"] <= -0.03).astype(float), np.nan)
    L = sub["L"].to_numpy()
    base = list(F.FEATURES)
    P = {}
    for name, target in (("L1_m0", "y_dump"), ("L2_m0", "y_d3")):
        f = OUT / f"oos_r13b_{name}.npy"
        if f.exists():
            P[name] = np.load(f)
        else:
            t1 = time.time(); P[name] = X.walk(sub, base, L, "cls", target); np.save(f, P[name]); log(f"{name}: {time.time() - t1:.0f}s")
    m1 = [np.load(OUT / "oos_r10_D3_8.npy"), np.load(OUT / "oos_r11_L1.npy"), np.load(OUT / "oos_r11_L2.npy")]
    m0 = [np.load(OUT / "oos_r9_A_m0.npy"), P["L1_m0"], P["L2_m0"]]
    s1, d1 = one_score(sub, m1)
    s0, d0 = one_score(sub, m0)
    # production shortlist for the morning model
    liq0 = np.load(OUT / "oos_r11_L4_m0.npy")
    day = sub["date"].to_numpy()
    sl = np.zeros(len(sub), bool)
    r_liq = pd.Series(np.where(L, liq0, np.nan)).groupby(day).rank(ascending=False, method="first").to_numpy()
    r_one = pd.Series(np.where(U, s0, np.nan)).groupby(day).rank(ascending=False, method="first").to_numpy()
    sl |= np.nan_to_num(r_liq, nan=1e9) <= 450
    sl |= np.nan_to_num(r_one, nan=1e9) <= 150
    hyb = [np.where(sl, a, b) for a, b in zip(m1, m0)]
    sh, dh = one_score(sub, hyb)
    res = {"tier_rule": {"high_max_deficit": HIGH, "medium_max_deficit": MEDIUM},
           "variants": {"O4_morning_pure": R13.evaluate_one(sub, s1, U), "O4_evening_m0": R13.evaluate_one(sub, s0, U),
                        "O4_morning_hybrid": R13.evaluate_one(sub, sh, U), "O0_production": R13.evaluate_one(sub, m1[0], U)},
           "tiers": {"O4_morning_hybrid": tiers(sub, sh, dh, U), "O4_evening_m0": tiers(sub, s0, d0, U),
                     "O4_morning_pure": tiers(sub, s1, d1, U)}}
    for k, v in res["variants"].items():
        log(f"{k}: " + " | ".join(f"{per} #1 hit {v[per]['top1']['hit']*100:.1f} big {v[per]['top1']['big10']*100:.1f} mean {v[per]['top1']['mean_oc']*100:+.2f} "
                                  f"net100k {v[per]['top1']['net100k']*100:+.2f} net500k {v[per]['top1']['net500k']*100:+.2f}" for per in ("dev", "confirm")))
    for k, v in res["tiers"].items():
        log(f"tiers {k}: " + json.dumps({t: {per: (x[per]['days'], round(x[per]['hit'], 3) if x[per]['hit'] is not None else None,
                                                    round(x[per]['mean_oc'], 4) if x[per]['mean_oc'] is not None else None)
                                             for per in ('dev', 'confirm')} for t, x in v.items()}))
    (OUT / "r13b.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
