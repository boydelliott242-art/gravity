"""Round 9 — what the morning run can know at ~9:00 ET, tested honestly.

Questions (each variant walk-forward, chosen on dev < 2025-11-01, reported on
the later confirmation window):

  A  M0                      close-only features (the evening model)
  B  M1 as it runs live      trained on the official 9:30 open gap, but live it
                             is fed the last extended-hours price (~9:00 ET);
                             names with no extended-hours trade fall back to M0
  B' M1 as backtested        same model, tested on the official open gap (what
                             the site has reported — information not available
                             at 9:05 ET, so optimistic)
  C  M1 honest               trained AND tested on the ~9:00 ET price
  D  C + extended-hours shape   high/low/fade/activity/after-hours move
  E  D + overnight filings   counts of filings accepted between the 16:00 close
                             and 9:00 ET, by type (same forms the live feed scans)

Then, for the best honest variant and for B: entry/exit timing from hourly
bars, and top-K baskets sized within each name's capacity.

Inputs: data/research/wide_lab.pkl (R1), ext_hours.pkl (fetch_ext_hours.py),
data/state/context.pkl (SEC events with acceptance times).
Output: data/research/r9.json, oos_r9_<variant>.npy
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import size_tier as R1  # noqa: E402
from gravity import capacity as CAP, cli, model as M  # noqa: E402
from gravity import features as F  # noqa: E402
from gravity.sources import prices, sec, universe  # noqa: E402

OUT = R1.OUT
SPLIT = pd.Timestamp("2025-11-01")
ET = "America/New_York"
QUICK = "--quick" in sys.argv
LEAK_FREE = "--leak-free" in sys.argv
EXT_FEATS = ["ext_hi", "ext_lo", "ext_fade", "n_ext", "n_pm", "ah_ret"]
ON_GROUPS = ["filings", "offer", "reg", "effect", "unreg", "delist", "finance", "current", "f144", "late"]
ON_FEATS = [f"on_{g}" for g in ON_GROUPS]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ── data ────────────────────────────────────────────────────────────────
def build_sub() -> pd.DataFrame:
    cache = OUT / "pit_sub.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    lab = pd.read_pickle(OUT / "wide_lab.pkl")
    uni = universe.load_universe(max_cap=float("inf")).drop_duplicates("symbol")
    shares_today = (uni.set_index("symbol")["market_cap"] / uni.set_index("symbol")["price"]).replace([np.inf, -np.inf], np.nan)
    small_today = set(uni.loc[uni["market_cap"] <= 2e9, "symbol"])
    splits = prices.load_splits(sorted(lab["symbol"].unique()))
    pit = cli._pit_market_cap(lab, sec.shares_frames(), sec.cik_map(), splits, shares_today)
    keep = ((pit <= 2e9) | (pit.isna() & lab["symbol"].isin(small_today))).to_numpy()
    sub = lab[keep & lab["y_oc"].notna().to_numpy()].reset_index(drop=True)
    del lab
    F._cross_sectional(sub)
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    sub.to_pickle(cache)
    return sub


def add_ext(sub: pd.DataFrame) -> pd.DataFrame:
    f = OUT / "ext_hours.pkl"
    if f.exists():
        ext = pd.read_pickle(f)
    else:                                         # smoke test before the fetch finishes
        assert QUICK, "ext_hours.pkl missing — run fetch_ext_hours.py first"
        ext = pd.concat([x for x in (pd.read_pickle(q) for q in sorted((OUT / "ext_parts").glob("*.pkl"))) if len(x)])
    ext["date"] = pd.to_datetime(ext["date"])
    have = set(ext["symbol"].unique())
    cal = np.sort(sub["date"].unique())
    pos = np.searchsorted(cal, sub["date"].to_numpy("datetime64[ns]"), side="right")
    sub["D"] = pd.NaT
    ok = pos < len(cal)
    sub.loc[ok, "D"] = cal[pos[ok]]
    ext = ext.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = ext.groupby("symbol")
    ext["prev_date"] = g["date"].shift(1)          # previous session in the HOURLY series
    ext["close_h_prev"] = g["close_h"].shift(1)    # its regular close, same fetch / split basis
    nxt = ext.rename(columns={"date": "D"})
    sub = sub.merge(nxt, on=["symbol", "D"], how="left")
    cur = ext[["symbol", "date", "close_h"]].rename(columns={"close_h": "close_h_t"})
    sub = sub.merge(cur, on=["symbol", "date"], how="left")
    close = sub["close"].to_numpy(float)
    open_d = close * (1 + sub["y_gap"].to_numpy(float))
    # Yahoo's hourly series must agree with the daily bars (same split basis)
    agree_t = np.abs(sub["close_h_t"].to_numpy(float) / close - 1) <= 0.03
    agree_o = np.abs(sub["h_open"].to_numpy(float) / open_d - 1) <= 0.03
    if LEAK_FREE:
        # leak-free (R9c): ratios to the hourly series' OWN prior close (one split basis), kept only when
        # the hourly previous session IS the panel date — no agreement check against the daily bars, whose
        # failures cluster on names that reverse-split later (future information)
        cp = sub["close_h_prev"].to_numpy(float)
        aligned = (pd.to_datetime(sub["prev_date"]).to_numpy("datetime64[ns]") == sub["date"].to_numpy("datetime64[ns]")) & np.isfinite(cp) & (cp > 0)
        sub["ext_ok"] = sub["symbol"].isin(have).to_numpy() & aligned
        base = cp
    else:
        sub["ext_ok"] = sub["symbol"].isin(have).to_numpy() & agree_t
        base = close
    sub["path_ok"] = (sub["symbol"].isin(have).to_numpy() & agree_t) & agree_o
    eo = sub["ext_ok"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        sub["gap_ext"] = np.where(eo, sub["ext_last"].to_numpy(float) / base - 1, np.nan)
        sub["ext_hi"] = np.where(eo, sub["ext_high"].to_numpy(float) / base - 1, np.nan)
        sub["ext_lo"] = np.where(eo, sub["ext_low"].to_numpy(float) / base - 1, np.nan)
        sub["ext_fade"] = np.where(eo, sub["ext_last"].to_numpy(float) / sub["ext_high"].to_numpy(float) - 1, np.nan)
        sub["ah_ret"] = np.where(eo, sub["ah_last"].to_numpy(float) / base - 1, np.nan)
    for c in ("n_ext", "n_pm"):
        sub[c] = np.where(eo, sub[c].fillna(0).to_numpy(float), np.nan)
    return sub


def add_overnight(sub: pd.DataFrame) -> pd.DataFrame:
    ctx = pickle.load(open(cli.CONTEXT_PATH, "rb"))
    events = ctx.get("events") or {}
    codes = {s: i for i, s in enumerate(sorted(sub["symbol"].unique()))}
    K = np.int64(10 ** 8)                         # minutes since epoch < 1e8 until ~2160
    keys = {g: [] for g in ON_GROUPS}
    for sym, evs in events.items():
        c = codes.get(sym)
        if c is None:
            continue
        for e in evs:
            form = str(e.get("form") or "").strip().upper()
            if form not in sec.CURRENT_FORMS or not e.get("accepted"):
                continue
            try:
                ts = pd.Timestamp(e["accepted"]).tz_convert("UTC")
            except Exception:
                continue
            mins = np.int64(ts.value // 60_000_000_000)
            for g in F.event_groups(e):
                if g in keys:
                    keys[g].append(c * K + mins)
    rc = sub["symbol"].map(codes).to_numpy(np.int64)
    t0 = pd.DatetimeIndex(sub["date"]).tz_localize(ET) + pd.Timedelta(hours=16)
    dd = pd.DatetimeIndex(sub["D"])
    t1 = dd.tz_localize(ET) + pd.Timedelta(hours=9)
    m0 = (t0.tz_convert("UTC").asi8 // 60_000_000_000).astype(np.int64)
    m1 = np.where(dd.isna(), m0, (t1.tz_convert("UTC").asi8 // 60_000_000_000)).astype(np.int64)
    has = sub["symbol"].isin(set(events)).to_numpy()
    for g in ON_GROUPS:
        ek = np.sort(np.asarray(keys[g], dtype=np.int64))
        lo = np.searchsorted(ek, rc * K + m0, side="left")
        hi = np.searchsorted(ek, rc * K + m1, side="left")
        sub[f"on_{g}"] = np.where(has, (hi - lo).astype(float), np.nan)
    return sub


# ── models ──────────────────────────────────────────────────────────────
def walk_forward(sub, feats, test_over=None, seed=0):
    """OOS P(dump) — production recipe (R1.walk_forward) but the TEST matrix
    may differ from the training one (``test_over``: column → array)."""
    s = M._settings(1, None)
    dates = np.unique(sub["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, sub["date"].to_numpy("datetime64[ns]"))
    X = sub[feats].to_numpy(np.float32)
    Xt = X
    if test_over:
        Xt = X.copy()
        for c, arr in test_over.items():
            Xt[:, feats.index(c)] = np.asarray(arr, np.float32)
    y = sub["y_dump"].to_numpy(float)
    oos = np.full(len(sub), np.nan)
    for fw in M._fold_windows(dates, s):
        fit_idx = np.flatnonzero(di <= fw["fit_end"])
        cal_idx = np.flatnonzero((di >= fw["cal_start"]) & (di <= fw["train_end"]))
        te_idx = np.flatnonzero((di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or len(cal_idx) < 200:
            continue
        sb = json.loads(json.dumps(s)); sb["hgb"]["random_state"] = seed
        clf = M._fit_classifier(X[fit_idx], y[fit_idx], sb, np.random.default_rng(seed * 100))
        rc, rt = M._raw(clf, Xt[cal_idx]), M._raw(clf, Xt[te_idx])
        iso = M._fit_iso(rc, y[cal_idx])
        cal = iso.predict(rt) if iso is not None else rt
        oos[te_idx] = (1 - M.TIE_EPS) * cal + M.TIE_EPS * rt
    return oos


# ── evaluation ──────────────────────────────────────────────────────────
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(p) & np.isfinite(y)
    return float(roc_auc_score(y[m], p[m])) if m.sum() > 100 and len(np.unique(y[m])) == 2 else None


def daily_top(sub, p, k=1):
    m = sub["pub"].to_numpy() & np.isfinite(p)
    d = sub.loc[m, ["date", "symbol", "y_dump", "y_oc", "y_oh", "dvx", "vol20", "spr", "cap", "close",
                    "h_open", "c1030", "c1230", "close_h", "hi1030", "hi_all", "path_ok", "gap_ext"]].assign(p=p[m])
    d = d.sort_values(["date", "p"], ascending=[True, False])
    d["rk"] = d.groupby("date").cumcount() + 1
    return d[d.rk <= k]


def summarize(top, label=""):
    r = {"days": int(top["date"].nunique()), "n": int(len(top)),
         "hit": float(top.y_dump.mean()), "mean_oc": float(top.y_oc.mean()), "median_oc": float(top.y_oc.median()),
         "squeeze": float((top.y_oh >= 0.20).mean())}
    for q in (10_000, 50_000):
        c = np.asarray(CAP.round_trip_cost(q, top.dvx.to_numpy(float), top.vol20.to_numpy(float), top.spr.to_numpy(float)), float)
        r[f"net_{q // 1000}k"] = float((-top.y_oc.to_numpy() - c).mean())
        r[f"win_{q // 1000}k"] = float(((-top.y_oc.to_numpy() - c) > 0).mean())
    return r


def evaluate(sub, p):
    out = {}
    for per, pm in (("dev", sub["date"] < SPLIT), ("confirm", sub["date"] >= SPLIT)):
        pmask = pm.to_numpy()
        pp = np.where(pmask, p, np.nan)
        res = {"auc_pub": auc(sub["y_dump"].to_numpy(float)[sub["pub"].to_numpy() & pmask], pp[sub["pub"].to_numpy() & pmask])}
        for k in (1, 3, 10):
            res[f"top{k}"] = summarize(daily_top(sub, pp, k))
        out[per] = res
    return out


def paired(sub, p_new, p_base):
    """Per-day #1 short return (−oc) new minus base: mean, bootstrap 90% CI, share of days the pick differs."""
    res = {}
    for per, pm in (("dev", sub["date"] < SPLIT), ("confirm", sub["date"] >= SPLIT)):
        pmask = pm.to_numpy()
        a = daily_top(sub, np.where(pmask, p_new, np.nan)).set_index("date")
        b = daily_top(sub, np.where(pmask, p_base, np.nan)).set_index("date")
        j = a.join(b, lsuffix="_n", rsuffix="_b", how="inner")
        diff = (-j.y_oc_n.to_numpy()) - (-j.y_oc_b.to_numpy())
        rng = np.random.default_rng(7)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)] if len(diff) else [np.nan]
        res[per] = {"days": int(len(j)), "differs": float((j.symbol_n != j.symbol_b).mean()) if len(j) else None,
                    "mean_diff": float(diff.mean()) if len(diff) else None,
                    "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                    "hit_diff": float(j.y_dump_n.mean() - j.y_dump_b.mean()) if len(j) else None}
    return res


def timing(sub, p):
    """Short entry/exit timing for the published #1 using hourly bars (rows
    whose hourly series agrees with the daily bars only)."""
    res = {}
    for per, pm in (("dev", sub["date"] < SPLIT), ("confirm", sub["date"] >= SPLIT)):
        t = daily_top(sub, np.where(pm.to_numpy(), p, np.nan))
        t = t[t.path_ok.astype(bool) & t.h_open.notna() & t.c1030.notna() & t.close_h.notna()]
        o, c10, c12, cl = (t[c].to_numpy(float) for c in ("h_open", "c1030", "c1230", "close_h"))
        r = {"n": int(len(t))}
        def st(x):
            x = x[np.isfinite(x)]
            return {"mean": float(x.mean()), "median": float(np.median(x)), "win": float((x > 0).mean()), "n": int(len(x))} if len(x) else None
        r["open→close"] = st(1 - cl / o)
        r["open→10:30"] = st(1 - c10 / o)
        r["open→12:30"] = st(1 - c12 / o)
        r["10:30→close"] = st(1 - cl / c10)
        r["10:30→close | up at 10:30"] = st(np.where(c10 > o, 1 - cl / c10, np.nan))
        r["10:30→close | down at 10:30"] = st(np.where(c10 <= o, 1 - cl / c10, np.nan))
        r["open→close | up at 10:30"] = st(np.where(c10 > o, 1 - cl / o, np.nan))
        r["open→close | down at 10:30"] = st(np.where(c10 <= o, 1 - cl / o, np.nan))
        res[per] = r
    return res


def baskets(sub, p):
    """Top-K basket, each name sized at min(capacity, S): $ deployed/day and
    net return per $ after each name's own estimated cost."""
    res = {}
    for per, pm in (("dev", sub["date"] < SPLIT), ("confirm", sub["date"] >= SPLIT)):
        t = daily_top(sub, np.where(pm.to_numpy(), p, np.nan), 10)
        for k in (1, 3, 5, 10):
            tk = t[t.rk <= k]
            for S in (10_000, 25_000, 50_000):
                size = np.minimum(tk.cap.to_numpy(float), S)
                size = np.where(size >= 1000, size, 0.0)
                cost = np.asarray(CAP.round_trip_cost(np.maximum(size, 1), tk.dvx.to_numpy(float), tk.vol20.to_numpy(float), tk.spr.to_numpy(float)), float)
                pnl = size * (-tk.y_oc.to_numpy(float) - cost)
                day = pd.DataFrame({"date": tk.date.to_numpy(), "dep": size, "pnl": pnl}).groupby("date").sum()
                day = day[day.dep > 0]
                rpd = day.pnl / day.dep
                res[f"{per}|top{k}|{S // 1000}k"] = {
                    "days": int(len(day)), "deployed_mean": float(day.dep.mean()), "deployed_median": float(day.dep.median()),
                    "net_per_dollar": float(day.pnl.sum() / day.dep.sum()), "day_win": float((day.pnl > 0).mean()),
                    "t_stat": float(rpd.mean() / (rpd.std(ddof=1) / np.sqrt(len(rpd)))) if len(rpd) > 2 else None,
                    "worst_day": float(rpd.min()), "pnl_per_day": float(day.pnl.mean())}
    return res


def main():
    t0 = time.time()
    sub = build_sub()
    # always the CURRENT cost model (the cached frame may carry an older spread rule)
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    log(f"PIT sub {len(sub):,} rows")
    if QUICK:
        keep = set(sorted(sub.symbol.unique())[:400:3])
        sub = sub[sub.symbol.isin(keep)].reset_index(drop=True)
        log(f"QUICK: {len(sub):,} rows")
    sub = add_ext(sub)
    sub = add_overnight(sub)
    sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    cov = {
        "ext_ok": float(sub.ext_ok.mean()), "gap_ext_finite": float(np.isfinite(sub.gap_ext).mean()),
        "gap_ext_finite_pub": float(np.isfinite(sub.gap_ext[sub.pub]).mean()),
        "path_ok": float(sub.path_ok.mean()),
        "corr_gap_ext_vs_open": float(pd.Series(sub.gap_ext).corr(sub.y_gap)),
        "by_year_ext": {str(y): float(np.isfinite(sub.gap_ext[sub.date.dt.year == y]).mean()) for y in sorted(sub.date.dt.year.unique())},
        "on_any_rate": float((sub.on_filings > 0).mean()), "on_offer_rate": float((sub.on_offer > 0).mean()),
        "dump_if_on_offer": float(sub.y_dump[sub.on_offer > 0].mean()) if (sub.on_offer > 0).any() else None,
        "dump_base": float(sub.y_dump.mean()),
    }
    log("coverage", json.dumps(cov))
    tag = "_quick" if QUICK else ""
    preds = {}

    def run(name, feats, test_over=None):
        f = OUT / f"oos_r9_{name}{tag}.npy"
        if f.exists():
            preds[name] = np.load(f)
            log(f"{name}: cached")
            return
        t1 = time.time()
        preds[name] = walk_forward(sub, feats, test_over)
        np.save(f, preds[name])
        log(f"{name}: {time.time() - t1:.0f}s")

    sub["gap_open"] = sub["y_gap"]
    base = list(F.FEATURES)
    if LEAK_FREE:
        tag2 = tag + "_lf"
        def run_lf(name, feats, test_over=None):
            run(name + "_lf", feats, test_over)
        run("A_m0", base)
        run_lf("B_m1_live", base + ["gap_open"], test_over={"gap_open": sub["gap_ext"].to_numpy(float)})
        sub["gap_open"] = sub["gap_ext"]
        run_lf("C_m1_honest", base + ["gap_open"])
        run_lf("D_ext_shape", base + ["gap_open"] + EXT_FEATS)
        run_lf("D2_no_counts", base + ["gap_open"] + [c for c in EXT_FEATS if c not in ("n_ext", "n_pm")])
        has_ext = np.isfinite(sub["gap_ext"].to_numpy(float))
        preds["B_m1_live_lf"] = np.where(has_ext, preds["B_m1_live_lf"], preds["A_m0"])
        res = {"coverage": {"ext_ok": float(sub.ext_ok.mean()), "gap_ext_finite": float(np.isfinite(sub.gap_ext).mean()),
                            "corr_gap_ext_vs_open": float(pd.Series(sub.gap_ext).corr(sub.y_gap))},
               "variants": {}, "paired_vs_B": {}, "timing": {}, "baskets": {}}
        for name, p in preds.items():
            res["variants"][name] = evaluate(sub, p)
            if name != "B_m1_live_lf":
                res["paired_vs_B"][name] = paired(sub, p, preds["B_m1_live_lf"])
            v = res["variants"][name]
            log(name, " | ".join(f"{per} hit {v[per]['top1']['hit']:.3f} mean {v[per]['top1']['mean_oc']:+.4f} net10k {v[per]['top1']['net_10k']:+.4f}" for per in ("dev", "confirm")))
        for name in ("D_ext_shape_lf", "D2_no_counts_lf", "B_m1_live_lf"):
            res["timing"][name] = timing(sub, preds[name])
            res["baskets"][name] = baskets(sub, preds[name])
        (OUT / f"r9c{tag}.json").write_text(json.dumps(res, indent=1, default=float))
        log(f"DONE {time.time() - t0:.0f}s")
        return
    run("A_m0", base)
    run("Bp_m1_official", base + ["gap_open"])
    run("B_m1_live", base + ["gap_open"], test_over={"gap_open": sub["gap_ext"].to_numpy(float)})
    sub["gap_open"] = sub["gap_ext"]
    run("C_m1_honest", base + ["gap_open"])
    run("D_ext_shape", base + ["gap_open"] + EXT_FEATS)
    run("E_overnight", base + ["gap_open"] + EXT_FEATS + ON_FEATS)
    run("F_overnight_only", base + ON_FEATS)

    has_ext = np.isfinite(sub["gap_ext"].to_numpy(float))
    # live mixing for B: M1 where an extended-hours price exists, else M0
    preds["B_m1_live"] = np.where(has_ext, preds["B_m1_live"], preds["A_m0"])
    # the new variants were trained WITH "no extended-hours trade" rows (NaN), so
    # production can score the whole shortlist with them; also report the old mix rule
    for k in ("C_m1_honest", "D_ext_shape", "E_overnight"):
        preds[k + "_mix"] = np.where(has_ext, preds[k], preds["A_m0"])
    res = {"coverage": cov, "variants": {}, "paired_vs_B": {}, "timing": {}, "baskets": {}}
    for name, p in preds.items():
        res["variants"][name] = evaluate(sub, p)
        log(name, json.dumps({per: {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}) for k, v in r.items()} for per, r in res["variants"][name].items()}))
        if name != "B_m1_live":
            res["paired_vs_B"][name] = paired(sub, p, preds["B_m1_live"])
            log("  vs B", json.dumps(res["paired_vs_B"][name]))
    for name in ("B_m1_live", "C_m1_honest", "D_ext_shape", "E_overnight", "A_m0"):
        res["timing"][name] = timing(sub, preds[name])
        res["baskets"][name] = baskets(sub, preds[name])
    (OUT / f"r9{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
