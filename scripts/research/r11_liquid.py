"""R11 — a board of 100 where EVERY name can take a $30K short without heavy
slippage. Rule: scripts/research/r9_selection_rule.txt (R11, pre-registered).

Universe L ("$30K-ready"): published rule (no SSR next session, >= $300K median
$ volume) AND capacity >= $30K (<= 5% of the conservative expected session
$ volume AND <= 0.5% one-way square-root impact) AND prior close >= $1.

Candidates (walk-forward on the production folds; pre-market model inputs =
close features + gap from the last pre-market trade before 8:00 ET):
  L0  production D3_8 (fit on every PIT row), ranked within L
  L1  dump classifier fit on L rows only
  L2  <= -3% classifier fit on L rows only
  L3  regression on the session-demeaned open->close (L rows, clipped +-30%)
  L4  regression on the within-session percentile of open->close (L rows)
  L5  rank-average of L0, L1 and L3
Stage 2 (automatic, same rule file): W_on (+ overnight filing counts) and W_m0
(close-only fallback). Output: data/research/r11.json, oos_r11_<name>.npy
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402
from gravity import features as F  # noqa: E402

OUT = R.OUT
SPLIT = R.SPLIT
SIZE = 30_000
BOARD = 100
BANDS = ((1, 1), (2, 10), (11, 25), (26, 50), (51, 100), (1, 10), (1, 100))
log = R.log


# ── data ────────────────────────────────────────────────────────────────
def prep() -> pd.DataFrame:
    sub = R.build_sub()
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    # D3_8 input: last pre-market trade before 8:00 ET vs the hourly series' own prior close (leak-free, as r10b)
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
    sub["D"] = D
    key = pd.DataFrame({"symbol": sub["symbol"].to_numpy(), "D": D})
    m = key.merge(ext.rename(columns={"date": "D"})[["symbol", "D", "prev_date", "close_h_prev", "pm_last8"]], on=["symbol", "D"], how="left")
    del ext
    cp = m["close_h_prev"].to_numpy(float)
    aligned = (pd.to_datetime(m["prev_date"]).to_numpy("datetime64[ns]") == sub["date"].to_numpy("datetime64[ns]")) & np.isfinite(cp) & (cp > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sub["gap_open"] = np.where(aligned, m["pm_last8"].to_numpy(float) / cp - 1, np.nan)
    # as-traded price (split-adjusted closes build in later reverse splits — review fix, R11 addendum)
    px = sub["price"].to_numpy(float)
    px = np.where(np.isfinite(px), px, sub["close"].to_numpy(float))
    sub["L"] = sub["pub"].to_numpy() & (sub["cap"].to_numpy(float) >= SIZE) & (px >= 1.0)
    sub["c30"] = np.asarray(CAP.round_trip_cost(SIZE, sub["dvx"].to_numpy(float), sub["vol20"].to_numpy(float), sub["spr"].to_numpy(float)), float)
    # cost if real spreads are half the assumed ones (sensitivity only; the rule uses c30)
    sub["c30h"] = np.asarray(CAP.round_trip_cost(SIZE, sub["dvx"].to_numpy(float), sub["vol20"].to_numpy(float), 0.5 * sub["spr"].to_numpy(float)), float)
    y = sub["y_oc"].to_numpy(float)
    Lm = sub["L"].to_numpy() & np.isfinite(y)
    t = pd.DataFrame({"date": sub["date"].to_numpy()[Lm], "y": y[Lm]})
    med = t.groupby("date")["y"].transform("median").to_numpy()
    pct = t.groupby("date")["y"].rank(pct=True).to_numpy()
    sub["y_dm"] = np.nan; sub["y_pct"] = np.nan
    sub.loc[Lm, "y_dm"] = np.clip(y[Lm] - med, -0.3, 0.3)
    sub.loc[Lm, "y_pct"] = pct
    sub["y_d3"] = np.where(np.isfinite(y), (y <= -0.03).astype(float), np.nan)
    return sub


# ── walk-forward (production folds and settings; training rows restricted by ``train``) ──
def walk(sub, feats, train, kind, target, seed=0):
    """Score where HIGHER = better short. kind 'cls' → calibrated P(target=1);
    'reg' → minus the predicted target (target is a return-like quantity)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    s = M._settings(1, None)
    dates = np.unique(sub["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, sub["date"].to_numpy("datetime64[ns]"))
    X = sub[feats].to_numpy(np.float32)
    y = sub[target].to_numpy(float)
    tr = np.asarray(train, bool) & np.isfinite(y)
    test_rows = sub["L"].to_numpy() | sub["pub"].to_numpy()   # score every published row (L ⊂ pub)
    oos = np.full(len(sub), np.nan)
    rng = np.random.default_rng(seed * 100)
    for fw in M._fold_windows(dates, s):
        fit_idx = np.flatnonzero(tr & (di <= fw["fit_end"]))
        cal_idx = np.flatnonzero(tr & (di >= fw["cal_start"]) & (di <= fw["train_end"]))
        te_idx = np.flatnonzero(test_rows & (di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or len(cal_idx) < 200:
            continue
        sb = json.loads(json.dumps(s)); sb["hgb"]["random_state"] = seed
        if kind == "cls":
            clf = M._fit_classifier(X[fit_idx], y[fit_idx], sb, rng)
            rc, rt = M._raw(clf, X[cal_idx]), M._raw(clf, X[te_idx])
            iso = M._fit_iso(rc, y[cal_idx])
            cal = iso.predict(rt) if iso is not None else rt
            oos[te_idx] = (1 - M.TIE_EPS) * cal + M.TIE_EPS * rt
        else:
            idx = np.r_[fit_idx, cal_idx]
            if len(idx) > sb["max_fit_rows"]:
                idx = np.sort(rng.choice(idx, int(sb["max_fit_rows"]), replace=False))
            reg = HistGradientBoostingRegressor(**sb["hgb"]).fit(X[idx], y[idx])
            oos[te_idx] = -reg.predict(X[te_idx])
    return oos


def rank_avg(sub, scores):
    """Within-session percentile of each score among L rows, averaged (NaN if any is missing)."""
    L = sub["L"].to_numpy()
    out = np.full(len(sub), np.nan)
    d = pd.DataFrame({"date": sub["date"].to_numpy()[L]})
    acc = np.zeros(L.sum()); ok = np.ones(L.sum(), bool)
    for sc in scores:
        v = np.asarray(sc, float)[L]
        ok &= np.isfinite(v)
        acc += pd.Series(v).groupby(d["date"].to_numpy()).rank(pct=True).fillna(0).to_numpy()
    out[np.flatnonzero(L)[ok]] = acc[ok] / len(scores)
    return out


# ── evaluation ──────────────────────────────────────────────────────────
def board(sub, score, mask, k=BOARD):
    m = np.asarray(mask, bool) & np.isfinite(score)
    d = sub.loc[m, ["date", "symbol", "y_dump", "y_oc", "y_oh", "c30", "c30h", "cap", "dvx", "close", "gap_open"]].assign(score=np.asarray(score)[m])
    d = d.sort_values(["date", "score"], ascending=[True, False])
    d["rk"] = d.groupby("date").cumcount() + 1
    return d[d.rk <= k]


def stats(t):
    if not len(t):
        return None
    net = -t.y_oc.to_numpy(float) - t.c30.to_numpy(float)
    neth = -t.y_oc.to_numpy(float) - t.c30h.to_numpy(float)
    return {"n": int(len(t)), "days": int(t.date.nunique()), "hit": float(t.y_dump.mean()),
            "mean_oc": float(t.y_oc.mean()), "median_oc": float(t.y_oc.median()),
            "squeeze": float((t.y_oh >= 0.20).mean()), "cost30": float(t.c30.mean()),
            "net30": float(net.mean()), "win30": float((net > 0).mean()), "net30_halfspread": float(neth.mean()),
            "worst_oc": float(t.y_oc.max())}


def daily(t):
    """Per-session equal-weight basket net at $30K per name."""
    t = t.assign(net=-t.y_oc - t.c30)
    return t.groupby("date")["net"].mean()


def evaluate(sub, score, mask=None):
    mask = sub["L"].to_numpy() if mask is None else mask
    b = board(sub, score, mask)
    out = {}
    for per, pm in (("dev", b.date < SPLIT), ("confirm", b.date >= SPLIT)):
        x = b[pm.to_numpy()]
        r = {f"{a}-{z}": stats(x[(x.rk >= a) & (x.rk <= z)]) for a, z in BANDS}
        dd = daily(x)
        r["basket"] = {"days": int(len(dd)), "mean": float(dd.mean()), "t": float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if len(dd) > 2 else None,
                       "day_win": float((dd > 0).mean()), "worst_day": float(dd.min()), "best_day": float(dd.max())}
        out[per] = r
    return out


def paired(sub, s_new, s_base, k=BOARD):
    res = {}
    L = sub["L"].to_numpy()
    for per, cut in (("dev", lambda d: d < SPLIT), ("confirm", lambda d: d >= SPLIT)):
        a = board(sub, s_new, L, k); b = board(sub, s_base, L, k)
        a = a[cut(a.date)]; b = b[cut(b.date)]
        da, db = daily(a), daily(b)
        j = pd.concat([da.rename("a"), db.rename("b")], axis=1, join="inner")
        diff = (j.a - j.b).to_numpy()
        rng = np.random.default_rng(7)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(4000)] if len(diff) else [np.nan]
        res[per] = {"days": int(len(diff)), "mean_diff": float(diff.mean()) if len(diff) else None,
                    "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}
    return res


def line(name, ev):
    def f(per):
        r = ev[per]
        return (f"{per} ALL100 net {r['1-100']['net30'] * 100:+.2f} (half-spread {r['1-100']['net30_halfspread'] * 100:+.2f}) "
                f"top10 net {r['1-10']['net30'] * 100:+.2f} #1 hit {r['1-1']['hit'] * 100:.1f} mean {r['1-1']['mean_oc'] * 100:+.2f} net {r['1-1']['net30'] * 100:+.2f} | "
                f"bands hit " + "/".join(f"{r[k]['hit'] * 100:.0f}" for k in ("1-1", "2-10", "11-25", "26-50", "51-100")) +
                " net " + "/".join(f"{r[k]['net30'] * 100:+.2f}" for k in ("1-1", "2-10", "11-25", "26-50", "51-100")))
    log(f"{name:10s} {f('dev')}\n{'':19s}{f('confirm')}")


def main():
    t0 = time.time()
    sub = prep()
    L = sub["L"].to_numpy()
    log(f"rows {len(sub):,}  L rows {L.sum():,}  L per session {pd.Series(L).groupby(sub['date'].to_numpy()).sum().median():.0f}  "
        f"gap coverage in L {np.isfinite(sub['gap_open'].to_numpy()[L]).mean():.2f}")
    base = list(F.FEATURES)
    D3 = base + ["gap_open"]
    preds = {}

    def run(name, feats, train, kind, target):
        f = OUT / f"oos_r11_{name}.npy"
        if f.exists():
            preds[name] = np.load(f); log(f"{name}: cached"); return
        t1 = time.time()
        preds[name] = walk(sub, feats, train, kind, target)
        np.save(f, preds[name]); log(f"{name}: {time.time() - t1:.0f}s")

    preds["L0"] = np.load(OUT / "oos_r10_D3_8.npy")
    preds["A_m0"] = np.load(OUT / "oos_r9_A_m0.npy")
    run("L1", D3, L, "cls", "y_dump")
    run("L2", D3, L, "cls", "y_d3")
    run("L3", D3, L, "reg", "y_dm")
    run("L4", D3, L, "reg", "y_pct")
    preds["L5"] = rank_avg(sub, [preds["L0"], preds["L1"], preds["L3"]])

    res = {"universe": {"rule": "published rule + capacity >= $30K (<=5% of conservative expected $vol, <=0.5% impact) + close >= $1",
                        "rows": int(L.sum()), "per_session_median": float(pd.Series(L).groupby(sub['date'].to_numpy()).sum().median())},
           "before": {"old_board_at_30k": evaluate(sub, preds["L0"], sub["pub"].to_numpy()),
                      "old_m0_board_at_30k": evaluate(sub, preds["A_m0"], sub["pub"].to_numpy())},
           "variants": {}, "paired_vs_L0": {}}
    line("OLD board", res["before"]["old_board_at_30k"])
    for name in ("L0", "L1", "L2", "L3", "L4", "L5", "A_m0"):
        res["variants"][name] = evaluate(sub, preds[name])
        if name not in ("L0", "A_m0"):
            res["paired_vs_L0"][name] = paired(sub, preds[name], preds["L0"])
        line(name, res["variants"][name])
        if name in res["paired_vs_L0"]:
            log(f"{'':19s}paired vs L0: {json.dumps(res['paired_vs_L0'][name])}")

    # ── rule (pre-registered) ──
    V = res["variants"]
    cands = ["L0", "L1", "L2", "L3", "L4", "L5"]
    best = max(V[c]["dev"]["1-100"]["net30"] for c in cands)
    tied = [c for c in cands if V[c]["dev"]["1-100"]["net30"] >= best - 0.0005]
    W = max(tied, key=lambda c: V[c]["dev"]["1-10"]["net30"])
    adopt = W == "L0" or (V[W]["confirm"]["1-100"]["net30"] >= V["L0"]["confirm"]["1-100"]["net30"]
                          and V[W]["confirm"]["1-10"]["net30"] >= V["L0"]["confirm"]["1-10"]["net30"] - 0.0025)
    chosen = W if adopt else "L0"
    res["decision"] = {"dev_winner": W, "adopted": chosen}
    log(f"RULE: dev winner {W}; adopted {chosen}")

    # ── stage 2 ──
    recipe = {"L0": (None, "cls", "y_dump"), "L1": (L, "cls", "y_dump"), "L2": (L, "cls", "y_d3"),
              "L3": (L, "reg", "y_dm"), "L4": (L, "reg", "y_pct")}
    # overnight filings counted only until the 8:00 ET pre-market cutoff the live morning run uses (review fix)
    sub = R.add_overnight(sub, cutoff_hour=8)
    every = np.isfinite(sub["y_dump"].to_numpy(float))
    if chosen == "L5":
        run("L1_on", D3 + R.ON_FEATS, L, "cls", "y_dump"); run("L3_on", D3 + R.ON_FEATS, L, "reg", "y_dm")
        run("L0_on", D3 + R.ON_FEATS, every, "cls", "y_dump")
        preds["W_on"] = rank_avg(sub, [preds["L0_on"], preds["L1_on"], preds["L3_on"]])
        run("L1_m0", base, L, "cls", "y_dump"); run("L3_m0", base, L, "reg", "y_dm")
        preds["W_m0"] = rank_avg(sub, [preds["A_m0"], preds["L1_m0"], preds["L3_m0"]])
    else:
        tr, kind, tgt = recipe[chosen]
        tr = every if tr is None else tr
        run(f"{chosen}_on", D3 + R.ON_FEATS, tr, kind, tgt); preds["W_on"] = preds[f"{chosen}_on"]
        if chosen == "L0":
            preds["W_m0"] = preds["A_m0"]
        else:
            run(f"{chosen}_m0", base, tr, kind, tgt); preds["W_m0"] = preds[f"{chosen}_m0"]
    for name in ("W_on", "W_m0"):
        res["variants"][name] = evaluate(sub, preds[name]); line(name, res["variants"][name])
    res["paired_W_on_vs_winner"] = paired(sub, preds["W_on"], preds[chosen])   # (not vs L0)
    on_ok = all(V["W_on"][per]["1-100"]["net30"] - V[chosen][per]["1-100"]["net30"] >= 0.0005 for per in ("dev", "confirm"))
    m0_ok = V["W_m0"]["confirm"]["1-100"]["net30"] >= V["A_m0"]["confirm"]["1-100"]["net30"]
    res["decision"].update({"overnight_filings": bool(on_ok), "fallback": "W_m0" if m0_ok else "A_m0"})
    log(f"RULE stage 2: overnight filings adopted {on_ok}; fallback {'W_m0' if m0_ok else 'A_m0'}")
    (OUT / "r11.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
