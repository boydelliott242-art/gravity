"""Research harness: can GRAVITY keep its edge on names that absorb a
$500K–$1M short without moving the price?

Walk-forward (monthly folds, 5-session embargo, isotonic calibration on a
later held-out slice — the production recipe), dump target, M0 features.
Variants are compared on a DEVELOPMENT window and the winner is reported on
an untouched CONFIRMATION window, so choosing among variants cannot inflate
the headline. Run:  ./.venv/bin/python scripts/research/size_tier.py
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
sys.path.insert(0, str(ROOT))

from gravity import cli, model as M  # noqa: E402
from gravity.features import FEATURES  # noqa: E402
from gravity.sources import prices  # noqa: E402

OUT = ROOT / "data" / "research"
OUT.mkdir(parents=True, exist_ok=True)

# ── capacity model (shared with production via gravity/capacity.py later) ──
RVOL_EDGES = np.array([0, 1, 1.5, 2, 3, 4, 6, 10, 15, 30])
NEXT_DV_Q25 = np.array([0.54, 0.77, 0.89, 0.99, 1.14, 1.32, 1.65, 2.12, 2.67, 6.57])
Y_IMPACT = 0.7          # square-root-law prefactor (literature range ~0.5–1)
MAX_PART = 0.05         # ≤ 5% of the session's expected $ volume per leg
MAX_IMPACT = 0.0050     # ≤ 50 bps one-way estimated impact


def exp_dvol(dv20, dv1):
    rv = np.where(dv20 > 0, dv1 / dv20, 0.0)
    k = np.clip(np.searchsorted(RVOL_EDGES, rv, side="right") - 1, 0, len(NEXT_DV_Q25) - 1)
    return dv20 * NEXT_DV_Q25[k]


def capacity(dvx, sigma):
    sig = np.maximum(sigma, 0.005)
    q_imp = dvx * (MAX_IMPACT / (Y_IMPACT * sig)) ** 2
    return np.minimum(MAX_PART * dvx, q_imp)


def impact(Q, dvx, sigma):
    return Y_IMPACT * np.maximum(sigma, 0.005) * np.sqrt(Q / np.maximum(dvx, 1.0))


# ── extra features from raw bars (as of the close of t; bars ≤ t only) ──
def extra_features(hist):
    out = []
    for s, df in hist.items():
        if df is None or len(df) < 30:
            continue
        d = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
        o, h, l, c, v = d.open, d.high, d.low, d.close, d.volume
        intra = c / o - 1
        gap = o / c.shift(1) - 1
        dv = c * v
        e = pd.DataFrame(index=d.index)
        e["oc_mean_20"] = intra.rolling(20, min_periods=10).mean()
        e["fade_rate_60"] = (intra <= -0.05).astype(float).rolling(60, min_periods=20).mean()
        e["pump_rate_60"] = (intra >= 0.05).astype(float).rolling(60, min_periods=20).mean()
        e["gap_mean_10"] = gap.rolling(10, min_periods=5).mean()
        e["intra_mean_10"] = intra.rolling(10, min_periods=5).mean()
        e["gap_fade_10"] = (gap.clip(lower=0) * (-intra).clip(lower=0)).rolling(10, min_periods=5).mean()
        r = np.log(c).diff().abs()
        e["amihud_log"] = np.log10((r / dv.replace(0, np.nan)).rolling(20, min_periods=10).mean() * 1e6)
        # Abdi–Ranaldo close-high-low spread estimator (20-day)
        eta = (np.log(h) + np.log(l)) / 2
        lc = np.log(c)
        prod = (lc - eta) * (lc - eta.shift(-1))  # uses t+1 → shift back so only ≤ t is used
        prod = prod.shift(1)                      # value at t now built from (t-1, t)
        e["spread_ar"] = 2 * np.sqrt(prod.rolling(20, min_periods=10).mean().clip(lower=0))
        e["oh_1"] = h / o - 1
        e["cl_vwap_1"] = c / ((h + l + c) / 3) - 1
        e["dv1"] = dv
        e["symbol"] = s
        out.append(e.reset_index().rename(columns={"index": "date"}))
    x = pd.concat(out, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    return x


EXTRA = ["oc_mean_20", "fade_rate_60", "pump_rate_60", "gap_mean_10", "intra_mean_10", "gap_fade_10",
         "amihud_log", "spread_ar", "oh_1", "cl_vwap_1", "log_dvx", "log_cap"]


def walk_forward(lab, feats, n_bag=1, train_mask=None, weights=None, seed=0):
    """OOS P(dump) for every test row (dump target, production recipe)."""
    s = M._settings(1, None)
    rng = np.random.default_rng(seed)
    dates = np.unique(lab["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, lab["date"].to_numpy("datetime64[ns]"))
    X = lab[feats].to_numpy(np.float32)
    y = lab["y_dump"].to_numpy(float)
    tm = np.ones(len(lab), bool) if train_mask is None else train_mask
    oos = np.full(len(lab), np.nan)
    for fw in M._fold_windows(dates, s):
        fit_idx = np.flatnonzero((di <= fw["fit_end"]) & tm)
        cal_idx = np.flatnonzero((di >= fw["cal_start"]) & (di <= fw["train_end"]) & tm)
        te_idx = np.flatnonzero((di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or len(cal_idx) < 200:
            continue
        raws_c, raws_t = [], []
        for b in range(n_bag):
            sb = json.loads(json.dumps(s)); sb["hgb"]["random_state"] = seed + b
            clf = M._fit_classifier(X[fit_idx], y[fit_idx], sb, np.random.default_rng(seed * 100 + b))
            raws_c.append(M._raw(clf, X[cal_idx])); raws_t.append(M._raw(clf, X[te_idx]))
        rc, rt = np.mean(raws_c, 0), np.mean(raws_t, 0)
        iso = M._fit_iso(rc, y[cal_idx])
        cal = iso.predict(rt) if iso is not None else rt
        oos[te_idx] = (1 - M.TIE_EPS) * cal + M.TIE_EPS * rt
    return oos


def evaluate(lab, prob, mask_dev, label):
    """Daily top-1 / top-3 for each size tier, gross and net of impact+spread."""
    res = {}
    base_ok = (~M.ssr_next(lab)) & np.isfinite(prob)
    tiers = {
        "micro (current rule)": base_ok & (lab["dvol20"].to_numpy() >= M.LIQ_FLOOR),
        "$500K": base_ok & (lab["close"].to_numpy() >= 1) & (lab["cap"].to_numpy() >= 5e5),
        "$1M": base_ok & (lab["close"].to_numpy() >= 1) & (lab["cap"].to_numpy() >= 1e6),
    }
    for period, pm in (("dev", mask_dev), ("confirm", ~mask_dev)):
        for tname, tmask in tiers.items():
            m = tmask & pm
            df = pd.DataFrame({"date": lab["date"].to_numpy()[m], "p": prob[m], "y": lab["y_dump"].to_numpy()[m],
                               "oc": lab["y_oc"].to_numpy()[m], "oh": lab["y_oh"].to_numpy()[m],
                               "dvx": lab["dvx"].to_numpy()[m], "sig": lab["vol20"].to_numpy()[m],
                               "spr": np.nan_to_num(lab["spread_ar"].to_numpy()[m], nan=0.01)})
            if not len(df):
                continue
            df = df.sort_values(["date", "p"], ascending=[True, False])
            top1 = df.groupby("date").head(1)
            top3 = df.groupby("date").head(3)
            Q = 5e5 if tname == "$500K" else (1e6 if tname == "$1M" else 5e4)
            cost1 = 2 * impact(Q, top1.dvx.to_numpy(), top1.sig.to_numpy()) + top1.spr.to_numpy()
            net1 = -top1.oc.to_numpy() - cost1
            q3 = Q / 3
            cost3 = 2 * impact(q3, top3.dvx.to_numpy(), top3.sig.to_numpy()) + top3.spr.to_numpy()
            net3 = (-top3.oc.to_numpy() - cost3)
            net3_day = pd.Series(net3).groupby(top3.date.to_numpy()).mean()
            eq = np.cumprod(1 + 0.10 * net1)
            res[f"{period}|{tname}"] = {
                "days": int(top1.date.nunique()), "names_per_day": float(df.groupby("date").size().median()),
                "tier_base_dump": float(df.y.mean()),
                "top1_hit": float(top1.y.mean()), "top1_mean_oc": float(top1.oc.mean()),
                "top1_median_oc": float(top1.oc.median()), "top1_squeeze": float((top1.oh >= 0.2).mean()),
                "top1_cost": float(np.mean(cost1)), "top1_net_mean": float(np.mean(net1)),
                "top1_net_win": float(np.mean(net1 > 0)),
                "top3_hit": float(top3.y.mean()), "top3_net_mean": float(net3_day.mean()),
                "comp10_mult": float(eq[-1]), "comp10_maxdd": float(np.max(1 - eq / np.maximum.accumulate(eq))),
                "auc_in_tier": M._auc(df.y.to_numpy(), df.p.to_numpy()),
            }
    print(f"\n=== {label}")
    for k, v in res.items():
        print(f"{k:28s} days {v['days']:>3} n/day {v['names_per_day']:>5.0f} base {v['tier_base_dump']:.3f} | "
              f"top1 hit {v['top1_hit']:.3f} oc {v['top1_mean_oc']:+.4f} med {v['top1_median_oc']:+.4f} sq {v['top1_squeeze']:.3f} "
              f"cost {v['top1_cost']*1e4:>4.0f}bp net {v['top1_net_mean']:+.4f} win {v['top1_net_win']:.3f} | "
              f"top3 hit {v['top3_hit']:.3f} net {v['top3_net_mean']:+.4f} | 10%comp {v['comp10_mult']:.2f}x dd {v['comp10_maxdd']:.2f} | auc {v['auc_in_tier']}", flush=True)
    return res


def main():
    t0 = time.time()
    panel = pickle.loads(cli.PANEL_PATH.read_bytes())
    lab = panel[panel["y_dump"].notna() & panel["y_oc"].notna()].copy()
    hist = prices.load_history(sorted(lab["symbol"].unique()), refresh=False)
    ex = extra_features(hist)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.merge(ex, on=["date", "symbol"], how="left")
    lab["dvx"] = exp_dvol(lab["dvol20"].to_numpy(float), lab["dv1"].to_numpy(float))
    lab["cap"] = capacity(lab["dvx"].to_numpy(float), lab["vol20"].to_numpy(float))
    lab["log_dvx"] = np.log10(lab["dvx"].clip(lower=1))
    lab["log_cap"] = np.log10(lab["cap"].clip(lower=1))
    lab = lab.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"prepared {len(lab):,} rows in {time.time()-t0:.0f}s; $500K-capable share {np.mean(lab.cap>=5e5):.3f}", flush=True)

    dates = np.sort(lab["date"].unique())
    split = pd.Timestamp("2025-11-01")
    mask_dev = (lab["date"] < split).to_numpy()
    liquid_train = (lab["dvx"].to_numpy() >= 2e6)
    variants = [
        ("V0 production features", FEATURES, dict()),
        ("V1 + new features", FEATURES + EXTRA, dict()),
        ("V2 + new features, 3-bag", FEATURES + EXTRA, dict(n_bag=3)),
        ("V3 + new features, liquid-only training", FEATURES + EXTRA, dict(train_mask=liquid_train)),
    ]
    allres = {}
    for name, feats, kw in variants:
        t1 = time.time()
        prob = walk_forward(lab, feats, **kw)
        np.save(OUT / f"oos_{name.split()[0]}.npy", prob)
        allres[name] = evaluate(lab, prob, mask_dev, f"{name}  ({time.time()-t1:.0f}s)")
    lab[["date", "symbol", "y_dump", "y_oc", "y_oh", "dvx", "cap", "vol20", "close", "dvol20", "spread_ar"]].to_pickle(OUT / "lab_meta.pkl")
    (OUT / "size_tier_results.json").write_text(json.dumps(allres, indent=1, default=float))
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
