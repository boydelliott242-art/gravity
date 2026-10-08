"""Walk-forward-validated dump / squeeze models (CONTRACTS §8, §12).

Two model sets, identical except for one column:

* **M0** — ``features.FEATURES`` only: everything known at the close of
  ``t``. Used the evening before and for every name in the morning.
* **M1** — ``FEATURES + M1_EXTRA`` (``gap_open``). In the backtest this is
  the real 9:30 open of ``t+1``; live it is a pre-market price proxy, so M1
  is only used for the shortlist names that have a pre-market print.

Each set has calibrated classifiers for ``y_dump`` (open→close ≤ −5 %),
``y_bigdump`` (≤ −15 %) and ``y_squeeze`` (open→high ≥ +20 %), plus a
regressor for the clipped open→close return (``exp_oc``).

Recipe (identical in every walk-forward fold and in production):

1. training window = every labeled session before the test window, minus
   an embargo of ``EMBARGO`` sessions (labels look one session ahead);
2. the last ``CAL_SESSIONS`` sessions of that window are held out;
   ``HistGradientBoosting`` (NaN-native) is fit on the sessions before them
   (minus another embargo) and isotonic regression maps its raw scores to
   probabilities on the held-out, *later* slice;
3. the ``exp_oc`` regressor is fit on the whole window (it is not calibrated).

Walk-forward = expanding window, test folds of ``fold_months`` calendar
months, out-of-sample predictions for every test session → AUC, Brier,
calibration deciles, daily top-1 / top-10 / top-decile hit rates and a
"short the #1 at the open, cover at the close" simulation. Production is
the same recipe with the window extended to the last labeled session.

Nothing here downloads anything; the panel comes from ``features``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import config
from .features import (FAMILIES, FEATURE_DOCS, FEATURES, FINRA_FEATURES, LABELS, M1_EXTRA,
                       PUMP_THRESHOLD, SWING_THRESHOLD)
from .util import clean

log = logging.getLogger(__name__)

# ── Settings ─────────────────────────────────────────────────────────────
TARGETS: Dict[str, str] = {"dump": "y_dump", "bigdump": "y_bigdump", "squeeze": "y_squeeze",
                           "swing": "y_swing", "pump": "y_pump"}
PROB_COLS: Dict[str, str] = {t: f"prob_{t}" for t in TARGETS}
OUT_COLUMNS: List[str] = ["prob_dump", "prob_bigdump", "prob_squeeze", "prob_swing", "prob_pump", "skew", "exp_oc"]
SWING_HOLD = 5                                # sessions a swing short is held (enter next open, cover 5th close)
M0_FEATURES: List[str] = list(FEATURES)
M1_FEATURES: List[str] = list(FEATURES) + list(M1_EXTRA)

OC_CLIP: Tuple[float, float] = (-0.5, 0.5)   # regressor target clip (one +3,000 % day must not dominate)
COST = 0.01                                   # assumed round-trip cost of a short, fraction of notional
SSR_DROP = 0.10                               # Rule 201: a ≥10% intraday drop restricts shorting the next session
LIQ_FLOOR = 300_000                           # published #1 needs ≥ $300k 20-day median dollar volume
CAP_FROM = 0.5                                # probabilities above this are capped at their realised OOS rate
SIM_FRACTION = 0.10                           # compounded curve: risk 10% of equity per day
REALISTIC_SIZES = (10_000, 50_000)            # per-pick cost model sizes reported in the sim
TIE_EPS = 1e-6                                # isotonic ties broken by the raw score (moves p by < 1e-6)
BUNDLE_NAME = "bundle.pkl"
DEFAULT_SITE_JSON = config.SITE_DATA / "model.json"
BUNDLE_VERSION = 1

DEFAULTS: Dict[str, Any] = {
    "embargo": 5,                  # sessions between the last training label and the first test/cal day
    "min_train_sessions": 252,     # first test fold starts after ~1 year of labeled sessions
    "cal_sessions": 63,            # held-out later slice for isotonic calibration (~1 quarter)
    "max_fit_rows": 400_000,       # per-fit row cap (see _subsample); keeps the full train ≲ 15 min
    "importance_rows": 40_000,     # rows used for permutation importance
    "hgb": {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": 0,
    },
    "seed": 0,
}

TARGET_TEXT = {
    "dump": f"next session open→close ≤ {config.DUMP_THRESHOLD:+.0%}",
    "bigdump": f"next session open→close ≤ {config.BIG_DUMP_THRESHOLD:+.0%}",
    "squeeze": f"next session open→high ≥ {config.SQUEEZE_THRESHOLD:+.0%}",
    "swing": f"close {SWING_HOLD} sessions later ≤ {SWING_THRESHOLD:+.0%} versus the next session's open "
             f"(short at the next open, cover at the close {SWING_HOLD} sessions later)",
    "pump": f"next session open→close ≥ {PUMP_THRESHOLD:+.0%}",
    "skew": "P(dump) ÷ (P(dump) + P(pump)): the downside share of a ±5% open→close move",
    "exp_oc": f"expected next-session open→close, each training outcome clipped to "
              f"[{OC_CLIP[0]:+.0%}, {OC_CLIP[1]:+.0%}]",
}

from . import features as _features_mod
_M1_CAVEAT = (
    "M1 uses the real 9:30 open in the backtest; live it uses a pre-market price as a proxy for the "
    "open, which can differ materially. Treat M1 live numbers as less reliable than its backtest."
    if _features_mod.MORNING_SET == "official" else
    "M1 is trained and tested on what the morning run actually knows at ~9:00 ET: the last after-hours / "
    "pre-market trade (Yahoo hourly bars), not the 9:30 open"
    + (", plus the shape of extended-hours trading (high, low, fade, after-hours move)" if _features_mod.MORNING_SET in ("ext", "ext_counts", "ext_on") else "")
    + (" and filings accepted overnight" if _features_mod.MORNING_SET == "ext_on" else "")
    + ". Live, those inputs come from same-day hourly bars, which Yahoo sometimes revises slightly later; the "
    "morning run records what it saw to measure that. Yahoo serves hourly history only for ~2 years, so older rows "
    "have no extended-hours inputs."
)
CAVEATS: List[str] = [
    "Point-in-time universe: training and this backtest use every listed common stock, keeping a row "
    "only while that company was ≤ $2B at the time — that day's traded price × the share count on its "
    "latest SEC cover page, adjusted for any split since. This removes the bias of using today's "
    "small-cap list (which over-represents stocks that already collapsed into it) without discarding "
    "serial diluters. It is still built from today's listings, so companies that delisted are missing "
    "(their collapses would have helped shorts) while names that later grew are included — on balance "
    "it leans slightly against shorts.",
    "No locate constraint: the simulation assumes the #1 name could be borrowed and shorted at the "
    "open every day. Many of these names are hard to borrow or unavailable, and borrow fees (often "
    "50-500%+ annualised) are not modelled beyond the flat cost.",
    "Fills are Yahoo's daily open and close (vendor data: some opens on thin names are substituted "
    "with the prior close). Real fills on thin names slip; the flat 1% round-trip cost may be optimistic "
    "— the cost-sensitivity table shows how fast the edge shrinks at 3% and 5%.",
    "SEC Rule 201: after a ≥10% intraday drop, shorts the next session may only be entered above the bid. "
    "Many raw #1 picks were in that state (see top1_ssr_rate / top1_gross_share_ssr); the published-rule "
    "numbers exclude them, and are the ones to look at.",
    "Dump odds are partly a volatility forecast: the same scores also rank big UP days well (pump_auc). "
    "The directional block (dump-vs-pump AUC, mean open→close by decile) shows how much is genuinely "
    "downside skew.",
    "Halts: sessions where the next day is halted or has no clean print have no label and are "
    "dropped. A short trapped in a halt is exactly the worst case, and it is not in these numbers.",
    "Uncapped risk: a short can lose more than 100% in one session (the data contains open→close "
    "moves above +1,000%). Daily P&L is summed at 1x notional per day, not compounded, with no stop.",
    _M1_CAVEAT,
    "Historical filing features are form/item based (no text matching); names without SEC history "
    "get blank filing features. Live text-matched catalysts are an overlay the model never trained on.",
    "Probabilities are calibrated on the most recent held-out quarter; they drift when the market "
    "regime changes. Out-of-sample results are history, not a promise.",
]


# ── Tradeability (shared with the live publication rule) ────────────────
def _numcol(df: pd.DataFrame, c: str) -> np.ndarray:
    """Column as float array; all-NaN when the column is absent."""
    if c not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[c], errors="coerce").to_numpy(float)


def ssr_next(df: pd.DataFrame) -> np.ndarray:
    """True where the NEXT session is under the SEC Rule 201 short-sale
    restriction: the day-t low was ≥10% below the prior close (shorts may
    then only be entered above the national best bid)."""
    low, close, r1 = _numcol(df, "low"), _numcol(df, "close"), _numcol(df, "r1")
    prev = close / (1.0 + r1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.isfinite(prev) & np.isfinite(low) & (low <= (1.0 - SSR_DROP) * prev)


def publishable(df: pd.DataFrame) -> np.ndarray:
    """The rule the site uses for its #1 that history can check: not under
    SSR next session and liquid (20-day median $ volume ≥ LIQ_FLOOR). Live
    it additionally needs IBKR borrow and squeeze danger < 70, which have no
    history — that part of the rule is unvalidated."""
    dv = _numcol(df, "dvol20")
    return (~ssr_next(df)) & np.isfinite(dv) & (dv >= LIQ_FLOOR)


# ── Small helpers ────────────────────────────────────────────────────────
def _settings(fold_months: int, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = json.loads(json.dumps(DEFAULTS))
    for k, v in (params or {}).items():
        if k == "hgb" and isinstance(v, dict):
            s["hgb"].update(v)
        else:
            s[k] = v
    s["fold_months"] = int(max(1, fold_months))
    return s


def _iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _day(d: Any) -> Optional[str]:
    if d is None:
        return None
    try:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _f(x: Any) -> Optional[float]:
    """float or None (NaN/inf → None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _matrix(rows: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    out = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        log.warning("model: %d feature column(s) missing from rows → NaN: %s", len(missing), missing[:8])
    for j, c in enumerate(cols):
        if c in rows.columns:
            v = pd.to_numeric(rows[c], errors="coerce").to_numpy(dtype=np.float64)
            v[~np.isfinite(v)] = np.nan
            out[:, j] = v
    return out


def _subsample(y: np.ndarray, cap: int, rng: np.random.Generator,
               classify: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Row indices (and weights) for one fit, at most ``cap`` rows.

    Classifiers keep every positive (up to half the cap) and a random share
    of negatives, re-weighted so the weighted base rate equals the true one;
    the regressor takes a uniform random sample. Isotonic calibration is fit
    afterwards on untouched, unweighted rows, so this never biases the
    reported probabilities."""
    n = len(y)
    if n <= cap:
        return np.arange(n), None
    if not classify:
        return np.sort(rng.choice(n, cap, replace=False)), None
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y != 1)
    k_pos = min(len(pos), cap // 2)
    k_neg = min(len(neg), cap - k_pos)
    p_idx = rng.choice(pos, k_pos, replace=False) if k_pos < len(pos) else pos
    n_idx = rng.choice(neg, k_neg, replace=False) if k_neg < len(neg) else neg
    idx = np.sort(np.r_[p_idx, n_idx])
    w = np.where(y[idx] == 1, len(pos) / max(k_pos, 1), len(neg) / max(k_neg, 1)).astype(np.float64)
    w /= w.mean()
    return idx, w


def _fit_classifier(X: np.ndarray, y: np.ndarray, s: Dict[str, Any], rng: np.random.Generator):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if len(y) == 0 or np.unique(y).size < 2:
        return None
    idx, w = _subsample(y, int(s["max_fit_rows"]), rng, classify=True)
    params = dict(s["hgb"])
    # early stopping needs enough positives in its random validation split
    if params.get("early_stopping") and (y[idx] == 1).sum() < 200:
        params["early_stopping"] = False
        params["max_iter"] = min(int(params["max_iter"]), 150)
    m = HistGradientBoostingClassifier(**params)
    m.fit(X[idx], y[idx], sample_weight=w)
    return m


def _fit_regressor(X: np.ndarray, y: np.ndarray, s: Dict[str, Any], rng: np.random.Generator):
    from sklearn.ensemble import HistGradientBoostingRegressor

    if len(y) < 50:
        return None
    idx, _ = _subsample(y, int(s["max_fit_rows"]), rng, classify=False)
    m = HistGradientBoostingRegressor(**s["hgb"])
    m.fit(X[idx], np.clip(y[idx], *OC_CLIP))
    return m


def _raw(clf, X: np.ndarray) -> np.ndarray:
    if clf is None or len(X) == 0:
        return np.full(len(X), np.nan)
    return clf.predict_proba(X)[:, 1]


def _fit_iso(raw: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    ok = np.isfinite(raw) & np.isfinite(y)
    if ok.sum() < 50 or np.unique(y[ok]).size < 2:
        return None
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw[ok], y[ok])


def _calibrated(cm: Optional[Dict[str, Any]], X: np.ndarray) -> np.ndarray:
    """Calibrated probability. Isotonic maps are step-like; exact ties are
    broken by the raw score with weight ``TIE_EPS`` so rankings are stable
    (the probability moves by less than 1e-6)."""
    if not cm or cm.get("clf") is None:
        return np.full(len(X), np.nan)
    raw = _raw(cm["clf"], X)
    iso = cm.get("iso")
    cal = iso.predict(raw) if iso is not None else raw
    cap = cm.get("cap")
    if cap is not None:  # never claim more certainty than the tail ever delivered out of sample
        cal = np.minimum(cal, cap)
    return np.clip((1.0 - TIE_EPS) * cal + TIE_EPS * raw, 0.0, 1.0)


def skew_of(prob_dump: np.ndarray, prob_pump: np.ndarray) -> np.ndarray:
    """P(dump) / (P(dump) + P(pump)) — NaN when either is unknown or both are 0."""
    d = np.asarray(prob_dump, dtype=float)
    u = np.asarray(prob_pump, dtype=float)
    tot = d + u
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.isfinite(tot) & (tot > 0), d / tot, np.nan)
    return out


def _predict_set(ms: Dict[str, Any], X: np.ndarray) -> Dict[str, np.ndarray]:
    out = {PROB_COLS[t]: _calibrated(ms.get(t), X) for t in TARGETS}
    # a ≤ −15 % day is also a ≤ −5 % day: keep the probabilities coherent
    out["prob_bigdump"] = np.fmin(out["prob_bigdump"], out["prob_dump"])
    out["skew"] = skew_of(out["prob_dump"], out["prob_pump"])
    reg = ms.get("oc")
    out["exp_oc"] = reg.predict(X) if (reg is not None and len(X)) else np.full(len(X), np.nan)
    return out


def _fit_set(X: np.ndarray, Y: Dict[str, np.ndarray], fit_idx: np.ndarray, cal_idx: np.ndarray,
             ncols: int, s: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """One model set (M0 when ``ncols == len(FEATURES)``, M1 with gap_open).

    Rows whose label for a target is unknown (``y_swing`` needs t+5) are
    left out of that target's fit and calibration only."""
    Xf = X[fit_idx, :ncols]
    Xc = X[cal_idx, :ncols]
    ms: Dict[str, Any] = {}
    for t, col in TARGETS.items():
        yf = Y[col][fit_idx]
        okf = np.isfinite(yf)
        clf = _fit_classifier(Xf[okf] if not okf.all() else Xf, yf[okf], s, rng)
        iso = _fit_iso(_raw(clf, Xc), Y[col][cal_idx]) if clf is not None else None
        ms[t] = {"clf": clf, "iso": iso}
    all_idx = np.r_[fit_idx, cal_idx]
    ms["oc"] = _fit_regressor(X[all_idx, :ncols], Y["y_oc"][all_idx], s, rng)
    return ms


# ── Windows ──────────────────────────────────────────────────────────────
def _fold_windows(dates: np.ndarray, s: Dict[str, Any]) -> List[Dict[str, int]]:
    """Walk-forward folds as indices into the sorted unique label dates."""
    n = len(dates)
    emb, cal = int(s["embargo"]), int(s["cal_sessions"])
    min_fit = 40
    start = max(int(s["min_train_sessions"]), cal + 2 * emb + min_fit)
    if n - start < 5:
        return []
    per = pd.DatetimeIndex(dates).to_period("M")
    months = (per.year * 12 + per.month - 1).to_numpy()
    # folds align to calendar months from the first test month; a stub first
    # fold (the start landed near a month end) is merged into the next one
    per_q = int(s["fold_months"])
    bucket = (months - months[start] + (months[start] % per_q if per_q > 1 else 0)) // per_q
    first = np.flatnonzero(bucket[start:] == bucket[start]) + start
    if len(first) < 10 and first[-1] + 1 < n:
        bucket[first] = bucket[first[-1] + 1]
    folds = []
    for b in np.unique(bucket[start:]):
        te = np.flatnonzero((bucket == b) & (np.arange(n) >= start))
        a, z = int(te[0]), int(te[-1])
        train_end = a - emb - 1                  # last training label date
        cal_start = train_end - cal + 1
        fit_end = cal_start - emb - 1
        if fit_end < min_fit:
            continue
        folds.append({"test_start": a, "test_end": z, "train_end": train_end,
                      "cal_start": cal_start, "fit_end": fit_end})
    return folds


# ── Metrics ──────────────────────────────────────────────────────────────
def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 10 or np.unique(y[ok]).size < 2:
        return None
    return float(roc_auc_score(y[ok], p[ok]))


def _daily_table(day: np.ndarray, prob: np.ndarray, y: np.ndarray, oc: np.ndarray,
                 sq: Optional[np.ndarray] = None, pub: Optional[np.ndarray] = None,
                 ssr: Optional[np.ndarray] = None, costs: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Per test day: the #1, the top-10 and the top decile by ``prob``; plus
    the #1 under the publication rule (``pub``) and whether the raw #1 was
    under Rule 201 (``ssr``)."""
    df = pd.DataFrame({"day": day, "p": prob, "y": y, "oc": oc,
                       "sq": sq if sq is not None else np.nan,
                       "pub": pub if pub is not None else True,
                       "ssr": ssr if ssr is not None else False})
    for k, v in (costs or {}).items():
        df[f"cost_{k}"] = v
    df = df[np.isfinite(df["p"]) & np.isfinite(df["y"]) & np.isfinite(df["oc"])]
    if not len(df):
        return pd.DataFrame(columns=["top1_y", "top1_oc", "top1_sq", "top1_ssr", "top10_y", "top10_oc",
                                     "dec_y", "uni_y", "uni_oc", "pub1_y", "pub1_oc", "pub1_sq", "n"])
    df = df.sort_values(["day", "p"], ascending=[True, False], kind="mergesort")
    g = df.groupby("day", sort=True)
    df["rk"] = g.cumcount() + 1
    df["n"] = g["p"].transform("size")
    top1 = df[df["rk"] == 1].set_index("day")
    top10 = df[df["rk"] <= 10].groupby("day")[["y", "oc"]].mean()
    dec = df[df["rk"] <= np.ceil(0.1 * df["n"])].groupby("day")["y"].mean()
    uni = g[["y", "oc"]].mean()
    pub1 = df[df["pub"].astype(bool)].groupby("day", sort=True).head(1).set_index("day")
    return pd.DataFrame({
        "top1_y": top1["y"], "top1_oc": top1["oc"], "top1_sq": top1["sq"],
        "top1_ssr": top1["ssr"].astype(float),
        "top10_y": top10["y"], "top10_oc": top10["oc"],
        "dec_y": dec, "uni_y": uni["y"], "uni_oc": uni["oc"],
        "pub1_y": pub1["y"], "pub1_oc": pub1["oc"], "pub1_sq": pub1["sq"],
        "n": g.size(),
        **{f"top1_cost_{k}": top1[f"cost_{k}"] for k in (costs or {})},
        **{f"pub1_cost_{k}": pub1[f"cost_{k}"] for k in (costs or {})},
    })


def _metrics(y: np.ndarray, p: np.ndarray, daily: pd.DataFrame) -> Dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(p)
    brier = float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else None
    m: Dict[str, Any] = {
        "auc": _auc(y, p),
        "brier": brier,
        "base_rate": float(np.mean(y[ok])) if ok.any() else None,
        "mean_pred": float(np.mean(p[ok])) if ok.any() else None,
        "n": int(ok.sum()),
        "days": int(len(daily)),
    }
    if len(daily):
        m.update({
            "top1_hit": _f(daily["top1_y"].mean()),
            "top10_hit": _f(daily["top10_y"].mean()),
            "top_decile_hit": _f(daily["dec_y"].mean()),
            "universe_daily_hit": _f(daily["uni_y"].mean()),
            "top1_mean_oc": _f(daily["top1_oc"].mean()),
            "top1_median_oc": _f(daily["top1_oc"].median()),
            "top10_mean_oc": _f(daily["top10_oc"].mean()),
            "universe_mean_oc": _f(daily["uni_oc"].mean()),
            "top1_squeeze_rate": _f(daily["top1_sq"].mean()),
            "top1_ssr_rate": _f(daily["top1_ssr"].mean()),
            "top1_gross_share_ssr": _f((-daily["top1_oc"] * daily["top1_ssr"]).sum() / (-daily["top1_oc"]).sum())
            if (-daily["top1_oc"]).sum() > 0 else None,
            "pub1_hit": _f(daily["pub1_y"].mean()),
            "pub1_mean_oc": _f(daily["pub1_oc"].mean()),
            "pub1_median_oc": _f(daily["pub1_oc"].median()),
            "pub1_squeeze_rate": _f(daily["pub1_sq"].mean()),
            "pub1_days": int(daily["pub1_oc"].notna().sum()),
        })
    else:
        m.update({k: None for k in ("top1_hit", "top10_hit", "top_decile_hit", "universe_daily_hit",
                                    "top1_mean_oc", "top1_median_oc", "top10_mean_oc",
                                    "universe_mean_oc", "top1_squeeze_rate")})
    return m


def _max_drawdown(pnl: np.ndarray) -> Optional[float]:
    if not len(pnl):
        return None
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float(np.max(peak - cum))


def _sim(daily: pd.DataFrame, cost: float = COST) -> Dict[str, Any]:
    """Short the #1 (and, separately, an equal-weight top-10 basket) at the
    open and cover at the close, every test day, 1x notional, summed."""
    if not len(daily):
        return {"daily": [], "gross_total": None, "net_total": None, "win_rate": None,
                "max_drawdown": None, "cost_assumption": cost}
    g1 = -daily["top1_oc"].to_numpy(float)
    g10 = -daily["top10_oc"].to_numpy(float)
    gu = -daily["uni_oc"].to_numpy(float)
    gp = -daily["pub1_oc"].to_numpy(float) if "pub1_oc" in daily else np.full(len(daily), np.nan)
    gp_ok = gp[np.isfinite(gp)]
    rows = [[_day(d), _f(a), _f(b), _f(c), _f(e), _f(k)] for d, a, b, c, e, k in
            zip(daily.index, daily["top1_oc"], daily["top10_oc"], daily["uni_oc"],
                daily["pub1_oc"] if "pub1_oc" in daily else [None] * len(daily),
                daily["pub1_cost_10000"] if "pub1_cost_10000" in daily else [None] * len(daily))]

    def compounded(g: np.ndarray) -> Dict[str, Any]:
        """Risk SIM_FRACTION of equity per day on the short (loss capped at total ruin)."""
        eq = np.cumprod(np.maximum(0.0, 1.0 + SIM_FRACTION * (g - cost)))
        peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
        return {"final_multiple": _f(eq[-1]) if len(eq) else None,
                "max_drawdown_pct": _f(np.max(1 - eq / peak)) if len(eq) else None,
                "curve": [_f(x) for x in eq[:: max(1, len(eq) // 120)]]}

    return {
        "daily": rows,
        "columns": ["date", "oc_top1", "oc_top10_mean", "universe_mean_oc", "oc_pub1", "pub1_cost_10k"],
        "pub_gross_total": _f(gp_ok.sum()) if len(gp_ok) else None,
        "pub_net_total": _f((gp_ok - cost).sum()) if len(gp_ok) else None,
        "pub_win_rate": _f(np.mean(gp_ok - cost > 0)) if len(gp_ok) else None,
        "pub_max_drawdown": _max_drawdown(gp_ok - cost),
        "pub_worst_day": _f(np.min(gp_ok)) if len(gp_ok) else None,
        "pub_best_day": _f(np.max(gp_ok)) if len(gp_ok) else None,
        "compounded_top1": compounded(g1),
        "compounded_pub": compounded(gp_ok) if len(gp_ok) else None,
        "compounded_fraction": SIM_FRACTION,
        "cost_sensitivity": [{"cost": c, "top1_net_total": _f((g1 - c).sum()),
                              "pub_net_total": _f((gp_ok - c).sum()) if len(gp_ok) else None}
                             for c in (0.01, 0.03, 0.05)],
        "top20_days_share": _f(np.sort(g1)[::-1][:20].sum() / g1.sum()) if g1.sum() > 0 else None,
        "realistic": _realistic(daily),
        "gross_total": float(g1.sum()),
        "net_total": float((g1 - cost).sum()),
        "win_rate": float(np.mean(g1 - cost > 0)),
        "gross_win_rate": float(np.mean(g1 > 0)),
        "max_drawdown": _max_drawdown(g1 - cost),
        "mean_daily_net": float(np.mean(g1 - cost)),
        "worst_day": float(np.min(g1)),
        "best_day": float(np.max(g1)),
        "top10_gross_total": float(g10.sum()),
        "top10_net_total": float((g10 - cost).sum()),
        "top10_win_rate": float(np.mean(g10 - cost > 0)),
        "top10_max_drawdown": _max_drawdown(g10 - cost),
        "top10_worst_day": float(np.min(g10)),
        "universe_gross_total": float(gu.sum()),
        "universe_net_total": float((gu - cost).sum()),
        "cost_assumption": cost,
        "units": "sum of daily returns at 1x notional (not compounded); short return = -(close/open - 1)",
    }


def _realistic(daily: pd.DataFrame) -> Dict[str, Any]:
    """Net of each pick's own estimated cost (capacity model: square-root
    impact on both legs + spread) at fixed position sizes, for the published
    #1 — the honest replacement for a flat cost assumption."""
    out: Dict[str, Any] = {}
    for size in REALISTIC_SIZES:
        col = f"pub1_cost_{size}"
        if col not in daily:
            continue
        m = daily["pub1_oc"].notna() & daily[col].notna()
        g = -daily.loc[m, "pub1_oc"].to_numpy(float)
        c = daily.loc[m, col].to_numpy(float)
        net = g - c
        eq = np.cumprod(1 + SIM_FRACTION * net) if len(net) else np.array([])
        out[str(size)] = {
            "days": int(m.sum()), "mean_cost": _f(c.mean()) if len(c) else None,
            "median_cost": _f(np.median(c)) if len(c) else None,
            "gross_mean": _f(g.mean()) if len(g) else None, "net_mean": _f(net.mean()) if len(net) else None,
            "net_total": _f(net.sum()) if len(net) else None, "win_rate": _f(np.mean(net > 0)) if len(net) else None,
            "compounded": _f(eq[-1]) if len(eq) else None,
            "max_drawdown_pct": _f(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else None,
        }
    return out


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < bins:
        return []
    yy, pp = y[ok], p[ok]
    r = pd.Series(pp).rank(method="first").to_numpy()
    b = np.minimum((r - 1) * bins // len(pp), bins - 1).astype(int)
    out = []
    for i in range(bins):
        m = b == i
        if m.any():
            out.append({"bin": i + 1, "pred": float(pp[m].mean()), "actual": float(yy[m].mean()),
                        "n": int(m.sum()), "pred_max": float(pp[m].max())})
    return out


# ── Swing (5-session) evaluation ─────────────────────────────────────────
def _swing_table(day: np.ndarray, sym: np.ndarray, prob: np.ndarray, y: np.ndarray, c5: np.ndarray,
                 h5: np.ndarray, pub: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per test day, ranked by ``prob`` among names whose 5-session outcome
    is known: the #1, the top-10 and the #1 under the publication rule.
    Returns (daily, picks) where ``picks`` holds every top-10 row."""
    df = pd.DataFrame({"day": day, "sym": sym, "p": prob, "y": y, "c5": c5, "h5": h5,
                       "pub": pub.astype(bool)})
    df = df[np.isfinite(df["p"]) & np.isfinite(df["y"]) & np.isfinite(df["c5"])]
    if not len(df):
        return pd.DataFrame(), df
    df = df.sort_values(["day", "p"], ascending=[True, False], kind="mergesort")
    g = df.groupby("day", sort=True)
    df["rk"] = g.cumcount() + 1
    top1 = df[df["rk"] == 1].set_index("day")
    top10 = df[df["rk"] <= 10]
    t10 = top10.groupby("day")[["y", "c5"]].mean()
    uni = g[["y", "c5"]].mean()
    pub1 = df[df["pub"]].groupby("day", sort=True).head(1).set_index("day")
    daily = pd.DataFrame({
        "top1_sym": top1["sym"], "top1_y": top1["y"], "top1_c5": top1["c5"], "top1_h5": top1["h5"],
        "top10_y": t10["y"], "top10_c5": t10["c5"], "uni_y": uni["y"], "uni_c5": uni["c5"],
        "pub1_sym": pub1["sym"], "pub1_y": pub1["y"], "pub1_c5": pub1["c5"], "pub1_h5": pub1["h5"],
    })
    return daily, top10


def _swing_sim(daily: pd.DataFrame, col: str, cost: float = COST) -> Dict[str, Any]:
    """Non-overlapping swing shorts: enter on every ``SWING_HOLD``-th test
    session only, short that day's #1 at the next open, cover at the close
    ``SWING_HOLD`` sessions later. 1x notional per trade, summed."""
    empty = {"n_trades": 0, "gross_total": None, "net_total": None, "win_rate": None, "worst_trade": None,
             "best_trade": None, "max_drawdown": None, "mean_trade_net": None, "trades": [],
             "cost_assumption": cost}
    if not len(daily) or col not in daily:
        return empty
    entries = daily.iloc[::SWING_HOLD]
    sym_col = col.replace("_c5", "_sym")
    rows = entries[np.isfinite(entries[col].to_numpy(float))]
    if not len(rows):
        return empty
    g = -rows[col].to_numpy(float)
    net = g - cost
    return {
        "n_trades": int(len(g)),
        "gross_total": _f(g.sum()),
        "net_total": _f(net.sum()),
        "win_rate": _f(np.mean(net > 0)),
        "worst_trade": _f(np.min(g)),
        "best_trade": _f(np.max(g)),
        "max_drawdown": _max_drawdown(net),
        "mean_trade_net": _f(np.mean(net)),
        "trades": [[_day(d), (str(s) if isinstance(s, str) else None), _f(c)]
                   for d, s, c in zip(rows.index, rows[sym_col] if sym_col in rows else [None] * len(rows),
                                      rows[col])],
        "trade_columns": ["entry_signal_date", "symbol", "c5"],
        "cost_assumption": cost,
        "units": (f"one trade every {SWING_HOLD} test sessions (never overlapping); short return = "
                  f"-(close {SWING_HOLD} sessions later / next open - 1); summed at 1x notional"),
    }


def _swing_report(day: np.ndarray, sym: np.ndarray, prob: np.ndarray, y: np.ndarray, c5: np.ndarray,
                  h5: np.ndarray, pub: np.ndarray, cost: float = COST) -> Dict[str, Any]:
    daily, picks = _swing_table(day, sym, prob, y, c5, h5, pub)
    if not len(daily):
        return {"days": 0}
    have_h5 = bool(np.isfinite(daily["top1_h5"].to_numpy(float)).any())

    def sq(x: pd.Series) -> Optional[float]:
        v = x.to_numpy(float)
        v = v[np.isfinite(v)]
        return _f(np.mean(v >= config.SQUEEZE_THRESHOLD)) if len(v) else None

    out: Dict[str, Any] = {
        "days": int(len(daily)),
        "hold_sessions": SWING_HOLD,
        "threshold": SWING_THRESHOLD,
        "base_rate": _f(np.nanmean(y[np.isfinite(prob) & np.isfinite(c5)])) if np.isfinite(y).any() else None,
        "top1_hit": _f(daily["top1_y"].mean()),
        "top1_mean_c5": _f(daily["top1_c5"].mean()),
        "top1_median_c5": _f(daily["top1_c5"].median()),
        "top10_hit": _f(daily["top10_y"].mean()),
        "top10_mean_c5": _f(picks["c5"].mean()),
        "top10_median_c5": _f(picks["c5"].median()),
        "universe_daily_hit": _f(daily["uni_y"].mean()),
        "universe_mean_c5": _f(daily["uni_c5"].mean()),
        "pub1_days": int(daily["pub1_c5"].notna().sum()),
        "pub1_hit": _f(daily["pub1_y"].mean()),
        "pub1_mean_c5": _f(daily["pub1_c5"].mean()),
        "pub1_median_c5": _f(daily["pub1_c5"].median()),
        "sim": _swing_sim(daily, "top1_c5", cost),
        "sim_pub": _swing_sim(daily, "pub1_c5", cost),
    }
    if have_h5:  # squeeze inside the holding window: highest high ≥ +20% above the entry open
        out["top1_squeeze_rate"] = sq(daily["top1_h5"])
        out["top10_squeeze_rate"] = sq(picks["h5"])
        out["pub1_squeeze_rate"] = sq(daily["pub1_h5"])
        out["squeeze_definition"] = (f"highest high over the {SWING_HOLD} sessions ≥ "
                                     f"{config.SQUEEZE_THRESHOLD:+.0%} above the entry open")
    return out


# ── Importance ───────────────────────────────────────────────────────────
def _importance(clf, X: np.ndarray, y: np.ndarray, cols: List[str],
                rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[float]]:
    """AUC drop when a feature (or a whole family, jointly) is shuffled."""
    base = _auc(y, _raw(clf, X)) if clf is not None else None
    if base is None:
        return [], [], None
    perm = rng.permutation(len(y))
    per: List[Dict[str, Any]] = []
    for j, c in enumerate(cols):
        Xp = X.copy()
        Xp[:, j] = X[perm, j]
        a = _auc(y, _raw(clf, Xp))
        per.append({"family": FEATURE_DOCS[c][0], "feature": c,
                    "importance": None if a is None else base - a, "doc": FEATURE_DOCS[c][1]})
    fam: List[Dict[str, Any]] = []
    for f in FAMILIES:
        js = [j for j, c in enumerate(cols) if FEATURE_DOCS[c][0] == f]
        if not js:
            continue
        Xp = X.copy()
        Xp[:, js] = X[perm][:, js]
        a = _auc(y, _raw(clf, Xp))
        fam.append({"family": f, "importance": None if a is None else base - a, "n_features": len(js)})
    per.sort(key=lambda d: -(d["importance"] if d["importance"] is not None else -1e9))
    fam.sort(key=lambda d: -(d["importance"] if d["importance"] is not None else -1e9))
    return per, fam, base


# ── Public API ───────────────────────────────────────────────────────────
def _label_arrays(lab: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Label arrays for training. ``y_swing`` / ``y_pump`` are derived from
    ``y_c5`` / ``y_oc`` (same definitions as ``features``) when an older
    panel lacks them; ``y_h5`` is NaN when absent (squeeze-in-window omitted)."""
    Y = {c: lab[c].to_numpy(dtype=np.float64) for c in ("y_oc", "y_dump", "y_bigdump", "y_squeeze")}
    c5 = _numcol(lab, "y_c5")
    Y["y_c5"] = c5
    with np.errstate(invalid="ignore"):
        Y["y_swing"] = (_numcol(lab, "y_swing") if "y_swing" in lab.columns
                        else np.where(np.isfinite(c5), (c5 <= SWING_THRESHOLD).astype(float), np.nan))
        Y["y_pump"] = (_numcol(lab, "y_pump") if "y_pump" in lab.columns
                       else np.where(np.isfinite(Y["y_oc"]), (Y["y_oc"] >= PUMP_THRESHOLD).astype(float), np.nan))
    Y["y_h5"] = _numcol(lab, "y_h5")
    return Y


def _decile_means(score: np.ndarray, v: np.ndarray) -> List[Optional[float]]:
    ok = np.isfinite(score) & np.isfinite(v)
    if ok.sum() < 10:
        return []
    dec = pd.qcut(pd.Series(score[ok]).rank(method="first"), 10, labels=False).to_numpy()
    return [_f(x) for x in pd.Series(v[ok]).groupby(dec).mean().to_numpy()]


def train(panel: pd.DataFrame, out_dir: Path = config.MODELS,
          site_json: Union[Path, str, None] = DEFAULT_SITE_JSON,
          fold_months: int = 1, params: Optional[Dict[str, Any]] = None,
          features: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Walk-forward evaluation + production fit. Returns the §12 report,
    writes it to ``site_json`` (unless None) and pickles the bundle to
    ``out_dir/bundle.pkl``. ``features`` overrides the M0 column list
    (default ``features.FEATURES``; M1 = it + ``M1_EXTRA``) — used to
    compare feature sets; the bundle records the list it was trained on."""
    import sklearn

    t_all = time.time()
    s = _settings(fold_months, params)
    rng = np.random.default_rng(int(s["seed"]))
    f0: List[str] = list(features) if features is not None else list(M0_FEATURES)
    f1: List[str] = f0 + [c for c in M1_EXTRA if c not in f0]
    unknown = [c for c in f1 if c not in FEATURE_DOCS]
    if unknown:
        raise ValueError(f"unknown feature columns: {unknown[:10]}")
    # morning-only extras (extended-hours shape, overnight filings) come from
    # sources outside the panel builder; if they are absent, M1 trains on NaN
    # for them (the model treats NaN as "not traded / unknown") rather than failing
    extra_absent = [c for c in M1_EXTRA if c != "gap_open" and c not in panel.columns]
    if extra_absent:
        log.warning("train: morning inputs %s absent from the panel — NaN in this fit", extra_absent)
        panel = panel.assign(**{c: np.nan for c in extra_absent})
    need = ["date", "symbol"] + [c for c in f1 if c not in FINRA_FEATURES] + ["y_oc", "y_dump", "y_bigdump", "y_squeeze"]
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing[:10]}")
    if any(c not in panel.columns for c in f1):   # FINRA columns on an old panel → NaN (warned in _matrix)
        log.warning("train: panel predates the FINRA features — they are all NaN in this fit")

    lab = panel[panel["y_dump"].notna() & panel["y_oc"].notna()]
    lab = lab.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    if len(lab) < 1000:
        raise ValueError(f"only {len(lab)} labeled rows — not enough to train")
    row_dates = pd.to_datetime(lab["date"]).to_numpy(dtype="datetime64[ns]")
    dates = np.unique(row_dates)
    di = np.searchsorted(dates, row_dates)
    X = _matrix(lab, f1)
    Y = _label_arrays(lab)
    n0, n1 = len(f0), len(f1)
    log.info("train: %d labeled rows, %d symbols, %d sessions (%s → %s)", len(lab), lab["symbol"].nunique(),
             len(dates), _day(dates[0]), _day(dates[-1]))

    # 1) walk-forward
    t_wf = time.time()
    folds = _fold_windows(dates, s)
    oos = {m: {c: np.full(len(lab), np.nan) for c in OUT_COLUMNS} for m in ("m0", "m1")}
    fold_rows: List[Dict[str, Any]] = []
    for k, fw in enumerate(folds):
        t_f = time.time()
        fit_idx = np.flatnonzero(di <= fw["fit_end"])
        cal_idx = np.flatnonzero((di >= fw["cal_start"]) & (di <= fw["train_end"]))
        te_idx = np.flatnonzero((di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or not len(cal_idx):
            continue
        row = {"fold": k + 1, "fit_end": _day(dates[fw["fit_end"]]),
               "cal_start": _day(dates[fw["cal_start"]]), "train_end": _day(dates[fw["train_end"]]),
               "test_start": _day(dates[fw["test_start"]]), "test_end": _day(dates[fw["test_end"]]),
               "embargo_sessions": int(fw["test_start"] - fw["train_end"] - 1),
               "n_fit": int(len(fit_idx)), "n_cal": int(len(cal_idx)), "n_test": int(len(te_idx)),
               "base_rate": _f(np.mean(Y["y_dump"][te_idx]))}
        for m, nc in (("m0", n0), ("m1", n1)):
            ms = _fit_set(X, Y, fit_idx, cal_idx, nc, s, rng)
            pr = _predict_set(ms, X[te_idx, :nc])
            for c in OUT_COLUMNS:
                oos[m][c][te_idx] = pr[c]
            row[f"auc_{m}"] = _auc(Y["y_dump"][te_idx], pr["prob_dump"])
        row["seconds"] = round(time.time() - t_f, 1)
        fold_rows.append(row)
        log.info("fold %d/%d test %s→%s  n_fit=%d n_test=%d  AUC m0 %.3f m1 %.3f  (%.1fs)", k + 1, len(folds),
                 row["test_start"], row["test_end"], row["n_fit"], row["n_test"],
                 row["auc_m0"] or float("nan"), row["auc_m1"] or float("nan"), row["seconds"])
    wf_s = time.time() - t_wf

    tested = np.isfinite(oos["m0"]["prob_dump"])
    pub_mask = publishable(lab)
    ssr_mask = ssr_next(lab)
    from . import capacity as CAP
    _dvx = CAP.exp_dvol(_numcol(lab, "dvol20"), _numcol(lab, "close") * _numcol(lab, "volume"))
    _spr = CAP.spread_used_vec(_numcol(lab, "spread_est"), _numcol(lab, "price"), _numcol(lab, "dvol20"))
    row_costs = {size: np.asarray(CAP.round_trip_cost(size, _dvx, _numcol(lab, "vol20"), _spr), float)
                 for size in REALISTIC_SIZES}
    y_up5 = Y["y_pump"]
    big_move = np.abs(Y["y_oc"]) >= 0.05
    directional: Dict[str, Any] = {}
    tail: Dict[str, Any] = {}
    caps: Dict[str, Dict[str, Optional[float]]] = {}
    report_oos: Dict[str, Any] = {}
    sims: Dict[str, Any] = {}
    calib: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    swing: Dict[str, Any] = {}
    sym_arr = lab["symbol"].astype(str).to_numpy()
    for m in ("m0", "m1"):
        p = oos[m]["prob_dump"]
        daily = _daily_table(row_dates[tested], p[tested], Y["y_dump"][tested], Y["y_oc"][tested],
                             Y["y_squeeze"][tested], pub_mask[tested], ssr_mask[tested],
                             {k: v[tested] for k, v in row_costs.items()})
        # Is it a DOWN forecast or just a big-move forecast? (the honest question)
        pt, yo = p[tested], Y["y_oc"][tested]
        dec = pd.qcut(pd.Series(pt).rank(method="first"), 10, labels=False)
        directional[m] = {
            "pump_auc": _auc(y_up5[tested], pt),
            "dump_vs_pump_auc": _auc((Y["y_dump"][tested][big_move[tested]]), pt[big_move[tested]]),
            "decile_mean_oc": [_f(v) for v in pd.Series(yo).groupby(dec.to_numpy()).mean().to_numpy()],
            "decile_up5_rate": [_f(v) for v in pd.Series(y_up5[tested]).groupby(dec.to_numpy()).mean().to_numpy()],
            "decile_dump_rate": [_f(v) for v in pd.Series(Y["y_dump"][tested]).groupby(dec.to_numpy()).mean().to_numpy()],
            "up5_base_rate": _f(np.mean(y_up5[tested])),
        }
        # the dedicated pump / swing models, and skew as a direction score
        pp_t = oos[m]["prob_pump"][tested]
        ps_t = oos[m]["prob_swing"][tested]
        sk_t = oos[m]["skew"][tested]
        ysw = Y["y_swing"][tested]
        bm = big_move[tested]
        directional[m].update({
            "pump_model_auc": _auc(y_up5[tested], pp_t),
            "pump_base_rate": _f(np.nanmean(y_up5[tested])),
            "skew_dump_vs_pump_auc": _auc(Y["y_dump"][tested][bm], sk_t[bm]),
            "swing_auc": _auc(ysw, ps_t),
            "swing_base_rate": _f(np.nanmean(ysw)) if np.isfinite(ysw).any() else None,
            "dump_score_swing_auc": _auc(ysw, pt),
            "swing_decile_mean_c5": _decile_means(ps_t, Y["y_c5"][tested]),
            "swing_decile_rate": _decile_means(ps_t, ysw),
        })
        # Tail calibration → caps: never show more certainty than the OOS tail delivered.
        caps[m] = {}
        tail[m] = {}
        for t in TARGETS:
            pp, yy = oos[m][PROB_COLS[t]][tested], Y[TARGETS[t]][tested]
            hi = np.isfinite(pp) & np.isfinite(yy) & (pp >= CAP_FROM)
            n_hi = int(hi.sum())
            realized = _f(np.mean(yy[hi])) if n_hi else None
            cap = max(realized, 0.3) if (n_hi >= 30 and realized is not None) else CAP_FROM + 0.1
            caps[m][t] = cap
            tail[m][t] = {"from": CAP_FROM, "n": n_hi, "mean_pred": _f(np.mean(pp[hi])) if n_hi else None,
                          "realized": realized, "cap": cap}
        report_oos[m] = _metrics(Y["y_dump"][tested], p[tested], daily)
        sims[m] = _sim(daily)
        calib[m] = _calibration(Y["y_dump"][tested], p[tested])
        for t in ("bigdump", "squeeze", "swing", "pump"):
            yy, pp = Y[TARGETS[t]][tested], oos[m][PROB_COLS[t]][tested]
            ok = np.isfinite(pp) & np.isfinite(yy)
            extra[f"{m}_{t}"] = {"auc": _auc(yy, pp),
                                 "brier": _f(np.mean((pp[ok] - yy[ok]) ** 2)) if ok.any() else None,
                                 "base_rate": _f(np.mean(yy[ok])) if ok.any() else None,
                                 "n": int(ok.sum()),
                                 "calibration": _calibration(yy, pp)}
        swing[m] = _swing_report(row_dates[tested], sym_arr[tested], oos[m]["prob_swing"][tested],
                                 Y["y_swing"][tested], Y["y_c5"][tested], Y["y_h5"][tested], pub_mask[tested])
        eo = oos[m]["exp_oc"][tested]
        yo = Y["y_oc"][tested]
        ok = np.isfinite(eo)
        extra[f"{m}_exp_oc"] = {
            "spearman": _f(pd.Series(eo[ok]).corr(pd.Series(np.clip(yo[ok], *OC_CLIP)), method="spearman"))
            if ok.sum() > 10 else None,
            "mae_clipped": _f(np.mean(np.abs(eo[ok] - np.clip(yo[ok], *OC_CLIP)))) if ok.any() else None,
        }
    report_oos["extra"] = extra

    # 2) production fit: same recipe, window extended to the last labeled session
    t_fin = time.time()
    last = len(dates) - 1
    cal_start = max(last - int(s["cal_sessions"]) + 1, 0)
    fit_end = cal_start - int(s["embargo"]) - 1
    if fit_end < 20:
        raise ValueError("not enough history for a production fit")
    fit_idx = np.flatnonzero(di <= fit_end)
    cal_idx = np.flatnonzero(di >= cal_start)
    prod = {m: _fit_set(X, Y, fit_idx, cal_idx, nc, s, rng) for m, nc in (("m0", n0), ("m1", n1))}
    for m in prod:
        for t in TARGETS:
            if prod[m].get(t) is not None:
                prod[m][t]["cap"] = caps.get(m, {}).get(t)
    fin_s = time.time() - t_fin

    # 3) permutation importance on the held-out calibration slice (not seen by the fit)
    t_imp = time.time()
    k = min(len(cal_idx), int(s["importance_rows"]))
    imp_idx = np.sort(rng.choice(cal_idx, k, replace=False)) if k < len(cal_idx) else cal_idx
    imp0, fam0, base0 = _importance(prod["m0"]["dump"]["clf"], X[imp_idx, :n0], Y["y_dump"][imp_idx], f0, rng)
    imp1, fam1, base1 = _importance(prod["m1"]["dump"]["clf"], X[imp_idx, :n1], Y["y_dump"][imp_idx], f1, rng)
    imp_s = time.time() - t_imp

    # 4) report
    all_dates = pd.to_datetime(panel["date"])
    later = all_dates[all_dates > pd.Timestamp(dates[-1])]
    trained_through = _day(later.min()) if len(later) else _day(dates[-1])
    n_iters = {f"{m}_{t}": int(getattr(prod[m][t]["clf"], "n_iter_", 0) or 0) for m in prod for t in TARGETS}
    notes = [
        f"HistGradientBoosting (NaN-native), {len(f0)} features for M0, "
        f"{len(f1)} for M1 (+ gap_open). learning_rate {s['hgb']['learning_rate']}, "
        f"≤ {s['hgb']['max_iter']} trees with early stopping on a random 10% of each fit window.",
        f"Walk-forward: expanding window, {s['fold_months']}-month test folds, embargo "
        f"{s['embargo']} sessions before every test fold and before every calibration slice; "
        f"first test fold after {s['min_train_sessions']} labeled sessions.",
        f"Calibration: isotonic regression on the last {s['cal_sessions']} sessions of each training "
        f"window (held out from the tree fit, later in time).",
        f"Row cap: at most {s['max_fit_rows']:,} rows per fit. Classifiers keep all positives (up to half "
        f"the cap) and a random share of negatives, re-weighted to the true base rate; regressors use a "
        f"uniform random sample. Calibration and evaluation always use every row.",
        f"exp_oc regressors are fit on the whole training window with outcomes clipped to "
        f"[{OC_CLIP[0]:+.0%}, {OC_CLIP[1]:+.0%}].",
        "prob_bigdump is capped at prob_dump (a −15% day is also a −5% day).",
        f"Production model: fit through {_day(dates[fit_end])}, calibrated on "
        f"{_day(dates[cal_start])} → {_day(dates[last])} — the exact recipe the walk-forward evaluated.",
        "Top-k and simulation rankings use each day's eligible names only (price ≥ $0.10, 20-day median "
        "dollar volume ≥ $50k, ≥ 60 prior sessions); no shortability filter.",
    ]
    report: Dict[str, Any] = {
        "trained_at": _iso_utc(),
        "trained_through": trained_through,
        "last_label_date": _day(dates[-1]),
        "n_rows": int(len(lab)),
        "n_symbols": int(lab["symbol"].nunique()),
        "n_days": int(len(dates)),
        "targets": TARGET_TEXT,
        "base_rate": {t: _f(np.nanmean(Y[c])) if np.isfinite(Y[c]).any() else None for t, c in TARGETS.items()},
        "oos": report_oos,
        "swing": swing,
        "calibration": calib,
        "importance": imp0,
        "importance_family": fam0,
        "importance_m1": imp1,
        "importance_m1_family": fam1,
        "importance_meta": {"model": "m0 dump (m1 in importance_m1*)", "metric": "AUC drop when shuffled",
                            "rows": int(len(imp_idx)), "slice": f"{_day(dates[cal_start])} → {_day(dates[last])}",
                            "base_auc_m0": base0, "base_auc_m1": base1},
        "sim": sims,
        "directional": directional,
        "tail_calibration": tail,
        "publication_rule": {
            "text": (f"#1 = highest P(dump) among names NOT under the Rule 201 short-sale restriction the next "
                     f"session (day-t low ≥ {SSR_DROP:.0%} below the prior close) and with a 20-day median dollar "
                     f"volume ≥ ${LIQ_FLOOR:,.0f}. Live, the #1 must also be borrowable at IBKR and have squeeze "
                     f"danger < 70 — those two conditions have no history and are not in this backtest."),
            "ssr_drop": SSR_DROP, "liq_floor": LIQ_FLOOR,
        },
        "walk_forward": {
            "n_folds": len(fold_rows), "fold_months": s["fold_months"], "embargo_sessions": s["embargo"],
            "min_train_sessions": s["min_train_sessions"], "cal_sessions": s["cal_sessions"],
            "oos_start": fold_rows[0]["test_start"] if fold_rows else None,
            "oos_end": fold_rows[-1]["test_end"] if fold_rows else None,
            "oos_rows": int(tested.sum()),
            "fits_per_fold": 2 * (len(TARGETS) + 1), "final_fits": 2 * (len(TARGETS) + 1),
        },
        "folds": fold_rows,
        "production": {"fit_through": _day(dates[fit_end]), "calibration_start": _day(dates[cal_start]),
                       "calibration_end": _day(dates[last]), "n_fit_rows": int(len(fit_idx)),
                       "n_cal_rows": int(len(cal_idx)), "n_iter": n_iters},
        "training_notes": notes,
        "caveats": CAVEATS,
        "params": {"hgb": s["hgb"], "max_fit_rows": s["max_fit_rows"], "cost": COST, "oc_clip": list(OC_CLIP)},
        "features": {"m0": f0, "m1": f1},
        "morning_set": _features_mod.MORNING_SET,
        "sklearn_version": sklearn.__version__,
        "timing": {"walk_forward_s": round(wf_s, 1), "final_fit_s": round(fin_s, 1),
                   "importance_s": round(imp_s, 1), "total_s": round(time.time() - t_all, 1)},
    }
    report = clean(report)

    bundle = {
        "version": BUNDLE_VERSION,
        "trained_at": report["trained_at"],
        "features": f0,
        "m1_features": f1,
        "m0": prod["m0"],
        "m1": prod["m1"],
        "report": report,
        "sklearn_version": sklearn.__version__,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (BUNDLE_NAME + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_dir / BUNDLE_NAME)
    if site_json is not None:
        p = Path(site_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        tj = p.with_suffix(".tmp")
        tj.write_text(json.dumps(report, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        os.replace(tj, p)
    log.info("train: done in %.1fs (walk-forward %.1fs, %d folds; final %.1fs; importance %.1fs)",
             time.time() - t_all, wf_s, len(fold_rows), fin_s, imp_s)
    return report


def load(out_dir: Path = config.MODELS) -> Optional[Dict[str, Any]]:
    """The production bundle, or None when missing/unreadable."""
    p = Path(out_dir) / BUNDLE_NAME
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            b = pickle.load(fh)
    except Exception as e:  # noqa: BLE001 — corrupt/incompatible pickle → no model
        log.warning("model: cannot load %s: %s", p, e)
        return None
    if not isinstance(b, dict) or "m0" not in b or "m1" not in b:
        log.warning("model: %s is not a GRAVITY bundle", p)
        return None
    try:
        import sklearn

        if b.get("sklearn_version") and b["sklearn_version"] != sklearn.__version__:
            log.warning("model: bundle trained with scikit-learn %s, running %s — retrain recommended",
                        b["sklearn_version"], sklearn.__version__)
    except ImportError:  # pragma: no cover
        pass
    return b


def predict(bundle: Dict[str, Any], rows: pd.DataFrame, use_open: bool) -> pd.DataFrame:
    """prob_dump, prob_bigdump, prob_squeeze, prob_swing, prob_pump, skew,
    exp_oc (+ ``model``: m0/m1), indexed like ``rows``. With ``use_open``
    rows that have a finite ``gap_open`` get M1; the rest fall back to M0.
    A bundle trained before a target existed gives NaN for it (and skew)."""
    n = len(rows)
    vals = {c: np.full(n, np.nan) for c in OUT_COLUMNS}
    which = np.full(n, "m0", dtype=object)
    if bundle is not None and n:
        f0, f1, X, use1 = _model_masks(bundle, rows, use_open)
        if use_open and "gap_open" not in rows.columns:
            log.warning("predict: use_open=True but rows have no gap_open — using M0 for every row")
        for m, mask, cols in (("m0", ~use1, len(f0)), ("m1", use1, len(f1))):
            if not mask.any():
                continue
            pr = _predict_set(bundle[m], X[mask, :cols])
            for c in OUT_COLUMNS:
                vals[c][mask] = pr[c]
            which[mask] = m
    out = pd.DataFrame(vals, index=rows.index)[OUT_COLUMNS]
    out["model"] = which
    return out


def _model_masks(bundle: Dict[str, Any], rows: pd.DataFrame, use_open: bool):
    f0 = list(bundle.get("features") or M0_FEATURES)
    f1 = list(bundle.get("m1_features") or (f0 + list(M1_EXTRA)))
    X = _matrix(rows, f1)
    use1 = np.zeros(len(rows), dtype=bool)
    if use_open and "m1_ok" in rows.columns:          # the morning run chose the M1 rows itself
        use1 = rows["m1_ok"].fillna(False).to_numpy(bool)
    elif use_open and "gap_open" in rows.columns and "gap_open" in f1:
        use1 = np.isfinite(X[:, f1.index("gap_open")])
    return f0, f1, X, use1


def attribute(bundle: Dict[str, Any], rows: pd.DataFrame, use_open: bool = False) -> pd.DataFrame:
    """Which feature families drive each row's dump score (CONTRACTS §14.2).

    For every family, all of its columns are replaced by their medians over
    ``rows`` (the day's scored universe — pass all of it, not one name) and
    the *raw* (uncalibrated) score of the dump classifier is recomputed;
    ``attr_<family>`` = max(0, raw − raw_replaced), normalised so a row's
    attributions sum to 1 (NaN when no family pushes the score up). One
    batched predict per family per model set. Descriptive, not causal: a
    tree model's families interact, and the shares answer "how much lower
    would the score be if this family looked typical for today".
    """
    cols = [f"attr_{f}" for f in FAMILIES]
    out = pd.DataFrame(np.nan, index=rows.index, columns=cols)
    if bundle is None or not len(rows):
        return out
    f0, f1, X, use1 = _model_masks(bundle, rows, use_open)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column → NaN median (stays unknown)
        med = np.nanmedian(X.astype(np.float64), axis=0).astype(np.float32)
    vals = np.full((len(rows), len(FAMILIES)), np.nan)
    for m, mask, names in (("m0", ~use1, f0), ("m1", use1, f1)):
        cm = (bundle.get(m) or {}).get("dump") or {}
        clf = cm.get("clf")
        if clf is None or not mask.any():
            continue
        Xm = X[mask, :len(names)]
        raw = _raw(clf, Xm)
        drops = np.zeros((len(Xm), len(FAMILIES)))
        for k, fam in enumerate(FAMILIES):
            js = [j for j, c in enumerate(names) if FEATURE_DOCS.get(c, ("",))[0] == fam]
            if not js:
                continue
            Xr = Xm.copy()
            Xr[:, js] = med[js]
            drops[:, k] = np.maximum(0.0, raw - _raw(clf, Xr))
        tot = drops.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where((tot > 0)[:, None], drops / tot[:, None], np.nan)
        vals[mask] = share
    out.loc[:, cols] = vals
    return out


def time_one_fit(lab: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Time one capped y_dump classifier fit at the size of ``lab`` (scale test)."""
    s = _settings(1, params)
    lab = lab[lab["y_dump"].notna()]
    X = _matrix(lab, M1_FEATURES)
    y = lab["y_dump"].to_numpy(dtype=np.float64)
    t = time.time()
    clf = _fit_classifier(X, y, s, np.random.default_rng(0))
    fit_s = time.time() - t
    return {"fit_s": fit_s, "n_rows": int(len(y)), "n_train": int(min(len(y), s["max_fit_rows"])),
            "n_iter": int(getattr(clf, "n_iter_", 0) or 0)}
