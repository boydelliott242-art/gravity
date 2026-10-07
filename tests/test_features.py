"""Tests for gravity.features (CONTRACTS.md §7).

The point of these tests is to *prove* there is no lookahead:

* changing, adding or deleting bars dated after ``t`` never changes a
  feature at ``t`` (for any symbol, including the cross-sectional ones);
* FilingEvents dated after ``t`` are invisible at ``t``; ones dated ``t`` are
  visible at ``t``;
* a split that happens after ``t`` (with history re-adjusted the way Yahoo
  does it) leaves every feature at ``t`` exactly as it looked at ``t``;
* labels at ``t`` describe session ``t+1`` exactly, and are NaN when ``t+1``
  is halted, missing or a bad print.

Everything here is synthetic and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gravity import config  # noqa: E402
from gravity import features as F  # noqa: E402

N = 160
DATES = pd.bdate_range("2024-01-02", periods=N)
SYMS = ["AAA", "BBB", "CCC", "DDD"]
FEAT = F.FEATURES


def make_bars(seed: int, n: int = N, p0: float = 5.0, vol: float = 1e6,
              dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = DATES[:n] if dates is None else dates
    c = p0 * np.exp(np.cumsum(rng.normal(0.0, 0.05, len(idx))))
    o = c * np.exp(rng.normal(0.0, 0.02, len(idx)))
    h = np.maximum(o, c) * (1.0 + rng.uniform(0.0, 0.04, len(idx)))
    lo = np.minimum(o, c) * (1.0 - rng.uniform(0.0, 0.04, len(idx)))
    v = rng.uniform(0.5, 1.5, len(idx)) * vol
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)


def universe(seed: int = 0) -> Dict[str, pd.DataFrame]:
    return {s: make_bars(seed + i) for i, s in enumerate(SYMS)}


def bench() -> pd.DataFrame:
    return make_bars(99, p0=200.0, vol=3e7)


STATIC = pd.DataFrame({"symbol": SYMS, "asia": [True, False, False, True],
                       "ipo_year": [2020, np.nan, 2019, 2025]})


def ev(sym: str, day: str, form: str, items=()) -> dict:
    return {"symbol": sym, "cik": 1, "date": day, "accepted": None, "form": form,
            "items": list(items), "category": "other", "url": None, "text_tags": []}


def build(hist, events=None, splits=None, static=STATIC, bm="default", min_date=None) -> pd.DataFrame:
    return F.build_panel(hist, {} if events is None else events, splits or {}, static,
                         bench() if isinstance(bm, str) else bm, min_date=min_date)


def keyed(p: pd.DataFrame) -> pd.DataFrame:
    return p.set_index(["date", "symbol"]).sort_index()


def assert_features_equal(a: pd.DataFrame, b: pd.DataFrame, cols: List[str] = FEAT) -> None:
    a, b = keyed(a), keyed(b)
    assert list(a.index) == list(b.index), "row sets differ"
    for c in cols:
        np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-6, atol=1e-9,
                                   equal_nan=True, err_msg=c)


# ── contract shape ───────────────────────────────────────────────────────
def test_column_contract():
    assert len(FEAT) == len(set(FEAT)) == 75 + (len(F.FINRA_FEATURES) if F.USE_FINRA else 0)
    assert F.M1_EXTRA == ["gap_open"]
    assert F.LABELS == ["y_oc", "y_co", "y_gap", "y_ol", "y_oh", "y_c5", "y_dump", "y_bigdump", "y_squeeze",
                        "y_swing", "y_pump", "y_h5"]
    assert F.PANEL_FEATURES[:len(FEAT)] == FEAT and set(F.FINRA_FEATURES) <= set(F.PANEL_FEATURES)
    assert set(F.FEATURE_DOCS) == set(F.PANEL_FEATURES) | set(F.M1_EXTRA)
    assert all(F.FEATURE_DOCS[c][0] == "flow" for c in F.FINRA_FEATURES)
    assert all(fam in F.FAMILIES and desc for fam, desc in F.FEATURE_DOCS.values())
    assert "gap_open" not in FEAT  # M0 must never see the open
    p = build(universe(), {s: [] for s in SYMS})
    assert list(p.columns) == F.INFO_COLUMNS + F.PANEL_FEATURES + F.M1_EXTRA + F.LABELS == F.PANEL_COLUMNS
    assert p[F.FINRA_FEATURES].isna().all().all()     # no short_vol passed → unknown, not zero
    assert not p.duplicated(["date", "symbol"]).any()
    X = p[FEAT].to_numpy(float)
    assert not np.isinf(X).any()
    rsi = p[["rsi14", "rsi2"]].to_numpy(float)
    assert np.nanmin(rsi) >= 0 and np.nanmax(rsi) <= 100
    fams = F.feature_families()
    assert sum(len(v) for v in fams.values()) == len(FEAT) + 1


# ── lookahead: bars ──────────────────────────────────────────────────────
@pytest.mark.parametrize("t_i", [70, 100, 140])
def test_future_bars_do_not_change_features(t_i):
    t = DATES[t_i]
    base = universe()
    evs = {s: [ev(s, str(DATES[20].date()), "424B5"), ev(s, str(DATES[t_i - 3].date()), "8-K", ["3.01"])] for s in SYMS}
    p0 = build(base, evs)

    mutated = {s: df.copy() for s, df in base.items()}
    rng = np.random.default_rng(7)
    after = mutated["BBB"].index > t
    k = int(after.sum())
    mult = np.exp(rng.normal(0.0, 0.5, k))
    for c in ("open", "high", "low", "close"):
        mutated["BBB"].loc[after, c] *= mult
    mutated["BBB"].loc[after, "volume"] *= rng.uniform(0.0, 50.0, k)
    mutated["CCC"] = mutated["CCC"][mutated["CCC"].index <= t]            # delisted after t
    mutated["AAA"].loc[after, ["open", "high", "low", "close"]] = np.nan   # halted after t
    mutated["EEE"] = make_bars(55, dates=DATES[t_i + 1:])                 # lists after t
    p1 = build(mutated, evs)

    a = p0[p0["date"] <= t]
    b = p1[p1["date"] <= t]
    assert len(a) > 0
    assert_features_equal(a, b)
    assert_features_equal(a, b, cols=["spread_est", "dvol20"])   # cost-model info columns: no lookahead either
    # sanity: the mutation really did change the future
    fa = keyed(p0[p0["date"] > t])
    fb = keyed(p1[p1["date"] > t])
    common = fa.index.intersection(fb.index)
    assert not np.allclose(fa.loc[common, "r1"].to_numpy(float), fb.loc[common, "r1"].to_numpy(float), equal_nan=True)


def test_labels_change_but_features_do_not_when_next_bar_changes():
    t_i = 90
    base = universe()
    p0 = build(base)
    mod = {s: df.copy() for s, df in base.items()}
    mod["DDD"].iloc[t_i + 1, :4] = mod["DDD"].iloc[t_i, 3] * np.array([1.5, 2.0, 1.4, 1.9])
    p1 = build(mod)
    t = DATES[t_i]
    assert_features_equal(p0[p0["date"] <= t], p1[p1["date"] <= t])
    r0 = keyed(p0).loc[(t, "DDD")]
    r1 = keyed(p1).loc[(t, "DDD")]
    assert r0["y_oc"] != r1["y_oc"]
    assert r1["y_gap"] == pytest.approx(0.5, rel=1e-5)
    assert r1["gap_open"] == r1["y_gap"]


# ── lookahead: filings ───────────────────────────────────────────────────
def test_future_events_invisible_and_same_day_events_visible():
    t_i = 100
    t = DATES[t_i]
    hist = universe()
    none = {s: [] for s in SYMS}
    later = dict(none)
    later["AAA"] = [
        ev("AAA", str(DATES[t_i + 1].date()), "424B5"),
        ev("AAA", str(DATES[t_i + 5].date()), "8-K", ["3.01", "3.02"]),
        ev("AAA", str(DATES[t_i + 2].date()), "S-1"),
        ev("AAA", "2099-01-01", "424B4"),
    ]
    p0 = build(hist, none)
    p1 = build(hist, later)
    assert_features_equal(p0[p0["date"] <= t], p1[p1["date"] <= t])
    k1 = keyed(p1)
    nxt = k1.loc[(DATES[t_i + 1], "AAA")]
    assert nxt["n_offer_30"] == 1 and nxt["sess_since_offer"] == 0
    assert np.isnan(nxt["sess_since_reg"]) and nxt["n_reg_30"] == 0
    assert k1.loc[(DATES[t_i + 2], "AAA"), "n_reg_30"] == 1
    assert k1.loc[(DATES[t_i + 5], "AAA"), "n_delist_90"] == 1
    assert k1.loc[(DATES[t_i + 5], "AAA"), "n_unreg_90"] == 1
    assert k1.loc[(DATES[t_i + 4], "AAA"), "n_delist_90"] == 0

    same = dict(none)
    same["AAA"] = [ev("AAA", str(t.date()), "424B5")]
    k2 = keyed(build(hist, same))
    assert k2.loc[(t, "AAA"), "n_offer_30"] == 1
    assert k2.loc[(t, "AAA"), "sess_since_offer"] == 0
    assert k2.loc[(DATES[t_i + 1], "AAA"), "sess_since_offer"] == 1
    assert k2.loc[(DATES[t_i - 1], "AAA"), "n_offer_30"] == 0
    # windows are calendar days: counted for 30 days, then gone
    d30 = t + pd.Timedelta(days=30)
    in_win = [d for d in DATES if t <= d < d30]
    out_win = [d for d in DATES if d >= d30]
    assert all(k2.loc[(d, "AAA"), "n_offer_30"] == 1 for d in in_win)
    assert k2.loc[(out_win[0], "AAA"), "n_offer_30"] == 0
    assert k2.loc[(out_win[0], "AAA"), "n_offer_90"] == 1


def test_events_absent_is_nan_empty_is_zero():
    hist = universe()
    p = keyed(build(hist, {"AAA": [], "BBB": [ev("BBB", str(DATES[10].date()), "424B5")]}))
    a = p.xs("AAA", level="symbol")
    b = p.xs("BBB", level="symbol")
    c = p.xs("CCC", level="symbol")
    assert (a["n_offer_90"] == 0).all() and a["sess_since_offer"].isna().all()
    assert b["sess_since_offer"].notna().all()
    assert c["n_offer_90"].isna().all() and c["n_filings_30"].isna().all()


def test_event_before_first_bar_counts_the_gap():
    hist = universe()
    first = DATES[0]
    pre = first - pd.Timedelta(days=28)                    # 4 weeks before the history starts
    p = keyed(build(hist, {"AAA": [ev("AAA", str(pre.date()), "S-1")]}))
    row = p.loc[(DATES[60], "AAA")]
    expected = 60 + np.busday_count(pre.date(), first.date())
    assert row["sess_since_reg"] == expected


# ── lookahead: splits ────────────────────────────────────────────────────
def test_future_split_leaves_point_in_time_features_unchanged():
    """World A = what you could see at t (raw bars up to t, no split yet).
    World B = seen later: a 1:10 reverse split at s > t, history
    re-adjusted Yahoo-style (prices ×10, volume ÷10 before s)."""
    raw = universe()
    raw["BBB"] = make_bars(33, p0=0.30)
    s_i, t_i = 120, 110
    s, t = DATES[s_i], DATES[t_i]
    raw["BBB"].iloc[s_i:, :4] *= 10.0                        # after the split it trades 10x higher
    raw["BBB"].iloc[s_i:, 4] /= 10.0
    adj = {k: v.copy() for k, v in raw.items()}
    adj["BBB"].iloc[:s_i, :4] *= 10.0
    adj["BBB"].iloc[:s_i, 4] /= 10.0
    splits = {"BBB": [{"date": str(s.date()), "ratio": 0.1}]}

    seen_at_t = {k: v[v.index <= t] for k, v in raw.items()}
    pa = build(seen_at_t)
    pb = build(adj, splits=splits)
    assert_features_equal(pa, pb[pb["date"] <= t])
    kb = keyed(pb)
    # as-traded price is restored (not the adjusted level), and the split
    # shows up only from its effective date
    np.testing.assert_allclose(kb.loc[(t, "BBB"), "price"], raw["BBB"].loc[t, "close"], rtol=1e-5)
    np.testing.assert_allclose(kb.loc[(t, "BBB"), "close"], adj["BBB"].loc[t, "close"], rtol=1e-5)
    assert kb.loc[(DATES[s_i - 1], "BBB"), "rs_count_1y"] == 0
    assert kb.loc[(s, "BBB"), "rs_count_1y"] == 1
    assert kb.loc[(s, "BBB"), "sess_since_rs"] == 0
    assert kb.loc[(s, "BBB"), "rs_last_factor"] == pytest.approx(10.0)
    assert kb.loc[(DATES[s_i + 3], "BBB"), "sess_since_rs"] == 3
    # the split-day label is not polluted: adjusted bars are continuous
    assert abs(kb.loc[(DATES[s_i - 1], "BBB"), "y_gap"]) < 0.5


def test_price_filter_uses_as_traded_price():
    raw = universe()
    raw["BBB"] = make_bars(34, p0=0.05)                       # sub-$0.10 as traded
    s_i = 130
    raw["BBB"].iloc[s_i:, :4] *= 50.0
    raw["BBB"].iloc[s_i:, 4] /= 50.0
    adj = {k: v.copy() for k, v in raw.items()}
    adj["BBB"].iloc[:s_i, :4] *= 50.0                         # adjusted level looks like $2.50
    adj["BBB"].iloc[:s_i, 4] /= 50.0
    p = build(adj, splits={"BBB": [{"date": str(DATES[s_i].date()), "ratio": 0.02}]})
    b = p[p["symbol"] == "BBB"]
    as_traded_before = raw["BBB"]["close"].iloc[:s_i]
    ok_before = as_traded_before[as_traded_before >= config.MIN_PRICE].index
    assert set(b.loc[b["date"] < DATES[s_i], "date"]) <= set(ok_before)
    assert (b["price"] >= config.MIN_PRICE).all()
    assert (b["date"] >= DATES[s_i]).any()


# ── labels ───────────────────────────────────────────────────────────────
def test_labels_describe_next_session_exactly():
    hist = universe()
    p = keyed(build(hist))
    df = hist["CCC"]
    for t_i in (61, 80, 120, N - 7):
        t, t1, t5 = DATES[t_i], DATES[t_i + 1], DATES[t_i + 5]
        r = p.loc[(t, "CCC")]
        O1, H1, L1, C1 = (df.loc[t1, k] for k in ("open", "high", "low", "close"))
        C0 = df.loc[t, "close"]
        np.testing.assert_allclose(r["y_oc"], C1 / O1 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["y_gap"], O1 / C0 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["y_co"], C1 / C0 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["y_ol"], L1 / O1 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["y_oh"], H1 / O1 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["y_c5"], df.loc[t5, "close"] / O1 - 1, rtol=1e-5)
        np.testing.assert_allclose(r["gap_open"], r["y_gap"], rtol=1e-7)
        assert r["y_dump"] == float(C1 / O1 - 1 <= config.DUMP_THRESHOLD)
        assert r["y_bigdump"] == float(C1 / O1 - 1 <= config.BIG_DUMP_THRESHOLD)
        assert r["y_squeeze"] == float(H1 / O1 - 1 >= config.SQUEEZE_THRESHOLD)
        assert r["y_swing"] == float(np.float32(df.loc[t5, "close"] / O1 - 1) <= F.SWING_THRESHOLD)
        assert r["y_pump"] == float(np.float32(C1 / O1 - 1) >= F.PUMP_THRESHOLD)
        np.testing.assert_allclose(r["y_h5"], df["high"].iloc[t_i + 1:t_i + 6].max() / O1 - 1, rtol=1e-5)
        # features at t use t's own bar
        np.testing.assert_allclose(r["r1"], C0 / df["close"].iloc[t_i - 1] - 1, rtol=1e-5)
        np.testing.assert_allclose(r["intraday"], C0 / df.loc[t, "open"] - 1, rtol=1e-5)
    last = p.xs("CCC", level="symbol").iloc[-1]
    assert np.isnan(last[F.LABELS].to_numpy(float)).all() and np.isnan(last["gap_open"])
    assert np.isnan(p.xs("CCC", level="symbol").iloc[-3]["y_c5"])
    # y_swing / y_h5 need t+5; y_pump only needs t+1
    r3 = p.xs("CCC", level="symbol").iloc[-3]
    assert np.isnan(r3["y_swing"]) and np.isnan(r3["y_h5"]) and np.isfinite(r3["y_pump"])


def test_threshold_edges():
    hist = universe()
    df = hist["AAA"]
    t_i = 100
    t1 = DATES[t_i + 1]
    df.loc[t1, ["open", "high", "low", "close"]] = [1.0, 1.25, 0.8, 0.95]    # oc = −5%, oh = +25%
    df.loc[DATES[t_i + 2], ["open", "high", "low", "close"]] = [1.0, 1.1, 0.8, 0.85]  # oc = −15%
    p = keyed(build(hist))
    r = p.loc[(DATES[t_i], "AAA")]
    assert r["y_dump"] == 1.0 and r["y_bigdump"] == 0.0 and r["y_squeeze"] == 1.0
    r2 = p.loc[(t1, "AAA")]
    assert r2["y_dump"] == 1.0 and r2["y_bigdump"] == 1.0 and r2["y_squeeze"] == 0.0
    assert r["y_pump"] == 0.0 and r2["y_pump"] == 0.0


def test_swing_and_pump_label_edges():
    hist = universe()
    df = hist["AAA"]
    t_i = 100
    o1 = float(df["open"].iloc[t_i + 1])
    # close 5 sessions later exactly −15% from the next open → swing; +5% open→close next day → pump
    df.iloc[t_i + 5, df.columns.get_loc("close")] = o1 * 0.85
    df.iloc[t_i + 5, df.columns.get_loc("low")] = min(df["low"].iloc[t_i + 5], o1 * 0.85)
    df.iloc[t_i + 1, df.columns.get_loc("close")] = o1 * 1.05
    df.iloc[t_i + 1, df.columns.get_loc("high")] = max(df["high"].iloc[t_i + 1], o1 * 1.05)
    df.iloc[t_i + 3, df.columns.get_loc("high")] = o1 * 1.60          # adverse excursion inside the window
    p = keyed(build(hist))
    r = p.loc[(DATES[t_i], "AAA")]
    assert r["y_swing"] == 1.0 and r["y_pump"] == 1.0
    np.testing.assert_allclose(r["y_h5"], 0.60, rtol=1e-5)
    # one cent less of a drop → not a swing
    df.iloc[t_i + 5, df.columns.get_loc("close")] = o1 * 0.86
    r = keyed(build(hist)).loc[(DATES[t_i], "AAA")]
    assert r["y_swing"] == 0.0
    # a halted session at t+5 → no swing label, pump still known
    df.iloc[t_i + 5, df.columns.get_loc("volume")] = 0.0
    r = keyed(build(hist)).loc[(DATES[t_i], "AAA")]
    assert np.isnan(r["y_swing"]) and np.isnan(r["y_h5"]) and r["y_pump"] == 1.0


@pytest.mark.parametrize("how", ["zero_volume", "nan_prices", "missing_row", "bad_print"])
def test_halted_or_missing_next_session_gives_nan_labels(how):
    hist = universe()
    t_i = 100
    t, t1 = DATES[t_i], DATES[t_i + 1]
    df = hist["BBB"]
    if how == "zero_volume":
        df.loc[t1, ["open", "high", "low", "close"]] = df.loc[t, "close"]
        df.loc[t1, "volume"] = 0.0
    elif how == "nan_prices":
        df.loc[t1, ["open", "high", "low", "close"]] = np.nan
        df.loc[t1, "volume"] = 0.0
    elif how == "missing_row":
        hist["BBB"] = df.drop(index=t1)
    else:
        df.loc[t1, ["open", "high", "low", "close"]] = df.loc[t, "close"] * 40.0
    p = keyed(build(hist))
    r = p.loc[(t, "BBB")]
    assert np.isnan(r[F.LABELS].to_numpy(float)).all(), how
    assert np.isnan(r["gap_open"])
    # the rows around it are unaffected
    assert np.isfinite(p.loc[(DATES[t_i - 1], "BBB"), "y_oc"]) or how == "missing_row"
    assert np.isfinite(p.loc[(t, "AAA"), "y_oc"])
    if how in ("zero_volume", "nan_prices"):
        # the halted session is still a row (flat bar, volume 0) with finite features
        h = p.loc[(t1, "BBB")]
        assert h["volume"] == 0 and h["halt_20"] >= 1
        assert h["close"] == pytest.approx(df.loc[t, "close"], rel=1e-6)
        # y_c5 from t−4: t+1 lies inside, but the label only needs the t−3 bar
        # and t+1 = (t−4)+5 is halted → NaN
        assert np.isnan(p.loc[(DATES[t_i - 4], "BBB"), "y_c5"])


# ── row filters ──────────────────────────────────────────────────────────
def test_row_filters():
    hist = universe()
    hist["LOWV"] = make_bars(11, p0=2.0, vol=1_000)            # ~$2k/day: below MIN_DOLLAR_VOLUME_20D
    hist["PENY"] = make_bars(12, p0=0.02)                      # below MIN_PRICE
    hist["YOUNG"] = make_bars(13, dates=DATES[-70:])            # only 70 bars
    p = build(hist)
    syms = set(p["symbol"])
    assert "LOWV" not in syms and "PENY" not in syms
    for s in SYMS:
        d = p.loc[p["symbol"] == s, "date"]
        assert d.min() == DATES[F.MIN_PRIOR_BARS]              # ≥ 60 prior bars
    y = p.loc[p["symbol"] == "YOUNG", "date"]
    assert len(y) == 70 - F.MIN_PRIOR_BARS
    assert (p["price"] >= config.MIN_PRICE).all()
    assert (p["dvol20"] >= config.MIN_DOLLAR_VOLUME_20D).all()
    pm = build(hist, min_date=str(DATES[120].date()))
    assert pm["date"].min() == DATES[120]
    # min_date drops rows but keeps look-back: features identical where both exist
    assert_features_equal(pm, p[p["date"] >= DATES[120]])


def test_static_columns():
    p = keyed(build(universe()))
    r = p.xs("AAA", level="symbol").iloc[0]
    assert r["asia"] == 1.0 and r["ipo_year"] == 2020 and r["ipo_age"] == 2024 - 2020
    b = p.xs("BBB", level="symbol").iloc[0]
    assert b["asia"] == 0.0 and np.isnan(b["ipo_age"])
    d = p.xs("DDD", level="symbol").iloc[0]
    assert np.isnan(d["ipo_age"])                               # IPO year after the row's year


def test_cross_sectional_only_uses_same_day():
    p = build(universe())
    day = p[p["date"] == DATES[100]]
    assert day["r1_rank"].max() == 1.0
    assert day.loc[day["r1"].idxmax(), "r1_rank"] == 1.0
    np.testing.assert_allclose(day["breadth_up"].iloc[0], (day["r1"] > 0).mean(), rtol=1e-6)
    np.testing.assert_allclose(day["univ_r1_med"].iloc[0], day["r1"].median(), rtol=1e-5)
    b = bench()
    np.testing.assert_allclose(day["iwm_r1"].iloc[0], b["close"].iloc[100] / b["close"].iloc[99] - 1, rtol=1e-5)


# ── live rows ────────────────────────────────────────────────────────────
def test_latest_rows_one_per_symbol_at_last_bar():
    hist = universe()
    hist["CCC"] = hist["CCC"].iloc[:-5]                         # stopped trading 5 sessions early
    hist["PENY"] = make_bars(21)
    hist["PENY"].iloc[-1, :4] = 0.05                            # last bar fails the price filter
    p = build(hist)
    lr = F.latest_rows(p)
    assert list(lr["symbol"]) == sorted(set(lr["symbol"]))
    assert set(lr["symbol"]) == set(SYMS)                       # PENY's stale row is not "latest"
    for _, r in lr.iterrows():
        assert r["date"] == hist[r["symbol"]].index[-1]
        assert r["is_last_bar"]
    assert np.isnan(lr[F.LABELS].to_numpy(float)).all()
    assert lr["gap_open"].isna().all()
    assert F.latest_rows(p.iloc[0:0]).empty


def test_live_row_equals_training_row():
    """Train/serve parity: the row built from history ending at t equals
    the training row for t built from the full history."""
    hist = universe()
    t = DATES[120]
    full = build(hist)
    live = F.latest_rows(build({s: df[df.index <= t] for s, df in hist.items()}))
    assert_features_equal(live, full[full["date"] == t])


# ── FINRA short volume (family "flow") ───────────────────────────────────
def finra_frame(hist: Dict[str, pd.DataFrame], seed: int = 3, share: float = 0.4,
                drop: Optional[set] = None) -> pd.DataFrame:
    """Synthetic FINRA history for every bar: total = share × volume, short ~ 30–70 %."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, df in hist.items():
        for d, v in df["volume"].items():
            if drop and (s, d) in drop:
                continue
            tot = share * float(v)
            rows.append((d, s, tot * rng.uniform(0.3, 0.7), tot))
    return pd.DataFrame(rows, columns=["date", "symbol", "short_volume", "total_volume"])


def build_sv(hist, sv, splits=None):
    return F.build_panel(hist, {}, splits or {}, STATIC, bench(), short_vol=sv)


def test_finra_features_exact_values():
    hist = universe()
    sv = finra_frame(hist)
    p = keyed(build_sv(hist, sv))
    g = sv[sv["symbol"] == "BBB"].set_index("date").sort_index()
    ratio = g["short_volume"] / g["total_volume"]
    for t_i in (80, 121):
        t = DATES[t_i]
        r = p.loc[(t, "BBB")]
        np.testing.assert_allclose(r["sv_ratio_1"], ratio.loc[t], rtol=1e-5)
        w5 = g.iloc[t_i - 4:t_i + 1]
        np.testing.assert_allclose(r["sv_ratio_5"], w5["short_volume"].sum() / w5["total_volume"].sum(), rtol=1e-5)
        w20 = g.iloc[t_i - 19:t_i + 1]
        np.testing.assert_allclose(r["sv_ratio_20"], w20["short_volume"].sum() / w20["total_volume"].sum(), rtol=1e-5)
        prior = ratio.iloc[t_i - 20:t_i]
        z = (ratio.loc[t] - prior.mean()) / max(prior.std(ddof=1), F.FINRA_Z_SD_FLOOR)
        np.testing.assert_allclose(r["sv_ratio_z"], z, rtol=1e-4)
        np.testing.assert_allclose(r["finra_share"], 0.4, rtol=1e-5)


def test_finra_future_rows_invisible_and_same_day_visible():
    hist = universe()
    t_i = 110
    t = DATES[t_i]
    sv = finra_frame(hist)
    p0 = build_sv(hist, sv)
    # rewrite, add and drop every FINRA row dated after t
    sv1 = sv.copy()
    after = pd.to_datetime(sv1["date"]) > t
    sv1.loc[after, "short_volume"] = sv1.loc[after, "total_volume"] * 0.99
    sv1 = sv1[~(after & (sv1["symbol"] == "CCC"))]
    extra = pd.DataFrame({"date": [DATES[t_i + 1]], "symbol": ["ZZZ"], "short_volume": [1.0], "total_volume": [2.0]})
    p1 = build_sv(hist, pd.concat([sv1, extra], ignore_index=True))
    cols = F.PANEL_FEATURES
    assert_features_equal(p0[p0["date"] <= t], p1[p1["date"] <= t], cols)
    fa, fb = keyed(p0[p0["date"] > t]), keyed(p1[p1["date"] > t])
    assert not np.allclose(fa["sv_ratio_1"].to_numpy(float), fb["sv_ratio_1"].to_numpy(float), equal_nan=True)
    # a FINRA row dated t is visible at t (published after t's close, used for t+1)
    sv2 = sv.copy()
    on_t = (pd.to_datetime(sv2["date"]) == t) & (sv2["symbol"] == "AAA")
    sv2.loc[on_t, "short_volume"] = sv2.loc[on_t, "total_volume"] * 0.95
    p2 = keyed(build_sv(hist, sv2))
    assert p2.loc[(t, "AAA"), "sv_ratio_1"] == pytest.approx(0.95, rel=1e-5)
    assert keyed(p0).loc[(DATES[t_i - 1], "AAA"), "sv_ratio_1"] == pytest.approx(
        p2.loc[(DATES[t_i - 1], "AAA"), "sv_ratio_1"], rel=1e-7)


def test_finra_missing_is_nan_not_zero():
    hist = universe()
    t = DATES[100]
    drop = {("AAA", d) for d in DATES[90:101]}
    p = keyed(build_sv(hist, finra_frame(hist, drop=drop)))
    r = p.loc[(t, "AAA")]
    assert np.isnan(r["sv_ratio_1"]) and np.isnan(r["sv_ratio_5"]) and np.isnan(r["finra_share"])
    assert np.isnan(r["sv_ratio_20"])                           # only 9 of the last 20 known (< 12)
    a = p.xs("AAA", level="symbol")
    assert np.isnan(a.loc[DATES[102], "sv_ratio_5"])            # 2 known of 5 (< 3)
    assert np.isfinite(a.loc[DATES[103], "sv_ratio_5"])         # 3 known
    assert np.isnan(a.loc[DATES[111], "sv_ratio_20"])           # 11 known
    assert np.isfinite(a.loc[DATES[112], "sv_ratio_20"])        # 12 known
    assert np.isnan(a.loc[DATES[101], "sv_ratio_z"])            # < 10 known in the prior 20
    assert np.isfinite(p.loc[(t, "BBB"), "sv_ratio_1"])


def test_finra_share_uses_as_traded_volume_after_future_split():
    """Yahoo scales pre-split volume by the split; FINRA reports as-traded
    shares. A 1-for-10 reverse split after t must not change finra_share at t."""
    hist = universe()
    s_i = 130
    sd = DATES[s_i]
    raw = hist["DDD"].copy()                          # as traded: price ×1 before, ×10 after
    raw.loc[raw.index >= sd, ["open", "high", "low", "close"]] *= 10.0
    raw.loc[raw.index >= sd, "volume"] /= 10.0
    sv = finra_frame({"DDD": raw})                    # FINRA shares = 0.4 × as-traded volume
    adj = raw.copy()                                  # Yahoo-adjusted history
    adj.loc[adj.index < sd, ["open", "high", "low", "close"]] *= 10.0
    adj.loc[adj.index < sd, "volume"] /= 10.0
    h = {**universe(), "DDD": adj}
    p = keyed(build_sv(h, sv, splits={"DDD": [{"date": str(sd.date()), "ratio": 0.1}]}))
    for t_i in (100, 129, 140):
        np.testing.assert_allclose(p.loc[(DATES[t_i], "DDD"), "finra_share"], 0.4, rtol=1e-5)


def test_finra_live_row_equals_training_row():
    hist = universe()
    sv = finra_frame(hist)
    t = DATES[120]
    full = build_sv(hist, sv)
    live = F.latest_rows(build_sv({s: df[df.index <= t] for s, df in hist.items()},
                                  sv[pd.to_datetime(sv["date"]) <= t]))
    assert_features_equal(live, full[full["date"] == t], F.PANEL_FEATURES)
