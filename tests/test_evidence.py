"""Tests for gravity.evidence (CONTRACTS.md §9).

Hand-built panels with known answers: counts, rates, lifts and the
date-clustered bootstrap are checked against numbers computed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gravity import config  # noqa: E402
from gravity import evidence as E  # noqa: E402

STUDY_KEYS = {"id", "title", "plain", "condition", "n", "n_symbols", "pct_red_oc", "pct_dump", "pct_bigdump",
              "pct_squeeze", "median_oc", "mean_oc", "median_c5", "ci_pct_dump", "lift_dump"}


def panel(n_days: int = 60, n_sym: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    df = pd.DataFrame({
        "date": np.repeat(dates.values, n_sym),
        "symbol": np.tile([f"S{i:02d}" for i in range(n_sym)], n_days),
    })
    n = len(df)
    df["r1"] = rng.normal(0, 0.03, n)
    df["r3"] = rng.normal(0, 0.05, n)
    df["y_oc"] = rng.normal(0.0, 0.03, n)
    df["y_oh"] = np.abs(rng.normal(0.02, 0.03, n))
    df["y_gap"] = rng.normal(0, 0.02, n)
    df["y_c5"] = rng.normal(0, 0.05, n)
    df["price"] = 3.0
    df["dd_52w"] = -0.3
    df["rsi14"] = 50.0
    df["asia"] = 0.0
    df["ipo_age"] = 5.0
    df["rs_count_2y"] = 0.0
    df["sess_since_rs"] = np.nan
    df["n_offer_30"] = 0.0
    df["sess_since_offer"] = np.nan
    df["n_reg_30"] = 0.0
    df["n_delist_90"] = 0.0
    return df


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    df["y_dump"] = np.where(df["y_oc"].notna(), (df["y_oc"] <= config.DUMP_THRESHOLD).astype(float), np.nan)
    df["y_bigdump"] = np.where(df["y_oc"].notna(), (df["y_oc"] <= config.BIG_DUMP_THRESHOLD).astype(float), np.nan)
    df["y_squeeze"] = np.where(df["y_oh"].notna(), (df["y_oh"] >= config.SQUEEZE_THRESHOLD).astype(float), np.nan)
    return df


def by_id(res):
    return {s["id"]: s for s in res["studies"]}


def test_schema_ids_and_strict_json():
    df = relabel(panel())
    res = E.run_studies(df, n_boot=200)
    for k in ("baseline", "studies", "generated_at", "n_rows"):
        assert k in res
    ids = [s["id"] for s in res["studies"]]
    for want in ("baseline", "up20", "up50", "up100", "up200", "run3_100", "gap30", "rs_5", "rs_30", "offer_1",
                 "reg_30", "delist_90", "sub1", "dd90", "rsi85", "asia_ipo2y", "inhd_profile"):
        assert want in ids
    for s in res["studies"] + [res["baseline"]]:
        assert STUDY_KEYS <= set(s), s["id"]
    # the integrator finds the fresh-offering lift by id
    hits = [s for s in res["studies"] if "offer" in s["id"] and "1" in s["id"]]
    assert len(hits) == 1 and hits[0]["id"] == "offer_1"
    json.dumps(res, allow_nan=False)            # raises on NaN/inf


def test_baseline_and_simple_study_exact():
    df = panel()
    df.loc[df.index[:7], "y_oc"] = np.nan                    # unlabeled rows are ignored
    pick = df.index[100:140]                                 # 40 "up 50%" rows
    df.loc[pick, "r1"] = 0.6
    df.loc[pick[:30], "y_oc"] = -0.08                        # 30 of them dump
    df.loc[pick[30:], "y_oc"] = 0.02
    df.loc[pick[:5], "y_oc"] = -0.2                          # 5 big dumps
    relabel(df)
    res = E.run_studies(df, n_boot=300)
    lab = df[df["y_oc"].notna()]
    b = res["baseline"]
    assert b["n"] == len(lab) == res["n_rows"]
    assert b["pct_dump"] == pytest.approx(lab["y_dump"].mean(), abs=1e-6)
    assert b["pct_red_oc"] == pytest.approx((lab["y_oc"] < 0).mean(), abs=1e-6)
    assert b["median_oc"] == pytest.approx(lab["y_oc"].median(), abs=1e-6)
    assert b["n_symbols"] == 20
    s = by_id(res)["up50"]
    assert s["n"] == 40
    assert s["pct_dump"] == pytest.approx(0.75)
    assert s["pct_bigdump"] == pytest.approx(5 / 40)
    assert s["pct_red_oc"] == pytest.approx(0.75)
    assert s["lift_dump"] == pytest.approx(0.75 / b["pct_dump"], rel=1e-4)
    assert s["n_symbols"] == len(set(df.loc[pick, "symbol"]))
    lo, hi = s["ci_pct_dump"]
    assert 0 <= lo <= 0.75 <= hi <= 1
    assert by_id(res)["up20"]["n"] >= 40
    # the sentence is generated from these numbers
    assert "40 cases" in s["plain"] and "75.0%" in s["plain"]


def test_gap_study_uses_the_open_gap_and_same_session_outcome():
    df = panel()
    pick = df.index[50:70]
    df.loc[pick, "y_gap"] = 0.45
    df.loc[pick, "y_oc"] = -0.12
    relabel(df)
    s = by_id(E.run_studies(df, n_boot=100))["gap30"]
    assert s["n"] == 20 and s["pct_dump"] == 1.0
    assert s["same_session_condition"] is True
    assert "same session" in s["plain"]


def test_filing_study_compares_against_rows_with_filing_data():
    df = panel()
    no_sec = df["symbol"].isin(["S00", "S01", "S02", "S03", "S04"])
    df.loc[no_sec, ["n_offer_30", "n_reg_30", "n_delist_90"]] = np.nan
    df.loc[no_sec, "y_oc"] = -0.2                               # names without SEC data all dump
    fresh = df.index[(df["symbol"] == "S10")][:10]
    df.loc[fresh, "sess_since_offer"] = [0, 1, 0, 1, 2, 3, 0, 1, 5, 9]
    df.loc[fresh, "n_offer_30"] = 1
    df.loc[fresh[[0, 1, 2]], "y_oc"] = -0.1
    relabel(df)
    res = E.run_studies(df, n_boot=100)
    s = by_id(res)["offer_1"]
    assert s["n"] == 6                                          # sess_since_offer ≤ 1
    assert s["pct_dump"] == pytest.approx(3 / 6)
    ref = df.loc[~no_sec & df["y_oc"].notna(), "y_dump"].mean()
    assert s["ref_pct_dump"] == pytest.approx(ref, abs=1e-6)
    assert s["lift_dump"] == pytest.approx(0.5 / ref, rel=1e-4)
    assert s["ref_pct_dump"] < res["baseline"]["pct_dump"]     # the no-SEC names are excluded from the reference


def test_empty_study_is_null_not_zero():
    df = relabel(panel())
    s = by_id(E.run_studies(df, n_boot=100))["up200"]
    assert s["n"] == 0
    for k in ("pct_dump", "pct_red_oc", "median_oc", "ci_pct_dump", "lift_dump"):
        assert s[k] is None
    assert "no qualifying" in s["plain"]


def test_bootstrap_resamples_dates_not_rows():
    """40 dates × 50 names, and every name on a date shares the outcome.
    Row-level resampling would give a CI of ±~2%; by date it is ±~15%."""
    df = panel(n_days=40, n_sym=50, seed=1)
    days = np.sort(df["date"].unique())
    dump_days = set(days[::2])                                   # exactly half the days dump
    df["y_oc"] = np.where(df["date"].isin(dump_days), -0.1, 0.01)
    relabel(df)
    b = E.run_studies(df, n_boot=1000)["baseline"]
    assert b["pct_dump"] == pytest.approx(0.5)
    lo, hi = b["ci_pct_dump"]
    assert hi - lo > 0.2
    assert lo < 0.5 < hi


def test_single_date_condition_has_no_ci():
    df = panel()
    first = df["date"] == df["date"].min()
    df.loc[first, "rsi14"] = 90.0
    relabel(df)
    s = by_id(E.run_studies(df, n_boot=100))["rsi85"]
    assert s["n"] == 20 and s["ci_pct_dump"] is None


def test_profiles():
    df = panel()
    inhd = df["symbol"] == "S07"
    df.loc[inhd, ["asia", "rs_count_2y", "dd_52w", "ipo_age", "price"]] = [1.0, 3.0, -0.95, 1.0, 0.5]
    relabel(df)
    st = by_id(E.run_studies(df, n_boot=100))
    assert st["inhd_profile"]["n"] == int(inhd.sum())
    assert st["asia_ipo2y"]["n"] == int(inhd.sum())
    assert st["sub1"]["n"] == int(inhd.sum())
    assert st["dd90"]["n"] == int(inhd.sum())
    assert st["inhd_profile"]["n_symbols"] == 1


def test_deterministic_and_empty_panel():
    df = relabel(panel())
    a = E.run_studies(df, n_boot=200)
    b = E.run_studies(df, n_boot=200)
    assert [s["ci_pct_dump"] for s in a["studies"]] == [s["ci_pct_dump"] for s in b["studies"]]
    empty = E.run_studies(df.iloc[0:0], n_boot=50)
    assert empty["n_rows"] == 0 and empty["baseline"]["n"] == 0
    assert all(s["n"] == 0 for s in empty["studies"])
