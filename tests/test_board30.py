"""The $30K board (round 11): the "$30K-ready" universe, the ranker's target,
the ranker in train → bundle → predict, and the live IBKR lendable gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gravity import capacity as CAP, cli, config  # noqa: E402
from gravity import model as M  # noqa: E402
from test_model import FAST, FOLD_MONTHS, synth_panel  # noqa: E402


def _row(close=5.0, volume=2_000_000, dvol20=8e6, vol20=0.04, low=None, r1=0.0):
    prev = close / (1 + r1)
    return {"close": close, "volume": volume, "dvol20": dvol20, "vol20": vol20,
            "low": prev * 0.97 if low is None else low, "r1": r1}


def test_liquid_mask_rules():
    df = pd.DataFrame([
        _row(),                                            # liquid, calm → ready
        _row(close=0.8, volume=12_000_000),                # under $1 → not ready
        _row(dvol20=200_000, volume=40_000),               # under the $300K publication floor
        _row(low=4.4),                                     # ≥10% intraday drop → Rule 201 next session
        _row(dvol20=400_000, volume=80_000, vol20=0.15),   # thin and wild: capacity ≪ $30K
        _row(vol20=np.nan),                                # no volatility → capacity unknown → not ready
    ])
    cap = M.market_capacity(df)
    assert cap[0] >= config.BOARD_MIN_USD and cap[4] < config.BOARD_MIN_USD and cap[5] == 0.0
    assert list(M.liquid_mask(df)) == [True, False, False, False, False, False]
    # the same capacity the cost model uses
    dvx = CAP.exp_dvol(8e6, 5.0 * 2_000_000)
    assert np.isclose(cap[0], float(CAP.market_capacity(dvx, 0.04)))
    assert list(M.liquid_mask(pd.DataFrame({"x": [1.0]}))) == [False]
    # the $1 floor reads the as-traded price, not the split-adjusted close (later reverse splits)
    rs = pd.DataFrame([_row(close=15.0, volume=700_000), _row(close=15.0, volume=700_000)])
    rs["price"] = [0.30, 15.0]
    assert list(M.liquid_mask(rs)) == [False, True]


def test_liq_target_is_within_session_percentile():
    day = np.array(["d1", "d1", "d1", "d1", "d2", "d2"])
    oc = np.array([-0.10, 0.00, 0.02, 0.50, 0.90, -0.01])
    liq = np.array([True, True, True, False, True, True])
    y = M._liq_target(day, oc, liq)
    assert np.isnan(y[3])                                   # not $30K-ready → no label
    np.testing.assert_allclose(y[:3], [1 / 3, 2 / 3, 1.0])  # ranked among d1's liquid rows only
    np.testing.assert_allclose(y[4:], [1.0, 0.5])


def _liquid_panel():
    p = synth_panel(n_sym=30, n_days=300, seed=3)
    rng = np.random.default_rng(5)
    sym_i = p["symbol"].str[1:].astype(int).to_numpy()
    liquid = sym_i % 2 == 0                                 # half the names can take $30K
    p["close"] = np.where(liquid, 8.0, 2.0)
    p["price"] = p["close"]
    p["dvol20"] = np.where(liquid, 2e7, 4e5)
    p["volume"] = p["dvol20"] / p["close"] * rng.uniform(0.8, 1.2, len(p))
    p["vol20"] = np.where(liquid, 0.03, 0.12)
    p["low"] = p["close"] * 0.98
    p["r1"] = p["r1"].clip(-0.05, 0.05)
    return p, liquid


def test_train_reports_liquid_board_and_predicts_edge(tmp_path):
    p, liquid = _liquid_panel()
    assert M.liquid_mask(p)[liquid].all() and not M.liquid_mask(p)[~liquid].any()
    rep = M.train(p, out_dir=tmp_path, site_json=None, fold_months=FOLD_MONTHS, params=FAST)
    liq = rep["liquid"]
    assert liq["size"] == config.BOARD_MIN_USD and "rule" in liq
    for m in ("m0", "m1"):
        bands = {(b["lo"], b["hi"]): b for b in liq[m]["bands"]}
        assert (1, 1) in bands and (1, 10) in bands
        for b in bands.values():
            assert 0.0 <= b["hit"] <= 1.0 and b["cost"] > 0 and np.isclose(b["net"], -b["mean_oc"] - b["cost"], atol=0.05)
        # the board only ever holds $30K-ready rows: 15 per session here
        assert bands[(1, 10)]["n_days"] * 10 == bands[(1, 10)]["n"]
    b = M.load(tmp_path)
    assert set(b["liq"]) == {"m0", "m1"} and b["liq"]["m0"] is not None
    # the morning ranker also reads overnight filing counts (R11 stage 2); absent columns are NaN
    assert b["liq_features"]["m1"] == b["m1_features"] + [c for c in M.LIQ_M1_EXTRA if c not in b["m1_features"]]
    assert b["liq_features"]["m0"] == b["features"]
    rows = p[p["date"] == p["date"].max()].set_index("symbol")
    pr = M.predict(b, rows, use_open=False)
    assert pr["liq_edge"].notna().all()
    # the synthetic truth: higher r1 → lower open→close → a better short
    g = rows.iloc[[0] * 2].copy()
    g["r1"] = [-0.05, 0.05]
    g.index = ["lo", "hi"]
    e = M.predict(b, g, use_open=False)["liq_edge"].to_numpy()
    assert e[1] > e[0]


def test_old_bundle_without_ranker_gives_nan_edge(tmp_path):
    p, _ = _liquid_panel()
    M.train(p, out_dir=tmp_path, site_json=None, fold_months=FOLD_MONTHS, params=FAST)
    b = M.load(tmp_path)
    b.pop("liq")
    pr = M.predict(b, p[p["date"] == p["date"].max()].set_index("symbol"), use_open=False)
    assert pr["liq_edge"].isna().all() and pr["prob_dump"].notna().all()


def test_ibkr_lendable_gate():
    borrow = {"AAA": {"available": 1_000}, "BBB": {"available": 100_000}, "CCC": {"available": None}}
    assert cli._lend_short("AAA", borrow, True, 5.0) == 5_000.0           # $5K lendable < $30K
    assert cli._lend_short("BBB", borrow, True, 5.0) is None              # $500K lendable: fine
    assert cli._lend_short("CCC", borrow, True, 5.0) is None              # unknown → not gated here
    assert cli._lend_short("ZZZ", borrow, True, 5.0) is None              # not listed → shortability gate handles it
    assert cli._lend_short("AAA", borrow, False, 5.0) is None             # no borrow file this run
    assert cli._lend_short("AAA", borrow, True, float("nan")) is None


def test_cost_table_has_board_size():
    blk = CAP.size_block(8.0, 2e7, 2e7, 0.03, np.full(30, 8.2), np.full(30, 7.8), np.full(30, 8.0), 1e6, 0.01)
    assert str(config.BOARD_MIN_USD) in blk["costs"] and str(config.BOARD_MIN_USD) in blk["participation"]


def test_overnight_counts_stop_at_the_morning_cutoff():
    from gravity import features as F
    ev = {"AAA": [
        {"form": "8-K", "items": ["8.01"], "accepted": "2026-10-12T16:30:00-04:00", "accession": "a1"},   # after the close
        {"form": "8-K", "items": ["8.01"], "accepted": "2026-10-13T07:59:00-04:00", "accession": "a2"},   # before 8:00 ET
        {"form": "8-K", "items": ["8.01"], "accepted": "2026-10-13T08:20:00-04:00", "accession": "a3"},   # after the cutoff
        {"form": "8-K", "items": ["8.01"], "accepted": "2026-10-12T15:59:00-04:00", "accession": "a4"},   # before the close
    ]}
    args = (np.array(["AAA", "BBB"], dtype=object), [pd.Timestamp("2026-10-12")] * 2, [pd.Timestamp("2026-10-13")] * 2, ev)
    eight = F.overnight_counts(*args, cutoff_hour=8)
    nine = F.overnight_counts(*args)
    assert eight["on_filings"][0] == 2 and nine["on_filings"][0] == 3
    assert np.isnan(eight["on_filings"][1])                      # no filing history at all → unknown, not zero
    assert config.MORNING_CUTOFF_ET == 8


# ── the ONE (round 13) ───────────────────────────────────────────────────
def test_one_score_rank_average_and_deficit():
    day = np.array(["d"] * 4 + ["e"] * 2)
    a = np.array([0.9, 0.5, 0.1, 0.8, 0.2, 0.3])
    b = np.array([0.7, 0.8, 0.1, 0.9, 0.6, 0.1])
    c = np.array([0.6, 0.5, 0.4, 0.1, np.nan, 0.2])
    liq = np.array([True, True, True, False, True, True])
    sc, dfc = M.one_score(day, [a, b, c], liq)
    assert np.isnan(sc[3]) and np.isnan(dfc[3])            # not $30K-ready
    assert np.isnan(sc[4])                                  # a part is missing
    # session d among liquid rows: a ranks 1,2,3; b ranks 2,1,3; c ranks 1,2,3
    np.testing.assert_allclose(dfc[:3], [1, 2, 6])
    assert sc[0] > sc[1] > sc[2]
    assert sc[5] == pytest.approx(1.0) and dfc[5] == 0      # alone in its session
    assert [M.one_tier(x) for x in (0, 2, 3, 7, 8, np.nan)] == ["high", "high", "medium", "medium", "low", None]


def test_one_mask_needs_the_bigger_size():
    df = pd.DataFrame([_row(), _row(dvol20=2.5e6, volume=500_000, vol20=0.04)])   # capacity ~$61K
    cap = M.market_capacity(df)
    assert (cap >= config.BOARD_MIN_USD).all() and cap[1] < config.ONE_MIN_USD <= cap[0]
    assert list(M.liquid_mask(df)) == [True, True] and list(M.one_mask(df)) == [True, False]


def test_train_reports_the_one_and_predicts_its_parts(tmp_path):
    p, liquid = _liquid_panel()
    rep = M.train(p, out_dir=tmp_path, site_json=None, fold_months=FOLD_MONTHS, params=FAST)
    one = rep["one"]
    assert one["size"] == config.ONE_MIN_USD and one["parts"] == list(M.ONE_PARTS)
    for m in ("m0", "m1"):
        o = one[m]["overall"]
        assert o["days"] > 0 and 0 <= o["hit"] <= 1 and str(config.ONE_MIN_USD) in {k[4:] for k in o if k.startswith("net_")}
        tiers = one[m]["tiers"]
        assert set(tiers) == {"high", "medium", "low"}
        assert sum(t["days"] for t in tiers.values()) == o["days"]
    b = M.load(tmp_path)
    assert set(b["one"]) == {"m0", "m1"} and all(b["one"][m]["dump"] is not None for m in ("m0", "m1"))
    rows = p[p["date"] == p["date"].max()].set_index("symbol")
    pr = M.predict(b, rows, use_open=False)
    assert pr["one_dump"].notna().all() and pr["one_d3"].notna().all()
    assert ((pr["one_dump"] >= 0) & (pr["one_dump"] <= 1)).all()


def test_one_lendable_gate_uses_its_own_size():
    borrow = {"AAA": {"available": 10_000}}                 # $50K lendable at $5
    assert cli._lend_short("AAA", borrow, True, 5.0) is None                          # fine for the $30K board
    assert cli._lend_short("AAA", borrow, True, 5.0, config.ONE_MIN_USD) == 50_000.0  # not for the $100K ONE
