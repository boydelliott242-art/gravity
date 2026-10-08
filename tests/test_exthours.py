"""Extended-hours features and overnight filing counts: exact time windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gravity import features as F
from gravity.sources import exthours as X


def _bars(rows):
    """rows: (ET timestamp string, o, h, l, c) → yfinance-like hourly frame (UTC index)."""
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, *_ in rows]).tz_convert("UTC")
    return pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[2] for r in rows], "Low": [r[3] for r in rows],
                         "Close": [r[4] for r in rows], "Volume": 0.0}, index=idx)


H = _bars([
    ("2026-10-05 15:30", 1.00, 1.02, 0.99, 1.00),   # previous session's last regular bar → close 1.00
    ("2026-10-05 16:00", 1.00, 1.05, 1.00, 1.04),   # after-hours
    ("2026-10-05 19:00", 1.04, 1.06, 1.03, 1.05),
    ("2026-10-06 04:00", 1.10, 1.30, 1.10, 1.25),   # pre-market spike
    ("2026-10-06 08:00", 1.20, 1.22, 1.15, 1.16),   # last trade before 9:00 → 1.16
    ("2026-10-06 09:00", 1.16, 1.40, 1.16, 1.39),   # 9:00–9:30 — NOT known at 9:00, must be ignored
    ("2026-10-06 09:30", 1.38, 1.40, 1.00, 1.01),   # regular session of D
    ("2026-10-06 15:30", 1.01, 1.02, 0.95, 0.96),
])


def test_ext_row_windows():
    f = X._frame(H)
    r = X.ext_row(f, pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-05"))
    assert r["pm_last"] == pytest.approx(1.16) and r["ext_last"] == pytest.approx(1.16)
    assert r["ah_last"] == pytest.approx(1.05)
    assert r["ext_high"] == pytest.approx(1.30) and r["ext_low"] == pytest.approx(1.00)
    assert r["n_ext"] == 4 and r["n_pm"] == 2
    assert r["close_h_prev"] == pytest.approx(1.00)


def test_reduce_and_training_alignment():
    store = X.reduce("AAA", H)
    assert list(store["date"]) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]
    rows = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": pd.to_datetime(["2026-10-05", "2026-10-06"]), "close": [1.00, 0.96]})
    f = X.training_features(rows, store)
    # the row dated 10-05 gets what the morning of 10-06 knew
    assert f.loc[0, "gap_ext"] == pytest.approx(0.16)
    assert f.loc[0, "ext_hi"] == pytest.approx(0.30) and f.loc[0, "ah_ret"] == pytest.approx(0.05)
    assert f.loc[0, "ext_fade"] == pytest.approx(1.16 / 1.30 - 1)
    assert np.isnan(f.loc[1, "gap_ext"])           # no next session in the data yet


def test_features_do_not_depend_on_the_daily_close_basis():
    # The daily close may be on another split basis (a later reverse split): features are ratios to the
    # hourly series' own prior close, so nothing is blanked — blanking would leak "this name splits later".
    store = X.reduce("AAA", H)
    a = X.training_features(pd.DataFrame({"symbol": ["AAA", "AAA"], "date": pd.to_datetime(["2026-10-05", "2026-10-06"]),
                                          "close": [1.0, 0.96]}), store)
    b = X.training_features(pd.DataFrame({"symbol": ["AAA", "AAA"], "date": pd.to_datetime(["2026-10-05", "2026-10-06"]),
                                          "close": [10.0, 9.6]}), store)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "gap_ext"] == pytest.approx(0.16)


def test_misaligned_previous_session_is_unusable():
    # the panel has a row on 10-05 but the hourly data's previous session for 10-06 is 10-02 (no hourly bars on 10-05)
    store = pd.DataFrame([{"symbol": "AAA", "date": pd.Timestamp("2026-10-06"), "prev": pd.Timestamp("2026-10-02"),
                           "ah_last": 1.0, "pm_last": 1.1, "ext_last": 1.1, "ext_high": 1.2, "ext_low": 1.0,
                           "n_ext": 3.0, "n_pm": 2.0, "close_h_prev": 1.0}])
    rows = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": pd.to_datetime(["2026-10-05", "2026-10-06"])})
    assert X.training_features(rows, store).isna().all(axis=None)


def test_derive_no_trades_is_nan_gap_but_zero_activity():
    d = X.derive([np.nan], [np.nan], [np.nan], [np.nan], [0.0], [0.0], [2.0], [True])
    assert np.isnan(d["gap_ext"][0]) and d["n_ext"][0] == 0.0


def test_overnight_counts_window_and_forms():
    ev = {"AAA": [
        {"form": "424B5", "accepted": "2026-10-05T16:30:00-04:00", "accession": "1"},   # after the close → counts
        {"form": "8-K", "accepted": "2026-10-05T15:59:00-04:00", "accession": "2", "items": ["1.01"]},  # before close
        {"form": "8-K", "accepted": "2026-10-06T08:59:00-04:00", "accession": "3", "items": ["3.01"]},  # counts
        {"form": "8-K", "accepted": "2026-10-06T09:01:00-04:00", "accession": "4"},     # after 9:00 → not
        {"form": "4", "accepted": "2026-10-05T18:00:00-04:00", "accession": "5"},       # not a scanned form
        {"form": "424B5", "accepted": "2026-10-05T16:30:00-04:00", "accession": "1"},   # duplicate
    ]}
    out = F.overnight_counts(["AAA", "BBB"], pd.to_datetime(["2026-10-05", "2026-10-05"]),
                             pd.to_datetime(["2026-10-06", "2026-10-06"]), ev)
    assert out["on_filings"][0] == 2 and out["on_offer"][0] == 1 and out["on_delist"][0] == 1
    assert out["on_finance"][0] == 0
    assert np.isnan(out["on_filings"][1])          # no event history → unknown


def test_overnight_counts_across_dst_change():
    # DST ends 2026-11-01: Friday 10-30 close (EDT) → Monday 11-02 9:00 (EST)
    ev = {"AAA": [{"form": "424B4", "accepted": "2026-11-02T08:30:00-05:00", "accession": "9"},
                  {"form": "424B4", "accepted": "2026-11-02T09:30:00-05:00", "accession": "10"}]}
    out = F.overnight_counts(["AAA"], pd.to_datetime(["2026-10-30"]), pd.to_datetime(["2026-11-02"]), ev)
    assert out["on_offer"][0] == 1


def test_update_store_merges_and_skips_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "STORE", tmp_path / "ext.pkl")
    monkeypatch.setattr(X, "MISSING", tmp_path / "missing.json")
    calls = []

    def fake_fetch(symbols, start, end, deadline_s=0):
        calls.append(sorted(symbols))
        return ({"AAA": H} if "AAA" in symbols else {}), set(symbols)

    monkeypatch.setattr(X, "fetch", fake_fetch)
    st = X.update_store(["AAA", "ZZZ"], today=pd.Timestamp("2026-10-06"))
    assert set(st["symbol"]) == {"AAA"} and len(st) == 2
    assert calls == [["AAA", "ZZZ"]]
    calls.clear()
    st = X.update_store(["AAA", "ZZZ"], today=pd.Timestamp("2026-10-07"))   # ZZZ skipped (no data last time)
    assert calls == [["AAA"]] and len(st) == 2                            # AAA re-fetched, rows replaced not duplicated


def test_early_close_days():
    from datetime import datetime as _dt
    from gravity.util import ET, market_phase, target_session
    assert market_phase(_dt(2026, 11, 27, 13, 30, tzinfo=ET)) == "after-hours"   # day after Thanksgiving: 13:00 close
    assert market_phase(_dt(2026, 11, 27, 12, 59, tzinfo=ET)) == "open"
    assert market_phase(_dt(2026, 11, 30, 13, 30, tzinfo=ET)) == "open"
    assert str(target_session(_dt(2026, 12, 24, 14, 0, tzinfo=ET))) == "2026-12-28"



def test_incremental_refresh_never_overwrites_a_complete_row(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "STORE", tmp_path / "ext.pkl")
    monkeypatch.setattr(X, "MISSING", tmp_path / "missing.json")
    full = X.reduce("AAA", H)                                  # 10-05 (no prev) and 10-06 (complete)
    full.to_pickle(tmp_path / "ext.pkl")
    tail = H[H.index >= pd.Timestamp("2026-10-06 08:00", tz="America/New_York")]   # a later fetch window starting on 10-06
    monkeypatch.setattr(X, "fetch", lambda symbols, start, end, deadline_s=0: ({"AAA": tail}, set(symbols)))
    st = X.update_store(["AAA"], today=pd.Timestamp("2026-10-07"))
    row = st[st["date"] == pd.Timestamp("2026-10-06")].iloc[0]
    assert row["close_h_prev"] == pytest.approx(1.00) and row["ah_last"] == pytest.approx(1.05)   # kept, not blanked


def test_failed_batch_is_not_marked_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "STORE", tmp_path / "ext.pkl")
    monkeypatch.setattr(X, "MISSING", tmp_path / "missing.json")
    monkeypatch.setattr(X, "fetch", lambda symbols, start, end, deadline_s=0: ({}, set()))   # rate-limited: nothing answered
    X.update_store(["ZZZ"], today=pd.Timestamp("2026-10-07"))
    import json as _json
    assert _json.loads((tmp_path / "missing.json").read_text()) == {}
