"""Tests for gravity.sources.prices (CONTRACTS.md §2) and scripts/warm_prices.py.

Unit tests replace every network touch point (``_yf_download``, ``_yf_one``,
``_yf_splits``, ``_nasdaq_history_raw``, ``_nasdaq_quote``,
``_yahoo_premarket``, ``net.get_json``) with deterministic fakes and pin the
clock via ``prices._now``. Live tests run only with ``GRAVITY_LIVE=1`` and
touch a handful of symbols, into a temporary cache.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Make the repo importable when pytest is run without a root conftest/ini.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gravity import config, net  # noqa: E402
from gravity.sources import prices as P  # noqa: E402
from gravity.util import ET, is_trading_day  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="set GRAVITY_LIVE=1 for live network tests")

TUE_EVENING = datetime(2026, 9, 29, 18, 0, tzinfo=ET)      # after Tuesday's close
WED_EVENING = datetime(2026, 9, 30, 18, 0, tzinfo=ET)
WED_PREMARKET = datetime(2026, 9, 30, 8, 15, tzinfo=ET)
WED_MIDDAY = datetime(2026, 9, 30, 12, 0, tzinfo=ET)


# ── fixtures & fakes ─────────────────────────────────────────────────────
class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _trading_days(start: date, end: date) -> List[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _price(d: date) -> float:
    """Deterministic 'true' split-adjusted close for a date."""
    return round(10.0 + 0.01 * (d.toordinal() % 500), 4)


def make_bars(start: date, end: date, drift: float = 1.0, raw_split: Optional[Dict[str, Any]] = None
              ) -> pd.DataFrame:
    """Daily bars on trading days in [start, end]. ``raw_split`` makes the
    bars *before* that split unadjusted (prices × 1/ratio, volume × ratio)."""
    days = _trading_days(start, end)
    rows = []
    for d in days:
        c = _price(d) * drift
        v = 100_000.0
        if raw_split and d < date.fromisoformat(raw_split["date"]):
            c = c * raw_split["ratio"]
            v = v / raw_split["ratio"]
        rows.append({"date": pd.Timestamp(d), "open": c * 0.99, "high": c * 1.02,
                     "low": c * 0.97, "close": c, "volume": v})
    return pd.DataFrame(rows).set_index("date")


class FakeYahoo:
    """Stands in for the yfinance layer (batch, single, splits)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.missing: set = set()           # symbols Yahoo has no data for
        self.splits: Dict[str, List[Dict[str, Any]]] = {}
        self.drift: Dict[str, float] = {}
        self.rate_limited_calls = 0         # next N batch calls answer "rate limited"
        self.batch_calls: List[tuple] = []
        self.single_calls: List[tuple] = []
        self.split_calls: List[str] = []
        self.splits_unknown: set = set()

    def _bars(self, sym: str, start: Optional[date]):
        now = self.clock()
        start = start or date(2015, 1, 1)
        df = P.clean_bars(make_bars(start, now.date(), self.drift.get(sym, 1.0)), now)
        sp = [s for s in self.splits.get(sym, []) if date.fromisoformat(s["date"]) >= start]
        return df, sp

    def download(self, symbols, start, end, max_workers, now):
        self.batch_calls.append((tuple(symbols), start))
        if self.rate_limited_calls > 0:
            self.rate_limited_calls -= 1
            msg = "YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
            return {}, {s: msg for s in symbols}
        frames, errors = {}, {}
        for s in symbols:
            if s in self.missing:
                errors[s] = "possibly delisted; no price data found"
            else:
                frames[s] = self._bars(s, start)
        return frames, errors

    def one(self, symbol, start, now):
        self.single_calls.append((symbol, start))
        if symbol in self.missing:
            return None, False
        return self._bars(symbol, start), False

    def splits_fn(self, symbol):
        self.split_calls.append(symbol)
        if symbol in self.splits_unknown:
            return None, False
        return list(self.splits.get(symbol, [])), False


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp cache, pinned clock, fake Yahoo, fake Nasdaq, no sleeping."""
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    clock = Clock(TUE_EVENING)
    fy = FakeYahoo(clock)
    sleeps: List[float] = []
    nasdaq: Dict[str, pd.DataFrame] = {}
    nasdaq_calls: List[str] = []

    def fake_nasdaq(symbol, start, end, now):
        nasdaq_calls.append(symbol)
        df = nasdaq.get(symbol)
        return P.empty_frame() if df is None else P.clean_bars(df, now)

    monkeypatch.setattr(P, "_now", clock)
    monkeypatch.setattr(P, "_yf_download", fy.download)
    monkeypatch.setattr(P, "_yf_one", fy.one)
    monkeypatch.setattr(P, "_yf_splits", fy.splits_fn)
    monkeypatch.setattr(P, "_nasdaq_history_raw", fake_nasdaq)
    monkeypatch.setattr(P, "_sleep", lambda s: sleeps.append(s))
    net.STATUS.pop(P.YAHOO_STATUS, None)
    net.STATUS.pop(P.NASDAQ_STATUS, None)

    class Env:
        pass

    e = Env()
    e.clock, e.yahoo, e.sleeps, e.nasdaq, e.nasdaq_calls, e.tmp = clock, fy, sleeps, nasdaq, nasdaq_calls, tmp_path
    return e


def assert_contract_frame(df: pd.DataFrame) -> None:
    assert list(df.columns) == P.PRICE_COLS
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df.index == df.index.normalize()).all()
    assert all(df[c].dtype == np.float64 for c in P.PRICE_COLS)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["volume"] >= 0).all()


# ── pure helpers ─────────────────────────────────────────────────────────
def test_period_start():
    today = date(2026, 9, 29)
    assert P.period_start("3y", today) == date(2023, 9, 29)
    assert P.period_start("6mo", today) == date(2026, 3, 29)
    assert P.period_start("2wk", today) == date(2026, 9, 15)
    assert P.period_start("ytd", today) == date(2026, 1, 1)
    assert P.period_start("max", today) is None
    assert P.period_start("5d", today) == date(2026, 9, 23)     # 5 sessions incl. today
    with pytest.raises(ValueError):
        P.period_start("3 years", today)


def test_nasdaq_symbol():
    assert P.nasdaq_symbol("BRK-A") == "BRK.A"
    assert P.nasdaq_symbol("inhd") == "INHD"


def test_clean_bars_normalises_everything():
    idx = pd.DatetimeIndex(["2026-09-25 00:00", "2026-09-24 00:00", "2026-09-24 00:00",
                            "2026-09-23 00:00", "2026-09-22 00:00", "2026-09-28 00:00"],
                           tz="America/New_York")
    raw = pd.DataFrame({
        "Open": [1.0, 2.0, 2.5, np.nan, 0.0, 3.0],
        "High": [1.1, 2.1, 2.6, 1.0, 1.0, 3.1],
        "Low": [0.9, 1.9, 2.4, 1.0, 1.0, 2.9],
        "Close": [1.0, 2.0, 2.5, 1.0, 1.0, 3.0],
        "Volume": [100, 200, 250, 10, 10, np.nan],
        "Dividends": 0.0,
    }, index=idx)
    df = P.clean_bars(raw, TUE_EVENING)
    assert_contract_frame(df)
    assert [d.strftime("%Y-%m-%d") for d in df.index] == ["2026-09-24", "2026-09-25", "2026-09-28"]
    assert df.loc["2026-09-24", "close"] == 2.5          # later duplicate wins
    assert df.loc["2026-09-28", "volume"] == 0.0         # NaN volume = halt → 0, row kept


def test_clean_bars_drops_the_in_progress_session():
    raw = make_bars(date(2026, 9, 28), date(2026, 9, 30))
    during = P.clean_bars(raw, WED_MIDDAY)
    after = P.clean_bars(raw, WED_EVENING)
    assert during.index.max() == pd.Timestamp("2026-09-29")
    assert after.index.max() == pd.Timestamp("2026-09-30")


def test_dollar_volume_median():
    df = pd.DataFrame({"close": [2.0] * 5, "volume": [0.0, 100.0, 200.0, 300.0, 400.0]},
                      index=pd.date_range("2026-09-01", periods=5))
    assert P.dollar_volume_median(df, n=3) == pytest.approx(600.0)     # 2×[200,300,400]
    assert P.dollar_volume_median(df, n=20) == pytest.approx(400.0)    # includes the halt ($0)
    assert math.isnan(P.dollar_volume_median(P.empty_frame()))
    assert math.isnan(P.dollar_volume_median(None))


def test_splits_from_series_and_merge():
    s = pd.Series([0.0, 0.05, 0.0, 2.0, 1.0],
                  index=pd.DatetimeIndex(["2026-05-01", "2026-05-04", "2026-05-05", "2026-06-01", "2026-06-02"],
                                         tz="America/New_York"))
    sp = P._splits_from_series(s)
    assert sp == [{"date": "2026-05-04", "ratio": 0.05}, {"date": "2026-06-01", "ratio": 2.0}]
    merged = P._merge_splits([{"date": "2026-06-01", "ratio": 2.0}], [{"date": "2025-01-02", "ratio": 0.1}])
    assert [m["date"] for m in merged] == ["2025-01-02", "2026-06-01"]


def test_adjust_for_splits_raw_reverse_split():
    split = {"date": "2026-05-04", "ratio": 0.05}               # 1:20 reverse
    raw = make_bars(date(2026, 4, 20), date(2026, 5, 15), raw_split=split)
    true = make_bars(date(2026, 4, 20), date(2026, 5, 15))
    adj, applied = P.adjust_for_splits(raw, [split])
    assert applied == ["2026-05-04"]
    pd.testing.assert_frame_equal(adj, true, check_exact=False, rtol=1e-9)


def test_adjust_for_splits_leaves_adjusted_data_alone():
    split = {"date": "2026-05-04", "ratio": 0.05}
    already = make_bars(date(2026, 4, 20), date(2026, 5, 15))
    adj, applied = P.adjust_for_splits(already, [split])
    assert applied == []
    pd.testing.assert_frame_equal(adj, already)


def test_adjust_for_splits_forward_split_and_out_of_window():
    split = {"date": "2026-05-04", "ratio": 2.0}                # 2:1 forward
    raw = make_bars(date(2026, 4, 20), date(2026, 5, 15), raw_split=split)
    adj, applied = P.adjust_for_splits(raw, [split, {"date": "2020-01-02", "ratio": 0.1}])
    assert applied == ["2026-05-04"]
    pd.testing.assert_frame_equal(adj, make_bars(date(2026, 4, 20), date(2026, 5, 15)),
                                  check_exact=False, rtol=1e-9)


def test_cache_freshness_rules():
    df = make_bars(date(2026, 9, 1), date(2026, 9, 29))
    e = {"df": df, "fetched_at": "2026-09-29T20:45:00+00:00"}        # 16:45 ET Tue
    assert P._is_current(e, TUE_EVENING)
    assert P._is_current(e, WED_PREMARKET)                          # last close still Tue
    assert not P._is_current(e, WED_EVENING)
    e_early = {"df": df, "fetched_at": "2026-09-29T20:10:00+00:00"}  # 16:10 ET: too soon
    assert not P._is_current(e_early, TUE_EVENING)
    # Missing Tuesday's bar: one late re-check, then accepted.
    e_nobar = {"df": df.iloc[:-1], "fetched_at": "2026-09-29T20:45:00+00:00"}
    assert not P._is_current(e_nobar, WED_PREMARKET)
    e_nobar_late = {"df": df.iloc[:-1], "fetched_at": "2026-09-30T00:30:00+00:00"}  # 20:30 ET
    assert P._is_current(e_nobar_late, WED_PREMARKET)
    # Weekend: last close is Friday.
    sat = datetime(2026, 10, 3, 10, 0, tzinfo=ET)
    assert P._last_close_dt(sat) == datetime(2026, 10, 2, 16, 0, tzinfo=ET)


# ── load_history (fakes) ─────────────────────────────────────────────────
def test_cold_load_fetches_full_period_and_caches(env):
    out = P.load_history(["inhd", "AAPL", "INHD", "BRK/B"], period="1y")
    assert set(out) == {"INHD", "AAPL", "BRK-B"}
    assert len(env.yahoo.batch_calls) == 1
    syms, start = env.yahoo.batch_calls[0]
    assert set(syms) == {"INHD", "AAPL", "BRK-B"} and start == date(2025, 9, 29)
    for s, df in out.items():
        assert_contract_frame(df)
        assert df.index.min() >= pd.Timestamp("2025-09-29")
        assert df.index.max() == pd.Timestamp("2026-09-29")
        assert df.attrs["source"] == "yahoo" and df.attrs["unadjusted"] is False
        assert df.attrs["stale"] is False and df.attrs["symbol"] == s
        assert (config.CACHE / "prices" / f"{s}.pkl").exists()
    assert P.LAST_RUN["full"] == 3 and P.LAST_RUN["failed"] == 0
    assert net.STATUS[P.YAHOO_STATUS]["ok"] is True


def test_current_cache_makes_no_requests(env):
    P.load_history(["INHD", "AAPL"], period="1y")
    env.yahoo.batch_calls.clear()
    env.clock.now = WED_PREMARKET
    out = P.load_history(["INHD", "AAPL"], period="1y")
    assert env.yahoo.batch_calls == []
    assert P.LAST_RUN["current"] == 2
    assert out["INHD"].index.max() == pd.Timestamp("2026-09-29")


def test_incremental_refresh_appends_recent_bars_only(env):
    P.load_history(["INHD", "AAPL"], period="3y")
    before = P.read_cache("INHD")["df"]
    env.yahoo.batch_calls.clear()
    env.clock.now = WED_EVENING
    out = P.load_history(["INHD", "AAPL"], period="3y")
    assert len(env.yahoo.batch_calls) == 1
    syms, start = env.yahoo.batch_calls[0]
    assert set(syms) == {"INHD", "AAPL"}
    assert start >= date(2026, 9, 14)                   # ~10 sessions, not 3 years
    assert P.LAST_RUN["incremental"] == 2 and P.LAST_RUN["full"] == 0
    df = out["INHD"]
    assert_contract_frame(df)
    assert df.index.max() == pd.Timestamp("2026-09-30")
    assert len(P.read_cache("INHD")["df"]) == len(before) + 1


def test_new_split_in_window_triggers_full_refetch(env):
    P.load_history(["INHD", "AAPL"], period="3y")
    env.yahoo.batch_calls.clear()
    env.clock.now = WED_EVENING
    env.yahoo.splits["INHD"] = [{"date": "2026-09-30", "ratio": 0.05}]
    out = P.load_history(["INHD", "AAPL"], period="3y")
    assert P.LAST_RUN["split_refetch"] == 1 and P.LAST_RUN["full"] == 1 and P.LAST_RUN["incremental"] == 1
    full_call = env.yahoo.batch_calls[-1]
    assert full_call == (("INHD",), date(2023, 9, 30))
    assert P.read_cache("INHD")["splits"] == [{"date": "2026-09-30", "ratio": 0.05}]
    assert out["INHD"].index.max() == pd.Timestamp("2026-09-30")
    assert P.load_splits(["INHD"])["INHD"] == [{"date": "2026-09-30", "ratio": 0.05}]


def test_overlap_drift_triggers_full_refetch(env):
    P.load_history(["INHD"], period="1y")
    env.clock.now = WED_EVENING
    env.yahoo.drift["INHD"] = 20.0                     # Yahoo re-adjusted history
    out = P.load_history(["INHD"], period="1y")
    assert P.LAST_RUN["drift_refetch"] == 1 and P.LAST_RUN["full"] == 1
    assert out["INHD"]["close"].iloc[0] == pytest.approx(_price(out["INHD"].index[0].date()) * 20.0)


def test_longer_period_than_cached_refetches(env):
    P.load_history(["INHD"], period="1y")
    env.yahoo.batch_calls.clear()
    out = P.load_history(["INHD"], period="3y")
    assert env.yahoo.batch_calls == [(("INHD",), date(2023, 9, 29))]
    assert out["INHD"].index.min() < pd.Timestamp("2024-01-01")
    env.yahoo.batch_calls.clear()
    short = P.load_history(["INHD"], period="6mo")    # shorter: served & trimmed from cache
    assert env.yahoo.batch_calls == []
    assert short["INHD"].index.min() >= pd.Timestamp("2026-03-29")


def test_refresh_false_serves_cache_without_network(env):
    P.load_history(["INHD"], period="1y")
    env.yahoo.batch_calls.clear()
    env.clock.now = datetime(2026, 10, 7, 18, 0, tzinfo=ET)
    out = P.load_history(["INHD", "AAPL"], period="1y", refresh=False)
    assert env.yahoo.batch_calls == [(("AAPL",), date(2025, 10, 7))]   # only the uncached one
    assert out["INHD"].attrs["stale"] is True
    assert out["INHD"].index.max() == pd.Timestamp("2026-09-29")


def test_batch_failure_retried_individually(env, monkeypatch):
    real = env.yahoo.download

    def flaky(symbols, start, end, max_workers, now):
        frames, errors = real(symbols, start, end, max_workers, now)
        if "SOUN" in frames:                          # batch drops SOUN, single call works
            del frames["SOUN"]
            errors["SOUN"] = "no data"
        return frames, errors

    monkeypatch.setattr(P, "_yf_download", flaky)
    out = P.load_history(["INHD", "SOUN"], period="1y")
    assert set(out) == {"INHD", "SOUN"}
    assert env.yahoo.single_calls == [("SOUN", date(2025, 9, 29))]
    assert P.LAST_RUN["individual"] == 1
    assert out["SOUN"].attrs["source"] == "yahoo"


def test_nasdaq_fallback_adjusts_raw_bars_with_known_splits(env):
    split = {"date": "2026-05-04", "ratio": 0.05}
    env.yahoo.missing.add("INHD")
    env.yahoo.splits["INHD"] = [split]                 # Ticker.splits still answers
    env.nasdaq["INHD"] = make_bars(date(2025, 9, 29), date(2026, 9, 29), raw_split=split)
    out = P.load_history(["INHD"], period="1y")
    df = out["INHD"]
    assert df.attrs["source"] == "nasdaq" and df.attrs["unadjusted"] is False
    pd.testing.assert_series_equal(df["close"], make_bars(date(2025, 9, 29), date(2026, 9, 29))["close"],
                                   check_exact=False, rtol=1e-9)
    assert P.LAST_RUN["nasdaq"] == 1
    assert P.read_cache("INHD")["source"] == "nasdaq"


def test_nasdaq_fallback_with_unknown_splits_is_flagged_and_refetched_later(env):
    env.yahoo.missing.add("INHD")
    env.yahoo.splits_unknown.add("INHD")
    env.nasdaq["INHD"] = make_bars(date(2025, 9, 29), date(2026, 9, 29))
    out = P.load_history(["INHD"], period="1y")
    assert out["INHD"].attrs["unadjusted"] is True
    assert "INHD" not in P.load_splits(["INHD"])      # unknown ≠ "no splits"
    # Next session Yahoo is back: the Nasdaq-sourced cache is replaced by a full Yahoo fetch.
    env.yahoo.missing.clear()
    env.yahoo.batch_calls.clear()
    env.clock.now = WED_EVENING
    out = P.load_history(["INHD"], period="1y")
    assert env.yahoo.batch_calls == [(("INHD",), date(2025, 9, 30))]
    assert out["INHD"].attrs["source"] == "yahoo" and out["INHD"].attrs["unadjusted"] is False


def test_total_failure_serves_stale_cache_or_omits(env):
    P.load_history(["INHD"], period="1y")
    env.clock.now = WED_EVENING
    env.yahoo.missing.update({"INHD", "GONE"})
    out = P.load_history(["INHD", "GONE"], period="1y")
    assert set(out) == {"INHD"}
    assert out["INHD"].attrs["stale"] is True
    assert out["INHD"].index.max() == pd.Timestamp("2026-09-29")
    assert P.LAST_RUN["stale"] == 1 and P.LAST_RUN["failed"] == 1
    assert "GONE" in P.LAST_RUN["failed_symbols"]
    assert net.STATUS[P.YAHOO_STATUS]["ok"] is False


def test_rate_limit_backs_off_then_recovers(env):
    env.yahoo.rate_limited_calls = 1
    out = P.load_history(["INHD", "AAPL"], period="1y")
    assert set(out) == {"INHD", "AAPL"}
    assert env.sleeps[:1] == [P.RATE_LIMIT_BACKOFF_S[0]]
    assert P.LAST_RUN["yahoo_rate_limited"] is False


def test_persistent_rate_limit_stops_calling_yahoo(env):
    P.load_history(["INHD"], period="1y")               # INHD has a cache
    env.clock.now = WED_EVENING
    env.yahoo.batch_calls.clear()
    env.yahoo.rate_limited_calls = 99
    env.nasdaq["NEWCO"] = make_bars(date(2026, 1, 5), date(2026, 9, 30))
    out = P.load_history(["INHD", "NEWCO"], period="1y")
    assert P.LAST_RUN["yahoo_rate_limited"] is True
    assert [s for s in env.sleeps if s >= 60] == list(P.RATE_LIMIT_BACKOFF_S)
    assert len(env.yahoo.batch_calls) == 1 + len(P.RATE_LIMIT_BACKOFF_S)   # then it stops
    assert env.yahoo.single_calls == []
    assert out["INHD"].attrs["stale"] is True            # cached → stale, not converted to Nasdaq
    assert env.nasdaq_calls == ["NEWCO"]                 # uncached → Nasdaq fallback
    assert out["NEWCO"].attrs["source"] == "nasdaq"
    st = net.STATUS[P.YAHOO_STATUS]
    assert st["ok"] is False and "rate-limited" in st["detail"]


def test_corrupt_cache_is_a_miss(env):
    (config.CACHE / "prices").mkdir(parents=True, exist_ok=True)
    (config.CACHE / "prices" / "INHD.pkl").write_bytes(b"not a pickle")
    assert P.read_cache("INHD") is None
    out = P.load_history(["INHD"], period="1y")
    assert "INHD" in out


def test_load_splits_from_cache_and_yahoo(env):
    env.yahoo.splits["INHD"] = [{"date": "2026-05-04", "ratio": 0.05}]
    env.yahoo.splits["AAPL"] = []
    P.load_history(["INHD"], period="1y")
    env.yahoo.split_calls.clear()
    got = P.load_splits(["INHD", "AAPL", "MYST"])
    assert got["INHD"] == [{"date": "2026-05-04", "ratio": 0.05}]   # from the price cache
    assert got["AAPL"] == []                                         # known: no splits
    assert "INHD" not in env.yahoo.split_calls
    assert env.yahoo.split_calls == ["AAPL", "MYST"]
    env.yahoo.split_calls.clear()
    P.load_splits(["AAPL"])                                          # 20 h JSON cache
    assert env.yahoo.split_calls == []


def test_benchmark_history_does_not_clobber_run_stats(env):
    P.load_history(["INHD"], period="1y")
    stats = dict(P.LAST_RUN)
    status = dict(net.STATUS[P.YAHOO_STATUS])
    bench = P.benchmark_history(period="1y")
    assert_contract_frame(bench)
    assert bench.attrs["symbol"] == "IWM"
    assert P.LAST_RUN == stats and net.STATUS[P.YAHOO_STATUS] == status


def test_real_yf_download_wrapper_parses_multiindex(monkeypatch):
    """``_yf_download`` against a fake ``yfinance.download`` shaped like the real one."""
    import yfinance as yf
    from yfinance import shared

    idx = pd.DatetimeIndex(["2026-09-24", "2026-09-25", "2026-09-28"], name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], fields], names=["Ticker", "Price"])
    data = np.full((3, len(cols)), np.nan)
    frame = pd.DataFrame(data, index=idx, columns=cols)
    frame[("AAA", "Open")] = [1.0, 1.1, 1.2]
    frame[("AAA", "High")] = [1.2, 1.3, 1.4]
    frame[("AAA", "Low")] = [0.9, 1.0, 1.1]
    frame[("AAA", "Close")] = [1.1, 1.2, 1.3]
    frame[("AAA", "Adj Close")] = [1.1, 1.2, 1.3]
    frame[("AAA", "Volume")] = [1000.0, 0.0, 500.0]
    frame[("AAA", "Dividends")] = 0.0
    frame[("AAA", "Stock Splits")] = [0.0, 0.05, 0.0]
    seen = {}

    def fake_download(tickers, **kw):
        seen["tickers"], seen["kw"] = tickers, kw
        shared._ERRORS = {"BBB": "possibly delisted; no price data found"}
        return frame

    monkeypatch.setattr(yf, "download", fake_download)
    frames, errors = P._yf_download(["AAA", "BBB"], date(2026, 9, 20), None, 3, TUE_EVENING)
    assert set(frames) == {"AAA"} and set(errors) == {"BBB"}
    bars, splits = frames["AAA"]
    assert_contract_frame(bars)
    assert bars["close"].tolist() == [1.1, 1.2, 1.3]          # unadjusted-for-dividends Close
    assert splits == [{"date": "2026-09-25", "ratio": 0.05}]
    kw = seen["kw"]
    assert kw["auto_adjust"] is False and kw["actions"] is True and kw["group_by"] == "ticker"
    assert kw["threads"] == 3 and kw["start"] == "2026-09-20" and kw["end"] == "2026-09-30"


def test_nasdaq_history_raw_parses_rows(monkeypatch):
    payload = {"data": {"symbol": "INHD", "totalRecords": 3, "tradesTable": {"rows": [
        {"date": "07/31/2026", "close": "$18.00", "volume": "3,305,994", "open": "$38.30",
         "high": "$38.46", "low": "$15.95"},
        {"date": "07/30/2026", "close": "$39.49", "volume": "N/A", "open": "$39.49",
         "high": "$39.49", "low": "$39.49"},
        {"date": "junk", "close": "$1", "volume": "1", "open": "$1", "high": "$1", "low": "$1"},
    ]}}}
    seen = {}

    def fake_get_json(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return payload

    monkeypatch.setattr(net, "get_json", fake_get_json)
    df = P._nasdaq_history_raw("BRK-B", date(2026, 7, 1), date(2026, 8, 1), TUE_EVENING)
    assert seen["url"].endswith("/BRK.B/historical")
    assert seen["kw"]["params"]["assetclass"] == "stocks" and seen["kw"]["headers"] == net.NASDAQ_HEADERS
    assert_contract_frame(df)
    assert df.index.tolist() == [pd.Timestamp("2026-07-30"), pd.Timestamp("2026-07-31")]
    assert df.loc["2026-07-30", "volume"] == 0.0              # halt: N/A → 0, row kept
    assert df.loc["2026-07-31", "volume"] == 3305994.0
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    assert P._nasdaq_history_raw("INHD", None, date(2026, 8, 1), TUE_EVENING).empty


# ── pre-market ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("Sep 29, 2026 6:57 PM ET", datetime(2026, 9, 29, 18, 57, tzinfo=ET)),
    ("Closed at Sep 29, 2026 4:00 PM ET", datetime(2026, 9, 29, 16, 0, tzinfo=ET)),
    ("Sep 30, 2026 12:05 AM ET", datetime(2026, 9, 30, 0, 5, tzinfo=ET)),
    ("Sep 30, 2026 12:30 PM ET", datetime(2026, 9, 30, 12, 30, tzinfo=ET)),
    ("Sep 30, 2026", datetime(2026, 9, 30, tzinfo=ET)),
    ("N/A", None), ("", None), (None, None),
])
def test_parse_nasdaq_timestamp(text, expected):
    assert P.parse_nasdaq_timestamp(text) == expected


def _quote(price, change="", ts="Sep 30, 2026 8:15 AM ET", volume="12,345", status="Pre-Market",
           sec_price="$3.17", sec_ts="Closed at Sep 29, 2026 4:00 PM ET", delta="up"):
    return {"data": {
        "symbol": "INHD", "marketStatus": status,
        "primaryData": {"lastSalePrice": price, "netChange": change, "percentageChange": "",
                        "deltaIndicator": delta, "lastTradeTimestamp": ts, "isRealTime": True,
                        "volume": volume},
        "secondaryData": {"lastSalePrice": sec_price, "lastTradeTimestamp": sec_ts,
                          "isRealTime": False} if sec_price is not None else None,
    }}


def test_parse_quote_premarket_gap(env):
    snap, why = P.parse_nasdaq_quote("INHD", _quote("$3.60", "+0.43"), WED_PREMARKET)
    assert why == "ok"
    assert snap["price"] == 3.6 and snap["prev_close"] == 3.17
    assert snap["gap_pct"] == pytest.approx(3.6 / 3.17 - 1)
    assert snap["volume"] == 12345.0 and snap["session"] == "pre-market"
    assert snap["source"] == "nasdaq" and snap["asof"].startswith("2026-09-30T08:15")


def test_parse_quote_after_hours_uses_close_block_and_hides_volume(env):
    q = _quote("$330.02", "+0.62", ts="Sep 29, 2026 7:09 PM ET", volume="38,454,612.97",
               status="After-Hours", sec_price="$329.40")
    snap, _ = P.parse_nasdaq_quote("AAPL", q, datetime(2026, 9, 29, 19, 10, tzinfo=ET))
    assert snap["session"] == "after-hours" and snap["prev_close"] == 329.40
    assert snap["gap_pct"] == pytest.approx(330.02 / 329.40 - 1)
    assert snap["volume"] is None                           # cumulative day volume, not AH volume


@pytest.mark.parametrize("quote,now,reason", [
    (_quote("$3.17", "", ts="Sep 29, 2026 7:09 PM ET", status="After-Hours", delta="unch"),
     datetime(2026, 9, 29, 19, 10, tzinfo=ET), "no extended-hours trade"),
    (_quote("$3.40", "+0.23", status="Market Open", ts="Sep 30, 2026 10:15 AM ET"),
     datetime(2026, 9, 30, 10, 16, tzinfo=ET), "regular session"),
    (_quote("$3.40", "+0.23", ts="Sep 25, 2026 7:59 PM ET", status="Closed"),
     WED_PREMARKET, "quote outside the extended-hours window"),
    (_quote("N/A"), WED_PREMARKET, "no quote"),
    ({"data": None}, WED_PREMARKET, "bad payload"),
])
def test_parse_quote_rejections(env, quote, now, reason):
    snap, why = P.parse_nasdaq_quote("INHD", quote, now)
    assert snap is None and why == reason


def test_parse_quote_prev_close_fallbacks(env):
    # secondaryData is for the wrong session → cached daily bar is used.
    P.load_history(["INHD"], period="1y")
    cached_close = P.read_cache("INHD")["df"].loc["2026-09-29", "close"]
    env.clock.now = WED_PREMARKET
    q = _quote("$20.00", "+9.99", sec_ts="Closed at Sep 25, 2026 4:00 PM ET")
    snap, _ = P.parse_nasdaq_quote("INHD", q, WED_PREMARKET)
    assert snap["prev_close"] == pytest.approx(cached_close)
    # no secondary block, no cache → price − netChange
    snap, _ = P.parse_nasdaq_quote("ZZZ", _quote("$3.60", "+0.43", sec_price=None), WED_PREMARKET)
    assert snap["prev_close"] == pytest.approx(3.17)
    # nothing known → still reported (a real pre-market trade) but gap unknown
    snap, _ = P.parse_nasdaq_quote("ZZZ", _quote("$3.60", "", sec_price=None, delta="up"), WED_PREMARKET)
    assert snap["prev_close"] is None and snap["gap_pct"] is None and snap["volume"] == 12345.0
    # flat price with pre-market volume counts as a trade (gap 0)
    snap, _ = P.parse_nasdaq_quote("ZZZ", _quote("$3.17", "", delta="unch"), WED_PREMARKET)
    assert snap["gap_pct"] == 0.0


def test_premarket_snapshot_with_yahoo_fallback(env, monkeypatch):
    env.clock.now = WED_PREMARKET
    quotes = {"AAA": _quote("$3.60", "+0.43"), "BBB": None,
              "CCC": _quote("$3.17", "", volume="", delta="unch")}
    monkeypatch.setattr(P, "_nasdaq_quote", lambda s: quotes[s])
    yahoo_calls = []

    def fake_yahoo(symbol, now):
        yahoo_calls.append(symbol)
        return {"price": 5.0, "prev_close": 4.0, "gap_pct": 0.25, "volume": None,
                "asof": "2026-09-30T08:10:00-04:00", "source": "yahoo", "session": "pre-market",
                "realtime": False, "market_status": None}, False

    monkeypatch.setattr(P, "_yahoo_premarket", fake_yahoo)
    snap = P.premarket_snapshot(["AAA", "BBB", "CCC"])
    assert set(snap) == {"AAA", "BBB"}                      # CCC: no pre-market trade → omitted
    assert snap["AAA"]["source"] == "nasdaq" and snap["BBB"]["source"] == "yahoo"
    assert yahoo_calls == ["BBB"]                           # only for the failed request
    for v in snap.values():
        assert {"price", "prev_close", "gap_pct", "volume", "asof", "source"} <= set(v)
    st = net.STATUS[P.NASDAQ_STATUS]
    assert st["ok"] is True and "2/3 quotes" in st["detail"]


# ── warm script ──────────────────────────────────────────────────────────
def _load_warm_module():
    spec = importlib.util.spec_from_file_location("warm_prices", ROOT / "scripts" / "warm_prices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_warm_script_end_to_end_with_fakes(env, monkeypatch, capsys):
    warm = _load_warm_module()
    uni = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"]})
    uni.attrs.update({"listed": 10, "fetched_at": "2026-09-29T22:00:00+00:00", "stale": False})
    monkeypatch.setattr(warm.universe, "load_universe", lambda: uni)
    monkeypatch.setattr(warm.time, "sleep", lambda s: None)
    env.yahoo.missing.add("DDD")
    env.yahoo.splits["BBB"] = [{"date": "2026-05-04", "ratio": 0.05}]
    rc = warm.main(["--limit", "4", "--batch", "2", "--pause", "0", "--batch-pause", "0", "--period", "1y"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "requested        4" in out and "ok               3" in out and "failed           1" in out
    assert "benchmark IWM" in out
    # Resumable: a second run is served entirely from cache.
    env.yahoo.batch_calls.clear()
    rc = warm.main(["--limit", "3", "--pause", "0", "--batch-pause", "0", "--period", "1y", "--no-benchmark"])
    assert rc == 0 and env.yahoo.batch_calls == []


# ── live (GRAVITY_LIVE=1) ────────────────────────────────────────────────
@pytest.fixture
def live_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    return tmp_path / "cache"


@pytest.mark.live
@live
def test_live_history_splits_incremental(live_cache):
    syms = ["INHD", "AAPL", "SOUN"]
    out = P.load_history(syms, period="1y")
    assert set(out) == set(syms), P.LAST_RUN
    for s, df in out.items():
        assert_contract_frame(df)
        assert len(df) > 150, s
        assert df.attrs["source"] == "yahoo"
    inhd = out["INHD"]
    splits = P.load_splits(["INHD"])["INHD"]
    assert {"date": "2026-05-04", "ratio": 0.05} in splits
    halted = inhd.loc["2026-06-09":"2026-07-29"]
    assert len(halted) > 0 and (halted["volume"] == 0).mean() > 0.9        # the June–July halt
    print(f"\nlive INHD: {len(inhd)} bars {inhd.index.min().date()}→{inhd.index.max().date()}, "
          f"last close {inhd['close'].iloc[-1]:.2f}, splits {splits}, "
          f"20d median $vol {P.dollar_volume_median(inhd):,.0f}, halted-zero-vol days {int((halted['volume'] == 0).sum())}")

    out2 = P.load_history(syms, period="1y")
    assert P.LAST_RUN["current"] == 3
    # Force an incremental refresh: pretend the cache is a week old and missing 3 bars.
    e = P.read_cache("INHD")
    e["fetched_at"] = "2026-01-01T00:00:00+00:00"
    e["df"] = e["df"].iloc[:-3]
    P._write_cache("INHD", e)
    out3 = P.load_history(["INHD"], period="1y")
    assert P.LAST_RUN["incremental"] == 1, P.LAST_RUN
    pd.testing.assert_frame_equal(out3["INHD"], out2["INHD"], check_exact=False, rtol=1e-6)
    print("live incremental:", P.LAST_RUN)


@pytest.mark.live
@live
def test_live_nasdaq_history_matches_yahoo(live_cache):
    y = P.load_history(["INHD"], period="1y")["INHD"]
    now = P._now()
    nd = P._nasdaq_history_raw("INHD", P.period_start("1y", now.date()), now.date(), now)
    assert len(nd) > 150
    adj, applied = P.adjust_for_splits(nd, P.load_splits(["INHD"])["INHD"])
    common = y.index.intersection(adj.index)
    ratio = (adj.loc[common, "close"] / y.loc[common, "close"]).median()
    print(f"\nlive Nasdaq INHD: {len(nd)} bars, splits applied {applied}, "
          f"median Nasdaq/Yahoo close ratio {ratio:.4f} over {len(common)} days")
    assert abs(ratio - 1) < 0.01


@pytest.mark.live
@live
def test_live_premarket_and_benchmark(live_cache):
    snap = P.premarket_snapshot(["AAPL", "TSLA", "INHD"])
    st = net.STATUS[P.NASDAQ_STATUS]
    print(f"\nlive premarket ({st['detail']}):", snap)
    assert st["ok"] is True
    for v in snap.values():
        assert v["price"] > 0 and v["source"] in ("nasdaq", "yahoo")
    bench = P.benchmark_history(period="1y")
    assert_contract_frame(bench)
    assert len(bench) > 200
    print(f"live IWM: {len(bench)} bars, last {bench.index.max().date()} close {bench['close'].iloc[-1]:.2f}")
