"""Tests for gravity.sources.shortside (CONTRACTS.md §4).

Unit tests replace the IBKR FTP download, ``net.get_text`` / ``net.get_json``
and the yfinance call with canned payloads shaped like the real feeds
(verified 2026-09-30). Live tests (``GRAVITY_LIVE=1``) touch one IBKR file,
five FINRA files, two Nasdaq short-interest calls and two Yahoo lookups,
into a temporary cache.
"""

from __future__ import annotations

import gzip
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.sources import shortside as S  # noqa: E402
from gravity.util import ET, is_trading_day  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="set GRAVITY_LIVE=1 for live network tests")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    for k in ("IBKR borrow", "FINRA short volume", "Nasdaq short interest", "Yahoo float"):
        net.STATUS.pop(k, None)
    monkeypatch.setattr(S, "_SI_STATS", {"ok": 0, "empty": 0, "fail": 0})
    return tmp_path / "cache"


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network touched")
    monkeypatch.setattr(net, "get", boom)
    monkeypatch.setattr(net, "get_text", boom)
    monkeypatch.setattr(net, "get_json", boom)
    monkeypatch.setattr(S, "_fetch_ibkr_text", boom)
    monkeypatch.setattr(S, "_yf_info", boom)


# ═════════════════════════════════════════════════════════════════════════
# IBKR
# ═════════════════════════════════════════════════════════════════════════
IBKR_TEXT = "\n".join([
    "#BOF|2026.09.30|08:22:51",
    "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|",
    "049323AB4|USD|CB ATLAS FINL HLDGS 06.625% 27|557991763|XXXXXXX3AB46|3.6300|0.2500|300000|BBG017QTG7P1|",
    "AAPL|USD|APPLE INC|265598|US0378331005|3.6300|0.2500|>10000000|BBG000B9XRY4|",
    "BRK B|USD|BERKSHIRE HATHAWAY INC-CL B|72063691|US0846707026|3.6300|0.2500|>10000000|BBG000DWG505|",
    "ABR PRD|USD|ARBOR REALTY TRUST PFD D|123|US0|3.1|0.5|4000|BBG0|",
    "INHD|USD|INNO HOLDINGS INC|879173582|XXXXXXXP4067|-94.9314|98.8114|35000|BBG01HMYBHL9|",
    "NOAV|USD|NOTHING AVAILABLE|1|X|-10.0|13.5|0|B|",
    "BLNK|USD|BLANK AVAIL|2|X|||   |B|",
    "AEGG.OLD|USD|LEGACY|3|X|1|1|100|B|",
    "",
    "#EOF|8",
])


@pytest.mark.parametrize("raw,expected", [
    ("INHD", "INHD"), ("brk b", "BRK-B"), ("ABR PRD", "ABR-PD"), ("AGM A", "AGM-A"),
    ("049323AB4", None), ("AEGG.OLD", None), ("3LI", None), ("", None), ("TOOLONGSYM", None),
])
def test_ibkr_symbol(raw, expected):
    assert S.ibkr_symbol(raw) == expected


def test_parse_ibkr_contract_shape():
    d = S.parse_ibkr(IBKR_TEXT)
    assert set(d) == {"AAPL", "BRK-B", "ABR-PD", "INHD", "NOAV", "BLNK"}
    assert d["INHD"] == {"fee_rate": 98.8114, "rebate_rate": -94.9314, "available": 35000,
                         "asof": "2026-09-30T08:22:51-04:00"}
    assert d["AAPL"]["available"] == 10_000_000 and isinstance(d["AAPL"]["available"], int)
    assert d["NOAV"]["available"] == 0
    assert d["BLNK"] == {"fee_rate": None, "rebate_rate": None, "available": None,
                         "asof": "2026-09-30T08:22:51-04:00"}


def test_parse_ibkr_rejects_truncated_or_headerless_files():
    assert S.parse_ibkr(IBKR_TEXT.replace("#EOF|8", "")) is None
    no_header = "\n".join(l for l in IBKR_TEXT.splitlines() if not l.startswith("#SYM"))
    assert S.parse_ibkr(no_header) is None
    assert S.parse_ibkr("") is None


def test_parse_ibkr_asof():
    assert S._parse_ibkr_asof("#BOF|2026.09.29|18:49:19") == "2026-09-29T18:49:19-04:00"
    assert S._parse_ibkr_asof("#BOF|2026.01.05|07:00:00") == "2026-01-05T07:00:00-05:00"
    assert S._parse_ibkr_asof("#BOF|junk") is None


def test_ibkr_borrow_caches_and_reports(tmp_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "_fetch_ibkr_text", lambda: calls.append(1) or IBKR_TEXT)
    b = S.ibkr_borrow()
    assert b["INHD"]["fee_rate"] == pytest.approx(98.8114)
    assert net.STATUS["IBKR borrow"]["ok"] is True and "6 symbols" in net.STATUS["IBKR borrow"]["detail"]
    assert S.ibkr_borrow() == b and len(calls) == 1        # 30-minute cache


def test_ibkr_borrow_failure_is_empty_not_cached(tmp_cache, monkeypatch):
    monkeypatch.setattr(S, "_fetch_ibkr_text", lambda: None)
    assert S.ibkr_borrow() == {}
    assert net.STATUS["IBKR borrow"]["ok"] is False
    monkeypatch.setattr(S, "_fetch_ibkr_text", lambda: IBKR_TEXT.replace("#EOF|8", ""))
    assert S.ibkr_borrow() == {}                           # truncated file never cached
    monkeypatch.setattr(S, "_fetch_ibkr_text", lambda: IBKR_TEXT)
    assert "INHD" in S.ibkr_borrow()


# ═════════════════════════════════════════════════════════════════════════
# FINRA
# ═════════════════════════════════════════════════════════════════════════
def _finra_file(d: date, rows=None) -> str:
    ymd = d.strftime("%Y%m%d")
    rows = rows if rows is not None else [
        ("INHD", "6715.979266", "0", "10319.412607"),
        ("AAPL", "2000000", "1000", "5000000"),
        ("BRK/B", "100", "0", "400"),
        ("ABRpD", "2968", "0", "4208.2213"),
        ("ZERO", "0", "0", "0"),
    ]
    lines = ["Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"]
    lines += [f"{ymd}|{s}|{sv}|{ex}|{tv}|B,Q,N" for s, sv, ex, tv in rows]
    lines.append(str(len(rows)))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("raw,expected", [
    ("INHD", "INHD"), ("BRK/B", "BRK-B"), ("ABRpD", "ABR-PD"), ("BRK.A", "BRK-A"), (" AAPL ", "AAPL"),
])
def test_finra_symbol(raw, expected):
    assert S.finra_symbol(raw) == expected


def test_parse_finra_columns_and_ratio():
    df = S.parse_finra(_finra_file(date(2026, 9, 29)))
    assert list(df.columns) == S.FINRA_COLUMNS
    assert set(df["symbol"]) == {"INHD", "AAPL", "BRK-B", "ABR-PD", "ZERO"}
    assert (df["date"] == "2026-09-29").all()
    inhd = df[df.symbol == "INHD"].iloc[0]
    assert inhd["short_ratio"] == pytest.approx(6715.979266 / 10319.412607)
    assert math.isnan(df[df.symbol == "ZERO"].iloc[0]["short_ratio"])     # 0/0 is unknown, not 0
    r = df["short_ratio"].dropna()
    assert ((r >= 0) & (r <= 1)).all()


def test_finra_complete_trailer():
    assert S._finra_complete(_finra_file(date(2026, 9, 29)))
    assert not S._finra_complete(_finra_file(date(2026, 9, 29)).rsplit("\n", 2)[0])
    assert not S._finra_complete("<html>403</html>")


def test_finra_candidate_days_skip_weekends_holidays_and_unpublished():
    # Wednesday 08:00 ET: today's file isn't out → Tue, Mon, Fri(25th) …
    days = S._finra_candidate_days(3, datetime(2026, 9, 30, 8, 0, tzinfo=ET))
    assert days == [date(2026, 9, 29), date(2026, 9, 28), date(2026, 9, 25)]
    # Tuesday 19:00 ET: Tuesday's file is out
    assert S._finra_candidate_days(1, datetime(2026, 9, 29, 19, 0, tzinfo=ET)) == [date(2026, 9, 29)]
    # Labor Day 2026-09-07 is skipped
    days = S._finra_candidate_days(2, datetime(2026, 9, 8, 7, 0, tzinfo=ET))
    assert days == [date(2026, 9, 4), date(2026, 9, 3)]
    assert all(is_trading_day(d) for d in days)


def test_finra_short_volume_fetch_cache_and_gaps(tmp_cache, monkeypatch):
    got = []
    published = {date(2026, 9, 29), date(2026, 9, 25), date(2026, 9, 24)}   # 09-28 missing (403)

    def fake_text(url, **kw):
        got.append(url)
        d = datetime.strptime(url.rsplit("CNMSshvol", 1)[1][:8], "%Y%m%d").date()
        return _finra_file(d) if d in published else None

    monkeypatch.setattr(net, "get_text", fake_text)
    monkeypatch.setattr(S, "now_et", lambda: datetime(2026, 9, 30, 8, 0, tzinfo=ET))
    df = S.finra_short_volume(3)
    assert list(df.columns) == S.FINRA_COLUMNS
    assert sorted(df["date"].unique()) == ["2026-09-24", "2026-09-25", "2026-09-29"]
    assert list(df.sort_values(["date", "symbol"]).index) == list(df.index)
    st = net.STATUS["FINRA short volume"]
    assert st["ok"] is True and "missing 2026-09-28" in st["detail"]
    assert got[0].endswith("CNMSshvol20260929.txt")
    # complete files are kept on disk forever: a second run only re-asks for the missing day
    got.clear()
    S.finra_short_volume(3)
    assert got == ["https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260928.txt"]
    assert gzip.decompress(S._finra_cache_path(date(2026, 9, 29)).read_bytes()).decode().startswith("Date|")


def test_finra_incomplete_file_is_used_but_not_cached(tmp_cache, monkeypatch):
    partial = _finra_file(date(2026, 9, 29)).rsplit("\n", 2)[0]
    monkeypatch.setattr(net, "get_text", lambda url, **kw: partial if "20260929" in url else None)
    monkeypatch.setattr(S, "now_et", lambda: datetime(2026, 9, 30, 8, 0, tzinfo=ET))
    df = S.finra_short_volume(1)
    assert len(df) == 5                               # every row; only the count trailer is missing
    assert not S._finra_cache_path(date(2026, 9, 29)).exists()


def test_finra_nothing_reachable(tmp_cache, monkeypatch):
    monkeypatch.setattr(net, "get_text", lambda url, **kw: None)
    df = S.finra_short_volume(2)
    assert df.empty and list(df.columns) == S.FINRA_COLUMNS
    assert net.STATUS["FINRA short volume"]["ok"] is False


# ═════════════════════════════════════════════════════════════════════════
# Nasdaq short interest
# ═════════════════════════════════════════════════════════════════════════
SI_PAYLOAD = {
    "data": {"symbol": "INHD", "shortInterestTable": {"headers": {}, "rows": [
        {"settlementDate": "08/31/2026", "interest": "108,982", "avgDailyShareVolume": "322,021", "daysToCover": 1.0},
        {"settlementDate": "09/15/2026", "interest": "49,699", "avgDailyShareVolume": "80,587", "daysToCover": 1.0},
        {"settlementDate": "08/14/2026", "interest": "185,748", "avgDailyShareVolume": "0", "daysToCover": 0.0},
        {"settlementDate": "bad", "interest": "1", "avgDailyShareVolume": "1", "daysToCover": 1},
    ]}},
    "message": None, "status": {"rCode": 200},
}


def test_parse_short_interest_newest_first_and_recomputed_dtc():
    rows = S.parse_short_interest(SI_PAYLOAD)
    assert [r["settlement_date"] for r in rows] == ["2026-09-15", "2026-08-31", "2026-08-14"]
    top = rows[0]
    assert top["interest"] == 49699 and top["avg_daily_volume"] == 80587
    assert top["days_to_cover"] == pytest.approx(49699 / 80587, abs=1e-4)
    assert top["days_to_cover_reported"] == 1.0
    assert rows[2]["avg_daily_volume"] is None and rows[2]["days_to_cover"] is None
    for r in rows:
        assert set(r) >= {"settlement_date", "interest", "avg_daily_volume", "days_to_cover"}


def test_parse_short_interest_empty_vs_unrecognised():
    assert S.parse_short_interest({"data": None, "status": {"rCode": 400}}) == []
    assert S.parse_short_interest({"data": {"shortInterestTable": None}}) == []
    assert S.parse_short_interest({"data": None, "status": {"rCode": 500}}) is None
    assert S.parse_short_interest(None) is None
    assert S.parse_short_interest("<html>") is None


def test_short_interest_request_and_cache(tmp_cache, monkeypatch):
    seen = []

    def fake(url, **kw):
        seen.append((url, kw.get("headers")))
        return SI_PAYLOAD if "INHD" in url else {"data": None, "status": {"rCode": 400}}

    monkeypatch.setattr(net, "get_json", fake)
    rows = S.short_interest("inhd")
    assert rows[0]["settlement_date"] == "2026-09-15"
    assert seen[0] == ("https://api.nasdaq.com/api/quote/INHD/short-interest?assetClass=stocks", net.NASDAQ_HEADERS)
    assert S.short_interest("BRK-B") == []
    assert seen[1][0].startswith("https://api.nasdaq.com/api/quote/BRK.B/")
    S.short_interest("INHD")
    assert len(seen) == 2                                      # cached 12h
    st = net.STATUS["Nasdaq short interest"]
    assert st["ok"] is True and "0 failed" in st["detail"]


def test_short_interest_failure_is_empty_and_retried_later(tmp_cache, monkeypatch):
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    assert S.short_interest("INHD") == []
    assert "1 failed" in net.STATUS["Nasdaq short interest"]["detail"]
    monkeypatch.setattr(net, "get_json", lambda url, **kw: SI_PAYLOAD)
    assert S.short_interest("INHD")                            # failure was not cached


# ═════════════════════════════════════════════════════════════════════════
# Float (yfinance)
# ═════════════════════════════════════════════════════════════════════════
INFO_INHD = {"floatShares": 2489628, "sharesOutstanding": 2520581, "shortPercentOfFloat": 0.0197,
             "sharesShort": 49699, "dateShortInterest": 1789430400}


def test_parse_float_info():
    r = S.parse_float_info(INFO_INHD)
    assert r["float"] == 2489628 and r["shares_out"] == 2520581
    assert r["short_pct_float"] == pytest.approx(0.0197) and r["source"] == "yahoo"
    assert r["si_date"] == datetime.utcfromtimestamp(1789430400).date().isoformat()
    only_implied = S.parse_float_info({"impliedSharesOutstanding": 100.0, "shortPercentOfFloat": -1})
    assert only_implied["shares_out"] == 100.0 and only_implied["short_pct_float"] is None
    assert only_implied["float"] is None
    assert S.parse_float_info({}) is None
    assert S.parse_float_info({"trailingPegRatio": None}) is None
    assert S.parse_float_info(None) is None


def test_float_shares_caches_hits_only_and_stops_when_throttled(tmp_cache, monkeypatch):
    calls = []

    def fake_info(sym):
        calls.append(sym)
        if sym == "EMPTY":
            return {}
        if sym == "LIMIT":
            raise S._RateLimited("Too Many Requests")
        return dict(INFO_INHD)

    monkeypatch.setattr(S, "_yf_info", fake_info)
    out = S.float_shares(["INHD", "inhd", "EMPTY", "LIMIT", "AAPL"], pause_s=0)
    assert set(out) == {"INHD"}                                # AAPL skipped after the rate limit
    assert calls == ["INHD", "EMPTY", "LIMIT"]
    st = net.STATUS["Yahoo float"]
    assert "stopped: Yahoo rate limit" in st["detail"]
    calls.clear()
    monkeypatch.setattr(S, "_yf_info", lambda s: calls.append(s) or dict(INFO_INHD))
    out = S.float_shares(["INHD", "EMPTY", "AAPL"], pause_s=0)
    assert calls == ["EMPTY", "AAPL"]                          # INHD from cache; EMPTY never cached
    assert set(out) == {"INHD", "EMPTY", "AAPL"}
    assert S.float_shares([], pause_s=0) == {}


# ═════════════════════════════════════════════════════════════════════════
# Live
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.live
@live
def test_live_ibkr_borrow(tmp_cache):
    b = S.ibkr_borrow()
    assert len(b) > 5000
    inhd = b["INHD"]
    assert isinstance(inhd["available"], int) and inhd["fee_rate"] is not None
    assert 20 <= inhd["fee_rate"] <= 500              # hard to borrow (≈98.8 %/yr on 2026-09-30)
    assert b["AAPL"]["available"] == 10_000_000
    print("\nIBKR:", len(b), "symbols; INHD", inhd)


@pytest.mark.live
@live
def test_live_finra_short_volume(tmp_cache):
    df = S.finra_short_volume(5)
    assert list(df.columns) == S.FINRA_COLUMNS
    assert df["date"].nunique() == 5 and len(df) > 20000
    r = df["short_ratio"].dropna()
    assert ((r >= 0) & (r <= 1)).all()
    inhd = df[df.symbol == "INHD"]
    assert len(inhd) >= 1
    print("\nFINRA:", sorted(df["date"].unique()), len(df), "rows; INHD\n",
          inhd[["date", "short_volume", "total_volume", "short_ratio"]].to_string(index=False))


@pytest.mark.live
@live
def test_live_short_interest_and_float(tmp_cache):
    si = S.short_interest("INHD")
    assert si, "Nasdaq short interest empty for INHD"
    dates = [r["settlement_date"] for r in si]
    assert dates == sorted(dates, reverse=True)
    fl = S.float_shares(["INHD"], pause_s=0)
    assert "INHD" in fl and (fl["INHD"]["float"] or fl["INHD"]["shares_out"])
    print("\nSI:", si[:2], "\nfloat:", fl["INHD"])
