"""Unit tests for gravity.sources.intel parsers, against real filing snippets
saved under tests/fixtures (no network)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from gravity.sources import intel

F = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (F / name).read_text()


def test_atm_prospectus_sizes():
    r = intel.parse_offering_text(_read("intel_424b5_inhd_atm_20260519.txt"), "424B5")
    assert r["type"] == "atm" and r["atm_capacity"] == 60_000_000 and r["price"] is None
    r2 = intel.parse_offering_text(_read("intel_424b5_fngr_atm_20251023.txt"), "424B5")
    assert r2["type"] == "atm" and r2["atm_capacity"] == 50_000_000


def test_priced_offering_price_shares_gross_and_warrants():
    r = intel.parse_offering_text(_read("intel_424b5_fngr_priced_20260831.txt"), "424B5")
    assert r["type"] == "priced"
    assert r["price"] == pytest.approx(0.24)
    assert r["shares"] == 3_958_055
    assert r["gross"] == 4_000_000
    assert r["warrants"]["shares"] == 8_275_594 and r["warrants"]["exercise_price"] == pytest.approx(1.64)


def test_registered_direct():
    r = intel.parse_offering_text(_read("intel_424b5_inhd_rd_20260120.txt"), "424B5")
    assert r["type"] == "registered_direct"
    assert r["price"] == pytest.approx(0.55) and r["shares"] == 1_332_000


def test_parser_never_invents_numbers():
    r = intel.parse_offering_text("This prospectus relates to nothing in particular.", "424B5")
    assert r["price"] is None and r["shares"] is None and r["gross"] is None and r["atm_capacity"] is None


def test_form4_parse():
    r = intel.parse_form4(_read("intel_form4_inhd_20260911.xml"))
    assert r["owners"][0]["name"] == "Wei Ding" and r["trades"] == []
    r2 = intel.parse_form4(_read("intel_form4_fngr_20251229.xml"))
    assert r2["owners"][0]["title"] == "Director"
    assert all({"date", "code", "shares"} <= set(t) for t in r2["trades"])
    assert intel.parse_form4("<not-xml") is None


def test_stocktwits_stream_rate_and_sentiment():
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    msgs = [{"created_at": f"2026-09-30T{h:02d}:00:00Z",
             "entities": {"sentiment": {"basic": "Bearish" if h % 3 == 0 else "Bullish"}}} for h in range(0, 12)]
    payload = {"symbol": {"watchlist_count": 1234}, "messages": msgs}
    r = intel.parse_stream("ABC", payload, now=now)
    assert r["watchers"] == 1234
    assert r["bull"] + r["bear"] == 12 and r["bear"] == 4
    assert r["msgs_per_day"] > 0
    assert intel.parse_stream("ABC", {"messages": []}, now=now)["msgs_per_day"] in (0, 0.0, None)


def test_lockup_window(monkeypatch):
    rows_ = [
        {"symbol": "NEWC", "name": "New Co", "priced_date": "2026-04-05", "price": 10.0, "shares": 1e6,
         "offer_amount": 1e7, "exchange": "NASDAQ", "source": "x"},
        {"symbol": "OLDC", "name": "Old Co", "priced_date": "2025-01-05", "price": 5.0, "shares": 1e6,
         "offer_amount": 5e6, "exchange": "NASDAQ", "source": "x"},
    ]
    monkeypatch.setattr(intel, "_ipo_months", lambda months, today: (rows_, True))
    rows = intel.lockups(date(2026, 9, 30), horizon_days=30, lookback_days=10)
    assert [r["symbol"] for r in rows] == ["NEWC"]
    assert rows[0]["lockup_date"] == "2026-10-02" and rows[0]["days_to"] == 2
    assert "180" in rows[0]["assumed"]


def test_downsample_keeps_ends():
    pts = [[i, float(i)] for i in range(1000)]
    d = intel.downsample(pts, 90)
    assert len(d) <= 90 and d[0] == pts[0] and d[-1] == pts[-1]


def test_par_value_is_never_an_offering_price():
    txt = ("Ordinary Shares, par value $0.0001 per share. We are offering 2,000,000 ordinary shares "
           "at a public offering price of $1.25 per share.")
    assert intel.parse_offering_text(txt, "424B4")["price"] == pytest.approx(1.25)
    only_par = "Class A Ordinary Shares, par value $0.0001 per share, offered from time to time."
    assert intel.parse_offering_text(only_par, "424B5")["price"] is None
