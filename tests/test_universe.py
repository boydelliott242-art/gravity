"""Tests for gravity.sources.universe (CONTRACTS.md §1).

Unit tests mock the network; the live test (``GRAVITY_LIVE=1``) makes one
real screener request into a temporary cache.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo importable when pytest is run without a root conftest/ini.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.sources import universe as U  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"


def _row(symbol, name, lastsale="$2.99", pct="-3.548%", volume="28703", cap="7536537.00",
         country="United States", ipo="2023", industry="Steel/Iron Ore", sector="Industrials"):
    return {"symbol": symbol, "name": name, "lastsale": lastsale, "netchange": "-0.11",
            "pctchange": pct, "volume": volume, "marketCap": cap, "country": country,
            "ipoyear": ipo, "industry": industry, "sector": sector,
            "url": f"/market-activity/stocks/{symbol.lower()}"}


SAMPLE = [
    _row("INHD", "Inno Holdings Inc. Common Stock"),
    _row("BRK/B", "Berkshire Hathaway Inc.", lastsale="$480.10", cap="1030000000000.00"),
    _row("HVT/A", "Haverty Furniture Companies Inc.", lastsale="$28.09", cap=""),
    _row("ABR^D", "Arbor Realty Trust 6.375% Series D Preferred Stock", lastsale="$18.00"),
    _row("DCOM^", "Dime Commercial Bancshares Inc. Preferred Stock Series A"),
    _row("BNCWW", "CEA Industries Inc. Warrant", lastsale="$0.40"),
    _row("IBACR", "IB Acquisition Corp. Right", lastsale="$0.20"),
    _row("TLACU", "Three Lions Acquisition Corp. Units", lastsale="$10.16", cap="0.00"),
    _row("TWOD", "Two Harbors Investments Corp 9.375% Senior Notes due 2030", lastsale="$25.10"),
    _row("NEEX", "NextEra Energy Inc. Series U Junior Subordinated Debentures due 2085"),
    _row("ACV", "Virtus Diversified Income & Convertible Fund Common Shares of Beneficial Interest",
         sector="Finance", industry="Finance/Investors Services"),
    _row("ECC  ", "Eagle Point Credit Company Common Share of Beneficial Interest",
         sector="Finance", industry="Trusts Except Educational Religious and Charitable"),
    _row("UE", "Urban Edge Properties Common Shares of Beneficial Interest", sector="Real Estate",
         industry="Real Estate Investment Trusts", cap="1500000000.00"),
    _row("CAPL", "CrossAmerica Partners LP Common Units representing limited partner interests"),
    _row("PPLC", "PPL Corporation Corporate Units"),
    _row("UNTY", "Unity Bancorp Inc. Common Stock", cap="500000000.00"),
    _row("CCS", "Century Communities Inc. Common Stock", cap="1800000000.00"),
    _row("MYND", "Mynd.ai Inc. American Depositary Shares", country="China", ipo=""),
    _row("TINY", "Tiny Penny Corp. Common Stock", lastsale="$0.05"),
    _row("BIGC", "Big Cap Corp. Common Stock", cap="2500000000.00"),
    _row("NOPX", "No Price Inc. Common Stock", lastsale=""),
    _row("ZVZZT", "NASDAQ TEST STOCK"),
    _row("PSNYW", "Polestar Automotive Holding UK PLC Class C-1 ADS (ADW)", lastsale="$0.30"),
    _row("MNYWW", "Moneylion Inc. Wts", lastsale="$0.30"),
]


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    net.STATUS.pop(U.SOURCE_NAME, None)
    return tmp_path / "cache"


# ── exclusion rules ──────────────────────────────────────────────────────
@pytest.mark.parametrize("symbol,name,sector,industry,expected", [
    ("ABR^D", "Arbor Realty Trust Series D", "", "", "preferred"),
    ("ATLCP", "Atlanticus Holdings 7.625% Series B Cumulative Perpetual Preferred Stock", "", "", "preferred"),
    ("GAB^H", "Gabelli Equity Trust Inc. (The) Pfd Ser H", "", "", "preferred"),
    ("BNCWW", "CEA Industries Inc. Warrant", "", "", "warrant"),
    ("OPENL", "Opendoor Technologies Inc Series A Warrants each whole warrant", "", "", "warrant"),
    ("PSNYW", "Polestar Automotive Holding UK PLC Class C-1 ADS (ADW)", "", "", "warrant"),
    ("KTWOR", "K2 Capital Acquisition Corporation Rights", "", "", "right"),
    ("AMPGZ", "Amplitech Group Inc. Series B Right", "", "", "right"),
    ("BLZRU", "Trailblazer Acquisition Corp. Unit", "", "", "unit"),
    ("XYZU", "XYZ Acquisition Corp Units, each consisting of one share and one right", "", "", "unit"),
    ("PPLC", "PPL Corporation Corporate Units", "", "", "unit"),
    ("TWOD", "Two Harbors 9.375% Senior Notes due 2030", "", "", "debt"),
    ("NEEU", "NextEra Energy Inc. Series U Junior Subordinated Debentures due 2085", "", "", "debt"),
    ("SAV", "Saratoga Investment Corp 7.50% Notes due 2031", "", "", "debt"),
    ("ADX", "Adams Diversified Equity Fund Inc.", "Finance", "Investment Managers", "fund"),
    ("BHV", "BlackRock Virginia Municipal Bond Trust", "Finance", "", "fund"),
    ("BCX", "BlackRock Resources Common Shares of Beneficial Interest", "Finance", "", "fund"),
    ("GUT", "Gabelli Utility Trust (The) Common Stock", "Finance",
     "Trusts Except Educational Religious and Charitable", "fund"),
    ("ZVZZT", "NASDAQ TEST STOCK", "", "", "test"),
])
def test_exclusion_reason_excludes(symbol, name, sector, industry, expected):
    assert U.exclusion_reason(symbol, name, sector, industry) == expected


@pytest.mark.parametrize("symbol,name,sector,industry", [
    ("INHD", "Inno Holdings Inc. Common Stock", "Industrials", "Steel/Iron Ore"),
    ("UNTY", "Unity Bancorp Inc. Common Stock", "Finance", "Major Banks"),          # "Unit" substring
    ("CCS", "Century Communities Inc. Common Stock", "", "Homebuilding"),           # "Unit" substring
    ("BRIG", "Brightview Holdings Common Stock", "", ""),                           # "Right" substring
    ("MYND", "Mynd.ai Inc. American Depositary Shares", "", ""),                    # ADRs are kept
    ("CAPL", "CrossAmerica Partners LP Common Units representing limited partner interests", "", ""),
    ("OZ", "Belpointe PREP LLC Class A Units", "Finance", "Real Estate"),
    ("PRT", "PermRock Royalty Trust Units of Beneficial Interest", "Energy", "Oil & Gas Production"),
    ("UE", "Urban Edge Properties Common Shares of Beneficial Interest", "Real Estate", "REITs"),
    ("BRK/A", "Berkshire Hathaway Inc.", "Finance", ""),
    ("AAC", "Ares Acquisition Corporation III Class A Ordinary Shares", "Industrials", ""),
    ("SOUNR", "Soundhound Robotics Common Stock", "", ""),                          # 5-letter R, name doesn't confirm
])
def test_exclusion_reason_keeps_common_equity(symbol, name, sector, industry):
    assert U.exclusion_reason(symbol, name, sector, industry) is None


# ── parsing ──────────────────────────────────────────────────────────────
def test_parse_rows_types_and_values():
    p = U.parse_rows(SAMPLE).set_index("symbol")
    inhd = p.loc["INHD"]
    assert inhd["price"] == pytest.approx(2.99)
    assert inhd["pct_change"] == pytest.approx(-0.03548)
    assert inhd["volume"] == 28703.0
    assert inhd["market_cap"] == pytest.approx(7536537.0)
    assert inhd["ipo_year"] == 2023.0
    assert inhd["country"] == "United States"
    assert not bool(inhd["asia"])
    assert bool(p.loc["MYND", "asia"])                          # China ∈ ASIA_COUNTRIES
    assert math.isnan(p.loc["MYND", "ipo_year"])
    assert math.isnan(p.loc["HVT-A", "market_cap"])            # "" → NaN
    assert math.isnan(p.loc["TLACU", "market_cap"])            # "0.00" → NaN
    assert math.isnan(p.loc["NOPX", "price"])
    assert "BRK-B" in p.index and "ECC" in p.index             # canonical + stripped
    assert p.loc["ABR^D", "reason"] == "preferred"
    assert p.loc["INHD", "reason"] is None


def test_filter_universe_contract_shape_and_exclusions():
    df = U.filter_universe(U.parse_rows(SAMPLE))
    assert list(df.columns) == U.COLUMNS
    syms = set(df["symbol"])
    assert {"INHD", "HVT-A", "UE", "CAPL", "UNTY", "CCS", "MYND"} <= syms
    for gone in ("BRK-B", "BNCWW", "IBACR", "TLACU", "TWOD", "NEEX", "ACV", "ECC", "PPLC",
                 "TINY", "BIGC", "NOPX", "ZVZZT", "PSNYW", "MNYWW"):
        assert gone not in syms, gone
    assert not any("^" in s or "/" in s for s in syms)
    assert df["symbol"].is_unique and df["symbol"].is_monotonic_increasing
    for c in ("price", "pct_change", "volume", "market_cap", "ipo_year"):
        assert df[c].dtype == np.float64
    assert df["asia"].dtype == bool
    assert (df["price"] >= config.MIN_PRICE).all()
    assert not (df["market_cap"] > config.MAX_MARKET_CAP).any()
    assert math.isnan(df.set_index("symbol").loc["HVT-A", "market_cap"])    # unknown cap kept
    ex = df.attrs["excluded"]
    assert ex["price_below_min"] == 1 and ex["market_cap_above_max"] == 2 and ex["no_price"] == 1
    assert ex["preferred"] == 2 and ex["fund"] == 2 and ex["debt"] == 2


def test_filter_universe_dedupes_canonical_symbols():
    rows = [_row("ABC.A", "Alpha Class A Common Stock"), _row("ABC/A", "Alpha Class A Common Stock")]
    df = U.filter_universe(U.parse_rows(rows))
    assert df["symbol"].tolist() == ["ABC-A"]
    assert df.attrs["excluded"]["duplicate_symbol"] == 1


# ── load_universe (network mocked) ───────────────────────────────────────
def _payload(rows):
    return {"data": {"asOf": None, "headers": {}, "rows": rows}, "message": None,
            "status": {"rCode": 200}}


def test_load_universe_fetches_caches_and_reports(tmp_cache, monkeypatch):
    monkeypatch.setattr(U, "_MIN_ROWS", 1)
    calls = []

    def fake_get_json(url, **kw):
        calls.append(url)
        assert kw["headers"] == net.NASDAQ_HEADERS
        return _payload(SAMPLE)

    monkeypatch.setattr(net, "get_json", fake_get_json)
    df = U.load_universe()
    assert len(calls) == 1 and calls[0] == U.SCREENER_URL
    assert "INHD" in set(df["symbol"])
    assert df.attrs["listed"] == len(SAMPLE)
    assert df.attrs["stale"] is False and df.attrs["fetched_at"]
    assert df.attrs["source_url"] == U.SCREENER_URL
    st = net.STATUS[U.SOURCE_NAME]
    assert st["ok"] is True and "listed" in st["detail"]

    df2 = U.load_universe()            # served from the disk cache
    assert len(calls) == 1
    pd.testing.assert_frame_equal(df, df2)


def test_load_universe_rejects_tiny_payload_and_uses_stale_cache(tmp_cache, monkeypatch):
    monkeypatch.setattr(U, "_MIN_ROWS", 1)
    monkeypatch.setattr(net, "get_json", lambda url, **kw: _payload(SAMPLE))
    U.load_universe()                  # populate cache

    monkeypatch.setattr(U, "_MIN_ROWS", 1000)   # now the (same, small) live payload is "broken"
    monkeypatch.setattr(net, "get_json", lambda url, **kw: _payload(SAMPLE[:3]))
    df = U.load_universe(max_age_s=0)   # force a live attempt
    assert df.attrs["stale"] is True
    assert "INHD" in set(df["symbol"])
    st = net.STATUS[U.SOURCE_NAME]
    assert st["ok"] is False and "cached copy" in st["detail"]


def test_load_universe_total_failure_returns_empty_contract_frame(tmp_cache, monkeypatch):
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    df = U.load_universe()
    assert df.empty and list(df.columns) == U.COLUMNS
    assert net.STATUS[U.SOURCE_NAME]["ok"] is False


# ── live ─────────────────────────────────────────────────────────────────
@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="set GRAVITY_LIVE=1 to hit the real Nasdaq screener")
def test_live_load_universe(tmp_cache):
    df = U.load_universe()
    assert df.attrs["listed"] > 5000
    assert 1500 < len(df) < df.attrs["listed"]
    assert df["symbol"].is_unique
    assert (df["price"] >= config.MIN_PRICE).all()
    assert not (df["market_cap"] > config.MAX_MARKET_CAP).any()
    assert not df["symbol"].str.contains(r"[\^/]").any()
    assert net.STATUS[U.SOURCE_NAME]["ok"] is True
    print(f"\nlive universe: listed={df.attrs['listed']} common={df.attrs['common']} "
          f"universe={len(df)} asia={int(df['asia'].sum())} excluded={df.attrs['excluded']}")
    if "INHD" in set(df["symbol"]):
        print("INHD row:", df.set_index("symbol").loc["INHD"].to_dict())
