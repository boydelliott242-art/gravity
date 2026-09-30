"""Tests for gravity.sources.street (CONTRACTS.md §5).

Unit tests replace ``net.get_json`` with canned Nasdaq payloads shaped like
the real API answers (checked 2026-09-30). Live tests (``GRAVITY_LIVE=1``)
make a handful of Nasdaq calls into a temporary cache; Danelfin is only
checked for "no key → no call".
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.sources import street as T  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="set GRAVITY_LIVE=1 for live network tests")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    for k in ("Nasdaq analyst ratings", "Nasdaq earnings calendar", "Nasdaq movers", "Danelfin API"):
        net.STATUS.pop(k, None)
    monkeypatch.setattr(T, "_ANALYST_STATS", {"ok": 0, "fail": 0})
    return tmp_path / "cache"


def _boom(*a, **k):
    raise AssertionError("network touched")


# ═════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("raw,expected", [
    ("$1.16", 1.16), ("($0.14)", -0.14), ("-$0.30", -0.30), ("$1,234.50", 1234.5),
    ("N/A", None), ("", None), (None, None), ("$116,460,503,000", 116460503000.0),
])
def test_money(raw, expected):
    assert T._money(raw) == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("raw,expected", [
    ("09/29/2026", "2026-09-29"), ("2026-09-01T00:00:00", "2026-09-01"), ("9/1/26", "2026-09-01"),
    ("junk", None), ("", None), (None, None),
])
def test_iso_date(raw, expected):
    assert T._iso_date(raw) == expected


# ═════════════════════════════════════════════════════════════════════════
# analyst
# ═════════════════════════════════════════════════════════════════════════
RATINGS_TSLA = {"data": {"symbol": "TSLA", "meanRatingType": "Hold",
                         "ratingsSummary": "Based on 32 analysts giving stock ratings to Tesla in the past 3 months",
                         "upgradesDowngrades": []}, "status": {"rCode": 200}}
TARGET_TSLA = {"data": {"symbol": "TSLA", "consensusOverview": {
    "lowPriceTarget": 24.86, "highPriceTarget": 505.0, "priceTarget": 391.4, "buy": 12, "sell": 2, "hold": 12},
    "historicalConsensus": [{"z": {"date": "08/01/2026", "consensus": 380.0}},
                            {"z": {"date": "09/01/2026", "consensus": 391.4}}]}, "status": {"rCode": 200}}
RATINGS_EMPTY = {"data": {"symbol": "INHD", "meanRatingType": "", "ratingsSummary": "", "upgradesDowngrades": None},
                 "status": {"rCode": 200}}
TARGET_EMPTY = {"data": None, "status": {"rCode": 200}}


def test_parse_analyst_covered():
    a = T.parse_analyst("TSLA", RATINGS_TSLA, TARGET_TSLA)
    assert a["mean_rating"] == "Hold" and a["n_analysts"] == 32
    assert a["price_target"] == 391.4 and a["price_target_date"] == "2026-09-01"
    assert (a["buy"], a["hold"], a["sell"]) == (12, 12, 2)
    assert a["covered"] is True and a["changes"] == []
    assert a["source_url"] == "https://www.nasdaq.com/market-activity/stocks/tsla/analyst-research"
    for k in ("mean_rating", "n_analysts", "changes", "price_target", "source_url"):
        assert k in a


def test_parse_analyst_uncovered_microcap_invents_nothing():
    a = T.parse_analyst("INHD", RATINGS_EMPTY, TARGET_EMPTY)
    assert a["covered"] is False
    for k in ("mean_rating", "n_analysts", "price_target", "price_target_low", "price_target_high",
              "price_target_date", "buy", "hold", "sell"):
        assert a[k] is None, k
    assert a["changes"] == []
    assert T.parse_analyst("INHD", None, None) is None


def test_parse_analyst_counts_from_buckets_when_summary_missing():
    r = {"data": {"meanRatingType": "Buy", "ratingsSummary": None}}
    t = {"data": {"consensusOverview": {"buy": 3, "hold": 1, "sell": 0, "priceTarget": "$5.00"}}}
    a = T.parse_analyst("ABC", r, t)
    assert a["n_analysts"] == 4 and a["price_target"] == 5.0 and a["price_target_date"] is None


def test_parse_rating_changes_tolerant_and_sorted():
    rows = [{"date": "08/01/2026", "brokerName": "B. Riley", "actionType": "Downgrade",
             "fromRating": "Buy", "toRating": "Neutral"},
            {"dateOfChange": "2026-09-15", "firm": "Maxim", "action": "Upgrade", "from": "Hold", "to": "Buy"},
            "junk"]
    ch = T.parse_rating_changes(rows)
    assert [c["date"] for c in ch] == ["2026-09-15", "2026-08-01"]
    assert ch[1] == {"date": "2026-08-01", "firm": "B. Riley", "action": "Downgrade", "from": "Buy", "to": "Neutral"}
    assert T.parse_rating_changes(None) == []


def test_analyst_fetch_cache_and_partial_not_cached(tmp_cache, monkeypatch):
    seen = []

    def fake(url, **kw):
        seen.append(url)
        assert kw["headers"] == net.NASDAQ_HEADERS
        return RATINGS_TSLA if url.endswith("/ratings") else TARGET_TSLA

    monkeypatch.setattr(net, "get_json", fake)
    a = T.analyst("TSLA")
    assert seen == ["https://api.nasdaq.com/api/analyst/TSLA/ratings",
                    "https://api.nasdaq.com/api/analyst/TSLA/targetprice"]
    assert T.analyst("TSLA") == a and len(seen) == 2                   # cached
    # partial answer (target endpoint down) is returned but not cached
    monkeypatch.setattr(net, "get_json", lambda url, **kw: RATINGS_TSLA if url.endswith("/ratings") else None)
    p = T.analyst("BRK-B")
    assert p["mean_rating"] == "Hold" and p["price_target"] is None
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    assert T.analyst("BRK-B") is None                                   # nothing answered → None
    st = net.STATUS["Nasdaq analyst ratings"]
    assert st["ok"] is True and st["detail"] == "3/4 symbols answered"


def test_analyst_uses_dotted_symbol(tmp_cache, monkeypatch):
    seen = []
    monkeypatch.setattr(net, "get_json", lambda url, **kw: seen.append(url) or None)
    T.analyst("BRK-B")
    assert seen[0] == "https://api.nasdaq.com/api/analyst/BRK.B/ratings"


# ═════════════════════════════════════════════════════════════════════════
# earnings calendar
# ═════════════════════════════════════════════════════════════════════════
EARNINGS = {"data": {"asOf": "Thu, Oct 1, 2026", "headers": {}, "rows": [
    {"lastYearRptDt": "10/02/2025", "lastYearEPS": "$0.33", "time": "time-pre-market", "symbol": "ACN",
     "name": "Accenture plc", "marketCap": "$116,460,503,000", "fiscalQuarterEnding": "Aug/2026",
     "epsForecast": "$3.19", "noOfEsts": "7"},
    {"time": "time-after-hours", "symbol": "NKE", "name": "Nike, Inc.", "marketCap": "$53,984,518,000",
     "fiscalQuarterEnding": "Aug/2026", "epsForecast": "$0.43", "noOfEsts": "12"},
    {"time": "time-not-supplied", "symbol": "TINY", "name": "Tiny Co", "marketCap": "N/A",
     "fiscalQuarterEnding": "Jun/2026", "epsForecast": "($0.14)", "noOfEsts": "N/A"},
    {"time": "time-pre-market", "symbol": "BRK.B", "name": "Berkshire", "marketCap": "", "epsForecast": "",
     "noOfEsts": "1"},
    {"symbol": "", "name": "blank"},
]}, "status": {"rCode": 200}}


def test_parse_earnings_rows():
    rows = T.parse_earnings(EARNINGS, date(2026, 10, 1))
    assert [r["symbol"] for r in rows] == ["ACN", "NKE", "TINY", "BRK-B"]
    acn = rows[0]
    assert acn == {"symbol": "ACN", "name": "Accenture plc", "date": "2026-10-01", "time": "pre-market",
                   "eps_forecast": 3.19, "n_ests": 7, "market_cap": 116460503000.0, "fiscal_quarter": "Aug/2026"}
    assert rows[1]["time"] == "after-hours"
    tiny = rows[2]
    assert tiny["time"] is None and tiny["eps_forecast"] == pytest.approx(-0.14)
    assert tiny["n_ests"] is None and tiny["market_cap"] is None
    assert rows[3]["eps_forecast"] is None
    for r in rows:
        assert {"symbol", "time", "eps_forecast", "n_ests", "market_cap"} <= set(r)


def test_parse_earnings_empty_vs_unrecognised():
    assert T.parse_earnings({"data": {"rows": None}}) == []
    assert T.parse_earnings({"data": None}) == []
    assert T.parse_earnings({"message": "x"}) is None
    assert T.parse_earnings(None) is None


def test_earnings_calendar_request_cache_and_failure(tmp_cache, monkeypatch):
    seen = []
    monkeypatch.setattr(net, "get_json", lambda url, **kw: seen.append((url, kw.get("params"))) or EARNINGS)
    rows = T.earnings_calendar(date(2026, 10, 1))
    assert len(rows) == 4
    assert seen == [("https://api.nasdaq.com/api/calendar/earnings", {"date": "2026-10-01"})]
    T.earnings_calendar(date(2026, 10, 1))
    assert len(seen) == 1
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    assert T.earnings_calendar(date(2026, 10, 2)) == []
    assert net.STATUS["Nasdaq earnings calendar"]["ok"] is False


# ═════════════════════════════════════════════════════════════════════════
# movers
# ═════════════════════════════════════════════════════════════════════════
def _block(rows, asof="Data as of Sep 29, 2026 4:15 PM ET"):
    return {"dataAsOf": asof, "table": {"rows": rows}}


MOVERS = {"data": {"STOCKS": {
    "MostAdvanced": _block([{"symbol": "ARBEW", "name": "Arbe Robotics Ltd.", "lastSalePrice": "$0.0582",
                             "lastSaleChange": "+0.0384", "change": "+193.9394%"},
                            {"symbol": "", "name": "blank", "lastSalePrice": "$1"}]),
    "MostDeclined": _block([{"symbol": "NIVF", "name": "NewGenIvf", "lastSalePrice": "$0.072",
                             "lastSaleChange": "-0.2735", "change": "-79.1183%"}]),
    "MostActiveByShareVolume": _block([{"symbol": "SLXN", "name": "Silexion", "lastSalePrice": "$0.3184",
                                        "lastSaleChange": "+0.07", "change": "253,089,230"},
                                       {"symbol": "NOPX", "name": "No Price", "lastSalePrice": "N/A",
                                        "lastSaleChange": "", "change": "1"}]),
}}}


def test_parse_movers_contract_shape():
    m = T.parse_movers(MOVERS, "preMarket")
    for k in ("premarket_gainers", "premarket_losers", "most_active"):
        assert isinstance(m[k], list)
        for it in m[k]:
            assert {"symbol", "price", "change_pct"} <= set(it)
    g = m["premarket_gainers"]
    assert g == [{"symbol": "ARBEW", "name": "Arbe Robotics Ltd.", "price": 0.0582, "change_pct": pytest.approx(1.939394)}]
    assert m["premarket_losers"][0]["change_pct"] == pytest.approx(-0.791183)
    act = m["most_active"]
    assert [a["symbol"] for a in act] == ["SLXN"]                     # no price → dropped
    assert act[0]["volume"] == 253089230
    assert act[0]["change_pct"] == pytest.approx(0.07 / (0.3184 - 0.07), rel=1e-4)
    assert m["asof"] == "2026-09-29T16:15:00-04:00" and m["board"] == "post-close"
    assert m["session"] == "preMarket"
    assert T.parse_movers({"data": None}, "x") is None


@pytest.mark.parametrize("asof,label", [
    ("2026-09-30T08:05:00-04:00", "pre-market"), ("2026-09-30T11:00:00-04:00", "intraday"),
    ("2026-09-29T16:15:00-04:00", "post-close"), (None, None), ("bad", None),
])
def test_board_label(asof, label):
    assert T.board_label(asof) == label


def test_movers_request_and_failure_shape(tmp_cache, monkeypatch):
    seen = []
    monkeypatch.setattr(T, "market_phase", lambda: "pre-market")
    monkeypatch.setattr(net, "get_json", lambda url, **kw: seen.append(kw["params"]) or MOVERS)
    m = T.movers()
    assert seen == [{"assetclass": "stocks", "exchangestatus": "preMarket", "limit": 25}]
    assert m["premarket_gainers"][0]["symbol"] == "ARBEW"
    assert "served post-close board" in net.STATUS["Nasdaq movers"]["detail"]
    monkeypatch.setattr(T, "market_phase", lambda: "open")
    monkeypatch.setattr(net, "get_json", lambda url, **kw: None)
    m = T.movers()
    assert m["premarket_gainers"] == [] and m["most_active"] == [] and m["asof"] is None
    assert m["session"] == "currentMarket"
    assert net.STATUS["Nasdaq movers"]["ok"] is False


# ═════════════════════════════════════════════════════════════════════════
# Danelfin
# ═════════════════════════════════════════════════════════════════════════
def test_danelfin_without_key_makes_no_network_call(tmp_cache, monkeypatch):
    monkeypatch.setattr(config, "DANELFIN_API_KEY", "")
    monkeypatch.setattr(net, "get", _boom)
    monkeypatch.setattr(net, "get_json", _boom)
    assert T.danelfin(["INHD", "TSLA"]) == {}
    st = net.STATUS["Danelfin API"]
    assert st["ok"] is False and "not set" in st["detail"]


def test_parse_danelfin_latest_date_and_ranges():
    payload = {"2026-09-28": {"aiscore": 5, "technical": 4, "fundamental": 6, "sentiment": 5, "low_risk": 3},
               "2026-09-29": {"aiscore": 3, "technical": 2, "fundamental": 11, "sentiment": "7", "low_risk": None},
               "2026-09-30": None}
    r = T.parse_danelfin("INHD", payload)
    assert r == {"ai_score": 3, "technical": 2, "fundamental": None, "sentiment": 7, "low_risk": None,
                 "date": "2026-09-29", "source_url": "https://danelfin.com/stock/INHD"}
    nested = {"2026-09-29": {"TSLA": {"aiscore": 8, "technical": 7, "fundamental": 6, "sentiment": 8, "low_risk": 5}}}
    assert T.parse_danelfin("TSLA", nested)["ai_score"] == 8
    assert T.parse_danelfin("X", {"2026-09-29": {"aiscore": None}}) is None
    assert T.parse_danelfin("X", {"error": "unauthorized"}) is None
    assert T.parse_danelfin("X", None) is None


def test_danelfin_with_key_probe_stops_on_dead_key(tmp_cache, monkeypatch):
    monkeypatch.setattr(config, "DANELFIN_API_KEY", "k")
    calls = []

    def fake(url, **kw):
        calls.append((url, kw["params"], kw["headers"]["x-api-key"]))
        return None

    monkeypatch.setattr(net, "get_json", fake)
    out = T.danelfin(["A", "B", "C", "D", "E"])
    assert out == {} and len(calls) == T.DANELFIN_PROBE
    assert calls[0] == (T.DANELFIN_API, {"ticker": "A"}, "k")
    assert "check key/quota" in net.STATUS["Danelfin API"]["detail"]


def test_danelfin_with_key_caches_hits(tmp_cache, monkeypatch):
    monkeypatch.setattr(config, "DANELFIN_API_KEY", "k")
    good = {"2026-09-29": {"aiscore": 4, "technical": 3, "fundamental": 5, "sentiment": 4, "low_risk": 2}}
    calls = []
    monkeypatch.setattr(net, "get_json", lambda url, **kw: calls.append(kw["params"]["ticker"]) or good)
    out = T.danelfin(["INHD", "TSLA", "INHD"], max_calls=5)
    assert set(out) == {"INHD", "TSLA"} and calls == ["INHD", "TSLA"]
    T.danelfin(["INHD"])
    assert calls == ["INHD", "TSLA"]
    out = T.danelfin(["X1", "X2", "X3"], max_calls=1)
    assert set(out) == {"X1"} and "per-run cap" in net.STATUS["Danelfin API"]["detail"]


# ═════════════════════════════════════════════════════════════════════════
# deep links
# ═════════════════════════════════════════════════════════════════════════
CONTRACT_LINKS = ["Zacks", "Danelfin", "Bloomberg", "WSJ", "Finviz", "TradingView", "Stocktwits", "Yahoo",
                  "Nasdaq", "SEC EDGAR", "Fintel", "iBorrowDesk", "TipRanks", "MarketBeat"]


def test_deep_links_order_and_well_formed_urls(monkeypatch):
    monkeypatch.setattr(net, "get", _boom)
    links = T.deep_links("INHD", 1961847)
    assert list(links)[: len(CONTRACT_LINKS)] == CONTRACT_LINKS
    assert "Trade halts" in links
    for label, url in links.items():
        u = urlparse(url)
        assert u.scheme == "https" and u.netloc and " " not in url, (label, url)
    assert links["SEC EDGAR"] == ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                                  "&CIK=1961847&type=&dateb=&owner=include&count=40")
    assert links["Zacks"] == "https://www.zacks.com/stock/quote/INHD"
    assert links["Bloomberg"] == "https://www.bloomberg.com/quote/INHD:US"
    assert links["iBorrowDesk"] == "https://www.iborrowdesk.com/report/INHD"
    assert links["Fintel"] == "https://fintel.io/ss/us/inhd"
    assert links["Yahoo"] == "https://finance.yahoo.com/quote/INHD/"


def test_deep_links_share_classes_exchange_and_no_cik():
    links = T.deep_links("brk-b", None, exchange="NYSE")
    assert links["Yahoo"] == "https://finance.yahoo.com/quote/BRK-B/"
    assert links["Zacks"] == "https://www.zacks.com/stock/quote/BRK.B"
    assert links["Bloomberg"] == "https://www.bloomberg.com/quote/BRK/B:US"
    assert links["TradingView"] == "https://www.tradingview.com/symbols/NYSE-BRK.B/"
    assert links["MarketBeat"] == "https://www.marketbeat.com/stocks/NYSE/BRK-B/"
    assert "CIK=BRK-B" in links["SEC EDGAR"]
    otc = T.deep_links("ABCDF", exchange="OTC")
    assert otc["OTC Markets"] == "https://www.otcmarkets.com/stock/ABCDF/overview"
    assert "OTC Markets" not in T.deep_links("INHD")
    assert T.deep_links("XYZ", exchange="NYSE American")["MarketBeat"].startswith(
        "https://www.marketbeat.com/stocks/NYSEAMERICAN/")


# ═════════════════════════════════════════════════════════════════════════
# Live
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.live
@live
def test_live_analyst(tmp_cache):
    a = T.analyst("TSLA")
    assert a is not None and a["covered"] and a["mean_rating"] and a["n_analysts"]
    b = T.analyst("INHD")
    assert b is None or (b["covered"] is False and b["mean_rating"] is None and b["price_target"] is None)
    print("\nTSLA:", {k: a[k] for k in ("mean_rating", "n_analysts", "price_target", "price_target_date")},
          "| INHD:", None if b is None else {k: b[k] for k in ("covered", "mean_rating", "n_analysts")})


@pytest.mark.live
@live
def test_live_earnings_and_movers(tmp_cache):
    rows = T.earnings_calendar(date(2026, 10, 1))
    assert isinstance(rows, list)
    assert net.STATUS["Nasdaq earnings calendar"]["ok"] is True
    for r in rows:
        assert r["symbol"] and r["time"] in ("pre-market", "after-hours", None)
    m = T.movers()
    assert m["premarket_gainers"] and m["premarket_losers"] and m["most_active"]
    assert all(g["change_pct"] > 0 for g in m["premarket_gainers"])
    assert all(x["change_pct"] < 0 for x in m["premarket_losers"])
    print("\nearnings 2026-10-01:", len(rows), [r["symbol"] for r in rows[:6]],
          "| movers:", m["session"], m["board"], m["asof"], m["premarket_gainers"][0])


@pytest.mark.live
@live
def test_live_danelfin_off_and_links(tmp_cache, monkeypatch):
    monkeypatch.setattr(config, "DANELFIN_API_KEY", "")
    monkeypatch.setattr(net, "get_json", _boom)
    assert T.danelfin(["INHD"]) == {}
    links = T.deep_links("INHD", 1961847)
    assert all(urlparse(u).scheme == "https" for u in links.values())
