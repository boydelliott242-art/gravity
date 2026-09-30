"""Tests for gravity.sources.sec (CONTRACTS.md §3).

Unit tests replace ``net.get`` with a URL router returning canned EDGAR
payloads (tickers map, submissions, companyfacts, efts full-text search,
the getcurrent Atom feed, filing documents and index pages), so the whole
stack — shared throttle, circuit breaker, caching, parsing — runs offline.

Live tests (``GRAVITY_LIVE=1`` and ``SEC_USER_AGENT`` set) hit EDGAR for
INHD and a 3-day full-text window. They pace themselves at 2 req/s (SEC's
fair-access cap is 10/s per client and a background job may be using 6)
and use the normal disk cache, exactly as the pipeline does.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net, util  # noqa: E402
from gravity.sources import sec  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"


# ═════════════════════════════════════════════════════════════════════════
# Fakes
# ═════════════════════════════════════════════════════════════════════════
class FakeResp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.content = self.text.encode("utf-8")
        self.status_code = status

    def json(self) -> Any:
        return json.loads(self.text)


class Router:
    """Stand-in for ``net.get``: first matching route answers; no match → None."""

    def __init__(self) -> None:
        self.routes: List[Any] = []
        self.calls: List[Any] = []

    def add(self, match: Any, body: Any) -> "Router":
        self.routes.append((match, body))
        return self

    def __call__(self, url: str, *, headers: Optional[Dict[str, str]] = None,
                 params: Optional[Dict[str, Any]] = None, **kw: Any) -> Optional[FakeResp]:
        params = dict(params or {})
        self.calls.append((url, params, headers))
        for match, body in self.routes:
            hit = match(url, params) if callable(match) else (url == match)
            if hit:
                b = body(url, params) if callable(body) else body
                return None if b is None else FakeResp(b)
        return None

    def urls(self) -> List[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated SEC state: temp cache, UA set, no pacing, fresh memo/breaker."""
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(config, "SEC_USER_AGENT", "Gravity Test test@example.com")
    monkeypatch.setattr(sec, "_SEC_GAP_S", 0.0)
    monkeypatch.setattr(sec, "_map_memo", {"at": 0.0, "by_sym": None, "by_cik": None})
    monkeypatch.setattr(sec, "_breaker", sec._Breaker())
    monkeypatch.setattr(sec, "_parts", {})
    net.STATUS.pop(sec.STATUS_NAME, None)
    router = Router()
    monkeypatch.setattr(net, "get", router)
    return router


TICKERS = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [1961847, "INNO HOLDINGS INC.", "INHDW", "Nasdaq"],   # derivative listed first on purpose
        [1961847, "INNO HOLDINGS INC.", "INHD", "Nasdaq"],
        [1067983, "BERKSHIRE HATHAWAY INC", "BRK.B", "NYSE"],
        [1786286, "Draganfly Inc.", "DPRO", "Nasdaq"],
        [1650101, "Autonomix Medical", "ATXG", "Nasdaq"],
        [1114483, "Integer Holdings Corp", "ITGR", "NYSE"],
        [1794338, "Insurance Co", "IGIC", "NYSE"],
        [1400118, "Sangamo", "SGMT", "Nasdaq"],
        [1099160, "Beasley", "BBGI", "Nasdaq"],
        [2000001, "Blank Ticker Co", "", "OTC"],
        [2000002, "Bad Row"],                                # short row → skipped
    ],
}


def tickers_route(router: Router, payload: Any = TICKERS) -> None:
    router.add(sec.TICKERS_URL, payload)


def _sub(rows: List[tuple], files: Optional[list] = None, **extra: Any) -> Dict[str, Any]:
    cols: Dict[str, list] = {k: [] for k in ("accessionNumber", "filingDate", "acceptanceDateTime",
                                             "form", "items", "primaryDocument")}
    for acc, d, acc_dt, form, items, doc in rows:
        cols["accessionNumber"].append(acc)
        cols["filingDate"].append(d)
        cols["acceptanceDateTime"].append(acc_dt)
        cols["form"].append(form)
        cols["items"].append(items)
        cols["primaryDocument"].append(doc)
    base = {
        "cik": "1961847", "name": "INNO HOLDINGS INC.", "entityType": "operating",
        "sic": "5990", "sicDescription": "Retail-Retail Stores, NEC",
        "tickers": ["INHD"], "exchanges": ["Nasdaq"], "fiscalYearEnd": "0930",
        "stateOfIncorporation": "TX", "stateOfIncorporationDescription": "TX",
        "category": "Non-accelerated filer<br>Smaller reporting company<br>Emerging growth company",
        "addresses": {
            "mailing": {"street1": "Room 1, 12/F", "city": "TSIM SHA TSUI", "stateOrCountry": "K3",
                        "stateOrCountryDescription": "HONG KONG", "isForeignLocation": 1, "countryCode": "K3"},
            "business": {"street1": "Room 1, 12/F", "city": "TSIM SHA TSUI", "stateOrCountry": "K3",
                         "stateOrCountryDescription": "HONG KONG", "isForeignLocation": 1, "countryCode": "K3"},
        },
        "formerNames": [],
        "filings": {"recent": cols, "files": files or []},
    }
    base.update(extra)
    return base


# INHD-shaped filing index (real accession numbers / items / times, 2025-12..2026-08).
INHD_ROWS = [
    ("0001493152-26-038354", "2026-08-14", "2026-08-14T20:10:29.000Z", "NT 10-Q", "", "formnt10-q.htm"),
    ("0001493152-26-024623", "2026-05-20", "2026-05-20T20:30:22.000Z", "8-K", "1.01,7.01,8.01,9.01", "form8-k.htm"),
    ("0001493152-26-024367", "2026-05-19", "2026-05-19T13:57:14.000Z", "424B5", "", "form424b5.htm"),
    ("0001493152-26-024367", "2026-05-19", "2026-05-19T13:57:14.000Z", "424B5", "", "form424b5.htm"),  # dup
    ("0001493152-26-021119", "2026-05-04", "2026-05-04T20:30:38.000Z", "8-K", "5.03,7.01,9.01", "form8-k.htm"),
    ("0001234567-26-000010", "2026-04-10", "2026-04-10T21:00:00.000Z", "4", "", "xslF345X05/wk-form4.xml"),
    ("0001234567-26-000011", "2026-04-09", "2026-04-09T21:00:00.000Z", "144", "", "xsl144X01/primary_doc.xml"),
    ("0001234567-26-000012", "2026-04-08", "2026-04-08T21:00:00.000Z", "S-1", "", "forms-1.htm"),
    ("0001234567-26-000013", "2026-04-07", "2026-04-07T21:00:00.000Z", "EFFECT", "", ""),
    ("0001234567-26-000014", "2026-04-06", "2026-04-06T21:00:00.000Z", "424B3", "", "form424b3.htm"),
    ("0001234567-26-000015", "2026-04-05", "2026-04-05T21:00:00.000Z", "8-K", "3.01,9.01", "form8-k.htm"),
    ("0001234567-26-000016", "2026-04-04", "2026-04-04T21:00:00.000Z", "8-K", "3.02", "form8-k.htm"),
    ("0001234567-26-000017", "2026-04-03", "2026-04-03T21:00:00.000Z", "10-K", "", "form10-k.htm"),
    ("0001234567-26-000018", "2026-04-02", "", "SC 13G", "", "sc13g.htm"),
    ("0001234567-26-000019", "2026-04-01", "2026-04-01T21:00:00.000Z", "", "", "x.htm"),       # no form → skipped
    ("0001493152-25-028854", "2025-12-22", "2025-12-22T22:30:40.000Z", "8-K", "5.03,9.01", "form8-k.htm"),
]


# ═════════════════════════════════════════════════════════════════════════
# Taxonomy
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("form,items,tags,expected", [
    ("424B5", [], [], "offering"),
    ("424B4", [], [], "offering"),
    ("424B1", [], [], "offering"),
    ("424B7", [], [], "offering"),
    ("424B5", [], ["atm"], "atm"),
    ("424B5", [], ["atm", "registered_direct"], "atm"),
    ("S-1MEF", [], [], "offering"),
    ("F-1MEF", [], [], "offering"),
    ("424B3", [], [], "resale"),
    ("S-1", [], [], "registration"),
    ("S-1/A", [], [], "registration"),
    ("F-1/A", [], [], "registration"),
    ("S-3", [], [], "registration"),
    ("F-3/A", [], [], "registration"),
    ("S-3ASR", [], [], "registration"),
    ("S-1", [], ["selling_stockholders"], "resale"),
    ("F-1", [], ["equity_line"], "toxic_financing"),
    ("EFFECT", [], [], "effective"),
    ("8-K", ["3.02"], [], "unregistered_sale"),
    ("8-K", ["3.01", "9.01"], [], "delisting_notice"),
    ("8-K", ["8.01"], ["bid_price_deficiency"], "delisting_notice"),
    ("6-K", [], ["delisting"], "delisting_notice"),
    ("8-K", ["5.03", "9.01"], ["reverse_split"], "reverse_split"),
    ("8-K", ["5.03", "9.01"], [], "charter_amendment"),        # item alone is not a reverse split
    ("8-K/A", ["5.03"], [], "charter_amendment"),
    ("6-K", [], ["reverse_split"], "reverse_split"),
    ("8-K", ["8.01", "9.01"], ["reverse_split"], "reverse_split"),   # press-release announcement
    ("8-K", ["5.07"], ["reverse_split"], "reverse_split"),           # stockholders approved it
    ("8-K", ["1.01", "3.02", "9.01"], ["reverse_split"], "unregistered_sale"),  # boilerplate in an agreement
    ("8-K", ["1.01", "9.01"], ["reverse_split"], "material_agreement"),
    ("8-K", ["1.01", "9.01"], [], "material_agreement"),
    ("8-K", ["1.01", "3.02"], ["registered_direct"], "offering"),
    ("8-K", ["8.01"], ["public_offering_priced"], "offering"),
    ("6-K", [], ["securities_purchase_agreement"], "offering"),
    ("8-K", ["1.01"], ["atm"], "atm"),
    ("8-K", ["1.01"], ["equity_line"], "toxic_financing"),
    ("6-K", [], ["convertible_note"], "toxic_financing"),
    ("8-K", ["1.01"], ["warrant_inducement"], "toxic_financing"),
    ("10-K", [], ["going_concern"], "going_concern"),
    ("10-Q/A", [], ["going_concern"], "going_concern"),
    ("20-F", [], ["going_concern"], "going_concern"),
    ("10-Q", [], ["registered_direct"], "other"),      # a 10-Q mentioning an old deal stays "other"
    ("10-K", [], [], "other"),
    ("NT 10-K", [], [], "late_filing"),
    ("NT 10-Q", [], [], "late_filing"),
    ("NT 20-F", [], [], "late_filing"),
    ("144", [], [], "insider_sale_notice"),
    ("4", [], [], "insider"),
    ("3/A", [], [], "insider"),
    ("SC 13G", [], [], "other"),
    ("", [], [], "other"),
    ("8-K", ["7.01"], [], "other"),
])
def test_categorize_taxonomy(form, items, tags, expected):
    got = sec.categorize(form, items, tags)
    assert got == expected
    assert got in sec.CATEGORIES


def test_category_from_tags_is_categorize_with_tags_first():
    assert sec.category_from_tags(["atm"], "424B5") == "atm"
    assert sec.category_from_tags([], "424B5") == "offering"
    assert sec.category_from_tags(["reverse_split"], "8-K", ["5.03"]) == "reverse_split"
    assert sec.category_from_tags(["reverse_split"], "8-K", ["1.01", "3.02"]) == "unregistered_sale"


def test_every_query_avoids_parentheses_and_every_tag_has_a_regex():
    """efts.sec.gov answers 0 hits for grouped queries (verified live)."""
    for tag, (q, forms, rx) in sec.PHRASES.items():
        for one in sec._tag_queries(tag):
            assert "(" not in one.replace('"5550(a)(2)"', ""), (tag, one)
            assert one.strip()
        assert forms and rx
    assert len(sec._tag_queries("public_offering_priced")) == 2
    contract_tags = {"registered_direct", "public_offering_priced", "atm", "reverse_split",
                     "bid_price_deficiency", "delisting", "going_concern", "equity_line",
                     "convertible_note", "warrant_inducement", "securities_purchase_agreement"}
    assert contract_tags <= set(sec.FULLTEXT_TAGS)


# ═════════════════════════════════════════════════════════════════════════
# Text classification
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("html_text,expected", [
    ("<p>On September 29, 2026 the Company entered into a Securities Purchase Agreement for a "
     "registered direct offering of 1,000,000 shares.</p>", {"registered_direct", "securities_purchase_agreement"}),
    ("<p>XYZ Announces Pricing of $5.0 Million Underwritten Public Offering</p>", {"public_offering_priced"}),
    ("<p>the public offering was priced at $1.00 per share</p>", {"public_offering_priced"}),
    ("<p>an &#8220;at the market offering&#8221; as defined in Rule 415(a)(4)</p>", {"atm"}),
    ("<p>At-The-Market Sales Agreement with H.C. Wainwright (the ATM program)</p>", {"atm"}),
    ("<p>effect a 1-for-20 reverse stock split of its common stock</p>", {"reverse_split"}),
    ("<p>a share consolidation of every ten ordinary shares</p>", {"reverse_split"}),
    ("<p>notice from the Nasdaq Listing Qualifications Department regarding the minimum bid "
     "price requirement under Listing Rule 5550(a)(2)</p>", {"delisting", "bid_price_deficiency"}),
    ("<p>substantial doubt about the Company&rsquo;s ability to continue as a going concern</p>", {"going_concern"}),
    ("<p>a Standby Equity Purchase Agreement (the &ldquo;SEPA&rdquo;) with Yorkville</p>", {"equity_line"}),
    ("<p>issued a Senior Secured Convertible Note in the principal amount of $2,000,000</p>", {"convertible_note"}),
    ("<p>entered into warrant inducement letters with holders</p>", {"warrant_inducement"}),
    ("<p>shares offered by the selling stockholders named herein</p>", {"selling_stockholders"}),
    ("<script>var x='registered direct';</script><style>.a{}</style><p>Quarterly results.</p>", set()),
    ("<p>The Company regained compliance and nothing else happened.</p>", set()),
])
def test_classify_text_phrases(html_text, expected):
    assert set(sec.classify_text(sec.html_to_text(html_text))) == expected


def test_html_to_text_normalises():
    t = sec.html_to_text("<div>A&nbsp;B\n\n<b>C</b>&amp;D’s</div>")
    assert t == "a b c &d's"


INDEX_PAGE = """<html><body><p>Document Format Files</p>
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td scope="row">1</td><td scope="row">REPORT</td>
<td scope="row"><a href="/ix?doc=/Archives/edgar/data/1786286/000149315226044983/form6-k.htm">form6-k.htm</a></td>
<td scope="row">6-K</td><td scope="row">16541</td></tr>
<tr class="evenRow"><td scope="row">2</td><td scope="row">GRAPHIC</td>
<td scope="row"><a href="/Archives/edgar/data/1786286/000149315226044983/img1.jpg">img1.jpg</a></td>
<td scope="row">GRAPHIC</td><td scope="row">471328</td></tr>
<tr><td scope="row">3</td><td scope="row">PRESS RELEASE</td>
<td scope="row"><a href="/Archives/edgar/data/1786286/000149315226044983/ex99-1.htm">ex99-1.htm</a></td>
<td scope="row">EX-99.1</td><td scope="row">9000</td></tr>
<tr><td scope="row">4</td><td scope="row">SPA</td>
<td scope="row"><a href="/Archives/edgar/data/1786286/000149315226044983/ex10-1.htm">ex10-1.htm</a></td>
<td scope="row">EX-10.1</td><td scope="row">90000</td></tr>
<tr class="evenRow"><td scope="row">&nbsp;</td><td scope="row">Complete submission text file</td>
<td scope="row"><a href="/Archives/edgar/data/1786286/000149315226044983/0001493152-26-044983.txt">0001493152-26-044983.txt</a></td>
<td scope="row">&nbsp;</td><td scope="row">879264</td></tr>
</table>
<p>Data Files</p><table><tr><td>1</td><td>x</td><td><a href="/Archives/x.xml">x.xml</a></td><td>EX-101.SCH</td></tr></table>
</body></html>"""

BASE = "https://www.sec.gov/Archives/edgar/data/1786286/000149315226044983/"
INDEX_URL = BASE + "0001493152-26-044983-index.htm"


def test_index_documents_primary_and_press_releases_only():
    docs = sec.index_documents(INDEX_PAGE)
    assert docs == [BASE + "form6-k.htm", BASE + "ex99-1.htm"]


def test_index_url_for_document_urls():
    assert sec.index_url_for(BASE + "form6-k.htm") == INDEX_URL
    assert sec.index_url_for(INDEX_URL) is None
    assert sec.index_url_for("https://example.com/x.htm") is None


LONG_424 = "<html>" + ("filler text about the company. " * 400) + \
    "an at-the-market offering pursuant to the sales agreement</html>"
COVER_6K = "<html>Report of Foreign Private Issuer. Exhibit 99.1 Press release.</html>"
PRESS = "<p>Draganfly announces US$6 million registered direct offering under a securities purchase agreement</p>"


def test_text_classify_long_document_reads_only_itself(env):
    url = "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/form424b5.htm"
    env.add(url, LONG_424)
    assert sec.text_classify(url) == ["atm"]
    assert env.urls() == [url]
    assert sec.text_classify(url) == ["atm"]           # cached: no second request
    assert len(env.calls) == 1
    assert env.calls[0][2]["User-Agent"] == config.SEC_USER_AGENT


def test_text_classify_cover_page_pulls_press_release(env):
    env.add(BASE + "form6-k.htm", COVER_6K).add(INDEX_URL, INDEX_PAGE).add(BASE + "ex99-1.htm", PRESS)
    tags = sec.text_classify(BASE + "form6-k.htm")
    assert tags == ["registered_direct", "securities_purchase_agreement"]
    assert env.urls() == [BASE + "form6-k.htm", INDEX_URL, BASE + "ex99-1.htm"]
    assert sec.category_from_tags(tags, "6-K") == "offering"


def test_text_classify_index_url_from_current_feed(env):
    env.add(INDEX_URL, INDEX_PAGE).add(BASE + "form6-k.htm", COVER_6K).add(BASE + "ex99-1.htm", PRESS)
    assert sec.text_classify(INDEX_URL) == ["registered_direct", "securities_purchase_agreement"]
    assert env.urls() == [INDEX_URL, BASE + "form6-k.htm", BASE + "ex99-1.htm"]


def test_text_classify_partial_read_is_returned_but_not_cached(env):
    env.add(INDEX_URL, INDEX_PAGE).add(BASE + "ex99-1.htm", PRESS)    # primary doc fails
    assert sec.text_classify(INDEX_URL) == ["registered_direct", "securities_purchase_agreement"]
    n = len(env.calls)
    sec.text_classify(INDEX_URL)
    assert len(env.calls) > n                                        # not cached → re-fetched


def test_text_classify_unreadable_is_none_and_never_cached(env):
    assert sec.text_classify(BASE + "missing.htm") is None
    assert sec.text_classify("") is None
    env.add(BASE + "missing.htm", LONG_424)
    assert sec.text_classify(BASE + "missing.htm") == ["atm"]


def test_text_classify_can_skip_or_force_exhibits(env):
    env.add(BASE + "form6-k.htm", COVER_6K).add(INDEX_URL, INDEX_PAGE).add(BASE + "ex99-1.htm", PRESS)
    assert sec.text_classify(BASE + "form6-k.htm", with_exhibits=False) == []
    assert env.urls() == [BASE + "form6-k.htm"]


# ═════════════════════════════════════════════════════════════════════════
# Ticker map
# ═════════════════════════════════════════════════════════════════════════
def test_cik_map_canonical_symbols_and_primary_ticker(env):
    tickers_route(env)
    m = sec.cik_map()
    assert m["INHD"] == {"cik": 1961847, "name": "INNO HOLDINGS INC.", "exchange": "Nasdaq"}
    assert "BRK-B" in m and m["BRK-B"]["cik"] == 1067983
    assert "INHDW" in m                                   # listed, but not the CIK's primary
    assert sec.symbol_for_cik(1961847) == "INHD"
    assert sec.symbol_for_cik(1067983) == "BRK-B"
    assert sec.symbol_for_cik(42) is None
    assert all(isinstance(v["cik"], int) for v in m.values())
    # memoised + disk cached: one request only
    sec.cik_map()
    assert env.urls().count(sec.TICKERS_URL) == 1
    assert net.STATUS[sec.STATUS_NAME]["ok"] is True


def test_cik_map_failure_is_empty_and_reported(env):
    assert sec.cik_map() == {}
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is False and "company_tickers" in st["detail"]


# ═════════════════════════════════════════════════════════════════════════
# Submissions → FilingEvents
# ═════════════════════════════════════════════════════════════════════════
def test_events_from_submissions_shapes_and_categories():
    evs = sec.events_from_submissions(_sub(INHD_ROWS), "INHD", 1961847)
    by_acc = {e["accession"]: e for e in evs}
    assert len(evs) == len(INHD_ROWS) - 2                    # duplicate + formless row dropped
    dates = [e["date"] for e in evs]
    assert dates == sorted(dates, reverse=True)                # newest first
    e = by_acc["0001493152-26-024367"]
    assert e == {
        "symbol": "INHD", "cik": 1961847, "date": "2026-05-19",
        "accepted": "2026-05-19T09:57:14-04:00",               # UTC → ET
        "form": "424B5", "items": [], "category": "offering",
        "url": "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/form424b5.htm",
        "text_tags": [], "accession": "0001493152-26-024367", "source": "sec_submissions",
    }
    assert by_acc["0001493152-26-038354"]["category"] == "late_filing"
    assert by_acc["0001493152-26-038354"]["accepted"] == "2026-08-14T16:10:29-04:00"
    assert by_acc["0001493152-26-021119"]["category"] == "charter_amendment"
    assert by_acc["0001493152-26-021119"]["items"] == ["5.03", "7.01", "9.01"]
    dec = by_acc["0001493152-25-028854"]
    assert dec["category"] == "charter_amendment" and dec["accepted"] == "2025-12-22T17:30:40-05:00"
    assert by_acc["0001493152-26-024623"]["category"] == "material_agreement"
    cats = {e["form"]: e["category"] for e in evs}
    assert cats["4"] == "insider" and cats["144"] == "insider_sale_notice"
    assert cats["S-1"] == "registration" and cats["EFFECT"] == "effective"
    assert cats["424B3"] == "resale" and cats["10-K"] == "other" and cats["SC 13G"] == "other"
    assert by_acc["0001234567-26-000015"]["category"] == "delisting_notice"
    assert by_acc["0001234567-26-000016"]["category"] == "unregistered_sale"
    # no primary document → index page URL; no acceptance time → None (not invented)
    assert by_acc["0001234567-26-000013"]["url"].endswith("/000123456726000013/0001234567-26-000013-index.htm")
    assert by_acc["0001234567-26-000018"]["accepted"] is None
    for ev in evs:
        assert ev["category"] in sec.CATEGORIES
        assert ev["url"].startswith("https://www.sec.gov/Archives/edgar/data/1961847/")


def test_events_from_submissions_since_filter_and_ragged_columns():
    sub = _sub(INHD_ROWS)
    since = sec.events_from_submissions(sub, "INHD", 1961847, since="2026-05-01")
    assert min(e["date"] for e in since) >= "2026-05-01"
    assert not any(e["date"] == "2025-12-22" for e in since)
    sub["filings"]["recent"]["items"] = sub["filings"]["recent"]["items"][:2]   # ragged
    del sub["filings"]["recent"]["acceptanceDateTime"]
    evs = sec.events_from_submissions(sub, "INHD", 1961847)
    assert all(e["accepted"] is None for e in evs)
    assert sec.events_from_submissions({}, "X", 1) == []


def test_apply_text_tags_rederives_category():
    evs = sec.events_from_submissions(_sub(INHD_ROWS), "INHD", 1961847)
    tagged = [{"accession": "0001493152-26-021119", "text_tags": ["reverse_split"]},
              {"accession": "0001493152-26-024367", "text_tags": ["atm"]},
              {"accession": "nope", "text_tags": ["atm"]}]
    sec.apply_text_tags(evs, tagged)
    by_acc = {e["accession"]: e for e in evs}
    assert by_acc["0001493152-26-021119"]["category"] == "reverse_split"
    assert by_acc["0001493152-26-024367"]["category"] == "atm"
    assert by_acc["0001493152-26-024367"]["text_tags"] == ["atm"]


def test_merge_events_one_per_filing_with_union_of_tags():
    idx = "https://www.sec.gov/Archives/edgar/data/1786286/000149315226044983/0001493152-26-044983-index.htm"
    doc = "https://www.sec.gov/Archives/edgar/data/1786286/000149315226044983/ex99-1.htm"
    feed = {"symbol": "DPRO", "cik": 1786286, "date": "2026-09-30", "accepted": "2026-09-30T08:01:00-04:00",
            "form": "6-K", "items": [], "category": "other", "url": idx, "text_tags": [],
            "accession": "0001493152-26-044983", "source": "sec_current"}
    ft = dict(feed, accepted=None, url=doc, text_tags=["registered_direct"], category="offering",
              source="sec_fulltext")
    other = dict(feed, accession="0001493152-26-000001", url=idx.replace("044983", "000001"), date="2026-09-29")
    out = sec.merge_events([feed, other], [ft])
    assert [e["accession"] for e in out] == ["0001493152-26-044983", "0001493152-26-000001"]
    m = out[0]
    assert m["text_tags"] == ["registered_direct"] and m["category"] == "offering"
    assert m["url"] == doc and m["accepted"] == "2026-09-30T08:01:00-04:00"
    assert feed["text_tags"] == [] and feed["url"] == idx          # inputs untouched
    assert sec.merge_events([], None) == []


def _sub_url(cik: int) -> str:
    return sec.SUBMISSIONS_URL.format(cik=cik)


def test_filing_events_end_to_end_and_text_enrichment(env):
    tickers_route(env)
    env.add(_sub_url(1961847), _sub(INHD_ROWS))
    evs = sec.filing_events("inhd")
    assert evs[0]["symbol"] == "INHD" and len(evs) == 14
    assert sec.filing_events("NOPE") == []
    # text check refines the 8-K 5.03 and the 424B5 (only text-checkable forms are read)
    rs = "https://www.sec.gov/Archives/edgar/data/1961847/000149315226021119/form8-k.htm"
    b5 = "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/form424b5.htm"
    env.add(rs, "<p>" + "x " * 4000 + "a 1-for-20 reverse stock split</p>")
    env.add(b5, LONG_424)
    evs = sec.filing_events("INHD", since="2026-05-01", fetch_text=True, max_text_docs=3)
    by_url = {e["url"]: e for e in evs}
    assert by_url[rs]["category"] == "reverse_split" and by_url[rs]["text_tags"] == ["reverse_split"]
    assert by_url[b5]["category"] == "atm"
    assert not any("formnt10-q" in u for u in env.urls())     # NT 10-Q is never text-checked
    assert net.STATUS[sec.STATUS_NAME]["ok"] is True


def test_submissions_merges_older_pages_inside_history_window(env, monkeypatch):
    monkeypatch.setattr(sec, "_history_cutoff", lambda: "2023-01-01")
    recent = _sub([("0000000001-24-000002", "2024-06-01", "2024-06-01T20:00:00.000Z", "8-K", "8.01", "a.htm")],
                  files=[{"name": "CIK0000000001-submissions-001.json", "filingCount": 2,
                          "filingFrom": "2023-01-05", "filingTo": "2024-05-01"},
                         {"name": "CIK0000000001-submissions-002.json", "filingCount": 1,
                          "filingFrom": "2019-01-01", "filingTo": "2022-12-31"}])
    page = {"accessionNumber": ["0000000001-24-000001", "0000000001-24-000002"],
            "filingDate": ["2024-05-01", "2024-06-01"], "form": ["424B5", "8-K"],
            "primaryDocument": ["b.htm", "a.htm"], "reportDate": ["", ""]}
    env.add(_sub_url(1), recent)
    env.add(sec.SUBMISSIONS_PAGE_URL.format(name="CIK0000000001-submissions-001.json"), page)
    sub = sec.submissions(1)
    rec = sub["filings"]["recent"]
    assert rec["accessionNumber"] == ["0000000001-24-000002", "0000000001-24-000001"]   # deduped
    assert rec["form"] == ["8-K", "424B5"]
    assert rec["items"] == ["8.01", None] and rec["reportDate"] == [None, ""]
    assert sub["_gravity"]["merged_files"] == ["CIK0000000001-submissions-001.json"]
    assert sub["_gravity"]["incomplete"] is False
    assert not any("submissions-002" in u for u in env.urls())   # entirely before the window
    evs = sec.events_from_submissions(sub, "X", 1)
    assert [e["form"] for e in evs] == ["8-K", "424B5"]
    # cached: a second call makes no request
    n = len(env.calls)
    sec.submissions(1)
    assert len(env.calls) == n


def test_events_for_universe_skips_unknown_and_failed(env):
    tickers_route(env)
    env.add(_sub_url(1961847), _sub(INHD_ROWS))
    out = sec.events_for_universe(["INHD", "INHDW", "DPRO", "ZZZZ"], max_workers=2,
                                  since="2026-01-01", incremental=False)
    assert set(out) == {"INHD", "INHDW"}             # DPRO fetch failed → absent (unknown), ZZZZ no CIK
    assert all(e["date"] >= "2026-01-01" for e in out["INHD"])
    assert out["INHDW"][0]["symbol"] == "INHDW"


# ═════════════════════════════════════════════════════════════════════════
# Issuer profile
# ═════════════════════════════════════════════════════════════════════════
def test_profile_hong_kong_issuer_is_asia():
    p = sec.profile_from_submissions(_sub([]))
    for k in sec.PROFILE_KEYS:
        assert k in p
    assert p["asia"] is True and p["business_country"] == "Hong Kong"
    assert p["business_city"] == "Tsim Sha Tsui" and p["state_of_inc"] == "TX"
    assert p["sic"] == "5990" and p["sic_desc"].startswith("Retail")
    assert p["filer_category"] == "Non-accelerated filer; Smaller reporting company; Emerging growth company"
    assert p["asia_basis"] == ["business address in Hong Kong (K3)"]
    assert p["offshore_inc"] is False
    assert p["source"] == "https://data.sec.gov/submissions/CIK0001961847.json"


def test_profile_us_issuer_with_cayman_incorporation_is_not_asia():
    us = {"city": "NEW YORK", "stateOrCountry": "NY", "isForeignLocation": 0}
    p = sec.profile_from_submissions(_sub([], stateOfIncorporation="E9",
                                          addresses={"business": us, "mailing": us}))
    assert p["asia"] is False and p["offshore_inc"] is True
    assert p["business_country"] == "United States" and p["business_state"] == "NY"


def test_profile_china_incorporation_and_mailing_fallback():
    p = sec.profile_from_submissions(_sub([], stateOfIncorporation="F4",
                                          addresses={"business": {}, "mailing": {"city": "BEIJING",
                                                                                 "stateOrCountry": "F4"}}))
    assert p["asia"] is True and p["business_country"] == "China"
    assert "incorporated in China (F4)" in p["asia_basis"]


def test_profile_unknown_everything_is_none():
    p = sec.profile_from_submissions({"addresses": {}, "cik": None})
    assert p["asia"] is None and p["business_country"] is None and p["state_of_inc"] is None
    assert p["source"] is None


def test_issuer_profile_failure_gives_all_none(env):
    p = sec.issuer_profile(1961847)
    assert set(p) == set(sec.PROFILE_KEYS) and all(v is None for v in p.values())
    env.add(_sub_url(1961847), _sub([]))
    assert sec.issuer_profile(1961847)["asia"] is True


# ═════════════════════════════════════════════════════════════════════════
# Full-text search
# ═════════════════════════════════════════════════════════════════════════
def _hit(adsh: str, doc: str, form: str, file_type: str, ciks: List[str], names: List[str],
         file_date: str = "2026-09-29", items: tuple = ()) -> Dict[str, Any]:
    return {"_id": f"{adsh}:{doc}", "_source": {
        "adsh": adsh, "ciks": ciks, "display_names": names, "form": form, "root_forms": [form],
        "file_date": file_date, "file_type": file_type, "items": list(items)}}


DPRO = ["Draganfly Inc.  (DPRO)  (CIK 0001786286)"]
EFTS_HITS: Dict[str, List[Dict[str, Any]]] = {
    '"registered direct"': [
        _hit("0001493152-26-044983", "ex99-1.htm", "6-K", "EX-99.1", ["0001786286"], DPRO, "2026-09-30"),
        _hit("0001493152-26-044983", "form6-k.htm", "6-K", "6-K", ["0001786286"], DPRO, "2026-09-30"),
    ],
    '"securities purchase agreement"': [
        _hit("0001493152-26-044983", "ex10-1.htm", "6-K", "EX-10.1", ["0001786286"], DPRO, "2026-09-30"),
    ],
    '"pricing of" "public offering"': [
        _hit("0001193125-26-000001", "d1.htm", "8-K", "EX-99.1", ["0001099160"],
             ["Beasley  (BBGI)  (CIK 0001099160)"], "2026-09-29", ("8.01", "9.01")),
    ],
    '"public offering" priced': [
        _hit("0001193125-26-000002", "d2.htm", "8-K", "8-K", ["0001400118"],
             ["Sangamo  (SGMT)  (CIK 0001400118)"], "2026-09-28", ("8.01",)),
    ],
    sec.PHRASES["reverse_split"][0]: [
        # CIK unknown to the ticker map → symbol from display_names, warrant ticker dropped
        _hit("0001493152-26-045028", "ex99-1.htm", "6-K", "EX-99.1", ["0001982012"],
             ["Top Leader Intl  (TLIH, TLIHW)  (CIK 0001982012)"], "2026-09-30"),
        _hit("0001493152-26-044997", "ex10-1.htm", "8-K", "EX-10.1", ["0001650101"],
             ["Autonomix  (ATXG)  (CIK 0001650101)"], "2026-09-30", ("1.01", "3.02", "9.01")),
        _hit("0001493152-26-000003", "f8k.htm", "8-K", "8-K", ["0001961847"],
             ["INNO HOLDINGS INC.  (INHD)  (CIK 0001961847)"], "2026-09-28", ("5.03", "9.01")),
        # no ticker anywhere → dropped
        _hit("0009999999-26-000001", "x.htm", "8-K", "8-K", ["0009999999"],
             ["Private Co  (CIK 0009999999)"], "2026-09-29", ("5.03",)),
    ],
    sec.PHRASES["going_concern"][0]: [
        _hit("0001114483-26-000001", "q.htm", "10-K", "10-K", ["0001114483"],
             ["Integer  (ITGR)  (CIK 0001114483)"], "2026-09-29"),
    ],
    sec.PHRASES["atm"][0]: [
        _hit("0001493152-26-000004", "b5.htm", "424B5", "424B5", ["0001961847"],
             ["INNO HOLDINGS INC.  (INHD)  (CIK 0001961847)"], "2026-09-29"),
    ],
}


def efts_route(router: Router, hits_by_q: Dict[str, list] = EFTS_HITS, fail_q: str = "") -> None:
    def answer(url: str, params: Dict[str, Any]) -> Any:
        if params.get("q") == fail_q:
            return None
        hits = hits_by_q.get(params.get("q"), [])
        off = int(params.get("from") or 0)
        return {"hits": {"total": {"value": len(hits), "relation": "eq"}, "hits": hits[off:off + 100]}}
    router.add(lambda u, p: u == sec.EFTS_URL, answer)


def test_fulltext_catalysts_events_tags_and_symbols(env):
    tickers_route(env)
    efts_route(env)
    out = sec.fulltext_catalysts("2026-09-28", "2026-09-30")
    by_sym = {e["symbol"]: e for e in out}
    assert set(by_sym) == {"DPRO", "BBGI", "SGMT", "TLIH", "ATXG", "INHD", "ITGR"}
    assert sorted(e["symbol"] for e in out).count("INHD") == 2
    dpro = by_sym["DPRO"]
    assert dpro["text_tags"] == ["registered_direct", "securities_purchase_agreement"]
    assert dpro["category"] == "offering" and dpro["form"] == "6-K" and dpro["cik"] == 1786286
    assert dpro["url"] == "https://www.sec.gov/Archives/edgar/data/1786286/000149315226044983/form6-k.htm"
    assert dpro["date"] == "2026-09-30" and dpro["accepted"] is None and dpro["source"] == "sec_fulltext"
    assert by_sym["BBGI"]["text_tags"] == ["public_offering_priced"] and by_sym["BBGI"]["category"] == "offering"
    assert by_sym["SGMT"]["category"] == "offering"
    assert by_sym["TLIH"]["cik"] == 1982012 and by_sym["TLIH"]["category"] == "reverse_split"
    assert by_sym["ATXG"]["category"] == "unregistered_sale"        # agreement boilerplate, not a split
    assert by_sym["ITGR"]["category"] == "going_concern"
    inhd = {e["accession"]: e for e in out if e["symbol"] == "INHD"}
    assert inhd["0001493152-26-000003"]["category"] == "reverse_split"
    assert inhd["0001493152-26-000004"]["category"] == "atm"
    assert all(e["symbol"] != "PRIVATE" for e in out) and len(out) == 8
    dates = [e["date"] for e in out]
    assert dates == sorted(dates, reverse=True)
    # both priced-offering queries ran, none with parentheses
    qs = [c[1]["q"] for c in env.calls if c[0] == sec.EFTS_URL]
    assert '"pricing of" "public offering"' in qs and '"public offering" priced' in qs
    params = [c[1] for c in env.calls if c[0] == sec.EFTS_URL][0]
    assert params["startdt"] == "2026-09-28" and params["enddt"] == "2026-09-30"
    assert params["dateRange"] == "custom"
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is True and "8 filings" in st["detail"]
    for e in out:
        assert set(e["text_tags"]) <= set(sec.FULLTEXT_TAGS) and e["category"] in sec.CATEGORIES


def test_fulltext_catalysts_failed_query_is_reported_not_hidden(env):
    tickers_route(env)
    efts_route(env, fail_q='"registered direct"')
    out = sec.fulltext_catalysts("2026-09-28", "2026-09-30", tags=["registered_direct", "atm"])
    assert [e["symbol"] for e in out] == ["INHD"]
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is False and "failed registered_direct" in st["detail"]


def test_fulltext_results_are_cached_per_query(env):
    tickers_route(env)
    efts_route(env)
    sec.fulltext_catalysts("2026-09-28", "2026-09-30", tags=["public_offering_priced"])
    n = len(env.calls)
    again = sec.fulltext_catalysts("2026-09-28", "2026-09-30", tags=["public_offering_priced"])
    assert len(env.calls) == n and {e["symbol"] for e in again} == {"BBGI", "SGMT"}


def test_efts_range_splits_oversized_ranges(monkeypatch):
    totals = {("2026-09-01", "2026-09-04"): 250, ("2026-09-01", "2026-09-02"): 120,
              ("2026-09-03", "2026-09-04"): 130, ("2026-09-05", "2026-09-05"): 900}
    seen = []

    def page(q, forms, start, end, offset):
        seen.append((start, end, offset))
        total = totals[(start, end)]
        n = max(0, min(100, total - offset))
        hits = [{"_id": f"{start}-{offset + i}:d.htm", "_source": {"adsh": f"{start}-{offset + i}"}}
                for i in range(n)]
        return {"total": {"value": total, "relation": "eq"}, "hits": hits}

    monkeypatch.setattr(sec, "_efts_page", page)
    hits, ok, trunc = sec._efts_range("q", "8-K", "2026-09-01", "2026-09-04", 2, [100])
    assert ok and not trunc and len(hits) == 250
    assert ("2026-09-01", "2026-09-02", 100) in seen and ("2026-09-03", "2026-09-04", 100) in seen
    hits, ok, trunc = sec._efts_range("q", "8-K", "2026-09-05", "2026-09-05", 2, [100])
    assert ok and trunc and len(hits) == 200                   # one day can't split: truncated, flagged
    hits, ok, trunc = sec._efts_range("q", "8-K", "2026-09-01", "2026-09-04", 2, [1])
    assert trunc                                                # request budget exhausted


def test_tickers_from_display_names():
    assert sec._tickers_from_display("Top Leader  (TLIH, TLIHW)  (CIK 0001982012)") == ["TLIH"]
    assert sec._tickers_from_display("Berkshire  (BRK.B, BRK.A)  (CIK 0001067983)") == ["BRK-B", "BRK-A"]
    assert sec._tickers_from_display("Unit Trust  (ABCU)  (CIK 0000000001)") == ["ABCU"]
    assert sec._tickers_from_display("Private Co  (CIK 0009999999)") == []
    assert sec._tickers_from_display("") == []


# ═════════════════════════════════════════════════════════════════════════
# Current-events feed
# ═════════════════════════════════════════════════════════════════════════
def _atom(entries: List[tuple]) -> str:
    parts = ['<?xml version="1.0" encoding="ISO-8859-1" ?>',
             '<feed xmlns="http://www.w3.org/2005/Atom"><title>Latest Filings</title>']
    for form, name, cik, role, acc, when, items in entries:
        nodash = acc.replace("-", "")
        item_html = "".join(f"&lt;br&gt;Item {i}: Something" for i in items)
        parts.append(
            f"<entry><title>{form} - {name} ({cik:010d}) ({role})</title>"
            f'<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/{cik}/'
            f'{nodash}/{acc}-index.htm"/>'
            f'<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; {when.date().isoformat()} &lt;b&gt;AccNo:&lt;/b&gt; '
            f"{acc} &lt;b&gt;Size:&lt;/b&gt; 253 KB{item_html}</summary>"
            f"<updated>{when.isoformat(timespec='seconds')}</updated>"
            f'<category scheme="https://www.sec.gov/" label="form type" term="{form}"/>'
            f"<id>urn:tag:sec.gov,2008:accession-number={acc}</id></entry>")
    parts.append("</feed>")
    return "".join(parts)


def test_parse_atom_entries():
    when = datetime(2026, 9, 30, 8, 21, 58, tzinfo=util.ET)
    rows = sec._parse_atom(_atom([("8-K", "Integer Holdings Corp", 1114483, "Filer",
                                   "0000950103-26-014831", when, ["8.01", "9.01"])]).encode())
    assert rows == [{
        "form": "8-K", "name": "Integer Holdings Corp", "cik": 1114483, "role": "Filer",
        "accession": "0000950103-26-014831", "date": "2026-09-30", "accepted_dt": when,
        "items": ["8.01", "9.01"],
        "url": "https://www.sec.gov/Archives/edgar/data/1114483/000095010326014831/0000950103-26-014831-index.htm",
    }]
    assert sec._parse_atom(b"<html>not a feed</html>") is None
    assert sec._parse_atom(b"garbage<") is None


def test_latest_filings_maps_filters_and_pages(env):
    tickers_route(env)
    now = datetime.now(util.ET).replace(microsecond=0)
    since = now - timedelta(hours=18)
    fresh, fresh2, old = now - timedelta(hours=1), now - timedelta(hours=2), now - timedelta(hours=30)
    pages = {
        ("8-K", 0): _atom([("8-K", "Integer Holdings Corp", 1114483, "Filer", "0000950103-26-014831", fresh, ["8.01", "9.01"])]
                          + [("8-K", f"Unlisted {i}", 5000000 + i, "Filer", f"0000000000-26-{i:06d}", fresh2, ["7.01"])
                             for i in range(99)]),
        ("8-K", 100): _atom([("8-K", "Draganfly Inc.", 1786286, "Filer", "0001493152-26-044900", fresh2, ["3.01"]),
                             ("8-K", "Autonomix", 1650101, "Filer", "0001493152-26-044901", old, ["1.01"])]),
        ("144", 0): _atom([("144", "Some Person", 1990001, "Reporting", "0001104659-26-112124", fresh, []),
                           ("144", "Insurance Co", 1794338, "Subject", "0001104659-26-112124", fresh, [])]),
        ("NT", 0): _atom([("NT 10-Q", "INNO HOLDINGS INC.", 1961847, "Filer", "0001493152-26-038354", fresh, [])]),
        ("424B5", 0): _atom([("424B5", "INNO HOLDINGS INC.", 1961847, "Filer", "0001493152-26-024367", fresh, []),
                             ("SC 13D", "INNO HOLDINGS INC.", 1961847, "Subject", "0001493152-26-024368", fresh, [])]),
    }

    def answer(url: str, p: Dict[str, Any]) -> str:
        assert p["action"] == "getcurrent" and p["output"] == "atom" and p["count"] == 100
        return pages.get((p["type"], p["start"]), _atom([]))

    env.add(lambda u, p: u == sec.CURRENT_URL, answer)
    evs = sec.latest_filings(since.astimezone(timezone.utc))
    got = [(e["symbol"], e["form"], e["category"]) for e in evs]
    assert ("ITGR", "8-K", "other") in got
    assert ("DPRO", "8-K", "delisting_notice") in got                   # found on page 2
    assert ("IGIC", "144", "insider_sale_notice") in got                # subject company, once
    assert ("INHD", "NT 10-Q", "late_filing") in got
    assert ("INHD", "424B5", "offering") in got
    assert not any(s == "ATXG" for s, _, _ in got)                      # older than since
    assert not any(f == "SC 13D" for _, f, _ in got)                    # not a form of interest
    assert len(evs) == 5
    itgr = next(e for e in evs if e["symbol"] == "ITGR")
    assert itgr["accepted"] == fresh.isoformat() and itgr["items"] == ["8.01", "9.01"]
    assert itgr["source"] == "sec_current" and itgr["url"].endswith("-index.htm")
    assert all(datetime.fromisoformat(e["accepted"]) >= since for e in evs)
    keys = [(e["date"], e["accepted"]) for e in evs]
    assert keys == sorted(keys, reverse=True)
    types = {c[1]["type"] for c in env.calls if c[0] == sec.CURRENT_URL}
    assert types == set(sec.CURRENT_TYPES)
    assert "424B2" not in types
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is True and "5 issuer filings" in st["detail"]


def test_latest_filings_reports_incomplete_feed(env):
    tickers_route(env)
    env.add(lambda u, p: u == sec.CURRENT_URL and p["type"] != "8-K", _atom([]))   # 8-K page fails
    evs = sec.latest_filings(datetime.now(timezone.utc) - timedelta(hours=2))
    assert evs == []
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is False and "feed incomplete" in st["detail"]


def test_latest_filings_accepts_naive_utc(env):
    tickers_route(env)
    env.add(lambda u, p: u == sec.CURRENT_URL, _atom([]))
    assert sec.latest_filings(datetime.utcnow() - timedelta(hours=1)) == []


# ═════════════════════════════════════════════════════════════════════════
# XBRL companyfacts
# ═════════════════════════════════════════════════════════════════════════
def _fact(end: str, val: float, filed: str, accn: str, form: str = "10-Q", start: Optional[str] = None) -> dict:
    r = {"end": end, "val": val, "filed": filed, "accn": accn, "form": form}
    if start:
        r["start"] = start
    return r


FACTS = {
    "cik": 1961847,
    "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _fact("2026-01-30", 8413224, "2026-02-03", "A1"),
            _fact("2026-04-30", 50413224, "2026-05-01", "A2"),
            _fact("2026-04-30", 2520611, "2026-08-18", "A3"),      # later restatement: ignored (point-in-time)
            _fact("2026-08-18", 2520581, "2026-08-18", "A3"),
            _fact("2026-08-18", 2520000, "2026-08-18", "A3"),      # duplicate in one filing → max kept
            _fact("2026-09-01", 0, "2026-09-02", "A4"),             # zero → dropped
        ]}}},
        "us-gaap": {
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                _fact("2025-12-31", 20_000_000, "2026-02-03", "A1"),
                _fact("2026-06-30", 33_238_616, "2026-08-18", "A3"),
            ]}},
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {"units": {"USD": [
                _fact("2026-06-30", 34_000_000, "2026-08-18", "A3"),   # same date, lower rank concept
            ]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                # fiscal year starts 2025-10-01; 10-Qs report year-to-date
                _fact("2025-12-31", -5_001_179, "2026-02-03", "A1", start="2025-10-01"),
                _fact("2026-03-31", -7_935_775, "2026-05-01", "A2", start="2025-10-01"),
                _fact("2026-06-30", -11_632_320, "2026-08-18", "A3", start="2025-10-01"),
                _fact("2025-09-30", -9_000_000, "2025-12-15", "K1", "10-K", start="2024-10-01"),
            ]}},
        },
    },
}


def test_shares_from_facts_point_in_time():
    rows = sec.shares_from_facts(FACTS)
    assert [(r["date"], r["shares"]) for r in rows] == [
        ("2026-01-30", 8413224.0), ("2026-04-30", 50413224.0), ("2026-08-18", 2520581.0)]
    assert rows[1]["filed"] == "2026-05-01" and rows[0]["concept"] == "dei:EntityCommonStockSharesOutstanding"
    for r in rows:
        assert set(r) >= {"date", "filed", "shares"}


def test_shares_fallback_concept_and_empty():
    facts = {"facts": {"us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [
        _fact("2026-06-30", 1000, "2026-08-01", "X")]}}}}}
    assert sec.shares_from_facts(facts)[0]["concept"] == "us-gaap:CommonStockSharesOutstanding"
    assert sec.shares_from_facts({"facts": {}}) == []


def test_runway_from_ytd_cash_flows():
    r = sec.runway_from_facts(FACTS, 1961847)
    assert r["cash"] == 33_238_616 and r["cash_date"] == "2026-06-30"
    assert r["cash_concept"] == "us-gaap:CashAndCashEquivalentsAtCarryingValue"
    q = {x["end"]: x for x in r["burn_quarters"]}
    assert q["2026-03-31"]["ocf"] == pytest.approx(-2_934_596) and q["2026-03-31"]["derived"] is True
    assert q["2026-06-30"]["ocf"] == pytest.approx(-3_696_545)
    assert q["2025-12-31"]["derived"] is False
    assert r["quarterly_burn"] == pytest.approx((2_934_596 + 3_696_545) / 2)
    assert r["runway_q"] == pytest.approx(33_238_616 / r["quarterly_burn"], abs=0.01)
    assert r["currency"] == "USD" and r["burn_date"] == "2026-06-30"
    assert r["source"] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0001961847.json"


def test_runway_semiannual_filer_scales_to_a_quarter_and_positive_ocf_has_no_runway():
    facts = {"facts": {"ifrs-full": {
        "CashAndCashEquivalents": {"units": {"USD": [_fact("2026-06-30", 1_000_000, "2026-09-01", "S")]}},
        "CashFlowsFromUsedInOperatingActivities": {"units": {"USD": [
            _fact("2026-06-30", -2_000_000, "2026-09-01", "S", start="2026-01-01")]}},
    }}}
    r = sec.runway_from_facts(facts)
    assert r["quarterly_burn"] == pytest.approx(2_000_000 * (365.25 / 4) / 180, rel=1e-3)
    assert r["runway_q"] == pytest.approx(1_000_000 / r["quarterly_burn"], abs=0.01)
    assert "scaled" in r["burn_basis"] and r["source"] is None
    facts["facts"]["ifrs-full"]["CashFlowsFromUsedInOperatingActivities"]["units"]["USD"][0]["val"] = 500_000
    r = sec.runway_from_facts(facts)
    assert r["quarterly_burn"] < 0 and r["runway_q"] is None       # generating cash: no runway claimed


def test_runway_none_without_cash_and_burn_none_without_flows():
    assert sec.runway_from_facts({"facts": {}}) is None
    facts = {"facts": {"us-gaap": {"CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
        _fact("2026-06-30", 5.0, "2026-08-01", "X")]}}}}}
    r = sec.runway_from_facts(facts)
    assert r["cash"] == 5.0 and r["quarterly_burn"] is None and r["runway_q"] is None


def test_shares_history_and_cash_runway_share_one_download(env):
    env.add(sec.FACTS_URL.format(cik=1961847), FACTS)
    sh = sec.shares_history(1961847)
    cr = sec.cash_runway(1961847)
    assert len(sh) == 3 and cr["runway_q"] > 0
    assert env.urls().count(sec.FACTS_URL.format(cik=1961847)) == 1


def test_facts_missing_is_empty_and_none(env):
    assert sec.shares_history(123) == [] and sec.cash_runway(123) is None


# ═════════════════════════════════════════════════════════════════════════
# Plumbing: disabled mode, breaker, time conversion
# ═════════════════════════════════════════════════════════════════════════
def test_everything_noops_without_user_agent(env, monkeypatch):
    monkeypatch.setattr(config, "SEC_USER_AGENT", "")

    def boom(*a, **k):
        raise AssertionError("network touched")

    monkeypatch.setattr(net, "get", boom)
    assert sec.cik_map() == {}
    assert sec.submissions(1) is None
    assert sec.filing_events("INHD") == []
    assert all(v is None for v in sec.issuer_profile(1).values())
    assert sec.latest_filings(datetime.now(timezone.utc)) == []
    assert sec.fulltext_catalysts("2026-09-28", "2026-09-30") == []
    assert sec.text_classify("https://www.sec.gov/x.htm") is None
    assert sec.events_for_universe(["INHD"]) == {}
    assert sec.shares_history(1) == [] and sec.cash_runway(1) is None
    st = net.STATUS[sec.STATUS_NAME]
    assert st["ok"] is False and st["detail"] == "SEC_USER_AGENT not set"


def test_breaker_stops_hammering_after_repeated_failures(env, monkeypatch):
    monkeypatch.setattr(sec, "_breaker", sec._Breaker(threshold=3, cooldown_s=60))
    for _ in range(3):
        assert sec._sec_get("https://www.sec.gov/fail") is None
    n = len(env.calls)
    assert sec._sec_get("https://www.sec.gov/fail") is None
    assert len(env.calls) == n                                  # open: no request made
    sec._note("x", True, "fine")
    assert net.STATUS[sec.STATUS_NAME]["ok"] is False
    assert "paused" in net.STATUS[sec.STATUS_NAME]["detail"]


def test_missing_ok_404s_do_not_trip_the_breaker(env, monkeypatch):
    monkeypatch.setattr(sec, "_breaker", sec._Breaker(threshold=2, cooldown_s=60))
    for _ in range(5):
        sec._sec_get("https://www.sec.gov/404", missing_ok=True)
    assert sec._breaker.allow()


def test_accepted_et_conversion():
    assert sec._accepted_et("2026-05-19T13:57:14.000Z") == "2026-05-19T09:57:14-04:00"
    assert sec._accepted_et("2026-01-20T22:00:28.000Z") == "2026-01-20T17:00:28-05:00"
    assert sec._accepted_et("") is None and sec._accepted_et("garbage") is None


def test_filing_url_forms():
    assert sec.filing_url(1961847, "0001493152-26-024367", "form424b5.htm") == \
        "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/form424b5.htm"
    assert sec.filing_url(1961847, "0001493152-26-024367") == \
        "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/0001493152-26-024367-index.htm"


# ═════════════════════════════════════════════════════════════════════════
# Live (GRAVITY_LIVE=1, SEC_USER_AGENT set) — a few dozen requests at 2/s
# ═════════════════════════════════════════════════════════════════════════
live = pytest.mark.skipif(not (LIVE and config.SEC_USER_AGENT),
                          reason="set GRAVITY_LIVE=1 and SEC_USER_AGENT to hit EDGAR")
INHD_CIK = 1961847


@pytest.fixture
def polite(monkeypatch):
    monkeypatch.setattr(sec, "_SEC_GAP_S", 0.5)
    monkeypatch.setattr(sec, "_breaker", sec._Breaker())


@pytest.mark.live
@live
def test_live_inhd_filing_events(polite):
    evs = sec.filing_events("INHD")
    assert len(evs) > 50
    key = {(e["date"], e["form"]): e for e in evs}
    assert key[("2026-05-19", "424B5")]["category"] == "offering"
    assert key[("2026-05-04", "8-K")]["category"] in ("reverse_split", "charter_amendment")
    assert key[("2025-12-22", "8-K")]["category"] in ("reverse_split", "charter_amendment")
    assert key[("2026-08-14", "NT 10-Q")]["category"] == "late_filing"
    for e in evs:
        assert e["symbol"] == "INHD" and e["cik"] == INHD_CIK and e["category"] in sec.CATEGORIES
        assert e["url"].startswith("https://www.sec.gov/Archives/edgar/data/1961847/")
    print("\nINHD events:", len(evs), [(e["date"], e["form"], e["category"]) for e in evs[:8]])


@pytest.mark.live
@live
def test_live_inhd_profile_shares_runway(polite):
    p = sec.issuer_profile(INHD_CIK)
    assert p["asia"] is True and p["business_country"] == "Hong Kong"
    sh = sec.shares_history(INHD_CIK)
    assert len(sh) >= 4 and all(r["shares"] > 0 for r in sh)
    assert [r["date"] for r in sh] == sorted(r["date"] for r in sh)
    cr = sec.cash_runway(INHD_CIK)
    assert cr is not None and cr["cash"] > 0 and cr["cash_date"] >= "2025-01-01"
    print("\nINHD profile:", p["business_city"], p["business_country"], p["state_of_inc"],
          "| shares:", sh[-3:], "| runway:", {k: cr[k] for k in ("cash", "cash_date", "quarterly_burn", "runway_q")})


@pytest.mark.live
@live
def test_live_text_classify_inhd_atm_prospectus(polite):
    url = "https://www.sec.gov/Archives/edgar/data/1961847/000149315226024367/form424b5.htm"
    tags = sec.text_classify(url)
    assert tags is not None and "atm" in tags
    assert sec.category_from_tags(tags, "424B5") == "atm"
    rs = sec.text_classify("https://www.sec.gov/Archives/edgar/data/1961847/000149315226021119/form8-k.htm")
    assert rs is not None and "reverse_split" in rs
    print("\n424B5 tags:", tags, "| 8-K 5.03 tags:", rs)


@pytest.mark.live
@live
def test_live_fulltext_catalysts(polite):
    out = sec.fulltext_catalysts("2026-09-28", "2026-09-30")
    assert len(out) > 20
    tags = {}
    for e in out:
        assert e["symbol"] and isinstance(e["cik"], int) and e["category"] in sec.CATEGORIES
        assert e["text_tags"] and set(e["text_tags"]) <= set(sec.FULLTEXT_TAGS)
        assert "2026-09-28" <= e["date"] <= "2026-09-30"
        for t in e["text_tags"]:
            tags[t] = tags.get(t, 0) + 1
    assert tags.get("public_offering_priced", 0) > 0          # the grouped-query bug is fixed
    assert tags.get("reverse_split", 0) > 0
    print("\nfulltext 09-28..09-30:", len(out), "filings; tags", tags)


@pytest.mark.live
@live
def test_live_latest_filings_and_index_text(polite):
    since = datetime.now(timezone.utc) - timedelta(hours=18)
    evs = sec.latest_filings(since)
    assert len(evs) > 20
    for e in evs:
        assert e["symbol"] and datetime.fromisoformat(e["accepted"]) >= since
        assert e["form"] in sec.CURRENT_FORMS and e["category"] in sec.CATEGORIES
    six = next((e for e in evs if e["form"] in ("6-K", "8-K")), None)
    assert six is not None
    tags = sec.text_classify(six["url"])
    assert tags is not None                                   # index page resolved to documents
    cats: Dict[str, int] = {}
    for e in evs:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print("\nlatest since", since.isoformat(timespec="minutes"), ":", len(evs), cats,
          "| text of", six["symbol"], six["form"], "→", tags)
