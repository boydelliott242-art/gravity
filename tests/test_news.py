"""Tests for gravity.sources.news (CONTRACTS.md §6).

The classifier is checked for precision on realistic small-cap headlines
(each with its exact expected tag set), including the traps that look like
catalysts but are not ("product offering", "regains compliance", a
terminated merger, a financing agreement dressed as a contract). Google
News is replaced by a canned RSS document; the live test (``GRAVITY_LIVE=1``)
makes one real query for INHD into a temporary cache.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import quote_plus

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.sources import news as N  # noqa: E402

LIVE = os.environ.get("GRAVITY_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="set GRAVITY_LIVE=1 for live network tests")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    net.STATUS.pop("Google News", None)
    monkeypatch.setattr(N, "_NEWS_STATS", {"ok": 0, "fail": 0})
    return tmp_path / "cache"


# ═════════════════════════════════════════════════════════════════════════
# classify — exact tag sets and polarity
# ═════════════════════════════════════════════════════════════════════════
HEADLINES = [
    # dilution / financing
    ("Inno Holdings Announces $5 Million Registered Direct Offering Priced At-the-Market", {"offering", "priced"}, -1),
    ("XYZ Corp. Announces Pricing of $10 Million Underwritten Public Offering", {"offering", "priced"}, -1),
    ("XYZ Corp. Announces Proposed Public Offering of Common Stock", {"offering"}, -1),
    ("XYZ Stock Plunges After Pricing Offering", {"offering", "priced"}, -1),
    ("XYZ Upsizes Offering to $15 Million", {"offering"}, -1),
    ("XYZ Announces Closing of $4.5 Million Private Placement Priced At-the-Market Under Nasdaq Rules",
     {"offering", "priced"}, -1),
    ("ABC Enters Into $20 Million Securities Purchase Agreement with Institutional Investor", {"offering"}, -1),
    ("ABC Therapeutics Enters into $50 Million At-The-Market Equity Offering Sales Agreement", {"offering", "atm"}, -1),
    ("XYZ Enters into Standby Equity Purchase Agreement with Yorkville Advisors", {"dilution"}, -1),
    ("XYZ Signs $10 Million Equity Line of Credit Agreement", {"dilution"}, -1),
    ("XYZ Files S-1 Registration Statement for Resale of Shares by Selling Stockholders", {"dilution"}, -1),
    ("XYZ Shares Fall as Company Announces Mixed Shelf Offering", {"dilution"}, -1),
    ("NOP Announces Exercise of Warrants for $3.2 Million Gross Proceeds", {"warrants"}, -1),
    ("UVW Prices $8 Million Offering to Fund Acquisition of Crypto Miner", {"offering", "priced", "acquisition"}, -1),
    # listing / capital structure
    ("Inno Holdings Inc. Announces 1-for-20 Reverse Stock Split", {"reverse_split"}, -1),
    ("XYZ Stockholders Approve Reverse Stock Split at Special Meeting", {"reverse_split"}, -1),
    ("XYZ Announces Share Consolidation of its Ordinary Shares", {"reverse_split"}, -1),
    ("KNOREX Receives Nasdaq Notification Regarding Minimum Bid Price Deficiency", {"deficiency"}, -1),
    ("XYZ Receives Nasdaq Delinquency Notification Letter", {"deficiency"}, -1),
    ("DEF Receives Nasdaq Delisting Determination; Plans to Appeal", {"delisting"}, -1),
    ("XYZ Granted Extension by Nasdaq Hearings Panel", {"delisting"}, -1),
    ("XYZ Faces Delisting After Failing to Meet Bid Price Rule", {"delisting", "deficiency"}, -1),
    ("Trading Halted in JKL Pending News", {"halt"}, -1),
    ("SEC Suspends Trading in MNO Shares", {"halt"}, -1),
    ("Nasdaq Halts XYZ", {"halt"}, -1),
    # governance / solvency / street
    ("Shareholder Alert: Pomerantz Law Firm Investigates Claims On Behalf of Investors of PQR",
     {"investigation", "lawsuit"}, -1),
    ("Class Action Lawsuit Filed Against STU Corp.; Investors Encouraged to Contact Firm", {"lawsuit"}, -1),
    ("VWX CFO Resigns Effective Immediately", {"resign"}, -1),
    ("YZA Discloses Substantial Doubt About Ability to Continue as Going Concern", {"going_concern"}, -1),
    ("BCD Files for Chapter 11 Bankruptcy Protection", {"default"}, -1),
    ("EFG Downgraded to Sell at Goldman Sachs", {"downgrade"}, -1),
    ("HIJ Q2 Earnings Miss Estimates as Revenue Falls", {"miss"}, -1),
    ("KLM Lowers Full-Year Guidance, Shares Tumble", {"guidance_cut"}, -1),
    ("ZAB Receives FDA Complete Response Letter for NDA", {"fda"}, -1),
    ("XYZ Receives Warning Letter from FDA", {"fda"}, -1),
    # bullish
    ("QRS Wins $12 Million Contract with U.S. Department of Defense", {"contract"}, 1),
    ("XYZ secures purchase order worth $2 million from major retailer", {"contract"}, 1),
    ("XYZ Signs Distribution Agreement with Walmart", {"contract"}, 1),
    ("TUV Announces Strategic Partnership with Microsoft to Deploy AI Agents", {"partnership"}, 1),
    ("WXY Receives FDA Approval for Lead Candidate", {"fda", "approval"}, 1),
    ("XYZ Stock Soars After FDA Grants Fast Track Designation", {"fda"}, 1),
    ("CDE Q3 EPS Beats Estimates, Revenue Tops Consensus", {"beat"}, 1),
    ("XYZ Reports Record Revenue for Second Quarter", {"beat"}, 1),
    ("FGH Upgraded to Buy at Maxim Group", {"upgrade"}, 1),
    ("Maxim raises XYZ price target to $10", {"upgrade"}, 1),
    ("IJK to Acquire LMN in All-Stock Deal", {"acquisition"}, 1),
    ("OPQ Board Authorizes $10 Million Share Repurchase Program", {"buyback"}, 1),
    ("RST Approved for Uplisting to Nasdaq Capital Market", {"approval", "uplisting"}, 1),
    # traps: nothing (or less) should fire
    ("GHI Regains Compliance with Nasdaq Minimum Bid Price Requirement", set(), 0),
    ("ABC Launches New Product Offering for Enterprise Customers", set(), 0),
    ("XYZ Announces Termination of Merger Agreement with ABC", set(), 0),
    ("XYZ Enters into $3 Million Loan Agreement with Lender", set(), 0),
    ("XYZ Terminates At-The-Market Offering Program", set(), 0),
    ("XYZ Cancels Proposed Public Offering", set(), 0),
    ("Inno Holdings Q3 Results: Sales up 90%, EPS improves to $(0.70)", set(), 0),
    ("Why Is XYZ Stock Down 50% Today?", set(), 0),
    ("Why XYZ Shares Are Trading Higher Today", set(), 0),
    ("XYZ to Present at H.C. Wainwright Investor Conference", set(), 0),
    ("", set(), 0),
]


@pytest.mark.parametrize("title,tags,polarity", HEADLINES)
def test_classify_headline(title, tags, polarity):
    got = N.classify(title)
    assert set(got["tags"]) == tags, got
    assert got["polarity"] == polarity, got


def test_classify_precision_summary():
    """Aggregate precision/recall over the labelled set (guards regressions
    that a single parametrised case might hide)."""
    tp = fp = fn = 0
    for title, tags, _ in HEADLINES:
        got = set(N.classify(title)["tags"])
        tp += len(got & tags)
        fp += len(got - tags)
        fn += len(tags - got)
    assert tp / max(1, tp + fp) >= 0.95 and tp / max(1, tp + fn) >= 0.95


def test_classify_output_shape_and_order():
    c = N.classify("XYZ Prices Offering, Announces Reverse Split and Wins Contract")
    assert list(c) == ["tags", "polarity"]
    order = N.BEARISH_TAGS + N.BULLISH_TAGS
    assert c["tags"] == sorted(c["tags"], key=order.index)
    assert c["polarity"] == -1                                  # dilution dominates
    assert set(c["tags"]) <= set(order)


def test_classify_handles_typographic_punctuation():
    assert N.classify("XYZ’s 1‑for‑25 reverse split — effective today")["tags"] == ["reverse_split"]


# ═════════════════════════════════════════════════════════════════════════
# query / mention helpers
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name,core", [
    ("Inno Holdings Inc.", "Inno Holdings"),
    ("KNOREX LTD. Class A Ordinary Shares", "KNOREX"),
    ("Mynd.ai Inc. American Depositary Shares", "Mynd.ai"),
    ("Apple Inc. Common Stock", "Apple"),
    ("", ""),
])
def test_company_core(name, core):
    assert N.company_core(name) == core


def test_build_query():
    assert N.build_query("INHD", "Inno Holdings Inc.") == '"INHD" stock OR "Inno Holdings" when:7d'
    assert N.build_query("KNRX", "", 3) == '"KNRX" stock when:3d'
    assert N.build_query("ABC", "ABC", 0) == '"ABC" stock when:1d'


@pytest.mark.parametrize("title,symbol,core,expected", [
    ("Inno Holdings Announces Reverse Split", "INHD", "Inno Holdings", True),
    ("INHD stock jumps", "INHD", "", True),
    ("Why $INHD is moving", "INHD", "", True),
    ("INHDX fund update", "INHD", "", False),
    ("Knorex Stock Skyrockets Monday", "KNRX", "KNOREX", True),
    ("Nanocap Stocks to Watch: IGC, KNRX, GYGY", "KNRX", "KNOREX", True),
    ("Here's Why GLXY Stock Is Rallying Today?", "INHD", "Inno Holdings", False),
    ("F shares fall", "F", "Ford Motor", False),             # 1-2 letter tickers need $F / (F) / :F
    ("Ford Motor (F) recalls trucks", "F", "Ford Motor", True),
    ("China stocks rally", "CJET", "China Jet", False),       # generic first word is not enough
    ("Berkshire (BRK.B) hits record", "BRK-B", "", True),     # share class written with a dot
    ("BRK-B and BRK/A diverge", "BRK-B", "", True),
    ("BRKXB is unrelated", "BRK-B", "", False),
])
def test_mentions(title, symbol, core, expected):
    assert N._mentions(title, symbol, core) is expected


# ═════════════════════════════════════════════════════════════════════════
# RSS parsing and headlines()
# ═════════════════════════════════════════════════════════════════════════
def _rss(items) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>',
           "<title>&quot;INHD&quot; stock - Google News</title>"]
    for title, when, source, link in items:
        pub = f"<pubDate>{format_datetime(when, usegmt=True)}</pubDate>" if when else ""
        src = f'<source url="https://{source.lower().replace(" ", "")}.com">{source}</source>' if source else ""
        out.append(f"<item><title>{title}</title><link>{link}</link>"
                   f'<guid isPermaLink="false">{link[-8:]}</guid>{pub}<description>x</description>{src}</item>')
    out.append("</channel></rss>")
    return "".join(out)


def test_parse_rss():
    when = datetime(2026, 9, 28, 12, 30, tzinfo=timezone.utc)
    rows = N.parse_rss(_rss([("Inno Holdings Announces 1-for-20 Reverse Stock Split - GlobeNewswire", when,
                              "GlobeNewswire", "https://news.google.com/rss/articles/A1?oc=5"),
                             ("No date item - Foo", None, "Foo", "https://news.google.com/rss/articles/A2"),
                             ("", when, "Foo", "https://x"),
                             ("No link", when, "Foo", "")]))
    assert rows == [
        {"published": "2026-09-28T12:30:00+00:00", "title": "Inno Holdings Announces 1-for-20 Reverse Stock Split",
         "source": "GlobeNewswire", "url": "https://news.google.com/rss/articles/A1?oc=5"},
        {"published": None, "title": "No date item", "source": "Foo",
         "url": "https://news.google.com/rss/articles/A2"},
    ]
    assert N.parse_rss("<html><body>blocked</body></html>") is None
    assert N.parse_rss("not xml <") is None


def test_headlines_filters_tags_sorts_and_limits(tmp_cache, monkeypatch):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    items = [
        ("Inno Holdings Announces 1-for-20 Reverse Stock Split - GlobeNewswire", now - timedelta(hours=5),
         "GlobeNewswire", "https://n/1"),
        ("INHD Prices $5 Million Registered Direct Offering - Stock Titan", now - timedelta(hours=1),
         "Stock Titan", "https://n/2"),
        ("INHD prices $5 million registered direct offering - Benzinga", now - timedelta(hours=2),
         "Benzinga", "https://n/3"),                                           # duplicate title
        ("Here's Why GLXY Stock Is Rallying Today? - Zacks", now - timedelta(hours=3), "Zacks", "https://n/4"),
        ("Inno Holdings Q3 Results - scanx", now - timedelta(days=9), "scanx", "https://n/5"),   # too old
        ("Inno Holdings undated - X", None, "X", "https://n/6"),                                # undated
    ] + [(f"INHD update number {i} - Wire", now - timedelta(hours=10 + i), "Wire", f"https://n/x{i}")
         for i in range(15)]
    seen = []

    def fake_get_text(url, **kw):
        seen.append(url)
        return _rss(items)

    monkeypatch.setattr(net, "get_text", fake_get_text)
    out = N.headlines("INHD", "Inno Holdings Inc.")
    q = quote_plus('"INHD" stock OR "Inno Holdings" when:7d')
    assert seen == [f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"]
    assert len(out) == 12
    assert [h["url"] for h in out[:2]] == ["https://n/2", "https://n/1"]
    pubs = [h["published"] for h in out]
    assert pubs == sorted(pubs, reverse=True)
    top = out[0]
    assert set(top) == {"published", "title", "source", "url", "tags", "polarity"}
    assert top["tags"] == ["offering", "priced"] and top["polarity"] == -1
    assert top["title"] == "INHD Prices $5 Million Registered Direct Offering" and top["source"] == "Stock Titan"
    assert datetime.fromisoformat(top["published"]).utcoffset() == timedelta(0)
    assert out[1]["tags"] == ["reverse_split"]
    urls = {h["url"] for h in out}
    assert not urls & {"https://n/3", "https://n/4", "https://n/5", "https://n/6"}
    # cached for 20 minutes
    N.headlines("INHD", "Inno Holdings Inc.")
    assert len(seen) == 1
    assert net.STATUS["Google News"]["ok"] is True


def test_headlines_failure_is_empty_and_reported(tmp_cache, monkeypatch):
    monkeypatch.setattr(net, "get_text", lambda url, **kw: None)
    assert N.headlines("INHD") == []
    st = net.STATUS["Google News"]
    assert st["ok"] is False and "1 failed" in st["detail"]
    monkeypatch.setattr(net, "get_text", lambda url, **kw: "<html>consent wall</html>")
    assert N.headlines("INHD") == []


# ═════════════════════════════════════════════════════════════════════════
# Live
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.live
@live
def test_live_headlines_inhd(tmp_cache):
    out = N.headlines("INHD", "Inno Holdings")
    assert net.STATUS["Google News"]["ok"] is True
    cutoff = datetime.now(timezone.utc) - timedelta(days=7, minutes=5)
    for h in out:
        dt = datetime.fromisoformat(h["published"])
        assert dt.utcoffset() == timedelta(0) and dt >= cutoff
        assert isinstance(h["tags"], list) and h["polarity"] in (-1, 0, 1)
        assert h["url"].startswith("https://") and h["title"]
    print("\nINHD headlines:", len(out))
    for h in out:
        print("  ", h["published"], h["source"], h["tags"], h["polarity"], h["title"][:90])
