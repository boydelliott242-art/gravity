"""GODMODE intel (CONTRACTS.md §14.1): dilution, insiders, chatter, IPO
lock-ups, earnings ahead, and intraday tape for the shortlist.

Sources and what each number means
----------------------------------
* **Dilution** — SEC EDGAR. Share counts are XBRL cover-page counts
  (``sec.shares_history``, point-in-time, *not* split-adjusted); cash and
  burn from ``sec.cash_runway``. The last offering is the newest
  424B1/424B2/424B4/424B5 (or an 8-K/6-K whose text reads like an offering)
  filed within 365 days; its cover page is regex-parsed for the per-share
  price, share count, stated gross proceeds, a stated at-the-market program
  size and warrant terms. Only the cover + "The Offering" summary are read
  (the rest of a prospectus retells *older* deals, which would contaminate
  the numbers). Anything not stated is ``None`` — never computed or guessed.
* **Insiders** — Form 4 XML (open-market/private ``S`` and ``P`` rows only)
  and the count of Form 144 notices, from the issuer's EDGAR submissions.
* **Chatter** — Stocktwits' public symbol stream (last 30 messages).
* **IPO calendar / lock-ups** — Nasdaq's IPO calendar (priced deals). The
  lock-up date is an *assumption*: the standard 180 calendar days after
  pricing. Real lock-ups vary (90 days, staged releases, early releases) —
  every row says so and points the reader to the prospectus.
* **Earnings ahead** — ``street.earnings_calendar`` for the next N sessions.
* **Intraday** — Nasdaq's quote chart (one sample per minute).

Every public function records one ``net.record_status`` row per call and
never raises on network failure (``None`` / ``{}`` / ``[]`` instead).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import xml.etree.ElementTree as XML
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .. import net
from ..util import ET, is_trading_day, next_trading_day, now_et, num
from . import sec, street

log = logging.getLogger(__name__)

# ── status rows (one per feed, shown on the site) ────────────────────────
ST_DILUTION = "SEC offering intel"
ST_INSIDER = "SEC Form 4 insiders"
ST_CHATTER = "Stocktwits chatter"
ST_IPO = "Nasdaq IPO calendar"
ST_LOCKUP = "IPO lock-ups (assumed 180d)"
ST_EARNINGS = "Nasdaq earnings calendar"
ST_INTRADAY = "Nasdaq intraday chart"

# ── caches ───────────────────────────────────────────────────────────────
NS_OFFER = "intel_offering"          # parsed cover facts per filed document (never changes)
PARSER_VERSION = 2                    # bump when parse_offering_text changes → cached parses are redone
NS_FORM4 = "intel_form4"             # parsed Form 4 trades per accession (never changes)
NS_STOCKTWITS = "intel_stocktwits"
NS_IPO = "intel_ipo"
NS_INTRADAY = "intel_intraday"
FOREVER_S = 3650 * 86400
STOCKTWITS_MAX_AGE_S = 30 * 60
IPO_RECENT_MAX_AGE_S = 12 * 3600     # current + previous month can still gain rows
IPO_SETTLED_MAX_AGE_S = 7 * 86400
INTRADAY_MAX_AGE_S = 60

# ── limits ───────────────────────────────────────────────────────────────
OFFERING_LOOKBACK_DAYS = 365
OFFERING_FORMS = ("424B1", "424B2", "424B4", "424B5")
CURRENT_FORMS = ("8-K", "6-K")
OFFER_ITEMS = {"1.01", "3.02", "8.01"}   # 8-K items under which offerings are announced
SAME_DEAL_DAYS = 5                   # an 8-K within this many days after a 424B is the same deal
MAX_OFFER_DOCS = 4                   # prospectus supplements parsed per symbol (last deal + ATM search)
MAX_CURRENT_CHECKS = 6               # 8-K/6-K text checks per symbol
DOC_PARSE_CHARS = 40_000             # cover + summary only
COVER_MIN, COVER_MAX = 1_500, 15_000
COVER_FALLBACK = 8_000
OFFERING_WINDOW = 3_000
MAX_FORM4 = 25
STOCKTWITS_RPS = 2.0
STOCKTWITS_PAGE = 30                 # the public stream returns the last 30 messages
MAX_POINTS = 90
LOCKUP_DAYS = 180
LOCKUP_ASSUMED = "180-day standard lock-up; check the prospectus"

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
STOCKTWITS_PAGE_URL = "https://stocktwits.com/symbol/{sym}"
IPO_URL = street.NASDAQ_API + "/ipo/calendar"
IPO_PAGE_URL = "https://www.nasdaq.com/market-activity/ipos"
CHART_URL = street.NASDAQ_API + "/quote/{sym}/chart"
EDGAR_BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include&count=40"

_STATS: Dict[str, Dict[str, int]] = {
    "dilution": {"ok": 0, "fail": 0},
    "insider": {"ok": 0, "fail": 0},
}
_stats_lock = threading.Lock()


def _bump(kind: str, ok: bool) -> Tuple[int, int]:
    with _stats_lock:
        s = _STATS.setdefault(kind, {"ok": 0, "fail": 0})
        s["ok" if ok else "fail"] += 1
        return s["ok"], s["fail"]


def _today(today: Optional[date]) -> date:
    return today or now_et().date()


def _d(s: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def _resolve_cik(symbol: str, cik: Optional[int]) -> Optional[int]:
    if cik is not None:
        try:
            return int(cik)
        except (TypeError, ValueError):
            return None
    row = sec.cik_map().get((symbol or "").strip().upper())
    return int(row["cik"]) if row and row.get("cik") is not None else None


def _sub_rows(sub: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Submissions ``filings.recent`` columns → row dicts, newest first."""
    rec = ((sub or {}).get("filings") or {}).get("recent") or {}
    forms = rec.get("form") or []
    cols = {k: rec.get(k) or [] for k in ("filingDate", "accessionNumber", "primaryDocument", "items")}
    out = []
    for i, f in enumerate(forms):
        def col(k: str) -> Any:
            v = cols[k]
            return v[i] if i < len(v) else None
        out.append({"form": (f or "").strip().upper(), "date": col("filingDate") or "",
                    "acc": col("accessionNumber") or "", "doc": col("primaryDocument") or "",
                    "items": [x.strip() for x in str(col("items") or "").split(",") if x.strip()]})
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def _sec_text(url: str) -> Optional[str]:
    """Raw document text via the SEC module's shared throttle + breaker
    (falls back to ``net.get_text`` with SEC headers)."""
    if not net.sec_enabled() or not url:
        return None
    getter = getattr(sec, "_sec_get", None)
    if getter is not None:
        r = getter(url, timeout=45.0)
        return None if r is None else r.text
    return net.get_text(url, headers=net.sec_headers(), timeout=45.0)


# ═════════════════════════════════════════════════════════════════════════
# Offering cover-page parsing
# ═════════════════════════════════════════════════════════════════════════
_N = r"(\d[\d,]*(?:\.\d+)?)"
_USD = r"(?:us)?\$\s?"
_SCALE = r"(?:\s*(million|billion))?"
_UNIT_WORD = r"(?:share|unit|ads|american depositary share|ordinary share|common share)"

_PRICE_RES = (
    re.compile(r"(?<!assumed )(?:combined )?(?:public )?offering price (?:is |of |will be |equal to )?"
               + _USD + _N + r" per " + _UNIT_WORD),
    re.compile(r"\b(?:at|for) an? (?:offering |purchase |public offering |combined )?price of "
               + _USD + _N + r" per " + _UNIT_WORD),
)
_OFFER_SECTION_PRICE_RE = re.compile(
    r"shares of (?:our )?(?:common stock|ordinary shares|common shares) at (?:a price of )?" + _USD + _N + r" per share")
_SHARES_RES = (
    re.compile(r"\b(?:we are|is|are) (?:hereby )?offering (?:up to )?(?:an aggregate of )?(\d[\d,]{2,}) "
               r"(?:shares|of (?:our|its) (?:ordinary|common) shares|ordinary shares|common shares|"
               r"american depositary shares|units)"),
    re.compile(r"\b(?:agreed to|will) (?:issue and )?sell[^.;]{0,80}?(?:an aggregate of |up to )?(\d[\d,]{2,}) "
               r"(?:shares|ordinary shares|common shares)"),
)
_OFFER_SECTION_SHARES_RE = re.compile(
    r"(?:common stock|ordinary shares|common shares|shares|securities) (?:being )?(?:offered|issued) by us:?\s*"
    r"(?:up to )?(?:an aggregate of )?(\d[\d,]{2,}) (?:shares|ordinary shares)")
_TITLE_SHARES_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+) (?:shares of (?:class a )?(?:common stock|ordinary shares)|"
                              r"ordinary shares|common shares|american depositary shares)")
_GROSS_RE = re.compile(
    r"(?:aggregate |total )?gross proceeds(?: to us)?(?: from (?:this|the) offering)?"
    r"(?: (?:will|are expected to|is expected to|would) be| of| are| were| is| totaling| totalling)?"
    r"(?: approximately| up to| about)? " + _USD + _N + _SCALE)
_TABLE_GROSS_RE = re.compile(r"public offering price(?: and proceeds[^$]{0,40})?((?:\s*\$\s?\d[\d,]*(?:\.\d+)?){2,4})")
_ATM_CAP_RE = re.compile(r"aggregate (?:gross )?(?:offering|sales) price of up to " + _USD + _N + _SCALE)
_ATM_TITLE_RE = re.compile(r"\bup to " + _USD + _N + _SCALE + r" (?:of )?(?:shares of )?(?:our )?"
                           r"(?:common stock|ordinary shares|common shares|american depositary shares)")
_ATM_RE = re.compile(r"\bat[- ]the[- ]market\b")
_ATM_AGREEMENT_RE = re.compile(r"\bsales agent\b|\bsales agreement\b|\bequity distribution agreement\b|"
                               r"\bcontrolled equity offering\b|\bat[- ]the[- ]market (?:offering|program|issuance)\b")
_WARRANT_RE = re.compile(r"((?:pre-funded|prefunded|common|series [a-z0-9-]+|placement agent|underwriter'?s'?|"
                         r"representative'?s'?)\s)?warrants to purchase (?:up to )?(?:an aggregate of )?"
                         r"(\d[\d,]{2,}) (?:shares|ordinary shares|common shares)")
_EXERCISE_RE = re.compile(r"exercise price (?:of |per share (?:of |equal to |is )?|equal to |is |will be )?"
                          + _USD + _N)
_OUTSTANDING_WARRANTS_RE = re.compile(
    r"(\d[\d,]{2,}) (?:shares (?:of (?:our )?common stock |of ordinary shares )?|ordinary shares )?issuable upon "
    r"(?:the )?exercise of (?:outstanding )?warrants(?: outstanding)?(?: as of [a-z]+ \d{1,2}, \d{4})?,? "
    r"(?:with|at|having) (?:a )?weighted[- ]average exercise price of " + _USD + _N)
_SHARES_AFTER_RE = re.compile(
    r"(?:common stock|ordinary shares|common shares|shares(?: of common stock)?) (?:to be )?outstanding "
    r"(?:immediately )?after (?:this|the) offering:?\s*(?:\(\d\)\s*)?(?:up to )?(?:approximately )?(\d[\d,]{2,}) shares")
_OFFERING_SECTION_RE = re.compile(
    r"\bthe offering (?:common stock|ordinary shares|shares|securities|common shares|units|"
    r"american depositary shares|pre-funded)")
_QUOTES_RE = re.compile(r"[“”\"]")
_SPACES_RE = re.compile(r"\s+")


def _num(s: Optional[str], scale: Optional[str] = None) -> Optional[float]:
    if not s:
        return None
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    if scale == "million":
        v *= 1e6
    elif scale == "billion":
        v *= 1e9
    return v


def normalize_text(text: str) -> str:
    """``sec.html_to_text`` output → quote-free, single-spaced lower case
    (EDGAR covers wrap defined terms in curly quotes: “at the market”)."""
    t = _QUOTES_RE.sub(" ", (text or "").lower()).replace("’", "'")
    return _SPACES_RE.sub(" ", t).strip()


def cover_and_summary(text: str) -> Tuple[str, str]:
    """(cover page, "The Offering" summary window) from normalised text.
    The cover ends at the first "table of contents" after the title block;
    later sections (recent developments, capitalization) describe *earlier*
    deals and are not read except for the explicit patterns that name them."""
    head = text[:DOC_PARSE_CHARS]
    i = head.find("table of contents", COVER_MIN)
    cover = head[:i] if COVER_MIN <= i <= COVER_MAX else head[:COVER_FALLBACK]
    m = _OFFERING_SECTION_RE.search(head, len(cover))
    summary = text[m.start(): m.start() + OFFERING_WINDOW] if m else ""
    return cover, summary


def _first(res: Iterable["re.Pattern[str]"], *texts: str) -> Optional["re.Match[str]"]:
    for t in texts:
        if not t:
            continue
        for rx in res:
            m = rx.search(t)
            if m:
                return m
    return None


def _price(cover: str, summary: str) -> Optional[float]:
    """Per-share deal price. Never the par value ("par value $0.0001 per
    share" sits on every cover page) and never a sub-cent figure."""
    for text, pats in ((cover, _PRICE_RES), (summary, (_OFFER_SECTION_PRICE_RE,) + _PRICE_RES)):
        if not text:
            continue
        for rx in pats:
            for m in rx.finditer(text):
                ctx = text[max(0, m.start() - 30): m.end()]
                if "par value" in ctx:
                    continue
                v = _num(m.group(1))
                if v is not None and 0.01 <= v < 100_000:
                    return v
    return None


def _shares(cover: str, summary: str) -> Optional[float]:
    m = _first(_SHARES_RES, cover) or _first((_OFFER_SECTION_SHARES_RE,) + _SHARES_RES, summary)
    if m is None:
        # title block: "fingermotion, inc. 3,958,055 shares of common stock pre-funded warrants …"
        for t in _TITLE_SHARES_RE.finditer(cover[:1500]):
            before = cover[max(0, t.start() - 40): t.start()]
            if not re.search(r"purchase|issuable|exercise|up to $", before):
                m = t
                break
    v = _num(m.group(1)) if m else None
    return v if v is not None and v >= 100 else None


def _gross(cover: str, summary: str) -> Optional[float]:
    for t in (cover, summary):
        if not t:
            continue
        m = _GROSS_RE.search(t)
        if m:
            v = _num(m.group(1), m.group(2))
            if v is not None and v >= 1_000:
                return v
        m = _TABLE_GROSS_RE.search(t)
        if m and "total" in t[max(0, m.start() - 300): m.start() + 20]:
            vals = [_num(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", m.group(1))]
            vals = [v for v in vals if v is not None]
            if vals and max(vals) >= 1_000:
                return max(vals)
    return None


def _atm_capacity(cover: str) -> Optional[float]:
    m = _ATM_CAP_RE.search(cover) or _ATM_TITLE_RE.search(cover[:1500])
    v = _num(m.group(1), m.group(2)) if m else None
    return v if v is not None and v >= 100_000 else None


def _warrants(cover: str, summary: str, head: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """(offered or outstanding warrant overhang, pre-funded warrants offered).
    Pre-funded warrants (exercise ≈ $0.0001) are economically shares, so they
    are reported with the offering, not as an overhang."""
    tranches: List[Dict[str, Any]] = []
    pf: Optional[float] = None
    seen = set()
    for t in (cover, summary):
        for m in _WARRANT_RE.finditer(t or ""):
            kind = (m.group(1) or "").strip()
            n = _num(m.group(2))
            if n is None or n < 100:
                continue
            if kind.startswith(("pre-funded", "prefunded")):
                pf = pf if pf is not None else n
                continue
            if kind.startswith(("placement agent", "underwriter", "representative")):
                continue
            ex = None
            for e in _EXERCISE_RE.finditer(t[m.end(): m.end() + 600]):
                v = _num(e.group(1))
                if v is not None and v >= 0.01:        # skip the pre-funded $0.0001 strike
                    ex = v
                    break
            key = (n, ex)
            if key in seen:
                continue
            seen.add(key)
            tranches.append({"shares": n, "exercise_price": ex, "series": kind or None})
    if tranches:
        return ({"shares": sum(x["shares"] for x in tranches),
                 "exercise_price": tranches[0]["exercise_price"],
                 "kind": "offered", "tranches": tranches}, pf)
    m = _OUTSTANDING_WARRANTS_RE.search(head)
    if m:
        n, ex = _num(m.group(1)), _num(m.group(2))
        if n is not None and n >= 100:
            return ({"shares": n, "exercise_price": ex, "kind": "outstanding",
                     "basis": "weighted-average exercise price of outstanding warrants, as stated in the filing"}, pf)
    return None, pf


def _shares_after(summary: str) -> Optional[float]:
    m = _SHARES_AFTER_RE.search(summary or "")
    v = _num(m.group(1)) if m else None
    return v if v is not None and v >= 100 else None


def offer_type(form: str, cover: str, parsed: Dict[str, Any], tags: Sequence[str] = ()) -> str:
    """'atm' | 'registered_direct' | 'priced' | 'shelf_takedown' | 'other'."""
    base = (form or "").upper().split("/")[0]
    atm_text = bool(_ATM_RE.search(cover)) or "atm" in tags
    if atm_text and (parsed.get("atm_capacity") is not None or _ATM_AGREEMENT_RE.search(cover)):
        return "atm"
    if "registered direct" in cover or "registered_direct" in tags:
        return "registered_direct"
    if base in CURRENT_FORMS:
        if "private placement" in cover and "public_offering_priced" not in tags:
            return "other"
        if "public_offering_priced" in tags or parsed.get("price") is not None:
            return "priced"
        return "other"
    if "securities purchase agreement" in cover and "underwrit" not in cover:
        return "registered_direct"
    if parsed.get("price") is not None or "underwriting agreement" in cover or "underwriters" in cover:
        return "priced"
    if base in ("424B2", "424B5"):
        return "shelf_takedown"
    return "other"


def parse_offering_text(text: str, form: str = "424B5", tags: Sequence[str] = ()) -> Dict[str, Any]:
    """Cover-page facts from a filing's text (``sec.html_to_text`` output or
    raw lower-case text). Every field is ``None`` unless stated."""
    t = normalize_text(text)
    cover, summary = cover_and_summary(t)
    head = t[:DOC_PARSE_CHARS]
    parsed: Dict[str, Any] = {
        "price": _price(cover, summary),
        "shares": _shares(cover, summary),
        "gross": _gross(cover, summary),
        "atm_capacity": None,
    }
    atm_cap = _atm_capacity(cover)
    if atm_cap is not None and _ATM_RE.search(cover):
        parsed["atm_capacity"] = atm_cap
    warrants, pf = _warrants(cover, summary, head)
    parsed["pf_warrants"] = pf
    parsed["warrants"] = warrants
    parsed["type"] = offer_type(form, cover, parsed, tags)
    parsed["shares_after"] = _shares_after(summary) if parsed["type"] != "atm" else None
    if parsed["type"] == "atm":
        # an ATM sells at market prices over time: a per-share or gross figure on
        # its cover is an example/assumption, not the deal
        parsed["price"] = None
        parsed["gross"] = None
        parsed["shares"] = None
    return parsed


# ═════════════════════════════════════════════════════════════════════════
# Dilution intel
# ═════════════════════════════════════════════════════════════════════════
def _doc_text(url: str, with_exhibits: bool) -> Optional[str]:
    """Primary document text (+ EX-99 exhibits when the primary document is
    a short cover page, typical of 6-K/8-K)."""
    raw = _sec_text(url)
    if raw is None:
        return None
    text = sec.html_to_text(raw[: sec.MAX_DOC_CHARS])
    if with_exhibits and len(text) < sec.COVER_MAX_CHARS:
        idx = sec.index_url_for(url)
        page = _sec_text(idx) if idx else None
        if page:
            for doc in [d for d in sec.index_documents(page) if d != url][: sec.MAX_INDEX_DOCS - 1]:
                more = _sec_text(doc)
                if more:
                    text += " " + sec.html_to_text(more[: sec.MAX_DOC_CHARS])
    return text


def _parse_filing(url: str, form: str, tags: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """Parsed cover facts for one filed document, cached forever (a filed
    document never changes). None when it could not be read."""
    key = f"v{PARSER_VERSION}|{url}|{form}|{','.join(sorted(tags))}"

    def fetch() -> Optional[Dict[str, Any]]:
        text = _doc_text(url, with_exhibits=form.upper().split("/")[0] in CURRENT_FORMS)
        return None if text is None else parse_offering_text(text, form, tags)

    return net.cached(NS_OFFER, key, FOREVER_S, fetch)


_OFFER_TAGS = {"registered_direct", "public_offering_priced", "atm", "securities_purchase_agreement"}


def _current_offering(row: Dict[str, Any], cik: int) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """(parsed, tags) when an 8-K/6-K reads like an offering, else None."""
    url = sec.filing_url(cik, row["acc"], row["doc"] or None)
    tags = sec.text_classify(url)
    hit = sorted(set(tags or ()) & _OFFER_TAGS)
    if not hit:
        return None
    parsed = _parse_filing(url, row["form"], hit)
    if parsed is None:
        return None
    strong = {"registered_direct", "public_offering_priced"} & set(hit)
    has_numbers = any(parsed.get(k) is not None for k in ("price", "shares", "gross", "atm_capacity"))
    if not strong and not has_numbers:
        return None            # an SPA/ATM mention with no deal terms (amendment, recap, …)
    return parsed, hit


def last_offerings(rows: List[Dict[str, Any]], cik: int, today: date) -> Dict[str, Any]:
    """{"last_offering", "atm", "warrants", "sources", "docs_read", "docs_failed"}."""
    start = (today - timedelta(days=OFFERING_LOOKBACK_DAYS)).isoformat()
    recent = [r for r in rows if r["date"] >= start and r["date"] <= today.isoformat()]
    prospectuses = [r for r in recent if r["form"] in OFFERING_FORMS and r["doc"]]
    out: Dict[str, Any] = {"last_offering": None, "atm": None, "warrants": None, "sources": [],
                           "docs_read": 0, "docs_failed": 0}

    # 8-K/6-K offerings newer than the newest prospectus (and not its own announcement)
    newest_pro = prospectuses[0]["date"] if prospectuses else None
    cutoff = None
    if newest_pro:
        cutoff = (_d(newest_pro) + timedelta(days=SAME_DEAL_DAYS)).isoformat()  # type: ignore[operator]
    currents = [r for r in recent if r["form"] in CURRENT_FORMS and r["doc"]
                and (r["form"] == "6-K" or set(r["items"]) & OFFER_ITEMS)
                and (newest_pro is None or r["date"] > cutoff)]
    for r in currents[:MAX_CURRENT_CHECKS]:
        got = _current_offering(r, cik)
        if got is None:
            continue
        parsed, _tags = got
        url = sec.filing_url(cik, r["acc"], r["doc"])
        out["last_offering"] = _offering_row(r, url, parsed)
        out["sources"].append(url)
        if parsed.get("warrants"):
            out["warrants"] = dict(parsed["warrants"], url=url, date=r["date"])
        if parsed["type"] == "atm" and parsed.get("atm_capacity") is not None:
            out["atm"] = {"capacity": parsed["atm_capacity"], "date": r["date"], "url": url}
        break

    for r in prospectuses[:MAX_OFFER_DOCS]:
        if out["last_offering"] is not None and out["atm"] is not None and out["warrants"] is not None:
            break
        url = sec.filing_url(cik, r["acc"], r["doc"])
        parsed = _parse_filing(url, r["form"])
        if parsed is None:
            out["docs_failed"] += 1
            continue
        out["docs_read"] += 1
        used = False
        if out["last_offering"] is None:
            out["last_offering"] = _offering_row(r, url, parsed)
            used = True
            if parsed.get("warrants") and out["warrants"] is None:
                out["warrants"] = dict(parsed["warrants"], url=url, date=r["date"])
        if out["atm"] is None and parsed["type"] == "atm" and parsed.get("atm_capacity") is not None:
            out["atm"] = {"capacity": parsed["atm_capacity"], "date": r["date"], "url": url}
            used = True
        if out["warrants"] is None and parsed.get("warrants") and parsed["warrants"].get("kind") == "outstanding":
            out["warrants"] = dict(parsed["warrants"], url=url, date=r["date"])
            used = True
        if used and url not in out["sources"]:
            out["sources"].append(url)
    return out


def _offering_row(r: Dict[str, Any], url: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {"date": r["date"], "form": r["form"], "url": url, "type": parsed.get("type") or "other",
            "price": parsed.get("price"), "shares": parsed.get("shares"), "gross": parsed.get("gross"),
            "pf_warrants": parsed.get("pf_warrants"), "shares_after": parsed.get("shares_after")}


def share_growth(hist: List[Dict[str, Any]], splits: Optional[List[Dict[str, Any]]] = None,
                 rows: Optional[List[Dict[str, Any]]] = None, days: int = 365) -> Dict[str, Any]:
    """1-year share-count change from point-in-time XBRL cover counts.

    Counts are *as reported* (not split-adjusted). With ``splits``
    ([{"date", "ratio"}], ratio < 1 = reverse) the older count is put on the
    newer count's basis. Without them, a split cannot be ruled out: growth is
    withheld (None) when the count fell below 80% of the old one or an 8-K
    item 5.03 (charter amendment — how reverse splits are effected) was filed
    between the two dates."""
    out: Dict[str, Any] = {"shares_now": None, "shares_asof": None, "shares_1y_ago": None,
                           "shares_1y_ago_date": None, "shares_growth_1y": None,
                           "split_factor": None, "split_checked": splits is not None, "shares_note": None}
    pts = sorted([h for h in hist or [] if _d(h.get("date")) and (h.get("shares") or 0) > 0],
                 key=lambda h: h["date"])
    if not pts:
        out["shares_note"] = "no XBRL share count on file"
        return out
    now = pts[-1]
    d_now = _d(now["date"])
    out["shares_now"], out["shares_asof"] = float(now["shares"]), now["date"]
    target = d_now - timedelta(days=days)  # type: ignore[operator]
    cands = [h for h in pts[:-1] if target - timedelta(days=120) <= _d(h["date"]) <= target + timedelta(days=45)]  # type: ignore[operator]
    if not cands:
        out["shares_note"] = "no share count about a year earlier"
        return out
    old = min(cands, key=lambda h: abs((_d(h["date"]) - target).days))  # type: ignore[operator]
    out["shares_1y_ago"], out["shares_1y_ago_date"] = float(old["shares"]), old["date"]
    raw = float(now["shares"]) / float(old["shares"])
    if splits is not None:
        factor = 1.0
        for s in splits:
            sd, ratio = _d(s.get("date")), num(s.get("ratio"))
            if sd is None or ratio is None or ratio <= 0:
                continue
            if _d(old["date"]) < sd <= d_now:  # type: ignore[operator]
                factor *= ratio
        out["split_factor"] = factor
        out["shares_growth_1y"] = round(raw / factor, 4)
        if factor != 1.0:
            out["shares_note"] = f"older count adjusted for splits (factor {factor:g})"
        return out
    charter = [r["date"] for r in rows or [] if r["form"] == "8-K" and "5.03" in r["items"]
               and old["date"] < r["date"] <= now["date"]]
    if raw < 0.8 or charter:
        why = "share count fell" if raw < 0.8 else f"8-K item 5.03 filed {charter[0]}"
        out["shares_note"] = f"possible split between the two counts ({why}); growth withheld — raw counts shown"
        return out
    out["shares_growth_1y"] = round(raw, 4)
    out["shares_note"] = "split list not supplied; no split sign found"
    return out


def dilution_intel(symbol: str, cik: Optional[int], splits: Optional[List[Dict[str, Any]]] = None,
                   *, today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Share growth, cash runway and the last offering's terms (§14.1).
    None when SEC is disabled or nothing at all could be read for the name."""
    if not net.sec_enabled():
        net.record_status(ST_DILUTION, False, "SEC_USER_AGENT not set")
        return None
    today = _today(today)
    cik = _resolve_cik(symbol, cik)
    if cik is None:
        ok, fail = _bump("dilution", False)
        net.record_status(ST_DILUTION, ok > 0, f"{ok} names read, {fail} failed (last: {symbol} has no CIK)")
        return None
    try:
        hist = sec.shares_history(cik)
        runway = sec.cash_runway(cik)
        sub = sec.submissions(cik)
        rows = _sub_rows(sub)
        growth = share_growth(hist, splits, rows)
        offers = last_offerings(rows, cik, today) if rows else None
    except Exception as e:  # noqa: BLE001 — a parser bug must not sink the run
        log.warning("dilution_intel %s: %s", symbol, e)
        ok, fail = _bump("dilution", False)
        net.record_status(ST_DILUTION, ok > 0, f"{ok} names read, {fail} failed (last: {symbol} error)")
        return None
    if not hist and runway is None and not rows:
        ok, fail = _bump("dilution", False)
        net.record_status(ST_DILUTION, ok > 0, f"{ok} names read, {fail} failed (last: {symbol} unavailable)")
        return None

    sources: List[str] = []
    if hist or runway is not None:
        sources.append(sec.FACTS_URL.format(cik=cik))
    out: Dict[str, Any] = {"symbol": symbol, "cik": cik}
    out.update({k: growth[k] for k in ("shares_now", "shares_1y_ago", "shares_growth_1y", "shares_asof",
                                       "shares_1y_ago_date", "split_factor", "split_checked", "shares_note")})
    rw = runway or {}
    out.update({"cash": rw.get("cash"), "cash_date": rw.get("cash_date"),
                "quarterly_burn": rw.get("quarterly_burn"), "runway_q": rw.get("runway_q"),
                "currency": rw.get("currency")})
    o = offers or {}
    out["last_offering"] = o.get("last_offering")
    atm = o.get("atm")
    out["atm_capacity"] = atm["capacity"] if atm else None
    out["atm_date"] = atm["date"] if atm else None
    out["atm_url"] = atm["url"] if atm else None
    out["warrants"] = o.get("warrants")
    sources.extend(u for u in o.get("sources") or [] if u not in sources)
    out["sources"] = sources
    out["offerings_checked_since"] = (today - timedelta(days=OFFERING_LOOKBACK_DAYS)).isoformat()
    out["asof"] = today.isoformat()
    ok, fail = _bump("dilution", True)
    lo = out["last_offering"]
    last = f"{lo['form']} {lo['date']} ({lo['type']})" if lo else "no offering in 365d"
    net.record_status(ST_DILUTION, True, f"{ok} names read, {fail} failed (last: {symbol}: {last})")
    return out


# ═════════════════════════════════════════════════════════════════════════
# Insider sales (Form 4 + Form 144)
# ═════════════════════════════════════════════════════════════════════════
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: Optional[XML.Element], *path: str) -> Optional[XML.Element]:
    cur = el
    for name in path:
        if cur is None:
            return None
        nxt = None
        for c in list(cur):
            if _local(c.tag) == name:
                nxt = c
                break
        cur = nxt
    return cur


def _text(el: Optional[XML.Element], *path: str) -> Optional[str]:
    node = _child(el, *path)
    if node is None:
        return None
    v = _child(node, "value")
    s = (v.text if v is not None else node.text) or ""
    s = s.strip()
    return s or None


def _flag(el: Optional[XML.Element], *path: str) -> bool:
    return (_text(el, *path) or "").lower() in ("1", "true")


def parse_form4(xml_text: str) -> Optional[Dict[str, Any]]:
    """Form 4 XML → {"owners": [...], "trades": [...]} with only ``S``/``P``
    non-derivative rows. Direction comes from the acquired/disposed code
    when the filer's P/S code contradicts it (seen in real filings); such
    rows carry ``"mismatch": True``. None when the XML cannot be parsed."""
    try:
        root = XML.fromstring(xml_text.strip().encode("utf-8"))
    except (XML.ParseError, ValueError, AttributeError):
        return None
    if _local(root.tag) != "ownershipDocument":
        return None
    owners = []
    for ro in [c for c in root if _local(c.tag) == "reportingOwner"]:
        name = _text(ro, "reportingOwnerId", "rptOwnerName")
        rel = _child(ro, "reportingOwnerRelationship")
        roles = []
        if _flag(rel, "isOfficer"):
            roles.append(_text(rel, "officerTitle") or "Officer")
        if _flag(rel, "isDirector"):
            roles.append("Director")
        if _flag(rel, "isTenPercentOwner"):
            roles.append("10% owner")
        if _flag(rel, "isOther"):
            roles.append(_text(rel, "otherText") or "Other")
        owners.append({"name": name, "title": ", ".join(roles) or None})
    plan = _text(root, "aff10b5One")
    trades = []
    table = _child(root, "nonDerivativeTable")
    for tx in [c for c in (list(table) if table is not None else []) if _local(c.tag) == "nonDerivativeTransaction"]:
        code = (_text(tx, "transactionCoding", "transactionCode") or "").upper()
        if code not in ("S", "P"):
            continue
        ad = (_text(tx, "transactionAmounts", "transactionAcquiredDisposedCode") or "").upper() or None
        shares = num(_text(tx, "transactionAmounts", "transactionShares"))
        price = num(_text(tx, "transactionAmounts", "transactionPricePerShare"))
        side = "sell" if code == "S" else "buy"
        mismatch = (code == "S" and ad == "A") or (code == "P" and ad == "D")
        if mismatch:
            side = "sell" if ad == "D" else "buy"
        trades.append({
            "date": _text(tx, "transactionDate"),
            "code": code, "ad": ad, "side": side, "mismatch": mismatch,
            "shares": shares,
            "price": price if price is not None and price > 0 else None,
            "value": round(shares * price, 2) if shares and price else None,
            "owned_after": num(_text(tx, "postTransactionAmounts", "sharesOwnedFollowingTransaction")),
        })
    return {"owners": owners, "trades": trades,
            "plan_10b5_1": None if plan is None else plan.lower() in ("1", "true")}


def form4_xml_url(cik: int, accession: str, primary_doc: str) -> Optional[str]:
    """Raw XML of a Form 4: the ``xslF345X0*/`` prefix is EDGAR's rendered
    view — drop it. None for pre-XML (text) filings."""
    base = (primary_doc or "").rsplit("/", 1)[-1]
    if not base.lower().endswith(".xml"):
        return None
    return sec.filing_url(cik, accession, base)


def _form4(cik: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = form4_xml_url(cik, row["acc"], row["doc"])
    if url is None:
        return None

    def fetch() -> Optional[Dict[str, Any]]:
        raw = _sec_text(url)
        return None if raw is None else parse_form4(raw)

    return net.cached(NS_FORM4, row["acc"], FOREVER_S, fetch)


def _is_fpi(rows: List[Dict[str, Any]], today: date) -> bool:
    start = (today - timedelta(days=730)).isoformat()
    forms = {r["form"] for r in rows if r["date"] >= start}
    return bool(forms & {"20-F", "6-K", "40-F"}) and not (forms & {"10-K", "10-Q"})


def insider_sales(symbol: str, cik: Optional[int], days: int = 90,
                  *, today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """Open-market insider trades (Form 4 ``S``/``P``) and Form 144 notices
    filed in the last ``days`` days. ``last_date`` = most recent sale date.
    For foreign private issuers (exempt from Section 16) the counts are None."""
    if not net.sec_enabled():
        net.record_status(ST_INSIDER, False, "SEC_USER_AGENT not set")
        return None
    today = _today(today)
    cik = _resolve_cik(symbol, cik)
    sub = sec.submissions(cik) if cik is not None else None
    if sub is None:
        ok, fail = _bump("insider", False)
        net.record_status(ST_INSIDER, ok > 0, f"{ok} names read, {fail} failed (last: {symbol} unavailable)")
        return None
    rows = _sub_rows(sub)
    start = (today - timedelta(days=days)).isoformat()
    window = [r for r in rows if start <= r["date"] <= today.isoformat()]
    f4 = [r for r in window if r["form"] == "4"]
    n_144 = sum(1 for r in window if r["form"] in ("144", "144/A"))
    base = {"symbol": symbol, "cik": cik, "days": days, "since": start, "n_144": n_144,
            "n_form4": len(f4), "source": EDGAR_BROWSE_URL.format(cik=cik, form="4")}
    if _is_fpi(rows, today) and not f4:
        ok, fail = _bump("insider", True)
        net.record_status(ST_INSIDER, True, f"{ok} names read, {fail} failed (last: {symbol} foreign private issuer)")
        return dict(base, n_sales=None, shares_sold=None, value_sold=None, last_date=None, n_buys=None,
                    value_bought=None, trades=[], n_form4_read=0, fpi=True,
                    note="foreign private issuer — insiders are exempt from Form 4")
    trades: List[Dict[str, Any]] = []
    read = failed = 0
    for r in f4[:MAX_FORM4]:
        doc = _form4(cik, r)  # type: ignore[arg-type]
        if doc is None:
            failed += 1
            continue
        read += 1
        owners = doc.get("owners") or []
        name = "; ".join(o["name"] for o in owners if o.get("name")) or None
        title = "; ".join(o["title"] for o in owners if o.get("title")) or None
        url = sec.filing_url(cik, r["acc"], r["doc"])  # type: ignore[arg-type]
        for t in doc.get("trades") or []:
            trades.append(dict(t, name=name, title=title, url=url, filed=r["date"],
                               plan_10b5_1=doc.get("plan_10b5_1")))
    trades.sort(key=lambda t: (t.get("date") or "", t.get("filed") or ""), reverse=True)
    sells = [t for t in trades if t["side"] == "sell"]
    buys = [t for t in trades if t["side"] == "buy"]

    def total(ts: List[Dict[str, Any]], k: str) -> Optional[float]:
        vals = [t[k] for t in ts if t.get(k) is not None]
        return round(sum(vals), 2) if vals else (0.0 if not ts else None)

    out = dict(base, n_sales=len(sells), shares_sold=total(sells, "shares"), value_sold=total(sells, "value"),
               last_date=sells[0]["date"] if sells else None, n_buys=len(buys),
               shares_bought=total(buys, "shares"), value_bought=total(buys, "value"),
               last_buy_date=buys[0]["date"] if buys else None,
               n_unpriced=sum(1 for t in trades if t.get("value") is None),
               n_form4_read=read, n_form4_failed=failed, truncated=len(f4) > MAX_FORM4, fpi=False,
               trades=trades)
    good = failed == 0 or read > 0
    ok, fail = _bump("insider", good)
    net.record_status(ST_INSIDER, ok > 0,
                      f"{ok} names read, {fail} failed (last: {symbol}: {len(sells)} sales, {len(buys)} buys, "
                      f"{n_144} Form 144 in {days}d; {read}/{len(f4[:MAX_FORM4])} Form 4 read)")
    return out


# ═════════════════════════════════════════════════════════════════════════
# Stocktwits chatter
# ═════════════════════════════════════════════════════════════════════════
_st_lock = threading.Lock()
_st_next = 0.0


def _st_throttle() -> None:
    global _st_next
    with _st_lock:
        now = time.monotonic()
        slot = max(now, _st_next)
        _st_next = slot + 1.0 / STOCKTWITS_RPS
    if slot > now:
        time.sleep(slot - now)


def _st_get(sym: str) -> Tuple[Optional[int], Any]:
    """(HTTP status or None, parsed JSON or None). One attempt: a 429 must
    stop the batch, not be retried into a longer ban."""
    _st_throttle()
    try:
        r = requests.get(STOCKTWITS_URL.format(sym=sym), timeout=15,
                         headers={"User-Agent": net.BROWSER_UA, "Accept": "application/json"})
    except requests.RequestException:
        return None, None
    try:
        return r.status_code, (r.json() if r.status_code == 200 else None)
    except ValueError:
        return r.status_code, None


def _iso_utc(s: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_stream(sym: str, payload: Any, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Stocktwits stream JSON → chatter row. ``msgs_per_day``: when the page
    is full (30 messages) and they all fall inside the last 24 h, the pace
    30 / span; otherwise the exact count of messages in the last 24 h (a
    full page older than 24 h covers the whole day)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return None
    now = now or datetime.now(timezone.utc)
    msgs = [m for m in payload["messages"] if isinstance(m, dict)]
    times = [t for t in (_iso_utc(m.get("created_at")) for m in msgs) if t is not None]
    bull = bear = 0
    for m in msgs:
        basic = (((m.get("entities") or {}).get("sentiment") or {}) or {}).get("basic")
        if basic == "Bullish":
            bull += 1
        elif basic == "Bearish":
            bear += 1
    watchers = (payload.get("symbol") or {}).get("watchlist_count")
    row: Dict[str, Any] = {"msgs_per_day": 0.0, "span_hours": None, "bull": bull, "bear": bear,
                           "n": len(msgs), "watchers": int(watchers) if isinstance(watchers, (int, float)) else None,
                           "latest": None, "basis": "count_24h", "asof": now.isoformat(timespec="seconds"),
                           "url": STOCKTWITS_PAGE_URL.format(sym=sym)}
    if not times:
        return row
    oldest, newest = min(times), max(times)
    span_h = max((now - oldest).total_seconds() / 3600.0, 0.0)
    row["span_hours"] = round(span_h, 2)
    row["latest"] = newest.isoformat(timespec="seconds")
    if len(times) >= STOCKTWITS_PAGE and span_h < 24:
        row["msgs_per_day"] = round(len(times) / max(span_h, 0.25) * 24.0, 1)
        row["basis"] = "pace_30_msgs"
    else:
        row["msgs_per_day"] = float(sum(1 for t in times if (now - t) <= timedelta(hours=24)))
    return row


def chatter(symbols: List[str], *, now: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
    """Stocktwits message pace, sentiment tags and watchers per symbol.
    ≤ 2 req/s, 30-min cache, stops the batch on HTTP 429."""
    out: Dict[str, Dict[str, Any]] = {}
    fetched = cached = failed = 0
    limited = False
    for sym in symbols:
        st_sym = (sym or "").strip().upper().replace("-", ".")
        if not st_sym:
            continue
        hit = net.cache_get(NS_STOCKTWITS, st_sym, STOCKTWITS_MAX_AGE_S)
        if hit is not None:
            out[sym] = hit
            cached += 1
            continue
        if limited:
            continue
        status, payload = _st_get(st_sym)
        if status == 429:
            limited = True
            continue
        row = parse_stream(st_sym, payload, now) if status == 200 else None
        if row is None:
            failed += 1
            continue
        net.cache_set(NS_STOCKTWITS, st_sym, row)
        out[sym] = row
        fetched += 1
    detail = f"{len(out)}/{len(symbols)} symbols ({fetched} fetched, {cached} cached, {failed} failed)"
    if limited:
        detail += "; stopped on HTTP 429 rate limit"
    net.record_status(ST_CHATTER, bool(out) or (not symbols), detail)
    return out


# ═════════════════════════════════════════════════════════════════════════
# IPO calendar and lock-ups
# ═════════════════════════════════════════════════════════════════════════
_SPAC_NAME_RE = re.compile(r"\bacquisition (?:corp|corporation|co|company|inc|ltd|limited)\b|\bblank check\b|"
                           r"\bspac\b|\bmerger (?:corp|corporation)\b|\bunits?\b", re.I)


def _is_spac_or_unit(sym: str, name: str) -> bool:
    s = (sym or "").upper()
    if len(s) == 5 and s[-1] in "UWR":
        return True
    if s.endswith((".U", "-U", ".WS", "-WS", ".W", "-W", ".R", "-R")):
        return True
    return bool(_SPAC_NAME_RE.search(name or ""))


def _us_date(s: Any) -> Optional[str]:
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", str(s or ""))
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat()
    except ValueError:
        return None


def parse_ipo_month(payload: Any) -> Optional[List[Dict[str, Any]]]:
    """Nasdaq IPO calendar JSON → priced operating-company IPOs (SPACs and
    units excluded). None = unrecognised payload; [] = none priced."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    priced = payload["data"].get("priced") or {}
    out = []
    for r in priced.get("rows") or []:
        if not isinstance(r, dict):
            continue
        sym = (r.get("proposedTickerSymbol") or "").strip().upper()
        name = (r.get("companyName") or "").strip()
        pd_ = _us_date(r.get("pricedDate"))
        if not sym or not pd_ or _is_spac_or_unit(sym, name):
            continue
        price = num(r.get("proposedSharePrice"))
        shares = num(r.get("sharesOffered"))
        amount = num(r.get("dollarValueOfSharesOffered"))
        out.append({"symbol": sym.replace(".", "-"), "name": name or None, "priced_date": pd_,
                    "price": price if price and price > 0 else None,
                    "shares": shares if shares and shares > 0 else None,
                    "offer_amount": amount if amount and amount > 0 else None,
                    "exchange": (r.get("proposedExchange") or "").strip() or None,
                    "source": IPO_PAGE_URL})
    return out


def _month_add(y: int, m: int, k: int) -> Tuple[int, int]:
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def _ipo_month(y: int, m: int, today: date) -> Optional[List[Dict[str, Any]]]:
    ym = f"{y:04d}-{m:02d}"
    age = (today.year * 12 + today.month) - (y * 12 + m)
    max_age = IPO_RECENT_MAX_AGE_S if age <= 1 else IPO_SETTLED_MAX_AGE_S

    def fetch() -> Optional[List[Dict[str, Any]]]:
        return parse_ipo_month(net.get_json(IPO_URL, headers=net.NASDAQ_HEADERS, params={"date": ym}))

    return net.cached(NS_IPO, ym, max_age, fetch)


def _ipo_months(months: Sequence[Tuple[int, int]], today: date) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    ok = 0
    seen = set()
    for y, m in months:
        got = _ipo_month(y, m, today)
        if got is None:
            continue
        ok += 1
        for r in got:
            k = (r["symbol"], r["priced_date"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    rows.sort(key=lambda r: (r["priced_date"], r["symbol"]), reverse=True)
    return rows, ok


def ipo_calendar(months: int = 9, *, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """Priced IPOs (operating companies) for the current month and the
    ``months - 1`` before it, newest first. 12-h cache per recent month."""
    today = _today(today)
    ms = [_month_add(today.year, today.month, -k) for k in range(max(1, months))]
    rows, ok = _ipo_months(ms, today)
    net.record_status(ST_IPO, ok > 0, f"{len(rows)} priced IPOs (SPACs/units excluded), {ok}/{len(ms)} months read")
    return rows


def lockups(session: date, horizon_days: int = 30, lookback_days: int = 10,
            *, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """IPOs whose *assumed* lock-up expiry (pricing + 180 calendar days, the
    standard term — actual terms vary, check the prospectus) falls in
    [session − lookback, session + horizon]. ``days_to`` < 0 = already past."""
    today = _today(today)
    lo = session - timedelta(days=lookback_days + LOCKUP_DAYS)
    hi = session + timedelta(days=horizon_days - LOCKUP_DAYS)
    months = []
    y, m = lo.year, lo.month
    while (y, m) <= (hi.year, hi.month):
        months.append((y, m))
        y, m = _month_add(y, m, 1)
    rows, ok = _ipo_months(months, today)
    out = []
    for r in rows:
        pd_ = _d(r["priced_date"])
        if pd_ is None or not (lo <= pd_ <= hi):
            continue
        lk = pd_ + timedelta(days=LOCKUP_DAYS)
        out.append({"symbol": r["symbol"], "name": r["name"], "ipo_date": r["priced_date"],
                    "ipo_price": r["price"], "lockup_date": lk.isoformat(), "days_to": (lk - session).days,
                    "shares": r["shares"], "offer_amount": r["offer_amount"], "exchange": r["exchange"],
                    "assumed": LOCKUP_ASSUMED, "source": r["source"]})
    out.sort(key=lambda r: (r["lockup_date"], r["symbol"]))
    net.record_status(ST_LOCKUP, ok == len(months),
                      f"{len(out)} assumed expiries {lo + timedelta(days=LOCKUP_DAYS)}…{hi + timedelta(days=LOCKUP_DAYS)}; "
                      f"{ok}/{len(months)} IPO months read")
    return out


# ═════════════════════════════════════════════════════════════════════════
# Earnings ahead
# ═════════════════════════════════════════════════════════════════════════
def earnings_ahead(start: date, sessions: int = 5) -> List[Dict[str, Any]]:
    """Nasdaq earnings calendar for ``sessions`` trading days starting at
    ``start`` (or the next trading day when ``start`` is not one)."""
    d = start if is_trading_day(start) else next_trading_day(start)
    days = []
    for _ in range(max(0, sessions)):
        days.append(d)
        d = next_trading_day(d)
    out: List[Dict[str, Any]] = []
    ok = 0
    for d in days:
        rows = street.earnings_calendar(d)
        st = net.STATUS.get(ST_EARNINGS) or {}
        if st.get("ok"):
            ok += 1
        for r in rows:
            out.append({"date": r.get("date") or d.isoformat(), "symbol": r.get("symbol"), "name": r.get("name"),
                        "time": r.get("time"), "eps_forecast": r.get("eps_forecast"), "n_ests": r.get("n_ests"),
                        "market_cap": r.get("market_cap")})
    if days:
        net.record_status(ST_EARNINGS, ok > 0,
                          f"{len(out)} reports over {len(days)} sessions {days[0]}…{days[-1]} ({ok}/{len(days)} days read)")
    return out


# ═════════════════════════════════════════════════════════════════════════
# Intraday tape
# ═════════════════════════════════════════════════════════════════════════
_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
_ASOF_RE = re.compile(r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*([AP]M)")
_MONTHS = {m: i + 1 for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                           "Oct", "Nov", "Dec"))}
_REG_OPEN, _REG_CLOSE = dtime(9, 30), dtime(16, 0)


def _clock(s: Any) -> Optional[dtime]:
    m = _CLOCK_RE.search(str(s or ""))
    if not m:
        return None
    h, mi = int(m.group(1)) % 12, int(m.group(2))
    if m.group(3).upper() == "PM":
        h += 12
    return dtime(h, mi)


def _parse_asof(s: Any) -> Optional[str]:
    m = _ASOF_RE.search(str(s or ""))
    if not m or m.group(1) not in _MONTHS:
        return None
    h = int(m.group(4)) % 12 + (12 if m.group(6) == "PM" else 0)
    try:
        return datetime(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)), h, int(m.group(5)),
                        tzinfo=ET).isoformat()
    except ValueError:
        return None


def downsample(points: List[List[float]], n: int = MAX_POINTS) -> List[List[float]]:
    """Evenly spaced subset of ≤ n points that always keeps the first, last,
    highest and lowest sample."""
    if len(points) <= n:
        return points
    ys = [p[1] for p in points]
    keep = {0, len(points) - 1, ys.index(max(ys)), ys.index(min(ys))}
    slots = n - len(keep)
    step = (len(points) - 1) / max(slots, 1)
    i = 0.0
    while len(keep) < n and i <= len(points) - 1:
        keep.add(int(round(i)))
        i += step
    return [points[k] for k in sorted(keep)][:n]


def parse_chart(payload: Any) -> Optional[Dict[str, Any]]:
    """Nasdaq chart JSON → intraday row. Nasdaq's ``x`` is the ET wall clock
    encoded as if it were UTC (4:00 AM ET → 04:00Z); it is converted to a true
    epoch when ``z.dateTime`` confirms that encoding. ``open`` is the first
    regular-session minute sample (≥ 9:30 ET) and ``high``/``low`` are over
    the minute samples — close to, not identical with, the official prints."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    chart = [p for p in data.get("chart") or [] if isinstance(p, dict)
             and isinstance(p.get("x"), (int, float)) and num(p.get("y")) is not None]
    wall = None
    for p in chart:
        c = _clock((p.get("z") or {}).get("dateTime"))
        if c is None:
            continue
        as_utc = datetime.fromtimestamp(p["x"] / 1000.0, timezone.utc)
        as_et = as_utc.astimezone(ET)
        if (as_utc.hour, as_utc.minute) == (c.hour, c.minute):
            wall = True
        elif (as_et.hour, as_et.minute) == (c.hour, c.minute):
            wall = False
        break
    pts: List[Tuple[datetime, float]] = []
    for p in chart:
        if wall is False:
            t_et = datetime.fromtimestamp(p["x"] / 1000.0, timezone.utc).astimezone(ET)
        else:
            t_et = datetime.fromtimestamp(p["x"] / 1000.0, timezone.utc).replace(tzinfo=ET)
        pts.append((t_et, float(num(p["y"]))))  # type: ignore[arg-type]
    pts.sort(key=lambda t: t[0])
    regular = [(t, y) for t, y in pts if _REG_OPEN <= t.time() < _REG_CLOSE]
    use = regular or pts
    series = [[int(t.timestamp() * 1000), y] for t, y in use]
    last = num(data.get("lastSalePrice"))
    return {
        "symbol": data.get("symbol"),
        "asof": _parse_asof(data.get("timeAsOf")),
        "session_date": pts[0][0].date().isoformat() if pts else None,
        "last": last if last is not None else (pts[-1][1] if pts else None),
        "open": regular[0][1] if regular else None,
        "high": max(y for _, y in regular) if regular else None,
        "low": min(y for _, y in regular) if regular else None,
        "last_regular": regular[-1][1] if regular else None,
        "prev_close": num(data.get("previousClose")),
        "points": downsample(series),
        "points_session": "regular" if regular else ("extended" if pts else None),
        "n_samples": len(pts),
        "source": "nasdaq",
    }


def intraday(symbol: str) -> Optional[Dict[str, Any]]:
    """Today's intraday tape for one symbol from nasdaq.com (60-s cache)."""
    sym = street._nasdaq_symbol(symbol)

    def fetch() -> Optional[Dict[str, Any]]:
        return parse_chart(net.get_json(CHART_URL.format(sym=sym), headers=net.NASDAQ_HEADERS,
                                        params={"assetclass": "stocks"}, retries=1, timeout=15.0))

    row = net.cached(NS_INTRADAY, sym, INTRADAY_MAX_AGE_S, fetch)
    if row is None:
        net.record_status(ST_INTRADAY, False, f"{symbol}: chart unavailable")
        return None
    net.record_status(ST_INTRADAY, True, f"{symbol}: {row.get('n_samples')} minute samples, asof {row.get('asof')}")
    return row


__all__ = [
    "dilution_intel", "insider_sales", "chatter", "ipo_calendar", "lockups", "earnings_ahead", "intraday",
    "parse_offering_text", "parse_form4", "parse_stream", "parse_ipo_month", "parse_chart", "share_growth",
    "form4_xml_url", "downsample", "offer_type", "normalize_text", "cover_and_summary", "last_offerings",
]
