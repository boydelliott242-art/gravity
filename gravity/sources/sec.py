"""SEC EDGAR: the dilution / delisting paper trail for every issuer.

Implements CONTRACTS.md §3. Everything reads public EDGAR endpoints through
``net`` (throttled, retried, never raising) and returns plain dicts:

* ``cik_map``             symbol → CIK / name / exchange (company_tickers_exchange.json)
* ``submissions``         raw per-issuer filing index from data.sec.gov, cached ~20 h,
                          older ``files`` pages merged in when the recent block is short
* ``filing_events``       FilingEvents for one symbol (form/item based; optional text check)
* ``events_for_universe`` bulk FilingEvents for the model's history; the nightly refresh
                          is incremental (EDGAR daily indexes + current feed say which
                          issuers filed since the cache was written)
* ``issuer_profile``      incorporation, business address, SIC, Asia link
* ``latest_filings``      overnight catalysts from the EDGAR "current events" Atom feed
* ``fulltext_catalysts``  EDGAR full-text search for dilution / delisting phrases
* ``text_classify``       the same phrase set applied to one filing document
* ``shares_history`` / ``cash_runway``  XBRL companyfacts (dei / us-gaap / ifrs-full)

Honesty rules: categories are derived from the form type and 8-K items
unless a phrase was actually found in the filing text (``text_tags`` says
which). An 8-K item 5.03 without a text match is ``charter_amendment`` —
it is *not* assumed to be a reverse split. Missing values are ``None``;
every event carries its filing date and a document URL.

Taxonomy note: ``charter_amendment`` is an addition to the contract table
(8-K item 5.03 without "reverse stock split" text). ``selling_stockholders``
is an extra text tag used only to tell resale S-1/F-1s from primary ones.
An 8-K *without* item 5.03 is ``reverse_split`` only when it is a plain
announcement (items ⊆ 5.07/7.01/8.01/9.01); agreement filings (1.01/3.02)
quote "reverse stock split" as anti-dilution boilerplate.

Politeness: all sec.gov hosts share one request budget of
``config.SEC_RPS`` per second (net's limiter is per host), and a circuit
breaker stops calling EDGAR for a while after repeated failures, so a
403/429 never turns into a hammering loop.
"""

from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as XML
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import config, net, util

log = logging.getLogger(__name__)

STATUS_NAME = "SEC EDGAR"

# ── Endpoints ────────────────────────────────────────────────────────────
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
CURRENT_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/master.{ymd}.idx"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{accession}-index.htm"

# ── Cache namespaces / ages ──────────────────────────────────────────────
NS_MAP = "sec_map"
NS_SUB = "sec_submissions"
NS_PAGES = "sec_submission_pages"
NS_FACTS = "sec_facts"
NS_TEXT = "sec_text"
NS_EFTS = "sec_fulltext"
NS_INDEX = "sec_daily_index"

MAP_MAX_AGE_S = 20 * 3600
SUBMISSIONS_MAX_AGE_S = 20 * 3600
FACTS_MAX_AGE_S = 20 * 3600
PAGE_MAX_AGE_S = 30 * 86400          # older "files" pages are keyed by their range → effectively immutable
TEXT_MAX_AGE_S = 180 * 86400         # a filed document never changes
INDEX_MAX_AGE_S = 400 * 86400        # a published daily index never changes
EFTS_RECENT_MAX_AGE_S = 3600         # ranges touching the last 3 days are still filling in
EFTS_SETTLED_MAX_AGE_S = 30 * 86400
REVALIDATE_MAX_AGE_S = 7 * 86400     # beyond this, always refetch instead of trusting the change log

MAX_DOC_CHARS = 4_000_000            # text_classify reads at most this much of a document
EFTS_PAGE = 100                      # efts.sec.gov returns 100 hits per page
EFTS_WINDOW = 10_000                 # Elasticsearch result window; larger ranges are split

# ── Category taxonomy (CONTRACTS.md §3 + charter_amendment) ──────────────
CATEGORIES = (
    "offering", "atm", "registration", "resale", "effective", "unregistered_sale",
    "delisting_notice", "reverse_split", "charter_amendment", "toxic_financing",
    "going_concern", "late_filing", "insider_sale_notice", "insider",
    "material_agreement", "other",
)

# tag → (EDGAR full-text query or tuple of queries whose hits are unioned,
# forms searched, local regex on lower-cased text).
# efts.sec.gov syntax (verified live 2026-09-30): quoted phrases, OR between
# phrases, and space-separated terms/phrases are ANDed; no stemming.
# Parentheses are NOT supported — a grouped query silently returns 0 hits —
# so alternatives that need grouping are separate queries.
PHRASES: Dict[str, Tuple[Any, str, str]] = {
    "registered_direct": (
        '"registered direct"',
        "8-K,6-K,424B4,424B5",
        r"\bregistered direct\b",
    ),
    "public_offering_priced": (
        ('"pricing of" "public offering"', '"public offering" priced'),
        "8-K,6-K",
        r"\bpric(?:ed|ing of)\b.{0,160}?\bpublic offering\b|\bpublic offering\b.{0,120}?\bpriced\b",
    ),
    "atm": (
        '"at-the-market offering" OR "at the market offering" OR "at-the-market program" OR '
        '"market issuance sales agreement" OR "equity distribution agreement" OR '
        '"open market sale agreement" OR "controlled equity offering"',
        "8-K,6-K,424B5",
        r"\bat[- ]the[- ]market (?:offering|program|facility|sales|issuance|equity)"
        r"|\bmarket issuance sales agreement\b|\bequity distribution agreement\b"
        r"|\bopen market sale agreement\b|\bcontrolled equity offering\b"
        r"|\batm (?:program|offering|facility|sales agreement)\b",
    ),
    "reverse_split": (
        '"reverse stock split" OR "reverse share split" OR "share consolidation"',
        "8-K,6-K",
        r"\breverse (?:stock|share) split|\bshare consolidation\b",
    ),
    "bid_price_deficiency": (
        '"minimum bid price" OR "5550(a)(2)" OR "bid price requirement"',
        "8-K,6-K",
        r"\bminimum bid price\b|\b5550\s*\(a\)\s*\(2\)|\bbid price requirement\b",
    ),
    "delisting": (
        '"Listing Qualifications" OR "delisting determination" OR "continued listing standard" OR '
        '"continued listing standards" OR "Hearings Panel" OR "notice of delisting"',
        "8-K,6-K",
        r"\blisting qualifications\b|\bdelisting determination\b"
        r"|\bcontinued listing (?:standards?|requirements?|rules?|criteria)\b"
        r"|\bhearings? panel\b|\bnotice of delisting\b",
    ),
    "going_concern": (
        '"substantial doubt" "going concern"',
        "10-K,10-Q,20-F",
        r"\bsubstantial doubt\b.{0,240}?\bgoing concern\b|\bgoing concern\b.{0,240}?\bsubstantial doubt\b",
    ),
    "equity_line": (
        '"equity line" OR "committed equity facility" OR "ELOC" OR '
        '"standby equity purchase agreement" OR "common stock purchase agreement"',
        "8-K,6-K,S-1,F-1",
        r"\bequity line\b|\bcommitted equity facility\b|\beloc\b"
        r"|\bstandby equity purchase agreement\b|\bcommon stock purchase agreement\b",
    ),
    "convertible_note": (
        '"convertible promissory note" OR "convertible promissory notes" OR '
        '"senior secured convertible note" OR "senior secured convertible notes" OR '
        '"convertible debenture" OR "convertible debentures"',
        "8-K,6-K",
        r"\bconvertible promissory notes?\b|\bsenior secured convertible notes?\b|\bconvertible debentures?\b",
    ),
    "warrant_inducement": (
        '"warrant inducement" OR "inducement letter" OR "warrant exercise inducement"',
        "8-K,6-K",
        r"\bwarrant (?:exercise )?inducement\b|\binducement letters?\b",
    ),
    "securities_purchase_agreement": (
        '"securities purchase agreement"',
        "8-K,6-K",
        r"\bsecurities purchase agreements?\b",
    ),
    "selling_stockholders": (
        '"selling stockholders" OR "selling stockholder" OR "selling shareholders" OR '
        '"selling shareholder" OR "selling securityholders"',
        "S-1,F-1",
        r"\bselling (?:stock|share|security)holders?\b",
    ),
}
FULLTEXT_TAGS: Tuple[str, ...] = tuple(PHRASES)
_PHRASE_RE = {tag: re.compile(rx) for tag, (_, _, rx) in PHRASES.items()}

_CURRENT = {"8-K", "6-K"}
_OFFERING_424 = {"424B1", "424B2", "424B4", "424B5", "424B7"}
_MEF = {"S-1MEF", "F-1MEF", "S-3MEF", "F-3MEF"}
_REG_S1 = {"S-1", "F-1"}
_REG_S3 = {"S-3", "F-3", "S-3ASR", "F-3ASR"}
_LATE = {"NT 10-K", "NT 10-Q", "NT 20-F"}
_INSIDER = {"3", "4", "5"}
_PERIODIC = {"10-K", "10-Q", "20-F", "10-KT", "10-QT", "40-F"}
_TOXIC_TAGS = {"equity_line", "convertible_note", "warrant_inducement"}
_RS_ANNOUNCE_ITEMS = {"5.07", "7.01", "8.01", "9.01"}

# ── EDGAR country codes (verified against sec.gov "EDGAR State and Country
# Codes", 2026-09-29). Names match config.ASIA_COUNTRIES exactly. ────────
ASIA_CODES: Dict[str, str] = {
    "F4": "China", "K3": "Hong Kong", "N5": "Macau", "F5": "Taiwan",
    "U0": "Singapore", "N8": "Malaysia", "M0": "Japan", "W1": "Thailand",
    "Q1": "Vietnam", "K8": "Indonesia", "R6": "Philippines",
    "E9": "Cayman Islands", "D8": "British Virgin Islands",
}
# Offshore holding jurisdictions: an Asia signal as a *business address*
# (config lists them), but incorporation there alone is not (US SPACs,
# Israeli and European issuers use them too).
OFFSHORE_CODES = {"E9", "D8"}
ASIA_CORE_CODES = set(ASIA_CODES) - OFFSHORE_CODES
_OTHER_COUNTRY_NAMES = {
    "L3": "Israel", "X0": "United Kingdom", "D0": "Bermuda", "1T": "Marshall Islands",
    "Z4": "Canada", "K7": "India", "M5": "South Korea", "C3": "Australia",
    "2M": "Germany", "I0": "France", "G7": "Greece", "H0": "Greece",
}

PROFILE_KEYS = (
    "state_of_inc", "business_country", "business_city", "sic", "sic_desc", "asia",
    "filer_category",
)

# ── Current-events feed (latest_filings) ───────────────────────────────
# Form-type prefixes queried (the feed matches by prefix). 424B2 is left out
# on purpose: it is thousands of bank structured-note supplements a day.
CURRENT_TYPES = ("8-K", "6-K", "424B1", "424B3", "424B4", "424B5", "424B7",
                 "S-1", "F-1", "S-3", "F-3", "EFFECT", "NT", "144")
CURRENT_FORMS = {
    "8-K", "8-K/A", "6-K", "6-K/A", "424B1", "424B3", "424B4", "424B5", "424B7",
    "S-1", "S-1/A", "S-1MEF", "F-1", "F-1/A", "F-1MEF",
    "S-3", "S-3/A", "S-3ASR", "S-3MEF", "F-3", "F-3/A", "F-3ASR", "F-3MEF",
    "EFFECT", "NT 10-K", "NT 10-Q", "NT 20-F", "NT 10-K/A", "NT 10-Q/A", "NT 20-F/A",
    "144", "144/A",
}
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_ATOM_PAGE = 100
_TITLE_RE = re.compile(r"^(?P<form>.+?) - (?P<name>.*) \((?P<cik>\d{1,10})\) \((?P<role>[^()]*)\)\s*$")
_FILED_RE = re.compile(r"Filed:\s*(?:</b>)?\s*(\d{4}-\d{2}-\d{2})")
_ACC_RE = re.compile(r"AccNo:\s*(?:</b>)?\s*(\d{10}-\d{2}-\d{6})")
_ITEM_RE = re.compile(r"Item\s+(\d{1,2}\.\d{2})")

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(?:-[A-Z0-9]{1,4})?$")
_DISPLAY_RE = re.compile(r"\(([A-Z0-9.\-]+(?:,\s*[A-Z0-9.\-]+)*)\)\s*\(CIK\s*(\d+)\)")
_DERIVATIVE_SUFFIXES = ("W", "WS", "WT", "U", "UN", "R", "RT", "-W", "-WS", "-WT", "-U", "-UN", "-R", "-RT")


# ═════════════════════════════════════════════════════════════════════════
# Plumbing: shared SEC throttle, circuit breaker, status
# ═════════════════════════════════════════════════════════════════════════
_SEC_GAP_S = 1.0 / config.SEC_RPS
_sec_lock = threading.Lock()
_sec_next_slot = 0.0


def _sec_throttle() -> None:
    """One request slot every 1/SEC_RPS s across *all* sec.gov hosts
    (net's limiter is per host; SEC's fair-access limit is per client)."""
    global _sec_next_slot
    with _sec_lock:
        now = time.monotonic()
        slot = max(now, _sec_next_slot)
        _sec_next_slot = slot + _SEC_GAP_S
    wait = slot - now
    if wait > 0:
        time.sleep(wait)


class _Breaker:
    """Stop calling EDGAR for ``cooldown_s`` after ``threshold`` consecutive
    failures (each already retried with backoff inside ``net.get``)."""

    def __init__(self, threshold: int = 6, cooldown_s: float = 600.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            return time.monotonic() >= self._open_until

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fails = 0
                return
            self._fails += 1
            if self._fails >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown_s
                self._fails = 0
                log.warning("SEC: %d consecutive failures — pausing EDGAR calls for %.0fs",
                            self.threshold, self.cooldown_s)

    def reset(self) -> None:
        with self._lock:
            self._fails = 0
            self._open_until = 0.0


_breaker = _Breaker()
_parts: Dict[str, Tuple[bool, str]] = {}
_parts_lock = threading.Lock()


def _note(part: str, ok: bool, detail: str) -> None:
    """Fold one sub-feed's outcome into the single "SEC EDGAR" status row."""
    with _parts_lock:
        _parts[part] = (bool(ok), detail)
        ok_all = all(v[0] for v in _parts.values())
        text = "; ".join(f"{k}: {v[1]}" for k, v in _parts.items())
    if not _breaker.allow():
        ok_all = False
        text += "; paused after repeated failures (possible SEC 403/429)"
    net.record_status(STATUS_NAME, ok_all, text)


def _disabled() -> None:
    net.record_status(STATUS_NAME, False, "SEC_USER_AGENT not set")


def _sec_get(url: str, params: Optional[Dict[str, Any]] = None, *, missing_ok: bool = False,
             retries: int = 2, timeout: float = 30.0) -> Any:
    """GET a sec.gov URL with the SEC UA, the shared throttle and the breaker.
    ``missing_ok`` = a None here is an expected answer (404), not a failure."""
    if not net.sec_enabled() or not _breaker.allow():
        return None
    _sec_throttle()
    r = net.get(url, headers=net.sec_headers(), params=params, retries=retries, timeout=timeout)
    if r is not None:
        _breaker.record(True)
    elif not missing_ok:
        _breaker.record(False)
    return r


def _sec_json(url: str, params: Optional[Dict[str, Any]] = None, **kw: Any) -> Any:
    r = _sec_get(url, params, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


# ═════════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════════
def _base_form(form: Optional[str]) -> str:
    f = (form or "").strip().upper()
    return f[:-2] if f.endswith("/A") else f


def categorize(form: str, items: Sequence[str] = (), tags: Iterable[str] = ()) -> str:
    """FilingEvent category from form type, 8-K items and text tags.

    Text tags only refine forms where the phrase means what it says
    (current reports, prospectuses, S-1/F-1, periodic reports for going
    concern); a 10-Q that *mentions* a past registered direct stays "other"."""
    b = _base_form(form)
    t = set(tags or ())
    it = set(items or ())
    if b in _CURRENT:
        if t & {"registered_direct", "public_offering_priced"}:
            return "offering"
        if "atm" in t:
            return "atm"
        if t & _TOXIC_TAGS:
            return "toxic_financing"
        if "securities_purchase_agreement" in t:
            return "offering"
        if "5.03" in it and "reverse_split" in t:
            return "reverse_split"
        if "3.01" in it or t & {"bid_price_deficiency", "delisting"}:
            return "delisting_notice"
        # Without item 5.03 the phrase counts only on a 6-K or on an 8-K that
        # is an announcement (7.01/8.01/5.07 vote/9.01); in agreement filings
        # (1.01/3.02 …) "reverse stock split" is anti-dilution boilerplate.
        if "reverse_split" in t and (b == "6-K" or not (it - _RS_ANNOUNCE_ITEMS)):
            return "reverse_split"
        if "3.02" in it:
            return "unregistered_sale"
        if "5.03" in it:
            return "charter_amendment"
        if "1.01" in it:
            return "material_agreement"
        return "other"
    if b in _OFFERING_424:
        return "atm" if "atm" in t else "offering"
    if b == "424B3":
        return "resale"
    if b in _MEF:
        return "offering"
    if b in _REG_S1:
        if "equity_line" in t:
            return "toxic_financing"
        if "selling_stockholders" in t:
            return "resale"
        return "registration"
    if b in _REG_S3:
        return "registration"
    if b == "EFFECT":
        return "effective"
    if b in _LATE:
        return "late_filing"
    if b == "144":
        return "insider_sale_notice"
    if b in _INSIDER:
        return "insider"
    if b in _PERIODIC:
        return "going_concern" if "going_concern" in t else "other"
    return "other"


def category_from_tags(tags: Iterable[str], form: str, items: Sequence[str] = ()) -> str:
    """``categorize`` with the argument order the integration layer uses
    (tags first): category for a filing after a text check."""
    return categorize(form, items, tags)


def _text_checkable(form: str) -> bool:
    """Forms whose category a text check can change (see ``categorize``)."""
    b = _base_form(form)
    return b in _CURRENT or b in _OFFERING_424 or b in _REG_S1


def filing_url(cik: int, accession: str, doc: Optional[str] = None) -> str:
    """Archive URL of a filing's document, or of its index page when the
    primary document is unknown."""
    acc = (accession or "").strip()
    nodash = acc.replace("-", "")
    if doc:
        return ARCHIVE_URL.format(cik=int(cik), acc=nodash, doc=doc)
    return INDEX_URL.format(cik=int(cik), acc=nodash, accession=acc)


def _split_items(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        vals = [str(x) for x in raw]
    else:
        vals = str(raw).split(",")
    return [v.strip() for v in vals if v and v.strip()]


def _accepted_et(raw: Optional[str]) -> Optional[str]:
    """data.sec.gov ``acceptanceDateTime`` ("…T13:57:14.000Z", true UTC —
    verified against the filing index page) → ET ISO string with offset."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(util.ET).isoformat(timespec="seconds")


def _history_years() -> int:
    """Years of filing history the model needs: the price history period
    plus one year of look-back for the 365-day filing-count features."""
    m = re.match(r"^(\d+)y$", str(config.HISTORY_PERIOD).strip())
    return (int(m.group(1)) if m else 3) + 1


def _history_cutoff() -> str:
    return (util.now_et().date() - timedelta(days=int(365.25 * _history_years()))).isoformat()


def _event_sort_key(e: Dict[str, Any]) -> Tuple[str, str]:
    return (e.get("date") or "", e.get("accepted") or "")


_intern_lock = threading.Lock()
_interned: Dict[str, str] = {}


def _i(s: Optional[str]) -> Optional[str]:
    """Share repeated strings (dates, forms, categories) across the ~10^5
    events a universe build holds in memory."""
    if s is None:
        return None
    with _intern_lock:
        return _interned.setdefault(s, s)


def _event(symbol: str, cik: int, date_: str, accepted: Optional[str], form: str,
           items: List[str], url: str, accession: str, source: str,
           tags: Iterable[str] = ()) -> Dict[str, Any]:
    tag_list = sorted(set(tags or ()))
    return {
        "symbol": _i(symbol),
        "cik": int(cik),
        "date": _i(date_),
        "accepted": accepted,
        "form": _i(form),
        "items": items,
        "category": _i(categorize(form, items, tag_list)),
        "url": url,
        "text_tags": tag_list,
        "accession": accession,
        "source": _i(source),
    }


# ═════════════════════════════════════════════════════════════════════════
# Ticker ↔ CIK map
# ═════════════════════════════════════════════════════════════════════════
_map_lock = threading.Lock()
_map_memo: Dict[str, Any] = {"at": 0.0, "by_sym": None, "by_cik": None}
_MAP_MEMO_S = 3600.0


def _is_derivative(sym: str, siblings: Sequence[str]) -> bool:
    """TICKW / TICK-WT / TICKU … when TICK is also listed for the same CIK."""
    for other in siblings:
        if other != sym and sym.startswith(other) and sym[len(other):] in _DERIVATIVE_SUFFIXES:
            return True
    return False


def _maps() -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
    """(symbol → info, cik → primary symbol), memoised in-process for an hour."""
    with _map_lock:
        if _map_memo["by_sym"] is not None and time.time() - _map_memo["at"] < _MAP_MEMO_S:
            return _map_memo["by_sym"], _map_memo["by_cik"]
        payload = net.cached(NS_MAP, "company_tickers_exchange", MAP_MAX_AGE_S,
                             lambda: _sec_json(TICKERS_URL))
        if not isinstance(payload, dict) or not payload.get("data"):
            _note("tickers", False, "company_tickers_exchange.json unavailable")
            return {}, {}
        fields = payload.get("fields") or ["cik", "name", "ticker", "exchange"]
        ix = {f: i for i, f in enumerate(fields)}
        by_sym: Dict[str, Dict[str, Any]] = {}
        per_cik: Dict[int, List[str]] = {}
        for row in payload["data"]:
            try:
                cik = int(row[ix["cik"]])
                tick = row[ix["ticker"]]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if not tick:
                continue
            sym = util.to_canonical(str(tick))
            if not _TICKER_RE.match(sym) or sym in by_sym:
                continue
            name = row[ix["name"]] if "name" in ix and ix["name"] < len(row) else None
            exch = row[ix["exchange"]] if "exchange" in ix and ix["exchange"] < len(row) else None
            by_sym[sym] = {"cik": cik, "name": name or None, "exchange": exch or None}
            per_cik.setdefault(cik, []).append(sym)
        by_cik: Dict[int, str] = {}
        for cik, syms in per_cik.items():
            primary = [s for s in syms if not _is_derivative(s, syms)]
            by_cik[cik] = (primary or syms)[0]
        _map_memo.update(at=time.time(), by_sym=by_sym, by_cik=by_cik)
        _note("tickers", True, f"{len(by_sym)} tickers")
        return by_sym, by_cik


def cik_map() -> Dict[str, Dict[str, Any]]:
    """Canonical symbol → {"cik": int, "name": str, "exchange": str|None}
    for every ticker in SEC's company_tickers_exchange.json (incl. OTC)."""
    if not net.sec_enabled():
        _disabled()
        return {}
    return _maps()[0]


def symbol_for_cik(cik: int) -> Optional[str]:
    """Primary (non-warrant/unit) canonical ticker for a CIK, or None."""
    if not net.sec_enabled():
        return None
    return _maps()[1].get(int(cik))


# ═════════════════════════════════════════════════════════════════════════
# Submissions (per-issuer filing index)
# ═════════════════════════════════════════════════════════════════════════
def _sub_key(cik: int) -> str:
    return f"CIK{int(cik):010d}"


def _extend_columns(recent: Dict[str, List[Any]], page: Dict[str, Any]) -> int:
    """Append an older page's parallel arrays to ``recent`` (dedupe by accession)."""
    accs = recent.get("accessionNumber") or []
    n_old = len(accs)
    seen = set(accs)
    src_acc = page.get("accessionNumber") or []
    keep = [j for j, a in enumerate(src_acc) if a and a not in seen]
    if not keep:
        return 0
    for k in set(recent) | {k for k, v in page.items() if isinstance(v, list)}:
        col = recent.get(k)
        if not isinstance(col, list):
            col = recent[k] = [None] * n_old
        elif len(col) < n_old:
            col.extend([None] * (n_old - len(col)))
        src = page.get(k) if isinstance(page.get(k), list) else []
        col.extend(src[j] if j < len(src) else None for j in keep)
    return len(keep)


def _merge_history_pages(d: Dict[str, Any]) -> Dict[str, Any]:
    """If the recent block (last ~1,000 filings) ends inside the history
    window, append the older ``filings.files`` pages that overlap it."""
    filings = d.get("filings") or {}
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        recent = {}
        filings["recent"] = recent
        d["filings"] = filings
    files = filings.get("files") or []
    cutoff = _history_cutoff()
    dates = [x for x in (recent.get("filingDate") or []) if x]
    merged: List[str] = []
    incomplete = False
    if files and (not dates or min(dates) > cutoff):
        for f in files:
            name = f.get("name")
            if not name or (f.get("filingTo") or "9999") < cutoff:
                continue
            key = f"{name}|{f.get('filingFrom')}|{f.get('filingTo')}|{f.get('filingCount')}"
            page = net.cached(NS_PAGES, key, PAGE_MAX_AGE_S,
                              lambda name=name: _sec_json(SUBMISSIONS_PAGE_URL.format(name=name)))
            if not isinstance(page, dict):
                incomplete = True
                continue
            _extend_columns(recent, page)
            merged.append(name)
    d["_gravity"] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_cutoff": cutoff,
        "merged_files": merged,
        "incomplete": incomplete,
    }
    return d


def submissions(cik: int, max_age_s: int = SUBMISSIONS_MAX_AGE_S) -> Optional[dict]:
    """Raw data.sec.gov submissions JSON for a CIK, cached on disk.

    ``filings.recent`` is extended in place with the older ``filings.files``
    pages when the recent block does not reach back ``HISTORY_PERIOD`` + 1y;
    ``_gravity`` records which pages were merged and when it was fetched."""
    if not net.sec_enabled():
        _disabled()
        return None
    cik = int(cik)

    def fetch() -> Optional[dict]:
        d = _sec_json(SUBMISSIONS_URL.format(cik=cik))
        return _merge_history_pages(d) if isinstance(d, dict) else None

    return net.cached(NS_SUB, _sub_key(cik), max_age_s, fetch)


def events_from_submissions(sub: Dict[str, Any], symbol: str, cik: int,
                            since: Optional[str] = None) -> List[Dict[str, Any]]:
    """FilingEvents (form/item based, no text) from a submissions JSON,
    newest first, deduplicated by accession, ``date >= since`` if given."""
    rec = ((sub or {}).get("filings") or {}).get("recent") or {}
    forms = rec.get("form") or []

    def col(k: str) -> List[Any]:
        v = rec.get(k)
        return v if isinstance(v, list) else []

    dates, accs, docs = col("filingDate"), col("accessionNumber"), col("primaryDocument")
    accepted, items = col("acceptanceDateTime"), col("items")
    out: Dict[str, Dict[str, Any]] = {}
    for i, form in enumerate(forms):
        d = dates[i] if i < len(dates) else None
        acc = accs[i] if i < len(accs) else None
        if not form or not d or not acc or acc in out:
            continue
        if since and d < since:
            continue
        doc = docs[i] if i < len(docs) else None
        out[acc] = _event(
            symbol, cik, d, _accepted_et(accepted[i] if i < len(accepted) else None),
            form, _split_items(items[i] if i < len(items) else None),
            filing_url(cik, acc, doc or None), acc, "sec_submissions",
        )
    return sorted(out.values(), key=_event_sort_key, reverse=True)


def apply_text_tags(events: List[Dict[str, Any]], tagged: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge ``text_tags`` from full-text hits (matched by accession) into
    form-based events and re-derive their category. Cheap: no requests.
    Mutates and returns ``events``."""
    by_acc: Dict[str, Set[str]] = {}
    for t in tagged:
        acc = t.get("accession")
        if acc and t.get("text_tags"):
            by_acc.setdefault(acc, set()).update(t["text_tags"])
    for e in events:
        extra = by_acc.get(e.get("accession") or "")
        if extra:
            e["text_tags"] = sorted(set(e.get("text_tags") or ()) | extra)
            e["category"] = categorize(e["form"], e.get("items") or [], e["text_tags"])
    return events


def merge_events(*event_lists: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One FilingEvent per filing across sources (e.g. ``latest_filings`` +
    ``fulltext_catalysts``), keyed by (accession, cik): ``text_tags`` are
    unioned and ``category`` re-derived; a document URL beats an index-page
    URL; known ``accepted`` / ``items`` fill in missing ones. Newest first.
    Inputs are not mutated."""
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for lst in event_lists:
        for e in lst or ():
            key = (e.get("accession") or e.get("url") or "", e.get("cik"))
            cur = out.get(key)
            if cur is None:
                out[key] = dict(e, text_tags=list(e.get("text_tags") or []), items=list(e.get("items") or []))
                continue
            cur["text_tags"] = sorted(set(cur["text_tags"]) | set(e.get("text_tags") or ()))
            if not cur.get("accepted") and e.get("accepted"):
                cur["accepted"] = e["accepted"]
            if not cur["items"] and e.get("items"):
                cur["items"] = list(e["items"])
            if e.get("url") and (not cur.get("url") or (_INDEX_PAGE_RE.search(cur["url"])
                                                         and not _INDEX_PAGE_RE.search(e["url"]))):
                cur["url"] = e["url"]
    for e in out.values():
        e["category"] = categorize(e.get("form") or "", e["items"], e["text_tags"])
    return sorted(out.values(), key=_event_sort_key, reverse=True)


def _text_enrich(events: List[Dict[str, Any]], max_docs: int) -> int:
    """Run ``text_classify`` on the newest text-checkable events."""
    n = 0
    for e in events:
        if n >= max_docs:
            break
        if not _text_checkable(e.get("form") or ""):
            continue
        n += 1
        tags = text_classify(e["url"])
        if tags is None:
            continue
        e["text_tags"] = sorted(set(e.get("text_tags") or ()) | set(tags))
        e["category"] = categorize(e["form"], e.get("items") or [], e["text_tags"])
    return n


def filing_events(symbol: str, since: Optional[str] = None, *, fetch_text: bool = False,
                  max_text_docs: int = 12) -> List[Dict[str, Any]]:
    """FilingEvents for one symbol, newest first (``date >= since`` if given).

    Categories are form/item based. ``fetch_text=True`` (shortlist only)
    additionally reads the primary document of the newest ``max_text_docs``
    8-K/6-K/424B/S-1/F-1 filings and refines category + ``text_tags``."""
    if not net.sec_enabled():
        _disabled()
        return []
    sym = util.to_canonical(symbol)
    info = cik_map().get(sym)
    if not info:
        return []
    sub = submissions(info["cik"])
    if sub is None:
        _note("filing_events", False, f"{sym}: submissions unavailable")
        return []
    evs = events_from_submissions(sub, sym, info["cik"], since)
    checked = _text_enrich(evs, max_text_docs) if fetch_text else 0
    _note("filing_events", True, f"{sym}: {len(evs)} events" + (f", {checked} text-checked" if checked else ""))
    return evs


# ═════════════════════════════════════════════════════════════════════════
# Issuer profile
# ═════════════════════════════════════════════════════════════════════════
def _address(a: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    a = a or {}
    state = (a.get("stateOrCountry") or "").strip().upper() or None
    code = (a.get("countryCode") or "").strip().upper() or state
    city = (a.get("city") or "").strip() or None
    foreign = a.get("isForeignLocation")
    if state and re.fullmatch(r"[A-Z]{2}", state) and not foreign:
        return {"code": state, "country": "United States", "state": state, "city": city, "us": True}
    if code:
        name = (a.get("country") or ASIA_CODES.get(code) or _OTHER_COUNTRY_NAMES.get(code)
                or (a.get("stateOrCountryDescription") or "").title() or None)
        return {"code": code, "country": name, "state": None, "city": city, "us": False}
    return {"code": None, "country": None, "state": None, "city": city, "us": None}


def _clean_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    parts = [p.strip() for p in re.split(r"<br\s*/?>", str(raw), flags=re.I)]
    s = "; ".join(p for p in parts if p)
    return s or None


def profile_from_submissions(sub: Dict[str, Any]) -> Dict[str, Any]:
    """Issuer profile (see ``issuer_profile``) from a submissions JSON."""
    addrs = sub.get("addresses") or {}
    biz = _address(addrs.get("business"))
    mail = _address(addrs.get("mailing"))
    if biz["code"] is None and mail["code"] is not None:
        biz = mail                                      # no business address on file
    inc = (sub.get("stateOfIncorporation") or "").strip().upper() or None
    inc_desc = (sub.get("stateOfIncorporationDescription") or "").strip() or None

    basis: List[str] = []
    if biz["code"] in ASIA_CODES:
        basis.append(f"business address in {ASIA_CODES[biz['code']]} ({biz['code']})")
    if mail["code"] in ASIA_CODES and mail["code"] != biz["code"]:
        basis.append(f"mailing address in {ASIA_CODES[mail['code']]} ({mail['code']})")
    if inc in ASIA_CORE_CODES:
        basis.append(f"incorporated in {ASIA_CODES[inc]} ({inc})")
    known = biz["code"] is not None or inc is not None
    asia: Optional[bool] = bool(basis) if known else None

    former = []
    for f in sub.get("formerNames") or []:
        former.append({"name": f.get("name"), "from": (f.get("from") or "")[:10] or None,
                       "to": (f.get("to") or "")[:10] or None})
    try:
        cik = int(sub.get("cik"))
    except (TypeError, ValueError):
        cik = None
    return {
        "state_of_inc": inc,
        "state_of_inc_desc": ASIA_CODES.get(inc or "") or inc_desc,
        "business_country": biz["country"],
        "business_country_code": biz["code"],
        "business_state": biz["state"],
        "business_city": biz["city"].title() if biz["city"] else None,
        "sic": (str(sub.get("sic")).strip() or None) if sub.get("sic") else None,
        "sic_desc": sub.get("sicDescription") or None,
        "asia": asia,
        "asia_basis": basis,
        "offshore_inc": (inc in OFFSHORE_CODES) if inc else None,
        "filer_category": _clean_category(sub.get("category")),
        "name": sub.get("name") or None,
        "entity_type": sub.get("entityType") or None,
        "tickers": list(sub.get("tickers") or []),
        "exchanges": list(sub.get("exchanges") or []),
        "fiscal_year_end": sub.get("fiscalYearEnd") or None,
        "former_names": former,
        "source": SUBMISSIONS_URL.format(cik=cik) if cik is not None else None,
    }


def issuer_profile(cik: int) -> Dict[str, Any]:
    """{"state_of_inc", "business_country", "business_city", "sic", "sic_desc",
    "asia", "filer_category"} plus provenance extras (``asia_basis`` lists
    the address / incorporation facts behind the Asia flag).

    ``asia`` is True when the business or mailing address is in an Asian
    jurisdiction from ``config.ASIA_COUNTRIES`` (EDGAR codes, incl. Cayman /
    BVI addresses) or the issuer is incorporated in one of the non-offshore
    ones; Cayman/BVI *incorporation* alone is reported as ``offshore_inc``.
    All values are None when EDGAR could not be read."""
    empty: Dict[str, Any] = {k: None for k in PROFILE_KEYS}
    if not net.sec_enabled():
        _disabled()
        return empty
    sub = submissions(cik)
    if sub is None:
        _note("issuer_profile", False, f"CIK {cik}: submissions unavailable")
        return empty
    return profile_from_submissions(sub)


# ═════════════════════════════════════════════════════════════════════════
# Full-text search (efts.sec.gov)
# ═════════════════════════════════════════════════════════════════════════
def _slim_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    s = h.get("_source") or {}
    return {
        "id": h.get("_id") or "",
        "adsh": s.get("adsh"),
        "ciks": s.get("ciks") or [],
        "display_names": s.get("display_names") or [],
        "form": s.get("form") or (s.get("root_forms") or [None])[0],
        "file_date": s.get("file_date"),
        "file_type": s.get("file_type"),
        "items": s.get("items") or [],
    }


def _efts_page(q: str, forms: str, start: str, end: str, offset: int) -> Optional[Dict[str, Any]]:
    d = _sec_json(EFTS_URL, {"q": q, "forms": forms, "dateRange": "custom",
                             "startdt": start, "enddt": end, "from": offset})
    if not isinstance(d, dict) or not isinstance(d.get("hits"), dict):
        return None
    return d["hits"]


def _efts_range(q: str, forms: str, start: str, end: str, max_pages: int,
                budget: List[int]) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """All hits for one query over [start, end] → (hits, ok, truncated).
    Ranges with more hits than one query can page through are split in half."""
    if budget[0] <= 0:
        return [], True, True
    budget[0] -= 1
    first = _efts_page(q, forms, start, end, 0)
    if first is None:
        return [], False, False
    tot = first.get("total") or {}
    total = int(tot.get("value") or 0)
    cap = min(max_pages * EFTS_PAGE, EFTS_WINDOW)
    too_many = total > cap or (tot.get("relation") == "gte" and total >= cap)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if too_many and d1 > d0:
        mid = d0 + (d1 - d0) // 2
        a, ok_a, tr_a = _efts_range(q, forms, start, mid.isoformat(), max_pages, budget)
        b, ok_b, tr_b = _efts_range(q, forms, (mid + timedelta(days=1)).isoformat(), end, max_pages, budget)
        return a + b, ok_a and ok_b, tr_a or tr_b
    hits = [_slim_hit(h) for h in first.get("hits") or []]
    offset = EFTS_PAGE
    while offset < min(total, cap) and len(hits) >= offset:
        if budget[0] <= 0:
            return hits, True, True
        budget[0] -= 1
        page = _efts_page(q, forms, start, end, offset)
        if page is None:
            return hits, False, False
        batch = page.get("hits") or []
        hits.extend(_slim_hit(h) for h in batch)
        if len(batch) < EFTS_PAGE:
            break
        offset += EFTS_PAGE
    return hits, True, too_many


def _tag_queries(tag: str) -> Tuple[str, ...]:
    q = PHRASES[tag][0]
    return (q,) if isinstance(q, str) else tuple(q)


def _efts_query(tag: str, start: str, end: str, max_pages: int,
                budget: List[int]) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """Hits for one tag (union over its queries) → (hits, ok, truncated).
    ``ok`` is False when any query failed (partial hits are still returned);
    only complete answers are cached."""
    _, forms, _ = PHRASES[tag]
    settled = end < (util.now_et().date() - timedelta(days=3)).isoformat()
    max_age = EFTS_SETTLED_MAX_AGE_S if settled else EFTS_RECENT_MAX_AGE_S
    out: List[Dict[str, Any]] = []
    ok_all, trunc_any = True, False
    for q in _tag_queries(tag):
        key = f"{q}|{forms}|{start}|{end}"
        hit = net.cache_get(NS_EFTS, key, max_age)
        if hit is not None:
            out.extend(hit)
            continue
        hits, ok, truncated = _efts_range(q, forms, start, end, max_pages, budget)
        out.extend(hits)
        ok_all = ok_all and ok
        trunc_any = trunc_any or truncated
        if ok and not truncated:
            net.cache_set(NS_EFTS, key, hits)
    return out, ok_all, trunc_any


def _tickers_from_display(name: str) -> List[str]:
    m = _DISPLAY_RE.search(name or "")
    if not m:
        return []
    ticks = [util.to_canonical(t) for t in m.group(1).split(",") if t.strip()]
    ticks = [t for t in ticks if _TICKER_RE.match(t)]
    primary = [t for t in ticks if not _is_derivative(t, ticks)]
    return primary or ticks


def _resolve_issuer(ciks: Sequence[Any], names: Sequence[str],
                    by_cik: Dict[int, str]) -> Tuple[Optional[str], Optional[int]]:
    """(symbol, cik) for a hit: SEC's ticker map first, then the ticker EDGAR
    printed in ``display_names`` ("Name  (TICK, TICKW)  (CIK 000…)")."""
    ints: List[int] = []
    for c in ciks:
        try:
            ints.append(int(c))
        except (TypeError, ValueError):
            continue
    for c in ints:
        if c in by_cik:
            return by_cik[c], c
    for i, n in enumerate(names):
        ticks = _tickers_from_display(n)
        if ticks:
            m = _DISPLAY_RE.search(n)
            cik = int(m.group(2)) if m else (ints[i] if i < len(ints) else None)
            return ticks[0], cik
    return None, (ints[0] if ints else None)


def fulltext_catalysts(start: str, end: str, *, tags: Optional[Iterable[str]] = None,
                       max_pages: int = 20, max_requests: int = 300) -> List[Dict[str, Any]]:
    """FilingEvents for filings (``start``..``end`` inclusive, "YYYY-MM-DD")
    whose text matches the catalyst phrases, one event per filing with
    ``text_tags`` = every phrase key it matched and ``category`` re-derived.
    Filings from issuers with no ticker are dropped. ``url`` points at the
    matched document (the primary document when it matched)."""
    if not net.sec_enabled():
        _disabled()
        return []
    _, by_cik = _maps()
    wanted = [t for t in (tags or FULLTEXT_TAGS) if t in PHRASES]
    budget = [max_requests]
    agg: Dict[Tuple[str, int], Dict[str, Any]] = {}
    primary: Set[Tuple[str, int]] = set()
    counts: Dict[str, int] = {}
    failed: List[str] = []
    truncated: List[str] = []
    for tag in wanted:
        hits, ok, trunc = _efts_query(tag, start, end, max_pages, budget)
        if not ok:
            failed.append(tag)
        if trunc:
            truncated.append(tag)
        seen: Set[Tuple[str, int]] = set()
        for h in hits:
            adsh = h.get("adsh") or (h.get("id") or "").split(":")[0]
            if not adsh:
                continue
            sym, cik = _resolve_issuer(h.get("ciks") or [], h.get("display_names") or [], by_cik)
            if not sym or cik is None:
                continue
            key = (adsh, cik)
            seen.add(key)
            doc = h["id"].split(":", 1)[1] if ":" in (h.get("id") or "") else None
            url = filing_url(cik, adsh, doc)
            is_primary = (h.get("file_type") or "").upper() == _base_form(h.get("form")) or \
                (h.get("file_type") or "").upper() == (h.get("form") or "").upper()
            ev = agg.get(key)
            if ev is None:
                ev = _event(sym, cik, h.get("file_date") or "", None, h.get("form") or "",
                            _split_items(h.get("items")), url, adsh, "sec_fulltext", [tag])
                agg[key] = ev
            else:
                ev["text_tags"] = sorted(set(ev["text_tags"]) | {tag})
            if is_primary and key not in primary:
                ev["url"] = url
                primary.add(key)
        counts[tag] = len(seen)
    out = []
    for ev in agg.values():
        ev["category"] = categorize(ev["form"], ev["items"], ev["text_tags"])
        out.append(ev)
    out.sort(key=_event_sort_key, reverse=True)
    detail = f"{start}..{end}: {len(out)} filings"
    if failed:
        detail += f", failed {','.join(failed)}"
    if truncated:
        detail += f", truncated {','.join(truncated)}"
    _note("fulltext", not failed, detail)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Filing text
# ═════════════════════════════════════════════════════════════════════════
_SCRIPT_RE = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(s: str) -> str:
    """Filing HTML → lower-cased single-spaced plain text."""
    s = _SCRIPT_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s).replace("\xa0", " ").replace("’", "'")
    return _WS_RE.sub(" ", s).strip().lower()


def classify_text(text: str) -> List[str]:
    """Phrase keys (``PHRASES``) found in already-normalised text."""
    return sorted(tag for tag, rx in _PHRASE_RE.items() if rx.search(text))


_INDEX_PAGE_RE = re.compile(r"-index\.html?$", re.I)
_ARCHIVE_DOC_RE = re.compile(r"^https?://www\.sec\.gov/Archives/edgar/data/(\d+)/(\d{18})/[^/?#]+$", re.I)
_ROW_RE = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr>")
_CELL_RE = re.compile(r"(?is)<td\b[^>]*>(.*?)</td>")
_HREF_RE = re.compile(r"""(?i)href\s*=\s*["']([^"']+)["']""")
_DOC_EXT_RE = re.compile(r"\.(?:htm|html|txt)$", re.I)
MAX_INDEX_DOCS = 3          # primary document + up to two EX-99 exhibits
COVER_MAX_CHARS = 6_000     # a primary document shorter than this is a cover page (typical 6-K)


def index_documents(page: str, max_docs: int = MAX_INDEX_DOCS) -> List[str]:
    """Filing index page (``…-index.htm``) → absolute URLs of the documents
    worth reading: the primary document (first row) and EX-99.x exhibits
    (press releases), in filing order. Graphics, XBRL and the complete
    submission text file are skipped."""
    i = page.find("Document Format Files")
    body = page[i:] if i >= 0 else page
    j = body.find("</table>")
    if j >= 0:
        body = body[:j]
    out: List[str] = []
    for row in _ROW_RE.findall(body):
        cells = _CELL_RE.findall(row)
        if len(cells) < 4:
            continue
        m = _HREF_RE.search(cells[2])
        if not m:
            continue
        href = html.unescape(m.group(1)).strip()
        if href.startswith("/ix?doc="):
            href = href[len("/ix?doc="):]
        typ = html_to_text(cells[3]).upper()
        if not typ or not _DOC_EXT_RE.search(href):
            continue
        if not out or typ.startswith("EX-99"):
            out.append(href if href.startswith("http") else "https://www.sec.gov" + href)
        if len(out) >= max_docs:
            break
    return out


def index_url_for(url: str) -> Optional[str]:
    """Index page of the filing an Archives document URL belongs to."""
    m = _ARCHIVE_DOC_RE.match(url or "")
    if not m or _INDEX_PAGE_RE.search(url):
        return None
    acc = m.group(2)
    return INDEX_URL.format(cik=int(m.group(1)), acc=acc, accession=f"{acc[:10]}-{acc[10:12]}-{acc[12:]}")


def _read_text(url: str) -> Optional[str]:
    r = _sec_get(url, timeout=45.0)
    return None if r is None else html_to_text(r.text[:MAX_DOC_CHARS])


def _classify_docs(urls: Sequence[str]) -> Tuple[Optional[Set[str]], bool]:
    """Union of phrase tags over documents → (tags or None if none could be
    read, every document read)."""
    tags: Set[str] = set()
    got = 0
    for u in urls:
        text = _read_text(u)
        if text is None:
            continue
        got += 1
        tags.update(classify_text(text))
    return (tags if got else None), got == len(urls)


def text_classify(url: str, with_exhibits: Optional[bool] = None) -> Optional[List[str]]:
    """Fetch a filing and return its ``text_tags`` (sorted phrase keys, same
    set as ``fulltext_catalysts``). None when nothing could be read.

    ``url`` may be a document (submissions-based events) or a filing index
    page (``latest_filings`` events). An index page is resolved to the
    primary document plus EX-99 exhibits (press releases). For a document,
    ``with_exhibits=None`` (auto) also reads the EX-99 exhibits when the
    document is only a cover page (typical 6-K); True always does, False
    never. Complete results are cached per URL (filed documents never
    change). Meant for shortlisted names only: 1–4 requests per filing."""
    if not net.sec_enabled():
        _disabled()
        return None
    if not url:
        return None
    key = url if with_exhibits is None else f"{url}|exhibits={bool(with_exhibits)}"
    hit = net.cache_get(NS_TEXT, key, TEXT_MAX_AGE_S)
    if hit is not None:
        return hit

    complete = True
    if _INDEX_PAGE_RE.search(url):
        r = _sec_get(url)
        if r is None:
            return None
        docs = index_documents(r.text)
        if not docs:
            return None
        tags, complete = _classify_docs(docs)
    else:
        text = _read_text(url)
        if text is None:
            return None
        tags = set(classify_text(text))
        want = with_exhibits if with_exhibits is not None else len(text) < COVER_MAX_CHARS
        idx = index_url_for(url) if want else None
        if idx:
            r = _sec_get(idx)
            if r is None:
                complete = False
            else:
                extra = [d for d in index_documents(r.text) if d != url][: MAX_INDEX_DOCS - 1]
                more, complete = _classify_docs(extra)
                tags |= more or set()
    if tags is None:
        return None
    out = sorted(tags)
    if complete:
        net.cache_set(NS_TEXT, key, out)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Current-events feed (overnight filings)
# ═════════════════════════════════════════════════════════════════════════
def _parse_atom(content: bytes) -> Optional[List[Dict[str, Any]]]:
    """EDGAR getcurrent Atom → [{form, name, cik, role, accession, date,
    accepted_dt, items, url}]. None if the payload is not the feed."""
    try:
        root = XML.fromstring(content)
    except XML.ParseError:
        return None
    if not root.tag.endswith("feed"):
        return None
    out = []
    for ent in root.findall("a:entry", _ATOM_NS):
        title = (ent.findtext("a:title", "", _ATOM_NS) or "").strip()
        m = _TITLE_RE.match(title)
        if not m:
            continue
        summary = ent.findtext("a:summary", "", _ATOM_NS) or ""
        updated = (ent.findtext("a:updated", "", _ATOM_NS) or "").strip()
        try:
            acc_dt = datetime.fromisoformat(updated)
        except ValueError:
            continue
        if acc_dt.tzinfo is None:
            acc_dt = acc_dt.replace(tzinfo=util.ET)
        cat = ent.find("a:category", _ATOM_NS)
        form = (cat.get("term") if cat is not None else None) or m.group("form")
        link = ent.find("a:link", _ATOM_NS)
        acc = _ACC_RE.search(summary)
        if not acc:
            eid = ent.findtext("a:id", "", _ATOM_NS) or ""
            acc = re.search(r"accession-number=(\d{10}-\d{2}-\d{6})", eid)
        if not acc:
            continue
        filed = _FILED_RE.search(summary)
        out.append({
            "form": form.strip(),
            "name": m.group("name").strip(),
            "cik": int(m.group("cik")),
            "role": m.group("role").strip(),
            "accession": acc.group(1),
            "date": filed.group(1) if filed else acc_dt.astimezone(util.ET).date().isoformat(),
            "accepted_dt": acc_dt,
            "items": _ITEM_RE.findall(summary),
            "url": link.get("href") if link is not None else None,
        })
    return out


def _atom_floor(now: Optional[datetime] = None) -> datetime:
    """Oldest acceptance time we trust the current feed to still hold.
    Observed retention is ~3 EDGAR business days (today + 2 prior); we only
    claim midnight ET of the previous weekday."""
    d = (now or util.now_et()).astimezone(util.ET).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return datetime.combine(d, dtime(0, 0), util.ET)


def _scan_current(since: datetime, types: Sequence[str] = CURRENT_TYPES,
                  max_pages: int = 40) -> Tuple[List[Dict[str, Any]], bool]:
    """Every feed entry accepted at/after ``since`` for the given form-type
    prefixes → (entries, complete). ``complete`` is False when a page failed
    or the feed ran out before reaching ``since``."""
    floor = _atom_floor()
    rows: List[Dict[str, Any]] = []
    complete = True
    for typ in types:
        reached = False
        for page in range(max_pages):
            r = _sec_get(CURRENT_URL, {"action": "getcurrent", "type": typ, "count": _ATOM_PAGE,
                                       "output": "atom", "start": page * _ATOM_PAGE})
            entries = _parse_atom(r.content) if r is not None else None
            if entries is None:
                complete = False
                reached = True
                break
            rows.extend(e for e in entries if e["accepted_dt"] >= since)
            if any(e["accepted_dt"] < since for e in entries):
                reached = True
                break
            if len(entries) < _ATOM_PAGE:
                reached = True
                if since < floor:
                    complete = False
                break
        if not reached:
            complete = False
    return rows, complete


def latest_filings(since_utc: datetime) -> List[Dict[str, Any]]:
    """FilingEvents for every filing of interest (8-K, 6-K, 424B*, S-1, F-1,
    S-3, F-3, EFFECT, NT 10-K/10-Q/20-F, 144) accepted since ``since_utc``,
    across all issuers that have a ticker, newest first. Category is form /
    item based (items come from the feed); ``accepted`` is the EDGAR
    acceptance time. 424B2 (bank structured notes) is not scanned."""
    if not net.sec_enabled():
        _disabled()
        return []
    since = since_utc if since_utc.tzinfo else since_utc.replace(tzinfo=timezone.utc)
    _, by_cik = _maps()
    rows, complete = _scan_current(since)
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        if r["form"] not in CURRENT_FORMS:
            continue
        sym = by_cik.get(r["cik"])
        key = (r["accession"], r["cik"])
        if not sym or key in out:
            continue
        out[key] = _event(sym, r["cik"], r["date"],
                          r["accepted_dt"].astimezone(util.ET).isoformat(timespec="seconds"),
                          r["form"], r["items"], r["url"] or filing_url(r["cik"], r["accession"]),
                          r["accession"], "sec_current")
    evs = sorted(out.values(), key=_event_sort_key, reverse=True)
    _note("latest", complete, f"{len(evs)} issuer filings since "
          f"{since.astimezone(timezone.utc).isoformat(timespec='minutes')}"
          + ("" if complete else " (feed incomplete)"))
    return evs


# ═════════════════════════════════════════════════════════════════════════
# Bulk history for the model (incremental)
# ═════════════════════════════════════════════════════════════════════════
def _daily_index_ciks(d: date) -> Optional[Set[int]]:
    """CIKs (filers *and* subject companies) in EDGAR's master index for one
    day; empty on weekends; None when the index is not (yet) published."""
    if d.weekday() >= 5:
        return set()
    url = DAILY_INDEX_URL.format(y=d.year, q=(d.month - 1) // 3 + 1, ymd=d.strftime("%Y%m%d"))

    def fetch() -> Optional[List[int]]:
        r = _sec_get(url, missing_ok=True, retries=1, timeout=60.0)
        if r is None:
            return None
        ciks: Set[int] = set()
        for line in r.text.splitlines():
            head = line.split("|", 1)[0]
            if "|" in line and head.isdigit():
                ciks.add(int(head))
        return sorted(ciks)

    got = net.cached(NS_INDEX, d.isoformat(), INDEX_MAX_AGE_S, fetch)
    return None if got is None else set(got)


def _changed_ciks_since(t0: datetime) -> Optional[Set[int]]:
    """CIKs with any filing accepted after ``t0``, or None when that cannot
    be established (then callers refetch everything).

    Published daily indexes cover whole filing dates; filings accepted after
    ~17:30 ET carry the next business day's date, so the tail after the last
    published index is covered from the current-events feed."""
    now = util.now_et()
    d = t0.astimezone(util.ET).date()
    changed: Set[int] = set()
    last_indexed: Optional[date] = None
    while d <= now.date():
        ciks = _daily_index_ciks(d)
        if ciks is not None:
            changed |= ciks
            if d.weekday() < 5:
                last_indexed = d   # earlier unpublished weekdays were EDGAR holidays
        d += timedelta(days=1)
    atom_since = t0 - timedelta(hours=1)
    if last_indexed is not None:
        atom_since = max(atom_since, datetime.combine(last_indexed, dtime(17, 0), util.ET))
    rows, complete = _scan_current(atom_since)
    if not complete:
        return None
    return changed | {r["cik"] for r in rows}


def _revalidate(ciks: Sequence[int], max_age_s: float) -> int:
    """Mark stale-but-recent cached submissions fresh when EDGAR's change log
    shows the issuer has not filed since they were fetched. Returns count."""
    now = time.time()
    stale: Dict[int, Any] = {}
    oldest = now
    for cik in ciks:
        p = net._cache_path(NS_SUB, _sub_key(cik))
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        if max_age_s < age <= REVALIDATE_MAX_AGE_S:
            stale[cik] = p
            oldest = min(oldest, mtime)
    if not stale:
        return 0
    changed = _changed_ciks_since(datetime.fromtimestamp(oldest, timezone.utc))
    if changed is None:
        log.info("SEC: change log incomplete — refetching %d stale submissions", len(stale))
        return 0
    n = 0
    for cik, p in stale.items():
        if cik in changed:
            continue
        try:
            os.utime(p, None)
            n += 1
        except OSError:
            continue
    return n


def events_for_universe(symbols: Sequence[str], max_workers: int = 4, since: Optional[str] = None, *,
                        max_age_s: int = SUBMISSIONS_MAX_AGE_S,
                        incremental: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """symbol → FilingEvents (form/item based, newest first) for every symbol
    with a CIK, from ``submissions`` (disk-cached ~20 h).

    ``since`` defaults to HISTORY_PERIOD + 1 year back. With ``incremental``,
    cached submissions older than ``max_age_s`` (≤ 7 days) are kept when the
    EDGAR daily indexes + current feed show no new filing by that issuer, so a
    nightly refresh only downloads issuers that actually filed. Symbols with
    no CIK or whose fetch failed are absent (unknown), not empty."""
    if not net.sec_enabled():
        _disabled()
        return {}
    by_sym = cik_map()
    if not by_sym:
        _note("universe", False, "ticker map unavailable")
        return {}
    since = since if since is not None else _history_cutoff()
    sym_cik: Dict[str, int] = {}
    for s in symbols:
        sym = util.to_canonical(s)
        info = by_sym.get(sym)
        if info:
            sym_cik[sym] = info["cik"]
    ciks = sorted(set(sym_cik.values()))
    t_start = time.time()
    reused = _revalidate(ciks, max_age_s) if incremental else 0
    subs: Dict[int, Optional[dict]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = {ex.submit(submissions, cik, max_age_s): cik for cik in ciks}
        for n, fut in enumerate(as_completed(futs), 1):
            cik = futs[fut]
            try:
                subs[cik] = fut.result()
            except Exception:  # pragma: no cover — defensive: never lose the batch
                log.exception("SEC submissions CIK %s", cik)
                subs[cik] = None
            if n % 250 == 0:
                log.info("SEC submissions %d/%d (%.0fs)", n, len(ciks), time.time() - t_start)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, cik in sym_cik.items():
        sub = subs.get(cik)
        if sub is not None:
            out[sym] = events_from_submissions(sub, sym, cik, since)
    failed = sum(1 for c in ciks if subs.get(c) is None)
    _note("universe", failed <= max(1, len(ciks) // 20),
          f"{len(out)}/{len(symbols)} symbols, {len(ciks)} CIKs, {failed} failed, "
          f"{reused} revalidated, {time.time() - t_start:.0f}s")
    return out


# ═════════════════════════════════════════════════════════════════════════
# XBRL companyfacts: shares outstanding and cash runway
# ═════════════════════════════════════════════════════════════════════════
SHARE_CONCEPTS = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("ifrs-full", "NumberOfSharesOutstanding"),
)
CASH_CONCEPTS = (
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
    ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ("us-gaap", "Cash"),
    ("ifrs-full", "CashAndCashEquivalents"),
)
OCF_CONCEPTS = (
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
)
_DAYS_PER_QUARTER = 365.25 / 4


def _units(facts: Dict[str, Any], tax: str, concept: str) -> Dict[str, List[Dict[str, Any]]]:
    node = ((facts.get("facts") or {}).get(tax) or {}).get(concept) or {}
    units = node.get("units") or {}
    return units if isinstance(units, dict) else {}


def _d(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10]) if s else None
    except ValueError:
        return None


def shares_from_facts(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Point-in-time share counts: for each as-of date the *first* value
    filed (later restatements, e.g. split-adjusted, are ignored so the series
    reflects what was known when). First concept with data wins."""
    for tax, concept in SHARE_CONCEPTS:
        rows = _units(facts, tax, concept).get("shares") or []
        per_filing: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            end, filed, val = r.get("end"), r.get("filed"), util.num(r.get("val"))
            if not end or not filed or val is None or val <= 0:
                continue
            k = (r.get("accn") or "", end)
            g = per_filing.get(k)
            if g is None or val > g["shares"]:          # duplicates in one filing: keep max
                per_filing[k] = {"date": end, "filed": filed, "shares": float(val),
                                 "form": r.get("form"), "accn": r.get("accn")}
        per_end: Dict[str, Dict[str, Any]] = {}
        for g in per_filing.values():
            cur = per_end.get(g["date"])
            if cur is None or g["filed"] < cur["filed"]:
                per_end[g["date"]] = g
        if per_end:
            tag = f"{tax}:{concept}"
            return [dict(v, concept=tag) for v in sorted(per_end.values(), key=lambda x: (x["date"], x["filed"]))]
    return []


def _pick_unit(units: Dict[str, List[Dict[str, Any]]], prefer: Optional[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not units:
        return None, []
    for u in (prefer, "USD"):
        if u and units.get(u):
            return u, units[u]
    u = sorted(units, key=lambda k: -len(units[k]))[0]
    return u, units[u]


def _latest_instant(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for rank, (tax, concept) in enumerate(CASH_CONCEPTS):
        unit, rows = _pick_unit(_units(facts, tax, concept), None)
        for r in rows:
            val = util.num(r.get("val"))
            if val is None or not r.get("end") or r.get("start"):
                continue
            cand = {"val": float(val), "end": r["end"], "filed": r.get("filed") or "", "unit": unit,
                    "concept": f"{tax}:{concept}", "rank": rank}
            if best is None or (cand["end"], -cand["rank"], cand["filed"]) > (best["end"], -best["rank"], best["filed"]):
                best = cand
    return best


def _quarterly_flows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discrete quarterly values from duration facts, undoing YTD cumulation:
    a 3-month fact is used as is; otherwise Q = YTD(start, end) −
    YTD(start, end − ~1 quarter). The latest-filed value per period wins."""
    per: Dict[Tuple[date, date], Tuple[str, float]] = {}
    for r in rows:
        s, e, v = _d(r.get("start")), _d(r.get("end")), util.num(r.get("val"))
        if s is None or e is None or v is None or e <= s:
            continue
        filed = r.get("filed") or ""
        cur = per.get((s, e))
        if cur is None or filed >= cur[0]:
            per[(s, e)] = (filed, float(v))
    quarters: Dict[date, Dict[str, Any]] = {}
    for (s, e), (_, v) in per.items():
        if 80 <= (e - s).days <= 100:
            quarters[e] = {"start": s.isoformat(), "end": e.isoformat(), "ocf": v, "derived": False}
    by_start: Dict[date, List[Tuple[date, float]]] = {}
    for (s, e), (_, v) in per.items():
        by_start.setdefault(s, []).append((e, v))
    for s, lst in by_start.items():
        lst.sort()
        for i, (e, v) in enumerate(lst):
            if e in quarters:
                continue
            for e2, v2 in lst[:i]:
                if 80 <= (e - e2).days <= 100:
                    quarters[e] = {"start": (e2 + timedelta(days=1)).isoformat(), "end": e.isoformat(),
                                   "ocf": v - v2, "derived": True}
                    break
    return [quarters[k] for k in sorted(quarters)]


def runway_from_facts(facts: Dict[str, Any], cik: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """{"cash", "cash_date", "quarterly_burn", "runway_q"} plus provenance.

    ``quarterly_burn`` = −(mean operating cash flow of the last one or two
    consecutive quarters), so positive means cash is being burned; with no
    quarterly data it falls back to the latest period scaled to 91 days.
    ``runway_q`` = cash / quarterly_burn, None when not burning. Amounts are
    in the filer's reporting currency (``currency``)."""
    cash = _latest_instant(facts)
    if cash is None:
        return None
    burn: Optional[float] = None
    basis: Optional[str] = None
    burn_date: Optional[str] = None
    recent: List[Dict[str, Any]] = []
    for tax, concept in OCF_CONCEPTS:
        unit, rows = _pick_unit(_units(facts, tax, concept), cash["unit"])
        if not rows or unit != cash["unit"]:
            continue
        qs = _quarterly_flows(rows)
        if qs:
            last = qs[-1]
            recent = [last]
            if len(qs) > 1:
                gap = (_d(last["start"]) - _d(qs[-2]["end"])).days  # type: ignore[operator]
                if 0 <= gap <= 5:
                    recent = [qs[-2], last]
            burn = -sum(q["ocf"] for q in recent) / len(recent)
            basis = f"mean of last {len(recent)} quarter(s) of {tax}:{concept}"
            burn_date = last["end"]
            recent = qs[-4:]
            break
        spans = []
        for r in rows:
            s, e, v = _d(r.get("start")), _d(r.get("end")), util.num(r.get("val"))
            if s and e and v is not None and (e - s).days >= 150:
                spans.append((e, (e - s).days, float(v)))
        if spans:
            e, days, v = max(spans)
            burn = -v * _DAYS_PER_QUARTER / days
            basis = f"{days}-day period of {tax}:{concept} scaled to one quarter"
            burn_date = e.isoformat()
            break
    runway = cash["val"] / burn if burn is not None and burn > 0 else None
    return {
        "cash": cash["val"],
        "cash_date": cash["end"],
        "quarterly_burn": round(burn, 2) if burn is not None else None,
        "runway_q": round(runway, 2) if runway is not None else None,
        "currency": cash["unit"],
        "cash_concept": cash["concept"],
        "cash_filed": cash["filed"] or None,
        "burn_date": burn_date,
        "burn_basis": basis,
        "burn_quarters": recent,
        "source": FACTS_URL.format(cik=int(cik)) if cik is not None else None,
    }


def _facts_summary(cik: int) -> Optional[Dict[str, Any]]:
    """Shares history + cash runway computed from one companyfacts download
    (the raw JSON is large, so only the derived summary is cached)."""
    cik = int(cik)

    def fetch() -> Optional[Dict[str, Any]]:
        facts = _sec_json(FACTS_URL.format(cik=cik), missing_ok=True, timeout=60.0)
        if not isinstance(facts, dict) or "facts" not in facts:
            return None
        return {"shares_history": shares_from_facts(facts), "cash_runway": runway_from_facts(facts, cik)}

    return net.cached(NS_FACTS, _sub_key(cik), FACTS_MAX_AGE_S, fetch)


def shares_history(cik: int) -> List[Dict[str, Any]]:
    """[{"date", "filed", "shares"} …] ascending, point-in-time (first value
    filed per as-of date), from dei:EntityCommonStockSharesOutstanding, else
    us-gaap:CommonStockSharesOutstanding, else ifrs-full:NumberOfSharesOutstanding.
    As reported — *not* split-adjusted. Extra keys: form, accn, concept."""
    if not net.sec_enabled():
        _disabled()
        return []
    s = _facts_summary(cik)
    if s is None:
        _note("xbrl", False, f"CIK {cik}: companyfacts unavailable")
        return []
    _note("xbrl", True, f"CIK {cik}: {len(s.get('shares_history') or [])} share counts")
    return list(s.get("shares_history") or [])


def cash_runway(cik: int) -> Optional[Dict[str, Any]]:
    """Latest cash and quarterly operating burn from XBRL companyfacts; see
    ``runway_from_facts``. None when no cash fact is available."""
    if not net.sec_enabled():
        _disabled()
        return None
    s = _facts_summary(cik)
    if s is None:
        _note("xbrl", False, f"CIK {cik}: companyfacts unavailable")
        return None
    return s.get("cash_runway")


__all__ = [
    "CATEGORIES", "FULLTEXT_TAGS", "PHRASES", "ASIA_CODES",
    "cik_map", "symbol_for_cik", "submissions", "issuer_profile", "filing_events",
    "events_for_universe", "latest_filings", "fulltext_catalysts", "shares_history",
    "cash_runway", "text_classify", "categorize", "category_from_tags", "apply_text_tags", "filing_url",
    "index_documents", "index_url_for", "merge_events",
    "events_from_submissions", "profile_from_submissions", "shares_from_facts",
    "runway_from_facts", "classify_text", "html_to_text",
]
