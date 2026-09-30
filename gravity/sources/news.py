"""Headlines per symbol from Google News RSS, tagged by a precise
keyword/regex classifier.

The classifier is deliberately conservative: a tag fires only on phrasing
that means the event (``"prices $5 million registered direct offering"``,
``"1-for-20 reverse stock split"``, ``"Nasdaq notification regarding
minimum bid price"``), not on loose words. Tags describe what a headline
*says*; they are inputs to the risk overlay, not predictions.

Headlines are kept only when the title itself names the ticker or the
company, because Google's full-text match often returns market round-ups
that mention the symbol in passing — tagging those would attribute other
companies' news to this stock.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple
from urllib.parse import quote_plus

from .. import net
from ..util import to_canonical

log = logging.getLogger(__name__)

GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
NEWS_MAX_AGE_S = 20 * 60

BEARISH_TAGS = (
    "offering", "priced", "dilution", "reverse_split", "delisting", "deficiency",
    "halt", "investigation", "lawsuit", "resign", "going_concern", "default",
    "downgrade", "miss", "guidance_cut", "atm", "warrants",
)
BULLISH_TAGS = (
    "contract", "partnership", "fda", "approval", "beat", "upgrade",
    "acquisition", "buyback", "uplisting",
)
# Corporate-finance / listing events that make a headline bearish no matter
# what else it says ("prices offering to fund acquisition" is an offering).
_DOMINANT_BEARISH = {
    "offering", "priced", "atm", "dilution", "warrants", "reverse_split",
    "delisting", "deficiency", "going_concern", "default", "halt",
}


def _rx(p: str) -> Pattern[str]:
    return re.compile(p, re.I)


_MONEY = r"\$\s?[\d.,]+\s*(?:k|m|mm|mln|million|b|bn|billion)?\b"

# Financing agreements look like commercial deals ("enters into $10M
# securities purchase agreement") — they are removed before bullish
# "contract" matching and routed to the bearish tags instead.
_FINANCING_AGREEMENT = _rx(
    r"\b(?:securities|stock|share|equity|standby equity|note|notes) purchase agreements?\b"
    r"|\b(?:equity distribution|controlled equity offering sales|at[- ]the[- ]market sales"
    r"|placement agency|underwriting|sales|credit|loan|forbearance|exchange|subscription"
    r"|(?:warrant )?inducement|equity line(?: of credit)?|(?:definitive )?merger) agreements?\b"
)

_RULES: List[Tuple[str, Pattern[str]]] = [
    # ── bearish ──────────────────────────────────────────────────────────
    ("offering", _rx(
        r"\b(?:registered direct|public|underwritten|secondary|follow[- ]on|best[- ]efforts"
        r"|private|confidentially marketed|overnight|proposed|upsized|equity|stock|share"
        r"|unit|common stock)\s+offerings?\b"
        rf"|{_MONEY}\s+offerings?\b"
        r"|\bprivate placement\b|\bpipe (?:financing|investment|transaction)\b"
        r"|\bsecurities purchase agreements?\b|\bbought deal\b"
        r"|\b(?:pric(?:es|ed|ing)|launch(?:es|ed|ing)?|clos(?:es|ed|ing)|complet(?:es|ed|ing)"
        r"|upsiz(?:es|ed|ing))\s+(?:of\s+)?(?:(?:an?|its|the|proposed)\s+)?offerings?\b"
        r"|\boffering of\s+(?:\$|[\d,.]+\s+(?:shares|units|adss?))")),
    ("atm", _rx(
        r"\bat[- ]the[- ]market (?:equity |stock )?(?:offerings?|programs?|facility|sales|issuance"
        r"|agreements?)\b|\batm (?:program|offering|facility|agreement|equity)\b"
        r"|\bequity distribution agreements?\b|\bcontrolled equity offering\b")),
    ("dilution", _rx(
        r"\bdilut\w*|\bshelf registration\b|\bmixed shelf\b"
        r"|\b(?:files?|filed|filing)\b[^.]{0,40}\b(?:shelf|s-1|s-3|f-1|f-3|resale|registration statement)\b"
        r"|\bresale (?:registration|prospectus|of up to)\b"
        r"|\bequity line\b|\b(?:standby )?equity purchase agreements?\b|\bcommitted equity facility\b"
        r"|\beloc\b|\bconvertible (?:promissory )?(?:notes?|debentures?)\b")),
    ("reverse_split", _rx(
        r"\breverse (?:stock |share )?splits?\b|\bshare consolidation\b"
        r"|\bconsolidation of (?:its )?(?:issued )?(?:ordinary |common )?shares\b")),
    ("delisting", _rx(
        r"\bdelist\w*|\bsuspension and delisting\b|\bform 25\b"
        r"|\b(?:removed|removal) from (?:the )?(?:nasdaq|nyse)\b|\bhearings? panel\b")),
    ("deficiency", _rx(
        r"\bminimum bid(?: price)?\b|\bbid price (?:requirement|rule|deficiency)\b"
        r"|\b(?:continued )?listing (?:rules?|requirements?|standards?)\b|\bdeficien\w*"
        r"|\bnon-?compliance\b|\bnot in compliance\b|\bnoncompliant\b"
        r"|\b(?:notification|notice|letter) (?:letter )?from (?:the )?(?:nasdaq|nyse)\b"
        r"|\b(?:nasdaq|nyse)(?: american)? (?:notification|notice|letter)\b"
        r"|\bnotice of (?:non-?compliance|deficiency|delinquency)\b"
        r"|\bstockholders'? equity requirement\b|\bmarket value of listed securities\b"
        r"|\blisting qualifications\b|\bdelinquen\w*")),
    ("halt", _rx(
        r"\btrading halt\w*|\bhalt(?:s|ed)? trading\b"
        r"|\b(?:stock|shares|trading)\s+(?:(?:is|was|remains?|been|gets?)\s+)?halted\b"
        r"|\bhalted\b|\btrading (?:pause|suspension)\b|\bsuspend(?:s|ed)? trading\b"
        r"|\bhalt pending\b|\bluld\b|\bcircuit breaker\b|\b(?:nasdaq|nyse|sec|finra)\s+halts\b")),
    ("investigation", _rx(
        r"\binvestigat\w*|\bprobes?\b|\bprobed\b|\bsubpoena\w*|\bwells notice\b"
        r"|\bsec (?:charges|inquiry|complaint)\b|\bfraud\b")),
    ("lawsuit", _rx(
        r"\blawsuits?\b|\bclass[- ]action\b|\bsues\b|\bsued\b|\blitigation\b"
        r"|\blegal action\b|\bshareholder alert\b|\bcomplaint\b")),
    ("resign", _rx(
        r"\bresign\w*|\bsteps? down\b|\bstepping down\b|\bdeparture\b|\bdeparts\b"
        r"|\bouste[dr]\b|\bfired\b"
        r"|\bdismiss(?:es|ed|al of)\b[^.]{0,20}\b(?:auditor|ceo|cfo|chief)\b"
        r"|\bterminat(?:es|ed|ion of)\b[^.]{0,20}\b(?:auditor|ceo|cfo|chief)\b")),
    ("going_concern", _rx(r"\bgoing[- ]concern\b|\bsubstantial doubt\b")),
    ("default", _rx(
        r"\bdefault(?:s|ed)?\b|\bbankrupt\w*|\bchapter (?:7|11)\b|\binsolven\w*"
        r"|\breceivership\b|\bforbearance\b|\bwind(?:s|ing)? down\b"
        r"|\bmisse[sd] (?:an? )?(?:interest|debt|loan|coupon) payment")),
    ("downgrade", _rx(
        r"\bdowngrad\w*"
        r"|\b(?:cuts?|lowers?|lowered|slashes|slashed|reduces?|reduced|trims?)\b[^.]{0,40}\b(?:price target|pt)\b"
        r"|\bprice target (?:cut|lowered|reduced|slashed)\b"
        r"|\binitiat\w*\b[^.]{0,50}\b(?:sell|underperform|underweight)\b")),
    ("miss", _rx(
        r"\bmiss(?:es|ed)?\b[^.]{0,40}\b(?:estimates?|expectations|consensus|forecasts?|views?|eps|revenue|targets?)\b"
        r"|\b(?:eps|earnings|revenue|profit|sales|results)\b[^.]{0,40}\bmiss(?:es|ed)?\b"
        r"|\bfalls? short of (?:estimates|expectations|consensus|forecasts?)\b"
        r"|\bworse[- ]than[- ]expected\b|\bbelow (?:estimates|expectations|consensus)\b")),
    ("guidance_cut", _rx(
        r"\b(?:cuts?|lowers?|lowered|slashes|slashed|reduces?|reduced|withdraws?|withdrawn"
        r"|suspends?|trims?|trimmed)\b[^.]{0,30}\b(?:guidance|outlook|forecast)\b"
        r"|\b(?:guidance|outlook|forecast)\b[^.]{0,15}\b(?:cut|lowered|reduced|slashed|withdrawn)\b"
        r"|\bprofit warning\b|\bwarns?\b[^.]{0,30}\b(?:revenue|sales|profit|earnings|results)\b")),
    ("warrants", _rx(r"\bwarrants?\b")),
    # ── bullish ──────────────────────────────────────────────────────────
    ("contract", _rx(
        r"\bcontracts?\b|\bpurchase orders?\b"
        r"|\b(?:wins?|won|secures?|secured|awarded|receives?|received|lands?|landed|signs?|signed"
        r"|inks?|inked)\b[^.]{0,60}\b(?:orders?|deals?|awards?|tenders?|agreements?)\b"
        r"|\b(?:supply|distribution|service|services|development|licensing|license|offtake"
        r"|framework|procurement|commercial) agreements?\b"
        rf"|{_MONEY}[^.]{{0,40}}\b(?:deals?|orders?|contracts?|agreements?)\b")),
    ("partnership", _rx(
        r"\bpartner(?:s|ship|ships|ing|ed)?\b|\bcollaborat\w*|\balliance\b|\bjoint venture\b"
        r"|\bteams? up\b|\bmou\b|\bmemorandum of understanding\b|\bstrategic cooperation\b"
        r"|\bcooperation agreement\b")),
    ("fda", _rx(
        r"\bfda\b|\b510\(k\)|\bbreakthrough (?:therapy|device) designation\b|\borphan drug\b"
        r"|\bfast track\b|\bpdufa\b|\bce mark\b")),
    ("approval", _rx(
        r"\bapprov(?:al|als|es|ed)\b|\bclearance\b|\bcleared by\b|\bauthoriz(?:ation|ed)\b"
        r"|\bpatent (?:granted|issued|allowance)\b|\bgranted (?:a )?(?:patent|approval|designation)\b")),
    ("beat", _rx(
        r"\bbeats?\b[^.]{0,40}\b(?:estimates?|expectations|consensus|forecasts?|views?|eps|revenue|street)\b"
        r"|\b(?:tops?|topped|exceeds?|exceeded|surpass\w*)\b[^.]{0,30}\b(?:estimates?|expectations|consensus|forecasts?|views?)\b"
        r"|\bbetter[- ]than[- ]expected\b"
        r"|\brecord (?:revenue|quarter|quarterly|sales|results|earnings|profit)\b"
        r"|\b(?:eps|earnings|revenue)\b[^.]{0,30}\bbeats?\b")),
    ("upgrade", _rx(
        r"\bupgrad(?:e|es|ed)\b[^.]{0,40}\b(?:to|at)\s+(?:buy|strong buy|outperform|overweight|neutral"
        r"|hold|market perform|equal[- ]weight|sector perform|accumulate|positive)\b"
        r"|\bupgraded by\b|\banalyst upgrade\b"
        r"|\b(?:raises?|raised|lifts?|lifted|boosts?|boosted|hikes?|hiked)\b[^.]{0,40}\b(?:price target|pt)\b"
        r"|\bprice target (?:raised|increased|lifted|hiked)\b"
        r"|\binitiat\w*\b[^.]{0,50}\b(?:buy|outperform|overweight|strong buy)\b"
        r"|\breiterat\w*\b[^.]{0,30}\b(?:buy|outperform|overweight)\b")),
    ("acquisition", _rx(
        r"\bacquir(?:e|es|ed|ing)\b|\bacquisitions?\b|\bmerger\b|\bmerg(?:e|es|ing)\b|\btakeover\b"
        r"|\bbuyout\b|\btender offer\b|\bto be acquired\b|\bgo(?:es|ing)?[- ]private\b")),
    ("buyback", _rx(
        r"\bbuy[- ]?backs?\b|\b(?:share|stock) repurchases?\b|\brepurchase (?:program|plan|authorization)\b"
        r"|\brepurchases?\b")),
    ("uplisting", _rx(
        r"\buplist\w*|\bapproved for listing on (?:the )?(?:nasdaq|nyse)\b"
        r"|\bbegins? trading on (?:the )?(?:nasdaq|nyse)\b|\bto (?:list|trade) on (?:the )?(?:nasdaq|nyse)\b")),
]

_ONE_FOR_N = _rx(r"\b1[- ]for[- ](\d{1,4})\b")
_PRICED = _rx(r"\bpric(?:es|ed|ing)\b")
_REGAINED = _rx(r"\bregain(?:s|ed|ing)?\b[^.]{0,40}\bcompliance\b|\bcompliance (?:regained|restored)\b")
_PRODUCT_OFFERING = _rx(r"\b(?:product|service|solution|menu|course|token|coin)s? offerings?\b")
_FDA_NEGATIVE = _rx(
    r"\bcomplete response letter\b|\bcrl\b|\bclinical hold\b|\brefus\w* to file\b"
    r"|\brejects?\b|\brejected\b|\bdeclines? to approve\b|\bnot approv\w*|\bfails?\b|\bfailed\b"
    r"|\bwarning letter\b|\bform 483\b")
# A called-off deal is not a deal; a cancelled offering/ATM is less dilution.
_TERMINATED_DEAL = _rx(
    r"\b(?:terminat\w*|cancel\w*|call(?:s|ed)? off|walks? away from|abandon\w*)\b[^.]{0,40}"
    r"\b(?:merger|acquisition|deal|agreement|partnership|contract|offer)s?\b")
_CANCELLED_FINANCING = _rx(
    r"\b(?:terminat\w*|cancel\w*|withdraw\w*|postpon\w*|abandon\w*)\b[^.]{0,40}"
    r"\b(?:offerings?|programs?|facility|equity line|atm|shelf|registration statement)\b")
_NEW_FINANCING = _rx(r"\bnew\b|\benter\w*|\breplac\w*|\bupsiz\w*")
# Approvals that are corporate actions, not regulatory wins.
_CORPORATE_ACTION = {"reverse_split", "offering", "priced", "dilution", "atm",
                     "warrants", "delisting", "deficiency"}


def _normalise(title: str) -> str:
    t = (title or "").replace("’", "'").replace("‘", "'")
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def classify(title: str) -> Dict[str, Any]:
    """Tag one headline. Returns ``{"tags": [...], "polarity": -1|0|1}``.

    Tags come from ``BEARISH_TAGS``/``BULLISH_TAGS``, in that fixed order.
    Polarity is −1 when any dilution/listing/solvency tag fires (those
    dominate: "prices offering to fund acquisition" is bearish), otherwise
    the sign of bullish minus bearish tag counts; negative FDA outcomes
    (CRL, clinical hold, rejection) are −1.
    """
    t = _normalise(title)
    if not t:
        return {"tags": [], "polarity": 0}
    tags = set()
    commercial = _FINANCING_AGREEMENT.sub(" ", t)
    for tag, rx in _RULES:
        text = commercial if tag == "contract" else t
        if rx.search(text):
            tags.add(tag)

    # Precision fixes ------------------------------------------------------
    m = _ONE_FOR_N.search(t)
    if m and int(m.group(1)) > 1:
        tags.add("reverse_split")
    if "offering" in tags and _PRODUCT_OFFERING.search(t) and not re.search(_MONEY, t):
        tags.discard("offering")
    if _PRICED.search(t) and ("offering" in tags or "atm" in tags or re.search(r"\bplacement\b", t, re.I)):
        tags.add("priced")
        tags.add("offering")
    if _REGAINED.search(t):
        tags.discard("deficiency")
        tags.discard("delisting")
    if tags & _CORPORATE_ACTION:
        tags.discard("approval")
    if "delisting" in tags and re.search(r"\bavoid\w*\b[^.]{0,20}\bdelist", t, re.I):
        tags.discard("delisting")
    if _TERMINATED_DEAL.search(t):
        tags -= {"acquisition", "contract", "partnership"}
    if _CANCELLED_FINANCING.search(t) and not _NEW_FINANCING.search(t):
        tags -= {"offering", "priced", "atm", "dilution"}      # less dilution, not more

    polarity: int
    fda_negative = "fda" in tags and bool(_FDA_NEGATIVE.search(t))
    if tags & _DOMINANT_BEARISH or fda_negative:
        polarity = -1
    else:
        bear = sum(1 for x in tags if x in BEARISH_TAGS)
        bull = sum(1 for x in tags if x in BULLISH_TAGS)
        polarity = (bull > bear) - (bull < bear)
    ordered = [x for x in BEARISH_TAGS + BULLISH_TAGS if x in tags]
    return {"tags": ordered, "polarity": polarity}


# ═════════════════════════════════════════════════════════════════════════
# Google News RSS
# ═════════════════════════════════════════════════════════════════════════
_NAME_SUFFIX = _rx(
    r"(?:,?\s+(?:inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|limited|plc|llc"
    r"|l\.?p\.?|n\.?v\.?|s\.?a\.?|ag|se|class [a-z](?: ordinary| common)? shares?"
    r"|ordinary shares?|common stock|common shares|american depositary shares?|ads|adr|\(the\)))\s*$")
_GENERIC_FIRST = {
    "american", "china", "global", "first", "united", "new", "national", "international",
    "general", "great", "north", "south", "west", "east", "the", "digital", "smart",
    "golden", "energy", "capital", "future", "world", "pacific", "atlantic", "green",
}


def company_core(name: str) -> str:
    """'Inno Holdings Inc.' → 'Inno Holdings'; 'KNOREX LTD. Class A Ordinary
    Shares' → 'KNOREX'. Strips legal/share-class suffixes only."""
    s = re.sub(r"\s+", " ", (name or "").strip())
    prev = None
    while prev != s:
        prev = s
        s = _NAME_SUFFIX.sub("", s).strip(" ,.-")
    return s


def _mentions(title: str, symbol: str, core: str) -> bool:
    """Does the headline itself name this stock?"""
    # share classes are written BRK.B or BRK-B (or BRK/B) in headlines
    alt = "[-./]".join(re.escape(p) for p in symbol.split("-"))
    if len(symbol) <= 2:
        pat = rf"(?:\(|\$|:\s?){alt}\b"
    else:
        pat = rf"(?<![A-Za-z0-9]){alt}(?![A-Za-z0-9])"
    if re.search(pat, title):
        return True
    if core:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(core)}(?![A-Za-z0-9])", title, re.I):
            return True
        first = core.split()[0]
        if len(first) >= 5 and first.lower() not in _GENERIC_FIRST and \
                re.search(rf"(?<![A-Za-z0-9]){re.escape(first)}(?![A-Za-z0-9])", title, re.I):
            return True
    return False


def build_query(symbol: str, company: str = "", max_age_days: int = 7) -> str:
    """Google News query: ``"SYM" stock`` OR the company's core name, limited
    to the last ``max_age_days`` with Google's ``when:`` operator."""
    q = f'"{symbol}" stock'
    core = company_core(company)
    if core and core.upper() != symbol.upper():
        q += f' OR "{core}"'
    return f"{q} when:{max(1, int(max_age_days))}d"


def parse_rss(xml_text: str) -> Optional[List[Dict[str, Any]]]:
    """Google News RSS → ``[{"published","title","source","url"}]``, or
    ``None`` if the document is not parseable RSS."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    if root.find("channel") is None:
        return None
    out: List[Dict[str, Any]] = []
    for it in root.iter("item"):
        title = _normalise(it.findtext("title") or "")
        src_el = it.find("source")
        source = _normalise(src_el.text or "") if src_el is not None else ""
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        published = None
        pd_txt = it.findtext("pubDate")
        if pd_txt:
            try:
                dt = parsedate_to_datetime(pd_txt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                published = dt.astimezone(timezone.utc).isoformat(timespec="seconds")
            except (TypeError, ValueError):
                published = None
        url = (it.findtext("link") or "").strip()
        if not title or not url:
            continue
        out.append({"published": published, "title": title, "source": source or None, "url": url})
    return out


_NEWS_STATS = {"ok": 0, "fail": 0}


def headlines(symbol: str, company: str = "", max_age_days: int = 7, limit: int = 12) -> List[Dict[str, Any]]:
    """Recent headlines that name ``symbol`` (or ``company``), newest first:
    ``[{"published": iso|None, "title", "source", "url", "tags", "polarity"}]``.

    Undated items are dropped (they cannot be placed in time), duplicates by
    title are collapsed, and the list is cut to ``limit``. Cached 20 minutes
    per query. Empty list when nothing matches or the feed is unreachable.
    """
    sym = to_canonical(symbol)
    core = company_core(company)
    query = build_query(sym, company, max_age_days)
    url = GNEWS_URL.format(q=quote_plus(query))

    def fetch() -> Optional[List[Dict[str, Any]]]:
        text = net.get_text(url, headers={"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"})
        return parse_rss(text) if text else None

    items = net.cached("news", query, NEWS_MAX_AGE_S, fetch)
    if items is None:
        _NEWS_STATS["fail"] += 1
        net.record_status("Google News", _NEWS_STATS["ok"] > 0,
                          f"{_NEWS_STATS['ok']} ok / {_NEWS_STATS['fail']} failed queries")
        return []
    _NEWS_STATS["ok"] += 1
    net.record_status("Google News", True,
                      f"{_NEWS_STATS['ok']} ok / {_NEWS_STATS['fail']} failed queries")

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        if not it.get("published"):
            continue
        try:
            if datetime.fromisoformat(it["published"]) < cutoff:
                continue
        except ValueError:
            continue
        if not _mentions(it["title"], sym, core):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", it["title"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({**it, **classify(it["title"])})
    out.sort(key=lambda h: h["published"], reverse=True)
    return out[:limit]
