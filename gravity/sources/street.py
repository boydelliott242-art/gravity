"""The Street's view: analyst consensus, the earnings calendar, market
movers, optional Danelfin AI scores, and deep links to every research site.

What we fetch vs. what we only link to
--------------------------------------
* Nasdaq's public JSON API (analyst ratings/targets, earnings calendar,
  movers) is fetched and parsed — it is the same data nasdaq.com renders.
* Danelfin is fetched **only** through its official REST API and **only**
  when the user supplies ``DANELFIN_API_KEY`` (free tier: 500 calls/month,
  10 calls/minute). Without a key ``danelfin()`` makes no network call.
* Zacks, Bloomberg, WSJ, TipRanks, MarketBeat and the rest are **links
  only** (``deep_links``). Their ratings are proprietary; we never scrape
  or republish them.

Micro-caps usually have no analyst coverage. That is reported as ``None``
fields / empty lists — never filled in.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .. import config, net
from ..util import ET, market_phase, num, to_canonical

log = logging.getLogger(__name__)

NASDAQ_API = "https://api.nasdaq.com/api"
DANELFIN_API = "https://apirest.danelfin.com/ranking"

ANALYST_MAX_AGE_S = 12 * 3600
EARNINGS_MAX_AGE_S = 3 * 3600
MOVERS_MAX_AGE_S = 5 * 60
DANELFIN_MAX_AGE_S = 20 * 3600       # scores are published once per day
DANELFIN_RPS = 10.0 / 60.0           # free tier: 10 calls per minute
DANELFIN_MAX_CALLS = 20              # per run; 500/month ≈ 22 per trading day
DANELFIN_PROBE = 3                   # first N calls all failing → key/quota problem, stop
DANELFIN_MAX_CONSEC_FAIL = 5         # later: stop after this many misses in a row


def _nasdaq_symbol(sym: str) -> str:
    """Canonical 'BRK-B' → Nasdaq API 'BRK.B'."""
    return sym.strip().upper().replace("-", ".")


def _money(x: Any) -> Optional[float]:
    """Parse Nasdaq money strings, honouring accounting negatives:
    '$1.16' → 1.16, '($0.14)' → -0.14, 'N/A'/'' → None."""
    v = num(x)
    if v is None:
        return None
    s = str(x)
    if "(" in s and ")" in s and v > 0:
        v = -v
    return v


def _int(x: Any) -> Optional[int]:
    v = num(x)
    return None if v is None else int(v)


# ═════════════════════════════════════════════════════════════════════════
# Analyst consensus (Nasdaq / Zacks-sourced consensus shown on nasdaq.com)
# ═════════════════════════════════════════════════════════════════════════
_ANALYST_STATS = {"ok": 0, "fail": 0}
_N_ANALYSTS = re.compile(r"Based on\s+([\d,]+)\s+analyst", re.I)


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _iso_date(s: Any) -> Optional[str]:
    """Accept 'MM/DD/YYYY', 'YYYY-MM-DD' or epoch seconds → ISO date."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(float(s), tz=timezone.utc).date().isoformat()
    txt = str(s).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(txt[:10] if fmt == "%Y-%m-%d" else txt, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_rating_changes(rows: Any) -> List[Dict[str, Any]]:
    """Nasdaq ``upgradesDowngrades`` → ``[{"date","firm","action","from","to"}]``
    newest first. Field names have varied over time, so several are tried.

    As of 2026-09 Nasdaq returns this list empty even for heavily covered
    names (TSLA, AAPL, NVDA), so ``changes`` is normally ``[]``; the parser
    stays tolerant in case the feed is repopulated."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        out.append({
            "date": _iso_date(_first(r, "date", "dateOfChange", "changeDate")),
            "firm": _first(r, "firm", "brokerName", "analystFirm", "broker"),
            "action": _first(r, "action", "actionType", "type", "changeType"),
            "from": _first(r, "from", "fromRating", "previousRating", "oldRating"),
            "to": _first(r, "to", "toRating", "newRating", "rating"),
        })
    out.sort(key=lambda c: c["date"] or "", reverse=True)
    return out


def _pos_num(x: Any) -> Optional[float]:
    v = num(x)
    return v if (v is not None and v > 0) else None


def _latest_consensus_date(hist: Any) -> Optional[str]:
    """Date of the newest ``historicalConsensus`` point (Nasdaq samples the
    consensus monthly), i.e. when the price target was last struck."""
    best: Optional[str] = None
    for pt in hist or []:
        z = pt.get("z") if isinstance(pt, dict) else None
        d = _iso_date(z.get("date")) if isinstance(z, dict) else None
        if d and (best is None or d > best):
            best = d
    return best


def parse_analyst(symbol: str, ratings: Any, target: Any) -> Optional[Dict[str, Any]]:
    """Combine the two Nasdaq analyst payloads. ``None`` only when neither
    answered at all; an answered-but-empty payload gives ``None`` fields and
    ``covered: False`` (typical for micro-caps — nothing is filled in)."""
    rd = ratings.get("data") if isinstance(ratings, dict) else None
    td = target.get("data") if isinstance(target, dict) else None
    answered = isinstance(ratings, dict) or isinstance(target, dict)
    if not answered:
        return None
    rd = rd if isinstance(rd, dict) else {}
    td = td if isinstance(td, dict) else {}
    co = td.get("consensusOverview") if isinstance(td.get("consensusOverview"), dict) else {}

    n_analysts: Optional[int] = None
    m = _N_ANALYSTS.search(str(rd.get("ratingsSummary") or ""))
    if m:
        n_analysts = int(m.group(1).replace(",", ""))
    buy, hold, sell = _int(co.get("buy")), _int(co.get("hold")), _int(co.get("sell"))
    if n_analysts is None and None not in (buy, hold, sell) and (buy + hold + sell) > 0:
        n_analysts = buy + hold + sell

    pt = _pos_num(co.get("priceTarget"))
    mean_rating = (str(rd.get("meanRatingType") or "").strip() or None)
    return {
        "mean_rating": mean_rating,
        "n_analysts": n_analysts,
        "changes": parse_rating_changes(rd.get("upgradesDowngrades")),
        "price_target": pt,
        "price_target_low": _pos_num(co.get("lowPriceTarget")),
        "price_target_high": _pos_num(co.get("highPriceTarget")),
        "price_target_date": _latest_consensus_date(td.get("historicalConsensus")) if pt else None,
        "buy": buy, "hold": hold, "sell": sell,
        "covered": bool(mean_rating or n_analysts or pt),
        "asof": datetime.now(ET).date().isoformat(),
        "source": "Nasdaq analyst research",
        "source_url": f"https://www.nasdaq.com/market-activity/stocks/{_nasdaq_symbol(symbol).lower()}/analyst-research",
    }


def analyst(symbol: str) -> Optional[Dict[str, Any]]:
    """Sell-side consensus for one symbol:
    ``{"mean_rating", "n_analysts", "changes", "price_target", "source_url",
    "covered", "price_target_low", "price_target_high", "price_target_date",
    "buy", "hold", "sell", "asof", "source"}``.

    Uncovered names (most micro-caps) give ``None`` fields and
    ``covered: False``; ``None`` overall means Nasdaq could not be reached.
    A full answer (both endpoints responded) is cached 12 hours; a partial
    one is returned but not cached, so a transient failure cannot hide a
    price target for half a day.
    """
    sym = to_canonical(symbol)
    nsym = quote(_nasdaq_symbol(sym))
    rec = net.cache_get("analyst", sym, ANALYST_MAX_AGE_S)
    if rec is None:
        ratings = net.get_json(f"{NASDAQ_API}/analyst/{nsym}/ratings", headers=net.NASDAQ_HEADERS)
        target = net.get_json(f"{NASDAQ_API}/analyst/{nsym}/targetprice", headers=net.NASDAQ_HEADERS)
        rec = parse_analyst(sym, ratings, target)
        if rec is not None and isinstance(ratings, dict) and isinstance(target, dict):
            net.cache_set("analyst", sym, rec)
    _ANALYST_STATS["ok" if rec is not None else "fail"] += 1
    n = sum(_ANALYST_STATS.values())
    net.record_status("Nasdaq analyst ratings", _ANALYST_STATS["ok"] > 0,
                      f"{_ANALYST_STATS['ok']}/{n} symbols answered")
    return rec


# ═════════════════════════════════════════════════════════════════════════
# Earnings calendar
# ═════════════════════════════════════════════════════════════════════════
_TIME_MAP = {
    "time-pre-market": "pre-market",
    "time-after-hours": "after-hours",
    "time-not-supplied": None,
}


def parse_earnings(payload: Any, d: Optional[date] = None) -> Optional[List[Dict[str, Any]]]:
    """Nasdaq earnings-calendar JSON → rows. ``None`` = unrecognised payload;
    ``[]`` = Nasdaq answered and nobody reports that day (``data.rows`` null)."""
    if not isinstance(payload, dict) or "data" not in payload:
        return None
    data = payload.get("data") or {}
    out: List[Dict[str, Any]] = []
    for r in data.get("rows") or []:
        if not isinstance(r, dict):
            continue
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        t = r.get("time")
        mc = _money(r.get("marketCap"))
        out.append({
            "symbol": to_canonical(sym),
            "name": r.get("name") or None,
            "date": d.isoformat() if d else None,
            "time": _TIME_MAP.get(t, None) if t else None,
            "eps_forecast": _money(r.get("epsForecast")),
            "n_ests": _int(r.get("noOfEsts")),
            "market_cap": mc if (mc is not None and mc > 0) else None,
            "fiscal_quarter": r.get("fiscalQuarterEnding") or None,
        })
    return out


def earnings_calendar(d: date) -> List[Dict[str, Any]]:
    """Companies reporting on ``d``:
    ``[{"symbol", "time": "pre-market"|"after-hours"|None, "eps_forecast",
    "n_ests", "market_cap", "name", "date", "fiscal_quarter"}]``. ``time``
    is ``None`` when the company has not said; ``eps_forecast`` keeps
    accounting negatives ('($0.14)' → -0.14). Cached 3 hours per date."""
    ds = d.isoformat()

    def fetch() -> Optional[List[Dict[str, Any]]]:
        return parse_earnings(net.get_json(f"{NASDAQ_API}/calendar/earnings",
                                           headers=net.NASDAQ_HEADERS, params={"date": ds}), d)

    rows = net.cached("earnings", ds, EARNINGS_MAX_AGE_S, fetch)
    if rows is None:
        net.record_status("Nasdaq earnings calendar", False, f"{ds} unavailable")
        return []
    net.record_status("Nasdaq earnings calendar", True, f"{len(rows)} reporting {ds}")
    return rows


# ═════════════════════════════════════════════════════════════════════════
# Market movers
# ═════════════════════════════════════════════════════════════════════════
_STATUS_FOR_PHASE = {"pre-market": "preMarket", "after-hours": "afterHours"}
_ASOF = re.compile(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]M)")


def _parse_asof(s: Any) -> Optional[str]:
    """'Data as of Sep 29, 2026 4:15 PM ET' → '2026-09-29T16:15:00-04:00'."""
    m = _ASOF.search(str(s or ""))
    if not m:
        return None
    try:
        dt = datetime.strptime(re.sub(r"\s+", " ", m.group(1)), "%b %d, %Y %I:%M %p")
    except ValueError:
        return None
    return dt.replace(tzinfo=ET).isoformat(timespec="seconds")


def _mover_rows(block: Any, change_is_pct: bool) -> List[Dict[str, Any]]:
    table = (block or {}).get("table") or {}
    out: List[Dict[str, Any]] = []
    for r in table.get("rows") or []:
        sym = (r.get("symbol") or "").strip()
        price = _money(r.get("lastSalePrice"))
        if not sym or price is None:
            continue
        chg = _money(r.get("lastSaleChange"))
        item: Dict[str, Any] = {"symbol": to_canonical(sym), "name": r.get("name") or None,
                                "price": price, "change_pct": None}
        if change_is_pct:
            pct = _money(r.get("change"))
            item["change_pct"] = None if pct is None else round(pct / 100.0, 6)
        else:
            # Most-active tables put share volume in "change"; derive the % move
            # from last price and net change (prev close = price − change).
            item["volume"] = _money(r.get("change"))
            if chg is not None and (price - chg) > 0:
                item["change_pct"] = round(chg / (price - chg), 6)
        out.append(item)
    return out


def board_label(asof: Optional[str]) -> Optional[str]:
    """What the served board actually is, from Nasdaq's own timestamp:
    ``"pre-market"`` (stamped before 09:30 ET), ``"intraday"`` (09:30–16:00)
    or ``"post-close"`` (16:00 or later — Nasdaq's closing board is stamped
    4:15 PM ET). ``None`` when the payload carried no timestamp."""
    if not asof:
        return None
    try:
        t = datetime.fromisoformat(asof).astimezone(ET).time()
    except ValueError:
        return None
    if t < dtime(9, 30):
        return "pre-market"
    if t < dtime(16, 0):
        return "intraday"
    return "post-close"


def parse_movers(payload: Any, requested: str) -> Optional[Dict[str, Any]]:
    """Nasdaq market-movers JSON → contract shape. ``None`` = unrecognised."""
    data = payload.get("data") if isinstance(payload, dict) else None
    stocks = data.get("STOCKS") if isinstance(data, dict) else None
    if not isinstance(stocks, dict):
        return None
    adv, dec = stocks.get("MostAdvanced"), stocks.get("MostDeclined")
    act = stocks.get("MostActiveByShareVolume")
    asof = None
    for blk in (adv, dec, act):
        if isinstance(blk, dict):
            asof = _parse_asof(blk.get("dataAsOf") or blk.get("lastTradeTimestamp"))
            if asof:
                break
    return {
        "premarket_gainers": _mover_rows(adv, True),
        "premarket_losers": _mover_rows(dec, True),
        "most_active": _mover_rows(act, False),
        "session": requested,
        "board": board_label(asof),
        "asof": asof,
        "source_url": "https://www.nasdaq.com/market-activity/most-active",
    }


def movers() -> Dict[str, Any]:
    """Nasdaq's top gainers / losers / most active (≤ 20 each):
    ``{"premarket_gainers", "premarket_losers", "most_active", "session",
    "board", "asof", "source_url"}``; items ``{"symbol","name","price",
    "change_pct"}`` (most-active items also carry ``volume``) with
    ``change_pct`` a fraction (+1.94 = +194 %).

    During pre-market (04:00–09:30 ET) the pre-market board is requested
    (``session`` = what was asked for). Nasdaq may still serve the prior
    closing board — verified 2026-09-29 after hours, when all three
    ``exchangestatus`` values returned the same 4:15 PM board — so
    ``board``/``asof`` (derived from Nasdaq's own "Data as of" stamp) say
    what the data really is and the site must label it from those fields,
    not from the key names. Includes warrants/rights/ETFs as Nasdaq lists
    them; filter against the universe downstream.
    """
    requested = _STATUS_FOR_PHASE.get(market_phase(), "currentMarket")

    def fetch() -> Optional[Dict[str, Any]]:
        payload = net.get_json(f"{NASDAQ_API}/marketmovers", headers=net.NASDAQ_HEADERS,
                               params={"assetclass": "stocks", "exchangestatus": requested,
                                       "limit": 25})
        return parse_movers(payload, requested)

    res = net.cached("movers", requested, MOVERS_MAX_AGE_S, fetch)
    if res is None:
        net.record_status("Nasdaq movers", False, f"{requested} board unavailable")
        return {"premarket_gainers": [], "premarket_losers": [], "most_active": [],
                "session": requested, "board": None, "asof": None, "source_url": None}
    net.record_status("Nasdaq movers", True,
                      f"requested {requested}; served {res.get('board')} board as of {res.get('asof')}")
    return res


# ═════════════════════════════════════════════════════════════════════════
# Danelfin (official API, opt-in)
# ═════════════════════════════════════════════════════════════════════════
def _score(v: Any) -> Optional[float]:
    x = num(v)
    return x if (x is not None and 1 <= x <= 10) else None


def parse_danelfin(symbol: str, payload: Any) -> Optional[Dict[str, Any]]:
    """Danelfin ``GET /ranking?ticker=`` answers a date-keyed object,
    ``{"2026-09-29": {"aiscore": 7, "technical": 6, "fundamental": 5,
    "sentiment": 7, "low_risk": 4}, ...}``; a date's value may be ``null``
    and the ticker-less (``date=``) form nests one level deeper under the
    ticker. Returns the latest non-empty dated entry, or ``None``."""
    if not isinstance(payload, dict):
        return None
    best_date, best = None, None
    for k, v in payload.items():
        d = _iso_date(k)
        if d is None or not isinstance(v, dict):
            continue
        if symbol in v and isinstance(v[symbol], dict):
            v = v[symbol]
        if best_date is None or d > best_date:
            best_date, best = d, v
    if best is None:
        return None
    rec = {
        "ai_score": _score(best.get("aiscore", best.get("ai_score"))),
        "technical": _score(best.get("technical")),
        "fundamental": _score(best.get("fundamental")),
        "sentiment": _score(best.get("sentiment")),
        "low_risk": _score(best.get("low_risk", best.get("lowrisk"))),
        "date": best_date,
        "source_url": f"https://danelfin.com/stock/{symbol}",
    }
    if all(rec[k] is None for k in ("ai_score", "technical", "fundamental", "sentiment", "low_risk")):
        return None
    return rec


def danelfin(symbols: List[str], max_calls: int = DANELFIN_MAX_CALLS) -> Dict[str, Dict[str, Any]]:
    """Danelfin AI Score (1–10) and sub-scores per symbol:
    ``{"ai_score","technical","fundamental","sentiment","low_risk","date","source_url"}``.

    Returns ``{}`` without touching the network unless
    ``config.DANELFIN_API_KEY`` is set. Official endpoint only:
    ``GET https://apirest.danelfin.com/ranking?ticker=SYM`` with header
    ``x-api-key``. At most ``max_calls`` uncached API calls per run (free
    tier = 500/month), paced at 10/min. ``net.get`` reports a 404 (ticker
    not covered) and a 401/403/429 (bad key, quota) the same way, so the
    run stops when the first ``DANELFIN_PROBE`` calls all fail (the key or
    quota is the problem) or after ``DANELFIN_MAX_CONSEC_FAIL`` failures in
    a row — never burning the quota on a dead key.

    Danelfin scores rank the odds of beating the market over ~3 months; a
    low score is not a dump call and is shown as context only. Cached 20
    hours per symbol; failures are never cached.
    """
    key = config.DANELFIN_API_KEY
    if not key:
        net.record_status("Danelfin API", False, "DANELFIN_API_KEY not set (optional)")
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    calls = ok = consecutive_fail = 0
    stopped = ""
    for raw in symbols:
        sym = to_canonical(raw)
        if sym in out:
            continue
        hit = net.cache_get("danelfin", sym, DANELFIN_MAX_AGE_S)
        if hit is not None:
            out[sym] = hit
            continue
        if stopped:
            continue
        if calls >= max_calls:
            stopped = f"per-run cap of {max_calls} calls reached"
            continue
        calls += 1
        payload = net.get_json(DANELFIN_API, headers={"x-api-key": key, "Accept": "application/json"},
                               params={"ticker": sym}, rps=DANELFIN_RPS, retries=0)
        rec = parse_danelfin(sym, payload)
        if rec is None:
            consecutive_fail += 1
        else:
            ok += 1
            consecutive_fail = 0
            net.cache_set("danelfin", sym, rec)
            out[sym] = rec
        if ok == 0 and calls >= DANELFIN_PROBE:
            stopped = f"first {calls} calls all failed (check key/quota)"
        elif consecutive_fail >= DANELFIN_MAX_CONSEC_FAIL:
            stopped = f"{consecutive_fail} failures in a row"
    uniq = len({to_canonical(s) for s in symbols})
    detail = f"{len(out)}/{uniq} symbols, {calls} API calls"
    if stopped:
        detail += f" (stopped: {stopped})"
    net.record_status("Danelfin API", bool(out) or calls == 0, detail)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Deep links
# ═════════════════════════════════════════════════════════════════════════
def _exchange_code(exchange: Optional[str]) -> Optional[str]:
    """Normalise an exchange label to NASDAQ | NYSE | AMEX | OTC | None."""
    e = (exchange or "").strip().upper().replace(" ", "")
    if not e:
        return None
    if "NASDAQ" in e or e in ("NMS", "NCM", "NGM", "NAS", "NASDAQCM", "NASDAQGS", "NASDAQGM"):
        return "NASDAQ"
    if any(t in e for t in ("AMERICAN", "AMEX", "NYSEMKT", "ASE", "ARCA", "CBOE", "BATS")):
        return "AMEX"
    if "NYSE" in e or e == "NYQ":
        return "NYSE"
    if "OTC" in e or e in ("PNK", "PINK", "OTCQB", "OTCQX"):
        return "OTC"
    return None


def deep_links(symbol: str, cik: Optional[int] = None, exchange: Optional[str] = None) -> Dict[str, str]:
    """Ordered label → URL for the research sites a trader would check.

    ``exchange`` (optional) sharpens TradingView and MarketBeat URLs; both
    redirect to the right listing when it is unknown. SEC EDGAR uses the
    CIK when known, else EDGAR's ticker lookup. These are links only —
    nothing behind them is fetched or republished.
    """
    sym = to_canonical(symbol)                 # Yahoo style: BRK-B
    dot = sym.replace("-", ".")                # BRK.B (Zacks, Nasdaq, TradingView, ...)
    low = dot.lower()
    ex = _exchange_code(exchange)
    tv_ex = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "AMEX": "AMEX", "OTC": "OTC"}.get(ex or "")
    mb_ex = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "AMEX": "NYSEAMERICAN", "OTC": "OTCMKTS"}.get(ex or "", "NASDAQ")
    edgar_id = str(int(cik)) if cik else sym
    links: Dict[str, str] = {
        "Zacks": f"https://www.zacks.com/stock/quote/{dot}",
        "Danelfin": f"https://danelfin.com/stock/{dot}",
        "Bloomberg": f"https://www.bloomberg.com/quote/{dot.replace('.', '/')}:US",
        "WSJ": f"https://www.wsj.com/market-data/quotes/{dot}",
        "Finviz": f"https://finviz.com/quote.ashx?t={sym}",
        "TradingView": (f"https://www.tradingview.com/symbols/{tv_ex}-{dot}/" if tv_ex
                        else f"https://www.tradingview.com/symbols/{dot}/"),
        "Stocktwits": f"https://stocktwits.com/symbol/{dot}",
        "Yahoo": f"https://finance.yahoo.com/quote/{sym}/",
        "Nasdaq": f"https://www.nasdaq.com/market-activity/stocks/{low}",
        "SEC EDGAR": ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                      f"&CIK={edgar_id}&type=&dateb=&owner=include&count=40"),
        "Fintel": f"https://fintel.io/ss/us/{low}",
        "iBorrowDesk": f"https://www.iborrowdesk.com/report/{dot}",
        "TipRanks": f"https://www.tipranks.com/stocks/{low}/forecast",
        "MarketBeat": f"https://www.marketbeat.com/stocks/{mb_ex}/{sym}/",
    }
    if ex == "OTC":
        links["OTC Markets"] = f"https://www.otcmarkets.com/stock/{sym}/overview"
    links["Trade halts"] = "https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts"
    return links
