"""Short-side market data: borrow cost, daily short volume, short interest
and float.

Four independent public feeds, each answering a different question a
short seller asks before touching a name:

* **IBKR borrow** (``ibkr_borrow``) — can I borrow it, and what does it
  cost? Interactive Brokers publishes its stock-loan inventory for every
  US symbol on an anonymous FTP server, refreshed roughly every 15 min.
* **FINRA Reg SHO daily short volume** (``finra_short_volume``) — how much
  of each day's *off-exchange* (FINRA-reported) volume was sold short.
  Note that ``total_volume`` here is FINRA TRF/ADF/ORF volume only, not
  consolidated tape volume, so ratios are comparable across days and
  symbols but are not "share of all trading".
* **Nasdaq short interest** (``short_interest``) — the twice-monthly
  exchange-reported short interest. Nasdaq's API only serves it for
  Nasdaq-listed names; NYSE/NYSE American symbols come back empty.
* **Float** (``float_shares``) — float / shares outstanding / short % of
  float from Yahoo's quote summary via yfinance, shortlist only.

Nothing here raises on network failure: callers get empty containers or
``None`` fields, and every function reports its feed through
``net.record_status`` so the site shows which sources were live.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import config, net
from ..util import ET, is_trading_day, now_et, num, to_canonical

log = logging.getLogger(__name__)

# ── Endpoints ────────────────────────────────────────────────────────────
IBKR_URL = "ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt"
FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
NASDAQ_SI_URL = "https://api.nasdaq.com/api/quote/{sym}/short-interest?assetClass=stocks"

# ── Cache policy ─────────────────────────────────────────────────────────
IBKR_MAX_AGE_S = 30 * 60
SI_MAX_AGE_S = 12 * 3600             # short interest changes twice a month
FLOAT_MAX_AGE_S = 3 * 24 * 3600
FINRA_PUBLISH_ET = dtime(18, 0)      # FINRA posts the day's file by ~6 pm ET
FINRA_SLACK_DAYS = 6                 # extra trading days to probe for gaps

FINRA_COLUMNS = ["date", "symbol", "short_volume", "total_volume", "short_ratio"]


# ═════════════════════════════════════════════════════════════════════════
# IBKR stock-loan availability
# ═════════════════════════════════════════════════════════════════════════
_IBKR_SYM = re.compile(r"^([A-Z]{1,6})(?: ([A-Z]{1,4}))?$")


def ibkr_symbol(raw: str) -> Optional[str]:
    """IBKR contract symbol → canonical, or ``None`` if it is not a ticker.

    IBKR writes share classes and preferreds with a space ("BRK B",
    "ABR PRD"). Rows whose symbol starts with a digit are bonds/CUSIPs and
    dotted names ("AEGG.OLD", "BHLL.USD") are legacy or secondary
    contracts; neither is a tradable US ticker, so both are skipped.
    """
    s = (raw or "").strip().upper()
    m = _IBKR_SYM.match(s)
    if not m:
        return None
    base, suffix = m.group(1), m.group(2)
    if not suffix:
        return base
    if suffix.startswith("PR"):          # "ABR PRD" → Yahoo "ABR-PD"
        return f"{base}-P{suffix[2:]}"
    return to_canonical(f"{base}-{suffix}")


def _parse_available(raw: str) -> Optional[int]:
    """'35000' → 35000, '>10000000' → 10000000, '' → None."""
    v = num(raw)
    return None if v is None else int(v)


def _parse_ibkr_asof(line: str) -> Optional[str]:
    """'#BOF|2026.09.29|18:49:19' → '2026-09-29T18:49:19-04:00' (IBKR stamps ET)."""
    parts = line.strip().split("|")
    if len(parts) < 3:
        return None
    try:
        dt = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=ET).isoformat(timespec="seconds")


def parse_ibkr(text: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse IBKR's ``usa.txt``. Returns ``None`` when the file is
    truncated (no ``#EOF``) or has no ``#SYM`` header, so a partial download
    is never cached or mistaken for "nothing is borrowable"."""
    lines = text.splitlines()
    asof: Optional[str] = None
    cols: Optional[List[str]] = None
    out: Dict[str, Dict[str, Any]] = {}
    complete = False
    for line in lines:
        if not line:
            continue
        if line.startswith("#BOF"):
            asof = _parse_ibkr_asof(line)
            continue
        if line.startswith("#SYM"):
            cols = [c.strip().lstrip("#").upper() for c in line.split("|")]
            continue
        if line.startswith("#EOF"):
            complete = True
            break
        if line.startswith("#") or cols is None:
            continue
        rec = dict(zip(cols, line.split("|")))
        sym = ibkr_symbol(rec.get("SYM", ""))
        if not sym:
            continue
        out[sym] = {
            "fee_rate": num(rec.get("FEERATE")),
            "rebate_rate": num(rec.get("REBATERATE")),
            "available": _parse_available(rec.get("AVAILABLE", "")),
            "asof": asof,
        }
    if not complete or cols is None:
        return None
    return out


def _fetch_ibkr_text(timeout: float = 60.0, attempts: int = 2) -> Optional[str]:
    """Download ``usa.txt`` over anonymous FTP (``requests`` cannot do FTP)."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(IBKR_URL, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            if "#EOF" in text:
                return text
            log.warning("IBKR usa.txt truncated (%d bytes), attempt %d", len(raw), attempt)
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            log.warning("IBKR FTP failed (%s), attempt %d", e, attempt)
        if attempt + 1 < attempts:
            time.sleep(5.0)
    return None


def ibkr_borrow() -> Dict[str, Dict[str, Any]]:
    """Every symbol IBKR can lend, keyed by canonical symbol:
    ``{"fee_rate": 98.81, "rebate_rate": -94.93, "available": 35000, "asof": iso}``.

    ``fee_rate``/``rebate_rate`` are annualised percentages as IBKR prints
    them; ``available`` is shares available to borrow (IBKR caps the figure
    at 10,000,000 and writes ``>10000000``). A symbol absent from the dict
    means IBKR has no lendable inventory listed for it — not that it is
    unshortable everywhere. Cached 30 minutes. Empty dict on failure.
    """
    def fetch() -> Optional[Dict[str, Dict[str, Any]]]:
        text = _fetch_ibkr_text()
        return parse_ibkr(text) if text else None

    data = net.cached("ibkr", "usa", IBKR_MAX_AGE_S, fetch)
    if not data:
        net.record_status("IBKR borrow", False, "usa.txt unavailable")
        return {}
    asof = next(iter(data.values())).get("asof")
    net.record_status("IBKR borrow", True, f"{len(data):,} symbols, file stamped {asof}")
    return data


# ═════════════════════════════════════════════════════════════════════════
# FINRA Reg SHO daily short volume
# ═════════════════════════════════════════════════════════════════════════
def finra_symbol(raw: str) -> str:
    """FINRA/CMS symbol → canonical: 'BRK/B' → 'BRK-B', 'ABRpD' → 'ABR-PD'
    (lower-case suffixes mark preferreds 'p', rights 'r', when-issued 'w')."""
    s = (raw or "").strip()
    m = re.match(r"^([A-Z0-9./]+?)([a-z].*)$", s)
    if m:
        s = f"{m.group(1)}-{m.group(2).upper()}"
    return to_canonical(s)


def _finra_cache_path(d: date) -> Path:
    p = config.CACHE / "finra"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"CNMSshvol{d:%Y%m%d}.txt.gz"


def _finra_complete(text: str) -> bool:
    """A finished FINRA file ends with a trailer line holding the row count."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3 or not lines[0].startswith("Date|"):
        return False
    trailer = lines[-1].strip()
    return trailer.isdigit() and int(trailer) == len(lines) - 2


def _finra_text(d: date) -> Optional[str]:
    """One day's raw file: disk cache first (complete files are kept
    forever — FINRA never revises them), else the CDN. Missing days
    (holidays, not yet published) answer 403/404 → ``None``."""
    path = _finra_cache_path(d)
    if path.exists():
        try:
            return gzip.decompress(path.read_bytes()).decode("utf-8")
        except (OSError, ValueError, EOFError):
            path.unlink(missing_ok=True)
    # retries=1: a 403 from FINRA's CDN means "no such file", not throttling.
    text = net.get_text(FINRA_URL.format(ymd=f"{d:%Y%m%d}"), retries=1)
    if not text or not text.startswith("Date|"):
        return None
    if _finra_complete(text):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(text.encode("utf-8")))
        tmp.replace(path)
    return text


def parse_finra(text: str) -> pd.DataFrame:
    """Parse one CNMSshvol file into the public column layout."""
    df = pd.read_csv(io.StringIO(text), sep="|", dtype={"Date": str, "Symbol": str},
                     keep_default_na=False)
    df = df[df["Date"].str.fullmatch(r"\d{8}") & (df["Symbol"] != "")]
    if df.empty:
        return pd.DataFrame(columns=FINRA_COLUMNS)
    sv = pd.to_numeric(df["ShortVolume"], errors="coerce")
    tv = pd.to_numeric(df["TotalVolume"], errors="coerce")
    uniq = {s: finra_symbol(s) for s in df["Symbol"].unique()}
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
        "symbol": df["Symbol"].map(uniq),
        "short_volume": sv.astype(float),
        "total_volume": tv.astype(float),
    })
    out["short_ratio"] = (out["short_volume"] / out["total_volume"]).where(out["total_volume"] > 0)
    return out.reset_index(drop=True)


def _finra_candidate_days(n: int, at: Optional[datetime] = None) -> List[date]:
    """Most recent ``n`` trading days whose file should exist by now."""
    at = (at or now_et()).astimezone(ET)
    d = at.date()
    if not (is_trading_day(d) and at.time() >= FINRA_PUBLISH_ET):
        d -= timedelta(days=1)
    days: List[date] = []
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    return days


def finra_short_volume(days: int = 20) -> pd.DataFrame:
    """The last ``days`` published sessions of FINRA daily short volume.

    Columns ``date`` (ISO str), ``symbol`` (canonical), ``short_volume``,
    ``total_volume`` (FINRA-reported off-exchange volume, may be fractional),
    ``short_ratio`` (short/total, NaN when total is 0). Sorted by date then
    symbol. Holidays and not-yet-published days are skipped; up to
    ``FINRA_SLACK_DAYS`` extra sessions are probed to fill gaps.
    """
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for d in _finra_candidate_days(days + FINRA_SLACK_DAYS):
        if len(frames) >= days:
            break
        text = _finra_text(d)
        if text is None:
            missing.append(d.isoformat())
            continue
        frames.append(parse_finra(text))
    if not frames:
        net.record_status("FINRA short volume", False, "no daily files reachable")
        return pd.DataFrame(columns=FINRA_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    detail = f"{len(frames)} sessions {df['date'].min()}→{df['date'].max()}"
    if missing:
        detail += f"; missing {', '.join(missing[:4])}"
    net.record_status("FINRA short volume", True, detail)
    return df


# ═════════════════════════════════════════════════════════════════════════
# Nasdaq short interest
# ═════════════════════════════════════════════════════════════════════════
_SI_STATS = {"ok": 0, "empty": 0, "fail": 0}


def _nasdaq_symbol(sym: str) -> str:
    """Canonical 'BRK-B' → Nasdaq API 'BRK.B'."""
    return sym.strip().upper().replace("-", ".")


def _iso_mdy(s: Any) -> Optional[str]:
    """'09/15/2026' → '2026-09-15'."""
    try:
        return datetime.strptime(str(s).strip(), "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


def parse_short_interest(payload: Any) -> Optional[List[Dict[str, Any]]]:
    """Nasdaq short-interest JSON → rows newest first. ``None`` means the
    payload was not a recognisable answer; ``[]`` means Nasdaq answered and
    has no data (e.g. non-Nasdaq listing).

    ``days_to_cover`` is recomputed as interest ÷ average daily volume:
    Nasdaq floors its own figure at 1.0 and prints 0.0 when volume is
    missing, both of which would misstate the ratio. The value Nasdaq
    printed is kept as ``days_to_cover_reported``.
    """
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if data is None:
        status = payload.get("status") or {}
        return [] if status.get("rCode") in (200, 400) else None
    table = (data or {}).get("shortInterestTable") or {}
    rows = table.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        sd = _iso_mdy(r.get("settlementDate"))
        if not sd:
            continue
        interest = num(r.get("interest"))
        adv = num(r.get("avgDailyShareVolume"))
        if adv is not None and adv <= 0:
            adv = None
        dtc = round(interest / adv, 4) if (interest is not None and adv) else None
        out.append({
            "settlement_date": sd,
            "interest": interest,
            "avg_daily_volume": adv,
            "days_to_cover": dtc,
            "days_to_cover_reported": num(r.get("daysToCover")),
        })
    out.sort(key=lambda x: x["settlement_date"], reverse=True)
    return out


def short_interest(symbol: str) -> List[Dict[str, Any]]:
    """Exchange-reported short interest history for one symbol, newest first:
    ``[{"settlement_date", "interest", "avg_daily_volume", "days_to_cover"}]``.
    Empty list when Nasdaq has none (non-Nasdaq listings) or on failure.
    Cached 12 hours per symbol."""
    sym = to_canonical(symbol)
    url = NASDAQ_SI_URL.format(sym=_nasdaq_symbol(sym))

    def fetch() -> Optional[List[Dict[str, Any]]]:
        return parse_short_interest(net.get_json(url, headers=net.NASDAQ_HEADERS))

    rows = net.cached("short_interest", sym, SI_MAX_AGE_S, fetch)
    if rows is None:
        _SI_STATS["fail"] += 1
        rows = []
    elif rows:
        _SI_STATS["ok"] += 1
    else:
        _SI_STATS["empty"] += 1
    n = sum(_SI_STATS.values())
    net.record_status(
        "Nasdaq short interest",
        _SI_STATS["ok"] > 0 or _SI_STATS["fail"] == 0,
        f"{_SI_STATS['ok']}/{n} symbols with data, {_SI_STATS['fail']} failed",
    )
    return rows


# ═════════════════════════════════════════════════════════════════════════
# Float / shares outstanding (yfinance)
# ═════════════════════════════════════════════════════════════════════════
def _epoch_to_iso(v: Any) -> Optional[str]:
    x = num(v)
    if x is None or x <= 0:
        return None
    return datetime.fromtimestamp(x, tz=timezone.utc).date().isoformat()


def _pos(v: Any) -> Optional[float]:
    x = num(v)
    return x if (x is not None and x > 0) else None


def parse_float_info(info: Any) -> Optional[Dict[str, Any]]:
    """yfinance ``Ticker.info`` → float record, or ``None`` if the payload
    carries none of the fields (empty/throttled answer — never cached).

    ``short_pct_float`` is a fraction as Yahoo reports it (0.0197 = 1.97 %),
    dated by ``si_date`` (Yahoo's ``dateShortInterest``)."""
    if not isinstance(info, dict) or not info:
        return None
    flt = _pos(info.get("floatShares"))
    so = _pos(info.get("sharesOutstanding")) or _pos(info.get("impliedSharesOutstanding"))
    spf = num(info.get("shortPercentOfFloat"))
    if spf is not None and spf < 0:
        spf = None
    if flt is None and so is None and spf is None:
        return None
    return {
        "float": flt,
        "shares_out": so,
        "short_pct_float": spf,
        "shares_short": _pos(info.get("sharesShort")),
        "si_date": _epoch_to_iso(info.get("dateShortInterest")),
        "source": "yahoo",
        "asof": datetime.now(timezone.utc).date().isoformat(),
    }


class _RateLimited(Exception):
    pass


def _yf_info(symbol: str) -> Optional[dict]:
    """One ``Ticker.info`` call. Raises ``_RateLimited`` when Yahoo throttles."""
    import yfinance as yf
    try:
        from yfinance.exceptions import YFRateLimitError  # type: ignore
    except ImportError:  # pragma: no cover — older yfinance
        YFRateLimitError = ()  # type: ignore
    try:
        return yf.Ticker(symbol).info
    except YFRateLimitError as e:  # type: ignore[misc]
        raise _RateLimited(str(e))
    except Exception as e:  # yfinance raises assorted errors on bad symbols
        msg = str(e).lower()
        if "too many requests" in msg or "rate limit" in msg or "429" in msg:
            raise _RateLimited(str(e))
        log.debug("yfinance info %s failed: %s", symbol, e)
        return None


def float_shares(symbols: List[str], pause_s: float = 0.4) -> Dict[str, Dict[str, Any]]:
    """Float, shares outstanding and short % of float for a shortlist.

    ``{sym: {"float", "shares_out", "short_pct_float", "shares_short",
    "si_date", "source": "yahoo", "asof"}}``; fields are ``None`` when Yahoo
    lacks them and symbols with no data at all are omitted. Sequential with
    a small pause; stops early (remaining symbols omitted) if Yahoo starts
    rate limiting. Each hit is cached three days; empty answers never are.
    """
    out: Dict[str, Dict[str, Any]] = {}
    fetched = failed = 0
    throttled = False
    for raw in symbols:
        sym = to_canonical(raw)
        if sym in out:
            continue
        hit = net.cache_get("float", sym, FLOAT_MAX_AGE_S)
        if hit is not None:
            out[sym] = hit
            continue
        if throttled:
            continue
        try:
            rec = parse_float_info(_yf_info(sym))
        except _RateLimited as e:
            log.warning("Yahoo rate-limited float lookups at %s: %s", sym, e)
            throttled = True
            continue
        fetched += 1
        if rec is None:
            failed += 1
        else:
            net.cache_set("float", sym, rec)
            out[sym] = rec
        time.sleep(pause_s)
    detail = f"{len(out)}/{len(set(map(to_canonical, symbols)))} symbols ({fetched} fetched, {failed} empty)"
    if throttled:
        detail += " (stopped: Yahoo rate limit)"
    net.record_status("Yahoo float", bool(out) or not symbols, detail)
    return out
