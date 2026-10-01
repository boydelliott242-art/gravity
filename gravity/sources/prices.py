"""Daily price history, split events, and pre-market quotes.

History comes from Yahoo Finance through the ``yfinance`` library (raw HTTP
to query1/query2 is rate-limited from this machine; yfinance's browser
impersonation is not). Frames are **split-adjusted, not dividend-adjusted**:
``auto_adjust=False`` OHLC, which Yahoo already scales for splits.

The per-symbol disk cache (``config.CACHE/"prices"/{SYM}.pkl``) makes the
daily refresh of ~3,500 names cheap:

* a cache written after the most recent session's close is *current* and is
  served without any request;
* otherwise only the last ~10 sessions are fetched (batched ``yf.download``),
  checked, and appended;
* a full-period refetch happens when the window shows a split the cache did
  not know about, when the overlapping bars disagree with the cache (Yahoo
  re-adjusted history), when the cache is too old or too short, or when it
  came from the Nasdaq fallback.

Failures degrade in a fixed order: batch → individual yfinance retry →
Nasdaq historical (full period) → the stale cached frame (flagged
``attrs["stale"]``) → omitted. On a Yahoo rate limit the run backs off hard
(60 s, then 180 s) and, if still refused, stops calling Yahoo for the rest
of the call and serves cached frames as stale. Nothing is imputed: halted
sessions stay in the frame with ``volume == 0``; rows without a positive
price are dropped; a session still in progress is never cached.

Nasdaq's historical endpoint has been seen returning split-adjusted bars
(INHD, 2026-09) although it was documented as raw, so the fallback tests
each known split instead of assuming either way (``adjust_for_splits``).

Every returned frame carries ``attrs``: ``symbol``, ``source``
(``"yahoo"``/``"nasdaq"``), ``unadjusted`` (True only when Nasdaq data could
not be checked against known splits), ``fetched_at`` (UTC ISO) and ``stale``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import config, net
from ..util import ET, is_trading_day, now_et, num, prev_trading_day, target_session, to_canonical

log = logging.getLogger(__name__)

YAHOO_STATUS = "Yahoo Finance prices"
NASDAQ_STATUS = "Nasdaq quotes"
BENCHMARK = "IWM"
PRICE_COLS = ["open", "high", "low", "close", "volume"]

NASDAQ_API = "https://api.nasdaq.com/api/quote"

# ── Tunables ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 60                 # symbols per yf.download call
CHUNK_PAUSE_S = 2.0             # polite pause between chunks
INDIVIDUAL_PAUSE_S = 0.5        # pause between single-symbol Yahoo retries
INCREMENTAL_SESSIONS = 10       # an incremental refresh fetches at least this many sessions
OVERLAP_SESSIONS = 5            # …and re-fetches at least this many already-cached bars
DRIFT_TOLERANCE = 0.03          # median |fresh/cached − 1| on overlap above this → full refetch
MAX_INCREMENTAL_GAP_DAYS = 120  # cache older than this (calendar days) → full refetch
COVERAGE_SLACK_DAYS = 7         # a cache may start this much later than asked and still "cover"
FRESH_AFTER_CLOSE_MIN = 30      # a cache written ≥ this long after the last close is current…
LATE_BAR_RECHECK_H = 3          # …unless it lacks that session's bar and was written < this long after
RATE_LIMIT_BACKOFF_S = (60.0, 180.0)   # sleeps before re-trying rate-limited symbols
STRAGGLERS = 3                         # ≤ this many refusals in a batch = bad tickers, not throttling
NASDAQ_FALLBACK_MAX = 250       # cap on Nasdaq history fallbacks per call (3 req/s host)
SPLITS_CACHE_S = 20 * 3600      # split lists fetched outside load_history
PREMARKET_WORKERS = 8
_CACHE_VERSION = 1

#: Stats of the most recent ``load_history`` call (for scripts and the site).
LAST_RUN: Dict[str, Any] = {}

_sleep = time.sleep             # monkeypatched in tests


def _now() -> datetime:
    """Current time in US/Eastern (indirection so tests can pin the clock)."""
    return now_et()


# ── Small helpers ────────────────────────────────────────────────────────
def _canon_list(symbols: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in symbols or []:
        if not s or not str(s).strip():
            continue
        c = to_canonical(str(s))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def nasdaq_symbol(symbol: str) -> str:
    """Canonical ``BRK-A`` → Nasdaq API ``BRK.A``."""
    return symbol.replace("-", ".").upper()


def _chunks(seq: Sequence[str], n: int) -> Iterator[List[str]]:
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


def _sessions_back(d: date, k: int) -> date:
    """The trading day ``k`` sessions before ``d`` (``d`` itself need not trade)."""
    for _ in range(max(0, k)):
        d = prev_trading_day(d)
    return d


def period_start(period: str, today: date) -> Optional[date]:
    """First calendar date covered by a yfinance-style period string.

    ``"3y"``, ``"6mo"``, ``"2wk"``, ``"10d"`` (trading sessions), ``"ytd"``;
    ``"max"`` → ``None`` (no lower bound). Raises ``ValueError`` otherwise.
    """
    p = (period or "").strip().lower()
    if p == "max":
        return None
    if p == "ytd":
        return date(today.year, 1, 1)
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", p)
    if not m:
        raise ValueError(f"unsupported period {period!r}")
    n, unit = int(m.group(1)), m.group(2)
    ts = pd.Timestamp(today)
    if unit == "y":
        return (ts - pd.DateOffset(years=n)).date()
    if unit == "mo":
        return (ts - pd.DateOffset(months=n)).date()
    if unit == "wk":
        return (ts - pd.DateOffset(weeks=n)).date()
    return _sessions_back(today, max(0, n - 1)) if is_trading_day(today) else _sessions_back(today, n)


def _last_close_dt(now: datetime) -> datetime:
    """16:00 ET of the most recent session that has already closed."""
    now = now.astimezone(ET)
    d = now.date()
    day = d if (is_trading_day(d) and now.time() >= dtime(16, 0)) else prev_trading_day(d)
    return datetime.combine(day, dtime(16, 0), tzinfo=ET)


def _session_open(now: datetime) -> bool:
    """True while today's regular session has not closed yet (its daily bar
    would be partial)."""
    now = now.astimezone(ET)
    return is_trading_day(now.date()) and now.time() < dtime(16, 0)


def _utc_iso() -> str:
    """Current time (per ``_now``) as a UTC ISO string — cache timestamps."""
    return _now().astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def dollar_volume_median(df: Optional[pd.DataFrame], n: int = 20) -> float:
    """Median of ``close × volume`` over the last ``n`` bars (halted
    sessions count as $0). NaN when there are no bars."""
    if df is None or len(df) == 0 or n <= 0:
        return float("nan")
    tail = df.iloc[-int(n):]
    dv = (tail["close"].astype(float) * tail["volume"].astype(float)).dropna()
    return float(dv.median()) if len(dv) else float("nan")


# ── Frame hygiene ────────────────────────────────────────────────────────
def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in PRICE_COLS})
    df.index = pd.DatetimeIndex([], name="date")
    return df


def clean_bars(df: Optional[pd.DataFrame], now: Optional[datetime] = None) -> pd.DataFrame:
    """Normalise any OHLCV frame to the contract shape.

    Lower-case ``open high low close volume`` float columns; tz-naive,
    midnight, ascending, unique ``DatetimeIndex`` named ``date`` (later
    duplicates win); rows with a missing or non-positive price dropped;
    missing volume → 0 (a halt); today's bar dropped while the session is
    still open (it would be partial).
    """
    if df is None or len(df) == 0:
        return empty_frame()
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if not set(("open", "high", "low", "close")).issubset(d.columns):
        return empty_frame()
    if "volume" not in d.columns:
        d["volume"] = np.nan
    d = d[PRICE_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(d.index))
    if idx.tz is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    d.index = idx.normalize()
    d.index.name = "date"
    px = d[["open", "high", "low", "close"]]
    d = d[px.notna().all(axis=1) & (px > 0).all(axis=1)]
    vol = d["volume"].where(d["volume"] >= 0)
    d["volume"] = vol.fillna(0.0)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    now = now or _now()
    if _session_open(now):
        d = d[d.index < pd.Timestamp(now.astimezone(ET).date())]
    return d


def _splits_from_series(s: Optional[pd.Series]) -> List[Dict[str, Any]]:
    """yfinance 'Stock Splits' column/series → ``[{"date", "ratio"}]`` ascending."""
    if s is None or len(s) == 0:
        return []
    s = pd.to_numeric(s, errors="coerce")
    s = s[s.notna() & (s > 0) & (s != 1.0)]
    out = {}
    for ts, v in s.items():
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert(ET).tz_localize(None)
        out[t.strftime("%Y-%m-%d")] = round(float(v), 6)
    return [{"date": k, "ratio": out[k]} for k in sorted(out)]


def _merge_splits(*lists: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    by_date: Dict[str, float] = {}
    for lst in lists:
        for sp in lst or []:
            by_date[str(sp["date"])] = float(sp["ratio"])
    return [{"date": d, "ratio": by_date[d]} for d in sorted(by_date)]


# ── Disk cache ───────────────────────────────────────────────────────────
def _cache_dir() -> Path:
    d = config.CACHE / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(symbol: str) -> Path:
    safe = re.sub(r"[^A-Z0-9\-]", "_", symbol.upper())
    return _cache_dir() / f"{safe}.pkl"


def read_cache(symbol: str) -> Optional[Dict[str, Any]]:
    """The cached entry for ``symbol`` or ``None`` (missing/corrupt)."""
    p = _cache_path(symbol)
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            e = pickle.load(fh)
    except Exception:  # corrupt/partial pickle → behave as a cache miss
        log.warning("prices: unreadable cache for %s, ignoring", symbol)
        return None
    if not isinstance(e, dict) or e.get("version") != _CACHE_VERSION or not isinstance(e.get("df"), pd.DataFrame):
        return None
    return e


def _write_cache(symbol: str, entry: Dict[str, Any]) -> None:
    p = _cache_path(symbol)
    tmp = p.with_suffix(".pkl.tmp")
    entry = dict(entry, version=_CACHE_VERSION, symbol=symbol)
    with tmp.open("wb") as fh:
        pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def _covers(entry: Dict[str, Any], want_start: Optional[date]) -> bool:
    cov = entry.get("coverage_start")
    if cov is None:
        return True
    if want_start is None:
        return False
    try:
        cov_d = date.fromisoformat(str(cov))
    except ValueError:
        return False
    return cov_d <= want_start + timedelta(days=COVERAGE_SLACK_DAYS)


def _is_current(entry: Dict[str, Any], now: datetime) -> bool:
    """A cache is current when it was written ≥ 30 min after the most recent
    close **and** either holds that session's bar or was written ≥ 3 h after
    the close (Yahoo publishes some illiquid names' bars late; one late
    re-check, then we accept that the bar does not exist, e.g. no trades)."""
    fetched = _parse_iso(entry.get("fetched_at"))
    if fetched is None:
        return False
    close_dt = _last_close_dt(now)
    if fetched < close_dt + timedelta(minutes=FRESH_AFTER_CLOSE_MIN):
        return False
    df = entry.get("df")
    if df is not None and len(df) and df.index.max().date() >= close_dt.date():
        return True
    return fetched >= close_dt + timedelta(hours=LATE_BAR_RECHECK_H)


def _frame_out(entry: Dict[str, Any], want_start: Optional[date], stale: bool = False) -> pd.DataFrame:
    df = entry["df"]
    if want_start is not None and len(df):
        df = df[df.index >= pd.Timestamp(want_start)]
    df = df.copy()
    df.attrs = {
        "symbol": entry.get("symbol"),
        "source": entry.get("source", "yahoo"),
        "unadjusted": bool(entry.get("unadjusted", False)),
        "fetched_at": entry.get("fetched_at"),
        "stale": bool(stale),
    }
    return df


# ── Yahoo (yfinance) ─────────────────────────────────────────────────────
@contextlib.contextmanager
def _quiet_yfinance() -> Iterator[None]:
    """yfinance logs every failed ticker at ERROR; we report failures ourselves."""
    lg = logging.getLogger("yfinance")
    old = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(old)


def _is_rate_limit(msg: Any) -> bool:
    m = str(msg or "")
    return ("RateLimit" in m) or ("Too Many Requests" in m) or ("Rate limited" in m) or (" 429" in m)


def _extract(raw: Optional[pd.DataFrame], symbol: str, n_requested: int) -> Optional[pd.DataFrame]:
    """One ticker's sub-frame out of a ``yf.download`` result."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        for lvl in range(cols.nlevels):
            vals = cols.get_level_values(lvl)
            if symbol in set(vals):
                return raw.xs(symbol, axis=1, level=lvl)
        return None
    return raw if n_requested == 1 else None


def _normalize_yf(sub: Optional[pd.DataFrame], now: datetime) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if sub is None or sub.empty:
        return empty_frame(), []
    sub = sub.copy()
    sub.columns = [str(c).strip().lower() for c in sub.columns]
    splits_col = sub["stock splits"] if "stock splits" in sub.columns else None
    idx = pd.DatetimeIndex(pd.to_datetime(sub.index))
    if idx.tz is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    if splits_col is not None:
        splits_col = splits_col.copy()
        splits_col.index = idx.normalize()
    bars = clean_bars(sub, now)
    return bars, _splits_from_series(splits_col)


def _yf_download(symbols: List[str], start: Optional[date], end: Optional[date],
                 max_workers: int, now: datetime
                 ) -> Tuple[Dict[str, Tuple[pd.DataFrame, List[Dict[str, Any]]]], Dict[str, str]]:
    """Batch download. Returns ``(frames, errors)``; ``frames[sym] = (bars,
    splits)`` only for symbols with at least one bar."""
    import yfinance as yf
    try:
        from yfinance import shared as yf_shared
    except ImportError:  # pragma: no cover
        yf_shared = None
    kw: Dict[str, Any] = dict(
        interval="1d", group_by="ticker", auto_adjust=False, actions=True,
        threads=max(1, int(max_workers)), progress=False, repair=False,
        keepna=False, timeout=20, multi_level_index=True,
    )
    if start is None:
        kw["period"] = "max"
    else:
        kw["start"] = start.isoformat()
        kw["end"] = (end or (now.date() + timedelta(days=1))).isoformat()
    errors: Dict[str, str] = {}
    with _quiet_yfinance():
        try:
            raw = yf.download(symbols if len(symbols) > 1 else symbols[0], **kw)
        except Exception as e:  # yfinance can raise on a total failure
            return {}, {s: repr(e) for s in symbols}
    if yf_shared is not None:
        errors.update({str(k).upper(): str(v) for k, v in (getattr(yf_shared, "_ERRORS", {}) or {}).items()})
    frames = {}
    for s in symbols:
        bars, splits = _normalize_yf(_extract(raw, s, len(symbols)), now)
        if len(bars):
            frames[s] = (bars, splits)
            errors.pop(s, None)
        else:
            errors.setdefault(s, "no data")
    return frames, errors


def _yf_one(symbol: str, start: Optional[date], now: datetime
            ) -> Tuple[Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]], bool]:
    """Single-symbol history. Returns ``((bars, splits) | None, rate_limited)``."""
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    kw: Dict[str, Any] = dict(interval="1d", auto_adjust=False, actions=True, timeout=20)
    if start is None:
        kw["period"] = "max"
    else:
        kw["start"] = start.isoformat()
        kw["end"] = (now.date() + timedelta(days=1)).isoformat()
    with _quiet_yfinance():
        try:
            h = yf.Ticker(symbol).history(**kw)
        except YFRateLimitError:
            return None, True
        except Exception as e:
            log.debug("yfinance history %s failed: %r", symbol, e)
            return None, False
    bars, splits = _normalize_yf(h, now)
    return ((bars, splits) if len(bars) else None), False


def _yf_splits(symbol: str) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """All-time split list from Yahoo. ``(None, rate_limited)`` on failure."""
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    with _quiet_yfinance():
        try:
            s = yf.Ticker(symbol).splits
        except YFRateLimitError:
            return None, True
        except Exception as e:
            log.debug("yfinance splits %s failed: %r", symbol, e)
            return None, False
    if s is None:
        return None, False
    return _splits_from_series(s), False


# ── Nasdaq historical fallback ───────────────────────────────────────────
def _nasdaq_history_raw(symbol: str, start: Optional[date], end: date, now: datetime) -> pd.DataFrame:
    """Nasdaq daily bars (volume ``N/A`` → 0, i.e. a halt). Empty on failure."""
    frm = start or (end - timedelta(days=int(365.25 * 10)))
    params = {
        "assetclass": "etf" if symbol == BENCHMARK else "stocks",
        "fromdate": frm.isoformat(), "todate": end.isoformat(), "limit": "9999",
    }
    data = net.get_json(f"{NASDAQ_API}/{nasdaq_symbol(symbol)}/historical",
                        headers=net.NASDAQ_HEADERS, params=params, timeout=30.0)
    try:
        rows = data["data"]["tradesTable"]["rows"] or []
    except (TypeError, KeyError):
        return empty_frame()
    recs = []
    for r in rows:
        try:
            d = datetime.strptime(str(r.get("date", "")).strip(), "%m/%d/%Y")
        except ValueError:
            continue
        vol = num(r.get("volume"))
        recs.append({"date": d, "open": num(r.get("open")), "high": num(r.get("high")),
                     "low": num(r.get("low")), "close": num(r.get("close")),
                     "volume": vol if vol is not None else 0.0})
    if not recs:
        return empty_frame()
    return clean_bars(pd.DataFrame(recs).set_index("date"), now)


def adjust_for_splits(df: pd.DataFrame, splits: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    """Split-adjust a frame that *may* be raw, one split at a time.

    Nasdaq's historical endpoint has been observed both ways, so each split
    inside the frame is tested rather than assumed: across the split date the
    open/previous-close jump of raw data sits near ``1/ratio`` (≈20× for a
    1:20 reverse split); adjusted data sits near 1. Only splits whose jump is
    closer (in log space) to ``1/ratio`` than to 1 are applied — bars before
    the split get prices ÷ ratio and volume × ratio. Returns the frame and
    the dates of the splits that were applied.
    """
    if df is None or df.empty or not splits:
        return df, []
    out = df.copy()
    applied = []
    for sp in sorted(splits, key=lambda x: x["date"]):
        r = float(sp.get("ratio") or 0)
        if r <= 0 or r == 1:
            continue
        sd = pd.Timestamp(sp["date"])
        before, after = out[out.index < sd], out[out.index >= sd]
        if before.empty or after.empty:
            continue
        jump = after["open"].iloc[0] / before["close"].iloc[-1]
        if not (jump > 0 and math.isfinite(jump)):
            continue
        if abs(math.log(jump) - math.log(1.0 / r)) < abs(math.log(jump)):
            m = out.index < sd
            out.loc[m, ["open", "high", "low", "close"]] = out.loc[m, ["open", "high", "low", "close"]] / r
            out.loc[m, "volume"] = out.loc[m, "volume"] * r
            applied.append(str(sp["date"]))
    return out, applied


# ── load_history ─────────────────────────────────────────────────────────
class _Run:
    """Bookkeeping for one ``load_history`` call."""

    def __init__(self, n: int) -> None:
        self.t0 = time.monotonic()
        self.n = n
        self.counts: Dict[str, int] = {k: 0 for k in (
            "current", "cache_only", "incremental", "full", "split_refetch",
            "drift_refetch", "individual", "nasdaq", "stale", "failed")}
        self.yahoo_tripped = False
        self.nasdaq_used = 0
        self.failed: List[str] = []

    def bump(self, k: str, by: int = 1) -> None:
        self.counts[k] = self.counts.get(k, 0) + by

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def _download_with_backoff(symbols: List[str], start: Optional[date], run: _Run,
                           max_workers: int, now: datetime
                           ) -> Tuple[Dict[str, Tuple[pd.DataFrame, List[Dict[str, Any]]]], List[str]]:
    """``_yf_download`` with hard back-off on rate limits. Rate-limited
    symbols are re-tried after each back-off; if Yahoo is still refusing,
    the run stops calling Yahoo altogether (``run.yahoo_tripped``)."""
    if run.yahoo_tripped:
        return {}, list(symbols)
    frames, errors = _yf_download(symbols, start, None, max_workers, now)
    limited = [s for s, e in errors.items() if _is_rate_limit(e)]
    if 0 < len(limited) <= STRAGGLERS and len(symbols) >= 20:
        # One or two refusals is usually a bad/odd ticker, not throttling —
        # don't stall the whole run for minutes over it.
        log.info("prices: %d straggler(s) refused by Yahoo (%s) — skipping without back-off",
                 len(limited), ", ".join(limited))
        limited = []
    for wait in RATE_LIMIT_BACKOFF_S:
        if not limited:
            break
        log.warning("prices: Yahoo rate-limited %d symbols; backing off %.0fs", len(limited), wait)
        _sleep(wait)
        more, errs2 = _yf_download(limited, start, None, max_workers, now)
        frames.update(more)
        for s in more:
            errors.pop(s, None)
        for s, e in errs2.items():
            errors[s] = e
        limited = [s for s in limited if s in errs2 and _is_rate_limit(errs2[s])]
    if limited:
        log.error("prices: Yahoo still rate-limiting after back-off; no more Yahoo calls this run")
        run.yahoo_tripped = True
    failed = [s for s in symbols if s not in frames]
    return frames, failed


def _retry_individually(symbols: List[str], start_of: Dict[str, Optional[date]], run: _Run,
                        now: datetime) -> Dict[str, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    got = {}
    for s in symbols:
        if run.yahoo_tripped:
            break
        res, limited = _yf_one(s, start_of.get(s), now)
        if limited:
            log.warning("prices: Yahoo rate-limited on single retry (%s); backing off %.0fs",
                        s, RATE_LIMIT_BACKOFF_S[0])
            _sleep(RATE_LIMIT_BACKOFF_S[0])
            res, limited = _yf_one(s, start_of.get(s), now)
            if limited:
                run.yahoo_tripped = True
                break
        if res is not None:
            got[s] = res
            run.bump("individual")
        _sleep(INDIVIDUAL_PAUSE_S)
    return got


def _known_splits(symbol: str, entry: Optional[Dict[str, Any]], hint: Optional[List[Dict[str, Any]]],
                  run: _Run) -> Optional[List[Dict[str, Any]]]:
    """Best split list we have without hammering Yahoo: cache entry (+ any
    split just seen in an incremental window), the 20 h splits cache, or one
    ``Ticker.splits`` call. ``None`` = unknown."""
    if entry is not None and entry.get("splits") is not None:
        return _merge_splits(entry["splits"], hint)
    hit = net.cache_get("splits", symbol, SPLITS_CACHE_S)
    if hit is not None:
        return _merge_splits(hit, hint)
    if run.yahoo_tripped:
        return _merge_splits(hint) if hint else None
    sp, limited = _yf_splits(symbol)
    if limited:
        run.yahoo_tripped = True
    if sp is not None:
        net.cache_set("splits", symbol, sp)
        return _merge_splits(sp, hint)
    return _merge_splits(hint) if hint else None


def _nasdaq_fallback(symbol: str, want_start: Optional[date], entry: Optional[Dict[str, Any]],
                     hint: Optional[List[Dict[str, Any]]], run: _Run, now: datetime
                     ) -> Optional[Dict[str, Any]]:
    """Full-period Nasdaq history, checked against known splits (see
    ``adjust_for_splits``). New cache entry, or ``None``. When the split
    list is unknown the entry is flagged ``unadjusted`` and will be
    re-fetched from Yahoo on the next refresh."""
    if run.nasdaq_used >= NASDAQ_FALLBACK_MAX:
        return None
    run.nasdaq_used += 1
    bars = _nasdaq_history_raw(symbol, want_start, now.date(), now)
    if bars.empty:
        return None
    splits = _known_splits(symbol, entry, hint, run)
    unadjusted = splits is None
    if splits:
        bars, applied = adjust_for_splits(bars, splits)
        if applied:
            log.info("prices: %s Nasdaq bars were raw across split(s) %s — adjusted", symbol, applied)
    run.bump("nasdaq")
    return {
        "df": bars, "splits": splits, "source": "nasdaq", "unadjusted": unadjusted,
        "coverage_start": want_start.isoformat() if want_start else None,
        "fetched_at": _utc_iso(), "full_fetched_at": _utc_iso(),
    }


def _drift(cached: pd.DataFrame, fresh: pd.DataFrame) -> Optional[float]:
    """Median |fresh/cached − 1| of closes on overlapping dates (None if no overlap)."""
    common = cached.index.intersection(fresh.index)
    if len(common) == 0:
        return None
    ratio = fresh.loc[common, "close"] / cached.loc[common, "close"]
    return float((ratio - 1.0).abs().median())


def _incremental_start(entry: Dict[str, Any], today: date) -> date:
    df = entry["df"]
    last = df.index.max().date() if len(df) else today
    return min(_sessions_back(last, OVERLAP_SESSIONS), _sessions_back(today, INCREMENTAL_SESSIONS))


def _plan(symbols: List[str], want_start: Optional[date], refresh: bool, now: datetime,
          run: _Run, out: Dict[str, pd.DataFrame]
          ) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Dict[str, Any]]]:
    """Split symbols into served-from-cache (written to ``out``), incremental
    and full-refetch groups. Returns ``(incremental, full, entries)``."""
    today = now.date()
    incr: Dict[str, Dict[str, Any]] = {}
    full: List[str] = []
    entries: Dict[str, Dict[str, Any]] = {}
    for s in symbols:
        e = read_cache(s)
        if e is not None:
            entries[s] = e
        if e is None or not _covers(e, want_start) or len(e["df"]) == 0:
            full.append(s)
            continue
        if _is_current(e, now):
            out[s] = _frame_out(e, want_start)
            run.bump("current")
            continue
        if not refresh:                       # caller asked for cache only
            out[s] = _frame_out(e, want_start, stale=True)
            run.bump("cache_only")
            continue
        last = e["df"].index.max().date()
        if (e.get("source") != "yahoo" or e.get("unadjusted")
                or (today - last).days > MAX_INCREMENTAL_GAP_DAYS):
            full.append(s)
        else:
            incr[s] = e
    return incr, full, entries


def load_history(symbols: List[str], period: str = config.HISTORY_PERIOD,
                 refresh: bool = True, max_workers: int = 4, *,
                 chunk_size: int = CHUNK_SIZE, pause_s: float = CHUNK_PAUSE_S,
                 report: bool = True) -> Dict[str, pd.DataFrame]:
    """Split-adjusted daily OHLCV for ``symbols`` → ``{symbol: frame}``.

    Frames: tz-naive ascending unique ``DatetimeIndex``; float columns
    ``open high low close volume``; trimmed to ``period``. Symbols with no
    data from any source are omitted (never filled in).

    ``refresh=False`` serves any cached symbol as-is and only fetches
    symbols that have no usable cache. ``refresh=True`` (default) brings
    every symbol up to the last completed session — incrementally when the
    cache allows it (see module docstring). ``max_workers`` caps yfinance's
    download threads. ``chunk_size``/``pause_s`` control batch pacing;
    ``report=False`` skips ``record_status`` (used for the benchmark).
    Progress is logged at INFO once per chunk; the call's stats land in
    ``LAST_RUN``.
    """
    syms = _canon_list(symbols)
    now = _now()
    want_start = period_start(period, now.date())
    run = _Run(len(syms))
    out: Dict[str, pd.DataFrame] = {}
    incr, full, entries = _plan(syms, want_start, refresh, now, run, out)
    log.info("prices: %d symbols — %d current in cache, %d incremental, %d full",
             len(syms), run.counts["current"], len(incr), len(full))

    # 1) Incremental refresh of healthy caches (batched). Anything that
    #    fails, shows a new split, or disagrees with the cache goes to the
    #    full-period path below — which is also the individual retry.
    retry_full: List[str] = []
    split_hints: Dict[str, List[Dict[str, Any]]] = {}
    order = sorted(incr, key=lambda s: _incremental_start(incr[s], now.date()))
    chunks = list(_chunks(order, chunk_size))
    for i, chunk in enumerate(chunks, 1):
        start = min(_incremental_start(incr[s], now.date()) for s in chunk)
        frames, failed = _download_with_backoff(chunk, start, run, max_workers, now)
        for s, (bars, splits) in frames.items():
            verdict = _apply_incremental(s, incr[s], bars, splits, want_start, now, out)
            if verdict == "ok":
                run.bump("incremental")
                continue
            run.bump(verdict)
            retry_full.append(s)
            if verdict == "split_refetch":
                split_hints[s] = splits
        retry_full.extend(failed)
        log.info("prices[incremental] chunk %d/%d (%d syms): %d fetched, %d failed — %.0fs",
                 i, len(chunks), len(chunk), len(frames), len(failed), run.elapsed())
        if i < len(chunks):
            _sleep(pause_s)

    # 2) Full-period fetches: batched, then each failure retried on its own.
    full_list = full + [s for s in retry_full if s not in out]
    failed_full: List[str] = []
    chunks = list(_chunks(full_list, chunk_size))
    for i, chunk in enumerate(chunks, 1):
        frames, failed = _download_with_backoff(chunk, want_start, run, max_workers, now)
        for s, (bars, splits) in frames.items():
            _store_full(s, bars, splits, want_start, out)
            run.bump("full")
        failed_full.extend(failed)
        log.info("prices[full] chunk %d/%d (%d syms): %d fetched, %d failed — %.0fs",
                 i, len(chunks), len(chunk), len(frames), len(failed), run.elapsed())
        if i < len(chunks):
            _sleep(pause_s)
    if failed_full:
        got = _retry_individually(failed_full, {s: want_start for s in failed_full}, run, now)
        for s, (bars, splits) in got.items():
            _store_full(s, bars, splits, want_start, out)
            run.bump("full")

    # 3) Still missing. If Yahoo is rate-limiting us, a cached frame is served
    #    as stale (converting healthy caches to Nasdaq data would only force
    #    full refetches tomorrow). Otherwise the problem is symbol-specific:
    #    Nasdaq historical, then the stale cache, then give up.
    for s in [s for s in failed_full if s not in out]:
        e = entries.get(s)
        has_cache = e is not None and len(e["df"]) > 0
        entry = None
        if not (run.yahoo_tripped and has_cache):
            entry = _nasdaq_fallback(s, want_start, e, split_hints.get(s), run, now)
        if entry is not None:
            _write_cache(s, entry)
            out[s] = _frame_out(dict(entry, symbol=s), want_start)
        elif has_cache:
            out[s] = _frame_out(e, want_start, stale=True)
            run.bump("stale")
        else:
            run.failed.append(s)
            run.bump("failed")

    stats = {
        "requested": len(syms), "returned": len(out), "elapsed_s": round(run.elapsed(), 1),
        "yahoo_rate_limited": run.yahoo_tripped, "failed_symbols": run.failed[:50], **run.counts,
    }
    log.info("prices: done — %s", stats)
    if report:
        LAST_RUN.clear()
        LAST_RUN.update(stats)
        _report(run, len(syms), len(out))
    return out


def _apply_incremental(symbol: str, entry: Dict[str, Any], bars: pd.DataFrame,
                       splits: List[Dict[str, Any]], want_start: Optional[date],
                       now: datetime, out: Dict[str, pd.DataFrame]) -> str:
    """Merge a recent window into a cached entry, or say why it can't be:
    ``"ok"`` | ``"split_refetch"`` (a split the cache doesn't know) |
    ``"drift_refetch"`` (overlapping closes disagree / no overlap)."""
    known = {sp["date"] for sp in (entry.get("splits") or [])}
    new_splits = [sp for sp in splits if sp["date"] not in known]
    if new_splits:
        log.info("prices: %s new split(s) %s → full refetch", symbol, new_splits)
        return "split_refetch"
    drift = _drift(entry["df"], bars)
    if drift is None or drift > DRIFT_TOLERANCE:
        log.info("prices: %s cached bars disagree with Yahoo (drift=%s) → full refetch", symbol,
                 "no overlap" if drift is None else f"{drift:.3f}")
        return "drift_refetch"
    merged = clean_bars(pd.concat([entry["df"], bars]), now)
    entry.update(df=merged, splits=_merge_splits(entry.get("splits"), splits), fetched_at=_utc_iso())
    _write_cache(symbol, entry)
    out[symbol] = _frame_out(entry, want_start)
    return "ok"


def _store_full(symbol: str, bars: pd.DataFrame, splits: List[Dict[str, Any]],
                want_start: Optional[date], out: Dict[str, pd.DataFrame]) -> None:
    entry = {
        "df": bars, "splits": splits, "source": "yahoo", "unadjusted": False,
        "coverage_start": want_start.isoformat() if want_start else None,
        "fetched_at": _utc_iso(), "full_fetched_at": _utc_iso(),
    }
    _write_cache(symbol, entry)
    out[symbol] = _frame_out(dict(entry, symbol=symbol), want_start)


def _report(run: _Run, n_req: int, n_out: int) -> None:
    c = run.counts
    fetched_yahoo = c["incremental"] + c["full"]
    detail = (f"{n_out:,}/{n_req:,} symbols: {c['current']:,} current in cache, "
              f"{c['cache_only']:,} cache-only, "
              f"{c['incremental']:,} incremental, {c['full']:,} full "
              f"({c['split_refetch']} split / {c['drift_refetch']} drift refetches), "
              f"{c['nasdaq']} via Nasdaq fallback, {c['stale']} stale, {c['failed']} failed")
    if run.yahoo_tripped:
        detail = "Yahoo rate-limited — " + detail
    attempted = n_req - c["current"] - c["cache_only"]
    ok = (not run.yahoo_tripped) and (attempted == 0 or fetched_yahoo >= 0.75 * attempted)
    net.record_status(YAHOO_STATUS, ok, detail)


# ── Splits ───────────────────────────────────────────────────────────────
def load_splits(symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Split events per symbol, ``[{"date": "YYYY-MM-DD", "ratio": 0.05}]``
    ascending (ratio < 1 = reverse split; 1:20 → 0.05).

    Served from the price cache when the symbol has one (splits inside the
    cached history window, kept current by ``load_history``); otherwise
    fetched from Yahoo (all-time) and cached for 20 h. A symbol whose splits
    could not be determined is **omitted** — an empty list means "no splits",
    absence means "unknown".
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    run = _Run(0)
    need = []
    for s in _canon_list(symbols):
        e = read_cache(s)
        if e is not None and e.get("splits") is not None:
            out[s] = list(e["splits"])
            continue
        hit = net.cache_get("splits", s, SPLITS_CACHE_S)
        if hit is not None:
            out[s] = hit
        else:
            need.append(s)
    for s in need:
        if run.yahoo_tripped:
            break
        sp, limited = _yf_splits(s)
        if limited:
            log.warning("prices: Yahoo rate-limited on splits; backing off %.0fs", RATE_LIMIT_BACKOFF_S[0])
            _sleep(RATE_LIMIT_BACKOFF_S[0])
            sp, limited = _yf_splits(s)
            run.yahoo_tripped = limited
        if sp is not None:
            net.cache_set("splits", s, sp)
            out[s] = sp
        _sleep(INDIVIDUAL_PAUSE_S)
    return out


# ── Benchmark ────────────────────────────────────────────────────────────
def benchmark_history(period: str = config.HISTORY_PERIOD) -> pd.DataFrame:
    """IWM (Russell 2000 ETF) daily bars, same shape and cache as
    ``load_history``. Empty frame if unavailable."""
    got = load_history([BENCHMARK], period=period, refresh=True, max_workers=1, report=False)
    df = got.get(BENCHMARK)
    if df is None:
        df = empty_frame()
    attrs = dict(df.attrs)
    # Yahoo sometimes publishes the latest IWM bar without a close (it is then
    # dropped). Market features for that session would go blank in live
    # scoring, so fill any missing completed session from Nasdaq.
    now = now_et()
    last_done = now.date() if (is_trading_day(now.date()) and now.time() >= dtime(16, 15)) else prev_trading_day(now.date())
    have = df.index.max().date() if len(df) else None
    if have is None or have < last_done:
        start = (have + timedelta(days=1)) if have else last_done - timedelta(days=10)
        extra = _nasdaq_history_raw(BENCHMARK, start, now.date(), now)
        if len(extra):
            extra = extra[extra.index.date <= last_done]
            df = pd.concat([df, extra[~extra.index.isin(df.index)]]).sort_index()
            log.info("benchmark: filled %d missing IWM bar(s) from Nasdaq (Yahoo had none through %s)", len(extra), have)
    have = df.index.max().date() if len(df) else None
    if have is None or have < last_done:
        bar = _benchmark_close_bar(last_done)
        if bar is not None:
            df = pd.concat([df, bar]).sort_index()
            log.info("benchmark: IWM %s bar built from Nasdaq's official close (history not published yet)", last_done)
    df.attrs = attrs
    return df


def _benchmark_close_bar(day: date) -> Optional[pd.DataFrame]:
    """One IWM bar for ``day`` from Nasdaq's quote (official 4 PM close). Only
    the close feeds GRAVITY's market features; open/high/low are set to it."""
    d = net.get_json(f"{NASDAQ_API}/{BENCHMARK}/info", headers=net.NASDAQ_HEADERS, params={"assetclass": "etf"})
    try:
        sec = d["data"]["secondaryData"] or {}
        prim = d["data"]["primaryData"] or {}
    except (TypeError, KeyError):
        return None
    ts = parse_nasdaq_timestamp(sec.get("lastTradeTimestamp"))
    close = num(sec.get("lastSalePrice"))
    if ts is None or close is None or ts.date() != day:
        return None
    vol = num(prim.get("volume"))
    return pd.DataFrame({"open": [close], "high": [close], "low": [close], "close": [close],
                         "volume": [vol if vol is not None else float("nan")]},
                        index=pd.DatetimeIndex([pd.Timestamp(day)]))


# ── Pre-market snapshot ──────────────────────────────────────────────────
_TS_RE = re.compile(
    r"([A-Z][a-z]{2})\w*\.?\s+(\d{1,2}),\s+(\d{4})(?:\s+(\d{1,2}):(\d{2})\s*([AP]M))?")


def parse_nasdaq_timestamp(s: Any) -> Optional[datetime]:
    """'Sep 29, 2026 6:57 PM ET' / 'Closed at Sep 29, 2026 4:00 PM ET' → aware ET datetime."""
    if not s:
        return None
    m = _TS_RE.search(str(s))
    if not m:
        return None
    mon, day, year, hh, mm, ap = m.groups()
    try:
        d = datetime.strptime(f"{mon} {day} {year}", "%b %d %Y")
    except ValueError:
        return None
    if hh is None:
        return d.replace(tzinfo=ET)
    h = int(hh) % 12 + (12 if ap == "PM" else 0)
    return d.replace(hour=h, minute=int(mm), tzinfo=ET)


def _ext_window(now: datetime) -> Tuple[date, date, datetime, datetime]:
    """(target session, previous session, window start, window end) for the
    extended-hours period that leads into the next regular open."""
    tgt = target_session(now)
    prev = prev_trading_day(tgt)
    return (tgt, prev, datetime.combine(prev, dtime(16, 0), tzinfo=ET),
            datetime.combine(tgt, dtime(9, 30), tzinfo=ET))


def _session_label(ts: datetime, tgt: date) -> str:
    return "pre-market" if ts.astimezone(ET).date() == tgt else "after-hours"


def _cached_close(symbol: str, day: date) -> Optional[float]:
    e = read_cache(symbol)
    if e is None or not len(e["df"]):
        return None
    ts = pd.Timestamp(day)
    if ts in e["df"].index:
        v = float(e["df"].loc[ts, "close"])
        return v if v > 0 else None
    return None


def parse_nasdaq_quote(symbol: str, payload: Any, now: datetime) -> Tuple[Optional[Dict[str, Any]], str]:
    """Nasdaq ``/info`` payload → snapshot dict, or ``(None, reason)``.

    Observed payload semantics (2026-09): outside regular hours
    ``primaryData`` is the extended-hours quote — ``lastSalePrice`` the last
    extended-hours trade (or the close when there was none), ``netChange``
    measured against the regular close (empty + ``deltaIndicator: "unch"``
    when unchanged), ``lastTradeTimestamp`` the quote's *as-of* time rather
    than a trade time, ``volume`` the day's cumulative volume — and
    ``secondaryData`` carries the regular close ("Closed at … 4:00 PM ET").

    ``prev_close`` is the previous regular close: Nasdaq's "Closed at" block
    when it is for the previous session, else the cached daily bar, else
    ``price − netChange``. A symbol counts as having an extended-hours trade
    when its price differs from that close, or (pre-market only, where the
    volume is pre-market volume) when volume > 0. Regular-session quotes and
    quotes older than the previous close are rejected.
    """
    try:
        d = payload["data"]
        pdata = d["primaryData"]
    except (TypeError, KeyError):
        return None, "bad payload"
    if not isinstance(pdata, dict):
        return None, "bad payload"
    status = str(d.get("marketStatus") or "")
    if "open" in status.lower():
        return None, "regular session"
    price = num(pdata.get("lastSalePrice"))
    ts = parse_nasdaq_timestamp(pdata.get("lastTradeTimestamp"))
    if price is None or price <= 0 or ts is None:
        return None, "no quote"
    tgt, prev, w0, w1 = _ext_window(now)
    if not (w0 < ts < w1):
        return None, "quote outside the extended-hours window"
    if "pre" in status.lower():
        session = "pre-market"
    elif "after" in status.lower():
        session = "after-hours"
    else:
        session = _session_label(ts, tgt)

    prev_close = None
    sd = d.get("secondaryData") or {}
    if isinstance(sd, dict):
        sts_raw = str(sd.get("lastTradeTimestamp") or "")
        sts = parse_nasdaq_timestamp(sts_raw)
        if sts is not None and "close" in sts_raw.lower() and sts.date() == prev:
            prev_close = num(sd.get("lastSalePrice"))
    if prev_close is None:
        prev_close = _cached_close(symbol, prev)
    if prev_close is None:
        chg = num(pdata.get("netChange"))
        if chg is None and str(pdata.get("deltaIndicator", "")).lower() == "unch":
            chg = 0.0
        if chg is not None and price - chg > 0:
            prev_close = price - chg
    if prev_close is not None and prev_close <= 0:
        prev_close = None

    # Cumulative volume is pre-market-only before the open; in the evening
    # it is regular + after-hours, so it is not reported for that session.
    vol = num(pdata.get("volume")) if session == "pre-market" else None
    moved = prev_close is not None and abs(price - prev_close) > 1e-9
    if not (moved or (vol is not None and vol > 0)):
        return None, "no extended-hours trade"
    return {
        "price": float(price),
        "prev_close": float(prev_close) if prev_close is not None else None,
        "gap_pct": (float(price) / prev_close - 1.0) if prev_close else None,
        "volume": float(vol) if vol is not None else None,
        "asof": ts.isoformat(),
        "source": "nasdaq",
        "session": session,
        "realtime": bool(pdata.get("isRealTime")),
        "market_status": status or None,
    }, "ok"


def _nasdaq_quote(symbol: str) -> Any:
    return net.get_json(f"{NASDAQ_API}/{nasdaq_symbol(symbol)}/info",
                        headers=net.NASDAQ_HEADERS, params={"assetclass": "stocks"}, timeout=20.0)


def _yahoo_premarket(symbol: str, now: datetime) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Fallback from yfinance 1-minute bars with ``prepost=True``.
    Returns ``(snapshot | None, rate_limited)``."""
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    tgt, prev, w0, w1 = _ext_window(now)
    with _quiet_yfinance():
        try:
            h = yf.Ticker(symbol).history(period="5d", interval="1m", prepost=True, auto_adjust=False)
        except YFRateLimitError:
            return None, True
        except Exception:
            return None, False
    if h is None or h.empty:
        return None, False
    idx = pd.DatetimeIndex(h.index)
    idx = idx.tz_localize(ET) if idx.tz is None else idx.tz_convert(ET)
    h = h.set_axis(idx)
    h.columns = [str(c).lower() for c in h.columns]
    win = h[(h.index >= pd.Timestamp(w0)) & (h.index < pd.Timestamp(w1)) & (h["close"] > 0)]
    if win.empty:
        return None, False
    last_ts = win.index[-1].to_pydatetime()
    price = float(win["close"].iloc[-1])
    session = _session_label(last_ts, tgt)
    vol = None
    if session == "pre-market":
        v = float(win[win.index >= pd.Timestamp(datetime.combine(tgt, dtime(4, 0), tzinfo=ET))]["volume"].sum())
        vol = v if v > 0 else None     # Yahoo often reports 0 for extended hours = unknown
    prev_close = _cached_close(symbol, prev)
    if prev_close is None:
        with _quiet_yfinance():
            try:
                dly = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
            except YFRateLimitError:
                dly = None
            except Exception:
                dly = None
        if dly is not None and not dly.empty:
            dly = clean_bars(dly, now)
            ts = pd.Timestamp(prev)
            if ts in dly.index:
                prev_close = float(dly.loc[ts, "close"])
    return {
        "price": price,
        "prev_close": prev_close,
        "gap_pct": (price / prev_close - 1.0) if prev_close else None,
        "volume": vol,
        "asof": last_ts.isoformat(),
        "source": "yahoo",
        "session": session,
        "realtime": False,
        "market_status": None,
    }, False


def premarket_snapshot(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Latest extended-hours price for each symbol (shortlist only, ≤ ~150).

    → ``{symbol: {"price", "prev_close", "gap_pct" (fraction vs the previous
    regular close), "volume" (pre-market volume; ``None`` when unknown or in
    the evening), "asof" (ISO, ET), "source": "nasdaq"|"yahoo", "session":
    "pre-market"|"after-hours", "realtime", "market_status"}}``.

    Symbols with no extended-hours trade since the previous close are
    omitted, as is everything while the regular session is open. Nasdaq
    ``/info`` first (see ``parse_nasdaq_quote``); yfinance 1-minute
    ``prepost`` bars only for symbols whose Nasdaq request failed outright.
    ``prev_close``/``gap_pct`` are ``None`` when the previous close cannot be
    established — never guessed.
    """
    syms = _canon_list(symbols)
    now = _now()
    out: Dict[str, Dict[str, Any]] = {}
    if not syms:
        net.record_status(NASDAQ_STATUS, True, "no symbols requested")
        return out
    with ThreadPoolExecutor(max_workers=PREMARKET_WORKERS) as ex:
        payloads = list(ex.map(lambda s: (s, _nasdaq_quote(s)), syms))
    failed, n_ok, statuses = [], 0, set()
    for s, payload in payloads:
        if payload is None or not isinstance(payload, dict) or not payload.get("data"):
            failed.append(s)
            continue
        n_ok += 1
        statuses.add(str((payload.get("data") or {}).get("marketStatus")))
        snap, _why = parse_nasdaq_quote(s, payload, now)
        if snap is not None:
            out[s] = snap
    n_nasdaq = len(out)
    n_yahoo = 0
    for s in failed:
        snap, limited = _yahoo_premarket(s, now)
        if limited:
            log.warning("premarket: Yahoo rate-limited; skipping remaining fallbacks")
            break
        if snap is not None:
            out[s] = snap
            n_yahoo += 1
        _sleep(INDIVIDUAL_PAUSE_S)
    detail = (f"{n_ok}/{len(syms)} quotes; {n_nasdaq} with extended-hours trades"
              f"{f', {n_yahoo} via Yahoo fallback' if n_yahoo else ''}; "
              f"market status {', '.join(sorted(statuses)) or 'unknown'}")
    net.record_status(NASDAQ_STATUS, n_ok >= max(1, len(syms) // 2), detail)
    return out
