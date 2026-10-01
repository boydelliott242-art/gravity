"""FINRA Reg SHO daily short-volume **history** (CONTRACTS §14.2).

``shortside.finra_short_volume`` serves the last few sessions for the live
page; this module keeps the long history the model trains on:

    load(start, end) -> DataFrame[date, symbol, short_volume, total_volume]

Source: ``https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt``
(one consolidated file per session, published after that session's close,
~330 KB). ``total_volume`` is FINRA-reported (TRF/ADF/ORF, i.e. off-exchange)
volume only — not consolidated tape volume.

Caching: each session's *parsed* file is kept forever under
``config.CACHE / "finra_hist"`` as ``YYYYMMDD.pkl.gz`` (FINRA never revises a
published file; parquet is not available in this runtime). Sessions that
answer 403/404 once they are safely in the past (holidays FINRA skipped, days
before the archive) get a ``YYYYMMDD.none`` marker so a re-run doesn't ask
again. Network errors are *not* marked — the next run retries them — so an
interrupted backfill simply resumes. Requests are throttled to ≤ 3 per second.

Nothing here raises on network failure.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .. import config, net
from ..util import is_trading_day, now_et
from .shortside import FINRA_URL, _finra_cache_path, _finra_complete, finra_symbol

log = logging.getLogger(__name__)

COLUMNS = ["date", "symbol", "short_volume", "total_volume"]
RPS = 3.0                       # politeness: ≤ 3 requests per second
MISSING_AFTER_DAYS = 10         # a 403/404 this old is final (holiday / not archived)
SOURCE_URL = "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files"

DateLike = Union[str, date, datetime, pd.Timestamp]


def cache_dir() -> Path:
    p = config.CACHE / "finra_hist"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _day(x: DateLike) -> date:
    return pd.Timestamp(x).date()


def _paths(d: date) -> Tuple[Path, Path]:
    base = cache_dir() / f"{d:%Y%m%d}"
    return base.with_suffix(".pkl.gz"), base.with_suffix(".none")


def candidate_days(start: DateLike, end: DateLike) -> List[date]:
    """Weekdays in [start, end] that are not known NYSE holidays.

    ``util`` only lists holidays from 2024 on; earlier holidays (e.g.
    2023-12-25) are probed once, answer 403, and get a ``.none`` marker."""
    s, e = _day(start), _day(end)
    out: List[date] = []
    d = s
    while d <= e:
        if d.weekday() < 5 and (d.year < 2024 or is_trading_day(d)):
            out.append(d)
        d += timedelta(days=1)
    return out


# ── Parsing ──────────────────────────────────────────────────────────────
def parse(text: str) -> pd.DataFrame:
    """One CNMSshvol file → DataFrame[date (datetime64), symbol (canonical),
    short_volume, total_volume]. Rows with unparseable numbers are dropped;
    duplicate symbols (two raw spellings mapping to one canonical) are summed."""
    if not text or not text.startswith("Date|"):
        return empty()
    try:
        df = pd.read_csv(io.StringIO(text), sep="|", dtype={"Date": str, "Symbol": str},
                         keep_default_na=False)
    except (ValueError, pd.errors.ParserError):
        return empty()
    need = {"Date", "Symbol", "ShortVolume", "TotalVolume"}
    if not need.issubset(df.columns):
        return empty()
    df = df[df["Date"].str.fullmatch(r"\d{8}") & (df["Symbol"].str.strip() != "")]
    if df.empty:
        return empty()
    sv = pd.to_numeric(df["ShortVolume"], errors="coerce")
    tv = pd.to_numeric(df["TotalVolume"], errors="coerce")
    uniq = {s: finra_symbol(s) for s in df["Symbol"].unique()}
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], format="%Y%m%d"),
        "symbol": df["Symbol"].map(uniq),
        "short_volume": sv.astype(float),
        "total_volume": tv.astype(float),
    })
    ok = out["short_volume"].notna() & out["total_volume"].notna() & (out["total_volume"] >= 0) \
        & (out["short_volume"] >= 0)
    out = out[ok]
    if out.duplicated(["date", "symbol"]).any():
        out = out.groupby(["date", "symbol"], as_index=False, sort=False)[["short_volume", "total_volume"]].sum()
    return out[COLUMNS].reset_index(drop=True)


def empty() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "symbol": pd.Series(dtype=object),
        "short_volume": pd.Series(dtype=float),
        "total_volume": pd.Series(dtype=float),
    })


# ── Network ──────────────────────────────────────────────────────────────
def _http_get(url: str) -> Tuple[Optional[int], Optional[str]]:
    """(status, text). status None = network error (retry later)."""
    host = url.split("/")[2]
    net._throttle(host, RPS)
    try:
        r = net._session().get(url, headers={"User-Agent": net.BROWSER_UA}, timeout=30)
    except Exception as e:  # requests.RequestException and friends
        log.debug("finra_hist GET %s failed: %s", url, e)
        return None, None
    return r.status_code, (r.text if 200 <= r.status_code < 300 else None)


def _raw_from_shortside_cache(d: date) -> Optional[str]:
    """Reuse a day already downloaded by ``shortside`` (same file, gzipped)."""
    p = _finra_cache_path(d)
    if not p.exists():
        return None
    try:
        return gzip.decompress(p.read_bytes()).decode("utf-8")
    except (OSError, ValueError, EOFError):
        return None


def _write(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=5) as fh:
        pickle.dump(df, fh, protocol=4)
    tmp.replace(path)


def _read(path: Path) -> Optional[pd.DataFrame]:
    try:
        with gzip.open(path, "rb") as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame) and set(COLUMNS).issubset(df.columns):
            return df[COLUMNS]
    except Exception as e:  # corrupt / truncated cache file → refetch
        log.debug("finra_hist: bad cache %s (%s)", path, e)
    try:
        path.unlink()
    except OSError:
        pass
    return None


def fetch_day(d: DateLike, today: Optional[date] = None, offline: bool = False) -> Optional[pd.DataFrame]:
    """One session: cache → shortside's raw cache → CDN. ``None`` when the
    file does not exist (holiday / not yet published) or is unreachable."""
    d = _day(d)
    pkl, none = _paths(d)
    if pkl.exists():
        df = _read(pkl)
        if df is not None:
            return df
    if none.exists():
        return None
    text = _raw_from_shortside_cache(d)
    if text is None:
        if offline:
            return None
        status, text = _http_get(FINRA_URL.format(ymd=f"{d:%Y%m%d}"))
        if status in (403, 404):
            today = today or now_et().date()
            if (today - d).days >= MISSING_AFTER_DAYS:
                none.touch()
            return None
        if status is None or text is None:
            return None
    if not text.startswith("Date|"):
        return None
    df = parse(text)
    if _finra_complete(text) and len(df):
        _write(pkl, df)           # only complete files are cached forever
    return df


# ── Public API ───────────────────────────────────────────────────────────
def load(start: DateLike, end: DateLike, symbols: Optional[Iterable[str]] = None,
         offline: bool = False, progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """FINRA daily short volume for every published session in [start, end].

    Returns DataFrame[date (datetime64, the session the volume traded on),
    symbol (canonical), short_volume, total_volume], sorted by date, symbol.
    ``symbols`` limits rows to that set (saves memory on a 3-year load).
    ``offline=True`` reads the cache only (no network). Days that are missing
    or unreachable are simply absent — callers treat them as unknown (NaN).
    """
    keep = None if symbols is None else set(symbols)
    days = candidate_days(start, end)
    frames: List[pd.DataFrame] = []
    got = miss = 0
    for i, d in enumerate(days):
        df = fetch_day(d, offline=offline)
        if df is None or not len(df):
            miss += 1
        else:
            got += 1
            if keep is not None:
                df = df[df["symbol"].isin(keep)]
            frames.append(df)
        if progress is not None:
            progress(i + 1, len(days))
    if not offline:
        net.record_status("FINRA short-volume history", got > 0,
                          f"{got} sessions {days[0] if days else '-'}→{days[-1] if days else '-'}; {miss} missing")
    if not frames:
        return empty()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def cached_range() -> Tuple[Optional[str], Optional[str], int]:
    """(first, last, n) sessions present in the forever cache."""
    names = sorted(p.name[:8] for p in cache_dir().glob("*.pkl.gz"))
    if not names:
        return None, None, 0
    f = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return f(names[0]), f(names[-1]), len(names)


def backfill(start: DateLike = "2023-12-01", end: Optional[DateLike] = None) -> Tuple[int, int]:
    """Download every missing session in [start, end]; resumable. → (ok, missing)."""
    end = end or now_et().date()
    days = candidate_days(start, end)
    ok = miss = 0
    for i, d in enumerate(days):
        df = fetch_day(d)
        if df is None:
            miss += 1
        else:
            ok += 1
        if (i + 1) % 50 == 0:
            log.info("finra_hist backfill: %d/%d (ok %d, missing %d)", i + 1, len(days), ok, miss)
    return ok, miss


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill FINRA daily short-volume history")
    ap.add_argument("--start", default="2023-12-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ok, miss = backfill(a.start, a.end)
    first, last, n = cached_range()
    print(f"finra_hist: ok={ok} missing={miss} cache={n} sessions {first}→{last}")
    return 0


if __name__ == "__main__":  # python -m gravity.sources.finra_hist
    sys.exit(main())
