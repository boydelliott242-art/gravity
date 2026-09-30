"""The tradeable universe: every US-listed common stock that is small enough
and priced high enough to matter, straight from Nasdaq's stock screener.

One request (~7,000 rows, ~2 MB) gives symbol, last sale, change, volume,
market cap, country, IPO year, sector and industry for every Nasdaq, NYSE
and NYSE American listing. We keep common equity (including ADRs) and drop
everything that is not: preferreds, warrants, rights, SPAC units, notes,
debentures and closed-end funds.

Nothing is imputed. Unparseable numbers become NaN; a zero market cap (the
screener's way of saying "unknown") becomes NaN; a row with no last sale is
dropped because it cannot be priced.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import config, net
from ..util import num, to_canonical

log = logging.getLogger(__name__)

SOURCE_NAME = "Nasdaq screener"
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"

#: Output columns, in contract order (CONTRACTS.md §1).
COLUMNS = [
    "symbol", "name", "price", "pct_change", "volume", "market_cap",
    "country", "ipo_year", "sector", "industry", "asia",
]

# A healthy payload has ~7,000 rows; anything tiny is a broken response and
# must not be cached or trusted.
_MIN_ROWS = 1000
# How old a cached payload may be when the live fetch fails.
_STALE_MAX_AGE_S = 7 * 24 * 3600

# ── Instrument filters ───────────────────────────────────────────────────
# Case-sensitive, word-bounded: "Unity Bancorp" and "Century Communities"
# must not match "Unit".
_WARRANT = re.compile(r"\bWarrants?\b|\bWts?\b|\(ADW\)")
_RIGHT = re.compile(r"\bRights?\b")
# SPAC-style units ("… Acquisition Corp. Units", "Units, each consisting of …")
# and mandatory-convertible "Corporate Units". MLP / LLC common units are
# common-equity exposure and are kept (see _EQUITY_UNIT).
_UNIT = re.compile(r"\bUnits?\s*$|\bUnits?\s*,?\s*each\b|\bCorporate Units?\b")
_EQUITY_UNIT = re.compile(
    r"\b(?:Common|Partnership|Partners|Class [A-Z]|L\.?P\.?|LLC)\s+Units?\b"
    r"|\bUnits? (?:of Beneficial Interest|representing)\b"
)
_PREFERRED = re.compile(r"\bPreferred\b|\bPreference\b|\bPfd\b|\bPref\b")
_DEBT = re.compile(r"\bNotes\b|\bNote due\b|\bDebentures?\b|\bBonds? due\b")
_PERCENT = re.compile(r"%")
_FUND = re.compile(
    r"\bFunds?\b|\bETF\b|\bETN\b|\bMunicipals?\b|\bClosed[- ]End\b"
    r"|\bTerm Trust\b|\bIncome Trust\b|\bBond Trust\b|\bTax[- ]Free\b"
)
_TRUST_INDUSTRY = "Trusts Except Educational Religious and Charitable"
# Nasdaq/NYSE test issues (ZVZZT, ZXZZT, NTEST …).
_TEST_SYMBOL = re.compile(r"^Z[A-Z]ZZT$|^[A-Z]TEST$")


def exclusion_reason(symbol: str, name: str, sector: str = "", industry: str = "") -> Optional[str]:
    """Why a screener row is not common equity, or ``None`` to keep it.

    ``symbol`` is the raw Nasdaq symbol (``ABR^D``, ``BRK/A``). Returns one of
    ``"preferred"``, ``"warrant"``, ``"right"``, ``"unit"``, ``"debt"``,
    ``"fund"``, ``"test"``. ADRs, MLP common units and dual-class shares are
    kept on purpose: they are common-equity exposure.
    """
    sym = (symbol or "").strip().upper()
    nm = " ".join((name or "").split())
    if not sym:
        return "test"
    if "^" in sym:
        return "preferred"
    if _TEST_SYMBOL.match(sym):
        return "test"
    if _PREFERRED.search(nm):
        return "preferred"
    if _WARRANT.search(nm):
        return "warrant"
    if _RIGHT.search(nm):
        return "right"
    if _UNIT.search(nm) and not _EQUITY_UNIT.search(nm):
        return "unit"
    if _DEBT.search(nm):
        return "debt"
    if _PERCENT.search(nm):
        return "debt"          # coupon-bearing paper ("7.50% … due 2031")
    if _FUND.search(nm):
        return "fund"
    if sector == "Finance" and "Beneficial Interest" in nm:
        return "fund"          # closed-end funds; REITs sit in "Real Estate"
    if industry == _TRUST_INDUSTRY and re.search(r"\bTrust\b", nm):
        return "fund"
    # 5-letter Nasdaq suffix convention (…W warrant, …R right, …U unit),
    # only when the name confirms it — plenty of real companies end in R/U.
    if len(sym) == 5 and sym[-1] in "WRU" and re.search(r"Warrant|Right|Unit|\bWt", nm, re.I):
        return {"W": "warrant", "R": "right", "U": "unit"}[sym[-1]]
    return None


def _fetch_payload() -> Optional[Dict[str, Any]]:
    """One live screener request → ``{"fetched_at", "as_of", "rows"}`` or
    ``None`` (never cached when ``None``)."""
    data = net.get_json(SCREENER_URL, headers=net.NASDAQ_HEADERS, timeout=60.0)
    try:
        rows = data["data"]["rows"]
    except (TypeError, KeyError):
        return None
    if not isinstance(rows, list) or len(rows) < _MIN_ROWS:
        log.warning("screener returned %s rows — treating as a failed fetch",
                    len(rows) if isinstance(rows, list) else "no")
        return None
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": (data.get("data") or {}).get("asOf"),
        "rows": rows,
    }


def _text(x: Any) -> str:
    return " ".join(str(x).split()) if x is not None else ""


def _positive(x: Any) -> float:
    v = num(x)
    return float(v) if v is not None and v > 0 else np.nan


def parse_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Raw screener rows → typed frame with contract columns plus ``reason``
    (exclusion reason, ``None`` for common equity). No filtering here."""
    out = []
    for r in rows:
        raw_sym = _text(r.get("symbol"))
        name = _text(r.get("name"))
        sector = _text(r.get("sector"))
        industry = _text(r.get("industry"))
        country = _text(r.get("country"))
        pct = num(r.get("pctchange"))
        vol = num(r.get("volume"))
        ipo = num(r.get("ipoyear"))
        out.append({
            "symbol": to_canonical(raw_sym) if raw_sym else "",
            "name": name,
            "price": _positive(r.get("lastsale")),
            "pct_change": pct / 100.0 if pct is not None else np.nan,
            "volume": float(vol) if vol is not None and vol >= 0 else np.nan,
            "market_cap": _positive(r.get("marketCap")),
            "country": country,
            "ipo_year": float(ipo) if ipo is not None and ipo > 1800 else np.nan,
            "sector": sector,
            "industry": industry,
            "asia": country in config.ASIA_COUNTRIES,
            "reason": exclusion_reason(raw_sym, name, sector, industry),
        })
    df = pd.DataFrame(out, columns=COLUMNS + ["reason"])
    for c in ("price", "pct_change", "volume", "market_cap", "ipo_year"):
        df[c] = df[c].astype(float)
    df["asia"] = df["asia"].astype(bool)
    return df


def filter_universe(parsed: pd.DataFrame) -> pd.DataFrame:
    """Apply the contract's exclusions to a ``parse_rows`` frame.

    Adds ``attrs["excluded"]`` = {reason: count} so the pipeline can show
    exactly what was dropped and why.
    """
    df = parsed.copy()
    reason = df["reason"].copy()
    reason[reason.isna() & df["price"].isna()] = "no_price"
    reason[reason.isna() & (df["price"] < config.MIN_PRICE)] = "price_below_min"
    reason[reason.isna() & (df["market_cap"] > config.MAX_MARKET_CAP)] = "market_cap_above_max"
    keep = reason.isna()
    excluded = {str(k): int(v) for k, v in reason[~keep].value_counts().items()}
    kept = df.loc[keep, COLUMNS]
    dupes = int(kept["symbol"].duplicated().sum())
    if dupes:
        excluded["duplicate_symbol"] = dupes
    kept = (kept.drop_duplicates("symbol", keep="first")
                .sort_values("symbol")
                .reset_index(drop=True))
    kept.attrs["excluded"] = excluded
    return kept


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})
    for c in ("price", "pct_change", "volume", "market_cap", "ipo_year"):
        df[c] = df[c].astype(float)
    df["asia"] = df["asia"].astype(bool)
    return df


def load_universe(max_age_s: int = 6 * 3600) -> pd.DataFrame:
    """Eligible small/micro-cap common stocks, one row per canonical symbol.

    Columns (CONTRACTS.md §1): ``symbol name price pct_change volume
    market_cap country ipo_year sector industry asia``. ``pct_change`` is a
    fraction (−0.035 = −3.5 %). Exclusions: non-common instruments (see
    ``exclusion_reason``), rows with no last sale, ``price < MIN_PRICE`` and
    ``market_cap > MAX_MARKET_CAP`` (an unknown market cap is kept as NaN).

    The raw payload is cached for ``max_age_s``. If the live fetch fails, a
    cached payload up to 7 days old is used and the source is reported as
    not live. ``attrs`` carries provenance: ``listed`` (raw row count),
    ``common`` (after instrument exclusions), ``excluded`` (reason → count),
    ``fetched_at``, ``stale``, ``source``, ``source_url``.
    """
    key = SCREENER_URL
    payload = net.cached("universe", key, max_age_s, _fetch_payload)
    stale = False
    if payload is None:
        payload = net.cache_get("universe", key, _STALE_MAX_AGE_S)
        stale = payload is not None
    if not payload or not payload.get("rows"):
        net.record_status(SOURCE_NAME, False, "screener unreachable and no cached copy")
        df = _empty()
        df.attrs.update({"listed": 0, "common": 0, "excluded": {}, "fetched_at": None,
                         "stale": False, "source": SOURCE_NAME, "source_url": SCREENER_URL})
        return df

    rows = payload["rows"]
    parsed = parse_rows(rows)
    df = filter_universe(parsed)
    common = int(parsed["reason"].isna().sum())
    df.attrs.update({
        "listed": len(rows),
        "common": common,
        "fetched_at": payload.get("fetched_at"),
        "stale": stale,
        "source": SOURCE_NAME,
        "source_url": SCREENER_URL,
    })
    detail = (f"{len(rows):,} listed → {common:,} common stocks → {len(df):,} in universe "
              f"(price ≥ ${config.MIN_PRICE:g}, cap ≤ ${config.MAX_MARKET_CAP / 1e9:g}B)")
    if stale:
        detail = f"live fetch failed; using cached copy from {payload.get('fetched_at')} — " + detail
    net.record_status(SOURCE_NAME, not stale, detail)
    log.info("universe: %s", detail)
    return df
