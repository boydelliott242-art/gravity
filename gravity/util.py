"""Small shared helpers: US-market clock, number parsing, symbol mapping."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — py3.9 ships zoneinfo
    ZoneInfo = None  # type: ignore

ET = ZoneInfo("America/New_York") if ZoneInfo else timezone(timedelta(hours=-4))

# NYSE full-day holidays 2024-2027 (early closes are treated as normal days).
_HOLIDAYS = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04",
    "2028-09-04", "2028-11-23", "2028-12-25",
}
HOLIDAYS_THROUGH = 2028


def holidays_covered(year: int) -> bool:
    """False once the calendar runs past the hard-coded holiday table."""
    return year <= HOLIDAYS_THROUGH


def now_et() -> datetime:
    return datetime.now(ET)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _HOLIDAYS


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    d = d - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def target_session(at: Optional[datetime] = None) -> date:
    """The session a run is picking for: today if we're before the 16:00 ET
    close on a trading day, else the next trading day."""
    at = (at or now_et()).astimezone(ET)
    d = at.date()
    if is_trading_day(d) and at.time() < time(16, 0):
        return d
    return next_trading_day(d)


def market_phase(at: Optional[datetime] = None) -> str:
    """'pre-market' | 'open' | 'after-hours' | 'closed'."""
    at = (at or now_et()).astimezone(ET)
    if not is_trading_day(at.date()):
        return "closed"
    t = at.time()
    if time(4, 0) <= t < time(9, 30):
        return "pre-market"
    if time(9, 30) <= t < time(16, 0):
        return "open"
    if time(16, 0) <= t < time(20, 0):
        return "after-hours"
    return "closed"


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def num(x: Any) -> Optional[float]:
    """Parse '$1,234.5', '12.3%', 'N/A', 1.2 → float or None. Percent strings
    are returned as the number shown (12.3), not a fraction."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)
    s = str(x).replace(",", "").replace("$", "").strip()
    m = _NUM.search(s)
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def clean(x: Any) -> Any:
    """Make a value JSON-safe: NaN/inf → None, numpy scalars → Python, round floats."""
    try:
        import numpy as np  # local import keeps util light
        if isinstance(x, np.generic):
            x = x.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 6)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (date, datetime)):
        return x.isoformat()
    return x


def to_yahoo(sym: str) -> str:
    """Nasdaq-style 'BRK/A' or 'BRK.A' → Yahoo 'BRK-A'."""
    return sym.replace("/", "-").replace(".", "-").upper()


def to_canonical(sym: str) -> str:
    """Canonical symbol used everywhere in GRAVITY = Yahoo style."""
    return to_yahoo(sym.strip())
