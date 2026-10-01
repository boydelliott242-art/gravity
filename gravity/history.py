"""Small public history stores the site can chart over time.

- ``docs/data/probs/<session>.json``  — every scored name's P(dump) per session
- ``docs/data/borrow/<date>.json``    — IBKR fee / availability for the universe

History starts the day GRAVITY started recording; nothing is back-filled.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config
from .util import clean

log = logging.getLogger(__name__)

PROBS = config.SITE_DATA / "probs"
BORROW = config.SITE_DATA / "borrow"
for _d in (PROBS, BORROW):
    _d.mkdir(parents=True, exist_ok=True)


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), separators=(",", ":")))
    tmp.replace(path)


def save_probs(session: str, features_asof: str, probs: pd.Series) -> None:
    """``probs``: symbol-indexed P(dump). Rounded to 4 dp to keep files small."""
    p = {str(k): round(float(v), 4) for k, v in probs.dropna().items()}
    _write(PROBS / f"{session}.json", {"session_date": session, "features_asof": features_asof, "p": p})
    _recent.cache_clear()


def save_borrow(day: str, borrow: Dict[str, dict], symbols: List[str]) -> None:
    rows = {}
    for s in symbols:
        b = borrow.get(s)
        if b:
            fee = b.get("fee_rate")
            rows[s] = [None if fee is None else round(float(fee), 2), b.get("available")]
    _write(BORROW / f"{day}.json", {"date": day, "b": rows})
    _recent.cache_clear()


@lru_cache(maxsize=4)
def _recent(kind: str, n: int) -> tuple:
    folder = PROBS if kind == "probs" else BORROW
    out = []
    for f in sorted(folder.glob("*.json"))[-n:]:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        out.append((f.stem, d.get("p") if kind == "probs" else d.get("b")))
    return tuple(out)


def prob_history(sym: str, n: int = 30) -> List[list]:
    return [[day, m[sym]] for day, m in _recent("probs", n) if m and sym in m]


def borrow_history(sym: str, n: int = 30) -> List[list]:
    return [[day, m[sym][0], m[sym][1]] for day, m in _recent("borrow", n) if m and sym in m]


def borrow_tightening(sym: str) -> Optional[str]:
    """Plain-English warning when lendable shares collapse or the fee jumps."""
    h = borrow_history(sym, 6)
    if len(h) < 2:
        return None
    (_, f0, a0), (_, f1, a1) = h[0], h[-1]
    if a0 and a1 is not None and a0 > 0 and a1 < 0.5 * a0:
        return f"Lendable shares fell from {int(a0):,} to {int(a1):,} since {h[0][0]} — borrow is tightening"
    if f0 is not None and f1 is not None and f1 >= max(20.0, 2 * f0):
        return f"Borrow fee jumped from {f0:.0f}% to {f1:.0f}% a year since {h[0][0]}"
    return None
