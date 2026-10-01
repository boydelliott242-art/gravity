"""Live tape: every ~15 minutes during the session, how today's #1 and the
board are trading versus their open. Unofficial — the official grade is the
evening one from completed daily bars."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import publish
from .util import market_phase

log = logging.getLogger(__name__)


def _row(sym: str, q: Optional[dict], session: Optional[str] = None) -> Optional[dict]:
    """open→now for ``sym`` — only from a quote of THIS session's regular
    hours (a halted name's chart can still show yesterday's session)."""
    if not q or not q.get("open") or not q.get("last"):
        return None
    if session is not None and (q.get("session_date") != session or q.get("points_session") not in (None, "regular")):
        return None
    o, last = float(q["open"]), float(q["last"])
    hi = float(q.get("high") or last)
    lo = float(q.get("low") or last)
    return {"symbol": sym, "open": o, "last": last, "high": hi, "low": lo,
            "oc_now": last / o - 1, "oh_now": hi / o - 1, "ol_now": lo / o - 1,
            "asof": q.get("asof")}


def snapshot(today: dict) -> Optional[dict]:
    """Build live.json content from today.json's #1 + board (≤ 26 names)."""
    from .sources import intel

    top = (today.get("top") or {}).get("symbol")
    syms: List[str] = [p["symbol"] for p in today.get("board", [])]
    if top and top not in syms:
        syms.insert(0, top)
    if not syms:
        return None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        quotes: Dict[str, Any] = dict(zip(syms, ex.map(lambda s: _safe_intraday(intel, s), syms)))
    session = today.get("session_date")
    rows = {s: _row(s, quotes.get(s), session) for s in syms}
    board = [rows[p["symbol"]] for p in today.get("board", []) if rows.get(p["symbol"])]
    top_row = rows.get(top) if top else None
    if top_row is not None:
        top_row["points"] = (quotes.get(top) or {}).get("points") or []
    oc = [b["oc_now"] for b in board]
    log.info("live: %d/%d quotes in %.0fs", sum(1 for r in rows.values() if r), len(syms), time.time() - t0)
    return {
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_date": today.get("session_date"),
        "phase": market_phase(),
        "top": top_row,
        "board": board,
        "board_mean_oc_now": sum(oc) / len(oc) if oc else None,
        "note": "Live and unofficial — the official grade uses completed daily bars after the close.",
    }


def _safe_intraday(intel, sym: str) -> Optional[dict]:
    try:
        return intel.intraday(sym)
    except Exception as e:  # noqa: BLE001
        log.warning("intraday %s failed: %s", sym, e)
        return None


def run(push: bool = True) -> int:
    phase = market_phase()
    if phase != "open":
        log.info("live: market %s — nothing to do", phase)
        return 0
    today = publish.read_json("today.json") or {}
    from .util import now_et
    if today.get("session_date") != now_et().date().isoformat():
        log.info("live: today.json is for %s, not today — skipping", today.get("session_date"))
        return 0
    snap = snapshot(today)
    if not snap:
        return 0
    publish.write_json("live.json", snap)
    if push:
        publish.push(f"live: {snap['session_date']} {snap['asof'][11:16]}Z")
    return 0
