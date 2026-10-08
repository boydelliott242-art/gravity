"""Live tape: every ~15 minutes during the session, how today's #1 and the
board are trading versus their open. Unofficial — the official grade is the
evening one from completed daily bars."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import json

from . import config, net, publish
from .util import market_phase, now_et, num

SPREAD_LOG = config.DATA / "spreads"          # private: real quotes to calibrate the cost model later


def quoted_spread(sym: str) -> Optional[dict]:
    """Live bid/ask from Nasdaq's quote (regular session only)."""
    d = net.get_json(f"https://api.nasdaq.com/api/quote/{sym}/info", headers=net.NASDAQ_HEADERS,
                     params={"assetclass": "stocks"}, timeout=15.0, retries=1)
    try:
        data = d["data"]
        prim = data["primaryData"] or {}
    except (TypeError, KeyError):
        return None
    if str(data.get("marketStatus") or "") not in ("Market Open", "Open"):
        return None
    bid, ask = num(prim.get("bidPrice")), num(prim.get("askPrice"))
    if not bid or not ask or ask <= bid:
        return None
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "spread": (ask - bid) / mid}


def _log_spreads(rows: Dict[str, dict]) -> None:
    if not rows:
        return
    SPREAD_LOG.mkdir(parents=True, exist_ok=True)
    stamp = now_et().isoformat(timespec="seconds")
    with open(SPREAD_LOG / f"{now_et().date().isoformat()}.jsonl", "a") as f:
        for sym, q in rows.items():
            f.write(json.dumps({"ts": stamp, "symbol": sym, **q}) + "\n")

log = logging.getLogger(__name__)


def recent_spreads(days: int = 5, min_quotes: int = 3) -> Dict[str, float]:
    """Median real bid/ask spread per symbol over the last ``days`` logged
    sessions (regular-hours quotes recorded by this job), for symbols with at
    least ``min_quotes`` quotes. Used by the cost model in place of the estimate."""
    import statistics
    files = sorted(SPREAD_LOG.glob("*.jsonl"))[-days:] if SPREAD_LOG.exists() else []
    acc: Dict[str, list] = {}
    for fp in files:
        try:
            for line in fp.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                v = r.get("spread")
                if isinstance(v, (int, float)) and 0 < v < 0.5 and r.get("symbol"):
                    acc.setdefault(r["symbol"], []).append(float(v))
        except OSError:
            continue
    return {s: statistics.median(v) for s, v in acc.items() if len(v) >= min_quotes}


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
    with ThreadPoolExecutor(max_workers=6) as ex:
        sp = dict(zip(syms, ex.map(lambda s: _safe_spread(s), syms)))
    sp = {k: v for k, v in sp.items() if v}
    _log_spreads(sp)
    for s, r in rows.items():
        if r is not None and s in sp:
            r["spread_now"] = sp[s]["spread"]
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
        "unquoted": [p["symbol"] for p in today.get("board", []) if not rows.get(p["symbol"])],
        "note": "Live and unofficial — the official grade uses completed daily bars after the close.",
    }


def _safe_spread(sym: str) -> Optional[dict]:
    try:
        return quoted_spread(sym)
    except Exception as e:  # noqa: BLE001
        log.warning("spread %s failed: %s", sym, e)
        return None


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
