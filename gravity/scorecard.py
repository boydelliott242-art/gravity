"""The public, append-only track record.

Each published run writes ``docs/data/history/<session>.json`` *before the
open* (the git commit timestamp proves when). After the close, ``grade``
fills in what actually happened using real daily bars and compares the
picks with the whole eligible universe on the same day. Nothing is ever
back-filled or edited after grading.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .util import ET, clean, now_et

log = logging.getLogger(__name__)


def _path(session: str) -> Path:
    return config.PICK_LOG / f"{session}.json"


def _read(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _write(p: Path, obj: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), separators=(",", ":")))
    tmp.replace(p)


def log_picks(today: dict) -> None:
    """Record (or refresh, until graded) the picks for ``today['session_date']``.

    Morning runs overwrite evening runs for the same session — the morning
    list is the one made with pre-market information. A graded file is
    frozen."""
    session = today.get("session_date")
    if not session or today.get("sample") or today.get("late"):
        return
    # Hard rule: a pick only counts if it was on record before that
    # session's 9:30 ET opening bell.
    open_bell = datetime.combine(date.fromisoformat(session), dtime(9, 30), tzinfo=ET)
    if now_et() >= open_bell:
        log.info("not logging picks for %s — the session has already opened", session)
        return
    p = _path(session)
    prev = _read(p) if p.exists() else None
    if prev and prev.get("outcome"):
        log.info("pick log %s already graded — leaving it frozen", session)
        return
    top = today.get("top") or {}
    rec = {
        "session_date": session,
        "run": today.get("run"),
        "published_at": today.get("generated_at"),
        "first_published_at": (prev or {}).get("first_published_at") or today.get("generated_at"),
        "model": (today.get("model") or {}).get("version"),
        "top": {k: top.get(k) for k in ("symbol", "prob_dump", "score", "premarket", "squeeze_danger")} if top else None,
        "board": [
            {"rank": b.get("rank"), "symbol": b.get("symbol"), "prob_dump": b.get("prob_dump")}
            for b in today.get("board", [])
        ],
        "outcome": None,
    }
    _write(p, rec)


def _bar(hist: Dict[str, pd.DataFrame], sym: str, day: pd.Timestamp) -> Optional[pd.Series]:
    df = hist.get(sym)
    if df is None or day not in df.index:
        return None
    r = df.loc[day]
    if not (r["open"] > 0 and r["close"] > 0) or r.get("volume", 0) == 0:
        return None
    return r


def _oc(r: pd.Series) -> dict:
    o = float(r["open"])
    return {
        "open": o, "high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"]),
        "oc": float(r["close"]) / o - 1, "ol": float(r["low"]) / o - 1, "oh": float(r["high"]) / o - 1,
    }


def grade(hist: Dict[str, pd.DataFrame], eligible: List[str]) -> int:
    """Grade every ungraded pick log whose session has a completed bar.
    Returns how many sessions were graded."""
    n = 0
    for p in sorted(config.PICK_LOG.glob("*.json")):
        rec = _read(p)
        if not rec or rec.get("outcome"):
            continue
        day = pd.Timestamp(rec["session_date"])
        # universe comparison for the same session
        uni = [_oc(r) for s in eligible if (r := _bar(hist, s, day)) is not None]
        if len(uni) < 200:  # bars for that day not in yet (or a data outage) — try later
            continue
        u_oc = np.array([x["oc"] for x in uni])
        out = {
            "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "universe_n": len(uni),
            "universe_mean_oc": float(u_oc.mean()),
            "universe_dump_rate": float((u_oc <= config.DUMP_THRESHOLD).mean()),
        }
        top = rec.get("top") or {}
        if top.get("symbol"):
            r = _bar(hist, top["symbol"], day)
            if r is not None:
                t = _oc(r)
                t["dump"] = t["oc"] <= config.DUMP_THRESHOLD
                t["squeezed"] = t["oh"] >= config.SQUEEZE_THRESHOLD
                out["top"] = t
            else:
                out["top"] = {"missing": True, "note": "no regular-session bar (halted or no trades)"}
        b_oc = []
        for b in rec.get("board", []):
            r = _bar(hist, b["symbol"], day)
            if r is not None:
                b_oc.append(_oc(r)["oc"])
        if b_oc:
            arr = np.array(b_oc)
            out["board_n"] = len(arr)
            out["board_mean_oc"] = float(arr.mean())
            out["board_dump_rate"] = float((arr <= config.DUMP_THRESHOLD).mean())
        rec["outcome"] = out
        _write(p, rec)
        n += 1
    return n


def summary() -> dict:
    """Aggregate all graded sessions into docs/data/scorecard.json."""
    days = []
    for p in sorted(config.PICK_LOG.glob("*.json")):
        rec = _read(p)
        if rec and rec.get("outcome"):
            days.append(rec)
    tops = [d["outcome"]["top"] for d in days if d["outcome"].get("top") and not d["outcome"]["top"].get("missing")]
    boards = [d["outcome"] for d in days if d["outcome"].get("board_mean_oc") is not None]
    live = {
        "n_days": len(days),
        "top_n": len(tops),
        "top_dump_rate": float(np.mean([t["dump"] for t in tops])) if tops else None,
        "top_mean_oc": float(np.mean([t["oc"] for t in tops])) if tops else None,
        "top_squeeze_rate": float(np.mean([t["squeezed"] for t in tops])) if tops else None,
        "board_dump_rate": float(np.mean([b["board_dump_rate"] for b in boards])) if boards else None,
        "board_mean_oc": float(np.mean([b["board_mean_oc"] for b in boards])) if boards else None,
        "universe_dump_rate": float(np.mean([d["outcome"]["universe_dump_rate"] for d in days])) if days else None,
        "universe_mean_oc": float(np.mean([d["outcome"]["universe_mean_oc"] for d in days])) if days else None,
    }
    return {
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": list(reversed(days))[:250],
        "live": live,
    }
