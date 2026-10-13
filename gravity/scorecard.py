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


def log_picks(today: dict) -> Optional[bool]:
    """Record (or refresh, until graded) the picks for ``today['session_date']``.

    Morning runs overwrite evening runs for the same session — the morning
    list is the one made with pre-market information — never the reverse. A
    graded file is frozen. Returns True when written, False when refused
    (after the bell / graded / morning already on record), None when not applicable."""
    session = today.get("session_date")
    if not session or today.get("sample") or today.get("late"):
        return None
    # Hard rule: a pick only counts if it was on record before that
    # session's 9:30 ET opening bell.
    open_bell = datetime.combine(date.fromisoformat(session), dtime(9, 30), tzinfo=ET)
    if now_et() >= open_bell:
        log.info("not logging picks for %s — the session has already opened", session)
        return False
    p = _path(session)
    prev = _read(p) if p.exists() else None
    if prev and prev.get("outcome"):
        log.info("pick log %s already graded — leaving it frozen", session)
        return False
    if (prev and prev.get("run") == "morning" and today.get("run") == "morning" and not prev.get("unverified")
            and (prev.get("top") or {}).get("model") == "m1" and (today.get("top") or {}).get("model_used") != "m1"):
        log.info("pick log %s keeps the earlier morning-model list (this run was close-only)", session)
        return False
    if prev and prev.get("run") == "morning" and today.get("run") == "evening" and not prev.get("unverified"):
        log.info("pick log %s already has the morning list — an evening build does not replace it", session)
        return False
    top = today.get("top") or {}
    size_key = str(config.BOARD_MIN_USD)
    cost_at = lambda p_: ((p_.get("size") or {}).get("costs") or {}).get(size_key)  # noqa: E731
    rec = {
        "session_date": session,
        "run": today.get("run"),
        "published_at": today.get("generated_at"),
        "first_published_at": (prev or {}).get("first_published_at") or today.get("generated_at"),
        "model": (today.get("model") or {}).get("version"),
        "top": ({k: top.get(k) for k in ("symbol", "prob_dump", "score", "premarket", "squeeze_danger")}
                | {"model": top.get("model_used")}
                | {"capacity": (top.get("size") or {}).get("capacity"), "breakeven": (top.get("size") or {}).get("breakeven"),
                   "cost30": cost_at(top), "liq_edge": top.get("liq_edge")}) if top else None,
        "board": [
            {"rank": b.get("rank"), "urank": b.get("urank"), "symbol": b.get("symbol"), "prob_dump": b.get("prob_dump"),
             "liq_edge": b.get("liq_edge"), "cost30": cost_at(b)}
            for b in today.get("board", [])
        ],
        "board_size": len(today.get("board", [])),
        # round 11: every board name could take this short; cost30 = its estimated round-trip cost at that size
        "board_min_usd": (today.get("model") or {}).get("board_min_usd"),
        "ranker": (today.get("model") or {}).get("board_ranker"),
        # names held back (no borrow / squeeze danger ≥ 70 / halted) — graded too, to measure the exclusion
        "zone": [{"symbol": z.get("symbol"), "urank": z.get("urank"), "prob_dump": z.get("prob_dump"),
                  "squeeze_danger": z.get("squeeze_danger"), "reason": z.get("zone_reason")}
                 for z in today.get("squeeze_zone", [])],
        "outcome": None,
    }
    _write(p, rec)
    return True


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


def _fresh_through(hist: Dict[str, pd.DataFrame], sym: str, day: pd.Timestamp) -> bool:
    """True when the symbol's frame is known to extend to ``day`` or later,
    so a missing bar on ``day`` genuinely means no trades (halt)."""
    df = hist.get(sym)
    return df is not None and len(df) > 0 and df.index.max() >= day and not df.attrs.get("stale", False)


def ungraded_symbols() -> List[str]:
    """#1 and board symbols of every ungraded pick log (to load their bars
    even if they have dropped out of the scoring universe)."""
    out: List[str] = []
    for p in sorted(config.PICK_LOG.glob("*.json")):
        rec = _read(p)
        if not rec or rec.get("outcome"):
            continue
        top = (rec.get("top") or {}).get("symbol")
        out += ([top] if top else []) + [b.get("symbol") for b in rec.get("board", []) + rec.get("zone", []) if b.get("symbol")]
    return sorted(set(out))


GIVE_UP_SESSIONS = 3   # sessions after which a #1 with no bar is recorded as "no data" instead of waiting


def grade(hist: Dict[str, pd.DataFrame], eligible: List[str], before: Optional[date] = None) -> int:
    """Grade every ungraded pick log whose session has complete data.

    Deferred (left ungraded for a later run) unless at least 80% of the
    names that traded on the previous session also have a bar for this one,
    and unless the #1's own data reaches the session — so a partial refresh
    can never freeze a wrong grade. Returns how many sessions were graded."""
    from .util import prev_trading_day

    n = 0
    for p in sorted(config.PICK_LOG.glob("*.json")):
        rec = _read(p)
        if not rec or rec.get("outcome"):
            continue
        if before is not None and date.fromisoformat(rec["session_date"]) >= before:
            continue   # graded the next evening: Yahoo revises a session's bar after its first print
        day = pd.Timestamp(rec["session_date"])
        prev = pd.Timestamp(prev_trading_day(day.date()))
        uni = [_oc(r) for s in eligible if (r := _bar(hist, s, day)) is not None]
        n_prev = sum(1 for s in eligible if hist.get(s) is not None and prev in hist[s].index)
        if len(uni) < max(200, int(0.8 * n_prev)):
            log.info("grade %s deferred: %d bars vs %d the session before", rec["session_date"], len(uni), n_prev)
            continue
        top_sym = (rec.get("top") or {}).get("symbol")
        give_up = False
        if top_sym and _bar(hist, top_sym, day) is None and not _fresh_through(hist, top_sym, day):
            from .util import next_trading_day
            later = sum(1 for s in eligible[:200] if hist.get(s) is not None and (hist[s].index > day).any())
            sessions_since, d_ = 0, day.date()
            while sessions_since < 10 and (d_ := next_trading_day(d_)) <= now_et().date():
                sessions_since += 1
            # only give up when there is NO frame at all for the #1 (after an explicit load attempt);
            # a frame that exists but is stale means "not refreshed yet", never "vanished"
            if sessions_since < GIVE_UP_SESSIONS or later == 0 or hist.get(top_sym) is not None:
                log.info("grade %s deferred: no fresh data for the #1 (%s) yet", rec["session_date"], top_sym)
                continue
            give_up = True       # the market has moved on; the #1 never got a bar → record it, don't hide it
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
                if isinstance(top.get("cost30"), (int, float)):
                    t["net30"] = -t["oc"] - float(top["cost30"])
                out["top"] = t
            elif give_up:
                out["top"] = {"missing": True, "halted": False,
                              "note": f"no price data for {top['symbol']} {GIVE_UP_SESSIONS}+ sessions later (delisted, renamed or under $0.10?)"}
            else:
                out["top"] = {"missing": True, "halted": True, "note": "no regular-session bar (halted or no trades)"}
        # per board-rank band (1, 2–10, 11–25, 26–50, 51–100) and the shadow-graded zone
        bands = []
        for lo, hi in ((1, 1), (2, 10), (11, 25), (26, 50), (51, 100)):
            xs, missing_ = [], []
            for b in rec.get("board", []):
                if b.get("rank") is not None and lo <= b["rank"] <= hi:
                    r = _bar(hist, b["symbol"], day)
                    (xs.append(_oc(r)["oc"]) if r is not None else missing_.append(b["symbol"]))
            if xs or missing_:
                a = np.array(xs) if xs else np.array([])
                bands.append({"lo": lo, "hi": hi, "n": int(len(a)), "missing": missing_,
                              "dump_rate": float((a <= config.DUMP_THRESHOLD).mean()) if len(a) else None,
                              "mean_oc": float(a.mean()) if len(a) else None,
                              "pump_rate": float((a >= 0.05).mean()) if len(a) else None})
        if bands:
            out["bands"] = bands
        zo, z_missing = [], []
        for z in rec.get("zone", []):
            r = _bar(hist, z.get("symbol"), day)
            (zo.append(r) if r is not None else z_missing.append(z.get("symbol")))
        if zo or z_missing:
            za = np.array([_oc(r)["oc"] for r in zo]) if zo else np.array([])
            out["zone"] = {"n": int(len(za)), "missing": z_missing,
                           "dump_rate": float((za <= config.DUMP_THRESHOLD).mean()) if len(za) else None,
                           "mean_oc": float(za.mean()) if len(za) else None,
                           "squeeze_rate": float((np.array([_oc(r)["oh"] for r in zo]) >= config.SQUEEZE_THRESHOLD).mean()) if zo else None}
        # "board" = the top 25 by rank (comparable with the 25-name boards logged
        # before 2026-10-12); "board100" = every logged board name
        ranked = sorted(rec.get("board", []), key=lambda b: (b.get("rank") or 10**6))
        for key, lim in (("board", 25), ("board100", None)):
            b_oc, b_net = [], []
            for b in ranked[:lim] if lim else ranked:
                r = _bar(hist, b["symbol"], day)
                if r is not None:
                    b_oc.append(_oc(r)["oc"])
                    if isinstance(b.get("cost30"), (int, float)):
                        b_net.append(-b_oc[-1] - float(b["cost30"]))
            if b_oc and (lim or len(ranked) > 25):
                arr = np.array(b_oc)
                out[f"{key}_n"] = len(arr)
                out[f"{key}_mean_oc"] = float(arr.mean())
                out[f"{key}_dump_rate"] = float((arr <= config.DUMP_THRESHOLD).mean())
                if b_net:   # net of each name's own estimated cost at the board size (round 11 logs on)
                    out[f"{key}_net30"] = float(np.mean(b_net))
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
    counted = [d for d in days if not d.get("unverified")]
    tops = [d["outcome"]["top"] for d in counted if d["outcome"].get("top") and not d["outcome"]["top"].get("missing")]
    halted = sum(1 for d in counted if (d["outcome"].get("top") or {}).get("missing"))
    boards = [d["outcome"] for d in days if not d.get("unverified") and d["outcome"].get("board_mean_oc") is not None]
    boards100 = [d["outcome"] for d in days if not d.get("unverified") and d["outcome"].get("board100_mean_oc") is not None]
    live = {
        "n_days": len(counted),
        "n_unverified": len(days) - len(counted),
        "top_n": len(tops),
        "top_halted": halted,
        "top_dump_rate": float(np.mean([t["dump"] for t in tops])) if tops else None,
        "top_mean_oc": float(np.mean([t["oc"] for t in tops])) if tops else None,
        "top_squeeze_rate": float(np.mean([t["squeezed"] for t in tops])) if tops else None,
        "board_dump_rate": float(np.mean([b["board_dump_rate"] for b in boards])) if boards else None,
        "board_mean_oc": float(np.mean([b["board_mean_oc"] for b in boards])) if boards else None,
        "board100_days": len(boards100),
        "board100_dump_rate": float(np.mean([b["board100_dump_rate"] for b in boards100])) if boards100 else None,
        "board100_mean_oc": float(np.mean([b["board100_mean_oc"] for b in boards100])) if boards100 else None,
        # round 11 on: every board name could take a $30K short; net of each name's estimated cost at $30K
        "net30_days": sum(1 for b in boards100 if b.get("board100_net30") is not None),
        "board100_net30": float(np.mean(v)) if (v := [b["board100_net30"] for b in boards100 if b.get("board100_net30") is not None]) else None,
        "board_net30": float(np.mean(v)) if (v := [b["board_net30"] for b in boards if b.get("board_net30") is not None]) else None,
        "top_net30": float(np.mean(v)) if (v := [t["net30"] for t in tops if t.get("net30") is not None]) else None,
        "universe_dump_rate": float(np.mean([d["outcome"]["universe_dump_rate"] for d in days])) if days else None,
        "universe_mean_oc": float(np.mean([d["outcome"]["universe_mean_oc"] for d in days])) if days else None,
    }
    # round 11: the board became $30K-only — the same rates for those sessions alone (the totals above mix both eras)
    d30 = [d for d in counted if d.get("ranker") == "liq"]
    t30 = [d["outcome"]["top"] for d in d30 if d["outcome"].get("top") and not d["outcome"]["top"].get("missing")]
    o30 = [d["outcome"] for d in d30]
    mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    live["board30"] = {
        "since": min(d["session_date"] for d in d30) if d30 else None,
        "n_days": len(d30), "top_n": len(t30),
        "top_dump_rate": mean([t["dump"] for t in t30]), "top_mean_oc": mean([t["oc"] for t in t30]),
        "top_squeeze_rate": mean([t["squeezed"] for t in t30]),
        "top_net30": mean([t["net30"] for t in t30 if t.get("net30") is not None]),
        "board_dump_rate": mean([o["board_dump_rate"] for o in o30 if o.get("board_dump_rate") is not None]),
        "board100_dump_rate": mean([o["board100_dump_rate"] for o in o30 if o.get("board100_dump_rate") is not None]),
        "board100_mean_oc": mean([o["board100_mean_oc"] for o in o30 if o.get("board100_mean_oc") is not None]),
        "board100_net30": mean([o["board100_net30"] for o in o30 if o.get("board100_net30") is not None]),
        "universe_dump_rate": mean([o["universe_dump_rate"] for o in o30]),
    }
    # the site reads each day's #1 and outcome; the full board lists stay in history/<date>.json
    slim_days = [{k: v for k, v in d.items() if k not in ("board", "zone")} for d in reversed(days)][:250]
    return {
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": slim_days,
        "live": live,
    }


def mark_unverified(session: str, reason: str) -> None:
    """A pick whose publication to GitHub could not be confirmed before the
    open is kept for transparency but excluded from the live rates."""
    p = _path(session)
    rec = _read(p) if p.exists() else None
    if rec and not rec.get("outcome"):
        rec["unverified"] = reason
        _write(p, rec)
