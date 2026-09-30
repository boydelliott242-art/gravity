"""Today's ranking: model probabilities + live-only evidence → the board.

Pipeline (called by ``cli``):

1. ``model.predict`` gives calibrated P(dump) / P(big dump) / P(squeeze)
   and an expected open→close for every eligible symbol (M0 = through
   yesterday's close; M1 = also knows the opening gap, proxied live by the
   pre-market price).
2. ``overlay_catalysts`` applies one transparent, evidence-derived odds
   adjustment for filings the model could not have seen (accepted after
   the feature date, or offering language found only by full-text search).
3. ``shortability`` / ``squeeze_danger`` describe whether the name can be
   shorted and how violently it could run the other way.
4. ``build_pick`` assembles the decomposed, sourced Pick object (§11).

Every number shown traces back to a dated fact; the probability is the
model's, never hand-tuned.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import config
from .util import clean

log = logging.getLogger(__name__)

# Categories that mean "new supply of shares is coming".
SUPPLY_CATS = {"offering", "atm", "toxic_financing", "unregistered_sale"}
OVERHANG_CATS = {"registration", "resale", "effective"}
DISTRESS_CATS = {"delisting_notice", "going_concern", "late_filing"}
CAT_LABEL = {
    "offering": "Offering",
    "atm": "At-the-market program",
    "toxic_financing": "Toxic financing (equity line / convertible)",
    "unregistered_sale": "Unregistered share sale",
    "registration": "Registration statement",
    "resale": "Resale registration (selling holders)",
    "effective": "Registration declared effective",
    "delisting_notice": "Exchange deficiency / delisting notice",
    "going_concern": "Going-concern doubt",
    "late_filing": "Late filing notice",
    "reverse_split": "Reverse split",
    "charter_amendment": "Charter amendment (8-K 5.03)",
    "insider_sale_notice": "Insider Form 144 sale notice",
    "insider": "Insider ownership filing",
    "material_agreement": "Material agreement",
}
FAMILIES = ("dilution", "exhaustion", "decay", "flow", "street", "news")


# ── small helpers ────────────────────────────────────────────────────────
def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def pct(x: Optional[float], digits: int = 0, signed: bool = True) -> str:
    if x is None:
        return "—"
    if digits == 0 and 0.99 < abs(x) < 1.0:  # never round −99.8% to "−100%"
        digits = 1
    s = f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"
    return s


def money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:.1f}M"
    if a >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.2f}"


def split_label(ratio: float) -> str:
    """0.05 → '1:20'; 2.0 → '2:1'."""
    if ratio <= 0:
        return "?"
    if ratio < 1:
        return f"1:{round(1 / ratio):d}"
    return f"{round(ratio):d}:1"


# ── shortability & squeeze danger ────────────────────────────────────────
def shortability(sym: str, borrow: Dict[str, dict], borrow_ok: bool) -> dict:
    """ETB / HTB / NONE from IBKR's public borrow file.

    IBKR lists every US name it can lend; a name missing from a *healthy*
    file is not borrowable there. If the file failed to load we say UNKNOWN
    rather than guessing."""
    if not borrow_ok:
        return {"status": "UNKNOWN", "available": None, "fee_rate": None, "asof": None}
    b = borrow.get(sym)
    if not b:
        return {"status": "NONE", "available": 0, "fee_rate": None, "asof": None}
    avail = b.get("available")
    fee = _f(b.get("fee_rate"))
    if avail is not None and avail <= 0:
        status = "NONE"
    elif fee is not None and fee >= 20:
        status = "HTB"
    else:
        status = "ETB"
    return {"status": status, "available": avail, "fee_rate": fee, "asof": b.get("asof")}


def squeeze_danger(
    prob_squeeze_pct: Optional[float],
    short: dict,
    si_pct_float: Optional[float],
    days_to_cover: Optional[float],
    float_shares: Optional[float],
    short_ratio_5d: Optional[float],
) -> tuple:
    """0–100 composite of what makes a short dangerous. Returns (score, parts).

    Components (weights renormalise over what is known):
      model P(squeeze) percentile 40 · borrow fee 15 · scarce lendable 10 ·
      short interest % float 15 · days to cover 5 · tiny float 10 ·
      FINRA short-volume share 5.
    """
    parts: Dict[str, Optional[float]] = {}
    parts["model"] = None if prob_squeeze_pct is None else prob_squeeze_pct / 100.0
    fee = short.get("fee_rate")
    parts["fee"] = None if fee is None else min(1.0, fee / 100.0)
    avail = short.get("available")
    if short.get("status") == "UNKNOWN" or avail is None:
        parts["scarcity"] = None
    else:
        parts["scarcity"] = 1.0 if avail < 10_000 else 0.6 if avail < 50_000 else 0.2 if avail < 250_000 else 0.0
    parts["si"] = None if si_pct_float is None else min(1.0, si_pct_float / 0.30)
    parts["dtc"] = None if days_to_cover is None else min(1.0, days_to_cover / 5.0)
    parts["float"] = None if float_shares is None else (1.0 if float_shares < 2e6 else 0.6 if float_shares < 5e6 else 0.2 if float_shares < 15e6 else 0.0)
    parts["short_vol"] = None if short_ratio_5d is None else max(0.0, min(1.0, (short_ratio_5d - 0.35) / 0.35))
    weights = {"model": 40, "fee": 15, "scarcity": 10, "si": 15, "dtc": 5, "float": 10, "short_vol": 5}
    num = sum(weights[k] * v for k, v in parts.items() if v is not None)
    den = sum(weights[k] for k, v in parts.items() if v is not None)
    score = None if den == 0 else int(round(100 * num / den))
    return score, parts


# ── catalyst overlay ─────────────────────────────────────────────────────
def overlay_catalysts(
    prob: float, fresh_events: List[dict], evidence_lift: Optional[float] = None
) -> tuple:
    """Flag supply events the model could not have seen — WITHOUT changing
    the probability.

    ``fresh_events`` are FilingEvents accepted after the feature date, or
    offering language found only by full-text search. Their effect on top
    of the model's own features has never been validated out of sample
    (an unconditional study lift would double-count risk the model already
    prices), so the probability stays the model's and the event is shown
    as separate, clearly-labelled evidence. Returns (prob, note or None).
    ``evidence_lift`` is accepted for context in the note only."""
    supply = [e for e in fresh_events if e.get("category") in SUPPLY_CATS]
    if not supply:
        return prob, None
    e = supply[0]
    label = CAT_LABEL.get(e.get("category"), e.get("category"))
    note = (f"Fresh {label.lower()} filing ({e.get('form')}, {e.get('date')}) found by the overnight "
            f"filing scan — not reflected in the probability")
    if evidence_lift:
        note += f"; historically, fresh offerings saw {evidence_lift:.1f}× the average dump rate"
    return prob, note


# ── families (descriptive decomposition) ─────────────────────────────────
def family_scores(
    rows: pd.DataFrame, feature_docs: Dict[str, tuple], signs: Dict[str, float]
) -> pd.DataFrame:
    """Percentile (0–100) per family, higher = more dump-like *by the
    direction each feature historically pointed* (``signs`` = sign of the
    feature's rank-correlation with y_dump in training). Purely descriptive:
    it explains the probability, it does not change it."""
    out = pd.DataFrame(index=rows.index)
    fam_map = {
        "dilution": ("dilution",),
        "exhaustion": ("exhaustion",),
        "decay": ("decay", "structure"),
        "flow": ("flow",),
    }
    for fam, src in fam_map.items():
        cols = [c for c, (f, _d) in feature_docs.items() if f in src and c in rows.columns and signs.get(c)]
        if not cols:
            out[fam] = np.nan
            continue
        ranks = []
        for c in cols:
            v = pd.to_numeric(rows[c], errors="coerce") * float(np.sign(signs[c]))
            ranks.append(v.rank(pct=True))
        out[fam] = (pd.concat(ranks, axis=1).mean(axis=1) * 100).round()
    return out


# ── reasons & flags ──────────────────────────────────────────────────────
def build_reasons(
    m: Dict[str, Any],
    pre: Optional[dict],
    filings: List[dict],
    splits: List[dict],
    news: List[dict],
    short: dict,
    session: date,
    catalyst_note: Optional[str],
) -> tuple:
    """Plain-English, dated, sourced reasons + short flag chips."""
    reasons: List[dict] = []
    flags: List[str] = []

    def add(fam: str, text: str, strength: int, url: Optional[str] = None) -> None:
        reasons.append({"family": fam, "text": text, "strength": strength, "url": url})

    r1, r3, rvol = _f(m.get("r1")), _f(m.get("r3")), _f(m.get("rvol1"))
    if r1 is not None and r1 >= 0.20:
        vol_txt = f" on {rvol:.0f}× its normal volume" if rvol and rvol >= 2 else ""
        add("exhaustion", f"Closed {pct(r1)} last session{vol_txt}", 3 if r1 >= 0.5 else 2)
        flags.append(f"PUMP {pct(r1)}")
    elif r3 is not None and r3 >= 0.5:
        add("exhaustion", f"Up {pct(r3)} over the last 3 sessions", 2)
        flags.append(f"RUN {pct(r3)}")
    if pre and _f(pre.get("gap_pct")) is not None:
        g = pre["gap_pct"]
        if abs(g) >= 0.10:
            add("exhaustion", f"Pre-market {pct(g)} at {str(pre.get('asof', ''))[11:16]} ET vs the last close", 3 if g >= 0.3 else 2)
            flags.append(f"GAP {pct(g)}")
    rsi = _f(m.get("rsi14"))
    if rsi is not None and rsi >= 80:
        add("exhaustion", f"RSI(14) {rsi:.0f} — stretched", 1)
    if r1 is not None and r1 <= -0.15:
        add("decay", f"Fell {pct(-r1, signed=False)} last session", 2 if r1 <= -0.25 else 1)
    vol, vol_pct = _f(m.get("vol20")), _f(m.get("vol20_pct"))
    if vol is not None and vol_pct is not None and vol_pct >= 0.85:
        add("decay", f"Typical daily swing ±{vol * 100:.0f}% (20-day volatility) — wilder than "
                         f"{vol_pct * 100:.0f}% of the universe; the model's strongest single driver", 2)
    d50 = _f(m.get("dist_ma50"))
    if d50 is not None and d50 <= -0.30:
        add("decay", f"Trading {pct(-d50, signed=False)} below its 50-day average", 1)

    # filings, newest first
    seen_cat = set()
    for e in filings:
        cat = e.get("category")
        if cat in seen_cat or cat not in (SUPPLY_CATS | OVERHANG_CATS | DISTRESS_CATS | {"reverse_split"}):
            continue
        age = (session - date.fromisoformat(e["date"])).days
        if cat in SUPPLY_CATS and age <= 45:
            add("dilution", f"{CAT_LABEL.get(cat, cat)} — {e.get('form')} filed {e['date']}", 3 if age <= 3 else 2, e.get("url"))
            flags.append("ATM" if cat == "atm" else "OFFERING" if cat == "offering" else "DILUTION")
        elif cat in OVERHANG_CATS and age <= 60:
            add("dilution", f"{CAT_LABEL.get(cat, cat)} — {e.get('form')} filed {e['date']}", 2 if age <= 14 else 1, e.get("url"))
            flags.append("SHELF" if e.get("form", "").startswith(("S-3", "F-3")) else "S-1/F-1" if cat == "registration" else "RESALE" if cat == "resale" else "EFFECTIVE")
        elif cat == "delisting_notice" and age <= 180:
            add("decay", f"Exchange deficiency notice filed {e['date']}", 2, e.get("url"))
            flags.append("DEFICIENCY")
        elif cat == "going_concern" and age <= 365:
            add("decay", f"Going-concern doubt disclosed {e['date']}", 1, e.get("url"))
            flags.append("GOING CONCERN")
        elif cat == "late_filing" and age <= 120:
            add("decay", f"Late-filing notice ({e.get('form')}) {e['date']}", 1, e.get("url"))
            flags.append("LATE FILER")
        else:
            continue
        seen_cat.add(cat)

    rs = [s for s in splits if _f(s.get("ratio")) and s["ratio"] < 1]
    if rs:
        last = rs[-1]
        age = (session - date.fromisoformat(last["date"])).days
        n2y = sum(1 for s in rs if (session - date.fromisoformat(s["date"])).days <= 730)
        txt = f"{split_label(last['ratio'])} reverse split on {last['date']}"
        if n2y >= 2:
            txt += f" ({n2y} reverse splits in 2 years)"
        add("decay", txt, 2 if age <= 60 or n2y >= 2 else 1)
        flags.append(f"R/S {split_label(last['ratio'])}")

    dd = _f(m.get("dd_52w"))
    if dd is not None and dd <= -0.80:
        add("decay", f"{pct(dd)} from its 52-week high", 1)
    price = _f(m.get("close"))
    if price is not None and price < 1:
        flags.append("SUB-$1")

    for n in news[:4]:
        if n.get("polarity", 0) < 0 and n.get("tags"):
            add("news", f"“{n['title'][:110]}” — {n.get('source', '')}", 2, n.get("url"))
            break

    sr = _f(m.get("short_ratio_5d"))
    if sr is not None and sr >= 0.55:
        add("flow", f"{pct(sr, signed=False)} of the last 5 sessions' reported volume was short sales (FINRA)", 1)

    if short.get("status") == "HTB" and short.get("fee_rate") is not None:
        flags.append(f"HTB {short['fee_rate']:.0f}%")
    elif short.get("status") == "NONE":
        flags.append("NO BORROW")
    if m.get("asia"):
        flags.append("ASIA")
    if _f(m.get("volume")) == 0:
        flags.append("HALTED")
    if catalyst_note:
        add("dilution", catalyst_note, 3)
        if "FRESH FILING" not in flags:
            flags.append("FRESH FILING")

    reasons.sort(key=lambda r: -r["strength"])
    # de-dup flags, keep order
    seen, fl = set(), []
    for f_ in flags:
        if f_ not in seen:
            seen.add(f_)
            fl.append(f_)
    return reasons, fl


def chart_rows(df: Optional[pd.DataFrame], n: int = config.CHART_SESSIONS) -> list:
    if df is None or df.empty:
        return []
    tail = df.tail(n)
    return [
        [d.strftime("%Y-%m-%d"), _f(r.open), _f(r.high), _f(r.low), _f(r.close), _f(r.volume)]
        for d, r in tail.iterrows()
    ]


def rank_percentile(values: pd.Series) -> pd.Series:
    return (values.rank(pct=True) * 100).round().astype("Int64")


def to_json_safe(x: Any) -> Any:
    return clean(x)


def iter_top(df: pd.DataFrame, n: int) -> Iterable:
    return df.head(n).itertuples()
