"""Evidence Lab: conditional base rates on the feature panel (CONTRACTS §9).

Each study answers one plain question — "after X, how often did the next
session dump?" — with nothing but counting on the labeled panel:

* condition on row ``t`` (known at the close of ``t``, except the gap-up
  study, whose condition is the ``t+1`` opening gap: it is known at 9:30,
  so its outcome is measured from that open);
* outcomes = the ``t+1`` labels (open→close, open→high, close five
  sessions later);
* a 95 % confidence interval for the dump rate from a bootstrap over
  **dates** (rows on the same day are correlated — one market-wide selloff
  is not 300 independent observations);
* ``lift_dump`` = the study's dump rate ÷ the dump rate of every row where
  the condition could be evaluated (e.g. filing studies compare against
  names that have SEC history).

No model, no fitting, no fabricated numbers: a study with no qualifying
rows reports ``n = 0`` and nulls.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .util import clean

log = logging.getLogger(__name__)

N_BOOT = 1000
SMALL_N = 100          # below this, the sentence says "small sample"
SEED = 7


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# Each study: id, title, condition text, plain lead-in, mask fn, defined fn.
# ``mask`` → rows meeting the condition; ``defined`` → rows where the
# condition could be evaluated (its inputs are not missing).
Cond = Callable[[pd.DataFrame], pd.Series]


def _studies() -> List[Dict[str, Any]]:
    def ge(c: str, v: float) -> Cond:
        return lambda p: _col(p, c) >= v

    def gt(c: str, v: float) -> Cond:
        return lambda p: _col(p, c) > v

    def lt(c: str, v: float) -> Cond:
        return lambda p: _col(p, c) < v

    def le(c: str, v: float) -> Cond:
        return lambda p: _col(p, c) <= v

    def has(*cs: str) -> Cond:
        return lambda p: pd.concat([_col(p, c).notna() for c in cs], axis=1).all(axis=1)

    out = [
        dict(id="up20", title="Up 20%+ on the day", condition="r1 ≥ +20%",
             lead="After a stock closed up 20% or more on the day", mask=ge("r1", 0.20), defined=has("r1")),
        dict(id="up50", title="Up 50%+ on the day", condition="r1 ≥ +50%",
             lead="After a stock closed up 50% or more on the day", mask=ge("r1", 0.50), defined=has("r1")),
        dict(id="up100", title="Up 100%+ on the day", condition="r1 ≥ +100%",
             lead="After a stock at least doubled on the day", mask=ge("r1", 1.00), defined=has("r1")),
        dict(id="up200", title="Up 200%+ on the day", condition="r1 ≥ +200%",
             lead="After a stock at least tripled on the day", mask=ge("r1", 2.00), defined=has("r1")),
        dict(id="run3_100", title="3-day run over +100%", condition="r3 > +100%",
             lead="After a stock more than doubled over three sessions", mask=gt("r3", 1.00), defined=has("r3")),
        dict(id="gap30", title="Gap-up over 30% (same session, from the open)",
             condition="next-session open ≥ 30% above the prior close; outcome measured from that open",
             lead="When a stock opened 30% or more above the prior close",
             mask=gt("y_gap", 0.30), defined=has("y_gap"), same_day=True),
        dict(id="rs_5", title="Reverse split in the last 5 sessions", condition="sess_since_rs ≤ 5",
             lead="Within five sessions of a reverse split", mask=le("sess_since_rs", 5), defined=has("r1")),
        dict(id="rs_30", title="Reverse split in the last 30 sessions", condition="sess_since_rs ≤ 30",
             lead="Within 30 sessions of a reverse split", mask=le("sess_since_rs", 30), defined=has("r1")),
        dict(id="offer_1", title="Offering filed ≤ 1 session ago",
             condition="424B1/2/4/5/7 or S-1MEF/F-1MEF filed today or the prior session (sess_since_offer ≤ 1)",
             lead="After an offering prospectus was filed that session or the one before",
             mask=le("sess_since_offer", 1), defined=has("n_offer_30")),
        dict(id="reg_30", title="Registration filed ≤ 30 days", condition="S-1/F-1/S-3/F-3 (+amendments) or 424B3 in the last 30 days",
             lead="Within 30 days of a registration statement", mask=ge("n_reg_30", 1), defined=has("n_reg_30")),
        dict(id="delist_90", title="Deficiency notice ≤ 90 days", condition="8-K item 3.01 in the last 90 days",
             lead="Within 90 days of a listing-deficiency notice (8-K item 3.01)",
             mask=ge("n_delist_90", 1), defined=has("n_delist_90")),
        dict(id="sub1", title="Price under $1", condition="as-traded close < $1.00",
             lead="When the stock closed under $1", mask=lt("price", 1.0), defined=has("price")),
        dict(id="dd90", title="52-week drawdown over 90%", condition="close < 10% of the 52-week high",
             lead="When the stock was down more than 90% from its 52-week high",
             mask=lt("dd_52w", -0.90), defined=has("dd_52w")),
        dict(id="rsi85", title="RSI(14) above 85", condition="rsi14 > 85",
             lead="When 14-day RSI was above 85", mask=gt("rsi14", 85), defined=has("rsi14")),
        dict(id="asia_ipo2y", title="Asia-linked, IPO ≤ 2 years", condition="asia = 1 and ipo_age ≤ 2",
             lead="For Asia-linked issuers within two years of their IPO",
             mask=lambda p: (_col(p, "asia") == 1) & (_col(p, "ipo_age") <= 2),
             defined=has("asia", "ipo_age")),
        dict(id="inhd_profile", title="INHD profile",
             condition="asia = 1, ≥ 2 reverse splits in 2 years, 52-week drawdown > 90%",
             lead="For INHD-like names (Asia-linked, two or more reverse splits in two years, down 90%+ from the high)",
             mask=lambda p: (_col(p, "asia") == 1) & (_col(p, "rs_count_2y") >= 2) & (_col(p, "dd_52w") < -0.90),
             defined=has("asia", "rs_count_2y", "dd_52w")),
    ]
    return out


def _date_ci(day_codes: np.ndarray, hits: np.ndarray, n_days: int, rng: np.random.Generator,
             n_boot: int = N_BOOT) -> Optional[List[float]]:
    """95 % CI of a pooled rate by resampling whole dates (vectorised)."""
    if len(hits) == 0:
        return None
    k = np.rint(np.bincount(day_codes, weights=hits, minlength=n_days)).astype(np.int64)
    n = np.bincount(day_codes, minlength=n_days).astype(np.int64)
    used = n > 0
    k, n = k[used], n[used]
    D = len(n)
    if D < 2:
        return None
    w = rng.multinomial(D, np.full(D, 1.0 / D), size=n_boot)          # (B, D) date multiplicities
    num, den = w @ k, w @ n                                            # integer matmul: exact counts
    rate = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(rate, [2.5, 97.5])
    return [float(lo), float(hi)]


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def _signed_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:+.1f}%"


def _stats(sub: pd.DataFrame) -> Dict[str, Any]:
    oc = _col(sub, "y_oc")
    c5 = _col(sub, "y_c5")

    def mean(s: pd.Series) -> Optional[float]:
        s = s.dropna()
        return float(s.mean()) if len(s) else None

    def med(s: pd.Series) -> Optional[float]:
        s = s.dropna()
        return float(s.median()) if len(s) else None

    return {
        "n": int(len(sub)),
        "n_symbols": int(sub["symbol"].nunique()) if len(sub) else 0,
        "n_days": int(sub["date"].nunique()) if len(sub) else 0,
        "pct_red_oc": mean((oc < 0).astype(float).where(oc.notna())),
        "pct_dump": mean(_col(sub, "y_dump")),
        "pct_bigdump": mean(_col(sub, "y_bigdump")),
        "pct_squeeze": mean(_col(sub, "y_squeeze")),
        "pct_up5": mean((oc >= 0.05).astype(float).where(oc.notna())),
        "median_oc": med(oc),
        "mean_oc": mean(oc),
        "median_c5": med(c5),
    }


def _sentence(lead: str, st: Dict[str, Any], ref: Optional[float], same_day: bool) -> str:
    """One plain-English sentence, generated only from the study's numbers."""
    n = st["n"]
    if n == 0:
        return f"{lead}: no qualifying sessions in the data, so there is nothing to measure."
    when = "that same session" if same_day else "the next session"
    ci = st.get("ci_pct_dump")
    ci_txt = f" (95% CI {_pct(ci[0])}–{_pct(ci[1])})" if ci else ""
    lift = st.get("lift_dump")
    if lift is None or ref is None:
        cmp_txt = ""
    elif 0.9 <= lift <= 1.1:
        cmp_txt = f", about the same as the {_pct(ref)} rate for comparable name-days"
    else:
        cmp_txt = f" — {lift:.2f}× the {_pct(ref)} rate for comparable name-days"
    small = f" Small sample (n = {n:,}) — treat as anecdotal." if n < SMALL_N else ""
    up = st.get("pct_up5")
    lu = st.get("lift_up5")
    if up is not None:
        both = (f" It also RAN UP 5%+ open→close {_pct(up)} of the time"
                + (f" ({lu:.2f}× normal)" if lu else "")
                + (" — so this is mostly a bigger-move signal, not a clean downside one." if st.get("skew") == "both"
                   else " — the tilt is to the upside." if st.get("skew") == "up" else " — the tilt is to the downside."))
    else:
        both = ""
    return (
        f"{lead} ({n:,} cases across {st['n_symbols']:,} stocks), {when} fell "
        f"{abs(config.DUMP_THRESHOLD):.0%}+ from open to close {_pct(st['pct_dump'])} of the time{ci_txt}"
        f"{cmp_txt}. Median open→close {_signed_pct(st['median_oc'])}; it closed below its open "
        f"{_pct(st['pct_red_oc'])} of the time and squeezed {config.SQUEEZE_THRESHOLD:.0%}+ above the open "
        f"{_pct(st['pct_squeeze'])} of the time.{both}{small}"
    )


def run_studies(panel: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """Baseline + event studies (CONTRACTS §9). Written by the caller to
    docs/data/evidence.json."""
    rng = np.random.default_rng(seed)
    lab = panel[_col(panel, "y_oc").notna() & _col(panel, "y_dump").notna()] if len(panel) else panel
    lab = lab.reset_index(drop=True)
    day = pd.to_datetime(lab["date"]) if len(lab) else pd.Series([], dtype="datetime64[ns]")
    codes, uniq = pd.factorize(day, sort=True)
    n_days = len(uniq)
    dump = _col(lab, "y_dump").to_numpy(float)

    def study(sid: str, title: str, condition: str, lead: str, mask: np.ndarray, defined: np.ndarray,
              same_day: bool, compare: bool = True) -> Dict[str, Any]:
        sub = lab[mask]
        st = _stats(sub)
        ref_rows = defined
        ref = float(np.mean(dump[ref_rows])) if ref_rows.any() else None
        st["ci_pct_dump"] = _date_ci(codes[mask], dump[mask], n_days, rng, n_boot) if mask.any() else None
        st["lift_dump"] = (st["pct_dump"] / ref) if (st["pct_dump"] is not None and ref) else None
        st["ref_pct_dump"] = ref
        st["ref_n"] = int(ref_rows.sum())
        oc_all = _col(lab, "y_oc").to_numpy(float)
        up_ref = float(np.nanmean(oc_all[ref_rows] >= 0.05)) if ref_rows.any() else None
        st["ref_pct_up5"] = up_ref
        st["lift_up5"] = (st["pct_up5"] / up_ref) if (st.get("pct_up5") is not None and up_ref) else None
        st["skew"] = ("down" if (st["lift_dump"] or 0) > (st["lift_up5"] or 0) * 1.1
                      else "up" if (st["lift_up5"] or 0) > (st["lift_dump"] or 0) * 1.1 else "both")
        out = {"id": sid, "title": title, "condition": condition, "same_session_condition": bool(same_day)}
        out.update(st)
        out["plain"] = _sentence(lead, st, ref if compare else None, same_day)
        return out

    everything = np.ones(len(lab), dtype=bool)
    base = study("baseline", "All eligible name-days",
                 "every labeled row (price ≥ $0.10, 20-day median $ volume ≥ $50k, ≥ 60 prior sessions)",
                 "Across every eligible small/micro-cap session", everything, everything, False, compare=False)
    studies = [base]
    for sd in _studies():
        m = sd["mask"](lab).fillna(False).to_numpy(bool) if len(lab) else np.zeros(0, bool)
        dfd = sd["defined"](lab).fillna(False).to_numpy(bool) if len(lab) else np.zeros(0, bool)
        studies.append(study(sd["id"], sd["title"], sd["condition"], sd["lead"], m & dfd, dfd,
                             bool(sd.get("same_day"))))
    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(lab)),
        "date_min": uniq.min().strftime("%Y-%m-%d") if n_days else None,
        "date_max": uniq.max().strftime("%Y-%m-%d") if n_days else None,
        "definitions": {
            "dump": f"next-session open→close ≤ {config.DUMP_THRESHOLD:+.0%}",
            "bigdump": f"next-session open→close ≤ {config.BIG_DUMP_THRESHOLD:+.0%}",
            "squeeze": f"next-session open→high ≥ {config.SQUEEZE_THRESHOLD:+.0%}",
            "median_c5": "close five sessions later ÷ next-session open − 1",
            "ci_pct_dump": f"95% bootstrap interval, {n_boot} resamples of whole dates",
            "lift_dump": "study dump rate ÷ dump rate of all rows where the condition could be evaluated",
        },
        "caveats": [
            "Conditional base rates, not a trading system: no borrow, fees or slippage.",
            "Universe = names listed today (survivorship bias); delisted collapses are missing.",
            "Filing-based studies only cover names with SEC filing history in the data.",
        ],
        "baseline": base,
        "studies": studies,
    }
    return clean(res)
