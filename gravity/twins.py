"""INHD lookalikes — names whose *profile* resembles the reference short.

Similarity is a weighted distance on robust-standardised profile traits
(size, price, reverse-split habit, drawdown, dilution cadence, geography,
listing age, volatility, liquidity). It says "structurally similar", not
"will behave the same".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

# trait → (weight, human label builder)
TRAITS = {
    "log_mcap": 1.4,
    "price_log": 0.8,
    "rs_count_2y": 1.4,
    "dd_52w": 1.2,
    "n_offer_365": 1.0,
    "asia": 1.2,
    "listing_age": 0.7,
    "vol60": 0.8,
    "dvol20_log": 0.6,
}


def _robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr / 1.349 if iqr and iqr > 0 else (s.std() or 1.0)
    return ((s - med) / scale).clip(-4, 4)


def profile_frame(rows: pd.DataFrame, session_year: int) -> pd.DataFrame:
    """Derive the trait columns from latest feature rows (+ universe cols)."""
    p = pd.DataFrame(index=rows.index)
    mcap = pd.to_numeric(rows.get("market_cap"), errors="coerce")
    p["log_mcap"] = np.log10(mcap.where(mcap > 0))
    p["price_log"] = pd.to_numeric(rows.get("price_log"), errors="coerce")
    p["rs_count_2y"] = pd.to_numeric(rows.get("rs_count_2y"), errors="coerce").fillna(0)
    p["dd_52w"] = pd.to_numeric(rows.get("dd_52w"), errors="coerce")
    off_col = next((c for c in ("n_offer_365", "n_offer_180", "n_offer_90") if c in rows.columns), None)
    p["n_offer_365"] = pd.to_numeric(rows[off_col], errors="coerce").fillna(0) if off_col else 0.0
    p["asia"] = rows.get("asia", pd.Series(False, index=rows.index)).astype(float)
    ipo = pd.to_numeric(rows.get("ipo_year"), errors="coerce")
    p["listing_age"] = (session_year - ipo).clip(lower=0, upper=30)
    vol_col = next((c for c in ("vol60", "vol20") if c in rows.columns), None)
    p["vol60"] = pd.to_numeric(rows[vol_col], errors="coerce") if vol_col else np.nan
    p["dvol20_log"] = pd.to_numeric(rows.get("dvol20_log"), errors="coerce")
    return p


def find_twins(
    rows: pd.DataFrame, ref: str = config.REFERENCE_SYMBOL, k: int = config.TWINS_SIZE,
    session_year: int = 2026,
) -> List[dict]:
    """Return the k most similar symbols to ``ref`` (excluding itself)."""
    if ref not in rows.index:
        return []
    prof = profile_frame(rows, session_year)
    z = prof.apply(_robust_z)
    w = pd.Series(TRAITS)
    target = z.loc[ref]
    diff = (z - target).pow(2)
    known = diff.notna() & target.notna()
    num = (diff.fillna(0) * w).where(known, 0).sum(axis=1)
    den = (known * w).sum(axis=1)
    # require most of the profile to be known
    ok = den >= 0.7 * w.sum()
    d = np.sqrt(num / den.replace(0, np.nan))
    sim = np.exp(-0.5 * d.pow(2) / 0.8)  # 1 = identical profile
    sim = sim[ok].drop(index=ref, errors="ignore").sort_values(ascending=False)

    out = []
    ref_p = prof.loc[ref]
    for sym, s in sim.head(k).items():
        p = prof.loc[sym]
        out.append({"symbol": sym, "similarity": round(float(s), 3), "reasons": _shared_traits(p, ref_p)})
    return out


def _shared_traits(p: pd.Series, ref: pd.Series) -> List[str]:
    reasons: List[str] = []
    if p.get("asia") == 1 and ref.get("asia") == 1:
        reasons.append("Asia-linked issuer")
    rs = p.get("rs_count_2y")
    if rs and rs >= 1:
        reasons.append(f"{int(rs)} reverse split{'s' if rs > 1 else ''} in 2 years")
    dd = p.get("dd_52w")
    if dd is not None and not math.isnan(dd) and dd <= -0.7:
        reasons.append(f"{dd * 100:.1f}% from 52-week high" if dd < -0.99 else f"{dd * 100:.0f}% from 52-week high")
    lm = p.get("log_mcap")
    if lm is not None and not math.isnan(lm) and lm < 8:  # < $100M
        reasons.append(f"~${10 ** lm / 1e6:.0f}M market cap")
    no = p.get("n_offer_365")
    if no and no >= 1:
        reasons.append(f"{int(no)} offering filing{'s' if no > 1 else ''} in a year")
    age = p.get("listing_age")
    if age is not None and not math.isnan(age) and age <= 3:
        reasons.append("listed within ~3 years")
    return reasons[:4]
