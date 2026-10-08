"""How much can you short without moving the price?

Capacity = the largest position (in dollars) that stays within BOTH
  • participation: ≤ 5% of the session's expected dollar volume per leg, and
  • impact: ≤ 50 bps one-way estimated with the square-root law
      impact ≈ Y · σ_daily · √(Q / V)        (Y = 0.7)
and, live, within what IBKR lists as lendable (shares available × price).

Expected session volume is deliberately conservative: the 20-day median
dollar volume times the 25th-percentile next-day/20-day ratio observed in
this universe for the same relative-volume bucket (a stock that just traded
30× its normal volume typically trades ~12× the next day; we assume 6.6×).

These are estimates for planning, not execution guarantees: real impact
depends on how the order is worked (auction vs. VWAP), the time of day and
the order book on that day.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# calibrated on 3y of daily bars for the ≤ $2B universe (scripts/research/size_tier.py)
RVOL_EDGES = np.array([0, 1, 1.5, 2, 3, 4, 6, 10, 15, 30])
NEXT_DV_Q25 = np.array([0.54, 0.77, 0.89, 0.99, 1.14, 1.32, 1.65, 2.12, 2.67, 6.57])
Y_IMPACT = 0.7
MAX_PART = 0.05
MAX_IMPACT = 0.0050
SIGMA_FLOOR = 0.005
TIERS = (500_000, 1_000_000)


def exp_dvol(dv20, dv1):
    """Conservative expected dollar volume for the next session."""
    dv20 = np.asarray(dv20, float)
    dv1 = np.asarray(dv1, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.where(dv20 > 0, dv1 / dv20, 0.0)
    rv = np.nan_to_num(rv, nan=0.0)
    k = np.clip(np.searchsorted(RVOL_EDGES, rv, side="right") - 1, 0, len(NEXT_DV_Q25) - 1)
    return np.nan_to_num(dv20, nan=0.0) * NEXT_DV_Q25[k]


def market_capacity(dvx, sigma):
    """Largest order (USD) within the participation and impact limits."""
    dvx = np.asarray(dvx, float)
    sig = np.maximum(np.nan_to_num(np.asarray(sigma, float), nan=0.05), SIGMA_FLOOR)
    q_imp = dvx * (MAX_IMPACT / (Y_IMPACT * sig)) ** 2
    return np.minimum(MAX_PART * dvx, q_imp)


def impact(q, dvx, sigma):
    """Estimated one-way price impact (fraction) of a q-dollar order."""
    sig = np.maximum(np.nan_to_num(np.asarray(sigma, float), nan=0.05), SIGMA_FLOOR)
    return Y_IMPACT * sig * np.sqrt(np.asarray(q, float) / np.maximum(np.asarray(dvx, float), 1.0))


def tick_floor(price) -> float:
    """One tick relative to the price — the narrowest possible spread.
    US equities quote in $0.0001 below $1 and in $0.01 at or above $1."""
    try:
        pr = float(price)
    except (TypeError, ValueError):
        return 0.001
    if not pr > 0:
        return 0.001
    return max(0.0005, (0.0001 if pr < 1 else 0.01) / pr)


def spread_cap(dv20) -> float:
    """Upper bound on the assumed spread by liquidity tier. The close-high-low
    estimator confuses volatility with spread on very volatile names (it reads
    ~5% for hot stocks trading $5M+/day), so it is only trusted inside
    [tick, cap]. Real quotes logged by the live job (data/spreads, from
    2026-10-07) show names under $250K/day quoting 4–9% wide, hence the 6% tier."""
    try:
        v = float(dv20)
    except (TypeError, ValueError):
        return 0.06
    if not v >= 2.5e5:
        return 0.06
    return 0.03 if v < 1e6 else (0.02 if v < 5e6 else 0.01)


def spread_used(estimate, price, dv20) -> float:
    """The spread the cost model assumes: the estimate clipped to
    [one tick, liquidity cap]; the cap itself (conservative) when the
    estimator can't read a spread — it reads ~0 exactly on the most volatile
    names, whose real spreads are wide."""
    lo, hi = tick_floor(price), spread_cap(dv20)
    hi = max(hi, lo)
    try:
        e = float(estimate)
    except (TypeError, ValueError):
        e = float("nan")
    if not (e >= 0.001):
        return hi
    return min(max(e, lo), hi)


def fallback_spread(price) -> float:  # kept for callers without volume data
    return spread_used(None, price, None)


def round_trip_cost(q, dvx, sigma, spread, hold_days: int = 0, borrow_fee_pct: Optional[float] = None):
    """Impact on both legs + one full spread + borrow for ``hold_days``
    trading days (IBKR annual % fee; intraday round trips pay no borrow).
    ``spread`` is used as given (callers pass ``spread_used`` or a real quote);
    only a floor of 0.1% applies."""
    spr = np.maximum(np.nan_to_num(np.asarray(spread, float), nan=0.005), 0.001)
    cost = 2 * impact(q, dvx, sigma) + spr
    if hold_days and borrow_fee_pct is not None:
        cost = cost + (borrow_fee_pct / 100.0) * hold_days / 252.0
    return cost


def borrow_capacity(available: Optional[float], price: Optional[float]) -> Optional[float]:
    """Dollar value IBKR can lend (None if unknown). IBKR reports '>10,000,000'
    as 10,000,000, so this is a floor for very liquid names."""
    if available is None or price is None or not np.isfinite(price):
        return None
    return float(available) * float(price)


def tier_of(cap_usd: Optional[float]) -> Optional[str]:
    if cap_usd is None or not np.isfinite(cap_usd):
        return None
    if cap_usd >= 1_000_000:
        return "$1M+"
    if cap_usd >= 500_000:
        return "$500K+"
    if cap_usd >= 100_000:
        return "$100K+"
    return "under $100K"


def ar_spread(high, low, close, n: int = 20) -> Optional[float]:
    """Abdi–Ranaldo (2017) close-high-low effective-spread estimate over the
    last ``n`` sessions (bars ≤ today only). None when there is too little data."""
    h = np.log(np.asarray(high, float)[-(n + 1):])
    l_ = np.log(np.asarray(low, float)[-(n + 1):])
    c = np.log(np.asarray(close, float)[-(n + 1):])
    if len(c) < 11:
        return None
    eta = (h + l_) / 2
    prod = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    prod = prod[np.isfinite(prod)]
    if len(prod) < 10:
        return None
    return float(2 * np.sqrt(max(prod.mean(), 0.0)))


def breakeven_size(avg_move: float, dvx, sigma, spread) -> float:
    """Largest position whose estimated round-trip cost (2·impact + spread)
    still fits inside ``avg_move`` (a positive fraction, e.g. 0.018).
    0 when the spread alone already eats the move."""
    spr = max(float(np.nan_to_num(np.nan if spread is None else spread, nan=0.005)), 0.001)
    sig = max(float(np.nan_to_num(np.nan if sigma is None else sigma, nan=0.05)), SIGMA_FLOOR)
    room = avg_move - spr
    if room <= 0 or not np.isfinite(dvx) or dvx <= 0:
        return 0.0
    return float(dvx * (room / (2 * Y_IMPACT * sig)) ** 2)


SIZES = (10_000, 50_000, 100_000, 500_000, 1_000_000)


def size_block(price, dv20, dv1, sigma, high, low, close, available, avg_move: Optional[float],
               quoted: Optional[float] = None) -> dict:
    """Everything the site shows about trade size for one name. ``quoted`` =
    the median real bid/ask spread the live job recorded for this name
    recently; when present it replaces the estimate (never below one tick)."""
    dvx = float(exp_dvol(dv20, dv1))
    cap_mkt = float(market_capacity(dvx, sigma))
    cap_b = borrow_capacity(available, price)
    cap = min(cap_mkt, cap_b) if cap_b is not None else cap_mkt
    est = ar_spread(high, low, close)
    estimated = est is not None and est >= 0.001
    spr = spread_used(est, price, dv20)
    source = "estimate" if estimated else "band cap"
    if quoted is not None and np.isfinite(quoted) and quoted > 0:
        spr = max(float(quoted), tick_floor(price))
        source = "quoted"
    limit = "borrow" if (cap_b is not None and cap_b < cap_mkt) else (
        "volume" if MAX_PART * dvx <= cap_mkt + 1e-9 else "impact")
    costs = {str(q): float(round_trip_cost(q, dvx, sigma, spr)) for q in SIZES}
    part = {str(q): (q / dvx if dvx > 0 else None) for q in SIZES}
    be = breakeven_size(avg_move, dvx, sigma, spr) if avg_move else None
    return {"capacity": cap, "cap_market": cap_mkt, "cap_borrow": cap_b, "limit": limit, "exp_dvol": dvx,
            "spread": spr, "spread_estimated": estimated, "spread_raw": est, "spread_source": source,
            "spread_band": [tick_floor(price), max(spread_cap(dv20), tick_floor(price))],
            "tier": tier_of(cap), "costs": costs, "participation": part,
            "breakeven": be, "avg_move": avg_move}


def spread_used_vec(estimate, price, dv20) -> np.ndarray:
    """Vectorised ``spread_used`` for backtests (same rule, row by row)."""
    est = np.asarray(estimate, float)
    pr = np.asarray(price, float)
    dv = np.asarray(dv20, float)
    tick = np.where(pr < 1, 0.0001, 0.01)
    lo = np.maximum(0.0005, np.where(pr > 0, tick / np.where(pr > 0, pr, 1.0), 0.001))
    hi = np.where(dv >= 5e6, 0.01, np.where(dv >= 1e6, 0.02, np.where(dv >= 2.5e5, 0.03, 0.06)))
    hi = np.maximum(hi, lo)
    ok = np.isfinite(est) & (est >= 0.001)
    return np.where(ok, np.clip(est, lo, hi), hi)
