"""Point-in-time feature panel + next-session labels (CONTRACTS §7).

One row per (``date``, ``symbol``). Features describe the stock **as of the
close of ``date``**; labels describe the **next** session (``t+1``, the next
row of that symbol's history). The same code builds the training panel and
the live feature vector, so there is no train/serve skew.

Lookahead discipline — the #1 risk in this module:

* a feature for row ``t`` may only use bars with index ``<= t`` and
  FilingEvents whose ``date <= t`` (an 8-K filed after the close of ``t``
  is visible at ``t``: the prediction is made that evening / next morning);
* labels use bars ``t+1 .. t+5`` only, and only when ``t+1`` is the next
  market session and a clean, traded bar;
* split-adjusted price *levels* silently embed knowledge of later splits
  (a stock that trades at $0.10 and later does a 1:20 reverse split shows
  $2.00 in adjusted history). Returns and dollar volume are unaffected, but
  the price level is not — so the as-traded price is rebuilt from the
  splits dated **after** ``t`` before it is used as a feature (``price_log``)
  or a filter (``MIN_PRICE``). This restores what was actually visible at
  ``t``; it does not add information.

Everything is vectorised over one long, (symbol, date)-sorted array: rolling
windows are computed on the concatenated series and invalidated where they
would cross a symbol boundary, so ~3,500 symbols x ~750 sessions build in
well under a minute.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# ── The contract: stable, explicit column lists ──────────────────────────
FAMILIES: Tuple[str, ...] = ("exhaustion", "decay", "dilution", "structure", "flow", "market")

FEATURES: List[str] = [
    # exhaustion — the stock just ran; how stretched is it?
    "r1", "r3", "r5", "r10", "r20",
    "gap", "intraday", "hi_ret", "range1", "clv", "upper_wick",
    "rsi14", "rsi2", "dist_ma20", "runup_52w",
    "max_r1_20", "sess_since_spike", "n_spike_60", "up_streak",
    "r1_rank", "r5_rank",
    # decay — long, grinding decline
    "r60", "r120", "dd_52w", "dist_ma50", "dist_ma200", "down_frac_60",
    # dilution — reverse splits and the paper trail of share issuance
    "rs_count_1y", "rs_count_2y", "sess_since_rs", "rs_last_factor", "rs_cum_log_2y",
    "n_offer_30", "n_offer_90", "n_offer_180", "n_offer_365", "sess_since_offer",
    "n_reg_30", "n_reg_90", "n_reg_180", "n_reg_365", "sess_since_reg",
    "n_unreg_90", "n_unreg_365",
    "n_delist_90", "n_delist_365", "sess_since_delist",
    "n_charter_365", "n_finance_90", "n_effect_90", "n_late_365", "n_f144_90",
    "n_current_30", "n_current_90", "n_filings_30",
    # structure — what kind of security this is
    "price_log", "vol20", "vol60", "range14", "asia", "ipo_age",
    # flow — volume and liquidity
    "rvol1", "rvol5", "dvol1_log", "dvol20_log", "dvol_trend", "halt_20", "rvol_rank",
    # market — the tape everyone is trading in
    "iwm_r1", "iwm_r5", "iwm_r20", "breadth_up", "breadth_ma50", "univ_r1_med", "spike_share",
]

# Only known at/after the 9:30 open of t+1 (live proxy = pre-market price).
M1_EXTRA: List[str] = ["gap_open"]

LABELS: List[str] = ["y_oc", "y_co", "y_gap", "y_ol", "y_oh", "y_c5", "y_dump", "y_bigdump", "y_squeeze"]

# Non-feature columns carried on every row (bars of t, the as-traded price,
# raw dollar volume, static attributes, and the live-row marker).
INFO_COLUMNS: List[str] = [
    "date", "symbol", "open", "high", "low", "close", "volume",
    "price", "dvol20", "ipo_year", "is_last_bar",
]

# name → (family, plain-English description). The site shows these.
FEATURE_DOCS: Dict[str, Tuple[str, str]] = {
    "r1": ("exhaustion", "Today's close-to-close return."),
    "r3": ("exhaustion", "Return over the last 3 sessions."),
    "r5": ("exhaustion", "Return over the last 5 sessions."),
    "r10": ("exhaustion", "Return over the last 10 sessions."),
    "r20": ("exhaustion", "Return over the last 20 sessions (about a month)."),
    "gap": ("exhaustion", "Today's opening gap versus the prior close."),
    "intraday": ("exhaustion", "Today's open-to-close move."),
    "hi_ret": ("exhaustion", "How far today's high got above the prior close."),
    "range1": ("exhaustion", "Today's high-low range as a share of the prior close."),
    "clv": ("exhaustion", "Where today closed inside its range (0 = at the low, 1 = at the high)."),
    "upper_wick": ("exhaustion", "Share of today's range above the candle body (selling off the high)."),
    "rsi14": ("exhaustion", "14-session RSI (Wilder); above 70-80 is conventionally 'overbought'."),
    "rsi2": ("exhaustion", "2-session RSI — very short-term overextension."),
    "dist_ma20": ("exhaustion", "Close versus its 20-session average."),
    "runup_52w": ("exhaustion", "Close versus the lowest low of the last 52 weeks (run-up off the bottom)."),
    "max_r1_20": ("exhaustion", "Largest single-day gain in the last 20 sessions."),
    "sess_since_spike": ("exhaustion", "Sessions since the last +50% day (capped at 252)."),
    "n_spike_60": ("exhaustion", "Number of +20% days in the last 60 sessions."),
    "up_streak": ("exhaustion", "Consecutive higher closes going into today (capped at 10)."),
    "r1_rank": ("exhaustion", "Today's return ranked against every eligible stock today (1 = biggest gainer)."),
    "r5_rank": ("exhaustion", "5-session return ranked against every eligible stock today."),
    "r60": ("decay", "Return over the last 60 sessions (about a quarter)."),
    "r120": ("decay", "Return over the last 120 sessions (about six months)."),
    "dd_52w": ("decay", "Close versus the highest high of the last 52 weeks (drawdown)."),
    "dist_ma50": ("decay", "Close versus its 50-session average."),
    "dist_ma200": ("decay", "Close versus its 200-session average (partial window for young listings)."),
    "down_frac_60": ("decay", "Share of the last 60 sessions that closed down."),
    "rs_count_1y": ("dilution", "Reverse splits in the last 365 days."),
    "rs_count_2y": ("dilution", "Reverse splits in the last 730 days."),
    "sess_since_rs": ("dilution", "Sessions since the most recent reverse split (blank = none on record)."),
    "rs_last_factor": ("dilution", "Size of the most recent reverse split (20 = 1-for-20)."),
    "rs_cum_log_2y": ("dilution", "Cumulative reverse-split consolidation over 2 years, log10 (1.0 = 10x fewer shares)."),
    "n_offer_30": ("dilution", "Offering prospectuses (424B1/2/4/5/7, ATM, MEF) filed in the last 30 days."),
    "n_offer_90": ("dilution", "Offering prospectuses filed in the last 90 days."),
    "n_offer_180": ("dilution", "Offering prospectuses filed in the last 180 days."),
    "n_offer_365": ("dilution", "Offering prospectuses filed in the last 365 days."),
    "sess_since_offer": ("dilution", "Sessions since the most recent offering prospectus (blank = none on record)."),
    "n_reg_30": ("dilution", "Registration statements (S-1/F-1/S-3/F-3 and amendments, resale 424B3) in the last 30 days."),
    "n_reg_90": ("dilution", "Registration statements in the last 90 days."),
    "n_reg_180": ("dilution", "Registration statements in the last 180 days."),
    "n_reg_365": ("dilution", "Registration statements in the last 365 days."),
    "sess_since_reg": ("dilution", "Sessions since the most recent registration statement (blank = none on record)."),
    "n_unreg_90": ("dilution", "8-K item 3.02 unregistered share sales in the last 90 days."),
    "n_unreg_365": ("dilution", "8-K item 3.02 unregistered share sales in the last 365 days."),
    "n_delist_90": ("dilution", "8-K item 3.01 listing-deficiency / delisting notices in the last 90 days."),
    "n_delist_365": ("dilution", "8-K item 3.01 listing-deficiency / delisting notices in the last 365 days."),
    "sess_since_delist": ("dilution", "Sessions since the most recent listing-deficiency notice (blank = none on record)."),
    "n_charter_365": ("dilution", "8-K item 5.03 charter amendments (often reverse-split mechanics) in the last 365 days."),
    "n_finance_90": ("dilution", "8-K item 1.01 material agreements (often financings) in the last 90 days."),
    "n_effect_90": ("dilution", "Registration statements declared effective (EFFECT) in the last 90 days."),
    "n_late_365": ("dilution", "Late-filing notices (NT 10-K/10-Q/20-F) in the last 365 days."),
    "n_f144_90": ("dilution", "Form 144 insider/affiliate sale notices in the last 90 days."),
    "n_current_30": ("dilution", "Current reports (8-K/6-K) in the last 30 days."),
    "n_current_90": ("dilution", "Current reports (8-K/6-K) in the last 90 days."),
    "n_filings_30": ("dilution", "All SEC filings in the last 30 days."),
    "price_log": ("structure", "log10 of the as-traded share price (not split-adjusted)."),
    "vol20": ("structure", "Daily volatility of log returns over 20 sessions."),
    "vol60": ("structure", "Daily volatility of log returns over 60 sessions."),
    "range14": ("structure", "Average daily high-low range over 14 sessions, as a share of the close."),
    "asia": ("structure", "Issuer is Asia-linked (country per the exchange listing / SEC address)."),
    "ipo_age": ("structure", "Years since the IPO year (blank = unknown)."),
    "rvol1": ("flow", "Today's volume versus the average of the prior 20 sessions."),
    "rvol5": ("flow", "Average volume of the last 5 sessions versus the last 60."),
    "dvol1_log": ("flow", "log10 of today's dollar volume."),
    "dvol20_log": ("flow", "log10 of the median daily dollar volume over 20 sessions."),
    "dvol_trend": ("flow", "log10 of 5-session versus 60-session average dollar volume."),
    "halt_20": ("flow", "Sessions with zero volume (halted / no trades) in the last 20."),
    "rvol_rank": ("flow", "Today's relative volume ranked against every eligible stock today."),
    "iwm_r1": ("market", "Russell 2000 ETF (IWM) return today."),
    "iwm_r5": ("market", "IWM return over the last 5 sessions."),
    "iwm_r20": ("market", "IWM return over the last 20 sessions."),
    "breadth_up": ("market", "Share of eligible small caps that closed up today."),
    "breadth_ma50": ("market", "Share of eligible small caps above their 50-session average."),
    "univ_r1_med": ("market", "Median return of eligible small caps today."),
    "spike_share": ("market", "Share of eligible small caps up 20%+ today (speculative froth)."),
    "gap_open": ("exhaustion", "Next session's opening gap versus today's close (known at 9:30; live proxy = pre-market price)."),
}

assert set(FEATURE_DOCS) == set(FEATURES) | set(M1_EXTRA), "FEATURE_DOCS out of sync"
assert all(f in FAMILIES for f, _ in FEATURE_DOCS.values())
assert len(set(FEATURES)) == len(FEATURES)

# ── Tunables (not in config: only this module needs them) ────────────────
MIN_PRIOR_BARS = 60              # rows need ≥ 60 bars before t
SPIKE_R1 = 0.50                  # "spike" day for sess_since_spike
BIG_UP_R1 = 0.20                 # +20% day for n_spike_60 / spike_share
SPIKE_CAP = 252
_KEY_STRIDE = 1 << 22            # symbol-code stride for (symbol, day) int keys
_CAL_MIN_SHARE = 0.05            # a date is a market session if ≥ 5% of symbols trade it
_BAD_GAP = np.log(20.0)          # |log gap| beyond 20x = bad print, not a move
_SPLIT_TOL = np.log(1.35)        # gap within ±35% of a split factor = unadjusted split
_NAT_DAY = np.iinfo(np.int64).min

# Filing groups are derived from ``form`` and 8-K ``items`` ONLY — never from
# ``category``/``text_tags``. The 3-year history is form/item based
# (CONTRACTS §3), while live shortlist events get text-matched categories
# (e.g. an 8-K re-labelled "offering"). Counting those would give live rows
# feature values the training rows could never have had (train/serve skew);
# text-matched evidence belongs to the live overlay in score.py instead.
_OFFER_FORMS = {"424B1", "424B2", "424B4", "424B5", "424B7", "S-1MEF", "F-1MEF"}
_REG_FORMS = {
    "S-1", "S-1/A", "F-1", "F-1/A", "S-3", "S-3/A", "F-3", "F-3/A",
    "S-3ASR", "F-3ASR", "424B3",
}
EVENT_GROUP_WINDOWS: Dict[str, Tuple[int, ...]] = {
    "offer": (30, 90, 180, 365),
    "reg": (30, 90, 180, 365),
    "unreg": (90, 365),
    "delist": (90, 365),
    "charter": (365,),
    "finance": (90,),
    "effect": (90,),
    "late": (365,),
    "f144": (90,),
    "current": (30, 90),
    "filings": (30,),
}
_SINCE_GROUPS = ("offer", "reg", "delist")


def _items(x: Any) -> Set[str]:
    if not x:
        return set()
    if isinstance(x, str):
        return {p.strip() for p in x.split(",") if p.strip()}
    return {str(p).strip() for p in x}


def event_groups(ev: dict) -> Set[str]:
    """Feature buckets a FilingEvent counts toward (see ``EVENT_GROUP_WINDOWS``).

    Uses ``form`` and ``items`` only (see the note on ``_OFFER_FORMS``), so a
    given filing lands in the same buckets whether it came from the 3-year
    history or from tonight's text-matched feed.
    """
    form = str(ev.get("form") or "").strip().upper()
    items = _items(ev.get("items"))
    base = form.split("/")[0]
    current = base in ("8-K", "6-K")
    g = {"filings"}
    if form in _OFFER_FORMS:
        g.add("offer")
    if form in _REG_FORMS:
        g.add("reg")
    if form == "EFFECT":
        g.add("effect")
    if current and "3.02" in items:
        g.add("unreg")
    if current and "3.01" in items:
        g.add("delist")
    if current and "5.03" in items:
        g.add("charter")
    if current and "1.01" in items:
        g.add("finance")
    if form.startswith("NT "):
        g.add("late")
    if base == "144":
        g.add("f144")
    if current:
        g.add("current")
    return g


# ── Grouped operations over one long (symbol, date)-sorted array ─────────
class _Blocks:
    """Index helper for contiguous per-symbol blocks in a long array.

    Every op returns a full-length float64 array and never lets a window,
    lag or lead read across a symbol boundary.
    """

    def __init__(self, codes: np.ndarray):
        n = len(codes)
        self.n = n
        self.codes = codes
        self.new = np.ones(n, dtype=bool)
        if n:
            self.new[1:] = codes[1:] != codes[:-1]
        starts = np.flatnonzero(self.new)
        lens = np.diff(np.r_[starts, n])
        self.start_of = np.repeat(starts, lens)
        self.end_of = np.repeat(starts + lens - 1, lens)
        self.pos = np.arange(n) - self.start_of
        self.idx = np.arange(n)

    def lag(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        out = np.full(self.n, np.nan)
        if k < self.n:
            out[k:] = x[:-k]
        out[self.pos < k] = np.nan
        return out

    def lead(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        out = np.full(self.n, np.nan)
        if k < self.n:
            out[:-k] = x[k:]
        out[self.idx + k > self.end_of] = np.nan
        return out

    def roll(self, x: np.ndarray, w: int, how: str, minp: Optional[int] = None) -> np.ndarray:
        """Trailing window of ``w`` rows ending at t (inclusive). With
        ``minp < w`` young symbols get a partial (expanding) window."""
        minp = w if minp is None else minp
        r = getattr(pd.Series(x).rolling(w, min_periods=minp), how)().to_numpy(dtype=float)
        early = self.pos < w - 1
        if early.any():
            if minp >= w:
                r[early] = np.nan
            else:
                r[early] = self._expanding(x, how)[early]
                r[self.pos < minp - 1] = np.nan
        return r

    def _expanding(self, x: np.ndarray, how: str) -> np.ndarray:
        s = pd.Series(x)
        g = s.groupby(self.codes, sort=False)
        if how == "max":
            return g.cummax().to_numpy(dtype=float)
        if how == "min":
            return g.cummin().to_numpy(dtype=float)
        if how in ("mean", "sum"):
            filled = s.fillna(0.0)
            cs = filled.groupby(self.codes, sort=False).cumsum().to_numpy(dtype=float)
            if how == "sum":
                return cs
            cnt = s.notna().astype(float).groupby(self.codes, sort=False).cumsum().to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                return cs / cnt
        raise ValueError(f"no expanding fallback for {how!r}")

    def ewm(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Per-symbol ``ewm(alpha, adjust=False).mean()`` without a Python loop.

        Run the recursion over the whole array, then remove the exact carry-in
        from the previous symbol: y_t = y'_t − (1−α)^(pos+1) · (y'_{s−1} − x_s).
        ``x`` must be NaN-free.
        """
        yp = pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)
        s = self.start_of
        prev = np.where(s > 0, yp[np.maximum(s - 1, 0)], x[s])
        carry = np.where(s > 0, prev - x[s], 0.0)
        return yp - np.power(1.0 - alpha, self.pos + 1.0) * carry

    def ffill_index(self, mask: np.ndarray) -> np.ndarray:
        """Global index of the most recent row (≤ t, same symbol) where mask is True; NaN if none."""
        last = np.where(mask, self.idx, -1)
        last = np.maximum.accumulate(last) if self.n else last
        out = last.astype(float)
        out[last < self.start_of] = np.nan
        return out


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _log10_pos(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(np.maximum(x, 1.0))


# ── Input normalisation ──────────────────────────────────────────────────
def _norm_bars(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    for c in ("open", "high", "low", "close", "volume"):
        if c not in d.columns:
            d[c] = np.nan
    d = d[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").astype(float)
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    px = d[["open", "high", "low", "close"]]
    d = d[~(px <= 0).any(axis=1)]          # non-positive prints are errors, not halts
    return d if len(d) else None


def _to_days(values: Iterable[Any]) -> np.ndarray:
    """'YYYY-MM-DD' / date / Timestamp → int days since epoch (NaT → _NAT_DAY)."""
    txt = pd.Series([str(v)[:10] if v is not None else "" for v in values], dtype=object)
    ts = pd.to_datetime(txt, format="%Y-%m-%d", errors="coerce")
    return ts.to_numpy(dtype="datetime64[D]").astype("int64")


def _static_maps(static: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """symbol → asia (1/0), symbol → ipo_year. Unknown values are left out (→ NaN)."""
    if static is None or len(static) == 0:
        return {}, {}
    st = static
    if "symbol" in st.columns:
        st = st.set_index("symbol")
    st = st[~st.index.duplicated(keep="first")]
    asia: Dict[str, float] = {}
    ipo: Dict[str, float] = {}
    if "asia" in st.columns:
        for s, v in st["asia"].items():
            if isinstance(v, (bool, np.bool_)) or (isinstance(v, (int, float, np.number)) and v in (0, 1)):
                asia[str(s)] = float(bool(v))
    if "ipo_year" in st.columns:
        for s, v in pd.to_numeric(st["ipo_year"], errors="coerce").items():
            if np.isfinite(v) and 1800 < v < 2100:
                ipo[str(s)] = float(v)
    return asia, ipo


# ── Events / splits, keyed by (symbol code, day) ─────────────────────────
def _event_features(
    row_key: np.ndarray, row_code: np.ndarray, has_events: np.ndarray,
    events: Dict[str, List[dict]], code_of: Dict[str, int],
) -> Dict[str, np.ndarray]:
    n = len(row_key)
    keys: Dict[str, List[int]] = {g: [] for g in EVENT_GROUP_WINDOWS}
    for sym, evs in events.items():
        code = code_of.get(sym)
        if code is None or not evs:
            continue
        days = _to_days(e.get("date") for e in evs)
        for e, day in zip(evs, days):
            if day == _NAT_DAY:
                continue
            for g in event_groups(e):
                if g in keys:
                    keys[g].append(code * _KEY_STRIDE + int(day))
    out: Dict[str, np.ndarray] = {}
    idx = np.arange(n)
    for g, windows in EVENT_GROUP_WINDOWS.items():
        ek = np.sort(np.asarray(keys[g], dtype=np.int64))
        hi = np.searchsorted(ek, row_key, side="right")          # events dated ≤ t
        for w in windows:
            lo = np.searchsorted(ek, row_key - w, side="right")  # events dated ≤ t−w
            out[f"n_{g}_{w}"] = np.where(has_events, (hi - lo).astype(float), np.nan)
        if g in _SINCE_GROUPS:
            j = hi - 1
            ok = has_events & (j >= 0)
            jj = np.clip(j, 0, max(len(ek) - 1, 0))
            if len(ek):
                ok &= (ek[jj] // _KEY_STRIDE) == row_code
                first_row = np.searchsorted(row_key, ek[jj], side="left")
                sess = (idx - first_row).astype(float)
                # An event dated before the symbol's first bar (S-1 months
                # before an IPO, or just before the history window starts)
                # lands on the first row; add the weekdays in between so the
                # count isn't understated as if it were filed on day one.
                fr = np.clip(first_row, 0, max(n - 1, 0))
                at_start = (fr == 0) | (row_code[np.maximum(fr - 1, 0)] != row_code[fr])
                ev_day = ek[jj] % _KEY_STRIDE
                first_day = row_key[fr] % _KEY_STRIDE
                pre = ok & at_start & (ev_day < first_day)
                if pre.any():
                    sess[pre] += np.busday_count(ev_day[pre].astype("datetime64[D]"),
                                                 first_day[pre].astype("datetime64[D]"))
            else:
                sess = np.full(n, np.nan)
            out[f"sess_since_{g}"] = np.where(ok, sess, np.nan)
    return out


def _split_arrays(splits: Dict[str, List[dict]], code_of: Dict[str, int], unadjusted: Set[str]):
    k_all, lr_all, k_rev, f_rev = [], [], [], []
    for sym, lst in (splits or {}).items():
        code = code_of.get(sym)
        if code is None or not lst:
            continue
        days = _to_days(s.get("date") for s in lst)
        for s, day in zip(lst, days):
            try:
                ratio = float(s.get("ratio"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ratio) or ratio <= 0 or ratio == 1 or day == _NAT_DAY:
                continue
            key = code * _KEY_STRIDE + int(day)
            if sym not in unadjusted:
                k_all.append(key)
                lr_all.append(np.log(ratio))
            if ratio < 1:
                k_rev.append(key)
                f_rev.append(1.0 / ratio)
    o = np.argsort(np.asarray(k_all, dtype=np.int64), kind="stable")
    ra = (np.asarray(k_all, dtype=np.int64)[o], np.asarray(lr_all, dtype=float)[o])
    o = np.argsort(np.asarray(k_rev, dtype=np.int64), kind="stable")
    rr = (np.asarray(k_rev, dtype=np.int64)[o], np.asarray(f_rev, dtype=float)[o])
    return ra, rr


def _split_features(row_key: np.ndarray, row_code: np.ndarray, rev: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
    ek, fac = rev
    n = len(row_key)
    hi = np.searchsorted(ek, row_key, side="right")
    lo1 = np.searchsorted(ek, row_key - 365, side="right")
    lo2 = np.searchsorted(ek, row_key - 730, side="right")
    out = {"rs_count_1y": (hi - lo1).astype(float), "rs_count_2y": (hi - lo2).astype(float)}
    cs = np.r_[0.0, np.cumsum(np.log10(fac))] if len(fac) else np.zeros(1)
    out["rs_cum_log_2y"] = cs[hi] - cs[lo2]
    j = hi - 1
    ok = j >= 0
    if len(ek):
        jj = np.clip(j, 0, len(ek) - 1)
        ok &= (ek[jj] // _KEY_STRIDE) == row_code
        first_row = np.searchsorted(row_key, ek[jj], side="left")
        out["sess_since_rs"] = np.where(ok, (np.arange(n) - first_row).astype(float), np.nan)
        out["rs_last_factor"] = np.where(ok, fac[jj], np.nan)
    else:
        out["sess_since_rs"] = np.full(n, np.nan)
        out["rs_last_factor"] = np.full(n, np.nan)
    return out


def _future_split_factor(row_key: np.ndarray, row_code: np.ndarray, allsp: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Product of split ratios dated strictly after t (same symbol).
    as-traded price_t = split-adjusted close_t × this factor."""
    ek, lr = allsp
    if not len(ek):
        return np.ones(len(row_key))
    cs = np.r_[0.0, np.cumsum(lr)]
    hi = np.searchsorted(ek, row_key, side="right")
    end = np.searchsorted(ek, (row_code.astype(np.int64) + 1) * _KEY_STRIDE, side="left")
    return np.exp(cs[end] - cs[hi])


def _split_label_guard(
    row_key: np.ndarray, B: _Blocks, gap_log: np.ndarray, co_log: np.ndarray,
    allsp_any: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Rows whose next-session move looks like an unadjusted split (±1 session
    around a split date and within ±35% of the split factor) → True (bad)."""
    ek, lr = allsp_any
    bad = np.zeros(B.n, dtype=bool)
    if not len(ek):
        return bad
    big = np.abs(lr) > np.log(1.5)
    ek, lr = ek[big], lr[big]
    if not len(ek):
        return bad
    first_row = np.searchsorted(row_key, ek, side="left")      # session the split took effect
    first_row = np.clip(first_row, 0, B.n - 1)
    same = (row_key[first_row] // _KEY_STRIDE) == (ek // _KEY_STRIDE)
    for off in (-2, -1, 0):                                     # label rows t with t+1 ∈ {s−1, s, s+1}
        t = first_row + off
        valid = same & (t >= 0) & (t < B.n)
        t = np.clip(t, 0, B.n - 1)
        valid &= B.codes[t] == B.codes[first_row]
        jump = -lr                                              # log(1/ratio): the unadjusted jump
        hit = (np.abs(gap_log[t] - jump) < _SPLIT_TOL) | (np.abs(co_log[t] - jump) < _SPLIT_TOL)
        bad[t[valid & hit]] = True
    return bad


def _market_calendar(dates: np.ndarray, bench: Optional[pd.DataFrame]) -> np.ndarray:
    u, cnt = np.unique(dates, return_counts=True)
    cal = u[cnt >= max(1.0, _CAL_MIN_SHARE * (cnt.max() if len(cnt) else 0))]
    if bench is not None and len(bench):
        b = _norm_bars(bench)
        if b is not None:
            cal = np.union1d(cal, b.index.to_numpy(dtype="datetime64[ns]"))
    return np.sort(cal)


def _bench_returns(bench: Optional[pd.DataFrame], dates: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(dates)
    out = {k: np.full(n, np.nan) for k in ("iwm_r1", "iwm_r5", "iwm_r20")}
    b = _norm_bars(bench) if bench is not None else None
    if b is None:
        return out
    c = b["close"].dropna()
    bd = c.index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(bd, dates)
    hit = (pos < len(bd)) & (bd[np.clip(pos, 0, len(bd) - 1)] == dates)
    for k, name in ((1, "iwm_r1"), (5, "iwm_r5"), (20, "iwm_r20")):
        r = (c / c.shift(k) - 1.0).to_numpy(dtype=float)
        out[name] = np.where(hit, r[np.clip(pos, 0, len(bd) - 1)], np.nan)
    return out


# ── Public API ───────────────────────────────────────────────────────────
def build_panel(
    hist: Dict[str, pd.DataFrame],
    events: Dict[str, List[dict]],
    splits: Dict[str, List[dict]],
    static: pd.DataFrame,
    bench: Optional[pd.DataFrame] = None,
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build the (date, symbol) panel of FEATURES + M1_EXTRA + LABELS.

    ``events``: symbol → FilingEvents. A symbol **absent** from ``events``
    gets NaN filing features (unknown, not zero); present with ``[]`` → zeros.
    ``splits``: symbol → [{date, ratio}] (absent = no splits on record).
    ``static``: indexed by symbol or with a ``symbol`` column; ``asia`` and
    ``ipo_year`` are used. ``bench``: IWM bars (optional → NaN iwm_*).
    ``min_date``: drop output rows before this date (history before it is
    still used as look-back).
    """
    t0 = time.time()
    events = events or {}
    splits = splits or {}
    if not events:
        log.warning("build_panel: no filing events supplied — all filing features will be NaN")

    # 1) long arrays, (symbol, date) sorted
    syms: List[str] = []
    frames: List[pd.DataFrame] = []
    unadjusted: Set[str] = set()
    for s in sorted(hist):
        d = _norm_bars(hist[s])
        if d is None:
            continue
        if getattr(hist[s], "attrs", {}).get("unadjusted"):
            unadjusted.add(s)
        syms.append(s)
        frames.append(d)
    if not frames:
        return _empty_panel()
    code_of = {s: i for i, s in enumerate(syms)}
    lens = np.array([len(f) for f in frames])
    codes = np.repeat(np.arange(len(syms), dtype=np.int64), lens)
    dates = np.concatenate([f.index.to_numpy(dtype="datetime64[ns]") for f in frames])
    raw = {c: np.concatenate([f[c].to_numpy(dtype=float) for f in frames]) for c in ("open", "high", "low", "close", "volume")}
    del frames

    # 2) halts / missing prints → flat bar at the last traded close, volume 0
    o, h, l, c = raw["open"], raw["high"], raw["low"], raw["close"]
    vol = np.nan_to_num(raw["volume"], nan=0.0)
    vol[vol < 0] = 0.0
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    B0 = _Blocks(codes)
    last_ok = B0.ffill_index(valid)
    keep = np.isfinite(last_ok)                   # drop leading rows with no print yet
    cf = np.where(keep, c[np.nan_to_num(last_ok, nan=0).astype(np.int64)], np.nan)
    o = np.where(valid, o, cf)
    h = np.where(valid, h, cf)
    l = np.where(valid, l, cf)
    c = np.where(valid, c, cf)
    vol = np.where(valid, vol, 0.0)
    if not keep.all():
        codes, dates, o, h, l, c, vol, valid = (a[keep] for a in (codes, dates, o, h, l, c, vol, valid))
        raw = {k: v[keep] for k, v in raw.items()}
    B = _Blocks(codes)
    n = B.n
    # bar used for features: make it internally consistent
    hf = np.maximum.reduce([h, o, c])
    lf = np.minimum.reduce([l, o, c])

    days = dates.astype("datetime64[D]").astype(np.int64)
    row_key = codes * _KEY_STRIDE + days

    f: Dict[str, np.ndarray] = {}
    # 3) price-action features (bars ≤ t only)
    cp = B.lag(c, 1)
    r1 = _safe_div(c, cp) - 1.0
    f["r1"] = r1
    for k in (3, 5, 10, 20, 60, 120):
        f[f"r{k}"] = _safe_div(c, B.lag(c, k)) - 1.0
    f["gap"] = _safe_div(o, cp) - 1.0
    f["intraday"] = _safe_div(c, o) - 1.0
    f["hi_ret"] = _safe_div(hf, cp) - 1.0
    f["range1"] = _safe_div(hf - lf, cp)
    rng = hf - lf
    with np.errstate(invalid="ignore", divide="ignore"):
        f["clv"] = np.where(rng > 0, (c - lf) / rng, 0.5)
        f["upper_wick"] = np.where(rng > 0, (hf - np.maximum(o, c)) / rng, 0.0)

    d1 = c - np.where(np.isfinite(cp), cp, c)
    for w, name in ((14, "rsi14"), (2, "rsi2")):
        ag = B.ewm(np.maximum(d1, 0.0), 1.0 / w)
        al = B.ewm(np.maximum(-d1, 0.0), 1.0 / w)
        with np.errstate(invalid="ignore", divide="ignore"):
            rsi = np.where(al > 0, 100.0 - 100.0 / (1.0 + ag / al), np.where(ag > 0, 100.0, 50.0))
        rsi[B.pos < w] = np.nan
        f[name] = rsi

    ma20 = B.roll(c, 20, "mean")
    ma50 = B.roll(c, 50, "mean")
    ma200 = B.roll(c, 200, "mean", MIN_PRIOR_BARS)
    f["dist_ma20"] = _safe_div(c, ma20) - 1.0
    f["dist_ma50"] = _safe_div(c, ma50) - 1.0
    f["dist_ma200"] = _safe_div(c, ma200) - 1.0
    f["dd_52w"] = _safe_div(c, B.roll(hf, 252, "max", MIN_PRIOR_BARS)) - 1.0
    f["runup_52w"] = _safe_div(c, B.roll(lf, 252, "min", MIN_PRIOR_BARS)) - 1.0

    f["max_r1_20"] = B.roll(r1, 20, "max")
    f["max_r1_20"][B.pos < 20] = np.nan
    last_spike = B.ffill_index(np.nan_to_num(r1, nan=0.0) >= SPIKE_R1)
    f["sess_since_spike"] = np.minimum(np.where(np.isfinite(last_spike), B.idx - last_spike, SPIKE_CAP), SPIKE_CAP).astype(float)
    big_up = (np.nan_to_num(r1, nan=0.0) >= BIG_UP_R1).astype(float)
    f["n_spike_60"] = B.roll(big_up, 60, "sum")
    up = np.nan_to_num(r1, nan=0.0) > 0
    reset = ~up | B.new
    run_start = np.maximum.accumulate(np.where(reset, B.idx, 0)) if n else B.idx
    f["up_streak"] = np.minimum(B.idx - run_start, 10).astype(float)
    down = np.where(np.isfinite(r1), (r1 < 0).astype(float), np.nan)
    f["down_frac_60"] = B.roll(down, 60, "mean", 59)
    f["down_frac_60"][B.pos < 60] = np.nan

    # 4) structure / flow
    lr = np.log(_safe_div(c, cp))
    f["vol20"] = B.roll(lr, 20, "std")
    f["vol60"] = B.roll(lr, 60, "std")
    f["vol20"][B.pos < 20] = np.nan
    f["vol60"][B.pos < 60] = np.nan
    f["range14"] = B.roll(_safe_div(hf - lf, c), 14, "mean")

    vmean20 = B.roll(vol, 20, "mean")
    f["rvol1"] = _safe_div(vol, B.lag(vmean20, 1))
    f["rvol5"] = _safe_div(B.roll(vol, 5, "mean"), B.roll(vol, 60, "mean"))
    dv = c * vol                                     # adjusted close × adjusted volume = as-traded $
    dvol20 = B.roll(dv, 20, "median")
    f["dvol1_log"] = _log10_pos(dv)
    f["dvol20_log"] = _log10_pos(dvol20)
    f["dvol_trend"] = np.log10(_safe_div(B.roll(dv, 5, "mean") + 1.0, B.roll(dv, 60, "mean") + 1.0))
    f["halt_20"] = B.roll((vol <= 0).astype(float), 20, "sum")

    # 5) splits: as-traded price + reverse-split history
    allsp, rev = _split_arrays(splits, code_of, unadjusted)
    price = c * _future_split_factor(row_key, codes, allsp)
    f["price_log"] = np.log10(price)
    f.update(_split_features(row_key, codes, rev))

    # 6) filings
    has_events = np.isin(codes, [code_of[s] for s in events if s in code_of])
    f.update(_event_features(row_key, codes, has_events, events, code_of))

    # 7) static
    asia_map, ipo_map = _static_maps(static)
    sym_arr = np.asarray(syms, dtype=object)
    asia_by_code = np.array([asia_map.get(s, np.nan) for s in syms], dtype=float)
    ipo_by_code = np.array([ipo_map.get(s, np.nan) for s in syms], dtype=float)
    f["asia"] = asia_by_code[codes]
    ipo_year = ipo_by_code[codes]
    year = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    age = year - ipo_year
    f["ipo_age"] = np.where(age >= 0, age, np.nan)

    # 8) market (bench) — cross-sectional ones come after the row filter
    f.update(_bench_returns(bench, dates))

    # 9) labels: session t+1 = next row, only if it is the next market session
    cal_days = _market_calendar(dates, bench).astype("datetime64[D]").astype(np.int64)
    ci = np.searchsorted(cal_days, days)
    in_cal = (ci < len(cal_days)) & (cal_days[np.clip(ci, 0, len(cal_days) - 1)] == days)

    def next_session(k: int) -> np.ndarray:
        j = ci + k
        ok = in_cal & (j < len(cal_days))
        return np.where(ok, cal_days[np.clip(j, 0, len(cal_days) - 1)], -1).astype(float)

    ro, rh, rl, rc = raw["open"], raw["high"], raw["low"], raw["close"]
    rv = np.nan_to_num(raw["volume"], nan=0.0)
    tol = 1e-3
    with np.errstate(invalid="ignore"):
        clean_bar = (
            np.isfinite(ro) & np.isfinite(rh) & np.isfinite(rl) & np.isfinite(rc)
            & (rl > 0) & (rv > 0)
            & (rh >= np.fmax(ro, rc) * (1 - tol)) & (rl <= np.fmin(ro, rc) * (1 + tol))
        ).astype(float)
    dayf = days.astype(float)
    o1, h1, l1, c1 = (B.lead(x, 1) for x in (ro, rh, rl, rc))
    ok1 = (B.lead(dayf, 1) == next_session(1)) & (B.lead(clean_bar, 1) == 1.0)
    ok1 &= np.isfinite(c) & (c > 0)
    gap_log = np.log(_safe_div(o1, c))
    co_log = np.log(_safe_div(c1, c))
    ok1 &= np.abs(np.nan_to_num(gap_log, nan=np.inf)) < _BAD_GAP
    every_split = _split_arrays(splits, code_of, set())[0]   # incl. unadjusted symbols
    ok1 &= ~_split_label_guard(row_key, B, gap_log, co_log, every_split)

    lab: Dict[str, np.ndarray] = {}
    lab["y_gap"] = np.where(ok1, o1 / c - 1.0, np.nan)
    lab["y_oc"] = np.where(ok1, c1 / o1 - 1.0, np.nan)
    lab["y_co"] = np.where(ok1, c1 / c - 1.0, np.nan)
    lab["y_ol"] = np.where(ok1, l1 / o1 - 1.0, np.nan)
    lab["y_oh"] = np.where(ok1, h1 / o1 - 1.0, np.nan)
    c5 = B.lead(rc, 5)
    ok5 = ok1 & (B.lead(dayf, 5) == next_session(5)) & (B.lead(clean_bar, 5) == 1.0)
    lab["y_c5"] = np.where(ok5, c5 / o1 - 1.0, np.nan)
    lab["y_dump"] = np.where(ok1, (lab["y_oc"] <= config.DUMP_THRESHOLD).astype(float), np.nan)
    lab["y_bigdump"] = np.where(ok1, (lab["y_oc"] <= config.BIG_DUMP_THRESHOLD).astype(float), np.nan)
    lab["y_squeeze"] = np.where(ok1, (lab["y_oh"] >= config.SQUEEZE_THRESHOLD).astype(float), np.nan)

    # 10) row filter
    mask = (
        (B.pos >= MIN_PRIOR_BARS)
        & np.isfinite(price) & (price >= config.MIN_PRICE)
        & np.isfinite(dvol20) & (dvol20 >= config.MIN_DOLLAR_VOLUME_20D)
    )
    if min_date is not None:
        mask &= dates >= np.datetime64(pd.Timestamp(min_date).normalize().to_datetime64())
    sel = np.flatnonzero(mask)

    out: Dict[str, Any] = {
        "date": dates[sel],
        "symbol": sym_arr[codes[sel]],
        "open": o[sel], "high": h[sel], "low": l[sel], "close": c[sel], "volume": vol[sel],
        "price": price[sel], "dvol20": dvol20[sel], "ipo_year": ipo_year[sel],
        "is_last_bar": (B.idx == B.end_of)[sel],
    }
    for k in FEATURES:
        if k in f:
            out[k] = f[k][sel]
    out["gap_open"] = lab["y_gap"][sel]
    for k in LABELS:
        out[k] = lab[k][sel]
    panel = pd.DataFrame(out)

    # 11) cross-sectional (per date, over the eligible universe that day)
    _cross_sectional(panel)

    for k in FEATURES + M1_EXTRA + LABELS + ["open", "high", "low", "close", "volume", "price", "dvol20", "ipo_year"]:
        panel[k] = panel[k].astype(np.float32)
    panel = panel[INFO_COLUMNS + FEATURES + M1_EXTRA + LABELS]
    panel.attrs["built_seconds"] = round(time.time() - t0, 2)
    log.info("build_panel: %d symbols, %d rows in, %d rows out (%.1fs)", len(syms), n, len(panel), time.time() - t0)
    return panel


def _cross_sectional(panel: pd.DataFrame) -> None:
    if not len(panel):
        for k in ("r1_rank", "r5_rank", "rvol_rank", "breadth_up", "breadth_ma50", "univ_r1_med", "spike_share"):
            panel[k] = np.array([], dtype=float)
        return
    g = panel.groupby("date", sort=False)
    panel["r1_rank"] = g["r1"].rank(pct=True)
    panel["r5_rank"] = g["r5"].rank(pct=True)
    panel["rvol_rank"] = g["rvol1"].rank(pct=True)
    r1 = panel["r1"]
    tmp = pd.DataFrame({
        "date": panel["date"],
        "up": np.where(r1.notna(), (r1 > 0).astype(float), np.nan),
        "ma50": np.where(panel["dist_ma50"].notna(), (panel["dist_ma50"] > 0).astype(float), np.nan),
        "spike": np.where(r1.notna(), (r1 >= BIG_UP_R1).astype(float), np.nan),
        "r1": r1,
    })
    gt = tmp.groupby("date", sort=False)
    panel["breadth_up"] = gt["up"].transform("mean").to_numpy()
    panel["breadth_ma50"] = gt["ma50"].transform("mean").to_numpy()
    panel["spike_share"] = gt["spike"].transform("mean").to_numpy()
    panel["univ_r1_med"] = gt["r1"].transform("median").to_numpy()


def _empty_panel() -> pd.DataFrame:
    cols = INFO_COLUMNS + FEATURES + M1_EXTRA + LABELS
    df = pd.DataFrame({c: pd.Series(dtype=np.float32) for c in cols})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    df["symbol"] = pd.Series(dtype=object)
    df["is_last_bar"] = pd.Series(dtype=bool)
    return df[cols]


def latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol at its last date (the live feature vector).

    "Last date" means the symbol's last bar in the history passed to
    ``build_panel``; a symbol whose last bar fails the row filters (price,
    liquidity, history length) is not eligible today and is left out rather
    than scored on a stale row. Labels on these rows are NaN (t+1 hasn't
    happened); ``gap_open`` is NaN until the integrator sets it from the
    pre-market price.
    """
    if panel is None or not len(panel):
        return _empty_panel()
    if "is_last_bar" in panel.columns:
        out = panel[panel["is_last_bar"].astype(bool)]
    else:  # pragma: no cover — panels from build_panel always carry it
        out = panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False).tail(1)
    return out.sort_values("symbol").reset_index(drop=True)


def feature_families() -> Dict[str, List[str]]:
    """family → feature names (FEATURES + M1_EXTRA), in FEATURES order."""
    fam: Dict[str, List[str]] = {f: [] for f in FAMILIES}
    for name in FEATURES + M1_EXTRA:
        fam[FEATURE_DOCS[name][0]].append(name)
    return fam
