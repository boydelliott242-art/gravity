"""Extended-hours prices (previous after-hours + pre-market) from Yahoo hourly bars.

What the morning run can know at ~9:00 ET about session D: the previous
session's after-hours trades (16:00–20:00 ET) and D's pre-market trades before
9:00 ET (hourly bars starting 4:00 … 8:00). Yahoo reports no volume for
extended hours but only emits a bar when something traded, so the number of
such bars is an activity proxy.

Every price input is a ratio to the SAME hourly series' regular close of the
previous session (``close_h_prev``) — one fetch, one split basis — so no
comparison with the daily bars is needed. (An earlier version blanked rows
whose hourly and daily closes disagreed by > 3%; those disagreements cluster on
names that reverse-split LATER, which leaked future information into training.)
A store row is used only when its hourly previous session is the panel date.

Training reads a stored history (``data/exthours/ext.pkl``, refreshed before
each retrain); the morning run fetches the same hourly bars for its shortlist
and records what it saw (``data/exthours/live/``) so the difference between
same-day bars and Yahoo's later-revised bars can be measured.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .. import config, net

log = logging.getLogger(__name__)

STORE = config.DATA / "exthours" / "ext.pkl"
MISSING = config.DATA / "exthours" / "missing.json"   # symbols Yahoo had no hourly bars for → {symbol: "YYYY-MM-DD"}
LIVE_LOG = config.DATA / "exthours" / "live"           # what the morning run actually saw (private)
MISSING_RETRY_DAYS = 30
STATUS = "Yahoo extended hours"
ET = "America/New_York"
BATCH = 40
EXT_FEATURES = ["gap_ext", "ext_hi", "ext_lo", "ext_fade", "n_ext", "n_pm", "ah_ret"]
_PM = (240, 540)          # bars starting 4:00 … 8:00 ET → trades before 9:00
_PM8 = (240, 480)         # bars starting 4:00 … 7:00 ET → trades before 8:00 (a 07:20 CT run sees it complete)
_AH = (960, 1200)         # 16:00 … 19:00
_REG = (570, 960)
_RAW = ("ah_last", "pm_last", "ext_last", "ext_high", "ext_low", "n_ext", "n_pm", "close_h_prev",
        "pm_last8", "ext_last8", "ext_high8", "ext_low8", "n_pm8")


def pm_only() -> bool:
    """MORNING_SET "pm": the morning model uses only the last PRE-MARKET trade
    (robust to Yahoo's after-hours outage since 2026-10-07)."""
    return str(getattr(config, "MORNING_SET", "")) == "pm"


def cutoff_suffix() -> str:
    """'' for the 9:00 ET pre-market cutoff, '8' for 8:00 ET (config.MORNING_CUTOFF_ET)."""
    return "8" if int(getattr(config, "MORNING_CUTOFF_ET", 9)) == 8 else ""


def _frame(h: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Hourly yfinance frame → [day, m (minutes after midnight ET), o, h, l, c]."""
    empty = pd.DataFrame(columns=["day", "m", "o", "h", "l", "c"])
    if h is None or not len(h):
        return empty
    cols = {str(c).lower(): c for c in h.columns}
    if not {"open", "high", "low", "close"} <= set(cols):
        return empty
    h = h.dropna(subset=[cols["close"]])
    h = h[h[cols["close"]] > 0]
    idx = pd.DatetimeIndex(h.index)
    idx = idx.tz_localize("UTC").tz_convert(ET) if idx.tz is None else idx.tz_convert(ET)
    return pd.DataFrame({"day": pd.DatetimeIndex(idx.date), "m": idx.hour * 60 + idx.minute,
                         "o": h[cols["open"]].to_numpy(float), "h": h[cols["high"]].to_numpy(float),
                         "l": h[cols["low"]].to_numpy(float), "c": h[cols["close"]].to_numpy(float)})


def ext_row(f: pd.DataFrame, d: pd.Timestamp, prev: Optional[pd.Timestamp]) -> dict:
    """Extended-hours aggregates for session ``d`` (after-hours of ``prev`` +
    pre-market of ``d`` before 9:00 ET) and the regular close of ``prev``."""
    a = f[(f["day"] == prev) & (f["m"] >= _AH[0]) & (f["m"] < _AH[1])] if prev is not None else f.iloc[:0]
    p = f[(f["day"] == d) & (f["m"] >= _PM[0]) & (f["m"] < _PM[1])]
    r = f[(f["day"] == prev) & (f["m"] >= _REG[0]) & (f["m"] < _REG[1])] if prev is not None else f.iloc[:0]
    ext = pd.concat([a, p])
    p8 = p[p["m"] < _PM8[1]]
    ext8 = pd.concat([a, p8])
    pm_last = float(p["c"].iloc[-1]) if len(p) else np.nan
    pm_last8 = float(p8["c"].iloc[-1]) if len(p8) else np.nan
    ah_last = float(a["c"].iloc[-1]) if len(a) else np.nan
    return {"prev": pd.Timestamp(prev) if prev is not None else pd.NaT,
            "ah_last": ah_last, "pm_last": pm_last,
            "ext_last": pm_last if np.isfinite(pm_last) else ah_last,
            "ext_high": float(ext["h"].max()) if len(ext) else np.nan,
            "ext_low": float(ext["l"].min()) if len(ext) else np.nan,
            "n_ext": float(len(ext)), "n_pm": float(len(p)),
            "pm_last8": pm_last8, "ext_last8": pm_last8 if np.isfinite(pm_last8) else ah_last,
            "ext_high8": float(ext8["h"].max()) if len(ext8) else np.nan,
            "ext_low8": float(ext8["l"].min()) if len(ext8) else np.nan, "n_pm8": float(len(p8)),
            "close_h_prev": float(r.sort_values("m")["c"].iloc[-1]) if len(r) and r["m"].max() >= 930 else np.nan}


def reduce(sym: str, h: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """Every regular session in ``h`` → one row of ``ext_row`` aggregates (the
    training store). The first session of a fetch has no previous session in
    the data, so ``drop_first`` discards it — incremental refreshes must, or
    they would overwrite a complete stored row with a half-empty one."""
    f = _frame(h)
    days = sorted(f.loc[(f["m"] >= _REG[0]) & (f["m"] < _REG[1]), "day"].unique())
    rows, prev = [], None
    for d in days:
        d = pd.Timestamp(d)
        rows.append({"symbol": sym, "date": d, **ext_row(f, d, prev)})
        prev = d
    out = pd.DataFrame(rows)
    return out.iloc[1:].reset_index(drop=True) if drop_first and len(out) else out


def derive(ext_last, ext_high, ext_low, ah_last, n_ext, n_pm, base, ok) -> Dict[str, np.ndarray]:
    """Model features: ratios to ``base`` (the hourly series' previous regular
    close); NaN where the row isn't usable (``ok`` False)."""
    c = np.asarray(base, float)
    ok = np.asarray(ok, bool) & np.isfinite(c) & (c > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        el, eh = np.asarray(ext_last, float), np.asarray(ext_high, float)
        out = {
            "gap_ext": el / c - 1, "ext_hi": eh / c - 1, "ext_lo": np.asarray(ext_low, float) / c - 1,
            "ext_fade": el / eh - 1, "ah_ret": np.asarray(ah_last, float) / c - 1,
            "n_ext": np.nan_to_num(np.asarray(n_ext, float), nan=0.0), "n_pm": np.nan_to_num(np.asarray(n_pm, float), nan=0.0),
        }
    return {k: np.where(ok, v, np.nan) for k, v in out.items()}


def _ysym(s: str) -> str:
    return s.replace(".", "-").replace("/", "-")


def fetch(symbols: Iterable[str], start: str, end: str, deadline_s: float = 3600.0) -> Tuple[Dict[str, pd.DataFrame], Set[str]]:
    """Hourly bars with extended hours for ``symbols`` (batches of 40).
    → (bars by symbol, symbols whose batch came back — a symbol in that set
    but not in the bars genuinely had no data; a failed batch proves nothing)."""
    import yfinance as yf
    syms = sorted(set(symbols))
    out: Dict[str, pd.DataFrame] = {}
    answered: Set[str] = set()
    t0 = time.time()
    for bi in range(0, len(syms), BATCH):
        if time.time() - t0 > deadline_s:
            log.warning("exthours: deadline reached after %d/%d symbols", bi, len(syms))
            break
        batch = syms[bi:bi + BATCH]
        ys = [_ysym(s) for s in batch]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(ys, start=start, end=end, interval="60m", prepost=True, group_by="ticker",
                                 threads=True, progress=False, auto_adjust=False)
                break
            except Exception as e:  # rate limit / network
                log.warning("exthours batch %d: %s", bi, e)
                time.sleep(15 * (attempt + 1))
        if df is None or not len(df):
            continue
        got_any = False
        for s, y in zip(batch, ys):
            try:
                h = df[y] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            h = h.dropna(how="all")
            if len(h):
                out[s] = h
                got_any = True
        if got_any:
            answered |= set(batch)
    return out, answered


def load_store() -> pd.DataFrame:
    try:
        st = pd.read_pickle(STORE)
    except Exception:  # missing, truncated or written by an incompatible pandas
        return pd.DataFrame(columns=["symbol", "date", "prev", *_RAW])
    if len(st) and "prev" not in st.columns:      # stores seeded before 'prev' existed
        st = st.sort_values(["symbol", "date"]).reset_index(drop=True)
        st["prev"] = st.groupby("symbol")["date"].shift(1)
    return st


def update_store(symbols: Iterable[str], today: Optional[date] = None, deadline_s: float = 3600.0) -> pd.DataFrame:
    """Append the latest sessions to the stored history: a full 727-day fetch
    for symbols not stored yet; for stored ones, from 15 days before their own
    last stored session, dropping the first fetched session (it has no
    previous session in the fetch). Yahoo only serves hourly bars for ~730
    days, so the store is what keeps older history."""
    syms = sorted(set(symbols))
    today = pd.Timestamp(today or pd.Timestamp.now(tz=ET).date())
    store = load_store()
    last_by = store.groupby("symbol")["date"].max() if len(store) else pd.Series(dtype="datetime64[ns]")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        missing = json.loads(MISSING.read_text())
    except (OSError, ValueError):
        missing = {}
    fresh_miss = {s for s, d in missing.items() if (today - pd.Timestamp(d)).days < MISSING_RETRY_DAYS}
    new_syms = [s for s in syms if s not in last_by.index and s not in fresh_miss]
    groups: List[Tuple[List[str], str, bool]] = [(new_syms, (today - pd.Timedelta(days=727)).strftime("%Y-%m-%d"), False)]
    by_start: Dict[str, List[str]] = {}
    for s in syms:
        if s in last_by.index:
            by_start.setdefault((pd.Timestamp(last_by[s]) - pd.Timedelta(days=15)).strftime("%Y-%m-%d"), []).append(s)
    groups += [(g, st, True) for st, g in sorted(by_start.items())]
    t0 = time.time()
    new_rows: List[pd.DataFrame] = []
    n_ok = 0
    for group, start, incremental in groups:
        if not group:
            continue
        bars, answered = fetch(group, start, end, deadline_s=max(60.0, deadline_s - (time.time() - t0)))
        n_ok += len(bars)
        if not incremental:
            for s in group:
                if s in answered and s not in bars:
                    missing[s] = today.strftime("%Y-%m-%d")
        for s, h in bars.items():
            r = reduce(s, h, drop_first=incremental)
            if len(r):
                new_rows.append(r)
    if new_rows:
        new = pd.concat(new_rows, ignore_index=True)
        new["date"] = pd.to_datetime(new["date"])
        if len(store):
            store = store.set_index(["symbol", "date"])
            new = new.set_index(["symbol", "date"])
            store = pd.concat([store[~store.index.isin(new.index)], new]).reset_index()
        else:
            store = new
        store = store.sort_values(["symbol", "date"]).reset_index(drop=True)
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        store.to_pickle(tmp)
        tmp.replace(STORE)
    try:
        MISSING.parent.mkdir(parents=True, exist_ok=True)
        MISSING.write_text(json.dumps(missing, sort_keys=True))
    except OSError:
        pass
    asked = len(syms) - len(fresh_miss & set(syms))
    net.record_status(STATUS, n_ok >= max(1, int(0.8 * asked)),
                      f"{n_ok}/{asked} symbols refreshed in {time.time() - t0:.0f}s; {len(store):,} sessions stored")
    return store


# ── feed health ──────────────────────────────────────────────────────────
# Yahoo silently stopped serving the 17:00–18:59 ET after-hours bars from the
# 2026-10-07 evening on: overnight ranges shrank by half for every name. A
# morning model trained on complete bars must not be fed that, so sessions are
# graded by how complete their after-hours data is.
GOOD_SESSION_AH = 0.5      # share of names with ≥ 2 after-hours bars on a normal session (~0.8)
GUARD_RATIO = 0.5          # live after-hours completeness must be ≥ half the same names' normal level
GUARD_PM_RATIO = 0.70      # live pre-market coverage must be ≥ 70% of normal: the pre-market model's ONLY
                           # input. Yahoo's pre-market bars thinned ~20–30% from 2026-10-07; R10d (pre-registered)
                           # showed the model still beats close-only with 25% of its inputs missing.


def session_quality(store: pd.DataFrame) -> pd.Series:
    """Per session date: share of stored names with ≥ 2 after-hours bars."""
    if not len(store):
        return pd.Series(dtype=float)
    ah = store["n_ext"].astype(float) - store["n_pm"].astype(float)
    return (ah >= 2).groupby(pd.to_datetime(store["date"])).mean()


def feed_check(live: pd.DataFrame, store: pd.DataFrame, n_ref: int = 40) -> Tuple[bool, str]:
    """Is this morning's extended-hours data as complete as the model's
    training data? Compares the shortlist's after-hours completeness and
    pre-market coverage with the SAME names on the last ``n_ref`` normal
    sessions in the store."""
    fetched = live[live["fetched"].astype(bool)] if "fetched" in live else live.iloc[:0]
    if len(fetched) < max(20, 0.8 * len(live)):
        return False, f"only {len(fetched)}/{len(live)} shortlist names fetched"
    q = session_quality(store)
    good = sorted(q[q >= GOOD_SESSION_AH].index)[-n_ref:]
    ref = store[pd.to_datetime(store["date"]).isin(good) & store["symbol"].isin(fetched.index)]
    if len(ref) < 200:
        return True, "no reference sessions for these names — not checked"
    pm_col = "n_pm" + cutoff_suffix() if ("n_pm" + cutoff_suffix()) in ref.columns else "n_pm"
    ref_ah = float(((ref["n_ext"] - ref["n_pm"]) >= 2).mean())
    ref_pm = float((ref[pm_col] >= 1).mean())
    live_ah = float(((fetched["n_ext"] - fetched["n_pm"]) >= 2).mean())
    live_pm = float((fetched["n_pm"] >= 1).mean())
    pm_ratio = GUARD_PM_RATIO if pm_only() else GUARD_RATIO
    ok = live_pm >= pm_ratio * ref_pm and (pm_only() or live_ah >= GUARD_RATIO * ref_ah)
    return ok, (f"after-hours completeness {live_ah:.0%} vs {ref_ah:.0%} normal, "
                f"pre-market coverage {live_pm:.0%} vs {ref_pm:.0%} for the same names")


def training_features(rows: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Features for the session AFTER each row's date (what the morning of
    that session knows), aligned to ``rows`` (needs symbol, date). A row is
    usable only when the store's hourly previous session IS the row's date."""
    from ..features import next_session_dates
    nxt = next_session_dates(rows["date"])
    key = pd.DataFrame({"symbol": rows["symbol"].to_numpy(), "date": nxt})
    cols = ["prev", *_RAW]
    if not len(store):
        m = pd.DataFrame({c: np.full(len(rows), np.nan) for c in _RAW})
        m["prev"] = pd.NaT
    else:
        have = [c for c in cols if c in store.columns]
        m = key.merge(store[["symbol", "date"] + have], on=["symbol", "date"], how="left")[have]
    for c in cols:
        if c not in m.columns:      # a store seeded before the 8:00 ET columns existed
            m[c] = np.nan
    aligned = pd.to_datetime(m["prev"]).to_numpy("datetime64[ns]") == pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
    x = cutoff_suffix()
    if pm_only():
        col = "n_pm" + x if ("n_pm" + x) in store.columns else "n_pm"
        cov = (store[col].astype(float) >= 1).groupby(pd.to_datetime(store["date"])).mean() if len(store) else pd.Series(dtype=float)
        if len(cov) > 20:   # vs the trailing 60-session median: a vendor gap, not "nobody traded"
            ref = cov.rolling(60, min_periods=20).median().shift(1)
            bad = set(cov[cov < GUARD_PM_RATIO * ref].index)
            if bad:
                aligned &= ~pd.Series(nxt).isin(bad).to_numpy()
        feats = derive(m["pm_last" + x], m["ext_high" + x], m["ext_low" + x], m["ah_last"], m["n_ext"], m["n_pm" + x],
                       m["close_h_prev"], aligned)
        return pd.DataFrame(feats, index=rows.index)
    q = session_quality(store)
    bad = set(q[q < GOOD_SESSION_AH].index) if len(q) else set()
    if bad:     # degraded feed sessions are unknown, not "quiet overnight"
        aligned &= ~pd.Series(nxt).isin(bad).to_numpy()
    feats = derive(m["ext_last" + x], m["ext_high" + x], m["ext_low" + x], m["ah_last"], m["n_ext"], m["n_pm" + x],
                   m["close_h_prev"], aligned)
    return pd.DataFrame(feats, index=rows.index)


def live_features(symbols: Iterable[str], session: date, prev_session: date,
                  closes: Optional[Dict[str, float]] = None, deadline_s: float = 120.0) -> pd.DataFrame:
    """The same features for this morning's shortlist (fetched now), plus a
    record of what was seen (data/exthours/live/<session>.jsonl).
    ``closes`` is accepted for backward compatibility and not used."""
    syms = sorted(set(symbols))
    start = (pd.Timestamp(prev_session) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(session) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t0 = time.time()
    bars, _ = fetch(syms, start, end, deadline_s=deadline_s)
    d, p = pd.Timestamp(session), pd.Timestamp(prev_session)
    rows = []
    for s in syms:
        f = _frame(bars.get(s))
        r = ext_row(f, d, p) if len(f) else {"prev": pd.NaT, **{k: np.nan for k in _RAW}}
        rows.append({"symbol": s, "fetched": s in bars, **r})
    df = pd.DataFrame(rows).set_index("symbol")
    x = cutoff_suffix()
    last = df["pm_last" + x] if pm_only() else df["ext_last" + x]
    feats = pd.DataFrame(derive(last, df["ext_high" + x], df["ext_low" + x], df["ah_last"], df["n_ext"],
                                df["n_pm" + x], df["close_h_prev"], df["fetched"].to_numpy(bool)), index=df.index)
    feats["fetched"] = df["fetched"].astype(bool)
    feats["n_ext_raw"] = df["n_ext"].astype(float)
    feats["n_pm_raw"] = df["n_pm" + x].astype(float)
    _record_live(session, df, feats)
    net.record_status(STATUS, len(bars) >= max(1, int(0.8 * len(syms))),
                      f"{len(bars)}/{len(syms)} shortlist names with hourly bars in {time.time() - t0:.0f}s; "
                      f"{int(np.isfinite(feats['gap_ext']).sum())} traded in extended hours")
    return feats


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _record_live(session: date, raw: pd.DataFrame, feats: pd.DataFrame) -> None:
    try:
        LIVE_LOG.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now(tz=ET).isoformat(timespec="seconds")
        with open(LIVE_LOG / f"{pd.Timestamp(session).date().isoformat()}.jsonl", "a") as fh:
            for s in raw.index:
                rec = {"asof": stamp, "symbol": s, **{k: _num(raw.at[s, k]) for k in _RAW}}
                rec.update({"f_" + k: _num(feats.at[s, k]) for k in feats.columns if k not in ("fetched",)})
                fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass
