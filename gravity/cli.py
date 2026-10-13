"""GRAVITY orchestrator.

    python -m gravity.cli train      # full refresh → walk-forward train → evidence (weekly)
    python -m gravity.cli evening    # after the close: refresh, grade today, publish tomorrow's watchlist
    python -m gravity.cli morning    # pre-market: overnight filings, pre-market gaps, borrow → today's #1
    python -m gravity.cli publish    # push docs/ as-is

The evening run does the heavy lifting (prices, filings, features) and
saves state so the morning run only has to add what happened overnight.
If the evening run was missed (laptop asleep), the morning run does it
first.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, net, publish, score, scorecard, twins
from .util import ET, clean, market_phase, now_et, prev_trading_day, target_session, wait_wall

log = logging.getLogger("gravity")

STATE = config.DATA / "state"
STATE.mkdir(parents=True, exist_ok=True)
PANEL_PATH = STATE / "panel.pkl"
LATEST_PATH = STATE / "latest.pkl"
CONTEXT_PATH = STATE / "context.pkl"
SIGNS_PATH = config.MODELS / "signs.json"
REGIME_PATH = STATE / "regime.json"
ENRICH_DEADLINE_S = 300
MIN_SCORED = 1500

DISCLAIMER = (
    "GRAVITY is an automated research tool, not investment advice. Probabilities describe how "
    "similar historical setups behaved; any single session can go either way. Short selling "
    "carries unlimited loss potential, squeezes, halts, borrow recalls and fees. Data comes from "
    "public sources that can be late or wrong. Do your own research."
)


# ── data refresh ─────────────────────────────────────────────────────────
def refresh_data(full_events: bool = True) -> dict:
    """Universe → prices → splits → filings → static profile → panel."""
    from .sources import prices, sec, universe
    from . import features

    t0 = time.time()
    uni = universe.load_universe()
    syms = uni["symbol"].tolist()
    log.info("universe: %d eligible symbols", len(syms))

    hist = prices.load_history(syms, refresh=True)
    hist = {s: df for s, df in hist.items() if df is not None and len(df) >= 30}
    log.info("prices: %d symbols with history (%.0fs)", len(hist), time.time() - t0)
    splits = prices.load_splits(list(hist))
    bench = prices.benchmark_history()

    events = sec.events_for_universe(list(hist)) if full_events else {}
    log.info("filings: %d symbols with events (%.0fs)", sum(1 for v in events.values() if v), time.time() - t0)

    static = uni.set_index("symbol")[["asia", "ipo_year", "market_cap", "name", "country", "sector", "industry", "price"]].copy()
    static = static[~static.index.duplicated()]
    # refine Asia flag with the SEC business address / incorporation
    cmap = sec.cik_map() if net.sec_enabled() else {}
    refined = 0
    for s in static.index:
        c = cmap.get(s, {}).get("cik")
        if not c:
            continue
        try:
            prof = sec.issuer_profile(c)
        except Exception:  # noqa: BLE001 — profile is best-effort
            prof = None
        if prof and prof.get("asia"):
            if not static.at[s, "asia"]:
                refined += 1
            static.at[s, "asia"] = True
    log.info("asia flag refined from SEC addresses: +%d", refined)

    # FINRA Reg SHO daily short volume history (cached forever per day)
    short_vol = None
    try:
        from .sources import finra_hist
        start = min(df.index.min() for df in hist.values()).date()
        short_vol = finra_hist.load(start, now_et().date(), symbols=list(hist))
        log.info("finra history: %d rows (%.0fs)", len(short_vol), time.time() - t0)
    except Exception as e:  # noqa: BLE001 — flow features are optional; NaN when missing
        log.warning("finra history unavailable: %s", e)

    panel = features.build_panel(hist, events, splits, static, bench, short_vol=short_vol)
    log.info("panel: %d rows × %d cols (%.0fs)", len(panel), panel.shape[1], time.time() - t0)
    ctx = {"universe": uni, "static": static, "splits": splits, "events": events, "cik_map": cmap,
           "built_at": datetime.now(timezone.utc).isoformat()}
    latest = features.latest_rows(panel)

    # Coverage gate BEFORE replacing the saved state: a throttled refresh
    # must never overwrite the last good state with a thin one.
    run = dict(getattr(prices, "LAST_RUN", {}) or {})
    fresh = int((pd.to_datetime(latest["date"]) == pd.to_datetime(latest["date"]).max()).sum()) if len(latest) else 0
    bad = int(run.get("stale", 0) or 0) + int(run.get("failed", 0) or 0)
    why = None
    # A throttled symbol or two is normal; judge by how much actually came back.
    if run.get("requested") and bad > 0.10 * run["requested"]:
        why = f"{bad} of {run['requested']} price refreshes were stale or failed"
    elif fresh < _coverage_floor():
        why = f"only {fresh} names have a bar on the latest date (floor {_coverage_floor()})"
    if why:
        raise CoverageError(why)
    _remember_coverage(fresh)

    # atomic writes; context first, `latest` last = the commit marker
    _atomic_pickle(CONTEXT_PATH, ctx)
    _atomic_pickle(PANEL_PATH, panel)
    _atomic_pickle(LATEST_PATH, latest)
    return {"hist": hist, "panel": panel, "latest": latest, **ctx}


class CoverageError(RuntimeError):
    """The data refresh came back too thin to publish honestly."""


COVERAGE_PATH = STATE / "coverage.json"


def _coverage_floor() -> int:
    """MIN_SCORED, or 85% of the median of the last 10 healthy runs."""
    try:
        hist = json.loads(COVERAGE_PATH.read_text())
    except (OSError, ValueError):
        hist = []
    med = float(np.median(hist[-10:])) if hist else 0.0
    return int(max(MIN_SCORED, 0.85 * med))


def _remember_coverage(n: int) -> None:
    try:
        hist = json.loads(COVERAGE_PATH.read_text())
    except (OSError, ValueError):
        hist = []
    COVERAGE_PATH.write_text(json.dumps((hist + [int(n)])[-30:]))


def _atomic_pickle(path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_state() -> Optional[dict]:
    from .sources import prices
    if not (LATEST_PATH.exists() and CONTEXT_PATH.exists()):
        return None
    try:
        with open(LATEST_PATH, "rb") as f:
            latest = pickle.load(f)
        with open(CONTEXT_PATH, "rb") as f:
            ctx = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError) as e:
        log.warning("saved state unreadable (%s) — will refresh", e)
        return None
    hist = prices.load_history(list(set(latest["symbol"])), refresh=False)
    return {"latest": latest, "hist": hist, **ctx}


def _listing_dates(hist: dict) -> Dict[str, Any]:
    """symbol → first daily bar (≈ first trading day for names listed inside the history)."""
    return {s: df.index.min() for s, df in hist.items() if df is not None and len(df)}


def update_regime(panel: pd.DataFrame) -> Optional[float]:
    """Mean model P(dump) per session over the last ~250 sessions (so today's
    reading can be placed in context) and the universe's realised dump rate
    over the last 20 graded sessions."""
    from . import model
    b = model.load()
    if b is None or panel is None or not len(panel):
        return None
    d = pd.to_datetime(panel["date"])
    days = np.sort(d.unique())[-250:]
    sub = panel[d.isin(days)]
    pr = model.predict(b, sub, use_open=False)["prob_dump"]
    means = pr.groupby(pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d").to_numpy()).mean()
    REGIME_PATH.write_text(json.dumps({k: round(float(v), 5) for k, v in means.items()}))
    lab = panel[panel["y_dump"].notna()]
    ld = pd.to_datetime(lab["date"])
    last20 = np.sort(ld.unique())[-20:]
    rate = float(lab.loc[ld.isin(last20), "y_dump"].mean()) if len(last20) else None
    try:
        ctx = pickle.loads(CONTEXT_PATH.read_bytes())
        ctx["universe_dump_rate_20d"] = rate
        _atomic_pickle(CONTEXT_PATH, ctx)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError):
        pass
    return rate


def compute_signs(panel: pd.DataFrame) -> Dict[str, float]:
    """Direction each feature historically pointed for y_dump (Spearman sign),
    used only to orient the descriptive family bars."""
    from . import features
    lab = panel.dropna(subset=["y_dump"])
    if len(lab) > 400_000:
        lab = lab.sample(400_000, random_state=7)
    y = lab["y_dump"].astype(float)
    signs = {}
    for c in features.FEATURES:
        if c in lab.columns:
            x = pd.to_numeric(lab[c], errors="coerce")
            if x.notna().sum() > 1000 and x.nunique() > 1:
                rho = x.rank().corr(y.rank())
                if rho is not None and not math.isnan(rho) and abs(rho) >= 0.005:
                    signs[c] = float(np.sign(rho))
    SIGNS_PATH.write_text(json.dumps(signs))
    return signs


# ── commands ─────────────────────────────────────────────────────────────
def build_training_panel() -> tuple:
    """Point-in-time training panel (survivorship / selection fix).

    Today's "≤ $2B" list over-represents stocks that *shrank* into small-cap
    range and leaves out the ones that grew out of it. Training therefore
    loads every listed common stock (no upper cap) and keeps a row only while
    that stock's market cap AT THE TIME was ≤ $2B:

        cap_t = as-traded price_t × shares outstanding as reported on the
                latest SEC cover page on or before t (XBRL frames), adjusted
                for any split between that report and t.

    Using today's share count instead would wildly overstate the past cap of
    serial diluters (the names that dump most) and wrongly drop them. Names in
    today's ≤ $2B universe with no SEC share count keep all their rows;
    others without one fall back to today's share count. Delisted names are
    still missing (no free history for them). Cross-sectional features are
    recomputed on the kept rows so they match live scoring."""
    from .sources import prices, sec, universe
    from . import features

    t0 = time.time()
    saved_status = dict(net.STATUS)            # keep today's feed health for the site
    uni = universe.load_universe(max_cap=config.TRAIN_MAX_CAP)
    uni = uni.drop_duplicates("symbol")
    small_today = set(uni.loc[uni["market_cap"] <= config.MAX_MARKET_CAP, "symbol"])
    shares_today = (uni.set_index("symbol")["market_cap"] / uni.set_index("symbol")["price"]).replace([np.inf, -np.inf], np.nan)
    syms = uni["symbol"].tolist()
    hist = prices.load_history(syms, refresh=True)
    hist = {s: df for s, df in hist.items() if df is not None and len(df) >= 30}
    splits = prices.load_splits(list(hist))
    bench = prices.benchmark_history()
    events = sec.events_for_universe(list(hist))
    cmap = sec.cik_map() if net.sec_enabled() else {}
    static = _static_frame(uni, cmap)
    short_vol = None
    try:
        from .sources import finra_hist
        start = min(df.index.min() for df in hist.values()).date()
        short_vol = finra_hist.load(start, now_et().date(), symbols=list(hist))
    except Exception as e:  # noqa: BLE001
        log.warning("finra history unavailable: %s", e)
    panel = features.build_panel(hist, events, splits, static, bench, short_vol=short_vol)

    pit = _pit_market_cap(panel, sec.shares_frames(), cmap, splits, shares_today)
    keep = (pit <= config.MAX_MARKET_CAP) | (pit.isna() & panel["symbol"].isin(small_today))
    log.info("training panel: %d names → %d of %d rows were ≤ $%.0fB at the time (%d with SEC point-in-time "
             "shares) (%.0fs)", len(hist), int(keep.sum()), len(panel), config.MAX_MARKET_CAP / 1e9,
             int(panel.attrs.get("pit_sec_rows", 0)), time.time() - t0)
    panel = panel[keep.to_numpy()].reset_index(drop=True)
    features._cross_sectional(panel)           # ranks / breadth over the kept universe, as live
    if features.MORNING_SET != "official":
        _add_morning_features(panel, events)
    net.STATUS.clear()
    net.STATUS.update(saved_status)
    return panel, hist


def _add_morning_features(panel: pd.DataFrame, events: Dict[str, List[dict]]) -> None:
    """Morning-model inputs for training, exactly as the 9:05 ET run sees them:
    gap_open = last after-hours / pre-market trade before 9:00 ET (not the
    9:30 open, which the morning run can't know), plus — per MORNING_SET — the
    shape of extended-hours trading and filings accepted overnight."""
    from . import features
    from .sources import exthours

    t0 = time.time()
    store = _safe(exthours.update_store, sorted(panel["symbol"].unique()), default=None)
    if store is None or not len(store):
        store = exthours.load_store()
    f = exthours.training_features(panel[["symbol", "date", "close"]], store)
    panel["gap_open"] = f["gap_ext"].astype(np.float32).to_numpy()
    for c in features.EXT_EXTRA:
        if c in features.M1_EXTRA:
            panel[c] = f[c].astype(np.float32).to_numpy()
    # overnight filing counts up to the morning cutoff: the $30K morning ranker's inputs (and M1's if ext_on)
    nxt = features.next_session_dates(panel["date"])
    on = features.overnight_counts(panel["symbol"].to_numpy(), panel["date"], nxt, events,
                                   cutoff_hour=int(getattr(config, "MORNING_CUTOFF_ET", 9)))
    for c in features.ON_FEATURES:
        panel[c] = on[c].astype(np.float32)
    log.info("morning features (%s): %.0f%% of rows have an extended-hours price (%.0fs)", features.MORNING_SET,
             100 * float(np.isfinite(panel["gap_open"]).mean()), time.time() - t0)


def _morning_inputs(latest: pd.DataFrame, shortlist: List[str], pre: Dict[str, dict],
                    events: Dict[str, List[dict]], overnight: Dict[str, List[dict]],
                    last_date: pd.Timestamp, session: date) -> pd.DataFrame:
    """The shortlist's morning-model rows (same definitions as training)."""
    from . import features
    from .sources import exthours

    sub = latest.loc[shortlist].copy()
    budget = max(15.0, min(120.0, _secs_until_bell(session)))
    xf = _safe(exthours.live_features, shortlist, session, last_date.date(), None, budget, default=None)
    xf = (xf if xf is not None else pd.DataFrame(index=shortlist)).reindex(shortlist)
    # exactly the training definition: no Nasdaq fallback (a finite gap with blank
    # extended-hours shape never occurs in training); Nasdaq's quote is display-only
    sub["gap_open"] = xf["gap_ext"].to_numpy(float) if "gap_ext" in xf else np.nan
    for c in features.EXT_EXTRA:
        if c in features.M1_EXTRA:
            sub[c] = xf[c].to_numpy(float) if c in xf else np.nan
    # filings accepted from the last close to the morning cutoff (as in training; later ones are ignored).
    # Names with no filing history at all stay unknown (NaN), exactly as training sees them.
    ev = {s: list(events.get(s, [])) + list(overnight.get(s, [])) for s in shortlist if s in events or s in overnight}
    n = len(shortlist)
    on = features.overnight_counts(np.array(shortlist, dtype=object), [last_date] * n, [pd.Timestamp(session)] * n, ev,
                                   cutoff_hour=int(getattr(config, "MORNING_CUTOFF_ET", 9)))
    for c in features.ON_FEATURES:
        sub[c] = on[c]
    # guard: only names Yahoo actually answered for, and only when the feed is as
    # complete as the data the model was trained on (Yahoo dropped bars on 2026-10-07)
    fetched = xf["fetched"].fillna(False).astype(bool).to_numpy() if "fetched" in xf else np.zeros(len(shortlist), bool)
    live = pd.DataFrame({"fetched": fetched,
                         "n_ext": xf["n_ext_raw"].to_numpy(float) if "n_ext_raw" in xf else np.nan,
                         "n_pm": xf["n_pm_raw"].to_numpy(float) if "n_pm_raw" in xf else np.nan}, index=shortlist)
    ok, why = _safe(exthours.feed_check, live, exthours.load_store(), default=(False, "feed check failed"))
    net.record_status("Extended-hours feed check", ok, why + ("" if ok else " — morning model off, close-only scores"))
    log.info("extended-hours feed check: %s (%s)", "ok" if ok else "FAILED", why)
    base_ok = np.ones(len(shortlist), bool) if ok else np.zeros(len(shortlist), bool)
    sub["m1_ok"] = base_ok if config.MORNING_SCOPE == "shortlist" else base_ok & np.isfinite(sub["gap_open"].to_numpy(float))
    return sub


def _pit_market_cap(panel: pd.DataFrame, frames: List[dict], cmap: dict, splits: dict,
                    shares_today: pd.Series) -> pd.Series:
    """Market cap at each row's date (NaN when unknown)."""
    cap = pd.Series(np.nan, index=panel.index)
    fr = pd.DataFrame(frames)
    by_cik = {c: g.sort_values("end") for c, g in fr.groupby("cik")} if len(fr) else {}
    sec_rows = 0
    adr_fixed = 0
    for sym, idx in panel.groupby("symbol").groups.items():
        rows = panel.loc[idx, ["date", "price", "close"]]
        dates = pd.to_datetime(rows["date"])
        cik = (cmap.get(sym) or {}).get("cik")
        g = by_cik.get(int(cik)) if cik else None
        if g is not None and len(g):
            ends = pd.to_datetime(g["end"]).to_numpy()
            vals = g["shares"].to_numpy(float)
            # latest report on/before t (fall back to the first report for earlier rows)
            k = np.searchsorted(ends, dates.to_numpy(), side="right") - 1
            k_eff = np.where(k < 0, 0, k)
            sh = vals[k_eff]
            rep_dates = ends[k_eff]
            # splits between the report date and t change the share count
            sp = sorted(splits.get(sym, []), key=lambda x: x["date"])
            factor = None
            if sp:
                sd = pd.to_datetime([x["date"] for x in sp]).to_numpy()
                cum = np.cumprod([float(x["ratio"]) for x in sp])
                def factor(t):
                    j = np.searchsorted(sd, t, side="right") - 1
                    return np.where(j >= 0, cum[np.clip(j, 0, None)], 1.0)
                sh = sh * factor(dates.to_numpy()) / factor(rep_dates)
            # ADRs: the SEC cover page counts ORDINARY shares while the listed
            # unit is the ADS (often 1 ADS = 5…1,000 ordinary). When Nasdaq's
            # implied unit count is under a third of the SEC count on today's
            # basis, rescale the whole SEC history by that ratio. (A higher
            # Nasdaq count usually means dilution since the last report — left as is.)
            st = shares_today.get(sym, np.nan) if sym in shares_today.index else np.nan
            if np.isfinite(st) and st > 0:
                today64 = np.datetime64(pd.Timestamp.now().normalize())
                last_adj = vals[-1] * (float(factor(today64) / factor(ends[-1:])[0]) if factor is not None else 1.0)
                ratio = float(st) / last_adj if last_adj > 0 else np.nan
                if np.isfinite(ratio) and ratio < 1 / 3:
                    sh = sh * ratio
                    adr_fixed += 1
            cap.loc[idx] = rows["price"].to_numpy(float) * sh
            sec_rows += len(idx)
        elif sym in shares_today.index and np.isfinite(shares_today.get(sym, np.nan)):
            # no SEC history: today's share count only matches the SPLIT-ADJUSTED
            # close (as-traded price × today's count is off by every split since)
            cap.loc[idx] = rows["close"].to_numpy(float) * float(shares_today[sym])
    panel.attrs["pit_sec_rows"] = sec_rows
    panel.attrs["pit_adr_rescaled"] = adr_fixed
    return cap


def _static_frame(uni: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """asia / ipo_year / descriptors per symbol, with the Asia flag refined
    from SEC business addresses and incorporation."""
    from .sources import sec
    static = uni.set_index("symbol")[["asia", "ipo_year", "market_cap", "name", "country", "sector", "industry", "price"]].copy()
    static = static[~static.index.duplicated()]
    for s in static.index:
        c = cmap.get(s, {}).get("cik")
        if not c:
            continue
        prof = _safe(sec.issuer_profile, c)
        if prof and prof.get("asia"):
            static.at[s, "asia"] = True
    return static


def cmd_train(_args: argparse.Namespace) -> int:
    from . import evidence, model
    state = refresh_data()                      # daily state stays current
    panel, whist = build_training_panel()       # point-in-time training rows
    report = model.train(panel)
    publish.write_json("model.json", report)
    ev = evidence.run_studies(panel, listing_dates=_listing_dates(whist))
    publish.write_json("evidence.json", ev)
    compute_signs(panel)
    _safe(update_regime, state["panel"])
    log.info("train done: %s", {k: report.get(k) for k in ("n_rows", "n_symbols", "trained_through")})
    return 0


def _model_stale(days: int = 7) -> bool:
    from . import model
    b = model.load()
    if not b:
        return True
    ts = b.get("trained_at") or (b.get("report") or {}).get("trained_at")
    if not ts:
        return True
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return True
    return age > timedelta(days=days)


def _is_late(session_iso: str) -> bool:
    """True once that session's 9:30 ET opening bell has rung."""
    from datetime import time as dtime
    open_bell = datetime.combine(date.fromisoformat(session_iso), dtime(9, 30), tzinfo=ET)
    return now_et() >= open_bell


def _withhold(reason: str, push: bool) -> None:
    """Leave the last good page up, but tell readers today's run was withheld."""
    cur = publish.read_json("today.json") or {}
    cur["withheld"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": reason}
    publish.write_json("today.json", cur)
    if push:
        publish.push(f"withheld: {reason[:80]}")


def _publish_today(today: dict, args: argparse.Namespace, message: str) -> int:
    """Write today.json, log the picks (pre-open only) and push.

    A late page never replaces a pre-open page for the same session — the
    board readers see must be the board the track record grades."""
    today["late"] = _is_late(today["session_date"])
    cur = publish.read_json("today.json") or {}
    if today["late"] and cur.get("session_date") == today["session_date"] and not cur.get("late") \
            and not cur.get("sample"):
        log.info("run is after the open — keeping the pre-open page for %s", today["session_date"])
        return 0
    if (cur.get("session_date") == today["session_date"] and cur.get("run") == "morning" and today.get("run") == "morning"
            and not cur.get("late") and (cur.get("top") or {}).get("model_used") == "m1"
            and (today.get("top") or {}).get("model_used") != "m1"):
        log.info("keeping the earlier morning-model page for %s (this run fell back to close-only scores)", today["session_date"])
        return 0
    if cur.get("session_date") == today["session_date"] and cur.get("run") == "morning" \
            and today.get("run") == "evening" and not cur.get("late"):
        log.info("an evening build must not replace the morning page for %s — keeping it", today["session_date"])
        return 0
    publish.write_json("today.json", today)
    _flush_side()
    if not today["late"]:
        logged = scorecard.log_picks(today)
        if logged is False and _is_late(today["session_date"]):
            # the bell rang between deciding "not late" and logging: say so on the page
            today["late"] = True
            publish.write_json("today.json", today)
        elif args.no_push:
            scorecard.mark_unverified(today["session_date"], "written by a run that did not push (--no-push)")
        from . import feed
        _safe(feed.write)
    else:
        log.info("published after the open — not logged to the track record")
    if args.no_push:
        return 0
    if publish.push(message):
        if not today["late"] and _is_late(today["session_date"]):
            scorecard.mark_unverified(today["session_date"], "push confirmed only after the 9:30 ET open")
        return 0
    if not today["late"]:
        scorecard.mark_unverified(today["session_date"], "push to GitHub not confirmed at publish time")
    return 3


def cmd_evening(args: argparse.Namespace) -> int:
    from . import evidence, model
    try:
        state = refresh_data()
    except CoverageError as e:
        log.error("coverage gate FAILED on refresh (%s) — no grading, previous page kept", e)
        _withhold(f"evening data refresh too thin ({e})", not args.no_push)
        return 2
    elig = state["latest"]["symbol"].tolist()
    hist_g = dict(state["hist"])
    missing = [s for s in scorecard.ungraded_symbols() if s not in hist_g]
    if missing:                       # a pick that fell out of the universe (sub-$0.10, delisted…) still gets graded
        from .sources import prices as _prices
        extra = _safe(_prices.load_history, missing, refresh=True, report=False, default={}) or {}
        hist_g.update({s: df for s, df in extra.items() if df is not None and len(df)})
        log.info("grading: loaded bars for %d/%d picks outside today's universe", len(extra), len(missing))
    n = scorecard.grade(hist_g, elig, before=now_et().date())
    log.info("graded %d session(s)", n)
    publish.write_json("scorecard.json", scorecard.summary())
    if args.retrain or _model_stale(9):
        try:
            log.info("model stale — retraining (walk-forward, point-in-time universe)")
            tpanel, whist = build_training_panel()
            publish.write_json("model.json", model.train(tpanel))
            publish.write_json("evidence.json", evidence.run_studies(tpanel, listing_dates=_listing_dates(whist)))
            compute_signs(tpanel)
        except Exception as e:  # noqa: BLE001 — keep the existing model and carry on
            log.exception("retrain failed — keeping the previous model")
            net.record_status("Model retrain", False, f"failed: {e}")
    state["universe_dump_rate_20d"] = _safe(update_regime, state["panel"])
    today = build_today(state, run="evening", live_extras=True)
    ok, why = healthy(today)
    if not ok:
        log.error("coverage gate FAILED (%s) — keeping the previous page", why)
        _withhold(f"watchlist withheld: {why}", not args.no_push)
        return 2
    return _publish_today(today, args, f"evening: watchlist for {today['session_date']}")


def cmd_morning(args: argparse.Namespace) -> int:
    state = load_state()
    session = target_session()
    refreshed = False

    def stale(st) -> bool:
        if st is None:
            return True
        d = pd.to_datetime(st["latest"]["date"])
        if d.max().date() < prev_trading_day(session) or (d == d.max()).mean() < 0.9:
            return True
        # the evening ran before IWM's bar existed → market features blank: refresh once
        iw = st["latest"].get("iwm_r1")
        return iw is not None and bool(pd.isna(iw[d == d.max()]).all())

    try:
        if stale(state):
            log.info("evening state missing/stale — running the full refresh first")
            state, refreshed = refresh_data(), True
            publish.write_json("scorecard.json", scorecard.summary())
        today = build_today(state, run="morning", live_extras=True)
        ok, why = healthy(today)
        if not ok and not refreshed:
            log.warning("coverage gate failed on saved state (%s) — refreshing once and retrying", why)
            state = refresh_data()
            today = build_today(state, run="morning", live_extras=True)
            ok, why = healthy(today)
    except CoverageError as e:
        ok, why = False, str(e)
    if not ok:
        log.error("coverage gate FAILED (%s) — keeping the previous page", why)
        _withhold(f"morning run withheld: {why}", not args.no_push)
        return 2
    top = (today.get("top") or {}).get("symbol", "none")
    rc = _publish_today(today, args, f"morning: {today['session_date']} #1 {top}")
    if rc == 0 and not args.no_push and today.get("top") and not today.get("late"):
        tp = today["top"]
        publish._notify(f"Today's #1: {tp['symbol']} — {(tp.get('prob_dump') or 0) * 100:.0f}% odds of a 5%+ open→close drop "
                        f"(squeeze odds {(tp.get('prob_squeeze') or 0) * 100:.0f}%). Research, not advice.")
    return rc


def healthy(today: dict) -> tuple:
    """Coverage gate: never overwrite a good page with a thin or stale one.

    Fails when fewer than MIN_SCORED names were scored, when under 60% of
    the eligible universe traded on the feature date, or when the features
    are older than the session before the one we're picking for."""
    uni = today.get("universe") or {}
    scored, eligible = int(uni.get("scored") or 0), int(uni.get("eligible") or 0)
    if scored < MIN_SCORED:
        return False, f"only {scored} names scored (< {MIN_SCORED})"
    if eligible and scored < 0.6 * eligible:
        return False, f"only {scored}/{eligible} eligible names have a bar on the feature date"
    session = date.fromisoformat(today["session_date"])
    asof = date.fromisoformat(today["features_asof"])
    if asof < prev_trading_day(session):
        return False, f"features as of {asof} are stale for session {session}"
    if not today.get("board"):
        return False, "empty board"
    return True, "ok"


def cmd_live(args: argparse.Namespace) -> int:
    from . import live
    return live.run(push=not args.no_push)


def cmd_publish(_args: argparse.Namespace) -> int:
    return 0 if publish.push("manual publish") else 1


# ── the build ────────────────────────────────────────────────────────────
def _evidence_lift() -> Optional[float]:
    ev = publish.read_json("evidence.json") or {}
    for st in ev.get("studies", []):
        sid = str(st.get("id", ""))
        if "offer" in sid and ("1" in sid or "fresh" in sid or "recent" in sid):
            return st.get("lift_dump")
    return None


def _filings_for(sym: str, events: Dict[str, List[dict]], overnight: Dict[str, List[dict]]) -> List[dict]:
    allf = list(overnight.get(sym, [])) + list(events.get(sym, []))
    seen, out = set(), []
    for e in sorted(allf, key=lambda e: (e.get("date") or "", e.get("accepted") or ""), reverse=True):
        key = e.get("url") or (e.get("form"), e.get("date"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _safe(fn, *a, default=None, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 — one enrichment failing must not sink the run
        log.warning("%s failed: %s", getattr(fn, "__name__", fn), e)
        return default


def _intel():
    try:
        from .sources import intel
        return intel
    except ImportError:  # pragma: no cover — optional until built
        return None


BOARD_POOL_MAX = 360        # deepest rank the board selection will walk to fill BOARD_SIZE eligible names
GATE_CHUNK = 60


def _nn(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _lend_short(sym: str, borrow: dict, borrow_ok: bool, price: Any, min_usd: Optional[float] = None) -> Optional[float]:
    """IBKR's lendable dollars when they are fewer than a ``min_usd`` short
    (default BOARD_MIN_USD) needs (None when enough, or when the borrow list /
    price is unknown)."""
    need = config.BOARD_MIN_USD if min_usd is None else min_usd
    if not borrow_ok:
        return None
    b = borrow.get(sym)
    try:
        avail, px = float(b["available"]), float(price)
    except (TypeError, KeyError, ValueError):
        return None
    if not (math.isfinite(avail) and math.isfinite(px)) or px <= 0:
        return None
    return avail * px if avail * px < need else None


def _gate_bad(sym: str, latest: pd.DataFrame, borrow: dict, borrow_ok: bool, E: dict,
              min_usd: Optional[float] = None) -> bool:
    """The board's exclusion rule (no borrow, IBKR can't lend a ``min_usd``
    short — BOARD_MIN_USD by default, ONE_MIN_USD for the ONE — halted, or
    squeeze danger ≥ 70), computed exactly as make_pick computes it."""
    r = latest.loc[sym]
    if _lend_short(sym, borrow, borrow_ok, _nn(r.get("close")), min_usd) is not None:
        return True
    short = score.shortability(sym, borrow, borrow_ok)
    si_rows = E["si"].get(sym) or []
    fl = E["floats"].get(sym) or {}
    si_pct = None
    if si_rows and fl.get("float"):
        si_pct = (si_rows[0].get("interest") or 0) / fl["float"] if fl["float"] else None
    if si_pct is None:
        si_pct = fl.get("short_pct_float")
    dtc = si_rows[0].get("days_to_cover") if si_rows else None
    sqd, _ = score.squeeze_danger(_nn(r.get("sq_pct")), short, si_pct, dtc, fl.get("float"), _nn(r.get("short_ratio_5d")))
    vol = _nn(r.get("volume"))
    return short.get("status") == "NONE" or (sqd or 0) >= 70 or (vol is not None and float(vol) == 0)


def _secs_until_bell(session: date, margin_min: int = 10) -> float:
    """Wall-clock seconds left before ``margin_min`` minutes ahead of that
    session's 9:30 ET open (inf for runs after the close of the previous day
    that are not racing the bell, i.e. more than 12 h before it)."""
    bell = datetime.combine(session, datetime.min.time(), tzinfo=ET).replace(hour=9, minute=30)
    left = (bell - now_et()).total_seconds() - margin_min * 60
    return float("inf") if left > 12 * 3600 else left


_KINDS = ("news", "si", "analyst", "dilution", "insider", "floats", "chatter")
CHUNK = 20                  # float / chatter lookups per future, so a deadline keeps partial results
CHATTER_FRESH_TOP = 50      # chatter refetched every run for the top of the list; older cache (12 h) below


def enrich(symbols: List[str], latest: pd.DataFrame, cmap: dict, splits: dict,
           deadline_s: float, deep: Optional[List[str]] = None, kinds: tuple = _KINDS) -> dict:
    """Per-name lookups (news, short interest, analyst, float, dilution,
    insider sales, chatter), all overlapped under ONE hard deadline — a
    slow feed must never hold the pre-market publish hostage. Anything
    unfinished is simply missing ("—")."""
    from .sources import news as newsmod
    from .sources import shortside, street

    intel = _intel()
    deep = set(deep or [])
    t_e = time.time()
    ex = ThreadPoolExecutor(max_workers=12)
    lane_fl = ThreadPoolExecutor(max_workers=1)   # Yahoo float: sequential source, own lane
    lane_ch = ThreadPoolExecutor(max_workers=1)   # Stocktwits chatter: sequential source, own lane
    chunks = [symbols[i:i + CHUNK] for i in range(0, len(symbols), CHUNK)]
    f_fl = [lane_fl.submit(_safe, shortside.float_shares, c, report=False, default={}) for c in chunks] if "floats" in kinds else []
    f_ch = ([lane_ch.submit(_safe, intel.chatter, c, report=False, default={},
                            **({} if i * CHUNK < CHATTER_FRESH_TOP else {"max_age_s": 12 * 3600}))
             for i, c in enumerate(chunks)] if intel is not None and "chatter" in kinds else [])
    F: Dict[str, Dict[str, Any]] = {k: {} for k in ("news", "si", "analyst", "dilution", "insider")}
    for s in symbols:
        name = str(latest.at[s, "name"] or "") if s in latest.index and "name" in latest.columns else ""
        cik = (cmap.get(s) or {}).get("cik")
        if "news" in kinds:
            F["news"][s] = ex.submit(_safe, newsmod.headlines, s, name, default=[])
        if "si" in kinds:
            F["si"][s] = ex.submit(_safe, shortside.short_interest, s, default=[])
        if "analyst" in kinds:
            F["analyst"][s] = ex.submit(_safe, street.analyst, s)
        if intel is not None and cik:
            if "dilution" in kinds:
                F["dilution"][s] = ex.submit(_safe, intel.dilution_intel, s, cik, splits.get(s, []))
            if s in deep and "insider" in kinds:
                F["insider"][s] = ex.submit(_safe, intel.insider_sales, s, cik)
    allf = [f for d in F.values() for f in d.values()] + f_fl + f_ch
    _done, pending = wait_wall(allf, deadline_s)
    ex.shutdown(wait=False, cancel_futures=True)
    lane_fl.shutdown(wait=False, cancel_futures=True)
    lane_ch.shutdown(wait=False, cancel_futures=True)

    def res(f, default):
        return (f.result() if f is not None and f.done() and not f.cancelled() else None) or default

    out = {k: {s: res(f, [] if k in ("news", "si") else None) for s, f in d.items()} for k, d in F.items()}
    out["floats"], out["chatter"] = {}, {}
    for f in f_fl:
        out["floats"].update(res(f, {}))
    for f in f_ch:
        out["chatter"].update(res(f, {}))
    # chunked sources record their health per chunk — restate it for the whole set
    if f_fl:
        n_f = sum(1 for v in out["floats"].values() if v)
        net.record_status("Yahoo float", n_f >= 0.6 * len(symbols), f"{n_f}/{len(symbols)} names with a float")
    if f_ch:
        n_c = sum(1 for v in out["chatter"].values() if v)
        net.record_status("Stocktwits chatter", n_c >= 0.5 * len(symbols), f"{n_c}/{len(symbols)} names with chatter")
    if pending:
        net.record_status("Enrichment deadline", False, f"{len(pending)} lookups unfinished after {deadline_s:.0f}s — shown as missing")
    log.info("enrichment for %d names in %.0fs (%d unfinished)", len(symbols), time.time() - t_e, len(pending))
    return out


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            dst.setdefault(k, {}).update(v)


def build_today(state: dict, run: str, live_extras: bool = True) -> dict:
    from . import capacity as CAP
    from . import features, history, model
    from .sources import prices, sec, shortside, street

    t0 = time.time()
    session = target_session()
    phase = market_phase()
    from .util import holidays_covered
    if not holidays_covered(session.year):
        net.record_status("Market calendar", False, "holiday table has run out — update gravity/util.py")
    latest: pd.DataFrame = state["latest"].copy()
    last_date = pd.Timestamp(latest["date"].max())
    # only names that actually traded on the last session are scoreable
    latest = latest[pd.to_datetime(latest["date"]) == last_date].set_index("symbol", drop=False)
    uni = state["universe"].set_index("symbol")
    uni = uni[~uni.index.duplicated()]
    for col in ("name", "country", "sector", "industry", "market_cap"):
        latest[col] = uni[col].reindex(latest.index)
    bundle = model.load()
    if bundle is None:
        raise SystemExit("no trained model — run `python -m gravity.cli train` first")
    cmap = state.get("cik_map", {})
    splits = state.get("splits", {})
    hist = state["hist"]
    events = state.get("events", {})

    # 1) overnight/live filings the feature history can't contain yet
    since = datetime.combine(last_date.date(), datetime.min.time(), tzinfo=ET).replace(hour=16)
    t_s = time.time()
    overnight_list = _safe(sec.latest_filings, since.astimezone(timezone.utc), default=[]) or []
    # was EDGAR's current feed read completely? (the $30K morning ranker counts overnight filings from it)
    sec_feed_ok = bool(((getattr(sec, "_parts", None) or {}).get("latest") or (False,))[0])
    ft = _safe(sec.fulltext_catalysts, last_date.date().isoformat(), session.isoformat(), default=[]) or []
    merged = _safe(sec.merge_events, overnight_list, ft, default=None)
    if merged is None:
        merged = overnight_list + ft
    overnight: Dict[str, List[dict]] = {}
    for e in merged:
        s = e.get("symbol")
        if s:
            overnight.setdefault(s, []).append(e)
    log.info("overnight filings: %d symbols (%d live + %d full-text) in %.0fs",
             len(overnight), len(overnight_list), len(ft), time.time() - t_s)

    # 2) model pass 1 (M0) on everything
    OUT = ["prob_dump", "prob_bigdump", "prob_squeeze", "exp_oc", "prob_swing", "prob_pump", "skew", "liq_edge",
           "one_dump", "one_d3"]
    pred = model.predict(bundle, latest, use_open=False)
    latest = latest.join(pred[[c for c in OUT if c in pred.columns]], rsuffix="_m")
    for c in OUT:
        if c not in latest.columns:
            latest[c] = np.nan
    latest["model_used"] = "m0"
    latest["prob_dump_m0"] = latest["prob_dump"]
    # the board universe (R11): names that take a BOARD_MIN_USD short without moving the price,
    # ranked by the $30K ranker (expected open→close shortfall versus the typical such name)
    latest["liq_ok"] = model.liquid_mask(latest)
    rank_col = "liq_edge" if latest["liq_edge"].notna().any() else "prob_dump"
    if rank_col != "liq_edge":
        log.warning("model bundle has no $30K ranker — board ranked by dump odds until the next train")
    log.info("$%s-ready universe: %d of %d names", f"{config.BOARD_MIN_USD:,}", int(latest["liq_ok"].sum()), len(latest))
    # the ONE (R13): consensus of three dump models among $30K-ready names; picked among $100K-ready ones
    latest["one_ok"] = model.one_mask(latest)

    def _one_scores() -> None:
        parts = [pd.to_numeric(latest[c], errors="coerce").to_numpy(float) for c in model.ONE_PARTS]
        sc, dfc = model.one_score(np.zeros(len(latest)), parts, latest["liq_ok"].to_numpy(bool))
        latest["one_score"], latest["one_deficit"] = sc, dfc

    _one_scores()
    log.info("$%s-ready (ONE) universe: %d names", f"{config.ONE_MIN_USD:,}", int(latest["one_ok"].sum()))

    # 3) shortlist → pre-market snapshot → M1 where we have a gap proxy
    order = latest[latest["liq_ok"]].sort_values(rank_col, ascending=False)
    # the morning model must score at least as deep as the board selection walks
    shortlist = list(order.index[:BOARD_POOL_MAX + 90])
    # ... and as deep as the ONE could reach: its top 150 by the close-only consensus
    one_pre = latest[latest["one_ok"] & latest["one_score"].notna()].sort_values("one_score", ascending=False)
    shortlist += [s for s in one_pre.index[:150] if s not in shortlist]
    supply = [s for s, evs in overnight.items()
              if s in latest.index and bool(latest.at[s, "liq_ok"]) and any(e.get("category") in score.SUPPLY_CATS for e in evs)]
    shortlist += [s for s in supply if s not in shortlist]
    pre: Dict[str, dict] = {}
    want_pre = phase == "pre-market" or run == "morning"
    if want_pre and _secs_until_bell(session) < 60:
        log.warning("pre-market snapshot skipped — too close to the 9:30 ET open (M0 only)")
        want_pre = False
    if want_pre:
        t_p = time.time()
        # Nasdaq's pre-market quote is display-only now (the model reads Yahoo's bars): top 150 + supply filers
        disp = list(dict.fromkeys(shortlist[:150] + supply))
        pre = _safe(prices.premarket_snapshot, disp, max(60.0, min(900.0, _secs_until_bell(session) - 600.0)), default={}) or {}
        log.info("pre-market snapshot: %d/%d names in %.0fs", len(pre), len(shortlist), time.time() - t_p)
    from . import features as _features
    # M1 was trained on extended-hours trades through 9:00 ET; an earlier run (07:20 CT)
    # would feed it a shorter window, so it scores close-only and the 9:05 run decides
    m1_window_ok = now_et().time() >= datetime.min.time().replace(hour=int(getattr(config, "MORNING_CUTOFF_ET", 9)))
    if _features.MORNING_SET != "official" and phase == "pre-market" and want_pre and not m1_window_ok:
        log.info("before %d:00 ET — close-only scores for now; a later run adds the pre-market model", int(getattr(config, "MORNING_CUTOFF_ET", 9)))
        use_open_syms, sub = [], latest.iloc[:0]
    elif _features.MORNING_SET != "official" and phase == "pre-market" and want_pre:
        sub = _morning_inputs(latest, shortlist, pre, events, overnight, last_date, session)
        use_open_syms = list(sub.index[sub["m1_ok"].astype(bool)])
        sub = sub.loc[use_open_syms]
    else:
        use_open_syms = [s for s in shortlist if s in pre and pre[s].get("gap_pct") is not None and phase == "pre-market"]
        sub = latest.loc[use_open_syms].copy()
        sub["gap_open"] = [pre[s]["gap_pct"] for s in use_open_syms]
    # the $30K morning ranker reads overnight filing counts: if EDGAR's current feed was incomplete it
    # would see "no filings" for names that did file, so it falls back to the close-only ranker
    liq_on_feed = bool(features.LIQ_M1_EXTRA) and any(c in features.LIQ_M1_EXTRA for c in features.ON_FEATURES)
    keep_liq_m0 = liq_on_feed and not sec_feed_ok
    if use_open_syms and liq_on_feed:
        net.record_status("Overnight filings for the $30K ranker", sec_feed_ok,
                          "EDGAR current feed complete" if sec_feed_ok else
                          "EDGAR current feed incomplete — the $30K board uses the close-only ranker this run")
    if use_open_syms:
        p1 = model.predict(bundle, sub, use_open=True)
        for c in OUT:
            if c in p1.columns and not (c == "liq_edge" and keep_liq_m0):
                latest.loc[use_open_syms, c] = p1[c].values
        latest.loc[use_open_syms, "model_used"] = "m1"
    _one_scores()          # the ONE's consensus with the morning scores (before the catalyst overlay below)
    latest["prob_dump_model"] = latest["prob_dump"]      # the ONE's tie-break: the model's own odds, pre-overlay

    # 4) flag supply filings the model hasn't seen (probability unchanged — see score.overlay_catalysts)
    lift = _evidence_lift()
    notes: Dict[str, str] = {}
    for s, evs in overnight.items():
        if s not in latest.index:
            continue
        fresh = [e for e in evs if (e.get("date") or "") > last_date.date().isoformat() or e.get("text_tags")]
        newp, note = score.overlay_catalysts(float(latest.at[s, "prob_dump"]), fresh, lift)
        if note:
            latest.at[s, "prob_dump"] = newp
            notes[s] = note

    base = ((bundle.get("report") or {}).get("base_rate") or {}).get("dump")
    latest["lift"] = latest["prob_dump"] / base if base else np.nan
    latest["score"] = score.rank_percentile(latest["prob_dump"])
    latest["sq_pct"] = latest["prob_squeeze"].rank(pct=True) * 100
    latest["range14_pct"] = pd.to_numeric(latest.get("range14"), errors="coerce").rank(pct=True)
    latest["ssr"] = model.ssr_next(latest)
    latest["publishable"] = model.publishable(latest)

    # 5) borrow + short volume for everyone (cheap bulk files)
    borrow = _safe(shortside.ibkr_borrow, default={}) or {}
    if borrow:
        names = {s_: str(latest.at[s_, "name"]) for s_ in latest.index if "name" in latest.columns and pd.notna(latest.at[s_, "name"])}
        mapped = _safe(shortside.map_by_name, borrow, names, set(latest.index), default={}) or {}
        for s_, ib in mapped.items():
            borrow[s_] = dict(borrow[ib], ibkr_symbol=ib)
        if mapped:
            log.info("IBKR borrow: %d names matched by company name under another ticker (%s)",
                     len(mapped), ", ".join(f"{k}→{v}" for k, v in list(mapped.items())[:8]))
    borrow_ok = len(borrow) > 1000
    sv = _safe(shortside.finra_short_volume, 5, default=None)
    if sv is not None and len(sv):
        g = sv.groupby("symbol")[["short_volume", "total_volume"]].sum()
        latest["short_ratio_5d"] = (g["short_volume"] / g["total_volume"].replace(0, np.nan)).reindex(latest.index)
    else:
        latest["short_ratio_5d"] = np.nan
    latest["borrow_status"] = [score.shortability(s, borrow, borrow_ok)["status"] for s in latest.index]
    today_iso, run_day = session.isoformat(), now_et().date().isoformat()
    _liq = ((bundle.get("report") or {}).get("liquid") or {}).get("m1" if use_open_syms else "m0") or {}
    _lb = {(b_.get("lo"), b_.get("hi")): b_ for b_ in (_liq.get("bands") or [])}
    _pub = ((bundle.get("report") or {}).get("oos") or {}).get("m0") or {}
    _mv = (_lb.get((1, 1)) or {}).get("mean_oc") if _lb else _pub.get("pub1_mean_oc")
    avg_move = (-float(_mv)) if (_mv is not None and _mv < 0) else None   # the board #1's historical average drop
    _m10 = (_lb.get((1, 10)) or {}).get("mean_oc") if _lb else _pub.get("top10_mean_oc")
    avg_move_10 = (-float(_m10)) if (_m10 is not None and _m10 < 0) else None  # a top-10 pick's

    from . import live as _live
    quoted_spreads = _safe(_live.recent_spreads, default={}) or {}

    def size_of(sym: str, m: dict, move: Optional[float] = None, basis: str = "top-10") -> Optional[dict]:
        df = hist.get(sym)
        if df is None or len(df) < 12:
            return None
        b = borrow.get(sym) if borrow_ok else None
        avail = b.get("available") if b else (0 if borrow_ok else None)
        dv1 = (m.get("close") or 0) * (m.get("volume") or 0)
        blk = _safe(CAP.size_block, m.get("close"), m.get("dvol20") or 0.0, dv1, m.get("vol20"),
                    df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), avail,
                    avg_move_10 if move is None else move, quoted_spreads.get(sym))
        if blk is not None:
            blk["move_basis"] = basis
        return blk

    # 6) families (descriptive) + model attribution (what the model actually weighed)
    feature_docs = getattr(features, "FEATURE_DOCS", {})
    try:
        signs = json.loads(SIGNS_PATH.read_text())
    except (OSError, ValueError):
        signs = {}
    latest = latest.join(score.family_scores(latest, feature_docs, signs))

    # 7) candidates: board pool, swing pool, provisional #1 (for twins) — all $30K-ready
    order = latest[latest["liq_ok"]].sort_values(rank_col, ascending=False)
    shortable = latest["borrow_status"].isin(["ETB", "HTB", "UNKNOWN"])
    swing_order = latest[shortable & latest["liq_ok"] & latest["prob_swing"].notna()].sort_values("prob_swing", ascending=False)
    swing_pool = list(swing_order.index[:25])
    one_order = latest[latest["one_ok"] & latest["one_score"].notna()].sort_values(
        ["one_score", "one_deficit", "prob_dump_model"], ascending=[False, True, False])
    provisional = next((s for s in one_order.index if latest.at[s, "borrow_status"] in ("ETB", "HTB")), None) or \
        next((s for s in order.index if latest.at[s, "publishable"] and latest.at[s, "borrow_status"] in ("ETB", "HTB")), None)
    feat_rows = latest.copy()
    tw = twins.find_twins(feat_rows, ref=provisional, session_year=session.year) if provisional else []

    E: Dict[str, Dict[str, Any]] = {k: {} for k in ("news", "si", "analyst", "dilution", "insider", "floats", "chatter")}
    budget = lambda: max(20.0, min(ENRICH_DEADLINE_S, _secs_until_bell(session)))  # noqa: E731
    # Stage A — fill the board: walk the ranking, fetch only what the squeeze gate
    # needs (short interest, float; cached), keep names that are shortable, not
    # halted and under squeeze danger 70, until BOARD_SIZE pass (12 more → zone).
    board_syms: List[str] = []
    zone_syms: List[str] = []
    walk = list(order.index)
    depth = 0
    while len(board_syms) < config.BOARD_SIZE and depth < min(BOARD_POOL_MAX, len(walk)):
        cand = walk[depth:depth + GATE_CHUNK]
        depth += len(cand)
        need = [s for s in cand if s not in E["si"]]
        if need and live_extras:
            _merge(E, enrich(need, latest, cmap, splits, budget(), kinds=("si", "floats")))
        for s in cand:
            if _gate_bad(s, latest, borrow, borrow_ok, E):
                if len(zone_syms) < 12:
                    zone_syms.append(s)
            elif len(board_syms) < config.BOARD_SIZE:
                board_syms.append(s)
    keep = set(board_syms) | set(zone_syms)
    board_pool = [s for s in walk[:depth] if s in keep]
    log.info("board fill: %d eligible from the top %d (+%d squeeze zone)", len(board_syms), depth, len(zone_syms))
    urank = {s: i + 1 for i, s in enumerate(walk)}
    # the ONE: walk its consensus order, same gates as the board but IBKR must lend ONE_MIN_USD
    one_walk = list(one_order.index[:40])
    one_sym: Optional[str] = None
    one_rank: Optional[int] = None
    one_held: List[str] = []
    for i in range(0, len(one_walk), 10):
        cand = one_walk[i:i + 10]
        need = [s for s in cand if s not in E["si"]]
        if need and live_extras:
            _merge(E, enrich(need, latest, cmap, splits, budget(), kinds=("si", "floats")))
        for k, s in enumerate(cand, start=i + 1):
            if _gate_bad(s, latest, borrow, borrow_ok, E, min_usd=config.ONE_MIN_USD):
                one_held.append(s)
            else:
                one_sym, one_rank = s, k
                break
        if one_sym:
            break
    log.info("the ONE: %s (consensus rank %s; %d held back: %s)", one_sym, one_rank, len(one_held), ", ".join(one_held[:8]))
    dossier = list(dict.fromkeys(([one_sym] if one_sym else []) + board_pool + swing_pool + [t["symbol"] for t in tw]))
    # Stage B — the full dossier treatment for every name that is shown
    if live_extras:
        # stage A's short interest / float stay authoritative for the gated names
        # (re-fetching could blank them on a deadline and change who passes the gate)
        _merge(E, enrich(dossier, latest, cmap, splits, budget(), deep=([one_sym] if one_sym else []) + board_pool + swing_pool[:5],
                         kinds=("news", "analyst", "dilution", "insider", "chatter")))
        extras = [s for s in dossier if s not in E["si"]]
        if extras:
            _merge(E, enrich(extras, latest, cmap, splits, budget(), kinds=("si", "floats")))
    danel = _safe(street.danelfin, dossier[:15], default={}) or {}

    # attribution for everything we'll show
    attr = pd.DataFrame(index=dossier)
    if hasattr(model, "attribute"):
        # medians over the WHOLE scored universe = "a typical name today"
        a = _safe(model.attribute, bundle, latest, False, default=None)
        if a is not None and len(a):
            attr = a.reindex(dossier)

    # text-classify recent 8-K/6-K for the top of the board so offering language is caught
    todo = []
    for s in ([one_sym] if one_sym and one_sym not in board_pool else []) + board_pool:
        for e in _filings_for(s, events, overnight)[:4]:
            if e.get("form") in ("8-K", "6-K", "424B5", "424B3") and not e.get("text_tags") and e.get("url"):
                if (session - date.fromisoformat(e["date"])).days <= 14:
                    todo.append(e)
    t_t = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        tagged = list(ex.map(lambda e: _safe(sec.text_classify, e["url"], default=None), todo))
    for e, tags in zip(todo, tagged):
        if tags:
            e["text_tags"] = tags
            cat = _safe(sec.categorize, e.get("form") or "", e.get("items") or [], tags, default=None)
            if cat:
                e["category"] = cat
    log.info("text-classified %d recent filings in %.0fs", len(todo), time.time() - t_t)

    def attribution(sym: str) -> Optional[dict]:
        if sym not in attr.index:
            return None
        row = {c.replace("attr_", ""): (None if pd.isna(v) else round(float(v), 4))
               for c, v in attr.loc[sym].items() if str(c).startswith("attr_")}
        return row if any(v for v in row.values()) else None

    def make_pick(sym: str, rank: Optional[int], with_chart: bool) -> dict:
        r = latest.loc[sym]
        m = {k: (None if pd.isna(v) else v) for k, v in r.items() if not isinstance(v, (list, dict))}
        short = score.shortability(sym, borrow, borrow_ok)
        si_rows = E["si"].get(sym) or []
        fl = E["floats"].get(sym) or {}
        si_pct = None
        if si_rows and fl.get("float"):
            si_pct = (si_rows[0].get("interest") or 0) / fl["float"] if fl["float"] else None
        if si_pct is None:
            si_pct = fl.get("short_pct_float")
        dtc = si_rows[0].get("days_to_cover") if si_rows else None
        sqd, sq_parts = score.squeeze_danger(
            m.get("sq_pct"), short, si_pct, dtc, fl.get("float"), m.get("short_ratio_5d"))
        filings = _filings_for(sym, events, overnight)[:12]
        nws = E["news"].get(sym, [])[:8]
        reasons, flags = score.build_reasons(
            m, pre.get(sym), filings, splits.get(sym, []), nws, short, session, notes.get(sym))
        b_now = borrow.get(sym) if borrow_ok else None
        b_pt = [run_day, b_now.get("fee_rate"), b_now.get("available")] if b_now else None
        tight = history.borrow_tightening(sym, b_pt)
        if tight:
            reasons.append({"family": "flow", "text": tight, "strength": 2, "url": None})
            flags.append("BORROW TIGHTENING")
        dil = E["dilution"].get(sym)
        if dil and (dil.get("shares_growth_1y") or 0) >= 2:
            reasons.append({"family": "dilution", "text": f"Share count up {dil['shares_growth_1y']:.1f}× in a year (SEC filings)",
                            "strength": 2, "url": (dil.get("sources") or [None])[0]})
        if dil and dil.get("runway_q") is not None and dil["runway_q"] < 2:
            reasons.append({"family": "dilution", "text": f"Cash runway about {dil['runway_q']:.1f} quarters at the recent burn — likely to raise",
                            "strength": 2, "url": (dil.get("sources") or [None])[0]})
        reasons.sort(key=lambda x: -x["strength"])
        fams = {f: (None if pd.isna(r.get(f, np.nan)) else int(r.get(f))) for f in ("dilution", "exhaustion", "decay", "flow")}
        a = E["analyst"].get(sym) or None
        fams["street"] = _street_score(a)
        fams["news"] = _news_score(nws) if sym in E["news"] else None
        cik = cmap.get(sym, {}).get("cik")
        price = m.get("close")
        return {
            "rank": rank, "urank": urank.get(sym), "symbol": sym, "name": m.get("name"),
            "exchange": cmap.get(sym, {}).get("exchange"), "country": m.get("country"),
            "sector": m.get("sector"), "industry": m.get("industry"),
            "price": price, "prev_close": price, "market_cap": m.get("market_cap"),
            "premarket": pre.get(sym) if sym in pre else None,
            "prob_dump": m.get("prob_dump"), "prob_bigdump": m.get("prob_bigdump"),
            "prob_squeeze": m.get("prob_squeeze"), "exp_oc": m.get("exp_oc"),
            "prob_swing": m.get("prob_swing"), "prob_pump": m.get("prob_pump"), "skew": m.get("skew"),
            "liq_edge": m.get("liq_edge"), "lend_short": _lend_short(sym, borrow, borrow_ok, price),
            "lift": m.get("lift"), "score": None if m.get("score") is None else int(m["score"]),
            "model_used": m.get("model_used"),
            "ssr": bool(m.get("ssr")), "publishable": bool(m.get("publishable")),
            "squeeze_danger": sqd, "squeeze_parts": sq_parts,
            "shortability": short, "families": fams, "reasons": reasons, "flags": flags,
            "attribution": attribution(sym),
            "size": size_of(sym, m),
            "metrics": {
                k: m.get(k) for k in (
                    "r1", "r3", "r5", "r20", "rvol1", "rsi14", "dist_ma20", "dist_ma50", "dd_52w",
                    "vol20", "rs_count_2y", "n_offer_90", "n_reg_90", "short_ratio_5d")
            } | {
                "dvol20": m.get("dvol20"),
                "si_pct_float": si_pct, "days_to_cover": dtc,
                "float": fl.get("float"), "shares_out": fl.get("shares_out"),
                "short_interest": si_rows[0] if si_rows else None,
            },
            "filings": filings, "news": nws,
            "street": {"analyst": a, "danelfin": danel.get(sym)},
            "dilution": dil, "insider": E["insider"].get(sym), "chatter": E["chatter"].get(sym),
            "borrow_history": _with_point(history.borrow_history(sym), b_pt),
            "prob_history": _with_point(history.prob_history(sym), [today_iso, m.get("prob_dump")] if m.get("prob_dump") is not None else None),
            "links": _safe(street.deep_links, sym, cik, default={}) or {},
            "chart": score.chart_rows(hist.get(sym)) if with_chart else [],
        }

    # display-only pre-market quotes for every name that will be shown (the first
    # snapshot covered the top 150 by the close-only ranking)
    if want_pre:
        shown = [s for s in dict.fromkeys(([one_sym] if one_sym else []) + board_pool + swing_pool + [t["symbol"] for t in tw])
                 if s not in pre]
        left = _secs_until_bell(session) - 300.0
        if shown and left > 30:
            more = _safe(prices.premarket_snapshot, shown, min(120.0, left), default={}) or {}
            pre.update(more)
            log.info("pre-market snapshot (shown names): %d/%d", len(more), len(shown))

    board, squeeze_zone = [], []
    for sym in board_pool:
        p = make_pick(sym, None, with_chart=True)
        bad = (p["shortability"]["status"] == "NONE" or (p["squeeze_danger"] or 0) >= 70 or "HALTED" in p["flags"]
               or p.get("lend_short") is not None)
        if bad:
            if len(squeeze_zone) < 12:
                p["zone_reason"] = _zone_reason(p)
                squeeze_zone.append(p)
        elif len(board) < config.BOARD_SIZE:
            p["rank"] = len(board) + 1
            board.append(p)
    net.record_status("Board fill", len(board) >= config.BOARD_SIZE,
                      f"{len(board)}/{config.BOARD_SIZE} eligible names from the top {depth} of "
                      f"{int(latest['liq_ok'].sum())} ${config.BOARD_MIN_USD // 1000}K-ready names "
                      f"({len(squeeze_zone)} held back in the squeeze zone)")
    # The #1 on the main page is the ONE (R13): the best consensus among names that take a ONE_MIN_USD
    # short; if none passed its gates, the board's first name (publication rule, borrow, squeeze < 70).
    top = None
    one_rep = ((bundle.get("report") or {}).get("one") or {})
    if one_sym:
        on_board = next((p for p in board if p["symbol"] == one_sym), None)
        top = dict(on_board) if on_board is not None else make_pick(one_sym, None, with_chart=True)
        m_used = "m1" if latest.at[one_sym, "model_used"] == "m1" else "m0"
        dfc = _nn(latest.at[one_sym, "one_deficit"])
        tier = model.one_tier(dfc)
        rec = (one_rep.get(m_used) or {})
        liq_rows = latest[latest["liq_ok"]]
        part_rank = {c: int((pd.to_numeric(liq_rows[c], errors="coerce") > float(latest.at[one_sym, c])).sum()) + 1
                     for c in model.ONE_PARTS if pd.notna(latest.at[one_sym, c])}
        top["one"] = {"score": _nn(latest.at[one_sym, "one_score"]), "deficit": dfc, "tier": tier,
                      "consensus_rank": one_rank, "held_back": one_held, "part_ranks": part_rank,
                      "universe": int(latest["one_ok"].sum()), "liq_universe": int(latest["liq_ok"].sum()),
                      "min_usd": config.ONE_MIN_USD, "model": m_used,
                      "record": rec.get("overall"), "tier_record": (rec.get("tiers") or {}).get(tier) if tier else None}
        mv = ((top["one"]["tier_record"] or {}).get("mean_oc"))
        basis = f"the ONE ({tier} conviction)" if (mv is not None and mv < 0 and tier) else None
        if basis is None:
            mv = (rec.get("overall") or {}).get("mean_oc")
            basis = "the ONE" if (mv is not None and mv < 0) else "top-10"
        _row = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in latest.loc[one_sym].items()}
        top["size"] = size_of(one_sym, _row, (-float(mv)) if basis != "top-10" else None, basis) or top.get("size")
    if top is None:
        top = next((p for p in board if p.get("publishable")), None)
        if top is not None:
            top = dict(top)
            top["one"] = None
            if avg_move is not None:
                _row = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in latest.loc[top["symbol"]].items()}
                top["size"] = size_of(top["symbol"], _row, avg_move, "#1") or top.get("size")
    net.record_status("The ONE", one_sym is not None,
                      f"{one_sym} — consensus rank {one_rank} of {int(latest['one_ok'].sum())} ${config.ONE_MIN_USD // 1000}K-ready names"
                      + (f" ({len(one_held)} above it held back: borrow, IBKR lendable, squeeze or halt)" if one_held else "")
                      if one_sym else "no $100K-ready name passed the borrow / squeeze / halt checks — the board's #1 is shown")
    tie_n = 0
    if top is not None:
        tie_n = int((np.abs(latest["prob_dump"].to_numpy(float) - float(top["prob_dump"])) < 1e-4).sum())

    # lookalikes are anchored to the FINAL #1 (re-run if enrichment changed it)
    anchor = top["symbol"] if top else None
    if anchor and anchor != provisional:
        tw = twins.find_twins(feat_rows, ref=anchor, session_year=session.year)
        new_syms = [t["symbol"] for t in tw if t["symbol"] not in dossier]
        if new_syms and live_extras:
            _merge(E, enrich(new_syms, latest, cmap, splits, 120))
    rank_of = {p["symbol"]: p["rank"] for p in board}

    # how much could be spread across today's top 10 before estimated costs
    # exceed the historical average move of a top-10 pick (each name capped
    # at its own capacity)
    size_summary = _safe(_size_summary, board, top, avg_move, avg_move_10, CAP)
    twin_cards = []
    for t in tw:
        s = t["symbol"]
        if s not in latest.index:
            continue
        r = latest.loc[s]
        pick = make_pick(s, rank_of.get(s), with_chart=True)
        twin_cards.append({
            "symbol": s, "name": r.get("name"), "similarity": t["similarity"], "reasons": t["reasons"],
            "price": r.get("close"), "market_cap": r.get("market_cap"), "country": r.get("country"),
            "prob_dump": r.get("prob_dump"), "rank": rank_of.get(s),
            "score": None if pd.isna(r.get("score")) else int(r.get("score")),
            "flags": pick["flags"], "pick": pick,
        })

    swing_board = []
    for s in swing_pool:
        pk = make_pick(s, None, with_chart=True)
        # a week-long short sits through every squeeze: same exclusions as the board
        if (pk["shortability"]["status"] == "NONE" or (pk["squeeze_danger"] or 0) >= 70 or "HALTED" in pk["flags"]
                or pk.get("lend_short") is not None):
            continue
        pk["rank"] = len(swing_board) + 1
        pk["board_rank"] = rank_of.get(s)
        swing_board.append(pk)
        if len(swing_board) >= 15:
            break

    # payload: a swing / lookalike entry whose name is also on the board keeps only
    # its list fields — the site opens dossiers from the board record
    on_board = {p["symbol"] for p in board}
    _SLIM = ("rank", "urank", "board_rank", "symbol", "name", "exchange", "country", "sector", "price", "market_cap",
             "prob_dump", "prob_swing", "prob_squeeze", "prob_pump", "skew", "lift", "score", "model_used", "ssr",
             "publishable", "squeeze_danger", "shortability", "flags", "premarket")
    slim = lambda p: {k: p.get(k) for k in _SLIM if k in p}  # noqa: E731
    swing_board = [slim(p) if p["symbol"] in on_board else p for p in swing_board]
    for t in twin_cards:
        if t["symbol"] in on_board:
            t["pick"] = slim(t["pick"])

    wire = _catalyst_wire(overnight, E["news"], pre, latest, session)
    earnings = []
    for e in _safe(street.earnings_calendar, session, default=[]) or []:
        e["in_universe"] = e.get("symbol") in latest.index
        earnings.append(e)
    earnings.sort(key=lambda e: (not e["in_universe"], e.get("symbol") or ""))

    report = (bundle.get("report") or {})
    today = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_date": session.isoformat(),
        "features_asof": last_date.date().isoformat(),
        "run": run,
        "market_phase": phase,
        "universe": {"listed": int(len(state["universe"])), "eligible": int(len(state["latest"])), "scored": int(len(latest))},
        "model": {
            "version": "m1" if use_open_syms else "m0",
            "use_open": bool(use_open_syms), "m1_names": len(use_open_syms),
            "base_rate_dump": base,
            "trained_through": report.get("trained_through"),
            "oos_auc": ((report.get("oos") or {}).get("m1" if use_open_syms else "m0") or {}).get("auc"),
            "prob_cap": {m_: ((bundle.get(m_) or {}).get("dump") or {}).get("cap") for m_ in ("m0", "m1")},
            "publication_rule": (report.get("publication_rule") or {}).get("text"),
            "board_rule": (report.get("liquid") or {}).get("rule"),
            "board_min_usd": config.BOARD_MIN_USD,
            "board_ranker": "liq" if rank_col == "liq_edge" else "dump",
            "board_universe": int(latest["liq_ok"].sum()),
            "one_rule": (report.get("one") or {}).get("rule"),
            "one_min_usd": config.ONE_MIN_USD,
            "one_universe": int(latest["one_ok"].sum()),
        },
        "top_tie_count": tie_n,
        "size_summary": size_summary,
        "top": top,
        "board": board,
        "swing_board": swing_board,
        "squeeze_zone": squeeze_zone,
        "catalyst_wire": wire,
        "twins_anchor": anchor,
        "twins": twin_cards,
        "market": _market_block(latest, state),
        "sectors": _sector_block(latest),
        "calendar": _calendar_block(session, latest, hist),
        "earnings": earnings[:40],
        "sources": [st for st in net.STATUS.values() if "(optional)" not in str(st.get("detail", ""))],
        "disclaimer": DISCLAIMER,
    }
    # side files (full-universe lookup + history stores) are written only when
    # this page is actually published — see _publish_today / _flush_side
    pick_sqd = {p["symbol"]: p["squeeze_danger"] for p in board + swing_board + squeeze_zone + [t["pick"] for t in twin_cards]
                + ([top] if top else [])}
    _PENDING_SIDE.clear()
    _PENDING_SIDE.update({
        "universe": (latest, hist, borrow, borrow_ok, events, splits, rank_of, session, pre, notes, pick_sqd, size_of),
        "probs": (today_iso, last_date.date().isoformat(), latest["prob_dump"].copy()),
        "borrow": (run_day, borrow, list(latest.index)) if borrow_ok else None,
    })
    log.info("built today (%s run) for %s in %.0fs — #1 %s", run, session, time.time() - t0, (top or {}).get("symbol"))
    return clean(today)


_PENDING_SIDE: Dict[str, Any] = {}


def _with_point(series: List[list], pt: Optional[list]) -> List[list]:
    """History + today's not-yet-saved point (replacing a same-day entry)."""
    if not pt or pt[1] is None:
        return series
    return [r for r in series if r[0] != pt[0]] + [pt]


def _flush_side() -> None:
    """Write the deferred side files of the page that was just published."""
    from . import history
    side = dict(_PENDING_SIDE)
    _PENDING_SIDE.clear()
    if side.get("universe"):
        _safe(_write_universe, *side["universe"])
    if side.get("probs"):
        _safe(history.save_probs, *side["probs"])
    if side.get("borrow"):
        _safe(history.save_borrow, *side["borrow"])


def _market_block(latest: pd.DataFrame, state: dict) -> dict:
    """Today's tape vs history: breadth, extremes, and how 'dumpy' the
    universe's odds are compared with the last year of sessions."""
    def col(c):
        return pd.to_numeric(latest[c], errors="coerce") if c in latest.columns else pd.Series(dtype=float)

    r1 = col("r1")
    mean_p = float(col("prob_dump_m0").mean()) if len(latest) else None
    regime_pct, regime = None, None
    try:
        hist_means = json.loads(REGIME_PATH.read_text())
    except (OSError, ValueError):
        hist_means = {}
    if hist_means and mean_p is not None:
        vals = np.array(list(hist_means.values())[-250:], dtype=float)
        regime_pct = float((vals < mean_p).mean() * 100)
        regime = "wild" if regime_pct >= 80 else "calm" if regime_pct <= 20 else "normal"
    return {
        "breadth_up": _f(col("breadth_up").median()) if "breadth_up" in latest.columns else _f((r1 > 0).mean()),
        "median_r1": _f(r1.median()),
        "n_up20": int((r1 >= 0.20).sum()), "n_down20": int((r1 <= -0.20).sum()),
        "iwm_r1": _f(col("iwm_r1").median()), "iwm_r5": _f(col("iwm_r5").median()),
        "mean_prob_dump": mean_p,
        "universe_dump_rate_20d": state.get("universe_dump_rate_20d"),
        "regime": regime, "regime_pct": regime_pct,
    }


def _sector_block(latest: pd.DataFrame) -> list:
    df = latest[["sector", "prob_dump"]].copy()
    df["sector"] = df["sector"].fillna("").replace("", "Other")
    top100 = set(latest.sort_values("prob_dump", ascending=False).index[:100])
    out = []
    for sec_name, g in df.groupby("sector"):
        if len(g) < 5:
            continue
        out.append({"sector": sec_name, "n": int(len(g)), "mean_prob": _f(g["prob_dump"].mean()),
                    "n_top100": int(sum(1 for s in g.index if s in top100)),
                    "top": list(g.sort_values("prob_dump", ascending=False).index[:3])})
    return sorted(out, key=lambda x: -(x["mean_prob"] or 0))


def _calendar_block(session: date, latest: pd.DataFrame, hist: dict) -> dict:
    intel = _intel()
    earn, locks = [], []
    if intel is not None:
        for e in _safe(intel.earnings_ahead, session, 5, default=[]) or []:
            s = e.get("symbol")
            e["in_universe"] = s in latest.index
            e["name"] = e.get("name") or (latest.at[s, "name"] if s in latest.index else None)
            e["prob_dump"] = _f(latest.at[s, "prob_dump"]) if s in latest.index else None
            earn.append(e)
        for lk in _safe(intel.lockups, session, default=[]) or []:
            s = lk.get("symbol")
            df = hist.get(s)
            price = _f(df["close"].iloc[-1]) if df is not None and len(df) else None
            lk["price"] = price
            lk["vs_ipo"] = (price / lk["ipo_price"] - 1) if (price and lk.get("ipo_price")) else None
            lk["prob_dump"] = _f(latest.at[s, "prob_dump"]) if s in latest.index else None
            locks.append(lk)
    earn.sort(key=lambda e: (e.get("date") or "", not e["in_universe"], -(e.get("prob_dump") or 0)))
    locks.sort(key=lambda x: x.get("days_to") if x.get("days_to") is not None else 999)
    return {"earnings": [e for e in earn if e["in_universe"]][:80], "lockups": locks[:40]}


def _write_universe(latest, hist, borrow, borrow_ok, events, splits, rank_of, session, pre, notes, pick_sqd=None, size_of=None) -> None:
    cols = ["symbol", "name", "price", "market_cap", "prob_dump", "prob_squeeze", "prob_swing", "skew", "score",
            "board_rank", "ssr", "publishable", "borrow_status", "fee_rate", "available", "squeeze_danger",
            "sector", "country", "flags", "spark", "capacity", "breakeven", "cost_50k", "cost_500k"]
    rows = []
    for s, r in latest.iterrows():
        m = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.items()}
        short = score.shortability(s, borrow, borrow_ok)
        if pick_sqd and s in pick_sqd:
            sqd = pick_sqd[s]          # the full score shown on the board/dossier
        else:                          # partial: no short-interest / float lookups for the long tail
            sqd, _ = score.squeeze_danger(m.get("sq_pct"), short, None, None, None, m.get("short_ratio_5d"))
        _, flags = score.build_reasons(m, pre.get(s), _filings_for(s, events, {})[:8], splits.get(s, []),
                                       [], short, session, notes.get(s))
        df = hist.get(s)
        spark = [round(float(x), 4) for x in df["close"].tail(30)] if df is not None and len(df) else []
        rows.append([s, m.get("name"), _f(m.get("close")), _f(m.get("market_cap")), _f(m.get("prob_dump")),
                     _f(m.get("prob_squeeze")), _f(m.get("prob_swing")), _f(m.get("skew")),
                     None if m.get("score") is None else int(m["score"]), rank_of.get(s),
                     bool(m.get("ssr")), bool(m.get("publishable")), short["status"], short.get("fee_rate"),
                     short.get("available"), sqd, m.get("sector"), m.get("country"), flags, spark]
                    + _size_cols(size_of(s, m) if size_of else None))
    rows.sort(key=lambda x: -(x[4] or 0))
    publish.write_json("universe.json", {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "session_date": session.isoformat(), "columns": cols, "rows": rows})


def _size_summary(board: list, top: Optional[dict], avg_move: Optional[float], avg_move_10: Optional[float], CAP) -> Optional[dict]:
    """How much could be spread across today's top 10 ELIGIBLE names (no SSR,
    ≥ $300K/day) before estimated costs exceed a top-10 pick's historical
    average move (each name capped at its own capacity; names that can't hold
    $1K are not counted)."""
    if avg_move_10 is None:
        return None
    tot, used = 0.0, 0
    elig = [p for p in board if p.get("publishable")][:10]      # the rule the backtest measured
    for p in elig:
        sz = p.get("size") or {}
        if not sz.get("exp_dvol"):
            continue
        be = sz.get("breakeven")
        if be is None:
            be = CAP.breakeven_size(avg_move_10, sz["exp_dvol"], (p.get("metrics") or {}).get("vol20"), sz.get("spread"))
        m = min(be, sz.get("capacity") or 0.0)
        if m >= 1000:                                             # a name that can't hold $1K adds nothing
            tot += m
            used += 1
    tsz = (top or {}).get("size") or {}
    return {"names": used, "deployable": tot, "avg_move_top10": avg_move_10, "avg_move_top1": avg_move,
            "top_move": tsz.get("avg_move"), "top_move_basis": tsz.get("move_basis"),
            "top_capacity": ((top or {}).get("size") or {}).get("capacity"),
            "top_breakeven": ((top or {}).get("size") or {}).get("breakeven")}


def _size_cols(sz: Optional[dict]) -> list:
    if not sz:
        return [None, None, None, None]
    c = sz.get("costs") or {}
    return [_f(sz.get("capacity")), _f(sz.get("breakeven")), _f(c.get("50000")), _f(c.get("500000"))]


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def _street_score(a: Optional[dict]) -> Optional[int]:
    if not a or not a.get("mean_rating"):
        return None
    mp = {"strong sell": 100, "sell": 80, "underperform": 75, "hold": 50, "neutral": 50,
          "buy": 25, "outperform": 30, "strong buy": 10}
    base = mp.get(str(a["mean_rating"]).lower())
    if base is None:
        return None
    downs = sum(1 for c in (a.get("changes") or []) if "down" in str(c.get("action", "")).lower())
    return int(min(100, base + 10 * downs))


def _news_score(items: List[dict]) -> Optional[int]:
    if not items:
        return 0
    neg = sum(1 for n in items if n.get("polarity", 0) < 0)
    pos = sum(1 for n in items if n.get("polarity", 0) > 0)
    return int(max(0, min(100, 50 + 20 * neg - 15 * pos))) if (neg or pos) else 50


def _zone_reason(p: dict) -> str:
    sh = p["shortability"]
    if sh["status"] == "NONE":
        return "No shares available to borrow at IBKR — you likely can't short it."
    if p.get("lend_short") is not None:
        return (f"IBKR can lend only about ${p['lend_short']:,.0f} — less than a "
                f"${config.BOARD_MIN_USD:,.0f} short needs.")
    if "HALTED" in p["flags"]:
        return "Halted / no trades last session."
    parts = p.get("squeeze_parts") or {}
    bits = []
    if (parts.get("fee") or 0) >= 0.5:
        bits.append(f"borrow fee {sh.get('fee_rate', 0):.0f}%/yr")
    if (parts.get("scarcity") or 0) >= 0.6:
        bits.append("very few shares to borrow")
    if (parts.get("float") or 0) >= 0.6:
        bits.append("tiny float")
    if (parts.get("si") or 0) >= 0.6:
        bits.append("crowded short interest")
    if (parts.get("model") or 0) >= 0.8:
        bits.append("model sees high odds of a +20% intraday spike")
    return "Squeeze danger " + str(p["squeeze_danger"]) + "/100: " + (", ".join(bits) or "multiple crowding signals")


def _ts(x: Any) -> float:
    """ISO string (any offset, or bare date) → epoch seconds; unknown → 0."""
    if not x:
        return 0.0
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if d.tzinfo is None:
        d = d.replace(tzinfo=ET)
    return d.timestamp()


def _catalyst_wire(overnight, news, pre, latest, session) -> list:
    sev = {"offering": 3, "atm": 3, "toxic_financing": 3, "unregistered_sale": 2, "delisting_notice": 2,
           "reverse_split": 2, "registration": 2, "resale": 2, "effective": 2, "going_concern": 2, "late_filing": 1}
    wire = []
    for sym, evs in overnight.items():
        if sym not in latest.index:
            continue
        for e in evs:
            cat = e.get("category")
            if cat not in sev:
                continue
            wire.append({
                "time": e.get("accepted") or e.get("date"), "symbol": sym, "kind": score.CAT_LABEL.get(cat, cat),
                "headline": f"{e.get('form')} — {score.CAT_LABEL.get(cat, cat)}" + (f" ({', '.join(e['text_tags'])})" if e.get("text_tags") else ""),
                "url": e.get("url"), "source": "SEC EDGAR", "severity": sev[cat],
            })
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=20)).isoformat()
    for sym, items in news.items():
        for n in items:
            if n.get("polarity", 0) < 0 and _ts(n.get("published")) >= _ts(cutoff):
                wire.append({"time": n.get("published"), "symbol": sym, "kind": "News",
                             "headline": n.get("title"), "url": n.get("url"), "source": n.get("source"),
                             "severity": 2 if set(n.get("tags", [])) & {"offering", "priced", "reverse_split", "delisting"} else 1})
    for sym, p in pre.items():
        g = p.get("gap_pct")
        if g is not None and abs(g) >= 0.2 and sym in latest.index:
            wire.append({"time": p.get("asof"), "symbol": sym, "kind": "Pre-market",
                         "headline": f"Pre-market {g * 100:+.0f}% vs last close", "url": None, "source": p.get("source"),
                         "severity": 2 if g >= 0.4 else 1})
    wire.sort(key=lambda w: _ts(w.get("time")), reverse=True)
    # de-dup identical headlines
    seen, out = set(), []
    for w in wire:
        k = (w["symbol"], w["headline"])
        if k not in seen:
            seen.add(k)
            out.append(w)
    # with 100+ names the news can crowd out SEC supply filings: keep the most
    # severe first, then show the survivors newest first
    keep = sorted(out, key=lambda w: (-(w.get("severity") or 0), -_ts(w.get("time"))))[:80]
    keep.sort(key=lambda w: _ts(w.get("time")), reverse=True)
    return keep


# ── entry ────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gravity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    e = sub.add_parser("evening")
    e.add_argument("--retrain", action="store_true")
    e.add_argument("--no-push", action="store_true")
    m = sub.add_parser("morning")
    m.add_argument("--no-push", action="store_true")
    sub.add_parser("publish")
    lv = sub.add_parser("live")
    lv.add_argument("--no-push", action="store_true")
    args = ap.parse_args(argv)

    logfile = config.LOGS / f"{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cmds = {"train": cmd_train, "evening": cmd_evening, "morning": cmd_morning, "publish": cmd_publish, "live": cmd_live}
    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
