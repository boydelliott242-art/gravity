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
from .util import ET, clean, market_phase, now_et, prev_trading_day, target_session

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
def cmd_train(_args: argparse.Namespace) -> int:
    from . import evidence, model
    state = refresh_data()
    panel = state["panel"]
    report = model.train(panel)
    publish.write_json("model.json", report)
    ev = evidence.run_studies(panel, listing_dates=_listing_dates(state["hist"]))
    publish.write_json("evidence.json", ev)
    compute_signs(panel)
    _safe(update_regime, panel)
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
    publish.write_json("today.json", today)
    _flush_side()
    if not today["late"]:
        scorecard.log_picks(today)
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
    n = scorecard.grade(state["hist"], elig)
    log.info("graded %d session(s)", n)
    publish.write_json("scorecard.json", scorecard.summary())
    if args.retrain or _model_stale(9):
        try:
            log.info("model stale — retraining (walk-forward)")
            publish.write_json("model.json", model.train(state["panel"]))
            publish.write_json("evidence.json", evidence.run_studies(state["panel"], listing_dates=_listing_dates(state["hist"])))
            compute_signs(state["panel"])
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


def enrich(symbols: List[str], latest: pd.DataFrame, cmap: dict, splits: dict,
           deadline_s: float, deep: Optional[List[str]] = None) -> dict:
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
    F: Dict[str, Dict[str, Any]] = {k: {} for k in ("news", "si", "analyst", "dilution", "insider")}
    for s in symbols:
        name = str(latest.at[s, "name"] or "") if s in latest.index and "name" in latest.columns else ""
        cik = (cmap.get(s) or {}).get("cik")
        F["news"][s] = ex.submit(_safe, newsmod.headlines, s, name, default=[])
        F["si"][s] = ex.submit(_safe, shortside.short_interest, s, default=[])
        F["analyst"][s] = ex.submit(_safe, street.analyst, s)
        if intel is not None and cik:
            F["dilution"][s] = ex.submit(_safe, intel.dilution_intel, s, cik, splits.get(s, []))
            if s in deep:
                F["insider"][s] = ex.submit(_safe, intel.insider_sales, s, cik)
    f_fl = ex.submit(_safe, shortside.float_shares, symbols, default={})
    f_ch = ex.submit(_safe, intel.chatter, symbols, default={}) if intel is not None else None
    allf = [f for d in F.values() for f in d.values()] + [f_fl] + ([f_ch] if f_ch else [])
    _done, pending = wait(allf, timeout=deadline_s)
    ex.shutdown(wait=False, cancel_futures=True)

    def res(f, default):
        return (f.result() if f is not None and f.done() and not f.cancelled() else None) or default

    out = {k: {s: res(f, [] if k in ("news", "si") else None) for s, f in d.items()} for k, d in F.items()}
    out["floats"] = res(f_fl, {})
    out["chatter"] = res(f_ch, {}) if f_ch else {}
    if pending:
        net.record_status("Enrichment deadline", False, f"{len(pending)} lookups unfinished after {deadline_s:.0f}s — shown as missing")
    log.info("enrichment for %d names in %.0fs (%d unfinished)", len(symbols), time.time() - t_e, len(pending))
    return out


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            dst.setdefault(k, {}).update(v)


def build_today(state: dict, run: str, live_extras: bool = True) -> dict:
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
    OUT = ["prob_dump", "prob_bigdump", "prob_squeeze", "exp_oc", "prob_swing", "prob_pump", "skew"]
    pred = model.predict(bundle, latest, use_open=False)
    latest = latest.join(pred[[c for c in OUT if c in pred.columns]], rsuffix="_m")
    for c in OUT:
        if c not in latest.columns:
            latest[c] = np.nan
    latest["model_used"] = "m0"
    latest["prob_dump_m0"] = latest["prob_dump"]

    # 3) shortlist → pre-market snapshot → M1 where we have a gap proxy
    order = latest.sort_values("prob_dump", ascending=False)
    shortlist = list(order.index[:300])
    for s, evs in overnight.items():
        if s in latest.index and any(e.get("category") in score.SUPPLY_CATS for e in evs) and s not in shortlist:
            shortlist.append(s)
    pre: Dict[str, dict] = {}
    if phase == "pre-market" or run == "morning":
        t_p = time.time()
        pre = _safe(prices.premarket_snapshot, shortlist, default={}) or {}
        log.info("pre-market snapshot: %d/%d names in %.0fs", len(pre), len(shortlist), time.time() - t_p)
    use_open_syms = [s for s in shortlist if s in pre and pre[s].get("gap_pct") is not None and phase == "pre-market"]
    if use_open_syms:
        sub = latest.loc[use_open_syms].copy()
        sub["gap_open"] = [pre[s]["gap_pct"] for s in use_open_syms]
        p1 = model.predict(bundle, sub, use_open=True)
        for c in OUT:
            if c in p1.columns:
                latest.loc[use_open_syms, c] = p1[c].values
        latest.loc[use_open_syms, "model_used"] = "m1"

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
    borrow_ok = len(borrow) > 1000
    sv = _safe(shortside.finra_short_volume, 5, default=None)
    if sv is not None and len(sv):
        g = sv.groupby("symbol")[["short_volume", "total_volume"]].sum()
        latest["short_ratio_5d"] = (g["short_volume"] / g["total_volume"].replace(0, np.nan)).reindex(latest.index)
    else:
        latest["short_ratio_5d"] = np.nan
    latest["borrow_status"] = [score.shortability(s, borrow, borrow_ok)["status"] for s in latest.index]
    today_iso, run_day = session.isoformat(), now_et().date().isoformat()

    # 6) families (descriptive) + model attribution (what the model actually weighed)
    feature_docs = getattr(features, "FEATURE_DOCS", {})
    try:
        signs = json.loads(SIGNS_PATH.read_text())
    except (OSError, ValueError):
        signs = {}
    latest = latest.join(score.family_scores(latest, feature_docs, signs))

    # 7) candidates: board pool, swing pool, provisional #1 (for twins)
    order = latest.sort_values("prob_dump", ascending=False)
    board_pool = list(order.index[: config.DOSSIER_SIZE])
    shortable = latest["borrow_status"].isin(["ETB", "HTB", "UNKNOWN"])
    swing_order = latest[shortable & latest["prob_swing"].notna()].sort_values("prob_swing", ascending=False)
    swing_pool = list(swing_order.index[:25])
    provisional = next((s for s in order.index if latest.at[s, "publishable"] and latest.at[s, "borrow_status"] in ("ETB", "HTB")), None)
    feat_rows = latest.copy()
    tw = twins.find_twins(feat_rows, ref=provisional, session_year=session.year) if provisional else []
    dossier = list(dict.fromkeys(board_pool + swing_pool + [t["symbol"] for t in tw]))

    E: Dict[str, Dict[str, Any]] = {k: {} for k in ("news", "si", "analyst", "dilution", "insider", "floats", "chatter")}
    if live_extras:
        _merge(E, enrich(dossier, latest, cmap, splits, ENRICH_DEADLINE_S, deep=board_pool[:30] + swing_pool[:5]))
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
    for s in board_pool[:30]:
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
            "rank": rank, "symbol": sym, "name": m.get("name"),
            "exchange": cmap.get(sym, {}).get("exchange"), "country": m.get("country"),
            "sector": m.get("sector"), "industry": m.get("industry"),
            "price": price, "prev_close": price, "market_cap": m.get("market_cap"),
            "premarket": pre.get(sym) if sym in pre else None,
            "prob_dump": m.get("prob_dump"), "prob_bigdump": m.get("prob_bigdump"),
            "prob_squeeze": m.get("prob_squeeze"), "exp_oc": m.get("exp_oc"),
            "prob_swing": m.get("prob_swing"), "prob_pump": m.get("prob_pump"), "skew": m.get("skew"),
            "lift": m.get("lift"), "score": None if m.get("score") is None else int(m["score"]),
            "model_used": m.get("model_used"),
            "ssr": bool(m.get("ssr")), "publishable": bool(m.get("publishable")),
            "squeeze_danger": sqd, "squeeze_parts": sq_parts,
            "shortability": short, "families": fams, "reasons": reasons, "flags": flags,
            "attribution": attribution(sym),
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

    board, squeeze_zone = [], []
    for sym in board_pool:
        p = make_pick(sym, None, with_chart=True)
        bad = p["shortability"]["status"] == "NONE" or (p["squeeze_danger"] or 0) >= 70 or "HALTED" in p["flags"]
        if bad:
            if len(squeeze_zone) < 12:
                p["zone_reason"] = _zone_reason(p)
                squeeze_zone.append(p)
        elif len(board) < config.BOARD_SIZE:
            p["rank"] = len(board) + 1
            board.append(p)
    # The #1 follows the publication rule the backtest measured: no Rule 201
    # restriction and ≥ $300k median daily volume (plus borrow + squeeze < 70).
    top = next((p for p in board if p.get("publishable")), None)
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
        if pk["shortability"]["status"] == "NONE" or (pk["squeeze_danger"] or 0) >= 70 or "HALTED" in pk["flags"]:
            continue
        pk["rank"] = len(swing_board) + 1
        pk["board_rank"] = rank_of.get(s)
        swing_board.append(pk)
        if len(swing_board) >= 15:
            break

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
        },
        "top_tie_count": tie_n,
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
    pick_sqd = {p["symbol"]: p["squeeze_danger"] for p in board + swing_board + squeeze_zone + [t["pick"] for t in twin_cards]}
    _PENDING_SIDE.clear()
    _PENDING_SIDE.update({
        "universe": (latest, hist, borrow, borrow_ok, events, splits, rank_of, session, pre, notes, pick_sqd),
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


def _write_universe(latest, hist, borrow, borrow_ok, events, splits, rank_of, session, pre, notes, pick_sqd=None) -> None:
    cols = ["symbol", "name", "price", "market_cap", "prob_dump", "prob_squeeze", "prob_swing", "skew", "score",
            "board_rank", "ssr", "publishable", "borrow_status", "fee_rate", "available", "squeeze_danger",
            "sector", "country", "flags", "spark"]
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
                     short.get("available"), sqd, m.get("sector"), m.get("country"), flags, spark])
    rows.sort(key=lambda x: -(x[4] or 0))
    publish.write_json("universe.json", {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "session_date": session.isoformat(), "columns": cols, "rows": rows})


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
    return out[:80]


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
