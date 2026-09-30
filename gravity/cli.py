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
ENRICH_DEADLINE_S = 300

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
    if config.REFERENCE_SYMBOL not in syms:
        syms.append(config.REFERENCE_SYMBOL)
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

    panel = features.build_panel(hist, events, splits, static, bench)
    log.info("panel: %d rows × %d cols (%.0fs)", len(panel), panel.shape[1], time.time() - t0)
    ctx = {"universe": uni, "static": static, "splits": splits, "events": events, "cik_map": cmap,
           "built_at": datetime.now(timezone.utc).isoformat()}
    with open(PANEL_PATH, "wb") as f:
        pickle.dump(panel, f, protocol=pickle.HIGHEST_PROTOCOL)
    latest = features.latest_rows(panel)
    latest.to_pickle(LATEST_PATH)
    with open(CONTEXT_PATH, "wb") as f:
        pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"hist": hist, "panel": panel, "latest": latest, **ctx}


def load_state() -> Optional[dict]:
    from .sources import prices
    if not (LATEST_PATH.exists() and CONTEXT_PATH.exists()):
        return None
    latest = pd.read_pickle(LATEST_PATH)
    with open(CONTEXT_PATH, "rb") as f:
        ctx = pickle.load(f)
    hist = prices.load_history(list(set(latest["symbol"]) | {config.REFERENCE_SYMBOL}), refresh=False)
    return {"latest": latest, "hist": hist, **ctx}


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
    ev = evidence.run_studies(panel)
    publish.write_json("evidence.json", ev)
    compute_signs(panel)
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


def cmd_evening(args: argparse.Namespace) -> int:
    from . import evidence, model
    state = refresh_data()
    elig = state["latest"]["symbol"].tolist()
    n = scorecard.grade(state["hist"], elig)
    log.info("graded %d session(s)", n)
    publish.write_json("scorecard.json", scorecard.summary())
    if args.retrain or _model_stale():
        log.info("model stale — retraining (walk-forward)")
        publish.write_json("model.json", model.train(state["panel"]))
        publish.write_json("evidence.json", evidence.run_studies(state["panel"]))
        compute_signs(state["panel"])
    today = build_today(state, run="evening", live_extras=True)
    publish.write_json("today.json", today)
    scorecard.log_picks(today)
    if not args.no_push:
        publish.push(f"evening: watchlist for {today['session_date']}")
    return 0


def cmd_morning(args: argparse.Namespace) -> int:
    state = load_state()
    session = target_session()
    stale = state is None or pd.Timestamp(state["latest"]["date"].max()).date() < prev_trading_day(session)
    if stale:
        log.info("evening state missing/stale — running the full refresh first")
        state = refresh_data()
        publish.write_json("scorecard.json", scorecard.summary())
    today = build_today(state, run="morning", live_extras=True)
    # A pick published after the open can't be graded fairly — show it,
    # but keep it out of the track record (the evening watchlist stands).
    if market_phase() != "pre-market":
        today["late"] = True
        log.info("morning run after the open (%s) — not logged to the track record", market_phase())
    publish.write_json("today.json", today)
    if not today.get("late"):
        scorecard.log_picks(today)
    if not args.no_push:
        top = (today.get("top") or {}).get("symbol", "none")
        publish.push(f"morning: {today['session_date']} #1 {top}")
    return 0


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


def build_today(state: dict, run: str, live_extras: bool = True) -> dict:
    from . import features, model
    from .sources import news as newsmod
    from .sources import prices, sec, shortside, street

    t0 = time.time()
    session = target_session()
    phase = market_phase()
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
    pred = model.predict(bundle, latest, use_open=False)
    latest = latest.join(pred[["prob_dump", "prob_bigdump", "prob_squeeze", "exp_oc"]], rsuffix="_m")
    latest["model_used"] = "m0"

    # 3) shortlist → pre-market snapshot → M1 where we have a gap proxy
    order = latest.sort_values("prob_dump", ascending=False)
    shortlist = list(order.index[:150])
    for s, evs in overnight.items():
        if s in latest.index and any(e.get("category") in score.SUPPLY_CATS for e in evs) and s not in shortlist:
            shortlist.append(s)
    if config.REFERENCE_SYMBOL in latest.index and config.REFERENCE_SYMBOL not in shortlist:
        shortlist.append(config.REFERENCE_SYMBOL)
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
        for c in ("prob_dump", "prob_bigdump", "prob_squeeze", "exp_oc"):
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
    latest["vol20_pct"] = pd.to_numeric(latest.get("vol20"), errors="coerce").rank(pct=True)

    # 5) borrow + short volume for everyone (cheap bulk files)
    borrow = _safe(shortside.ibkr_borrow, default={}) or {}
    borrow_ok = len(borrow) > 1000
    sv = _safe(shortside.finra_short_volume, 5, default=None)
    if sv is not None and len(sv):
        g = sv.groupby("symbol")[["short_volume", "total_volume"]].sum()
        latest["short_ratio_5d"] = (g["short_volume"] / g["total_volume"].replace(0, np.nan)).reindex(latest.index)
    else:
        latest["short_ratio_5d"] = np.nan

    # 6) families
    feature_docs = getattr(features, "FEATURE_DOCS", {})
    try:
        signs = json.loads(SIGNS_PATH.read_text())
    except (OSError, ValueError):
        signs = {}
    fam = score.family_scores(latest, feature_docs, signs)
    latest = latest.join(fam)

    # 7) rank + dossier enrichment for the names we'll show
    order = latest.sort_values("prob_dump", ascending=False)
    dossier = list(order.index[: config.DOSSIER_SIZE])
    feat_rows = latest.copy()
    feat_rows["market_cap"] = uni["market_cap"].reindex(feat_rows.index)
    tw = twins.find_twins(feat_rows, session_year=session.year)
    extra = [t["symbol"] for t in tw] + [config.REFERENCE_SYMBOL]
    for s in extra:
        if s not in dossier and s in latest.index:
            dossier.append(s)

    news: Dict[str, list] = {}
    si: Dict[str, list] = {}
    analyst: Dict[str, Any] = {}
    floats: Dict[str, dict] = {}
    if live_extras:
        # Each source is throttled per host inside net.get, so overlapping
        # them only hides latency (Nasdaq answers in ~2-3 s per call).
        # Hard deadline: a slow feed must never hold the pre-market publish
        # hostage — anything unfinished is shown as missing ("—").
        t_e = time.time()
        ex = ThreadPoolExecutor(max_workers=12)
        f_news = {s: ex.submit(_safe, newsmod.headlines, s, str(latest.at[s, "name"] or ""), default=[]) for s in dossier}
        f_si = {s: ex.submit(_safe, shortside.short_interest, s, default=[]) for s in dossier}
        f_an = {s: ex.submit(_safe, street.analyst, s) for s in dossier}
        f_fl = ex.submit(_safe, shortside.float_shares, dossier, default={})
        allf = list(f_news.values()) + list(f_si.values()) + list(f_an.values()) + [f_fl]
        _done, pending = wait(allf, timeout=ENRICH_DEADLINE_S)
        ex.shutdown(wait=False, cancel_futures=True)

        def res(f, default):
            return (f.result() if f.done() and not f.cancelled() else None) or default

        news = {s: res(f, []) for s, f in f_news.items()}
        si = {s: res(f, []) for s, f in f_si.items()}
        analyst = {s: res(f, None) for s, f in f_an.items()}
        floats = res(f_fl, {})
        if pending:
            net.record_status("Enrichment deadline", False, f"{len(pending)} lookups unfinished after {ENRICH_DEADLINE_S}s — shown as missing")
        log.info("enrichment for %d names in %.0fs (%d unfinished)", len(dossier), time.time() - t_e, len(pending))
    danel = _safe(street.danelfin, dossier[:15], default={}) or {}
    # text-classify recent 8-K/6-K for the top of the board so offering language is caught
    events = state.get("events", {})
    todo = []
    for s in dossier[:30]:
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

    cmap = state.get("cik_map", {})
    splits = state.get("splits", {})
    hist = state["hist"]

    def make_pick(sym: str, rank: Optional[int], with_chart: bool) -> dict:
        r = latest.loc[sym]
        m = {k: (None if pd.isna(v) else v) for k, v in r.items() if not isinstance(v, (list, dict))}
        short = score.shortability(sym, borrow, borrow_ok)
        si_rows = si.get(sym) or []
        fl = floats.get(sym) or {}
        si_pct = None
        if si_rows and fl.get("float"):
            si_pct = (si_rows[0].get("interest") or 0) / fl["float"] if fl["float"] else None
        if si_pct is None:
            si_pct = fl.get("short_pct_float")
        dtc = si_rows[0].get("days_to_cover") if si_rows else None
        sqd, sq_parts = score.squeeze_danger(
            m.get("sq_pct"), short, si_pct, dtc, fl.get("float"), m.get("short_ratio_5d"))
        filings = _filings_for(sym, events, overnight)[:12]
        nws = news.get(sym, [])[:8]
        reasons, flags = score.build_reasons(
            m, pre.get(sym), filings, splits.get(sym, []), nws, short, session, notes.get(sym))
        fams = {f: (None if pd.isna(r.get(f, np.nan)) else int(r.get(f))) for f in ("dilution", "exhaustion", "decay", "flow")}
        a = analyst.get(sym) or None
        fams["street"] = _street_score(a)
        fams["news"] = _news_score(nws) if sym in news else None
        cik = cmap.get(sym, {}).get("cik")
        price = m.get("close")
        pick = {
            "rank": rank, "symbol": sym, "name": m.get("name"),
            "exchange": cmap.get(sym, {}).get("exchange"), "country": m.get("country"),
            "sector": m.get("sector"), "industry": m.get("industry"),
            "price": price, "prev_close": price, "market_cap": m.get("market_cap"),
            "premarket": pre.get(sym) if sym in pre else None,
            "prob_dump": m.get("prob_dump"), "prob_bigdump": m.get("prob_bigdump"),
            "prob_squeeze": m.get("prob_squeeze"), "exp_oc": m.get("exp_oc"),
            "lift": m.get("lift"), "score": None if m.get("score") is None else int(m["score"]),
            "model_used": m.get("model_used"),
            "squeeze_danger": sqd, "squeeze_parts": sq_parts,
            "shortability": short, "families": fams, "reasons": reasons, "flags": flags,
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
            "links": _safe(street.deep_links, sym, cik, default={}) or {},
            "chart": score.chart_rows(hist.get(sym)) if with_chart else [],
            "inhd_similarity": sim_map.get(sym),
        }
        return pick

    sim_map = {t["symbol"]: t["similarity"] for t in tw}
    sim_map[config.REFERENCE_SYMBOL] = 1.0

    board, squeeze_zone = [], []
    for sym in order.index[: config.DOSSIER_SIZE]:
        p = make_pick(sym, None, with_chart=True)
        bad = p["shortability"]["status"] == "NONE" or (p["squeeze_danger"] or 0) >= 70 or "HALTED" in p["flags"]
        if bad:
            if len(squeeze_zone) < 12:
                p["zone_reason"] = _zone_reason(p)
                squeeze_zone.append(p)
        elif len(board) < config.BOARD_SIZE:
            p["rank"] = len(board) + 1
            board.append(p)
    top = board[0] if board else None

    rank_of = {p["symbol"]: p["rank"] for p in board}
    twin_cards = []
    for t in tw:
        s = t["symbol"]
        if s not in latest.index:
            continue
        r = latest.loc[s]
        twin_cards.append({
            "symbol": s, "name": r.get("name"), "similarity": t["similarity"], "reasons": t["reasons"],
            "price": r.get("close"), "market_cap": r.get("market_cap"), "country": r.get("country"),
            "prob_dump": r.get("prob_dump"), "rank": rank_of.get(s),
            "score": None if pd.isna(r.get("score")) else int(r.get("score")),
            "pick": make_pick(s, rank_of.get(s), with_chart=True),
        })
        twin_cards[-1]["flags"] = twin_cards[-1]["pick"]["flags"]

    reference = _reference(latest, hist, borrow, borrow_ok, make_pick, rank_of, events, overnight, news, splits)
    wire = _catalyst_wire(overnight, news, pre, latest, session)
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
        },
        "top": top,
        "board": board,
        "squeeze_zone": squeeze_zone,
        "catalyst_wire": wire,
        "twins": twin_cards,
        "reference": reference,
        "earnings": earnings[:40],
        "sources": [st for st in net.STATUS.values() if "(optional)" not in str(st.get("detail", ""))],
        "disclaimer": DISCLAIMER,
    }
    log.info("built today (%s run) for %s in %.0fs — #1 %s", run, session, time.time() - t0, (top or {}).get("symbol"))
    return clean(today)


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


def _reference(latest, hist, borrow, borrow_ok, make_pick, rank_of, events, overnight, news, state_splits=None) -> dict:
    state_splits = state_splits or {}
    sym = config.REFERENCE_SYMBOL
    df = hist.get(sym)
    ref_date = config.REFERENCE_SHORT_DATE
    ref_price = None
    if df is not None and len(df):
        on = df[df.index <= pd.Timestamp(ref_date)]
        if len(on):
            ref_price = float(on["close"].iloc[-1])
    price = float(df["close"].iloc[-1]) if df is not None and len(df) else None
    pick = make_pick(sym, rank_of.get(sym), with_chart=False) if sym in latest.index else None
    notes = []
    if pick and pick["shortability"].get("fee_rate") is not None:
        notes.append(f"Borrow currently costs about {pick['shortability']['fee_rate']:.0f}% a year at IBKR "
                     f"(~{pick['shortability']['fee_rate'] / 12:.1f}% of the position per month).")
    if ref_price and price:
        notes.append(f"{sym} is {((price / ref_price) - 1) * 100:+.0f}% since {ref_date} "
                     f"(${ref_price:.2f} → ${price:.2f}); a short from then is up about {(1 - price / ref_price) * 100:.0f}% before fees.")
    allf = _filings_for(sym, events, overnight)
    supply = [e for e in allf if e.get("category") in score.SUPPLY_CATS | {"atm"}]
    if supply:
        e = supply[0]
        notes.append(f"Most recent supply filing: {score.CAT_LABEL.get(e['category'], e['category'])} "
                     f"({e.get('form')}, {e.get('date')}) — the company can keep issuing shares into the market "
                     f"while it stays effective.")
    rs = [x for x in state_splits.get(sym, []) if (x.get("ratio") or 1) < 1]
    if len(rs) >= 2:
        notes.append(f"{len(rs)} reverse splits on record ({', '.join(score.split_label(x['ratio']) + ' ' + x['date'] for x in rs[-3:])}) — "
                     f"a pattern that usually comes with repeated dilution.")
    late = [e for e in allf if e.get("category") == "late_filing"]
    if late:
        notes.append(f"Filed a late-filing notice ({late[0].get('form')}) on {late[0].get('date')}.")
    return {
        "symbol": sym,
        "name": (pick or {}).get("name") or "Inno Holdings Inc.",
        "price": price,
        "prev_close": price,
        "short_ref_date": ref_date,
        "short_ref_price": ref_price,
        "change_since_ref": (price / ref_price - 1) if (price and ref_price) else None,
        "chart": score.chart_rows(
            df[df.index >= pd.Timestamp(ref_date) - pd.Timedelta(days=21)] if df is not None else None, 400),
        "borrow": score.shortability(sym, borrow, borrow_ok),
        "pick": pick,
        "rank": rank_of.get(sym),
        "filings": _filings_for(sym, events, overnight)[:12],
        "news": news.get(sym, [])[:8],
        "notes": notes,
    }


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
    args = ap.parse_args(argv)

    logfile = config.LOGS / f"{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
    )
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cmds = {"train": cmd_train, "evening": cmd_evening, "morning": cmd_morning, "publish": cmd_publish}
    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
