#!/usr/bin/env python3
"""Build a realistic development dataset and run the statistical core on it.

Stages (each cached under ``data/cache/dev_modeling/`` so re-runs are cheap):

1. **universe** — Nasdaq screener; pick ~250 US-listed names with market cap
   < $300M and price > $0.10 (seeded random sample, INHD always included).
2. **bars** — 3 years of daily bars + split lists straight from yfinance
   (``yf.download`` chunks of 50, ``auto_adjust=False``, ``actions=True``,
   polite pauses, one back-off then give up on rate limits). IWM too.
3. **filings** — optional: SEC submissions for the sample (only when
   ``SEC_USER_AGENT`` is set; ≤ 6 req/s through ``net``), turned into
   minimal form/item-based FilingEvents.
4. **model** — ``features.build_panel`` → ``model.train`` →
   ``evidence.run_studies``; prints the headline numbers and timings.
5. **scale** (``--scale``) — times ``build_panel`` on the sample replicated to
   ~3,500 symbols and one capped HistGradientBoosting fit at production
   size, then extrapolates the full-data walk-forward + refit runtime.

With ``--no-fetch`` nothing touches the network: symbols missing from the
dev bar cache are filled from the production price cache
(``data/cache/prices``) and filings come from the production SEC
submissions cache (``data/cache/sec_submissions``) — whatever those hold.

Nothing here touches ``docs/`` or ``data/models/``: reports are written to
the dev cache so the dev model can never be published by accident.

    export SEC_USER_AGENT="…"          # optional; never printed or logged
    ./.venv/bin/python scripts/dev_modeling_sample.py            # stages 1–4
    ./.venv/bin/python scripts/dev_modeling_sample.py --scale    # + stage 5
    ./.venv/bin/python scripts/dev_modeling_sample.py --fetch-only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.util import clean, to_canonical  # noqa: E402

log = logging.getLogger("dev_modeling")

DEV = config.CACHE / "dev_modeling"
BARS = DEV / "bars"
SEC_DIR = DEV / "sec"
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"

MAX_MCAP = 300e6
MIN_PRICE = 0.10
N_SAMPLE = 250
SEED = 42
CHUNK = 50
PAUSE_S = 4.0
BAR_MAX_AGE_S = 20 * 3600

_EXCLUDE_NAME = re.compile(
    r"\b(warrants?|rights?|units?|preferred|notes|debentures?|depositary shares? representing .*preferred|"
    r"trust preferred|fund|etf|%)\b", re.I)

ASIA_SEC = {c.upper() for c in config.ASIA_COUNTRIES} | {"CHINA", "HONG KONG", "KOREA, REPUBLIC OF"}

OFFER_FORMS = {"424B1", "424B2", "424B4", "424B5", "424B7", "S-1MEF", "F-1MEF"}
REG_FORMS = {"S-1", "S-1/A", "F-1", "F-1/A", "S-3", "S-3/A", "F-3", "F-3/A", "S-3ASR"}


# ── 1) universe ──────────────────────────────────────────────────────────
def load_screener() -> pd.DataFrame:
    """Raw Nasdaq screener rows as a DataFrame (cached 20 h)."""
    raw = net.cached("dev_modeling_screener", SCREENER_URL, 20 * 3600,
                     lambda: net.get_json(SCREENER_URL, headers=net.NASDAQ_HEADERS, timeout=60))
    rows = (((raw or {}).get("data") or {}).get("rows")) or []
    if not rows:
        raise SystemExit("Nasdaq screener unavailable — cannot build the dev sample")
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["lastsale"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df["market_cap"] = pd.to_numeric(df["marketCap"].astype(str).str.replace(",", ""), errors="coerce")
    df["ipo_year"] = pd.to_numeric(df.get("ipoyear"), errors="coerce")
    df["country"] = df.get("country", "").fillna("").astype(str)
    return df


def pick_sample(scr: pd.DataFrame, n: int = N_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """Seeded random sample of common-stock-like micro caps (+ INHD)."""
    ok = (
        scr["price"].gt(MIN_PRICE)
        & scr["market_cap"].gt(0) & scr["market_cap"].lt(MAX_MCAP)
        & ~scr["symbol"].str.contains(r"[\^/ ]", regex=True)
        & ~scr["name"].astype(str).str.contains(_EXCLUDE_NAME)
    )
    elig = scr[ok].copy()
    syms = sorted(elig["symbol"])
    rng = random.Random(seed)
    chosen = set(rng.sample(syms, min(n, len(syms))))
    chosen.add(config.REFERENCE_SYMBOL)
    out = scr[scr["symbol"].isin(chosen)].drop_duplicates("symbol").copy()
    out["symbol"] = out["symbol"].map(to_canonical)
    out["asia"] = out["country"].isin(config.ASIA_COUNTRIES)
    log.info("screener: %d rows, %d eligible (<$300M, >$0.10), sampled %d", len(scr), len(elig), len(out))
    return out.set_index("symbol")[["name", "price", "market_cap", "country", "ipo_year", "asia", "sector", "industry"]]


# ── 2) bars ──────────────────────────────────────────────────────────────
def _bar_path(sym: str) -> Path:
    return BARS / f"{sym}.pkl"


def _fresh(p: Path, max_age_s: float = BAR_MAX_AGE_S) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < max_age_s


def _split_frame(sub: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """yfinance per-ticker frame → (OHLCV split-adjusted, [{date, ratio}])."""
    sub = sub.copy()
    sub.columns = [str(c).lower() for c in sub.columns]
    splits: List[dict] = []
    if "stock splits" in sub.columns:
        ss = pd.to_numeric(sub["stock splits"], errors="coerce")
        for d, r in ss[ss.notna() & (ss > 0) & (ss != 1)].items():
            splits.append({"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "ratio": float(r)})
    bars = sub.reindex(columns=["open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(bars.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    bars.index = idx.normalize()
    px = bars[["open", "high", "low", "close"]]
    traded = px.notna().all(axis=1)
    if not traded.any():
        return bars.iloc[0:0], splits
    bars = bars.loc[traded.idxmax():]                       # drop pre-listing padding
    last = traded[::-1].idxmax()
    bars = bars.loc[:last]
    halted = bars[["open", "high", "low", "close"]].isna().any(axis=1)
    bars.loc[halted, "volume"] = 0.0                         # halts kept, volume 0
    bars = bars[~(bars[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    return bars.astype(float), splits


def fetch_bars(symbols: List[str], period: str = "3y") -> Dict[str, dict]:
    """Download (or read cached) bars + splits for ``symbols``."""
    import yfinance as yf

    BARS.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not _fresh(_bar_path(s))]
    log.info("bars: %d cached, %d to download", len(symbols) - len(todo), len(todo))
    backed_off = False
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(chunk, period=period, interval="1d", auto_adjust=False, actions=True,
                                  group_by="ticker", threads=True, progress=False)
            except Exception as e:  # noqa: BLE001 — yfinance raises many types
                msg = str(e)
                log.warning("yf.download chunk %d failed: %s", i // CHUNK, msg[:120])
                raw = None
                if "Rate" in msg or "Too Many" in msg:
                    if backed_off:
                        log.error("rate-limited twice — stopping downloads (no hammering)")
                        return _read_bars(symbols)
                    backed_off = True
                    time.sleep(60)
                    continue
            break
        if raw is None or raw.empty:
            log.warning("chunk %d: empty result", i // CHUNK)
        else:
            for s in chunk:
                try:
                    sub = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                bars, splits = _split_frame(sub)
                if len(bars) >= 30:
                    with open(_bar_path(s), "wb") as f:
                        pickle.dump({"bars": bars, "splits": splits,
                                     "fetched": datetime.now(timezone.utc).isoformat()}, f)
        log.info("bars: chunk %d/%d done", i // CHUNK + 1, (len(todo) + CHUNK - 1) // CHUNK)
        time.sleep(PAUSE_S)
    return _read_bars(symbols)


def _read_bars(symbols: List[str]) -> Dict[str, dict]:
    out = {}
    for s in symbols:
        p = _bar_path(s)
        if p.exists():
            with open(p, "rb") as f:
                out[s] = pickle.load(f)
    return out


# ── 3) filings ───────────────────────────────────────────────────────────
def _category(form: str, items: List[str]) -> str:
    """Form/item-only category (CONTRACTS §3 taxonomy, no text matching)."""
    f = form.upper()
    if f in OFFER_FORMS:
        return "offering"
    if f in REG_FORMS:
        return "registration"
    if f == "424B3":
        return "resale"
    if f == "EFFECT":
        return "effective"
    if f.startswith("NT "):
        return "late_filing"
    if f in ("144", "144/A"):
        return "insider_sale_notice"
    if f in ("3", "4", "5", "3/A", "4/A", "5/A"):
        return "insider"
    if f.split("/")[0] == "8-K":
        if "3.01" in items:
            return "delisting_notice"
        if "3.02" in items:
            return "unregistered_sale"
        if "5.03" in items:
            return "charter_amendment"
        if "1.01" in items:
            return "material_agreement"
    return "other"


def _events_from_block(sym: str, cik: int, block: dict) -> List[dict]:
    n = len(block.get("form", []))
    out = []
    for i in range(n):
        form = str(block["form"][i])
        items_raw = (block.get("items") or [""] * n)[i] or ""
        items = [x.strip() for x in str(items_raw).split(",") if x.strip()]
        acc = str(block.get("accessionNumber", [""] * n)[i])
        doc = str((block.get("primaryDocument") or [""] * n)[i] or "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
               if acc and doc else None)
        out.append({
            "symbol": sym, "cik": cik, "date": str(block["filingDate"][i])[:10],
            "accepted": (block.get("acceptanceDateTime") or [None] * n)[i],
            "form": form, "items": items, "category": _category(form, items),
            "url": url, "text_tags": [],
        })
    return out


def fetch_filings(symbols: List[str], since: str) -> Tuple[Dict[str, List[dict]], Dict[str, bool]]:
    """symbol → FilingEvents (form/item based) and symbol → Asia address flag.
    Returns ({}, {}) when SEC access is not configured."""
    if not net.sec_enabled():
        log.info("filings: SEC_USER_AGENT not set — skipping (filing features will be NaN)")
        return {}, {}
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    tick = net.cached("dev_modeling_sec", SEC_TICKERS_URL, 7 * 86400,
                      lambda: net.get_json(SEC_TICKERS_URL, headers=net.sec_headers(), timeout=60))
    if not tick:
        log.warning("filings: SEC ticker map unavailable")
        return {}, {}
    cik_of = {to_canonical(str(v["ticker"])): int(v["cik_str"]) for v in tick.values()}
    events: Dict[str, List[dict]] = {}
    asia: Dict[str, bool] = {}
    n_req = 0
    for s in symbols:
        cik = cik_of.get(s)
        if cik is None:
            continue
        name = f"CIK{cik:010d}.json"
        url = SEC_SUBMISSIONS_URL.format(name=name)

        def _fetch(u: str = url) -> Any:
            return net.get_json(u, headers=net.sec_headers(), timeout=30)

        sub = net.cached("dev_modeling_sec", url, 20 * 3600, _fetch)
        n_req += 1
        if not sub:
            continue
        rec = (sub.get("filings") or {}).get("recent") or {}
        evs = _events_from_block(s, cik, rec)
        oldest = min((e["date"] for e in evs), default="9999")
        if oldest > since:
            for page in (sub.get("filings") or {}).get("files") or []:
                if str(page.get("filingTo", "")) < since:
                    continue
                purl = SEC_SUBMISSIONS_URL.format(name=page["name"])
                blk = net.cached("dev_modeling_sec", purl, 7 * 86400,
                                 lambda u=purl: net.get_json(u, headers=net.sec_headers(), timeout=30))
                n_req += 1
                if blk:
                    evs += _events_from_block(s, cik, blk)
        events[s] = sorted([e for e in evs if e["date"] >= since], key=lambda e: e["date"])
        addr = ((sub.get("addresses") or {}).get("business") or {})
        where = str(addr.get("stateOrCountryDescription") or "").upper()
        inc = str(sub.get("stateOfIncorporationDescription") or "").upper()
        asia[s] = where in ASIA_SEC or inc in ASIA_SEC
    log.info("filings: %d symbols with SEC history (%d submission requests)", len(events), n_req)
    return events, asia


def _pipeline_bars(symbols: List[str]) -> Dict[str, dict]:
    """Offline: bars + splits from the production price cache
    (``config.CACHE/"prices"``) for symbols the dev cache lacks. No network."""
    from gravity.sources import prices

    out: Dict[str, dict] = {}
    for s in symbols:
        e = prices.read_cache(s)
        if not e or e.get("df") is None or len(e["df"]) < 30:
            continue
        df = e["df"].copy()
        if e.get("unadjusted"):
            df.attrs["unadjusted"] = True
        out[s] = {"bars": df, "splits": list(e.get("splits") or [])}
    return out


def _pipeline_filings(symbols: List[str], since: str) -> Tuple[Dict[str, List[dict]], Dict[str, bool]]:
    """Offline: FilingEvents from the production SEC submissions cache
    (whatever has been downloaded so far). Symbols without a cached
    submissions file are left out → NaN filing features. No network."""
    from gravity.sources import sec

    payload = net.cache_get(sec.NS_MAP, "company_tickers_exchange", 1e12)
    if not isinstance(payload, dict) or not payload.get("data"):
        return {}, {}
    ix = {f: i for i, f in enumerate(payload.get("fields") or ["cik", "name", "ticker", "exchange"])}
    cik_of: Dict[str, int] = {}
    for row in payload["data"]:
        try:
            cik_of.setdefault(to_canonical(str(row[ix["ticker"]])), int(row[ix["cik"]]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    events: Dict[str, List[dict]] = {}
    asia: Dict[str, bool] = {}
    for s in symbols:
        cik = cik_of.get(s)
        sub = net.cache_get(sec.NS_SUB, sec._sub_key(cik), 1e12) if cik else None
        if not sub:
            continue
        events[s] = sorted(sec.events_from_submissions(sub, s, cik, since=since), key=lambda e: e["date"])
        addr = ((sub.get("addresses") or {}).get("business") or {})
        where = str(addr.get("stateOrCountryDescription") or "").upper()
        inc = str(sub.get("stateOfIncorporationDescription") or "").upper()
        asia[s] = where in ASIA_SEC or inc in ASIA_SEC
    log.info("filings (offline, pipeline cache): %d/%d symbols have cached SEC submissions",
             len(events), len(symbols))
    return events, asia


# ── 4) the statistical core ──────────────────────────────────────────────
def build_inputs(fetch: bool = True) -> Dict[str, Any]:
    DEV.mkdir(parents=True, exist_ok=True)
    sample_path = DEV / "sample.pkl"
    if sample_path.exists() and (_fresh(sample_path, 7 * 86400) or not fetch):
        static = pd.read_pickle(sample_path)
    else:
        static = pick_sample(load_screener())
        static.to_pickle(sample_path)
    syms = sorted(static.index)
    got = fetch_bars(syms + ["IWM"]) if fetch else _read_bars(syms + ["IWM"])
    missing = [s for s in syms + ["IWM"] if s not in got]
    if missing:                      # fill from the production cache (offline)
        extra = _pipeline_bars(missing)
        got.update(extra)
        log.info("bars: %d symbols filled from the production price cache (offline)", len(extra))
    bench = got.pop("IWM", {}).get("bars")
    hist = {s: v["bars"] for s, v in got.items()}
    splits = {s: v["splits"] for s, v in got.items() if v["splits"]}
    since = (date.today() - timedelta(days=3 * 365 + 30)).isoformat()
    events, asia_sec = fetch_filings(sorted(hist), since) if fetch else ({}, {})
    if not events:
        events, asia_sec = _pipeline_filings(sorted(hist), since)
    static = static.copy()
    for s, a in asia_sec.items():
        if a and s in static.index:
            static.at[s, "asia"] = True
    log.info("inputs: %d symbols with bars, %d with splits (%d reverse), %d with filings, IWM=%s",
             len(hist), len(splits), sum(any(x["ratio"] < 1 for x in v) for v in splits.values()),
             len(events), bench is not None)
    return {"hist": hist, "splits": splits, "events": events, "static": static, "bench": bench}


def run_core(inp: Dict[str, Any], fold_months: int = 1) -> Dict[str, Any]:
    from gravity import evidence, features, model

    t = time.time()
    panel = features.build_panel(inp["hist"], inp["events"], inp["splits"], inp["static"], inp["bench"])
    t_panel = time.time() - t
    panel.to_pickle(DEV / "panel.pkl")
    log.info("panel: %d rows x %d cols, %d symbols, %d dates (%.1fs)", len(panel), panel.shape[1],
             panel["symbol"].nunique(), panel["date"].nunique(), t_panel)

    t = time.time()
    report = model.train(panel, out_dir=DEV / "models", site_json=None, fold_months=fold_months)
    t_train = time.time() - t
    (DEV / "model_report.json").write_text(json.dumps(clean(report), indent=1))

    t = time.time()
    ev = evidence.run_studies(panel)
    t_ev = time.time() - t
    (DEV / "evidence.json").write_text(json.dumps(clean(ev), indent=1))

    bundle = model.load(DEV / "models")
    latest = features.latest_rows(panel)
    t = time.time()
    pred = model.predict(bundle, latest, use_open=False)
    t_pred = time.time() - t
    return {"panel": panel, "report": report, "evidence": ev, "latest": latest, "pred": pred,
            "timing": {"build_panel_s": t_panel, "train_s": t_train, "evidence_s": t_ev, "predict_s": t_pred}}


def print_headline(res: Dict[str, Any]) -> None:
    rep, ev, panel = res["report"], res["evidence"], res["panel"]
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    num = lambda x, f=".3f": "—" if x is None else format(x, f)  # noqa: E731
    print("\n══ GRAVITY dev modeling sample ═══════════════════════════════════")
    print(f"panel: {rep['n_rows']:,} labeled rows · {rep['n_symbols']} symbols · {rep['n_days']} sessions · "
          f"through {rep['trained_through']}")
    br = rep["base_rate"]
    print(f"base rates (next-session open→close): dump ≤−5% {pct(br['dump'])} · big dump ≤−15% "
          f"{pct(br['bigdump'])} · squeeze open→high ≥+20% {pct(br['squeeze'])}")
    wf = rep.get("walk_forward", {})
    print(f"walk-forward: {wf.get('n_folds')} folds of {wf.get('fold_months')} month(s), embargo "
          f"{wf.get('embargo_sessions')} sessions, OOS {wf.get('oos_start')} → {wf.get('oos_end')}")
    for m in ("m0", "m1"):
        o = rep["oos"].get(m) or {}
        s = rep["sim"].get(m) or {}
        print(f"  {m}: AUC {num(o.get('auc'))} · Brier {num(o.get('brier'), '.4f')} · OOS base "
              f"{pct(o.get('base_rate'))} · top-1 hit {pct(o.get('top1_hit'))} (mean oc "
              f"{pct(o.get('top1_mean_oc'))}) · top-10 hit {pct(o.get('top10_hit'))} · top-decile "
              f"{pct(o.get('top_decile_hit'))} · days {o.get('days')}")
        print(f"      sim short #1 open→close: gross {num(s.get('gross_total'), '+.2f')} · net(1%) "
              f"{num(s.get('net_total'), '+.2f')} · win {pct(s.get('win_rate'))} · maxDD "
              f"{num(s.get('max_drawdown'), '.2f')} (units of 1× notional, summed daily)")
        print(f"      sim short top-10 basket: gross {num(s.get('top10_gross_total'), '+.2f')} · net "
              f"{num(s.get('top10_net_total'), '+.2f')} · win {pct(s.get('top10_win_rate'))}")
    extra = rep["oos"].get("extra") or {}
    for k, v in extra.items():
        print(f"  OOS {k}: AUC {num(v.get('auc'))} (base {pct(v.get('base_rate'))})")
    fam = rep.get("importance_family") or []
    print("importance by family (AUC drop when permuted): " +
          ", ".join(f"{f['family']} {f['importance']:+.4f}" for f in fam))
    print("top features: " + ", ".join(f"{i['feature']} {i['importance']:+.4f}" for i in rep["importance"][:8]))
    print("\nEvidence Lab (next-session outcomes):")
    for s in ev["studies"]:
        print(f"  {s['id']:<13} n={s['n']:>7,}  dump {pct(s['pct_dump']):>6} CI "
              f"[{pct((s['ci_pct_dump'] or [None])[0])}, {pct((s['ci_pct_dump'] or [None, None])[1])}]  "
              f"lift {num(s['lift_dump'], '.2f')}  red {pct(s['pct_red_oc'])}")
    print(f"\nINHD in latest rows: {'INHD' in set(res['latest']['symbol'])}")
    if "INHD" in set(res["latest"]["symbol"]):
        r = res["latest"].set_index("symbol").loc["INHD"]
        p = res["pred"].loc[res["latest"].index[res["latest"]["symbol"] == "INHD"][0]]
        print(f"  INHD {pd.Timestamp(r['date']).date()} close {r['close']:.3f} price {r['price']:.3f} "
              f"rs_count_2y {r['rs_count_2y']} dd_52w {r['dd_52w']:.2f} → prob_dump {p['prob_dump']:.3f} "
              f"prob_squeeze {p['prob_squeeze']:.3f} exp_oc {p['exp_oc']:+.3f}")
    tm = res["timing"]
    print("\ntiming: " + ", ".join(f"{k} {v:.1f}s" for k, v in tm.items()))
    print("model timing detail: " + json.dumps(rep.get("timing", {})))


# ── 5) scale test ────────────────────────────────────────────────────────
def scale_test(inp: Dict[str, Any], res: Dict[str, Any], target_symbols: int = 3500,
               target_sessions_rows: int = 2_500_000) -> Dict[str, float]:
    """Time the production-size pieces and extrapolate the full train."""
    from gravity import features, model

    hist, events, splits = inp["hist"], inp["events"], inp["splits"]
    reps = max(1, int(np.ceil(target_symbols / max(1, len(hist)))))
    big_hist, big_ev, big_sp = {}, {}, {}
    for r in range(reps):
        for s, df in hist.items():
            k = f"{s}.{r}"
            big_hist[k] = df
            if s in events:
                big_ev[k] = events[s]
            if s in splits:
                big_sp[k] = splits[s]
    st = inp["static"].copy()
    st = pd.concat([st.rename(index=lambda s, r=r: f"{s}.{r}") for r in range(reps)])
    t = time.time()
    big = features.build_panel(big_hist, big_ev, big_sp, st, inp["bench"])
    t_build = time.time() - t
    log.info("scale: build_panel %d symbols → %d rows in %.1fs", len(big_hist), len(big), t_build)
    del big_hist

    # one capped fit on production-size data (the replicated panel)
    lab = big.dropna(subset=["y_dump"])
    t = time.time()
    info = model.time_one_fit(lab)
    t_fit = time.time() - t
    del big, lab
    rep = res["report"]
    n_folds_monthly = rep["walk_forward"]["n_folds"]
    per_fold_fits = rep["walk_forward"]["fits_per_fold"]
    final_fits = rep["walk_forward"]["final_fits"]
    est = {
        "build_panel_s": t_build,
        "one_capped_fit_s": info["fit_s"],
        "fit_rows": info["n_train"],
        "fit_iters": info["n_iter"],
        "folds_monthly": n_folds_monthly,
        "fits_per_fold": per_fold_fits,
        "final_fits": final_fits,
    }
    est["walk_forward_est_s"] = n_folds_monthly * per_fold_fits * info["fit_s"]
    est["final_refit_est_s"] = final_fits * info["fit_s"] + 60.0      # + permutation importance
    est["total_est_s"] = t_build + est["walk_forward_est_s"] + est["final_refit_est_s"]
    return est


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch-only", action="store_true", help="stages 1–3 only")
    ap.add_argument("--no-fetch", action="store_true", help="use cached bars only")
    ap.add_argument("--scale", action="store_true", help="also run the production-size timing test")
    ap.add_argument("--fold-months", type=int, default=1, help="walk-forward test-fold length")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    inp = build_inputs(fetch=not args.no_fetch)
    if args.fetch_only:
        return 0
    res = run_core(inp, fold_months=args.fold_months)
    print_headline(res)
    if args.scale:
        est = scale_test(inp, res)
        print("\nscale test (production size): " + json.dumps({k: round(v, 1) if isinstance(v, float) else v
                                                             for k, v in est.items()}))
        print(f"extrapolated full train (build + walk-forward + final refit): "
              f"{est['total_est_s'] / 60:.1f} min")
        (DEV / "scale_test.json").write_text(json.dumps(clean(est), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
