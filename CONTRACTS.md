# GRAVITY — module contracts

This file is the single source of truth for how the pieces fit. Every module
is built against it; if code and contract disagree, the contract wins until
it is deliberately changed.

Runtime: **system Python 3.9** (`~/gravity/.venv`, has pandas 2.2, numpy 2.0,
yfinance 1.2, scikit-learn 1.6, requests, pytest). Use
`from __future__ import annotations`; no `match`, no `X | Y` at runtime.
Run tests with `./.venv/bin/pytest -q`.

Shared helpers you MUST use (already written, do not rewrite):

- `gravity/config.py` — paths, env vars, thresholds (read it).
- `gravity/net.py` — `get / get_json / get_text` (throttled, retried, never
  raise, return `None` on failure), `NASDAQ_HEADERS`, `sec_headers()`,
  `sec_enabled()`, JSON disk cache `cached(ns, key, max_age_s, fetch)`,
  and `record_status(name, ok, detail)` which every source calls once per
  run so the site can show which feeds were live.
- `gravity/util.py` — `num()` parsing, `clean()` JSON-safety, market clock
  (`now_et`, `target_session`, `market_phase`, trading-day helpers),
  `to_canonical()` symbol mapping.

**Canonical symbol** = Yahoo style (`BRK-A`). Convert at the edges.

**Honesty rules (non-negotiable, apply to code, labels and UI copy):**
nothing is fabricated; missing data is `None`/`null` and shown as "—", never
imputed silently in outputs; every fact carries its date and source URL when
one exists; model outputs are calibrated probabilities with the base rate
shown next to them; no "guaranteed", "will dump", "sure thing" language.

---

## 1. `gravity/sources/universe.py`

```python
def load_universe(max_age_s: int = 6*3600) -> pd.DataFrame
```
Nasdaq screener (`https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true`,
headers `NASDAQ_HEADERS`; ~7,000 rows, 2 MB). Returns one row per common
stock with columns:

| column | type | notes |
|---|---|---|
| `symbol` | str | canonical |
| `name` | str | |
| `price` | float | last sale |
| `pct_change` | float | fraction (−0.035 for −3.5%) |
| `volume` | float | |
| `market_cap` | float or NaN | |
| `country` | str or "" | |
| `ipo_year` | float or NaN | |
| `sector`, `industry` | str | |
| `asia` | bool | `country in config.ASIA_COUNTRIES` (SEC business-address refinement happens later in sec.py) |

Exclusions: ETFs/funds/notes, warrants/rights/units/preferreds
(symbol contains `^`; name matches Warrant|Right|Unit|Preferred|Notes|
Debenture|Trust Preferred|%; 5-letter symbols ending W/R/U when the name
confirms), `price < config.MIN_PRICE`, `market_cap > config.MAX_MARKET_CAP`.
Keep ADRs (they are common-equity exposure). Cache the raw payload via
`net.cached("universe", ...)`. `record_status("Nasdaq screener", ...)`.

## 2. `gravity/sources/prices.py`

```python
def load_history(symbols: list[str], period: str = config.HISTORY_PERIOD,
                 refresh: bool = True, max_workers: int = 4) -> dict[str, pd.DataFrame]
def load_splits(symbols: list[str]) -> dict[str, list[dict]]   # [{"date": "YYYY-MM-DD", "ratio": 0.05}]  ratio<1 = reverse split (1:20 → 0.05)
def premarket_snapshot(symbols: list[str]) -> dict[str, dict]
def benchmark_history(period: str = config.HISTORY_PERIOD) -> pd.DataFrame  # IWM daily, same shape
```

History frames: `DatetimeIndex` (tz-naive dates, ascending, unique), columns
`open high low close volume` — **split-adjusted** (yfinance `auto_adjust=False`,
use split-adjusted OHLC; do not dividend-adjust). Source = yfinance batch
`yf.download` (chunks ~60, `threads=True`, pause between chunks, retry
failed symbols individually, back off hard on rate-limit). Disk cache per
symbol under `config.CACHE/"prices"` (pickle) plus the split list; refresh
is incremental (fetch recent ~10 sessions, append, dedupe) **unless a new
split appeared since the cache was written**, in which case refetch the full
period (adjustment changed history). Zero-volume/NaN rows (halts) are kept
with `volume = 0` so downstream can see halts; rows with non-positive prices
are dropped. Fallback when yfinance fails for a symbol: Nasdaq historical
(`/api/quote/{SYM}/historical?assetclass=stocks&fromdate=...&todate=...&limit=9999`)
which is **NOT split-adjusted** — adjust it with `load_splits` data if known,
else mark `df.attrs["unadjusted"] = True`.

`premarket_snapshot` → per symbol
`{"price": float, "prev_close": float, "gap_pct": float (fraction), "volume": float|None, "asof": iso str, "source": "nasdaq"|"yahoo"}`
from Nasdaq `/api/quote/{SYM}/info?assetclass=stocks` (`primaryData` is
live during pre-market; `marketStatus` tells you the phase), falling back
to yfinance 1-minute bars with `prepost=True`. Only called for the
shortlist (≤ 150 names). Omit symbols with no pre-market trade.

`record_status("Yahoo Finance prices", ...)`, `record_status("Nasdaq quotes", ...)`.

## 3. `gravity/sources/sec.py`

```python
def cik_map() -> dict[str, dict]            # symbol → {"cik": int, "name": str, "exchange": str}
def submissions(cik: int, max_age_s: int = 20*3600) -> dict | None   # raw data.sec.gov JSON (+ merges the "files" pages when needed for ≥3y of history)
def issuer_profile(cik: int) -> dict        # {"state_of_inc", "business_country", "business_city", "sic", "sic_desc", "asia": bool, "filer_category"}
def filing_events(symbol: str, since: str = None) -> list[dict]
def latest_filings(since_utc: datetime) -> list[dict]   # everything accepted since (overnight catalysts), across all issuers
def fulltext_catalysts(start: str, end: str) -> list[dict]
def shares_history(cik: int) -> list[dict]  # [{"date": "YYYY-MM-DD", "filed": "YYYY-MM-DD", "shares": float}] from companyfacts dei:EntityCommonStockSharesOutstanding
def cash_runway(cik: int) -> dict | None     # {"cash": float, "cash_date": str, "quarterly_burn": float|None, "runway_q": float|None}
```

**FilingEvent** (dict) — the unit every other module consumes:

```json
{
  "symbol": "INHD", "cik": 1961847,
  "date": "2026-05-19",                 // filingDate (point-in-time key)
  "accepted": "2026-05-19T17:02:11-04:00",  // acceptanceDateTime when known, else null
  "form": "424B5", "items": ["1.01","3.02"],
  "category": "offering",               // see taxonomy
  "url": "https://www.sec.gov/Archives/edgar/data/1961847/000.../form424b5.htm",
  "text_tags": []                        // filled by full-text search when matched
}
```

Category taxonomy (exact strings):

| category | how |
|---|---|
| `offering` | 424B1/424B2/424B4/424B5/424B7, S-1MEF/F-1MEF, 8-K/6-K text-matched "registered direct"/"public offering … priced"/"securities purchase agreement" |
| `atm` | 424B5/8-K/6-K text-matched "at-the-market"/"at the market offering"/"sales agreement" |
| `registration` | S-1, S-1/A, F-1, F-1/A, S-3, S-3/A, F-3, F-3/A, S-3ASR |
| `resale` | 424B3; S-1/F-1 whose text says "selling stockholders" when known |
| `effective` | EFFECT |
| `unregistered_sale` | 8-K item 3.02 |
| `delisting_notice` | 8-K item 3.01; text-matched "5550(a)(2)", "minimum bid price", "Listing Qualifications" |
| `reverse_split` | 8-K item 5.03 text-matched "reverse stock split"/"share consolidation"; text-matched 6-K |
| `toxic_financing` | text-matched "equity line", "committed equity facility", "ELOC", "convertible promissory note", "warrant inducement" |
| `going_concern` | text-matched "substantial doubt" / "going concern" in 10-K/10-Q/20-F |
| `late_filing` | NT 10-K, NT 10-Q, NT 20-F |
| `insider_sale_notice` | 144 |
| `insider` | 3, 4, 5 |
| `material_agreement` | 8-K item 1.01 without offering text |
| `other` | everything else |

History: `submissions` gives `filings.recent` (last ~1,000) — enough for
3y for almost all micro-caps; follow `filings.files[]` pages only if the
recent block ends inside the 3-year window. 8-K items come from the
`items` field. Text matching is only done for (a) `fulltext_catalysts`
and (b) the ≤ 150 shortlist names in the live run — never for the whole
3-year history (too many requests); historical categories are form/item based.

`fulltext_catalysts(start, end)` runs `efts.sec.gov/LATEST/search-index`
queries (quoted phrases above; forms 8-K,6-K,424B1-5,S-1,F-1,10-Q,10-K,20-F)
over the date range and returns FilingEvents with `text_tags` set to the
matched phrase keys (`registered_direct`, `public_offering_priced`,
`atm`, `reverse_split`, `bid_price_deficiency`, `delisting`,
`going_concern`, `equity_line`, `convertible_note`, `warrant_inducement`,
`securities_purchase_agreement`) and `category` re-derived from tags.
Map `display_names` "(TICK, TICKW)" → canonical symbol.

`latest_filings(since)` → EDGAR "current events" Atom feed
(`/cgi-bin/browse-edgar?action=getcurrent&type=&count=100&output=atom`,
paging with `start=`) for forms of interest (8-K, 6-K, 424B*, S-1, F-1, S-3,
F-3, EFFECT, NT*, 144) until entries are older than `since`; map CIK →
symbol via `cik_map`.

Everything no-ops (returns empty) with `record_status("SEC EDGAR", False,
"SEC_USER_AGENT not set")` when `net.sec_enabled()` is False. Max 6 req/s.

## 4. `gravity/sources/shortside.py`

```python
def ibkr_borrow() -> dict[str, dict]      # {"fee_rate": float (annual %, 98.81), "rebate_rate": float, "available": int|None, "asof": str}
def finra_short_volume(days: int = 20) -> pd.DataFrame   # columns date, symbol, short_volume, total_volume, short_ratio
def short_interest(symbol: str) -> list[dict]   # [{"settlement_date","interest","avg_daily_volume","days_to_cover"}], newest first
def float_shares(symbols: list[str]) -> dict[str, dict]   # {"float": float|None, "shares_out": float|None, "short_pct_float": float|None, "source": str}
```

IBKR: anonymous FTP `ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt`
(pipe-delimited, header row starts `#SYM`; `AVAILABLE` can be `>10000000`
→ 10000000). Cache 30 min. FINRA: `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`
(skip weekends/holidays/404s; cache each day's file forever once complete).
Short interest: Nasdaq `/api/quote/{SYM}/short-interest?assetClass=stocks`.
Float: yfinance `Ticker.info` (`floatShares`, `sharesOutstanding`,
`shortPercentOfFloat`) with 3-day cache; shortlist only; never cache a
throttled/empty response.

## 5. `gravity/sources/street.py`

```python
def analyst(symbol: str) -> dict | None    # {"mean_rating": str|None, "n_analysts": int|None, "changes": [{"date","firm","action","from","to"}], "price_target": float|None, "source_url": str}
def earnings_calendar(d: date) -> list[dict]   # [{"symbol","time","eps_forecast","n_ests","market_cap"}]
def movers() -> dict        # {"premarket_gainers": [...], "premarket_losers": [...], "most_active": [...]}  each item {"symbol","price","change_pct"}
def danelfin(symbols: list[str]) -> dict[str, dict]   # {} unless config.DANELFIN_API_KEY; {"ai_score","technical","fundamental","sentiment","low_risk","date"}
def deep_links(symbol: str, cik: int | None = None) -> dict[str, str]
```

`deep_links` returns ordered label → URL for: Zacks, Danelfin, Bloomberg,
WSJ, Finviz, TradingView, Stocktwits, Yahoo, Nasdaq, SEC EDGAR (by CIK when
known), Fintel (short data), iBorrowDesk (IBKR borrow history), TipRanks,
MarketBeat, OTC/Nasdaq halts. We **link** to Zacks/Bloomberg/Danelfin
pages; we never scrape or republish their proprietary ratings (only the
Danelfin API, and only when the user supplies a key).

## 6. `gravity/sources/news.py`

```python
def headlines(symbol: str, company: str = "", max_age_days: int = 7, limit: int = 12) -> list[dict]
def classify(title: str) -> dict   # {"tags": [...], "polarity": -1|0|1}
```

Google News RSS (`https://news.google.com/rss/search?q=...&hl=en-US&gl=US&ceid=US:en`),
query = `"{SYMBOL}" stock` plus company name when given; parse with stdlib
`xml.etree`. Item → `{"published": iso, "title", "source", "url", "tags", "polarity"}`.
Bearish tags: `offering`, `priced`, `dilution`, `reverse_split`,
`delisting`, `deficiency`, `halt`, `investigation`, `lawsuit`, `resign`,
`going_concern`, `default`, `downgrade`, `miss`, `guidance_cut`,
`atm`, `warrants`. Bullish: `contract`, `partnership`, `fda`, `approval`,
`beat`, `upgrade`, `acquisition`, `buyback`, `uplisting`. Cache 20 min.

## 7. `gravity/features.py`

```python
FEATURES: list[str]          # exact model input columns, stable order
M1_EXTRA: list[str]          # ["gap_open"] — only known at/after the open (live proxy = pre-market price)
LABELS: list[str]            # y_oc, y_co, y_gap, y_ol, y_oh, y_c5, y_dump, y_bigdump, y_squeeze
def build_panel(hist: dict[str, pd.DataFrame], events: dict[str, list[dict]],
                splits: dict[str, list[dict]], static: pd.DataFrame,
                bench: pd.DataFrame | None = None, min_date: str | None = None) -> pd.DataFrame
def latest_rows(panel: pd.DataFrame) -> pd.DataFrame     # one row per symbol at its last date (the live feature vector)
```

Panel = one row per (`date`, `symbol`) (columns, not index), features as
of the **close of `date`** using only bars ≤ date and FilingEvents with
`date` ≤ that date; labels describe the **next** session `t+1`:

- `y_gap = O[t+1]/C[t] − 1`, `y_oc = C[t+1]/O[t+1] − 1`, `y_co = C[t+1]/C[t] − 1`,
  `y_ol = L[t+1]/O[t+1] − 1`, `y_oh = H[t+1]/O[t+1] − 1`, `y_c5 = C[t+5]/O[t+1] − 1`
- `y_dump = y_oc ≤ DUMP_THRESHOLD`, `y_bigdump = y_oc ≤ BIG_DUMP_THRESHOLD`,
  `y_squeeze = y_oh ≥ SQUEEZE_THRESHOLD`
- labels are NaN when t+1 is missing, halted (volume 0) or has bad prints.
- `gap_open` (M1 feature) = `y_gap` — it is known at 9:30 on t+1, so the
  M1 model may use it; M0 may not.

Row filters: close ≥ MIN_PRICE, 20-day median dollar volume ≥
MIN_DOLLAR_VOLUME_20D, ≥ 60 prior bars. Static columns carried: `asia`,
`ipo_year`. Features must be computed by **the same code** for training and
live scoring (no train/serve skew). Include at minimum: multi-horizon
returns, today's gap / intraday / range / close-location / upper wick,
relative volume, log dollar volume, log price, realized vol, RSI14/RSI2,
distance from MA20/50/200, 52-week drawdown & run-up, spike recency,
reverse-split count/recency/size, filing-derived counts by category over
30/90/180/365 days and days-since-last-offering, cross-sectional ranks per
date (r1, rvol), universe breadth and IWM return.

## 8. `gravity/model.py`

```python
def train(panel: pd.DataFrame, out_dir: Path = config.MODELS) -> dict   # returns the report dict (also written to docs/data/model.json)
def load() -> dict | None
def predict(bundle: dict, rows: pd.DataFrame, use_open: bool) -> pd.DataFrame  # adds prob_dump, prob_bigdump, prob_squeeze, exp_oc
```

Models: M0 (FEATURES) and M1 (FEATURES + M1_EXTRA), each with
classifiers for `y_dump`, `y_bigdump`, `y_squeeze` and a regressor for
clipped `y_oc`. sklearn `HistGradientBoosting*`, isotonic calibration
fit on a held-out later slice. **Walk-forward evaluation** (expanding
window, monthly test folds, ≥ 5-session embargo) producing
out-of-sample predictions for every test day → report: AUC, Brier, base
rate, calibration table (deciles), top-decile / top-10 / top-1 daily hit
rates and mean `y_oc`, a daily "short the #1 at the open, cover at the
close" simulation (gross and with 1 % round-trip cost), feature
importances (permutation, grouped by family). Final production model is
refit on all data. Report written to `docs/data/model.json` — schema in §12.

## 9. `gravity/evidence.py`

```python
def run_studies(panel: pd.DataFrame) -> dict   # written to docs/data/evidence.json
```

Event studies = conditional base rates on the panel, each:
`{"id","title","plain","condition","n","n_symbols","pct_red_oc","pct_dump","pct_bigdump","pct_squeeze","median_oc","mean_oc","median_c5","ci_pct_dump":[lo,hi],"lift_dump"}`
plus `baseline` (all rows). Bootstrap CI (by date clusters). Studies:
baseline; +20/+50/+100/+200% day; 3-day run > +100%; gap-up > 30% (M1:
same-day open→close); reverse split within 5/30 sessions; offering filed
≤ 1 session ago; registration filed ≤ 30 days; delisting notice ≤ 90 days;
price < $1; 52-week drawdown > 90%; RSI14 > 85; Asia-linked & IPO ≤ 2y;
INHD-profile (Asia, ≥ 2 reverse splits in 2y, drawdown > 90%).

## 10. `gravity/score.py`, `twins.py`, `scorecard.py`, `publish.py`, `cli.py` (integration layer)

Owned by the integrator. `score.py` combines model probabilities, live-only
overlays (overnight filings, full-text catalysts, pre-market gap, news
tags) and shortability; `twins.py` ranks INHD lookalikes; `scorecard.py`
appends the daily pick log and grades it against real OHLC;
`publish.py` writes §11–§13 and pushes; `cli.py`: `python -m gravity.cli evening|morning|train|publish`.

## 11. `docs/data/today.json` (the site's main feed)

```jsonc
{
  "generated_at": "2026-09-30T12:15:00+00:00",
  "session_date": "2026-09-30",          // the trading session these picks are FOR
  "run": "morning",                       // morning | evening
  "market_phase": "pre-market",
  "universe": {"listed": 7017, "eligible": 3400, "scored": 3100},
  "model": {"version": "m1", "use_open": true, "base_rate_dump": 0.118, "trained_through": "2026-09-29", "oos_auc": 0.68},
  "top": Pick,                            // the #1 (shortable, not in squeeze zone), or null
  "board": [Pick, ...],                   // BOARD_SIZE, ranked by prob_dump
  "squeeze_zone": [Pick, ...],            // high prob_dump but squeeze_danger ≥ 70 or unborrowable
  "catalyst_wire": [{"time": iso, "symbol", "kind", "headline", "url", "source", "severity": 1-3}],
  "twins": [Twin, ...],
  "reference": ReferencePosition,         // INHD
  "earnings": [{"symbol","time","eps_forecast","n_ests","in_universe": bool}],
  "sources": [{"name","ok","detail","asof"}],
  "disclaimer": "…"
}
```

**Pick**

```jsonc
{
  "rank": 1, "symbol": "XXXX", "name": "…", "exchange": "NASDAQ", "country": "Hong Kong",
  "sector": "…", "industry": "…",
  "price": 2.31, "prev_close": 2.31, "market_cap": 12000000,
  "premarket": {"price": 2.9, "gap_pct": 0.255, "volume": 120000, "asof": "…"} | null,
  "prob_dump": 0.31, "prob_bigdump": 0.09, "prob_squeeze": 0.12, "exp_oc": -0.041,
  "lift": 2.6,                             // prob_dump / base_rate_dump
  "score": 94,                             // percentile of prob_dump within today's scored universe (0–100)
  "squeeze_danger": 34,                    // 0–100
  "shortability": {"status": "HTB"|"ETB"|"NONE"|"UNKNOWN", "available": 35000, "fee_rate": 98.8, "asof": "…"},
  "families": {"dilution": 0-100, "exhaustion": 0-100, "decay": 0-100, "flow": 0-100, "street": 0-100|null, "news": 0-100|null},
  "reasons": [{"family": "dilution", "text": "424B5 prospectus supplement filed 2026-09-29 after the close", "url": "…", "strength": 1-3}],
  "flags": ["OFFERING", "R/S 1:20", "DEFICIENCY", "PUMP +140%", "HTB 99%", "ASIA", "HALTED"],
  "metrics": {"r1": .., "r5": .., "r20": .., "rvol1": .., "rsi14": .., "dist_ma20": .., "dd_52w": .., "vol20": .., "dvol20": .., "rs_count_2y": .., "n_offer_90": .., "short_ratio_5d": .., "si_pct_float": .., "days_to_cover": .., "float": ..},
  "filings": [FilingEvent, ...],           // newest first, ≤ 12
  "news": [Headline, ...],                 // ≤ 8
  "street": {"analyst": {...}|null, "danelfin": {...}|null},
  "links": {"Zacks": "…", "Danelfin": "…", "Bloomberg": "…", ...},
  "chart": [["2026-05-01", o, h, l, c, v], ...],   // CHART_SESSIONS bars (board + reference only)
  "inhd_similarity": 0.83
}
```

**Twin** = `{"symbol","name","similarity" (0–1),"price","market_cap","country","reasons":[str],"prob_dump","rank"|null,"flags"}`

**ReferencePosition** =
`{"symbol":"INHD","name","price","prev_close","short_ref_date","short_ref_price","change_since_ref","chart":[...],"borrow":{...},"pick":Pick|null,"rank":int|null,"filings":[...],"news":[...],"notes":[str]}`

## 12. `docs/data/model.json`

```jsonc
{
  "trained_at", "trained_through", "n_rows", "n_symbols", "n_days",
  "targets": {"dump": "open→close ≤ −5%", ...},
  "base_rate": {"dump": .., "bigdump": .., "squeeze": ..},
  "oos": {"m0": Metrics, "m1": Metrics},
  "calibration": {"m1": [{"bin": 1, "pred": .., "actual": .., "n": ..}, ...]},
  "importance": [{"family": "exhaustion", "feature": "r1", "importance": ..}, ...],
  "sim": {"m0": Sim, "m1": Sim},
  "caveats": [str]
}
```
`Metrics = {"auc", "brier", "top1_hit", "top10_hit", "top_decile_hit", "top1_mean_oc", "top10_mean_oc", "days"}`;
`Sim = {"daily": [["2025-01-02", oc_top1, oc_top10_mean, universe_mean_oc], ...], "gross_total", "net_total", "win_rate", "max_drawdown", "cost_assumption": 0.01}`

## 13. `docs/data/scorecard.json`, `docs/data/evidence.json`, `docs/data/history/YYYY-MM-DD.json`

History file (written at publish, graded after the close):
`{"session_date","published_at","top": {"symbol","prob_dump","score","premarket"}, "board": [{"rank","symbol","prob_dump"}], "outcome": null | {"graded_at","top": {"open","high","low","close","oc","ol","oh","dump": bool}, "board_mean_oc", "board_dump_rate", "universe_mean_oc", "universe_dump_rate"}}`

Scorecard = `{"asof","days":[...graded history rows...],"live": {"n_days","top_dump_rate","top_mean_oc","board_dump_rate","universe_dump_rate"} }`.
