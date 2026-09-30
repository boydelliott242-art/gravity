"""Generate schema-exact SAMPLE feeds so the static site renders before the
pipeline has published anything.

Writes ``today.json``, ``model.json``, ``evidence.json`` and
``scorecard.json`` into ``docs/data/`` (CONTRACTS.md §11–§13). Every file
carries a top-level ``"sample": true`` and the site shows an unmistakable
"SAMPLE DATA" banner whenever it sees that flag.

Honesty rules for the sample:

- Every ticker contains a digit (``SMPL1``, ``DEMO7`` …) and every company
  name contains "Sample" or "Demo", so no sample row can ever be mistaken for
  (or libel) a real US listing.
- The one real ticker is the reference position ``INHD`` (the user's short).
  Its prices are round placeholder numbers, its notes say so, and it gets
  **no** filings, news, analyst data or model pick — we never invent facts
  about a real company.
- Every URL that would point at a source document points at ``example.com``
  (reserved for documentation), except the research deep links, which use the
  real public URL patterns so the link row can be exercised.

The script refuses to overwrite a feed that is *not* a sample (i.e. real
pipeline output) unless ``--force`` is given.

Usage::

    ./.venv/bin/python scripts/make_sample_data.py
    ./.venv/bin/python scripts/make_sample_data.py --session-date 2026-09-30 --now 2026-09-30T11:05:00+00:00
    ./.venv/bin/python scripts/make_sample_data.py --late   # exercise the "published after the open" banner
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gravity import config  # noqa: E402
from gravity.util import ET, clean, prev_trading_day, target_session  # noqa: E402

FILES = ("today.json", "model.json", "evidence.json", "scorecard.json")
BASE_RATE_DUMP = 0.118
BASE_RATE_BIGDUMP = 0.021
BASE_RATE_SQUEEZE = 0.064
EXAMPLE = "https://example.com/gravity-sample"

DISCLAIMER = (
    "GRAVITY is an automated research tool, not investment advice. Probabilities describe how "
    "similar historical setups behaved; any single session can go either way. Short selling "
    "carries unlimited loss potential, squeezes, halts, borrow recalls and fees. Data comes from "
    "public sources that can be late or wrong. Do your own research."
)

# (symbol, name, country, sector, industry, exchange, chart shape)
ROSTER: List[Tuple[str, str, str, str, str, str, str]] = [
    ("SMPL1", "Sample Holdings Ltd", "Hong Kong", "Industrials", "Engineering & Construction", "NASDAQ", "spike_end"),
    ("DEMO1", "Demo Robotics (Sample) Inc", "United States", "Technology", "Computer Hardware", "NASDAQ", "pump_fade"),
    ("DEMO2", "Demo Bio Sample Corp", "United States", "Health Care", "Biotechnology", "NASDAQ", "spike_end"),
    ("SMPL2", "Sample Logistics Group Ltd", "Singapore", "Industrials", "Trucking", "NASDAQ", "decay"),
    ("DEMO3", "Demo Clean Energy Sample Inc", "China", "Utilities", "Renewable Utilities", "NASDAQ", "pump_fade"),
    ("DEMO4", "Demo Payments Sample Ltd", "Cayman Islands", "Financials", "Credit Services", "NASDAQ", "spike_end"),
    ("SMPL3", "Sample Therapeutics Inc", "United States", "Health Care", "Biotechnology", "NASDAQ", "decay"),
    ("DEMO5", "Demo Media Sample Holdings", "Malaysia", "Communication Services", "Internet Content", "NASDAQ", "dead_cat"),
    ("DEMO6", "Demo AI Sample Corp", "United States", "Technology", "Software", "NASDAQ", "spike_end"),
    ("SMPL4", "Sample Foods Group Ltd", "Hong Kong", "Consumer Staples", "Packaged Foods", "NASDAQ", "decay"),
    ("DEMO7", "Demo Mining Sample Inc", "United States", "Materials", "Gold", "NYSE American", "pump_fade"),
    ("DEMO8", "Demo EV Sample Holdings", "China", "Consumer Discretionary", "Auto Parts", "NASDAQ", "dead_cat"),
    ("SMPL5", "Sample Education Ltd", "China", "Consumer Discretionary", "Education", "NASDAQ", "decay"),
    ("DEMO9", "Demo Quantum Sample Inc", "United States", "Technology", "Semiconductors", "NASDAQ", "spike_end"),
    ("SMPL6", "Sample Shipping Corp", "Singapore", "Industrials", "Marine Shipping", "NASDAQ", "grind"),
    ("DEMO10", "Demo Health Sample Inc", "United States", "Health Care", "Medical Devices", "NASDAQ", "grind"),
    ("SMPL7", "Sample Apparel Holdings Ltd", "Hong Kong", "Consumer Discretionary", "Apparel", "NASDAQ", "pump_fade"),
    ("DEMO11", "Demo Crypto Sample Corp", "United States", "Financials", "Capital Markets", "NASDAQ", "spike_end"),
    ("SMPL8", "Sample Pharma (Demo) Ltd", "Israel", "Health Care", "Drug Manufacturers", "NASDAQ", "decay"),
    ("DEMO12", "Demo Fintech Sample Ltd", "Singapore", "Financials", "Software", "NASDAQ", "grind"),
    ("SMPL9", "Sample Water Tech Inc", "United States", "Utilities", "Water Utilities", "NASDAQ", "dead_cat"),
    ("DEMO13", "Demo Drone Sample Inc", "United States", "Industrials", "Aerospace & Defense", "NASDAQ", "spike_end"),
    ("SMPL10", "Sample Beverage Group", "Japan", "Consumer Staples", "Beverages", "NASDAQ", "grind"),
    ("DEMO14", "Demo Genomics Sample Corp", "United States", "Health Care", "Diagnostics", "NASDAQ", "decay"),
    ("SMPL11", "Sample Property Holdings Ltd", "Hong Kong", "Real Estate", "Real Estate Services", "NASDAQ", "pump_fade"),
    # squeeze-zone names
    ("SQZ1", "Sample Squeeze Candidate Inc", "United States", "Technology", "Software", "NASDAQ", "spike_end"),
    ("SQZ2", "Demo Low-Float Sample Ltd", "Hong Kong", "Industrials", "Consulting", "NASDAQ", "spike_end"),
    ("SQZ3", "Sample Halted Holdings Corp", "United States", "Health Care", "Biotechnology", "NASDAQ", "pump_fade"),
    ("SQZ4", "Demo No-Borrow Sample Inc", "China", "Technology", "Electronics", "NASDAQ", "spike_end"),
    # twins that are not on the board
    ("TWIN1", "Sample Twin Holdings Ltd", "Hong Kong", "Industrials", "Building Products", "NASDAQ", "decay"),
    ("TWIN2", "Demo Twin Group Ltd", "Singapore", "Consumer Discretionary", "Leisure", "NASDAQ", "decay"),
    ("TWIN3", "Sample Twin Technology Inc", "Malaysia", "Technology", "IT Services", "NASDAQ", "dead_cat"),
]
ASIA = config.ASIA_COUNTRIES


# ── time helpers ─────────────────────────────────────────────────────────
def trading_days_before(session: date, n: int) -> List[date]:
    """The ``n`` trading days strictly before ``session``, ascending."""
    out: List[date] = []
    d = session
    while len(out) < n:
        d = prev_trading_day(d)
        out.append(d)
    return list(reversed(out))


def iso_et(d: date, hh: int, mm: int, ss: int = 0) -> str:
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=ET).isoformat()


# ── synthetic price paths ────────────────────────────────────────────────
def _closes(rng: random.Random, n: int, shape: str, end: float) -> List[float]:
    """A plausible micro-cap close path of length ``n`` ending near ``end``."""
    lr: List[float] = []
    for i in range(n):
        vol = 0.055
        mu = -0.004
        if shape == "decay":
            mu = -0.009
        elif shape == "grind":
            mu, vol = -0.005, 0.035
        elif shape == "pump_fade":
            if n - 14 <= i < n - 9:
                mu, vol = 0.16, 0.06
            elif i >= n - 9:
                mu, vol = -0.05, 0.06
        elif shape == "dead_cat":
            if i < n * 0.35:
                mu = -0.022
            elif i >= n - 6:
                mu, vol = 0.05, 0.06
        elif shape == "spike_end":
            if i < n - 16:
                mu = -0.011
            elif i < n - 3:
                mu, vol = 0.0, 0.03
        lr.append(rng.gauss(mu, vol))
    if shape == "spike_end":  # the three-day ramp a shorter waits for
        lr[-3], lr[-2], lr[-1] = math.log(1.18), math.log(1.35), math.log(1.0 + rng.uniform(0.9, 1.45))
    elif shape == "dead_cat":
        lr[-1] = math.log(1.0 + rng.uniform(0.25, 0.6))
    path = [0.0]
    for x in lr[1:]:
        path.append(path[-1] + x)
    shift = math.log(end) - path[-1]
    return [math.exp(p + shift) for p in path]


def _round_px(x: float) -> float:
    return round(x, 4) if x < 10 else round(x, 2)


def synth_chart(rng: random.Random, days: Sequence[date], shape: str, end: float,
                base_volume: float) -> List[list]:
    """[[date, o, h, l, c, v], ...] with internally consistent OHLC."""
    closes = _closes(rng, len(days), shape, end)
    rows: List[list] = []
    prev = closes[0] * math.exp(rng.gauss(0.01, 0.03))
    for d, c in zip(days, closes):
        ret = c / prev - 1.0
        o = prev * math.exp(rng.gauss(0.35 * math.log(c / prev), 0.025))
        top = max(o, c) * (1.0 + abs(rng.gauss(0, 0.035)) + (0.08 if ret > 0.3 else 0.0))
        bot = min(o, c) * (1.0 - abs(rng.gauss(0, 0.03)))
        v = base_volume * math.exp(rng.gauss(0, 0.45)) * (1.0 + 14.0 * abs(ret))
        rows.append([d.isoformat(), _round_px(o), _round_px(top), _round_px(bot), _round_px(c), float(int(v))])
        prev = c
    for r in rows:  # enforce l ≤ o,c ≤ h after rounding
        r[2] = max(r[1], r[2], r[3], r[4])
        r[3] = min(r[1], r[2], r[3], r[4])
    return rows


def _rsi(closes: Sequence[float], n: int = 14) -> Optional[float]:
    if len(closes) <= n:
        return None
    gains, losses = 0.0, 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        ch = b - a
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    if losses == 0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100.0 - 100.0 / (1.0 + rs)


def chart_metrics(chart: List[list]) -> Dict[str, Optional[float]]:
    c = [r[4] for r in chart]
    v = [r[5] for r in chart]

    def ret(k: int) -> Optional[float]:
        return c[-1] / c[-1 - k] - 1.0 if len(c) > k else None

    def ma(k: int) -> Optional[float]:
        return sum(c[-k:]) / k if len(c) >= k else None

    rets = [b / a - 1.0 for a, b in zip(c[-21:-1], c[-20:])]
    mean = sum(rets) / len(rets)
    vol20 = math.sqrt(sum((x - mean) ** 2 for x in rets) / (len(rets) - 1))
    avg_v = sum(v[-21:-1]) / 20.0
    ma20, ma50 = ma(20), ma(50)
    dv = sorted(a * b for a, b in zip(c[-20:], v[-20:]))
    return {
        "r1": ret(1), "r3": ret(3), "r5": ret(5), "r20": ret(20),
        "rvol1": v[-1] / avg_v if avg_v else None,
        "rsi14": _rsi(c),
        "dist_ma20": c[-1] / ma20 - 1.0 if ma20 else None,
        "dist_ma50": c[-1] / ma50 - 1.0 if ma50 else None,
        "dd_52w": c[-1] / max(r[2] for r in chart) - 1.0,
        "vol20": vol20,
        "dvol20": dv[len(dv) // 2],
    }


# ── building blocks ──────────────────────────────────────────────────────
def deep_links(sym: str) -> Dict[str, str]:
    """Public research pages, same set and order as street.deep_links."""
    lo = sym.lower()
    return {
        "Zacks": f"https://www.zacks.com/stock/quote/{sym}",
        "Danelfin": f"https://danelfin.com/stock/{sym}",
        "Bloomberg": f"https://www.bloomberg.com/quote/{sym}:US",
        "WSJ": f"https://www.wsj.com/market-data/quotes/{sym}",
        "Finviz": f"https://finviz.com/quote.ashx?t={sym}",
        "TradingView": f"https://www.tradingview.com/symbols/{sym}/",
        "Stocktwits": f"https://stocktwits.com/symbol/{sym}",
        "Yahoo": f"https://finance.yahoo.com/quote/{sym}",
        "Nasdaq": f"https://www.nasdaq.com/market-activity/stocks/{lo}",
        "SEC EDGAR": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sym}&type=&dateb=&owner=include&count=40",
        "Fintel": f"https://fintel.io/ss/us/{lo}",
        "iBorrowDesk": f"https://iborrowdesk.com/report/{sym}",
        "TipRanks": f"https://www.tipranks.com/stocks/{lo}",
        "MarketBeat": f"https://www.marketbeat.com/stocks/NASDAQ/{sym}/",
        "Halts": "https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts",
    }


FORM_FOR = {
    "offering": ("424B5", []), "atm": ("424B5", []), "registration": ("S-1", []),
    "resale": ("424B3", []), "effective": ("EFFECT", []), "unregistered_sale": ("8-K", ["3.02"]),
    "delisting_notice": ("8-K", ["3.01"]), "reverse_split": ("8-K", ["5.03"]),
    "toxic_financing": ("8-K", ["1.01"]), "going_concern": ("10-Q", []),
    "late_filing": ("NT 10-Q", []), "insider_sale_notice": ("144", []), "insider": ("4", []),
    "material_agreement": ("8-K", ["1.01"]), "other": ("8-K", ["7.01"]),
}
TAGS_FOR = {"offering": ["registered_direct"], "atm": ["atm"], "reverse_split": ["reverse_split"],
            "delisting_notice": ["bid_price_deficiency"], "toxic_financing": ["equity_line"]}


def filing(sym: str, cik: int, d: date, cat: str, accepted: Optional[str], n: int,
           asia: bool) -> dict:
    form, items = FORM_FOR[cat]
    if asia and form == "8-K":
        form, items = "6-K", []
    return {
        "symbol": sym, "cik": cik, "date": d.isoformat(), "accepted": accepted,
        "form": form, "items": items, "category": cat,
        "url": f"{EXAMPLE}/filings/{sym.lower()}-{n}.htm",
        "text_tags": TAGS_FOR.get(cat, []) if cat in TAGS_FOR and n % 2 == 0 else [],
    }


NEWS_TEMPLATES = [
    ("{name} prices $4.0 million registered direct offering (sample headline)", ["offering", "priced"], -1),
    ("{name} receives Nasdaq minimum bid price deficiency notice (sample headline)", ["deficiency", "delisting"], -1),
    ("{name} announces 1-for-20 reverse share split (sample headline)", ["reverse_split"], -1),
    ("{name} enters at-the-market sales agreement (sample headline)", ["atm", "dilution"], -1),
    ("{name} signs distribution partnership (sample headline)", ["partnership"], 1),
    ("{name} shares surge on heavy volume (sample headline)", [], 0),
    ("{name} announces warrant inducement (sample headline)", ["warrants", "dilution"], -1),
    ("{name} wins supply contract (sample headline)", ["contract"], 1),
]


def news_items(rng: random.Random, sym: str, name: str, gen: datetime, k: int) -> List[dict]:
    out = []
    picks = rng.sample(NEWS_TEMPLATES, k)
    for j, (title, tags, pol) in enumerate(picks):
        t = gen - timedelta(hours=rng.uniform(1, 150))
        out.append({
            "published": t.replace(microsecond=0).isoformat(), "title": title.format(name=name),
            "source": "Sample Wire", "url": f"{EXAMPLE}/news/{sym.lower()}-{j}",
            "tags": tags, "polarity": pol,
        })
    out.sort(key=lambda x: x["published"], reverse=True)
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def squeeze_bits(rng: random.Random, short: dict, fl: float, si: Optional[float],
                 dtc: Optional[float], sr: Optional[float], model_pct: float) -> Tuple[Optional[int], dict]:
    """Same weighting as score.squeeze_danger so the sample is consistent."""
    parts: Dict[str, Optional[float]] = {"model": model_pct / 100.0}
    fee = short.get("fee_rate")
    parts["fee"] = None if fee is None else min(1.0, fee / 100.0)
    avail = short.get("available")
    parts["scarcity"] = None if avail is None else (1.0 if avail < 10_000 else 0.6 if avail < 50_000 else 0.2 if avail < 250_000 else 0.0)
    parts["si"] = None if si is None else min(1.0, si / 0.30)
    parts["dtc"] = None if dtc is None else min(1.0, dtc / 5.0)
    parts["float"] = 1.0 if fl < 2e6 else 0.6 if fl < 5e6 else 0.2 if fl < 15e6 else 0.0
    parts["short_vol"] = None if sr is None else max(0.0, min(1.0, (sr - 0.35) / 0.35))
    w = {"model": 40, "fee": 15, "scarcity": 10, "si": 15, "dtc": 5, "float": 10, "short_vol": 5}
    num = sum(w[k] * v for k, v in parts.items() if v is not None)
    den = sum(w[k] for k, v in parts.items() if v is not None)
    return (int(round(100 * num / den)) if den else None), parts


class Sampler:
    """Holds the RNG and the clock so every generated object agrees."""

    def __init__(self, session: date, now: datetime, seed: int):
        self.rng = random.Random(seed)
        self.session = session
        self.now = now
        self.last = prev_trading_day(session)
        self.days = trading_days_before(session, config.CHART_SESSIONS)
        self.cik = 9_900_000

    # one full Pick ------------------------------------------------------
    def pick(self, row: tuple, rank: Optional[int], prob: float, *, zone: Optional[str] = None,
             fresh_offer: bool = False, sub_dollar: bool = False, twin_sim: Optional[float] = None) -> dict:
        rng = self.rng
        sym, name, country, sector, industry, exch, shape = row
        self.cik += rng.randint(3, 97)
        cik = self.cik
        asia = country in ASIA
        end = rng.uniform(0.35, 0.95) if sub_dollar else rng.uniform(1.2, 6.5)
        if sym == "SMPL1":
            end = 2.41
        base_vol = rng.uniform(1.5e5, 2.5e6)
        chart = synth_chart(rng, self.days, shape, end, base_vol)
        m = chart_metrics(chart)
        price = chart[-1][4]
        shares_out = rng.uniform(2.5e6, 4.0e7)
        flt = shares_out * rng.uniform(0.35, 0.9)
        halted = zone == "halted"
        if halted:
            chart[-1][5] = 0.0

        # shortability
        if zone == "none":
            short = {"status": "NONE", "available": 0, "fee_rate": None, "asof": None}
        else:
            fee = round(rng.choice([rng.uniform(0.3, 8), rng.uniform(25, 140), rng.uniform(60, 400)]), 2)
            if zone == "squeeze":
                fee = round(rng.uniform(250, 420), 2)
            avail = int(rng.choice([rng.uniform(4e3, 4e4), rng.uniform(5e4, 9e5), 10_000_000]))
            short = {"status": "HTB" if fee >= 20 else "ETB", "available": avail, "fee_rate": fee,
                     "asof": (self.now - timedelta(minutes=rng.randint(6, 40))).replace(microsecond=0).isoformat()}
        if sym == "SMPL1":
            short = {"status": "HTB", "available": 35_000, "fee_rate": 98.81,
                     "asof": (self.now - timedelta(minutes=14)).replace(microsecond=0).isoformat()}

        # pre-market
        pre = None
        if rng.random() < 0.7 or sym == "SMPL1":
            gap = 0.224 if sym == "SMPL1" else rng.gauss(0.04, 0.12)
            pre = {"price": _round_px(price * (1 + gap)), "prev_close": price, "gap_pct": round(gap, 4),
                   "volume": float(int(rng.uniform(2e4, 9e5))),
                   "asof": (self.now - timedelta(minutes=rng.randint(3, 25))).replace(microsecond=0).isoformat(),
                   "source": "nasdaq"}

        # squeeze danger
        si = rng.uniform(0.02, 0.35)
        dtc = rng.uniform(0.2, 4.5)
        sr = rng.uniform(0.3, 0.7)
        sq_pct = rng.uniform(40, 99) if zone == "squeeze" else rng.uniform(10, 85)
        if zone == "squeeze":
            flt = rng.uniform(0.8e6, 1.8e6)
            si, dtc = 0.34, 4.8
        sqd, parts = squeeze_bits(rng, short, flt, si, dtc, sr, sq_pct)
        if zone == "squeeze":
            sqd = max(sqd or 0, 78)
        elif sqd is not None and sqd >= 70:
            sqd = 62  # board names are by definition outside the squeeze zone

        # filings, newest first
        cats = rng.sample(["registration", "resale", "delisting_notice", "reverse_split", "effective",
                           "going_concern", "insider_sale_notice", "material_agreement", "atm", "other"],
                          rng.randint(2, 6))
        fl: List[dict] = []
        for j, cat in enumerate(cats):
            d = self.days[-rng.randint(2, len(self.days) - 1)]
            fl.append(filing(sym, cik, d, cat, iso_et(d, rng.randint(7, 19), rng.randint(0, 59)), j + 1, asia))
        if fresh_offer:
            acc = self.now - timedelta(hours=rng.uniform(2, 14))
            ad = acc.astimezone(ET).date()
            fl.append(filing(sym, cik, ad, "offering", acc.astimezone(ET).replace(microsecond=0).isoformat(), 0, asia))
        fl.sort(key=lambda e: (e["date"], e["accepted"] or ""), reverse=True)
        fl = fl[:12]

        nws = news_items(rng, sym, name, self.now, rng.randint(0, 5))

        # reasons + flags (mirrors score.build_reasons wording)
        reasons: List[dict] = []
        flags: List[str] = []
        if m["r1"] is not None and m["r1"] >= 0.2:
            rv = f" on {m['rvol1']:.0f}× its normal volume" if (m["rvol1"] or 0) >= 2 else ""
            reasons.append({"family": "exhaustion", "text": f"Closed {_pct(m['r1'])} last session{rv}",
                            "url": None, "strength": 3 if m["r1"] >= 0.5 else 2})
            flags.append(f"PUMP {_pct(m['r1'])}")
        elif m["r3"] is not None and m["r3"] >= 0.5:
            reasons.append({"family": "exhaustion", "text": f"Up {_pct(m['r3'])} over the last 3 sessions",
                            "url": None, "strength": 2})
            flags.append(f"RUN {_pct(m['r3'])}")
        if pre and abs(pre["gap_pct"]) >= 0.10:
            reasons.append({"family": "exhaustion",
                            "text": f"Pre-market {_pct(pre['gap_pct'])} at "
                                    f"{datetime.fromisoformat(pre['asof']).astimezone(ET):%H:%M} ET vs the last close (sample)",
                            "url": None, "strength": 3 if pre["gap_pct"] >= 0.3 else 2})
            flags.append(f"GAP {_pct(pre['gap_pct'])}")
        for e in fl:
            cat = e["category"]
            if cat == "offering":
                reasons.append({"family": "dilution", "text": f"Offering — {e['form']} filed {e['date']} (sample filing)",
                                "url": e["url"], "strength": 3})
                flags.append("OFFERING")
            elif cat == "atm":
                reasons.append({"family": "dilution", "text": f"At-the-market program — {e['form']} filed {e['date']} (sample filing)",
                                "url": e["url"], "strength": 2})
                flags.append("ATM")
            elif cat == "delisting_notice":
                reasons.append({"family": "decay", "text": f"Exchange deficiency notice filed {e['date']} (sample filing)",
                                "url": e["url"], "strength": 2})
                flags.append("DEFICIENCY")
            elif cat == "reverse_split":
                reasons.append({"family": "decay", "text": f"1:20 reverse split on {e['date']} (sample)",
                                "url": e["url"], "strength": 2})
                flags.append("R/S 1:20")
        if (m["dd_52w"] or 0) <= -0.8:
            reasons.append({"family": "decay", "text": f"{_pct(m['dd_52w'])} from its high in the chart window",
                            "url": None, "strength": 1})
        if sr >= 0.55:
            reasons.append({"family": "flow", "text": f"{sr * 100:.0f}% of the last 5 sessions' reported volume was short sales (sample)",
                            "url": None, "strength": 1})
        if price < 1:
            flags.append("SUB-$1")
        if short["status"] == "HTB" and short["fee_rate"] is not None:
            flags.append(f"HTB {short['fee_rate']:.0f}%")
        elif short["status"] == "NONE":
            flags.append("NO BORROW")
        if asia:
            flags.append("ASIA")
        if halted:
            flags.append("HALTED")
        reasons.sort(key=lambda r: -r["strength"])
        seen: set = set()
        flags = [f for f in flags if not (f in seen or seen.add(f))]

        fams = {
            "dilution": min(100, 30 + 12 * len([e for e in fl if e["category"] in ("offering", "atm", "registration", "resale")]) + rng.randint(0, 20)),
            "exhaustion": int(max(0, min(100, 50 + 60 * (m["r5"] or 0) + rng.randint(-10, 10)))),
            "decay": int(max(0, min(100, 40 - 50 * (m["dd_52w"] or 0) + rng.randint(-10, 10)))),
            "flow": rng.randint(20, 95),
            "street": None if rng.random() < 0.7 else rng.choice([50, 60, 75, 80]),
            "news": None if not nws else int(max(0, min(100, 50 + 20 * sum(1 for n in nws if n["polarity"] < 0) - 15 * sum(1 for n in nws if n["polarity"] > 0)))),
        }
        analyst = None
        if fams["street"] is not None:
            analyst = {"mean_rating": rng.choice(["Hold", "Buy", "Sell"]), "n_analysts": rng.randint(1, 3),
                       "changes": [{"date": self.days[-rng.randint(3, 60)].isoformat(), "firm": "Sample Securities",
                                    "action": rng.choice(["Downgrade", "Initiated", "Reiterated"]),
                                    "from": "Buy", "to": "Hold"}],
                       "price_target": round(price * rng.uniform(1.2, 3.0), 2),
                       "source_url": f"{EXAMPLE}/analyst/{sym.lower()}"}

        bigdump = prob * rng.uniform(0.18, 0.35)
        psq = min(0.6, BASE_RATE_SQUEEZE * (sq_pct / 50.0) * rng.uniform(0.8, 1.4))
        exp_oc = -0.012 - 0.12 * (prob - BASE_RATE_DUMP) + rng.gauss(0, 0.004)
        si_interest = si * flt
        return {
            "rank": rank, "symbol": sym, "name": name, "exchange": exch, "country": country,
            "sector": sector, "industry": industry,
            "price": price, "prev_close": price, "market_cap": round(price * shares_out, 0),
            "premarket": pre,
            "prob_dump": round(prob, 4), "prob_bigdump": round(bigdump, 4), "prob_squeeze": round(psq, 4),
            "exp_oc": round(exp_oc, 4), "lift": round(prob / BASE_RATE_DUMP, 3),
            "score": int(min(100, round(80 + 70 * (prob - 0.12)))),
            "model_used": "m1" if pre else "m0",
            "squeeze_danger": sqd, "squeeze_parts": {k: (None if v is None else round(v, 3)) for k, v in parts.items()},
            "shortability": short, "families": fams, "reasons": reasons, "flags": flags,
            "metrics": {
                **{k: (None if v is None else round(v, 4)) for k, v in m.items() if k != "dvol20"},
                "rs_count_2y": rng.randint(0, 3), "n_offer_90": sum(1 for e in fl if e["category"] in ("offering", "atm")),
                "n_reg_90": sum(1 for e in fl if e["category"] == "registration"),
                "short_ratio_5d": round(sr, 4), "dvol20": round(m["dvol20"] or 0, 0),
                "si_pct_float": round(si, 4), "days_to_cover": round(dtc, 2),
                "float": round(flt, 0), "shares_out": round(shares_out, 0),
                "short_interest": {"settlement_date": self.days[-8].isoformat(), "interest": round(si_interest, 0),
                                   "avg_daily_volume": round(base_vol, 0), "days_to_cover": round(dtc, 2)},
            },
            "filings": fl, "news": nws,
            "street": {"analyst": analyst, "danelfin": None},
            "links": deep_links(sym),
            "chart": chart,
            "inhd_similarity": twin_sim,
            **({"zone_reason": zone_text(zone, short, sqd, parts)} if zone else {}),
        }


def zone_text(zone: str, short: dict, sqd: Optional[int], parts: Optional[dict] = None) -> str:
    """Same wording rules as the pipeline's ``cli._zone_reason`` (driven by the squeeze parts)."""
    if zone == "none":
        return "No shares available to borrow at IBKR — you likely can't short it."
    if zone == "halted":
        return "Halted / no trades last session."
    parts = parts or {}
    bits = []
    if (parts.get("fee") or 0) >= 0.5:
        bits.append(f"borrow fee {short.get('fee_rate') or 0:.0f}%/yr")
    if (parts.get("scarcity") or 0) >= 0.6:
        bits.append("very few shares to borrow")
    if (parts.get("float") or 0) >= 0.6:
        bits.append("tiny float")
    if (parts.get("si") or 0) >= 0.6:
        bits.append("crowded short interest")
    if (parts.get("model") or 0) >= 0.8:
        bits.append("model sees high odds of a +20% intraday spike")
    return f"Squeeze danger {sqd}/100: " + (", ".join(bits) or "multiple crowding signals")


# ── feeds ────────────────────────────────────────────────────────────────
def build_today(s: Sampler) -> dict:
    rng = s.rng
    probs = [0.312] + sorted((rng.uniform(0.14, 0.29) for _ in range(24)), reverse=True)
    sims = {"SMPL1": 0.83, "SMPL4": 0.77, "SMPL7": 0.71, "SMPL11": 0.66}
    board = []
    for i, row in enumerate(ROSTER[:25]):
        board.append(s.pick(row, i + 1, probs[i], fresh_offer=row[0] in ("SMPL1", "DEMO4", "SMPL3"),
                            sub_dollar=row[0] in ("SMPL2", "DEMO5", "SMPL5", "DEMO12", "SMPL9"),
                            twin_sim=sims.get(row[0])))
    zone_rows = {"SQZ1": "squeeze", "SQZ2": "squeeze", "SQZ3": "halted", "SQZ4": "none"}
    squeeze_zone = [s.pick(r, None, rng.uniform(0.2, 0.36), zone=zone_rows[r[0]]) for r in ROSTER[25:29]]

    twins = []
    twin_rows = [(ROSTER[0], 0.83), (ROSTER[9], 0.77), ROSTER_TWIN(0, 0.74), (ROSTER[16], 0.71),
                 ROSTER_TWIN(1, 0.69), (ROSTER[24], 0.66), ROSTER_TWIN(2, 0.61)]
    by_sym = {p["symbol"]: p for p in board}
    for row, sim in twin_rows:
        p = by_sym.get(row[0]) or s.pick(row, None, rng.uniform(0.09, 0.2), twin_sim=sim)
        twins.append({
            "symbol": p["symbol"], "name": p["name"], "similarity": sim,
            "reasons": ["Asia-linked issuer" if p["country"] in ASIA else "US issuer",
                        "Market cap under $25M" if (p["market_cap"] or 0) < 25e6 else "Micro-cap",
                        "Repeated reverse splits (sample)", "Down more than 80% from its high"][: rng.randint(2, 4)],
            "price": p["price"], "market_cap": p["market_cap"], "country": p["country"],
            "prob_dump": p["prob_dump"], "rank": p["rank"], "score": p["score"], "flags": p["flags"],
            "pick": p,
        })

    wire = []
    for p in board[:10] + squeeze_zone[:2]:
        for e in p["filings"][:2]:
            if e["category"] in ("offering", "atm", "delisting_notice", "reverse_split", "registration", "resale"):
                sev = 3 if e["category"] in ("offering", "atm") else 2
                wire.append({"time": e["accepted"] or e["date"], "symbol": p["symbol"],
                             "kind": {"offering": "Offering", "atm": "At-the-market program",
                                      "delisting_notice": "Exchange deficiency / delisting notice",
                                      "reverse_split": "Reverse split", "registration": "Registration statement",
                                      "resale": "Resale registration (selling holders)"}[e["category"]],
                             "headline": f"{e['form']} — sample filing" + (f" ({', '.join(e['text_tags'])})" if e["text_tags"] else ""),
                             "url": e["url"], "source": "SEC EDGAR (sample)", "severity": sev})
        if p["premarket"] and abs(p["premarket"]["gap_pct"]) >= 0.2:
            wire.append({"time": p["premarket"]["asof"], "symbol": p["symbol"], "kind": "Pre-market",
                         "headline": f"Pre-market {p['premarket']['gap_pct'] * 100:+.0f}% vs last close",
                         "url": None, "source": "nasdaq", "severity": 2 if p["premarket"]["gap_pct"] >= 0.4 else 1})
        for n in p["news"][:1]:
            if n["polarity"] < 0:
                wire.append({"time": n["published"], "symbol": p["symbol"], "kind": "News", "headline": n["title"],
                             "url": n["url"], "source": n["source"], "severity": 2})
    def _instant(ts: str) -> datetime:  # real instants: offsets differ; date-only sorts as midnight ET
        d = datetime.fromisoformat(ts)
        return d if d.tzinfo else d.replace(tzinfo=ET)

    wire.sort(key=lambda w: _instant(w["time"]), reverse=True)

    ref = build_reference(s)
    gen = s.now.replace(microsecond=0).isoformat()
    sources = [
        ("Nasdaq screener", True), ("Yahoo Finance prices", True), ("Nasdaq quotes", True),
        ("SEC EDGAR", True), ("SEC full-text search", True), ("IBKR borrow", True),
        ("FINRA short volume", True), ("Nasdaq short interest", True), ("Yahoo float", True),
        ("Google News", True), ("Nasdaq analyst", True), ("Danelfin API", False),
    ]
    return {
        "sample": True,
        "generated_at": gen,
        "session_date": s.session.isoformat(),
        "features_asof": s.last.isoformat(),
        "run": "morning",
        "market_phase": "pre-market",
        "universe": {"listed": 7017, "eligible": 3400, "scored": 3112},
        "model": {"version": "m1", "use_open": True, "m1_names": 131, "base_rate_dump": BASE_RATE_DUMP,
                  "trained_through": s.last.isoformat(), "oos_auc": 0.68},
        "top": board[0],
        "board": board,
        "squeeze_zone": squeeze_zone,
        "catalyst_wire": wire,
        "twins": twins,
        "reference": ref,
        "earnings": [
            {"symbol": "DEMO2", "time": "time-pre-market", "eps_forecast": -0.21, "n_ests": 1, "market_cap": 18e6, "in_universe": True},
            {"symbol": "SMPL8", "time": "time-after-hours", "eps_forecast": -0.05, "n_ests": 2, "market_cap": 41e6, "in_universe": True},
            {"symbol": "DEMO30", "time": "time-not-supplied", "eps_forecast": None, "n_ests": None, "market_cap": 3.2e9, "in_universe": False},
        ],
        "sources": [{"name": n, "ok": ok,
                     "detail": "sample — no request was made" if ok else "DANELFIN_API_KEY not set (sample)",
                     "asof": gen} for n, ok in sources],
        "disclaimer": DISCLAIMER,
    }


def ROSTER_TWIN(i: int, sim: float) -> Tuple[tuple, float]:  # noqa: N802 — reads like a constant lookup
    return ROSTER[29 + i], sim


def build_reference(s: Sampler) -> dict:
    """INHD with round, clearly-labelled placeholder numbers and no invented facts."""
    days = trading_days_before(s.session, 260)
    ref_date = date.fromisoformat(config.REFERENCE_SHORT_DATE)
    ref_i = max(i for i, d in enumerate(days) if d <= ref_date)
    rng = random.Random(7)  # independent stream: INHD placeholder never shifts with roster edits
    # placeholder path: 2.00 at the reference date, 1.50 at the last bar
    lr = [rng.gauss(-0.006, 0.05) for _ in days]
    path = [0.0]
    for x in lr[1:]:
        path.append(path[-1] + x)
    # pin two anchors exactly by piecewise linear re-levelling in log space
    a0 = math.log(2.0) - path[ref_i]
    a1 = math.log(1.5) - path[-1]
    closes = []
    for i, p in enumerate(path):
        adj = a0 if i <= ref_i else a0 + (a1 - a0) * (i - ref_i) / (len(path) - 1 - ref_i)
        closes.append(math.exp(p + adj))
    chart = []
    prev = closes[0]
    for d, c in zip(days, closes):
        o = prev * math.exp(rng.gauss(0, 0.02))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.03)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.03)))
        chart.append([d.isoformat(), _round_px(o), _round_px(h), _round_px(lo), _round_px(c),
                      float(int(4e5 * math.exp(rng.gauss(0, 0.5))))])
        prev = c
    chart[ref_i][4] = 2.0
    chart[-1][4] = 1.5
    for r in chart:
        r[2] = max(r[1:5])
        r[3] = min(r[1:5])
    return {
        "symbol": config.REFERENCE_SYMBOL,
        "name": "Inno Holdings Inc.",
        "price": 1.5,
        "prev_close": 1.5,
        "short_ref_date": config.REFERENCE_SHORT_DATE,
        "short_ref_price": 2.0,
        "change_since_ref": -0.25,
        "chart": chart,
        "borrow": {"status": "HTB", "available": 10_000, "fee_rate": 100.0, "asof": None},
        "pick": None,
        "rank": None,
        "filings": [],
        "news": [],
        "notes": [
            "PLACEHOLDER — sample mode. Every INHD number here (prices, chart, borrow fee, availability) is a "
            "round synthetic value, not a quote. Real figures appear after the first pipeline run.",
            "No filings or news are shown for INHD in sample mode: we never invent facts about a real company.",
        ],
    }


def build_model(s: Sampler) -> dict:
    rng = s.rng
    n_days = 480
    days: List[date] = []
    d = s.last
    for _ in range(n_days):
        days.append(d)
        d = prev_trading_day(d)
    days.reverse()

    def sim(edge: float) -> dict:
        daily = []
        for dd in days:
            oc1 = rng.gauss(-edge, 0.09)
            if rng.random() < 0.05:
                oc1 = rng.uniform(0.15, 0.6)  # the squeezes that make shorting hard
            daily.append([dd.isoformat(), round(oc1, 4), round(rng.gauss(-edge * 0.5, 0.03), 4),
                          round(rng.gauss(-0.002, 0.006), 4)])
        pnl = [-r[1] for r in daily]
        net = [p - 0.01 for p in pnl]
        eq, peak, mdd = 0.0, 0.0, 0.0
        for x in net:
            eq += x
            peak = max(peak, eq)
            mdd = min(mdd, eq - peak)
        return {"daily": daily, "gross_total": round(sum(pnl), 4), "net_total": round(sum(net), 4),
                "win_rate": round(sum(1 for p in pnl if p > 0) / len(pnl), 4),
                "max_drawdown": round(mdd, 4), "cost_assumption": 0.01}

    def metrics(auc: float, top1: float) -> dict:
        return {"auc": auc, "brier": round(0.093 - (auc - 0.6) * 0.05, 4), "top1_hit": top1,
                "top10_hit": round(top1 * 0.82, 4), "top_decile_hit": round(top1 * 0.66, 4),
                "top1_mean_oc": round(-0.021 - (top1 - 0.3) * 0.1, 4), "top10_mean_oc": -0.014, "days": n_days}

    def calib(skew: float) -> List[dict]:
        out = []
        for b in range(1, 11):
            pred = 0.02 + 0.042 * (b - 1) + (0.05 if b == 10 else 0.0)
            out.append({"bin": b, "pred": round(pred, 4),
                        "actual": round(max(0.0, pred * (1 + rng.gauss(skew, 0.08))), 4),
                        "n": int(185_000 + rng.randint(-900, 900))})
        return out

    feats = [("dilution", "n_offer_90"), ("dilution", "days_since_offer"), ("dilution", "n_reg_180"),
             ("exhaustion", "r1"), ("exhaustion", "gap_open"), ("exhaustion", "rsi2"), ("exhaustion", "upper_wick"),
             ("decay", "dd_52w"), ("decay", "rs_count_2y"), ("decay", "dist_ma200"), ("decay", "price_log"),
             ("flow", "rvol1"), ("flow", "dvol20_log"), ("flow", "r1_rank"),
             ("market", "iwm_r1"), ("market", "breadth")]
    imp = sorted(({"family": f, "feature": c, "importance": round(rng.uniform(0.0005, 0.02), 5)} for f, c in feats),
                 key=lambda r: -r["importance"])
    return {
        "sample": True,
        "trained_at": (s.now - timedelta(days=2)).replace(microsecond=0).isoformat(),
        "trained_through": s.last.isoformat(),
        "n_rows": 1_842_117, "n_symbols": 3_406, "n_days": 748,
        "targets": {"dump": "open→close ≤ −5%", "bigdump": "open→close ≤ −15%", "squeeze": "open→high ≥ +20%"},
        "base_rate": {"dump": BASE_RATE_DUMP, "bigdump": BASE_RATE_BIGDUMP, "squeeze": BASE_RATE_SQUEEZE},
        "oos": {"m0": metrics(0.64, 0.29), "m1": metrics(0.68, 0.34)},
        "calibration": {"m0": calib(-0.05), "m1": calib(0.0)},
        "importance": imp,
        "sim": {"m0": sim(0.012), "m1": sim(0.02)},
        "caveats": [
            "SAMPLE — synthetic numbers for layout only. No model has been trained yet.",
            "Backtest assumes you could borrow the #1 at the open every day; real borrow is often unavailable.",
            "A 1% round-trip cost is assumed; hard-to-borrow fees and slippage on thin names can be far higher.",
        ],
    }


STUDIES = [
    ("up20", "Up 20%+ yesterday", "Names that closed up 20% or more the day before.", 0.19),
    ("up50", "Up 50%+ yesterday", "Names that closed up 50% or more the day before.", 0.24),
    ("up100", "Up 100%+ yesterday", "Names that at least doubled the day before.", 0.29),
    ("up200", "Up 200%+ yesterday", "Names that tripled the day before.", 0.33),
    ("run3_100", "3-day run over +100%", "Up more than 100% over three sessions.", 0.26),
    ("gap30", "Gap-up over 30% (same day)", "Opened 30%+ above the prior close; open→close that same day.", 0.31),
    ("rs_5", "Reverse split in the last 5 sessions", "A reverse split took effect within 5 sessions.", 0.21),
    ("rs_30", "Reverse split in the last 30 sessions", "A reverse split took effect within 30 sessions.", 0.17),
    ("offer_1", "Offering filed ≤ 1 session ago", "A prospectus/offering filing landed at most one session earlier.", 0.27),
    ("reg_30", "Registration filed ≤ 30 days", "An S-1/F-1/S-3/F-3 was filed in the last 30 days.", 0.15),
    ("delist_90", "Deficiency notice ≤ 90 days", "An exchange deficiency or delisting notice in the last 90 days.", 0.16),
    ("sub1", "Price under $1", "Closed below one dollar.", 0.14),
    ("dd90", "52-week drawdown over 90%", "More than 90% below the 52-week high.", 0.15),
    ("rsi85", "RSI(14) above 85", "Fourteen-day RSI above 85.", 0.22),
    ("asia_ipo2y", "Asia-linked, IPO ≤ 2 years", "Asia-linked issuers that listed within two years.", 0.18),
    ("inhd_profile", "INHD profile", "Asia-linked, 2+ reverse splits in 2 years, drawdown over 90%.", 0.2),
]


def study(rng: random.Random, sid: str, title: str, plain: str, p: float, n: int) -> dict:
    half = 1.96 * math.sqrt(p * (1 - p) / max(n / 6, 1))  # cluster-ish widening
    return {
        "id": sid, "title": title, "plain": f"{plain} (sample)", "condition": sid, "n": n,
        "n_symbols": max(1, int(n / rng.uniform(2.5, 9))),
        "pct_red_oc": round(min(0.95, 0.52 + p), 4), "pct_dump": round(p, 4),
        "pct_bigdump": round(p * 0.24, 4), "pct_squeeze": round(0.04 + p * 0.3, 4),
        "median_oc": round(-0.004 - 0.1 * (p - BASE_RATE_DUMP), 4),
        "mean_oc": round(-0.002 - 0.08 * (p - BASE_RATE_DUMP), 4),
        "median_c5": round(-0.01 - 0.3 * (p - BASE_RATE_DUMP), 4),
        "ci_pct_dump": [round(max(0.0, p - half), 4), round(min(1.0, p + half), 4)],
        "lift_dump": round(p / BASE_RATE_DUMP, 3),
    }


def build_evidence(s: Sampler) -> dict:
    rng = s.rng
    base = study(rng, "baseline", "All eligible name-days", "Every eligible small/micro-cap session",
                 BASE_RATE_DUMP, 1_842_117)
    studies = [base] + [study(rng, sid, t, pl, p, int(rng.uniform(180, 9000)))
                        for sid, t, pl, p in STUDIES]
    return {"sample": True, "generated_at": (s.now - timedelta(days=2)).replace(microsecond=0).isoformat(),
            "baseline": base, "studies": studies}


def build_scorecard(s: Sampler, today: dict) -> dict:
    rng = s.rng
    days_out = []
    d = s.last
    for k in range(12):
        sym = ROSTER[rng.randint(0, 24)][0]
        o = round(rng.uniform(0.6, 5.0), 4)
        oc = rng.gauss(-0.035, 0.09)
        missing = k == 7
        top_out: Dict[str, Any]
        if missing:
            top_out = {"missing": True, "note": "no regular-session bar (halted or no trades)"}
        else:
            c = o * (1 + oc)
            hi = max(o, c) * (1 + abs(rng.gauss(0, 0.06)))
            lo = min(o, c) * (1 - abs(rng.gauss(0, 0.05)))
            top_out = {"open": o, "high": round(hi, 4), "low": round(lo, 4), "close": round(c, 4),
                       "oc": round(oc, 4), "ol": round(lo / o - 1, 4), "oh": round(hi / o - 1, 4),
                       "dump": oc <= config.DUMP_THRESHOLD, "squeezed": hi / o - 1 >= config.SQUEEZE_THRESHOLD}
        pub = datetime(d.year, d.month, d.day, 7, 55, tzinfo=ET).isoformat()
        days_out.append({
            "session_date": d.isoformat(), "run": "morning", "published_at": pub, "first_published_at": pub,
            "model": "m1",
            "top": {"symbol": sym, "prob_dump": round(rng.uniform(0.22, 0.4), 4), "score": rng.randint(95, 100),
                    "premarket": None, "squeeze_danger": rng.randint(15, 65)},
            "board": [{"rank": i + 1, "symbol": ROSTER[i][0], "prob_dump": round(0.3 - 0.006 * i, 4)} for i in range(10)],
            "outcome": {
                "graded_at": datetime(d.year, d.month, d.day, 17, 40, tzinfo=ET).isoformat(),
                "universe_n": 3100 + rng.randint(-60, 60),
                "universe_mean_oc": round(rng.gauss(-0.003, 0.004), 4),
                "universe_dump_rate": round(rng.uniform(0.09, 0.14), 4),
                "top": top_out,
                "board_n": 25, "board_mean_oc": round(rng.gauss(-0.018, 0.02), 4),
                "board_dump_rate": round(rng.uniform(0.16, 0.36), 4),
            },
        })
        d = prev_trading_day(d)
    tops = [x["outcome"]["top"] for x in days_out if not x["outcome"]["top"].get("missing")]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "sample": True,
        "asof": s.now.replace(microsecond=0).isoformat(),
        "days": days_out,
        "live": {
            "n_days": len(days_out), "top_n": len(tops),
            "top_dump_rate": mean([1.0 if t["dump"] else 0.0 for t in tops]),
            "top_mean_oc": mean([t["oc"] for t in tops]),
            "top_squeeze_rate": mean([1.0 if t["squeezed"] else 0.0 for t in tops]),
            "board_dump_rate": mean([x["outcome"]["board_dump_rate"] for x in days_out]),
            "board_mean_oc": mean([x["outcome"]["board_mean_oc"] for x in days_out]),
            "universe_dump_rate": mean([x["outcome"]["universe_dump_rate"] for x in days_out]),
            "universe_mean_oc": mean([x["outcome"]["universe_mean_oc"] for x in days_out]),
        },
    }


# ── entry ────────────────────────────────────────────────────────────────
def generate(session: date, now: datetime, seed: int = 42, late: bool = False) -> Dict[str, dict]:
    """Build all four sample feeds in memory (deterministic for a given input).

    ``late`` sets the optional ``today["late"]`` flag the pipeline writes when a
    morning run publishes after the open (shown, but kept out of the record)."""
    s = Sampler(session, now, seed)
    today = build_today(s)
    if late:
        today["late"] = True
        today["market_phase"] = "open"
    return {
        "today.json": today,
        "model.json": build_model(s),
        "evidence.json": build_evidence(s),
        "scorecard.json": build_scorecard(s, today),
    }


def is_real_feed(p: Path) -> bool:
    """True when ``p`` exists and is not a sample (i.e. real pipeline output)."""
    if not p.exists():
        return False
    try:
        return not bool(json.loads(p.read_text()).get("sample"))
    except (OSError, ValueError, AttributeError):
        return True  # unreadable ≠ sample: don't clobber what we can't identify


def write(feeds: Dict[str, dict], out: Path, force: bool = False) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    blocked = [n for n in feeds if is_real_feed(out / n)]
    if blocked and not force:
        raise SystemExit(f"refusing to overwrite real pipeline output: {', '.join(blocked)} (use --force)")
    written = []
    for name, obj in feeds.items():
        p = out / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        tmp.replace(p)
        written.append(p)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=config.SITE_DATA)
    ap.add_argument("--session-date", type=date.fromisoformat, default=None)
    ap.add_argument("--now", type=datetime.fromisoformat, default=None, help="ISO timestamp with offset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite real (non-sample) feeds")
    ap.add_argument("--late", action="store_true", help='mark today.json as published after the open ("late": true)')
    a = ap.parse_args(argv)
    now = a.now or (datetime.now(timezone.utc) - timedelta(minutes=7))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    session = a.session_date or target_session(now.astimezone(ET))
    for p in write(generate(session, now, a.seed, a.late), a.out, a.force):
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
