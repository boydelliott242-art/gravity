"""Assemble docs/data/size_research.json from the saved research outputs
(no hand-typed numbers): r7 (same-day tiers, corrected point-in-time,
production costs), r3 + r6 (20-session before/after), r6 (main #1)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data" / "research"
r7 = json.loads((R / "size_tier_r7.json").read_text())
r6 = json.loads((R / "size_tier_r6.json").read_text())
r3 = json.loads((R / "size_tier_r3.json").read_text())["H20 reg liquid"]


def dc(get):
    return {"dev": get("dev"), "confirm": get("confirm")}


intraday = []
for key, label, q in (("micro", "Micro-cap (today's #1 rule)", "10000"), ("$500K", "Takes $500K", "500000"), ("$1M", "Takes $1M", "1000000")):
    row = lambda p, k: r7[f"{p}|{key}"].get(k)  # noqa: E731
    intraday.append({"tier": label, "size_tested": {"10000": "$10K", "500000": "$500K", "1000000": "$1M"}[q],
                     "names_per_day": dc(lambda p: row(p, "names_per_day")), "base_dump": dc(lambda p: row(p, "base_dump")),
                     "top1_hit": dc(lambda p: row(p, "top1_hit")), "top1_mean_oc": dc(lambda p: row(p, "top1_mean_oc")),
                     "cost": dc(lambda p: row(p, f"cost_{q}")), "net": dc(lambda p: row(p, f"net_{q}")),
                     "net_tight": dc(lambda p: row(p, f"net_tick_{q}"))})
multi = {t: {"biased_net": dc(lambda p: r3[f"{p}|{t}"]["net"]), "biased_tier": dc(lambda p: r3[f"{p}|{t}"]["tier_mean_r"]),
             "pit_net": dc(lambda p: r6[f"h20|{p}|{t}"]["net"]), "pit_tier": dc(lambda p: r6[f"h20|{p}|{t}"]["tier_mean"]),
             "pit_t": dc(lambda p: r6[f"h20|{p}|{t}"]["t"]), "indep": dc(lambda p: r6[f"h20|{p}|{t}"]["indep_trades"])}
         for t in ("$500K", "$1M")}
main = {"pit": {p: {"hit": r6[f"main|{p}"]["hit"], "mean_oc": r6[f"main|{p}"]["mean_oc"], "median_oc": r6[f"main|{p}"]["median_oc"],
                    "auc": r6[f"main|{p}"]["auc"], "base": r6[f"main|{p}"]["base"],
                    "cost10k": r7[f"{p}|micro"]["cost_10000"], "net10k": r7[f"{p}|micro"]["net_10000"],
                    "net10k_tight": r7[f"{p}|micro"]["net_tick_10000"]} for p in ("dev", "confirm")}}
m = json.loads((ROOT / "docs" / "data" / "model.json").read_text())


def sg(x, digits=1):
    """Signed percent with a true minus; ~0 reads as 'about 0%'."""
    if x is None:
        return "—"
    v = x * 100
    if abs(v) < 0.05:
        return "about 0%"
    return f"{'+' if v > 0 else '−'}{abs(v):.{digits}f}%"
o, sim = m["oos"]["m0"], m["sim"]["m0"]
re10 = (sim.get("realistic") or {}).get("10000") or {}
doc = {
    "generated": m.get("trained_at", "")[:10],
    "question": "Can GRAVITY's daily #1 be shorted at $500K–$1M without moving the price — and keep its accuracy?",
    "method": [
        "Walk-forward, out of sample: monthly test folds, 5-session embargo (21 for 20-session holds), isotonic calibration on a later held-out slice — the production recipe.",
        "Variants were chosen on a DEVELOPMENT window (test months Jan–Oct 2025) and reported on an untouched CONFIRMATION window (Nov 2025 – Sep 2026), so picking the best of many cannot inflate the result.",
        "Point-in-time universe: every listed stock, a row kept only while the company was ≤ $2B at the time (traded price × the share count on its latest SEC cover page, split-adjusted). Delisted stocks are still missing.",
        "Capacity: ≤ 5% of the session's expected dollar volume per leg and ≤ 0.5% estimated one-way impact (square-root law, Y = 0.7, daily volatility), with a conservative next-session volume and, live, IBKR's lendable shares.",
        "Costs: impact on both legs + one spread. The spread is the least certain input: the close-high-low estimate is trusted only between one tick ($0.0001 under $1, $0.01 above) and a liquidity cap (3% under $1M/day, 2% to $5M, 1% above). 'Tight spread' shows the result if real spreads sit near the tick.",
        "About 25 variants tested: production features; liquidity / intraday-habit features; bagging; liquid-only training; the pre-market gap; dump-minus-pump ranking; direct return models; protective stops; 5/10/20-session holds; two universe definitions.",
    ],
    "intraday": intraday,
    "multi_day_20": multi,
    "main_pit": main,
    "official": {"pub1_hit": o.get("pub1_hit"), "pub1_mean_oc": o.get("pub1_mean_oc"), "pub1_median_oc": o.get("pub1_median_oc"),
                 "base": o.get("universe_daily_hit"), "net10k": re10.get("net_mean"), "cost10k": re10.get("mean_cost")},
    "verdict": [
        f"The daily #1 still finds dumps — {o['pub1_hit']*100:.0f}% of published #1s fell 5%+ open→close vs {o['universe_daily_hit']*100:.0f}% for the average name (point-in-time backtest), averaging {sg(o['pub1_mean_oc'])} before costs.",
        f"But these are thin stocks. With each pick's own estimated cost, the average trade nets {sg(re10.get('net_mean'))} even at $10K (estimated cost ≈ {re10.get('mean_cost', 0)*100:.1f}% round trip) — roughly breakeven. If real spreads are near the tick it is better; the live tape now records real bid/ask spreads to settle this.",
        "Names that can take $500K–$1M: the model still finds dump candidates at about 5× the normal rate, but their average same-day move is too small to pay the costs at that size. No variant — stops, direction-aware ranking, the pre-market gap, liquid-only models — fixed that.",
        "20-session shorts in liquid names looked excellent (+6–12% per trade) on today's small-cap list, but that list over-represents stocks that had already collapsed. On a point-in-time universe the result is mixed and statistically indistinguishable from zero (about 11 independent trades per period). Not confirmed, not shipped.",
        "So the honest answer to “$500K–$1M without losing accuracy” is no — not with these signals and free data. GRAVITY shows every pick's capacity, cost at your size and breakeven, so you can size each trade to what the stock can actually absorb.",
    ],
}
(ROOT / "docs" / "data" / "size_research.json").write_text(json.dumps(doc, indent=1))
print("written", len(json.dumps(doc)))
