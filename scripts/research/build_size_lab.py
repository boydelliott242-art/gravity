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
_lifts = [r7[f"{p}|{t}"]["top1_hit"] / r7[f"{p}|{t}"]["base_dump"] for p in ("dev", "confirm") for t in ("$500K", "$1M")
          if r7[f"{p}|{t}"].get("base_dump")]
lift_lo, lift_hi = min(_lifts), max(_lifts)
_bn = [r3[f"{p}|{t}"]["net"] for p in ("dev", "confirm") for t in ("$500K", "$1M")]
bn_lo, bn_hi = min(_bn), max(_bn)
_ind = sorted({int(r6[f"h20|{p}|{t}"]["indep_trades"]) for p in ("dev", "confirm") for t in ("$500K", "$1M")})
indep_txt = f"{_ind[0]}" if len(_ind) == 1 else f"{_ind[0]}–{_ind[-1]}"
re10 = (sim.get("realistic") or {}).get("10000") or {}
doc = {
    "generated": m.get("trained_at", "")[:10],
    "question": "Can GRAVITY's daily #1 be shorted at $500K–$1M without moving the price — and keep its accuracy?",
    "method": [
        "Walk-forward, out of sample: monthly test folds, 5-session embargo (21 for 20-session holds), isotonic calibration on a later held-out slice — the production recipe.",
        "Variants were chosen on a DEVELOPMENT window (test months Jan–Oct 2025) and reported on an untouched CONFIRMATION window (Nov 2025 – Sep 2026), so picking the best of many cannot inflate the result.",
        "Point-in-time universe: every listed stock, a row kept only while the company was ≤ $2B at the time (traded price × the share count on its latest SEC cover page, split-adjusted). Delisted stocks are still missing.",
        "Capacity: ≤ 5% of the session's expected dollar volume per leg and ≤ 0.5% estimated one-way impact (square-root law, Y = 0.7, daily volatility), with a conservative next-session volume and, live, IBKR's lendable shares.",
        "Costs: impact on both legs + one spread. The spread is the least certain input: the close-high-low estimate is trusted only between one tick ($0.0001 under $1, $0.01 above) and a liquidity-tier cap (6% under $250K/day, 3% under $1M, 2% under $5M, 1% above); when the estimator can't read a spread the cap itself is used, and live the site uses the median real bid/ask the tape recorded when it has one. 'Tight spread' shows the result if real spreads sit near the tick.",
        "About 35 variants tested: production features; liquidity / intraday-habit features; bagging; liquid-only training; the pre-market gap; dump-minus-pump ranking; direct return models; protective stops; 5/10/20-session holds; two universe definitions; and (round 9) seven morning models built only from what is knowable at 9:00 ET.",
        "Round 9 (the morning model): Yahoo hourly bars with extended hours give, for each stock and morning, the last after-hours / pre-market trade before 9:00 ET, the extended-hours high and low, and how many hours had any trade (Yahoo reports no extended-hours volume). Overnight filings use EDGAR acceptance times. The winner was fixed by a rule written down before the results were seen (best development-window average move, then it had to match or beat the old morning model in the confirmation window).",
        "The tables in this section are research runs (their own universe build); the Track Record shows the production model's walk-forward with the same cost model. A single daily #1 is a noisy statistic (about 210–230 sessions per window): small changes to the universe moved the same model's average by about a point, so read differences smaller than that as noise. The morning-model comparison is paired — every variant on the same days and universe — which is what makes its ranking meaningful.",
    ],
    "intraday": intraday,
    "multi_day_20": multi,
    "main_pit": main,
    "official": {"pub1_hit": o.get("pub1_hit"), "pub1_mean_oc": o.get("pub1_mean_oc"), "pub1_median_oc": o.get("pub1_median_oc"),
                 "base": o.get("universe_daily_hit"), "net10k": re10.get("net_mean"), "cost10k": re10.get("mean_cost")},
    "verdict": [
        f"The close-only model (the evening watchlist, and the morning fallback) still finds dumps — {o['pub1_hit']*100:.0f}% of its #1s fell 5%+ open→close vs {o['universe_daily_hit']*100:.0f}% for the average name (point-in-time backtest), averaging {sg(o['pub1_mean_oc'])} before costs.",
        f"But these are thin stocks. With each pick's own estimated cost, its average trade nets {sg(re10.get('net_mean'))} at $10K (estimated cost ≈ {re10.get('mean_cost', 0)*100:.1f}% round trip). The live tape records real bid/ask spreads to keep checking the cost model.",
        f"Names that can take $500K–$1M: the model still finds dump candidates at about {lift_lo:.0f}–{lift_hi:.0f}× the normal rate, but their average same-day move is too small to pay the costs at that size. No variant — stops, direction-aware ranking, the pre-market gap, liquid-only models — fixed that.",
        f"20-session shorts in liquid names looked excellent ({sg(bn_lo, 0)} to {sg(bn_hi, 0)} per trade) on today's small-cap list, but that list over-represents stocks that had already collapsed. On a point-in-time universe the result is mixed and statistically indistinguishable from zero (about {indep_txt} independent trades per period). Not confirmed, not shipped.",
        "So the honest answer to “$500K–$1M without losing accuracy” is no — not with these signals and free data. GRAVITY shows every pick's capacity, cost at your size and breakeven, so you can size each trade to what the stock can actually absorb.",
    ],
}
# ── Round 9: what the morning run knows at ~9:00 ET (honest M1), timing, baskets ──
# Primary source: the LEAK-FREE re-run (r9c; features relative to the hourly series' own prior
# close), re-scored with the current cost model; r9 supplies the "official open" reference row,
# r9b the close-only baskets, r9d the 10:30 exit-rule test. Selection: data/research/r9_selection_rule.txt.
def _load(name):
    f = R / name
    return json.loads(f.read_text()) if f.exists() else None


r9, r9b, r9c, r9d = _load("r9.json"), _load("r9b.json"), _load("r9c.json"), _load("r9d.json")
if r9c and r9:
    V = dict(r9c["variants"])
    V["Bp_m1_official"] = r9["variants"]["Bp_m1_official"]
    if r9b:  # same predictions, current cost model
        for per in ("dev", "confirm"):
            for f_ in ("net_10k", "net_50k"):
                V["Bp_m1_official"][per]["top1"][f_] = r9b["variants"]["Bp_m1_official"][per][f_]
    LABELS = {
        "A_m0": "Close only (evening model)",
        "Bp_m1_official": "Morning model scored on the real 9:30 open (what the site showed — not knowable at 9:05)",
        "B_m1_live_lf": "Morning model as it ran live (trained on the 9:30 open, fed the pre-market price)",
        "C_m1_honest_lf": "Honest: trained and tested on the last trade before 9:00 ET",
        "D_ext_shape_lf": "Honest + extended-hours shape + how many hours traded",
        "D2_no_counts_lf": "Honest + extended-hours shape (high, low, fade, after-hours move)",
    }
    B = "B_m1_live_lf"
    t1 = lambda k, per: V[k][per]["top1"]  # noqa: E731
    # addendum rule 1: keep D if it still matches/beats the live model in confirmation
    keep_d = t1("D_ext_shape_lf", "confirm")["mean_oc"] <= t1(B, "confirm")["mean_oc"] and \
        t1("D_ext_shape_lf", "confirm")["hit"] >= t1(B, "confirm")["hit"] - 0.02
    chosen = "D_ext_shape_lf" if keep_d else "C_m1_honest_lf"
    # addendum rule 2: prefer the version without bar counts if it loses nothing meaningful (≤ 0.5 pt)
    if keep_d and all(t1("D2_no_counts_lf", per)["mean_oc"] - t1("D_ext_shape_lf", per)["mean_oc"] <= 0.005 for per in ("dev", "confirm")):
        chosen = "D2_no_counts_lf"

    def vrow(k):
        g = lambda per, f: (V[k][per]["top1"] or {}).get(f)  # noqa: E731
        pv = (r9c.get("paired_vs_B") or {}).get(k)
        return {"key": k, "label": LABELS.get(k, k),
                "hit": dc(lambda per: g(per, "hit")), "mean_oc": dc(lambda per: g(per, "mean_oc")),
                "median_oc": dc(lambda per: g(per, "median_oc")), "squeeze": dc(lambda per: g(per, "squeeze")),
                "net10k": dc(lambda per: g(per, "net_10k")), "net50k": dc(lambda per: g(per, "net_50k")),
                "auc": dc(lambda per: V[k][per].get("auc_pub")), "vs_live": pv}

    rows9 = [vrow(k) for k in ("A_m0", "Bp_m1_official", B, "C_m1_honest_lf", "D_ext_shape_lf", "D2_no_counts_lf") if k in V]
    bk_m1 = (r9c.get("baskets") or {}).get(chosen) or (r9c.get("baskets") or {}).get("D2_no_counts_lf") or {}
    bk_m0 = ((r9b or {}).get("baskets") or {}).get("A_m0") or {}
    basket = {m: {f"top{k}": {per: src.get(f"{per}|top{k}|50k") for per in ("dev", "confirm")} for k in (1, 3, 5, 10)}
              for m, src in (("m1", bk_m1), ("m0", bk_m0))}
    exit_rule = ((r9d or {}).get("exit_rule") or {}).get(chosen)
    cov = dict(r9c.get("coverage") or {})
    cov.setdefault("corr_gap_ext_vs_open", (r9.get("coverage") or {}).get("corr_gap_ext_vs_open"))
    pv = (r9c.get("paired_vs_B") or {}).get(chosen) or {}
    doc["morning"] = {"rows": rows9, "chosen": chosen, "kept_shape": keep_d, "coverage": cov,
                      "timing": (r9c.get("timing") or {}).get(chosen), "timing_variant": chosen, "exit_rule": exit_rule,
                      "paired": pv}
    doc["basket"] = basket
    cv = lambda per, f: t1(chosen, per)[f]  # noqa: E731
    bv = lambda per, f: t1(B, per)[f]  # noqa: E731
    ci = lambda per: (pv.get(per) or {}).get("ci90") or [None, None]  # noqa: E731
    doc["verdict"] = [
        f"New: the morning #1 now uses only what the 9:05 ET run can know — after-hours and pre-market trading. "
        f"Its #1 fell 5%+ open→close on {cv('dev', 'hit')*100:.0f}% / {cv('confirm', 'hit')*100:.0f}% of sessions (development / confirmation) "
        f"vs {bv('dev', 'hit')*100:.0f}% / {bv('confirm', 'hit')*100:.0f}% for the morning model it replaces, averaging "
        f"{sg(cv('dev', 'mean_oc'))} / {sg(cv('confirm', 'mean_oc'))} before costs (old: {sg(bv('dev', 'mean_oc'))} / {sg(bv('confirm', 'mean_oc'))}).",
        f"Paired day by day against the old morning model, the average short gained {sg(pv.get('dev', {}).get('mean_diff'))} per trade in development "
        f"(90% interval {sg(ci('dev')[0])} to {sg(ci('dev')[1])}) and {sg(pv.get('confirm', {}).get('mean_diff'))} in confirmation "
        f"(interval {sg(ci('confirm')[0])} to {sg(ci('confirm')[1])}) — a clear gain in development and in the dump rate; in confirmation the average move is about the same.",
        f"After each pick's own estimated cost: {sg(cv('dev', 'net_10k'))} / {sg(cv('confirm', 'net_10k'))} per trade at $10K and "
        f"{sg(cv('dev', 'net_50k'))} / {sg(cv('confirm', 'net_50k'))} at $50K (old morning model: {sg(bv('dev', 'net_10k'))} / {sg(bv('confirm', 'net_10k'))} at $10K, "
        f"{sg(bv('dev', 'net_50k'))} / {sg(bv('confirm', 'net_50k'))} at $50K). Costs assume wide spreads whenever the estimator can't read one — real quotes logged since 7 Oct showed thin names quoting 4–9% wide.",
        "An earlier version of this test looked better because of a subtle leak (rows blanked for disagreeing with the daily bars were mostly names that reverse-split later); these are the leak-free numbers.",
    ] + doc["verdict"]
    if exit_rule:
        e = exit_rule
        doc["verdict"].insert(4, f"Covering at 10:30 whenever the #1 is above its open did not help: on the days with hourly data it changed the $10K net from "
                                 f"{sg(e['dev']['hold_net10k'])} to {sg(e['dev']['rule_net10k'])} (development) and from "
                                 f"{sg(e['confirm']['hold_net10k'])} to {sg(e['confirm']['rule_net10k'])} (confirmation). It did trim the worst single day "
                                 f"(from {sg(e['confirm']['hold_worst'], 0)} to {sg(e['confirm']['rule_worst'], 0)} in confirmation) — a risk-for-return trade, not a free improvement. Holding to the close stays the default.")
    # tail risk: production walk-forward (model.json) + the pre-registered guard test (r9e)
    try:
        import numpy as _np
        cols = m["sim"]["m0"]["columns"]
        ip = cols.index("oc_pub1")
        o0 = _np.array([r[ip] for r in m["sim"]["m0"]["daily"] if r[ip] is not None], float)
        o1 = _np.array([r[ip] for r in m["sim"]["m1"]["daily"] if r[ip] is not None], float)
        if len(o0) and len(o1):
            doc["morning"]["tails"] = {"m0_worst": float(o0.max()), "m1_worst": float(o1.max()),
                                       "m0_median": float(_np.median(o0)), "m1_median": float(_np.median(o1)),
                                       "m0_mean": float(o0.mean()), "m1_mean": float(o1.mean())}
            doc["verdict"].insert(3, f"The catch: in the production walk-forward the morning #1 dumped more often ({m['oos']['m1']['pub1_hit']*100:.0f}% vs "
                                     f"{m['oos']['m0']['pub1_hit']*100:.0f}% for the close-only model) with a better median ({sg(_np.median(o1))} vs {sg(_np.median(o0))}), "
                                     f"but its squeezes are bigger (worst day {sg(o1.max(), 0)} vs {sg(o0.max(), 0)}; spiked 20%+ on "
                                     f"{m['oos']['m1'].get('pub1_squeeze_rate', 0)*100:.0f}% vs {m['oos']['m0'].get('pub1_squeeze_rate', 0)*100:.0f}% of days), so its plain average is about the same "
                                     f"({sg(o1.mean())} vs {sg(o0.mean())}). Size and stops accordingly.")
    except (KeyError, ValueError, TypeError):
        pass
    r9e = _load("r9e.json")
    if r9e:
        doc["verdict"].insert(4, "Skipping the morning #1 when it is a big pre-market gapper (four thresholds tested, chosen in development) cut the tail but "
                                 "removed most of the edge in confirmation — not adopted. The big gappers are where much of the edge is.")
        doc["morning"]["guards"] = r9e
    print("R9 (leak-free) → chosen", chosen, "(kept shape)" if keep_d else "(fell back to honest gap only)")

(ROOT / "docs" / "data" / "size_research.json").write_text(json.dumps(doc, indent=1))
print("written", len(json.dumps(doc)))
