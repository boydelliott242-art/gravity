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
        "Costs: impact on both legs + one spread. The spread is the least certain input: the close-high-low estimate is trusted only between one tick ($0.0001 under $1, $0.01 above) and a liquidity-tier cap (6% under $250K/day, 3% under $1M; from round 14 the caps above $1M/day are set from real quotes of the whole liquid universe: 1.45% under $5M, 0.9% under $20M, 0.55% above — before that 2% and 1%); when the estimator can't read a spread the cap itself is used, and live the site uses the median real bid/ask the tape recorded when it has one. 'Tight spread' shows the result if real spreads sit near the tick.",
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

# ── Round 10: the morning model that can actually run (07:20 CT, pre-market before 8:00 ET) ──
r10, r10b, r10c = _load("r10.json"), _load("r10b.json"), _load("r10c.json")
if r10 and r10b and (r10b.get("decision") or {}).get("adopt_pm_only") and r9c:
    V10 = {**r10["variants"], **r10b["variants"]}
    V10["B_m1_live_lf"] = r9c["variants"]["B_m1_live_lf"]
    LAB10 = {
        "A_m0": "Close only (evening model; the morning fallback)",
        "B_m1_live_lf": "Old morning model as it ran live (trained on the 9:30 open, fed the pre-market price)",
        "D2_9": "Extended-hours shape, trades before 9:00 ET (needs the 8:05 CT run)",
        "D2_8": "Extended-hours shape, trades before 8:00 ET",
        "D3_8": "Last pre-market trade before 8:00 ET only",
    }

    def vrow10(k):
        g = lambda per, f: (V10[k][per]["top1"] or {}).get(f)  # noqa: E731
        return {"key": k, "label": LAB10[k], "hit": dc(lambda per: g(per, "hit")), "mean_oc": dc(lambda per: g(per, "mean_oc")),
                "median_oc": dc(lambda per: g(per, "median_oc")), "squeeze": dc(lambda per: g(per, "squeeze")),
                "net10k": dc(lambda per: g(per, "net_10k")), "net50k": dc(lambda per: g(per, "net_50k")),
                "auc": dc(lambda per: V10[k][per].get("auc_pub")), "vs_close_only": (r10b.get("paired_vs_A") or {}).get(k)}

    rows10 = [vrow10(k) for k in ("A_m0", "B_m1_live_lf", "D2_9", "D2_8", "D3_8")]
    doc.setdefault("morning", {})
    doc["morning"].update({"rows": rows10, "chosen": "D3_8", "round": 10, "cutoff_et": "8:00",
                           "paired": (r10b.get("paired_vs_A") or {}).get("D3_8")})
    if r10c:
        bk = lambda src: {f"top{k}": {per: src.get(f"{per}|top{k}|50k") for per in ("dev", "confirm")} for k in (1, 3, 5, 10)}  # noqa: E731
        doc["basket"] = {"m1": bk(r10c["D3_8"]["baskets"]), "m0": bk(r10c["A_m0"]["baskets"])}
    c = lambda per, f: V10["D3_8"][per]["top1"][f]  # noqa: E731
    a = lambda per, f: V10["A_m0"][per]["top1"][f]  # noqa: E731
    d9 = lambda per, f: V10["D2_9"][per]["top1"][f]  # noqa: E731
    pv = (r10b.get("paired_vs_A") or {}).get("D3_8") or {}
    ci = lambda per: (pv.get(per) or {}).get("ci90") or [None, None]  # noqa: E731
    head = [
        f"New: the morning #1 now adds the last PRE-MARKET trade before 8:00 ET, so it can run at 7:20 CT. In the backtest its #1 fell 5%+ "
        f"open→close on {c('dev', 'hit')*100:.0f}% / {c('confirm', 'hit')*100:.0f}% of sessions (development / confirmation) vs "
        f"{a('dev', 'hit')*100:.0f}% / {a('confirm', 'hit')*100:.0f}% for the close-only model, averaging {sg(c('dev', 'mean_oc'))} / {sg(c('confirm', 'mean_oc'))} "
        f"(close-only {sg(a('dev', 'mean_oc'))} / {sg(a('confirm', 'mean_oc'))}); after each pick's estimated cost {sg(c('dev', 'net_10k'))} / {sg(c('confirm', 'net_10k'))} per trade at $10K.",
        f"Paired day by day with the close-only model: {sg(pv.get('dev', {}).get('mean_diff'))} per trade in development (90% interval {sg(ci('dev')[0])} to {sg(ci('dev')[1])}) "
        f"and {sg(pv.get('confirm', {}).get('mean_diff'))} in confirmation ({sg(ci('confirm')[0])} to {sg(ci('confirm')[1])}) — better in both windows, though each interval still includes zero.",
        f"Why not the richer 9:00 ET version (which did better: {d9('confirm', 'hit')*100:.0f}% dumped, {sg(d9('confirm', 'mean_oc'))} in confirmation)? It needs the 8:05 CT run, "
        "and on 8 and 9 October that run could not finish before the bell while the Mac slept; Yahoo also stopped serving after-hours bars (17:00–18:59 ET) on 7 October. "
        "The pre-market-only version needs neither — and a feed check switches the morning model off (close-only scores) on any day its inputs look abnormal.",
    ]
    old_morning = [v for v in doc["verdict"] if v.startswith(("New: the morning #1", "Paired day by day", "After each pick's own estimated cost", "The catch:", "Skipping the morning", "An earlier version", "Covering at 10:30"))]
    doc["verdict"] = head + [v for v in doc["verdict"] if v not in old_morning]
    print("R10: morning model = pre-market only, 8:00 ET cutoff")

# ── Round 11/12: every board name able to take a $30K short ──
f11, f12, f11c = R / "r11.json", R / "r12.json", R / "r11c.json"
if f11.exists():
    r11 = json.loads(f11.read_text())
    r12 = json.loads(f12.read_text()) if f12.exists() else {}
    r11c = json.loads(f11c.read_text()) if f11c.exists() else {}
    rc = json.loads((R / "r11_costs.json").read_text()) if (R / "r11_costs.json").exists() else {}
    won = r11["decision"]["adopted"]
    on = bool(r11c.get("decision", {}).get("overnight_filings")) if r11c else bool(r11["decision"].get("overnight_filings"))
    V, VC = r11["variants"], r11c.get("variants", {})
    # what production actually publishes each morning: close-only ranker for every $30K-ready name, the morning
    # ranker for the top 450 of them (R11c hybrid); the evening watchlist / feed-check fallback = close-only ranker
    ship_key = ("hybrid_L4_on_allsym" if on else "hybrid_L4") if VC else ("W_on" if on else won)
    new = VC.get(ship_key) or V[won]
    m0v = VC.get("L4_m0") or V.get("W_m0")
    old = r11["before"]["old_board_at_30k"]
    dump_l = V["L0"]                                    # the dump-odds model ranked within the same $30K-ready names
    rows30 = []
    for k, lab in (("1-1", "#1"), ("2-10", "Ranks 2–10"), ("11-25", "Ranks 11–25"), ("26-50", "Ranks 26–50"),
                   ("51-100", "Ranks 51–100"), ("1-100", "All 100")):
        g = lambda src, f: dc(lambda per: src[per][k][f])  # noqa: E731
        rows30.append({"band": lab, "old_hit": g(old, "hit"), "old_mean": g(old, "mean_oc"), "old_net": g(old, "net30"),
                       "new_hit": g(new, "hit"), "new_mean": g(new, "mean_oc"), "new_cost": g(new, "cost30"),
                       "new_net": g(new, "net30"), "new_net_half": g(new, "net30_halfspread"), "new_squeeze": g(new, "squeeze")})
    nv = lambda per, k, f: new[per][k][f]  # noqa: E731
    ov = lambda per, k, f: old[per][k][f]  # noqa: E731
    two = lambda src, k, f, fmt=sg: f"{fmt(src['dev'][k][f])} / {fmt(src['confirm'][k][f])}"  # noqa: E731
    pc = lambda x: f"{x*100:.0f}%"  # noqa: E731
    pt2 = lambda x: f"{'+' if x >= 0 else '−'}{abs(x)*100:.2f} pt"  # noqa: E731
    allv = {c: dc(lambda per: V[c][per]["1-100"]["net30"]) for c in ("L0", "L1", "L2", "L3", "L4", "L5")}
    lo_all = min(min(v.values()) for v in allv.values()); hi_all = max(max(v.values()) for v in allv.values())
    pv = r11.get("paired_vs_L0", {})
    worse = [c for c, x in pv.items() if c in allv and any(x[per]["ci90"][1] < 0 for per in ("dev", "confirm"))]
    better = [c for c, x in pv.items() if c in allv and any(x[per]["ci90"][0] > 0 for per in ("dev", "confirm"))]
    names = {"L0": "the production dump model", "L1": "a dump classifier trained only on these names",
             "L2": "a 3%-drop classifier", "L3": "a regression on each stock's move versus its peers",
             "L4": "a regression on its within-day percentile", "L5": "an average of three of them"}
    andj = lambda xs: (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else (xs[0] if xs else "")  # noqa: E731
    hz = {H: dc(lambda per: r12[f"H{H}"][per]["1-100"]["net"]) for H in (5, 10) if f"H{H}" in r12}
    hz_lo = min((min(v.values()) for v in hz.values()), default=None); hz_hi = max((max(v.values()) for v in hz.values()), default=None)
    nb = rc.get("net_by_spread_multiple", {})
    verdict = [
        f"New: all 100 board names (and the #1) pass a $30K test — at most 5% of the stock's expected volume, at most 0.5% "
        f"estimated price impact, an as-traded price of $1 or more and no short-sale restriction; live, each board name must "
        f"also be lendable at IBKR for $30K or more whenever IBKR's list is available (otherwise borrow shows as unknown). "
        f"About {r11['universe']['per_session_median']:.0f} stocks pass the volume, impact and price test on a typical day, "
        f"before the borrow, squeeze and halt checks.",
        f"The honest catch: the #1 dumps far less often. In the backtest the $30K board's #1 fell 5%+ open→close on "
        f"{two(new, '1-1', 'hit', pc)} of sessions (old #1: {two(old, '1-1', 'hit', pc)}); ranks 51–100 on "
        f"{two(new, '51-100', 'hit', pc)} (old: {two(old, '51-100', 'hit', pc)}). Most of that is liquidity — the dump-odds "
        f"model's own #1 among the same $30K-ready names fell 5%+ on {two(dump_l, '1-1', 'hit', pc)} — and part is the ranker, "
        f"which was chosen for all 100 names, not for the #1: its #1 dumps less often and netted "
        + ("better" if all(new[p]['1-1']['net30'] > dump_l[p]['1-1']['net30'] for p in ('dev', 'confirm')) else
           "worse" if all(new[p]['1-1']['net30'] < dump_l[p]['1-1']['net30'] - 0.0005 for p in ('dev', 'confirm')) else "no better")
        + f" at $30K ({two(new, '1-1', 'net30')} versus {two(dump_l, '1-1', 'net30')} for the dump-odds #1). Same-or-better "
        f"accuracy is not available at this size: the big "
        f"intraday dumps happen in stocks too thin to take $30K.",
        f"The old #1 itself, even charged its large estimated cost, netted {two(old, '1-1', 'net30')} at $30K — better than "
        f"the new #1's {two(new, '1-1', 'net30')}. But it could take $30K within the volume and impact limits on only "
        + (f"{pc(rc['old_top1_cap30_share']['dev'])} / {pc(rc['old_top1_cap30_share']['confirm'])} of days"
           if rc.get("old_top1_cap30_share") else "a minority of days")
        + " — the rest of the time a $30K short was exactly the heavy slippage this change exists to avoid, and the cost "
        "model is least reliable for such oversized orders.",
        f"Across all 100 at $30K a name, the new board is far better than the old one: per position after estimated costs "
        f"{two(new, '1-100', 'net30')}, versus {two(old, '1-100', 'net30')} for the old board, whose thin names would pay "
        f"about {ov('confirm', '1-100', 'cost30')*100:.1f}% to trade $30K.",
        f"But shorting all 100 at $30K still lost money in the backtest: the average name moved only "
        f"{sg(nv('confirm', '1-100', 'mean_oc'))} open→close. Under the cost model of 12 October (about "
        f"{nv('confirm', '1-100', 'cost30')*100:.1f}% round trip; the table above) that was {two(new, '1-100', 'net30')} per position"
        + (f"; with the spread caps re-measured from real quotes (round 14, about "
           f"{(rc['mean_spread_used'] + rc['mean_two_way_impact'])*100:.1f}% round trip) it is still "
           f"{sg(nb['1.0']['dev'])} / {sg(nb['1.0']['confirm'])} ({sg(nb['0.5']['confirm'])} if real spreads are half of that, "
           f"{sg(nb['2.0']['confirm'])} if double)" if nb.get("1.0") and nb.get("0.5") and nb.get("2.0") else "")
        + f". The top 10 come closest ({two(new, '1-10', 'net30')} under the old costs). Treat the lower ranks as a watchlist.",
    ]
    if m0v:
        verdict.append(
            f"The evening watchlist, and the morning page whenever the pre-market feed check fails, use the close-only version "
            f"of the ranker: its #1 fell 5%+ on {two(m0v, '1-1', 'hit', pc)} and netted {two(m0v, '1-1', 'net30')}; all 100 "
            f"netted {two(m0v, '1-100', 'net30')} per position. The morning version above re-scores the top 450 $30K-ready names "
            f"with the pre-market gap" + (" and overnight filings" if on else "") + " — exactly as the live run does.")
    rule = (f"Six ranking models were compared under a rule written down before the run: rank by development 100-name net, "
            f"treat gaps under 0.05 pt as ties, break ties by development top-10 net, then require confirmation no worse than "
            f"the production model. It picked {names.get(won, won)}. The six spanned {sg(hi_all)} to {sg(lo_all)} per position")
    rule += ("; " + (f"{andj([names.get(c, c) for c in worse])} {'was' if len(worse) == 1 else 'were'} measurably worse than "
                     f"production in at least one window (paired 90% interval below zero)" if worse else "")
             + ("; " if worse and better else "")
             + (f"{andj([names.get(c, c) for c in better])} measurably better in at least one window" if better else "")
             + ("; the rest were statistically indistinguishable from production" if (worse or better) else
                "; none was statistically distinguishable from production")) if pv else ""
    pw = pv.get(won)
    if pw:
        spans = all(pw[per]["ci90"][0] <= 0 <= pw[per]["ci90"][1] for per in ("dev", "confirm"))
        rule += (f"; against production, the adopted ranker's paired 90% interval {'spans zero in both windows' if spans else 'excludes zero in at least one window'}")
    rule += "."
    if r11c:
        g0 = {per: VC["L4_on_allsym"][per]["1-100"]["net30"] - VC["L4"][per]["1-100"]["net30"] for per in ("dev", "confirm")}
        rule += (f" A second pre-registered step tested adding counts of SEC filings accepted overnight (up to 8:00 ET). Its "
                 f"first run took filing histories only for today's ≤ $2B universe — a look-ahead (names that later grew had "
                 f"none), caught in review — so it was re-run with every stock's history: {pt2(g0['dev'])} / "
                 f"{pt2(g0['confirm'])} per position against a bar of +0.05 pt in both windows, so overnight filings are "
                 + ("ON in the morning ranker." if on else "NOT used."))
    verdict.append(rule)
    if hz_lo is not None:
        h5, h10 = r12.get("H5", {}), r12.get("H10", {})
        sent = (f"Holding 5 or 10 sessions instead (tested with the first round's peer-relative regression) did not fix the "
                f"100-name board ({sg(hz_hi)} to {sg(hz_lo)} per position).")
        if h5.get("dev", {}).get("1-1") and h10.get("dev", {}).get("1-1"):
            sent += (f" Exploratory only — that test was designed after looking at 5-session moves and the top ranks were not "
                     f"its question: the #1 netted {sg(h5['dev']['1-1']['net'])} / {sg(h5['confirm']['1-1']['net'])} held 5 "
                     f"sessions and {sg(h10['dev']['1-1']['net'])} / {sg(h10['confirm']['1-1']['net'])} held 10; the top 10 "
                     f"{sg(h5['dev']['1-10']['net'])} / {sg(h5['confirm']['1-10']['net'])} and {sg(h10['dev']['1-10']['net'])} / "
                     f"{sg(h10['confirm']['1-10']['net'])}. A 10-session #1 is only about {h10['confirm']['1-1']['n'] // 10} "
                     f"independent trades per window, and 1 in 20 of its positions lost "
                     f"{h10['dev']['1-1']['p95_adverse']*100:.0f}% / {h10['confirm']['1-1']['p95_adverse']*100:.0f}% or more "
                     f"before costs. Not shipped, not a recommendation.")
        verdict.append(sent)
    if rc:
        verdict.append(
            f"Costs remain the biggest uncertainty, in both directions. For {rc['share_at_cap']*100:.0f}% of backtest board names "
            f"the spread used is the liquidity-tier cap ({rc['share_unreadable']*100:.0f}% because the estimator could not read a "
            f"spread, {rc['share_read_wider_clipped']*100:.0f}% because it read a wider one that was clipped down). Round 14 set "
            f"those caps from real quotes — one session, three rounds of mid-session quotes of every liquid name — but spreads "
            f"at the open and close, when these trades happen, were not sampled yet.")
    doc["board30k"] = {
        "title": "Round 11 — every name on the board can take a $30K short",
        "note": ("Walk-forward, out of sample, development / confirmation. Old board = the previous ranking (dump odds) among "
                 "publishable names (no Rule 201 next session, ≥ $300K a day) whatever their capacity, measured as if $30K "
                 "went into each; $30K board = the morning board as the live run builds it (close-only ranker for every "
                 "$30K-ready name, the morning ranker for its top 450). Net = open→close short minus each name's own "
                 "estimated round-trip cost at $30K (square-root impact on both legs + spread)."),
        "rows": rows30, "adopted": won + (" + overnight filings" if on else ""), "shipped_variant": ship_key,
        "per_session": r11["universe"]["per_session_median"],
        "variants_all100": allv, "horizons_all100": hz, "costs": rc, "verdict": verdict,
    }
    doc["question"] = "How much can each pick take without moving the price — and does accuracy survive the size?"
    print("R11: $30K board block added (shipped", ship_key + ")")

# ── Round 13: the ONE — the single main-page pick, shortable at $100K+ ──
f13, f13b = R / "r13.json", R / "r13b.json"
if f13.exists() and f13b.exists():
    r13, r13b = json.loads(f13.read_text()), json.loads(f13b.read_text())
    names13 = {"O0": "Production dump model (any liquidity)", "O1": "Dump model trained on $30K-ready names",
               "O2": "Dump model trained on $100K-ready names", "O3": "O1 averaged over 5 seeds",
               "O4": "Consensus of three dump models (rank-average)", "O5": "O1 + overnight filings and earnings releases",
               "O6": "≥10%-drop model"}
    V13 = r13["variants"]
    rows13 = [{"key": k, "label": names13.get(k, k),
               **{f: dc(lambda per: V13[k][per]["top1"][f]) for f in ("hit", "big10", "mean_oc", "net100k")}}
              for k in ("O0", "O1", "O2", "O3", "O4", "O5", "O6")]
    hy = r13b["variants"]["O4_morning_hybrid"]
    ev = r13b["variants"]["O4_evening_m0"]
    th = r13b["tiers"]["O4_morning_hybrid"]
    won13 = r13["decision"]["adopted"]
    pc = lambda x: f"{x*100:.0f}%"  # noqa: E731
    tw = lambda src, f, fmt=sg: f"{fmt(src['dev']['top1'][f])} / {fmt(src['confirm']['top1'][f])}"  # noqa: E731
    fl = r13.get("floors", {})
    v13 = [
        f"New: the main page shows ONE stock — the most accurate dump call found among stocks that can take a $100K short "
        f"without moving the price (≤ 5% of expected volume, ≤ 0.5% estimated impact, $1+, IBKR able to lend $100K). Across "
        f"the whole sample {r13['universe']['base_dump']*100:.1f}% of those stocks fell 5%+ open→close on an average day.",
        f"Seven ways to pick it were compared under a rule written down before the run (best development #1 dump rate, "
        f"ties within 2 points broken by the top-3 rate; then in confirmation the #1 no more than 2 points worse and the "
        f"top 3 no worse than production). It picked {names13.get(won13, won13).lower()}: the stock that the production "
        f"dump model, a ≥5%-drop model and a ≥3%-drop model (both trained only on liquid names) jointly rank highest. Its #1 fell 5%+ on {tw(V13[won13], 'hit', pc)} of sessions versus "
        f"{tw(V13['O0'], 'hit', pc)} for the production model alone.",
        f"Measured as the live morning run ranks it (before the live-only borrow, IBKR-lendable, squeeze and halt checks, "
        f"which have no history), the ONE fell 5%+ on {tw(hy, 'hit', pc)} of sessions and 10%+ "
        f"on {tw(hy, 'big10', pc)}, averaging {tw(hy, 'mean_oc')} open→close. The evening (close-only) version: "
        f"{tw(ev, 'hit', pc)}.",
        f"Conviction tiers (how closely the three models agree it is the top name): high-conviction days fell 5%+ on "
        f"{pc(th['high']['dev']['hit'])} / {pc(th['high']['confirm']['hit'])} ({th['high']['dev']['days']} / "
        f"{th['high']['confirm']['days']} days), medium {pc(th['medium']['dev']['hit'])} / {pc(th['medium']['confirm']['hit'])}, "
        f"low {pc(th['low']['dev']['hit'])} / {pc(th['low']['confirm']['hit'])}. The high tier is a small sample and was "
        f"not monotone in development — read it as a hint, not a promise.",
        f"Super-high accuracy at this size is not something the data supports: the ONE is about 8× the base rate, but it "
        f"is still wrong about two days in three, and on the days it is wrong the stock often rises (that is why the "
        f"average move is small). After each pick's own estimated cost it netted {tw(hy, 'net100k')} per trade at $100K "
        f"and {tw(hy, 'net500k')} at $500K.",
    ]
    if fl:
        v13.append("Requiring more capacity lowers accuracy: with a $250K floor the same rule's #1 fell 5%+ on "
                   f"{tw(fl['250000'], 'hit', pc)}, at $500K {tw(fl['500000'], 'hit', pc)}, at $1M {tw(fl['1000000'], 'hit', pc)} "
                   "— so the floor is $100K and the page shows how much more each day's pick can take.")
    doc["one13"] = {"title": "Round 13 — the ONE: one stock, the highest accuracy we can find, shortable at $100K+",
                    "note": ("Walk-forward, out of sample, development / confirmation. #1 = each session's top-ranked "
                             "$100K-shortable name under each method. Net = open→close short minus each pick's own "
                             "estimated round-trip cost at $100K."),
                    "rows": rows13, "adopted": won13,
                    "shipped": {f: dc(lambda per: hy[per]["top1"][f]) for f in ("hit", "big10", "mean_oc", "net100k", "net500k")},
                    "tiers": th, "verdict": v13}
    print("R13: ONE block added (adopted", won13 + ")")

(ROOT / "docs" / "data" / "size_research.json").write_text(json.dumps(doc, indent=1))
print("written", len(json.dumps(doc)))
